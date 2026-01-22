"""
Silent Voice Bridge - Streamlit Dashboard
Real-time ASL with Full-Screen Video
"""

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import cv2
import torch
import numpy as np
import pickle
from pathlib import Path
from collections import deque
import sys

sys.path.append(str(Path(__file__).parent / 'src'))

from src.feature_extraction import HandLandmarkExtractor
from src.model import ASLFingerSpellingLSTM
from src.utils import load_config, get_idx_to_class
from src.sentence_inference import SentenceBuilder

# Page Config
st.set_page_config(
    page_title="Silent Voice Bridge",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS - Full screen video
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main, .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Make video FULL SIZE */
    video {
        width: 100% !important;
        height: auto !important;
        border-radius: 12px;
    }
    
    .text-display {
        background: #050508;
        border: 2px solid #667eea;
        border-radius: 12px;
        padding: 32px;
        min-height: 300px;
        font-size: 48px;
        font-weight: 300;
        color: #ffffff;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.3);
    }
    
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
    }
    
    .cursor {
        display: inline-block;
        width: 3px;
        height: 1em;
        background: #667eea;
        margin-left: 8px;
        animation: blink 1s infinite;
    }
    
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #e94bb9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    h3 { color: #ffffff; font-weight: 600; }
    
    .status-badge {
        background: rgba(16, 185, 129, 0.2);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #10b981;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 600;
    }
    
    .prediction-box {
        background: rgba(102, 126, 234, 0.15);
        border: 2px solid #667eea;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 16px 0;
    }
    
    .prediction-letter {
        font-size: 72px;
        font-weight: 700;
        color: #667eea;
        text-shadow: 0 0 30px rgba(102, 126, 234, 0.8);
    }
    
    .history-item {
        background: rgba(255, 255, 255, 0.05);
        border-left: 3px solid #667eea;
        padding: 12px 16px;
        margin-bottom: 8px;
        border-radius: 8px;
        color: #ffffff;
    }
    
    .history-time {
        color: #667eea;
        font-size: 12px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize
if 'sentence_builder' not in st.session_state:
    st.session_state.sentence_builder = SentenceBuilder()
    st.session_state.current_prediction = None
    st.session_state.current_confidence = 0.0
    st.session_state.frame_count = 0

# Load Model
@st.cache_resource
def load_model():
    config = load_config('config/config.yaml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_path = "models/checkpoints/best_model.pth"
    checkpoint = torch.load(model_path, map_location=device)
    
    model = ASLFingerSpellingLSTM(
        input_size=config['model']['input_size'],
        hidden_size_1=config['model']['lstm_hidden_1'],
        hidden_size_2=config['model']['lstm_hidden_2'],
        num_classes=config['model']['num_classes'],
        dropout=config['model']['dropout'],
        bidirectional=config['model']['bidirectional']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    scaler_path = Path("data/processed/scaler.pkl")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    idx_to_class = get_idx_to_class(config['data_collection']['classes'])
    
    return model, scaler, config, device, idx_to_class

try:
    model, scaler, config, device, idx_to_class = load_model()
    extractor = HandLandmarkExtractor()
except Exception as e:
    st.error(f"❌ Error: {e}")
    st.stop()

# Buffers
if 'sequence_buffer' not in st.session_state:
    st.session_state.sequence_buffer = deque(maxlen=config['data_collection']['sequence_length'])
    st.session_state.prediction_buffer = deque(maxlen=config['inference']['smoothing_window'])

# Video callback with LANDMARKS
class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        # Extract features AND DRAW LANDMARKS
        features, results = extractor.extract_landmarks(img)
        
        # IMPORTANT: Draw the landmarks on the image
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                extractor.mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    extractor.mp_hands.HAND_CONNECTIONS,
                    extractor.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    extractor.mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)
                )
        
        # Add to buffer
        st.session_state.sequence_buffer.append(features)
        
        # Predict every 3 frames
        if len(st.session_state.sequence_buffer) == config['data_collection']['sequence_length']:
            if st.session_state.frame_count % 3 == 0:
                sequence = np.array(st.session_state.sequence_buffer)
                sequence_reshaped = sequence.reshape(-1, config['features']['num_features'])
                sequence_normalized = scaler.transform(sequence_reshaped)
                sequence_normalized = sequence_normalized.reshape(1, config['data_collection']['sequence_length'], 
                                                                 config['features']['num_features'])
                tensor = torch.FloatTensor(sequence_normalized).to(device)
                
                with torch.no_grad():
                    output = model(tensor)
                    probabilities = torch.softmax(output, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, 1)
                
                pred_class = idx_to_class[predicted_idx.item()]
                conf_val = confidence.item()
                
                st.session_state.prediction_buffer.append((pred_class, conf_val))
                
                from collections import Counter
                predictions = [p[0] for p in st.session_state.prediction_buffer]
                prediction_counts = Counter(predictions)
                most_common = prediction_counts.most_common(1)[0][0]
                avg_confidence = np.mean([c for p, c in st.session_state.prediction_buffer if p == most_common])
                
                st.session_state.current_prediction = most_common
                st.session_state.current_confidence = float(avg_confidence)
                
                if st.session_state.current_prediction and st.session_state.current_confidence > 0.7:
                    st.session_state.sentence_builder.add_detection(
                        st.session_state.current_prediction, st.session_state.current_confidence
                    )
        
        st.session_state.frame_count += 1
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Header
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("🤟 Silent Voice Bridge")
    st.markdown("### ASL Fingerspelling to English Translation")
with col_h2:
    st.markdown('<div class="status-badge">● Live</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main Layout - FULL SCREEN VIDEO
col1, col2 = st.columns([1.2, 1], gap="large")

with col1:
    st.markdown("### 📹 Camera Feed")
    
    # WebRTC Video Stream - FULL SIZE
    webrtc_ctx = webrtc_streamer(
        key="asl-camera",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 1280},
                "height": {"ideal": 720}
            },
            "audio": False
        },
        async_processing=True,
    )
    
    # Show current prediction BELOW video
    if st.session_state.current_prediction and st.session_state.current_confidence > 0.6:
        st.markdown(
            f'''<div class="prediction-box">
                <div class="prediction-letter">{st.session_state.current_prediction}</div>
                <div style="color: #a0a0a0; margin-top: 12px; font-size: 20px; font-weight: 600;">{int(st.session_state.current_confidence * 100)}% Confidence</div>
            </div>''',
            unsafe_allow_html=True
        )

with col2:
    st.markdown("### 💬 English Translation")
    
    # Text Display
    current_sentence = st.session_state.sentence_builder.get_current_sentence()
    if current_sentence:
        st.markdown(
            f'<div class="text-display">{current_sentence}<span class="cursor"></span></div>', 
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="text-display" style="color: #666; font-style: italic;">Start signing...</div>', 
            unsafe_allow_html=True
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Controls
    col_b1, col_b2, col_b3 = st.columns(3)
    
    with col_b1:
        if st.button("␣ Space", key="space", use_container_width=True):
            st.session_state.sentence_builder.add_space()
            st.rerun()
    
    with col_b2:
        if st.button("🔙 Back", key="backspace", use_container_width=True):
            st.session_state.sentence_builder.backspace()
            st.rerun()
    
    with col_b3:
        if st.button("🗑️ Clear", key="clear", use_container_width=True):
            st.session_state.sentence_builder.clear()
            st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("💾 Save Sentence", key="save", use_container_width=True, type="primary"):
        st.session_state.sentence_builder.finish_sentence()
        st.rerun()
    
    # History
    if st.session_state.sentence_builder.sentence_history:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📝 Saved Sentences")
        for timestamp, sentence in reversed(st.session_state.sentence_builder.sentence_history[-5:]):
            st.markdown(
                f'<div class="history-item"><span class="history-time">{timestamp}</span><br>{sentence}</div>',
                unsafe_allow_html=True
            )

# Reference
st.markdown("<br><br>", unsafe_allow_html=True)
with st.expander("🤟 ASL Alphabet Reference", expanded=False):
    st.markdown("**Hold each sign steady for 2-3 seconds • Press Space for word breaks**")
    st.image("docs/asl-alphabet-reference.png", use_container_width=True)
