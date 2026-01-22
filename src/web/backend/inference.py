import cv2
import torch
import numpy as np
import pickle
from pathlib import Path
from collections import deque
import base64
import sys
import os

# Add src to path to import existing modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from feature_extraction import HandLandmarkExtractor
from model import ASLFingerSpellingLSTM
from utils import load_config, get_idx_to_class
from sentence_inference import SentenceBuilder

class WebASLInference:
    """ASL Inference Engine for Web Socket"""
    
    def __init__(self, model_path: str, config_path: str, scaler_path: str):
        self.config = load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = ASLFingerSpellingLSTM(
            input_size=self.config['model']['input_size'],
            hidden_size_1=self.config['model']['lstm_hidden_1'],
            hidden_size_2=self.config['model']['lstm_hidden_2'],
            num_classes=self.config['model']['num_classes'],
            dropout=self.config['model']['dropout'],
            bidirectional=self.config['model']['bidirectional']
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
            
        # Feature extractor
        self.extractor = HandLandmarkExtractor()
        
        # Configuration
        self.sequence_length = self.config['data_collection']['sequence_length']
        self.num_features = self.config['features']['num_features']
        self.confidence_threshold = self.config['inference']['confidence_threshold']
        self.smoothing_window = self.config['inference']['smoothing_window']
        
        # Class mapping
        self.idx_to_class = get_idx_to_class(self.config['data_collection']['classes'])
        
        # Buffers
        self.sequence_buffer = deque(maxlen=self.sequence_length)
        self.prediction_buffer = deque(maxlen=self.smoothing_window)
        
        # Sentence builder
        self.sentence_builder = SentenceBuilder()
        self.sentence_builder.min_repeats_for_add = 10  # Increase stability buffer
        
        self.frame_count = 0

    def process_frame(self, image_bytes):
        """Process a single frame bytes and return state"""
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return None
            
        # Flip frame (mirror effect)
        frame = cv2.flip(frame, 1)
        
        # Extract features
        features, results = self.extractor.extract_landmarks(frame)
        
        # Draw landmarks (optional, if we want to send back processed frame)
        # For now, let's just return key info
        
        # Add to buffer
        self.sequence_buffer.append(features)
        
        current_prediction = None
        current_confidence = 0.0
        status_message = self.sentence_builder.get_status_message()
        is_right_hand = True
        
        # Check Handedness
        if results.handedness:
            # MediaPipe returns 'Left' or 'Right'
            # Since we flipped the frame, 'Right' corresponds to user's Right hand
            hand_label = results.handedness[0][0].category_name
            
            if hand_label == 'Left':
                is_right_hand = False
                status_message = "⚠️ PLEASE USE RIGHT HAND"
        
        # Make prediction
        if len(self.sequence_buffer) == self.sequence_length:
            if self.frame_count % 2 == 0:  # Predict every 2 frames for smoother web experience
                sequence = np.array(self.sequence_buffer)
                
                # Preprocess
                sequence_reshaped = sequence.reshape(-1, self.num_features)
                sequence_normalized = self.scaler.transform(sequence_reshaped)
                sequence_normalized = sequence_normalized.reshape(1, self.sequence_length, self.num_features)
                tensor = torch.FloatTensor(sequence_normalized).to(self.device)
                
                with torch.no_grad():
                    output = self.model(tensor)
                    probabilities = torch.softmax(output, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, 1)
                
                pred_class = self.idx_to_class[predicted_idx.item()]
                conf_val = confidence.item()
                
                self.prediction_buffer.append((pred_class, conf_val))
                
                # Smooth prediction
                from collections import Counter
                predictions = [p[0] for p in self.prediction_buffer]
                prediction_counts = Counter(predictions)
                most_common = prediction_counts.most_common(1)[0][0]
                avg_confidence = np.mean([c for p, c in self.prediction_buffer if p == most_common])
                
                current_prediction = most_common
                current_confidence = float(avg_confidence)
                
                # Add to sentence
                if current_prediction:
                    added = self.sentence_builder.add_detection(
                        current_prediction, current_confidence
                    )
                    if added:
                        self.sentence_builder.auto_correct_current_word()
        
        self.frame_count += 1
        
        # Extract landmarks if hand detected
        landmarks = None
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
        
        return {
            "prediction": current_prediction,
            "confidence": current_confidence,
            "current_sentence": self.sentence_builder.get_current_sentence(),
            "last_added": self.sentence_builder.last_added_letter,
            "history": self.sentence_builder.sentence_history,
            "landmarks": landmarks,
            "is_right_hand": is_right_hand,
            "status_message": status_message
        }

    def handle_command(self, command):
        """Handle control commands"""
        if command == 'SPACE':
            self.sentence_builder.add_space()
        elif command == 'BACKSPACE':
            self.sentence_builder.backspace()
        elif command == 'ENTER':
            self.sentence_builder.finish_sentence()
        elif command == 'CLEAR':
            self.sentence_builder.clear()
