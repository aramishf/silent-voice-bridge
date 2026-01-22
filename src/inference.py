"""
Real-time inference for ASL Fingerspelling Recognition
"""
import cv2
import torch
import numpy as np
import pickle
from pathlib import Path
from collections import deque
import sys

from feature_extraction import HandLandmarkExtractor
from model import ASLFingerSpellingLSTM
from utils import load_config, get_idx_to_class

sys.path.append(str(Path(__file__).parent.parent))


class ASLInference:
    """Real-time ASL fingerspelling inference"""
    
    def __init__(self, 
                 model_path: str,
                 config_path: str = "../config/config.yaml"):
        """
        Initialize inference system
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        print(f"Loading model from {model_path}...")
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
        print("✓ Model loaded successfully")
        
        # Load scaler
        scaler_path = Path(__file__).parent.parent / "data" / "processed" / "scaler.pkl"
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print("✓ Scaler loaded successfully")
        
        # Feature extractor
        self.extractor = HandLandmarkExtractor()
        
        # Configuration
        self.sequence_length = self.config['data_collection']['sequence_length']
        self.num_features = self.config['features']['num_features']
        self.confidence_threshold = self.config['inference']['confidence_threshold']
        self.smoothing_window = self.config['inference']['smoothing_window']
        self.prediction_interval = self.config['inference']['prediction_interval']
        self.show_landmarks = self.config['inference']['show_landmarks']
        self.show_fps = self.config['inference']['show_fps']
        
        # Class mapping
        self.idx_to_class = get_idx_to_class(self.config['data_collection']['classes'])
        
        # Buffers
        self.sequence_buffer = deque(maxlen=self.sequence_length)
        self.prediction_buffer = deque(maxlen=self.smoothing_window)
        
        # State
        self.frame_count = 0
        self.current_prediction = None
        self.current_confidence = 0.0
        self.paused = False
        
    def preprocess_sequence(self, sequence: np.ndarray) -> torch.Tensor:
        """
        Preprocess sequence for model input
        
        Args:
            sequence: Raw sequence of features
            
        Returns:
            Preprocessed tensor
        """
        # Reshape for scaling
        sequence_reshaped = sequence.reshape(-1, self.num_features)
        
        # Normalize
        sequence_normalized = self.scaler.transform(sequence_reshaped)
        
        # Reshape back
        sequence_normalized = sequence_normalized.reshape(1, self.sequence_length, self.num_features)
        
        # Convert to tensor
        tensor = torch.FloatTensor(sequence_normalized).to(self.device)
        
        return tensor
    
    def predict(self, sequence: np.ndarray) -> tuple:
        """
        Make prediction on sequence
        
        Args:
            sequence: Input sequence
            
        Returns:
            Tuple of (predicted_class, confidence)
        """
        # Preprocess
        input_tensor = self.preprocess_sequence(sequence)
        
        # Predict
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        predicted_class = self.idx_to_class[predicted_idx.item()]
        confidence_value = confidence.item()
        
        return predicted_class, confidence_value
    
    def get_smoothed_prediction(self) -> tuple:
        """
        Get smoothed prediction from buffer
        
        Returns:
            Tuple of (predicted_class, confidence)
        """
        if len(self.prediction_buffer) == 0:
            return None, 0.0
        
        # Get most common prediction
        predictions = [p[0] for p in self.prediction_buffer]
        confidences = [p[1] for p in self.prediction_buffer]
        
        # Majority voting
        from collections import Counter
        prediction_counts = Counter(predictions)
        most_common = prediction_counts.most_common(1)[0][0]
        
        # Average confidence for most common prediction
        avg_confidence = np.mean([c for p, c in self.prediction_buffer if p == most_common])
        
        return most_common, avg_confidence
    
    def draw_ui(self, frame: np.ndarray) -> np.ndarray:
        """Draw UI elements on frame"""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        
        # Top bar - Prediction
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        
        if self.current_prediction and self.current_confidence > self.confidence_threshold:
            # Prediction text
            pred_text = f"Prediction: {self.current_prediction}"
            cv2.putText(overlay, pred_text, (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            
            # Confidence bar
            conf_text = f"Confidence: {self.current_confidence:.2%}"
            cv2.putText(overlay, conf_text, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Confidence bar
            bar_width = int((w - 250) * self.current_confidence)
            cv2.rectangle(overlay, (240, 70), (240 + bar_width, 95), (0, 255, 0), -1)
            cv2.rectangle(overlay, (240, 70), (w - 10, 95), (255, 255, 255), 2)
        else:
            # No prediction
            no_pred_text = "No prediction (low confidence or no hand detected)"
            cv2.putText(overlay, no_pred_text, (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Bottom bar - Controls
        cv2.rectangle(overlay, (0, h - 80), (w, h), (0, 0, 0), -1)
        
        controls = "Controls: P - Pause/Resume | R - Reset | Q - Quit"
        cv2.putText(overlay, controls, (10, h - 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Buffer status
        buffer_text = f"Buffer: {len(self.sequence_buffer)}/{self.sequence_length}"
        cv2.putText(overlay, buffer_text, (10, h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # FPS
        if self.show_fps:
            fps_text = f"Device: {self.device}"
            cv2.putText(overlay, fps_text, (w - 200, h - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Pause indicator
        if self.paused:
            pause_text = "PAUSED"
            text_size = cv2.getTextSize(pause_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h // 2
            cv2.putText(overlay, pause_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
        
        # Blend overlay
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        return frame
    
    def run(self):
        """Main inference loop"""
        cap = cv2.VideoCapture(0)
        
        print("\n" + "="*60)
        print("ASL FINGERSPELLING REAL-TIME INFERENCE")
        print("="*60)
        print("\nControls:")
        print("  P: Pause/Resume")
        print("  R: Reset buffers")
        print("  Q: Quit")
        print("\n" + "="*60 + "\n")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                if not self.paused:
                    # Extract features
                    features, results = self.extractor.extract_landmarks(frame)
                    
                    # Draw landmarks
                    if self.show_landmarks:
                        frame = self.extractor.draw_landmarks(frame, results)
                    
                    # Add to buffer
                    self.sequence_buffer.append(features)
                    
                    # Make prediction
                    if len(self.sequence_buffer) == self.sequence_length:
                        if self.frame_count % self.prediction_interval == 0:
                            # Convert buffer to array
                            sequence = np.array(self.sequence_buffer)
                            
                            # Predict
                            pred_class, confidence = self.predict(sequence)
                            
                            # Add to prediction buffer
                            self.prediction_buffer.append((pred_class, confidence))
                            
                            # Get smoothed prediction
                            self.current_prediction, self.current_confidence = \
                                self.get_smoothed_prediction()
                
                # Draw UI
                frame = self.draw_ui(frame)
                
                # Display
                cv2.imshow('ASL Fingerspelling Inference', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Reset buffers
                    self.sequence_buffer.clear()
                    self.prediction_buffer.clear()
                    self.current_prediction = None
                    self.current_confidence = 0.0
                    print("Buffers reset")
                elif key == ord('p'):
                    # Toggle pause
                    self.paused = not self.paused
                    if self.paused:
                        print("⏸ Paused")
                    else:
                        print("▶ Resumed")
                
                self.frame_count += 1
                
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.extractor.close()
            print("\nInference stopped")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ASL Fingerspelling Real-time Inference')
    parser.add_argument('--model', type=str, 
                       default='../models/checkpoints/best_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str,
                       default='../config/config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    inference = ASLInference(args.model, args.config)
    inference.run()
