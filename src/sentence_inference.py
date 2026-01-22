"""
Enhanced real-time inference with sentence formation
Features:
- Deduplication (H, H, H → H)
- Auto-correction (HELPO → HELLO)
- Split-screen UI (video left, text right)
- Sentence history
"""
import cv2
import torch
import numpy as np
import pickle
from pathlib import Path
from collections import deque
import sys
from datetime import datetime

from feature_extraction import HandLandmarkExtractor
from model import ASLFingerSpellingLSTM
from utils import load_config, get_idx_to_class

sys.path.append(str(Path(__file__).parent.parent))


class SentenceBuilder:
    """Build sentences from detected letters with deduplication and auto-correction"""
    
    def __init__(self):
        self.current_sentence = []
        self.sentence_history = []
        self.last_letter = None
        self.last_letter_count = 0
        self.last_added_letter = None  # Track what was actually added
        self.min_repeats_for_add = 3  # Need 3 consecutive detections to add letter
        self.release_detected = True  # Start ready to add
        self.waiting_for_release = False
        
        # Common word corrections
        self.corrections = {
            'HELPO': 'HELLO',
            'WROLD': 'WORLD',
            'THNAK': 'THANK',
            'YUOR': 'YOUR',
            'TAHT': 'THAT',
            'WAHT': 'WHAT',
            'HELO': 'HELLO',
            'WOLRD': 'WORLD',
        }
    
    def add_detection(self, letter: str, confidence: float, threshold: float = 0.8):
        """Add a detected letter with deduplication, release detection, and smart disambiguation"""
        if confidence < threshold:
            # Low confidence = release detected
            if self.waiting_for_release:
                self.release_detected = True
                self.waiting_for_release = False
            self.last_letter_count = 0
            return False
        
        # Smart disambiguation based on context
        letter = self._disambiguate(letter)
        
        # Check if this is a different letter than last detected
        if letter != self.last_letter:
            # Different letter = release detected
            if self.waiting_for_release and self.last_letter == self.last_added_letter:
                self.release_detected = True
                self.waiting_for_release = False
            self.last_letter = letter
            self.last_letter_count = 1
        else:
            # Same letter as before
            self.last_letter_count += 1
        
        # Check if we should add this letter
        if self.last_letter_count == self.min_repeats_for_add:
            # If this is the same letter we just added, check if released first
            if letter == self.last_added_letter:
                if self.release_detected:
                    # Released and re-held - add it again!
                    self.current_sentence.append(letter)
                    self.last_added_letter = letter
                    self.release_detected = False
                    self.waiting_for_release = True
                    return True
                else:
                    # Still holding - don't add again
                    self.waiting_for_release = True
                    return False
            else:
                # Different letter - add it
                self.current_sentence.append(letter)
                self.last_added_letter = letter
                self.release_detected = False
                self.waiting_for_release = True
                return True
        
        return False
    
    def _disambiguate(self, letter: str) -> str:
        """
        Smart disambiguation between similar letters and digits
        Based on context of current word being formed
        
        Similar pairs:
        - O/0 (letter O vs digit zero)
        - B/4 (letter B vs digit four)
        - D/1 (letter D vs digit one)
        - F/9 (letter F vs digit nine)
        - V/2 (letter V vs digit two)
        - W/6 (letter W vs digit six)
        """
        # Get current word context
        current_word = self.get_current_word()
        
        # If word is empty, can't determine context yet
        if not current_word:
            return letter
        
        # Check if current word contains letters or digits
        has_letters = any(c.isalpha() for c in current_word)
        has_digits = any(c.isdigit() for c in current_word)
        
        # Disambiguation rules
        disambiguations = {
            # If in word context (has letters), prefer letters
            '0': 'O' if has_letters and not has_digits else '0',
            '4': 'B' if has_letters and not has_digits else '4',  # Actually B looks like 8, but keeping your mapping
            '1': 'D' if has_letters and not has_digits else '1',
            '9': 'F' if has_letters and not has_digits else '9',
            '2': 'V' if has_letters and not has_digits else '2',
            '6': 'W' if has_letters and not has_digits else '6',
            
            # If in number context (has digits), prefer digits
            'O': '0' if has_digits and not has_letters else 'O',
            'B': '4' if has_digits and not has_letters else 'B',
            'D': '1' if has_digits and not has_letters else 'D',
            'F': '9' if has_digits and not has_letters else 'F',
            'V': '2' if has_digits and not has_letters else 'V',
            'W': '6' if has_digits and not has_letters else 'W',
        }
        
        return disambiguations.get(letter, letter)
    
    def get_status_message(self):
        """Get status message for user guidance"""
        if self.waiting_for_release and self.last_added_letter:
            return f"Release hand to add another '{self.last_added_letter}'"
        return None
    
    def add_space(self):
        """Add a space to separate words"""
        if self.current_sentence and self.current_sentence[-1] != ' ':
            self.current_sentence.append(' ')
    
    def backspace(self):
        """Remove last character"""
        if self.current_sentence:
            self.current_sentence.pop()
    
    def get_current_word(self):
        """Get the current word being formed"""
        sentence_str = ''.join(self.current_sentence)
        words = sentence_str.split(' ')
        return words[-1] if words else ''
    
    def auto_correct_current_word(self):
        """Auto-correct the current word if it matches a known mistake"""
        current_word = self.get_current_word()
        
        if current_word in self.corrections:
            # Remove current word
            word_len = len(current_word)
            for _ in range(word_len):
                if self.current_sentence:
                    self.current_sentence.pop()
            
            # Add corrected word
            corrected = self.corrections[current_word]
            self.current_sentence.extend(list(corrected))
            return corrected
        
        return None
    
    def finish_sentence(self):
        """Finish current sentence and add to history"""
        if self.current_sentence:
            sentence = ''.join(self.current_sentence).strip()
            if sentence:
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.sentence_history.append((timestamp, sentence))
                self.current_sentence = []
                self.last_letter = None
                self.last_letter_count = 0
    
    def get_current_sentence(self):
        """Get current sentence being formed"""
        return ''.join(self.current_sentence)
    
    def clear(self):
        """Clear current sentence"""
        self.current_sentence = []
        self.last_letter = None
        self.last_letter_count = 0


class EnhancedASLInference:
    """Enhanced ASL inference with sentence formation and split-screen UI"""
    
    def __init__(self, model_path: str, config_path: str = "../config/config.yaml"):
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
        
        # Class mapping
        self.idx_to_class = get_idx_to_class(self.config['data_collection']['classes'])
        
        # Buffers
        self.sequence_buffer = deque(maxlen=self.sequence_length)
        self.prediction_buffer = deque(maxlen=self.smoothing_window)
        
        # Sentence builder
        self.sentence_builder = SentenceBuilder()
        
        # State
        self.frame_count = 0
        self.paused = False
        
        # UI dimensions
        self.video_width = 640
        self.video_height = 480
        self.text_panel_width = 400
        self.window_width = self.video_width + self.text_panel_width
        self.window_height = self.video_height
    
    def preprocess_sequence(self, sequence: np.ndarray) -> torch.Tensor:
        """Preprocess sequence for model input"""
        sequence_reshaped = sequence.reshape(-1, self.num_features)
        sequence_normalized = self.scaler.transform(sequence_reshaped)
        sequence_normalized = sequence_normalized.reshape(1, self.sequence_length, self.num_features)
        tensor = torch.FloatTensor(sequence_normalized).to(self.device)
        return tensor
    
    def predict(self, sequence: np.ndarray) -> tuple:
        """Make prediction on sequence"""
        input_tensor = self.preprocess_sequence(sequence)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        predicted_class = self.idx_to_class[predicted_idx.item()]
        confidence_value = confidence.item()
        
        return predicted_class, confidence_value
    
    def get_smoothed_prediction(self) -> tuple:
        """Get smoothed prediction from buffer"""
        if len(self.prediction_buffer) == 0:
            return None, 0.0
        
        from collections import Counter
        predictions = [p[0] for p in self.prediction_buffer]
        confidences = [p[1] for p in self.prediction_buffer]
        
        prediction_counts = Counter(predictions)
        most_common = prediction_counts.most_common(1)[0][0]
        avg_confidence = np.mean([c for p, c in self.prediction_buffer if p == most_common])
        
        return most_common, avg_confidence
    
    def draw_split_screen_ui(self, video_frame: np.ndarray, prediction: str, confidence: float):
        """Draw split-screen UI with video on left and text on right"""
        # Resize video frame
        video_frame = cv2.resize(video_frame, (self.video_width, self.video_height))
        
        # Create text panel (white background)
        text_panel = np.ones((self.window_height, self.text_panel_width, 3), dtype=np.uint8) * 255
        
        # Draw current sentence section
        cv2.rectangle(text_panel, (0, 0), (self.text_panel_width, 150), (240, 240, 240), -1)
        cv2.putText(text_panel, "Current Sentence:", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Current sentence (word wrap)
        current = self.sentence_builder.get_current_sentence()
        y_offset = 55
        max_chars = 25
        for i in range(0, len(current), max_chars):
            line = current[i:i+max_chars]
            cv2.putText(text_panel, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 0), 2)
            y_offset += 30
        
        # Current prediction
        if prediction and confidence > self.confidence_threshold:
            pred_text = f"Detecting: {prediction} ({confidence:.0%})"
            cv2.putText(text_panel, pred_text, (10, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Status message (e.g., "Release hand to add another L")
        status_msg = self.sentence_builder.get_status_message()
        if status_msg:
            cv2.putText(text_panel, status_msg, (10, 145),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        
        # History section
        cv2.line(text_panel, (0, 150), (self.text_panel_width, 150), (200, 200, 200), 2)
        cv2.putText(text_panel, "History:", (10, 175),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Show last 10 sentences
        y_offset = 200
        for timestamp, sentence in self.sentence_builder.sentence_history[-10:]:
            # Timestamp
            cv2.putText(text_panel, timestamp, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
            y_offset += 20
            
            # Sentence (word wrap)
            for i in range(0, len(sentence), max_chars):
                line = sentence[i:i+max_chars]
                cv2.putText(text_panel, line, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                y_offset += 20
            y_offset += 5
        
        # Controls at bottom
        controls_y = self.window_height - 80
        cv2.rectangle(text_panel, (0, controls_y), (self.text_panel_width, self.window_height),
                     (240, 240, 240), -1)
        
        controls = [
            "SPACE: Add space",
            "BACKSPACE: Delete",
            "ENTER: Finish sentence",
            "C: Clear current",
            "P: Pause | Q: Quit"
        ]
        
        y = controls_y + 15
        for control in controls:
            cv2.putText(text_panel, control, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
            y += 15
        
        # Combine video and text panel
        combined = np.hstack([video_frame, text_panel])
        
        # Pause indicator
        if self.paused:
            cv2.rectangle(combined, (self.window_width//2 - 100, self.window_height//2 - 30),
                         (self.window_width//2 + 100, self.window_height//2 + 30), (0, 0, 0), -1)
            cv2.putText(combined, "PAUSED", (self.window_width//2 - 80, self.window_height//2 + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        
        return combined
    
    def run(self):
        """Main inference loop"""
        cap = cv2.VideoCapture(0)
        
        print("\n" + "="*70)
        print("ASL FINGERSPELLING - SENTENCE FORMATION MODE")
        print("="*70)
        print("\nFeatures:")
        print("  ✓ Automatic deduplication (H,H,H → H)")
        print("  ✓ Auto-correction for common mistakes")
        print("  ✓ Split-screen UI (video + text)")
        print("\nControls:")
        print("  SPACE: Add space between words")
        print("  BACKSPACE: Delete last character")
        print("  ENTER: Finish sentence (add to history)")
        print("  C: Clear current sentence")
        print("  P: Pause/Resume")
        print("  Q: Quit")
        print("\n" + "="*70 + "\n")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                current_prediction = None
                current_confidence = 0.0
                
                if not self.paused:
                    # Extract features
                    features, results = self.extractor.extract_landmarks(frame)
                    frame = self.extractor.draw_landmarks(frame, results)
                    
                    # Add to buffer
                    self.sequence_buffer.append(features)
                    
                    # Make prediction
                    if len(self.sequence_buffer) == self.sequence_length:
                        if self.frame_count % 10 == 0:
                            sequence = np.array(self.sequence_buffer)
                            pred_class, confidence = self.predict(sequence)
                            self.prediction_buffer.append((pred_class, confidence))
                            
                            # Get smoothed prediction
                            current_prediction, current_confidence = self.get_smoothed_prediction()
                            
                            # Add to sentence with deduplication
                            if current_prediction:
                                added = self.sentence_builder.add_detection(
                                    current_prediction, current_confidence
                                )
                                
                                # Auto-correct after adding letter
                                if added:
                                    corrected = self.sentence_builder.auto_correct_current_word()
                                    if corrected:
                                        print(f"Auto-corrected to: {corrected}")
                
                # Draw UI
                display_frame = self.draw_split_screen_ui(frame, current_prediction, current_confidence)
                cv2.imshow('ASL Sentence Formation', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    self.sentence_builder.add_space()
                elif key == 8 or key == 127:  # Backspace
                    self.sentence_builder.backspace()
                elif key == 13:  # Enter
                    self.sentence_builder.finish_sentence()
                elif key == ord('c'):
                    self.sentence_builder.clear()
                elif key == ord('p'):
                    self.paused = not self.paused
                
                self.frame_count += 1
                
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.extractor.close()
            print("\nInference stopped")
            
            # Print final history
            if self.sentence_builder.sentence_history:
                print("\n" + "="*70)
                print("SENTENCE HISTORY")
                print("="*70)
                for timestamp, sentence in self.sentence_builder.sentence_history:
                    print(f"[{timestamp}] {sentence}")
                print("="*70 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ASL Sentence Formation')
    parser.add_argument('--model', type=str,
                       default='../models/checkpoints/best_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str,
                       default='../config/config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    inference = EnhancedASLInference(args.model, args.config)
    inference.run()
