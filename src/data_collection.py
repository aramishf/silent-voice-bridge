"""
Interactive data collection tool for ASL fingerspelling
Allows recording sequences of hand gestures for training
"""
import cv2
import numpy as np
import os
import sys
from pathlib import Path
from feature_extraction import HandLandmarkExtractor
from utils import load_config, ensure_dir, count_samples_per_class

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


class DataCollector:
    """Interactive data collection for ASL gestures"""
    
    def __init__(self, config_path: str = "../config/config.yaml"):
        """
        Initialize data collector
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.extractor = HandLandmarkExtractor()
        
        # Configuration
        self.classes = self.config['data_collection']['classes']
        self.sequence_length = self.config['data_collection']['sequence_length']
        self.fps = self.config['data_collection']['fps']
        self.countdown_seconds = self.config['data_collection']['countdown_seconds']
        self.samples_per_class = self.config['data_collection']['samples_per_class']
        
        # Data directory
        self.data_dir = Path(__file__).parent.parent / "data" / "raw"
        ensure_dir(str(self.data_dir))
        
        # State
        self.current_class_idx = 0
        self.recording = False
        self.countdown = 0
        self.sequence_buffer = []
        self.paused = False
        
    def get_current_class(self) -> str:
        """Get current class name"""
        return self.classes[self.current_class_idx]
    
    def get_class_dir(self, class_name: str) -> Path:
        """Get directory for specific class"""
        class_dir = self.data_dir / class_name
        ensure_dir(str(class_dir))
        return class_dir
    
    def get_next_filename(self, class_name: str) -> str:
        """Get next available filename for class"""
        class_dir = self.get_class_dir(class_name)
        existing_files = list(class_dir.glob("*.npy"))
        next_idx = len(existing_files)
        return str(class_dir / f"{class_name}_{next_idx:04d}.npy")
    
    def save_sequence(self, sequence: np.ndarray, class_name: str):
        """Save sequence to disk"""
        filename = self.get_next_filename(class_name)
        np.save(filename, sequence)
        print(f"✓ Saved: {filename}")
    
    def draw_ui(self, frame: np.ndarray) -> np.ndarray:
        """Draw UI elements on frame"""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        
        # Top bar - Current class and instructions
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        
        current_class = self.get_current_class()
        class_dir = self.get_class_dir(current_class)
        num_samples = len(list(class_dir.glob("*.npy")))
        
        # Class info
        class_text = f"Class: {current_class} ({self.current_class_idx + 1}/{len(self.classes)})"
        cv2.putText(overlay, class_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Sample count
        sample_text = f"Samples: {num_samples}/{self.samples_per_class}"
        cv2.putText(overlay, sample_text, (10, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Progress bar
        progress = min(num_samples / self.samples_per_class, 1.0)
        bar_width = int((w - 20) * progress)
        cv2.rectangle(overlay, (10, 75), (10 + bar_width, 90), (0, 255, 0), -1)
        cv2.rectangle(overlay, (10, 75), (w - 10, 90), (255, 255, 255), 2)
        
        # Bottom bar - Controls
        cv2.rectangle(overlay, (0, h - 120), (w, h), (0, 0, 0), -1)
        
        controls = [
            "SPACE: Start Recording",
            "LEFT/RIGHT: Change Class",
            "P: Pause/Resume",
            "Q: Quit"
        ]
        
        y_offset = h - 100
        for control in controls:
            cv2.putText(overlay, control, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
        
        # Recording indicator
        if self.recording:
            if self.countdown > 0:
                # Countdown
                countdown_text = str(self.countdown)
                text_size = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX, 
                                           4, 5)[0]
                text_x = (w - text_size[0]) // 2
                text_y = (h + text_size[1]) // 2
                cv2.putText(overlay, countdown_text, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 5)
            else:
                # Recording in progress
                rec_text = "RECORDING"
                frames_left = self.sequence_length - len(self.sequence_buffer)
                rec_info = f"{rec_text} - {frames_left} frames left"
                cv2.putText(overlay, rec_info, (w // 2 - 200, h // 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Recording circle
                cv2.circle(overlay, (w - 50, 50), 20, (0, 0, 255), -1)
        
        # Pause indicator
        if self.paused:
            pause_text = "PAUSED"
            text_size = cv2.getTextSize(pause_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
            text_x = (w - text_size[0]) // 2
            text_y = 50
            cv2.putText(overlay, pause_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
        
        # Blend overlay
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        return frame
    
    def run(self):
        """Main data collection loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        print("\n" + "="*60)
        print("ASL FINGERSPELLING DATA COLLECTION")
        print("="*60)
        print("\nControls:")
        print("  SPACE: Start recording sequence")
        print("  LEFT/RIGHT ARROW: Change class")
        print("  P: Pause/Resume")
        print("  Q: Quit")
        print("\n" + "="*60 + "\n")
        
        frame_count = 0
        
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
                    frame = self.extractor.draw_landmarks(frame, results)
                    
                    # Handle recording
                    if self.recording:
                        if self.countdown > 0:
                            # Countdown phase
                            if frame_count % self.fps == 0:
                                self.countdown -= 1
                        else:
                            # Recording phase
                            self.sequence_buffer.append(features)
                            
                            if len(self.sequence_buffer) >= self.sequence_length:
                                # Save sequence
                                sequence = np.array(self.sequence_buffer)
                                self.save_sequence(sequence, self.get_current_class())
                                
                                # Reset
                                self.recording = False
                                self.sequence_buffer = []
                
                # Draw UI
                frame = self.draw_ui(frame)
                
                # Display
                cv2.imshow('ASL Data Collection', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' ') and not self.recording and not self.paused:
                    # Start recording
                    self.recording = True
                    self.countdown = self.countdown_seconds
                    self.sequence_buffer = []
                    print(f"Recording {self.get_current_class()}...")
                elif key == 81 or key == 2:  # Left arrow
                    if not self.recording:
                        self.current_class_idx = (self.current_class_idx - 1) % len(self.classes)
                        print(f"Class: {self.get_current_class()}")
                elif key == 83 or key == 3:  # Right arrow
                    if not self.recording:
                        self.current_class_idx = (self.current_class_idx + 1) % len(self.classes)
                        print(f"Class: {self.get_current_class()}")
                elif key == ord('p'):
                    # Toggle pause
                    self.paused = not self.paused
                    if self.paused:
                        print("⏸ Paused")
                    else:
                        print("▶ Resumed")
                
                frame_count += 1
                
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.extractor.close()
            
            # Print summary
            print("\n" + "="*60)
            print("DATA COLLECTION COMPLETE")
            print("="*60)
            counts = count_samples_per_class(str(self.data_dir), self.classes)
            total = sum(counts.values())
            print(f"\nTotal samples collected: {total}")
            print(f"Average per class: {total/len(self.classes):.1f}")
            print("="*60 + "\n")


if __name__ == "__main__":
    collector = DataCollector()
    collector.run()
