"""
Bulk data collection tool for ASL fingerspelling
Continuously captures sequences while you move your hand around
MUCH faster than individual recording!
"""
import cv2
import numpy as np
import os
import sys
from pathlib import Path
from feature_extraction import HandLandmarkExtractor
from utils import load_config, ensure_dir, count_samples_per_class

sys.path.append(str(Path(__file__).parent.parent))


class BulkDataCollector:
    """Bulk data collection - capture many samples quickly"""
    
    def __init__(self, config_path: str = "../config/config.yaml"):
        """
        Initialize bulk data collector
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.extractor = HandLandmarkExtractor()
        
        # Configuration
        self.classes = self.config['data_collection']['classes']
        self.sequence_length = self.config['data_collection']['sequence_length']
        self.fps = self.config['data_collection']['fps']
        
        # Bulk collection settings
        self.capture_interval = 5  # Capture every 5 frames (6 samples/second)
        self.target_samples = 150  # Target samples per class
        
        # Data directory
        self.data_dir = Path(__file__).parent.parent / "data" / "raw"
        ensure_dir(str(self.data_dir))
        
        # State
        self.current_class_idx = 0
        self.recording = False
        self.sequence_buffer = []
        self.paused = False
        self.samples_collected_this_session = 0
        
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
        self.samples_collected_this_session += 1
    
    def draw_ui(self, frame: np.ndarray) -> np.ndarray:
        """Draw UI elements on frame"""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        
        # Top bar - Current class and instructions
        cv2.rectangle(overlay, (0, 0), (w, 140), (0, 0, 0), -1)
        
        current_class = self.get_current_class()
        class_dir = self.get_class_dir(current_class)
        num_samples = len(list(class_dir.glob("*.npy")))
        
        # Class info
        class_text = f"Class: {current_class} ({self.current_class_idx + 1}/{len(self.classes)})"
        cv2.putText(overlay, class_text, (10, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Sample count
        sample_text = f"Total: {num_samples} | This session: {self.samples_collected_this_session}"
        cv2.putText(overlay, sample_text, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Progress bar
        progress = min(num_samples / self.target_samples, 1.0)
        bar_width = int((w - 20) * progress)
        cv2.rectangle(overlay, (10, 85), (10 + bar_width, 105), (0, 255, 0), -1)
        cv2.rectangle(overlay, (10, 85), (w - 10, 105), (255, 255, 255), 2)
        
        # Progress percentage
        progress_text = f"{int(progress * 100)}% ({num_samples}/{self.target_samples})"
        cv2.putText(overlay, progress_text, (10, 125), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Bottom bar - Instructions
        cv2.rectangle(overlay, (0, h - 150), (w, h), (0, 0, 0), -1)
        
        if not self.recording:
            instructions = [
                "BULK COLLECTION MODE - Fast Data Collection!",
                "",
                "SPACE: Start recording (move hand around while holding gesture)",
                "LEFT/RIGHT: Change class | P: Pause | Q: Quit"
            ]
        else:
            instructions = [
                "RECORDING - Keep moving your hand around!",
                "",
                f"Auto-capturing every {self.capture_interval} frames (~6 samples/sec)",
                "SPACE: Stop recording | Buffer: " + str(len(self.sequence_buffer))
            ]
        
        y_offset = h - 130
        for i, instruction in enumerate(instructions):
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            size = 0.7 if i == 0 else 0.5
            cv2.putText(overlay, instruction, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, size, color, 1 if i == 0 else 1)
            y_offset += 30
        
        # Recording indicator
        if self.recording:
            # Recording circle (pulsing)
            import time
            pulse = int(abs(np.sin(time.time() * 5) * 20))
            cv2.circle(overlay, (w - 50, 50), 20 + pulse, (0, 0, 255), -1)
            
            # Recording text
            rec_text = "REC"
            cv2.putText(overlay, rec_text, (w - 90, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
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
        """Main bulk collection loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        print("\n" + "="*70)
        print("ASL FINGERSPELLING BULK DATA COLLECTION")
        print("="*70)
        print("\n🚀 FAST MODE: Collect 100+ samples in under a minute!")
        print("\nHow it works:")
        print("  1. Select your class (letter/digit)")
        print("  2. Press SPACE to start recording")
        print("  3. Hold the gesture and MOVE YOUR HAND AROUND")
        print("     - Different positions")
        print("     - Different angles")
        print("     - Different distances from camera")
        print("  4. System auto-captures ~6 samples per second")
        print("  5. Press SPACE again to stop")
        print("\nControls:")
        print("  SPACE: Start/Stop recording")
        print("  LEFT/RIGHT ARROW: Change class")
        print("  P: Pause/Resume")
        print("  Q: Quit")
        print("\n" + "="*70 + "\n")
        
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
                    
                    # Add to buffer
                    self.sequence_buffer.append(features)
                    
                    # Maintain buffer at sequence_length
                    if len(self.sequence_buffer) > self.sequence_length:
                        self.sequence_buffer.pop(0)
                    
                    # Auto-capture during recording
                    if self.recording and len(self.sequence_buffer) == self.sequence_length:
                        if frame_count % self.capture_interval == 0:
                            # Save current sequence
                            sequence = np.array(self.sequence_buffer)
                            self.save_sequence(sequence, self.get_current_class())
                
                # Draw UI
                frame = self.draw_ui(frame)
                
                # Display
                cv2.imshow('ASL Bulk Data Collection', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    # Toggle recording
                    self.recording = not self.recording
                    if self.recording:
                        self.samples_collected_this_session = 0
                        print(f"\n🔴 Recording {self.get_current_class()}... Move your hand around!")
                    else:
                        print(f"⏹ Stopped. Collected {self.samples_collected_this_session} samples this session")
                elif key == 81 or key == 2:  # Left arrow
                    if not self.recording:
                        self.current_class_idx = (self.current_class_idx - 1) % len(self.classes)
                        self.samples_collected_this_session = 0
                        print(f"Class: {self.get_current_class()}")
                elif key == 83 or key == 3:  # Right arrow
                    if not self.recording:
                        self.current_class_idx = (self.current_class_idx + 1) % len(self.classes)
                        self.samples_collected_this_session = 0
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
            print("\n" + "="*70)
            print("BULK DATA COLLECTION COMPLETE")
            print("="*70)
            counts = count_samples_per_class(str(self.data_dir), self.classes)
            total = sum(counts.values())
            print(f"\nTotal samples collected: {total}")
            print(f"Average per class: {total/len(self.classes):.1f}")
            
            # Show classes that need more data
            print("\nClasses needing more data:")
            for cls in self.classes:
                count = counts.get(cls, 0)
                if count < self.target_samples:
                    needed = self.target_samples - count
                    print(f"  {cls}: {count}/{self.target_samples} ({needed} more needed)")
            
            print("="*70 + "\n")


if __name__ == "__main__":
    collector = BulkDataCollector()
    collector.run()
