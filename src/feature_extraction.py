"""
Feature extraction using MediaPipe Hands (Tasks API - MediaPipe 0.10.31)
Extracts hand landmarks from images/video frames
"""
import cv2
import numpy as np
from typing import Tuple, Optional
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path


class HandLandmarkExtractor:
    """Extract hand landmarks using MediaPipe Tasks API"""
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe Hands
        
        Args:
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        # Path to the hand landmarker model
        model_path = Path(__file__).parent.parent / "models" / "mediapipe" / "hand_landmarker.task"
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Hand landmarker model not found at {model_path}\\n"
                "Please download it from: "
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )
        
        # Create hand landmarker
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=vision.RunningMode.VIDEO
        )
        
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.frame_timestamp_ms = 0
        
    def extract_landmarks(self, image: np.ndarray) -> Tuple[np.ndarray, any]:
        """
        Extract hand landmarks from image
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            Tuple of (features_array, results_object)
            - features_array: Shape (258,) containing all features
            - results_object: MediaPipe results for visualization
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Process image
        self.frame_timestamp_ms += 33  # ~30 FPS
        results = self.landmarker.detect_for_video(mp_image, self.frame_timestamp_ms)
        
        # Initialize feature array (258 features)
        features = np.zeros(258)
        
        if results.hand_landmarks:
            # Track which hand is which (left/right)
            left_hand_landmarks = None
            right_hand_landmarks = None
            
            for idx, hand_landmarks in enumerate(results.hand_landmarks):
                # Get hand label (Left or Right)
                if idx < len(results.handedness):
                    hand_label = results.handedness[idx][0].category_name
                    
                    # Extract landmarks (21 landmarks with x, y, z)
                    landmarks_array = np.array([[lm.x, lm.y, lm.z] 
                                               for lm in hand_landmarks])
                    
                    if hand_label == "Left":
                        left_hand_landmarks = landmarks_array
                    else:
                        right_hand_landmarks = landmarks_array
            
            # Fill feature array
            # Left hand: features[0:63]
            if left_hand_landmarks is not None:
                features[0:63] = left_hand_landmarks.flatten()
                features[126] = 1.0  # Left hand presence flag
            
            # Right hand: features[63:126]
            if right_hand_landmarks is not None:
                features[63:126] = right_hand_landmarks.flatten()
                features[127] = 1.0  # Right hand presence flag
        
        return features, results
    
    def draw_landmarks(self, image: np.ndarray, results) -> np.ndarray:
        """
        Draw hand landmarks on image
        
        Args:
            image: Input image
            results: MediaPipe results object
            
        Returns:
            Image with landmarks drawn
        """
        if results and results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                # Draw landmarks
                for landmark in hand_landmarks:
                    x = int(landmark.x * image.shape[1])
                    y = int(landmark.y * image.shape[0])
                    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
                
                # Draw connections
                connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
                    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
                    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
                    (5, 9), (9, 13), (13, 17)  # Palm
                ]
                
                for connection in connections:
                    start_idx, end_idx = connection
                    start = hand_landmarks[start_idx]
                    end = hand_landmarks[end_idx]
                    
                    start_point = (int(start.x * image.shape[1]), int(start.y * image.shape[0]))
                    end_point = (int(end.x * image.shape[1]), int(end.y * image.shape[0]))
                    
                    cv2.line(image, start_point, end_point, (255, 255, 255), 2)
        
        return image
    
    def close(self):
        """Release resources"""
        self.landmarker.close()


# Test the module
if __name__ == "__main__":
    print("Testing Hand Landmark Extractor...")
    try:
        extractor = HandLandmarkExtractor()
        print("✓ Hand landmarker initialized successfully!")
        print("✓ Model loaded from models/mediapipe/hand_landmarker.task")
        extractor.close()
    except Exception as e:
        print(f"✗ Error: {e}")
