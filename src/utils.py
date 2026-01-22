"""
Utility functions for ASL Fingerspelling Interpreter
"""
import os
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


def load_config(config_path: str = "../config/config.yaml") -> Dict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).parent.parent


def ensure_dir(directory: str) -> None:
    """
    Create directory if it doesn't exist
    
    Args:
        directory: Path to directory
    """
    os.makedirs(directory, exist_ok=True)


def get_class_to_idx(classes: List[str]) -> Dict[str, int]:
    """
    Create mapping from class names to indices
    
    Args:
        classes: List of class names
        
    Returns:
        Dictionary mapping class names to indices
    """
    return {cls: idx for idx, cls in enumerate(classes)}


def get_idx_to_class(classes: List[str]) -> Dict[int, str]:
    """
    Create mapping from indices to class names
    
    Args:
        classes: List of class names
        
    Returns:
        Dictionary mapping indices to class names
    """
    return {idx: cls for idx, cls in enumerate(classes)}


def count_samples_per_class(data_dir: str, classes: List[str]) -> Dict[str, int]:
    """
    Count number of samples collected for each class
    
    Args:
        data_dir: Directory containing raw data
        classes: List of class names
        
    Returns:
        Dictionary with sample counts per class
    """
    counts = {}
    for cls in classes:
        class_dir = os.path.join(data_dir, cls)
        if os.path.exists(class_dir):
            counts[cls] = len([f for f in os.listdir(class_dir) if f.endswith('.npy')])
        else:
            counts[cls] = 0
    return counts


def print_data_summary(data_dir: str, classes: List[str]) -> None:
    """
    Print summary of collected data
    
    Args:
        data_dir: Directory containing raw data
        classes: List of class names
    """
    counts = count_samples_per_class(data_dir, classes)
    total = sum(counts.values())
    
    print("\n" + "="*50)
    print("DATA COLLECTION SUMMARY")
    print("="*50)
    print(f"Total samples: {total}")
    print(f"Total classes: {len(classes)}")
    print(f"Average per class: {total/len(classes):.1f}")
    print("\nPer-class breakdown:")
    print("-"*50)
    
    # Print in groups (letters and digits)
    print("\nLetters (A-Z):")
    for cls in classes[:26]:
        count = counts.get(cls, 0)
        bar = "█" * (count // 5)
        print(f"  {cls}: {count:3d} {bar}")
    
    print("\nDigits (0-9):")
    for cls in classes[26:]:
        count = counts.get(cls, 0)
        bar = "█" * (count // 5)
        print(f"  {cls}: {count:3d} {bar}")
    
    print("="*50 + "\n")


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize landmarks relative to wrist position
    
    Args:
        landmarks: Array of shape (21, 3) containing hand landmarks
        
    Returns:
        Normalized landmarks
    """
    if landmarks.size == 0:
        return landmarks
    
    # Wrist is landmark 0
    wrist = landmarks[0].copy()
    normalized = landmarks - wrist
    
    return normalized


def pad_sequence(sequence: np.ndarray, target_length: int) -> np.ndarray:
    """
    Pad or truncate sequence to target length
    
    Args:
        sequence: Input sequence of shape (seq_len, features)
        target_length: Desired sequence length
        
    Returns:
        Padded/truncated sequence of shape (target_length, features)
    """
    current_length = sequence.shape[0]
    
    if current_length == target_length:
        return sequence
    elif current_length > target_length:
        # Truncate
        return sequence[:target_length]
    else:
        # Pad with zeros
        padding = np.zeros((target_length - current_length, sequence.shape[1]))
        return np.vstack([sequence, padding])
