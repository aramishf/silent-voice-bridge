"""
Data preprocessing pipeline
Loads raw sequences, normalizes, and creates train/val/test splits
"""
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from typing import Tuple, List
from utils import load_config, ensure_dir, pad_sequence


class DataPreprocessor:
    """Preprocess raw sequences for model training"""
    
    def __init__(self, config_path: str = "../config/config.yaml"):
        """
        Initialize preprocessor
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.classes = self.config['data_collection']['classes']
        self.sequence_length = self.config['data_collection']['sequence_length']
        self.num_features = self.config['features']['num_features']
        
        # Directories
        self.raw_dir = Path(__file__).parent.parent / "data" / "raw"
        self.processed_dir = Path(__file__).parent.parent / "data" / "processed"
        ensure_dir(str(self.processed_dir))
        
        # Scaler
        self.scaler = StandardScaler()
        
    def load_raw_data(self) -> Tuple[List[np.ndarray], List[int]]:
        """
        Load all raw sequences from disk
        
        Returns:
            Tuple of (sequences, labels)
        """
        sequences = []
        labels = []
        
        print("Loading raw data...")
        for class_idx, class_name in enumerate(self.classes):
            class_dir = self.raw_dir / class_name
            
            if not class_dir.exists():
                print(f"Warning: No data found for class {class_name}")
                continue
            
            # Load all .npy files for this class
            npy_files = list(class_dir.glob("*.npy"))
            
            for npy_file in npy_files:
                try:
                    sequence = np.load(str(npy_file))
                    
                    # Validate shape
                    if sequence.shape[1] != self.num_features:
                        print(f"Warning: Skipping {npy_file} - wrong feature dimension")
                        continue
                    
                    # Pad/truncate to target length
                    sequence = pad_sequence(sequence, self.sequence_length)
                    
                    sequences.append(sequence)
                    labels.append(class_idx)
                    
                except Exception as e:
                    print(f"Error loading {npy_file}: {e}")
            
            print(f"  {class_name}: {len(npy_files)} samples")
        
        print(f"\nTotal samples loaded: {len(sequences)}")
        return sequences, labels
    
    def normalize_data(self, sequences: List[np.ndarray], 
                      fit: bool = True) -> np.ndarray:
        """
        Normalize sequences using StandardScaler
        
        Args:
            sequences: List of sequences
            fit: Whether to fit the scaler
            
        Returns:
            Normalized sequences as array
        """
        # Convert to array
        X = np.array(sequences)  # Shape: (num_samples, seq_len, features)
        
        # Reshape for scaling
        num_samples, seq_len, num_features = X.shape
        X_reshaped = X.reshape(-1, num_features)  # Shape: (num_samples * seq_len, features)
        
        # Fit and transform or just transform
        if fit:
            X_normalized = self.scaler.fit_transform(X_reshaped)
        else:
            X_normalized = self.scaler.transform(X_reshaped)
        
        # Reshape back
        X_normalized = X_normalized.reshape(num_samples, seq_len, num_features)
        
        return X_normalized
    
    def create_splits(self, X: np.ndarray, y: np.ndarray) -> Tuple:
        """
        Create train/validation/test splits
        
        Args:
            X: Feature array
            y: Label array
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        train_split = self.config['training']['train_split']
        val_split = self.config['training']['val_split']
        test_split = self.config['training']['test_split']
        random_seed = self.config['training']['random_seed']
        
        # First split: train + val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=test_split,
            random_state=random_seed,
            stratify=y
        )
        
        # Second split: train vs val
        val_ratio = val_split / (train_split + val_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=random_seed,
            stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_processed_data(self, X_train, X_val, X_test, 
                           y_train, y_val, y_test):
        """Save processed data to disk"""
        print("\nSaving processed data...")
        
        # Save arrays
        np.save(self.processed_dir / "X_train.npy", X_train)
        np.save(self.processed_dir / "X_val.npy", X_val)
        np.save(self.processed_dir / "X_test.npy", X_test)
        np.save(self.processed_dir / "y_train.npy", y_train)
        np.save(self.processed_dir / "y_val.npy", y_val)
        np.save(self.processed_dir / "y_test.npy", y_test)
        
        # Save scaler
        with open(self.processed_dir / "scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"  Saved to: {self.processed_dir}")
        print(f"  Train: {X_train.shape}")
        print(f"  Val: {X_val.shape}")
        print(f"  Test: {X_test.shape}")
    
    def process(self):
        """Main preprocessing pipeline"""
        print("\n" + "="*60)
        print("DATA PREPROCESSING")
        print("="*60 + "\n")
        
        # Load raw data
        sequences, labels = self.load_raw_data()
        
        if len(sequences) == 0:
            print("Error: No data found!")
            return
        
        # Normalize
        print("\nNormalizing data...")
        X = self.normalize_data(sequences, fit=True)
        y = np.array(labels)
        
        # Create splits
        print("\nCreating train/val/test splits...")
        X_train, X_val, X_test, y_train, y_val, y_test = self.create_splits(X, y)
        
        # Save
        self.save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test)
        
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE")
        print("="*60 + "\n")


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.process()
