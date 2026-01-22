"""
Training script for ASL Fingerspelling LSTM model
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import sys

from model import create_model
from utils import load_config, ensure_dir, get_idx_to_class

sys.path.append(str(Path(__file__).parent.parent))


class Trainer:
    """Train ASL Fingerspelling model"""
    
    def __init__(self, config_path: str = "../config/config.yaml"):
        """
        Initialize trainer
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Directories
        self.processed_dir = Path(__file__).parent.parent / "data" / "processed"
        self.checkpoint_dir = Path(__file__).parent.parent / "models" / "checkpoints"
        ensure_dir(str(self.checkpoint_dir))
        
        # Training config
        self.batch_size = self.config['training']['batch_size']
        self.learning_rate = self.config['training']['learning_rate']
        self.epochs = self.config['training']['epochs']
        self.patience = self.config['training']['early_stopping_patience']
        
        # Model
        self.model = create_model(self.config).to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
    def load_data(self):
        """Load preprocessed data"""
        print("Loading preprocessed data...")
        
        X_train = np.load(self.processed_dir / "X_train.npy")
        X_val = np.load(self.processed_dir / "X_val.npy")
        X_test = np.load(self.processed_dir / "X_test.npy")
        y_train = np.load(self.processed_dir / "y_train.npy")
        y_val = np.load(self.processed_dir / "y_val.npy")
        y_test = np.load(self.processed_dir / "y_test.npy")
        
        print(f"  Train: {X_train.shape}")
        print(f"  Val: {X_val.shape}")
        print(f"  Test: {X_test.shape}")
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train)
        X_val = torch.FloatTensor(X_val)
        X_test = torch.FloatTensor(X_test)
        y_train = torch.LongTensor(y_train)
        y_val = torch.LongTensor(y_val)
        y_test = torch.LongTensor(y_test)
        
        # Create datasets
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )
        
    def train_epoch(self) -> tuple:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self) -> tuple:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in tqdm(self.val_loader, desc="Validation"):
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accs, label='Train Acc')
        ax2.plot(self.val_accs, label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.checkpoint_dir / 'training_history.png', dpi=150)
        print(f"  Saved training history plot")
    
    def evaluate_test_set(self):
        """Evaluate on test set and generate confusion matrix"""
        print("\nEvaluating on test set...")
        
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in tqdm(self.test_loader, desc="Testing"):
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = self.model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        # Calculate accuracy
        accuracy = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
        print(f"\nTest Accuracy: {accuracy:.2f}%")
        
        # Classification report
        idx_to_class = get_idx_to_class(self.config['data_collection']['classes'])
        class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
        
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        plt.figure(figsize=(16, 14))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.checkpoint_dir / 'confusion_matrix.png', dpi=150)
        print(f"  Saved confusion matrix")
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*60)
        print("TRAINING ASL FINGERSPELLING MODEL")
        print("="*60)
        print(f"\nDevice: {self.device}")
        print(f"Model parameters: {self.model.get_num_parameters():,}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Epochs: {self.epochs}")
        print(f"Early stopping patience: {self.patience}\n")
        
        # Load data
        self.load_data()
        
        print("\n" + "="*60)
        print("Starting training...")
        print("="*60 + "\n")
        
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            print("-" * 60)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Learning rate scheduler
            self.scheduler.step(val_loss)
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pth')
                print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # Save final model
        self.save_checkpoint('final_model.pth')
        
        # Plot training history
        print("\nGenerating plots...")
        self.plot_training_history()
        
        # Evaluate on test set
        # Load best model
        checkpoint = torch.load(self.checkpoint_dir / 'best_model.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.evaluate_test_set()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"\nBest Validation Accuracy: {self.best_val_acc:.2f}%")
        print(f"Best Validation Loss: {self.best_val_loss:.4f}")
        print(f"\nModel saved to: {self.checkpoint_dir / 'best_model.pth'}")
        print("="*60 + "\n")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
