"""
LSTM Model for ASL Fingerspelling Recognition
"""
import torch
import torch.nn as nn


class ASLFingerSpellingLSTM(nn.Module):
    """LSTM-based sequence classifier for ASL fingerspelling"""
    
    def __init__(self, 
                 input_size: int = 258,
                 hidden_size_1: int = 128,
                 hidden_size_2: int = 64,
                 num_classes: int = 36,
                 dropout: float = 0.3,
                 bidirectional: bool = True):
        """
        Initialize LSTM model
        
        Args:
            input_size: Number of input features
            hidden_size_1: Hidden size for first LSTM layer
            hidden_size_2: Hidden size for second LSTM layer
            num_classes: Number of output classes
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super(ASLFingerSpellingLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        
        # First LSTM layer
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size_1,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=bidirectional
        )
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        
        # Second LSTM layer
        lstm2_input_size = hidden_size_1 * 2 if bidirectional else hidden_size_1
        self.lstm2 = nn.LSTM(
            input_size=lstm2_input_size,
            hidden_size=hidden_size_2,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=bidirectional
        )
        
        # Dropout
        self.dropout2 = nn.Dropout(dropout)
        
        # Fully connected layer
        fc_input_size = hidden_size_2 * 2 if bidirectional else hidden_size_2
        self.fc = nn.Linear(fc_input_size, num_classes)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # First LSTM layer
        lstm1_out, _ = self.lstm1(x)  # (batch, seq_len, hidden_size_1 * 2)
        lstm1_out = self.dropout1(lstm1_out)
        
        # Second LSTM layer
        lstm2_out, _ = self.lstm2(lstm1_out)  # (batch, seq_len, hidden_size_2 * 2)
        lstm2_out = self.dropout2(lstm2_out)
        
        # Take output from last time step
        last_output = lstm2_out[:, -1, :]  # (batch, hidden_size_2 * 2)
        
        # Fully connected layer
        output = self.fc(last_output)  # (batch, num_classes)
        
        return output
    
    def get_num_parameters(self):
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config: dict) -> ASLFingerSpellingLSTM:
    """
    Create model from configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized model
    """
    model = ASLFingerSpellingLSTM(
        input_size=config['model']['input_size'],
        hidden_size_1=config['model']['lstm_hidden_1'],
        hidden_size_2=config['model']['lstm_hidden_2'],
        num_classes=config['model']['num_classes'],
        dropout=config['model']['dropout'],
        bidirectional=config['model']['bidirectional']
    )
    
    return model


if __name__ == "__main__":
    # Test model
    from utils import load_config
    
    config = load_config("../config/config.yaml")
    model = create_model(config)
    
    print("Model Architecture:")
    print(model)
    print(f"\nTotal parameters: {model.get_num_parameters():,}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 30
    input_size = 258
    
    x = torch.randn(batch_size, seq_len, input_size)
    output = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ Model test passed!")
