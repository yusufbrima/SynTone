import torch
import torch.nn as nn
import torch.nn.functional as F

class WaveformClassifier(nn.Module):
    """Convolutional network for classifying waveforms."""
    
    def __init__(self):
        """
        Initialize convolutional layers, fully connected layers, and flatten layer.
        """
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2) 
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 11025, 64)  
        self.fc2 = nn.Linear(64, 4)
        
        # Flatten layer
        self.flatten = nn.Flatten()

    def forward(self, x):
        """
        Run input tensor through convolutional and fully connected layers 
        to obtain waveform classification.
        
        Args:
            x (tensor): Batch of 1D waveform inputs
        
        Returns:
            x (tensor): Class scores 
        """
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = self.flatten(x)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    pass