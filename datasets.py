import torch
from torch.utils.data import Dataset
import torchaudio.transforms as T 
class WaveformDataset(Dataset):
    """PyTorch dataset for waveform data."""
    
    def __init__(self, x, y, transform=None):
        """
        Args:
            x (ndarray): Waveform data
            y (ndarray): Corresponding labels
        """
        
        # Convert to PyTorch tensors
        self.x = torch.from_numpy(x).float() 
        self.y = torch.from_numpy(y).long()
        self.transform = transform
        
    def __len__(self):
        """Returns length of dataset."""
        return len(self.x)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of sample to retrieve
        
        Returns:
            x (Tensor): Waveform sample 
            y (int): Corresponding class label
        """
        
        # Get waveform and reshape to 1D
        x = self.x[idx]
        if self.transform:
            max_amplitude = torch.max(torch.abs(x))
            x =  x / max_amplitude
        x = x.reshape(1, -1) 
        
        # Get label
        y = self.y[idx]
        
        return x, y


class SpectrogramDataset(Dataset):
    """PyTorch dataset for audio waveforms and spectrograms."""
    
    def __init__(self, waveforms, labels):
        """
        Args:
            waveforms (ndarray): Audio waveform data 
            labels (ndarray): Corresponding labels
        """
        self.x = torch.from_numpy(waveforms).float()
        self.y = torch.from_numpy(labels).long()
        
        # Spectrogram transform
        self.spectrogram = T.Spectrogram(n_fft=1024,
                                          win_length=512,
                                          center=True,
                                          pad_mode="reflect",
                                          power=2.0)
                                          
        # Inverse spectrogram transform 
        self.griffinlim = T.GriffinLim(n_fft=1024, 
                                       win_length=512,
                                       n_iter=200)
        
    def __len__(self):
        """Returns length of dataset."""
        return len(self.x)

    def __getitem__(self, idx):
        """Retrieves waveform and converts to spectrogram at given index.
        
        Args:
            idx (int): Index 
        
        Returns:
            waveform (Tensor): Waveform 
            spectrogram (Tensor): Corresponding spectrogram
            label (int): Class label
        """
        waveform = self.x[idx].unsqueeze(0).float()
        spec = self.spectrogram(waveform)
        label = self.y[idx]
        
        return waveform, spec, label

if __name__ == "__main__":
    pass