import torch
from torch.utils.data import Dataset
import torchaudio.transforms as T
import librosa
import numpy as np 
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
    
    def __init__(self, waveforms, labels, sr =  16000):
        """
        Args:
            waveforms (ndarray): Audio waveform data 
            labels (ndarray): Corresponding labels
        """
        # self.x = torch.from_numpy(waveforms).float()
        self.x = waveforms
        self.y = torch.from_numpy(labels).long()
        self.sr = sr
        
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
        waveform = self.x[idx]
        mel_spec = librosa.feature.melspectrogram(y=self.x[idx], sr=self.sr)
        # mel_spec_db  = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_normalized = torch.from_numpy(librosa.util.normalize(mel_spec)).float().unsqueeze(0) 
        label = self.y[idx]
        
        return waveform, mel_spec_normalized, label
    
    # create a static method to invert the spectrogram
    @staticmethod
    def invert_melspectrogram(spectrogram, sr = 16000):
        """Inverts a spectrogram back to a waveform.
        
        Args:
            spectrogram (Tensor): Spectrogram to invert
            sr (int): Sampling rate
        
        Returns:
            waveform (Tensor): Inverted waveform
        """
        # Convert to numpy array
        spectrogram = spectrogram.squeeze(0).numpy()
        
        # Convert to power
        # spectrogram = librosa.db_to_power(spectrogram)
        
        # Invert
        waveform = librosa.feature.inverse.mel_to_audio(spectrogram, sr=sr, n_fft=2048, hop_length=512, n_iter=512)
        
        return waveform #torch.from_numpy(waveform).float()


class DisentanglementDataset(Dataset):
    """PyTorch dataset for audio waveforms and spectrograms."""
    
    def __init__(self, filepath, sr =  16000):
        """
        Args:
            waveforms (ndarray): Audio waveform data 
            labels (ndarray): Corresponding labels
        """
        # self.x = torch.from_numpy(waveforms).float()
        data = np.load(filepath, allow_pickle=True)
        waveforms = data['x']
        self.metadata = data['metadata'].tolist()
        self.CLASSES = data['classes'].tolist()
        self.frequences = torch.from_numpy(self.metadata['freqs']).float()
        self.amps = torch.from_numpy(self.metadata['amps']).float()
        self.waveforms = torch.from_numpy(np.array([self.CLASSES.index(x) for x in self.metadata['waveforms']])).long()
        self.x = waveforms
        self.sr = sr

        
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
        waveform = self.x[idx]
        mel_spec = librosa.feature.melspectrogram(y=self.x[idx], sr=self.sr)
        # mel_spec_db  = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_normalized = torch.from_numpy(librosa.util.normalize(mel_spec)).float().unsqueeze(0) 
        label = self.waveforms[idx]
        frequency = self.frequences[idx]
        amplitude = self.amps[idx]
        
        return waveform, mel_spec_normalized, label, frequency, amplitude
    
    # create a static method to invert the spectrogram
    @staticmethod
    def invert_melspectrogram(spectrogram, sr = 16000):
        """Inverts a spectrogram back to a waveform.
        
        Args:
            spectrogram (Tensor): Spectrogram to invert
            sr (int): Sampling rate
        
        Returns:
            waveform (Tensor): Inverted waveform
        """
        # Convert to numpy array
        spectrogram = spectrogram.squeeze(0).numpy()
        
        # Convert to power
        # spectrogram = librosa.db_to_power(spectrogram)
        
        # Invert
        waveform = librosa.feature.inverse.mel_to_audio(spectrogram, sr=sr, n_fft=2048, hop_length=512, n_iter=512)
        
        return waveform #torch.from_numpy(waveform).float()
if __name__ == "__main__":
    pass