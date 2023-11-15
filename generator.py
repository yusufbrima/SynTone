import numpy as np
from scipy import signal 
from tqdm import tqdm

class WaveformGenerator:
    """Class for generating synthetic waveform dataset"""
    
    def __init__(self, 
                 sample_rate=44100,
                 duration=1,
                 waveforms=['sine', 'triangle', 'sawtooth', 'square'],  
                 freqs=np.linspace(440, 22050, num=100),
                 amps=np.linspace(0.1, 1.0, num=20)):

        """
        Initialize the waveform dataset generator.
        
        Args:
            sample_rate (int): Sample rate in Hz
            duration (float): Duration of each waveform in seconds
            waveforms (list): List of waveform types ('sine', 'triangle', etc)
            freqs (ndarray): Frequencies to generate waveforms at
            amps (ndarray): Amplitudes to scale waveforms by 
        """
        print("Initializing WaveformGenerator...")        
        self.sample_rate = sample_rate
        self.duration = duration
        self.waveforms = waveforms
        self.freqs = freqs
        self.amps = amps
        self.classes = waveforms

        # Calculate number of total samples
        self.num_samples = len(self.waveforms) * len(self.freqs) * len(self.amps)

        # Time axis 
        self.t = np.linspace(0, self.duration, self.sample_rate * self.duration)

        # Initialize output arrays
        self.x = np.zeros((self.num_samples, self.sample_rate * self.duration))
        self.y = np.zeros(self.num_samples, dtype=int)

        # Lists to store factors for each sample
        self.freqs_list = []
        self.amps_list = []
        self.waveforms_list = []
        
    def generate(self):
        """
        Generate the synthetic waveform dataset.
        
        Returns:
            x (ndarray): Waveform data 
            y (ndarray): Corresponding class labels
            metadata (dict): Metadata with freqs, amps, waveforms
        """
            
        sample_idx = 0
        total = len(self.freqs) * len(self.amps) * len(self.waveforms)
        print(f"Generating {total} waveforms at {self.sample_rate} for {self.duration} second(s) each...")
        with tqdm(total=total) as pbar:
            for freq in self.freqs:
                for amp in self.amps:
                    for waveform in self.waveforms:
                    
                        # Generate waveform
                        if waveform == 'sine':
                            self.x[sample_idx] = amp * np.sin(2*np.pi*freq*self.t)
                        elif waveform == 'triangle':
                            self.x[sample_idx] = amp * signal.sawtooth(2*np.pi*freq*self.t, 0.5)
                        elif waveform == 'sawtooth':
                            self.x[sample_idx] = amp * signal.sawtooth(2*np.pi*freq*self.t)
                        else:
                            self.x[sample_idx] = amp * signal.square(2*np.pi*freq*self.t)
                            
                        # Store factors
                        self.freqs_list.append(freq)
                        self.amps_list.append(amp)
                        self.waveforms_list.append(waveform)
                        
                        # Class label
                        self.y[sample_idx] = self.waveforms.index(waveform)
                        sample_idx += 1
                        
                        # Update progress bar
                        pbar.update(1)
                        
        # Package metadata
        self.metadata = {
            'freqs': np.array(self.freqs_list),
            'amps': np.array(self.amps_list), 
            'waveforms': np.array(self.waveforms_list)
        }
        print("Generated waveforms successfully.")
        return self.x, self.y, self.metadata
        
    def export(self, file_name):
        """
        Export generated dataset to a compressed .npz file.
        
        Args:
            file_name (str): File path to save to
        """

        x, y, metadata = self.generate()
        print(f"Exporting to {file_name}...")
        with tqdm(total=1) as pbar:
            np.savez_compressed(file_name, x=x, y=y, metadata=metadata, classes = self.classes) 
            pbar.update(1)
        print(f"Exported successfully to {file_name}.")

if __name__ == "__main__":
    pass