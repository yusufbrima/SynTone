import numpy as np # For numerical operations
from generator import WaveformGenerator


from pathlib import Path

file_path = Path("/net/projects/scratch/summer/valid_until_31_January_2024/ybrima/data/learning/SyncSpeech/dataset.npz")

if __name__ == "__main__":

    sr = 16000
    generator = WaveformGenerator(sample_rate=sr, freqs=np.linspace(440, sr/2, num=400), amps =np.linspace(0.1, 1.0, num=40))
    
    # Generate and export synthetic data 
    generator.export(file_name=file_path)   
    
    # Load saved data
    data = np.load(file_path)
    x = data['x']
    y = data['y']
    CLASSES = data['classes']

    print(x.shape)