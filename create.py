import numpy as np
from generator import WaveformGenerator
from pathlib import Path
import argparse

# Setup argument parser
parser = argparse.ArgumentParser(description='Generate and process waveforms.')
parser.add_argument('--sample_rate', type=int, default=16000, help='Sample rate for waveform generation')
parser.add_argument('--freq_start', type=float, default=440, help='Start frequency for linear space')
parser.add_argument('--freq_end', type=float, help='End frequency for linear space, default is half of sample rate')
parser.add_argument('--num_freqs', type=int, default=400, help='Number of frequencies in linear space')
parser.add_argument('--amp_start', type=float, default=0.1, help='Start amplitude for linear space')
parser.add_argument('--amp_end', type=float, default=1.0, help='End amplitude for linear space')
parser.add_argument('--num_amps', type=int, default=20, help='Number of amplitudes in linear space')
parser.add_argument('--file_path', type=str, default='./Dataset/dataset_32K.npz', help='Path to save the dataset')

# Parse arguments
args = parser.parse_args()

# Update variables with parsed arguments
sr = args.sample_rate
freq_end = args.freq_end if args.freq_end else sr / 2
file_path = Path(args.file_path)

# Waveform generation
generator = WaveformGenerator(
    sample_rate=sr, 
    freqs=np.linspace(args.freq_start, freq_end, num=args.num_freqs), 
    amps=np.linspace(args.amp_start, args.amp_end, num=args.num_amps)
)

if __name__ == "__main__":
    # Generate and export synthetic data 
    generator.export(file_name=file_path)   
    
    # Load saved data
    data = np.load(file_path)
    x = data['x']
    y = data['y']
    CLASSES = data['classes']

    print(x.shape)