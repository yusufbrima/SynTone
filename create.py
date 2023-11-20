import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import librosa
from pathlib import Path
from generator import WaveformGenerator
from Models.classifier import WaveformClassifier
from datasets import WaveformDataset
from utils import train_model
from tqdm import tqdm

file_path = Path("/net/projects/scratch/summer/valid_until_31_January_2024/ybrima/data/learning/SyncSpeech/dataset_16k_sine.npz")

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    generator = WaveformGenerator(sample_rate=16000, freqs=np.linspace(440, 16000/2, num=400), waveforms=['sine'])
    generator.export(file_name=file_path)
    # Load data
    data = np.load(file_path)
    x = data['x']
    y = data['y']
    CLASSES = data['classes']

    print(x.shape)