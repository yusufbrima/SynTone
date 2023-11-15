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
from Models.classifier import WaveformClassifier
from datasets import WaveformDataset
from utils import train_model
from tqdm import tqdm

file_path = Path("/net/projects/scratch/summer/valid_until_31_January_2024/ybrima/data/learning/SyncSpeech/dataset.npz")

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    # Load data
    data = np.load(file_path)
    x = data['x']
    y = data['y']
    CLASSES = data['classes']

    # # Split data
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    # Create datasets
    train_dataset = WaveformDataset(x_train, y_train)
    val_dataset = WaveformDataset(x_val, y_val)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    # Train n models
    num_models = 100
    train_accs_over_models = []
    val_accs_over_models = []
    train_losses_over_models = []
    val_losses_over_models = []
    for i in tqdm(range(num_models)):
      
      train_losses, train_accs, val_losses, val_accs = train_model(WaveformClassifier, train_loader, val_loader,device, num_epochs=20)
      train_accs_over_models.append(train_accs)
      val_accs_over_models.append(val_accs)
      train_losses_over_models.append(train_losses)
      val_losses_over_models.append(val_losses)
    
    # Plot training and validation accuracy over epochs

    # Plot results
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Train Accuracy',  fontsize=14)
    for i in range(num_models):
      plt.plot(train_accs_over_models[i], alpha=0.5)
    
    plt.title(f'Train Accuracy for {num_models} Randomly Initialized Networks', fontsize=14)
    plt.savefig(f'./Figures/train_accs_{num_models}.png')
    plt.close()

    # Plot results
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Validation Accuracy', fontsize=14)
    for i in range(num_models):
      plt.plot(val_accs_over_models[i], alpha=0.5)
    plt.title(f'Validation Accuracy for {num_models} Randomly Initialized Networks', fontsize=14)
    plt.savefig(f'./Figures/val_accs_{num_models}.png')
    plt.close()





