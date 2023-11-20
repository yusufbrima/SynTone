import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import librosa
from pathlib import Path
from Models.vae import VAE2D,VAEDeep,VAEDeeper
from datasets import SpectrogramDataset
from utils import train_with_validation
from Losses.vae import BetaVAELoss
from tqdm import tqdm

file_path = Path("/net/projects/scratch/summer/valid_until_31_January_2024/ybrima/data/learning/SyncSpeech/dataset_16k.npz")

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
    train_dataset = SpectrogramDataset(x_train, y_train)
    val_dataset = SpectrogramDataset(x_val, y_val)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    # Train 50 models
    num_models = 1

    latent_dim = 16
    
    # Get input shape 
    x_batch,x_spec_batch, y_batch = next(iter(train_loader))
    input_shape = x_spec_batch.shape

    train_losses_over_models = []
    val_losses_over_models = []
    for i in tqdm(range(num_models)):
      vae = VAEDeep(latent_dim, input_shape).to(device)
      loss_fn = BetaVAELoss(beta = 4)
      optimizer = optim.Adam(vae.parameters(), lr=0.001, betas = (0.9, 0.999))
      train_losses, val_losses = train_with_validation(vae, train_loader, val_loader, optimizer, loss_fn, num_epochs=200, device=device)
      train_losses_over_models.append(train_losses)
      val_losses_over_models.append(val_losses)