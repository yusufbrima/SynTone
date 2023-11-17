import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import librosa
from pathlib import Path
from Models.vae import VAE
from datasets import WaveformDataset
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
    train_dataset = WaveformDataset(x_train, y_train)
    val_dataset = WaveformDataset(x_val, y_val)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    # Train 50 models
    num_models = 1

    latent_dim = 16
    
    # Get input shape 
    x_batch, y_batch = next(iter(train_loader))
    input_shape = x_batch.shape
    train_losses_over_models = []
    val_losses_over_models = []
    for i in tqdm(range(num_models)):
      vae = VAE(latent_dim, input_shape).to(device)
      loss_fn = BetaVAELoss(beta = 4)
      optimizer = optim.Adam(vae.parameters(), lr=0.001, betas = (0.9, 0.999))
      train_losses, val_losses = train_with_validation(vae, train_loader, val_loader, optimizer, loss_fn, num_epochs=20, device=device)
      train_losses_over_models.append(train_losses)
      val_losses_over_models.append(val_losses)


    # Plot training and validation accuracy over epochs
    # Plot results
    # plt.figure(1)
    # plt.xlabel('Epoch', fontsize=14)
    # plt.ylabel('Loss',  fontsize=14)
    # for i in range(num_models):
    #   plt.plot(train_losses_over_models[i], alpha=0.5)
    
    # plt.title(f'Train Loss for {num_models} Randomly Initialized Networks', fontsize=14)
    # plt.savefig(f'./Figures/train_vae_loss_{num_models}.png')
    # plt.close()

    # plt.figure(2)
    # plt.xlabel('Epoch', fontsize=14)
    # plt.ylabel('Loss',  fontsize=14)
    # for i in range(num_models):
    #   plt.plot(val_losses_over_models[i], alpha=0.5)
    
    # plt.title(f'Validation Loss for {num_models} Randomly Initialized Networks', fontsize=14)
    # plt.savefig(f'./Figures/val_vae_loss_{num_models}.png')
    # plt.close()







