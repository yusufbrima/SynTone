import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from pathlib import Path
from Models.vae import VAEDeep
from Losses.vae import BetaVAELoss, BTCVAELoss, FactorKLoss
from datasets import SpectrogramDataset
from utils import train_with_validation_general
from tqdm import tqdm
import argparse

# Setup argument parser
parser = argparse.ArgumentParser(description='Train various VAE models on spectrogram dataset.')
parser.add_argument('--file_path', type=str, default='./Dataset/dataset.npz', help='Path to the dataset file')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and validation')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for training')
parser.add_argument('--latent_dim', type=int, default=8, help='Dimension of the latent space in VAE')
parser.add_argument('--model_paths', nargs='+', default=[f'./Exports/vae2deep_{8}.pth', f'./Exports/betavae2deep_{8}.pth', f'./Exports/btcvae2deep_{8}.pth', f'./Exports/factorvae2deep_{8}.pth'], help='List of model paths')

# Parse arguments
args = parser.parse_args()

# Update variables with parsed arguments
file_path = Path(args.file_path)
batch_size = args.batch_size
epochs = args.epochs
latent_dim = args.latent_dim
model_list = args.model_paths

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # Load dataset
    data = np.load(file_path)
    x = data['x']
    y = data['y']
    CLASSES = data['classes']

    # Split data into train and validation
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    # Create datasets
    train_dataset = SpectrogramDataset(x_train, y_train)
    val_dataset = SpectrogramDataset(x_val, y_val)

    train_losses_over_models = []
    val_losses_over_models = []

    loss_fn = [BetaVAELoss, BetaVAELoss, BTCVAELoss, FactorKLoss]
    for i, model in enumerate(model_list):
        if i == 3:  # FactorVAE
            epochs *= 2
            batch_size *= 2

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        _, x_spec_batch, _ = next(iter(train_loader))
        input_shape = x_spec_batch.shape

        vae = VAEDeep(latent_dim, input_shape).to(device)
        optimizer = optim.Adam(vae.parameters(), lr=0.001, betas=(0.9, 0.999))

        # Set the loss function
        # ...
        if i == 0:
            loss_fn = BetaVAELoss(beta = 1) # When beta = 1, it is equivalent to standard VAE
        elif i == 1:
            loss_fn = BetaVAELoss(beta = 4)
        elif i == 2:
            loss_fn = BTCVAELoss(beta = 4, gamma = 1, n_data=len(train_dataset))
        else:
            factor_G = 6
            lr_disc = 5e-5
            loss_fn = FactorKLoss(device=device, gamma=factor_G, latent_dim=latent_dim, optim_kwargs=dict(lr=lr_disc, betas=(0.5, 0.9)))

        train_loader = tqdm(train_loader, desc=f'Training Model {i}', leave=False)

        train_losses, val_losses = train_with_validation_general(vae, train_loader, val_loader, optimizer, loss_fn, num_epochs=epochs, device=device, idx=i, filename=model_list[i])
