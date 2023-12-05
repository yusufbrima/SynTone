import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
from Models.vae import VAEDeep
from Losses.vae import FactorKLoss
from datasets import SpectrogramDataset
from utils import train_with_validation_general
from tqdm import tqdm

def main():
    # File path for the dataset
    file_path = Path("/net/projects/scratch/summer/valid_until_31_January_2024/ybrima/data/learning/SyncSpeech/dataset_16k.npz")

    # Set device (cuda if available, else cpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    print("Loading dataset...")
    data = np.load(file_path)
    x = data['x']
    y = data['y']

    # Split data into train and validation
    print("Splitting data into train and validation...")
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    # Create datasets
    print("Creating datasets...")
    train_dataset = SpectrogramDataset(x_train, y_train)
    val_dataset = SpectrogramDataset(x_val, y_val)

    # Common parameters for FactorVAE
    batch_size = 128  # Adjusted for FactorVAE
    epochs = 40  # Adjusted for FactorVAE
    latent_dim = 8

    # Create dataloader objects
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Get input shape from train data
    _, x_spec_batch, _ = next(iter(train_loader))
    input_shape = x_spec_batch.shape

    # Create FactorVAE model, optimizer, and loss function
    vae = VAEDeep(latent_dim, input_shape).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.001, betas=(0.9, 0.999))
    factor_G = 6
    lr_disc = 5e-5
    loss_fn = FactorKLoss(device=device, gamma=factor_G, latent_dim=latent_dim,
                          optim_kwargs=dict(lr=lr_disc, betas=(0.5, 0.9)))

    # Apply tqdm to the training loader for a progress bar
    print("Training...")
    train_loader = tqdm(train_loader, desc='Training FactorVAE', leave=False)

    # Training loop for FactorVAE only
    train_losses, val_losses = train_with_validation_general(vae, train_loader, val_loader, optimizer,
                                                             loss_fn, num_epochs=epochs, device=device,
                                                             idx=3, filename='./Exports/factorvae2deep_8.pth')

    print("FactorVAE training complete.")
    print(f"Train Losses: {train_losses}")
    print(f"Validation Losses: {val_losses}")

    print("FactorVAE trained successfully.")

if __name__ == "__main__":
    main()
