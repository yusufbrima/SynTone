import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
from Models.vae import VAEDeep
from Losses.vae import BetaVAELoss, BTCVAELoss, FactorKLoss
from datasets import SpectrogramDataset
from utils import train_with_validation_general
from tqdm import tqdm

def main():
    # File path for the dataset
    file_path = Path("./Datasets/my_dataset.npz")

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

    # Common parameters
    batch_size = 64
    epochs = 50
    latent_dim = 8
    model_list = [f'./Exports/vae2deep_{latent_dim}_{epochs}.pth', f'./Exports/betavae2deep_{latent_dim}_{epochs}.pth',
                  f'./Exports/btcvae2deep_{latent_dim}_{epochs}.pth', f'./Exports/factorvae2deep_{latent_dim}_{epochs}.pth']

    for i, model_path in enumerate(model_list):
        print(f"\nTraining Model {i}...")
        # Adjust parameters for FactorVAE
        if i == 3:
            epochs *= 2
            batch_size *= 2

        # Create dataloader objects
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Get input shape from train data
        _, x_spec_batch, _ = next(iter(train_loader))
        input_shape = x_spec_batch.shape

        # Create model, optimizer, and loss function
        vae = VAEDeep(latent_dim, input_shape).to(device)
        optimizer = torch.optim.Adam(vae.parameters(), lr=0.001, betas=(0.9, 0.999))

        # Select loss function based on model index
        if i == 0:
            loss_fn = BetaVAELoss(beta=1)
        elif i == 1:
            loss_fn = BetaVAELoss(beta=4)
        elif i == 2:
            loss_fn = BTCVAELoss(beta=4, gamma=1, n_data=len(train_dataset))
        else:
            factor_G = 6
            lr_disc = 5e-5
            loss_fn = FactorKLoss(device=device, gamma=factor_G, latent_dim=latent_dim,
                                  optim_kwargs=dict(lr=lr_disc, betas=(0.5, 0.9)))

        # Apply tqdm to the training loader for a progress bar
        print("Training...")
        train_loader = tqdm(train_loader, desc=f'Training Model {i}', leave=False)

        # Training loop
        train_losses, val_losses = train_with_validation_general(vae, train_loader, val_loader, optimizer,
                                                                 loss_fn, num_epochs=epochs, device=device,
                                                                 idx=i, filename=model_path)

        print(f"Model {i} training complete.")
        print(f"Train Losses: {train_losses}")
        print(f"Validation Losses: {val_losses}")

    print("All models trained successfully.")

if __name__ == "__main__":
    main()
