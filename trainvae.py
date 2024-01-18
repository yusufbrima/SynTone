import numpy as np # For numerical operations
import torch # Pytorch framework
from torch.utils.data import DataLoader # For loading data into batches
import torch.nn as nn # Neural network modules 
import torch.optim as optim # Optimizers like SGD, Adam etc
from sklearn.model_selection import train_test_split # Splitting data into train and test
from pathlib import Path # For path manipulation

# Custom model and loss modules
from Models.vae import VAEDeep  
from Losses.vae import BetaVAELoss,BTCVAELoss,FactorKLoss

# Custom dataset and utility functions
from datasets import SpectrogramDataset  
from utils import train_with_validation,train_with_validation_general 

from tqdm import tqdm # For progress bar

file_path = Path("/net/projects/scratch/summer/valid_until_31_January_2024/ybrima/data/learning/SyncSpeech/dataset_16k.npz") 

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
    batch_size = 64
    epochs = 20
    latent_dim = 8
    model_list = [f'./Exports/vae2deep_{latent_dim}.pth', f'./Exports/betavae2deep_{latent_dim}.pth',f'./Exports/btcvae2deep_{latent_dim}.pth', f'./Exports/factorvae2deep_{latent_dim}.pth' ]
    loss_fn = [BetaVAELoss, BetaVAELoss, BTCVAELoss, FactorKLoss]
    for i,  model in enumerate(model_list):
        if i == 3: # FactorVAE
            epochs *= 2
            batch_size *= 2
            # Create dataloader objects
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Get input shape from train data
        _,x_spec_batch, _ = next(iter(train_loader))
        input_shape = x_spec_batch.shape
        
        vae = VAEDeep(latent_dim, input_shape).to(device)
        optimizer = optim.Adam(vae.parameters(), lr=0.001, betas = (0.9, 0.999))

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

        train_losses, val_losses = train_with_validation_general(vae, train_loader, val_loader, optimizer, loss_fn, num_epochs=epochs, device=device, idx = i, filename = model_list[i])