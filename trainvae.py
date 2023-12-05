import numpy as np # For numerical operations
import torch # Pytorch framework
from torch.utils.data import DataLoader # For loading data into batches
import torch.nn as nn # Neural network modules 
import torch.optim as optim # Optimizers like SGD, Adam etc
from sklearn.model_selection import train_test_split # Splitting data into train and test
from pathlib import Path # For path manipulation

# Custom model and loss modules
from Models.vae import VAE,VAEDeep,VAEDeeper  
from Losses.vae import BetaVAELoss,BTCVAELoss

# Custom dataset and utility functions
from datasets import SpectrogramDataset  
from utils import train_with_validation  

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

    # Create dataloader objects
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) 
    val_loader = DataLoader(val_dataset, batch_size=64)
    
    # Train multiple models
    num_models = 1
    latent_dim = 8

    # Get input shape from train data
    _,x_spec_batch, _ = next(iter(train_loader))
    input_shape = x_spec_batch.shape
    
    train_losses_over_models = []
    val_losses_over_models = []
    
    for i in tqdm(range(num_models)):

        vae = VAEDeep(latent_dim, input_shape).to(device) 
        loss_fn = BetaVAELoss(beta = 4) 
        # loss_fn = BTCVAELoss(beta = 4, gamma = 1, n_data=len(train_dataset)) 
        optimizer = optim.Adam(vae.parameters(), lr=0.001, betas = (0.9, 0.999))
        
        train_losses, val_losses = train_with_validation(
            vae, train_loader, val_loader, optimizer, loss_fn, 
            num_epochs=50, device=device
        )
        
        train_losses_over_models.append(train_losses) 
        val_losses_over_models.append(val_losses)