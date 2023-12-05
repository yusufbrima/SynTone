import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from Models.vae import VAEDeep
from datasets import DisentanglementDataset
from torch.utils.data import DataLoader
from Metrics.mig import mig
from Metrics.jemmig import jemmig
from Metrics.dci import dci
from Metrics.sap import sap
from Metrics.dcimig import dcimig
from Metrics.modularity import modularity

def compute_metrics(vae, dataloader, device):
    """
    Compute disentanglement metrics for a given VAE model.

    Parameters:
    - vae (torch.nn.Module): The VAE model.
    - dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
    - device (str): Device to run the computations ('cuda' or 'cpu').

    Returns:
    - metrics (dict): Dictionary containing disentanglement metrics.
    """
    z_list, y_list, frequency_list, amplitude_list = [], [], [], []

    with torch.inference_mode():
        for x_batch, x_spec_batch, y_batch, frequency_batch, amplitude_batch in dataloader:
            x_spec_batch = x_spec_batch.to(device)
            x_hat, z_mean, z_logvar = vae(x_spec_batch)
            z = vae.reparameterize(z_mean, z_logvar)

            # Append current batch of latent vectors to the list
            z_list.append(z.cpu().detach().numpy())

            # Append current batch of y, frequency, and amplitude to their respective lists
            y_list.append(y_batch.cpu().detach().numpy())
            frequency_list.append(frequency_batch.cpu().detach().numpy())
            amplitude_list.append(amplitude_batch.cpu().detach().numpy())

    # Convert lists to numpy arrays
    v_factors = np.concatenate(z_list, axis=0)
    all_labels = np.concatenate(y_list, axis=0)
    all_frequency = np.concatenate(frequency_list, axis=0)
    all_amplitude = np.concatenate(amplitude_list, axis=0)
    z_factors = np.column_stack((all_labels, all_frequency, all_amplitude))

    # Compute metrics
    metrics = {
        'mig': mig(v_factors, z_factors),
        'jemmig': jemmig(v_factors, z_factors),
        'dcimig': dcimig(v_factors, z_factors),
        'modularity': modularity(v_factors, z_factors),
        'sap': sap(v_factors, z_factors),
    }

    return metrics

def main():
    # File path for the dataset
    file_path = Path("/net/projects/scratch/summer/valid_until_31_January_2024/ybrima/data/learning/SyncSpeech/dataset_16k.npz")

    # Create DisentanglementDataset and DataLoader
    dataset = DisentanglementDataset(file_path)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Number of models and latent dimension
    num_models = 1
    latent_dim = 8

    # Get a batch of data to determine input shape
    _, x_spec_batch, _, _, _ = next(iter(train_loader))
    input_shape = x_spec_batch.shape

    # Sampling rate and hop length for librosa
    sr = dataset.sr
    hop_length = 512

    # Random index for visualization
    idx = np.random.randint(0, x_spec_batch.shape[0])

    # Device (cuda if available, else cpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # List of model paths to evaluate
    model_list = ['./Exports/vae2deep_8.pth', './Exports/betavae2deep_8.pth', './Exports/btcvae2deep_8.pth','./Exports/factorvae2deep_8.pth']

    # Dictionary to store computed metrics
    history = {'model': ['vae', 'betavae', 'btcvae','factorvae'], 'mig': [], 'jemmig': [], 'dcimig': [], 'modularity': [], 'sap': []}

    # Evaluate metrics for each model
    for model_path in model_list:
        print(f"\nEvaluating Model: {model_path}")
        vae = VAEDeep(latent_dim, input_shape).to(device)
        state_dict = torch.load(model_path)
        vae.load_state_dict(state_dict)

        # Compute metrics for the current model
        metrics = compute_metrics(vae, train_loader, device)

        # Store metrics in the history dictionary
        for metric_name, value in metrics.items():
            history[metric_name].append(value)

    # Create a DataFrame from the computed metrics and save to CSV
    df = pd.DataFrame(history)
    df.to_csv('./Exports/metrics.csv', index=False)

    # Print DataFrame in LaTeX format
    print("Metrics:\n")
    print(df.to_latex(index=False))

if __name__ == "__main__":
    main()
