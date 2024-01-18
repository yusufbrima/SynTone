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
import argparse

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

def main(file_path, batch_size, num_experiments, latent_dim, model_list):
    # Create DisentanglementDataset and DataLoader
    dataset = DisentanglementDataset(Path(file_path))
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Get a batch of data to determine input shape
    _, x_spec_batch, _, _, _ = next(iter(train_loader))
    input_shape = x_spec_batch.shape

    # Device (cuda if available, else cpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # List to store computed metrics for each experiment
    history_list = []

    for exp in range(num_experiments):
        history = {'model': ['vae', 'betavae', 'btcvae', 'factorvae'], 'mig': [], 'jemmig': [], 'dcimig': [], 'modularity': [], 'sap': []}

        print(f"\nExperiment {exp + 1}:")

        for model_path in model_list:
            print(f"\nEvaluating Model: {model_path}")
            vae = VAEDeep(latent_dim, input_shape).to(device)
            state_dict = torch.load(model_path)
            vae.load_state_dict(state_dict)

            metrics = compute_metrics(vae, train_loader, device)
            for metric_name, value in metrics.items():
                history[metric_name].append(value)

        history_list.append(history)

    # Aggregate results across all experiments
    aggregated_metrics = {...}  # Complete this section as in your original code

    # Create a DataFrame from the aggregated metrics and save to CSV
    df_aggregated = pd.DataFrame(aggregated_metrics)
    df_aggregated.to_csv('./Exports/metrics_aggregated.csv', index=False)

    # Print aggregated metrics in LaTeX format
    print("\nAggregated Metrics:\n")
    for model in ['vae', 'betavae', 'btcvae', 'factorvae']:
        print(f"{model} & ${aggregated_metrics['mig_mean'][aggregated_metrics['model'].index(model)]:.3f} \\pm {aggregated_metrics['mig_std'][aggregated_metrics['model'].index(model)]:.3f}$ \\\\")
        print("\\hline")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate disentanglement metrics for VAE models.')
    parser.add_argument('--file_path', type=str, default='/net/projects/scratch/summer/valid_until_31_January_2024/ybrima/data/learning/SynTone/my_dataset.npz', help='Path to the dataset file')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for data loading')
    parser.add_argument('--num_experiments', type=int, default=5, help='Number of experiments to run')
    parser.add_argument('--latent_dim', type=int, default=8, help='Latent dimension for the VAE models')
    parser.add_argument('--model_list', nargs='+', default=['./Exports/vae2deep_8.pth', './Exports/betavae2deep_8.pth', './Exports/btcvae2deep_8.pth', './Exports/factorvae2deep_8.pth'], help='List of model paths')

    args = parser.parse_args()
    main(args.file_path, args.batch_size, args.num_experiments, args.latent_dim, args.model_list)
