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
    # ... (unchanged code)

def main(n_runs = 10):
    # ... (unchanged code)

    # Number of experiments to run
    num_experiments = n_runs # Change this to the desired number of experiments

    # List to store computed metrics for each experiment
    history_list = []

    for exp in range(num_experiments):
        # Dictionary to store computed metrics for the current experiment
        history = {'model': ['vae', 'betavae', 'btcvae', 'factorvae'], 'mig': [], 'jemmig': [], 'dcimig': [], 'modularity': [], 'sap': []}

        print(f"\nExperiment {exp + 1}:")

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

        # Append the metrics for the current experiment to the history list
        history_list.append(history)

    # Aggregate results across all experiments
    aggregated_metrics = {'model': [], 'mig_mean': [], 'mig_std': [], 'jemmig_mean': [], 'jemmig_std': [],
                          'dcimig_mean': [], 'dcimig_std': [], 'modularity_mean': [], 'modularity_std': [],
                          'sap_mean': [], 'sap_std': []}

    for model in ['vae', 'betavae', 'btcvae', 'factorvae']:
        for metric_name in ['mig', 'jemmig', 'dcimig', 'modularity', 'sap']:
            # Extract metric values for the current model and metric
            metric_values = [exp[metric_name] for exp in history_list]

            # Compute mean and std
            mean_value = np.mean(metric_values)
            std_value = np.std(metric_values)

            # Append results to the aggregated_metrics dictionary
            aggregated_metrics['model'].append(model)
            aggregated_metrics[f'{metric_name}_mean'].append(mean_value)
            aggregated_metrics[f'{metric_name}_std'].append(std_value)

    # Create a DataFrame from the aggregated metrics and save to CSV
    df_aggregated = pd.DataFrame(aggregated_metrics)
    df_aggregated.to_csv('./Exports/metrics_aggregated.csv', index=False)

    # Print aggregated metrics in LaTeX format
    print("\nAggregated Metrics:\n")
    for model in ['vae', 'betavae', 'btcvae', 'factorvae']:
        print(f"{model} & ${aggregated_metrics['mig_mean'][aggregated_metrics['model'].index(model)]:.3f} \\pm {aggregated_metrics['mig_std'][aggregated_metrics['model'].index(model)]:.3f}$ \\\\")
        print("\\hline")

if __name__ == "__main__":
    main()
