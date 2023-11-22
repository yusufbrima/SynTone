import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from celluloid import Camera
import librosa
import torch
from pathlib import Path
from Models.vae import VAEDeep
from datasets import SpectrogramDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def load_data(file_path):
    data = np.load(file_path, allow_pickle=True)
    x = data['x']
    y = data['y']
    metadata = data['metadata'].tolist()
    CLASSES = data['classes']
    return x, y, metadata, CLASSES

def split_data(x, y):
    return train_test_split(x, y, test_size=0.2, random_state=42)

def create_datasets(x_train, y_train, x_val, y_val):
    train_dataset = SpectrogramDataset(x_train, y_train)
    val_dataset = SpectrogramDataset(x_val, y_val)
    return train_dataset, val_dataset

def create_dataloaders(train_dataset, val_dataset, batch_size=64):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader

def load_model(latent_dim, input_shape, model_path):
    vae = VAEDeep(latent_dim, input_shape).to(device)
    state_dict = torch.load(model_path)
    vae.load_state_dict(state_dict)
    return vae

def process_batch(x_spec_batch, vae, device, pbar):
    with torch.inference_mode():
        x_hat, z_mean, z_logvar = vae(x_spec_batch.to(device))
    return x_hat, z_mean, z_logvar



def visualize_results(x_spec_batch, x_hat, y_batch, CLASSES, sr, hop_length, n_samples=10, alpha=1.0, color='b', filename="Original_vs_Reconstructed"):
    fig, axs = plt.subplots(2, 3, figsize=(12, 6)) 
    fig.patch.set_facecolor('none')

    camera = Camera(fig)
    # take n_samples from the batch
    x_spec_batch = x_spec_batch[:n_samples]
    for i in range(len(x_spec_batch)):
        pbar.update(1)
        pbar.set_description(f"Processing {i+1} of {len(x_batch)}")
        
        mel_spec = x_spec_batch[i].squeeze().detach().cpu().numpy()
        mel_spec_recon = x_hat[i].squeeze().detach().cpu().numpy()

        waveform = librosa.feature.inverse.mel_to_audio(mel_spec, sr=sr, n_fft=2048, hop_length=512, n_iter=512)
        # Clear the axis before plotting

        librosa.display.waveshow(waveform, sr=sr, color=color, ax=axs[0,0], alpha=alpha)
        axs[0,0].set_xlabel('Time(s)')
        axs[0,0].set_ylabel('Amplitude')
        axs[0,0].set_title('Time domain', fontsize=12)

        waveform_recon = librosa.feature.inverse.mel_to_audio(mel_spec_recon, sr=sr, n_fft=2048, hop_length=512, n_iter=512)  
        librosa.display.waveshow(waveform_recon, sr=sr, color=color, ax=axs[1,0], alpha=alpha)
        axs[1,0].set_xlabel('Time(s)')
        axs[1,0].set_ylabel('Amplitude')

        ft = np.fft.fft(waveform)
        axs[0,1].plot(np.abs(ft)[:len(ft)//2], color=color)
        axs[0,1].set_xlabel('Frequency')
        axs[0,1].set_ylabel('Magnitude')
        axs[0,1].set_title('Fourier Transform', fontsize=12)

        ft_recon = np.fft.fft(waveform_recon)
        axs[1,1].plot(np.abs(ft_recon)[:len(ft_recon)//2], color=color) 
        axs[1,1].set_xlabel('Frequency')
        axs[1,1].set_ylabel('Magnitude')
        
        
        librosa.display.specshow(librosa.power_to_db(mel_spec, ref=np.max),
                        y_axis='mel', x_axis='s', sr=sr, 
                        hop_length=hop_length, ax=axs[0,2])
        axs[0,2].set_title('Mel spectrogram', fontsize=12)

        librosa.display.specshow(librosa.power_to_db(mel_spec_recon, ref=np.max),
                        y_axis='mel', x_axis='s', sr=sr,
                        hop_length=hop_length, ax=axs[1,2])
                        
        fig.suptitle(f'Original (top) vs. Reconstructed (bottom) of a {CLASSES[y_batch[i]].capitalize()}', fontsize=16)
        camera.snap()


    animation = camera.animate()
    animation.save(f"./Figures/{filename}.gif")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sr = 16000
    hop_length = 512

    file_path = Path("/net/projects/scratch/summer/valid_until_31_January_2024/ybrima/data/learning/SyncSpeech/dataset_16k_sine.npz")

    x, y, metadata, CLASSES = load_data(file_path)
    x_train, x_val, y_train, y_val = split_data(x, y)

    train_dataset, val_dataset = create_datasets(x_train, y_train, x_val, y_val)
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset)

    latent_dim = 8

    x_batch, x_spec_batch, y_batch = next(iter(train_loader))
    input_shape = x_spec_batch.shape

    pbar = tqdm(total=len(x_batch))

    vae = load_model(latent_dim, input_shape, './Exports/vae2deep_{}.pth'.format(latent_dim))

    x_hat, z_mean, z_logvar = process_batch(x_spec_batch, vae, device, pbar)

    visualize_results(x_spec_batch, x_hat, y_batch, CLASSES, sr, hop_length)

    pbar.close()
    print("Processing complete!")
