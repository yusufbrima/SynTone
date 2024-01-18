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

def animate_generated_samples(x_generated_batch, sr, hop_length, n_samples=10, alpha=1.0, color='b', output_path="./Figures/Generated_Samples.gif"):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor('none')

    camera = Camera(fig)
    # take n_samples from the batch
    x_generated_batch = x_generated_batch[:n_samples]

    with tqdm(total=len(x_generated_batch), desc="Generating Samples") as pbar:
        for i in range(len(x_generated_batch)):
            mel_spec_generated = x_generated_batch[i].squeeze().detach().cpu().numpy()

            waveform_generated = librosa.feature.inverse.mel_to_audio(mel_spec_generated, sr=sr, n_fft=2048, hop_length=512, n_iter=512)
            librosa.display.waveshow(waveform_generated, sr=sr, color=color, ax=axs[0], alpha=alpha)
            axs[0].set_xlabel('Time(s)')
            axs[0].set_ylabel('Amplitude')
            axs[0].set_title('Time domain', fontsize=12)

            ft_generated = np.fft.fft(waveform_generated)
            axs[1].plot(np.abs(ft_generated)[:len(ft_generated)//2], color=color)
            axs[1].set_xlabel('Frequency')
            axs[1].set_ylabel('Magnitude')
            axs[1].set_title('Fourier Transform', fontsize=12)


            librosa.display.specshow(librosa.power_to_db(mel_spec_generated, ref=np.max),
                                    y_axis='mel', x_axis='s', sr=sr,
                                    hop_length=hop_length, ax=axs[2])
            axs[2].set_title('Mel Spectrogram', fontsize=12)

            # fig.suptitle(f'Generated Sample {i+1}', fontsize=16)
            camera.snap()
            pbar.update(1)

    animation = camera.animate()
    animation.save(output_path)
    plt.close(fig)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sr = 16000
    hop_length = 512

    file_path = Path("/path/to/dataset.npz")

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
    
    num_samples = 100  # Number of sounds to generate
    latent_samples = torch.randn(num_samples, latent_dim).to(device)  # Generate random samples

    # Decode the latent samples to generate new sounds
    with torch.no_grad():
        vae.eval()
        generated_mel_spectrograms = vae.decode(latent_samples)  # Decode the latent samples

    num_samples =  len(generated_mel_spectrograms)
    animate_generated_samples(generated_mel_spectrograms, sr=sr, hop_length = hop_length, n_samples=num_samples)

    pbar.close()
    print("Processing complete!")
