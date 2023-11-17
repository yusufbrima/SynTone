import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """Variational Autoencoder model."""
    
    def __init__(self, latent_dim, input_shape):
        """
        Args:
            latent_dim (int): Dimensionality of latent space.
            input_shape (tuple): Shape of the input tensor (num_channels, sequence_length)
        """
        super().__init__()

        _, num_channels, sequence_length = input_shape

        # Encoder
        self.conv1 = nn.Conv1d(num_channels, 16, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1) 

        # Compute the size for fully connected layers dynamically
        fc_size = 32 * (sequence_length // 4)
        self.flatten = nn.Flatten()
        self.fc_enc1 = nn.Linear(fc_size, 64)
        self.fc_enc2_mean = nn.Linear(64, latent_dim)
        self.fc_enc2_logvar = nn.Linear(64, latent_dim)
        
        # Decoder
        self.fc_dec1 = nn.Linear(latent_dim, 64)
        self.fc_dec2 = nn.Linear(64, 1 * sequence_length)
        self.unflatten = nn.Unflatten(1, (1, sequence_length))
        self.deconv1 = nn.ConvTranspose1d(1, 16, kernel_size=5, padding=2)
        self.deconv2 = nn.ConvTranspose1d(16, num_channels, kernel_size=3, padding=1)
        
        # Non-linearity
        self.neg_factor = 0.01
        
    def encode(self, x):
        """
        Encodes the input data into mean and log variance of the latent space.
        """
        
        # Encoding layers
        x = F.leaky_relu(self.conv1(x), negative_slope=self.neg_factor)
        x = self.pool(x)
        x = F.leaky_relu(self.conv2(x), negative_slope=self.neg_factor)
        x = self.pool(x)
        x = self.flatten(x)
        x = F.leaky_relu(self.fc_enc1(x), negative_slope=self.neg_factor)
        
        # Output parameters of latent distribution
        z_mean = self.fc_enc2_mean(x)
        z_logvar = self.fc_enc2_logvar(x)
        
        return z_mean, z_logvar

    
    def reparameterize(self, mean, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    
    def decode(self, z):
        """
        Maps the given latent code onto the output.
        """
        
        # Decoding layers
        z = F.leaky_relu(self.fc_dec1(z), negative_slope=self.neg_factor)
        z = F.leaky_relu(self.fc_dec2(z), negative_slope=self.neg_factor)
        z = self.unflatten(z)
        z = F.leaky_relu(self.deconv1(z), negative_slope=self.neg_factor)
        x_hat = torch.tanh(self.deconv2(z))
        
        return x_hat

    
    def forward(self, x):
        """
        Forward pass of the model.
        """
        z_mean, z_logvar = self.encode(x)
        z = self.reparameterize(z_mean, z_logvar) 
        x_hat = self.decode(z)
        
        return x_hat, z_mean, z_logvar




if __name__ == "__main__":
    pass