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





class VAEDeep(nn.Module):
    """Variational Autoencoder model for 2D data (e.g., spectrograms)."""

    def __init__(self, latent_dim, input_shape):
        super().__init__()

        _, num_channels, height, width = input_shape

        # Encoder
        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)


        # Compute the size for fully connected layers dynamically
        fc_size = 64 * (height // 8) * (width // 8)
        self.flatten = nn.Flatten()
        self.fc_enc1 = nn.Linear(fc_size, 128)
        self.fc_enc2 = nn.Linear(128, 64)
        self.fc_enc3_mean = nn.Linear(64, latent_dim)
        self.fc_enc3_logvar = nn.Linear(64, latent_dim)

        # Decoder
        self.fc_dec1 = nn.Linear(latent_dim, 64)
        self.fc_dec2 = nn.Linear(64, 128)
        self.fc_dec3 = nn.Linear(128, fc_size)
        self.unflatten = nn.Unflatten(1, (64, height // 8, width // 8))
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(16, num_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

        # Non-linearity
        self.neg_factor = 0.01

    def encode(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=self.neg_factor)
        x = self.pool(x)
        x = F.leaky_relu(self.conv2(x), negative_slope=self.neg_factor)
        x = self.pool(x)
        x = F.leaky_relu(self.conv3(x), negative_slope=self.neg_factor)
        x = self.pool(x)
        x = self.flatten(x)
        x = F.leaky_relu(self.fc_enc1(x), negative_slope=self.neg_factor)
        x = F.leaky_relu(self.fc_enc2(x), negative_slope=self.neg_factor)

        z_mean = self.fc_enc3_mean(x)
        z_logvar = self.fc_enc3_logvar(x)

        return z_mean, z_logvar

    def decode(self, z):
        z = F.leaky_relu(self.fc_dec1(z), negative_slope=self.neg_factor)
        z = F.leaky_relu(self.fc_dec2(z), negative_slope=self.neg_factor)
        z = F.leaky_relu(self.fc_dec3(z), negative_slope=self.neg_factor)
        z = self.unflatten(z)
        z = F.leaky_relu(self.deconv1(z), negative_slope=self.neg_factor)
        z = F.leaky_relu(self.deconv2(z), negative_slope=self.neg_factor)
        x_hat = torch.sigmoid(self.deconv3(z))

        return x_hat

    def forward(self, x):
        z_mean, z_logvar = self.encode(x)
        z = self.reparameterize(z_mean, z_logvar)
        x_hat = self.decode(z)

        return x_hat, z_mean, z_logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std


class VAEDeeper(nn.Module):
    """Variational Autoencoder model for 2D data (e.g., spectrograms)."""

    def __init__(self, latent_dim, input_shape):
        super().__init__()

        _, num_channels, height, width = input_shape

        # Encoder
        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)

        # Compute the size for fully connected layers dynamically
        fc_size = 128 * (height // 16) * (width // 16)
        self.flatten = nn.Flatten()
        self.fc_enc1 = nn.Linear(fc_size, 256)
        self.fc_enc2 = nn.Linear(256, 128)
        self.fc_enc3_mean = nn.Linear(128, latent_dim)
        self.fc_enc3_logvar = nn.Linear(128, latent_dim)

        # Decoder
        self.fc_dec1 = nn.Linear(latent_dim, 128)
        self.fc_dec2 = nn.Linear(128, 256)
        self.fc_dec3 = nn.Linear(256, fc_size)
        self.unflatten = nn.Unflatten(1, (128, height // 16, width // 16))
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(16, num_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

        # Non-linearity
        self.neg_factor = 0.01

    def encode(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=self.neg_factor)
        x = self.pool(x)
        x = F.leaky_relu(self.conv2(x), negative_slope=self.neg_factor)
        x = self.pool(x)
        x = F.leaky_relu(self.conv3(x), negative_slope=self.neg_factor)
        x = self.pool(x)
        x = F.leaky_relu(self.conv4(x), negative_slope=self.neg_factor)
        x = self.pool(x)
        x = self.flatten(x)
        x = F.leaky_relu(self.fc_enc1(x), negative_slope=self.neg_factor)
        x = F.leaky_relu(self.fc_enc2(x), negative_slope=self.neg_factor)

        z_mean = self.fc_enc3_mean(x)
        z_logvar = self.fc_enc3_logvar(x)

        return z_mean, z_logvar

    def decode(self, z):
        z = F.leaky_relu(self.fc_dec1(z), negative_slope=self.neg_factor)
        z = F.leaky_relu(self.fc_dec2(z), negative_slope=self.neg_factor)
        z = F.leaky_relu(self.fc_dec3(z), negative_slope=self.neg_factor)
        z = self.unflatten(z)
        z = F.leaky_relu(self.deconv1(z), negative_slope=self.neg_factor)
        z = F.leaky_relu(self.deconv2(z), negative_slope=self.neg_factor)
        z = F.leaky_relu(self.deconv3(z), negative_slope=self.neg_factor)
        x_hat = torch.sigmoid(self.deconv4(z))

        return x_hat

    def forward(self, x):
        z_mean, z_logvar = self.encode(x)
        z = self.reparameterize(z_mean, z_logvar)
        x_hat = self.decode(z)

        return x_hat, z_mean, z_logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std


if __name__ == "__main__":
    pass