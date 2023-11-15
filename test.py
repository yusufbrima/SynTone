import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAE(nn.Module):

    def __init__(self, latent_dim):
        super().__init__()

        # Encoder
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        
        self.fc_enc1 = nn.Linear(32 * 128 * 43, 64)
        self.fc_enc2_mean = nn.Linear(64, latent_dim)
        self.fc_enc2_logvar = nn.Linear(64, latent_dim)

        # Decoder
        self.fc_dec1 = nn.Linear(latent_dim, 64)
        self.fc_dec2 = nn.Linear(64, 32 * 128 * 44)
        
        self.deconv2 = nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1)
        self.deconv1 = nn.ConvTranspose2d(32, 1, kernel_size=6, padding=2)

        self.neg_factor = 0.01

    def encode(self, x):
        # Encoding layers
        x = F.leaky_relu(self.conv1(x), negative_slope=self.neg_factor)
        x = self.pool1(x)
        x = F.leaky_relu(self.conv2(x), negative_slope=self.neg_factor)
        x = self.pool2(x)
        x = torch.flatten(x, start_dim=1)
        
        x = F.leaky_relu(self.fc_enc1(x), negative_slope=self.neg_factor)
        z_mean = self.fc_enc2_mean(x)
        z_logvar = self.fc_enc2_logvar(x)
        
        return z_mean, z_logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std 

    def decode(self, z):
        z = F.leaky_relu(self.fc_dec1(z), negative_slope=self.neg_factor)
        z = F.leaky_relu(self.fc_dec2(z), negative_slope=self.neg_factor)

        z = z.view(-1, 32, 128, 44)
        z = F.leaky_relu(self.deconv2(z), negative_slope=self.neg_factor)
        x_hat = torch.tanh(self.deconv1(z))
        return x_hat

    def forward(self, x):
        z_mean, z_logvar = self.encode(x)
        z = self.reparameterize(z_mean, z_logvar) 
        x_hat = self.decode(z)
        return x_hat, z_mean, z_logvar

if __name__ == "__main__":

    # Generate random of size torch.Size([64, 1, 513, 173])
    latent_dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X  = torch.randn(1, 1, 513, 173)

    cvae = CVAE(latent_dim).to(device)
    x_hat, z_mean, z_logvar = cvae(X.to(device))

    print(x_hat.shape)