import torch
import torch.nn as nn
import torch.nn.functional as F

class BetaVAELoss(nn.Module):
    """Custom loss function for VAE using MSE and KL divergence."""
    
    def __init__(self, beta=1):
        """
        Args:
            beta (float): Weight factor for KL loss term.
        """
        super(BetaVAELoss, self).__init__()
        self.beta = beta

    def forward(self, recon_x, x, mu, logvar):
        """
        Computes the VAE loss.
        
        Args:
            recon_x (Tensor): Reconstructed input
            x (Tensor): Original input
            mu (Tensor): Mean from the latent distribution
            logvar (Tensor): Log variance from the latent distribution
        
        Returns:
            loss (Tensor): Overall VAE loss
        """
        
        # Reconstruction loss (MSE)
        reconstruction_loss = F.mse_loss(recon_x, x, reduction='mean')
        
        # KL divergence loss
        kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Combine losses
        total_loss = reconstruction_loss + self.beta * kl_divergence_loss
        
        return total_loss


class VAELoss(nn.Module):
    """Alternative VAE loss without beta parameter."""

    def __init__(self):
        super(VAELoss, self).__init__()

    def forward(self, recon_x, x, mu, logvar):
        
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence loss
        kl_div = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
        
        # Total loss
        loss = recon_loss + 0.001 * kl_div
        
        return loss