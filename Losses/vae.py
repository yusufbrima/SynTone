import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import (log_density_gaussian, log_importance_weight_matrix,
                               matrix_log_density_gaussian)

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
        reconstruction_loss = F.mse_loss(recon_x, x, reduction='sum')
        # reconstruction_loss = F.l1_loss(recon_x, x, reduction='mean')

        sample_dim = x.shape[-1]
        kl_d_weight = 1 / sample_dim
        # KL divergence loss
        # kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_divergence_loss =  torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        
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

class BTCVAELoss(nn.Module):
    """Custom loss function for VAE using MSE and KL divergence."""
    
    def __init__(self, beta=1,alpha=1., gamma=1., n_data=1, is_mss=True):
        """
        Args:
            beta (float): Weight factor for KL loss term.
        """
        super(BTCVAELoss, self).__init__()
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.n_data = n_data
        self.is_mss = is_mss

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
        reconstruction_loss = F.mse_loss(recon_x, x, reduction='sum')
        # reconstruction_loss = F.l1_loss(recon_x, x, reduction='mean')

        latent_dist = (mu, logvar)
        latent_sample = mu + torch.randn_like(logvar) * torch.exp(0.5 * logvar)

        log_pz, log_qz, log_prod_qzi, log_q_zCx = _get_log_pz_qz_prodzi_qzCx(latent_sample,
                                                                             latent_dist,
                                                                             self.n_data,
                                                                             is_mss=self.is_mss)
        # I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
        mi_loss = (log_q_zCx - log_qz).mean()
        # TC[z] = KL[q(z)||\prod_i z_i]
        tc_loss = (log_qz - log_prod_qzi).mean()
        # dw_kl_loss is KL[q(z)||p(z)] instead of usual KL[q(z|x)||p(z))]
        dw_kl_loss = (log_prod_qzi - log_pz).mean()
        # total loss
        total_loss = reconstruction_loss + (self.alpha * mi_loss + self.beta * tc_loss + self.gamma * dw_kl_loss)

        
        return total_loss


def _get_log_pz_qz_prodzi_qzCx(latent_sample, latent_dist, n_data, is_mss=True):
    batch_size, hidden_dim = latent_sample.shape

    # calculate log q(z|x)
    log_q_zCx = log_density_gaussian(latent_sample, *latent_dist).sum(dim=1)

    # calculate log p(z)
    # mean and log var is 0
    zeros = torch.zeros_like(latent_sample)
    log_pz = log_density_gaussian(latent_sample, zeros, zeros).sum(1)

    mat_log_qz = matrix_log_density_gaussian(latent_sample, *latent_dist)

    if is_mss:
        # use stratification
        log_iw_mat = log_importance_weight_matrix(batch_size, n_data).to(latent_sample.device)
        mat_log_qz = mat_log_qz + log_iw_mat.view(batch_size, batch_size, 1)

    log_qz = torch.logsumexp(mat_log_qz.sum(2), dim=1, keepdim=False)
    log_prod_qzi = torch.logsumexp(mat_log_qz, dim=1, keepdim=False).sum(1)

    return log_pz, log_qz, log_prod_qzi, log_q_zCx