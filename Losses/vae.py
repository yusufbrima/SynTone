import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import (log_density_gaussian, log_importance_weight_matrix,
                               matrix_log_density_gaussian)
from Models.discriminator import Discriminator

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


class FactorKLoss(nn.Module):
    """
    Factor-VAE loss implementation.

    Parameters
    ----------
    device : torch.device
        Device on which the model and loss computations should be performed.

    gamma : float, optional
        Weight of the TC loss term.

    latent_dim : int, optional
        Dimensionality of the latent space.

    optim_kwargs : dict, optional
        Additional arguments for the Adam optimizer.

    References
    ----------
    [1] Kim, Hyunjik, and Andriy Mnih. "Disentangling by factorising."
    arXiv preprint arXiv:1802.05983 (2018).
    """

    def __init__(self, device, gamma=10., latent_dim=10, optim_kwargs=dict(lr=5e-5, betas=(0.5, 0.9), t_max=10000)):
        super().__init__()
        self.gamma = gamma
        self.device = device
        self.latent_dim = latent_dim
        self.discriminator = Discriminator(latent_dim=self.latent_dim).to(self.device)
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), **optim_kwargs)

    def _permute_dims(self, latent_sample):
        """
        Permutes dimensions of the latent sample.

        Parameters
        ----------
        latent_sample : torch.Tensor
            Latent sample from the reparameterization trick.

        Returns
        -------
        torch.Tensor
            Permuted latent sample.
        """
        perm = torch.zeros_like(latent_sample)
        batch_size, dim_z = perm.size()

        for z in range(dim_z):
            pi = torch.randperm(batch_size).to(latent_sample.device)
            perm[:, z] = latent_sample[pi, z]
        
        return perm

    def _compute_vae_loss(self, data, model):
        """
        Compute the VAE loss.

        Parameters
        ----------
        data : torch.Tensor
            Input data.

        model : torch.nn.Module
            The VAE model.

        Returns
        -------
        torch.Tensor
            VAE loss.
        """
        recon_batch, mu, logvar = model(data)
        latent_sample = model.reparameterize(mu, logvar)
        rec_loss = F.mse_loss(recon_batch, data, reduction='sum')
        kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)

        return rec_loss, kl_loss, latent_sample

    def _compute_discriminator_loss(self, d_z, d_z_perm, half_batch_size):
        """
        Compute the discriminator loss.

        Parameters
        ----------
        d_z : torch.Tensor
            Output of the discriminator for the original latent sample.

        d_z_perm : torch.Tensor
            Output of the discriminator for the permuted latent sample.

        half_batch_size : int
            Half of the batch size.

        Returns
        -------
        torch.Tensor
            Discriminator loss.
        """
        ones = torch.ones(half_batch_size, dtype=torch.long, device=self.device)
        zeros = torch.zeros_like(ones)
        d_tc_loss = 0.5 * (F.cross_entropy(d_z, zeros) + F.cross_entropy(d_z_perm, ones))

        return d_tc_loss

    def forward(self, data, model, optimizer):
        """
        Compute and optimize the Factor-VAE loss.

        Parameters
        ----------
        data : torch.Tensor
            Input data.

        model : torch.nn.Module
            The VAE model.

        optimizer : torch.optim
            The optimizer for updating model parameters.

        Returns
        -------
        torch.Tensor
            Factor-VAE loss.
        """
        # Pre-processing steps
        batch_size = data.size(dim=0)
        half_batch_size = batch_size // 2
        data = data.split(half_batch_size)
        data1, data2 = data[0], data[1]

        # VAE Loss Calculation
        rec_loss, kl_loss, latent_sample1 = self._compute_vae_loss(data1, model)

        d_z = self.discriminator(latent_sample1)
        tc_loss = (d_z[:, 0] - d_z[:, 1]).mean()
        anneal_reg = 1
        vae_loss = rec_loss + kl_loss + anneal_reg * self.gamma * tc_loss

        # Return loss if not training
        if not model.training:
            return vae_loss

        # Backpropagation for VAE
        optimizer.zero_grad()
        vae_loss.backward(retain_graph=True)

        # Discriminator Loss Calculation
        latent_sample2 = model.sample_latent(data2)
        z_perm = self._permute_dims(latent_sample2).detach()
        d_z_perm = self.discriminator(z_perm)

        d_tc_loss = self._compute_discriminator_loss(d_z, d_z_perm, half_batch_size)

        # Backpropagation for Discriminator
        self.optimizer_d.zero_grad()
        d_tc_loss.backward()

        # Update parameters
        optimizer.step()
        self.optimizer_d.step()

        return vae_loss
