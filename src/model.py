import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config

"""
Probably there should be different files for different models.
"""


def mc_kl_divergence(self, z, mu, std):
    # stolen from https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed
    # --------------------------
    # Monte carlo KL divergence
    # --------------------------
    # 1. define the first two probabilities (in this case Normal for both)
    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    q = torch.distributions.Normal(mu, std)

    # 2. get the probabilities from the equation
    log_qzx = q.log_prob(z)
    log_pz = p.log_prob(z)

    # kl
    kl = (log_qzx - log_pz)
    
    # sum over last dim to go from single dim distribution to multi-dim
    kl = kl.sum(-1)
    return kl

class VariationalEncoder(nn.Module):
    def __init__(self, data_dim, latent_dim):
        super(VariationalEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        print(data_dim)
        self.data_dim_flat = torch.prod(torch.as_tensor(data_dim))
        print(self.data_dim_flat)

        layers = [
            torch.nn.Flatten(),
            nn.Linear(self.data_dim_flat, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        ]
        self.seq = nn.Sequential(*layers)

        self.fc_z_mu = nn.Linear(512, latent_dim)
        self.fc_z_log_var = nn.Linear(512, latent_dim)
        
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(config.device)
        self.N.scale = self.N.scale.to(config.device)
    
    def forward(self, x):
        #x = torch.flatten(x, start_dim=1)
        # x = torch.flatten(x, 1)
        x = self.seq(x)
        z_mu = self.fc_z_mu(x)
        z_log_var = self.fc_z_log_var(x)
        z_var = torch.exp(z_log_var)
        z_std = torch.exp(z_log_var/2)
        z = z_mu + z_std*self.N.sample(z_mu.shape)
        kl = torch.mean(-0.5 * torch.sum(1 + z_log_var - z_mu ** 2 - z_var, dim = 1), dim = 0)
        return z, kl


class Decoder(nn.Module):
    def __init__(self, data_dim, latent_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.data_dim_flat = torch.prod(torch.as_tensor(data_dim))
        layers = [
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, self.data_dim_flat),
            torch.nn.Unflatten(1, self.data_dim)
        ]
        self.seq = nn.Sequential(*layers)
        
    def forward(self, z):
        z = self.seq(z)
        return z


class VariationalAutoencoder(nn.Module):
    def __init__(self, data_dim, latent_dim):
        super(VariationalAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.data_dim_flat = torch.prod(torch.as_tensor(data_dim))
        self.encoder = VariationalEncoder(data_dim, latent_dim)
        self.decoder = Decoder(data_dim, latent_dim)

    def forward(self, x):
        z, kl = self.encoder(x)
        return self.decoder(z), kl

