from pickletools import StackObject
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Probably there should be different files for different models.
"""


def mc_kl_divergence(z, z_mu, z_std):
    # stolen from https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed
    # --------------------------
    # Monte carlo KL divergence
    # --------------------------
    # 1. define the first two probabilities (in this case Normal for both)
    p = torch.distributions.Normal(torch.zeros_like(z_mu), torch.ones_like(z_std))
    q = torch.distributions.Normal(z_mu, z_std)

    # 2. get the probabilities from the equation
    log_qzx = q.log_prob(z)
    log_pz = p.log_prob(z)

    # kl
    kl = (log_qzx - log_pz)
    
    # sum over last dim to go from single dim distribution to multi-dim
    kl = kl.sum(-1)
    return kl
def kl_divergence(z_mu, z_log_var):
    z_var = torch.exp(z_log_var)
    kl = torch.mean(-0.5 * torch.sum(1 + z_log_var - z_mu ** 2 - z_var, dim = 1), dim = 0)
    return kl

class SimpleVariationalEncoder(nn.Module):
    def __init__(self, data_dim, latent_dim, device):
        super(SimpleVariationalEncoder, self).__init__()
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
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)
    
    def forward(self, x):
        #x = torch.flatten(x, start_dim=1)
        # x = torch.flatten(x, 1)
        x = self.seq(x)
        z_mu = self.fc_z_mu(x)
        z_log_var = self.fc_z_log_var(x)
        z_var = torch.exp(z_log_var)
        z_std = torch.exp(z_log_var/2)
        z = z_mu + z_std*self.N.sample(z_mu.shape)
        kl = kl_divergence(z_mu, z_log_var)
        return z, kl


class SimpleDecoder(nn.Module):
    def __init__(self, data_dim, latent_dim):
        super(SimpleDecoder, self).__init__()
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


class SimpleVariationalAutoencoder(nn.Module):
    def __init__(self, data_dim, latent_dim, device):
        super(SimpleVariationalAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.data_dim_flat = torch.prod(torch.as_tensor(data_dim))
        self.encoder = SimpleVariationalEncoder(data_dim, latent_dim, device)
        self.decoder = SimpleDecoder(data_dim, latent_dim, device)

    def forward(self, x):
        z, kl = self.encoder(x)
        return self.decoder(z), kl

class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, batch_norm=True, activation_fn=nn.ReLU):
        super().__init__()
        layers = [
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim) if batch_norm else nn.Identity(),
            activation_fn()
        ]
        self.seq = nn.Sequential(*layers)
    def forward(self, x):
        return self.seq(x)
class StackedLinearBlock(nn.Module):
    def __init__(self, dims, *args):
        super().__init__()
        layers = [LinearBlock(in_dim, out_dim, *args) for in_dim, out_dim in zip(dims[:-1], dims[1:])]
        self.seq = nn.Sequential(*layers)
    def forward(self, x):
        return self.seq(x)


class LinearVariationalEncoder(nn.Module):
    def __init__(self, data_dim, latent_dim, hidden_dims, device, *args):
        super().__init__()
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.data_dim_flat = torch.prod(torch.as_tensor(data_dim))

        self.flatten = nn.Flatten()

        dims = [self.data_dim_flat] + hidden_dims
        self.seq = StackedLinearBlock(dims, *args)

        last_hiddem_dim = hidden_dims[-1]
        self.fc_z_mu = nn.Linear(last_hiddem_dim, latent_dim)
        self.fc_z_log_var = nn.Linear(last_hiddem_dim, latent_dim)
        
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)
    
    def forward(self, x):
        #x = torch.flatten(x, start_dim=1)
        # x = torch.flatten(x, 1)
        x = self.flatten(x)
        x = self.seq(x)
        z_mu = self.fc_z_mu(x)
        z_log_var = self.fc_z_log_var(x)
        z_var = torch.exp(z_log_var)
        z_std = torch.exp(z_log_var/2)
        z = z_mu + z_std*self.N.sample(z_mu.shape)
        kl = kl_divergence(z_mu, z_log_var)
        return z, kl


class LinearDecoder(nn.Module):
    def __init__(self, data_dim, latent_dim, hidden_dims, *args):
        super().__init__()
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.data_dim_flat = torch.prod(torch.as_tensor(data_dim))
        
        dims = [latent_dim] + hidden_dims
        last_hiddem_dim = hidden_dims[-1]
        self.seq = nn.Sequential(
            StackedLinearBlock(dims, *args),
            nn.Linear(last_hiddem_dim, self.data_dim_flat),
            nn.Unflatten(1, self.data_dim)
        )
        
    def forward(self, z):
        z = self.seq(z)
        return z


class LinearVariationalAutoencoder(nn.Module):
    def __init__(self, data_dim, latent_dim, hidden_dims, device, *args):
        super().__init__()
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.data_dim_flat = torch.prod(torch.as_tensor(data_dim))
        self.encoder = LinearVariationalEncoder(data_dim, latent_dim, hidden_dims, device, *args)
        self.decoder = LinearDecoder(data_dim, latent_dim, list(reversed(hidden_dims)), *args)

    def forward(self, x):
        z, kl = self.encoder(x)
        return self.decoder(z), kl

def get_model(config, data_dim):
    if config.model_type == "simple_vae":
        model = SimpleVariationalAutoencoder(data_dim, config.latent_dim, config.device)
    
    elif config.model_type == "linear_vae":
        model = LinearVariationalAutoencoder(data_dim, config.latent_dim, config.hidden_dims, config.device)

    return model.to(config.device)
