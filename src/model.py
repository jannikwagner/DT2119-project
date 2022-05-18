from pickletools import StackObject
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

"""
Probably there should be different files for different models.
"""

def get_model(config, data_dim):
    if config.model_type == "simple_vae":
        model = SimpleVariationalAutoencoder(data_dim, config.latent_dim, config.device)
    
    elif config.model_type == "linear_vae":
        model = LinearVariationalAutoencoder(data_dim, config.latent_dim, config.hidden_dims, config.device)

    elif config.model_type == "conv2d_vae":
        model = conv2d_builder(data_dim, config.latent_dim, config.device, config.channels, config.strides, config.kernel_sizes, config.paddings, config.nonlinearity, config.batch_norm)
    return model.to(config.device)


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
        self.fc_z_log_var = F.tanh(nn.Linear(512, latent_dim))
        
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
    def __init__(self, in_dim, out_dim, batch_norm=True, nonlinearity="relu"):
        super().__init__()
        nonlinearity = NONLINEARITIES[nonlinearity]
        layers = [
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim) if batch_norm else nn.Identity(),
            nonlinearity
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
        z_log_var = F.tanh(self.fc_z_log_var(x))
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


NONLINEARITIES = {
    "relu" : nn.ReLU(),
    "tanh" : nn.Tanh(),
    None : nn.Identity()
}
def split_list(x):
    # print(type(x), isinstance(x, list))
    x_h, x_w = x if isinstance(x, (list, tuple)) else (x, x)
    # print(x_h, x_w)
    return x_h, x_w
def update_h(h, p, k, s):
    return (h+2*p-(k-1)-1) // s + 1
def update_chw(h, w, c2, p, k, s):
    p_h, p_w = split_list(p)
    k_h, k_w = split_list(k)
    s_h, s_w = split_list(s)
    return c2, update_h(h, p_h, k_h, s_h), update_h(w, p_w, k_w, s_w)

class Conv2dStack(nn.Module):
    def __init__(self, data_dim, channels, strides=None, kernel_sizes=None, paddings=None, nonlinearity="relu", batch_norm=True):
        super().__init__()
        if strides is None: strides = [1] * len(channels)
        if kernel_sizes is None: kernel_sizes = [3] * len(channels)
        if paddings is None: paddings = [0] * len(channels)
        nonlinearity = NONLINEARITIES[nonlinearity]
        *_, c, h, w = data_dim
        print(c, h, w)

        layers = []
        for c2, k, s, p in zip(channels, kernel_sizes, strides, paddings):
            layers.append(nn.Conv2d(c, c2, k, s, p))
            layers.append(nonlinearity)
            if batch_norm:
                layers.append(nn.BatchNorm2d(c2))
            c, h, w = update_chw(h, w, c2, p, k, s)
            print(c, h, w)
        
        layers.append(nn.Flatten())

        self.layers = layers
        self.seq = nn.Sequential(*layers)

        self.dim = torch.prod(torch.as_tensor([c, h, w]))
    def forward(self, x):
        # for l in self.layers:
        #     x = l(x)
        #     print(x.shape)
        x = self.seq(x)
        return x

class TransposedConv2dStack(nn.Module):
    def __init__(self, data_dim, channels, strides=None, kernel_sizes=None, paddings=None, nonlinearity="relu", batch_norm=True):
        super().__init__()
        if strides is None: strides = [1] * len(channels)
        if kernel_sizes is None: kernel_sizes = [3] * len(channels)
        if paddings is None: paddings = [0] * len(channels)
        nonlinearity = NONLINEARITIES[nonlinearity]
        *_, c, h, w = data_dim
        print(c, h, w)

        layers = []
        for i, (c2, k, s, p) in enumerate(zip(channels, kernel_sizes, strides, paddings)):
            layers.append(nn.Upsample((h,w)))
            if i!= 0:
                if batch_norm:
                    layers.append(nn.BatchNorm2d(c))
                layers.append(nonlinearity)
            layers.append(nn.ConvTranspose2d(c2, c, k, s, p))
            c, h, w = update_chw(h, w, c2, p, k, s)
        
        layers.append(nn.Unflatten(1, (c, h, w)))

        layers = list(reversed(layers))

        self.layers = layers
        self.seq = nn.Sequential(*layers)

        self.dim = torch.prod(torch.as_tensor([c, h, w]))
    def forward(self, x):
        # for l in self.layers:
        #     x = l(x)
        #     print(x.shape)
        x = self.seq(x)
        return x

class Encoder(nn.Module):
    def __init__(self, stack, latent_dim):
        super().__init__()
        self.stack = stack
        dim = stack.dim

        self.fc_z_mu = nn.Linear(dim, latent_dim)
        self.fc_z_log_var = nn.Linear(dim, latent_dim)
    
    def forward(self, x):
        #x = torch.flatten(x, start_dim=1)
        # x = torch.flatten(x, 1)
        x = self.stack(x)
        z_mu = self.fc_z_mu(x)
        z_log_var = F.tanh(self.fc_z_log_var(x))
        return z_mu, z_log_var


class Decoder(nn.Module):
    def __init__(self, stack, latent_dim):
        super().__init__()
        dim = stack.dim
        self.fc = nn.Linear(latent_dim, dim)

        self.stack = stack
        
    def forward(self, z):
        z = self.fc(z)
        x_rec = self.stack(z)
        return x_rec


class VariationalAutoencoder(nn.Module):
    def __init__(self, encoder, decoder, latent_dim, device):
        super(VariationalAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = encoder
        self.decoder = decoder
        
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)

    def forward(self, x):
        z_mu, z_log_var = self.encoder(x)

        z_std = torch.exp(z_log_var/2)
        z = z_mu + z_std*self.N.sample(z_mu.shape)
        kl = kl_divergence(z_mu, z_log_var)

        x_rec = self.decoder(z)
        return x_rec, kl

    def sample(self, n):
        z_sample = self.N.sample((n, self.latent_dim))
        x_sample = self.decoder(z_sample)
        return x_sample

    def to(self, device, *args):
        old_device = self.N.loc.device
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)
        model = super().to(device, *args)

        self.N.loc = self.N.loc.to(old_device)
        self.N.scale = self.N.scale.to(old_device)
        return model
        
def conv2d_builder(data_dim, latent_dim, device, channels, strides=None, kernel_sizes=None, paddings=None, nonlinearity="relu", batch_norm=True):
    encoder_stack = Conv2dStack(data_dim, channels, strides, kernel_sizes, paddings, nonlinearity, batch_norm)
    decoder_stack = TransposedConv2dStack(data_dim, channels, strides, kernel_sizes, paddings, nonlinearity, batch_norm)
    encoder = Encoder(encoder_stack, latent_dim)
    decoder = Decoder(decoder_stack, latent_dim)
    vae = VariationalAutoencoder(encoder, decoder, latent_dim, device)
    return vae
