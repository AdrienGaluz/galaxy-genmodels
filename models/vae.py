"""
models/vae.py — Variational Autoencoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEEncoder(nn.Module):
    def __init__(self, ndf=64, nc=3, latent_dim=128, image_size=64):
        super().__init__()
        self.latent_dim = latent_dim

        if image_size == 64:
            channels = [ndf, ndf*2, ndf*4, ndf*8]
        elif image_size == 128:
            channels = [ndf, ndf*2, ndf*4, ndf*8, ndf*8]
        else:  # 256
            channels = [ndf, ndf*2, ndf*4, ndf*8, ndf*8, ndf*8]

        conv_layers = [
            nn.Conv2d(nc, channels[0], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for i in range(len(channels) - 1):
            conv_layers += [
                nn.Conv2d(channels[i], channels[i+1], 4, 2, 1, bias=False),
                nn.BatchNorm2d(channels[i+1]),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        self.conv = nn.Sequential(*conv_layers)
        self.flat_dim = channels[-1] * 4 * 4
        self.fc_mu     = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x):
        h      = self.conv(x)
        h_flat = h.view(h.size(0), -1)
        mu     = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar


class VAEDecoder(nn.Module):
    def __init__(self, ngf=64, nc=3, latent_dim=128, image_size=64):
        super().__init__()

        if image_size == 64:
            channels = [ngf*8, ngf*4, ngf*2, ngf]
        elif image_size == 128:
            channels = [ngf*8, ngf*8, ngf*4, ngf*2, ngf]
        else:  # 256
            channels = [ngf*8, ngf*8, ngf*8, ngf*4, ngf*2, ngf]

        self.fc_in = nn.Linear(latent_dim, channels[0] * 4 * 4)
        layers = []
        for i in range(len(channels) - 1):
            layers += [
                nn.ConvTranspose2d(channels[i], channels[i+1], 4, 2, 1, bias=False),
                nn.BatchNorm2d(channels[i+1]),
                nn.ReLU(True),
            ]
        layers += [
            nn.ConvTranspose2d(channels[-1], nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        ]
        self.main = nn.Sequential(*layers)
        self.channels0 = channels[0]

    def forward(self, z):
        h = self.fc_in(z)
        h = h.view(h.size(0), self.channels0, 4, 4)
        return self.main(h)


class VAE(nn.Module):
    def __init__(self, ndf=64, ngf=64, nc=3, latent_dim=128,
                 image_size=64, beta=1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta       = beta
        self.encoder = VAEEncoder(ndf=ndf, nc=nc, latent_dim=latent_dim,
                                   image_size=image_size)
        self.decoder = VAEDecoder(ngf=ngf, nc=nc, latent_dim=latent_dim,
                                   image_size=image_size)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z          = self.reparameterize(mu, logvar)
        recon      = self.decoder(z)
        return recon, mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def sample(self, n_samples, device):
        z = torch.randn(n_samples, self.latent_dim, device=device)
        return self.decode(z)


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(recon_x, x, reduction="sum") / x.size(0)
    kl_loss    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    total      = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss