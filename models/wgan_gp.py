"""
models/wgan_gp.py — Wasserstein GAN with Gradient Penalty
"""
import torch
import torch.nn as nn


class WGANGenerator(nn.Module):
    def __init__(self, latent_dim=100, ngf=64, nc=3, image_size=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        layers = [
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
        ]

        if image_size == 64:
            channels = [ngf * 8, ngf * 4, ngf * 2, ngf]
        elif image_size == 128:
            channels = [ngf * 8, ngf * 8, ngf * 4, ngf * 2, ngf]
        else:  # 256
            channels = [ngf * 8, ngf * 8, ngf * 8, ngf * 4, ngf * 2, ngf]

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
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, z):
        z = z.view(z.size(0), -1, 1, 1)
        return self.main(z)


class WGANCritic(nn.Module):
    def __init__(self, ndf=64, nc=3, image_size=64):
        super().__init__()
        self.image_size = image_size

        layers = [
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        if image_size == 64:
            channels = [ndf, ndf * 2, ndf * 4, ndf * 8]
        elif image_size == 128:
            channels = [ndf, ndf * 2, ndf * 4, ndf * 8, ndf * 8]
        else:  # 256
            channels = [ndf, ndf * 2, ndf * 4, ndf * 8, ndf * 8, ndf * 8]

        for i in range(len(channels) - 1):
            layers += [
                nn.Conv2d(channels[i], channels[i+1], 4, 2, 1, bias=False),
                nn.InstanceNorm2d(channels[i+1], affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        layers += [
            nn.Conv2d(channels[-1], 1, 4, 1, 0, bias=False),
        ]
        self.main = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)

    def forward(self, x):
        return self.main(x).view(-1)


def compute_gradient_penalty(critic, real, fake, device):
    B = real.size(0)
    alpha = torch.rand(B, 1, 1, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interpolated = critic(interpolated)
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(B, -1)
    grad_norm = gradients.norm(2, dim=1)
    penalty = ((grad_norm - 1) ** 2).mean()
    return penalty