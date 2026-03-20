"""
models/dcgan.py — Deep Convolutional GAN
"""
import torch
import torch.nn as nn


def _weights_init(m):
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class DCGANGenerator(nn.Module):
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
        self.apply(_weights_init)

    def forward(self, z):
        z = z.view(z.size(0), -1, 1, 1)
        return self.main(z)


class DCGANDiscriminator(nn.Module):
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
                nn.BatchNorm2d(channels[i+1]),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        layers += [
            nn.Conv2d(channels[-1], 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        ]
        self.main = nn.Sequential(*layers)
        self.apply(_weights_init)

    def forward(self, x):
        return self.main(x).view(-1)