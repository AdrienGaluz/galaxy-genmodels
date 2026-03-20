import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import cfg
from models.dcgan import DCGANGenerator, DCGANDiscriminator
from models.wgan_gp import WGANGenerator, WGANCritic
from models.vae import VAE

device = cfg.DEVICE

G_dc = DCGANGenerator(image_size=64).to(device)
D_dc = DCGANDiscriminator(image_size=64).to(device)
z = torch.randn(4, 100, device=device)
out = G_dc(z)
print('DC-GAN Generator output :', out.shape)
print('DC-GAN Discriminator    :', D_dc(out).shape)

G_wg = WGANGenerator(image_size=64).to(device)
C_wg = WGANCritic(image_size=64).to(device)
out2 = G_wg(z)
print('WGAN-GP Generator output:', out2.shape)
print('WGAN-GP Critic          :', C_wg(out2).shape)

vae = VAE(image_size=64).to(device)
x = torch.randn(4, 3, 64, 64, device=device)
recon, mu, logvar = vae(x)
print('VAE reconstruction      :', recon.shape)
print('VAE mu                  :', mu.shape)

print('')
print('All 3 models running on', device)