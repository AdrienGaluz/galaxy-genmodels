"""
training/train_dcgan.py — DC-GAN Training Loop
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import cfg
from models.dcgan import DCGANGenerator, DCGANDiscriminator
from data.dataset import get_dataloader
from utils.visualize import save_image_grid, save_loss_curves


def train_dcgan(
    image_size  = cfg.IMAGE_SIZE,
    batch_size  = cfg.BATCH_SIZE,
    num_epochs  = cfg.NUM_EPOCHS,
    lr_g        = cfg.DCGAN_LR_G,
    lr_d        = cfg.DCGAN_LR_D,
    latent_dim  = cfg.LATENT_DIM,
    ngf         = cfg.NGF,
    ndf         = cfg.NDF,
    max_samples = cfg.MAX_SAMPLES,
    device      = cfg.DEVICE,
    save_dir    = cfg.CHECKPOINT_DIR,
    sample_dir  = cfg.SAMPLE_DIR,
    run_name    = "dcgan",
):
    print(f"\n{'='*60}")
    print(f"  Training DC-GAN | {image_size}x{image_size} | Device: {device}")
    print(f"{'='*60}")

    loader = get_dataloader(
        image_size=image_size, batch_size=batch_size,
        max_samples=max_samples, num_workers=cfg.NUM_WORKERS
    )

    G = DCGANGenerator(latent_dim=latent_dim, ngf=ngf, nc=cfg.NC,
                        image_size=image_size).to(device)
    D = DCGANDiscriminator(ndf=ndf, nc=cfg.NC,
                            image_size=image_size).to(device)

    criterion = nn.BCELoss()
    opt_G = optim.Adam(G.parameters(), lr=lr_g,
                        betas=(cfg.DCGAN_BETA1, cfg.DCGAN_BETA2))
    opt_D = optim.Adam(D.parameters(), lr=lr_d,
                        betas=(cfg.DCGAN_BETA1, cfg.DCGAN_BETA2))

    fixed_noise  = torch.randn(64, latent_dim, device=device)
    writer       = SummaryWriter(log_dir=f"runs/{run_name}")
    G_losses, D_losses = [], []
    real_label, fake_label = 1.0, 0.0

    print(f"Generator params    : {sum(p.numel() for p in G.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in D.parameters()):,}")
    print(f"Starting training for {num_epochs} epochs...\n")

    for epoch in range(1, num_epochs + 1):
        G.train(); D.train()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        t0 = time.time()

        for real_imgs, _ in loader:
            real_imgs = real_imgs.to(device)
            B = real_imgs.size(0)

            D.zero_grad()
            labels_real = torch.full((B,), real_label, device=device)
            out_real    = D(real_imgs)
            loss_d_real = criterion(out_real, labels_real)

            z         = torch.randn(B, latent_dim, device=device)
            fake_imgs = G(z)
            labels_fake = torch.full((B,), fake_label, device=device)
            out_fake    = D(fake_imgs.detach())
            loss_d_fake = criterion(out_fake, labels_fake)

            loss_D = loss_d_real + loss_d_fake
            loss_D.backward()
            opt_D.step()

            G.zero_grad()
            labels_gen = torch.full((B,), real_label, device=device)
            out_gen    = D(fake_imgs)
            loss_G     = criterion(out_gen, labels_gen)
            loss_G.backward()
            opt_G.step()

            epoch_g_loss += loss_G.item()
            epoch_d_loss += loss_D.item()

        avg_g = epoch_g_loss / len(loader)
        avg_d = epoch_d_loss / len(loader)
        G_losses.append(avg_g)
        D_losses.append(avg_d)

        elapsed = time.time() - t0
        print(f"Epoch [{epoch:3d}/{num_epochs}] "
              f"G_loss: {avg_g:.4f} | D_loss: {avg_d:.4f} | "
              f"Time: {elapsed:.1f}s")

        writer.add_scalar("Loss/Generator",     avg_g, epoch)
        writer.add_scalar("Loss/Discriminator", avg_d, epoch)

        if epoch % cfg.SAVE_EVERY == 0 or epoch == num_epochs:
            G.eval()
            with torch.no_grad():
                samples = G(fixed_noise).cpu()
            save_image_grid(
                samples,
                path=os.path.join(sample_dir, f"{run_name}_epoch{epoch:03d}.png"),
                nrow=8, title=f"DC-GAN Epoch {epoch}"
            )
            torch.save({
                "epoch":    epoch,
                "G_state":  G.state_dict(),
                "D_state":  D.state_dict(),
                "G_losses": G_losses,
                "D_losses": D_losses,
            }, os.path.join(save_dir, f"{run_name}_epoch{epoch:03d}.pt"))

    save_loss_curves(
        {"Generator": G_losses, "Discriminator": D_losses},
        path=os.path.join(cfg.RESULTS_DIR, f"{run_name}_losses.png"),
        title="DC-GAN Training Losses"
    )
    writer.close()
    print(f"\n✓ DC-GAN training complete.")
    return G_losses, D_losses, G, D


if __name__ == "__main__":
    train_dcgan()