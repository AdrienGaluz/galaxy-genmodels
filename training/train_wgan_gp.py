"""
training/train_wgan_gp.py — WGAN-GP Training Loop
"""
import os
import time
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import cfg
from models.wgan_gp import WGANGenerator, WGANCritic, compute_gradient_penalty
from data.dataset import get_dataloader
from utils.visualize import save_image_grid, save_loss_curves


def train_wgan_gp(
    image_size  = cfg.IMAGE_SIZE,
    batch_size  = cfg.BATCH_SIZE,
    num_epochs  = cfg.NUM_EPOCHS,
    lr          = cfg.WGAN_LR,
    latent_dim  = cfg.LATENT_DIM,
    ngf         = cfg.NGF,
    ndf         = cfg.NDF,
    lambda_gp   = cfg.WGAN_LAMBDA,
    n_critic    = cfg.WGAN_CRITIC,
    max_samples = cfg.MAX_SAMPLES,
    device      = cfg.DEVICE,
    save_dir    = cfg.CHECKPOINT_DIR,
    sample_dir  = cfg.SAMPLE_DIR,
    run_name    = "wgan_gp",
):
    print(f"\n{'='*60}")
    print(f"  Training WGAN-GP | {image_size}x{image_size} | Device: {device}")
    print(f"{'='*60}")

    loader = get_dataloader(
        image_size=image_size, batch_size=batch_size,
        max_samples=max_samples, num_workers=cfg.NUM_WORKERS
    )

    G = WGANGenerator(latent_dim=latent_dim, ngf=ngf, nc=cfg.NC,
                       image_size=image_size).to(device)
    C = WGANCritic(ndf=ndf, nc=cfg.NC, image_size=image_size).to(device)

    opt_G = optim.Adam(G.parameters(), lr=lr,
                        betas=(cfg.WGAN_BETA1, cfg.WGAN_BETA2))
    opt_C = optim.Adam(C.parameters(), lr=lr,
                        betas=(cfg.WGAN_BETA1, cfg.WGAN_BETA2))

    fixed_noise = torch.randn(64, latent_dim, device=device)
    writer      = SummaryWriter(log_dir=f"runs/{run_name}")
    G_losses, C_losses, W_distances = [], [], []

    print(f"Generator params: {sum(p.numel() for p in G.parameters()):,}")
    print(f"Critic params   : {sum(p.numel() for p in C.parameters()):,}")
    print(f"Starting training for {num_epochs} epochs...\n")

    for epoch in range(1, num_epochs + 1):
        G.train(); C.train()
        epoch_g_loss = 0.0
        epoch_c_loss = 0.0
        epoch_w_dist = 0.0
        t0 = time.time()

        data_iter = iter(loader)
        n_batches = len(loader) // (n_critic + 1)

        for _ in range(n_batches):
            c_loss_accum = 0.0
            for _ in range(n_critic):
                try:
                    real_imgs, _ = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    real_imgs, _ = next(data_iter)

                real_imgs = real_imgs.to(device)
                B = real_imgs.size(0)

                C.zero_grad()
                z         = torch.randn(B, latent_dim, device=device)
                fake_imgs = G(z).detach()

                score_real = C(real_imgs)
                score_fake = C(fake_imgs)
                w_dist     = score_real.mean() - score_fake.mean()
                gp         = compute_gradient_penalty(C, real_imgs, fake_imgs, device)
                loss_C     = -w_dist + lambda_gp * gp
                loss_C.backward()
                opt_C.step()

                c_loss_accum += loss_C.item()
                epoch_w_dist += w_dist.item()

            epoch_c_loss += c_loss_accum / n_critic

            try:
                real_imgs, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                real_imgs, _ = next(data_iter)

            B = real_imgs.size(0)
            G.zero_grad()
            z         = torch.randn(B, latent_dim, device=device)
            fake_imgs = G(z)
            loss_G    = -C(fake_imgs).mean()
            loss_G.backward()
            opt_G.step()
            epoch_g_loss += loss_G.item()

        avg_g = epoch_g_loss / n_batches
        avg_c = epoch_c_loss / n_batches
        avg_w = epoch_w_dist / (n_batches * n_critic)
        G_losses.append(avg_g)
        C_losses.append(avg_c)
        W_distances.append(avg_w)

        elapsed = time.time() - t0
        print(f"Epoch [{epoch:3d}/{num_epochs}] "
              f"G_loss: {avg_g:.4f} | C_loss: {avg_c:.4f} | "
              f"W_dist: {avg_w:.4f} | Time: {elapsed:.1f}s")

        writer.add_scalar("Loss/Generator",     avg_g, epoch)
        writer.add_scalar("Loss/Critic",         avg_c, epoch)
        writer.add_scalar("Metrics/Wasserstein", avg_w, epoch)

        if epoch % cfg.SAVE_EVERY == 0 or epoch == num_epochs:
            G.eval()
            with torch.no_grad():
                samples = G(fixed_noise).cpu()
            save_image_grid(
                samples,
                path=os.path.join(sample_dir, f"{run_name}_epoch{epoch:03d}.png"),
                nrow=8, title=f"WGAN-GP Epoch {epoch}"
            )
            torch.save({
                "epoch":       epoch,
                "G_state":     G.state_dict(),
                "C_state":     C.state_dict(),
                "G_losses":    G_losses,
                "C_losses":    C_losses,
                "W_distances": W_distances,
            }, os.path.join(save_dir, f"{run_name}_epoch{epoch:03d}.pt"))

    save_loss_curves(
        {"Generator": G_losses, "Critic": C_losses,
         "Wasserstein Distance": W_distances},
        path=os.path.join(cfg.RESULTS_DIR, f"{run_name}_losses.png"),
        title="WGAN-GP Training Losses"
    )
    writer.close()
    print(f"\n✓ WGAN-GP training complete.")
    return G_losses, C_losses, W_distances, G, C


if __name__ == "__main__":
    train_wgan_gp()