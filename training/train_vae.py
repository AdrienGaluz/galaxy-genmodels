"""
training/train_vae.py — VAE Training Loop
"""
import os
import time
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import cfg
from models.vae import VAE, vae_loss
from data.dataset import get_dataloader
from utils.visualize import save_image_grid, save_loss_curves, save_vae_reconstructions


def train_vae(
    image_size  = cfg.IMAGE_SIZE,
    batch_size  = cfg.BATCH_SIZE,
    num_epochs  = cfg.NUM_EPOCHS,
    lr          = cfg.VAE_LR,
    latent_dim  = cfg.VAE_LATENT,
    ngf         = cfg.NGF,
    ndf         = cfg.NDF,
    beta        = cfg.VAE_BETA,
    max_samples = cfg.MAX_SAMPLES,
    device      = cfg.DEVICE,
    save_dir    = cfg.CHECKPOINT_DIR,
    sample_dir  = cfg.SAMPLE_DIR,
    run_name    = "vae",
):
    print(f"\n{'='*60}")
    print(f"  Training VAE | {image_size}x{image_size} | β={beta} | Device: {device}")
    print(f"{'='*60}")

    loader = get_dataloader(
        image_size=image_size, batch_size=batch_size,
        max_samples=max_samples, num_workers=cfg.NUM_WORKERS
    )

    vae = VAE(ndf=ndf, ngf=ngf, nc=cfg.NC, latent_dim=latent_dim,
              image_size=image_size, beta=beta).to(device)

    optimizer = optim.Adam(vae.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    fixed_batch, _ = next(iter(loader))
    fixed_batch    = fixed_batch[:16].to(device)
    fixed_noise    = torch.randn(64, latent_dim, device=device)
    writer         = SummaryWriter(log_dir=f"runs/{run_name}")

    total_losses, recon_losses, kl_losses = [], [], []

    print(f"VAE params   : {sum(p.numel() for p in vae.parameters()):,}")
    print(f"Latent dim   : {latent_dim}")
    print(f"Beta (KL wt) : {beta}")
    print(f"Starting training for {num_epochs} epochs...\n")

    for epoch in range(1, num_epochs + 1):
        vae.train()
        epoch_total = 0.0
        epoch_recon = 0.0
        epoch_kl    = 0.0
        t0 = time.time()

        for real_imgs, _ in loader:
            real_imgs = real_imgs.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = vae(real_imgs)
            loss, rl, kl = vae_loss(recon, real_imgs, mu, logvar, beta=beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_total += loss.item()
            epoch_recon += rl.item()
            epoch_kl    += kl.item()

        avg_total = epoch_total / len(loader)
        avg_recon = epoch_recon / len(loader)
        avg_kl    = epoch_kl    / len(loader)
        total_losses.append(avg_total)
        recon_losses.append(avg_recon)
        kl_losses.append(avg_kl)
        scheduler.step(avg_total)

        elapsed = time.time() - t0
        print(f"Epoch [{epoch:3d}/{num_epochs}] "
              f"Total: {avg_total:.2f} | Recon: {avg_recon:.2f} | "
              f"KL: {avg_kl:.4f} | Time: {elapsed:.1f}s")

        writer.add_scalar("Loss/Total",          avg_total, epoch)
        writer.add_scalar("Loss/Reconstruction", avg_recon, epoch)
        writer.add_scalar("Loss/KL",             avg_kl,   epoch)

        if epoch % cfg.SAVE_EVERY == 0 or epoch == num_epochs:
            vae.eval()
            with torch.no_grad():
                samples = vae.sample(64, device).cpu()
                save_image_grid(
                    samples,
                    path=os.path.join(sample_dir, f"{run_name}_samples_epoch{epoch:03d}.png"),
                    nrow=8, title=f"VAE Samples Epoch {epoch}"
                )
                recon_imgs, _, _ = vae(fixed_batch)
                save_vae_reconstructions(
                    original=fixed_batch.cpu(),
                    reconstructed=recon_imgs.cpu(),
                    path=os.path.join(sample_dir, f"{run_name}_recon_epoch{epoch:03d}.png"),
                    title=f"VAE Reconstructions Epoch {epoch}"
                )
            torch.save({
                "epoch":        epoch,
                "vae_state":    vae.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "total_losses": total_losses,
                "recon_losses": recon_losses,
                "kl_losses":    kl_losses,
            }, os.path.join(save_dir, f"{run_name}_epoch{epoch:03d}.pt"))

    save_loss_curves(
        {"Total Loss": total_losses, "Reconstruction": recon_losses, "KL": kl_losses},
        path=os.path.join(cfg.RESULTS_DIR, f"{run_name}_losses.png"),
        title="VAE Training Losses"
    )
    writer.close()
    print(f"\n✓ VAE training complete.")
    return total_losses, recon_losses, kl_losses, vae


if __name__ == "__main__":
    train_vae()