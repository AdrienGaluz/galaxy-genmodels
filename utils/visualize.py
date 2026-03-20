"""
utils/visualize.py — Visualization Utilities
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def _to_numpy_grid(tensor, nrow=8):
    grid = make_grid(tensor.clamp(-1, 1), nrow=nrow, normalize=True,
                     value_range=(-1, 1), padding=2)
    return grid.permute(1, 2, 0).numpy()


def save_image_grid(images, path, nrow=8, title=""):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    grid_np = _to_numpy_grid(images[:nrow*nrow], nrow=nrow)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(grid_np)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=14, pad=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[visualize] Saved grid → {path}")


def save_vae_reconstructions(original, reconstructed, path, title="", n=8):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    orig  = original[:n].clamp(-1, 1)
    recon = reconstructed[:n].clamp(-1, 1)
    combined = torch.cat([orig, recon], dim=0)
    grid_np = _to_numpy_grid(combined, nrow=n)
    fig, ax = plt.subplots(figsize=(n * 1.5, 3))
    ax.imshow(grid_np)
    ax.axis("off")
    ax.set_title(title or "Top: Original | Bottom: Reconstructed", fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[visualize] Saved reconstructions → {path}")


def save_loss_curves(losses_dict, path, title="Training Losses"):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, values in losses_dict.items():
        ax.plot(range(1, len(values) + 1), values, label=label, linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[visualize] Saved loss curves → {path}")


def save_combined_loss_comparison(all_losses, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
    for ax, (name, losses) in zip(axes, all_losses.items()):
        for i, (label, values) in enumerate(losses.items()):
            ax.plot(range(1, len(values) + 1), values,
                    label=label, color=colors[i % len(colors)], linewidth=2)
        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Training Loss Curves — All Models", fontsize=15,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[visualize] Saved combined losses → {path}")


def save_latent_interpolation(vae, device, path, n_steps=10):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    vae.eval()
    with torch.no_grad():
        z1 = torch.randn(1, vae.latent_dim, device=device)
        z2 = torch.randn(1, vae.latent_dim, device=device)
        alphas = torch.linspace(0, 1, n_steps, device=device)
        imgs = []
        for alpha in alphas:
            z = (1 - alpha) * z1 + alpha * z2
            img = vae.decode(z).cpu()
            imgs.append(img)
        imgs = torch.cat(imgs, dim=0)
    grid_np = _to_numpy_grid(imgs, nrow=n_steps)
    fig, ax = plt.subplots(figsize=(n_steps * 1.5, 2))
    ax.imshow(grid_np)
    ax.axis("off")
    ax.set_title("VAE Latent Space Interpolation", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[visualize] Saved interpolation → {path}")


def save_model_comparison_grid(samples_dict, path, n_per_model=16):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    n_models = len(samples_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(n_models * 6, 6))
    if n_models == 1:
        axes = [axes]
    for ax, (name, images) in zip(axes, samples_dict.items()):
        nrow = int(n_per_model ** 0.5)
        grid_np = _to_numpy_grid(images[:n_per_model], nrow=nrow)
        ax.imshow(grid_np)
        ax.axis("off")
        ax.set_title(name, fontsize=14, fontweight="bold", pad=8)
    fig.suptitle("Generated Galaxy Images — Model Comparison",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[visualize] Saved model comparison → {path}")


def save_metric_bar_chart(results_dict, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    models   = list(results_dict.keys())
    fid_vals = [results_dict[m]["fid"] for m in models]
    is_vals  = [results_dict[m]["inception_score_mean"] for m in models]
    x     = np.arange(len(models))
    width = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    bars1 = ax1.bar(x, fid_vals, width,
                    color=["#2196F3", "#FF5722", "#4CAF50"],
                    edgecolor="black", linewidth=0.8)
    ax1.set_title("Fréchet Inception Distance (FID) ↓", fontsize=13,
                  fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=11)
    ax1.set_ylabel("FID Score")
    ax1.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars1, fid_vals):
        if val is not None:
            ax1.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.5,
                     f"{val:.1f}", ha="center", va="bottom", fontsize=10)
    bars2 = ax2.bar(x, is_vals, width,
                    color=["#2196F3", "#FF5722", "#4CAF50"],
                    edgecolor="black", linewidth=0.8)
    ax2.set_title("Inception Score (IS) ↑", fontsize=13, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=11)
    ax2.set_ylabel("IS Score")
    ax2.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars2, is_vals):
        if val is not None:
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.01,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=10)
    fig.suptitle("Quantitative Metrics — DC-GAN vs WGAN-GP vs VAE",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[visualize] Saved metric chart → {path}")