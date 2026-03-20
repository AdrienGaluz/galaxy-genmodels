"""
evaluation/ablation.py — Hyperparameter Ablation Study
"""
import os
import json
import torch
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg
from data.dataset import get_dataloader
from evaluation.metrics import (generate_samples_for_eval,
                                  generate_real_samples_for_eval,
                                  compute_fid_is)

ABLATION_EPOCHS = 20


def _quick_train_dcgan(latent_dim=100, lr_g=2e-4, lr_d=2e-4, ngf=64,
                        epochs=ABLATION_EPOCHS, tag=""):
    from training.train_dcgan import train_dcgan
    G_losses, D_losses, G, D = train_dcgan(
        num_epochs=epochs, lr_g=lr_g, lr_d=lr_d,
        latent_dim=latent_dim, ngf=ngf,
        run_name=f"ablation_dcgan_{tag}"
    )
    return G, "gan", latent_dim


def _quick_train_vae(latent_dim=128, lr=1e-3, ngf=64, beta=1.0,
                      epochs=ABLATION_EPOCHS, tag=""):
    from training.train_vae import train_vae
    _, _, _, vae = train_vae(
        num_epochs=epochs, lr=lr,
        latent_dim=latent_dim, ngf=ngf, beta=beta,
        run_name=f"ablation_vae_{tag}"
    )
    return vae, "vae", latent_dim


def ablation_latent_dim(real_dir, results_dir="./results/ablation"):
    print("\n" + "="*60)
    print("  ABLATION 1: Latent Dimension")
    print("="*60)
    os.makedirs(results_dir, exist_ok=True)
    dims    = cfg.ABLATION_LATENT
    results = {}

    for dim in dims:
        tag = f"latent{dim}"
        print(f"\n── Latent dim = {dim} ──────────────")
        G, mtype, ld = _quick_train_dcgan(latent_dim=dim, tag=f"dcgan_{tag}")
        fake_dir = os.path.join(results_dir, f"dcgan_{tag}_fake")
        generate_samples_for_eval(G, mtype, 500, ld, cfg.DEVICE, fake_dir)
        results[f"DCGAN_latent{dim}"] = compute_fid_is(real_dir, fake_dir, cfg.DEVICE)

        vae, mtype, ld = _quick_train_vae(latent_dim=dim, tag=tag)
        fake_dir = os.path.join(results_dir, f"vae_{tag}_fake")
        generate_samples_for_eval(vae, mtype, 500, ld, cfg.DEVICE, fake_dir)
        results[f"VAE_latent{dim}"] = compute_fid_is(real_dir, fake_dir, cfg.DEVICE)

    out = os.path.join(results_dir, "ablation_latent_dim.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Latent dim ablation saved → {out}")
    return results


def ablation_learning_rate(real_dir, results_dir="./results/ablation"):
    print("\n" + "="*60)
    print("  ABLATION 2: Learning Rate")
    print("="*60)
    os.makedirs(results_dir, exist_ok=True)
    lrs     = cfg.ABLATION_LR
    results = {}

    for lr in lrs:
        tag = f"lr{lr:.0e}"
        print(f"\n── LR = {lr} ──────────────")
        G, mtype, ld = _quick_train_dcgan(lr_g=lr, lr_d=lr, tag=f"dcgan_{tag}")
        fake_dir = os.path.join(results_dir, f"dcgan_{tag}_fake")
        generate_samples_for_eval(G, mtype, 500, ld, cfg.DEVICE, fake_dir)
        results[f"DCGAN_lr{lr}"] = compute_fid_is(real_dir, fake_dir, cfg.DEVICE)

    out = os.path.join(results_dir, "ablation_lr.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Learning rate ablation saved → {out}")
    return results


def ablation_vae_beta(real_dir, results_dir="./results/ablation"):
    print("\n" + "="*60)
    print("  ABLATION 3: VAE Beta (KL weight)")
    print("="*60)
    os.makedirs(results_dir, exist_ok=True)
    betas   = cfg.ABLATION_BETA_VAE
    results = {}

    for beta in betas:
        tag = f"beta{beta}"
        print(f"\n── Beta = {beta} ──────────────")
        vae, mtype, ld = _quick_train_vae(beta=beta, tag=tag)
        fake_dir = os.path.join(results_dir, f"vae_{tag}_fake")
        generate_samples_for_eval(vae, mtype, 500, ld, cfg.DEVICE, fake_dir)
        results[f"VAE_beta{beta}"] = compute_fid_is(real_dir, fake_dir, cfg.DEVICE)

    out = os.path.join(results_dir, "ablation_vae_beta.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ VAE beta ablation saved → {out}")
    return results


def ablation_architecture(real_dir, results_dir="./results/ablation"):
    print("\n" + "="*60)
    print("  ABLATION 4: Architecture Depth (NGF/NDF)")
    print("="*60)
    os.makedirs(results_dir, exist_ok=True)
    configs = {"shallow": 32, "default": 64}
    results = {}

    for name, ngf in configs.items():
        tag = f"ngf{ngf}"
        print(f"\n── NGF = {ngf} ({name}) ──────────────")
        G, mtype, ld = _quick_train_dcgan(ngf=ngf, tag=f"dcgan_{tag}")
        fake_dir = os.path.join(results_dir, f"dcgan_{tag}_fake")
        generate_samples_for_eval(G, mtype, 500, ld, cfg.DEVICE, fake_dir)
        results[f"DCGAN_{name}"] = compute_fid_is(real_dir, fake_dir, cfg.DEVICE)

    out = os.path.join(results_dir, "ablation_architecture.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Architecture ablation saved → {out}")
    return results


def print_ablation_summary(results_dict, title="Ablation Results"):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  {'Config':<25} {'FID ↓':>10}  {'IS ↑':>12}")
    print(f"  {'-'*50}")
    for name, r in results_dict.items():
        fid = f"{r['fid']:.2f}" if r.get('fid') is not None else "N/A"
        isc = f"{r['inception_score_mean']:.2f}" if r.get('inception_score_mean') is not None else "N/A"
        print(f"  {name:<25} {fid:>10}  {isc:>12}")


if __name__ == "__main__":
    real_dir = "./results/fid_real"
    ablation_latent_dim(real_dir)
    ablation_learning_rate(real_dir)
    ablation_vae_beta(real_dir)
    ablation_architecture(real_dir)