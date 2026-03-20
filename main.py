import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

"""
main.py — Full Pipeline Orchestrator
======================================
Runs the complete project end-to-end:
  1. Train DC-GAN
  2. Train WGAN-GP
  3. Train VAE
  4. Evaluate FID and IS for all models
  5. Generate all paper figures
  6. Run ablation studies

Usage:
  python main.py                    # full pipeline
  python main.py --models dcgan     # train single model only
  python main.py --eval-only        # evaluation only (models must be trained)
  python main.py --ablation-only    # ablation only

Flags:
  --size 64 or --size 128           # override IMAGE_SIZE from config
"""

import os
import time
import argparse
import torch
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import cfg
from data.dataset import get_dataloader
from training.train_dcgan   import train_dcgan
from training.train_wgan_gp import train_wgan_gp
from training.train_vae     import train_vae
from evaluation.metrics import (evaluate_all_models,
                                  generate_real_samples_for_eval)
from evaluation.ablation    import (ablation_latent_dim, ablation_learning_rate,
                                     ablation_vae_beta, ablation_architecture)
from utils.visualize import (save_model_comparison_grid, save_metric_bar_chart,
                               save_combined_loss_comparison,
                               save_latent_interpolation)


def parse_args():
    parser = argparse.ArgumentParser(description="Galaxy Generative Model Comparison")
    parser.add_argument("--models",       default="all",
                        help="Models to train: all | dcgan | wgan_gp | vae")
    parser.add_argument("--eval-only",    action="store_true")
    parser.add_argument("--ablation-only",action="store_true")
    parser.add_argument("--no-ablation",  action="store_true",
                        help="Skip ablation study (saves ~30min)")
    parser.add_argument("--size",         type=int, default=None,
                        help="Override IMAGE_SIZE (64 or 128)")
    parser.add_argument("--epochs",       type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Apply CLI overrides ────────────────────────────────────────────────────
    image_size = args.size or cfg.IMAGE_SIZE
    num_epochs = args.epochs or cfg.NUM_EPOCHS

    print(f"\n{'#'*60}")
    print(f"  Galaxy Generative Model Comparison")
    print(f"  Resolution : {image_size}x{image_size}")
    print(f"  Device     : {cfg.DEVICE}")
    print(f"  Epochs     : {num_epochs}")
    print(f"{'#'*60}\n")

    pipeline_start = time.time()

    trained_models = {}
    all_losses     = {}

    # ── Phase 1: Training ──────────────────────────────────────────────────────
    if not args.eval_only and not args.ablation_only:

        if args.models in ("all", "dcgan"):
            g_losses, d_losses, G_dcgan, D_dcgan = train_dcgan(
                image_size=image_size, num_epochs=num_epochs, run_name="dcgan"
            )
            trained_models["DC-GAN"]  = {"model": G_dcgan, "type": "gan"}
            all_losses["DC-GAN"]      = {"Generator": g_losses, "Discriminator": d_losses}

        if args.models in ("all", "wgan_gp"):
            g_losses, c_losses, w_dists, G_wgan, C_wgan = train_wgan_gp(
                image_size=image_size, num_epochs=num_epochs, run_name="wgan_gp"
            )
            trained_models["WGAN-GP"] = {"model": G_wgan, "type": "gan"}
            all_losses["WGAN-GP"]     = {"Generator": g_losses, "Critic": c_losses,
                                          "Wasserstein Distance": w_dists}

        if args.models in ("all", "vae"):
            t_losses, r_losses, kl_losses, vae_model = train_vae(
                image_size=image_size, num_epochs=num_epochs, run_name="vae"
            )
            trained_models["VAE"]     = {"model": vae_model, "type": "vae"}
            all_losses["VAE"]         = {"Total Loss": t_losses,
                                          "Reconstruction": r_losses, "KL": kl_losses}

        # Save combined loss figure (for paper)
        if all_losses:
            save_combined_loss_comparison(
                all_losses,
                path=os.path.join(cfg.RESULTS_DIR, "all_models_losses.png")
            )

    # ── Phase 2: Load checkpoints (if eval-only) ──────────────────────────────
    if args.eval_only and not trained_models:
        print("[main] Loading latest checkpoints for evaluation...")
        trained_models = _load_latest_checkpoints(image_size)

    # ── Phase 3: Evaluation ───────────────────────────────────────────────────
    if trained_models:
        real_dir = os.path.join(cfg.RESULTS_DIR, "fid_real")

        # Save real images for FID reference (only needed once)
        if not os.path.exists(real_dir) or len(os.listdir(real_dir)) < cfg.FID_BATCH:
            loader = get_dataloader(image_size=image_size, batch_size=64,
                                    max_samples=cfg.MAX_SAMPLES)
            generate_real_samples_for_eval(loader, cfg.FID_BATCH, real_dir)

        # Compute FID & IS
        eval_results = evaluate_all_models(
            models_dict=trained_models,
            real_dir=real_dir,
            n_samples=cfg.FID_BATCH,
            latent_dim=cfg.LATENT_DIM,
            device=cfg.DEVICE,
        )

        # Bar chart for paper
        save_metric_bar_chart(
            eval_results,
            path=os.path.join(cfg.RESULTS_DIR, "fid_is_comparison.png")
        )

        # Visual comparison grid
        samples_dict = {}
        for name, info in trained_models.items():
            m = info["model"].to(cfg.DEVICE)
            m.eval()
            with torch.no_grad():
                if info["type"] == "gan":
                    z = torch.randn(16, cfg.LATENT_DIM, device=cfg.DEVICE)
                    imgs = m(z).cpu()
                else:
                    imgs = m.sample(16, cfg.DEVICE).cpu()
            samples_dict[name] = imgs

        save_model_comparison_grid(
            samples_dict,
            path=os.path.join(cfg.RESULTS_DIR, "model_comparison_grid.png")
        )

        # VAE latent interpolation
        if "VAE" in trained_models:
            save_latent_interpolation(
                trained_models["VAE"]["model"],
                device=cfg.DEVICE,
                path=os.path.join(cfg.RESULTS_DIR, "vae_latent_interpolation.png")
            )

    # ── Phase 4: Ablation Study ───────────────────────────────────────────────
    if not args.no_ablation or args.ablation_only:
        real_dir = os.path.join(cfg.RESULTS_DIR, "fid_real")
        ablation_results = {}

        print("\n[main] Running ablation studies...")
        ablation_results["latent_dim"]    = ablation_latent_dim(real_dir)
        ablation_results["learning_rate"] = ablation_learning_rate(real_dir)
        ablation_results["vae_beta"]      = ablation_vae_beta(real_dir)
        ablation_results["architecture"]  = ablation_architecture(real_dir)

        import json
        with open(os.path.join(cfg.RESULTS_DIR, "all_ablation_results.json"), "w") as f:
            json.dump(ablation_results, f, indent=2)
        print("\n✓ All ablation studies complete.")

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed_total = time.time() - pipeline_start
    hours   = int(elapsed_total // 3600)
    minutes = int((elapsed_total % 3600) // 60)

    print(f"\n{'#'*60}")
    print(f"  ✓ Pipeline complete!")
    print(f"  Total time  : {hours}h {minutes}m")
    print(f"  Results saved to: {cfg.RESULTS_DIR}/")
    print(f"  Samples  saved to: {cfg.SAMPLE_DIR}/")
    print(f"{'#'*60}\n")


def _load_latest_checkpoints(image_size):
    from models.dcgan   import DCGANGenerator
    from models.wgan_gp import WGANGenerator
    from models.vae     import VAE

    models   = {}
    ckpt_dir = cfg.CHECKPOINT_DIR

    for name, ModelClass, mtype in [
        ("DC-GAN",  DCGANGenerator, "gan"),
        ("WGAN-GP", WGANGenerator,  "gan"),
        ("VAE",     None,           "vae"),
    ]:
        prefix = "dcgan" if name == "DC-GAN" else ("wgan_gp" if name == "WGAN-GP" else "vae")
        ckpts  = sorted([f for f in os.listdir(ckpt_dir) if f.startswith(prefix)])

        if not ckpts:
            print(f"[main] No checkpoint found for {name}, skipping.")
            continue

        ckpt_path = os.path.join(ckpt_dir, ckpts[-1])
        ckpt      = torch.load(ckpt_path, map_location=cfg.DEVICE)
        print(f"[main] Loaded {name} from {ckpt_path}")

        if name == "VAE":
            model = VAE(image_size=image_size, latent_dim=cfg.VAE_LATENT).to(cfg.DEVICE)
            model.load_state_dict(ckpt["vae_state"])
        else:
            model = ModelClass(latent_dim=cfg.LATENT_DIM, image_size=image_size).to(cfg.DEVICE)
            model.load_state_dict(ckpt["G_state"])

        models[name] = {"model": model, "type": mtype}

    return models


if __name__ == "__main__":
    main()