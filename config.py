import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

"""
config.py — Central configuration for Galaxy Generative Model Comparison
==========================================================================
RESOLUTION UPGRADE PATH:
  - Laptop (RTX 4070, 8GB VRAM)  → set IMAGE_SIZE = 64
  - PC     (RTX 4080S, 16GB VRAM) → set IMAGE_SIZE = 128 or 256

Change ONLY the IMAGE_SIZE line below. Everything else auto-adjusts.
"""

import torch

# ─────────────────────────────────────────────
#  ★  CHANGE THIS LINE TO SWITCH RESOLUTION  ★
IMAGE_SIZE = 256       # 64 = laptop | 128 = standard | 256 = max quality
# ─────────────────────────────────────────────

class Config:
    # ── Resolution ───────────────────────────
    IMAGE_SIZE   = IMAGE_SIZE
    NC           = 3

    # ── Dataset ──────────────────────────────
    DATA_DIR     = "./data"
    NUM_WORKERS  = 0
    MAX_SAMPLES  = 17736                      # full dataset

    # ── Training ─────────────────────────────
    BATCH_SIZE   = (64 if IMAGE_SIZE == 64
                    else 32 if IMAGE_SIZE == 128
                    else 16)                  # 16 for 256×256
    NUM_EPOCHS   = 50

    # ── Latent space ─────────────────────────
    LATENT_DIM   = 100
    VAE_LATENT   = 128

    # ── Architecture widths ──────────────────
    NGF = 64
    NDF = 64

    # ── DC-GAN hyperparameters ───────────────
    DCGAN_LR_G   = 2e-4
    DCGAN_LR_D   = 1e-5
    DCGAN_BETA1  = 0.5
    DCGAN_BETA2  = 0.999

    # ── WGAN-GP hyperparameters ──────────────
    WGAN_LR      = 1e-4
    WGAN_BETA1   = 0.0
    WGAN_BETA2   = 0.9
    WGAN_LAMBDA  = 10
    WGAN_CRITIC  = 5

    # ── VAE hyperparameters ──────────────────
    VAE_LR       = 1e-3
    VAE_BETA     = 1.0

    # ── Paths ─────────────────────────────────
    CHECKPOINT_DIR = "./checkpoints"
    RESULTS_DIR    = "./results"
    SAMPLE_DIR     = "./samples"

    # ── Device ───────────────────────────────
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Evaluation ───────────────────────────
    FID_BATCH     = 500
    EVAL_EVERY    = 10
    SAVE_EVERY    = 10

    # ── Ablation grid ─────────────────────────
    ABLATION_LR       = [1e-3, 2e-4, 1e-4]
    ABLATION_LATENT   = [64, 100, 256]
    ABLATION_BETA_VAE = [0.5, 1.0, 2.0]

# Create output directories on import
for d in [Config.CHECKPOINT_DIR, Config.RESULTS_DIR,
          Config.SAMPLE_DIR, Config.DATA_DIR]:
    os.makedirs(d, exist_ok=True)

cfg = Config()

if __name__ == "__main__":
    print(f"Device      : {cfg.DEVICE}")
    print(f"Image size  : {cfg.IMAGE_SIZE}x{cfg.IMAGE_SIZE}")
    print(f"Batch size  : {cfg.BATCH_SIZE}")
    print(f"Epochs      : {cfg.NUM_EPOCHS}")