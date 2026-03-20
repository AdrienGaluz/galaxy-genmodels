# Galaxy Generative Model Comparison
### DC-GAN vs. WGAN-GP vs. VAE on Galaxy10 DECals

**COGS185 – Advanced Machine Learning Methods | UC San Diego**

---

## Overview

This project trains and compares three deep generative models on the **Galaxy10 DECals** dataset — a collection of 17,736 galaxy images across 10 morphological classes (spiral, elliptical, merging, etc.).

The central question: *How do adversarial and probabilistic generative approaches differ in their ability to model the complex structure of astronomical imagery?*

---

## Models

| Model | Type | Key Loss | Key Advantage |
|---|---|---|---|
| **DC-GAN** | Adversarial | Binary Cross-Entropy | Sharp images, fast training |
| **WGAN-GP** | Adversarial | Wasserstein + Gradient Penalty | Training stability, no mode collapse |
| **VAE** | Probabilistic | ELBO (Recon + KL) | Smooth latent space, reconstructions |

---

## Dataset

**Galaxy10 DECals** — 17,736 color galaxy images (256×256 RGB), 10 morphological classes.  
Downloaded automatically via `astroNN` on first run.

| Class | Description |
|---|---|
| 0 | Disturbed Galaxies |
| 1 | Merging Galaxies |
| 2 | Round Smooth Galaxies |
| 3 | In-between Round Smooth |
| 4 | Cigar Shaped Smooth |
| 5 | Barred Spiral Galaxies |
| 6 | Unbarred Tight Spiral |
| 7 | Unbarred Loose Spiral |
| 8 | Edge-on without Bulge |
| 9 | Edge-on with Bulge |

---

## Quantitative Results

| Model | FID ↓ | IS ↑ |
|---|---|---|
| DC-GAN | TBD | TBD |
| WGAN-GP | TBD | TBD |
| VAE | TBD | TBD |

*Results will be updated after training completion.*

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set resolution (config.py line 16)
```python
IMAGE_SIZE = 64    # Laptop (RTX 4070, 8GB VRAM)
IMAGE_SIZE = 128   # Desktop (RTX 4080 Super, 16GB VRAM)
```

### 3. Run full pipeline
```bash
python main.py
```

### 4. Train individual models
```bash
python main.py --models dcgan
python main.py --models wgan_gp
python main.py --models vae
```

### 5. Evaluation only (after training)
```bash
python main.py --eval-only
```

---

## Project Structure

```
galaxy_genmodels/
├── config.py               ← All hyperparameters + resolution switch
├── main.py                 ← Full pipeline orchestrator
├── requirements.txt
├── data/
│   └── dataset.py          ← Galaxy10 DECals loader + augmentation
├── models/
│   ├── dcgan.py            ← DC-GAN Generator + Discriminator
│   ├── wgan_gp.py          ← WGAN-GP Generator + Critic + GP
│   └── vae.py              ← VAE Encoder + Decoder + ELBO loss
├── training/
│   ├── train_dcgan.py      ← DC-GAN training loop
│   ├── train_wgan_gp.py    ← WGAN-GP training loop
│   └── train_vae.py        ← VAE training loop + LR scheduler
├── evaluation/
│   ├── metrics.py          ← FID + Inception Score computation
│   └── ablation.py         ← Hyperparameter ablation study
├── utils/
│   └── visualize.py        ← Image grids, loss curves, interpolation
├── checkpoints/            ← Saved model weights (auto-created)
├── results/                ← Evaluation figures (auto-created)
└── samples/                ← Training sample images (auto-created)
```

---

## Hyperparameter Ablation Studies

The project includes 4 systematic ablation experiments:

1. **Latent dimension**: 64 vs. 100 vs. 256
2. **Learning rate**: 1e-3 vs. 2e-4 vs. 1e-4
3. **VAE β coefficient**: 0.5 vs. 1.0 vs. 2.0
4. **Architecture depth**: NGF=32 (shallow) vs. NGF=64 (default)

---

## Key Visualizations

- `results/model_comparison_grid.png` — Side-by-side generated galaxy grids
- `results/all_models_losses.png` — Training loss curves for all 3 models
- `results/fid_is_comparison.png` — FID and IS bar chart
- `results/vae_latent_interpolation.png` — Smooth latent space walk
- `samples/` — Per-epoch sample grids showing training progression

---

## Requirements

- Python 3.9+
- PyTorch 2.0+ with CUDA
- NVIDIA GPU (8GB+ VRAM for 64×64, 16GB+ for 128×128)

---

## References

1. Radford et al. (2015) — *Unsupervised Representation Learning with Deep Convolutional GANs* (DC-GAN)
2. Gulrajani et al. (2017) — *Improved Training of Wasserstein GANs* (WGAN-GP)
3. Kingma & Welling (2013) — *Auto-Encoding Variational Bayes* (VAE)
4. Leung & Bovy (2019) — *Galaxy10 DECals Dataset* (astroNN)

---

*COGS185 Final Project — UC San Diego*
