"""
evaluation/metrics.py — FID and Inception Score
"""
import os
import torch
from torchvision.utils import save_image

try:
    import torch_fidelity
    FIDELITY_AVAILABLE = True
except ImportError:
    FIDELITY_AVAILABLE = False
    print("[metrics] torch-fidelity not found. Run: pip install torch-fidelity")


def save_images_for_fid(images_tensor, save_dir, prefix="img"):
    os.makedirs(save_dir, exist_ok=True)
    imgs = (images_tensor.clamp(-1, 1) + 1) / 2.0
    for i, img in enumerate(imgs):
        save_image(img, os.path.join(save_dir, f"{prefix}_{i:05d}.png"))


def generate_samples_for_eval(model, model_type, n_samples, latent_dim,
                               device, save_dir):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    batch_size = 64
    saved = 0

    with torch.no_grad():
        while saved < n_samples:
            n = min(batch_size, n_samples - saved)
            if model_type == "gan":
                z    = torch.randn(n, latent_dim, device=device)
                imgs = model(z).cpu()
            elif model_type == "vae":
                imgs = model.sample(n, device).cpu()
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
            save_images_for_fid(imgs, save_dir, prefix=f"gen_{saved}")
            saved += n

    print(f"[metrics] Saved {saved} generated images to {save_dir}")


def generate_real_samples_for_eval(dataloader, n_samples, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    saved = 0
    for imgs, _ in dataloader:
        if saved >= n_samples:
            break
        n = min(imgs.size(0), n_samples - saved)
        save_images_for_fid(imgs[:n], save_dir, prefix=f"real_{saved}")
        saved += n
    print(f"[metrics] Saved {saved} real images to {save_dir}")


def compute_fid_is(real_dir, fake_dir, device="cuda"):
    if not FIDELITY_AVAILABLE:
        print("[metrics] torch-fidelity not available. Skipping FID/IS.")
        return {"fid": None, "inception_score_mean": None, "inception_score_std": None}

    print(f"[metrics] Computing FID and IS...")
    metrics = torch_fidelity.calculate_metrics(
        input1=real_dir,
        input2=fake_dir,
        cuda=(device == "cuda" or device == torch.device("cuda")),
        isc=True,
        fid=True,
        verbose=False,
    )

    fid     = metrics.get("frechet_inception_distance", None)
    is_mean = metrics.get("inception_score_mean", None)
    is_std  = metrics.get("inception_score_std",  None)

    print(f"[metrics] FID: {fid:.2f} | IS: {is_mean:.2f} ± {is_std:.2f}")
    return {"fid": fid, "inception_score_mean": is_mean, "inception_score_std": is_std}


def evaluate_all_models(models_dict, real_dir, n_samples=1000,
                        latent_dim=100, device=None, results_dir="./results"):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_results = {}

    for name, info in models_dict.items():
        print(f"\n── Evaluating {name} ──────────────────────────")
        fake_dir = os.path.join(results_dir,
                                f"fid_fake_{name.lower().replace('-','_')}")
        generate_samples_for_eval(
            model=info["model"],
            model_type=info["type"],
            n_samples=n_samples,
            latent_dim=latent_dim,
            device=device,
            save_dir=fake_dir,
        )
        result = compute_fid_is(real_dir=real_dir, fake_dir=fake_dir, device=device)
        result["model"] = name
        all_results[name] = result

    print(f"\n{'='*55}")
    print(f"  QUANTITATIVE EVALUATION SUMMARY")
    print(f"{'='*55}")
    print(f"  {'Model':<12} {'FID ↓':>10}  {'IS ↑ (mean ± std)':>20}")
    print(f"  {'-'*50}")
    for name, r in all_results.items():
        fid_str = f"{r['fid']:.2f}" if r['fid'] is not None else "N/A"
        is_str  = (f"{r['inception_score_mean']:.2f} ± {r['inception_score_std']:.2f}"
                   if r['inception_score_mean'] is not None else "N/A")
        print(f"  {name:<12} {fid_str:>10}  {is_str:>20}")
    print(f"{'='*55}")

    return all_results