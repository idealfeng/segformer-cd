"""
Day 1: DINOv2 Zero-Shot Baseline (LEVIRCDDataset version)
目标：20张可视化，判断 score map 是否有意义
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path

from config import cfg
from dataset import LEVIRCDDataset, get_val_transforms
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATCH = 14

# ===== 1) Load DINOv2 =====
print("Loading DINOv2...")
dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")  # 推荐 reg
dino.eval().to(DEVICE)


@torch.no_grad()
def resize_to_patch_multiple(x: torch.Tensor, patch=14):
    """x: (B,3,H,W) -> resize so H,W are multiples of patch"""
    B, C, H, W = x.shape
    H2 = max(patch, (H // patch) * patch)
    W2 = max(patch, (W // patch) * patch)
    if H2 == H and W2 == W:
        return x
    return F.interpolate(x, size=(H2, W2), mode="bilinear", align_corners=False)


@torch.no_grad()
def extract_dense_features(img: torch.Tensor):
    """
    img: (B,3,H,W) normalized
    return: feat_map (B,C,h,w), resized_hw (H',W')
    """
    img = resize_to_patch_multiple(img, PATCH)
    B, _, H, W = img.shape
    h, w = H // PATCH, W // PATCH

    with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
        out = dino.forward_features(img)

    if "x_norm_patchtokens" not in out:
        raise KeyError(f"forward_features keys: {list(out.keys())}")

    tokens = out["x_norm_patchtokens"]  # (B, N, C)
    B2, N, C = tokens.shape
    assert N == h * w, f"N={N} != h*w={h*w}. Check resize/patch."

    feat = tokens.transpose(1, 2).reshape(B, C, h, w)
    feat = F.normalize(feat, dim=1)
    return feat, (H, W)


@torch.no_grad()
def compute_change_score(img_a: torch.Tensor, img_b: torch.Tensor):
    """return score_map (H0,W0) numpy"""
    H0, W0 = img_a.shape[-2], img_a.shape[-1]

    fa, (Ha, Wa) = extract_dense_features(img_a)
    fb, (Hb, Wb) = extract_dense_features(img_b)
    assert (Ha, Wa) == (Hb, Wb)

    sim = (fa * fb).sum(dim=1, keepdim=True)
    dist = 1 - sim  # (B,1,h,w)

    score = F.interpolate(dist, size=(Ha, Wa), mode="bilinear", align_corners=False)
    score = F.interpolate(score, size=(H0, W0), mode="bilinear", align_corners=False)

    return score.squeeze(0).squeeze(0).float().cpu().numpy()


# ===== 2) Dataset (建议用 280 而不是 256，减少 resize 误差) =====
crop_size = 1024  # ✅ 14的倍数；如果你显存足够也可以 560
test_dataset = LEVIRCDDataset(
    root_dir=cfg.DATA_ROOT,
    split="test",
    transform=get_val_transforms(crop_size=crop_size),
    crop_size=crop_size,
)

# ===== 3) Run 20 samples =====
out_dir = Path("outputs/day1_dinov2_levir")
out_dir.mkdir(parents=True, exist_ok=True)

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

indices = np.random.choice(len(test_dataset), min(20, len(test_dataset)), replace=False)

print(f"Testing {len(indices)} samples...")

for i, idx in enumerate(indices):
    sample = test_dataset[idx]
    img_a = (
        sample["img_a"].unsqueeze(0).to(DEVICE, non_blocking=True)
    )  # (1,3,H,W) normalized
    img_b = sample["img_b"].unsqueeze(0).to(DEVICE, non_blocking=True)
    label = sample["label"]  # (H,W) torch

    score = compute_change_score(img_a, img_b)

    thr = np.percentile(score, 95)
    pred = (score > thr).astype(np.uint8)

    # --- vis ---
    img_a_np = img_a.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    img_b_np = img_b.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    img_a_np = (img_a_np * std + mean).clip(0, 1)
    img_b_np = (img_b_np * std + mean).clip(0, 1)

    gt = label.detach().cpu().numpy().astype(np.uint8)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(img_a_np)
    axes[0, 0].set_title("T1")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(img_b_np)
    axes[0, 1].set_title("T2")
    axes[0, 1].axis("off")

    im = axes[0, 2].imshow(score, cmap="jet")
    axes[0, 2].set_title(f"Score (max={score.max():.3f})")
    axes[0, 2].axis("off")
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046)

    axes[1, 0].imshow(pred, cmap="gray")
    axes[1, 0].set_title(f"Pred (top5%, thr={thr:.3f})")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(gt, cmap="gray")
    axes[1, 1].set_title("GT")
    axes[1, 1].axis("off")

    overlay = img_b_np.copy()
    overlay[pred == 1] = [1, 0, 0]
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title("Overlay")
    axes[1, 2].axis("off")

    plt.tight_layout()
    save_path = out_dir / f"sample_{i:02d}_{sample['name']}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(
        f"[{i+1}/{len(indices)}] saved {save_path.name} | score [{score.min():.3f}, {score.max():.3f}]"
    )

print(f"\n✅ Done. Check: {out_dir}")
