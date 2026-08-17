"""
Core utilities for the DINOv2 change detection head.
Provides model definition, dataloaders, metrics, and helpers shared by train/eval scripts.
234行
"""
import json
import os
import random
import time
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from config import cfg as project_cfg
except Exception:
    project_cfg = None

try:
    from dataset import (
        LEVIRCDDataset,
        get_train_transforms,
        get_val_transforms,
        get_test_transforms_full,
        worker_init_fn,
    )
except Exception as e:
    raise RuntimeError(
        "???? dataset.py?LEVIRCDDataset/get_*_transforms/worker_init_fn??????????????? PYTHONPATH ?????\n"
        f"Import error: {e}"
    )


def _cfg_value(name: str, default):
    if project_cfg is not None and hasattr(project_cfg, name):
        return getattr(project_cfg, name)
    return default


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p


def style_perturb(x: torch.Tensor, sigma: float = 0.15, blur_prob: float = 0.3) -> torch.Tensor:
    """
    Style/appearance perturbation: channel-wise scale/bias + noise and optional blur.
    Keeps semantics while altering low-level style for counterfactual consistency.
    """
    noise = torch.randn_like(x) * sigma
    scale = 1.0 + torch.randn(x.shape[0], x.shape[1], 1, 1, device=x.device) * 0.1
    bias = torch.randn(x.shape[0], x.shape[1], 1, 1, device=x.device) * 0.05
    x = x * scale + bias + noise
    if torch.rand(1, device=x.device).item() < blur_prob:
        x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    return x.clamp(-3, 3)


def nuisance_perturb(
    x: torch.Tensor,
    illumination_strength: float = 0.35,
    max_shift: float = 2.0,
    blur_prob: float = 0.3,
) -> torch.Tensor:
    """Target-like no-change perturbation with smooth illumination and mild misregistration."""
    x01 = _denorm_imagenet(x).clamp(0.0, 1.0)
    bsz, channels, height, width = x01.shape

    gamma = torch.empty((bsz, 1, 1, 1), device=x.device, dtype=x.dtype).uniform_(0.7, 1.4)
    x01 = x01.clamp_min(1e-4).pow(gamma)
    gain = torch.empty((bsz, channels, 1, 1), device=x.device, dtype=x.dtype).uniform_(0.8, 1.2)
    bias = torch.empty((bsz, channels, 1, 1), device=x.device, dtype=x.dtype).uniform_(-0.08, 0.08)
    x01 = x01 * gain + bias

    field = torch.randn((bsz, channels, 4, 4), device=x.device, dtype=x.dtype)
    field = F.interpolate(field, size=(height, width), mode="bicubic", align_corners=False)
    field = field / field.std(dim=(2, 3), keepdim=True).clamp_min(1e-4)
    x01 = x01 * (1.0 + float(illumination_strength) * field).clamp(0.5, 1.5)

    if blur_prob > 0 and torch.rand((), device=x.device).item() < blur_prob:
        x01 = F.avg_pool2d(x01, kernel_size=3, stride=1, padding=1)

    if max_shift > 0:
        shift_x = torch.empty((bsz,), device=x.device, dtype=x.dtype).uniform_(-max_shift, max_shift)
        shift_y = torch.empty((bsz,), device=x.device, dtype=x.dtype).uniform_(-max_shift, max_shift)
        theta = torch.zeros((bsz, 2, 3), device=x.device, dtype=x.dtype)
        theta[:, 0, 0] = 1.0
        theta[:, 1, 1] = 1.0
        theta[:, 0, 2] = 2.0 * shift_x / max(1, width - 1)
        theta[:, 1, 2] = 2.0 * shift_y / max(1, height - 1)
        grid = F.affine_grid(theta, x01.shape, align_corners=False)
        x01 = F.grid_sample(x01, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return _norm_imagenet(x01.clamp(0.0, 1.0))


def registration_perturb(x: torch.Tensor, max_shift: float = 2.0) -> torch.Tensor:
    """Apply only a mild subpixel translation, preserving image appearance."""
    if max_shift <= 0:
        return x
    bsz, _, height, width = x.shape
    shift_x = torch.empty((bsz,), device=x.device, dtype=x.dtype).uniform_(
        -max_shift, max_shift
    )
    shift_y = torch.empty((bsz,), device=x.device, dtype=x.dtype).uniform_(
        -max_shift, max_shift
    )
    theta = torch.zeros((bsz, 2, 3), device=x.device, dtype=x.dtype)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = 2.0 * shift_x / max(1, width - 1)
    theta[:, 1, 2] = 2.0 * shift_y / max(1, height - 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(
        x, grid, mode="bilinear", padding_mode="border", align_corners=False
    )


def confusion_update(pred: torch.Tensor, gt: torch.Tensor, cm: Dict[str, int]):
    pred = pred.view(-1).to(torch.int64)
    gt = gt.view(-1).to(torch.int64)
    cm["TP"] += int(((pred == 1) & (gt == 1)).sum().item())
    cm["FP"] += int(((pred == 1) & (gt == 0)).sum().item())
    cm["FN"] += int(((pred == 0) & (gt == 1)).sum().item())
    cm["TN"] += int(((pred == 0) & (gt == 0)).sum().item())


def compute_metrics_from_cm(cm: Dict[str, int]) -> Dict[str, float]:
    TP, FP, FN, TN = cm["TP"], cm["FP"], cm["FN"], cm["TN"]
    eps = 1e-12
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = TP / (TP + FP + FN + eps)
    oa = (TP + TN) / (TP + TN + FP + FN + eps)
    total = TP + TN + FP + FN + eps
    po = (TP + TN) / total
    pe = ((TP + FP) * (TP + FN) + (FN + TN) * (FP + TN)) / (total * total)
    kappa = (po - pe) / (1 - pe + eps)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "oa": float(oa),
        "kappa": float(kappa),
        "TP": int(TP),
        "FP": int(FP),
        "FN": int(FN),
        "TN": int(TN),
    }


def otsu_threshold(score_map: np.ndarray) -> float:
    s = score_map.astype(np.float32)
    lo = np.percentile(s, 0.5)
    hi = np.percentile(s, 99.5)
    s = np.clip(s, lo, hi)
    hist, bin_edges = np.histogram(s.ravel(), bins=256)
    hist = hist.astype(np.float64)
    p = hist / (hist.sum() + 1e-12)
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mu_t = mu[-1]
    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1 - omega) + 1e-12)
    k = int(np.nanargmax(sigma_b2))
    return float(bin_edges[k])


def threshold_map(prob: np.ndarray, mode: str, fixed_thr: float, topk: float) -> Tuple[np.ndarray, float]:
    if mode == "fixed":
        thr = float(fixed_thr)
    elif mode == "topk":
        thr = float(np.quantile(prob, 1.0 - topk))
    elif mode == "otsu":
        thr = otsu_threshold(prob)
    else:
        raise ValueError(f"Unknown thr_mode: {mode}")
    pred = (prob > thr).astype(np.uint8)
    return pred, thr


def filter_small_cc(mask: np.ndarray, min_area: int = 256) -> np.ndarray:
    try:
        import cv2
    except Exception:
        return mask
    mask = np.asarray(mask)
    # cv2.connectedComponentsWithStats expects a 2D single-channel 8-bit image.
    # Be defensive: squeeze trivial singleton dims; otherwise skip filtering.
    if mask.ndim == 3:
        if mask.shape[0] == 1:
            mask = mask[0]
        elif mask.shape[-1] == 1:
            mask = mask[..., 0]
        else:
            return mask
    if mask.ndim != 2:
        return mask
    mask_u8 = ((mask > 0).astype(np.uint8) * 255).copy()
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    out = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 1
    return out


def dice_loss_with_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(2, 3))
    union = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def tversky_loss_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.7,
    beta: float = 0.3,
    eps: float = 1e-6,
) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    target = target.float()
    tp = (prob * target).sum(dim=(2, 3))
    fp = (prob * (1.0 - target)).sum(dim=(2, 3))
    fn = ((1.0 - prob) * target).sum(dim=(2, 3))
    score = (tp + eps) / (tp + float(alpha) * fp + float(beta) * fn + eps)
    return 1.0 - score.mean()


def _symmetric_local_cosine(
    feat_a: torch.Tensor,
    feat_b: torch.Tensor,
    radius: int = 0,
) -> torch.Tensor:
    """Best local cosine correspondence, averaged in both temporal directions."""
    feat_a = F.normalize(feat_a, dim=1)
    feat_b = F.normalize(feat_b, dim=1)
    radius = int(max(0, radius))
    if radius == 0:
        return (feat_a * feat_b).sum(dim=1, keepdim=True)

    bsz, channels, height, width = feat_a.shape
    kernel = 2 * radius + 1

    def _best(reference: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        patches = F.unfold(candidate, kernel_size=kernel, padding=radius)
        patches = patches.view(bsz, channels, kernel * kernel, height, width)
        corr = (reference.unsqueeze(2) * patches).sum(dim=1)

        valid = F.unfold(
            torch.ones((bsz, 1, height, width), device=reference.device, dtype=reference.dtype),
            kernel_size=kernel,
            padding=radius,
        ).view(bsz, kernel * kernel, height, width)
        corr = corr.masked_fill(valid < 0.5, torch.finfo(corr.dtype).min)
        return corr.amax(dim=1, keepdim=True)

    return 0.5 * (_best(feat_a, feat_b) + _best(feat_b, feat_a))


def temporal_dense_contrastive_loss(
    pair_features,
    target: torch.Tensor,
    margin: float = 0.2,
    radius: int = 1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Pull unchanged temporal features together and push changed features apart."""
    if not pair_features:
        return target.new_zeros(())

    losses = []
    for feat_a, feat_b in pair_features:
        similarity = _symmetric_local_cosine(feat_a, feat_b, radius=radius)
        changed = F.interpolate(target.float(), size=similarity.shape[-2:], mode="area").clamp(0, 1)
        unchanged = 1.0 - changed

        pull = ((1.0 - similarity) * unchanged).sum() / unchanged.sum().clamp_min(eps)
        push = (F.relu(similarity - float(margin)) * changed).sum() / changed.sum().clamp_min(eps)
        losses.append(pull + push)
    return torch.stack(losses).mean()


def pair_feature_invariance_loss(clean_pair_features, augmented_pair_features) -> torch.Tensor:
    """Align augmented adapter features to detached clean-view anchors."""
    if len(clean_pair_features) != len(augmented_pair_features):
        raise ValueError("Clean and augmented pair features must have the same layer count")
    losses = []
    for clean_pair, augmented_pair in zip(clean_pair_features, augmented_pair_features):
        for clean_feat, augmented_feat in zip(clean_pair, augmented_pair):
            clean_anchor = F.normalize(clean_feat.detach(), dim=1)
            augmented_norm = F.normalize(augmented_feat, dim=1)
            losses.append(1.0 - (clean_anchor * augmented_norm).sum(dim=1).mean())
    if not losses:
        raise ValueError("Pair feature lists must not be empty")
    return torch.stack(losses).mean()


def pair_feature_preservation_loss(base_pair_features, adapted_pair_features) -> torch.Tensor:
    """Penalize residual drift on clean inputs relative to a frozen base path."""
    if len(base_pair_features) != len(adapted_pair_features):
        raise ValueError("Base and adapted pair features must have the same layer count")
    losses = []
    for base_pair, adapted_pair in zip(base_pair_features, adapted_pair_features):
        for base_feat, adapted_feat in zip(base_pair, adapted_pair):
            anchor = base_feat.detach()
            scale = anchor.square().mean().clamp_min(1e-6)
            losses.append((adapted_feat - anchor).square().mean() / scale)
    if not losses:
        raise ValueError("Pair feature lists must not be empty")
    return torch.stack(losses).mean()


def alignment_feature_preservation_loss(alignment_features) -> torch.Tensor:
    """Keep local-alignment blending close to the frozen comparison path on clean pairs."""
    losses = []
    for base_compare, blended_compare in alignment_features:
        anchor = base_compare.detach()
        scale = anchor.square().mean().clamp_min(1e-6)
        losses.append((blended_compare - anchor).square().mean() / scale)
    if not losses:
        raise ValueError("Alignment feature list must not be empty")
    return torch.stack(losses).mean()


def mask_to_boundary(mask: torch.Tensor, dilation: int = 3) -> torch.Tensor:
    """
    Extract binary boundary map from a 1/0 mask using Sobel gradients + dilation.
    """
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    m = mask.float()
    if m.max() > 1:
        m = (m > 0).float()
    device = m.device
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=m.dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=m.dtype).view(1, 1, 3, 3)
    gx = F.conv2d(m, kx, padding=1)
    gy = F.conv2d(m, ky, padding=1)
    g = torch.sqrt(gx * gx + gy * gy)
    boundary = (g > 0).float()
    if dilation and dilation > 1:
        pad = dilation // 2
        boundary = F.max_pool2d(boundary, kernel_size=dilation, stride=1, padding=pad)
    return boundary.clamp(0, 1)


@dataclass
class HeadCfg:
    data_root: str = str(_cfg_value("DATA_ROOT", Path("data/LEVIR-CD")))
    out_dir: str = str(Path(_cfg_value("OUTPUT_ROOT", Path("outputs"))) / "dino_head_cd")
    seed: int = int(_cfg_value("SEED", 42))
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # training
    epochs: int = int(_cfg_value("NUM_EPOCHS", 100))
    batch_size: int = int(_cfg_value("BATCH_SIZE", 8))
    num_workers: int = int(_cfg_value("NUM_WORKERS", 4))
    crop_size: int = int(_cfg_value("CROP_SIZE", 256))
    lr: float = float(_cfg_value("LEARNING_RATE", 5e-4))
    weight_decay: float = float(_cfg_value("WEIGHT_DECAY", 1e-2))
    grad_accum: int = int(_cfg_value("GRADIENT_ACCUMULATION_STEPS", 2))
    bce_weight: float = float(_cfg_value("LOSS_WEIGHT_BCE", 1.0))
    dice_weight: float = float(_cfg_value("LOSS_WEIGHT_DICE", 1.0))
    tversky_weight: float = 0.0
    tversky_alpha: float = 0.7  # FP penalty; larger means more conservative predictions.
    tversky_beta: float = 0.3
    fp_penalty_weight: float = 0.0
    hard_negative_weight: float = 0.0
    hard_negative_ratio: float = 0.1
    boundary_weight: float = 0.0
    boundary_dilation: int = 3
    lambda_consis: float = 0.0  # counterfactual consistency weight
    lambda_domain: float = 0.0  # domain confusion weight
    self_sup_weight: float = 0.0  # auxiliary supervised weight on augmented view
    style_aug_prob: float = 0.0
    style_aug_sigma: float = 0.2
    style_blur_prob: float = 0.3
    identity_neg_weight: float = 0.0
    identity_neg_prob: float = 0.0
    contrastive_weight: float = 0.0
    contrastive_margin: float = 0.2
    contrastive_radius: int = 1
    nuisance_real_weight: float = 0.0
    nuisance_synth_weight: float = 0.0
    nuisance_synth_prob: float = 0.0
    nuisance_illumination: float = 0.35
    nuisance_max_shift: float = 2.0
    nuisance_pair_aug_weight: float = 0.0
    nuisance_pair_aug_prob: float = 0.0
    nuisance_pair_consistency: float = 0.0
    nuisance_feature_consistency: float = 0.0
    nuisance_clean_feature_preservation: float = 0.0
    registration_neg_weight: float = 0.0
    registration_neg_prob: float = 0.0
    registration_max_shift: float = 2.0
    alignment_clean_preservation: float = 0.0
    spatial_fusion_uniform_weight: float = 0.0
    head_aux_weight: float = 0.25
    head_cons_weight: float = 0.0

    # eval
    full_eval: bool = False  # fast eval by default; enable full_eval via CLI if needed
    eval_crop: int = int(_cfg_value("CROP_SIZE", 256))
    thr_mode: str = "fixed"
    thr: float = 0.5
    topk: float = 0.01
    smooth_k: int = 3
    use_minarea: bool = False
    min_area: int = 256
    use_ensemble_pred: bool = False

    # model
    arch: str = "dlv"  # dlv (default) | a0 (frozen backbone + single 1x1 head)
    dino_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    # 1-based transformer block indices to use for multi-layer features.
    # For 12-layer ViT (e.g., vits16/vitb16): (3, 6, 9, 12)
    # For 24-layer ViT (e.g., vitl16): (6, 12, 18, 24)
    # For 32-layer ViT (e.g., vith16plus): (8, 16, 24, 32)
    selected_layers: Tuple[int, ...] = (3, 6, 9, 12)
    fuse_mode: str = "abs+sum"
    align_radius: int = 0
    align_temperature: float = 0.1
    use_whiten: bool = False
    use_domain_adv: bool = False
    domain_hidden: int = 256
    domain_grl: float = 1.0
    use_style_norm: bool = False
    proto_path: str = ""
    proto_weight: float = 0.0
    boundary_dim: int = 0
    use_layer_ensemble: bool = False
    layer_head_ch: int = 128
    use_nuisance_gate: bool = False
    nuisance_hidden: int = 64
    nuisance_gate_weight: float = 2.0
    nuisance_use_image_cues: bool = False
    train_nuisance_only: bool = False
    train_adapters_only: bool = False
    use_residual_style_adapter: bool = False
    style_adapter_hidden: int = 64
    style_adapter_scale: float = 1.0
    train_style_adapter_only: bool = False
    learnable_align_blend: bool = False
    train_alignment_only: bool = False
    use_spatial_head_fusion: bool = False
    spatial_fusion_indices: Tuple[int, ...] = (2, 3, 4)
    spatial_fusion_hidden: int = 16
    train_spatial_fusion_only: bool = False
    a0_layer: int = 12  # only used when arch == "a0"
    ft_mode: str = "frozen"  # frozen | shallow | deep | full (backbone fine-tuning)
    ft_k: int = 4  # number of blocks to unfreeze for shallow/deep
    backbone_lr: float = 1e-5  # lr for unfrozen backbone blocks

    # saving / logging
    save_best: bool = True
    save_last: bool = True
    selection_metric: str = "f1"  # f1 | fbeta | precision
    selection_beta: float = 1.0  # beta < 1 favors precision when using fbeta
    vis_every: int = 5
    vis_n: int = 8
    log_every: int = 50

    # scheduler
    scheduler: str = "cosine"  # cosine | none
    warmup_epochs: int = 5
    min_lr: float = 1e-6

    # eval sliding window (None 代表不用滑窗)
    eval_window: Optional[int] = None
    eval_stride: Optional[int] = None


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: HeadCfg):
    if cfg.scheduler.lower() != "cosine":
        return None
    base_lr = cfg.lr
    min_lr = cfg.min_lr
    warmup = max(0, cfg.warmup_epochs)
    total = max(1, cfg.epochs)

    def lr_lambda(epoch: int):
        if epoch < warmup:
            return (epoch + 1) / max(1, warmup)
        t = (epoch - warmup) / max(1, total - warmup)
        cosine = 0.5 * (1 + math.cos(math.pi * t))
        return (min_lr / base_lr) + (1 - (min_lr / base_lr)) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _ensemble_prob_from_logits_all(
    logits_all: torch.Tensor, ensemble_cfg: Optional[Dict] = None
) -> torch.Tensor:
    """
    logits_all: [K,B,1,H,W]
    returns prob: [B,1,H,W]
    """
    cfg = ensemble_cfg or {}
    mode = str(cfg.get("mode", "mean_prob"))
    indices = cfg.get("indices")
    weights = cfg.get("weights")
    min_k = int(cfg.get("min_k", 2) or 2)
    max_k = int(cfg.get("max_k", 0) or 0)
    unc_power = float(cfg.get("unc_power", 1.0) or 1.0)

    if indices is not None and mode not in ("uw_gate",):
        idx = torch.as_tensor(indices, device=logits_all.device, dtype=torch.long)
        logits_all = logits_all.index_select(0, idx)

    if mode == "mean_prob":
        return torch.sigmoid(logits_all).mean(dim=0)
    if mode == "mean_logit":
        return torch.sigmoid(logits_all.mean(dim=0))
    if mode == "min_prob":
        # Conservative consensus: every selected head must clear the threshold.
        return torch.sigmoid(logits_all).amin(dim=0)
    if mode == "soft_min":
        # Fixed-strength compromise between average fusion and strict consensus.
        probs = torch.sigmoid(logits_all)
        strength = max(0.0, min(1.0, float(cfg.get("strength", 0.5))))
        return (1.0 - strength) * probs.mean(dim=0) + strength * probs.amin(dim=0)
    if mode == "max_rejection":
        # Suppress a locally over-confident head while preserving consensus pixels.
        probs = torch.sigmoid(logits_all)
        strength = max(0.0, min(1.0, float(cfg.get("strength", 0.5))))
        mean = probs.mean(dim=0)
        return (mean - strength * (probs.amax(dim=0) - mean)).clamp_(0.0, 1.0)
    if mode == "weighted_logit":
        if weights is None:
            raise ValueError("ensemble_cfg.mode='weighted_logit' requires ensemble_cfg['weights']")
        w = torch.as_tensor(weights, device=logits_all.device, dtype=logits_all.dtype)
        if w.ndim != 1 or w.numel() != logits_all.shape[0]:
            raise ValueError(
                f"weights must be shape [K'] matching selected heads, got {tuple(w.shape)} vs K'={logits_all.shape[0]}"
            )
        w = w / w.sum().clamp_min(1e-12)
        w = w.view(-1, 1, 1, 1, 1)
        return torch.sigmoid((w * logits_all).sum(dim=0))
    if mode == "ugls":
        # Uncertainty-guided layer selection (UGLS):
        # - compute pixel-wise uncertainty as variance across head probabilities
        # - map uncertainty to a dynamic top-k (in ranked head order) per pixel
        # - output sigmoid(mean_logit of selected heads)
        K = int(logits_all.shape[0])
        if K <= 1:
            return torch.sigmoid(logits_all.squeeze(0))

        if max_k <= 0 or max_k > K:
            max_k = K
        min_k = max(1, min(int(min_k), max_k))

        probs = torch.sigmoid(logits_all)  # [K,B,1,H,W]
        unc = probs.var(dim=0, unbiased=False)  # [B,1,H,W] in [0, 0.25]
        unc_norm = (unc / 0.25).clamp(0.0, 1.0)
        if unc_power != 1.0:
            unc_norm = unc_norm.clamp_min(0.0).pow(float(unc_power))

        span = max_k - min_k
        if span <= 0:
            # fixed k == min_k == max_k
            sum_logits = logits_all[:max_k].sum(dim=0)
            return torch.sigmoid(sum_logits / float(max_k))

        # Map unc_norm in [0,1] to integer k in [min_k, max_k] (inclusive), using full range.
        # Use (span+1) bins so max_k is reachable without requiring unc_norm==1 exactly.
        k_off = torch.floor(unc_norm * float(span + 1)).to(dtype=torch.long).clamp(min=0, max=span)  # [B,1,H,W]
        k_map = (min_k + k_off).to(dtype=torch.long)  # [B,1,H,W]
        k_map = k_map.clamp(min=min_k, max=max_k)

        cum = logits_all.cumsum(dim=0)  # [K,B,1,H,W]
        gather_idx = (k_map - 1).clamp(min=0, max=K - 1).unsqueeze(0)  # [1,B,1,H,W]
        sum_logits = cum.gather(0, gather_idx).squeeze(0)  # [B,1,H,W]
        mean_logits = sum_logits / k_map.to(dtype=logits_all.dtype)
        return torch.sigmoid(mean_logits)
    if mode == "consis2":
        # Consistency-driven dynamic weighting between two heads.
        # Expects selected heads K'==2 (e.g., [deepest_layer_head, fused]).
        # Let d = |p0 - p1|. Use d to control blend weight w in [0, max_w].
        # Output sigmoid((1-w)*logit1 + w*logit0), where logit1 is treated as the "anchor" (usually fused).
        if logits_all.shape[0] != 2:
            raise ValueError(f"consis2 requires exactly 2 heads after selection, got K'={int(logits_all.shape[0])}")
        d0 = float(cfg.get("d0", 0.05))
        d1 = float(cfg.get("d1", 0.25))
        gamma = float(cfg.get("gamma", 1.0))
        max_w = float(cfg.get("max_w", 0.5))
        eps = 1e-6
        if d1 <= d0:
            d1 = d0 + 1e-3
        p = torch.sigmoid(logits_all)  # [2,B,1,H,W]
        d = (p[0] - p[1]).abs()  # [B,1,H,W] in [0,1]
        g = ((d - d0) / (d1 - d0 + eps)).clamp(0.0, 1.0)
        if gamma != 1.0:
            g = g.pow(gamma)
        w = (max_w * g).to(dtype=logits_all.dtype)  # [B,1,H,W]
        logit = (1.0 - w) * logits_all[1] + w * logits_all[0]
        return torch.sigmoid(logit)
    if mode == "consisk":
        # Consistency-driven per-pixel soft weighting over K' heads.
        # Weights are computed by how close each head's prob is to the mean prob (consensus),
        # with a temperature that increases with uncertainty (var across heads).
        temp0 = float(cfg.get("temp0", 0.03))
        temp1 = float(cfg.get("temp1", 0.20))
        gamma = float(cfg.get("gamma", 1.0))
        fused_local_idx = int(cfg.get("fused_local_idx", -1))
        fused_bias = float(cfg.get("fused_bias", 0.0))
        eps = 1e-6

        probs = torch.sigmoid(logits_all)  # [K',B,1,H,W]
        p_mean = probs.mean(dim=0, keepdim=True)  # [1,B,1,H,W]
        dev = (probs - p_mean).abs()  # [K',B,1,H,W]
        score = -dev  # higher is better (closer to consensus)
        if fused_bias and fused_local_idx >= 0 and fused_local_idx < int(score.shape[0]):
            score[fused_local_idx] = score[fused_local_idx] + float(fused_bias)

        unc = probs.var(dim=0, unbiased=False)  # [B,1,H,W] in [0,0.25]
        unc_norm = (unc / 0.25).clamp(0.0, 1.0)
        if gamma != 1.0:
            unc_norm = unc_norm.pow(gamma)
        temp = (temp0 + temp1 * unc_norm).clamp_min(eps).to(dtype=logits_all.dtype)  # [B,1,H,W]
        w = torch.softmax(score / temp.unsqueeze(0), dim=0)  # [K',B,1,H,W]
        logit = (w * logits_all).sum(dim=0)
        return torch.sigmoid(logit)
    if mode == "uw_gate":
        # Uncertainty-weighted inference gate:
        # - compute pixel-wise uncertainty u(x)=Var_k(p_k(x)) over unc_indices
        # - map u to gate g in [0, max_w]
        # - blend two logits: (1-g)*anchor + g*other
        fuse_indices = cfg.get("fuse_indices", None)
        unc_indices = cfg.get("unc_indices", None)
        u0 = float(cfg.get("u0", 0.10))
        u1 = float(cfg.get("u1", 0.60))
        gamma = float(cfg.get("gamma", 1.0))
        max_w = float(cfg.get("max_w", 0.50))
        gate_smooth_k = int(cfg.get("gate_smooth_k", 0) or 0)

        K = int(logits_all.shape[0])
        if fuse_indices is None:
            fuse_indices = [max(0, K - 2), max(0, K - 1)]
        if not (isinstance(fuse_indices, (list, tuple)) and len(fuse_indices) == 2):
            raise ValueError("uw_gate requires fuse_indices=[other, anchor] with length 2")
        other_idx, anchor_idx = int(fuse_indices[0]), int(fuse_indices[1])
        if any(i < 0 or i >= K for i in (other_idx, anchor_idx)):
            raise ValueError(f"uw_gate fuse_indices out of range: {fuse_indices} for K={K}")

        if unc_indices is None:
            unc_indices = list(range(K))
        if not isinstance(unc_indices, (list, tuple)) or len(unc_indices) < 2:
            raise ValueError("uw_gate requires unc_indices with length >= 2")
        unc_indices = [int(i) for i in unc_indices]
        if any(i < 0 or i >= K for i in unc_indices):
            raise ValueError(f"uw_gate unc_indices out of range: {unc_indices} for K={K}")

        probs_unc = torch.sigmoid(logits_all.index_select(0, torch.as_tensor(unc_indices, device=logits_all.device)))
        u = probs_unc.var(dim=0, unbiased=False)  # [B,1,H,W] in [0,0.25]
        u_norm = (u / 0.25).clamp(0.0, 1.0)
        if u1 <= u0:
            u1 = u0 + 1e-3
        g = ((u_norm - u0) / (u1 - u0)).clamp(0.0, 1.0)
        if gamma != 1.0:
            g = g.pow(gamma)
        g = (float(max_w) * g).to(dtype=logits_all.dtype)  # [B,1,H,W]
        if gate_smooth_k and gate_smooth_k > 1:
            pad = gate_smooth_k // 2
            g = F.avg_pool2d(g, kernel_size=gate_smooth_k, stride=1, padding=pad)

        logit_other = logits_all[other_idx]
        logit_anchor = logits_all[anchor_idx]
        logit = (1.0 - g) * logit_anchor + g * logit_other
        return torch.sigmoid(logit)
    raise ValueError(f"Unknown ensemble mode: {mode}")


def _prob_from_out(out, use_ensemble: bool = False, ensemble_cfg: Optional[Dict] = None) -> torch.Tensor:
    if isinstance(out, dict):
        if use_ensemble and out.get("logits_all") is not None:
            return _ensemble_prob_from_logits_all(out["logits_all"], ensemble_cfg=ensemble_cfg)
        return torch.sigmoid(out["pred"])
    return torch.sigmoid(out)

# ImageNet normalization used by dataset.py
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def _denorm_imagenet(x: torch.Tensor) -> torch.Tensor:
    mean = _IMAGENET_MEAN.to(device=x.device, dtype=x.dtype)
    std = _IMAGENET_STD.to(device=x.device, dtype=x.dtype)
    return x * std + mean


def _norm_imagenet(x: torch.Tensor) -> torch.Tensor:
    mean = _IMAGENET_MEAN.to(device=x.device, dtype=x.dtype)
    std = _IMAGENET_STD.to(device=x.device, dtype=x.dtype)
    return (x - mean) / std


def pairwise_contrast_canonicalize(
    img_a: torch.Tensor,
    img_b: torch.Tensor,
    eps: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Remove global temporal brightness/contrast mismatch using shared pair moments."""
    a01 = _denorm_imagenet(img_a).clamp(0.0, 1.0)
    b01 = _denorm_imagenet(img_b).clamp(0.0, 1.0)
    mu_a = a01.mean(dim=(2, 3), keepdim=True)
    mu_b = b01.mean(dim=(2, 3), keepdim=True)
    std_a = a01.std(dim=(2, 3), keepdim=True).clamp_min(eps)
    std_b = b01.std(dim=(2, 3), keepdim=True).clamp_min(eps)

    shared_mu = 0.5 * (mu_a + mu_b)
    shared_std = torch.sqrt(0.5 * (std_a.square() + std_b.square())).clamp_min(eps)
    a_shared = ((a01 - mu_a) / std_a * shared_std + shared_mu).clamp(0.0, 1.0)
    b_shared = ((b01 - mu_b) / std_b * shared_std + shared_mu).clamp(0.0, 1.0)
    return _norm_imagenet(a_shared), _norm_imagenet(b_shared)


def _jpeg_compress_batch(x01: torch.Tensor, quality: int) -> torch.Tensor:
    """
    x01: [B,3,H,W] in [0,1] (float). Returns float in [0,1].
    Uses PIL JPEG encode/decode on CPU for realism.
    """
    from io import BytesIO

    try:
        from PIL import Image
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"PIL is required for JPEG corruption: {e}")

    quality = int(max(1, min(95, int(quality))))
    x_cpu = (x01.clamp(0, 1) * 255.0).to(dtype=torch.uint8, device="cpu")
    out_list: List[torch.Tensor] = []
    for i in range(int(x_cpu.shape[0])):
        arr = x_cpu[i].permute(1, 2, 0).contiguous().numpy()
        img = Image.fromarray(arr, mode="RGB")
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        buf.seek(0)
        img2 = Image.open(buf).convert("RGB")
        arr2 = np.asarray(img2, dtype=np.uint8).copy()
        t = torch.from_numpy(arr2).permute(2, 0, 1).contiguous()
        out_list.append(t)
    out = torch.stack(out_list, dim=0).to(dtype=torch.float32) / 255.0
    return out.to(device=x01.device)


def apply_corruption_pair(
    img_a: torch.Tensor,
    img_b: torch.Tensor,
    *,
    mode: str = "none",
    pair_mode: str = "correlated",
    seed: int = 0,
    gaussian_sigma: float = 0.0,
    bc_brightness: float = 0.0,
    bc_contrast: float = 0.0,
    jpeg_quality: int = 75,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply synthetic test-time corruptions on normalized tensors.

    img_a/img_b: [B,3,H,W], assumed ImageNet-normalized.

    mode:
      - none
      - gaussian: add N(0, sigma^2) in [0,1] space
      - bc: brightness/contrast: y = (x-0.5)*c + 0.5 + b
      - jpeg: JPEG encode/decode in [0,1] space

    pair_mode:
      - correlated: same random params per sample for (A,B)
      - uncorrelated: independent params for A and B
    """
    mode = str(mode or "none").lower()
    if mode in ("none", "off", "0", "false"):
        return img_a, img_b

    pair_mode = str(pair_mode or "correlated").lower()
    if pair_mode not in ("correlated", "uncorrelated"):
        raise ValueError("pair_mode must be correlated|uncorrelated")

    g = torch.Generator(device=img_a.device)
    g.manual_seed(int(seed))

    a01 = _denorm_imagenet(img_a).clamp(0.0, 1.0)
    b01 = _denorm_imagenet(img_b).clamp(0.0, 1.0)

    B = int(a01.shape[0])
    if mode == "gaussian":
        sigma = float(gaussian_sigma)
        if sigma <= 0:
            return img_a, img_b
        if pair_mode == "correlated":
            n = torch.randn(a01.shape, device=a01.device, generator=g, dtype=a01.dtype)
            a01 = (a01 + sigma * n).clamp(0.0, 1.0)
            b01 = (b01 + sigma * n).clamp(0.0, 1.0)
        else:
            na = torch.randn(a01.shape, device=a01.device, generator=g, dtype=a01.dtype)
            nb = torch.randn(b01.shape, device=b01.device, generator=g, dtype=b01.dtype)
            a01 = (a01 + sigma * na).clamp(0.0, 1.0)
            b01 = (b01 + sigma * nb).clamp(0.0, 1.0)
    elif mode in ("bc", "brightness_contrast", "bri_contrast"):
        bmax = float(bc_brightness)
        cmax = float(bc_contrast)
        if bmax <= 0 and cmax <= 0:
            return img_a, img_b

        def _sample_bc():
            bb = (torch.rand((B, 1, 1, 1), device=a01.device, generator=g, dtype=a01.dtype) * 2 - 1.0) * bmax
            if cmax > 0:
                cc = 1.0 + (torch.rand((B, 1, 1, 1), device=a01.device, generator=g, dtype=a01.dtype) * 2 - 1.0) * cmax
                cc = cc.clamp(0.1, 3.0)
            else:
                cc = torch.ones((B, 1, 1, 1), device=a01.device, dtype=a01.dtype)
            return bb, cc

        if pair_mode == "correlated":
            bb, cc = _sample_bc()
            a01 = ((a01 - 0.5) * cc + 0.5 + bb).clamp(0.0, 1.0)
            b01 = ((b01 - 0.5) * cc + 0.5 + bb).clamp(0.0, 1.0)
        else:
            bb1, cc1 = _sample_bc()
            bb2, cc2 = _sample_bc()
            a01 = ((a01 - 0.5) * cc1 + 0.5 + bb1).clamp(0.0, 1.0)
            b01 = ((b01 - 0.5) * cc2 + 0.5 + bb2).clamp(0.0, 1.0)
    elif mode == "jpeg":
        q = int(jpeg_quality)
        # same quality; encode/decode separately per image
        a01 = _jpeg_compress_batch(a01, quality=q).clamp(0.0, 1.0)
        b01 = _jpeg_compress_batch(b01, quality=q).clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unknown corruption mode: {mode}")

    return _norm_imagenet(a01), _norm_imagenet(b01)


def _apply_tta_d4(x: torch.Tensor, *, k: int, hflip: bool) -> torch.Tensor:
    """
    Apply D4 transform: rotate by 90*k, then optional horizontal flip.
    x: [..., H, W]
    """
    k = int(k) % 4
    if k:
        x = torch.rot90(x, k=k, dims=(-2, -1))
    if hflip:
        x = torch.flip(x, dims=(-1,))
    return x


def _invert_tta_d4(x: torch.Tensor, *, k: int, hflip: bool) -> torch.Tensor:
    """Inverse of _apply_tta_d4."""
    k = int(k) % 4
    if hflip:
        x = torch.flip(x, dims=(-1,))
    if k:
        x = torch.rot90(x, k=(4 - k) % 4, dims=(-2, -1))
    return x


@torch.no_grad()
def tta_inference_prob(
    model: nn.Module,
    img_a: torch.Tensor,
    img_b: torch.Tensor,
    *,
    device: str,
    window: Optional[int] = None,
    stride: Optional[int] = None,
    use_ensemble: bool = False,
    ensemble_cfg: Optional[Dict] = None,
    tta_mode: str = "none",
) -> torch.Tensor:
    """
    TTA inference returning prob map [B,1,H,W].

    tta_mode:
      - "none": no augmentation
      - "pair_contrast": shared pairwise brightness/contrast canonicalization
      - "flip": 4-way (id, hflip, vflip, hvflip)
      - "flip_median": 4-way flip TTA merged by the conservative lower median
      - "flip_min": 4-way flip TTA merged by per-pixel minimum probability
      - "d4":   8-way D4 (rot0/90/180/270, each with optional hflip)

    Merges by averaging probabilities (after inverting each transform back).
    """
    tta_mode = str(tta_mode or "none").lower()
    if tta_mode == "pair_contrast":
        img_a, img_b = pairwise_contrast_canonicalize(img_a, img_b)
        tta_mode = "none"
    if tta_mode in ("none", "off", "0", "false"):
        if window is not None and stride is not None:
            return sliding_window_inference(
                model=model,
                img_a=img_a,
                img_b=img_b,
                window=int(window),
                stride=int(stride),
                device=device,
                use_ensemble=use_ensemble,
                ensemble_cfg=ensemble_cfg,
            )
        out = model(img_a, img_b)
        return _prob_from_out(out, use_ensemble=use_ensemble, ensemble_cfg=ensemble_cfg)

    merge_mode = "mean"
    if tta_mode in ("flip", "flip_median", "flip_min"):
        # id/hflip/rot180/vflip (equivalent set of 4 flip variants)
        aug_list = [(0, False), (0, True), (2, False), (2, True)]
        if tta_mode == "flip_median":
            merge_mode = "median"
        elif tta_mode == "flip_min":
            merge_mode = "min"
    elif tta_mode in ("d4", "rot90", "flip_rot90"):
        aug_list = [(k, f) for k in (0, 1, 2, 3) for f in (False, True)]
    else:
        raise ValueError(f"Unknown tta_mode: {tta_mode}")

    probs = []
    for k, f in aug_list:
        a_aug = _apply_tta_d4(img_a, k=k, hflip=f)
        b_aug = _apply_tta_d4(img_b, k=k, hflip=f)
        if window is not None and stride is not None:
            prob_aug = sliding_window_inference(
                model=model,
                img_a=a_aug,
                img_b=b_aug,
                window=int(window),
                stride=int(stride),
                device=device,
                use_ensemble=use_ensemble,
                ensemble_cfg=ensemble_cfg,
            )
        else:
            out = model(a_aug, b_aug)
            prob_aug = _prob_from_out(out, use_ensemble=use_ensemble, ensemble_cfg=ensemble_cfg)
        prob_aug = _invert_tta_d4(prob_aug, k=k, hflip=f)
        probs.append(prob_aug)

    prob_stack = torch.stack(probs, dim=0)
    if merge_mode == "min":
        return prob_stack.amin(dim=0)
    if merge_mode == "median":
        # torch.median returns the lower middle value for an even number of views.
        return prob_stack.median(dim=0).values
    return prob_stack.mean(dim=0)


@torch.no_grad()
def sliding_window_inference(
    model: nn.Module,
    img_a: torch.Tensor,
    img_b: torch.Tensor,
    window: int,
    stride: int,
    device: str,
    use_ensemble: bool = False,
    ensemble_cfg: Optional[Dict] = None,
) -> torch.Tensor:
    """
    Sliding-window inference for large images.
    Returns prob map (1,1,H,W) on device.
    """
    _, _, H, W = img_a.shape
    if use_ensemble and str((ensemble_cfg or {}).get("mode")) == "max_rejection":
        probs_all = sliding_window_inference_probs_all(
            model=model,
            img_a=img_a,
            img_b=img_b,
            window=window,
            stride=stride,
            device=device,
        )
        cfg = ensemble_cfg or {}
        indices = cfg.get("indices")
        if indices is not None:
            index = torch.as_tensor(indices, device=probs_all.device, dtype=torch.long)
            probs_all = probs_all.index_select(0, index)
        strength = max(0.0, min(1.0, float(cfg.get("strength", 0.5))))
        mean = probs_all.mean(dim=0)
        return (mean - strength * (probs_all.amax(dim=0) - mean)).clamp_(0.0, 1.0)
    if H <= window and W <= window:
        out = model(img_a, img_b)
        return _prob_from_out(out, use_ensemble=use_ensemble, ensemble_cfg=ensemble_cfg)

    prob_sum = torch.zeros((1, 1, H, W), device=device)
    count_map = torch.zeros((1, 1, H, W), device=device)

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y_end = min(y + window, H)
            x_end = min(x + window, W)
            y_start = max(0, y_end - window)
            x_start = max(0, x_end - window)

            patch_a = img_a[..., y_start:y_end, x_start:x_end]
            patch_b = img_b[..., y_start:y_end, x_start:x_end]

            out = model(patch_a, patch_b)
            prob = _prob_from_out(out, use_ensemble=use_ensemble, ensemble_cfg=ensemble_cfg)  # (1,1,h,w)

            prob_sum[..., y_start:y_end, x_start:x_end] += prob
            count_map[..., y_start:y_end, x_start:x_end] += 1

    prob = prob_sum / count_map.clamp_min(1e-6)
    return prob


@torch.no_grad()
def sliding_window_inference_probs_all(
    model: nn.Module,
    img_a: torch.Tensor,
    img_b: torch.Tensor,
    window: int,
    stride: int,
    device: str,
) -> torch.Tensor:
    """
    Returns probs_all [K,B,1,H,W] from model's logits_all, with sliding-window stitching if needed.
    Note: sliding-window path assumes B==1.
    """
    B, _, H, W = img_a.shape
    if H <= window and W <= window:
        out = model(img_a, img_b)
        if not isinstance(out, dict) or out.get("logits_all") is None:
            raise RuntimeError("Model output has no logits_all; enable use_layer_ensemble during training/eval.")
        return torch.sigmoid(out["logits_all"])

    if B != 1:
        raise ValueError("sliding_window_inference_probs_all currently supports B==1 only.")

    prob_sum = None  # [K,1,H,W]
    count_map = torch.zeros((1, 1, H, W), device=device)

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y_end = min(y + window, H)
            x_end = min(x + window, W)
            y_start = max(0, y_end - window)
            x_start = max(0, x_end - window)

            patch_a = img_a[..., y_start:y_end, x_start:x_end]
            patch_b = img_b[..., y_start:y_end, x_start:x_end]

            out = model(patch_a, patch_b)
            if not isinstance(out, dict) or out.get("logits_all") is None:
                raise RuntimeError("Model output has no logits_all; enable use_layer_ensemble during training/eval.")
            probs_all = torch.sigmoid(out["logits_all"])  # [K,1,1,h,w]

            if prob_sum is None:
                K = probs_all.shape[0]
                prob_sum = torch.zeros((K, 1, H, W), device=device)

            prob_patch = probs_all[:, 0]  # [K,1,h,w]
            prob_sum[..., y_start:y_end, x_start:x_end] += prob_patch
            count_map[..., y_start:y_end, x_start:x_end] += 1

    probs_all = prob_sum / count_map.clamp_min(1e-6)  # [K,1,H,W]
    return probs_all.unsqueeze(1)  # [K,1,1,H,W]


@torch.no_grad()
def sliding_window_inference_logits_all(
    model: nn.Module,
    img_a: torch.Tensor,
    img_b: torch.Tensor,
    window: int,
    stride: int,
    device: str,
) -> torch.Tensor:
    """
    Returns logits_all [K,B,1,H,W] from model's logits_all, with sliding-window stitching if needed.
    Note: sliding-window path assumes B==1.
    """
    B, _, H, W = img_a.shape
    if H <= window and W <= window:
        out = model(img_a, img_b)
        if not isinstance(out, dict) or out.get("logits_all") is None:
            raise RuntimeError("Model output has no logits_all; enable use_layer_ensemble during training/eval.")
        return out["logits_all"]

    if B != 1:
        raise ValueError("sliding_window_inference_logits_all currently supports B==1 only.")

    logit_sum = None  # [K,1,H,W]
    count_map = torch.zeros((1, 1, H, W), device=device)

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y_end = min(y + window, H)
            x_end = min(x + window, W)
            y_start = max(0, y_end - window)
            x_start = max(0, x_end - window)

            patch_a = img_a[..., y_start:y_end, x_start:x_end]
            patch_b = img_b[..., y_start:y_end, x_start:x_end]

            out = model(patch_a, patch_b)
            if not isinstance(out, dict) or out.get("logits_all") is None:
                raise RuntimeError("Model output has no logits_all; enable use_layer_ensemble during training/eval.")
            logits_all = out["logits_all"]  # [K,1,1,h,w]

            if logit_sum is None:
                K = logits_all.shape[0]
                logit_sum = torch.zeros((K, 1, H, W), device=device)

            logit_patch = logits_all[:, 0]  # [K,1,h,w]
            logit_sum[..., y_start:y_end, x_start:x_end] += logit_patch
            count_map[..., y_start:y_end, x_start:x_end] += 1

    logits_all = logit_sum / count_map.clamp_min(1e-6)  # [K,1,H,W]
    return logits_all.unsqueeze(1)  # [K,1,1,H,W]

try:
    from models.dinov2_head import DinoSiameseHead, DinoFrozenA0Head
except ImportError:
    from models.dinov2_head import DinoSiameseHead
    DinoFrozenA0Head = None

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: str,
    bce_w: float,
    dice_w: float,
    boundary_w: float,
    boundary_dilation: int,
    grad_accum: int,
    log_every: int,
    tversky_w: float = 0.0,
    tversky_alpha: float = 0.7,
    tversky_beta: float = 0.3,
    fp_penalty_w: float = 0.0,
    hard_negative_weight: float = 0.0,
    hard_negative_ratio: float = 0.1,
    lambda_consis: float = 0.0,
    lambda_domain: float = 0.0,
    self_sup_weight: float = 0.0,
    style_aug_prob: float = 0.0,
    style_aug_sigma: float = 0.2,
    style_blur_prob: float = 0.3,
    identity_neg_weight: float = 0.0,
    identity_neg_prob: float = 0.0,
    contrastive_weight: float = 0.0,
    contrastive_margin: float = 0.2,
    contrastive_radius: int = 1,
    nuisance_real_weight: float = 0.0,
    nuisance_synth_weight: float = 0.0,
    nuisance_synth_prob: float = 0.0,
    nuisance_illumination: float = 0.35,
    nuisance_max_shift: float = 2.0,
    nuisance_pair_aug_weight: float = 0.0,
    nuisance_pair_aug_prob: float = 0.0,
    nuisance_pair_consistency: float = 0.0,
    nuisance_feature_consistency: float = 0.0,
    nuisance_clean_feature_preservation: float = 0.0,
    registration_neg_weight: float = 0.0,
    registration_neg_prob: float = 0.0,
    registration_max_shift: float = 2.0,
    alignment_clean_preservation: float = 0.0,
    spatial_fusion_uniform_weight: float = 0.0,
    head_aux_weight: float = 0.0,
    head_cons_weight: float = 0.0,
):
    model.train()
    bce = nn.BCEWithLogitsLoss()
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()
    for it, batch in enumerate(loader, 1):
        img_a = batch["img_a"].to(device, non_blocking=True)
        img_b = batch["img_b"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        if label.ndim == 3:
            label = label.unsqueeze(1)
        elif label.ndim == 4 and label.shape[1] != 1:
            label = label[:, :1]
        label = label.float()
        boundary_gt = None
        if boundary_w > 0:
            boundary_gt = mask_to_boundary(label, dilation=boundary_dilation)
        with torch.cuda.amp.autocast(enabled=(device.startswith("cuda") and torch.cuda.is_available())):
            out = model(img_a, img_b)
            logits = out["pred"] if isinstance(out, dict) else out
            prob = torch.sigmoid(logits)
            loss_bce = bce(logits, label)
            loss_dice = dice_loss_with_logits(logits, label)
            loss_tv = tversky_loss_with_logits(logits, label, alpha=tversky_alpha, beta=tversky_beta)
            loss_fp = (prob * (1.0 - label)).mean()
            loss = (bce_w * loss_bce + dice_w * loss_dice + tversky_w * loss_tv + fp_penalty_w * loss_fp) / grad_accum
            if hard_negative_weight > 0:
                negative_loss = F.softplus(logits)[label < 0.5]
                if negative_loss.numel() > 0:
                    k = max(1, int(math.ceil(float(hard_negative_ratio) * negative_loss.numel())))
                    k = min(k, negative_loss.numel())
                    hard_negative_loss = torch.topk(negative_loss, k=k, sorted=False).values.mean()
                    loss = loss + hard_negative_weight * hard_negative_loss / grad_accum
            if contrastive_weight > 0 and isinstance(out, dict) and out.get("pair_features"):
                loss_contrast = temporal_dense_contrastive_loss(
                    out["pair_features"],
                    label,
                    margin=contrastive_margin,
                    radius=contrastive_radius,
                )
                loss = loss + contrastive_weight * loss_contrast / grad_accum
            if alignment_clean_preservation > 0:
                if not isinstance(out, dict) or not out.get("alignment_features"):
                    raise ValueError(
                        "alignment_clean_preservation requires alignment_features"
                    )
                alignment_preservation = alignment_feature_preservation_loss(
                    out["alignment_features"]
                )
                loss = loss + (
                    alignment_clean_preservation
                    * alignment_preservation
                    / grad_accum
                )
            if spatial_fusion_uniform_weight > 0:
                fusion_weights = out.get("fusion_weights") if isinstance(out, dict) else None
                if fusion_weights is None:
                    raise ValueError(
                        "spatial_fusion_uniform_weight requires fusion_weights"
                    )
                uniform = 1.0 / float(fusion_weights.shape[1])
                fusion_regularization = (fusion_weights - uniform).square().mean()
                loss = loss + (
                    spatial_fusion_uniform_weight
                    * fusion_regularization
                    / grad_accum
                )
            if nuisance_real_weight > 0 and isinstance(out, dict) and out.get("nuisance_logit") is not None:
                with torch.no_grad():
                    raw_pred = out.get("raw_pred", logits)
                    nuisance_target = (
                        (torch.sigmoid(raw_pred) > 0.5) & (label < 0.5)
                    ).to(dtype=out["nuisance_logit"].dtype)
                nuisance_loss_map = F.binary_cross_entropy_with_logits(
                    out["nuisance_logit"], nuisance_target, reduction="none"
                )
                positive = nuisance_target > 0.5
                negative = ~positive
                positive_loss = (
                    nuisance_loss_map[positive].mean()
                    if positive.any()
                    else nuisance_loss_map.new_zeros(())
                )
                negative_loss = nuisance_loss_map[negative].mean()
                nuisance_real_loss = 0.5 * (positive_loss + negative_loss)
                loss = loss + nuisance_real_weight * nuisance_real_loss / grad_accum
            if boundary_w > 0 and isinstance(out, dict) and out.get("boundary") is not None:
                b_logit = out["boundary"]
                loss = loss + boundary_w * bce(b_logit, boundary_gt) / grad_accum
            if isinstance(out, dict) and out.get("logits_all") is not None:
                logits_all = out["logits_all"]
                if logits_all.ndim == 5 and logits_all.shape[0] > 1:
                    head_loss = torch.tensor(0.0, device=logits_all.device)
                    for k in range(logits_all.shape[0] - 1):
                        loss_bce_k = bce(logits_all[k], label)
                        loss_dice_k = dice_loss_with_logits(logits_all[k], label)
                        loss_tv_k = tversky_loss_with_logits(
                            logits_all[k],
                            label,
                            alpha=tversky_alpha,
                            beta=tversky_beta,
                        )
                        prob_k = torch.sigmoid(logits_all[k])
                        loss_fp_k = (prob_k * (1.0 - label)).mean()
                        head_loss = head_loss + (
                            bce_w * loss_bce_k
                            + dice_w * loss_dice_k
                            + tversky_w * loss_tv_k
                            + fp_penalty_w * loss_fp_k
                        )
                    if head_aux_weight > 0:
                        loss = loss + head_aux_weight * head_loss / grad_accum
                    if head_cons_weight > 0:
                        prob_all = torch.sigmoid(logits_all)
                        prob_mean = prob_all.mean(dim=0)
                        cons_loss = ((prob_all - prob_mean.unsqueeze(0)) ** 2).mean()
                        loss = loss + head_cons_weight * cons_loss / grad_accum

            do_cf = style_aug_prob > 0 and (lambda_consis > 0 or lambda_domain > 0 or self_sup_weight > 0)
            if do_cf and torch.rand(1, device=img_a.device).item() < style_aug_prob:
                img_a_cf = style_perturb(img_a, sigma=style_aug_sigma, blur_prob=style_blur_prob)
                img_b_cf = style_perturb(img_b, sigma=style_aug_sigma, blur_prob=style_blur_prob)
                out_cf = model(img_a_cf, img_b_cf)
                logits_cf = out_cf["pred"] if isinstance(out_cf, dict) else out_cf
                prob_cf = torch.sigmoid(logits_cf)

                if self_sup_weight > 0:
                    cf_bce = bce(logits_cf, label)
                    cf_dice = dice_loss_with_logits(logits_cf, label)
                    cf_tv = tversky_loss_with_logits(logits_cf, label, alpha=tversky_alpha, beta=tversky_beta)
                    loss = loss + self_sup_weight * (cf_bce + cf_dice + tversky_w * cf_tv) / grad_accum
                if boundary_w > 0 and isinstance(out_cf, dict) and out_cf.get("boundary") is not None:
                    loss = loss + boundary_w * bce(out_cf["boundary"], boundary_gt) / grad_accum

                if lambda_consis > 0:
                    loss = loss + lambda_consis * F.l1_loss(prob, prob_cf) / grad_accum

                if lambda_domain > 0:
                    dom_logits_list = []
                    dom_labels_list = []
                    if isinstance(out, dict) and out.get("domain_logit") is not None:
                        dom_logits_list.append(out["domain_logit"].view(-1))
                        dom_labels_list.append(torch.zeros_like(out["domain_logit"].view(-1)))
                    if isinstance(out_cf, dict) and out_cf.get("domain_logit") is not None:
                        dom_logits_list.append(out_cf["domain_logit"].view(-1))
                        dom_labels_list.append(torch.ones_like(out_cf["domain_logit"].view(-1)))
                    if dom_logits_list:
                        dom_logits = torch.cat(dom_logits_list, dim=0)
                        dom_labels = torch.cat(dom_labels_list, dim=0)
                        dom_loss = F.binary_cross_entropy_with_logits(dom_logits, dom_labels)
                        loss = loss + lambda_domain * dom_loss / grad_accum

            do_nuisance_pair_aug = (
                (
                    nuisance_pair_aug_weight > 0
                    or nuisance_pair_consistency > 0
                    or nuisance_feature_consistency > 0
                    or nuisance_clean_feature_preservation > 0
                )
                and nuisance_pair_aug_prob > 0
                and torch.rand((), device=img_a.device).item() < nuisance_pair_aug_prob
            )
            if do_nuisance_pair_aug:
                # Independent photometric changes preserve the source label; geometric
                # shifts are intentionally disabled because they invalidate its boundary.
                img_a_aug = nuisance_perturb(
                    img_a,
                    illumination_strength=nuisance_illumination,
                    max_shift=0.0,
                    blur_prob=style_blur_prob,
                )
                img_b_aug = nuisance_perturb(
                    img_b,
                    illumination_strength=nuisance_illumination,
                    max_shift=0.0,
                    blur_prob=style_blur_prob,
                )
                out_aug = model(img_a_aug, img_b_aug)
                logits_aug = out_aug["pred"] if isinstance(out_aug, dict) else out_aug
                prob_aug = torch.sigmoid(logits_aug)
                aug_loss = (
                    bce_w * bce(logits_aug, label)
                    + dice_w * dice_loss_with_logits(logits_aug, label)
                    + tversky_w
                    * tversky_loss_with_logits(
                        logits_aug,
                        label,
                        alpha=tversky_alpha,
                        beta=tversky_beta,
                    )
                    + fp_penalty_w * (prob_aug * (1.0 - label)).mean()
                )
                if isinstance(out_aug, dict) and out_aug.get("logits_all") is not None:
                    logits_all_aug = out_aug["logits_all"]
                    if logits_all_aug.ndim == 5 and logits_all_aug.shape[0] > 1:
                        aux_aug = logits_aug.new_zeros(())
                        for k in range(logits_all_aug.shape[0] - 1):
                            logits_k = logits_all_aug[k]
                            prob_k = torch.sigmoid(logits_k)
                            aux_aug = aux_aug + (
                                bce_w * bce(logits_k, label)
                                + dice_w * dice_loss_with_logits(logits_k, label)
                                + tversky_w
                                * tversky_loss_with_logits(
                                    logits_k,
                                    label,
                                    alpha=tversky_alpha,
                                    beta=tversky_beta,
                                )
                                + fp_penalty_w * (prob_k * (1.0 - label)).mean()
                            )
                        aug_loss = aug_loss + head_aux_weight * aux_aug
                if nuisance_pair_aug_weight > 0:
                    loss = loss + nuisance_pair_aug_weight * aug_loss / grad_accum
                if nuisance_pair_consistency > 0:
                    loss = loss + (
                        nuisance_pair_consistency
                        * F.l1_loss(prob_aug, prob.detach())
                        / grad_accum
                    )
                if nuisance_feature_consistency > 0:
                    if not (
                        isinstance(out, dict)
                        and isinstance(out_aug, dict)
                        and out.get("pair_features")
                        and out_aug.get("pair_features")
                    ):
                        raise ValueError(
                            "nuisance_feature_consistency requires model pair_features"
                        )
                    clean_anchors = out.get("base_pair_features") or out["pair_features"]
                    feature_loss = pair_feature_invariance_loss(
                        clean_anchors, out_aug["pair_features"]
                    )
                    loss = loss + nuisance_feature_consistency * feature_loss / grad_accum
                if nuisance_clean_feature_preservation > 0:
                    if not (
                        isinstance(out, dict)
                        and out.get("base_pair_features")
                        and out.get("pair_features")
                    ):
                        raise ValueError(
                            "nuisance_clean_feature_preservation requires base_pair_features"
                        )
                    preservation_loss = pair_feature_preservation_loss(
                        out["base_pair_features"], out["pair_features"]
                    )
                    loss = loss + (
                        nuisance_clean_feature_preservation
                        * preservation_loss
                        / grad_accum
                    )

            do_identity_neg = (
                identity_neg_weight > 0
                and identity_neg_prob > 0
                and torch.rand(1, device=img_a.device).item() < identity_neg_prob
            )
            if do_identity_neg:
                # Same-scene appearance perturbations are hard no-change examples.
                base = img_a if torch.rand(1, device=img_a.device).item() < 0.5 else img_b
                base_cf = style_perturb(
                    base,
                    sigma=style_aug_sigma,
                    blur_prob=style_blur_prob,
                )
                out_neg = model(base, base_cf)
                logits_neg = out_neg["pred"] if isinstance(out_neg, dict) else out_neg
                neg_loss = F.binary_cross_entropy_with_logits(
                    logits_neg, torch.zeros_like(logits_neg)
                )
                if isinstance(out_neg, dict) and out_neg.get("logits_all") is not None:
                    logits_all_neg = out_neg["logits_all"]
                    if logits_all_neg.ndim == 5:
                        aux_neg = torch.stack(
                            [
                                F.binary_cross_entropy_with_logits(
                                    logits_all_neg[k], torch.zeros_like(logits_all_neg[k])
                                )
                                for k in range(logits_all_neg.shape[0])
                            ]
                        ).mean()
                        neg_loss = neg_loss + head_aux_weight * aux_neg
                loss = loss + identity_neg_weight * neg_loss / grad_accum

            do_registration_neg = (
                registration_neg_weight > 0
                and registration_neg_prob > 0
                and torch.rand((), device=img_a.device).item() < registration_neg_prob
            )
            if do_registration_neg:
                base = img_a if torch.rand((), device=img_a.device).item() < 0.5 else img_b
                shifted = registration_perturb(base, max_shift=registration_max_shift)
                out_registration = model(base, shifted)
                logits_registration = (
                    out_registration["pred"]
                    if isinstance(out_registration, dict)
                    else out_registration
                )
                registration_loss = F.binary_cross_entropy_with_logits(
                    logits_registration, torch.zeros_like(logits_registration)
                )
                loss = loss + registration_neg_weight * registration_loss / grad_accum

            do_nuisance_synth = (
                nuisance_synth_weight > 0
                and nuisance_synth_prob > 0
                and isinstance(out, dict)
                and out.get("nuisance_logit") is not None
                and torch.rand((), device=img_a.device).item() < nuisance_synth_prob
            )
            if do_nuisance_synth:
                base = img_a if torch.rand((), device=img_a.device).item() < 0.5 else img_b
                base_cf = nuisance_perturb(
                    base,
                    illumination_strength=nuisance_illumination,
                    max_shift=nuisance_max_shift,
                    blur_prob=style_blur_prob,
                )
                out_nuisance = model(base, base_cf)
                nuisance_logit = out_nuisance["nuisance_logit"]
                gate_loss = F.binary_cross_entropy_with_logits(
                    nuisance_logit, torch.ones_like(nuisance_logit)
                )
                suppress_loss = F.binary_cross_entropy_with_logits(
                    out_nuisance["pred"], torch.zeros_like(out_nuisance["pred"])
                )
                loss = loss + nuisance_synth_weight * (gate_loss + suppress_loss) / grad_accum
        scaler.scale(loss).backward()
        if it % grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                for module in model.modules():
                    strength = getattr(module, "align_strength", None)
                    if isinstance(strength, nn.Parameter):
                        strength.clamp_(0.0, 1.0)
            optimizer.zero_grad(set_to_none=True)
        if it % log_every == 0:
            dt = time.time() - t0
            print(f"  [train] iter={it}/{len(loader)} loss={loss.item()*grad_accum:.4f} time={dt:.1f}s")
            t0 = time.time()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    thr_mode: str,
    thr: float,
    topk: float,
    smooth_k: int,
    use_minarea: bool,
    min_area: int,
    print_every: int = 0,
    window: Optional[int] = 256,
    stride: Optional[int] = 256,
    use_ensemble: bool = False,
    ensemble_cfg: Optional[Dict] = None,
    tta_mode: str = "none",
    corrupt: str = "none",
    corrupt_pair: str = "correlated",
    corrupt_seed: int = 0,
    gaussian_sigma: float = 0.0,
    bc_brightness: float = 0.0,
    bc_contrast: float = 0.0,
    jpeg_quality: int = 75,
) -> Dict[str, float]:
    model.eval()
    cm = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for i, batch in enumerate(loader, 1):
        img_a = batch["img_a"].to(device, non_blocking=True)
        img_b = batch["img_b"].to(device, non_blocking=True)
        if corrupt and str(corrupt).lower() not in ("none", "off", "0", "false"):
            img_a, img_b = apply_corruption_pair(
                img_a,
                img_b,
                mode=str(corrupt),
                pair_mode=str(corrupt_pair),
                seed=int(corrupt_seed) + int(i),
                gaussian_sigma=float(gaussian_sigma),
                bc_brightness=float(bc_brightness),
                bc_contrast=float(bc_contrast),
                jpeg_quality=int(jpeg_quality),
            )
        gt = batch["label"]
        if gt.ndim == 3:
            gt = gt.unsqueeze(1)
        elif gt.ndim == 4 and gt.shape[1] != 1:
            gt = gt[:, :1]
        prob = tta_inference_prob(
            model=model,
            img_a=img_a,
            img_b=img_b,
            device=device,
            window=window,
            stride=stride,
            use_ensemble=use_ensemble,
            ensemble_cfg=ensemble_cfg,
            tta_mode=tta_mode,
        )
        if smooth_k and smooth_k > 1:
            pad = smooth_k // 2
            prob = F.avg_pool2d(prob, kernel_size=smooth_k, stride=1, padding=pad)
        # batch-safe evaluation
        B = int(prob.shape[0])
        for bi in range(B):
            prob_np = prob[bi, 0].float().detach().cpu().numpy()
            gt_np = gt[bi, 0].detach().cpu().numpy().astype(np.uint8)
            pred_np, _ = threshold_map(prob_np, thr_mode, thr, topk)
            if use_minarea:
                pred_np = filter_small_cc(pred_np, min_area=min_area)
            pred_t = torch.from_numpy(pred_np.astype(np.uint8))
            gt_t = torch.from_numpy((gt_np > 0).astype(np.uint8))
            confusion_update(pred_t, gt_t, cm)
        if print_every and (i % print_every == 0):
            print(f"[{i}/{len(loader)}] TP={cm['TP']} FP={cm['FP']} FN={cm['FN']} TN={cm['TN']}")
    return compute_metrics_from_cm(cm)


def save_vis_samples(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    out_dir: str,
    n: int,
    thr_mode: str,
    thr: float,
    topk: float,
    smooth_k: int,
    window: Optional[int] = None,
    stride: Optional[int] = None,
    use_ensemble: bool = False,
    ensemble_cfg: Optional[Dict] = None,
    tta_mode: str = "none",
    corrupt: str = "none",
    corrupt_pair: str = "correlated",
    corrupt_seed: int = 0,
    gaussian_sigma: float = 0.0,
    bc_brightness: float = 0.0,
    bc_contrast: float = 0.0,
    jpeg_quality: int = 75,
):
    import matplotlib.pyplot as plt
    ensure_dir(out_dir)
    model.eval()
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    saved = 0
    for batch in loader:
        img_a = batch["img_a"].to(device)
        img_b = batch["img_b"].to(device)
        if corrupt and str(corrupt).lower() not in ("none", "off", "0", "false"):
            img_a, img_b = apply_corruption_pair(
                img_a,
                img_b,
                mode=str(corrupt),
                pair_mode=str(corrupt_pair),
                seed=int(corrupt_seed) + int(saved),
                gaussian_sigma=float(gaussian_sigma),
                bc_brightness=float(bc_brightness),
                bc_contrast=float(bc_contrast),
                jpeg_quality=int(jpeg_quality),
            )
        gt = batch["label"]
        names = batch["name"]
        prob = tta_inference_prob(
            model=model,
            img_a=img_a,
            img_b=img_b,
            device=device,
            window=window,
            stride=stride,
            use_ensemble=use_ensemble,
            ensemble_cfg=ensemble_cfg,
            tta_mode=tta_mode,
        )
        if smooth_k and smooth_k > 1:
            pad = smooth_k // 2
            prob = F.avg_pool2d(prob, kernel_size=smooth_k, stride=1, padding=pad)
        B = img_a.shape[0]
        for bi in range(B):
            prob_np = prob[bi].squeeze().detach().cpu().numpy()
            pred_np, used_thr = threshold_map(prob_np, thr_mode, thr, topk)
            a_np = img_a[bi].detach().cpu().permute(1, 2, 0).numpy()
            b_np = img_b[bi].detach().cpu().permute(1, 2, 0).numpy()
            a_np = (a_np * std + mean).clip(0, 1)
            b_np = (b_np * std + mean).clip(0, 1)
            gt_np = gt[bi].detach().cpu().numpy().astype(np.uint8)
            overlay = b_np.copy()
            overlay[pred_np == 1] = [1, 0, 0]
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes[0, 0].imshow(a_np); axes[0, 0].set_title("T1"); axes[0, 0].axis("off")
            axes[0, 1].imshow(b_np); axes[0, 1].set_title("T2"); axes[0, 1].axis("off")
            im = axes[0, 2].imshow(prob_np, cmap="jet"); axes[0, 2].set_title("Prob"); axes[0, 2].axis("off"); fig.colorbar(im, ax=axes[0, 2], fraction=0.046)
            axes[1, 0].imshow(pred_np, cmap="gray"); axes[1, 0].set_title(f"Pred ({thr_mode} thr={used_thr:.4f})"); axes[1, 0].axis("off")
            axes[1, 1].imshow(gt_np, cmap="gray"); axes[1, 1].set_title("GT"); axes[1, 1].axis("off")
            axes[1, 2].imshow(overlay); axes[1, 2].set_title("Overlay"); axes[1, 2].axis("off")
            plt.tight_layout()
            name = names[bi] if isinstance(names, list) else names
            save_path = os.path.join(out_dir, f"{saved:03d}_{name}.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            saved += 1
            if saved >= n:
                return


def build_dataloaders(
    cfg: HeadCfg,
    require_train: bool = True,
    require_val: bool = True,
    load_val: bool = True,
) -> Tuple[Optional[DataLoader], Optional[DataLoader], DataLoader]:
    if require_val and not load_val:
        raise ValueError("require_val=True is incompatible with load_val=False")
    root = Path(cfg.data_root)
    train_tf = get_train_transforms(crop_size=cfg.crop_size)
    eval_tf = get_test_transforms_full() if cfg.full_eval else get_val_transforms(crop_size=cfg.eval_crop)

    def _make_dataset(split: str, transform) -> LEVIRCDDataset:
        return LEVIRCDDataset(
            root_dir=root,
            split=split,
            transform=transform,
            crop_size=cfg.crop_size if split == "train" else cfg.eval_crop,
        )

    loader_kwargs = dict(
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
    )
    if cfg.num_workers > 0:
        loader_kwargs["worker_init_fn"] = worker_init_fn

    train_loader = None
    if require_train:
        train_loader = DataLoader(
            _make_dataset("train", train_tf),
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
            **loader_kwargs,
        )
    eval_batch_size = 1 if cfg.full_eval else cfg.batch_size
    val_ds = None
    if load_val:
        if require_val:
            val_ds = _make_dataset("val", eval_tf)
        else:
            try:
                val_ds = _make_dataset("val", eval_tf)
            except FileNotFoundError:
                val_ds = None

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=eval_batch_size,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )
    test_loader = DataLoader(
        _make_dataset("test", eval_tf),
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    return train_loader, val_loader, test_loader

__all__ = [
    "HeadCfg",
    "DinoSiameseHead",
    "DinoFrozenA0Head",
    "build_dataloaders",
    "seed_everything",
    "ensure_dir",
    "train_one_epoch",
    "evaluate",
    "save_vis_samples",
    "sliding_window_inference",
    "sliding_window_inference_probs_all",
    "threshold_map",
    "filter_small_cc",
    "mask_to_boundary",
    "build_scheduler",
]
