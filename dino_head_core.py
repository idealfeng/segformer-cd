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
    mask_u8 = mask.astype(np.uint8) * 255
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
    lambda_consis: float = 0.0  # counterfactual consistency weight
    lambda_domain: float = 0.0  # domain confusion weight
    self_sup_weight: float = 0.0  # auxiliary supervised weight on augmented view
    style_aug_prob: float = 0.0
    style_aug_sigma: float = 0.2
    style_blur_prob: float = 0.3

    # eval
    full_eval: bool = False  # fast eval by default; enable full_eval via CLI if needed
    eval_crop: int = int(_cfg_value("CROP_SIZE", 256))
    thr_mode: str = "fixed"
    thr: float = 0.5
    topk: float = 0.01
    smooth_k: int = 3
    use_minarea: bool = False
    min_area: int = 256

    # model
    dino_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    fuse_mode: str = "abs+sum"
    use_whiten: bool = False
    use_domain_adv: bool = False
    domain_hidden: int = 256
    domain_grl: float = 1.0
    use_style_norm: bool = False
    proto_path: str = ""
    proto_weight: float = 0.0

    # saving / logging
    save_best: bool = True
    save_last: bool = True
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


@torch.no_grad()
def sliding_window_inference(
    model: nn.Module,
    img_a: torch.Tensor,
    img_b: torch.Tensor,
    window: int,
    stride: int,
    device: str,
) -> torch.Tensor:
    """
    Sliding-window inference for large images.
    Returns prob map (1,1,H,W) on device.
    """
    _, _, H, W = img_a.shape
    if H <= window and W <= window:
        out = model(img_a, img_b)
        logits = out["pred"] if isinstance(out, dict) else out
        return torch.sigmoid(logits)

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
            logits = out["pred"] if isinstance(out, dict) else out
            prob = torch.sigmoid(logits)  # (1,1,h,w)

            prob_sum[..., y_start:y_end, x_start:x_end] += prob
            count_map[..., y_start:y_end, x_start:x_end] += 1

    prob = prob_sum / count_map.clamp_min(1e-6)
    return prob


from models.dinov2_head import DinoSiameseHead

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: str,
    bce_w: float,
    dice_w: float,
    grad_accum: int,
    log_every: int,
    lambda_consis: float = 0.0,
    lambda_domain: float = 0.0,
    self_sup_weight: float = 0.0,
    style_aug_prob: float = 0.0,
    style_aug_sigma: float = 0.2,
    style_blur_prob: float = 0.3,
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
        with torch.cuda.amp.autocast(enabled=(device.startswith("cuda") and torch.cuda.is_available())):
            out = model(img_a, img_b)
            logits = out["pred"] if isinstance(out, dict) else out
            prob = torch.sigmoid(logits)
            loss_bce = bce(logits, label)
            loss_dice = dice_loss_with_logits(logits, label)
            loss = (bce_w * loss_bce + dice_w * loss_dice) / grad_accum

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
                    loss = loss + self_sup_weight * (cf_bce + cf_dice) / grad_accum

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
        scaler.scale(loss).backward()
        if it % grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
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
    window: int = 256,
    stride: int = 256,
) -> Dict[str, float]:
    model.eval()
    cm = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for i, batch in enumerate(loader, 1):
        img_a = batch["img_a"].to(device, non_blocking=True)
        img_b = batch["img_b"].to(device, non_blocking=True)
        gt = batch["label"]
        if window is not None and stride is not None:
            prob = sliding_window_inference(
                model=model,
                img_a=img_a,
                img_b=img_b,
                window=window,
                stride=stride,
                device=device,
            )
        else:
            out = model(img_a, img_b)
            logits = out["pred"] if isinstance(out, dict) else out
            prob = torch.sigmoid(logits)
        if smooth_k and smooth_k > 1:
            pad = smooth_k // 2
            prob = F.avg_pool2d(prob, kernel_size=smooth_k, stride=1, padding=pad)
        prob_np = prob.squeeze(0).squeeze(0).float().cpu().numpy()
        gt_np = gt.squeeze(0).cpu().numpy().astype(np.uint8) if gt.ndim == 3 else gt.cpu().numpy().astype(np.uint8)
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
        gt = batch["label"]
        names = batch["name"]
        if window is not None and stride is not None:
            prob = sliding_window_inference(
                model=model,
                img_a=img_a,
                img_b=img_b,
                window=window,
                stride=stride,
                device=device,
            )
        else:
            out = model(img_a, img_b)
            logits = out["pred"] if isinstance(out, dict) else out
            prob = torch.sigmoid(logits)
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


def build_dataloaders(cfg: HeadCfg) -> Tuple[DataLoader, DataLoader, DataLoader]:
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

    train_loader = DataLoader(
        _make_dataset("train", train_tf),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    eval_batch_size = 1 if cfg.full_eval else cfg.batch_size
    val_loader = DataLoader(
        _make_dataset("val", eval_tf),
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
    "build_dataloaders",
    "seed_everything",
    "ensure_dir",
    "train_one_epoch",
    "evaluate",
    "save_vis_samples",
    "sliding_window_inference",
    "threshold_map",
    "filter_small_cc",
    "build_scheduler",
]
