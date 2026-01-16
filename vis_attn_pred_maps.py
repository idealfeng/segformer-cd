"""
Visualize per-layer ViT self-attention maps (DINOv3) and prediction maps.

Outputs (per sample) can be saved as separate images, a combined grid, or both.

Notes:
- Attention maps are extracted from DINOv3 (HF AutoModel) attentions using CLS->patch attention
  averaged over heads, for selected transformer blocks (1-based indices).
- For DINOv2 (torch.hub) backbones, attention extraction is not implemented (script will warn and
  only save prediction maps).

Example:
python vis_attn_pred_maps.py --checkpoint "outputs/ablation/best/BEST_whu--levir/dino_head_cd/best.pt" ^
  --data_root data/LEVIR-CD --split test --full_eval ^
  --indices 0,1,2 --layers 3,6,9,12 --save_mode both --out_dir outputs/ablation/attn_vis ^
  --thr_mode fixed --thr 0.5 --smooth_k 3 --use_minarea --min_area 256
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Avoid slow/blocked network calls in restricted environments.
os.environ.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from dino_head_core import (
    HeadCfg,
    DinoSiameseHead,
    DinoFrozenA0Head,
    build_dataloaders,
    seed_everything,
    sliding_window_inference,
    threshold_map,
    filter_small_cc,
)


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _parse_int_list(s: Optional[str]) -> Optional[List[int]]:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    return [int(x) for x in s.split(",") if str(x).strip() != ""]


def _denorm_img(x: torch.Tensor) -> np.ndarray:
    mean = torch.as_tensor(_IMAGENET_MEAN, device=x.device).view(3, 1, 1)
    std = torch.as_tensor(_IMAGENET_STD, device=x.device).view(3, 1, 1)
    img = (x * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).detach().cpu().numpy()


def _overlay_heat(img: np.ndarray, heat: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """
    img: [H,W,3] in [0,1]
    heat: [H,W] in [0,1]
    returns: [H,W,3] in [0,1]
    """
    import matplotlib

    heat = np.asarray(heat, dtype=np.float32)
    heat = np.nan_to_num(heat, nan=0.0, posinf=1.0, neginf=0.0)
    heat = np.clip(heat, 0.0, 1.0)

    cmap = matplotlib.cm.get_cmap("magma")
    h3 = cmap(heat)[..., :3].astype(np.float32)
    out = (1.0 - alpha) * img.astype(np.float32) + alpha * h3
    return np.clip(out, 0.0, 1.0)


def _make_pred_overlay(img: np.ndarray, pred: np.ndarray, gt: Optional[np.ndarray] = None) -> np.ndarray:
    out = img.copy()
    m = (pred > 0).astype(bool)
    if m.any():
        out[m] = 0.55 * out[m] + 0.45 * np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if gt is not None:
        try:
            import cv2

            gt_u8 = (gt > 0).astype(np.uint8)
            contours, _ = cv2.findContours(gt_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            out_u8 = (out * 255.0).astype(np.uint8)
            cv2.drawContours(out_u8, contours, -1, (0, 255, 0), 2)
            out = out_u8.astype(np.float32) / 255.0
        except Exception:
            pass
    return out


@torch.no_grad()
def _pad_to_patch_multiple(x: torch.Tensor, patch: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    _, _, H, W = x.shape
    H2 = ((H + patch - 1) // patch) * patch
    W2 = ((W + patch - 1) // patch) * patch
    pad_h, pad_w = H2 - H, W2 - W
    if pad_h == 0 and pad_w == 0:
        return x, (H, W)
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, (H, W)


@torch.no_grad()
def _extract_attn_maps_dinov3(
    model: torch.nn.Module,
    img: torch.Tensor,
    layers_1based: List[int],
    max_side: int = 512,
) -> Dict[int, np.ndarray]:
    """
    Returns per-layer attention map in [0,1] at image resolution (H,W).
    Uses CLS->patch attention averaged over heads.
    """
    if not hasattr(model, "backbone") or not getattr(model, "use_hf", False):
        raise RuntimeError("Backbone does not support HF attentions (need DINOv3).")
    backbone = model.backbone
    patch = int(getattr(model, "patch", 14))
    num_reg = int(getattr(model, "num_reg", 0))

    # Attention weights scale as O(N^2) with number of tokens; for large images this can OOM.
    # We compute attention on a resized view (keep aspect ratio) and upsample back to (H,W).
    H_in, W_in = img.shape[-2:]
    if max_side and max_side > 0 and max(H_in, W_in) > max_side:
        scale = float(max_side) / float(max(H_in, W_in))
        H_rs = max(1, int(round(H_in * scale)))
        W_rs = max(1, int(round(W_in * scale)))
        img_attn = F.interpolate(img, size=(H_rs, W_rs), mode="bilinear", align_corners=False)
    else:
        img_attn = img

    x, (H0, W0) = _pad_to_patch_multiple(img_attn, patch=patch)
    _, _, Hp, Wp = x.shape
    h, w = Hp // patch, Wp // patch

    # Transformers may default to SDPA/Flash attention, which does NOT support returning attentions.
    # Switch to eager attention for visualization, then restore previous mode to avoid affecting
    # subsequent forward passes (and to reduce memory use for prediction).
    prev_impl = None
    if hasattr(backbone, "config") and hasattr(backbone.config, "attn_implementation"):
        prev_impl = getattr(backbone.config, "attn_implementation", None)
    if hasattr(backbone, "set_attn_implementation"):
        try:
            backbone.set_attn_implementation("eager")
        except Exception:
            pass
    if hasattr(backbone, "config") and hasattr(backbone.config, "attn_implementation"):
        try:
            backbone.config.attn_implementation = "eager"
        except Exception:
            pass

    try:
        out = backbone(pixel_values=x, output_attentions=True, return_dict=True)
    finally:
        if prev_impl is not None and hasattr(backbone, "config") and hasattr(backbone.config, "attn_implementation"):
            try:
                backbone.config.attn_implementation = prev_impl
            except Exception:
                pass
    attns = getattr(out, "attentions", None)
    if attns is None:
        raise RuntimeError("HF backbone did not return attentions (output_attentions=True ineffective).")

    maps: Dict[int, np.ndarray] = {}
    for L in layers_1based:
        li = max(0, min(int(L) - 1, len(attns) - 1))
        a = attns[li]  # [B, heads, T, T]
        # CLS -> patch tokens, skip register tokens
        patch_start = 1 + num_reg
        cls2p = a[:, :, 0, patch_start:]  # [B, heads, Npatch]
        cls2p = cls2p.mean(dim=1)  # [B, Npatch]
        if cls2p.shape[1] != h * w:
            cls2p = cls2p[:, -h * w :]
        m = cls2p.reshape(1, h, w)
        # normalize per-map to [0,1]
        m = m - m.min()
        m = m / (m.max().clamp_min(1e-6))
        m = m.unsqueeze(1)  # [1,1,h,w]
        m = F.interpolate(m, size=(Hp, Wp), mode="bilinear", align_corners=False)[0, 0]
        m = m[..., :H0, :W0]
        if m.shape[-2:] != img_attn.shape[-2:]:
            m = F.interpolate(m.view(1, 1, *m.shape), size=img_attn.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        # Upsample to original image resolution for overlay.
        if m.shape[-2:] != (H_in, W_in):
            m = F.interpolate(m.view(1, 1, *m.shape), size=(H_in, W_in), mode="bilinear", align_corners=False)[0, 0]
        maps[int(L)] = m.detach().cpu().numpy().astype(np.float32)
    return maps


@torch.no_grad()
def _infer_pred_maps(
    model: torch.nn.Module,
    img_a: torch.Tensor,
    img_b: torch.Tensor,
    smooth_k: int,
    window: Optional[int],
    stride: Optional[int],
) -> Tuple[np.ndarray, Optional[List[np.ndarray]]]:
    """
    Returns:
      prob_fused: [H,W] float32
      probs_all:  list of [H,W] float32 for each head (if logits_all exists), else None
    """
    if window is not None and stride is not None:
        prob = sliding_window_inference(
            model=model,
            img_a=img_a,
            img_b=img_b,
            window=int(window),
            stride=int(stride),
            device=str(img_a.device),
            use_ensemble=False,
            ensemble_cfg=None,
        )
        out = None
    else:
        out = model(img_a, img_b)
        logits = out["pred"] if isinstance(out, dict) else out
        prob = torch.sigmoid(logits)
    if smooth_k and smooth_k > 1:
        pad = smooth_k // 2
        prob = F.avg_pool2d(prob, kernel_size=smooth_k, stride=1, padding=pad)
    prob_np = prob[0, 0].detach().cpu().numpy().astype(np.float32)

    probs_all = None
    if isinstance(out, dict) and out.get("logits_all") is not None:
        pa = torch.sigmoid(out["logits_all"])  # [K,B,1,H,W]
        if smooth_k and smooth_k > 1:
            K, B, _, H, W = pa.shape
            pa2 = pa.view(K * B, 1, H, W)
            pad = smooth_k // 2
            pa2 = F.avg_pool2d(pa2, kernel_size=smooth_k, stride=1, padding=pad)
            pa = pa2.view(K, B, 1, H, W)
        probs_all = [pa[k, 0, 0].detach().cpu().numpy().astype(np.float32) for k in range(pa.shape[0])]
    return prob_np, probs_all


def _save_image(path: str, arr: np.ndarray):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(4, 4))
    if arr.ndim == 2:
        plt.imshow(arr, cmap="magma", vmin=0.0, vmax=1.0)
    else:
        plt.imshow(arr)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_grid(
    out_path: str,
    t1: np.ndarray,
    t2: np.ndarray,
    gt: np.ndarray,
    pred_overlay: np.ndarray,
    attn_overlays: List[Tuple[str, np.ndarray]],
    *,
    font_scale: float = 1.0,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = [("T1", t1), ("T2", t2), ("GT", gt), ("Prediction", pred_overlay)] + attn_overlays
    ncol = len(cols)
    fig, axes = plt.subplots(1, ncol, figsize=(3.7 * ncol, 4.0))
    if ncol == 1:
        axes = [axes]

    font_scale = float(font_scale) if font_scale else 1.0
    title_fs = int(round(14 * font_scale))

    for ax, (title, img) in zip(axes, cols):
        ax.set_title(title, fontsize=title_fs)
        if img.ndim == 2:
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_paper_grid(
    out_path: str,
    rows: List[Dict[str, np.ndarray]],
    attn_titles: List[str],
    *,
    font_scale: float = 1.0,
    show_row_tags: bool = True,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    font_scale = float(font_scale) if font_scale else 1.0

    def _fs(x: float) -> int:
        return int(round(float(x) * font_scale))

    col_titles = ["T1", "T2", "GT", "Prediction"] + list(attn_titles)
    nrows = len(rows)
    ncols = len(col_titles)

    fig_w = 3.2 * ncols
    fig_h = 3.2 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=_fs(14))

    for r, item in enumerate(rows):
        axes[r, 0].imshow(item["t1"])
        axes[r, 1].imshow(item["t2"])
        axes[r, 2].imshow(item["gt"], cmap="gray", vmin=0, vmax=1)
        axes[r, 3].imshow(item["pred_overlay"])

        for j, t in enumerate(attn_titles):
            img = item.get(f"attn::{t}")
            ax = axes[r, 4 + j]
            if img is None:
                ax.imshow(item["t2"])
            else:
                ax.imshow(img)

        for c in range(ncols):
            axes[r, c].axis("off")

        if show_row_tags:
            row_tag = chr(ord("a") + r)
            axes[r, 0].text(
                -0.08,
                0.5,
                f"({row_tag})",
                transform=axes[r, 0].transAxes,
                va="center",
                ha="right",
                fontsize=_fs(22),
                color="black",
                clip_on=False,
            )

    plt.tight_layout(rect=[0.02, 0.0, 1.0, 1.0])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    base = HeadCfg()
    p = argparse.ArgumentParser(description="Visualize DINOv3 attention maps and prediction maps")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--split", type=str, default="test", choices=["val", "test"])
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="auto", help="cuda|cpu|auto")
    p.add_argument("--seed", type=int, default=base.seed)
    p.add_argument("--full_eval", dest="full_eval", action="store_true")
    p.add_argument("--no_full_eval", dest="full_eval", action="store_false")
    p.set_defaults(full_eval=True)
    p.add_argument("--eval_crop", type=int, default=base.eval_crop)
    p.add_argument("--indices", type=str, default="0", help="Comma-separated dataset indices to visualize")
    p.add_argument("--layers", type=str, default="3,6,9,12", help="1-based transformer block indices for attention maps")
    p.add_argument("--attn_max_side", type=int, default=512, help="Resize input for attention extraction to avoid OOM (0 disables).")
    p.add_argument("--save_mode", type=str, choices=["separate", "grid", "both"], default="both")
    p.add_argument("--paper_grid", action="store_true", help="Also save a single multi-row grid (paper-style) for all --indices.")
    p.add_argument("--no_row_tags", dest="show_row_tags", action="store_false", help="Disable (a)(b)(c)... row tags in --paper_grid.")
    p.set_defaults(show_row_tags=True)
    p.add_argument("--font_scale", type=float, default=1.35, help="Scale all font sizes for grid/paper figures.")
    p.add_argument("--window", type=int, default=None, help="Optional sliding-window size for prediction (recommended for full_eval large images).")
    p.add_argument("--stride", type=int, default=None, help="Optional sliding-window stride for prediction (must be set with --window).")

    # prediction post-processing (align with eval protocol)
    p.add_argument("--thr_mode", type=str, choices=["fixed", "topk", "otsu"], default=base.thr_mode)
    p.add_argument("--thr", type=float, default=base.thr)
    p.add_argument("--topk", type=float, default=base.topk)
    p.add_argument("--smooth_k", type=int, default=base.smooth_k)
    p.add_argument("--use_minarea", action="store_true", default=base.use_minarea)
    p.add_argument("--min_area", type=int, default=base.min_area)

    # Paper Figure 8 helper: 8 rows in a fixed order.
    # (a,c): LEVIR->LEVIR (in-domain)
    # (b,d): WHU->LEVIR (cross-domain)
    # (e,g): WHU->WHU (in-domain)
    # (f,h): LEVIR->WHU (cross-domain)
    p.add_argument("--figure8", action="store_true", help="Build Figure 8 (8-row) paper grid from two checkpoints and two datasets.")
    p.add_argument("--ckpt_levir", type=str, default=None, help="Checkpoint trained on LEVIR (used for LEVIR->LEVIR and LEVIR->WHU).")
    p.add_argument("--ckpt_whu", type=str, default=None, help="Checkpoint trained on WHU (used for WHU->WHU and WHU->LEVIR).")
    p.add_argument("--levir_root", type=str, default=None, help="LEVIR-CD dataset root.")
    p.add_argument("--whu_root", type=str, default=None, help="WHU-CD dataset root.")
    p.add_argument("--idx_levir_in", type=str, default="0,1", help="Two indices for LEVIR->LEVIR rows (a,c).")
    p.add_argument("--idx_whu2levir", type=str, default="0,1", help="Two indices for WHU->LEVIR rows (b,d).")
    p.add_argument("--idx_whu_in", type=str, default="0,1", help="Two indices for WHU->WHU rows (e,g).")
    p.add_argument("--idx_levir2whu", type=str, default="0,1", help="Two indices for LEVIR->WHU rows (f,h).")

    return p.parse_args()


def _load_model(ckpt_path: str, device: str) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device)
    load_cfg = ckpt.get("cfg") if isinstance(ckpt, dict) else None
    if isinstance(load_cfg, dict):
        arch = str(load_cfg.get("arch", "dlv"))
        if arch == "a0":
            model = DinoFrozenA0Head(
                dino_name=load_cfg.get("dino_name", "facebook/dinov3-vitb16-pretrain-lvd1689m"),
                layer=int(load_cfg.get("a0_layer", 12)),
                use_whiten=bool(load_cfg.get("use_whiten", False)),
            ).to(device)
        else:
            model = DinoSiameseHead(
                dino_name=load_cfg.get("dino_name", "facebook/dinov3-vitb16-pretrain-lvd1689m"),
                use_whiten=bool(load_cfg.get("use_whiten", False)),
                use_domain_adv=bool(load_cfg.get("use_domain_adv", False)),
                domain_hidden=int(load_cfg.get("domain_hidden", 256)),
                domain_grl=float(load_cfg.get("domain_grl", 1.0)),
                use_style_norm=bool(load_cfg.get("use_style_norm", False)),
                proto_path=str(load_cfg.get("proto_path", "")) or None,
                proto_weight=float(load_cfg.get("proto_weight", 0.0)),
                boundary_dim=int(load_cfg.get("boundary_dim", 0)),
                use_layer_ensemble=bool(load_cfg.get("use_layer_ensemble", False)),
                layer_head_ch=int(load_cfg.get("layer_head_ch", 128)),
            ).to(device)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    else:
        model = DinoSiameseHead().to(device)
        model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
    model.eval()
    return model


def _require_args(args, names: List[str]):
    missing = [n for n in names if not getattr(args, n)]
    if missing:
        raise ValueError(f"Missing required args: {', '.join('--' + m.replace('_', '-') for m in missing)}")


def _make_dataset(root_dir: str, split: str, *, full_eval: bool, eval_crop: int):
    from dataset import LEVIRCDDataset, get_val_transforms, get_test_transforms_full

    tf = get_test_transforms_full() if full_eval else get_val_transforms(crop_size=int(eval_crop))
    return LEVIRCDDataset(root_dir=Path(root_dir), split=str(split), transform=tf, crop_size=int(eval_crop))


def _get_sample(ds, idx: int) -> dict:
    item = ds[int(idx)]
    # Ensure batch dimension for model forward.
    img_a = item["img_a"].unsqueeze(0)
    img_b = item["img_b"].unsqueeze(0)
    label = item["label"]
    if label.ndim == 2:
        label = label.unsqueeze(0).unsqueeze(0)
    elif label.ndim == 3:
        label = label.unsqueeze(1)
    return {"img_a": img_a, "img_b": img_b, "label": label, "name": item.get("name", str(idx))}


def _build_row(
    *,
    model: torch.nn.Module,
    sample: dict,
    device: str,
    layers: List[int],
    attn_max_side: int,
    smooth_k: int,
    window: Optional[int],
    stride: Optional[int],
    thr_mode: str,
    thr: float,
    topk: float,
    use_minarea: bool,
    min_area: int,
) -> Tuple[
    Dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[Tuple[str, np.ndarray]],
    Dict[int, np.ndarray],
]:
    img_a = sample["img_a"].to(device)
    img_b = sample["img_b"].to(device)
    gt_t = sample["label"]
    if gt_t.ndim == 3:
        gt_t = gt_t.unsqueeze(1)
    gt_np = (gt_t[0, 0].detach().cpu().numpy() > 0).astype(np.uint8)

    prob_np, _ = _infer_pred_maps(
        model,
        img_a,
        img_b,
        smooth_k=int(smooth_k),
        window=window,
        stride=stride,
    )
    pred_np, _ = threshold_map(prob_np, thr_mode, float(thr), float(topk))
    if use_minarea:
        pred_np = filter_small_cc(pred_np, min_area=int(min_area))

    t1 = _denorm_img(img_a[0])
    t2 = _denorm_img(img_b[0])
    pred_overlay = _make_pred_overlay(t2, pred_np, gt=gt_np)

    attn_maps: Dict[int, np.ndarray] = {}
    if layers:
        try:
            attn_maps = _extract_attn_maps_dinov3(
                model,
                img_b,
                layers_1based=layers,
                max_side=int(attn_max_side),
            )
        except Exception as e:
            print(f"[Attn] Skip attention maps (reason: {e})")
            attn_maps = {}
            if str(device).startswith("cuda"):
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    attn_overlays: List[Tuple[str, np.ndarray]] = []
    for L in layers:
        title = f"Attn L{int(L)}"
        m = attn_maps.get(int(L))
        if m is None:
            ov = t2.copy()
        else:
            ov = _overlay_heat(t2, m)
        attn_overlays.append((title, ov))

    row: Dict[str, np.ndarray] = {
        "t1": t1,
        "t2": t2,
        "gt": gt_np.astype(np.uint8),
        "pred_overlay": pred_overlay,
    }
    for title, img in attn_overlays:
        row[f"attn::{title}"] = img

    return row, t1, t2, gt_np, prob_np, pred_overlay, attn_overlays, attn_maps


def main():
    args = parse_args()
    seed_everything(int(args.seed))
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if (args.window is None) ^ (args.stride is None):
        raise ValueError("--window and --stride must be set together (or both omitted).")

    layers = _parse_int_list(args.layers) or []

    if args.figure8:
        _require_args(args, ["ckpt_levir", "ckpt_whu", "levir_root", "whu_root"])
        idx_levir_in = _parse_int_list(args.idx_levir_in) or []
        idx_whu2levir = _parse_int_list(args.idx_whu2levir) or []
        idx_whu_in = _parse_int_list(args.idx_whu_in) or []
        idx_levir2whu = _parse_int_list(args.idx_levir2whu) or []
        if not (len(idx_levir_in) == len(idx_whu2levir) == len(idx_whu_in) == len(idx_levir2whu) == 2):
            raise ValueError("Figure8 requires exactly 2 indices for each of: idx_levir_in, idx_whu2levir, idx_whu_in, idx_levir2whu")

        model_levir = _load_model(args.ckpt_levir, device=device)
        model_whu = _load_model(args.ckpt_whu, device=device)
        ds_levir = _make_dataset(args.levir_root, args.split, full_eval=bool(args.full_eval), eval_crop=int(args.eval_crop))
        ds_whu = _make_dataset(args.whu_root, args.split, full_eval=bool(args.full_eval), eval_crop=int(args.eval_crop))

        # Fixed row order for Figure 8: a,b,c,d,e,f,g,h
        # a,c: LEVIR->LEVIR; b,d: WHU->LEVIR; e,g: WHU->WHU; f,h: LEVIR->WHU
        plan = [
            ("a", model_levir, ds_levir, idx_levir_in[0], "LEVIR2LEVIR"),
            ("b", model_whu, ds_levir, idx_whu2levir[0], "WHU2LEVIR"),
            ("c", model_levir, ds_levir, idx_levir_in[1], "LEVIR2LEVIR"),
            ("d", model_whu, ds_levir, idx_whu2levir[1], "WHU2LEVIR"),
            ("e", model_whu, ds_whu, idx_whu_in[0], "WHU2WHU"),
            ("f", model_levir, ds_whu, idx_levir2whu[0], "LEVIR2WHU"),
            ("g", model_whu, ds_whu, idx_whu_in[1], "WHU2WHU"),
            ("h", model_levir, ds_whu, idx_levir2whu[1], "LEVIR2WHU"),
        ]

        paper_rows: List[Dict[str, np.ndarray]] = []
        for tag, model, ds, idx, scenario in plan:
            sample = _get_sample(ds, int(idx))
            name = sample["name"]
            row, t1, t2, gt_np, prob_np, pred_overlay, attn_overlays, attn_maps = _build_row(
                model=model,
                sample=sample,
                device=device,
                layers=layers,
                attn_max_side=int(args.attn_max_side),
                smooth_k=int(args.smooth_k),
                window=args.window,
                stride=args.stride,
                thr_mode=str(args.thr_mode),
                thr=float(args.thr),
                topk=float(args.topk),
                use_minarea=bool(args.use_minarea),
                min_area=int(args.min_area),
            )

            sample_dir = out_dir / f"{tag}_{scenario}_idx{int(idx):04d}_{name}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            if args.save_mode in ("separate", "both"):
                _save_image(str(sample_dir / "t1.png"), t1)
                _save_image(str(sample_dir / "t2.png"), t2)
                _save_image(str(sample_dir / "gt.png"), gt_np.astype(np.uint8))
                _save_image(str(sample_dir / "pred_prob.png"), np.clip(prob_np, 0.0, 1.0))
                _save_image(str(sample_dir / "pred_overlay.png"), pred_overlay)

                for L in layers:
                    title = f"Attn L{int(L)}"
                    ov = row.get(f"attn::{title}")
                    if ov is not None:
                        try:
                            _save_image(str(sample_dir / f"attn_L{int(L):02d}_overlay.png"), ov)
                        except Exception:
                            pass
                    m = attn_maps.get(int(L))
                    if m is not None:
                        try:
                            _save_image(str(sample_dir / f"attn_L{int(L):02d}.png"), np.clip(np.nan_to_num(m, nan=0.0), 0.0, 1.0))
                        except Exception:
                            pass

            if args.save_mode in ("grid", "both"):
                grid_path = sample_dir / "grid.png"
                _plot_grid(
                    out_path=str(grid_path),
                    t1=t1,
                    t2=t2,
                    gt=gt_np.astype(np.uint8),
                    pred_overlay=pred_overlay,
                    attn_overlays=attn_overlays,
                    font_scale=float(args.font_scale),
                )

            paper_rows.append(row)
            print(f"[OK] Saved {sample_dir}")

        attn_titles_all = [f"Attn L{int(L)}" for L in layers]
        out_path = out_dir / "paper_grid.png"
        _plot_paper_grid(
            out_path=str(out_path),
            rows=paper_rows,
            attn_titles=attn_titles_all,
            font_scale=float(args.font_scale),
            show_row_tags=bool(args.show_row_tags),
        )
        print(f"[OK] Saved {out_path}")

        manifest = {
            "mode": "figure8",
            "ckpt_levir": str(args.ckpt_levir),
            "ckpt_whu": str(args.ckpt_whu),
            "levir_root": str(args.levir_root),
            "whu_root": str(args.whu_root),
            "split": str(args.split),
            "layers": layers,
            "idx_levir_in": idx_levir_in,
            "idx_whu2levir": idx_whu2levir,
            "idx_whu_in": idx_whu_in,
            "idx_levir2whu": idx_levir2whu,
            "args": vars(args),
        }
        (out_dir / "manifest.json").write_text(str(manifest), encoding="utf-8")
        return

    if args.checkpoint is None:
        raise ValueError("--checkpoint is required unless --figure8 is set.")
    model = _load_model(args.checkpoint, device=device)
    indices = _parse_int_list(args.indices) or [0]

    data_root = args.data_root if args.data_root is not None else HeadCfg().data_root
    cfg = HeadCfg(
        data_root=str(data_root),
        out_dir=str(out_dir),
        seed=int(args.seed),
        device=device,
        full_eval=bool(args.full_eval),
        eval_crop=int(args.eval_crop),
    )
    cfg.batch_size = 1
    _, val_loader, test_loader = build_dataloaders(cfg, require_train=False, require_val=(args.split == "val"))
    loader = val_loader if args.split == "val" else test_loader

    # Collect requested indices while preserving the requested order.
    wanted = set(indices)
    found: Dict[int, Dict[str, np.ndarray]] = {}
    meta: Dict[
        int,
        Tuple[
            str,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            List[Tuple[str, np.ndarray]],
            Dict[int, np.ndarray],
        ],
    ] = {}
    for idx, batch in enumerate(loader):
        if idx not in wanted:
            continue
        name = batch["name"][0] if isinstance(batch["name"], list) else batch["name"]
        sample = {
            "img_a": batch["img_a"],
            "img_b": batch["img_b"],
            "label": batch["label"],
            "name": name,
        }
        row, t1, t2, gt_np, prob_np, pred_overlay, attn_overlays, attn_maps = _build_row(
            model=model,
            sample=sample,
            device=device,
            layers=layers,
            attn_max_side=int(args.attn_max_side),
            smooth_k=int(args.smooth_k),
            window=args.window,
            stride=args.stride,
            thr_mode=str(args.thr_mode),
            thr=float(args.thr),
            topk=float(args.topk),
            use_minarea=bool(args.use_minarea),
            min_area=int(args.min_area),
        )
        found[int(idx)] = row
        meta[int(idx)] = (str(name), t1, t2, gt_np, prob_np, pred_overlay, attn_overlays, attn_maps)
        if len(found) >= len(wanted):
            break

    paper_rows: List[Dict[str, np.ndarray]] = []
    for idx in indices:
        if int(idx) not in found:
            continue
        row = found[int(idx)]
        name, t1, t2, gt_np, prob_np, pred_overlay, attn_overlays, attn_maps = meta[int(idx)]

        sample_dir = out_dir / f"idx{int(idx):04d}_{name}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        if args.save_mode in ("separate", "both"):
            _save_image(str(sample_dir / "t1.png"), t1)
            _save_image(str(sample_dir / "t2.png"), t2)
            _save_image(str(sample_dir / "gt.png"), gt_np.astype(np.uint8))
            _save_image(str(sample_dir / "pred_prob.png"), np.clip(prob_np, 0.0, 1.0))
            _save_image(str(sample_dir / "pred_overlay.png"), pred_overlay)
            for L in layers:
                title = f"Attn L{int(L)}"
                ov = row.get(f"attn::{title}")
                if ov is not None:
                    try:
                        _save_image(str(sample_dir / f"attn_L{int(L):02d}_overlay.png"), ov)
                    except Exception:
                        pass
                m = attn_maps.get(int(L))
                if m is not None:
                    try:
                        _save_image(str(sample_dir / f"attn_L{int(L):02d}.png"), np.clip(np.nan_to_num(m, nan=0.0), 0.0, 1.0))
                    except Exception:
                        pass

        if args.save_mode in ("grid", "both"):
            grid_path = sample_dir / "grid.png"
            _plot_grid(
                out_path=str(grid_path),
                t1=t1,
                t2=t2,
                gt=gt_np.astype(np.uint8),
                pred_overlay=pred_overlay,
                attn_overlays=attn_overlays,
                font_scale=float(args.font_scale),
            )

        if args.paper_grid:
            paper_rows.append(row)

        print(f"[OK] Saved {sample_dir}")

    if args.paper_grid and paper_rows:
        attn_titles_all = [f"Attn L{int(L)}" for L in layers]
        out_path = out_dir / "paper_grid.png"
        _plot_paper_grid(
            out_path=str(out_path),
            rows=paper_rows,
            attn_titles=attn_titles_all,
            font_scale=float(args.font_scale),
            show_row_tags=bool(args.show_row_tags),
        )
        print(f"[OK] Saved {out_path}")

    # Save a small manifest
    manifest = {
        "mode": "single",
        "checkpoint": str(args.checkpoint),
        "data_root": str(data_root),
        "split": str(args.split),
        "indices": indices,
        "layers": layers,
        "args": vars(args),
    }
    (out_dir / "manifest.json").write_text(str(manifest), encoding="utf-8")


if __name__ == "__main__":
    main()
