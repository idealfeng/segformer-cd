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
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = [("T1", t1), ("T2", t2), ("GT", gt), ("Prediction", pred_overlay)] + attn_overlays
    ncol = len(cols)
    fig, axes = plt.subplots(1, ncol, figsize=(3.7 * ncol, 4.0))
    if ncol == 1:
        axes = [axes]

    for ax, (title, img) in zip(axes, cols):
        ax.set_title(title, fontsize=12, fontweight="bold")
        if img.ndim == 2:
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    base = HeadCfg()
    p = argparse.ArgumentParser(description="Visualize DINOv3 attention maps and prediction maps")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_root", type=str, default=base.data_root)
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
    p.add_argument("--window", type=int, default=None, help="Optional sliding-window size for prediction (recommended for full_eval large images).")
    p.add_argument("--stride", type=int, default=None, help="Optional sliding-window stride for prediction (must be set with --window).")

    # prediction post-processing (align with eval protocol)
    p.add_argument("--thr_mode", type=str, choices=["fixed", "topk", "otsu"], default=base.thr_mode)
    p.add_argument("--thr", type=float, default=base.thr)
    p.add_argument("--topk", type=float, default=base.topk)
    p.add_argument("--smooth_k", type=int, default=base.smooth_k)
    p.add_argument("--use_minarea", action="store_true", default=base.use_minarea)
    p.add_argument("--min_area", type=int, default=base.min_area)

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


def main():
    args = parse_args()
    seed_everything(int(args.seed))
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = _load_model(args.checkpoint, device=device)
    indices = _parse_int_list(args.indices) or [0]
    layers = _parse_int_list(args.layers) or []
    if (args.window is None) ^ (args.stride is None):
        raise ValueError("--window and --stride must be set together (or both omitted).")

    cfg = HeadCfg(
        data_root=args.data_root,
        out_dir=str(out_dir),
        seed=int(args.seed),
        device=device,
        full_eval=bool(args.full_eval),
        eval_crop=int(args.eval_crop),
    )
    cfg.batch_size = 1
    _, val_loader, test_loader = build_dataloaders(cfg)
    loader = val_loader if args.split == "val" else test_loader

    # For quick lookup, we iterate once and keep requested indices.
    wanted = set(indices)
    found = 0
    for idx, batch in enumerate(loader):
        if idx not in wanted:
            continue
        found += 1
        name = batch["name"][0] if isinstance(batch["name"], list) else batch["name"]

        img_a = batch["img_a"].to(device)
        img_b = batch["img_b"].to(device)
        gt_t = batch["label"]
        if gt_t.ndim == 3:
            gt_t = gt_t.unsqueeze(1)
        gt_np = (gt_t[0, 0].detach().cpu().numpy() > 0).astype(np.uint8)

        prob_np, _ = _infer_pred_maps(
            model,
            img_a,
            img_b,
            smooth_k=int(args.smooth_k),
            window=args.window,
            stride=args.stride,
        )
        pred_np, _ = threshold_map(prob_np, args.thr_mode, float(args.thr), float(args.topk))
        if args.use_minarea:
            pred_np = filter_small_cc(pred_np, min_area=int(args.min_area))

        t1 = _denorm_img(img_a[0])
        t2 = _denorm_img(img_b[0])
        pred_overlay = _make_pred_overlay(t2, pred_np, gt=gt_np)

        sample_dir = out_dir / f"idx{idx:04d}_{name}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # save prediction maps
        if args.save_mode in ("separate", "both"):
            _save_image(str(sample_dir / "t1.png"), t1)
            _save_image(str(sample_dir / "t2.png"), t2)
            _save_image(str(sample_dir / "gt.png"), gt_np.astype(np.uint8))
            _save_image(str(sample_dir / "pred_prob.png"), np.clip(prob_np, 0.0, 1.0))
            _save_image(str(sample_dir / "pred_overlay.png"), pred_overlay)

        attn_overlays: List[Tuple[str, np.ndarray]] = []
        if layers:
            try:
                attn_maps = _extract_attn_maps_dinov3(
                    model,
                    img_b,
                    layers_1based=layers,
                    max_side=int(args.attn_max_side),
                )
                for L in layers:
                    m = attn_maps.get(int(L))
                    if m is None:
                        continue
                    # normalize to [0,1] (already), overlay on T2 for readability
                    ov = _overlay_heat(t2, np.clip(m, 0.0, 1.0))
                    title = f"Attn L{int(L)}"
                    attn_overlays.append((title, ov))
                    if args.save_mode in ("separate", "both"):
                        _save_image(str(sample_dir / f"attn_L{int(L):02d}_overlay.png"), ov)
                        _save_image(str(sample_dir / f"attn_L{int(L):02d}.png"), np.clip(m, 0.0, 1.0))
            except Exception as e:
                print(f"[Attn] Skip attention maps for idx={idx} (reason: {e})")
                if device.startswith("cuda"):
                    try:
                        torch.cuda.empty_cache()
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
            )

        print(f"[OK] Saved {sample_dir}")
        if found >= len(wanted):
            break

    # Save a small manifest
    manifest = {
        "checkpoint": str(args.checkpoint),
        "data_root": str(args.data_root),
        "split": str(args.split),
        "indices": indices,
        "layers": layers,
        "args": vars(args),
    }
    (out_dir / "manifest.json").write_text(str(manifest), encoding="utf-8")


if __name__ == "__main__":
    main()
