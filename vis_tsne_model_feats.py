"""
t-SNE visualization for a trained checkpoint (no retraining).

This script extracts feature vectors from a trained model on two datasets (e.g., LEVIR vs WHU),
then runs PCA and optionally t-SNE to visualize the feature space.

Two feature sources are supported:
  - backbone: DINOv3 hidden-state tokens mean (like vis_domain_shift.py)
  - head:     change-head feature map ("feat" in DinoSiameseHead forward) pooled to a vector

Example:
  python vis_tsne_model_feats.py ^
    --checkpoint outputs/ablation/best/BEST_whu--levir/dino_head_cd/best.pt ^
    --src_root data/LEVIR-CD --tgt_root data/WHUCD --split test ^
    --feature head --which both_avg --num_samples 400 --out_dir outputs/tsne_head
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    raise RuntimeError(f"matplotlib is required for vis_tsne_model_feats.py: {e}")

try:
    from sklearn.manifold import TSNE  # type: ignore
except Exception:
    TSNE = None

try:
    from transformers import AutoModel  # type: ignore
except Exception:
    AutoModel = None

from dataset import LEVIRCDDataset, get_val_transforms, worker_init_fn
from dino_head_core import HeadCfg, seed_everything
from models.dinov2_head import DinoSiameseHead, DinoFrozenA0Head


@dataclass
class PointMeta:
    dataset: str  # "src" | "tgt"
    changed: int  # 0/1


def _pca_2d(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=0, keepdims=True)
    xc = x - mean
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    comps = vt[:2]
    proj = xc @ comps.T
    return proj.astype(np.float32), mean.squeeze(0).astype(np.float32), comps.astype(np.float32)


def _pca_reduce(x: np.ndarray, k: int) -> np.ndarray:
    """
    PCA dimensionality reduction via SVD (centered), returns [N,k].
    """
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=0, keepdims=True)
    xc = x - mean
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    k = int(max(1, min(int(k), vt.shape[0])))
    comps = vt[:k]  # [k,D]
    proj = xc @ comps.T  # [N,k]
    return proj.astype(np.float32)


def _load_model(ckpt_path: str, device: str) -> torch.nn.Module:
    # Newer PyTorch warns about the default weights_only=False; be explicit to keep logs clean.
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
    arch = str(cfg.get("arch", "dlv"))
    if arch == "a0":
        model = DinoFrozenA0Head(
            dino_name=str(cfg.get("dino_name", HeadCfg().dino_name)),
            layer=int(cfg.get("a0_layer", 12)),
            use_whiten=bool(cfg.get("use_whiten", False)),
        ).to(device)
    else:
        model = DinoSiameseHead(
            dino_name=str(cfg.get("dino_name", HeadCfg().dino_name)),
            use_whiten=bool(cfg.get("use_whiten", False)),
            use_domain_adv=bool(cfg.get("use_domain_adv", False)),
            domain_hidden=int(cfg.get("domain_hidden", 256)),
            domain_grl=float(cfg.get("domain_grl", 1.0)),
            use_style_norm=bool(cfg.get("use_style_norm", False)),
            proto_path=str(cfg.get("proto_path", "")) or None,
            proto_weight=float(cfg.get("proto_weight", 0.0)),
            boundary_dim=int(cfg.get("boundary_dim", 0)),
            use_layer_ensemble=bool(cfg.get("use_layer_ensemble", False)),
            layer_head_ch=int(cfg.get("layer_head_ch", 128)),
            backbone_grad=False,
        ).to(device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    elif isinstance(ckpt, dict):
        model.load_state_dict(ckpt)
    model.eval()
    return model


def _load_pretrained_dinov3(dino_name: str, device: str) -> torch.nn.Module:
    """
    Load official pretrained DINOv3 encoder (frozen) via HF AutoModel.
    Supports local directory checkpoints (recommended in offline environments).
    """
    if AutoModel is None:
        raise ImportError("transformers is required for pretrained DINOv3 feature extraction.")
    dino_name = str(dino_name)
    try:
        backbone = AutoModel.from_pretrained(dino_name, trust_remote_code=True, local_files_only=True).to(device)
    except TypeError:
        backbone = AutoModel.from_pretrained(dino_name, trust_remote_code=True).to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone


def _make_tsne(*, perplexity: float, iters: int, seed: int):
    if TSNE is None:
        raise RuntimeError("Requested --tsne but scikit-learn is not available.")
    import inspect

    sig = inspect.signature(TSNE.__init__)
    params = sig.parameters
    kw = dict(
        n_components=2,
        perplexity=float(perplexity),
        init="pca",
        learning_rate="auto",
        random_state=int(seed),
    )
    # scikit-learn 1.6 prefers max_iter; older versions used n_iter.
    if "max_iter" in params:
        kw["max_iter"] = int(iters)
    elif "n_iter" in params:
        kw["n_iter"] = int(iters)
    else:
        raise RuntimeError("Unsupported scikit-learn TSNE API (missing max_iter/n_iter).")
    return TSNE(**kw)


def _changed_flag(mask: torch.Tensor, thr: float) -> int:
    m = mask
    if m.ndim == 3:
        m = m.unsqueeze(1)
    frac = float((m > 0).float().mean().item())
    return int(frac >= float(thr))


@torch.no_grad()
def _embed_dinov3_patch_mean(
    backbone: torch.nn.Module,
    imgs: torch.Tensor,
    *,
    layer: int,
    num_reg: int,
) -> torch.Tensor:
    """
    Global vector per image from pretrained DINOv3:
      mean pooling over patch tokens (exclude CLS and register tokens).

    Layer convention: 1-based transformer block index (e.g., 3/6/9/12 for ViT-B/ViT-S).
    """
    cfg = getattr(backbone, "config", None)
    num_layers = int(getattr(cfg, "num_hidden_layers", 0) or 0)
    layer = int(layer)

    # If requesting the last layer, avoid returning all hidden states for speed/memory.
    if num_layers and layer >= num_layers:
        out = backbone(pixel_values=imgs, output_hidden_states=False, return_dict=True)
        feat = out.last_hidden_state  # [B, 1+reg+N, C]
    else:
        out = backbone(pixel_values=imgs, output_hidden_states=True, return_dict=True)
        hs = out.hidden_states
        if hs is None:
            hs = [out.last_hidden_state]
        real_idx = min(layer, len(hs) - 1)
        feat = hs[real_idx]  # [B, 1+reg+N, C]

    tokens = feat[:, 1 + int(num_reg) :, :]
    return tokens.mean(dim=1)  # [B,C]


@torch.no_grad()
def _embed_backbone_dinov3(model: DinoSiameseHead, imgs: torch.Tensor, layer: int) -> torch.Tensor:
    out = model.backbone(pixel_values=imgs, output_hidden_states=True, return_dict=True)
    hs = out.hidden_states
    if hs is None:
        hs = [out.last_hidden_state]
    real_idx = min(int(layer), len(hs) - 1)
    feat = hs[real_idx]  # [B, 1+reg+N, C]
    num_reg = int(getattr(model, "num_reg", 0) or 0)
    tokens = feat[:, 1 + num_reg :, :]
    return tokens.mean(dim=1)  # [B,C]


@torch.no_grad()
def _embed_head_feat(model: DinoSiameseHead, img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
    out = model(img_a, img_b)
    feat = out.get("feat", None) if isinstance(out, dict) else None
    if feat is None:
        raise RuntimeError("Model forward did not return 'feat' (requires DinoSiameseHead).")
    # feat: [B,C,h,w] -> [B,C]
    return feat.mean(dim=(2, 3))


@torch.no_grad()
def _embed_batch(
    model: torch.nn.Module,
    batch: Dict,
    *,
    feature: str,
    which: str,
    layer: int,
    device: str,
) -> torch.Tensor:
    img_a = batch["img_a"].to(device, non_blocking=True)
    img_b = batch["img_b"].to(device, non_blocking=True)
    which = str(which).lower()

    if feature == "head":
        if not isinstance(model, DinoSiameseHead):
            raise RuntimeError("--feature head requires a DinoSiameseHead checkpoint (arch=dlv).")
        return _embed_head_feat(model, img_a, img_b)

    if feature != "backbone":
        raise ValueError("--feature must be one of: backbone | head")

    if not (isinstance(model, DinoSiameseHead) and getattr(model, "use_hf", False)):
        raise RuntimeError("--feature backbone currently supports DINOv3 (HF) checkpoints only.")

    if which == "t1":
        imgs = img_a
        emb = _embed_backbone_dinov3(model, imgs, layer=layer)
    elif which == "t2":
        imgs = img_b
        emb = _embed_backbone_dinov3(model, imgs, layer=layer)
    elif which == "mix":
        B = img_a.shape[0]
        sel = (torch.rand(B, device=img_a.device) < 0.5).view(B, 1, 1, 1)
        imgs = torch.where(sel, img_a, img_b)
        emb = _embed_backbone_dinov3(model, imgs, layer=layer)
    elif which == "both_avg":
        imgs = torch.cat([img_a, img_b], dim=0)
        emb2 = _embed_backbone_dinov3(model, imgs, layer=layer)
        B = img_a.shape[0]
        emb = 0.5 * (emb2[:B] + emb2[B:])
    else:
        raise ValueError("--which must be one of: t1 | t2 | mix | both_avg")
    return emb


def _collect(
    *,
    model: torch.nn.Module,
    data_root: str,
    split: str,
    dataset_tag: str,
    crop: int,
    batch_size: int,
    num_workers: int,
    num_samples: int,
    feature: str,
    which: str,
    layer: int,
    device: str,
    seed: int,
    change_thr: float,
) -> Tuple[np.ndarray, List[PointMeta]]:
    tf = get_val_transforms(crop_size=int(crop))
    ds = LEVIRCDDataset(root_dir=data_root, split=split, transform=tf, crop_size=int(crop))

    g = torch.Generator()
    g.manual_seed(int(seed))
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        generator=g,
    )

    feats: List[np.ndarray] = []
    meta: List[PointMeta] = []
    got = 0
    for batch in loader:
        emb = _embed_batch(
            model,
            batch,
            feature=str(feature),
            which=str(which),
            layer=int(layer),
            device=str(device),
        )
        emb_np = emb.detach().float().cpu().numpy()
        flags = [_changed_flag(m, thr=float(change_thr)) for m in batch["label"]]

        if got + emb_np.shape[0] > num_samples:
            keep = max(0, int(num_samples - got))
            emb_np = emb_np[:keep]
            flags = flags[:keep]

        feats.append(emb_np)
        meta.extend([PointMeta(dataset=str(dataset_tag), changed=int(f)) for f in flags])
        got += emb_np.shape[0]
        if got >= num_samples:
            break

    if got == 0:
        raise RuntimeError(f"No samples collected from {data_root} split={split}.")
    return np.concatenate(feats, axis=0), meta


def _scatter_2d(ax, xy: np.ndarray, meta: List[PointMeta], *, title: str, color_by: str):
    ds = np.array([m.dataset for m in meta])
    changed = np.array([m.changed for m in meta]).astype(int)

    if color_by == "dataset":
        colors = np.where(ds == "src", "#1f77b4", "#ff7f0e")
        markers = np.where(changed == 1, "o", "^")
        for mk in ["o", "^"]:
            sel = markers == mk
            ax.scatter(xy[sel, 0], xy[sel, 1], s=10, alpha=0.75, c=colors[sel], marker=mk)
        ax.scatter([], [], c="#1f77b4", marker="o", label="src (changed)")
        ax.scatter([], [], c="#1f77b4", marker="^", label="src (no-change)")
        ax.scatter([], [], c="#ff7f0e", marker="o", label="tgt (changed)")
        ax.scatter([], [], c="#ff7f0e", marker="^", label="tgt (no-change)")
    elif color_by == "change":
        colors = np.where(changed == 1, "#d62728", "#2ca02c")
        markers = np.where(ds == "src", "o", "^")
        for mk in ["o", "^"]:
            sel = markers == mk
            ax.scatter(xy[sel, 0], xy[sel, 1], s=10, alpha=0.75, c=colors[sel], marker=mk)
        ax.scatter([], [], c="#d62728", marker="o", label="changed (src)")
        ax.scatter([], [], c="#d62728", marker="^", label="changed (tgt)")
        ax.scatter([], [], c="#2ca02c", marker="o", label="no-change (src)")
        ax.scatter([], [], c="#2ca02c", marker="^", label="no-change (tgt)")
    else:
        raise ValueError("--color_by must be one of: dataset | change")

    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper right", frameon=True, fontsize=8, markerscale=1.5)


def parse_args():
    p = argparse.ArgumentParser("t-SNE/PCA visualization for a single checkpoint")
    # Preferred (paper) mode: pretrained DINOv3 encoder over multiple datasets (>=2).
    p.add_argument("--data_roots", type=str, default=None, help="Comma-separated dataset roots (>=2).")
    p.add_argument("--data_names", type=str, default=None, help="Comma-separated dataset names (optional).")
    p.add_argument("--dino_name", type=str, default="dinov3-vitb16", help="HF id or local dir for official pretrained DINOv3.")

    # Legacy mode: a trained checkpoint with two datasets (src/tgt).
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--src_root", type=str, default=None, help="Legacy: source dataset root")
    p.add_argument("--tgt_root", type=str, default=None, help="Legacy: target dataset root")
    p.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--crop", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--num_samples", type=int, default=300, help="Per-dataset sample count.")

    p.add_argument("--feature", type=str, choices=["backbone", "head"], default="head")
    p.add_argument("--which", type=str, choices=["t1", "t2", "mix", "both_avg"], default="mix")
    p.add_argument("--layer", type=int, default=12, help="Backbone layer (only for --feature backbone)")
    p.add_argument("--change_thr", type=float, default=0.01, help="Mask fraction threshold for changed vs no-change tag.")
    p.add_argument("--color_by", type=str, choices=["dataset", "change"], default="dataset")
    p.add_argument("--pca_dim", type=int, default=50, help="PCA dim before t-SNE (standard is 50).")

    p.add_argument("--tsne", action="store_true", help="Also run t-SNE (requires scikit-learn).")
    p.add_argument("--tsne_perplexity", type=float, default=30.0)
    p.add_argument("--tsne_iter", type=int, default=1000)
    return p.parse_args()


def _parse_csv(s: Optional[str]) -> List[str]:
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _scatter_by_dataset(ax, xy: np.ndarray, meta: List[PointMeta], *, title: str):
    ds = np.array([m.dataset for m in meta], dtype=object)
    uniq = list(dict.fromkeys(ds.tolist()))
    cmap = plt.get_cmap("tab10")
    for i, name in enumerate(uniq):
        sel = ds == name
        ax.scatter(
            xy[sel, 0],
            xy[sel, 1],
            s=10,
            alpha=0.78,
            c=[cmap(i % 10)],
            label=str(name),
        )
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper right", frameon=True, fontsize=9, markerscale=1.7)


@torch.no_grad()
def _collect_pretrained(
    *,
    backbone: torch.nn.Module,
    data_root: str,
    dataset_name: str,
    split: str,
    crop: int,
    batch_size: int,
    num_workers: int,
    num_samples: int,
    which: str,
    layer: int,
    device: str,
    seed: int,
) -> Tuple[np.ndarray, List[PointMeta]]:
    tf = get_val_transforms(crop_size=int(crop))
    ds = LEVIRCDDataset(root_dir=data_root, split=split, transform=tf, crop_size=int(crop))

    g = torch.Generator()
    g.manual_seed(int(seed))
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        generator=g,
    )

    cfg = getattr(backbone, "config", None)
    num_reg = int(getattr(cfg, "num_register_tokens", 0) or 0)

    feats: List[np.ndarray] = []
    meta: List[PointMeta] = []
    got = 0
    for batch in loader:
        img_a = batch["img_a"].to(device, non_blocking=True)
        img_b = batch["img_b"].to(device, non_blocking=True)
        which2 = str(which).lower()
        if which2 == "t1":
            imgs = img_a
        elif which2 == "t2":
            imgs = img_b
        elif which2 == "mix":
            B = img_a.shape[0]
            sel = (torch.rand(B, device=img_a.device) < 0.5).view(B, 1, 1, 1)
            imgs = torch.where(sel, img_a, img_b)
        else:
            raise ValueError("--which must be one of: t1 | t2 | mix (for pretrained mode)")

        emb = _embed_dinov3_patch_mean(
            backbone,
            imgs,
            layer=int(layer),
            num_reg=int(num_reg),
        )
        emb_np = emb.detach().float().cpu().numpy()

        if got + emb_np.shape[0] > num_samples:
            keep = max(0, int(num_samples - got))
            emb_np = emb_np[:keep]

        feats.append(emb_np)
        meta.extend([PointMeta(dataset=str(dataset_name), changed=0) for _ in range(emb_np.shape[0])])
        got += emb_np.shape[0]
        if got >= num_samples:
            break

    if got == 0:
        raise RuntimeError(f"No samples collected from {data_root} split={split}.")
    return np.concatenate(feats, axis=0), meta


def _run_pretrained(args):
    device = str(args.device)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    roots = _parse_csv(args.data_roots)
    if len(roots) < 2:
        raise ValueError("--data_roots must contain at least 2 dataset roots (paper: 3).")

    names = _parse_csv(args.data_names)
    if names and len(names) != len(roots):
        raise ValueError("--data_names must match --data_roots length (or be omitted).")
    if not names:
        names = [Path(r).name for r in roots]

    backbone = _load_pretrained_dinov3(str(args.dino_name), device=device)

    xs: List[np.ndarray] = []
    meta: List[PointMeta] = []
    for i, (r, n) in enumerate(zip(roots, names)):
        x_i, m_i = _collect_pretrained(
            backbone=backbone,
            data_root=str(r),
            dataset_name=str(n),
            split=str(args.split),
            crop=int(args.crop),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            num_samples=int(args.num_samples),
            which=str(args.which),
            layer=int(args.layer),
            device=str(device),
            seed=int(args.seed) + 17 * i,
        )
        xs.append(x_i)
        meta.extend(m_i)

    x = np.concatenate(xs, axis=0)

    np.savez_compressed(
        str(out_dir / "embeddings.npz"),
        x=x.astype(np.float32),
        dataset=np.array([m.dataset for m in meta], dtype=object),
    )

    pca_xy, _, _ = _pca_2d(x)
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 5.6), dpi=160)
    _scatter_by_dataset(
        ax,
        pca_xy,
        meta,
        title=f"PCA-2D (pretrained DINOv3 L{int(args.layer)}, patch-mean; which={args.which})",
    )
    fig.tight_layout()
    fig.savefig(str(out_dir / "pca.png"), bbox_inches="tight")
    plt.close(fig)

    report: Dict[str, object] = {
        "mode": "pretrained",
        "dino_name": str(args.dino_name),
        "layer": int(args.layer),
        "pooling": "patch_mean (exclude CLS and register tokens)",
        "which": str(args.which),
        "split": str(args.split),
        "crop": int(args.crop),
        "num_samples_per_dataset": int(args.num_samples),
        "datasets": [{"name": str(n), "root": str(r)} for n, r in zip(names, roots)],
    }

    if args.tsne:
        x50 = _pca_reduce(x, k=int(args.pca_dim))
        tsne = _make_tsne(
            perplexity=float(args.tsne_perplexity),
            iters=int(args.tsne_iter),
            seed=int(args.seed),
        )
        tsne_xy = tsne.fit_transform(x50.astype(np.float32))
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 5.6), dpi=160)
        _scatter_by_dataset(
            ax,
            tsne_xy,
            meta,
            title=f"t-SNE (PCA{int(args.pca_dim)}→2D; perplexity={float(args.tsne_perplexity):g})",
        )
        fig.tight_layout()
        fig.savefig(str(out_dir / "tsne.png"), bbox_inches="tight")
        plt.close(fig)
        report["tsne"] = {
            "pca_dim": int(args.pca_dim),
            "perplexity": float(args.tsne_perplexity),
            "iter": int(args.tsne_iter),
            "seed": int(args.seed),
        }

    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] Saved pretrained PCA/t-SNE outputs to: {out_dir}")

def main():
    args = parse_args()
    seed_everything(int(args.seed))

    # Preferred (paper) mode: pretrained encoder over multiple datasets.
    if args.data_roots:
        _run_pretrained(args)
        return

    device = str(args.device)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.checkpoint:
        raise ValueError("--checkpoint is required unless --data_roots is provided.")
    if not args.src_root or not args.tgt_root:
        raise ValueError("--src_root and --tgt_root are required unless --data_roots is provided.")

    model = _load_model(str(args.checkpoint), device=device)
    src_x, src_meta = _collect(
        model=model,
        data_root=str(args.src_root),
        split=str(args.split),
        dataset_tag="src",
        crop=int(args.crop),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        num_samples=int(args.num_samples),
        feature=str(args.feature),
        which=str(args.which),
        layer=int(args.layer),
        device=str(device),
        seed=int(args.seed),
        change_thr=float(args.change_thr),
    )
    tgt_x, tgt_meta = _collect(
        model=model,
        data_root=str(args.tgt_root),
        split=str(args.split),
        dataset_tag="tgt",
        crop=int(args.crop),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        num_samples=int(args.num_samples),
        feature=str(args.feature),
        which=str(args.which),
        layer=int(args.layer),
        device=str(device),
        seed=int(args.seed) + 1,
        change_thr=float(args.change_thr),
    )

    x = np.concatenate([src_x, tgt_x], axis=0)
    meta = src_meta + tgt_meta

    # Save embeddings for reuse.
    np.savez_compressed(
        str(out_dir / "embeddings.npz"),
        x=x.astype(np.float32),
        dataset=np.array([m.dataset for m in meta]),
        changed=np.array([m.changed for m in meta]).astype(np.int64),
    )

    # PCA
    pca_xy, mean, comps = _pca_2d(x)
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 5.6), dpi=160)
    _scatter_2d(
        ax,
        pca_xy,
        meta,
        title=f"PCA ({args.feature}, {args.which})",
        color_by=str(args.color_by),
    )
    fig.tight_layout()
    fig.savefig(str(out_dir / "pca.png"), bbox_inches="tight")
    plt.close(fig)

    report: Dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "src_root": str(args.src_root),
        "tgt_root": str(args.tgt_root),
        "split": str(args.split),
        "feature": str(args.feature),
        "which": str(args.which),
        "layer": int(args.layer),
        "num_samples_per_dataset": int(args.num_samples),
        "pca": {"mean": mean.tolist(), "components": comps.tolist()},
    }

    # t-SNE (optional)
    if args.tsne:
        x50 = _pca_reduce(x, k=int(args.pca_dim))
        tsne = _make_tsne(
            perplexity=float(args.tsne_perplexity),
            iters=int(args.tsne_iter),
            seed=int(args.seed),
        )
        tsne_xy = tsne.fit_transform(x50.astype(np.float32))
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 5.6), dpi=160)
        _scatter_2d(
            ax,
            tsne_xy,
            meta,
            title=f"t-SNE ({args.feature}, {args.which})",
            color_by=str(args.color_by),
        )
        fig.tight_layout()
        fig.savefig(str(out_dir / "tsne.png"), bbox_inches="tight")
        plt.close(fig)
        report["tsne"] = {
            "perplexity": float(args.tsne_perplexity),
            "iter": int(args.tsne_iter),
        }

    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] Saved PCA/t-SNE outputs to: {out_dir}")


if __name__ == "__main__":
    # Avoid slow/blocked network calls in restricted environments.
    os.environ.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    main()
