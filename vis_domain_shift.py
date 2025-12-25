"""
Visualize feature domain shift (source vs target) for different fine-tuning modes.

Example (4 checkpoints -> 2x2 PCA / t-SNE grids):
  python vis_domain_shift.py `
    --src_root data/WHUCD --tgt_root data/LEVIR-CD --split test `
    --ckpt_frozen outputs/ablation/best/frozen/best.pt `
    --ckpt_shallow outputs/ablation/best/ft_shallow/best.pt `
    --ckpt_deep outputs/ablation/best/ft_deep/best.pt `
    --ckpt_full outputs/ablation/best/ft_full/best.pt `
    --layer 12 --which t2 --num_samples 200 --out_dir outputs/ablation/domain_shift
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    raise RuntimeError(f"matplotlib is required for vis_domain_shift.py: {e}")

try:
    from sklearn.manifold import TSNE  # type: ignore
except Exception:
    TSNE = None

from dataset import LEVIRCDDataset, get_val_transforms, worker_init_fn
from dino_head_core import HeadCfg
from models.dinov2_head import DinoSiameseHead, DinoFrozenA0Head


@dataclass
class Variant:
    name: str
    ckpt: str


def _pca_2d(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      proj: [N,2]
      mean: [D]
      components: [2,D]
    """
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=0, keepdims=True)
    xc = x - mean
    # SVD on centered data
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    comps = vt[:2]  # [2,D]
    proj = xc @ comps.T  # [N,2]
    return proj.astype(np.float32), mean.squeeze(0).astype(np.float32), comps.astype(np.float32)


def _load_model(ckpt_path: str, device: str) -> torch.nn.Module:
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


@torch.no_grad()
def _embed_batch_dinov3(
    model: DinoSiameseHead,
    imgs: torch.Tensor,
    layer: int,
) -> torch.Tensor:
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
def _embed_batch(model: torch.nn.Module, batch: Dict, which: str, layer: int, device: str) -> torch.Tensor:
    img_a = batch["img_a"].to(device, non_blocking=True)
    img_b = batch["img_b"].to(device, non_blocking=True)
    which = str(which).lower()
    if which == "t1":
        imgs = img_a
    elif which == "t2":
        imgs = img_b
    elif which == "both_avg":
        imgs = torch.cat([img_a, img_b], dim=0)
    else:
        raise ValueError("--which must be one of: t1 | t2 | both_avg")

    if isinstance(model, DinoSiameseHead) and getattr(model, "use_hf", False):
        emb = _embed_batch_dinov3(model, imgs, layer=layer)
    else:
        raise RuntimeError("vis_domain_shift.py currently supports DINOv3 (HF) checkpoints only.")

    if which == "both_avg":
        B = img_a.shape[0]
        emb = 0.5 * (emb[:B] + emb[B:])
    return emb


def _collect_embeddings(
    model: torch.nn.Module,
    data_root: str,
    split: str,
    crop: int,
    batch_size: int,
    num_workers: int,
    num_samples: int,
    which: str,
    layer: int,
    device: str,
    seed: int,
) -> np.ndarray:
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
    got = 0
    for batch in loader:
        emb = _embed_batch(model, batch, which=which, layer=layer, device=device)
        emb_np = emb.detach().float().cpu().numpy()
        if got + emb_np.shape[0] > num_samples:
            emb_np = emb_np[: max(0, num_samples - got)]
        feats.append(emb_np)
        got += emb_np.shape[0]
        if got >= num_samples:
            break
    if got == 0:
        raise RuntimeError(f"No samples collected from {data_root} split={split}.")
    return np.concatenate(feats, axis=0)


def _plot_2d(
    ax,
    src_xy: np.ndarray,
    tgt_xy: np.ndarray,
    title: str,
    show_legend: bool,
):
    ax.scatter(src_xy[:, 0], src_xy[:, 1], s=10, alpha=0.75, c="#1f77b4", label="source")
    ax.scatter(tgt_xy[:, 0], tgt_xy[:, 1], s=10, alpha=0.75, c="#ff7f0e", label="target")
    src_c = src_xy.mean(axis=0)
    tgt_c = tgt_xy.mean(axis=0)
    ax.scatter([src_c[0]], [src_c[1]], s=90, c="#1f77b4", marker="x")
    ax.scatter([tgt_c[0]], [tgt_c[1]], s=90, c="#ff7f0e", marker="x")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    if show_legend:
        ax.legend(loc="upper right", frameon=True, fontsize=9)


def parse_args():
    p = argparse.ArgumentParser("PCA/t-SNE domain shift visualization")
    p.add_argument("--src_root", type=str, required=True)
    p.add_argument("--tgt_root", type=str, required=True)
    p.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--crop", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--which", type=str, choices=["t1", "t2", "both_avg"], default="t2")
    p.add_argument("--layer", type=int, default=12, help="DINO layer index (same convention as your selected_layers, e.g. 3/6/9/12)")

    p.add_argument("--ckpt_frozen", type=str, required=True)
    p.add_argument("--ckpt_shallow", type=str, required=True)
    p.add_argument("--ckpt_deep", type=str, required=True)
    p.add_argument("--ckpt_full", type=str, required=True)

    p.add_argument("--tsne", action="store_true", help="Also run t-SNE (requires scikit-learn)")
    p.add_argument("--tsne_perplexity", type=float, default=30.0)
    p.add_argument("--tsne_iter", type=int, default=1000)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    variants = [
        Variant("Frozen", args.ckpt_frozen),
        Variant("Shallow FT", args.ckpt_shallow),
        Variant("Deep FT", args.ckpt_deep),
        Variant("Full FT", args.ckpt_full),
    ]

    report: Dict[str, Dict] = {}
    pca_fig, pca_axes = plt.subplots(2, 2, figsize=(9.5, 8.5), dpi=150)
    pca_axes = pca_axes.reshape(-1)

    tsne_fig = None
    tsne_axes = None
    if args.tsne:
        if TSNE is None:
            raise RuntimeError("Requested --tsne but scikit-learn is not available.")
        tsne_fig, tsne_axes = plt.subplots(2, 2, figsize=(9.5, 8.5), dpi=150)
        tsne_axes = tsne_axes.reshape(-1)

    for i, v in enumerate(variants):
        model = _load_model(v.ckpt, device=device)
        src = _collect_embeddings(
            model=model,
            data_root=args.src_root,
            split=args.split,
            crop=args.crop,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_samples=args.num_samples,
            which=args.which,
            layer=args.layer,
            device=device,
            seed=args.seed + 17,
        )
        tgt = _collect_embeddings(
            model=model,
            data_root=args.tgt_root,
            split=args.split,
            crop=args.crop,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_samples=args.num_samples,
            which=args.which,
            layer=args.layer,
            device=device,
            seed=args.seed + 23,
        )

        x = np.concatenate([src, tgt], axis=0)
        proj, mean, comps = _pca_2d(x)
        src_xy = proj[: src.shape[0]]
        tgt_xy = proj[src.shape[0] :]
        centroid_dist = float(np.linalg.norm(src.mean(axis=0) - tgt.mean(axis=0)))
        report[v.name] = {
            "ckpt": v.ckpt,
            "num_src": int(src.shape[0]),
            "num_tgt": int(tgt.shape[0]),
            "layer": int(args.layer),
            "which": args.which,
            "centroid_l2": centroid_dist,
            "pca_components": comps.tolist(),
            "pca_mean": mean.tolist(),
        }
        _plot_2d(
            pca_axes[i],
            src_xy=src_xy,
            tgt_xy=tgt_xy,
            title=f"{v.name} (PCA)  Δμ={centroid_dist:.3f}",
            show_legend=(i == 0),
        )

        if tsne_fig is not None and tsne_axes is not None:
            tsne = TSNE(
                n_components=2,
                perplexity=float(args.tsne_perplexity),
                n_iter=int(args.tsne_iter),
                init="pca",
                learning_rate="auto",
                random_state=int(args.seed),
            )
            xy = tsne.fit_transform(x.astype(np.float32))
            src_xy = xy[: src.shape[0]]
            tgt_xy = xy[src.shape[0] :]
            _plot_2d(
                tsne_axes[i],
                src_xy=src_xy,
                tgt_xy=tgt_xy,
                title=f"{v.name} (t-SNE)",
                show_legend=(i == 0),
            )

        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    pca_fig.tight_layout()
    pca_path = os.path.join(args.out_dir, "pca_grid.png")
    pca_fig.savefig(pca_path, bbox_inches="tight")
    plt.close(pca_fig)

    if tsne_fig is not None and tsne_axes is not None:
        tsne_fig.tight_layout()
        tsne_path = os.path.join(args.out_dir, "tsne_grid.png")
        tsne_fig.savefig(tsne_path, bbox_inches="tight")
        plt.close(tsne_fig)

    with open(os.path.join(args.out_dir, "domain_shift_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "src_root": args.src_root,
                "tgt_root": args.tgt_root,
                "split": args.split,
                "crop": int(args.crop),
                "num_samples": int(args.num_samples),
                "layer": int(args.layer),
                "which": args.which,
                "variants": report,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[OK] Saved: {pca_path}")
    if args.tsne:
        print(f"[OK] Saved: {os.path.join(args.out_dir, 'tsne_grid.png')}")
    print(f"[OK] Saved: {os.path.join(args.out_dir, 'domain_shift_metrics.json')}")


if __name__ == "__main__":
    main()

