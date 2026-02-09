"""
Frequency-domain energy analysis for layer-wise heads (and fused head).

Goal: support paper-style analysis like:
  - shallow heads preserve more high-frequency (edges)
  - deeper/fused heads concentrate on low-frequency (semantics)
  - cross-domain shift changes spectral energy distribution

This is a *no-retraining* tool: it runs inference on an existing checkpoint and
computes radial-averaged FFT power spectra on the head probability maps.

Typical usage (center crop 256 for speed):
  python vis_freq_energy_layers.py ^
    --checkpoint outputs/ablation/best/Best_levir--whu/best.pt ^
    --levir_root data/LEVIR-CD --whu_root data/WHUCD --split test ^
    --crop 256 --num_samples 200 --batch_size 4 --out_dir outputs/freq_energy
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

try:
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    raise RuntimeError(f"matplotlib is required for vis_freq_energy_layers.py: {e}")

from dataset import LEVIRCDDataset, get_val_transforms, worker_init_fn
from dino_head_core import HeadCfg, seed_everything
from models.dinov2_head import DinoSiameseHead, DinoFrozenA0Head


@dataclass
class SpectrumStats:
    freq: np.ndarray  # [nbins] in [0,1]
    energy: np.ndarray  # [nbins], sums to 1
    hf_ratio: float
    centroid: float
    n_maps: int


def _fftshift2(x: torch.Tensor) -> torch.Tensor:
    # torch.fft.fftshift exists in recent PyTorch; keep a robust fallback.
    if hasattr(torch.fft, "fftshift"):
        return torch.fft.fftshift(x, dim=(-2, -1))
    h = int(x.shape[-2])
    w = int(x.shape[-1])
    return torch.roll(x, shifts=(h // 2, w // 2), dims=(-2, -1))


def _make_radial_bins(h: int, w: int, nbins: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    yy = torch.arange(h, device=device, dtype=torch.float32) - (h // 2)
    xx = torch.arange(w, device=device, dtype=torch.float32) - (w // 2)
    y, x = torch.meshgrid(yy, xx, indexing="ij")
    r = torch.sqrt(x * x + y * y)
    r_norm = (r / (r.max().clamp_min(1e-6))).clamp(0.0, 1.0)  # [H,W] in [0,1]
    bin_idx = (r_norm * float(nbins - 1)).round().to(dtype=torch.long)  # [H,W] in [0..nbins-1]
    freq = torch.linspace(0.0, 1.0, steps=nbins, device=device, dtype=torch.float32)
    return bin_idx, freq


def _radial_energy_1map(x2d: torch.Tensor, bin_idx: torch.Tensor, nbins: int) -> torch.Tensor:
    """
    x2d: [H,W] float
    returns energy distribution [nbins] normalized to sum=1
    """
    x2d = x2d - x2d.mean()
    fft = torch.fft.fft2(x2d)
    power = _fftshift2(fft).abs().pow(2)  # [H,W]

    b = bin_idx.reshape(-1)
    p = power.reshape(-1)
    sums = torch.zeros((nbins,), device=x2d.device, dtype=torch.float64)
    sums.scatter_add_(0, b, p.to(dtype=torch.float64))
    total = sums.sum().clamp_min(1e-12)
    return (sums / total).to(dtype=torch.float32)


class _DiffHook:
    def __init__(self, n: int):
        self.n = int(n)
        self.buf: List[torch.Tensor] = []

    def clear(self):
        self.buf = []

    def hook(self, _module, _inp, out):
        # out: [B,C,h,w]
        self.buf.append(out)

    def get(self) -> List[torch.Tensor]:
        if len(self.buf) != self.n:
            raise RuntimeError(f"Expected {self.n} diff features, got {len(self.buf)} (hooks may not have fired).")
        return self.buf


@torch.no_grad()
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
def _collect_spectra(
    *,
    model: torch.nn.Module,
    data_root: str,
    split: str,
    crop: int,
    batch_size: int,
    num_workers: int,
    num_samples: int,
    device: str,
    fft_device: str,
    seed: int,
    nbins: int,
    hf_start: float,
    source: str,
) -> Dict[str, SpectrumStats]:
    tf = get_val_transforms(crop_size=int(crop))
    ds = LEVIRCDDataset(root_dir=data_root, split=split, transform=tf, crop_size=int(crop))
    g = torch.Generator()
    g.manual_seed(int(seed))
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=int(num_workers),
        pin_memory=True,
        persistent_workers=(int(num_workers) > 0),
        worker_init_fn=worker_init_fn if int(num_workers) > 0 else None,
        generator=g,
    )

    source = str(source).lower()
    if source not in ("prob", "logit", "diff"):
        raise ValueError("--source must be one of: prob | logit | diff")

    # For diff mode, attach hooks once (no code changes, no retraining).
    diff_hook = None
    diff_handles: List[torch.utils.hooks.RemovableHandle] = []
    diff_layers: List[int] = []
    if source == "diff":
        if not isinstance(model, DinoSiameseHead):
            raise RuntimeError("--source diff requires a DinoSiameseHead checkpoint (arch=dlv).")
        diff_layers = [int(x) for x in getattr(model, "selected_layers", [])]
        diff_hook = _DiffHook(n=len(model.diff_modules))
        for m in model.diff_modules:
            diff_handles.append(m.register_forward_hook(diff_hook.hook))

    # We compute spectra on either:
    #  - prob/logit: layer-wise head maps at output resolution
    #  - diff:       |D^l| magnitude maps from DifferenceModule outputs (patch-grid resolution)
    head_names: List[str] = []
    sum_energy: List[torch.Tensor] = []
    sum_hf: List[float] = []
    sum_centroid: List[float] = []
    n_maps = 0

    fft_dev = torch.device(str(fft_device))
    bin_idx = None
    freq = None
    b_flat = None
    hf_mask = None

    got = 0
    try:
        for batch in loader:
            if diff_hook is not None:
                diff_hook.clear()
            img_a = batch["img_a"].to(device, non_blocking=True)
            img_b = batch["img_b"].to(device, non_blocking=True)
            out = model(img_a, img_b)

            if source in ("prob", "logit"):
                if not isinstance(out, dict) or out.get("logits_all") is None:
                    raise RuntimeError(
                        "Model output has no logits_all; prob/logit spectrum requires a DinoSiameseHead with use_layer_ensemble=True."
                    )
                logits_all = out["logits_all"]  # [K,B,1,H,W]
                probs_all = torch.sigmoid(logits_all)
                K, B, _, H, W = probs_all.shape
                diff_feats_local = None
            else:
                assert source == "diff"
                assert diff_hook is not None
                diff_feats_local = diff_hook.get()  # list of [B,C,h,w]
                K = len(diff_feats_local)
                B = int(diff_feats_local[0].shape[0])
                H, W = int(diff_feats_local[0].shape[-2]), int(diff_feats_local[0].shape[-1])

            if bin_idx is None or freq is None:
                # FFT stats are computed on fft_device (CPU by default for Windows CUDA compatibility).
                bin_idx, freq = _make_radial_bins(int(H), int(W), nbins=int(nbins), device=fft_dev)
                b_flat = bin_idx.reshape(1, -1)  # [1,HW]
                hf_mask = (freq >= float(hf_start)).to(dtype=torch.float32)  # [nbins]
                if source == "diff":
                    if len(diff_layers) == K:
                        head_names = [f"layer{l}" for l in diff_layers]
                    else:
                        head_names = [f"diff{k}" for k in range(K)]
                else:
                    # label heads: selected layers + fused
                    if isinstance(model, DinoSiameseHead):
                        layers = [int(x) for x in getattr(model, "selected_layers", [])]
                    else:
                        layers = []
                    if len(layers) == K - 1:
                        head_names = [f"layer{l}" for l in layers] + ["fused"]
                    else:
                        head_names = [f"head{k}" for k in range(K - 1)] + ["fused"]

                sum_energy = [torch.zeros((nbins,), device="cpu", dtype=torch.float64) for _ in range(K)]
                sum_hf = [0.0 for _ in range(K)]
                sum_centroid = [0.0 for _ in range(K)]

            # limit sample count
            keep = int(min(int(num_samples) - got, B))
            if keep <= 0:
                break
            if source in ("prob", "logit"):
                probs_all = probs_all[:, :keep]  # type: ignore[name-defined]
                logits_all = logits_all[:, :keep]  # type: ignore[name-defined]
            else:
                assert diff_feats_local is not None
                diff_feats_local = [d[:keep] for d in diff_feats_local]

            assert b_flat is not None and hf_mask is not None and freq is not None
            b_expand = b_flat.expand(keep, -1)  # [keep,HW]

            for k in range(int(K)):
                if source == "prob":
                    maps = probs_all[k, :, 0].detach().to(device=fft_dev)  # type: ignore[name-defined]
                elif source == "logit":
                    maps = logits_all[k, :, 0].detach().to(device=fft_dev)  # type: ignore[name-defined]
                else:
                    assert source == "diff"
                    assert diff_feats_local is not None
                    d = diff_feats_local[k].detach().to(device=fft_dev)  # [keep,C,h,w]
                    maps = d.abs().mean(dim=1)  # [keep,h,w]
                maps = maps - maps.mean(dim=(1, 2), keepdim=True)
                fft = torch.fft.fft2(maps)  # [keep,H,W]
                power = _fftshift2(fft).abs().pow(2)  # [keep,H,W]
                p_flat = power.reshape(keep, -1).to(dtype=torch.float64)
                sums = torch.zeros((keep, int(nbins)), device=fft_dev, dtype=torch.float64)
                sums.scatter_add_(1, b_expand, p_flat)
                total = sums.sum(dim=1, keepdim=True).clamp_min(1e-12)
                e = (sums / total).to(dtype=torch.float32)  # [keep,nbins]

                sum_energy[k] += e.sum(dim=0).detach().to(device="cpu", dtype=torch.float64)
                sum_hf[k] += float((e * hf_mask.view(1, -1)).sum(dim=1).sum().item())
                sum_centroid[k] += float((e * freq.view(1, -1)).sum(dim=1).sum().item())

            got += keep
            n_maps += keep
            if got >= int(num_samples):
                break
    finally:
        for h in diff_handles:
            try:
                h.remove()
            except Exception:
                pass

    assert bin_idx is not None and freq is not None
    assert b_flat is not None and hf_mask is not None
    assert n_maps > 0
    out_stats: Dict[str, SpectrumStats] = {}
    freq_np = freq.detach().cpu().numpy().astype(np.float32)
    for k, name in enumerate(head_names):
        e_mean = (sum_energy[k] / float(n_maps)).numpy().astype(np.float32)
        e_mean = e_mean / (e_mean.sum() + 1e-12)
        out_stats[name] = SpectrumStats(
            freq=freq_np,
            energy=e_mean,
            hf_ratio=float(sum_hf[k] / float(n_maps)),
            centroid=float(sum_centroid[k] / float(n_maps)),
            n_maps=int(n_maps),
        )
    return out_stats


def _plot_curves(out_path: Path, title: str, stats: Dict[str, SpectrumStats]):
    # Plot energy curves for each head.
    plt.figure(figsize=(6.6, 4.8), dpi=160)
    for name, st in stats.items():
        plt.plot(st.freq, st.energy, linewidth=2.0, label=name)
    plt.xlabel("Normalized spatial frequency (radial)")
    plt.ylabel("Energy fraction")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(str(out_path), bbox_inches="tight")
    plt.close()


def _plot_metrics(out_path: Path, title: str, stats: Dict[str, SpectrumStats]):
    names = list(stats.keys())
    hf = [stats[n].hf_ratio for n in names]
    cent = [stats[n].centroid for n in names]
    x = np.arange(len(names))

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), dpi=160)
    axes[0].bar(x, hf, color="#1f77b4")
    axes[0].set_title("High-frequency ratio")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=30, ha="right")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(x, cent, color="#ff7f0e")
    axes[1].set_title("Spectral centroid")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=30, ha="right")
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser("Frequency energy distribution for layer-wise heads")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--levir_root", type=str, required=True)
    p.add_argument("--whu_root", type=str, required=True)
    p.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--fft_device", type=str, choices=["cpu", "cuda", "auto"], default="cpu")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--crop", type=int, default=256, help="Center crop size before inference")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--num_samples", type=int, default=200, help="Per-dataset sample count")

    p.add_argument("--nbins", type=int, default=64, help="Radial frequency bins")
    p.add_argument("--hf_start", type=float, default=0.50, help="High-frequency start in normalized freq [0,1]")
    p.add_argument(
        "--source",
        type=str,
        choices=["prob", "logit", "diff"],
        default="diff",
        help="Spectrum source: prob/logit from logits_all, or diff feature magnitude from DifferenceModule outputs",
    )
    return p.parse_args()


def _stats_to_jsonable(stats: Dict[str, SpectrumStats]) -> Dict[str, Dict]:
    d: Dict[str, Dict] = {}
    for k, st in stats.items():
        d[str(k)] = {
            "freq": st.freq.tolist(),
            "energy": st.energy.tolist(),
            "hf_ratio": float(st.hf_ratio),
            "centroid": float(st.centroid),
            "n_maps": int(st.n_maps),
        }
    return d


def main():
    args = parse_args()
    seed_everything(int(args.seed))

    device = str(args.device)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    fft_device = str(args.fft_device)
    if fft_device == "auto":
        # Prefer CPU FFT: avoids CUDA nvrtc toolchain issues on some Windows setups.
        fft_device = "cpu"
    if fft_device == "cuda" and not torch.cuda.is_available():
        fft_device = "cpu"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = _load_model(str(args.checkpoint), device=device)
    levir_stats = _collect_spectra(
        model=model,
        data_root=str(args.levir_root),
        split=str(args.split),
        crop=int(args.crop),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        num_samples=int(args.num_samples),
        device=str(device),
        fft_device=str(fft_device),
        seed=int(args.seed),
        nbins=int(args.nbins),
        hf_start=float(args.hf_start),
        source=str(args.source),
    )
    whu_stats = _collect_spectra(
        model=model,
        data_root=str(args.whu_root),
        split=str(args.split),
        crop=int(args.crop),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        num_samples=int(args.num_samples),
        device=str(device),
        fft_device=str(fft_device),
        seed=int(args.seed) + 1,
        nbins=int(args.nbins),
        hf_start=float(args.hf_start),
        source=str(args.source),
    )

    _plot_curves(out_dir / "freq_energy_levir.png", "LEVIR (layer-wise heads)", levir_stats)
    _plot_curves(out_dir / "freq_energy_whu.png", "WHU (layer-wise heads)", whu_stats)
    _plot_metrics(out_dir / "freq_metrics_levir.png", "LEVIR (metrics)", levir_stats)
    _plot_metrics(out_dir / "freq_metrics_whu.png", "WHU (metrics)", whu_stats)

    report = {
        "checkpoint": str(args.checkpoint),
        "split": str(args.split),
        "crop": int(args.crop),
        "num_samples_per_dataset": int(args.num_samples),
        "nbins": int(args.nbins),
        "hf_start": float(args.hf_start),
        "fft_device": str(fft_device),
        "source": str(args.source),
        "levir": _stats_to_jsonable(levir_stats),
        "whu": _stats_to_jsonable(whu_stats),
    }
    (out_dir / "freq_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] Saved frequency analysis to: {out_dir}")


if __name__ == "__main__":
    main()
