"""
Visualize raw RGB channel statistics (mean/std) for two CD datasets (e.g., LEVIR vs WHU).

No training required: reads images from the existing dataset folders and computes per-image
channel mean/std in [0,1], then plots distributions (hist + boxplot).

Example:
  python vis_dataset_channel_stats.py ^
    --levir_root data/LEVIR-CD --whu_root data/WHUCD --split train ^
    --which both --max_samples 2000 --out_dir outputs/dataset_stats
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    raise RuntimeError(f"matplotlib is required for vis_dataset_channel_stats.py: {e}")

from dataset import LEVIRCDDataset


@dataclass
class Stats:
    mean: np.ndarray  # [N,3]
    std: np.ndarray  # [N,3]


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _as_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _pick_imgs(sample: Dict, which: str) -> List[torch.Tensor]:
    which = str(which).lower()
    if which == "t1":
        return [sample["img_a"]]
    if which == "t2":
        return [sample["img_b"]]
    if which == "both":
        return [sample["img_a"], sample["img_b"]]
    raise ValueError("--which must be one of: t1 | t2 | both")


@torch.no_grad()
def _collect_stats(root: str, split: str, which: str, max_samples: int, seed: int) -> Stats:
    ds = LEVIRCDDataset(root_dir=root, split=split, transform=None, crop_size=256)
    idxs = list(range(len(ds)))
    rng = random.Random(int(seed))
    rng.shuffle(idxs)
    if max_samples and max_samples > 0:
        idxs = idxs[: int(max_samples)]

    means: List[np.ndarray] = []
    stds: List[np.ndarray] = []
    for idx in idxs:
        sample = ds[int(idx)]
        for img in _pick_imgs(sample, which=which):
            if not isinstance(img, torch.Tensor):
                img = torch.as_tensor(img)
            img = img.float()
            if img.ndim != 3 or img.shape[0] != 3:
                raise ValueError(f"Expected CHW RGB tensor, got {tuple(img.shape)} for idx={idx}")
            mu = img.mean(dim=(1, 2))
            sd = img.std(dim=(1, 2), unbiased=False)
            means.append(_as_np(mu))
            stds.append(_as_np(sd))

    if not means:
        raise RuntimeError(f"No images collected from {root} split={split}.")
    return Stats(mean=np.stack(means, axis=0), std=np.stack(stds, axis=0))


def _plot_hist_3ch(
    out_path: str,
    s1: Stats,
    s2: Stats,
    *,
    title: str,
    key: str,  # "mean" | "std"
    name1: str,
    name2: str,
    bins: int = 50,
):
    x1 = getattr(s1, key)
    x2 = getattr(s2, key)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.5), dpi=160, sharey=True)
    channels = ["R", "G", "B"]
    colors = ["#d62728", "#2ca02c", "#1f77b4"]
    for c in range(3):
        ax = axes[c]
        ax.hist(x1[:, c], bins=bins, density=True, alpha=0.55, color=colors[c], label=name1)
        ax.hist(x2[:, c], bins=bins, density=True, alpha=0.35, color="black", label=name2)
        ax.set_title(f"{title} - {channels[c]}")
        ax.set_xlabel(key)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
        if c == 0:
            ax.set_ylabel("density")
        if c == 2:
            ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_box_3ch(
    out_path: str,
    s1: Stats,
    s2: Stats,
    *,
    title: str,
    key: str,  # "mean" | "std"
    name1: str,
    name2: str,
):
    x1 = getattr(s1, key)
    x2 = getattr(s2, key)
    fig, ax = plt.subplots(1, 1, figsize=(9.5, 3.6), dpi=160)
    data = []
    labels = []
    for ch, ch_name in enumerate(["R", "G", "B"]):
        data.append(x1[:, ch])
        labels.append(f"{name1}-{ch_name}")
        data.append(x2[:, ch])
        labels.append(f"{name2}-{ch_name}")
    bp = ax.boxplot(data, labels=labels, showfliers=False)
    for element in ["boxes", "whiskers", "caps", "medians"]:
        for item in bp[element]:
            item.set(color="#333333", linewidth=1.2)
    ax.set_title(title)
    ax.set_ylabel(key)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser("LEVIR vs WHU raw RGB channel stats")
    p.add_argument("--levir_root", type=str, required=True)
    p.add_argument("--whu_root", type=str, required=True)
    p.add_argument("--split", type=str, choices=["train", "val", "test"], default="train")
    p.add_argument("--which", type=str, choices=["t1", "t2", "both"], default="both")
    p.add_argument("--max_samples", type=int, default=0, help="Max samples (per dataset); 0 = all.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bins", type=int, default=50)
    p.add_argument("--out_dir", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    _seed_everything(int(args.seed))

    os.makedirs(args.out_dir, exist_ok=True)
    out_dir = Path(args.out_dir)

    levir = _collect_stats(
        root=str(args.levir_root),
        split=str(args.split),
        which=str(args.which),
        max_samples=int(args.max_samples),
        seed=int(args.seed),
    )
    whu = _collect_stats(
        root=str(args.whu_root),
        split=str(args.split),
        which=str(args.which),
        max_samples=int(args.max_samples),
        seed=int(args.seed) + 1,
    )

    _plot_hist_3ch(
        str(out_dir / "rgb_mean_hist.png"),
        levir,
        whu,
        title="Channel mean distribution",
        key="mean",
        name1="LEVIR-CD",
        name2="WHU-CD",
        bins=int(args.bins),
    )
    _plot_hist_3ch(
        str(out_dir / "rgb_std_hist.png"),
        levir,
        whu,
        title="Channel std distribution",
        key="std",
        name1="LEVIR-CD",
        name2="WHU-CD",
        bins=int(args.bins),
    )
    _plot_box_3ch(
        str(out_dir / "rgb_mean_box.png"),
        levir,
        whu,
        title="Channel mean (boxplot)",
        key="mean",
        name1="LEVIR-CD",
        name2="WHU-CD",
    )
    _plot_box_3ch(
        str(out_dir / "rgb_std_box.png"),
        levir,
        whu,
        title="Channel std (boxplot)",
        key="std",
        name1="LEVIR-CD",
        name2="WHU-CD",
    )

    # Also save numerical summaries for tables.
    report = {
        "split": str(args.split),
        "which": str(args.which),
        "levir_root": str(args.levir_root),
        "whu_root": str(args.whu_root),
        "levir": {
            "mean_mean": levir.mean.mean(axis=0).tolist(),
            "mean_std": levir.mean.std(axis=0).tolist(),
            "std_mean": levir.std.mean(axis=0).tolist(),
            "std_std": levir.std.std(axis=0).tolist(),
            "n_images": int(levir.mean.shape[0]),
        },
        "whu": {
            "mean_mean": whu.mean.mean(axis=0).tolist(),
            "mean_std": whu.mean.std(axis=0).tolist(),
            "std_mean": whu.std.mean(axis=0).tolist(),
            "std_std": whu.std.std(axis=0).tolist(),
            "n_images": int(whu.mean.shape[0]),
        },
    }
    import json

    (out_dir / "rgb_stats_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] Saved figures + summary to: {out_dir}")


if __name__ == "__main__":
    main()
