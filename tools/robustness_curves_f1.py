"""
Run and plot robustness curves (F1) under synthetic test-time corruptions.

This is an inference-only analysis: no retraining.

Default: cross-domain only (LEVIR->WHU, WHU->LEVIR), with 3 corruption types:
  - gaussian noise
  - brightness/contrast (BC), correlated between (A,B)
  - jpeg compression

It will:
  1) run eval_dino_head.py for each point (skip if eval_results.json exists)
  2) aggregate F1 into summary.jsonl
  3) generate two figures:
       - robustness_f1.png  (absolute F1 vs severity)
       - robustness_df1.png (delta F1 vs clean, per setting)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


try:
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    raise RuntimeError(f"matplotlib is required: {e}")


@dataclass(frozen=True)
class Scenario:
    name: str
    ckpt: str
    data_root: str


@dataclass(frozen=True)
class Point:
    scenario: str
    corruption: str  # none|gaussian|bc|jpeg
    severity_x: float
    severity_label: str
    params: Dict[str, object]
    out_dir: str
    f1: Optional[float] = None


def _run_eval(
    *,
    point: Point,
    ckpt: str,
    data_root: str,
    device: str,
    crop: int,
    batch_size: int,
    num_workers: int,
    thr: float,
    smooth_k: int,
    min_area: int,
    ensemble_indices: str,
) -> float:
    out_dir = Path(point.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "eval_results.json"
    if out_json.is_file():
        rep = json.loads(out_json.read_text(encoding="utf-8"))
        return float(rep["f1"])

    args = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "eval_dino_head.py"),
        "--checkpoint",
        ckpt,
        "--data_root",
        data_root,
        "--out_dir",
        str(out_dir),
        "--device",
        device,
        "--batch_size",
        str(int(batch_size)),
        "--num_workers",
        str(int(num_workers)),
        "--no_full_eval",
        "--eval_crop",
        str(int(crop)),
        "--thr_mode",
        "fixed",
        "--thr",
        str(float(thr)),
        "--smooth_k",
        str(int(smooth_k)),
        "--use_minarea",
        "--min_area",
        str(int(min_area)),
        "--use_ensemble_pred",
        "--ensemble_strategy",
        "mean_logit",
        "--ensemble_indices",
        str(ensemble_indices),
        "--tta",
        "none",
    ]

    corrupt = str(point.corruption)
    if corrupt != "none":
        args += ["--corrupt", corrupt, "--corrupt_pair", "correlated", "--corrupt_seed", "123"]
        if corrupt == "gaussian":
            args += ["--gaussian_sigma", str(float(point.params["gaussian_sigma"]))]
        elif corrupt == "bc":
            args += [
                "--bc_brightness",
                str(float(point.params["bc_brightness"])),
                "--bc_contrast",
                str(float(point.params["bc_contrast"])),
            ]
        elif corrupt == "jpeg":
            args += ["--jpeg_quality", str(int(point.params["jpeg_quality"]))]
        else:
            raise ValueError(f"Unknown corruption: {corrupt}")

    print(f"[Run] {point.scenario} {point.corruption} {point.severity_label} -> {out_dir}")
    env = os.environ.copy()
    # ensure no version/network checks
    env.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    subprocess.run(args, check=True, env=env)

    rep = json.loads(out_json.read_text(encoding="utf-8"))
    return float(rep["f1"])


def _default_points(base_out: Path) -> List[Tuple[str, str, float, str, Dict[str, object]]]:
    """
    Returns list of (corrupt, x, label, params).
    x is a numeric severity axis; label is for legends/logs.
    """
    pts: List[Tuple[str, str, float, str, Dict[str, object]]] = []
    pts.append(("none", "clean", 0.0, "clean", {}))

    # Gaussian sigma in [0,1] intensity space.
    for s in (0.01, 0.02, 0.03, 0.05):
        pts.append(("gaussian", f"sigma{s:g}", float(s), f"σ={s:g}", {"gaussian_sigma": float(s)}))

    # Brightness/contrast: y=(x-0.5)*c+0.5+b; b in [-bmax,+bmax], c in [1-cmax,1+cmax]
    # Use the common setting bmax=cmax in {0.1,0.2,0.3,0.4} for clean interpretation.
    for v in (0.10, 0.20, 0.30, 0.40):
        x = float(v)
        pts.append(
            (
                "bc",
                f"b{v:g}_c{v:g}",
                x,
                f"bmax=cmax={v:g}",
                {"bc_brightness": float(v), "bc_contrast": float(v)},
            )
        )

    # JPEG: severity increases as quality decreases. Use x=(95-q).
    # We treat Q=95 as the clean baseline (shown at severity 0 via the clean point).
    for q in (75, 55, 35):
        pts.append(("jpeg", f"q{q}", float(95 - q), f"Q={q}", {"jpeg_quality": int(q)}))
    return pts


def _plot(
    out_png: Path,
    out_png_df1: Path,
    points: List[Point],
    *,
    title: str,
):
    # group
    by_corrupt: Dict[str, List[Point]] = {}
    for p in points:
        by_corrupt.setdefault(p.corruption, []).append(p)

    # We'll plot only gaussian/bc/jpeg (clean is included as x=0 point for each curve).
    corrupts = ["gaussian", "bc", "jpeg"]
    scen_names = sorted({p.scenario for p in points})

    def _display_scenario(name: str) -> str:
        # Paper-friendly labels.
        name = str(name)
        name = name.replace("LEVIR", "LEVIR-CD").replace("WHU", "WHU-CD")
        return name

    # Build clean F1 per scenario from corruption==none.
    clean_f1: Dict[str, float] = {}
    for p in points:
        if p.corruption == "none":
            assert p.f1 is not None
            clean_f1[p.scenario] = float(p.f1)
    if any(s not in clean_f1 for s in scen_names):
        raise RuntimeError(f"Missing clean point for scenarios: {scen_names} vs {list(clean_f1.keys())}")

    colors = {}
    markers = {}
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    mks = ["o", "^", "s", "D"]
    for i, s in enumerate(scen_names):
        colors[s] = palette[i % len(palette)]
        markers[s] = mks[i % len(mks)]

    def _select(corrupt: str, scenario: str) -> List[Point]:
        pts = [p for p in points if p.corruption == corrupt and p.scenario == scenario]
        pts = sorted(pts, key=lambda x: float(x.severity_x))
        return pts

    panel_labels = ["(a)", "(b)", "(c)"]

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9), dpi=180)
    for ax, corrupt, plab in zip(axes, corrupts, panel_labels):
        for s in scen_names:
            pts = _select(corrupt, s)
            xs = [0.0] + [float(p.severity_x) for p in pts]
            ys = [float(clean_f1[s])] + [float(p.f1) for p in pts]  # type: ignore[arg-type]
            ax.plot(
                xs,
                ys,
                marker=markers[s],
                color=colors[s],
                linewidth=2.0,
                markersize=5,
                label=_display_scenario(s),
            )
        if corrupt == "jpeg":
            ax.set_xlabel("Severity (95 - JPEG quality)")
        elif corrupt == "gaussian":
            ax.set_xlabel("Severity (Gaussian σ)")
        else:
            ax.set_xlabel("Severity (BC brightness bmax)")
        ax.set_title(corrupt.upper())
        ax.grid(True, alpha=0.25)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("F1" if corrupt == "gaussian" else "")
        ax.text(0.5, -0.32, plab, transform=ax.transAxes, ha="center", va="top", fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(scen_names), frameon=True, fontsize=10)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0.0, 0.08, 1.0, 0.95])
    fig.savefig(str(out_png), bbox_inches="tight")
    fig.savefig(str(out_png.with_suffix(".pdf")), bbox_inches="tight")
    plt.close(fig)

    # Delta-F1 plot
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9), dpi=180)
    for ax, corrupt, plab in zip(axes, corrupts, panel_labels):
        for s in scen_names:
            pts = _select(corrupt, s)
            xs = [0.0] + [float(p.severity_x) for p in pts]
            ys = [0.0] + [float(p.f1) - float(clean_f1[s]) for p in pts]  # type: ignore[arg-type]
            ax.plot(
                xs,
                ys,
                marker=markers[s],
                color=colors[s],
                linewidth=2.0,
                markersize=5,
                label=_display_scenario(s),
            )
        if corrupt == "jpeg":
            ax.set_xlabel("Severity (95 - JPEG quality)")
        elif corrupt == "gaussian":
            ax.set_xlabel("Severity (Gaussian σ)")
        else:
            ax.set_xlabel("Severity (BC brightness bmax)")
        ax.set_title(corrupt.upper())
        ax.grid(True, alpha=0.25)
        ax.set_ylabel("ΔF1 vs clean" if corrupt == "gaussian" else "")
        ax.text(0.5, -0.32, plab, transform=ax.transAxes, ha="center", va="top", fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(scen_names), frameon=True, fontsize=10)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0.0, 0.08, 1.0, 0.95])
    fig.savefig(str(out_png_df1), bbox_inches="tight")
    fig.savefig(str(out_png_df1.with_suffix(".pdf")), bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser("Run robustness curves and plot F1")
    p.add_argument("--out_dir", type=str, default="outputs/robustness_curves_f1")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--crop", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--smooth_k", type=int, default=3)
    p.add_argument("--min_area", type=int, default=256)
    p.add_argument("--ensemble_indices", type=str, default="3,4")
    p.add_argument("--ckpt_levir2whu", type=str, default="outputs/ablation/best/Best_levir--whu/best.pt")
    p.add_argument("--ckpt_whu2levir", type=str, default="outputs/ablation/best/Best_whu--levir/best.pt")
    p.add_argument("--levir_root", type=str, default="data/LEVIR-CD")
    p.add_argument("--whu_root", type=str, default="data/WHUCD")
    p.add_argument("--include_in_domain", action="store_true", help="Also run LEVIR->LEVIR and WHU->WHU points (more runs).")
    return p.parse_args()


def main():
    args = parse_args()
    base_out = Path(args.out_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    scenarios = [
        Scenario(name="LEVIR→WHU", ckpt=str(args.ckpt_levir2whu), data_root=str(args.whu_root)),
        Scenario(name="WHU→LEVIR", ckpt=str(args.ckpt_whu2levir), data_root=str(args.levir_root)),
    ]
    if args.include_in_domain:
        scenarios = [
            *scenarios,
            Scenario(name="LEVIR→LEVIR", ckpt=str(args.ckpt_levir2whu), data_root=str(args.levir_root)),
            Scenario(name="WHU→WHU", ckpt=str(args.ckpt_whu2levir), data_root=str(args.whu_root)),
        ]

    grid = _default_points(base_out)
    all_points: List[Point] = []
    summary_path = base_out / "summary.jsonl"

    # Load existing summary to skip already recorded points.
    seen = set()
    if summary_path.is_file():
        for line in summary_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            seen.add((obj["scenario"], obj["corruption"], obj["severity_label"]))

    with summary_path.open("a", encoding="utf-8") as fsum:
        for sc in scenarios:
            for corrupt, run_name, x, label, params in grid:
                out_dir = base_out / sc.name.replace("→", "2").replace(" ", "") / corrupt / run_name
                pt = Point(
                    scenario=sc.name,
                    corruption=corrupt,
                    severity_x=float(x),
                    severity_label=str(label),
                    params=dict(params),
                    out_dir=str(out_dir),
                    f1=None,
                )

                key = (pt.scenario, pt.corruption, pt.severity_label)
                if key in seen:
                    # still load f1 for plotting
                    out_json = Path(pt.out_dir) / "eval_results.json"
                    if out_json.is_file():
                        rep = json.loads(out_json.read_text(encoding="utf-8"))
                        pt = Point(**{**asdict(pt), "f1": float(rep["f1"])})
                        all_points.append(pt)
                    continue

                f1 = _run_eval(
                    point=pt,
                    ckpt=sc.ckpt,
                    data_root=sc.data_root,
                    device=str(args.device),
                    crop=int(args.crop),
                    batch_size=int(args.batch_size),
                    num_workers=int(args.num_workers),
                    thr=float(args.thr),
                    smooth_k=int(args.smooth_k),
                    min_area=int(args.min_area),
                    ensemble_indices=str(args.ensemble_indices),
                )
                pt = Point(**{**asdict(pt), "f1": float(f1)})
                all_points.append(pt)
                fsum.write(json.dumps(asdict(pt), ensure_ascii=False) + "\n")
                fsum.flush()
                seen.add(key)

    # Reload all points from summary to ensure completeness.
    pts: List[Point] = []
    for line in summary_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        pts.append(Point(**obj))

    _plot(
        base_out / "robustness_f1.png",
        base_out / "robustness_df1.png",
        pts,
        title=f"Robustness under synthetic corruptions (crop={int(args.crop)}, ensemble={args.ensemble_indices})",
    )
    (base_out / "config.json").write_text(
        json.dumps(
            {
                "device": str(args.device),
                "crop": int(args.crop),
                "batch_size": int(args.batch_size),
                "num_workers": int(args.num_workers),
                "thr": float(args.thr),
                "smooth_k": int(args.smooth_k),
                "min_area": int(args.min_area),
                "ensemble_indices": str(args.ensemble_indices),
                "scenarios": [asdict(s) for s in scenarios],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[OK] Wrote summary to: {summary_path}")
    print(f"[OK] Saved figures: {base_out / 'robustness_f1.png'} and {base_out / 'robustness_df1.png'}")


if __name__ == "__main__":
    main()
