import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import LEVIRCDDataset, get_test_transforms_full
from dino_head_core import (
    HeadCfg,
    seed_everything,
    evaluate,
    filter_small_cc,
    sliding_window_inference,
    sliding_window_inference_probs_all,
)

try:
    from models.dinov2_head import DinoSiameseHead, DinoFrozenA0Head
except Exception as e:
    raise RuntimeError(f"Failed to import model heads from models/dinov2_head.py: {e}")


_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _device_from_arg(device: str) -> str:
    device = str(device).lower()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _to_rgb(img_chw: torch.Tensor) -> np.ndarray:
    img = img_chw.detach().cpu().float().permute(1, 2, 0).numpy()
    img = (img * _STD + _MEAN).clip(0.0, 1.0)
    return img


def _to_mask_2d(mask: torch.Tensor) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    return (mask.detach().cpu().numpy() > 0).astype(np.uint8)


def _confusion_overlay_rgb(
    base_rgb: Optional[np.ndarray],
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    alpha: float = 0.55,
    tp_color: Tuple[float, float, float] = (0.0, 1.0, 0.0),
    fp_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    fn_color: Tuple[float, float, float] = (0.15, 0.55, 1.0),
) -> np.ndarray:
    """
    Render a paper-friendly overlay that encodes TP/FP/FN with colors on top of a base image.

    - TP: pred=1 & gt=1  (default green)
    - FP: pred=1 & gt=0  (default red)
    - FN: pred=0 & gt=1  (default blue)

    base_rgb: float RGB image in [0,1], shape [H,W,3]. If None, uses black background.
    """
    if pred.ndim != 2 or gt.ndim != 2:
        raise ValueError("pred and gt must be 2D masks.")
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch: {pred.shape} vs {gt.shape}")

    h, w = pred.shape
    if base_rgb is None:
        base = np.zeros((h, w, 3), dtype=np.float32)
    else:
        base = base_rgb.astype(np.float32, copy=False)
        if base.shape[:2] != (h, w) or base.shape[2] != 3:
            raise ValueError(f"base_rgb must be [H,W,3], got {base.shape}")
        base = base.clip(0.0, 1.0)

    pred_u = (pred > 0).astype(np.uint8)
    gt_u = (gt > 0).astype(np.uint8)

    tp = (pred_u == 1) & (gt_u == 1)
    fp = (pred_u == 1) & (gt_u == 0)
    fn = (pred_u == 0) & (gt_u == 1)

    overlay = np.zeros((h, w, 3), dtype=np.float32)
    overlay[tp] = np.array(tp_color, dtype=np.float32)
    overlay[fp] = np.array(fp_color, dtype=np.float32)
    overlay[fn] = np.array(fn_color, dtype=np.float32)

    mask_any = (tp | fp | fn).astype(np.float32)[..., None]
    a = float(alpha) * mask_any
    out = base * (1.0 - a) + overlay * a
    return out.clip(0.0, 1.0)


def _confusion_counts(pred: np.ndarray, gt: np.ndarray) -> Dict[str, int]:
    pred_u = (pred > 0).astype(np.uint8)
    gt_u = (gt > 0).astype(np.uint8)
    tp = int(((pred_u == 1) & (gt_u == 1)).sum())
    fp = int(((pred_u == 1) & (gt_u == 0)).sum())
    fn = int(((pred_u == 0) & (gt_u == 1)).sum())
    tn = int(((pred_u == 0) & (gt_u == 0)).sum())
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


def _infer_prob_and_uncertainty(
    model: torch.nn.Module,
    img_a: torch.Tensor,
    img_b: torch.Tensor,
    device: str,
    window: Optional[int],
    stride: Optional[int],
    thr: float,
    smooth_k: int,
    use_minarea: bool,
    min_area: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    img_a = img_a.to(device)
    img_b = img_b.to(device)

    # Uncertainty: prefer layer-ensemble disagreement variance; fall back to Bernoulli variance p(1-p).
    prob = None
    unc = None
    probs_all = None

    if window is not None and stride is not None:
        try:
            probs_all = sliding_window_inference_probs_all(
                model=model, img_a=img_a, img_b=img_b, window=window, stride=stride, device=device
            )  # [K,1,1,H,W]
        except Exception:
            probs_all = None

        if probs_all is not None:
            K = int(probs_all.shape[0])
            prob = probs_all[-1, 0, 0]  # fused head prob
            if K >= 2:
                unc = probs_all[: K - 1, 0, 0].var(dim=0, unbiased=False)
            else:
                unc = prob * (1.0 - prob)
        else:
            prob = sliding_window_inference(
                model=model,
                img_a=img_a,
                img_b=img_b,
                window=window,
                stride=stride,
                device=device,
                use_ensemble=False,
                ensemble_cfg=None,
            )[0, 0]
            unc = prob * (1.0 - prob)
    else:
        out = model(img_a, img_b)
        if isinstance(out, dict):
            prob = torch.sigmoid(out["pred"])[0, 0]
            if out.get("logits_all") is not None:
                probs_all = torch.sigmoid(out["logits_all"])  # [K,1,1,H,W]
        else:
            prob = torch.sigmoid(out)[0, 0]

        if probs_all is not None:
            K = int(probs_all.shape[0])
            if K >= 2:
                unc = probs_all[: K - 1, 0, 0].var(dim=0, unbiased=False)
            else:
                unc = prob * (1.0 - prob)
        else:
            unc = prob * (1.0 - prob)

    smooth_k = int(smooth_k)
    if smooth_k and smooth_k > 1:
        pad = smooth_k // 2
        prob = F.avg_pool2d(prob.view(1, 1, *prob.shape), kernel_size=smooth_k, stride=1, padding=pad)[0, 0]
        unc = F.avg_pool2d(unc.view(1, 1, *unc.shape), kernel_size=smooth_k, stride=1, padding=pad)[0, 0]

    pred = (prob > float(thr)).to(dtype=torch.uint8).detach().cpu().numpy().astype(np.uint8)
    if use_minarea:
        pred = filter_small_cc(pred, min_area=int(min_area))

    return (
        prob.detach().cpu().numpy().astype(np.float32),
        pred,
        unc.detach().cpu().numpy().astype(np.float32),
    )


def _load_model_from_ckpt(ckpt_path: Path, device: str) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    ckpt = torch.load(str(ckpt_path), map_location=device)
    load_cfg = ckpt.get("cfg") if isinstance(ckpt, dict) else None
    load_cfg = load_cfg if isinstance(load_cfg, dict) else {}

    arch = load_cfg.get("arch", "dlv")
    if arch == "a0":
        model = DinoFrozenA0Head(
            dino_name=load_cfg.get("dino_name", HeadCfg().dino_name),
            layer=int(load_cfg.get("a0_layer", getattr(HeadCfg(), "a0_layer", 12))),
            use_whiten=bool(load_cfg.get("use_whiten", False)),
        ).to(device)
    else:
        model = DinoSiameseHead(
            dino_name=load_cfg.get("dino_name", HeadCfg().dino_name),
            selected_layers=tuple(int(x) for x in load_cfg.get("selected_layers", HeadCfg().selected_layers)),
            use_whiten=bool(load_cfg.get("use_whiten", False)),
            use_domain_adv=bool(load_cfg.get("use_domain_adv", False)),
            domain_hidden=int(load_cfg.get("domain_hidden", HeadCfg().domain_hidden)),
            domain_grl=float(load_cfg.get("domain_grl", HeadCfg().domain_grl)),
            use_style_norm=bool(load_cfg.get("use_style_norm", False)),
            proto_path=load_cfg.get("proto_path", HeadCfg().proto_path),
            proto_weight=float(load_cfg.get("proto_weight", HeadCfg().proto_weight)),
            boundary_dim=int(load_cfg.get("boundary_dim", HeadCfg().boundary_dim)),
            use_layer_ensemble=bool(load_cfg.get("use_layer_ensemble", False)),
            layer_head_ch=int(load_cfg.get("layer_head_ch", HeadCfg().layer_head_ch)),
        ).to(device)

    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model, load_cfg


def _build_test_loader(data_root: str, split: str, batch_size: int, num_workers: int) -> Tuple[LEVIRCDDataset, DataLoader]:
    ds = LEVIRCDDataset(
        root_dir=Path(data_root),
        split=split,
        transform=get_test_transforms_full(),
        crop_size=256,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=int(num_workers),
        pin_memory=True,
    )
    return ds, loader


def _pick_indices(n_total: int, n_pick: int) -> List[int]:
    n_pick = int(n_pick)
    if n_total <= n_pick:
        return list(range(n_total))
    xs = np.linspace(0, n_total - 1, n_pick)
    return [int(round(x)) for x in xs]


def _compare_one_dataset(
    dataset_name: str,
    data_root: str,
    split: str,
    ckpt1: Path,
    ckpt2: Path,
    ckpt1_label: str,
    ckpt2_label: str,
    out_dir: Path,
    device: str,
    window: Optional[int],
    stride: Optional[int],
    thr_mode: str,
    thr: float,
    topk: float,
    smooth_k: int,
    use_minarea: bool,
    min_area: int,
    n_vis: int,
    seed: int,
    skip_eval: bool,
    pred_viz: str,
    overlay_base: str,
    overlay_alpha: float,
    show_stats: bool,
    show_legend: bool,
    unc_vmax: Optional[float],
    unc_colorbar: bool,
):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    out_dir.mkdir(parents=True, exist_ok=True)
    ds, loader = _build_test_loader(data_root=data_root, split=split, batch_size=1, num_workers=0)

    model1, cfg1 = _load_model_from_ckpt(ckpt1, device=device)
    model2, cfg2 = _load_model_from_ckpt(ckpt2, device=device)

    metrics1: Optional[Dict[str, Any]] = None
    metrics2: Optional[Dict[str, Any]] = None
    if not skip_eval:
        # Evaluate (fused pred by default).
        eval_cfg = HeadCfg(
            data_root=data_root,
            out_dir=str(out_dir),
            device=device,
            full_eval=True,
            thr_mode=thr_mode,
            thr=float(thr),
            topk=float(topk),
            smooth_k=int(smooth_k),
            use_minarea=bool(use_minarea),
            min_area=int(min_area),
            use_ensemble_pred=False,
        )
        metrics1 = evaluate(
            model=model1,
            loader=loader,
            device=device,
            thr_mode=eval_cfg.thr_mode,
            thr=eval_cfg.thr,
            topk=eval_cfg.topk,
            smooth_k=eval_cfg.smooth_k,
            use_minarea=eval_cfg.use_minarea,
            min_area=eval_cfg.min_area,
            print_every=0,
            window=window,
            stride=stride,
            use_ensemble=False,
            ensemble_cfg=None,
        )
        metrics2 = evaluate(
            model=model2,
            loader=loader,
            device=device,
            thr_mode=eval_cfg.thr_mode,
            thr=eval_cfg.thr,
            topk=eval_cfg.topk,
            smooth_k=eval_cfg.smooth_k,
            use_minarea=eval_cfg.use_minarea,
            min_area=eval_cfg.min_area,
            print_every=0,
            window=window,
            stride=stride,
            use_ensemble=False,
            ensemble_cfg=None,
        )

        with open(out_dir / f"metrics_{dataset_name}_ckpt1.json", "w", encoding="utf-8") as f:
            json.dump({"dataset": dataset_name, "checkpoint": str(ckpt1), "metrics": metrics1, "cfg": cfg1}, f, ensure_ascii=False, indent=2)
        with open(out_dir / f"metrics_{dataset_name}_ckpt2.json", "w", encoding="utf-8") as f:
            json.dump({"dataset": dataset_name, "checkpoint": str(ckpt2), "metrics": metrics2, "cfg": cfg2}, f, ensure_ascii=False, indent=2)

    seed_everything(seed)
    indices = _pick_indices(len(ds), n_vis)

    # Precompute predictions and uncertainties to enable shared uncertainty scaling across all panels.
    cases: List[Dict[str, Any]] = []
    for idx in indices:
        sample = ds[idx]
        img_a = sample["img_a"].unsqueeze(0)
        img_b = sample["img_b"].unsqueeze(0)
        gt = sample["label"]
        name = sample.get("name", str(idx))

        rgb_a = _to_rgb(sample["img_a"])
        rgb_b = _to_rgb(sample["img_b"])
        gt_np = _to_mask_2d(gt)

        prob1, pred1, unc1 = _infer_prob_and_uncertainty(
            model=model1,
            img_a=img_a,
            img_b=img_b,
            device=device,
            window=window,
            stride=stride,
            thr=float(thr),
            smooth_k=int(smooth_k),
            use_minarea=bool(use_minarea),
            min_area=int(min_area),
        )
        prob2, pred2, unc2 = _infer_prob_and_uncertainty(
            model=model2,
            img_a=img_a,
            img_b=img_b,
            device=device,
            window=window,
            stride=stride,
            thr=float(thr),
            smooth_k=int(smooth_k),
            use_minarea=bool(use_minarea),
            min_area=int(min_area),
        )

        if pred_viz == "overlay":
            base1 = rgb_b if overlay_base == "t2" else (rgb_a if overlay_base == "t1" else None)
            base2 = base1
            pred1_vis = _confusion_overlay_rgb(base1, pred1, gt_np, alpha=float(overlay_alpha))
            pred2_vis = _confusion_overlay_rgb(base2, pred2, gt_np, alpha=float(overlay_alpha))
        elif pred_viz == "mask":
            pred1_vis = pred1
            pred2_vis = pred2
        else:
            raise ValueError(f"Unknown pred_viz={pred_viz!r}")

        cases.append(
            {
                "name": name,
                "rgb_a": rgb_a,
                "rgb_b": rgb_b,
                "gt": gt_np,
                "pred1": pred1,
                "pred2": pred2,
                "pred1_vis": pred1_vis,
                "pred2_vis": pred2_vis,
                "unc1": unc1,
                "unc2": unc2,
                "cm1": _confusion_counts(pred1, gt_np),
                "cm2": _confusion_counts(pred2, gt_np),
            }
        )

    if unc_vmax is None:
        uncert_stack = np.stack([x["unc1"] for x in cases] + [x["unc2"] for x in cases], axis=0)
        vmax_unc = float(np.quantile(uncert_stack, 0.995))
        if not np.isfinite(vmax_unc) or vmax_unc <= 0:
            vmax_unc = float(np.max(uncert_stack) if np.max(uncert_stack) > 0 else 1.0)
    else:
        vmax_unc = float(unc_vmax)

    rows = len(indices)
    cols = 7
    fig_w = 3.2 * cols
    fig_h = 3.2 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    titles = [
        "T1",
        "T2",
        "GT",
        f"Pred ({ckpt1_label})",
        f"Pred ({ckpt2_label})",
        f"Unc ({ckpt1_label})",
        f"Unc ({ckpt2_label})",
    ]
    for c in range(cols):
        axes[0, c].set_title(titles[c], fontsize=12)

    last_unc_im = None
    for r, case in enumerate(cases):
        axes[r, 0].imshow(case["rgb_a"])
        axes[r, 1].imshow(case["rgb_b"])
        axes[r, 2].imshow(case["gt"], cmap="gray", vmin=0, vmax=1)

        if pred_viz == "overlay":
            axes[r, 3].imshow(case["pred1_vis"])
            axes[r, 4].imshow(case["pred2_vis"])
        else:
            axes[r, 3].imshow(case["pred1_vis"], cmap="gray", vmin=0, vmax=1)
            axes[r, 4].imshow(case["pred2_vis"], cmap="gray", vmin=0, vmax=1)

        im5 = axes[r, 5].imshow(case["unc1"], cmap="magma", vmin=0.0, vmax=vmax_unc)
        im6 = axes[r, 6].imshow(case["unc2"], cmap="magma", vmin=0.0, vmax=vmax_unc)
        last_unc_im = im6

        for c in range(cols):
            axes[r, c].axis("off")

        axes[r, 0].text(
            0.01,
            0.99,
            f"{case['name']}",
            transform=axes[r, 0].transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox=dict(facecolor="black", alpha=0.35, pad=2, edgecolor="none"),
            color="white",
        )
        row_tag = chr(ord("a") + r)
        axes[r, 0].text(
            -0.08,
            0.5,
            f"({row_tag})",
            transform=axes[r, 0].transAxes,
            va="center",
            ha="right",
            fontsize=12,
            color="black",
            clip_on=False,
        )

        if show_stats:
            cm1 = case["cm1"]
            cm2 = case["cm2"]
            axes[r, 3].text(
                0.01,
                0.99,
                f"FP={cm1['FP']}  FN={cm1['FN']}",
                transform=axes[r, 3].transAxes,
                va="top",
                ha="left",
                fontsize=10,
                bbox=dict(facecolor="black", alpha=0.35, pad=2, edgecolor="none"),
                color="white",
            )
            axes[r, 4].text(
                0.01,
                0.99,
                f"FP={cm2['FP']}  FN={cm2['FN']}",
                transform=axes[r, 4].transAxes,
                va="top",
                ha="left",
                fontsize=10,
                bbox=dict(facecolor="black", alpha=0.35, pad=2, edgecolor="none"),
                color="white",
            )

    if metrics1 is not None and metrics2 is not None:
        title = f"{dataset_name} ({split}) | ckpt1 F1={metrics1['f1']:.4f} | ckpt2 F1={metrics2['f1']:.4f}"
    else:
        title = f"{dataset_name} ({split})"
    fig.suptitle(title, fontsize=14, y=0.995)

    if show_legend and pred_viz == "overlay":
        handles = [
            Patch(facecolor=(0.0, 1.0, 0.0), edgecolor="none", label="TP"),
            Patch(facecolor=(1.0, 0.0, 0.0), edgecolor="none", label="FP"),
            Patch(facecolor=(0.15, 0.55, 1.0), edgecolor="none", label="FN"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.01))
        plt.tight_layout(rect=[0.0, 0.04, 1.0, 0.98])
    else:
        plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])

    if unc_colorbar and last_unc_im is not None:
        fig.colorbar(last_unc_im, ax=axes[:, 5:7].ravel().tolist(), fraction=0.02, pad=0.01)

    out_path = out_dir / f"compare_{dataset_name}_{split}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[{dataset_name}] Saved figure to {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Compare two checkpoints: metrics + 5x7 visualization grid (T1,T2,GT,pred1,pred2,unc1,unc2).")
    p.add_argument("--ckpt1", type=str, default=r"outputs\ablation\best\Best_levir--whu\best.pt")
    p.add_argument("--ckpt2", type=str, default=r"outputs\ablation\best\Best_whu--levir\best.pt")
    p.add_argument("--levir_root", type=str, default=r"data\LEVIR-CD")
    p.add_argument("--whu_root", type=str, default=r"data\WHUCD")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--out_dir", type=str, default=r"outputs\compare_ckpts")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_vis", type=int, default=5)
    p.add_argument("--thr_mode", type=str, default="fixed", choices=["fixed", "topk", "otsu"])
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--topk", type=float, default=0.01)
    p.add_argument("--smooth_k", type=int, default=3)
    p.add_argument("--use_minarea", dest="use_minarea", action="store_true")
    p.add_argument("--no_minarea", dest="use_minarea", action="store_false")
    p.set_defaults(use_minarea=True)
    p.add_argument("--min_area", type=int, default=256)
    p.add_argument("--window", type=int, default=256)
    p.add_argument("--stride", type=int, default=256)
    p.add_argument("--skip_eval", action="store_true", help="Skip full-dataset evaluation (faster; useful when only generating figures).")
    p.add_argument(
        "--pred_viz",
        type=str,
        default="overlay",
        choices=["overlay", "mask"],
        help="How to visualize predictions: 'overlay' colors TP/FP/FN on top of an image; 'mask' shows a binary mask.",
    )
    p.add_argument(
        "--overlay_base",
        type=str,
        default="t2",
        choices=["t1", "t2", "none"],
        help="Background for overlay visualization (default: t2).",
    )
    p.add_argument("--overlay_alpha", type=float, default=0.55, help="Alpha for TP/FP/FN overlay (default: 0.55).")
    p.add_argument("--show_stats", action="store_true", help="Show per-case FP/FN counts on pred panels.")
    p.add_argument("--no_legend", dest="show_legend", action="store_false", help="Disable TP/FP/FN legend for overlay mode.")
    p.set_defaults(show_legend=True)
    p.add_argument(
        "--unc_vmax",
        type=float,
        default=None,
        help="If set, fixes uncertainty vmax; otherwise uses 99.5% quantile across shown cases.",
    )
    p.add_argument("--unc_colorbar", action="store_true", help="Add a single colorbar for uncertainty panels.")
    return p.parse_args()


def main():
    args = parse_args()
    device = _device_from_arg(args.device)
    seed_everything(args.seed)

    ckpt1 = Path(args.ckpt1)
    ckpt2 = Path(args.ckpt2)
    out_dir = Path(args.out_dir)

    window = int(args.window) if args.window and int(args.window) > 0 else None
    stride = int(args.stride) if args.stride and int(args.stride) > 0 else None

    ckpt1_label = "LEVIR-CD-->WHU-CD"
    ckpt2_label = "WHU-CD-->LEVIR-CD"

    _compare_one_dataset(
        dataset_name="LEVIR-CD",
        data_root=args.levir_root,
        split=args.split,
        ckpt1=ckpt1,
        ckpt2=ckpt2,
        ckpt1_label=ckpt1_label,
        ckpt2_label=ckpt2_label,
        out_dir=out_dir / "LEVIR-CD",
        device=device,
        window=window,
        stride=stride,
        thr_mode=args.thr_mode,
        thr=args.thr,
        topk=args.topk,
        smooth_k=args.smooth_k,
        use_minarea=args.use_minarea,
        min_area=args.min_area,
        n_vis=args.n_vis,
        seed=args.seed,
        skip_eval=bool(args.skip_eval),
        pred_viz=args.pred_viz,
        overlay_base=args.overlay_base,
        overlay_alpha=args.overlay_alpha,
        show_stats=bool(args.show_stats),
        show_legend=bool(args.show_legend),
        unc_vmax=args.unc_vmax,
        unc_colorbar=bool(args.unc_colorbar),
    )

    _compare_one_dataset(
        dataset_name="WHUCD",
        data_root=args.whu_root,
        split=args.split,
        ckpt1=ckpt1,
        ckpt2=ckpt2,
        ckpt1_label=ckpt1_label,
        ckpt2_label=ckpt2_label,
        out_dir=out_dir / "WHUCD",
        device=device,
        window=window,
        stride=stride,
        thr_mode=args.thr_mode,
        thr=args.thr,
        topk=args.topk,
        smooth_k=args.smooth_k,
        use_minarea=args.use_minarea,
        min_area=args.min_area,
        n_vis=args.n_vis,
        seed=args.seed,
        skip_eval=bool(args.skip_eval),
        pred_viz=args.pred_viz,
        overlay_base=args.overlay_base,
        overlay_alpha=args.overlay_alpha,
        show_stats=bool(args.show_stats),
        show_legend=bool(args.show_legend),
        unc_vmax=args.unc_vmax,
        unc_colorbar=bool(args.unc_colorbar),
    )


if __name__ == "__main__":
    main()
