"""
Paper-style visualization for DINO change-detection head.

Creates N rows with 5 columns:
  T1 | T2 | GT | Prediction | Uncertainty

Default case selection (up to 5 cases):
  - success: highest IoU
  - boundary: highest boundary-F1 (excluding already chosen)
  - fp: highest false-positive pixels (excluding already chosen)
  - fn: highest false-negative pixels (excluding already chosen)
  - failure: lowest IoU

Works with checkpoints trained with or without multi-head ensemble. If logits_all is available and
multiple heads are selected, uncertainty defaults to per-pixel variance across heads; otherwise uses
entropy of the (ensemble) probability.

Example (zero-shot style: fixed head indices, no target-val fitting):
python vis_paper_cases.py --checkpoint outputs/dino_head_cd/best.pt --data_root data/LEVIR-CD \
  --split test --out_dir outputs/vis_paper --num_cases 5 --save_mode rows \
  --ensemble_strategy mean_logit --ensemble_indices 3,4 \
  --thr_mode fixed --thr 0.5 --smooth_k 3 --use_minarea --min_area 256 --full_eval
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F

# Avoid slow network calls (common in restricted environments).
# Must be set before importing albumentations/transformers via downstream modules.
os.environ.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from dino_head_core import (
    HeadCfg,
    DinoSiameseHead,
    build_dataloaders,
    seed_everything,
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
    try:
        return [int(x) for x in s.split(",") if str(x).strip() != ""]
    except Exception as e:
        raise ValueError(f"Invalid int list: {s}") from e


def _parse_float_list(s: Optional[str]) -> Optional[List[float]]:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    try:
        return [float(x) for x in s.split(",") if str(x).strip() != ""]
    except Exception as e:
        raise ValueError(f"Invalid float list: {s}") from e


def _denorm_img(x: torch.Tensor) -> np.ndarray:
    """
    x: [3,H,W] normalized
    returns: [H,W,3] float in [0,1]
    """
    mean = torch.as_tensor(_IMAGENET_MEAN, device=x.device).view(3, 1, 1)
    std = torch.as_tensor(_IMAGENET_STD, device=x.device).view(3, 1, 1)
    img = (x * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).detach().cpu().numpy()


def _entropy(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(eps, 1.0 - eps)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))


def _boundary_from_mask(mask: np.ndarray, dilation: int = 1) -> np.ndarray:
    """
    mask: [H,W] uint8 {0,1}
    returns boundary map [H,W] uint8 {0,1}
    """
    mask = (mask > 0).astype(np.uint8)
    if dilation <= 0:
        dilation = 1
    try:
        import cv2

        k = np.ones((3, 3), dtype=np.uint8)
        dil = cv2.dilate(mask, k, iterations=dilation)
        ero = cv2.erode(mask, k, iterations=dilation)
        b = (dil != ero).astype(np.uint8)
        return b
    except Exception:
        # Numpy fallback: approximate erosion by 3x3 min-filter
        m = mask
        for _ in range(dilation):
            pad = np.pad(m, 1, mode="edge")
            neigh = [
                pad[0:-2, 0:-2],
                pad[0:-2, 1:-1],
                pad[0:-2, 2:],
                pad[1:-1, 0:-2],
                pad[1:-1, 1:-1],
                pad[1:-1, 2:],
                pad[2:, 0:-2],
                pad[2:, 1:-1],
                pad[2:, 2:],
            ]
            m = np.minimum.reduce(neigh).astype(np.uint8)
        b = (mask != m).astype(np.uint8)
        return b


def _cm_from_masks(pred: np.ndarray, gt: np.ndarray) -> Tuple[int, int, int, int]:
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())
    return tp, fp, fn, tn


def _f1_iou(tp: int, fp: int, fn: int) -> Tuple[float, float]:
    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    return float(f1), float(iou)


@torch.no_grad()
def _infer_probs(
    model: torch.nn.Module,
    img_a: torch.Tensor,
    img_b: torch.Tensor,
    device: str,
    window: Optional[int],
    stride: Optional[int],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Returns:
      prob_fused: [1,1,H,W]
      logits_all: [K,1,1,H,W] or None
    """
    if window is not None and stride is not None:
        # sliding-window path (B==1)
        from dino_head_core import sliding_window_inference, sliding_window_inference_logits_all

        prob = sliding_window_inference(
            model=model,
            img_a=img_a,
            img_b=img_b,
            window=window,
            stride=stride,
            device=device,
            use_ensemble=False,
            ensemble_cfg=None,
        )
        logits_all = None
        try:
            logits_all = sliding_window_inference_logits_all(
                model=model,
                img_a=img_a,
                img_b=img_b,
                window=window,
                stride=stride,
                device=device,
            )
        except Exception:
            logits_all = None
        return prob, logits_all

    out = model(img_a, img_b)
    if isinstance(out, dict):
        prob = torch.sigmoid(out["pred"])
        logits_all = out.get("logits_all")
        return prob, logits_all
    return torch.sigmoid(out), None


def _ensemble_prob_from_logits_all(
    logits_all: torch.Tensor,
    indices: List[int],
    mode: str,
    weights: Optional[List[float]] = None,
) -> torch.Tensor:
    idx = torch.as_tensor(indices, device=logits_all.device, dtype=torch.long)
    sel = logits_all.index_select(0, idx)  # [K',B,1,H,W]
    if mode == "mean_prob":
        return torch.sigmoid(sel).mean(dim=0)
    if mode == "mean_logit":
        return torch.sigmoid(sel.mean(dim=0))
    if mode == "weighted_logit":
        if weights is None:
            raise ValueError("weighted_logit requires --ensemble_weights")
        w = torch.as_tensor(weights, device=logits_all.device, dtype=sel.dtype)
        if w.ndim != 1 or w.numel() != sel.shape[0]:
            raise ValueError("ensemble_weights length must match ensemble_indices length")
        w = (w / w.sum().clamp_min(1e-12)).view(-1, 1, 1, 1, 1)
        return torch.sigmoid((w * sel).sum(dim=0))
    raise ValueError(f"Unknown ensemble_mode: {mode}")


def _pick_cases(
    records: List[dict],
    n: int = 5,
) -> List[dict]:
    """
    records: list of dicts with keys: idx, name, iou, boundary_f1, fp, fn
    returns list of dicts: [{"role": str, ...record...}, ...] ordered for display.
    """
    if len(records) == 0:
        raise RuntimeError("Empty records; cannot pick cases.")
    if n <= 0:
        raise ValueError("--num_cases must be > 0")

    def _pick_one(cand: List[dict], key, reverse: bool = True) -> Optional[dict]:
        if not cand:
            return None
        return sorted(cand, key=key, reverse=reverse)[0]

    picks: List[dict] = []

    # Always try to pick "success" first.
    success = _pick_one(records, key=lambda r: (r.get("iou", 0.0), r.get("boundary_f1", 0.0)), reverse=True)
    if success is not None:
        picks.append({"role": "success", **success})
    if n == 1:
        return picks

    # "failure" last by default; pick it now but append later.
    failure = _pick_one(records, key=lambda r: (r.get("iou", 0.0), -r.get("boundary_f1", 0.0)), reverse=False)

    reserved = set()
    if success is not None:
        reserved.add(int(success["idx"]))
    if failure is not None:
        reserved.add(int(failure["idx"]))

    # For n==2, just show best vs worst IoU.
    if n == 2:
        if failure is not None and (success is None or int(failure["idx"]) != int(success["idx"])):
            picks.append({"role": "failure", **failure})
        return picks

    used = set(reserved)

    # boundary: highest boundary_f1 among remaining, break ties by iou
    cand = [r for r in records if int(r["idx"]) not in used]
    boundary = _pick_one(cand, key=lambda r: (r.get("boundary_f1", 0.0), r.get("iou", 0.0)), reverse=True)
    if boundary is None:
        boundary = success
    if boundary is not None and int(boundary["idx"]) not in used:
        used.add(int(boundary["idx"]))
    if boundary is not None:
        picks.append({"role": "boundary", **boundary})

    # fp/fn picks to complement the story (cross-domain errors often show up as FP/FN clusters).
    if n >= 4:
        cand = [r for r in records if int(r["idx"]) not in used]
        fp_case = _pick_one(cand, key=lambda r: (r.get("fp", 0), r.get("iou", 0.0)), reverse=True)
        if fp_case is not None:
            used.add(int(fp_case["idx"]))
            picks.append({"role": "fp", **fp_case})
    if n >= 5:
        cand = [r for r in records if int(r["idx"]) not in used]
        fn_case = _pick_one(cand, key=lambda r: (r.get("fn", 0), r.get("iou", 0.0)), reverse=True)
        if fn_case is not None:
            used.add(int(fn_case["idx"]))
            picks.append({"role": "fn", **fn_case})

    # Append failure at the end if possible.
    if failure is not None and (success is None or int(failure["idx"]) != int(success["idx"])):
        used.add(int(failure["idx"]))
        picks.append({"role": "failure", **failure})

    # If still not enough (rare), fill by evenly-spaced IoU ranks for diversity.
    if len(picks) < n:
        remaining = [r for r in records if int(r["idx"]) not in used]
        remaining_sorted = sorted(remaining, key=lambda r: r.get("iou", 0.0))
        if remaining_sorted:
            need = n - len(picks)
            for j in range(need):
                k = int(round((j + 1) * (len(remaining_sorted) - 1) / max(1, need)))
                r = remaining_sorted[k]
                if int(r["idx"]) in used:
                    continue
                used.add(int(r["idx"]))
                picks.append({"role": f"extra{j+1}", **r})

    return picks[:n]


def parse_args():
    base = HeadCfg()
    p = argparse.ArgumentParser(description="Paper-style visualization for DINO head")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_root", type=str, default=base.data_root)
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--device", type=str, default=base.device, help="cuda|cpu|auto")
    p.add_argument("--seed", type=int, default=base.seed)
    p.add_argument("--full_eval", dest="full_eval", action="store_true")
    p.add_argument("--no_full_eval", dest="full_eval", action="store_false")
    p.set_defaults(full_eval=True)
    p.add_argument("--eval_crop", type=int, default=base.eval_crop)
    p.add_argument("--window", type=int, default=None)
    p.add_argument("--stride", type=int, default=None)

    # prediction protocol
    p.add_argument("--thr_mode", type=str, choices=["fixed", "topk", "otsu"], default=base.thr_mode)
    p.add_argument("--thr", type=float, default=base.thr)
    p.add_argument("--topk", type=float, default=base.topk)
    p.add_argument("--smooth_k", type=int, default=base.smooth_k)
    p.add_argument("--use_minarea", action="store_true", default=base.use_minarea)
    p.add_argument("--min_area", type=int, default=base.min_area)

    # ensemble for prediction/uncertainty (fixed; no val fitting here)
    p.add_argument(
        "--ensemble_strategy",
        type=str,
        default="mean_logit",
        choices=["fused", "mean_prob", "mean_logit", "weighted_logit"],
        help="How to form probability for Prediction column.",
    )
    p.add_argument("--ensemble_indices", type=str, default=None, help="Comma-separated head indices, e.g. '3,4'")
    p.add_argument("--ensemble_weights", type=str, default=None, help="Comma-separated weights for weighted_logit")
    p.add_argument(
        "--uncertainty",
        type=str,
        default="var",
        choices=["var", "entropy"],
        help="Uncertainty map: var = per-pixel variance over selected heads' probabilities (if >=2 heads); otherwise entropy of the (ensemble) probability.",
    )
    p.add_argument(
        "--save_mode",
        type=str,
        default="rows",
        choices=["rows", "grid", "both"],
        help="rows: save one 1x5 figure per selected case; grid: save Nx5 grid; both: save both.",
    )

    # selection constraints
    p.add_argument("--num_cases", type=int, default=5, help="How many cases to visualize (default: 5).")
    p.add_argument("--min_pos", type=int, default=200, help="Min positive pixels in GT to consider a sample")
    p.add_argument("--boundary_dilation", type=int, default=2, help="Boundary thickness for boundary-F1 selection")

    args = p.parse_args()
    args.ensemble_indices = _parse_int_list(args.ensemble_indices)
    args.ensemble_weights = _parse_float_list(args.ensemble_weights)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.window is not None and args.window <= 0:
        args.window = None
    if args.stride is not None and args.stride <= 0:
        args.stride = None
    if (args.window is None) ^ (args.stride is None):
        raise ValueError("--window and --stride must be set together (or both omitted).")
    return args


def _plot_grid(
    rows: List[dict],
    out_path: str,
    unc_vmax: float,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col_titles = ["T1", "T2", "GT", "Prediction", "Uncertainty"]

    def _row_title(item: dict) -> str:
        role = str(item.get("role", "")).upper()
        name = str(item.get("name", ""))
        iou = float(item.get("iou", 0.0))
        f1 = float(item.get("f1", 0.0))
        bf1 = float(item.get("boundary_f1", 0.0))
        fp = int(item.get("fp", 0))
        fn = int(item.get("fn", 0))
        if str(item.get("role")) in ("fp", "fn"):
            return f"{role}  ({name})  FP={fp}  FN={fn}  IoU={iou:.3f}  F1={f1:.3f}"
        return f"{role}  ({name})  IoU={iou:.3f}  F1={f1:.3f}  bF1={bf1:.3f}"

    row_titles = [_row_title(r) for r in rows]

    nrows = len(rows)
    fig_h = max(4.0, 3.2 * nrows)
    fig, axes = plt.subplots(nrows, 5, figsize=(18, fig_h))
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)
    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=14, fontweight="bold")

    for r in range(nrows):
        axes[r, 0].set_ylabel(row_titles[r], fontsize=11, rotation=90, labelpad=18)
        axes[r, 0].yaxis.set_label_position("left")

    for r, item in enumerate(rows):
        axes[r, 0].imshow(item["t1"])
        axes[r, 1].imshow(item["t2"])
        axes[r, 2].imshow(item["gt"], cmap="gray", vmin=0, vmax=1)
        axes[r, 3].imshow(item["pred_overlay"])
        im = axes[r, 4].imshow(item["uncert"], cmap="magma", vmin=0.0, vmax=unc_vmax)

        for c in range(5):
            axes[r, c].axis("off")

    # Leave room on the right for a global colorbar (avoid overlapping the uncertainty panels).
    plt.tight_layout(rect=[0.0, 0.0, 0.90, 1.0])
    cax = fig.add_axes([0.92, 0.12, 0.015, 0.76])
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=9)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_row(case: dict, out_path: str, unc_vmax: float):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col_titles = ["T1", "T2", "GT", "Prediction", "Uncertainty"]
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for c, t in enumerate(col_titles):
        axes[c].set_title(t, fontsize=14, fontweight="bold")
        axes[c].axis("off")

    axes[0].imshow(case["t1"])
    axes[1].imshow(case["t2"])
    axes[2].imshow(case["gt"], cmap="gray", vmin=0, vmax=1)
    axes[3].imshow(case["pred_overlay"])
    im = axes[4].imshow(case["uncert"], cmap="magma", vmin=0.0, vmax=unc_vmax)

    plt.tight_layout(rect=[0.0, 0.0, 0.90, 0.92])
    cax = fig.add_axes([0.92, 0.18, 0.015, 0.64])
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=9)
    role = str(case.get("role", "")).upper()
    title = f"{role}  {case['name']}  IoU={case['iou']:.3f}  F1={case['f1']:.3f}  bF1={case['boundary_f1']:.3f}"
    fig.suptitle(title, fontsize=12, y=0.98)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _make_overlay(
    img: np.ndarray,
    mask: np.ndarray,
    color=(1.0, 0.0, 0.0),
    alpha=0.45,
    gt: Optional[np.ndarray] = None,
    gt_color=(0.0, 1.0, 0.0),
) -> np.ndarray:
    out = img.copy()
    m = mask.astype(bool)
    if m.any():
        out[m] = (1.0 - alpha) * out[m] + alpha * np.array(color, dtype=np.float32)
    if gt is not None:
        try:
            import cv2

            gt_u8 = (gt > 0).astype(np.uint8)
            contours, _ = cv2.findContours(gt_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            out_u8 = (out * 255.0).astype(np.uint8)
            # cv2 uses BGR
            bgr = (int(gt_color[2] * 255), int(gt_color[1] * 255), int(gt_color[0] * 255))
            cv2.drawContours(out_u8, contours, -1, bgr, 2)
            out = out_u8.astype(np.float32) / 255.0
        except Exception:
            pass
    return out


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = args.device

    ckpt = torch.load(args.checkpoint, map_location=device)
    load_cfg = ckpt.get("cfg") if isinstance(ckpt, dict) else None

    out_dir = args.out_dir or str(Path(args.checkpoint).parent / "paper_vis")
    os.makedirs(out_dir, exist_ok=True)

    cfg = HeadCfg(
        data_root=args.data_root,
        out_dir=out_dir,
        device=device,
        seed=args.seed,
        full_eval=bool(args.full_eval),
        eval_crop=int(args.eval_crop),
        use_ensemble_pred=False,
    )
    # allow overriding window/stride for full images
    cfg.eval_window = args.window
    cfg.eval_stride = args.stride

    # build loaders with batch_size=1 for deterministic case selection
    cfg.batch_size = 1
    _, val_loader, test_loader = build_dataloaders(cfg)
    loader = {"train": None, "val": val_loader, "test": test_loader}[args.split]
    if loader is None:
        raise ValueError("split=train not supported for this visualization script.")

    if isinstance(load_cfg, dict):
        model = DinoSiameseHead(
            dino_name=load_cfg.get("dino_name", cfg.dino_name),
            use_whiten=load_cfg.get("use_whiten", cfg.use_whiten),
            use_domain_adv=load_cfg.get("use_domain_adv", cfg.use_domain_adv),
            domain_hidden=load_cfg.get("domain_hidden", cfg.domain_hidden),
            domain_grl=load_cfg.get("domain_grl", cfg.domain_grl),
            use_style_norm=load_cfg.get("use_style_norm", cfg.use_style_norm),
            proto_path=load_cfg.get("proto_path", cfg.proto_path),
            proto_weight=load_cfg.get("proto_weight", cfg.proto_weight),
            boundary_dim=load_cfg.get("boundary_dim", cfg.boundary_dim),
            use_layer_ensemble=load_cfg.get("use_layer_ensemble", cfg.use_layer_ensemble),
            layer_head_ch=load_cfg.get("layer_head_ch", cfg.layer_head_ch),
        ).to(device)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    else:
        model = DinoSiameseHead().to(device)
        model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
    model.eval()

    # first pass: compute per-sample metrics for selection
    records: List[dict] = []
    for idx, batch in enumerate(loader):
        img_a = batch["img_a"].to(device)
        img_b = batch["img_b"].to(device)
        gt_t = batch["label"]
        name = batch["name"][0] if isinstance(batch["name"], list) else batch["name"]
        if gt_t.ndim == 3:
            gt_t = gt_t.unsqueeze(1)
        gt_np = (gt_t[0, 0].detach().cpu().numpy() > 0).astype(np.uint8)
        if int(gt_np.sum()) < int(args.min_pos):
            continue

        prob_fused, logits_all = _infer_probs(model, img_a, img_b, device=device, window=args.window, stride=args.stride)
        if args.smooth_k and args.smooth_k > 1:
            pad = args.smooth_k // 2
            prob_fused = F.avg_pool2d(prob_fused, kernel_size=args.smooth_k, stride=1, padding=pad)

        # choose prob for prediction (fixed; no val fitting)
        prob = prob_fused
        if args.ensemble_strategy != "fused" and logits_all is not None:
            if args.ensemble_indices is None:
                # default: fused only
                indices = [int(logits_all.shape[0]) - 1]
            else:
                indices = args.ensemble_indices
            prob = _ensemble_prob_from_logits_all(
                logits_all=logits_all,
                indices=indices,
                mode=args.ensemble_strategy,
                weights=args.ensemble_weights,
            )
            if args.smooth_k and args.smooth_k > 1:
                pad = args.smooth_k // 2
                prob = F.avg_pool2d(prob, kernel_size=args.smooth_k, stride=1, padding=pad)

        prob_np = prob[0, 0].detach().cpu().numpy().astype(np.float32)
        pred_np, _ = threshold_map(prob_np, args.thr_mode, args.thr, args.topk)
        if args.use_minarea:
            pred_np = filter_small_cc(pred_np, min_area=args.min_area)

        tp, fp, fn, _ = _cm_from_masks(pred_np, gt_np)
        f1, iou = _f1_iou(tp, fp, fn)

        # boundary f1 for selection
        b_gt = _boundary_from_mask(gt_np, dilation=int(args.boundary_dilation))
        b_pr = _boundary_from_mask(pred_np, dilation=int(args.boundary_dilation))
        btp, bfp, bfn, _ = _cm_from_masks(b_pr, b_gt)
        b_f1, _ = _f1_iou(btp, bfp, bfn)

        records.append(
            {
                "idx": idx,
                "name": str(name),
                "f1": f1,
                "iou": iou,
                "boundary_f1": b_f1,
                "fp": int(fp),
                "fn": int(fn),
            }
        )

    picks = _pick_cases(records, n=int(args.num_cases))
    picked_indices = {p["idx"] for p in picks}
    meta_path = os.path.join(out_dir, "paper_cases.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"picks": picks, "args": vars(args)}, f, ensure_ascii=False, indent=2)
    print(f"[PaperVis] Picked cases saved to {meta_path}")

    # second pass: render selected cases
    rows = []
    for idx, batch in enumerate(loader):
        if idx not in picked_indices:
            continue
        img_a = batch["img_a"].to(device)
        img_b = batch["img_b"].to(device)
        gt_t = batch["label"]
        name = batch["name"][0] if isinstance(batch["name"], list) else batch["name"]
        if gt_t.ndim == 3:
            gt_t = gt_t.unsqueeze(1)
        gt_np = (gt_t[0, 0].detach().cpu().numpy() > 0).astype(np.uint8)

        prob_fused, logits_all = _infer_probs(model, img_a, img_b, device=device, window=args.window, stride=args.stride)
        if args.smooth_k and args.smooth_k > 1:
            pad = args.smooth_k // 2
            prob_fused = F.avg_pool2d(prob_fused, kernel_size=args.smooth_k, stride=1, padding=pad)

        if args.ensemble_indices is None and logits_all is not None:
            indices = [int(logits_all.shape[0]) - 1]
        else:
            indices = args.ensemble_indices or []

        prob = prob_fused
        prob_sel = None
        if args.ensemble_strategy != "fused" and logits_all is not None and indices:
            prob = _ensemble_prob_from_logits_all(
                logits_all=logits_all,
                indices=indices,
                mode=args.ensemble_strategy,
                weights=args.ensemble_weights,
            )
            # for uncertainty variance: keep per-head probs
            if len(indices) > 1:
                idx_t = torch.as_tensor(indices, device=logits_all.device, dtype=torch.long)
                sel = logits_all.index_select(0, idx_t)
                prob_sel = torch.sigmoid(sel)  # [K',1,1,H,W]

        if args.smooth_k and args.smooth_k > 1:
            pad = args.smooth_k // 2
            prob = F.avg_pool2d(prob, kernel_size=args.smooth_k, stride=1, padding=pad)
            if prob_sel is not None:
                Kp, Bp, _, Hp, Wp = prob_sel.shape
                prob_sel = prob_sel.view(Kp * Bp, 1, Hp, Wp)
                prob_sel = F.avg_pool2d(prob_sel, kernel_size=args.smooth_k, stride=1, padding=pad)
                prob_sel = prob_sel.view(Kp, Bp, 1, Hp, Wp)

        prob_np = prob[0, 0].detach().cpu().numpy().astype(np.float32)
        pred_np, _ = threshold_map(prob_np, args.thr_mode, args.thr, args.topk)
        if args.use_minarea:
            pred_np = filter_small_cc(pred_np, min_area=args.min_area)

        tp, fp, fn, _ = _cm_from_masks(pred_np, gt_np)
        f1, iou = _f1_iou(tp, fp, fn)
        b_gt = _boundary_from_mask(gt_np, dilation=int(args.boundary_dilation))
        b_pr = _boundary_from_mask(pred_np, dilation=int(args.boundary_dilation))
        btp, bfp, bfn, _ = _cm_from_masks(b_pr, b_gt)
        b_f1, _ = _f1_iou(btp, bfp, bfn)

        # uncertainty map
        if args.uncertainty == "var" and prob_sel is not None and prob_sel.shape[0] > 1:
            uncert_t = prob_sel.var(dim=0)  # [1,1,H,W]
        else:
            uncert_t = _entropy(prob)  # [1,1,H,W]
        uncert = uncert_t[0, 0].detach().cpu().numpy().astype(np.float32)

        t1 = _denorm_img(img_a[0])
        t2 = _denorm_img(img_b[0])
        # Overlay prediction (red) and GT contour (green) for paper-friendly comparison.
        pred_overlay = _make_overlay(t2, pred_np, color=(1.0, 0.0, 0.0), alpha=0.45, gt=gt_np, gt_color=(0.0, 1.0, 0.0))

        rows.append(
            {
                "idx": int(idx),
                "name": str(name),
                "t1": t1,
                "t2": t2,
                "gt": gt_np,
                "pred": pred_np,
                "pred_overlay": pred_overlay,
                "uncert": uncert,
                "f1": f1,
                "iou": iou,
                "boundary_f1": b_f1,
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
            }
        )

    rows_by_idx = {int(r["idx"]): r for r in rows}
    ordered = []
    for p in picks:
        ridx = int(p["idx"])
        if ridx not in rows_by_idx:
            continue
        item = dict(rows_by_idx[ridx])
        item["role"] = str(p.get("role", "case"))
        ordered.append(item)
    if not ordered:
        raise RuntimeError("No selected rows to render (unexpected).")

    # Shared uncertainty scale across selected cases.
    uncert_stack = np.stack([x["uncert"] for x in ordered], axis=0)
    unc_vmax = float(np.quantile(uncert_stack, 0.995))
    if not np.isfinite(unc_vmax) or unc_vmax <= 0:
        unc_vmax = float(np.max(uncert_stack) if np.max(uncert_stack) > 0 else 1.0)

    if args.save_mode in ("rows", "both"):
        for item in ordered:
            role = str(item.get("role", "case"))
            out_path = os.path.join(out_dir, f"{role}.png")
            _plot_row(item, out_path=out_path, unc_vmax=unc_vmax)
        print(f"[PaperVis] Saved rows to {out_dir}")

    if args.save_mode in ("grid", "both"):
        out_path = os.path.join(out_dir, "paper_cases.png")
        _plot_grid(ordered, out_path=out_path, unc_vmax=unc_vmax)
        print(f"[PaperVis] Saved: {out_path}")


if __name__ == "__main__":
    main()
