import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Avoid slow/blocked network calls (common in restricted environments).
os.environ.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

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


def _parse_csv_items(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def _resolve_dpi(
    *,
    fig_w_in: float,
    fig_h_in: float,
    dpi: int,
    auto_dpi: bool,
    max_megapixels: float,
) -> int:
    dpi = int(dpi)
    if dpi <= 0:
        dpi = 240
    if not auto_dpi:
        return dpi

    fig_w_in = float(fig_w_in)
    fig_h_in = float(fig_h_in)
    max_megapixels = float(max_megapixels)
    if fig_w_in <= 0 or fig_h_in <= 0 or max_megapixels <= 0:
        return dpi

    target_pixels = max_megapixels * 1_000_000.0
    cur_pixels = (fig_w_in * dpi) * (fig_h_in * dpi)
    if cur_pixels <= target_pixels:
        return dpi

    new_dpi = int((target_pixels / (fig_w_in * fig_h_in)) ** 0.5)
    return max(72, min(dpi, new_dpi))


def _safe_mean(x: np.ndarray) -> float:
    x = np.asarray(x)
    if x.size == 0:
        return 0.0
    v = float(np.mean(x))
    if not np.isfinite(v):
        return 0.0
    return v


def _iou_from_cm(cm: Dict[str, int]) -> float:
    tp = float(cm.get("TP", 0))
    fp = float(cm.get("FP", 0))
    fn = float(cm.get("FN", 0))
    return float(tp / (tp + fp + fn + 1e-12))


def _select_paper_cases(
    pool: List[Dict[str, Any]],
    *,
    n_vis: int,
    min_gt_pixels: int,
    min_err_pixels: int,
) -> List[Dict[str, Any]]:
    """
    Pick visually-informative cases for paper-style qualitative figures.

    Assumes:
      - model1 is in-domain, model2 is cross-domain (as passed by --paper_fig45 mode).
      - Each pool item contains cm1/cm2, pred2/gt, unc2.
    """
    if not pool:
        return []

    # Add selection features.
    for c in pool:
        gt = c["gt"]
        pred2 = c["pred2"]
        cm1 = c["cm1"]
        cm2 = c["cm2"]
        gt_pos = int((gt > 0).sum())
        fp2 = int(cm2.get("FP", 0))
        fn2 = int(cm2.get("FN", 0))
        err2 = fp2 + fn2
        iou1 = _iou_from_cm(cm1)
        iou2 = _iou_from_cm(cm2)
        delta_iou = float(iou1 - iou2)
        err_mask2 = (pred2.astype(np.uint8) != (gt > 0).astype(np.uint8))
        unc2 = c["unc2"]
        unc2_err_mean = _safe_mean(unc2[err_mask2]) if np.any(err_mask2) else 0.0
        unc2_mean = _safe_mean(unc2)
        c["_sel"] = {
            "gt_pos": gt_pos,
            "fp2": fp2,
            "fn2": fn2,
            "err2": err2,
            "iou1": iou1,
            "iou2": iou2,
            "delta_iou": delta_iou,
            "unc2_err_mean": unc2_err_mean,
            "unc2_mean": unc2_mean,
        }

    # Prefer cases with either meaningful GT or clear cross-domain errors.
    candidates = [
        c
        for c in pool
        if c["_sel"]["gt_pos"] >= int(min_gt_pixels) or c["_sel"]["err2"] >= int(min_err_pixels)
    ]
    if len(candidates) < n_vis:
        candidates = list(pool)

    picks: List[Dict[str, Any]] = []
    reserved: set[int] = set()

    def _pick_one(where, key, reverse: bool = True) -> Optional[Dict[str, Any]]:
        xs = [c for c in candidates if int(c["idx"]) not in reserved and where(c)]
        if not xs:
            return None
        best = sorted(xs, key=key, reverse=reverse)[0]
        reserved.add(int(best["idx"]))
        picks.append(best)
        return best

    # (a) "domain shift" case: in-domain good, cross-domain clearly worse.
    _pick_one(
        lambda c: c["_sel"]["gt_pos"] >= int(min_gt_pixels) and c["_sel"]["iou1"] >= 0.55 and c["_sel"]["delta_iou"] > 0.05,
        key=lambda c: (c["_sel"]["delta_iou"], c["_sel"]["err2"], c["_sel"]["gt_pos"]),
        reverse=True,
    )
    # (b) FP-heavy cross-domain.
    _pick_one(
        lambda c: c["_sel"]["fp2"] >= int(min_err_pixels) // 2,
        key=lambda c: (c["_sel"]["fp2"], c["_sel"]["delta_iou"], c["_sel"]["err2"]),
        reverse=True,
    )
    # (c) FN-heavy cross-domain.
    _pick_one(
        lambda c: c["_sel"]["fn2"] >= int(min_err_pixels) // 2,
        key=lambda c: (c["_sel"]["fn2"], c["_sel"]["delta_iou"], c["_sel"]["err2"]),
        reverse=True,
    )
    # (d) worst cross-domain IoU (but not empty GT).
    _pick_one(
        lambda c: c["_sel"]["gt_pos"] >= int(min_gt_pixels),
        key=lambda c: (c["_sel"]["iou2"], -c["_sel"]["err2"]),
        reverse=False,
    )
    # (e) uncertainty highlights errors (high uncertainty on error pixels).
    _pick_one(
        lambda c: c["_sel"]["err2"] >= int(min_err_pixels) // 2,
        key=lambda c: (c["_sel"]["unc2_err_mean"] * (1.0 + c["_sel"]["err2"]), c["_sel"]["delta_iou"]),
        reverse=True,
    )

    # Fill remaining slots with high-error, high-gap cases.
    while len(picks) < int(n_vis):
        nxt = _pick_one(
            lambda c: True,
            key=lambda c: (c["_sel"]["err2"], c["_sel"]["delta_iou"], c["_sel"]["gt_pos"]),
            reverse=True,
        )
        if nxt is None:
            break

    # Keep ordering as picked above.
    return picks[: int(n_vis)]


def _infer_train_domain_from_path(p: Path) -> Optional[str]:
    s = str(p).replace("\\", "/").lower()
    if "levir--whu" in s:
        return "levir"
    if "whu--levir" in s:
        return "whu"
    has_levir = "levir" in s
    has_whu = "whu" in s
    if has_levir and not has_whu:
        return "levir"
    if has_whu and not has_levir:
        return "whu"
    return None


def _domain_display(domain: str) -> str:
    d = str(domain).lower()
    if d == "levir":
        return "LEVIR-CD"
    if d == "whu":
        return "WHU-CD"
    return str(domain)


def _arrow_label(src_domain: Optional[str], tgt_domain: str) -> str:
    if src_domain is None:
        return tgt_domain
    return f"{_domain_display(src_domain)}→{tgt_domain}"


def _estimate_unc_vmax_for_dataset(
    *,
    data_root: str,
    split: str,
    ckpt1: Path,
    ckpt2: Path,
    device: str,
    window: Optional[int],
    stride: Optional[int],
    thr: float,
    smooth_k: int,
    use_minarea: bool,
    min_area: int,
    n_vis: int,
    seed: int,
) -> float:
    seed_everything(seed)
    ds, _ = _build_test_loader(data_root=data_root, split=split, batch_size=1, num_workers=0)
    indices = _pick_indices(len(ds), n_vis)

    model1, _ = _load_model_from_ckpt(ckpt1, device=device)
    model2, _ = _load_model_from_ckpt(ckpt2, device=device)

    uncs: List[np.ndarray] = []
    for idx in indices:
        sample = ds[idx]
        img_a = sample["img_a"].unsqueeze(0)
        img_b = sample["img_b"].unsqueeze(0)

        _, _, unc1 = _infer_prob_and_uncertainty(
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
        _, _, unc2 = _infer_prob_and_uncertainty(
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
        uncs.append(unc1)
        uncs.append(unc2)

    if not uncs:
        return 1.0
    stack = np.stack(uncs, axis=0)
    vmax_unc = float(np.quantile(stack, 0.995))
    if not np.isfinite(vmax_unc) or vmax_unc <= 0:
        vmax_unc = float(np.max(stack) if np.max(stack) > 0 else 1.0)
    return vmax_unc


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
    font_scale: float,
    *,
    fig_title: Optional[str] = None,
    out_name: Optional[str] = None,
    show_case_names: bool = True,
    show_row_tags: bool = True,
    case_strategy: str = "uniform",
    case_pool: int = 200,
    min_gt_pixels: int = 256,
    min_err_pixels: int = 256,
    indices_override: Optional[List[str]] = None,
    dpi: int = 300,
    auto_dpi: bool = True,
    max_megapixels: float = 24.0,
    tight_bbox: bool = True,
    panel_size: float = 3.2,
):
    import matplotlib

    matplotlib.use("Agg")
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
    if indices_override is not None:
        name_to_idx: Dict[str, int] = {}
        if hasattr(ds, "img_names") and isinstance(getattr(ds, "img_names"), list):
            name_to_idx = {str(n): i for i, n in enumerate(getattr(ds, "img_names"))}

        indices: List[int] = []
        unresolved: List[str] = []
        for raw in indices_override:
            tok = str(raw).strip()
            if not tok:
                continue

            # 1) Direct name match.
            if tok in name_to_idx:
                indices.append(int(name_to_idx[tok]))
                continue

            # 2) Try common dataset name patterns for numeric tokens.
            if tok.isdigit():
                cand_names = [f"test_{tok}", f"val_{tok}", f"train_{tok}", f"0_{tok}"]
                hit = next((nm for nm in cand_names if nm in name_to_idx), None)
                if hit is not None:
                    indices.append(int(name_to_idx[hit]))
                    continue

                # Unique suffix match like "*_253" -> "0_253".
                suffix = f"_{tok}"
                suffix_hits = [nm for nm in name_to_idx.keys() if nm.endswith(suffix)]
                if len(suffix_hits) == 1:
                    indices.append(int(name_to_idx[suffix_hits[0]]))
                    continue

            # 3) Fallback: treat as index (0-based).
            try:
                idx0 = int(tok)
            except Exception:
                idx0 = None
            if idx0 is not None and 0 <= idx0 < len(ds):
                indices.append(int(idx0))
                continue

            # 4) Fallback: treat as 1-based index.
            if idx0 is not None and 1 <= idx0 <= len(ds):
                indices.append(int(idx0 - 1))
                continue

            unresolved.append(tok)

        if unresolved:
            raise ValueError(
                f"Could not resolve indices for {dataset_name}: {unresolved}. "
                f"Hint: pass sample names (e.g., test_128 or 0_253) or valid indices in [0,{len(ds)-1}]."
            )
        case_strategy = "fixed"
    else:
        case_strategy = str(case_strategy or "uniform").lower()
        if case_strategy not in ("uniform", "paper"):
            raise ValueError(f"Unknown case_strategy={case_strategy!r} (expected 'uniform' or 'paper').")

        if case_strategy == "paper":
            # Build a candidate pool, then select n_vis cases from it.
            pool_n = int(case_pool) if case_pool and int(case_pool) > 0 else int(n_vis)
            pool_n = min(pool_n, len(ds))
            pool_indices = _pick_indices(len(ds), pool_n)
            indices = pool_indices
        else:
            indices = _pick_indices(len(ds), n_vis)

    # Precompute predictions and uncertainties (and selection stats if needed).
    cases_all: List[Dict[str, Any]] = []
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

        cases_all.append(
            {
                "idx": int(idx),
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

    if case_strategy == "paper":
        selected = _select_paper_cases(
            cases_all,
            n_vis=int(n_vis),
            min_gt_pixels=int(min_gt_pixels),
            min_err_pixels=int(min_err_pixels),
        )
        if selected:
            cases = selected
        else:
            cases = cases_all[: int(n_vis)]
    else:
        cases = cases_all

    if unc_vmax is None:
        uncert_stack = np.stack([x["unc1"] for x in cases] + [x["unc2"] for x in cases], axis=0)
        vmax_unc = float(np.quantile(uncert_stack, 0.995))
        if not np.isfinite(vmax_unc) or vmax_unc <= 0:
            vmax_unc = float(np.max(uncert_stack) if np.max(uncert_stack) > 0 else 1.0)
    else:
        vmax_unc = float(unc_vmax)

    rows = len(cases)
    cols = 7
    panel_size = float(panel_size) if panel_size else 2.8
    fig_w = panel_size * cols
    fig_h = panel_size * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    font_scale = float(font_scale) if font_scale else 1.0

    def _fs(x: float) -> int:
        return int(round(float(x) * font_scale))

    titles = [
        "T1",
        "T2",
        "GT",
        f"Pred\n({ckpt1_label})",
        f"Pred\n({ckpt2_label})",
        f"Unc\n({ckpt1_label})",
        f"Unc\n({ckpt2_label})",
    ]
    for c in range(cols):
        axes[0, c].set_title(titles[c], fontsize=_fs(12))

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

        if show_case_names:
            axes[r, 0].text(
                0.01,
                0.99,
                f"{case['name']}",
                transform=axes[r, 0].transAxes,
                va="top",
                ha="left",
                fontsize=_fs(10),
                bbox=dict(facecolor="black", alpha=0.35, pad=2, edgecolor="none"),
                color="white",
            )
        if show_row_tags:
            row_tag = chr(ord("a") + r)
            axes[r, 0].text(
                -0.08,
                0.5,
                f"({row_tag})",
                transform=axes[r, 0].transAxes,
                va="center",
                ha="right",
                fontsize=_fs(12),
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
                fontsize=_fs(10),
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
                fontsize=_fs(10),
                bbox=dict(facecolor="black", alpha=0.35, pad=2, edgecolor="none"),
                color="white",
            )

    if fig_title is not None:
        title = str(fig_title)
    elif metrics1 is not None and metrics2 is not None:
        title = f"{dataset_name} ({split}) | ckpt1 F1={metrics1['f1']:.4f} | ckpt2 F1={metrics2['f1']:.4f}"
    else:
        title = f"{dataset_name} ({split})"
    fig.suptitle(title, fontsize=_fs(14), y=0.995)

    if show_legend and pred_viz == "overlay":
        handles = [
            Patch(facecolor=(0.0, 1.0, 0.0), edgecolor="none", label="TP"),
            Patch(facecolor=(1.0, 0.0, 0.0), edgecolor="none", label="FP"),
            Patch(facecolor=(0.15, 0.55, 1.0), edgecolor="none", label="FN"),
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.01),
            prop={"size": _fs(11)},
        )
        right = 0.92 if unc_colorbar else 1.0
        plt.tight_layout(rect=[0.0, 0.04, right, 0.98])
    else:
        right = 0.92 if unc_colorbar else 1.0
        plt.tight_layout(rect=[0.0, 0.0, right, 0.98])

    if unc_colorbar and last_unc_im is not None:
        fig.canvas.draw()
        unc_axes = axes[:, 5:7].ravel().tolist()
        bboxes = [ax.get_position() for ax in unc_axes]
        y0 = min(bb.y0 for bb in bboxes)
        y1 = max(bb.y1 for bb in bboxes)
        x1 = max(bb.x1 for bb in bboxes)
        cax = fig.add_axes([x1 + 0.008, y0, 0.012, y1 - y0])
        cbar = fig.colorbar(last_unc_im, cax=cax)
        cbar.ax.tick_params(labelsize=_fs(10))

    out_path = out_dir / (str(out_name) if out_name else f"compare_{dataset_name}_{split}.png")
    resolved_dpi = _resolve_dpi(
        fig_w_in=fig_w,
        fig_h_in=fig_h,
        dpi=int(dpi),
        auto_dpi=bool(auto_dpi),
        max_megapixels=float(max_megapixels),
    )
    save_kwargs = {"dpi": resolved_dpi}
    if bool(tight_bbox):
        save_kwargs["bbox_inches"] = "tight"
    fig.savefig(out_path, **save_kwargs)
    plt.close(fig)
    print(f"[{dataset_name}] Saved figure to {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Compare two checkpoints: metrics + 5x7 visualization grid (T1,T2,GT,pred1,pred2,unc1,unc2).")
    p.add_argument("--ckpt1", type=str, default=r"outputs\ablation\best\Best_levir--whu\best.pt")
    p.add_argument("--ckpt2", type=str, default=r"outputs\ablation\best\Best_whu--levir\best.pt")
    p.add_argument("--ckpt1_label", type=str, default=None, help="Optional label shown in figure for ckpt1 (default: auto).")
    p.add_argument("--ckpt2_label", type=str, default=None, help="Optional label shown in figure for ckpt2 (default: auto).")
    p.add_argument(
        "--ckpt1_train_domain",
        type=str,
        default=None,
        choices=["levir", "whu"],
        help="Optional training/source domain for ckpt1 (used by --paper_fig45).",
    )
    p.add_argument(
        "--ckpt2_train_domain",
        type=str,
        default=None,
        choices=["levir", "whu"],
        help="Optional training/source domain for ckpt2 (used by --paper_fig45).",
    )
    p.add_argument("--levir_root", type=str, default=r"data\LEVIR-CD")
    p.add_argument("--whu_root", type=str, default=r"data\WHUCD")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--out_dir", type=str, default=r"outputs\compare_ckpts_final")
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
        help="If set, fixes uncertainty vmax; otherwise uses 99.5%% quantile across shown cases.",
    )
    p.add_argument(
        "--unc_vmax_shared",
        action="store_true",
        help="If set (and --unc_vmax is not set), uses a single shared unc_vmax for both datasets in this run.",
    )
    p.add_argument("--unc_colorbar", action="store_true", help="Add a single colorbar for uncertainty panels.")
    p.add_argument("--font_scale", type=float, default=1.35, help="Scale all font sizes for the output figure.")
    p.add_argument("--paper_fig45", action="store_true", help="Generate paper Figures 4-5 (in-domain vs cross-domain) with correct ordering/titles.")
    p.add_argument("--no_case_names", dest="show_case_names", action="store_false", help="Hide sample name tags on the leftmost panel.")
    p.set_defaults(show_case_names=True)
    p.add_argument("--no_row_tags", dest="show_row_tags", action="store_false", help="Hide (a)(b)(c)... row tags.")
    p.set_defaults(show_row_tags=True)
    p.add_argument("--levir_indices", type=str, default=None, help="Comma-separated indices for LEVIR rows (used by --paper_fig45).")
    p.add_argument("--whu_indices", type=str, default=None, help="Comma-separated indices for WHU rows (used by --paper_fig45).")
    p.add_argument(
        "--case_strategy",
        type=str,
        default="auto",
        choices=["auto", "uniform", "paper"],
        help="Case selection strategy: uniform spacing or paper-oriented (prefers clear in-vs-cross differences).",
    )
    p.add_argument("--case_pool", type=int, default=200, help="Pool size scanned for --case_strategy paper (default: 200).")
    p.add_argument("--min_gt_pixels", type=int, default=256, help="Minimum GT positive pixels preferred for paper cases (default: 256).")
    p.add_argument("--min_err_pixels", type=int, default=256, help="Minimum cross-domain error pixels preferred for paper cases (default: 256).")
    p.add_argument("--dpi", type=int, default=300, help="Output DPI for saved figures (default: 300).")
    p.add_argument("--no_auto_dpi", dest="auto_dpi", action="store_false", help="Disable auto DPI downscaling to avoid OOM.")
    p.set_defaults(auto_dpi=True)
    p.add_argument("--max_megapixels", type=float, default=24.0, help="Max megapixels when --auto_dpi is enabled (default: 24).")
    p.add_argument("--no_tight_bbox", dest="tight_bbox", action="store_false", help="Disable bbox_inches='tight' when saving.")
    p.set_defaults(tight_bbox=True)
    p.add_argument("--panel_size", type=float, default=3.2, help="Panel size in inches (default: 3.2).")
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

    ckpt1_train = args.ckpt1_train_domain or _infer_train_domain_from_path(ckpt1)
    ckpt2_train = args.ckpt2_train_domain or _infer_train_domain_from_path(ckpt2)
    ckpt1_label = args.ckpt1_label or ckpt1.stem
    ckpt2_label = args.ckpt2_label or ckpt2.stem

    shared_unc_vmax = args.unc_vmax
    if shared_unc_vmax is None and bool(args.unc_vmax_shared):
        vmax_levir = _estimate_unc_vmax_for_dataset(
            data_root=args.levir_root,
            split=args.split,
            ckpt1=ckpt1,
            ckpt2=ckpt2,
            device=device,
            window=window,
            stride=stride,
            thr=float(args.thr),
            smooth_k=int(args.smooth_k),
            use_minarea=bool(args.use_minarea),
            min_area=int(args.min_area),
            n_vis=int(args.n_vis),
            seed=int(args.seed),
        )
        vmax_whu = _estimate_unc_vmax_for_dataset(
            data_root=args.whu_root,
            split=args.split,
            ckpt1=ckpt1,
            ckpt2=ckpt2,
            device=device,
            window=window,
            stride=stride,
            thr=float(args.thr),
            smooth_k=int(args.smooth_k),
            use_minarea=bool(args.use_minarea),
            min_area=int(args.min_area),
            n_vis=int(args.n_vis),
            seed=int(args.seed),
        )
        shared_unc_vmax = float(max(vmax_levir, vmax_whu))

    if args.paper_fig45:
        case_strategy = "paper" if str(args.case_strategy).lower() == "auto" else str(args.case_strategy).lower()
        levir_indices = _parse_csv_items(args.levir_indices)
        whu_indices = _parse_csv_items(args.whu_indices)
        # Figure 4: test on LEVIR-CD, compare LEVIR->LEVIR (in-domain) vs WHU->LEVIR (cross-domain).
        tgt_levir = "LEVIR-CD"
        if ckpt1_train == "levir":
            in_ckpt_levir, cross_ckpt_levir = ckpt1, ckpt2
            cross_train_levir = ckpt2_train
        elif ckpt2_train == "levir":
            in_ckpt_levir, cross_ckpt_levir = ckpt2, ckpt1
            cross_train_levir = ckpt1_train
        else:
            in_ckpt_levir, cross_ckpt_levir = ckpt1, ckpt2
            cross_train_levir = ckpt2_train

        _compare_one_dataset(
            dataset_name=tgt_levir,
            data_root=args.levir_root,
            split=args.split,
            ckpt1=in_ckpt_levir,
            ckpt2=cross_ckpt_levir,
            ckpt1_label=_arrow_label("levir", tgt_levir),
            ckpt2_label=_arrow_label(cross_train_levir, tgt_levir),
            out_dir=out_dir / "Figure4_LEVIR-CD",
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
            unc_vmax=shared_unc_vmax,
            unc_colorbar=bool(args.unc_colorbar),
            font_scale=float(args.font_scale),
            fig_title="Figure 4. Qualitative comparison on LEVIR-CD (in-domain vs cross-domain)",
            out_name="figure4.png",
            show_case_names=bool(args.show_case_names),
            show_row_tags=bool(args.show_row_tags),
            case_strategy=case_strategy,
            case_pool=int(args.case_pool),
            min_gt_pixels=int(args.min_gt_pixels),
            min_err_pixels=int(args.min_err_pixels),
            indices_override=levir_indices,
            dpi=int(args.dpi),
            auto_dpi=bool(args.auto_dpi),
            max_megapixels=float(args.max_megapixels),
            tight_bbox=bool(args.tight_bbox),
            panel_size=float(args.panel_size),
        )

        # Figure 5: test on WHU-CD, compare WHU->WHU (in-domain) vs LEVIR->WHU (cross-domain).
        tgt_whu = "WHU-CD"
        if ckpt1_train == "whu":
            in_ckpt_whu, cross_ckpt_whu = ckpt1, ckpt2
            cross_train_whu = ckpt2_train
        elif ckpt2_train == "whu":
            in_ckpt_whu, cross_ckpt_whu = ckpt2, ckpt1
            cross_train_whu = ckpt1_train
        else:
            in_ckpt_whu, cross_ckpt_whu = ckpt1, ckpt2
            cross_train_whu = ckpt2_train

        _compare_one_dataset(
            dataset_name=tgt_whu,
            data_root=args.whu_root,
            split=args.split,
            ckpt1=in_ckpt_whu,
            ckpt2=cross_ckpt_whu,
            ckpt1_label=_arrow_label("whu", tgt_whu),
            ckpt2_label=_arrow_label(cross_train_whu, tgt_whu),
            out_dir=out_dir / "Figure5_WHU-CD",
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
            unc_vmax=shared_unc_vmax,
            unc_colorbar=bool(args.unc_colorbar),
            font_scale=float(args.font_scale),
            fig_title="Figure 5. Qualitative comparison on WHU-CD (in-domain vs cross-domain)",
            out_name="figure5.png",
            show_case_names=bool(args.show_case_names),
            show_row_tags=bool(args.show_row_tags),
            case_strategy=case_strategy,
            case_pool=int(args.case_pool),
            min_gt_pixels=int(args.min_gt_pixels),
            min_err_pixels=int(args.min_err_pixels),
            indices_override=whu_indices,
            dpi=int(args.dpi),
            auto_dpi=bool(args.auto_dpi),
            max_megapixels=float(args.max_megapixels),
            tight_bbox=bool(args.tight_bbox),
            panel_size=float(args.panel_size),
        )
        return

    case_strategy_default = "uniform" if str(args.case_strategy).lower() == "auto" else str(args.case_strategy).lower()
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
        unc_vmax=shared_unc_vmax,
        unc_colorbar=bool(args.unc_colorbar),
        font_scale=float(args.font_scale),
        show_case_names=bool(args.show_case_names),
        case_strategy=case_strategy_default,
        case_pool=int(args.case_pool),
        min_gt_pixels=int(args.min_gt_pixels),
        min_err_pixels=int(args.min_err_pixels),
        dpi=int(args.dpi),
        auto_dpi=bool(args.auto_dpi),
        max_megapixels=float(args.max_megapixels),
        tight_bbox=bool(args.tight_bbox),
        panel_size=float(args.panel_size),
    )

    _compare_one_dataset(
        dataset_name="WHU-CD",
        data_root=args.whu_root,
        split=args.split,
        ckpt1=ckpt1,
        ckpt2=ckpt2,
        ckpt1_label=ckpt1_label,
        ckpt2_label=ckpt2_label,
        out_dir=out_dir / "WHU-CD",
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
        unc_vmax=shared_unc_vmax,
        unc_colorbar=bool(args.unc_colorbar),
        font_scale=float(args.font_scale),
        show_case_names=bool(args.show_case_names),
        case_strategy=case_strategy_default,
        case_pool=int(args.case_pool),
        min_gt_pixels=int(args.min_gt_pixels),
        min_err_pixels=int(args.min_err_pixels),
        dpi=int(args.dpi),
        auto_dpi=bool(args.auto_dpi),
        max_megapixels=float(args.max_megapixels),
        tight_bbox=bool(args.tight_bbox),
        panel_size=float(args.panel_size),
    )


if __name__ == "__main__":
    main()
