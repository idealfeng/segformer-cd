"""
Evaluate a DINO change-detection checkpoint over multiple fixed thresholds.

This is intended for exploratory target-domain analysis. For strict zero-shot
reporting, use the pre-declared threshold (usually 0.5) and treat other target
thresholds as oracle / diagnostic results.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Any
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dino_head_core import (
    HeadCfg,
    DinoFrozenA0Head,
    DinoSiameseHead,
    build_dataloaders,
    compute_metrics_from_cm,
    confusion_update,
    filter_small_cc,
    seed_everything,
    tta_inference_prob,
)


def _csv_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def _csv_ints(s: str | None) -> list[int] | None:
    if not s:
        return None
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fixed-threshold sweep for DINO CD checkpoints.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Dataset split to evaluate.")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--full_eval", action="store_true")
    p.add_argument("--eval_crop", type=int, default=256)
    p.add_argument("--window", type=int, default=256)
    p.add_argument("--stride", type=int, default=256)
    p.add_argument("--thresholds", type=str, default="0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    p.add_argument("--smooth_k", type=int, default=3)
    p.add_argument("--use_minarea", action="store_true")
    p.add_argument("--min_area", type=int, default=256)
    p.add_argument("--morph", type=str, default="none", choices=["none", "erode", "dilate", "open", "close"])
    p.add_argument("--morph_kernel", type=int, default=3)
    p.add_argument("--morph_iters", type=int, default=1)
    p.add_argument("--tta", type=str, default="none", choices=["none", "flip", "d4"])
    p.add_argument("--use_ensemble_pred", action="store_true")
    p.add_argument("--ensemble_strategy", type=str, default="mean_logit", choices=["mean_prob", "mean_logit"])
    p.add_argument("--ensemble_indices", type=str, default=None)
    p.add_argument("--max_samples", type=int, default=0, help="Optional cap for a quick subset run. 0 means all samples.")
    p.add_argument("--save_every", type=int, default=50, help="Write partial results every N processed samples.")
    return p.parse_args()


def _apply_morph(mask: np.ndarray, mode: str, kernel_size: int, iters: int) -> np.ndarray:
    mode = str(mode).lower()
    if mode in ("none", "off", "0", "false"):
        return mask
    try:
        import cv2
    except Exception:
        return mask
    kernel_size = int(max(1, kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    iters = int(max(1, iters))
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    m = (mask > 0).astype(np.uint8)
    if mode == "erode":
        out = cv2.erode(m, kernel, iterations=iters)
    elif mode == "dilate":
        out = cv2.dilate(m, kernel, iterations=iters)
    elif mode == "open":
        out = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=iters)
    elif mode == "close":
        out = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=iters)
    else:
        out = m
    return (out > 0).astype(np.uint8)


def _build_model_from_ckpt(ckpt: dict, cfg: HeadCfg, device: str) -> torch.nn.Module:
    load_cfg = ckpt.get("cfg")
    if isinstance(load_cfg, dict):
        cfg.arch = load_cfg.get("arch", getattr(cfg, "arch", "dlv"))
        cfg.a0_layer = load_cfg.get("a0_layer", getattr(cfg, "a0_layer", 12))
        cfg.use_layer_ensemble = load_cfg.get("use_layer_ensemble", cfg.use_layer_ensemble)
        cfg.layer_head_ch = load_cfg.get("layer_head_ch", cfg.layer_head_ch)
        if load_cfg.get("selected_layers") is not None:
            cfg.selected_layers = tuple(int(x) for x in load_cfg["selected_layers"])

    arch = load_cfg.get("arch", getattr(cfg, "arch", "dlv")) if isinstance(load_cfg, dict) else "dlv"
    if arch == "a0":
        model = DinoFrozenA0Head(
            dino_name=load_cfg.get("dino_name", cfg.dino_name) if isinstance(load_cfg, dict) else cfg.dino_name,
            layer=load_cfg.get("a0_layer", cfg.a0_layer) if isinstance(load_cfg, dict) else cfg.a0_layer,
            use_whiten=load_cfg.get("use_whiten", cfg.use_whiten) if isinstance(load_cfg, dict) else cfg.use_whiten,
        ).to(device)
    else:
        model = DinoSiameseHead(
            dino_name=load_cfg.get("dino_name", cfg.dino_name) if isinstance(load_cfg, dict) else cfg.dino_name,
            selected_layers=cfg.selected_layers,
            use_whiten=load_cfg.get("use_whiten", cfg.use_whiten) if isinstance(load_cfg, dict) else cfg.use_whiten,
            use_domain_adv=load_cfg.get("use_domain_adv", cfg.use_domain_adv) if isinstance(load_cfg, dict) else cfg.use_domain_adv,
            domain_hidden=load_cfg.get("domain_hidden", cfg.domain_hidden) if isinstance(load_cfg, dict) else cfg.domain_hidden,
            domain_grl=load_cfg.get("domain_grl", cfg.domain_grl) if isinstance(load_cfg, dict) else cfg.domain_grl,
            use_style_norm=load_cfg.get("use_style_norm", cfg.use_style_norm) if isinstance(load_cfg, dict) else cfg.use_style_norm,
            proto_path=load_cfg.get("proto_path", cfg.proto_path) if isinstance(load_cfg, dict) else cfg.proto_path,
            proto_weight=load_cfg.get("proto_weight", cfg.proto_weight) if isinstance(load_cfg, dict) else cfg.proto_weight,
            boundary_dim=load_cfg.get("boundary_dim", cfg.boundary_dim) if isinstance(load_cfg, dict) else cfg.boundary_dim,
            use_layer_ensemble=load_cfg.get("use_layer_ensemble", cfg.use_layer_ensemble) if isinstance(load_cfg, dict) else cfg.use_layer_ensemble,
            layer_head_ch=load_cfg.get("layer_head_ch", cfg.layer_head_ch) if isinstance(load_cfg, dict) else cfg.layer_head_ch,
        ).to(device)

    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()
    return model


def _make_payload(
    *,
    args: argparse.Namespace,
    cfg: HeadCfg,
    thresholds: list[float],
    cms: dict[float, dict[str, int]],
    processed: int,
    ensemble_cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    results = []
    for thr in thresholds:
        m = compute_metrics_from_cm(cms[thr])
        m["thr"] = float(thr)
        results.append(m)
    results.sort(key=lambda x: x["thr"])
    best = max(results, key=lambda x: x["f1"]) if results else None
    return {
        "checkpoint": args.checkpoint,
        "data_root": args.data_root,
        "split": args.split,
        "processed_samples": int(processed),
        "max_samples": int(args.max_samples),
        "cfg": asdict(cfg),
        "thresholds": thresholds,
        "tta": args.tta,
        "morph": {"mode": args.morph, "kernel": int(args.morph_kernel), "iters": int(args.morph_iters)},
        "use_ensemble_pred": bool(args.use_ensemble_pred),
        "ensemble_cfg": ensemble_cfg,
        "results": results,
        "best_by_f1": best,
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    thresholds = _csv_floats(args.thresholds)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    seed_everything(42)
    cfg = HeadCfg(
        data_root=args.data_root,
        out_dir=str(out_dir),
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        full_eval=args.full_eval,
        eval_crop=args.eval_crop,
        smooth_k=args.smooth_k,
        use_minarea=args.use_minarea,
        min_area=args.min_area,
    )
    _, val_loader, test_loader = build_dataloaders(cfg, require_train=False, require_val=(args.split == "val"))
    if args.split == "val":
        if val_loader is None:
            raise RuntimeError(f"No val split found under {args.data_root}")
        eval_loader = val_loader
    else:
        eval_loader = test_loader

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = _build_model_from_ckpt(ckpt, cfg, device)

    ensemble_cfg = None
    if args.use_ensemble_pred:
        indices = _csv_ints(args.ensemble_indices)
        ensemble_cfg = {"mode": args.ensemble_strategy}
        if indices is not None:
            ensemble_cfg["indices"] = indices

    cms = {thr: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for thr in thresholds}
    window = args.window if args.full_eval and args.window > 0 else None
    stride = args.stride if args.full_eval and args.stride > 0 else None

    total = len(eval_loader)
    if args.max_samples and args.max_samples > 0:
        total = min(total, int(args.max_samples))
    processed = 0
    partial_path = out_dir / "threshold_sweep_partial.json"
    final_path = out_dir / "threshold_sweep_results.json"

    for batch in tqdm(eval_loader, desc=f"eval-{args.split}", total=total):
        if args.max_samples and processed >= int(args.max_samples):
            break
        img_a = batch["img_a"].to(device)
        img_b = batch["img_b"].to(device)
        gt = batch["label"]
        prob = tta_inference_prob(
            model=model,
            img_a=img_a,
            img_b=img_b,
            device=device,
            window=window,
            stride=stride,
            use_ensemble=bool(args.use_ensemble_pred),
            ensemble_cfg=ensemble_cfg,
            tta_mode=args.tta,
        )
        if args.smooth_k and args.smooth_k > 1:
            pad = args.smooth_k // 2
            prob = F.avg_pool2d(prob, kernel_size=args.smooth_k, stride=1, padding=pad)

        for bi in range(prob.shape[0]):
            prob_np = prob[bi, 0].detach().float().cpu().numpy()
            gt_np = gt[bi].detach().cpu().numpy().astype(np.uint8)
            if gt_np.ndim == 3:
                gt_np = gt_np.squeeze(0)
            gt_t = torch.from_numpy((gt_np > 0).astype(np.uint8))
            for thr in thresholds:
                pred_np = (prob_np > float(thr)).astype(np.uint8)
                pred_np = _apply_morph(pred_np, args.morph, args.morph_kernel, args.morph_iters)
                if args.use_minarea:
                    pred_np = filter_small_cc(pred_np, min_area=args.min_area)
                pred_t = torch.from_numpy(pred_np)
                confusion_update(pred_t, gt_t, cms[thr])
        processed += int(prob.shape[0])
        if args.save_every > 0 and processed % int(args.save_every) == 0:
            payload = _make_payload(
                args=args,
                cfg=cfg,
                thresholds=thresholds,
                cms=cms,
                processed=processed,
                ensemble_cfg=ensemble_cfg,
            )
            partial_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    payload = _make_payload(
        args=args,
        cfg=cfg,
        thresholds=thresholds,
        cms=cms,
        processed=processed,
        ensemble_cfg=ensemble_cfg,
    )
    final_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    partial_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"processed_samples": processed, "best_by_f1": payload["best_by_f1"], "out": str(final_path)}, indent=2))


if __name__ == "__main__":
    main()
