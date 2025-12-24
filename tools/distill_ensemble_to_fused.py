"""
Ensemble-to-fused distillation fine-tune.

Goal:
  - Keep inference single-head (fused pred) for sharp in-domain performance
  - Transfer some cross-domain robustness from an ensemble teacher

Typical usage (PowerShell):
  python tools/distill_ensemble_to_fused.py `
    --teacher_ckpt outputs/dino_head_cd/best.pt `
    --student_ckpt outputs/dino_head_cd/best.pt `
    --data_root data/WHUCD `
    --out_dir outputs/dino_head_cd_distill `
    --epochs 20 `
    --tune fused_classifier `
    --teacher_mode mean_logit --teacher_indices 3,4 `
    --alpha_sup 0.5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dino_head_core import (
    HeadCfg,
    build_dataloaders,
    dice_loss_with_logits,
    ensure_dir,
    evaluate,
    seed_everything,
)
from models.dinov2_head import DinoSiameseHead


def _parse_int_list(s: Optional[str]) -> Optional[List[int]]:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out or None


def _ensemble_teacher_prob(
    logits_all: torch.Tensor, mode: str, indices: Optional[List[int]], weights: Optional[List[float]], temp: float
) -> torch.Tensor:
    """
    logits_all: [K,B,1,H,W]
    returns prob: [B,1,H,W]
    """
    if indices is not None:
        idx = torch.as_tensor(indices, device=logits_all.device, dtype=torch.long)
        logits_all = logits_all.index_select(0, idx)
    mode = str(mode)
    temp = float(max(1e-6, temp))

    if mode == "mean_prob":
        return torch.sigmoid(logits_all / temp).mean(dim=0)
    if mode == "mean_logit":
        return torch.sigmoid(logits_all.mean(dim=0) / temp)
    if mode == "weighted_logit":
        if weights is None:
            raise ValueError("teacher_mode=weighted_logit requires --teacher_weights")
        w = torch.as_tensor(weights, device=logits_all.device, dtype=logits_all.dtype)
        if w.ndim != 1 or w.numel() != logits_all.shape[0]:
            raise ValueError(f"teacher_weights must match selected heads K={logits_all.shape[0]}, got {w.numel()}")
        w = w / w.sum().clamp_min(1e-12)
        w = w.view(-1, 1, 1, 1, 1)
        return torch.sigmoid(((w * logits_all).sum(dim=0)) / temp)
    raise ValueError(f"Unknown teacher_mode: {mode}")


def _set_trainable(student: torch.nn.Module, tune: str) -> Dict[str, int]:
    """
    Freeze everything then selectively unfreeze.
    Returns stats dict: total/trainable parameter counts.
    """
    for p in student.parameters():
        p.requires_grad = False

    tune = str(tune).lower().strip()

    def _unfreeze(module: Optional[torch.nn.Module]):
        if module is None:
            return
        for p in module.parameters():
            p.requires_grad = True

    if tune == "fused_classifier":
        _unfreeze(getattr(student, "fused_classifier", None))
    elif tune == "fused_classifier+fused_head":
        _unfreeze(getattr(student, "fused_classifier", None))
        _unfreeze(getattr(student, "fused_head", None))
    elif tune == "ensemble_heads_only":
        _unfreeze(getattr(student, "layer_heads", None))
        _unfreeze(getattr(student, "fused_decoder", None))
        _unfreeze(getattr(student, "fused_head", None))
        _unfreeze(getattr(student, "fused_classifier", None))
    else:
        raise ValueError("Unknown --tune. Use fused_classifier | fused_classifier+fused_head | ensemble_heads_only")

    total = sum(p.numel() for p in student.parameters())
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    return {"total_params": int(total), "trainable_params": int(trainable)}


def _build_model_from_ckpt(ckpt: Dict[str, Any], device: str) -> DinoSiameseHead:
    cfg = ckpt.get("cfg") if isinstance(ckpt, dict) else None
    if not isinstance(cfg, dict):
        cfg = {}
    model = DinoSiameseHead(
        dino_name=cfg.get("dino_name", "facebook/dinov3-vitb16-pretrain-lvd1689m"),
        use_whiten=cfg.get("use_whiten", False),
        use_domain_adv=cfg.get("use_domain_adv", False),
        domain_hidden=cfg.get("domain_hidden", 256),
        domain_grl=cfg.get("domain_grl", 1.0),
        use_style_norm=cfg.get("use_style_norm", False),
        proto_path=cfg.get("proto_path", None) or None,
        proto_weight=cfg.get("proto_weight", 0.0),
        boundary_dim=cfg.get("boundary_dim", 0),
        use_layer_ensemble=cfg.get("use_layer_ensemble", False),
        layer_head_ch=cfg.get("layer_head_ch", 128),
    ).to(device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    return model


def parse_args():
    base = HeadCfg()
    p = argparse.ArgumentParser(description="Ensemble-to-fused distillation fine-tune")
    p.add_argument("--teacher_ckpt", type=str, required=True)
    p.add_argument("--student_ckpt", type=str, required=True, help="Initialization ckpt for student (usually same as teacher)")
    p.add_argument("--data_root", type=str, required=True, help="Source domain data_root used for supervised + distill training")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=base.seed)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=base.batch_size)
    p.add_argument("--num_workers", type=int, default=base.num_workers)
    p.add_argument("--crop_size", type=int, default=base.crop_size)
    p.add_argument("--eval_crop", type=int, default=base.eval_crop)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_accum", type=int, default=base.grad_accum)
    p.add_argument("--alpha_sup", type=float, default=0.5, help="Weight for supervised loss; (1-alpha) for distill loss")
    p.add_argument("--bce_weight", type=float, default=base.bce_weight)
    p.add_argument("--dice_weight", type=float, default=base.dice_weight)
    p.add_argument(
        "--tune",
        type=str,
        default="fused_classifier",
        choices=["fused_classifier", "fused_classifier+fused_head", "ensemble_heads_only"],
    )
    p.add_argument(
        "--teacher_mode",
        type=str,
        default="mean_logit",
        choices=["mean_logit", "mean_prob", "weighted_logit"],
    )
    p.add_argument(
        "--teacher_indices",
        type=str,
        default="3,4",
        help="Comma-separated indices into logits_all (e.g., '3,4' = deepest layer head + fused head)",
    )
    p.add_argument(
        "--teacher_weights",
        type=str,
        default=None,
        help="Comma-separated weights for weighted_logit teacher (must match teacher_indices after selection).",
    )
    p.add_argument("--teacher_temp", type=float, default=1.0, help="Temperature for teacher probability")
    p.add_argument(
        "--distill_loss",
        type=str,
        default="bce_soft",
        choices=["bce_soft", "kl_teacher_to_student"],
        help="bce_soft uses BCEWithLogits(student_logit, teacher_prob). kl_teacher_to_student uses KL(teacher||student) over Bernoulli.",
    )
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--save_best", action="store_true", default=True)
    p.add_argument("--thr_mode", type=str, choices=["fixed", "topk", "otsu"], default=base.thr_mode)
    p.add_argument("--thr", type=float, default=base.thr)
    p.add_argument("--topk", type=float, default=base.topk)
    p.add_argument("--smooth_k", type=int, default=base.smooth_k)
    p.add_argument("--use_minarea", action="store_true", default=base.use_minarea)
    p.add_argument("--min_area", type=int, default=base.min_area)
    p.add_argument("--full_eval", dest="full_eval", action="store_true")
    p.add_argument("--no_full_eval", dest="full_eval", action="store_false")
    p.set_defaults(full_eval=True)
    args = p.parse_args()
    args.teacher_indices = _parse_int_list(args.teacher_indices) or [3, 4]
    if args.teacher_weights:
        args.teacher_weights = [float(x) for x in str(args.teacher_weights).split(",") if str(x).strip() != ""]
    else:
        args.teacher_weights = None
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def main():
    args = parse_args()
    seed_everything(args.seed)
    ensure_dir(args.out_dir)
    device = args.device

    cfg = HeadCfg(
        data_root=args.data_root,
        out_dir=args.out_dir,
        seed=args.seed,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        crop_size=args.crop_size,
        eval_crop=args.eval_crop,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_accum=args.grad_accum,
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        full_eval=args.full_eval,
        thr_mode=args.thr_mode,
        thr=args.thr,
        topk=args.topk,
        smooth_k=args.smooth_k,
        use_minarea=args.use_minarea,
        min_area=args.min_area,
        use_ensemble_pred=False,
    )

    train_loader, val_loader, _ = build_dataloaders(cfg)

    teacher_ckpt = torch.load(args.teacher_ckpt, map_location=device)
    student_ckpt = torch.load(args.student_ckpt, map_location=device)
    teacher = _build_model_from_ckpt(teacher_ckpt, device=device)
    student = _build_model_from_ckpt(student_ckpt, device=device)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    stats = _set_trainable(student, tune=args.tune)
    print(f"[Distill] device={device} seed={args.seed}")
    print(f"[Distill] teacher_ckpt={args.teacher_ckpt}")
    print(f"[Distill] student_ckpt={args.student_ckpt}")
    print(f"[Distill] data_root={args.data_root} out_dir={args.out_dir}")
    print(f"[Distill] epochs={args.epochs} batch={args.batch_size} crop={args.crop_size} grad_accum={args.grad_accum}")
    print(f"[Distill] tune={args.tune} trainable_params={stats['trainable_params']}/{stats['total_params']}")
    print(f"[Distill] teacher_mode={args.teacher_mode} indices={args.teacher_indices} temp={args.teacher_temp}")
    print(f"[Distill] alpha_sup={args.alpha_sup} distill_loss={args.distill_loss}")

    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.startswith("cuda") and torch.cuda.is_available()))

    metrics_path = os.path.join(cfg.out_dir, "metrics.jsonl")
    best_path = os.path.join(cfg.out_dir, "best.pt")
    last_path = os.path.join(cfg.out_dir, "last.pt")
    best_f1 = -1.0

    bce = torch.nn.BCEWithLogitsLoss()
    alpha = float(min(1.0, max(0.0, args.alpha_sup)))

    for ep in range(1, cfg.epochs + 1):
        student.train()
        opt.zero_grad(set_to_none=True)
        t0 = time.time()

        for it, batch in enumerate(train_loader, 1):
            img_a = batch["img_a"].to(device, non_blocking=True)
            img_b = batch["img_b"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            if y.ndim == 3:
                y = y.unsqueeze(1)
            elif y.ndim == 4 and y.shape[1] != 1:
                y = y[:, :1]
            y = (y > 0).float()

            with torch.no_grad():
                out_t = teacher(img_a, img_b)
                if not isinstance(out_t, dict) or out_t.get("logits_all") is None:
                    raise RuntimeError("Teacher model has no logits_all; ensure it was trained with --use_layer_ensemble.")
                p_t = _ensemble_teacher_prob(
                    logits_all=out_t["logits_all"],
                    mode=args.teacher_mode,
                    indices=args.teacher_indices,
                    weights=args.teacher_weights,
                    temp=args.teacher_temp,
                ).detach()

            with torch.cuda.amp.autocast(enabled=(device.startswith("cuda") and torch.cuda.is_available())):
                out_s = student(img_a, img_b)
                if not isinstance(out_s, dict) or out_s.get("pred") is None:
                    raise RuntimeError("Student forward expected dict with 'pred'.")
                z_s = out_s["pred"]  # [B,1,H,W] (fused logit when use_layer_ensemble=True)

                sup = cfg.bce_weight * bce(z_s, y) + cfg.dice_weight * dice_loss_with_logits(z_s, y)
                if args.distill_loss == "bce_soft":
                    dist = bce(z_s, p_t)
                else:
                    # KL(teacher || student) for Bernoulli at each pixel
                    eps = 1e-6
                    p_s = torch.sigmoid(z_s).clamp(eps, 1.0 - eps)
                    p_t2 = p_t.clamp(eps, 1.0 - eps)
                    dist = (p_t2 * (p_t2 / p_s).log() + (1.0 - p_t2) * ((1.0 - p_t2) / (1.0 - p_s)).log()).mean()

                loss = (alpha * sup + (1.0 - alpha) * dist) / float(cfg.grad_accum)

            scaler.scale(loss).backward()

            if it % cfg.grad_accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            if it % max(1, (len(train_loader) // 3)) == 0:
                dt = time.time() - t0
                print(f"  [ep {ep:03d}] it={it}/{len(train_loader)} loss={loss.item()*cfg.grad_accum:.4f} time={dt:.1f}s")
                t0 = time.time()

        # save last
        torch.save(
            {
                "epoch": ep,
                "model": student.state_dict(),
                "optimizer": opt.state_dict(),
                "best_f1": best_f1,
                "cfg": asdict(cfg),
                "distill": {
                    "teacher_ckpt": args.teacher_ckpt,
                    "student_ckpt": args.student_ckpt,
                    "teacher_mode": args.teacher_mode,
                    "teacher_indices": args.teacher_indices,
                    "teacher_weights": args.teacher_weights,
                    "teacher_temp": args.teacher_temp,
                    "alpha_sup": alpha,
                    "distill_loss": args.distill_loss,
                    "tune": args.tune,
                },
            },
            last_path,
        )

        do_eval = (ep % max(1, int(args.eval_every)) == 0) or (ep == cfg.epochs)
        if do_eval:
            val_m = evaluate(
                model=student,
                loader=val_loader,
                device=device,
                thr_mode=cfg.thr_mode,
                thr=cfg.thr,
                topk=cfg.topk,
                smooth_k=cfg.smooth_k,
                use_minarea=cfg.use_minarea,
                min_area=cfg.min_area,
                print_every=0,
                window=None if cfg.full_eval else cfg.eval_window,
                stride=None if cfg.full_eval else cfg.eval_stride,
                use_ensemble=False,
                ensemble_cfg=None,
            )
            print(
                f"[Epoch {ep:03d}] VAL P={val_m['precision']:.4f} R={val_m['recall']:.4f} "
                f"F1={val_m['f1']:.4f} IoU={val_m['iou']:.4f}"
            )
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"epoch": ep, "split": "val", **val_m, "time": time.time()}, ensure_ascii=False) + "\n")
            if args.save_best and val_m["f1"] > best_f1:
                best_f1 = float(val_m["f1"])
                torch.save(
                    {
                        "epoch": ep,
                        "model": student.state_dict(),
                        "optimizer": opt.state_dict(),
                        "best_f1": best_f1,
                        "cfg": asdict(cfg),
                    },
                    best_path,
                )
                print(f"  -> Saved BEST {best_path} (best_f1={best_f1:.4f})")

    print(f"Done. best_f1={best_f1:.4f} out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
