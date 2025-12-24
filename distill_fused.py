"""
Distill multi-head ensemble back into the fused head (student) for unified in/out-domain performance.

Workflow:
1) Start from a trained multi-head checkpoint (use_layer_ensemble=True, logits_all available).
2) Freeze most parameters; only fine-tune fused head branch.
3) Use layer-head ensemble (teacher) to provide soft targets; train fused head (student) with mixed hard+soft loss.

Example:
python distill_fused.py `
  --checkpoint outputs/dino_head_cd/best.pt `
  --data_root data/LEVIR-CD `
  --out_dir outputs/dino_head_cd_distill `
  --epochs 30 --lr 1e-4 `
  --alpha 0.5 --teacher_topk 2 --teacher_mode mean_logit
"""

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from dino_head_core import (
    HeadCfg,
    DinoSiameseHead,
    build_dataloaders,
    seed_everything,
    ensure_dir,
    evaluate,
    threshold_map,
    filter_small_cc,
    confusion_update,
    compute_metrics_from_cm,
    dice_loss_with_logits,
    build_scheduler,
)


def _parse_int_list(s: str | None) -> list[int] | None:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    try:
        return [int(x) for x in s.split(",") if str(x).strip() != ""]
    except Exception as e:
        raise ValueError(f"Invalid int list: {s}") from e


def parse_args():
    base = HeadCfg()
    parser = argparse.ArgumentParser(description="Distill ensemble teacher into fused head (student)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to multi-head checkpoint (best.pt/last.pt)")
    parser.add_argument("--data_root", type=str, default=base.data_root)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=base.device, help="cuda|cpu|auto")
    parser.add_argument("--seed", type=int, default=base.seed)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=base.batch_size)
    parser.add_argument("--num_workers", type=int, default=base.num_workers)
    parser.add_argument("--crop_size", type=int, default=base.crop_size)
    parser.add_argument("--eval_crop", type=int, default=base.eval_crop)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_accum", type=int, default=base.grad_accum)
    parser.add_argument("--log_every", type=int, default=base.log_every)
    parser.add_argument("--eval_every", type=int, default=1)

    # distillation knobs
    parser.add_argument("--alpha", type=float, default=0.5, help="Hard-label weight; (1-alpha) is soft distill weight")
    parser.add_argument(
        "--teacher_mode",
        type=str,
        default="mean_logit",
        choices=["mean_logit", "weighted_logit"],
        help="How to combine teacher layer heads (logit-space)",
    )
    parser.add_argument(
        "--teacher_indices",
        type=str,
        default=None,
        help="Comma-separated layer-head indices for teacher (e.g. '0,2'); excludes fused automatically. If not set, picks top-k by VAL F1.",
    )
    parser.add_argument("--teacher_topk", type=int, default=2, help="If --teacher_indices not set: pick top-k layer heads by VAL F1")
    parser.add_argument(
        "--teacher_weights",
        type=str,
        default=None,
        help="Comma-separated weights for teacher indices (same length). Only used when teacher_mode=weighted_logit.",
    )

    # evaluation / threshold
    parser.add_argument("--thr_mode", type=str, choices=["fixed", "topk", "otsu"], default=base.thr_mode)
    parser.add_argument("--thr", type=float, default=base.thr)
    parser.add_argument("--topk", type=float, default=base.topk)
    parser.add_argument("--smooth_k", type=int, default=base.smooth_k)
    parser.add_argument("--use_minarea", action="store_true", default=base.use_minarea)
    parser.add_argument("--min_area", type=int, default=base.min_area)
    parser.add_argument("--full_eval", dest="full_eval", action="store_true")
    parser.add_argument("--no_full_eval", dest="full_eval", action="store_false")
    parser.set_defaults(full_eval=base.full_eval)

    # freezing policy
    parser.add_argument(
        "--train_fused",
        type=str,
        default="branch",
        choices=["classifier", "branch"],
        help="'classifier': only fused_classifier; 'branch': fused_decoder+fused_head+fused_classifier",
    )

    args = parser.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.teacher_indices = _parse_int_list(args.teacher_indices)
    if args.teacher_weights is not None:
        w = [float(x) for x in str(args.teacher_weights).split(",") if str(x).strip() != ""]
        args.teacher_weights = w
    alpha = float(args.alpha)
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("--alpha must be in [0,1]")
    return args


def _set_trainable_for_distill(model: nn.Module, train_fused: str):
    for p in model.parameters():
        p.requires_grad = False
    if not hasattr(model, "fused_classifier") or model.fused_classifier is None:
        raise RuntimeError("Model has no fused_classifier; ensure checkpoint was trained with use_layer_ensemble.")
    if train_fused == "classifier":
        for p in model.fused_classifier.parameters():
            p.requires_grad = True
        return
    # branch
    for name in ("fused_decoder", "fused_head", "fused_classifier"):
        m = getattr(model, name, None)
        if m is None:
            raise RuntimeError(f"Model missing {name}; ensure checkpoint was trained with use_layer_ensemble.")
        for p in m.parameters():
            p.requires_grad = True


@torch.no_grad()
def _score_heads_on_val(
    model: nn.Module,
    loader,
    device: str,
    thr_mode: str,
    thr: float,
    topk: float,
    smooth_k: int,
    use_minarea: bool,
    min_area: int,
):
    model.eval()
    cms = None
    for batch in loader:
        img_a = batch["img_a"].to(device, non_blocking=True)
        img_b = batch["img_b"].to(device, non_blocking=True)
        gt = batch["label"]
        out = model(img_a, img_b)
        if not isinstance(out, dict) or out.get("logits_all") is None:
            raise RuntimeError("Model output has no logits_all; enable use_layer_ensemble during training/eval.")
        probs_all = torch.sigmoid(out["logits_all"])  # [K,B,1,H,W]
        K, B, _, H, W = probs_all.shape
        if cms is None:
            cms = [{"TP": 0, "FP": 0, "FN": 0, "TN": 0} for _ in range(K)]

        if smooth_k and smooth_k > 1:
            pad = smooth_k // 2
            probs_all = probs_all.view(K * B, 1, H, W)
            probs_all = F.avg_pool2d(probs_all, kernel_size=smooth_k, stride=1, padding=pad)
            probs_all = probs_all.view(K, B, 1, H, W)

        for bi in range(B):
            gt_np = gt[bi].detach().cpu().numpy().astype("uint8")
            if gt_np.ndim == 3:
                gt_np = gt_np.squeeze(0)
            gt_t = torch.from_numpy((gt_np > 0).astype("uint8"))

            for k in range(K):
                prob_np = probs_all[k, bi, 0].detach().float().cpu().numpy()
                pred_np, _ = threshold_map(prob_np, thr_mode, thr, topk)
                if use_minarea:
                    pred_np = filter_small_cc(pred_np, min_area=min_area)
                pred_t = torch.from_numpy(pred_np.astype("uint8"))
                confusion_update(pred_t, gt_t, cms[k])
    if cms is None:
        raise RuntimeError("Empty val loader; cannot score heads.")
    return [compute_metrics_from_cm(cm) for cm in cms]


def _teacher_from_logits_all(
    logits_all: torch.Tensor,
    indices: list[int],
    mode: str,
    weights: list[float] | None,
) -> torch.Tensor:
    """
    logits_all: [K,B,1,H,W]
    indices: selected teacher head indices (must not include fused)
    returns teacher_prob: [B,1,H,W] (detached outside)
    """
    idx = torch.as_tensor(indices, device=logits_all.device, dtype=torch.long)
    sel = logits_all.index_select(0, idx)  # [K',B,1,H,W]
    if mode == "mean_logit":
        t_logit = sel.mean(dim=0)
        return torch.sigmoid(t_logit)
    if mode == "weighted_logit":
        if weights is None:
            raise ValueError("teacher_mode=weighted_logit requires --teacher_weights")
        w = torch.as_tensor(weights, device=logits_all.device, dtype=sel.dtype)
        if w.ndim != 1 or w.numel() != sel.shape[0]:
            raise ValueError(f"teacher_weights must match indices length, got {w.numel()} vs {sel.shape[0]}")
        w = (w / w.sum().clamp_min(1e-12)).view(-1, 1, 1, 1, 1)
        t_logit = (w * sel).sum(dim=0)
        return torch.sigmoid(t_logit)
    raise ValueError(f"Unknown teacher_mode: {mode}")


def train_one_epoch_distill(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: str,
    teacher_indices: list[int],
    teacher_mode: str,
    teacher_weights: list[float] | None,
    alpha: float,
    grad_accum: int,
    log_every: int,
):
    model.train()
    bce_hard = nn.BCEWithLogitsLoss()
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()
    for it, batch in enumerate(loader, 1):
        img_a = batch["img_a"].to(device, non_blocking=True)
        img_b = batch["img_b"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        if label.ndim == 3:
            label = label.unsqueeze(1)
        elif label.ndim == 4 and label.shape[1] != 1:
            label = label[:, :1]
        label = label.float()

        with torch.cuda.amp.autocast(enabled=(device.startswith("cuda") and torch.cuda.is_available())):
            out = model(img_a, img_b)
            if not isinstance(out, dict) or out.get("logits_all") is None:
                raise RuntimeError("Model output has no logits_all; enable use_layer_ensemble during training/eval.")
            logits_all = out["logits_all"]  # [K,B,1,H,W]
            student_logit = logits_all[-1]  # fused
            teacher_prob = _teacher_from_logits_all(
                logits_all=logits_all,
                indices=teacher_indices,
                mode=teacher_mode,
                weights=teacher_weights,
            ).detach()

            hard = bce_hard(student_logit, label) + dice_loss_with_logits(student_logit, label)
            soft = F.binary_cross_entropy_with_logits(student_logit, teacher_prob)
            loss = (alpha * hard + (1.0 - alpha) * soft) / max(1, grad_accum)

        scaler.scale(loss).backward()
        if it % grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        if log_every and (it % log_every == 0):
            dt = time.time() - t0
            print(f"  [distill] iter={it}/{len(loader)} loss={loss.item()*max(1,grad_accum):.4f} time={dt:.1f}s")
            t0 = time.time()


def main():
    args = parse_args()
    seed_everything(args.seed)
    ensure_dir(args.out_dir)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable, fallback to CPU")
        device = "cpu"

    ckpt = torch.load(args.checkpoint, map_location=device)
    load_cfg = ckpt.get("cfg")
    if not isinstance(load_cfg, dict):
        raise RuntimeError("Checkpoint missing cfg dict; please use checkpoints saved by train_dino_head.py.")
    if not bool(load_cfg.get("use_layer_ensemble", False)):
        raise RuntimeError("Checkpoint cfg.use_layer_ensemble is False; distillation requires logits_all.")

    cfg = HeadCfg(
        data_root=args.data_root,
        out_dir=args.out_dir,
        seed=args.seed,
        device=device,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        crop_size=int(args.crop_size),
        eval_crop=int(args.eval_crop),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        grad_accum=int(args.grad_accum),
        log_every=int(args.log_every),
        thr_mode=args.thr_mode,
        thr=float(args.thr),
        topk=float(args.topk),
        smooth_k=int(args.smooth_k),
        use_minarea=bool(args.use_minarea),
        min_area=int(args.min_area),
        full_eval=bool(args.full_eval),
        # model
        dino_name=load_cfg.get("dino_name", "facebook/dinov3-vitb16-pretrain-lvd1689m"),
        use_whiten=bool(load_cfg.get("use_whiten", False)),
        use_domain_adv=bool(load_cfg.get("use_domain_adv", False)),
        domain_hidden=int(load_cfg.get("domain_hidden", 256)),
        domain_grl=float(load_cfg.get("domain_grl", 1.0)),
        use_style_norm=bool(load_cfg.get("use_style_norm", False)),
        proto_path=str(load_cfg.get("proto_path", "")),
        proto_weight=float(load_cfg.get("proto_weight", 0.0)),
        boundary_dim=int(load_cfg.get("boundary_dim", 0)),
        use_layer_ensemble=bool(load_cfg.get("use_layer_ensemble", True)),
        layer_head_ch=int(load_cfg.get("layer_head_ch", 128)),
    )
    with open(os.path.join(cfg.out_dir, "distill_config.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "cfg": asdict(cfg)}, f, ensure_ascii=False, indent=2)

    train_loader, val_loader, _ = build_dataloaders(cfg)

    model = DinoSiameseHead(
        dino_name=cfg.dino_name,
        use_whiten=cfg.use_whiten,
        use_domain_adv=cfg.use_domain_adv,
        domain_hidden=cfg.domain_hidden,
        domain_grl=cfg.domain_grl,
        use_style_norm=cfg.use_style_norm,
        proto_path=cfg.proto_path,
        proto_weight=cfg.proto_weight,
        boundary_dim=cfg.boundary_dim,
        use_layer_ensemble=cfg.use_layer_ensemble,
        layer_head_ch=cfg.layer_head_ch,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint from {args.checkpoint}")

    # determine teacher indices (layer heads only)
    with torch.no_grad():
        dummy = next(iter(val_loader))
        img_a = dummy["img_a"].to(device)
        img_b = dummy["img_b"].to(device)
        out = model(img_a, img_b)
        if not isinstance(out, dict) or out.get("logits_all") is None:
            raise RuntimeError("Model output has no logits_all; enable use_layer_ensemble during training/eval.")
        K_all = int(out["logits_all"].shape[0])
        fused_idx = K_all - 1

    if args.teacher_indices is None:
        print("[Teacher] Scoring heads on VAL to pick teacher...")
        metrics = _score_heads_on_val(
            model=model,
            loader=val_loader,
            device=device,
            thr_mode=cfg.thr_mode,
            thr=cfg.thr,
            topk=cfg.topk,
            smooth_k=cfg.smooth_k,
            use_minarea=cfg.use_minarea,
            min_area=cfg.min_area,
        )
        f1s = [m["f1"] for m in metrics]
        layer_indices = list(range(0, max(0, fused_idx)))
        layer_sorted = sorted(layer_indices, key=lambda i: f1s[i], reverse=True)
        picked = layer_sorted[: max(1, min(int(args.teacher_topk), len(layer_sorted)))]
        teacher_indices = picked
        print(f"[Teacher] Picked top-{len(picked)} layer heads by VAL F1: {teacher_indices}")
    else:
        teacher_indices = [int(i) for i in args.teacher_indices if int(i) != fused_idx]
        if any((i < 0 or i >= fused_idx) for i in teacher_indices):
            raise ValueError(f"--teacher_indices must be within [0,{fused_idx-1}] (layer heads only), got {teacher_indices}")
        if len(teacher_indices) <= 0:
            raise ValueError("Empty teacher_indices after excluding fused head.")
        print(f"[Teacher] Using provided layer heads: {teacher_indices}")

    teacher_weights = None
    if args.teacher_mode == "weighted_logit":
        if args.teacher_weights is None:
            raise ValueError("--teacher_mode weighted_logit requires --teacher_weights")
        if len(args.teacher_weights) != len(teacher_indices):
            raise ValueError("--teacher_weights length must match selected teacher_indices length")
        teacher_weights = [float(x) for x in args.teacher_weights]
        print(f"[Teacher] Weights={['{:.3f}'.format(x) for x in teacher_weights]}")

    _set_trainable_for_distill(model, train_fused=args.train_fused)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")

    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.startswith("cuda") and torch.cuda.is_available())
    scheduler = build_scheduler(optimizer, cfg)

    best_f1 = -1.0
    best_path = os.path.join(cfg.out_dir, "best.pt")
    last_path = os.path.join(cfg.out_dir, "last.pt")
    metrics_path = os.path.join(cfg.out_dir, "metrics.jsonl")

    print("\n===== Start Distillation =====")
    print(f"device={device}")
    print(f"data_root={cfg.data_root}")
    print(f"out_dir={cfg.out_dir}")
    print(f"epochs={cfg.epochs} batch={cfg.batch_size} grad_accum={cfg.grad_accum} lr={cfg.lr}")
    print(f"alpha={args.alpha} teacher_mode={args.teacher_mode} teacher_indices={teacher_indices}")
    print("==============================\n")

    for ep in range(1, cfg.epochs + 1):
        train_one_epoch_distill(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            teacher_indices=teacher_indices,
            teacher_mode=args.teacher_mode,
            teacher_weights=teacher_weights,
            alpha=float(args.alpha),
            grad_accum=cfg.grad_accum,
            log_every=cfg.log_every,
        )
        if scheduler is not None:
            scheduler.step()

        if (ep % int(args.eval_every) == 0) or (ep == cfg.epochs):
            val_m = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                thr_mode=cfg.thr_mode,
                thr=cfg.thr,
                topk=cfg.topk,
                smooth_k=cfg.smooth_k,
                use_minarea=cfg.use_minarea,
                min_area=cfg.min_area,
                print_every=0,
                window=cfg.eval_window,
                stride=cfg.eval_stride,
                use_ensemble=False,
                ensemble_cfg=None,
            )
            print(
                f"[Epoch {ep:03d}] VAL  "
                f"P={val_m['precision']:.4f} R={val_m['recall']:.4f} F1={val_m['f1']:.4f} "
                f"IoU={val_m['iou']:.4f} OA={val_m['oa']:.4f} Kappa={val_m['kappa']:.4f}"
            )
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"epoch": ep, "split": "val", **val_m, "time": time.time()}, ensure_ascii=False) + "\n")
            if val_m["f1"] > best_f1:
                best_f1 = float(val_m["f1"])
                torch.save({"model": model.state_dict(), "epoch": ep, "best_f1": best_f1, "cfg": asdict(cfg)}, best_path)
                print(f"Saved best to {best_path} (best_f1={best_f1:.4f})")

        torch.save({"model": model.state_dict(), "epoch": ep, "best_f1": best_f1, "cfg": asdict(cfg)}, last_path)


if __name__ == "__main__":
    main()

