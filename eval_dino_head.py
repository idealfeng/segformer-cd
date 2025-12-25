"""
python eval_dino_head.py --checkpoint outputs/dino_head_cd/best.pt --data_root data/LEVIR-CD --out_dir outputs/eval --thr_mode fixed --smooth_k 3 --use_minarea --min_area 256 --vis --vis_n 10 --vis_dir outputs/eval/vis --full_eval
"""

import argparse
import json
import os
from pathlib import Path
from dataclasses import asdict

import torch

from dino_head_core import (
    HeadCfg,
    DinoSiameseHead,
    DinoFrozenA0Head,
    build_dataloaders,
    seed_everything,
    evaluate,
    save_vis_samples,
    sliding_window_inference_logits_all,
    sliding_window_inference_probs_all,
    threshold_map,
    filter_small_cc,
    confusion_update,
    compute_metrics_from_cm,
)

try:
    from config import cfg as project_cfg
    _DEFAULT_WINDOW = int(getattr(project_cfg, "IMAGE_SIZE", 256))
except Exception:
    _DEFAULT_WINDOW = 256


def parse_args():
    base = HeadCfg()
    beta_default = getattr(base, "beta_prior", 0.01)
    parser = argparse.ArgumentParser(description="Evaluate DINOv2 change-detection head")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (best.pt/last.pt)")
    parser.add_argument("--data_root", type=str, default=base.data_root)
    parser.add_argument(
        "--calib_root",
        type=str,
        default=None,
        help="Optional calibration data_root for selecting heads / fitting ensemble weights (e.g., source domain). If not set, uses --data_root.",
    )
    parser.add_argument("--out_dir", type=str, default=None, help="目录为空则默认与 checkpoint 同级")
    parser.add_argument("--device", type=str, default=base.device, help="cuda|cpu|auto")
    parser.add_argument("--batch_size", type=int, default=2, help="Eval batch size (use 1 for full images)")
    parser.add_argument("--full_eval", dest="full_eval", action="store_true")
    parser.add_argument("--no_full_eval", dest="full_eval", action="store_false")
    parser.set_defaults(full_eval=base.full_eval)
    parser.add_argument("--eval_crop", type=int, default=base.eval_crop)
    parser.add_argument("--thr_mode", type=str, choices=["fixed", "topk", "otsu"], default=base.thr_mode)
    parser.add_argument("--thr", type=float, default=base.thr)
    parser.add_argument("--topk", type=float, default=base.topk)
    parser.add_argument("--beta_prior", type=float, default=beta_default, help="Expected change ratio prior for beta mixture")
    parser.add_argument("--smooth_k", type=int, default=base.smooth_k)
    parser.add_argument("--use_minarea", action="store_true", default=base.use_minarea)
    parser.add_argument("--min_area", type=int, default=base.min_area)
    parser.add_argument("--use_ensemble_pred", action="store_true", default=base.use_ensemble_pred, help="use ensemble mean for evaluation")
    parser.add_argument(
        "--selected_layers",
        type=int,
        nargs="+",
        default=list(base.selected_layers),
        help="1-based transformer block indices used by the head (must match checkpoint).",
    )
    parser.add_argument(
        "--ensemble_strategy",
        type=str,
        default="mean_prob",
        choices=["mean_prob", "mean_logit", "topk", "weighted_logit", "cvx_nll"],
        help="Ensemble strategy when --use_ensemble_pred is set",
    )
    parser.add_argument(
        "--ensemble_indices",
        type=str,
        default=None,
        help="Optional comma-separated head indices to use (e.g., '3,4'). Overrides topk selection.",
    )
    parser.add_argument(
        "--ensemble_topk",
        type=int,
        default=2,
        help="For topk/weighted_logit: pick top-k layer heads (excluding fused); 0 means use all heads",
    )
    parser.add_argument(
        "--ensemble_weight_norm",
        type=str,
        default="softmax",
        choices=["softmax", "linear"],
        help="How to turn val F1 into weights for weighted_logit",
    )
    parser.add_argument(
        "--ensemble_weight_temp",
        type=float,
        default=1.0,
        help="Softmax temperature for weighted_logit (smaller -> more peaky)",
    )
    parser.add_argument("--cvx_lambda", type=float, default=1e-3, help="L2 regularization for cvx_nll weights (ensures unique optimum)")
    parser.add_argument("--cvx_steps", type=int, default=200, help="Optimization steps for cvx_nll")
    parser.add_argument("--cvx_lr", type=float, default=0.5, help="Step size for projected gradient descent (cvx_nll)")
    parser.add_argument("--cvx_max_pixels", type=int, default=400000, help="Max sampled pixels from val for cvx_nll")
    parser.add_argument(
        "--cvx_pos_weight",
        type=str,
        default="auto",
        choices=["auto", "none"],
        help="Use pos_weight for BCE in cvx_nll (auto uses neg/pos from sampled pixels)",
    )
    parser.add_argument("--vis", action="store_true", help="Save visualization samples")
    parser.add_argument("--vis_n", type=int, default=base.vis_n)
    parser.add_argument("--vis_dir", type=str, default=None)
    parser.add_argument("--print_every", type=int, default=0)
    parser.add_argument(
        "--window",
        type=int,
        default=_DEFAULT_WINDOW,
        help=f"滑窗窗口大小，默认 {_DEFAULT_WINDOW}（设为0或负值则禁用滑窗）",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=_DEFAULT_WINDOW,
        help="滑窗步长，默认等于 window",
    )
    args = parser.parse_args()
    if args.ensemble_indices:
        try:
            args.ensemble_indices = [int(x) for x in str(args.ensemble_indices).split(",") if str(x).strip() != ""]
        except Exception:
            raise ValueError("--ensemble_indices must be a comma-separated list of ints, e.g. '3,4'")
        if len(args.ensemble_indices) <= 0:
            args.ensemble_indices = None
    if (args.ensemble_strategy != "mean_prob") and (not args.use_ensemble_pred):
        print("[Ensemble] Detected --ensemble_strategy != mean_prob; auto-enabling --use_ensemble_pred.")
        args.use_ensemble_pred = True
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = HeadCfg(
        data_root=args.data_root,
        out_dir=args.out_dir or str(Path(args.checkpoint).parent),
        device=device,
        full_eval=args.full_eval,
        eval_crop=args.eval_crop,
        thr_mode=args.thr_mode,
        thr=args.thr,
        topk=args.topk,
        smooth_k=args.smooth_k,
        use_minarea=args.use_minarea,
        min_area=args.min_area,
        use_ensemble_pred=args.use_ensemble_pred,
        selected_layers=tuple(int(x) for x in args.selected_layers),
    )
    if args.window is not None and args.window <= 0:
        args.window = None
    if args.stride is not None and args.stride <= 0:
        args.stride = None
    return args, cfg


def _project_simplex(v: torch.Tensor) -> torch.Tensor:
    """
    Euclidean projection of v onto the probability simplex {w>=0, sum w = 1}.
    v: 1D tensor
    """
    if v.ndim != 1:
        raise ValueError("project_simplex expects 1D tensor")
    n = v.numel()
    if n == 1:
        return torch.ones_like(v)
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0) - 1
    ind = torch.arange(1, n + 1, device=v.device, dtype=v.dtype)
    cond = u - cssv / ind > 0
    if not bool(cond.any()):
        return torch.full_like(v, 1.0 / n)
    rho = int(torch.nonzero(cond, as_tuple=False)[-1].item()) + 1
    theta = cssv[rho - 1] / float(rho)
    w = (v - theta).clamp_min(0.0)
    w = w / w.sum().clamp_min(1e-12)
    return w


@torch.no_grad()
def _collect_val_pixels_for_cvx(
    model: torch.nn.Module,
    loader,
    device: str,
    indices: list[int],
    window: int | None,
    stride: int | None,
    max_pixels: int,
):
    """
    Collect sampled pixels (logits per head, label) from val.
    Returns:
      logits_mat: [N, K] float32 on device
      labels: [N] float32 on device
    """
    logits_buf = None
    y_buf = None
    max_pixels = int(max(1000, max_pixels))

    for batch in loader:
        img_a = batch["img_a"].to(device, non_blocking=True)
        img_b = batch["img_b"].to(device, non_blocking=True)
        gt = batch["label"].to(device)
        if gt.ndim == 3:
            gt = gt.unsqueeze(1)
        if gt.ndim == 4 and gt.shape[1] != 1:
            gt = gt[:, :1]
        y = (gt > 0).float()  # [B,1,H,W]

        if window is not None and stride is not None:
            if img_a.shape[0] != 1:
                raise ValueError("cvx_nll with sliding window currently supports batch_size==1.")
            logits_all = sliding_window_inference_logits_all(
                model=model,
                img_a=img_a,
                img_b=img_b,
                window=window,
                stride=stride,
                device=device,
            )  # [Kall,1,1,H,W]
        else:
            out = model(img_a, img_b)
            if not isinstance(out, dict) or out.get("logits_all") is None:
                raise RuntimeError("Model output has no logits_all; enable use_layer_ensemble during training/eval.")
            logits_all = out["logits_all"]  # [Kall,B,1,H,W]

        idx = torch.as_tensor(indices, device=logits_all.device, dtype=torch.long)
        logits_sel = logits_all.index_select(0, idx)  # [K,B,1,H,W]
        K, B, _, H, W = logits_sel.shape
        logits_mat = logits_sel.permute(1, 3, 4, 0, 2).reshape(B * H * W, K)  # [N,K]
        y_flat = y.reshape(B * H * W)  # [N]

        # merge buffer with cap
        if logits_buf is None:
            logits_buf = logits_mat
            y_buf = y_flat
        else:
            logits_buf = torch.cat([logits_buf, logits_mat], dim=0)
            y_buf = torch.cat([y_buf, y_flat], dim=0)

        if logits_buf.shape[0] > max_pixels:
            perm = torch.randperm(logits_buf.shape[0], device=logits_buf.device)[:max_pixels]
            logits_buf = logits_buf.index_select(0, perm)
            y_buf = y_buf.index_select(0, perm)

    if logits_buf is None or y_buf is None:
        raise RuntimeError("Empty val loader; cannot fit cvx_nll weights.")
    return logits_buf.to(dtype=torch.float32), y_buf.to(dtype=torch.float32)


def _fit_cvx_nll_weights(
    logits_mat: torch.Tensor,
    labels: torch.Tensor,
    l2_lambda: float,
    steps: int,
    lr: float,
    pos_weight_mode: str,
) -> torch.Tensor:
    """
    Solve:
      min_{w in simplex} BCEWithLogitsLoss(sum_k w_k z_k, y) + lambda * ||w||_2^2
    Convex in w; with lambda>0 it's strongly convex => unique optimum.
    """
    if logits_mat.ndim != 2:
        raise ValueError("logits_mat must be [N,K]")
    N, K = logits_mat.shape
    if labels.ndim != 1 or labels.shape[0] != N:
        raise ValueError("labels must be [N] matching logits_mat")
    steps = int(max(50, steps))
    lr = float(max(1e-6, lr))
    l2_lambda = float(max(0.0, l2_lambda))

    if pos_weight_mode == "auto":
        pos = float(labels.sum().item())
        neg = float(labels.numel() - pos)
        if pos > 0:
            pos_weight = torch.tensor([neg / max(1.0, pos)], device=logits_mat.device, dtype=torch.float32)
        else:
            pos_weight = None
    else:
        pos_weight = None

    w = torch.full((K,), 1.0 / float(K), device=logits_mat.device, dtype=torch.float32, requires_grad=True)

    for _ in range(steps):
        z = (logits_mat * w.view(1, K)).sum(dim=1)  # [N]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, labels, pos_weight=pos_weight)
        if l2_lambda > 0:
            loss = loss + l2_lambda * (w * w).sum()
        loss.backward()
        with torch.no_grad():
            grad = w.grad
            w_new = _project_simplex(w - lr * grad)
            w.copy_(w_new)
            w.grad = None

    return w.detach()


@torch.no_grad()
def score_heads_on_loader(
    model: torch.nn.Module,
    loader,
    device: str,
    thr_mode: str,
    thr: float,
    topk: float,
    smooth_k: int,
    use_minarea: bool,
    min_area: int,
    window: int,
    stride: int,
):
    model.eval()
    cms = None  # list of dicts
    for batch in loader:
        img_a = batch["img_a"].to(device, non_blocking=True)
        img_b = batch["img_b"].to(device, non_blocking=True)
        gt = batch["label"]

        if window is not None and stride is not None:
            probs_all = sliding_window_inference_probs_all(
                model=model,
                img_a=img_a,
                img_b=img_b,
                window=window,
                stride=stride,
                device=device,
            )  # [K,B,1,H,W]
        else:
            out = model(img_a, img_b)
            if not isinstance(out, dict) or out.get("logits_all") is None:
                raise RuntimeError("Model output has no logits_all; enable use_layer_ensemble during training/eval.")
            probs_all = torch.sigmoid(out["logits_all"])

        K, B, _, H, W = probs_all.shape
        if cms is None:
            cms = [{"TP": 0, "FP": 0, "FN": 0, "TN": 0} for _ in range(K)]

        if smooth_k and smooth_k > 1:
            pad = smooth_k // 2
            probs_all = probs_all.view(K * B, 1, H, W)
            probs_all = torch.nn.functional.avg_pool2d(probs_all, kernel_size=smooth_k, stride=1, padding=pad)
            probs_all = probs_all.view(K, B, 1, H, W)

        for bi in range(B):
            gt_np = (
                gt[bi].detach().cpu().numpy().astype("uint8")
                if gt.ndim == 4
                else gt[bi].detach().cpu().numpy().astype("uint8")
            )
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
        raise RuntimeError("Empty loader; cannot score heads.")
    metrics = [compute_metrics_from_cm(cm) for cm in cms]
    return metrics


def main():
    args, cfg = parse_args()
    seed_everything(cfg.seed)
    device = cfg.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable, fallback to CPU")
        device = "cpu"
        cfg.device = device
    cfg.batch_size = args.batch_size
    # eval loaders: use --data_root
    _, val_loader, test_loader = build_dataloaders(cfg)
    # calibration loaders: optionally use --calib_root (e.g., source domain)
    val_loader_calib = val_loader
    if args.calib_root:
        cfg_calib = HeadCfg(
            data_root=args.calib_root,
            out_dir=cfg.out_dir,
            device=cfg.device,
            full_eval=cfg.full_eval,
            eval_crop=cfg.eval_crop,
            thr_mode=cfg.thr_mode,
            thr=cfg.thr,
            topk=cfg.topk,
            smooth_k=cfg.smooth_k,
            use_minarea=cfg.use_minarea,
            min_area=cfg.min_area,
            use_ensemble_pred=cfg.use_ensemble_pred,
        )
        cfg_calib.batch_size = cfg.batch_size
        try:
            _, val_loader_calib, _ = build_dataloaders(cfg_calib)
            print(f"[Calib] Using calib_root={args.calib_root} (val size={len(val_loader_calib.dataset)})")
        except Exception as e:
            print(f"[Calib] Failed to build calib loaders from {args.calib_root}: {e}; fallback to --data_root val.")
            val_loader_calib = val_loader
    ckpt = torch.load(args.checkpoint, map_location=device)
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
        if DinoFrozenA0Head is None:
            raise ImportError(
                "DinoFrozenA0Head is not available. Please update models/dinov2_head.py to a version that defines DinoFrozenA0Head, "
                "or evaluate a non-A0 checkpoint."
            )
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
    print(f"Loaded checkpoint from {args.checkpoint}")
    print(f"val/test sizes: {len(val_loader.dataset)}/{len(test_loader.dataset)}")

    # Convenience: allow passing transformer layer numbers in --ensemble_indices.
    # If any provided index is >= K (K = len(selected_layers)+1), treat them as layer numbers.
    if args.ensemble_indices is not None and isinstance(cfg.selected_layers, (list, tuple)) and len(cfg.selected_layers) > 0:
        K = len(cfg.selected_layers) + 1
        if any(int(i) >= K for i in args.ensemble_indices):
            layer_to_head = {int(layer): idx for idx, layer in enumerate(cfg.selected_layers)}
            mapped = []
            for v in args.ensemble_indices:
                vv = int(v)
                if vv not in layer_to_head:
                    raise ValueError(
                        f"--ensemble_indices contains {vv}, which is not in selected_layers={list(cfg.selected_layers)}. "
                        f"Provide head indices 0..{K-1}, or use layer numbers from selected_layers."
                    )
                mapped.append(layer_to_head[vv])
            print(
                f"[Ensemble] Mapped transformer layers {args.ensemble_indices} -> head indices {mapped} "
                f"(selected_layers={list(cfg.selected_layers)})"
            )
            args.ensemble_indices = mapped

    ensemble_cfg = None
    if args.use_ensemble_pred:
        if args.ensemble_strategy in ("mean_prob", "mean_logit"):
            ensemble_cfg = {"mode": args.ensemble_strategy}
            if args.ensemble_indices is not None:
                ensemble_cfg["indices"] = args.ensemble_indices
                print(f"[Ensemble] Using fixed heads indices={args.ensemble_indices} with mode={args.ensemble_strategy}")
        else:
            if not getattr(cfg, "use_layer_ensemble", False):
                print("[Ensemble] use_layer_ensemble=False; cannot score heads for topk/weighted/cvx strategies. Disabling ensemble.")
                args.use_ensemble_pred = False

    if args.use_ensemble_pred and ensemble_cfg is None and args.ensemble_strategy not in ("mean_prob", "mean_logit"):
            print("\n[Ensemble] Scoring each head on VAL to derive selection/weights...")
            head_metrics = score_heads_on_loader(
                model=model,
                loader=val_loader_calib,
                device=device,
                thr_mode=cfg.thr_mode,
                thr=cfg.thr,
                topk=cfg.topk,
                smooth_k=cfg.smooth_k,
                use_minarea=cfg.use_minarea,
                min_area=cfg.min_area,
                window=args.window,
                stride=args.stride,
            )
            f1s = [m["f1"] for m in head_metrics]
            K = len(f1s)
            fused_idx = K - 1
            print("[Ensemble] VAL F1 per head (0..K-2 are layer heads, K-1 is fused):")
            for i, f1 in enumerate(f1s):
                tag = "fused" if i == fused_idx else f"layer{i}"
                print(f"  head[{i}] ({tag}): F1={f1:.4f}")

            if args.ensemble_indices is not None:
                indices = [int(i) for i in args.ensemble_indices]
            else:
                layer_indices = list(range(0, max(0, fused_idx)))
                if args.ensemble_topk and args.ensemble_topk > 0 and layer_indices:
                    layer_sorted = sorted(layer_indices, key=lambda i: f1s[i], reverse=True)
                    picked_layers = layer_sorted[: min(args.ensemble_topk, len(layer_sorted))]
                else:
                    picked_layers = layer_indices
                indices = picked_layers + [fused_idx]

            # validate indices
            indices = [int(i) for i in indices]
            if any((i < 0 or i >= K) for i in indices):
                raise ValueError(f"--ensemble_indices out of range: got {indices}, but K={K}")
            # ensure fused included by default
            if (args.ensemble_indices is None) and (fused_idx not in indices):
                indices = indices + [fused_idx]

            if args.ensemble_strategy == "topk":
                ensemble_cfg = {"mode": "mean_logit", "indices": indices}
                print(f"[Ensemble] Using mean_logit over heads indices={indices}")
            elif args.ensemble_strategy == "weighted_logit":
                f1_sel = torch.tensor([f1s[i] for i in indices], dtype=torch.float32)
                if args.ensemble_weight_norm == "linear":
                    w = (f1_sel - float(f1_sel.min())).clamp_min(0.0) + 1e-6
                    w = (w / w.sum()).tolist()
                else:
                    temp = max(1e-6, float(args.ensemble_weight_temp))
                    w = torch.softmax(f1_sel / temp, dim=0).tolist()
                ensemble_cfg = {"mode": "weighted_logit", "indices": indices, "weights": w}
                print(f"[Ensemble] Using weighted_logit over heads indices={indices}")
                print(f"[Ensemble] Weights={['{:.3f}'.format(x) for x in w]}")
            elif args.ensemble_strategy == "cvx_nll":
                print(f"[Ensemble] Fitting convex NLL weights on VAL (indices={indices})...")
                logits_mat, y_flat = _collect_val_pixels_for_cvx(
                    model=model,
                    loader=val_loader_calib,
                    device=device,
                    indices=indices,
                    window=args.window,
                    stride=args.stride,
                    max_pixels=args.cvx_max_pixels,
                )
                w = _fit_cvx_nll_weights(
                    logits_mat=logits_mat,
                    labels=y_flat,
                    l2_lambda=args.cvx_lambda,
                    steps=args.cvx_steps,
                    lr=args.cvx_lr,
                    pos_weight_mode=args.cvx_pos_weight,
                ).tolist()
                ensemble_cfg = {"mode": "weighted_logit", "indices": indices, "weights": w, "solver": "cvx_nll"}
                print(f"[Ensemble] Using cvx_nll weighted_logit over heads indices={indices}")
                print(f"[Ensemble] Weights={['{:.3f}'.format(x) for x in w]}")
            else:
                raise ValueError(f"Unknown ensemble_strategy: {args.ensemble_strategy}")
            try:
                os.makedirs(cfg.out_dir, exist_ok=True)
                with open(os.path.join(cfg.out_dir, "ensemble_cfg.json"), "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "strategy": args.ensemble_strategy,
                            "indices": indices,
                            "f1s": f1s,
                            "cfg": ensemble_cfg,
                            "cvx": {
                                "lambda": float(args.cvx_lambda),
                                "steps": int(args.cvx_steps),
                                "lr": float(args.cvx_lr),
                                "max_pixels": int(args.cvx_max_pixels),
                                "pos_weight": str(args.cvx_pos_weight),
                            }
                            if args.ensemble_strategy == "cvx_nll"
                            else None,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
            except Exception:
                pass

    metrics = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        thr_mode=cfg.thr_mode,
        thr=cfg.thr,
        topk=cfg.topk,
        smooth_k=cfg.smooth_k,
        use_minarea=cfg.use_minarea,
        min_area=cfg.min_area,
        print_every=args.print_every,
        window=args.window,
        stride=args.stride,
        use_ensemble=args.use_ensemble_pred,
        ensemble_cfg=ensemble_cfg,
    )
    print("\n====== Test Metrics ======")
    for k in ["precision", "recall", "f1", "iou", "oa", "kappa"]:
        print(f"{k}: {metrics[k]:.4f}")
    print(f"TP={metrics['TP']} FP={metrics['FP']} FN={metrics['FN']} TN={metrics['TN']}")
    out_dir = cfg.out_dir
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "eval_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"split": "test", **metrics, "cfg": asdict(cfg)}, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics to {results_path}")
    if args.vis:
        vis_dir = args.vis_dir or os.path.join(out_dir, "vis_eval")
        save_vis_samples(
            model=model,
            loader=val_loader,
            device=device,
            out_dir=vis_dir,
            n=args.vis_n,
            thr_mode=cfg.thr_mode,
            thr=cfg.thr,
            topk=cfg.topk,
            smooth_k=cfg.smooth_k,
            use_ensemble=args.use_ensemble_pred,
            ensemble_cfg=ensemble_cfg,
        )
        print(f"Saved visualizations to {vis_dir}")


if __name__ == "__main__":
    main()
