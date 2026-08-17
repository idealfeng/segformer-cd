from __future__ import annotations

"""
python eval_dino_head.py --checkpoint outputs/dino_head_cd/best.pt --data_root data/LEVIR-CD --out_dir outputs/eval --thr_mode fixed --smooth_k 3 --use_minarea --min_area 256 --vis --vis_n 10 --vis_dir outputs/eval/vis --full_eval
"""

import argparse
import json
import os
import time
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
    sliding_window_inference,
    sliding_window_inference_logits_all,
    sliding_window_inference_probs_all,
    tta_inference_prob,
    apply_corruption_pair,
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


class ProbabilityCheckpointEnsemble(torch.nn.Module):
    """Average checkpoint probabilities while preserving the model output API."""

    def __init__(self, models, weights=None):
        super().__init__()
        if len(models) < 2:
            raise ValueError("ProbabilityCheckpointEnsemble requires at least two models")
        self.models = torch.nn.ModuleList(models)
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        if len(weights) != len(models) or any(float(w) < 0 for w in weights):
            raise ValueError("Model ensemble weights must be non-negative and match model count")
        weight_sum = sum(float(w) for w in weights)
        if weight_sum <= 0:
            raise ValueError("Model ensemble weights must have a positive sum")
        self.register_buffer(
            "weights",
            torch.tensor([float(w) / weight_sum for w in weights], dtype=torch.float32),
        )

    def _mean_prob_logit(self, logits):
        prob_stack = torch.stack([torch.sigmoid(x) for x in logits], dim=0)
        shape = (len(logits),) + (1,) * (prob_stack.ndim - 1)
        probs = (prob_stack * self.weights.to(prob_stack).view(shape)).sum(dim=0)
        return torch.logit(probs.clamp(1e-6, 1.0 - 1e-6))

    def forward(self, img_a, img_b):
        outputs = [model(img_a, img_b) for model in self.models]
        result = dict(outputs[0])
        result["pred"] = self._mean_prob_logit([out["pred"] for out in outputs])

        logits_all = [out.get("logits_all") for out in outputs]
        if all(x is not None for x in logits_all):
            shapes = {tuple(x.shape) for x in logits_all}
            if len(shapes) != 1:
                raise RuntimeError(f"Checkpoint ensemble logits_all shapes differ: {sorted(shapes)}")
            result["logits_all"] = self._mean_prob_logit(logits_all)
        else:
            result["logits_all"] = None
        return result


def parse_args():
    base = HeadCfg()
    beta_default = getattr(base, "beta_prior", 0.01)
    parser = argparse.ArgumentParser(description="Evaluate DINOv2 change-detection head")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (best.pt/last.pt)")
    parser.add_argument(
        "--model_ensemble_checkpoints",
        type=str,
        nargs="*",
        default=None,
        help="Additional checkpoints to fuse in probability space; heterogeneous fuse modes are supported.",
    )
    parser.add_argument(
        "--model_ensemble_weights",
        type=str,
        default=None,
        help="Comma-separated probability weights for primary plus additional checkpoints.",
    )
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
    parser.add_argument(
        "--num_workers",
        type=int,
        default=base.num_workers,
        help="DataLoader workers. Set to 0 if multiprocessing is not permitted in your environment.",
    )
    parser.add_argument("--full_eval", dest="full_eval", action="store_true")
    parser.add_argument("--no_full_eval", dest="full_eval", action="store_false")
    parser.set_defaults(full_eval=base.full_eval)
    parser.add_argument("--eval_crop", type=int, default=base.eval_crop)
    parser.add_argument("--thr_mode", type=str, choices=["fixed", "topk", "otsu", "val_best"], default=base.thr_mode)
    parser.add_argument("--thr", type=float, default=base.thr)
    parser.add_argument("--topk", type=float, default=base.topk)
    parser.add_argument(
        "--val_best_max_pixels",
        type=int,
        default=400000,
        help="For thr_mode=val_best: max sampled pixels from VAL to search best fixed threshold.",
    )
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
        choices=["mean_prob", "mean_logit", "min_prob", "soft_min", "max_rejection", "topk", "weighted_logit", "cvx_nll", "ugls", "consis2", "consisk", "uwi"],
        help="Ensemble strategy when --use_ensemble_pred is set",
    )
    parser.add_argument(
        "--ensemble_indices",
        type=str,
        default=None,
        help="Optional comma-separated head indices to use (e.g., '3,4'). Overrides topk selection.",
    )
    parser.add_argument(
        "--consensus_strength",
        type=float,
        default=0.5,
        help="For soft_min/max_rejection: conservative fusion strength in [0,1].",
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
    parser.add_argument("--ugls_min_k", type=int, default=2, help="UGLS: min number of heads per pixel (ranked by VAL F1)")
    parser.add_argument("--ugls_max_k", type=int, default=0, help="UGLS: max heads per pixel (0 = use all selected heads)")
    parser.add_argument("--ugls_unc_power", type=float, default=1.0, help="UGLS: exponent on normalized uncertainty (>=1 is more conservative)")
    parser.add_argument("--consis2_d0", type=float, default=0.05, help="consis2: disagreement low threshold")
    parser.add_argument("--consis2_d1", type=float, default=0.25, help="consis2: disagreement high threshold")
    parser.add_argument("--consis2_gamma", type=float, default=1.0, help="consis2: nonlinearity on mapped disagreement")
    parser.add_argument("--consis2_max_w", type=float, default=0.5, help="consis2: max blend weight towards the non-anchor head")
    parser.add_argument("--consisk_temp0", type=float, default=0.03, help="consisk: base softmax temperature")
    parser.add_argument("--consisk_temp1", type=float, default=0.20, help="consisk: temperature scale with uncertainty")
    parser.add_argument("--consisk_gamma", type=float, default=1.0, help="consisk: exponent on uncertainty normalization")
    parser.add_argument("--consisk_fused_bias", type=float, default=0.0, help="consisk: additive bias to fused head score (after selection)")
    parser.add_argument("--uwi_unc_indices", type=str, default=None, help="UWI: comma-separated head indices for uncertainty Var(p) (default: all heads)")
    parser.add_argument("--uwi_u0", type=float, default=0.10, help="UWI: uncertainty lower threshold (normalized to [0,1])")
    parser.add_argument("--uwi_u1", type=float, default=0.60, help="UWI: uncertainty upper threshold (normalized to [0,1])")
    parser.add_argument("--uwi_gamma", type=float, default=1.0, help="UWI: nonlinearity on mapped uncertainty")
    parser.add_argument("--uwi_max_w", type=float, default=0.50, help="UWI: max blend weight towards the non-anchor head")
    parser.add_argument("--uwi_gate_smooth_k", type=int, default=0, help="UWI: optional spatial smoothing kernel for gate g(x)")
    parser.add_argument("--vis", action="store_true", help="Save visualization samples")
    parser.add_argument("--vis_n", type=int, default=base.vis_n)
    parser.add_argument("--vis_dir", type=str, default=None)
    parser.add_argument(
        "--tta",
        type=str,
        default="none",
        choices=["none", "pair_contrast", "flip", "flip_median", "flip_min", "d4"],
        help="Test-time augmentation and probability consensus mode.",
    )
    parser.add_argument(
        "--corrupt",
        type=str,
        default="none",
        choices=["none", "gaussian", "bc", "jpeg"],
        help="Test-time corruption applied to inputs (image-space, then re-normalized).",
    )
    parser.add_argument("--corrupt_pair", type=str, default="correlated", choices=["correlated", "uncorrelated"])
    parser.add_argument("--corrupt_seed", type=int, default=0)
    parser.add_argument("--gaussian_sigma", type=float, default=0.0, help="Gaussian noise std in [0,1] space")
    parser.add_argument("--bc_brightness", type=float, default=0.0, help="Brightness shift magnitude in [0,1] space")
    parser.add_argument("--bc_contrast", type=float, default=0.0, help="Contrast delta: c ~ U(1-d,1+d)")
    parser.add_argument("--jpeg_quality", type=int, default=75, help="JPEG quality (1..95)")
    parser.add_argument("--print_every", type=int, default=0)
    parser.add_argument(
        "--strict_zeroshot",
        action="store_true",
        help=(
            "Do not construct or use the target validation split. Requires a distinct "
            "--calib_root and rejects target-label visualizations."
        ),
    )
    parser.add_argument(
        "--exploratory_target_tuned",
        action="store_true",
        help="Record that target labels were inspected to select inference hyperparameters.",
    )
    parser.add_argument("--profile_complexity", action="store_true", help="Add Params/FLOPs/latency/FPS to eval_results.json.")
    parser.add_argument("--profile_size", type=int, default=256, help="Input crop size for complexity profiling.")
    parser.add_argument("--profile_warmup", type=int, default=10, help="Warmup iterations for latency profiling.")
    parser.add_argument("--profile_iters", type=int, default=30, help="Timed iterations for latency profiling.")
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

    def _has_ab_label(p: Path) -> bool:
        return (p / "A").is_dir() and (p / "B").is_dir() and (p / "label").is_dir()

    def _normalize_data_root(p_str: str | None, arg_name: str) -> str | None:
        if not p_str:
            return p_str
        p = Path(p_str)
        name = p.name.lower()
        if name in ("train", "val", "test"):
            parent = p.parent
            parent_has_splits = any((parent / s).is_dir() for s in ("train", "val", "test"))
            looks_like_split = _has_ab_label(p) or _has_ab_label(p / p.name)
            if parent_has_splits and looks_like_split:
                print(f"[Data] Detected {arg_name} points to split folder '{p.name}'; using parent dataset root: {parent}")
                return str(parent)
        return p_str

    args.data_root = _normalize_data_root(args.data_root, "--data_root")
    args.calib_root = _normalize_data_root(args.calib_root, "--calib_root")

    if args.strict_zeroshot:
        if not args.calib_root:
            raise ValueError("--strict_zeroshot requires a source-domain --calib_root")
        target_root = Path(args.data_root).resolve()
        calib_root = Path(args.calib_root).resolve()
        if target_root == calib_root:
            raise ValueError("--strict_zeroshot requires --calib_root to differ from --data_root")
        if args.vis:
            raise ValueError("--strict_zeroshot rejects --vis because it renders target labels")

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


def _build_model_from_checkpoint_cfg(load_cfg: dict, cfg: HeadCfg, device: str):
    arch = load_cfg.get("arch", getattr(cfg, "arch", "dlv"))
    if arch == "a0":
        if DinoFrozenA0Head is None:
            raise ImportError(
                "DinoFrozenA0Head is not available. Please update models/dinov2_head.py "
                "or evaluate a non-A0 checkpoint."
            )
        return DinoFrozenA0Head(
            dino_name=load_cfg.get("dino_name", cfg.dino_name),
            layer=load_cfg.get("a0_layer", cfg.a0_layer),
            use_whiten=load_cfg.get("use_whiten", cfg.use_whiten),
        ).to(device)

    selected_layers = tuple(
        int(x) for x in load_cfg.get("selected_layers", cfg.selected_layers)
    )
    return DinoSiameseHead(
        dino_name=load_cfg.get("dino_name", cfg.dino_name),
        selected_layers=selected_layers,
        fuse_mode=load_cfg.get("fuse_mode", cfg.fuse_mode),
        align_radius=load_cfg.get("align_radius", cfg.align_radius),
        align_temperature=load_cfg.get("align_temperature", cfg.align_temperature),
        learnable_align_blend=load_cfg.get(
            "learnable_align_blend", cfg.learnable_align_blend
        ),
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
        use_nuisance_gate=load_cfg.get("use_nuisance_gate", cfg.use_nuisance_gate),
        nuisance_hidden=load_cfg.get("nuisance_hidden", cfg.nuisance_hidden),
        nuisance_gate_weight=load_cfg.get("nuisance_gate_weight", cfg.nuisance_gate_weight),
        nuisance_use_image_cues=load_cfg.get("nuisance_use_image_cues", cfg.nuisance_use_image_cues),
        use_residual_style_adapter=load_cfg.get(
            "use_residual_style_adapter", cfg.use_residual_style_adapter
        ),
        style_adapter_hidden=load_cfg.get("style_adapter_hidden", cfg.style_adapter_hidden),
        style_adapter_scale=load_cfg.get("style_adapter_scale", cfg.style_adapter_scale),
        use_spatial_head_fusion=load_cfg.get(
            "use_spatial_head_fusion", cfg.use_spatial_head_fusion
        ),
        spatial_fusion_indices=tuple(
            int(x)
            for x in load_cfg.get("spatial_fusion_indices", cfg.spatial_fusion_indices)
        ),
        spatial_fusion_hidden=load_cfg.get(
            "spatial_fusion_hidden", cfg.spatial_fusion_hidden
        ),
    ).to(device)


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
def _collect_pixels_for_val_best_thr(
    model: torch.nn.Module,
    loader,
    device: str,
    window: int | None,
    stride: int | None,
    smooth_k: int,
    max_pixels: int,
    use_ensemble: bool,
    ensemble_cfg: dict | None,
    tta_mode: str = "none",
    corrupt: str = "none",
    corrupt_pair: str = "correlated",
    corrupt_seed: int = 0,
    gaussian_sigma: float = 0.0,
    bc_brightness: float = 0.0,
    bc_contrast: float = 0.0,
    jpeg_quality: int = 75,
):
    """
    Collect sampled pixels (probability, label) from loader for selecting a single global threshold.
    Returns:
      probs: [N] float32 on device
      labels: [N] float32 on device
    """
    probs_buf = None
    y_buf = None
    max_pixels = int(max(1000, max_pixels))

    for batch in loader:
        img_a = batch["img_a"].to(device, non_blocking=True)
        img_b = batch["img_b"].to(device, non_blocking=True)
        if corrupt and str(corrupt).lower() not in ("none", "off", "0", "false"):
            img_a, img_b = apply_corruption_pair(
                img_a,
                img_b,
                mode=str(corrupt),
                pair_mode=str(corrupt_pair),
                seed=int(corrupt_seed) + int(img_a.shape[0]),
                gaussian_sigma=float(gaussian_sigma),
                bc_brightness=float(bc_brightness),
                bc_contrast=float(bc_contrast),
                jpeg_quality=int(jpeg_quality),
            )
        gt = batch["label"].to(device)
        if gt.ndim == 3:
            gt = gt.unsqueeze(1)
        if gt.ndim == 4 and gt.shape[1] != 1:
            gt = gt[:, :1]
        y = (gt > 0).float()  # [B,1,H,W]

        if window is not None and stride is not None and img_a.shape[0] != 1:
            raise ValueError("val_best with sliding window currently supports batch_size==1.")
        prob = tta_inference_prob(
            model=model,
            img_a=img_a,
            img_b=img_b,
            device=device,
            window=window,
            stride=stride,
            use_ensemble=use_ensemble,
            ensemble_cfg=ensemble_cfg,
            tta_mode=tta_mode,
        )

        if smooth_k and smooth_k > 1:
            pad = smooth_k // 2
            prob = torch.nn.functional.avg_pool2d(prob, kernel_size=smooth_k, stride=1, padding=pad)

        probs_mat = prob[:, 0].reshape(-1)
        y_flat = y.reshape(-1).to(dtype=torch.float32)

        if probs_buf is None:
            probs_buf = probs_mat
            y_buf = y_flat
        else:
            probs_buf = torch.cat([probs_buf, probs_mat], dim=0)
            y_buf = torch.cat([y_buf, y_flat], dim=0)

        if probs_buf.numel() > max_pixels:
            perm = torch.randperm(probs_buf.numel(), device=probs_buf.device)[:max_pixels]
            probs_buf = probs_buf.index_select(0, perm)
            y_buf = y_buf.index_select(0, perm)

    if probs_buf is None or y_buf is None:
        raise RuntimeError("Empty loader; cannot select val_best threshold.")
    return probs_buf.to(dtype=torch.float32), y_buf.to(dtype=torch.float32)


def _best_thr_from_samples(probs: torch.Tensor, labels: torch.Tensor) -> tuple[float, float, int]:
    """
    Find a single threshold that maximizes pixel-wise F1 on sampled data.
    preds are defined as (prob > thr).
    Returns (thr, best_f1, k_pred_pos).
    """
    if probs.ndim != 1 or labels.ndim != 1 or probs.numel() != labels.numel():
        raise ValueError("probs/labels must be 1D and same length")
    if probs.numel() == 0:
        return 0.5, 0.0, 0

    mask = torch.isfinite(probs)
    probs = probs[mask]
    labels = labels[mask]
    if probs.numel() == 0:
        return 0.5, 0.0, 0

    total_pos = float(labels.sum().item())
    if total_pos <= 0:
        return 1.0, 0.0, 0

    order = torch.argsort(probs, descending=True)
    p = probs.index_select(0, order)
    y = labels.index_select(0, order)

    tp = torch.cumsum(y, dim=0)
    k = torch.arange(1, tp.numel() + 1, device=tp.device, dtype=tp.dtype)
    fp = k - tp
    fn = total_pos - tp
    denom = (2 * tp + fp + fn).clamp_min(1e-12)
    f1 = (2 * tp) / denom
    best_i = int(torch.argmax(f1).item())
    best_f1 = float(f1[best_i].item())
    k_pos = int(k[best_i].item())

    # Choose threshold between p[best_i] and p[best_i+1] so that exactly k_pos are > thr (ignoring ties).
    if best_i >= p.numel() - 1:
        thr = float(p[best_i].item()) - 1e-6
    else:
        thr = 0.5 * (float(p[best_i].item()) + float(p[best_i + 1].item()))
    thr = float(max(0.0, min(1.0, thr)))
    return thr, best_f1, k_pos


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


@torch.no_grad()
def profile_complexity(
    model: torch.nn.Module,
    device: str,
    input_size: int,
    warmup: int,
    iters: int,
    window: int | None,
    stride: int | None,
    use_ensemble: bool,
    ensemble_cfg: dict | None,
    tta_mode: str,
) -> dict:
    model.eval()
    input_size = int(max(16, input_size))
    warmup = int(max(0, warmup))
    iters = int(max(1, iters))
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    img_a = torch.randn(1, 3, input_size, input_size, device=device)
    img_b = torch.randn(1, 3, input_size, input_size, device=device)
    prof_window = window
    prof_stride = stride
    if prof_window is not None and int(prof_window) >= input_size and prof_stride is not None:
        prof_window = input_size
        prof_stride = input_size

    for _ in range(warmup):
        _ = tta_inference_prob(
            model=model,
            img_a=img_a,
            img_b=img_b,
            device=device,
            window=prof_window,
            stride=prof_stride,
            use_ensemble=use_ensemble,
            ensemble_cfg=ensemble_cfg,
            tta_mode=tta_mode,
        )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = tta_inference_prob(
            model=model,
            img_a=img_a,
            img_b=img_b,
            device=device,
            window=prof_window,
            stride=prof_stride,
            use_ensemble=use_ensemble,
            ensemble_cfg=ensemble_cfg,
            tta_mode=tta_mode,
        )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) * 1000.0 / float(iters)

    flops = None
    flop_note = "torch.profiler with_flops=True; unsupported ops may be omitted."
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.startswith("cuda"):
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, with_flops=True) as prof:
            _ = tta_inference_prob(
                model=model,
                img_a=img_a,
                img_b=img_b,
                device=device,
                window=prof_window,
                stride=prof_stride,
                use_ensemble=use_ensemble,
                ensemble_cfg=ensemble_cfg,
                tta_mode=tta_mode,
            )
        flops = int(sum(getattr(evt, "flops", 0) or 0 for evt in prof.key_averages()))
    except Exception as e:
        flop_note = f"FLOPs profiling failed: {e}"

    return {
        "params": int(total_params),
        "params_m": float(total_params / 1e6),
        "trainable_params": int(trainable_params),
        "trainable_params_m": float(trainable_params / 1e6),
        "flops": flops,
        "flops_g": None if flops is None else float(flops / 1e9),
        "latency_ms_per_pair": float(latency_ms),
        "fps_pairs": float(1000.0 / latency_ms) if latency_ms > 0 else None,
        "profile_input_size": int(input_size),
        "profile_warmup": int(warmup),
        "profile_iters": int(iters),
        "profile_window": None if prof_window is None else int(prof_window),
        "profile_stride": None if prof_stride is None else int(prof_stride),
        "profile_tta": str(tta_mode),
        "profile_use_ensemble": bool(use_ensemble),
        "profile_ensemble_cfg": ensemble_cfg,
        "note": flop_note,
    }


def main():
    args, cfg = parse_args()
    seed_everything(cfg.seed)
    device = cfg.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable, fallback to CPU")
        device = "cpu"
        cfg.device = device
    cfg.batch_size = args.batch_size
    cfg.num_workers = int(getattr(args, "num_workers", cfg.num_workers))
    # eval loaders: use --data_root
    _, val_loader, test_loader = build_dataloaders(
        cfg,
        require_train=False,
        require_val=False,
        load_val=not args.strict_zeroshot,
    )
    if val_loader is None and not args.strict_zeroshot:
        print("[Data] No val split found; using test split as val for calibration/visualization.")
        val_loader = test_loader
    # calibration loaders: optionally use --calib_root (e.g., source domain)
    val_loader_calib = val_loader
    if args.calib_root:
        cfg_calib = HeadCfg(
            data_root=args.calib_root,
            out_dir=cfg.out_dir,
            device=cfg.device,
            num_workers=cfg.num_workers,
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
            _, val_loader_calib, test_loader_calib = build_dataloaders(
                cfg_calib,
                require_train=False,
                require_val=args.strict_zeroshot,
            )
            if val_loader_calib is None:
                val_loader_calib = test_loader_calib
                print("[Calib] No val split found under calib_root; using test split for calibration.")
            print(f"[Calib] Using calib_root={args.calib_root} (val size={len(val_loader_calib.dataset)})")
        except Exception as e:
            if args.strict_zeroshot:
                raise RuntimeError(
                    f"Strict zero-shot calibration loader failed for {args.calib_root}: {e}"
                ) from e
            print(f"[Calib] Failed to build calib loaders from {args.calib_root}: {e}; fallback to --data_root val.")
            val_loader_calib = val_loader
    ckpt = torch.load(args.checkpoint, map_location=device)
    load_cfg = ckpt.get("cfg")
    if isinstance(load_cfg, dict):
        cfg.arch = load_cfg.get("arch", getattr(cfg, "arch", "dlv"))
        cfg.a0_layer = load_cfg.get("a0_layer", getattr(cfg, "a0_layer", 12))
        cfg.fuse_mode = load_cfg.get("fuse_mode", cfg.fuse_mode)
        cfg.align_radius = load_cfg.get("align_radius", cfg.align_radius)
        cfg.align_temperature = load_cfg.get("align_temperature", cfg.align_temperature)
        cfg.learnable_align_blend = load_cfg.get(
            "learnable_align_blend", cfg.learnable_align_blend
        )
        cfg.use_nuisance_gate = load_cfg.get("use_nuisance_gate", cfg.use_nuisance_gate)
        cfg.nuisance_hidden = load_cfg.get("nuisance_hidden", cfg.nuisance_hidden)
        cfg.nuisance_gate_weight = load_cfg.get("nuisance_gate_weight", cfg.nuisance_gate_weight)
        cfg.nuisance_use_image_cues = load_cfg.get("nuisance_use_image_cues", cfg.nuisance_use_image_cues)
        cfg.use_residual_style_adapter = load_cfg.get(
            "use_residual_style_adapter", cfg.use_residual_style_adapter
        )
        cfg.style_adapter_hidden = load_cfg.get(
            "style_adapter_hidden", cfg.style_adapter_hidden
        )
        cfg.style_adapter_scale = load_cfg.get(
            "style_adapter_scale", cfg.style_adapter_scale
        )
        cfg.use_layer_ensemble = load_cfg.get("use_layer_ensemble", cfg.use_layer_ensemble)
        cfg.layer_head_ch = load_cfg.get("layer_head_ch", cfg.layer_head_ch)
        cfg.use_spatial_head_fusion = load_cfg.get(
            "use_spatial_head_fusion", cfg.use_spatial_head_fusion
        )
        cfg.spatial_fusion_hidden = load_cfg.get(
            "spatial_fusion_hidden", cfg.spatial_fusion_hidden
        )
        if load_cfg.get("spatial_fusion_indices") is not None:
            cfg.spatial_fusion_indices = tuple(
                int(x) for x in load_cfg["spatial_fusion_indices"]
            )
        if load_cfg.get("selected_layers") is not None:
            cfg.selected_layers = tuple(int(x) for x in load_cfg["selected_layers"])

    load_cfg = load_cfg if isinstance(load_cfg, dict) else {}
    model = _build_model_from_checkpoint_cfg(load_cfg, cfg, device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    print(f"Loaded checkpoint from {args.checkpoint}")
    model_checkpoint_paths = [args.checkpoint]
    model_ensemble_weights = None
    if args.model_ensemble_checkpoints:
        models = [model]
        for checkpoint_path in args.model_ensemble_checkpoints:
            extra_ckpt = torch.load(checkpoint_path, map_location=device)
            extra_cfg = extra_ckpt.get("cfg") if isinstance(extra_ckpt, dict) else None
            extra_cfg = extra_cfg if isinstance(extra_cfg, dict) else {}
            extra_model = _build_model_from_checkpoint_cfg(extra_cfg, cfg, device)
            extra_model.load_state_dict(extra_ckpt["model"] if "model" in extra_ckpt else extra_ckpt)
            models.append(extra_model)
            model_checkpoint_paths.append(checkpoint_path)
            print(f"Loaded ensemble checkpoint from {checkpoint_path}")
        if args.model_ensemble_weights:
            model_ensemble_weights = [
                float(x) for x in args.model_ensemble_weights.split(",") if x.strip()
            ]
        model = ProbabilityCheckpointEnsemble(models, weights=model_ensemble_weights).to(device)
        model_ensemble_weights = model.weights.detach().cpu().tolist()
        print(
            f"[Model Ensemble] Probability fusion over {len(models)} checkpoints "
            f"with weights={model_ensemble_weights}"
        )
    val_size = "not loaded" if val_loader is None else str(len(val_loader.dataset))
    print(f"val/test sizes: {val_size}/{len(test_loader.dataset)}")

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
        if args.ensemble_strategy in ("mean_prob", "mean_logit", "min_prob", "soft_min", "max_rejection"):
            ensemble_cfg = {"mode": args.ensemble_strategy}
            if args.ensemble_strategy in ("soft_min", "max_rejection"):
                if not 0.0 <= float(args.consensus_strength) <= 1.0:
                    raise ValueError("--consensus_strength must be in [0,1]")
                ensemble_cfg["strength"] = float(args.consensus_strength)
            if args.ensemble_indices is not None:
                ensemble_cfg["indices"] = args.ensemble_indices
                print(f"[Ensemble] Using fixed heads indices={args.ensemble_indices} with mode={args.ensemble_strategy}")
        elif args.ensemble_strategy in ("consis2", "consisk"):
            # Consistency-driven dynamic weighting (no labels; no target calibration).
            if not getattr(cfg, "use_layer_ensemble", False):
                print("[Ensemble] use_layer_ensemble=False; cannot use consis2/consisk. Disabling ensemble.")
                args.use_ensemble_pred = False
            else:
                # Default indices:
                # - consis2: [deepest_layer_head, fused]
                # - consisk: all heads
                K = len(cfg.selected_layers) + 1 if isinstance(cfg.selected_layers, (list, tuple)) else None
                if args.ensemble_indices is not None:
                    indices = [int(i) for i in args.ensemble_indices]
                else:
                    if args.ensemble_strategy == "consis2":
                        if K is None:
                            raise ValueError("consis2 requires selected_layers to be known.")
                        indices = [K - 2, K - 1]
                    else:
                        if K is None:
                            raise ValueError("consisk requires selected_layers to be known.")
                        indices = list(range(0, K))

                if args.ensemble_strategy == "consis2":
                    if len(indices) != 2:
                        raise ValueError(f"consis2 requires exactly 2 indices, got {indices}")
                    if K is not None:
                        fused_idx = K - 1
                        if fused_idx in indices and indices[-1] != fused_idx:
                            # reorder to [other, fused] so anchor is fused in dino_head_core
                            other = [i for i in indices if i != fused_idx][0]
                            indices = [other, fused_idx]
                    ensemble_cfg = {
                        "mode": "consis2",
                        "indices": indices,
                        "d0": float(args.consis2_d0),
                        "d1": float(args.consis2_d1),
                        "gamma": float(args.consis2_gamma),
                        "max_w": float(args.consis2_max_w),
                    }
                    print(f"[Ensemble] Using consis2 indices={indices} (anchor=head[1])")
                else:
                    fused_local_idx = -1
                    if K is not None:
                        fused_idx = K - 1
                        if fused_idx in indices:
                            fused_local_idx = int(indices.index(fused_idx))
                    ensemble_cfg = {
                        "mode": "consisk",
                        "indices": indices,
                        "temp0": float(args.consisk_temp0),
                        "temp1": float(args.consisk_temp1),
                        "gamma": float(args.consisk_gamma),
                        "fused_local_idx": int(fused_local_idx),
                        "fused_bias": float(args.consisk_fused_bias),
                    }
                    print(
                        f"[Ensemble] Using consisk indices={indices} (fused_local_idx={fused_local_idx}, "
                        f"temp0={float(args.consisk_temp0):.3g} temp1={float(args.consisk_temp1):.3g})"
                    )
        elif args.ensemble_strategy == "uwi":
            if not getattr(cfg, "use_layer_ensemble", False):
                print("[Ensemble] use_layer_ensemble=False; cannot use uwi. Disabling ensemble.")
                args.use_ensemble_pred = False
            else:
                # Fuse indices: default deepest + fused, or use --ensemble_indices (must be 2).
                K = len(cfg.selected_layers) + 1 if isinstance(cfg.selected_layers, (list, tuple)) else None
                if args.ensemble_indices is not None:
                    fuse_indices = [int(i) for i in args.ensemble_indices]
                else:
                    if K is None:
                        raise ValueError("uwi requires selected_layers to be known.")
                    fuse_indices = [K - 2, K - 1]
                if len(fuse_indices) != 2:
                    raise ValueError(f"uwi requires exactly 2 fuse indices, got {fuse_indices}")
                if K is not None:
                    fused_idx = K - 1
                    if fused_idx in fuse_indices and fuse_indices[-1] != fused_idx:
                        other = [i for i in fuse_indices if i != fused_idx][0]
                        fuse_indices = [other, fused_idx]  # [other, anchor=fused]

                unc_indices = None
                if args.uwi_unc_indices:
                    unc_indices = [int(x) for x in str(args.uwi_unc_indices).split(",") if str(x).strip() != ""]
                elif K is not None:
                    unc_indices = list(range(0, K))

                ensemble_cfg = {
                    "mode": "uw_gate",
                    "fuse_indices": fuse_indices,
                    "unc_indices": unc_indices,
                    "u0": float(args.uwi_u0),
                    "u1": float(args.uwi_u1),
                    "gamma": float(args.uwi_gamma),
                    "max_w": float(args.uwi_max_w),
                    "gate_smooth_k": int(args.uwi_gate_smooth_k),
                }
                print(f"[Ensemble] Using UWI fuse_indices={fuse_indices} unc_indices={unc_indices}")
        else:
            if not getattr(cfg, "use_layer_ensemble", False):
                print("[Ensemble] use_layer_ensemble=False; cannot score heads for topk/weighted/cvx/ugls strategies. Disabling ensemble.")
                args.use_ensemble_pred = False

    if args.use_ensemble_pred and ensemble_cfg is None and args.ensemble_strategy not in ("mean_prob", "mean_logit", "min_prob", "soft_min"):
            print("\n[Ensemble] Scoring each head on VAL to derive selection/weights...")
            score_thr_mode = cfg.thr_mode
            score_thr = cfg.thr
            if score_thr_mode == "val_best":
                score_thr_mode = "fixed"
                score_thr = float(args.thr)
            head_metrics = score_heads_on_loader(
                model=model,
                loader=val_loader_calib,
                device=device,
                thr_mode=score_thr_mode,
                thr=score_thr,
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
            elif args.ensemble_strategy == "ugls":
                # UGLS uses a ranked list of candidate heads.
                # Default: pick top-(ensemble_topk) layer heads by VAL F1 (excluding fused), then add fused.
                layer_indices = list(range(0, max(0, fused_idx)))
                if args.ensemble_topk and args.ensemble_topk > 0 and layer_indices:
                    layer_sorted = sorted(layer_indices, key=lambda i: f1s[i], reverse=True)
                    picked_layers = layer_sorted[: min(args.ensemble_topk, len(layer_sorted))]
                else:
                    picked_layers = layer_indices
                indices = picked_layers + [fused_idx]
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
            elif args.ensemble_strategy == "ugls":
                ranked = sorted(indices, key=lambda i: f1s[int(i)], reverse=True)
                max_k = int(args.ugls_max_k) if int(args.ugls_max_k) > 0 else len(ranked)
                max_k = max(1, min(max_k, len(ranked)))
                min_k = max(1, min(int(args.ugls_min_k), max_k))
                ensemble_cfg = {
                    "mode": "ugls",
                    "indices": ranked,
                    "min_k": int(min_k),
                    "max_k": int(max_k),
                    "unc_power": float(args.ugls_unc_power),
                }
                indices = ranked
                print(f"[Ensemble] Using UGLS ranked indices={ranked}")
                print(f"[Ensemble] UGLS min_k={min_k} max_k={max_k} unc_power={float(args.ugls_unc_power):.3g}")
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

    if args.thr_mode == "val_best":
        probs, y = _collect_pixels_for_val_best_thr(
            model=model,
            loader=val_loader_calib,
            device=device,
            window=args.window,
            stride=args.stride,
            smooth_k=cfg.smooth_k,
            max_pixels=args.val_best_max_pixels,
            use_ensemble=args.use_ensemble_pred,
            ensemble_cfg=ensemble_cfg,
            tta_mode=str(args.tta),
            corrupt=str(args.corrupt),
            corrupt_pair=str(args.corrupt_pair),
            corrupt_seed=int(args.corrupt_seed),
            gaussian_sigma=float(args.gaussian_sigma),
            bc_brightness=float(args.bc_brightness),
            bc_contrast=float(args.bc_contrast),
            jpeg_quality=int(args.jpeg_quality),
        )
        best_thr, best_f1, k_pos = _best_thr_from_samples(probs, y)
        print(f"[val_best] Selected global threshold thr={best_thr:.4f} (sampled_pixels={int(probs.numel())}, pred_pos={k_pos}, best_F1={best_f1:.4f})")
        cfg.thr_mode = "fixed"
        cfg.thr = float(best_thr)

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
        tta_mode=str(args.tta),
        corrupt=str(args.corrupt),
        corrupt_pair=str(args.corrupt_pair),
        corrupt_seed=int(args.corrupt_seed),
        gaussian_sigma=float(args.gaussian_sigma),
        bc_brightness=float(args.bc_brightness),
        bc_contrast=float(args.bc_contrast),
        jpeg_quality=int(args.jpeg_quality),
    )
    print("\n====== Test Metrics ======")
    for k in ["precision", "recall", "f1", "iou", "oa", "kappa"]:
        print(f"{k}: {metrics[k]:.4f}")
    print(f"TP={metrics['TP']} FP={metrics['FP']} FN={metrics['FN']} TN={metrics['TN']}")
    complexity = None
    if args.profile_complexity:
        print("\n====== Complexity ======")
        complexity = profile_complexity(
            model=model,
            device=device,
            input_size=int(args.profile_size),
            warmup=int(args.profile_warmup),
            iters=int(args.profile_iters),
            window=args.window,
            stride=args.stride,
            use_ensemble=args.use_ensemble_pred,
            ensemble_cfg=ensemble_cfg,
            tta_mode=str(args.tta),
        )
        print(f"Params: {complexity['params_m']:.2f}M")
        if complexity["flops_g"] is not None:
            print(f"FLOPs: {complexity['flops_g']:.2f}G @ {complexity['profile_input_size']}x{complexity['profile_input_size']}")
        else:
            print("FLOPs: unavailable")
        print(f"Latency: {complexity['latency_ms_per_pair']:.2f} ms/pair")
        print(f"FPS: {complexity['fps_pairs']:.2f} pairs/s")
    out_dir = cfg.out_dir
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "eval_results.json")
    payload = {
        "split": "test",
        **metrics,
        "cfg": asdict(cfg),
        "protocol": {
            "strict_zeroshot": bool(args.strict_zeroshot),
            "exploratory_target_tuned": bool(args.exploratory_target_tuned),
            "uses_target_labels_for_fusion_selection": bool(
                args.exploratory_target_tuned
            ),
            "calib_root": args.calib_root,
            "target_val_loaded": val_loader is not None,
            "ensemble_cfg": ensemble_cfg,
            "model_ensemble_checkpoints": model_checkpoint_paths,
            "model_ensemble_weights": model_ensemble_weights,
            "checkpoint_test_time_adapted": bool(load_cfg.get("test_time_adapted", False)),
            "checkpoint_test_time_adapt_method": load_cfg.get("test_time_adapt_method"),
            "checkpoint_test_time_adapt_uses_target_labels": load_cfg.get(
                "test_time_adapt_uses_target_labels"
            ),
        },
    }
    if complexity is not None:
        payload["complexity"] = complexity
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
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
            window=args.window,
            stride=args.stride,
            use_ensemble=args.use_ensemble_pred,
            ensemble_cfg=ensemble_cfg,
            tta_mode=str(args.tta),
            corrupt=str(args.corrupt),
            corrupt_pair=str(args.corrupt_pair),
            corrupt_seed=int(args.corrupt_seed),
            gaussian_sigma=float(args.gaussian_sigma),
            bc_brightness=float(args.bc_brightness),
            bc_contrast=float(args.bc_contrast),
            jpeg_quality=int(args.jpeg_quality),
        )
        print(f"Saved visualizations to {vis_dir}")


if __name__ == "__main__":
    main()
