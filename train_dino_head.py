"""
python train_dino_head.py --data_root data/WHUCD --out_dir outputs/dino_head_cd --device auto --epochs 100 --batch_size 8 --crop_size 256 --bce_weight 0.5 --dice_weight 0.5 --thr_mode fixed --thr 0.5
--boundary_dim 192 --boundary_weight 0.5 --boundary_dilation 3
"""

import argparse
import inspect
import json
import os
import time
from dataclasses import asdict
from dataclasses import fields

import torch

from dino_head_core import (
    HeadCfg,
    DinoSiameseHead,
    DinoFrozenA0Head,
    build_dataloaders,
    seed_everything,
    ensure_dir,
    train_one_epoch,
    evaluate,
    save_vis_samples,
    build_scheduler,  # ← 这一行补上
)

def _load_cfg_defaults_from_json(path: str) -> HeadCfg:
    base = HeadCfg()
    if not path:
        return base
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load --config JSON: {path}: {e}")
    if not isinstance(raw, dict):
        raise RuntimeError(f"--config must be a JSON object: {path}")
    keys = {f.name for f in fields(HeadCfg)}
    filtered = {k: v for k, v in raw.items() if k in keys}
    return HeadCfg(**filtered)

def _get_backbone_blocks(backbone):
    """
    Best-effort access to the ViT block list for (HF) DINOv3 or (hub) DINOv2.
    Returns a list of blocks, or None if unknown.
    """
    candidates = ["encoder.layers", "encoder.layer", "layers", "blocks"]
    for path in candidates:
        obj = backbone
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok and isinstance(obj, (list, tuple, torch.nn.ModuleList)):
            return list(obj)
    return None


def _configure_backbone_finetune(model: torch.nn.Module, ft_mode: str, ft_k: int) -> int:
    """
    Configure which backbone blocks are trainable.
    Returns number of trainable backbone parameters.
    """
    if not hasattr(model, "backbone"):
        return 0
    ft_mode = str(ft_mode).lower()
    ft_k = int(ft_k)

    for p in model.backbone.parameters():
        p.requires_grad = False

    if ft_mode == "frozen":
        return 0
    if ft_mode == "full":
        for p in model.backbone.parameters():
            p.requires_grad = True
        return sum(int(p.requires_grad) for p in model.backbone.parameters())

    blocks = _get_backbone_blocks(model.backbone)
    if not blocks:
        print("[FT] Warning: unable to locate backbone blocks; falling back to full fine-tune.")
        for p in model.backbone.parameters():
            p.requires_grad = True
        return sum(int(p.requires_grad) for p in model.backbone.parameters())

    n = len(blocks)
    k = max(0, min(ft_k, n))
    if ft_mode == "shallow":
        idxs = range(0, k)
    elif ft_mode == "deep":
        idxs = range(n - k, n)
    else:
        raise ValueError(f"Unknown ft_mode: {ft_mode} (expected frozen|shallow|deep|full)")

    for i in idxs:
        for p in blocks[i].parameters():
            p.requires_grad = True
    return sum(int(p.requires_grad) for p in model.backbone.parameters())


def _checkpoint_selection_score(metrics: dict, metric: str, beta: float) -> float:
    metric = str(metric).lower()
    if metric == "f1":
        return float(metrics["f1"])
    if metric == "precision":
        return float(metrics["precision"])
    if metric == "fbeta":
        beta = float(beta)
        if beta <= 0:
            raise ValueError("selection_beta must be > 0")
        precision = float(metrics["precision"])
        recall = float(metrics["recall"])
        beta2 = beta * beta
        return (1.0 + beta2) * precision * recall / (beta2 * precision + recall + 1e-8)
    raise ValueError(f"Unknown selection_metric={metric!r}")


def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None, help="Path to a JSON config (e.g., out_dir/config.json) to use as defaults.")
    pre_args, _ = pre.parse_known_args()
    base = _load_cfg_defaults_from_json(str(pre_args.config)) if pre_args.config else HeadCfg()

    parser = argparse.ArgumentParser(description="Train DINOv2 change-detection head")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON config to use as defaults.")
    parser.add_argument("--data_root", type=str, default=base.data_root)
    parser.add_argument("--out_dir", type=str, default=base.out_dir)
    parser.add_argument("--device", type=str, default=base.device, help="cuda|cpu|auto")
    parser.add_argument("--seed", type=int, default=base.seed)
    parser.add_argument("--epochs", type=int, default=base.epochs)
    parser.add_argument("--batch_size", type=int, default=base.batch_size)
    parser.add_argument("--num_workers", type=int, default=base.num_workers)
    parser.add_argument("--crop_size", type=int, default=base.crop_size)
    parser.add_argument("--lr", type=float, default=base.lr)
    parser.add_argument("--weight_decay", type=float, default=base.weight_decay)
    parser.add_argument("--grad_accum", type=int, default=base.grad_accum)
    parser.add_argument("--scheduler", type=str, choices=["cosine", "none"], default=base.scheduler)
    parser.add_argument("--warmup_epochs", type=int, default=base.warmup_epochs)
    parser.add_argument("--min_lr", type=float, default=base.min_lr)
    parser.add_argument("--bce_weight", type=float, default=base.bce_weight)
    parser.add_argument("--dice_weight", type=float, default=base.dice_weight)
    parser.add_argument("--tversky_weight", type=float, default=base.tversky_weight, help="FP/FN-asymmetric Tversky loss weight")
    parser.add_argument("--tversky_alpha", type=float, default=base.tversky_alpha, help="Tversky FP penalty")
    parser.add_argument("--tversky_beta", type=float, default=base.tversky_beta, help="Tversky FN penalty")
    parser.add_argument("--fp_penalty_weight", type=float, default=base.fp_penalty_weight, help="Direct penalty on predicted probability over negative pixels")
    parser.add_argument("--hard_negative_weight", type=float, default=base.hard_negative_weight, help="OHEM weight on highest-loss unchanged pixels")
    parser.add_argument("--hard_negative_ratio", type=float, default=base.hard_negative_ratio, help="Fraction of unchanged pixels retained by OHEM")
    parser.add_argument("--boundary_weight", type=float, default=base.boundary_weight, help="aux boundary loss weight")
    parser.add_argument("--boundary_dilation", type=int, default=base.boundary_dilation, help="boundary thickness (px) for supervision")
    parser.add_argument("--lambda_consis", type=float, default=base.lambda_consis, help="counterfactual consistency weight")
    parser.add_argument("--lambda_domain", type=float, default=base.lambda_domain, help="domain confusion weight")
    parser.add_argument("--self_sup_weight", type=float, default=base.self_sup_weight, help="aux supervised weight on perturbed view")
    parser.add_argument("--style_aug_prob", type=float, default=base.style_aug_prob, help="probability to apply style perturbation")
    parser.add_argument("--style_aug_sigma", type=float, default=base.style_aug_sigma, help="noise scale for style perturbation")
    parser.add_argument("--style_blur_prob", type=float, default=base.style_blur_prob, help="blur probability for style perturbation")
    parser.add_argument(
        "--identity_neg_weight",
        type=float,
        default=base.identity_neg_weight,
        help="Loss weight for same-image appearance-perturbed no-change pairs.",
    )
    parser.add_argument(
        "--identity_neg_prob",
        type=float,
        default=base.identity_neg_prob,
        help="Per-batch probability of applying the identity no-change branch.",
    )
    parser.add_argument(
        "--contrastive_weight",
        type=float,
        default=base.contrastive_weight,
        help="Weight for dense temporal contrastive supervision on adapter features.",
    )
    parser.add_argument(
        "--contrastive_margin",
        type=float,
        default=base.contrastive_margin,
        help="Maximum desired cosine similarity for changed feature pairs.",
    )
    parser.add_argument(
        "--contrastive_radius",
        type=int,
        default=base.contrastive_radius,
        help="Local correspondence radius in backbone feature pixels.",
    )
    parser.add_argument("--nuisance_real_weight", type=float, default=base.nuisance_real_weight)
    parser.add_argument("--nuisance_synth_weight", type=float, default=base.nuisance_synth_weight)
    parser.add_argument("--nuisance_synth_prob", type=float, default=base.nuisance_synth_prob)
    parser.add_argument("--nuisance_illumination", type=float, default=base.nuisance_illumination)
    parser.add_argument("--nuisance_max_shift", type=float, default=base.nuisance_max_shift)
    parser.add_argument("--nuisance_pair_aug_weight", type=float, default=base.nuisance_pair_aug_weight, help="Supervised loss weight for independently photometrically perturbed real pairs")
    parser.add_argument("--nuisance_pair_aug_prob", type=float, default=base.nuisance_pair_aug_prob, help="Per-batch probability of the supervised nuisance pair branch")
    parser.add_argument("--nuisance_pair_consistency", type=float, default=base.nuisance_pair_consistency, help="Clean/perturbed prediction consistency weight")
    parser.add_argument("--nuisance_feature_consistency", type=float, default=base.nuisance_feature_consistency, help="Adapter feature invariance weight between clean and perturbed real pairs")
    parser.add_argument("--nuisance_clean_feature_preservation", type=float, default=base.nuisance_clean_feature_preservation, help="Preserve clean residual-adapter features relative to the frozen base path")
    parser.add_argument("--registration_neg_weight", type=float, default=base.registration_neg_weight, help="No-change loss weight for same-image translated pairs")
    parser.add_argument("--registration_neg_prob", type=float, default=base.registration_neg_prob, help="Per-batch probability of translated no-change supervision")
    parser.add_argument("--registration_max_shift", type=float, default=base.registration_max_shift, help="Maximum synthetic translation in input pixels")
    parser.add_argument("--alignment_clean_preservation", type=float, default=base.alignment_clean_preservation, help="Preserve the unaligned comparison path on clean source pairs")
    parser.add_argument("--spatial_fusion_uniform_weight", type=float, default=base.spatial_fusion_uniform_weight, help="Regularize spatial fusion weights toward their uniform initialization")
    parser.add_argument("--head_aux_weight", type=float, default=base.head_aux_weight, help="aux supervision weight for layer heads")
    parser.add_argument("--head_cons_weight", type=float, default=base.head_cons_weight, help="consistency weight across layer heads")
    parser.add_argument("--eval_crop", type=int, default=base.eval_crop)
    parser.add_argument("--window", type=int, default=base.eval_window, help="滑窗窗口（默认不用滑窗）")
    parser.add_argument("--stride", type=int, default=base.eval_stride, help="滑窗步长（需与window同时设置）")
    parser.add_argument("--thr_mode", type=str, choices=["fixed", "topk", "otsu"], default=base.thr_mode)
    parser.add_argument("--thr", type=float, default=base.thr)
    parser.add_argument("--topk", type=float, default=base.topk)
    parser.add_argument("--smooth_k", type=int, default=base.smooth_k)
    parser.add_argument("--min_area", type=int, default=base.min_area)
    parser.add_argument("--vis_every", type=int, default=base.vis_every)
    parser.add_argument("--vis_n", type=int, default=base.vis_n)
    parser.add_argument("--log_every", type=int, default=base.log_every)
    parser.add_argument("--use_ensemble_pred", action="store_true", default=base.use_ensemble_pred, help="use ensemble mean for eval/vis")
    parser.add_argument("--arch", type=str, choices=["dlv", "a0"], default=base.arch, help="Model variant: dlv (default) | a0 (frozen backbone + single 1x1 head)")
    parser.add_argument("--dino_name", type=str, default=base.dino_name)
    parser.add_argument(
        "--selected_layers",
        type=int,
        nargs="+",
        default=list(base.selected_layers),
        help="1-based transformer block indices to use (e.g., 3 6 9 12 for 12-layer; 6 12 18 24 for 24-layer; 8 16 24 32 for 32-layer)",
    )
    parser.add_argument(
        "--fuse_mode",
        type=str,
        choices=["abs", "norm_abs", "abs+sum", "cat4"],
        default=base.fuse_mode,
    )
    parser.add_argument("--align_radius", type=int, default=base.align_radius, help="Local feature-alignment radius in DINO tokens")
    parser.add_argument("--align_temperature", type=float, default=base.align_temperature, help="Soft-correlation alignment temperature")
    parser.add_argument("--learnable_align_blend", action="store_true", default=base.learnable_align_blend)
    parser.add_argument("--use_nuisance_gate", action="store_true", default=base.use_nuisance_gate)
    parser.add_argument("--nuisance_hidden", type=int, default=base.nuisance_hidden)
    parser.add_argument("--nuisance_gate_weight", type=float, default=base.nuisance_gate_weight)
    parser.add_argument("--nuisance_use_image_cues", action="store_true", default=base.nuisance_use_image_cues)
    parser.add_argument("--train_nuisance_only", action="store_true", default=base.train_nuisance_only)
    parser.add_argument("--train_adapters_only", action="store_true", default=base.train_adapters_only)
    parser.add_argument("--use_residual_style_adapter", action="store_true", default=base.use_residual_style_adapter)
    parser.add_argument("--style_adapter_hidden", type=int, default=base.style_adapter_hidden)
    parser.add_argument("--style_adapter_scale", type=float, default=base.style_adapter_scale)
    parser.add_argument("--train_style_adapter_only", action="store_true", default=base.train_style_adapter_only)
    parser.add_argument("--train_alignment_only", action="store_true", default=base.train_alignment_only)
    parser.add_argument("--use_spatial_head_fusion", action="store_true", default=base.use_spatial_head_fusion)
    parser.add_argument("--spatial_fusion_indices", type=int, nargs="+", default=list(base.spatial_fusion_indices))
    parser.add_argument("--spatial_fusion_hidden", type=int, default=base.spatial_fusion_hidden)
    parser.add_argument("--train_spatial_fusion_only", action="store_true", default=base.train_spatial_fusion_only)
    parser.add_argument("--use_whiten", action="store_true", default=base.use_whiten)
    parser.add_argument("--use_domain_adv", action="store_true", default=base.use_domain_adv)
    parser.add_argument("--domain_hidden", type=int, default=base.domain_hidden)
    parser.add_argument("--domain_grl", type=float, default=base.domain_grl)
    parser.add_argument("--use_style_norm", action="store_true", default=base.use_style_norm)
    parser.add_argument("--proto_path", type=str, default=base.proto_path, help="npy path for prototype vectors [K,C]")
    parser.add_argument("--proto_weight", type=float, default=base.proto_weight, help="weight for prototype change logit")
    parser.add_argument("--boundary_dim", type=int, default=base.boundary_dim, help="embed dim for boundary decoder")
    parser.add_argument("--use_layer_ensemble", action="store_true", default=base.use_layer_ensemble, help="enable layer-wise ensemble heads")
    parser.add_argument("--layer_head_ch", type=int, default=base.layer_head_ch, help="channel width for fused ensemble head")
    parser.add_argument("--a0_layer", type=int, default=base.a0_layer, help="Backbone layer index for A0 baseline (default: 12)")
    parser.add_argument("--ft_mode", type=str, choices=["frozen", "shallow", "deep", "full"], default=base.ft_mode, help="Backbone fine-tune mode (dlv only)")
    parser.add_argument("--ft_k", type=int, default=base.ft_k, help="Number of ViT blocks to unfreeze for shallow/deep")
    parser.add_argument("--backbone_lr", type=float, default=base.backbone_lr, help="Learning rate for unfrozen backbone parameters")
    parser.add_argument("--full_eval", dest="full_eval", action="store_true")
    parser.add_argument("--no_full_eval", dest="full_eval", action="store_false")
    parser.set_defaults(full_eval=base.full_eval)
    parser.add_argument("--use_minarea", dest="use_minarea", action="store_true")
    parser.add_argument("--no_minarea", dest="use_minarea", action="store_false")
    parser.set_defaults(use_minarea=base.use_minarea)
    parser.add_argument("--save_best", dest="save_best", action="store_true")
    parser.add_argument("--no_save_best", dest="save_best", action="store_false")
    parser.set_defaults(save_best=base.save_best)
    parser.add_argument("--save_last", dest="save_last", action="store_true")
    parser.add_argument("--no_save_last", dest="save_last", action="store_false")
    parser.set_defaults(save_last=base.save_last)
    parser.add_argument(
        "--selection_metric",
        type=str,
        choices=["f1", "fbeta", "precision"],
        default=base.selection_metric,
        help="Source-validation metric used to select best.pt.",
    )
    parser.add_argument(
        "--selection_beta",
        type=float,
        default=base.selection_beta,
        help="Beta for --selection_metric fbeta; beta < 1 favors precision.",
    )
    parser.add_argument("--resume", type=str, default=None, help="断点续训 ckpt 路径（last.pt/best.pt）")
    parser.add_argument("--eval_every", type=int, default=1, help="每多少个 epoch 验证一次（默认每个 epoch）")
    parser.add_argument(
        "--reset_best_on_resume",
        action="store_true",
        help="Reset inherited best score when changing the resumed objective or architecture.",
    )
    parser.add_argument(
        "--reset_optimizer_on_resume",
        action="store_true",
        help="Resume model weights but keep the newly configured optimizer and scheduler.",
    )
    args = parser.parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = HeadCfg(
        data_root=args.data_root,
        out_dir=args.out_dir,
        seed=args.seed,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        crop_size=args.crop_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_accum=args.grad_accum,
        scheduler=args.scheduler,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        tversky_weight=args.tversky_weight,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
        fp_penalty_weight=args.fp_penalty_weight,
        hard_negative_weight=args.hard_negative_weight,
        hard_negative_ratio=args.hard_negative_ratio,
        boundary_weight=args.boundary_weight,
        boundary_dilation=args.boundary_dilation,
        lambda_consis=args.lambda_consis,
        lambda_domain=args.lambda_domain,
        self_sup_weight=args.self_sup_weight,
        style_aug_prob=args.style_aug_prob,
        style_aug_sigma=args.style_aug_sigma,
        style_blur_prob=args.style_blur_prob,
        identity_neg_weight=args.identity_neg_weight,
        identity_neg_prob=args.identity_neg_prob,
        contrastive_weight=args.contrastive_weight,
        contrastive_margin=args.contrastive_margin,
        contrastive_radius=args.contrastive_radius,
        nuisance_real_weight=args.nuisance_real_weight,
        nuisance_synth_weight=args.nuisance_synth_weight,
        nuisance_synth_prob=args.nuisance_synth_prob,
        nuisance_illumination=args.nuisance_illumination,
        nuisance_max_shift=args.nuisance_max_shift,
        nuisance_pair_aug_weight=args.nuisance_pair_aug_weight,
        nuisance_pair_aug_prob=args.nuisance_pair_aug_prob,
        nuisance_pair_consistency=args.nuisance_pair_consistency,
        nuisance_feature_consistency=args.nuisance_feature_consistency,
        nuisance_clean_feature_preservation=args.nuisance_clean_feature_preservation,
        registration_neg_weight=args.registration_neg_weight,
        registration_neg_prob=args.registration_neg_prob,
        registration_max_shift=args.registration_max_shift,
        alignment_clean_preservation=args.alignment_clean_preservation,
        spatial_fusion_uniform_weight=args.spatial_fusion_uniform_weight,
        head_aux_weight=args.head_aux_weight,
        head_cons_weight=args.head_cons_weight,
        full_eval=args.full_eval,
        eval_crop=args.eval_crop,
        eval_window=args.window,
        eval_stride=args.stride,
        thr_mode=args.thr_mode,
        thr=args.thr,
        topk=args.topk,
        smooth_k=args.smooth_k,
        use_minarea=args.use_minarea,
        min_area=args.min_area,
        use_ensemble_pred=args.use_ensemble_pred,
        arch=args.arch,
        dino_name=args.dino_name,
        selected_layers=tuple(int(x) for x in args.selected_layers),
        fuse_mode=args.fuse_mode,
        align_radius=args.align_radius,
        align_temperature=args.align_temperature,
        learnable_align_blend=args.learnable_align_blend,
        use_nuisance_gate=args.use_nuisance_gate,
        nuisance_hidden=args.nuisance_hidden,
        nuisance_gate_weight=args.nuisance_gate_weight,
        nuisance_use_image_cues=args.nuisance_use_image_cues,
        train_nuisance_only=args.train_nuisance_only,
        train_adapters_only=args.train_adapters_only,
        use_residual_style_adapter=args.use_residual_style_adapter,
        style_adapter_hidden=args.style_adapter_hidden,
        style_adapter_scale=args.style_adapter_scale,
        train_style_adapter_only=args.train_style_adapter_only,
        train_alignment_only=args.train_alignment_only,
        use_spatial_head_fusion=args.use_spatial_head_fusion,
        spatial_fusion_indices=tuple(int(x) for x in args.spatial_fusion_indices),
        spatial_fusion_hidden=args.spatial_fusion_hidden,
        train_spatial_fusion_only=args.train_spatial_fusion_only,
        use_whiten=args.use_whiten,
        use_domain_adv=args.use_domain_adv,
        domain_hidden=args.domain_hidden,
        domain_grl=args.domain_grl,
        use_style_norm=args.use_style_norm,
        proto_path=args.proto_path,
        proto_weight=args.proto_weight,
        boundary_dim=args.boundary_dim,
        use_layer_ensemble=args.use_layer_ensemble,
        layer_head_ch=args.layer_head_ch,
        a0_layer=args.a0_layer,
        ft_mode=args.ft_mode,
        ft_k=args.ft_k,
        backbone_lr=args.backbone_lr,
        save_best=args.save_best,
        save_last=args.save_last,
        selection_metric=args.selection_metric,
        selection_beta=args.selection_beta,
        vis_every=args.vis_every,
        vis_n=args.vis_n,
        log_every=args.log_every,
    )
    return (
        cfg,
        args.resume,
        args.eval_every,
        args.reset_best_on_resume,
        args.reset_optimizer_on_resume,
    )


def main():
    (
        cfg,
        resume_ckpt,
        eval_every,
        reset_best_on_resume,
        reset_optimizer_on_resume,
    ) = parse_args()
    seed_everything(cfg.seed)
    device = cfg.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable, fallback to CPU")
        device = "cpu"
        cfg.device = device
    ensure_dir(cfg.out_dir)
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    with open(os.path.join(cfg.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    if cfg.arch == "a0":
        if DinoFrozenA0Head is None:
            raise ImportError(
                "DinoFrozenA0Head is not available. Please update models/dinov2_head.py to a version that defines DinoFrozenA0Head, "
                "or use --arch dlv."
            )
        if cfg.use_layer_ensemble or cfg.boundary_dim or cfg.use_domain_adv or cfg.use_style_norm or cfg.proto_weight:
            print("[A0] Note: ignoring DLF/MHE/auxiliary modules; using frozen backbone + single 1x1 head only.")
        model = DinoFrozenA0Head(
            dino_name=cfg.dino_name,
            layer=cfg.a0_layer,
            use_whiten=cfg.use_whiten,
        ).to(device)
    else:
        head_kwargs = dict(
            dino_name=cfg.dino_name,
            selected_layers=cfg.selected_layers,
            fuse_mode=cfg.fuse_mode,
            align_radius=cfg.align_radius,
            align_temperature=cfg.align_temperature,
            learnable_align_blend=cfg.learnable_align_blend,
            use_spatial_head_fusion=cfg.use_spatial_head_fusion,
            spatial_fusion_indices=cfg.spatial_fusion_indices,
            spatial_fusion_hidden=cfg.spatial_fusion_hidden,
            use_nuisance_gate=cfg.use_nuisance_gate,
            nuisance_hidden=cfg.nuisance_hidden,
            nuisance_gate_weight=cfg.nuisance_gate_weight,
            nuisance_use_image_cues=cfg.nuisance_use_image_cues,
            use_residual_style_adapter=cfg.use_residual_style_adapter,
            style_adapter_hidden=cfg.style_adapter_hidden,
            style_adapter_scale=cfg.style_adapter_scale,
            use_whiten=cfg.use_whiten,
            backbone_grad=(cfg.ft_mode != "frozen"),
            use_domain_adv=cfg.use_domain_adv,
            domain_hidden=cfg.domain_hidden,
            domain_grl=cfg.domain_grl,
            use_style_norm=cfg.use_style_norm,
            proto_path=cfg.proto_path,
            proto_weight=cfg.proto_weight,
            boundary_dim=cfg.boundary_dim,
            use_layer_ensemble=cfg.use_layer_ensemble,
            layer_head_ch=cfg.layer_head_ch,
        )
        sig = inspect.signature(DinoSiameseHead.__init__)
        head_kwargs = {k: v for k, v in head_kwargs.items() if k in sig.parameters}
        if (cfg.ft_mode != "frozen") and ("backbone_grad" not in sig.parameters):
            print("[FT] Warning: this DinoSiameseHead version has no backbone_grad; backbone fine-tuning may not work as expected.")
        model = DinoSiameseHead(**head_kwargs).to(device)

        n_trainable_bb = _configure_backbone_finetune(model, cfg.ft_mode, cfg.ft_k)
        if cfg.ft_mode != "frozen":
            print(
                f"[FT] mode={cfg.ft_mode} ft_k={cfg.ft_k} backbone_lr={cfg.backbone_lr} "
                f"trainable_backbone_params={n_trainable_bb}"
            )

    train_only_modes = sum(
        int(flag)
        for flag in (
            cfg.train_nuisance_only,
            cfg.train_adapters_only,
            cfg.train_style_adapter_only,
            cfg.train_alignment_only,
            cfg.train_spatial_fusion_only,
        )
    )
    if train_only_modes > 1:
        raise ValueError("Only one train-*-only mode may be enabled")
    if cfg.train_nuisance_only:
        for p in model.parameters():
            p.requires_grad = False
        if getattr(model, "nuisance_head", None) is None:
            raise ValueError("--train_nuisance_only requires --use_nuisance_gate")
        for p in model.nuisance_head.parameters():
            p.requires_grad = True
        print("[Nuisance] Frozen base model; training nuisance_head only.")
    elif cfg.train_adapters_only:
        for p in model.parameters():
            p.requires_grad = False
        if not hasattr(model, "adapters"):
            raise ValueError("--train_adapters_only requires a model with adapters")
        for p in model.adapters.parameters():
            p.requires_grad = True
        print("[Adapters] Frozen backbone and output heads; training adapters only.")
    elif cfg.train_style_adapter_only:
        for p in model.parameters():
            p.requires_grad = False
        if getattr(model, "style_adapters", None) is None:
            raise ValueError(
                "--train_style_adapter_only requires --use_residual_style_adapter"
            )
        for p in model.style_adapters.parameters():
            p.requires_grad = True
        print("[Style] Frozen source model; training residual style adapters only.")
    elif cfg.train_alignment_only:
        for p in model.parameters():
            p.requires_grad = False
        strengths = [
            module.align_strength
            for module in getattr(model, "diff_modules", [])
            if getattr(module, "align_strength", None) is not None
        ]
        if not strengths:
            raise ValueError(
                "--train_alignment_only requires --align_radius > 0 and "
                "--learnable_align_blend"
            )
        for strength in strengths:
            strength.requires_grad = True
        print(f"[Alignment] Frozen source model; training {len(strengths)} blend scalars only.")
    elif cfg.train_spatial_fusion_only:
        for p in model.parameters():
            p.requires_grad = False
        if getattr(model, "spatial_head_fusion", None) is None:
            raise ValueError(
                "--train_spatial_fusion_only requires --use_spatial_head_fusion"
            )
        for p in model.spatial_head_fusion.parameters():
            p.requires_grad = True
        print("[Fusion] Frozen source model; training spatial head fusion only.")

    head_params = []
    backbone_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append(p)
        else:
            head_params.append(p)
    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": cfg.lr})
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": cfg.backbone_lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.startswith("cuda") and torch.cuda.is_available())
    scheduler = build_scheduler(optimizer, cfg)
    best_f1 = -1.0
    best_selection_score = -1.0
    start_ep = 1
    if resume_ckpt:
        ckpt = torch.load(resume_ckpt, map_location=device)
        allow_new_modules = (
            cfg.use_nuisance_gate
            or cfg.use_residual_style_adapter
            or cfg.learnable_align_blend
            or cfg.use_spatial_head_fusion
        )
        incompatible = model.load_state_dict(ckpt["model"], strict=not allow_new_modules)
        if allow_new_modules and incompatible.missing_keys:
            print(f"[Resume] Initialized new parameters: {incompatible.missing_keys}")
        if "optimizer" in ckpt and not cfg.train_nuisance_only and not reset_optimizer_on_resume:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                print(f"[Resume] Warning: failed to load optimizer state (param groups may differ): {e}")
        if (
            not cfg.train_nuisance_only
            and not reset_optimizer_on_resume
            and scheduler is not None
            and ckpt.get("scheduler") is not None
        ):
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
            except Exception:
                pass
        if (
            not cfg.train_nuisance_only
            and not reset_optimizer_on_resume
            and isinstance(scaler, torch.cuda.amp.GradScaler)
            and ckpt.get("scaler") is not None
        ):
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception:
                pass
        best_f1 = ckpt.get("best_f1", best_f1)
        if cfg.selection_metric == "f1":
            best_selection_score = ckpt.get("best_selection_score", best_f1)
        else:
            best_selection_score = ckpt.get("best_selection_score", best_selection_score)
        start_ep = ckpt.get("epoch", 0) + 1
        if reset_best_on_resume:
            best_f1 = -1.0
            best_selection_score = -1.0
            print("[Resume] Reset inherited best score for the new objective.")
        if reset_optimizer_on_resume:
            print("[Resume] Kept freshly initialized optimizer and scheduler state.")
        print(
            f"Resumed from {resume_ckpt}: start_ep={start_ep}, best_f1={best_f1:.4f}, "
            f"best_{cfg.selection_metric}={best_selection_score:.4f}"
        )
    best_path = os.path.join(cfg.out_dir, "best.pt")
    last_path = os.path.join(cfg.out_dir, "last.pt")
    metrics_path = os.path.join(cfg.out_dir, "metrics.jsonl")
    print("\n===== Start Training =====")
    print(f"device={device}")
    print(f"data_root={cfg.data_root}")
    print(f"out_dir={cfg.out_dir}")
    print(f"dataset sizes: train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={len(test_loader.dataset)}")
    print(f"epochs={cfg.epochs} batch={cfg.batch_size} crop={cfg.crop_size} grad_accum={cfg.grad_accum}")
    print(f"dino={cfg.dino_name} fuse={cfg.fuse_mode} whiten={cfg.use_whiten}")
    print(
        f"loss weights: bce={cfg.bce_weight} dice={cfg.dice_weight} "
        f"tversky={cfg.tversky_weight}(a={cfg.tversky_alpha},b={cfg.tversky_beta}) "
        f"fp_penalty={cfg.fp_penalty_weight} hard_negative={cfg.hard_negative_weight}@{cfg.hard_negative_ratio} "
        f"boundary={cfg.boundary_weight} (dilation={cfg.boundary_dilation}) "
        f"contrastive={cfg.contrastive_weight}(margin={cfg.contrastive_margin},radius={cfg.contrastive_radius})"
    )
    print(f"eval: full_eval={cfg.full_eval} thr_mode={cfg.thr_mode} thr={cfg.thr} topk={cfg.topk} smooth_k={cfg.smooth_k}")
    print(f"minarea: {cfg.use_minarea} (min_area={cfg.min_area})")
    print(
        f"nuisance pair aug: weight={cfg.nuisance_pair_aug_weight} "
        f"prob={cfg.nuisance_pair_aug_prob} illumination={cfg.nuisance_illumination} "
        f"consistency={cfg.nuisance_pair_consistency} "
        f"feature_consistency={cfg.nuisance_feature_consistency} "
        f"clean_preservation={cfg.nuisance_clean_feature_preservation}"
    )
    print(
        f"registration: weight={cfg.registration_neg_weight} "
        f"prob={cfg.registration_neg_prob} max_shift={cfg.registration_max_shift} "
        f"clean_preservation={cfg.alignment_clean_preservation}"
    )
    print(
        f"spatial fusion: enabled={cfg.use_spatial_head_fusion} "
        f"indices={cfg.spatial_fusion_indices} hidden={cfg.spatial_fusion_hidden} "
        f"uniform_weight={cfg.spatial_fusion_uniform_weight}"
    )
    print(f"checkpoint selection: metric={cfg.selection_metric} beta={cfg.selection_beta}")
    print("==========================\n")
    for ep in range(start_ep, cfg.epochs + 1):
        train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            bce_w=cfg.bce_weight,
            dice_w=cfg.dice_weight,
            boundary_w=cfg.boundary_weight,
            boundary_dilation=cfg.boundary_dilation,
            grad_accum=cfg.grad_accum,
            log_every=cfg.log_every,
            tversky_w=cfg.tversky_weight,
            tversky_alpha=cfg.tversky_alpha,
            tversky_beta=cfg.tversky_beta,
            fp_penalty_w=cfg.fp_penalty_weight,
            hard_negative_weight=cfg.hard_negative_weight,
            hard_negative_ratio=cfg.hard_negative_ratio,
            lambda_consis=cfg.lambda_consis,
            lambda_domain=cfg.lambda_domain,
            self_sup_weight=cfg.self_sup_weight,
            style_aug_prob=cfg.style_aug_prob,
            style_aug_sigma=cfg.style_aug_sigma,
            style_blur_prob=cfg.style_blur_prob,
            identity_neg_weight=cfg.identity_neg_weight,
            identity_neg_prob=cfg.identity_neg_prob,
            contrastive_weight=cfg.contrastive_weight,
            contrastive_margin=cfg.contrastive_margin,
            contrastive_radius=cfg.contrastive_radius,
            nuisance_real_weight=cfg.nuisance_real_weight,
            nuisance_synth_weight=cfg.nuisance_synth_weight,
            nuisance_synth_prob=cfg.nuisance_synth_prob,
            nuisance_illumination=cfg.nuisance_illumination,
            nuisance_max_shift=cfg.nuisance_max_shift,
            nuisance_pair_aug_weight=cfg.nuisance_pair_aug_weight,
            nuisance_pair_aug_prob=cfg.nuisance_pair_aug_prob,
            nuisance_pair_consistency=cfg.nuisance_pair_consistency,
            nuisance_feature_consistency=cfg.nuisance_feature_consistency,
            nuisance_clean_feature_preservation=cfg.nuisance_clean_feature_preservation,
            registration_neg_weight=cfg.registration_neg_weight,
            registration_neg_prob=cfg.registration_neg_prob,
            registration_max_shift=cfg.registration_max_shift,
            alignment_clean_preservation=cfg.alignment_clean_preservation,
            spatial_fusion_uniform_weight=cfg.spatial_fusion_uniform_weight,
            head_aux_weight=cfg.head_aux_weight,
            head_cons_weight=cfg.head_cons_weight,
        )
        alignment_strengths = [
            float(module.align_strength.detach().cpu())
            for module in getattr(model, "diff_modules", [])
            if getattr(module, "align_strength", None) is not None
        ]
        if alignment_strengths:
            print(f"[Alignment] strengths={alignment_strengths}")
        do_eval = (ep % eval_every == 0) or (ep == cfg.epochs)
        if do_eval:
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
                use_ensemble=cfg.use_ensemble_pred,
                ensemble_cfg=None,
            )
            print(
                f"[Epoch {ep:03d}] VAL  "
                f"P={val_m['precision']:.4f} R={val_m['recall']:.4f} F1={val_m['f1']:.4f} "
                f"IoU={val_m['iou']:.4f} OA={val_m['oa']:.4f} Kappa={val_m['kappa']:.4f} | "
                f"TP={val_m['TP']} FP={val_m['FP']} FN={val_m['FN']} TN={val_m['TN']}"
            )
            selection_score = _checkpoint_selection_score(
                val_m, cfg.selection_metric, cfg.selection_beta
            )
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "epoch": ep,
                            "split": "val",
                            **val_m,
                            "selection_metric": cfg.selection_metric,
                            "selection_score": selection_score,
                            "time": time.time(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            is_best = selection_score > best_selection_score
            if is_best:
                best_selection_score = selection_score
                best_f1 = float(val_m["f1"])
            if cfg.save_last:
                torch.save(
                    {
                        "epoch": ep,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "scaler": scaler.state_dict() if isinstance(scaler, torch.cuda.amp.GradScaler) else None,
                        "best_f1": best_f1,
                        "best_selection_score": best_selection_score,
                        "cfg": asdict(cfg),
                    },
                    last_path,
                )
            if cfg.save_best and is_best:
                torch.save(
                    {
                        "epoch": ep,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "scaler": scaler.state_dict() if isinstance(scaler, torch.cuda.amp.GradScaler) else None,
                        "best_f1": best_f1,
                        "best_selection_score": best_selection_score,
                        "cfg": asdict(cfg),
                    },
                    best_path,
                )
                print(
                    f"  -> Saved BEST {best_path} "
                    f"({cfg.selection_metric}={best_selection_score:.4f}, f1={best_f1:.4f})"
                )
        if scheduler is not None:
            scheduler.step()
        if cfg.vis_every > 0 and (ep % cfg.vis_every == 0):
            vis_dir = os.path.join(cfg.out_dir, "vis", f"epoch_{ep:03d}")
            save_vis_samples(
                model=model,
                loader=val_loader,
                device=device,
                out_dir=vis_dir,
                n=cfg.vis_n,
                thr_mode=cfg.thr_mode,
                thr=cfg.thr,
                topk=cfg.topk,
                smooth_k=cfg.smooth_k,
                window=cfg.eval_window,
                stride=cfg.eval_stride,
                use_ensemble=cfg.use_ensemble_pred,
                ensemble_cfg=None,
            )
            print(f"  -> Saved VIS {vis_dir}")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(
            f"\nLoaded BEST checkpoint: {best_path} | "
            f"best_{cfg.selection_metric}={ckpt.get('best_selection_score', ckpt.get('best_f1', -1)):.4f} | "
            f"best_f1={ckpt.get('best_f1', -1):.4f}"
        )
    test_m = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        thr_mode=cfg.thr_mode,
        thr=cfg.thr,
        topk=cfg.topk,
        smooth_k=cfg.smooth_k,
        use_minarea=cfg.use_minarea,
        min_area=cfg.min_area,
        print_every=max(1, len(test_loader) // 5),
        window=cfg.eval_window,
        stride=cfg.eval_stride,
        use_ensemble=cfg.use_ensemble_pred,
        ensemble_cfg=None,
    )
    print("\n====== Final Metrics (Test) ======")
    print(f"THR_MODE={cfg.thr_mode} | TOPK={cfg.topk} | FIXED_THR={cfg.thr}")
    print(f"USE_WHITEN={cfg.use_whiten} | SMOOTH_K={cfg.smooth_k} | MINAREA={cfg.use_minarea}({cfg.min_area})")
    print("--------------------------------------")
    print(f"precision: {test_m['precision']:.4f}")
    print(f"recall   : {test_m['recall']:.4f}")
    print(f"F1       : {test_m['f1']:.4f}")
    print(f"IoU      : {test_m['iou']:.4f}")
    print(f"OA       : {test_m['oa']:.4f}")
    print(f"Kappa    : {test_m['kappa']:.4f}")
    print("--------------------------------------")
    print(f"TP={test_m['TP']} FP={test_m['FP']} FN={test_m['FN']} TN={test_m['TN']}")
    print("======================================\n")
    final_path = os.path.join(cfg.out_dir, "final_test.json")
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump({"split": "test", **test_m, "time": time.time(), "cfg": asdict(cfg)}, f, ensure_ascii=False, indent=2)
    print(f"Saved final metrics to: {final_path}")


if __name__ == "__main__":
    main()
