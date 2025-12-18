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
    build_dataloaders,
    seed_everything,
    evaluate,
    save_vis_samples,
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
        "--ensemble_strategy",
        type=str,
        default="mean_prob",
        choices=["mean_prob", "mean_logit", "topk", "weighted_logit"],
        help="Ensemble strategy when --use_ensemble_pred is set",
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
    )
    if args.window is not None and args.window <= 0:
        args.window = None
    if args.stride is not None and args.stride <= 0:
        args.stride = None
    return args, cfg


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
    # build loaders (only need val/test; train ignored)
    _, val_loader, test_loader = build_dataloaders(cfg)
    ckpt = torch.load(args.checkpoint, map_location=device)
    load_cfg = ckpt.get("cfg")
    if isinstance(load_cfg, dict):
        cfg.use_layer_ensemble = load_cfg.get("use_layer_ensemble", cfg.use_layer_ensemble)
        cfg.layer_head_ch = load_cfg.get("layer_head_ch", cfg.layer_head_ch)
    model = DinoSiameseHead(
        dino_name=load_cfg.get("dino_name", cfg.dino_name) if isinstance(load_cfg, dict) else cfg.dino_name,
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

    ensemble_cfg = None
    if args.use_ensemble_pred:
        if args.ensemble_strategy in ("mean_prob", "mean_logit"):
            ensemble_cfg = {"mode": args.ensemble_strategy}
        else:
            if not getattr(cfg, "use_layer_ensemble", False):
                print("Warning: use_layer_ensemble=False in checkpoint cfg; logits_all may be None, ensemble will fallback to pred.")
            print("\n[Ensemble] Scoring each head on VAL to derive selection/weights...")
            head_metrics = score_heads_on_loader(
                model=model,
                loader=val_loader,
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

            layer_indices = list(range(0, max(0, fused_idx)))
            if args.ensemble_topk and args.ensemble_topk > 0 and layer_indices:
                layer_sorted = sorted(layer_indices, key=lambda i: f1s[i], reverse=True)
                picked_layers = layer_sorted[: min(args.ensemble_topk, len(layer_sorted))]
            else:
                picked_layers = layer_indices

            indices = picked_layers + [fused_idx]
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
            else:
                raise ValueError(f"Unknown ensemble_strategy: {args.ensemble_strategy}")
            try:
                os.makedirs(cfg.out_dir, exist_ok=True)
                with open(os.path.join(cfg.out_dir, "ensemble_cfg.json"), "w", encoding="utf-8") as f:
                    json.dump({"strategy": args.ensemble_strategy, "indices": indices, "f1s": f1s, "cfg": ensemble_cfg}, f, ensure_ascii=False, indent=2)
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
