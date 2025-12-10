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
)

try:
    from config import cfg as project_cfg
    _DEFAULT_WINDOW = int(getattr(project_cfg, "IMAGE_SIZE", 256))
except Exception:
    _DEFAULT_WINDOW = 256


def parse_args():
    base = HeadCfg()
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
    parser.add_argument("--smooth_k", type=int, default=base.smooth_k)
    parser.add_argument("--use_minarea", action="store_true", default=base.use_minarea)
    parser.add_argument("--min_area", type=int, default=base.min_area)
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
    )
    if args.window is not None and args.window <= 0:
        args.window = None
    if args.stride is not None and args.stride <= 0:
        args.stride = None
    return args, cfg


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
    model = DinoSiameseHead(
        dino_name=load_cfg.get("dino_name", cfg.dino_name) if isinstance(load_cfg, dict) else cfg.dino_name,
        use_whiten=load_cfg.get("use_whiten", cfg.use_whiten) if isinstance(load_cfg, dict) else cfg.use_whiten,
    ).to(device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    print(f"Loaded checkpoint from {args.checkpoint}")
    print(f"val/test sizes: {len(val_loader.dataset)}/{len(test_loader.dataset)}")
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
        )
        print(f"Saved visualizations to {vis_dir}")


if __name__ == "__main__":
    main()
