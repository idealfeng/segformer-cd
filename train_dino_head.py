"""
python train_dino_head.py --data_root data/WHUCD --out_dir outputs/dino_head_cd --device auto --epochs 100 --batch_size 8 --crop_size 256 --bce_weight 0.5 --dice_weight 0.5 --thr_mode fixed --thr 0.5
--boundary_dim 192 --boundary_weight 0.5 --boundary_dilation 3
"""

import argparse
import json
import os
import time
from dataclasses import asdict

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


def parse_args():
    base = HeadCfg()
    parser = argparse.ArgumentParser(description="Train DINOv2 change-detection head")
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
    parser.add_argument("--bce_weight", type=float, default=base.bce_weight)
    parser.add_argument("--dice_weight", type=float, default=base.dice_weight)
    parser.add_argument("--boundary_weight", type=float, default=base.boundary_weight, help="aux boundary loss weight")
    parser.add_argument("--boundary_dilation", type=int, default=base.boundary_dilation, help="boundary thickness (px) for supervision")
    parser.add_argument("--lambda_consis", type=float, default=base.lambda_consis, help="counterfactual consistency weight")
    parser.add_argument("--lambda_domain", type=float, default=base.lambda_domain, help="domain confusion weight")
    parser.add_argument("--self_sup_weight", type=float, default=base.self_sup_weight, help="aux supervised weight on perturbed view")
    parser.add_argument("--style_aug_prob", type=float, default=base.style_aug_prob, help="probability to apply style perturbation")
    parser.add_argument("--style_aug_sigma", type=float, default=base.style_aug_sigma, help="noise scale for style perturbation")
    parser.add_argument("--style_blur_prob", type=float, default=base.style_blur_prob, help="blur probability for style perturbation")
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
    parser.add_argument("--fuse_mode", type=str, choices=["abs", "abs+sum", "cat4"], default=base.fuse_mode)
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
    parser.add_argument("--resume", type=str, default=None, help="断点续训 ckpt 路径（last.pt/best.pt）")
    parser.add_argument("--eval_every", type=int, default=1, help="每多少个 epoch 验证一次（默认每个 epoch）")
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
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        boundary_weight=args.boundary_weight,
        boundary_dilation=args.boundary_dilation,
        lambda_consis=args.lambda_consis,
        lambda_domain=args.lambda_domain,
        self_sup_weight=args.self_sup_weight,
        style_aug_prob=args.style_aug_prob,
        style_aug_sigma=args.style_aug_sigma,
        style_blur_prob=args.style_blur_prob,
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
        save_best=args.save_best,
        save_last=args.save_last,
        vis_every=args.vis_every,
        vis_n=args.vis_n,
        log_every=args.log_every,
    )
    return cfg, args.resume, args.eval_every


def main():
    cfg, resume_ckpt, eval_every = parse_args()
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
        if cfg.use_layer_ensemble or cfg.boundary_dim or cfg.use_domain_adv or cfg.use_style_norm or cfg.proto_weight:
            print("[A0] Note: ignoring DLF/MHE/auxiliary modules; using frozen backbone + single 1x1 head only.")
        model = DinoFrozenA0Head(
            dino_name=cfg.dino_name,
            layer=cfg.a0_layer,
            use_whiten=cfg.use_whiten,
        ).to(device)
    else:
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
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.startswith("cuda") and torch.cuda.is_available())
    scheduler = build_scheduler(optimizer, cfg)
    best_f1 = -1.0
    start_ep = 1
    if resume_ckpt:
        ckpt = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and ckpt.get("scheduler") is not None:
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
            except Exception:
                pass
        if isinstance(scaler, torch.cuda.amp.GradScaler) and ckpt.get("scaler") is not None:
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception:
                pass
        best_f1 = ckpt.get("best_f1", best_f1)
        start_ep = ckpt.get("epoch", 0) + 1
        print(f"Resumed from {resume_ckpt}: start_ep={start_ep}, best_f1={best_f1:.4f}")
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
    print(f"loss weights: bce={cfg.bce_weight} dice={cfg.dice_weight} boundary={cfg.boundary_weight} (dilation={cfg.boundary_dilation})")
    print(f"eval: full_eval={cfg.full_eval} thr_mode={cfg.thr_mode} thr={cfg.thr} topk={cfg.topk} smooth_k={cfg.smooth_k}")
    print(f"minarea: {cfg.use_minarea} (min_area={cfg.min_area})")
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
            lambda_consis=cfg.lambda_consis,
            lambda_domain=cfg.lambda_domain,
            self_sup_weight=cfg.self_sup_weight,
            style_aug_prob=cfg.style_aug_prob,
            style_aug_sigma=cfg.style_aug_sigma,
            style_blur_prob=cfg.style_blur_prob,
            head_aux_weight=cfg.head_aux_weight,
            head_cons_weight=cfg.head_cons_weight,
        )
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
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"epoch": ep, "split": "val", **val_m, "time": time.time()}, ensure_ascii=False) + "\n")
            if cfg.save_last:
                torch.save(
                    {
                        "epoch": ep,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "scaler": scaler.state_dict() if isinstance(scaler, torch.cuda.amp.GradScaler) else None,
                        "best_f1": best_f1,
                        "cfg": asdict(cfg),
                    },
                    last_path,
                )
            if cfg.save_best and val_m["f1"] > best_f1:
                best_f1 = val_m["f1"]
                torch.save(
                    {
                        "epoch": ep,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "scaler": scaler.state_dict() if isinstance(scaler, torch.cuda.amp.GradScaler) else None,
                        "best_f1": best_f1,
                        "cfg": asdict(cfg),
                    },
                    best_path,
                )
                print(f"  -> Saved BEST {best_path} (best_f1={best_f1:.4f})")
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
        print(f"\nLoaded BEST checkpoint: {best_path} | best_f1={ckpt.get('best_f1', -1):.4f}")
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
