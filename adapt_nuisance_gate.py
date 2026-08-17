from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from dino_head_core import (
    HeadCfg,
    build_dataloaders,
    pairwise_contrast_canonicalize,
    seed_everything,
)
from eval_dino_head import _build_model_from_checkpoint_cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Unlabeled nuisance-gate test-time adaptation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--drop_margin", type=float, default=0.1)
    parser.add_argument("--stable_negative", type=float, default=0.1)
    parser.add_argument("--parameter_reg", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    checkpoint = torch.load(args.checkpoint, map_location=device)
    checkpoint_cfg = checkpoint.get("cfg", {})
    if not checkpoint_cfg.get("use_nuisance_gate", False):
        raise ValueError("Checkpoint must contain a trained nuisance gate")

    cfg = HeadCfg(
        data_root=args.data_root,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        full_eval=False,
        eval_crop=256,
    )
    _, _, test_loader = build_dataloaders(cfg, require_train=False, require_val=False)
    model = _build_model_from_checkpoint_cfg(checkpoint_cfg, cfg, device)
    model.load_state_dict(checkpoint["model"])
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.nuisance_head.parameters():
        parameter.requires_grad = True

    model.eval()
    model.nuisance_head.train()
    initial_gate = {
        name: parameter.detach().clone()
        for name, parameter in model.nuisance_head.named_parameters()
    }
    optimizer = torch.optim.AdamW(model.nuisance_head.parameters(), lr=args.lr, weight_decay=0.0)

    total_unstable = 0
    total_stable_negative = 0
    total_pixels = 0
    total_loss = 0.0
    updates = 0
    for _ in range(args.epochs):
        for batch in test_loader:
            # Labels may exist in the dataset object but are intentionally never read here.
            img_a = batch["img_a"].to(device, non_blocking=True)
            img_b = batch["img_b"].to(device, non_blocking=True)
            norm_a, norm_b = pairwise_contrast_canonicalize(img_a, img_b)

            with torch.no_grad():
                raw = torch.sigmoid(model(img_a, img_b)["raw_pred"])
                raw_normalized = torch.sigmoid(model(norm_a, norm_b)["raw_pred"])
                unstable = (raw > 0.5) & ((raw - raw_normalized) > args.drop_margin)
                stable_negative = (raw < args.stable_negative) & (
                    raw_normalized < args.stable_negative
                )

            nuisance_logit = model(img_a, img_b)["nuisance_logit"]
            loss_map = F.binary_cross_entropy_with_logits(
                nuisance_logit,
                unstable.to(dtype=nuisance_logit.dtype),
                reduction="none",
            )
            positive_loss = (
                loss_map[unstable].mean() if unstable.any() else loss_map.new_zeros(())
            )
            negative_loss = loss_map[stable_negative].mean()
            loss = 0.5 * (positive_loss + negative_loss)
            if args.parameter_reg > 0:
                reg = loss.new_zeros(())
                for name, parameter in model.nuisance_head.named_parameters():
                    reg = reg + (parameter - initial_gate[name]).square().mean()
                loss = loss + args.parameter_reg * reg

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_unstable += int(unstable.sum())
            total_stable_negative += int(stable_negative.sum())
            total_pixels += unstable.numel()
            total_loss += float(loss.detach())
            updates += 1

    adapted_cfg = dict(checkpoint_cfg)
    adapted_cfg.update(
        {
            "test_time_adapted": True,
            "test_time_adapt_method": "pair_contrast_unstable_nuisance_gate",
            "test_time_adapt_source_checkpoint": args.checkpoint,
            "test_time_adapt_uses_target_labels": False,
            "test_time_adapt_data_root": args.data_root,
            "test_time_adapt_drop_margin": args.drop_margin,
            "test_time_adapt_stable_negative": args.stable_negative,
            "test_time_adapt_epochs": args.epochs,
        }
    )
    output = {
        "model": model.state_dict(),
        "cfg": adapted_cfg,
        "epoch": checkpoint.get("epoch", 0),
        "best_f1": checkpoint.get("best_f1", -1.0),
        "tta_adaptation": {
            "uses_target_labels": False,
            "updates": updates,
            "mean_loss": total_loss / max(1, updates),
            "unstable_pixels": total_unstable,
            "stable_negative_pixels": total_stable_negative,
            "total_pixels": total_pixels,
            "args": vars(args),
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, out_path)
    report_path = out_path.with_suffix(".json")
    report_path.write_text(
        json.dumps(output["tta_adaptation"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output["tta_adaptation"], ensure_ascii=False, indent=2))
    print(f"Saved adapted checkpoint to {out_path}")


if __name__ == "__main__":
    main()
