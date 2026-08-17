import argparse
import json
import os
from dataclasses import fields

import numpy as np
import torch
import torch.nn.functional as F

from dino_head_core import (
    HeadCfg,
    build_dataloaders,
    compute_metrics_from_cm,
    confusion_update,
    filter_small_cc,
)
from eval_dino_head import _build_model_from_checkpoint_cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select a conservative fixed head-fusion rule on source validation data."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--source_f1_floor", type=float, default=0.65)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--eval_crop", type=int, default=256)
    parser.add_argument("--thr", type=float, default=0.5)
    parser.add_argument("--smooth_k", type=int, default=3)
    parser.add_argument("--min_area", type=int, default=256)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    load_cfg = checkpoint.get("cfg") or {}
    cfg_keys = {field.name for field in fields(HeadCfg)}
    cfg_values = {key: value for key, value in load_cfg.items() if key in cfg_keys}
    cfg_values.update(
        data_root=args.data_root,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        full_eval=False,
        eval_crop=args.eval_crop,
    )
    cfg = HeadCfg(**cfg_values)
    _, val_loader, _ = build_dataloaders(cfg, require_train=False, require_val=True)
    model = _build_model_from_checkpoint_cfg(load_cfg, cfg, args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    candidates = [("soft_min", float(strength)) for strength in args.strengths]
    candidates.extend([("mean_logit", None), ("geometric", None), ("lower_two_mean", None)])
    cms = [{"TP": 0, "FP": 0, "FN": 0, "TN": 0} for _ in candidates]
    index = torch.as_tensor(args.indices, device=args.device, dtype=torch.long)

    for batch_number, batch in enumerate(val_loader, 1):
        out = model(
            batch["img_a"].to(args.device, non_blocking=True),
            batch["img_b"].to(args.device, non_blocking=True),
        )
        logits = out["logits_all"].index_select(0, index)
        probabilities = torch.sigmoid(logits)
        mean_probability = probabilities.mean(dim=0)
        minimum_probability = probabilities.amin(dim=0)

        gt = batch["label"]
        if gt.ndim == 3:
            gt = gt.unsqueeze(1)
        elif gt.ndim == 4 and gt.shape[1] != 1:
            gt = gt[:, :1]

        fused_probabilities = []
        for mode, strength in candidates:
            if mode == "soft_min":
                probability = (1.0 - strength) * mean_probability + strength * minimum_probability
            elif mode == "mean_logit":
                probability = torch.sigmoid(logits.mean(dim=0))
            elif mode == "geometric":
                probability = torch.exp(torch.log(probabilities.clamp_min(1e-6)).mean(dim=0))
            else:
                probability = probabilities.topk(k=2, dim=0, largest=False).values.mean(dim=0)
            if args.smooth_k > 1:
                probability = F.avg_pool2d(
                    probability,
                    kernel_size=args.smooth_k,
                    stride=1,
                    padding=args.smooth_k // 2,
                )
            fused_probabilities.append(probability)

        for candidate_index, probability in enumerate(fused_probabilities):
            for item in range(probability.shape[0]):
                prediction = (
                    probability[item, 0].detach().cpu().numpy() >= args.thr
                ).astype(np.uint8)
                if args.min_area > 0:
                    prediction = filter_small_cc(prediction, min_area=args.min_area)
                target = (gt[item, 0].detach().cpu().numpy() > 0).astype(np.uint8)
                confusion_update(
                    torch.from_numpy(prediction),
                    torch.from_numpy(target),
                    cms[candidate_index],
                )
        if batch_number % 20 == 0:
            print(f"[{batch_number}/{len(val_loader)}] scored fusion curve")

    results = []
    for (mode, strength), cm in zip(candidates, cms):
        results.append(
            {
                "mode": mode,
                "strength": strength,
                "indices": args.indices,
                **compute_metrics_from_cm(cm),
            }
        )
    valid = [result for result in results if result["f1"] >= args.source_f1_floor]
    selected = max(
        valid,
        key=lambda result: (
            result["precision"],
            result["f1"],
        ),
        default=None,
    )
    payload = {
        "protocol": {
            "selection_domain": args.data_root,
            "uses_target_labels": False,
            "checkpoint": args.checkpoint,
            "indices": args.indices,
            "threshold": args.thr,
            "source_f1_floor": args.source_f1_floor,
        },
        "selected": selected,
        "results": results,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    for result in sorted(results, key=lambda item: item["precision"], reverse=True):
        print(
            f"{result['mode']:14s} strength={str(result['strength']):>4s} "
            f"P={result['precision']:.4f} R={result['recall']:.4f} F1={result['f1']:.4f}"
        )
    print(f"Selected: {selected}")
    print(f"Saved fusion curve to {args.out}")


if __name__ == "__main__":
    main()
