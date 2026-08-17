import argparse
import itertools
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
        description="Exhaustively rank DLV-CD layer-head subsets on source validation data."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--eval_crop", type=int, default=256)
    parser.add_argument("--thr", type=float, default=0.5)
    parser.add_argument("--smooth_k", type=int, default=3)
    parser.add_argument("--min_area", type=int, default=256)
    parser.add_argument("--selection_beta", type=float, default=0.75)
    parser.add_argument("--top_n", type=int, default=20)
    return parser.parse_args()


def _fbeta(precision: float, recall: float, beta: float) -> float:
    beta2 = beta * beta
    return (1.0 + beta2) * precision * recall / (
        beta2 * precision + recall + 1e-12
    )


def _head_names(load_cfg: dict, count: int):
    layers = [int(x) for x in load_cfg.get("selected_layers", [])]
    names = [f"layer_{layer}" for layer in layers]
    names.append("fused")
    if len(names) != count:
        names = [f"head_{i}" for i in range(count - 1)] + ["fused"]
    return names


@torch.no_grad()
def main():
    args = parse_args()
    device = args.device
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    load_cfg = checkpoint.get("cfg") or {}

    cfg_keys = {field.name for field in fields(HeadCfg)}
    cfg_values = {key: value for key, value in load_cfg.items() if key in cfg_keys}
    cfg_values.update(
        data_root=args.data_root,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        full_eval=False,
        eval_crop=args.eval_crop,
    )
    cfg = HeadCfg(**cfg_values)
    _, val_loader, _ = build_dataloaders(cfg, require_train=False, require_val=True)

    model = _build_model_from_checkpoint_cfg(load_cfg, cfg, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    candidates = None
    cms = None
    head_names = None
    for batch_index, batch in enumerate(val_loader, 1):
        img_a = batch["img_a"].to(device, non_blocking=True)
        img_b = batch["img_b"].to(device, non_blocking=True)
        out = model(img_a, img_b)
        logits_all = out.get("logits_all") if isinstance(out, dict) else None
        if logits_all is None:
            raise RuntimeError("Checkpoint must contain layer-ensemble heads")

        head_count, batch_size, _, height, width = logits_all.shape
        if candidates is None:
            subsets = [
                tuple(indices)
                for size in range(1, head_count + 1)
                for indices in itertools.combinations(range(head_count), size)
            ]
            candidates = [
                (mode, indices)
                for mode in ("mean_logit", "mean_prob")
                for indices in subsets
            ]
            cms = [
                {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
                for _ in candidates
            ]
            head_names = _head_names(load_cfg, head_count)

        gt = batch["label"]
        if gt.ndim == 3:
            gt = gt.unsqueeze(1)
        elif gt.ndim == 4 and gt.shape[1] != 1:
            gt = gt[:, :1]

        probs_all = torch.sigmoid(logits_all)
        for candidate_index, (mode, indices) in enumerate(candidates):
            index = torch.as_tensor(indices, device=device, dtype=torch.long)
            selected_logits = logits_all.index_select(0, index)
            if mode == "mean_logit":
                probability = torch.sigmoid(selected_logits.mean(dim=0))
            else:
                probability = probs_all.index_select(0, index).mean(dim=0)
            if args.smooth_k > 1:
                pad = args.smooth_k // 2
                probability = F.avg_pool2d(
                    probability,
                    kernel_size=args.smooth_k,
                    stride=1,
                    padding=pad,
                )

            for batch_item in range(batch_size):
                prediction = (
                    probability[batch_item, 0].detach().cpu().numpy() >= args.thr
                ).astype(np.uint8)
                if args.min_area > 0:
                    prediction = filter_small_cc(prediction, min_area=args.min_area)
                target = (gt[batch_item, 0].detach().cpu().numpy() > 0).astype(
                    np.uint8
                )
                confusion_update(
                    torch.from_numpy(prediction),
                    torch.from_numpy(target),
                    cms[candidate_index],
                )

        if batch_index % 20 == 0:
            print(f"[{batch_index}/{len(val_loader)}] scored {len(candidates)} subsets")

    results = []
    for (mode, indices), cm in zip(candidates, cms):
        metrics = compute_metrics_from_cm(cm)
        metrics["fbeta"] = _fbeta(
            metrics["precision"], metrics["recall"], args.selection_beta
        )
        results.append(
            {
                "mode": mode,
                "indices": list(indices),
                "heads": [head_names[index] for index in indices],
                **metrics,
            }
        )

    by_f1 = sorted(results, key=lambda item: item["f1"], reverse=True)
    by_fbeta = sorted(results, key=lambda item: item["fbeta"], reverse=True)
    payload = {
        "protocol": {
            "selection_domain": args.data_root,
            "uses_target_labels": False,
            "checkpoint": args.checkpoint,
            "threshold": args.thr,
            "smooth_k": args.smooth_k,
            "min_area": args.min_area,
            "selection_beta": args.selection_beta,
            "head_names": head_names,
        },
        "top_by_f1": by_f1[: args.top_n],
        "top_by_fbeta": by_fbeta[: args.top_n],
        "all_results": results,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    print("Top source-validation subsets by F1:")
    for item in by_f1[: min(10, args.top_n)]:
        print(
            f"  {item['mode']:10s} {item['heads']} "
            f"P={item['precision']:.4f} R={item['recall']:.4f} "
            f"F1={item['f1']:.4f} F{args.selection_beta:g}={item['fbeta']:.4f}"
        )
    print(f"Saved {len(results)} candidates to {args.out}")


if __name__ == "__main__":
    main()
