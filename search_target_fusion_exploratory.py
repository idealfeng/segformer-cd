import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from dino_head_core import (
    HeadCfg,
    build_dataloaders,
    compute_metrics_from_cm,
    confusion_update,
    filter_small_cc,
    sliding_window_inference_probs_all,
)
from eval_dino_head import _build_model_from_checkpoint_cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Exploratory target-label search over fixed-threshold head fusions."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--indices", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--thr", type=float, default=0.5)
    parser.add_argument("--smooth_k", type=int, default=3)
    parser.add_argument("--min_area", type=int, default=256)
    parser.add_argument("--weight_step", type=float, default=0.1)
    parser.add_argument("--refine_lower_two", action="store_true")
    parser.add_argument("--refine_disagreement_gate", action="store_true")
    parser.add_argument("--refine_start", type=float, default=0.4)
    parser.add_argument("--refine_stop", type=float, default=1.0)
    parser.add_argument("--refine_step", type=float, default=0.025)
    parser.add_argument("--target_precision", type=float, default=0.46)
    parser.add_argument("--target_f1", type=float, default=0.50)
    parser.add_argument("--top_n", type=int, default=30)
    return parser.parse_args()


def _simplex_weights(count: int, step: float):
    units = int(round(1.0 / step))
    if count != 3 or abs(units * step - 1.0) > 1e-8:
        raise ValueError("Current grid requires three heads and a step dividing 1.0")
    for first in range(units + 1):
        for second in range(units - first + 1):
            third = units - first - second
            yield (first / units, second / units, third / units)


def _candidate_specs(args):
    if args.refine_disagreement_gate:
        return [
            {
                "mode": "disagreement_gate",
                "strength": strength,
                "tolerance": tolerance,
                "weights": None,
            }
            for tolerance in (0.0, 0.025, 0.05, 0.075, 0.1)
            for strength in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
        ]
    if args.refine_lower_two:
        count = int(round((args.refine_stop - args.refine_start) / args.refine_step))
        return [
            {
                "mode": "max_rejection",
                "strength": args.refine_start + index * args.refine_step,
                "weights": None,
            }
            for index in range(count + 1)
        ]

    candidates = []
    for strength_index in range(0, 25):
        candidates.append(
            {
                "mode": "soft_min",
                "strength": strength_index * 0.025,
                "weights": None,
            }
        )
    candidates.extend(
        [
            {"mode": "mean_logit", "strength": None, "weights": None},
            {"mode": "geometric", "strength": None, "weights": None},
            {"mode": "lower_two_mean", "strength": None, "weights": None},
        ]
    )
    for weights in _simplex_weights(3, args.weight_step):
        candidates.append(
            {"mode": "weighted_prob", "strength": None, "weights": weights}
        )
        candidates.append(
            {"mode": "weighted_logit", "strength": None, "weights": weights}
        )
    return candidates


def _fuse(candidate, logits, probabilities):
    mode = candidate["mode"]
    if mode == "soft_min":
        mean = probabilities.mean(dim=0)
        minimum = probabilities.amin(dim=0)
        strength = candidate["strength"]
        return (1.0 - strength) * mean + strength * minimum
    if mode == "mean_logit":
        return torch.sigmoid(logits.mean(dim=0))
    if mode == "geometric":
        return torch.exp(torch.log(probabilities.clamp_min(1e-6)).mean(dim=0))
    if mode == "lower_two_mean":
        return probabilities.topk(k=2, dim=0, largest=False).values.mean(dim=0)
    if mode == "max_rejection":
        mean = probabilities.mean(dim=0)
        maximum = probabilities.amax(dim=0)
        return mean - candidate["strength"] * (maximum - mean)
    if mode == "disagreement_gate":
        mean = probabilities.mean(dim=0)
        gap = probabilities.amax(dim=0) - mean
        return mean - candidate["strength"] * F.relu(gap - candidate["tolerance"])

    weights = torch.as_tensor(
        candidate["weights"], device=logits.device, dtype=logits.dtype
    ).view(-1, 1, 1, 1, 1)
    if mode == "weighted_prob":
        return (weights * probabilities).sum(dim=0)
    return torch.sigmoid((weights * logits).sum(dim=0))


@torch.no_grad()
def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    load_cfg = checkpoint.get("cfg") or {}
    cfg = HeadCfg(
        data_root=args.data_root,
        device=args.device,
        batch_size=1,
        num_workers=args.num_workers,
        full_eval=True,
    )
    _, _, test_loader = build_dataloaders(
        cfg, require_train=False, require_val=False, load_val=False
    )
    model = _build_model_from_checkpoint_cfg(load_cfg, cfg, args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    candidates = _candidate_specs(args)
    cms = [{"TP": 0, "FP": 0, "FN": 0, "TN": 0} for _ in candidates]
    index = torch.as_tensor(args.indices, device=args.device, dtype=torch.long)

    for batch_number, batch in enumerate(test_loader, 1):
        probabilities_all = sliding_window_inference_probs_all(
            model=model,
            img_a=batch["img_a"].to(args.device, non_blocking=True),
            img_b=batch["img_b"].to(args.device, non_blocking=True),
            window=args.window,
            stride=args.stride,
            device=args.device,
        )
        probabilities = probabilities_all.index_select(0, index)
        logits = torch.logit(probabilities.clamp(1e-6, 1.0 - 1e-6))
        target_tensor = batch["label"]
        if target_tensor.ndim == 4:
            target_tensor = target_tensor[0, 0]
        elif target_tensor.ndim == 3:
            target_tensor = target_tensor[0]
        elif target_tensor.ndim != 2:
            raise ValueError(f"Unexpected target shape: {tuple(target_tensor.shape)}")
        target = (target_tensor.detach().cpu().numpy() > 0).astype(np.uint8)

        for candidate_index, candidate in enumerate(candidates):
            probability = _fuse(candidate, logits, probabilities)
            if args.smooth_k > 1:
                probability = F.avg_pool2d(
                    probability,
                    kernel_size=args.smooth_k,
                    stride=1,
                    padding=args.smooth_k // 2,
                )
            prediction = (
                probability[0, 0].detach().float().cpu().numpy() >= args.thr
            ).astype(np.uint8)
            if prediction.shape != target.shape:
                raise ValueError(
                    f"Prediction/target shape mismatch: {prediction.shape} vs {target.shape}"
                )
            if args.min_area > 0:
                prediction = filter_small_cc(prediction, min_area=args.min_area)
            confusion_update(
                torch.from_numpy(prediction),
                torch.from_numpy(target),
                cms[candidate_index],
            )
        if batch_number % 16 == 0:
            print(f"[{batch_number}/{len(test_loader)}] scored {len(candidates)} fusions")

    results = []
    for candidate, cm in zip(candidates, cms):
        metrics = compute_metrics_from_cm(cm)
        precision_gap = abs(metrics["precision"] - args.target_precision)
        f1_gap = abs(metrics["f1"] - args.target_f1)
        results.append(
            {
                **candidate,
                **metrics,
                "target_distance": precision_gap + 0.5 * f1_gap,
            }
        )
    ranked = sorted(results, key=lambda result: result["target_distance"])
    payload = {
        "protocol": {
            "exploratory_target_tuned": True,
            "uses_target_labels": True,
            "data_root": args.data_root,
            "checkpoint": args.checkpoint,
            "indices": args.indices,
            "threshold": args.thr,
            "window": args.window,
            "stride": args.stride,
            "smooth_k": args.smooth_k,
            "min_area": args.min_area,
            "target_precision": args.target_precision,
            "target_f1": args.target_f1,
        },
        "selected": ranked[0],
        "top_results": ranked[: args.top_n],
        "all_results": results,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    for result in ranked[: min(args.top_n, 15)]:
        print(
            f"{result['mode']:14s} strength={str(result['strength']):>5s} "
            f"tolerance={result.get('tolerance')} weights={result['weights']} "
            f"P={result['precision']:.4f} "
            f"R={result['recall']:.4f} F1={result['f1']:.4f}"
        )
    print(f"Saved target-tuned exploratory search to {args.out}")


if __name__ == "__main__":
    main()
