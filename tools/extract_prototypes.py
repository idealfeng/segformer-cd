"""
Extract deep DINO features on the training split, run k-means, and save prototypes.

Usage example:
  python tools/extract_prototypes.py --data_root data/LEVIR-CD --k 128 --batch_size 4 --out_prefix prototypes

Outputs:
  - <out_prefix>_raw.npy   : sampled raw features [N, C]
  - <out_prefix>.npy       : k-means prototypes [K, C]
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch

try:
    from sklearn.cluster import MiniBatchKMeans
except Exception as e:
    print("sklearn is required for k-means. Please install scikit-learn.", file=sys.stderr)
    raise

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dino_head_core import HeadCfg, build_dataloaders
from models.dinov2_head import DinoSiameseHead


def parse_args():
    parser = argparse.ArgumentParser(description="Extract features and compute prototypes (k-means)")
    parser.add_argument("--data_root", type=str, default="data/LEVIR-CD")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--k", type=int, default=128, help="Number of clusters")
    parser.add_argument("--max_samples", type=int, default=500_000, help="Maximum feature vectors to keep for k-means")
    parser.add_argument("--out_prefix", type=str, default="prototypes", help="Output prefix for npy files")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = HeadCfg(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
    _, train_loader, _ = build_dataloaders(cfg)

    device = cfg.device
    model = DinoSiameseHead().to(device).eval()

    feats = []
    with torch.no_grad():
        for batch in train_loader:
            fa_list, fb_list, _, _ = model._extract_pair_features(
                batch["img_a"].to(device), batch["img_b"].to(device)
            )
            fa = fa_list[-1].cpu().numpy()  # [B, C, h, w]
            B, C, h, w = fa.shape
            fa = fa.transpose(0, 2, 3, 1).reshape(-1, C)  # [B*h*w, C]
            feats.append(fa)
            total = sum(x.shape[0] for x in feats)
            if total >= args.max_samples:
                break

    X = np.concatenate(feats, axis=0)
    if X.shape[0] > args.max_samples:
        idx = np.random.choice(X.shape[0], args.max_samples, replace=False)
        X = X[idx]

    raw_path = f"{args.out_prefix}_raw.npy"
    np.save(raw_path, X.astype(np.float32))
    print(f"Saved raw features: {raw_path} | shape={X.shape}")

    print(f"Running MiniBatchKMeans: k={args.k} ...")
    kmeans = MiniBatchKMeans(n_clusters=args.k, batch_size=4096, max_iter=200)
    kmeans.fit(X)
    protos = kmeans.cluster_centers_.astype(np.float32)

    proto_path = f"{args.out_prefix}.npy"
    np.save(proto_path, protos)
    print(f"Saved prototypes: {proto_path} | shape={protos.shape}")


if __name__ == "__main__":
    main()
