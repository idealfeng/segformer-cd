"""
DINOv2 Zero-Shot Change Detection - Full Test Evaluation
输出：precision / recall / F1 / IoU / OA / Kappa + TP/FP/FN/TN

用法：
    python eval_dinov2_zeroshot_fulltest.py
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import cfg
from dataset import LEVIRCDDataset, get_test_transforms_full  # 你工程里的

# ----------------------------
# Config (你主要改这些就够了)
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DINO_NAME = "dinov2_vits14_reg"  # 推荐 reg
PATCH = 14

BATCH_SIZE = 2  # 4060 建议 1；如果你显存够可改 2/4
NUM_WORKERS = 0  # Windows 通常 0 更省事

# 阈值策略：
#   "topk": 每张图取 top-k% 的像素为变化（最稳，跨域也常用）
#   "fixed": 固定阈值（需要你自己对 score 分布很熟）
THR_MODE = "topk"
TOPK = 0.1  # 0.003/0.005/0.01/0.02 都可以试；LEVIR 0.01 通常比 0.05 更合理
FIXED_THR = 0.5  # 仅在 THR_MODE="fixed" 时有效

# score 后处理（强烈建议开平滑，能显著减少碎点）
SMOOTH_K = 5  # 1=不平滑；3或5常用

# 特征白化（跨域友好，in-domain不一定更高）
USE_WHITEN = False

# 如果你想把结果二值图做一点去碎片过滤（可选；需要opencv）
USE_MINAREA = True
MIN_AREA = 256  # 128/256/512试试


# ----------------------------
# Optional: connected component filter (opencv)
# ----------------------------
def filter_small_cc(mask_np: np.ndarray, min_area=256) -> np.ndarray:
    import cv2

    mask = mask_np.astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 1
    return out


# ----------------------------
# DINO helpers
# ----------------------------
@torch.no_grad()
def pad_to_patch_multiple(x: torch.Tensor, patch=14):
    """pad (not resize) so H,W are multiples of patch; return padded_x and original (H,W)."""
    B, C, H, W = x.shape
    H2 = ((H + patch - 1) // patch) * patch
    W2 = ((W + patch - 1) // patch) * patch
    pad_h = H2 - H
    pad_w = W2 - W
    if pad_h == 0 and pad_w == 0:
        return x, (H, W)
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, (H, W)


@torch.no_grad()
def spatial_whiten(feat: torch.Tensor, eps=1e-6):
    """
    feat: (B,C,h,w)
    在空间维度 (h,w) 上做 per-channel 标准化（每张图独立）
    """
    mean = feat.mean(dim=(2, 3), keepdim=True)
    std = feat.std(dim=(2, 3), keepdim=True)
    return (feat - mean) / (std + eps)


@torch.no_grad()
def smooth_score(score: torch.Tensor, k: int):
    """score: (B,1,H,W) -> avg pool 平滑"""
    if k <= 1:
        return score
    return F.avg_pool2d(score, kernel_size=k, stride=1, padding=k // 2)


@torch.no_grad()
def extract_dense_features(dino, img: torch.Tensor):
    """
    img: (B,3,H,W) normalized
    return: feat (B,C,h,w), padded_hw (Hp,Wp), orig_hw (H0,W0)
    """
    img, (H0, W0) = pad_to_patch_multiple(img, PATCH)
    B, _, Hp, Wp = img.shape
    h, w = Hp // PATCH, Wp // PATCH

    with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
        out = dino.forward_features(img)

    tokens = out["x_norm_patchtokens"]  # (B, N, C)
    B2, N, C = tokens.shape
    assert N == h * w, f"N={N} != h*w={h*w}. Check pad/patch."

    feat = tokens.transpose(1, 2).reshape(B, C, h, w)

    if USE_WHITEN:
        feat = spatial_whiten(feat)

    feat = F.normalize(feat, dim=1)  # (B,C,h,w)
    return feat, (Hp, Wp), (H0, W0)


@torch.no_grad()
def compute_score_map(dino, img_a: torch.Tensor, img_b: torch.Tensor):
    """
    return score: (B,1,H0,W0) in torch, already cropped to original size
    """
    fa, (Ha, Wa), (H0, W0) = extract_dense_features(dino, img_a)
    fb, (Hb, Wb), _ = extract_dense_features(dino, img_b)
    assert (Ha, Wa) == (Hb, Wb)

    # cosine distance: 1 - cos
    sim = (fa * fb).sum(dim=1, keepdim=True)  # (B,1,h,w)
    dist = 1 - sim

    # upsample to padded size, then crop back to original
    score = F.interpolate(dist, size=(Ha, Wa), mode="bilinear", align_corners=False)
    score = score[..., :H0, :W0]  # (B,1,H0,W0)

    score = smooth_score(score, SMOOTH_K)
    return score


@torch.no_grad()
def threshold_topk(score: torch.Tensor, topk: float):
    """
    score: (B,1,H,W)
    return thr: (B,1,1,1)
    """
    B, _, H, W = score.shape
    flat = score.view(B, -1)
    n = flat.shape[1]
    # 取 1-topk 分位点：把最大的 topk 当正类
    k = int(np.ceil((1.0 - topk) * n))
    k = max(1, min(k, n))
    thr = []
    for b in range(B):
        # kthvalue 是第 k 小；我们需要第 k 小当阈值（k 越大阈值越大）
        v = torch.kthvalue(flat[b], k).values
        thr.append(v)
    thr = torch.stack(thr).view(B, 1, 1, 1)
    return thr


# ----------------------------
# Metrics
# ----------------------------
@torch.no_grad()
def update_confusion(pred: torch.Tensor, gt: torch.Tensor, tp, fp, fn, tn):
    """
    pred, gt: (B,H,W) uint8 (0/1)
    """
    pred = pred.bool()
    gt = gt.bool()
    tp += (pred & gt).sum().item()
    fp += (pred & (~gt)).sum().item()
    fn += ((~pred) & gt).sum().item()
    tn += ((~pred) & (~gt)).sum().item()
    return tp, fp, fn, tn


def compute_metrics(tp, fp, fn, tn, eps=1e-12):
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    oa = (tp + tn) / (tp + fp + fn + tn + eps)

    total = tp + fp + fn + tn + eps
    po = (tp + tn) / total
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (total * total)
    kappa = (po - pe) / (1 - pe + eps)

    return precision, recall, f1, iou, oa, kappa


# ----------------------------
# Main
# ----------------------------
def main():
    print("Loading DINOv2...")
    dino = torch.hub.load("facebookresearch/dinov2", DINO_NAME)
    dino.eval().to(DEVICE)

    print("Building TEST dataset...")
    test_dataset = LEVIRCDDataset(
        root_dir=cfg.DATA_ROOT,
        split="test",
        transform=get_test_transforms_full(),
        crop_size=1024,  # 不影响 test_full transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    tp = fp = fn = tn = 0

    for it, batch in enumerate(test_loader):
        img_a = batch["img_a"].to(DEVICE, non_blocking=True)  # (B,3,H,W)
        img_b = batch["img_b"].to(DEVICE, non_blocking=True)
        gt = batch["label"]  # (B,H,W) or (H,W)

        if gt.dim() == 2:  # 兼容 batch_size=1 时可能没 batch 维
            gt = gt.unsqueeze(0)
        gt = gt.to(DEVICE, non_blocking=True).to(torch.uint8)

        # score: (B,1,H,W)
        score = compute_score_map(dino, img_a, img_b)

        # threshold
        if THR_MODE == "topk":
            thr = threshold_topk(score, TOPK)
        elif THR_MODE == "fixed":
            thr = torch.tensor(FIXED_THR, device=DEVICE).view(1, 1, 1, 1)
        else:
            raise ValueError(f"Unknown THR_MODE={THR_MODE}")

        pred = (score > thr).squeeze(1).to(torch.uint8)  # (B,H,W)

        # optional cc filter (opencv, cpu)
        if USE_MINAREA:
            pred_np = pred.detach().cpu().numpy()
            pred_np2 = np.zeros_like(pred_np)
            for b in range(pred_np.shape[0]):
                pred_np2[b] = filter_small_cc(pred_np[b], min_area=MIN_AREA)
            pred = torch.from_numpy(pred_np2).to(DEVICE).to(torch.uint8)

        # update confusion
        tp, fp, fn, tn = update_confusion(pred, gt, tp, fp, fn, tn)

        if (it + 1) % 20 == 0 or (it + 1) == len(test_loader):
            print(f"[{it+1}/{len(test_loader)}] TP={tp} FP={fp} FN={fn} TN={tn}")

    precision, recall, f1, iou, oa, kappa = compute_metrics(tp, fp, fn, tn)

    print("\n====== Final Metrics (Full Test) ======")
    print(f"THR_MODE={THR_MODE} | TOPK={TOPK} | FIXED_THR={FIXED_THR}")
    print(
        f"USE_WHITEN={USE_WHITEN} | SMOOTH_K={SMOOTH_K} | MINAREA={USE_MINAREA}({MIN_AREA})"
    )
    print("--------------------------------------")
    print(f"precision: {precision:.4f}")
    print(f"recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"IoU      : {iou:.4f}")
    print(f"OA       : {oa:.4f}")
    print(f"Kappa    : {kappa:.4f}")
    print("--------------------------------------")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    print("======================================\n")


if __name__ == "__main__":
    main()
