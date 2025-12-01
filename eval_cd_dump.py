"""
Unified evaluation + dump for Change Detection (SegFormer baseline)

python eval_cd_dump.py --checkpoint outputs\checkpoints\best_model.pth --dataset levir --data-root data\LEVIR-CD --exp-name levir2levir --window 256 --stride 256 --num-vis 10
python eval_cd_dump.py --checkpoint outputs\checkpoints\best_model.pth --dataset whucd --data-root data\WHUCD --exp-name levir2whu --window 256 --stride 256 --num-vis 10
python eval_cd_dump.py --checkpoint outputs\checkpoints\best_model.pth --dataset whucd --data-root data\WHUCD --exp-name whu2whu --window 256 --stride 256 --num-vis 10
python eval_cd_dump.py --checkpoint outputs\checkpoints\best_model.pth --dataset levir --data-root data\LEVIR-CD --exp-name whu2levir --window 256 --stride 256 --num-vis 10

python eval_cd_dump.py --checkpoint outputs\checkpoints\best_model.pth --dataset levir --data-root data\LEVIR-CD --exp-name levir2levir_autoThr --window 256 --stride 256 --auto-threshold --auto-bins 512 --num-vis 10 --no-dump-all

Run examples:
  # levir->levir
  python eval_cd_dump.py --checkpoint ckpt_levir.pth --dataset levir --data-root data/LEVIR-CD \
    --exp-name levir2levir --window 256 --stride 256 --num-vis 10

  # levir->whu
  python eval_cd_dump.py --checkpoint ckpt_levir.pth --dataset whucd --data-root data/WHU-CD \
    --exp-name levir2whu --window 256 --stride 256 --num-vis 10

  # whu->whu
  python eval_cd_dump.py --checkpoint ckpt_whu.pth --dataset whucd --data-root data/WHU-CD \
    --exp-name whu2whu --window 256 --stride 256 --num-vis 10

  # whu->levir
  python eval_cd_dump.py --checkpoint ckpt_whu.pth --dataset levir --data-root data/LEVIR-CD \
    --exp-name whu2levir --window 256 --stride 256 --num-vis 10
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cv2
except Exception:
    cv2 = None

# ====== your project imports ======
from config import cfg
from models.segformer import build_model


# -----------------------------
# Utils
# -----------------------------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def to_tensor_norm(img_rgb_uint8: np.ndarray) -> torch.Tensor:
    """HWC uint8 -> CHW float, normalized (ImageNet)"""
    x = img_rgb_uint8.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(x).permute(2, 0, 1)  # CHW
    return x

def denorm_to_numpy(img_chw: torch.Tensor) -> np.ndarray:
    """CHW normalized torch -> HWC float [0,1]"""
    x = img_chw.detach().cpu().float().permute(1, 2, 0).numpy()
    x = (x * IMAGENET_STD + IMAGENET_MEAN).clip(0, 1)
    return x

def ensure_dirs(base: Path):
    (base / "score_np").mkdir(parents=True, exist_ok=True)   # npz float
    (base / "mask").mkdir(parents=True, exist_ok=True)       # png 0/255
    (base / "vis").mkdir(parents=True, exist_ok=True)        # panel png
    (base / "score_vis").mkdir(parents=True, exist_ok=True)  # optional colored heatmap

def otsu_threshold(score01: np.ndarray) -> float:
    """Otsu threshold for float map in [0,1] (works OK even if not perfect)."""
    s = score01.astype(np.float32)
    hist, bin_edges = np.histogram(s.ravel(), bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    prob = hist / (hist.sum() + 1e-12)
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]
    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1 - omega) + 1e-12)
    k = int(np.nanargmax(sigma_b2))
    thr = float(bin_edges[k])
    return thr

def mean_std_threshold(score01: np.ndarray, k: float = 2.0) -> float:
    m, sd = float(score01.mean()), float(score01.std())
    return m + k * sd

def filter_small_cc(mask01: np.ndarray, min_area: int = 0) -> np.ndarray:
    """Remove small connected components. Requires cv2."""
    if min_area <= 0 or cv2 is None:
        return mask01
    mask_u8 = (mask01.astype(np.uint8) * 255)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    out = np.zeros_like(mask01, dtype=np.uint8)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 1
    return out


# -----------------------------
# Dataset (robust folder guesses)
# Expect split folders: train/val/test
# Inside split: A,B,label (or some common aliases)
# -----------------------------
def _resolve_subdir(split_dir: Path, candidates: List[str]) -> Path:
    for c in candidates:
        p = split_dir / c
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find any of {candidates} under {split_dir}")

def _list_stems(dir_path: Path) -> List[str]:
    exts = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"]
    files = []
    for e in exts:
        files += list(dir_path.glob(e))
    stems = sorted(list({f.stem for f in files}))
    if len(stems) == 0:
        raise FileNotFoundError(f"No images found in {dir_path}")
    return stems

def _find_file(dir_path: Path, stem: str) -> Path:
    exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]
    for ext in exts:
        p = dir_path / f"{stem}{ext}"
        if p.exists():
            return p
    # fallback: search by glob
    cand = list(dir_path.glob(stem + ".*"))
    if cand:
        return cand[0]
    raise FileNotFoundError(f"Cannot find file for stem={stem} in {dir_path}")


class PairCDDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "test"):
        self.root_dir = Path(root_dir)
        self.split = split
        split_dir = self.root_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split dir not found: {split_dir}")

        self.dir_a = _resolve_subdir(
            split_dir, ["A", "imageA", "t1", "T1", "img1", "images1"]
        )
        self.dir_b = _resolve_subdir(
            split_dir, ["B", "imageB", "t2", "T2", "img2", "images2"]
        )
        self.dir_l = _resolve_subdir(
            split_dir, ["label", "labels", "gt", "GT", "mask", "masks"]
        )

        # ✅ 用交集，避免缺文件
        stems_a = set(_list_stems(self.dir_a))
        stems_b = set(_list_stems(self.dir_b))
        stems_l = set(_list_stems(self.dir_l))
        self.names = sorted(list(stems_a & stems_b & stems_l))
        if len(self.names) == 0:
            raise RuntimeError(
                f"No paired samples found under {split_dir} (check A/B/label names)."
            )

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx: int):
        name = self.names[idx]
        pa = _find_file(self.dir_a, name)
        pb = _find_file(self.dir_b, name)
        pl = _find_file(self.dir_l, name)

        img_a = np.array(Image.open(pa).convert("RGB"))
        img_b = np.array(Image.open(pb).convert("RGB"))
        lab = np.array(Image.open(pl).convert("L"))

        # ✅ 鲁棒二值化：同时兼容 0/1、0/255、以及 jpg 脏值(254/1)
        mx = int(lab.max())
        if mx <= 1:
            lab01 = (lab > 0).astype(np.uint8)
        else:
            lab01 = (lab >= 128).astype(np.uint8)  # 254/255 -> 1, 0/1 -> 0

        ta = to_tensor_norm(img_a)
        tb = to_tensor_norm(img_b)
        tl = torch.from_numpy(lab01).long()

        return {"img_a": ta, "img_b": tb, "label": tl, "name": name}


# -----------------------------
# Evaluator
# -----------------------------
class Evaluator:
    def __init__(
        self,
        checkpoint_path: str,
        dataset_root: str,
        device: str = "cuda",
        window: int = 256,
        stride: int = 256,
        thresh_mode: str = "fixed",
        threshold: float = 0.5,
        meanstd_k: float = 2.0,
        min_area: int = 0,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.window = int(window)
        self.stride = int(stride)

        self.thresh_mode = thresh_mode
        self.threshold = float(threshold)
        self.meanstd_k = float(meanstd_k)
        self.min_area = int(min_area)

        self.model = self._load_model(checkpoint_path).to(self.device).eval()
        self.dataset_root = Path(dataset_root)
        self.dataset = PairCDDataset(str(self.dataset_root), split="test")

        self.loader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    def _load_model(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        config = ckpt.get("config", {})

        model_type = config.get("model_type", getattr(cfg, "MODEL_TYPE", "segformer_b0"))
        num_classes = config.get("num_classes", getattr(cfg, "NUM_CLASSES", 1))
        variant = model_type.split("_")[-1]

        model = build_model(variant=variant, pretrained=False, num_classes=num_classes)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        return model

    @torch.no_grad()
    def sliding_window_prob(self, img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
        """Return prob map: (1, H, W) in [0,1]"""
        _, _, H, W = img_a.shape
        win, stride = self.window, self.stride

        if H <= win and W <= win:
            out = self.model(img_a, img_b)
            prob = torch.sigmoid(out["pred"].squeeze(1))  # (B,H,W)
            return prob

        pred_sum = torch.zeros((1, H, W), device=self.device)
        count_map = torch.zeros((1, H, W), device=self.device)

        for y in range(0, H, stride):
            for x in range(0, W, stride):
                y_end = min(y + win, H)
                x_end = min(x + win, W)
                y_start = max(0, y_end - win)
                x_start = max(0, x_end - win)

                pa = img_a[:, :, y_start:y_end, x_start:x_end]
                pb = img_b[:, :, y_start:y_end, x_start:x_end]

                out = self.model(pa, pb)
                prob_patch = torch.sigmoid(out["pred"].squeeze(1))  # (1,h,w)

                pred_sum[:, y_start:y_end, x_start:x_end] += prob_patch
                count_map[:, y_start:y_end, x_start:x_end] += 1

        prob = pred_sum / count_map.clamp(min=1)
        return prob

    def _threshold(self, score01: np.ndarray) -> Tuple[np.ndarray, float]:
        if self.thresh_mode == "fixed":
            thr = self.threshold
        elif self.thresh_mode == "otsu":
            thr = otsu_threshold(score01)
        elif self.thresh_mode == "meanstd":
            thr = mean_std_threshold(score01, k=self.meanstd_k)
        else:
            raise ValueError(f"Unknown thresh_mode: {self.thresh_mode}")

        mask01 = (score01 > thr).astype(np.uint8)
        mask01 = filter_small_cc(mask01, min_area=self.min_area)
        return mask01, float(thr)
    
    def _build_loader(self, split: str):
        ds = PairCDDataset(str(self.dataset.root_dir), split=split)
        return DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    @torch.no_grad()
    def _val_hist(self, split: str = "val", bins: int = 512):
        """
        用直方图累计 pos/neg 的 score 分布，从而快速扫阈值找最优 F1。
        不需要把所有 score map 存到内存里，速度/内存都稳。
        """
        loader = self._build_loader(split)
        hist_pos = np.zeros((bins,), dtype=np.int64)
        hist_neg = np.zeros((bins,), dtype=np.int64)
        total_pos = 0
        total_neg = 0

        for batch in loader:
            img_a = batch["img_a"].to(self.device)
            img_b = batch["img_b"].to(self.device)
            gt = batch["label"].to(self.device).squeeze(0).detach().cpu().numpy().astype(np.uint8)  # (H,W)

            prob = self.sliding_window_prob(img_a, img_b)  # (1,H,W)
            score = prob.squeeze(0).detach().cpu().numpy().astype(np.float32)  # [0,1]

            s = score.reshape(-1)
            g = gt.reshape(-1)

            pos = s[g == 1]
            neg = s[g == 0]

            total_pos += pos.size
            total_neg += neg.size

            # 直方图累计（固定分箱 0~1）
            hp, _ = np.histogram(pos, bins=bins, range=(0.0, 1.0))
            hn, _ = np.histogram(neg, bins=bins, range=(0.0, 1.0))
            hist_pos += hp
            hist_neg += hn

        return hist_pos, hist_neg, total_pos, total_neg

    @torch.no_grad()
    def auto_select_threshold(self, bins: int = 512, split: str = "val"):
        """
        在 val 上寻找 best F1 的阈值（pred=score>=thr）。
        """
        hist_pos, hist_neg, total_pos, total_neg = self._val_hist(split=split, bins=bins)

        # 从高到低累计：thr 越小，预测为 1 的像素越多
        pos_cum = np.cumsum(hist_pos[::-1])[::-1].astype(np.float64)  # tp(thr_i)
        neg_cum = np.cumsum(hist_neg[::-1])[::-1].astype(np.float64)  # fp(thr_i)

        # 对应阈值（用 bin 的左边界近似）
        edges = np.linspace(0.0, 1.0, bins + 1, dtype=np.float32)[:-1]  # len=bins

        tp = pos_cum
        fp = neg_cum
        fn = float(total_pos) - tp

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        best_i = int(np.nanargmax(f1))
        best_thr = float(edges[best_i])
        best_f1 = float(f1[best_i])
        best_p = float(precision[best_i])
        best_r = float(recall[best_i])

        return best_thr, best_f1, best_p, best_r

    @torch.no_grad()
    def run(self, save_dir: Path, num_vis: int = 10, visualize_all: bool = False, dump_all: bool = True):
        ensure_dirs(save_dir)

        total_tp = total_fp = total_fn = total_tn = 0
        vis_count = 0

        t0 = time.time()
        for batch in self.loader:
            img_a = batch["img_a"].to(self.device)  # (1,3,H,W)
            img_b = batch["img_b"].to(self.device)  # (1,3,H,W)

            label = batch["label"].to(self.device).squeeze(0)  # (H,W)

            name = batch["name"][0]

            prob = self.sliding_window_prob(img_a, img_b)        # (1,H,W) float
            score = prob.squeeze(0).detach().cpu().numpy().astype(np.float32)  # [0,1]

            pred01, thr_used = self._threshold(score)
            gt01 = label.detach().cpu().numpy().astype(np.uint8)

            # confusion
            tp = int(((pred01 == 1) & (gt01 == 1)).sum())
            fp = int(((pred01 == 1) & (gt01 == 0)).sum())
            fn = int(((pred01 == 0) & (gt01 == 1)).sum())
            tn = int(((pred01 == 0) & (gt01 == 0)).sum())

            total_tp += tp; total_fp += fp; total_fn += fn; total_tn += tn

            # dump per-sample
            if dump_all:
                np.savez_compressed(save_dir / "score_np" / f"{name}.npz", score=score.astype(np.float16))

                Image.fromarray((pred01 * 255).astype(np.uint8)).save(save_dir / "mask" / f"{name}.png")

                # optional colored score heatmap (方便快速翻）
                if cv2 is not None:
                    s = score
                    s_u8 = (255 * (s - s.min()) / (s.max() - s.min() + 1e-8)).astype(np.uint8)
                    heat = cv2.applyColorMap(s_u8, cv2.COLORMAP_JET)[:, :, ::-1]  # -> RGB
                    Image.fromarray(heat).save(save_dir / "score_vis" / f"{name}.png")

            # panel visualize
            if visualize_all or (vis_count < num_vis):
                img_a_np = denorm_to_numpy(img_a.squeeze(0))
                img_b_np = denorm_to_numpy(img_b.squeeze(0))

                overlay = img_b_np.copy()
                overlay[pred01 == 1] = [1, 0, 0]

                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                axes[0, 0].imshow(img_a_np); axes[0, 0].set_title("T1"); axes[0, 0].axis("off")
                axes[0, 1].imshow(img_b_np); axes[0, 1].set_title("T2"); axes[0, 1].axis("off")
                im = axes[0, 2].imshow(score, cmap="jet"); axes[0, 2].set_title(f"Score (thr={thr_used:.3f})"); axes[0, 2].axis("off")
                fig.colorbar(im, ax=axes[0, 2], fraction=0.046)

                axes[1, 0].imshow(pred01 * 255, cmap="gray"); axes[1, 0].set_title("Pred"); axes[1, 0].axis("off")
                axes[1, 1].imshow(gt01 * 255, cmap="gray"); axes[1, 1].set_title("GT"); axes[1, 1].axis("off")
                axes[1, 2].imshow(overlay); axes[1, 2].set_title("Overlay"); axes[1, 2].axis("off")

                plt.tight_layout()
                plt.savefig(save_dir / "vis" / f"{name}.png", dpi=150, bbox_inches="tight")
                plt.close()
                vis_count += 1

        # metrics
        precision = total_tp / (total_tp + total_fp + 1e-8)
        recall = total_tp / (total_tp + total_fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        iou = total_tp / (total_tp + total_fp + total_fn + 1e-8)
        oa = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + 1e-8)

        total = total_tp + total_fp + total_fn + total_tn
        pe = ((total_tp + total_fp) * (total_tp + total_fn) +
              (total_fn + total_tn) * (total_fp + total_tn)) / (total * total + 1e-8)
        kappa = (oa - pe) / (1 - pe + 1e-8)

        results = {
            "Precision": float(precision),
            "Recall": float(recall),
            "F1": float(f1),
            "IoU": float(iou),
            "OA": float(oa),
            "Kappa": float(kappa),
            "TP": int(total_tp),
            "FP": int(total_fp),
            "FN": int(total_fn),
            "TN": int(total_tn),
            "window": int(self.window),
            "stride": int(self.stride),
            "thresh_mode": self.thresh_mode,
            "threshold": float(self.threshold),
            "meanstd_k": float(self.meanstd_k),
            "min_area": int(self.min_area),
            "elapsed_sec": float(time.time() - t0),
        }

        with open(save_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--dataset", type=str, default="levir", choices=["levir", "whucd"])
    ap.add_argument("--data-root", type=str, required=True, help="test dataset root (contains test/A, test/B, test/label etc.)")
    ap.add_argument("--exp-name", type=str, default="exp")
    ap.add_argument("--out-dir", type=str, default="outputs/baseline_eval")

    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--stride", type=int, default=256)

    ap.add_argument("--thresh-mode", type=str, default="fixed", choices=["fixed", "otsu", "meanstd"])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--meanstd-k", type=float, default=2.0)
    ap.add_argument("--min-area", type=int, default=0, help="connected component filter area, 0 to disable")

    ap.add_argument("--num-vis", type=int, default=10)
    ap.add_argument("--visualize-all", action="store_true")
    ap.add_argument("--no-dump-all", action="store_true", help="disable dumping score/mask for all samples")

    ap.add_argument("--auto-threshold", action="store_true", help="select best threshold on val, then eval on test")
    ap.add_argument("--auto-bins", type=int, default=512, help="bins for auto threshold search on val")

    args = ap.parse_args()

    save_dir = Path(args.out_dir) / args.exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    evaluator = Evaluator(
        checkpoint_path=args.checkpoint,
        dataset_root=args.data_root,
        device=getattr(cfg, "DEVICE", "cuda"),
        window=args.window,
        stride=args.stride,
        thresh_mode=args.thresh_mode,
        threshold=args.threshold,
        meanstd_k=args.meanstd_k,
        min_area=args.min_area,
    )
    if args.auto_threshold:
        # Needs a val split under data-root/val
        best_thr, best_f1, best_p, best_r = evaluator.auto_select_threshold(
            bins=args.auto_bins, split="val"
        )
        evaluator.thresh_mode = "fixed"
        evaluator.threshold = best_thr
        print(
            f"\n[AUTO-THR] best_thr={best_thr:.4f}  val_F1={best_f1:.4f}  val_P={best_p:.4f}  val_R={best_r:.4f}\n"
        )

    res = evaluator.run(
        save_dir=save_dir,
        num_vis=args.num_vis,
        visualize_all=args.visualize_all,
        dump_all=(not args.no_dump_all),
    )

    print("\n===== DONE =====")
    for k, v in res.items():
        if isinstance(v, float):
            print(f"{k:>12}: {v:.4f}")
        else:
            print(f"{k:>12}: {v}")
    print(f"\nSaved to: {save_dir.resolve()}")


if __name__ == "__main__":
    main()
