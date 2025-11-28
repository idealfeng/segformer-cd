# -*- coding: utf-8 -*-
"""
Generate WHU-CD change labels by XOR'ing building masks of T1(2012) and T2(2016).

You have (new structure):
  WHUCD/
    train/A/*.tif
    train/B/*.tif
    train/label/   (empty, will be filled)
    val/A ...
    test/A ...

You also have (source building masks) in some folders, e.g. original WHU format:
  .../2012/splited_images/{train,val,test}/label/*.tif
  .../2016/splited_images/{train,val,test}/label/*.tif

This script:
  - Uses filenames from dst split/A as the "index"
  - Finds corresponding T1/T2 building masks (by stem)
  - change = XOR(mask_t1, mask_t2)
  - Saves to dst split/label/<same_name_as_A> (default .tif)

Run example (PowerShell one-line!):
  python tools/make_whu_change_labels.py --dst-root data/WHUCD `
    --t1-label-root "data/Building change detection dataset_add/1. The two-period image data/2012/splited_images" `
    --t2-label-root "data/Building change detection dataset_add/1. The two-period image data/2016/splited_images"
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, List
import numpy as np

# Try PIL first, fallback tifffile if needed
from PIL import Image

EXTS = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]


def find_by_stem(folder: Path, stem: str) -> Optional[Path]:
    for ext in EXTS:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def list_files(folder: Path) -> List[Path]:
    files = []
    for ext in EXTS:
        files.extend(folder.glob(f"*{ext}"))
        files.extend(folder.glob(f"*{ext.upper()}"))
    files = sorted(set(files))
    return files


def read_mask(path: Path) -> np.ndarray:
    """
    Read a single-channel mask (tif/png/jpg).
    Returns uint8 array.
    """
    # PIL can read most small tifs; if your tif is weird, install tifffile and use it.
    img = Image.open(path)
    # Some tifs may be "I;16" / etc; convert to L to be safe
    if img.mode not in ("L", "1"):
        img = img.convert("L")
    arr = np.array(img)
    return arr


def save_mask_tif(path: Path, arr_u8: np.ndarray):
    """
    Save as .tif (uint8). Use tifffile if available for best compatibility.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import tifffile  # pip install tifffile

        tifffile.imwrite(str(path), arr_u8, photometric="minisblack")
    except Exception:
        # PIL fallback
        Image.fromarray(arr_u8).save(str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dst-root",
        type=str,
        required=True,
        help="Your reorganized WHUCD root: contains train/val/test with A/B/label.",
    )
    ap.add_argument(
        "--t1-label-root",
        type=str,
        required=True,
        help="Root that contains {train,val,test}/label for T1(2012) building masks. "
        "Example: .../2012/splited_images",
    )
    ap.add_argument(
        "--t2-label-root",
        type=str,
        required=True,
        help="Root that contains {train,val,test}/label for T2(2016) building masks. "
        "Example: .../2016/splited_images",
    )
    ap.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated splits to process. Default: train,val,test",
    )
    ap.add_argument(
        "--thr",
        type=int,
        default=127,
        help="Threshold to binarize building masks. Default 127 (works for 0/255 masks).",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing change labels in dst label folder.",
    )
    args = ap.parse_args()

    dst_root = Path(args.dst_root)
    t1_root = Path(args.t1_label_root)
    t2_root = Path(args.t2_label_root)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    print("=== WHU Change Label Generation (XOR) ===")
    print(f"dst_root       : {dst_root}")
    print(f"t1_label_root  : {t1_root}")
    print(f"t2_label_root  : {t2_root}")
    print(f"splits         : {splits}")
    print(f"thr            : {args.thr}")
    print("========================================")

    for split in splits:
        a_dir = dst_root / split / "A"
        out_dir = dst_root / split / "label"

        # Source building-mask folders (original WHU format)
        t1_lbl_dir = t1_root / split / "label"
        t2_lbl_dir = t2_root / split / "label"

        if not a_dir.exists():
            raise FileNotFoundError(f"Missing dst A folder: {a_dir}")
        if not t1_lbl_dir.exists():
            raise FileNotFoundError(f"Missing T1 label folder: {t1_lbl_dir}")
        if not t2_lbl_dir.exists():
            raise FileNotFoundError(f"Missing T2 label folder: {t2_lbl_dir}")

        a_files = list_files(a_dir)
        if len(a_files) == 0:
            raise RuntimeError(f"No images found in {a_dir}")

        out_dir.mkdir(parents=True, exist_ok=True)

        ok, miss = 0, 0
        for a_path in a_files:
            stem = a_path.stem

            t1p = find_by_stem(t1_lbl_dir, stem)
            t2p = find_by_stem(t2_lbl_dir, stem)
            if t1p is None or t2p is None:
                miss += 1
                continue

            out_path = out_dir / a_path.name  # keep exact filename (e.g. 0_0.tif)
            if out_path.exists() and not args.overwrite:
                ok += 1
                continue

            m1 = read_mask(t1p)
            m2 = read_mask(t2p)
            if m1.shape != m2.shape:
                raise RuntimeError(
                    f"Shape mismatch for {stem}: {m1.shape} vs {m2.shape}"
                )

            b1 = m1 > args.thr
            b2 = m2 > args.thr
            change = (b1 ^ b2).astype(np.uint8) * 255  # 0/255

            save_mask_tif(out_path, change)
            ok += 1

        print(f"[{split}] processed: {ok}, missing_pairs: {miss}, out: {out_dir}")

    print("✅ Done. Change labels are written into each split/label/.")


if __name__ == "__main__":
    main()
