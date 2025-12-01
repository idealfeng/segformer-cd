# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
from PIL import Image
import argparse

EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        required=True,
        help="dataset root containing train/val/test with label/",
    )
    ap.add_argument("--splits", type=str, default="train,val,test")
    ap.add_argument("--thr", type=int, default=128, help="threshold for 0~255 masks")
    ap.add_argument(
        "--keep-old",
        action="store_true",
        help="keep original files (default keeps and also writes png)",
    )
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing png")
    args = ap.parse_args()

    root = Path(args.root)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    for split in splits:
        label_dir = root / split / "label"
        if not label_dir.exists():
            print(f"[WARN] missing: {label_dir}")
            continue

        files = []
        for ext in EXTS:
            files += list(label_dir.glob(f"*{ext}"))
        files = sorted(files)

        print(f"[{split}] found {len(files)} label files in {label_dir}")

        for p in files:
            # 输出 png（同 stem）
            out_png = label_dir / f"{p.stem}.png"
            if out_png.exists() and (not args.overwrite):
                continue

            arr = np.array(Image.open(p).convert("L"))

            # 兼容 0/1 和 0/255，同时避免 jpg 的 1 被当成前景
            if arr.max() <= 1:
                bin01 = arr > 0
            else:
                bin01 = arr >= args.thr

            out = bin01.astype(np.uint8) * 255
            Image.fromarray(out).save(out_png)

            # 可选：删除原始有损 label
            if (not args.keep_old) and p.suffix.lower() != ".png":
                try:
                    p.unlink()
                except Exception:
                    pass

        print(f"[{split}] done.")


if __name__ == "__main__":
    main()
