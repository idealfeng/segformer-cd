# -*- coding: utf-8 -*-
"""
Split WHU-CD train into train/val, keep official test untouched.

Root expected (your case):
root/
  2012/splited_images/train/image/*.tif
  2012/splited_images/train/label/*.tif
  2016/splited_images/train/image/*.tif
  2016/splited_images/train/label/*.tif
  2012/splited_images/test/image/*.tif
  2012/splited_images/test/label/*.tif
  2016/splited_images/test/image/*.tif
  2016/splited_images/test/label/*.tif

This script will create (optional, by move/copy):
  2012/splited_images/val/image, label
  2016/splited_images/val/image, label

and write split lists:
  root/splits/whu_train.txt
  root/splits/whu_val.txt
  root/splits/whu_test.txt
  root/splits/whu_split.json

IMPORTANT:
- Do NOT subtract test stems from train stems (train/test can share same names).
- Use --mode move to avoid val leakage when training by directory glob.
"""

from __future__ import annotations
import argparse
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

EXTS = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]


def list_stems(folder: Path) -> List[str]:
    stems = []
    if not folder.exists():
        return stems
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in EXTS:
            stems.append(p.stem)
    stems.sort()
    return stems


def find_file_by_stem(folder: Path, stem: str) -> Optional[Path]:
    for ext in EXTS:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def clear_dir(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    elif mode == "symlink":
        os.symlink(str(src), str(dst))
    else:
        raise ValueError(f"Unsupported mode: {mode}")


@dataclass
class WHUPaths:
    root: Path
    a_img_train: Path
    a_lbl_train: Path
    b_img_train: Path
    b_lbl_train: Path
    a_img_val: Path
    a_lbl_val: Path
    b_img_val: Path
    b_lbl_val: Path
    a_img_test: Path
    a_lbl_test: Path
    b_img_test: Path
    b_lbl_test: Path
    splits_dir: Path


def build_paths(root: Path) -> WHUPaths:
    root = Path(root)

    def p(*xs):
        return root.joinpath(*xs)

    return WHUPaths(
        root=root,
        a_img_train=p("2012", "splited_images", "train", "image"),
        a_lbl_train=p("2012", "splited_images", "train", "label"),
        b_img_train=p("2016", "splited_images", "train", "image"),
        b_lbl_train=p("2016", "splited_images", "train", "label"),
        a_img_val=p("2012", "splited_images", "val", "image"),
        a_lbl_val=p("2012", "splited_images", "val", "label"),
        b_img_val=p("2016", "splited_images", "val", "image"),
        b_lbl_val=p("2016", "splited_images", "val", "label"),
        a_img_test=p("2012", "splited_images", "test", "image"),
        a_lbl_test=p("2012", "splited_images", "test", "label"),
        b_img_test=p("2016", "splited_images", "test", "image"),
        b_lbl_test=p("2016", "splited_images", "test", "label"),
        splits_dir=p("splits"),
    )


def paired_stems(img_dir: Path, lbl_dir: Path) -> set[str]:
    return set(list_stems(img_dir)) & set(list_stems(lbl_dir))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        required=True,
        help="Path to '1. The two-period image data' folder",
    )
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument(
        "--mode", type=str, default="move", choices=["move", "copy", "symlink"]
    )
    ap.add_argument(
        "--reset-val",
        action="store_true",
        help="Delete existing val folders before split (recommended).",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    paths = build_paths(Path(args.root))

    # sanity
    req = [
        paths.a_img_train,
        paths.a_lbl_train,
        paths.b_img_train,
        paths.b_lbl_train,
        paths.a_img_test,
        paths.a_lbl_test,
        paths.b_img_test,
        paths.b_lbl_test,
    ]
    for p in req:
        if not p.exists():
            raise FileNotFoundError(f"Missing folder: {p}")

    # ---- compute paired stems for train and test (DO NOT subtract each other) ----
    train_a = paired_stems(paths.a_img_train, paths.a_lbl_train)
    train_b = paired_stems(paths.b_img_train, paths.b_lbl_train)
    train_stems_all = sorted(list(train_a & train_b))

    test_a = paired_stems(paths.a_img_test, paths.a_lbl_test)
    test_b = paired_stems(paths.b_img_test, paths.b_lbl_test)
    test_stems = sorted(list(test_a & test_b))

    if not train_stems_all:
        raise RuntimeError(
            "No paired TRAIN samples found. Check names/extensions in train folders."
        )
    if not test_stems:
        print("[WARN] No paired TEST samples found (maybe labels missing in test?).")

    n_total = len(train_stems_all)
    n_val = int(round(n_total * float(args.val_ratio)))
    n_val = max(1, n_val)
    n_val = min(n_total - 1, n_val)

    rng = random.Random(args.seed)
    shuffled = train_stems_all[:]
    rng.shuffle(shuffled)
    val_stems = sorted(shuffled[:n_val])
    train_stems = sorted(shuffled[n_val:])

    # info
    overlap = len(set(train_stems_all) & set(test_stems))
    print(f"[WHU] paired TRAIN: {len(train_stems_all)}")
    print(f"[WHU] paired TEST : {len(test_stems)}")
    if overlap:
        print(
            f"[WHU] NOTE: {overlap} stems appear in BOTH train and test (same name, different folders). This is OK."
        )

    print(
        f"[WHU] split: train={len(train_stems)} val={len(val_stems)} (val_ratio={args.val_ratio}, seed={args.seed})"
    )
    print(f"[WHU] mode={args.mode} reset_val={args.reset_val} dry_run={args.dry_run}")

    # create/clear val dirs + splits dir
    if not args.dry_run:
        ensure_dir(paths.splits_dir)
        if args.reset_val:
            clear_dir(paths.a_img_val)
            clear_dir(paths.a_lbl_val)
            clear_dir(paths.b_img_val)
            clear_dir(paths.b_lbl_val)
        else:
            ensure_dir(paths.a_img_val)
            ensure_dir(paths.a_lbl_val)
            ensure_dir(paths.b_img_val)
            ensure_dir(paths.b_lbl_val)

    # move/copy val samples from TRAIN->VAL
    missing = 0
    for stem in val_stems:
        a_img = find_file_by_stem(paths.a_img_train, stem)
        a_lbl = find_file_by_stem(paths.a_lbl_train, stem)
        b_img = find_file_by_stem(paths.b_img_train, stem)
        b_lbl = find_file_by_stem(paths.b_lbl_train, stem)
        if not all([a_img, a_lbl, b_img, b_lbl]):
            missing += 1
            continue

        if args.dry_run:
            continue

        link_or_copy(a_img, paths.a_img_val / a_img.name, args.mode)
        link_or_copy(a_lbl, paths.a_lbl_val / a_lbl.name, args.mode)
        link_or_copy(b_img, paths.b_img_val / b_img.name, args.mode)
        link_or_copy(b_lbl, paths.b_lbl_val / b_lbl.name, args.mode)

    if missing:
        print(
            f"[WARN] {missing} val stems missing paired files in train folders (skipped)."
        )

    # write split lists
    split_json = {
        "root": str(paths.root),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "mode": args.mode,
        "reset_val": args.reset_val,
        "n_train_total_paired_before_split": n_total,
        "n_train": len(train_stems),
        "n_val": len(val_stems),
        "n_test": len(test_stems),
        "train_stems": train_stems,
        "val_stems": val_stems,
        "test_stems": test_stems,
    }

    if not args.dry_run:
        (paths.splits_dir / "whu_train.txt").write_text(
            "\n".join(train_stems), encoding="utf-8"
        )
        (paths.splits_dir / "whu_val.txt").write_text(
            "\n".join(val_stems), encoding="utf-8"
        )
        (paths.splits_dir / "whu_test.txt").write_text(
            "\n".join(test_stems), encoding="utf-8"
        )
        (paths.splits_dir / "whu_split.json").write_text(
            json.dumps(split_json, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    print(f"Done. Splits saved to: {paths.splits_dir}")


if __name__ == "__main__":
    main()
