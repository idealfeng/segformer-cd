"""
Prepare an xBD disaster subset for source-only / target-free change-detection evaluation.

This converts the raw xBD layout:
  data/xBD/
    images/
    labels/

into the project's standard evaluation layout:
  data/xBD-CD/
    test/
      A/
      B/
      label/

Label definition:
  positive pixels = post-disaster building polygons whose subtype is in
  {minor-damage, major-damage, destroyed} by default.

This keeps xBD strictly as an evaluation-only target domain. No xBD image or
label is used for training or model selection.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare xBD disaster subset for change detection evaluation.")
    p.add_argument("--raw_root", type=str, default="data/xBD", help="Raw xBD root with images/ and labels/.")
    p.add_argument("--out_root", type=str, default="data/xBD-CD", help="Output dataset root in A/B/label layout.")
    p.add_argument("--split", type=str, default="test", help="Output split name. Default: test")
    p.add_argument(
        "--positive_subtypes",
        type=str,
        default="minor-damage,major-damage,destroyed",
        help="Comma-separated post-disaster subtypes treated as changed.",
    )
    p.add_argument(
        "--ignore_subtypes",
        type=str,
        default="un-classified",
        help="Comma-separated post-disaster subtypes ignored during rasterization.",
    )
    p.add_argument(
        "--disasters",
        type=str,
        default="",
        help="Optional comma-separated disaster prefixes to keep, e.g. hurricane-harvey,socal-fire",
    )
    p.add_argument(
        "--copy_mode",
        type=str,
        default="hardlink",
        choices=["hardlink", "copy"],
        help="How to place images into the prepared dataset.",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing prepared files.")
    return p.parse_args()


def _parse_csv(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _pair_stem_from_post_label(path: Path) -> str:
    suffix = "_post_disaster.json"
    if not path.name.endswith(suffix):
        raise ValueError(f"Unexpected label filename: {path.name}")
    return path.name[: -len(suffix)]


def _parse_polygon_rings(wkt: str) -> List[List[tuple[float, float]]]:
    wkt = str(wkt).strip()
    if not wkt.startswith("POLYGON"):
        raise ValueError(f"Unsupported geometry: {wkt[:32]}")
    start = wkt.find("((")
    end = wkt.rfind("))")
    if start < 0 or end < 0 or end <= start + 2:
        raise ValueError(f"Malformed POLYGON WKT: {wkt[:64]}")
    body = wkt[start + 2 : end]
    ring_strs = re.split(r"\)\s*,\s*\(", body)
    rings: List[List[tuple[float, float]]] = []
    for ring_str in ring_strs:
        pts: List[tuple[float, float]] = []
        for tok in ring_str.split(","):
            parts = tok.strip().split()
            if len(parts) < 2:
                continue
            pts.append((float(parts[0]), float(parts[1])))
        if len(pts) >= 3:
            rings.append(pts)
    return rings


def _rasterize_damage_mask(post_json: Path, positive_subtypes: set[str], ignore_subtypes: set[str]) -> tuple[Image.Image, dict]:
    obj = json.loads(post_json.read_text(encoding="utf-8"))
    meta = obj.get("metadata", {})
    width = int(meta.get("width", meta.get("original_width", 1024)))
    height = int(meta.get("height", meta.get("original_height", 1024)))
    feats = obj.get("features", {}).get("xy", [])

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    stats = {
        "buildings_total": 0,
        "buildings_positive": 0,
        "buildings_ignored": 0,
    }
    subtype_counts: Counter[str] = Counter()

    for feat in feats:
        props = feat.get("properties", {}) or {}
        subtype = str(props.get("subtype", "no-damage")).strip()
        subtype_counts[subtype] += 1
        stats["buildings_total"] += 1
        if subtype in ignore_subtypes:
            stats["buildings_ignored"] += 1
            continue
        if subtype not in positive_subtypes:
            continue
        rings = _parse_polygon_rings(feat.get("wkt", ""))
        if not rings:
            continue
        draw.polygon(rings[0], fill=1, outline=1)
        for hole in rings[1:]:
            draw.polygon(hole, fill=0, outline=0)
        stats["buildings_positive"] += 1

    stats["subtype_counts"] = dict(subtype_counts)
    return mask, stats


def _link_or_copy(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            return
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def _save_mask(mask: Image.Image, dst: Path, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    mask.save(dst)


def _iter_post_labels(raw_root: Path, disasters: Sequence[str]) -> Iterable[Path]:
    label_dir = raw_root / "labels"
    keep = set(disasters)
    for path in sorted(label_dir.glob("*_post_disaster.json")):
        stem = _pair_stem_from_post_label(path)
        if keep and not any(stem.startswith(prefix) for prefix in keep):
            continue
        yield path


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    split_root = out_root / args.split
    img_dir = raw_root / "images"

    positive_subtypes = set(_parse_csv(args.positive_subtypes))
    ignore_subtypes = set(_parse_csv(args.ignore_subtypes))
    disasters = _parse_csv(args.disasters)

    out_a = split_root / "A"
    out_b = split_root / "B"
    out_label = split_root / "label"
    out_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "raw_root": str(raw_root),
        "out_root": str(out_root),
        "split": args.split,
        "positive_subtypes": sorted(positive_subtypes),
        "ignore_subtypes": sorted(ignore_subtypes),
        "disasters": disasters,
        "copy_mode": args.copy_mode,
        "samples_total": 0,
        "samples_positive": 0,
        "samples_negative": 0,
        "missing_pairs": 0,
        "pixel_positives": 0,
        "subtype_counts": Counter(),
        "disaster_counts": Counter(),
    }

    for post_json in _iter_post_labels(raw_root, disasters):
        pair_stem = _pair_stem_from_post_label(post_json)
        disaster_name = pair_stem.rsplit("_", 1)[0]
        pre_img = img_dir / f"{pair_stem}_pre_disaster.png"
        post_img = img_dir / f"{pair_stem}_post_disaster.png"
        if not pre_img.is_file() or not post_img.is_file():
            summary["missing_pairs"] += 1
            continue

        mask, stats = _rasterize_damage_mask(post_json, positive_subtypes, ignore_subtypes)
        pixel_pos = int(sum(mask.getdata()))
        sample_name = f"{pair_stem}.png"

        _link_or_copy(pre_img, out_a / sample_name, mode=args.copy_mode, overwrite=args.overwrite)
        _link_or_copy(post_img, out_b / sample_name, mode=args.copy_mode, overwrite=args.overwrite)
        _save_mask(mask, out_label / sample_name, overwrite=args.overwrite)

        summary["samples_total"] += 1
        summary["pixel_positives"] += pixel_pos
        summary["disaster_counts"][disaster_name] += 1
        summary["subtype_counts"].update(stats["subtype_counts"])
        if pixel_pos > 0:
            summary["samples_positive"] += 1
        else:
            summary["samples_negative"] += 1

    summary["subtype_counts"] = dict(summary["subtype_counts"])
    summary["disaster_counts"] = dict(summary["disaster_counts"])
    summary_path = out_root / "prepare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"[OK] Prepared xBD-CD dataset under: {out_root}")
    print(f"[OK] Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
