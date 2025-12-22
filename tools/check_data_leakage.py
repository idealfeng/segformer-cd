import argparse
import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]


def _find_existing(folder: Path, stem: str) -> Optional[Path]:
    for ext in EXTS:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _stems(split_dir: Path) -> Set[str]:
    a_dir = split_dir / "A"
    if not a_dir.exists():
        return set()
    out: Set[str] = set()
    for ext in EXTS:
        out.update({p.stem for p in a_dir.glob(f"*{ext}")})
    return set(sorted(out))


def _pair_hash(root: Path, split: str, stem: str) -> Optional[str]:
    split_dir = root / split
    a = _find_existing(split_dir / "A", stem)
    b = _find_existing(split_dir / "B", stem)
    y = _find_existing(split_dir / "label", stem)
    if not a or not b or not y:
        return None
    return "|".join([_md5(a), _md5(b), _md5(y)])


def _check_by_name(tr: Set[str], va: Set[str], te: Set[str]) -> Dict[str, Set[str]]:
    return {
        "train∩val": tr & va,
        "train∩test": tr & te,
        "val∩test": va & te,
    }


def _check_by_hash(root: Path, split_a: str, split_b: str, stems: Iterable[str], limit: int) -> Tuple[int, List[str]]:
    same = 0
    examples: List[str] = []
    for i, stem in enumerate(stems):
        if limit and i >= limit:
            break
        ha = _pair_hash(root, split_a, stem)
        hb = _pair_hash(root, split_b, stem)
        if ha is None or hb is None:
            continue
        if ha == hb:
            same += 1
            if len(examples) < 10:
                examples.append(stem)
    return same, examples


def main() -> int:
    ap = argparse.ArgumentParser(description="Quick checks for potential split leakage (by filename and optional hash).")
    ap.add_argument("--data_root", type=str, required=True, help="Dataset root containing train/val/test subfolders.")
    ap.add_argument("--hash_check", action="store_true", help="Also compare content hash for overlapping stems.")
    ap.add_argument(
        "--hash_limit",
        type=int,
        default=200,
        help="Max number of overlapping stems to hash-check per split pair (0 means no limit).",
    )
    args = ap.parse_args()

    root = Path(args.data_root)
    tr = _stems(root / "train")
    va = _stems(root / "val")
    te = _stems(root / "test")

    print(f"root={root}")
    print(f"counts: train={len(tr)} val={len(va)} test={len(te)}")

    overlaps = _check_by_name(tr, va, te)
    for k, s in overlaps.items():
        print(f"{k}: {len(s)}")
        if s:
            print(f"  examples: {sorted(list(s))[:10]}")

    if not args.hash_check:
        print("\nNOTE: filename overlap does NOT necessarily mean leakage if different splits reuse names.")
        print("      Use --hash_check to verify identical (A,B,label) content on overlapping stems.")
        return 0

    print("\nHash-checking overlapping stems (A,B,label md5)...")
    for (sa, sb, key) in [("train", "val", "train∩val"), ("train", "test", "train∩test"), ("val", "test", "val∩test")]:
        stems = sorted(list(overlaps[key]))
        same, examples = _check_by_hash(root, sa, sb, stems, limit=int(args.hash_limit))
        print(f"{sa} vs {sb}: identical_pairs={same} / checked={min(len(stems), int(args.hash_limit) if args.hash_limit else len(stems))}")
        if examples:
            print(f"  identical examples: {examples}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

