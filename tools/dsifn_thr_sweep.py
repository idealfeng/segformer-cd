"""
Sweep label-free thresholding modes (fixed/otsu/topk) on DSIFN for cross-domain checkpoints.

This is for analysis/debugging. Selecting the best setting on the DSIFN *test* after seeing results
is not a fair evaluation protocol for a paper; treat this as exploratory.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


@dataclass(frozen=True)
class RunCfg:
    ckpt_name: str
    ckpt_path: str
    mhe: str  # "off" | "on"
    thr_mode: str  # fixed|otsu|topk
    thr: Optional[float] = None
    topk: Optional[float] = None


def _run_eval(
    *,
    exe: str,
    out_dir: Path,
    checkpoint: str,
    data_root: str,
    device: str,
    thr_mode: str,
    thr: Optional[float],
    topk: Optional[float],
    use_mhe: bool,
    ensemble_indices: str,
    smooth_k: int,
    use_minarea: bool,
    min_area: int,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "eval_results.json"
    if out_json.is_file():
        return json.loads(out_json.read_text(encoding="utf-8"))

    args: List[str] = [
        sys.executable,
        exe,
        "--checkpoint",
        checkpoint,
        "--data_root",
        data_root,
        "--out_dir",
        str(out_dir),
        "--device",
        device,
        "--batch_size",
        "1",
        "--num_workers",
        "0",
        "--full_eval",
        "--thr_mode",
        str(thr_mode),
        "--smooth_k",
        str(int(smooth_k)),
        "--tta",
        "none",
    ]

    if thr_mode == "fixed":
        assert thr is not None
        args += ["--thr", str(float(thr))]
    elif thr_mode == "topk":
        assert topk is not None
        args += ["--topk", str(float(topk))]
    elif thr_mode == "otsu":
        pass
    else:
        raise ValueError(f"Unknown thr_mode: {thr_mode}")

    if use_minarea:
        args += ["--use_minarea", "--min_area", str(int(min_area))]

    if use_mhe:
        args += [
            "--use_ensemble_pred",
            "--ensemble_strategy",
            "mean_logit",
            "--ensemble_indices",
            str(ensemble_indices),
        ]

    env = os.environ.copy()
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")

    subprocess.run(args, check=True, env=env)
    return json.loads(out_json.read_text(encoding="utf-8"))


def parse_args():
    p = argparse.ArgumentParser("DSIFN threshold sweep (fixed/otsu/topk)")
    p.add_argument("--data_root", type=str, default="data/DSIFN-Dataset")
    p.add_argument("--out_dir", type=str, default="outputs/dsifn_thr_sweep")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--smooth_k", type=int, default=3)
    p.add_argument("--use_minarea", action="store_true", default=True)
    p.add_argument("--min_area", type=int, default=256)
    p.add_argument("--ensemble_indices", type=str, default="3,4")
    p.add_argument("--ckpt_levir", type=str, default="outputs/ablation/best/Best_levir--whu/best.pt")
    p.add_argument("--ckpt_whu", type=str, default="outputs/ablation/best/Best_whu--levir/best.pt")
    return p.parse_args()


def main():
    args = parse_args()
    exe = str(Path(__file__).resolve().parents[1] / "eval_dino_head.py")

    fixed_thrs = [0.05, 0.10, 0.20, 0.30, 0.50]
    topks = [0.002, 0.005, 0.01, 0.02, 0.05]

    run_list: List[RunCfg] = []
    for ckpt_name, ckpt_path in [
        ("LEVIR→DSIFN", str(args.ckpt_levir)),
        ("WHU→DSIFN", str(args.ckpt_whu)),
    ]:
        for mhe in ("off", "on"):
            for t in fixed_thrs:
                run_list.append(RunCfg(ckpt_name, ckpt_path, mhe, "fixed", thr=float(t)))
            run_list.append(RunCfg(ckpt_name, ckpt_path, mhe, "otsu"))
            for k in topks:
                run_list.append(RunCfg(ckpt_name, ckpt_path, mhe, "topk", topk=float(k)))

    base_out = Path(args.out_dir)
    base_out.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []

    for rc in run_list:
        tag = rc.ckpt_name.replace("→", "2").replace(" ", "")
        if rc.thr_mode == "fixed":
            sub = f"fixed_thr{rc.thr:g}"
        elif rc.thr_mode == "topk":
            sub = f"topk_{rc.topk:g}"
        else:
            sub = "otsu"
        out_dir = base_out / tag / f"mhe_{rc.mhe}" / sub

        rep = _run_eval(
            exe=exe,
            out_dir=out_dir,
            checkpoint=rc.ckpt_path,
            data_root=str(args.data_root),
            device=str(args.device),
            thr_mode=rc.thr_mode,
            thr=rc.thr,
            topk=rc.topk,
            use_mhe=(rc.mhe == "on"),
            ensemble_indices=str(args.ensemble_indices),
            smooth_k=int(args.smooth_k),
            use_minarea=bool(args.use_minarea),
            min_area=int(args.min_area),
        )
        rows.append(
            {
                "run": str(out_dir.relative_to(base_out)).replace("\\", "/"),
                "ckpt": rc.ckpt_name,
                "mhe": rc.mhe,
                "thr_mode": rc.thr_mode,
                "thr": rc.thr,
                "topk": rc.topk,
                "precision": float(rep.get("precision", 0.0)),
                "recall": float(rep.get("recall", 0.0)),
                "f1": float(rep.get("f1", 0.0)),
                "iou": float(rep.get("iou", 0.0)),
            }
        )

    (base_out / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    # Print best per (ckpt,mhe).
    def _key(r):
        return (r["ckpt"], r["mhe"])

    best: Dict[Tuple[str, str], Dict[str, object]] = {}
    for r in rows:
        k = (str(r["ckpt"]), str(r["mhe"]))
        if k not in best or float(r["f1"]) > float(best[k]["f1"]):
            best[k] = r

    print("\n== Best F1 on DSIFN test (exploratory) ==")
    for k in sorted(best.keys()):
        r = best[k]
        extra = ""
        if r["thr_mode"] == "fixed":
            extra = f"thr={r['thr']}"
        elif r["thr_mode"] == "topk":
            extra = f"topk={r['topk']}"
        print(
            f"{r['ckpt']} | MHE={r['mhe']} | {r['thr_mode']} {extra} | "
            f"F1={float(r['f1']):.4f} P={float(r['precision']):.4f} R={float(r['recall']):.4f} | {r['run']}"
        )


if __name__ == "__main__":
    main()

