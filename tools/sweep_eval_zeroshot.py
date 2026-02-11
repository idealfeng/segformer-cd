"""
Zero-shot evaluation sweep helper.

Runs eval_dino_head.py across a grid of:
  - thresholding (fixed / otsu / topk)
  - TTA (none / flip / d4)
  - ensemble strategies (none / mean_logit / consis2 / consisk / uwi)

This is intended for quick experimentation to find a high-F1 inference setting
WITHOUT using target labels. If an ensemble strategy needs calibration, pass
--calib_root pointing to the *source* domain (allowed in zero-shot transfer).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _slug(s: str) -> str:
    s = str(s)
    s = s.replace("\\", "/")
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "run"


def _parse_csv_floats(s: str) -> List[float]:
    out: List[float] = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    return out


def _parse_csv_str(s: str) -> List[str]:
    return [t.strip() for t in str(s).split(",") if t.strip()]


@dataclass(frozen=True)
class SweepCfg:
    ckpt_name: str
    checkpoint: str


@dataclass(frozen=True)
class RunCfg:
    ckpt_name: str
    mode: str  # none|mean_logit|consis2|consisk|uwi
    indices: Optional[str]
    thr_mode: str  # fixed|otsu|topk
    thr: Optional[float]
    topk: Optional[float]
    tta: str  # none|flip|d4
    smooth_k: int
    use_minarea: bool
    min_area: int


def _call_eval(
    *,
    exe: Path,
    out_dir: Path,
    checkpoint: str,
    data_root: str,
    calib_root: Optional[str],
    device: str,
    batch_size: int,
    num_workers: int,
    eval_crop: int,
    window: int,
    stride: int,
    thr_mode: str,
    thr: Optional[float],
    topk: Optional[float],
    tta: str,
    smooth_k: int,
    use_minarea: bool,
    min_area: int,
    ensemble_mode: str,
    ensemble_indices: Optional[str],
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "eval_results.json"
    if out_json.is_file():
        return json.loads(out_json.read_text(encoding="utf-8"))

    args: List[str] = [
        sys.executable,
        str(exe),
        "--checkpoint",
        str(checkpoint),
        "--data_root",
        str(data_root),
        "--out_dir",
        str(out_dir),
        "--device",
        str(device),
        "--batch_size",
        str(int(batch_size)),
        "--num_workers",
        str(int(num_workers)),
        "--full_eval",
        "--eval_crop",
        str(int(eval_crop)),
        "--window",
        str(int(window)),
        "--stride",
        str(int(stride)),
        "--tta",
        str(tta),
        "--thr_mode",
        str(thr_mode),
        "--smooth_k",
        str(int(smooth_k)),
    ]

    if calib_root:
        args += ["--calib_root", str(calib_root)]

    if thr_mode == "fixed":
        assert thr is not None
        args += ["--thr", str(float(thr))]
    elif thr_mode == "topk":
        assert topk is not None
        args += ["--topk", str(float(topk))]
    elif thr_mode == "otsu":
        pass
    else:
        raise ValueError(f"Unknown thr_mode={thr_mode}")

    if use_minarea:
        args += ["--use_minarea", "--min_area", str(int(min_area))]

    if ensemble_mode != "none":
        args += ["--use_ensemble_pred", "--ensemble_strategy", str(ensemble_mode)]
        if ensemble_indices:
            args += ["--ensemble_indices", str(ensemble_indices)]

    env = os.environ.copy()
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")

    subprocess.run(args, check=True, env=env)
    return json.loads(out_json.read_text(encoding="utf-8"))


def _preset(name: str) -> Dict[str, object]:
    name = str(name or "medium").lower()
    if name == "fixed_only":
        return {
            "fixed_thrs": [0.5],
            "topks": [],
            "otsu": False,
            "ttas": ["none"],
            "modes": ["none", "mean_logit"],
        }
    if name == "fast":
        return {
            "fixed_thrs": [0.1, 0.2, 0.3, 0.5],
            "topks": [0.01, 0.02, 0.05],
            "otsu": True,
            "ttas": ["none", "flip"],
            "modes": ["none", "mean_logit", "uwi"],
        }
    if name == "full":
        return {
            "fixed_thrs": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
            "topks": [0.002, 0.005, 0.01, 0.02, 0.05, 0.1],
            "otsu": True,
            "ttas": ["none", "flip", "d4"],
            "modes": ["none", "mean_logit", "consis2", "uwi", "consisk"],
        }
    # medium default
    return {
        "fixed_thrs": [0.05, 0.1, 0.2, 0.3, 0.5],
        "topks": [0.005, 0.01, 0.02, 0.05],
        "otsu": True,
        "ttas": ["none", "flip"],
        "modes": ["none", "mean_logit", "consis2", "uwi", "consisk"],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Zero-shot evaluation sweep")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--calib_root", type=str, default="", help="Optional calibration root (use SOURCE domain for zero-shot).")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--eval_crop", type=int, default=256)
    # Match eval_dino_head defaults (windowed inference), but allow disabling via <=0.
    p.add_argument("--window", type=int, default=256)
    p.add_argument("--stride", type=int, default=256)

    p.add_argument("--checkpoints", type=str, required=True, help="Comma-separated checkpoint paths.")
    p.add_argument("--ckpt_names", type=str, default="", help="Comma-separated names matching --checkpoints.")

    p.add_argument("--preset", type=str, default="medium", choices=["fixed_only", "fast", "medium", "full"])
    p.add_argument("--ensemble_indices", type=str, default="3,4", help="Comma-separated indices or transformer layer numbers.")

    p.add_argument("--smooth_ks", type=str, default="3", help="Comma-separated ints.")
    p.add_argument("--use_minarea", action="store_true", default=True)
    p.add_argument("--min_areas", type=str, default="256", help="Comma-separated ints (effective when --use_minarea).")

    # Optional explicit override of thresholds/topk
    p.add_argument("--fixed_thrs", type=str, default="")
    p.add_argument("--topks", type=str, default="")
    p.add_argument("--no_otsu", action="store_true", default=False)
    p.add_argument("--ttas", type=str, default="")
    p.add_argument("--modes", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    exe = Path(__file__).resolve().parents[1] / "eval_dino_head.py"

    ckpts = _parse_csv_str(args.checkpoints)
    if len(ckpts) <= 0:
        raise ValueError("--checkpoints is empty")

    ckpt_names = _parse_csv_str(args.ckpt_names) if args.ckpt_names else []
    if ckpt_names and len(ckpt_names) != len(ckpts):
        raise ValueError("--ckpt_names must have the same length as --checkpoints when provided")
    if not ckpt_names:
        ckpt_names = [Path(p).parent.name for p in ckpts]

    sweep = _preset(args.preset)
    fixed_thrs: List[float] = list(sweep["fixed_thrs"])
    topks: List[float] = list(sweep["topks"])
    use_otsu: bool = bool(sweep["otsu"])
    ttas: List[str] = list(sweep["ttas"])
    modes: List[str] = list(sweep["modes"])

    if args.fixed_thrs:
        fixed_thrs = _parse_csv_floats(args.fixed_thrs)
    if args.topks:
        topks = _parse_csv_floats(args.topks)
    if args.no_otsu:
        use_otsu = False
    if args.ttas:
        ttas = _parse_csv_str(args.ttas)
    if args.modes:
        modes = _parse_csv_str(args.modes)

    smooth_ks = [int(x) for x in _parse_csv_str(args.smooth_ks)]
    min_areas = [int(x) for x in _parse_csv_str(args.min_areas)]
    use_minarea = bool(args.use_minarea)

    base_out = Path(args.out_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []

    run_list: List[Tuple[SweepCfg, RunCfg]] = []
    for ckpt_name, ckpt_path in zip(ckpt_names, ckpts):
        for mode in modes:
            mode = str(mode)
            for tta in ttas:
                for smooth_k in smooth_ks:
                    for min_area in (min_areas if use_minarea else [0]):
                        for t in fixed_thrs:
                            run_list.append(
                                (
                                    SweepCfg(ckpt_name=ckpt_name, checkpoint=ckpt_path),
                                    RunCfg(
                                        ckpt_name=ckpt_name,
                                        mode=mode,
                                        indices=(str(args.ensemble_indices) if mode != "none" else None),
                                        thr_mode="fixed",
                                        thr=float(t),
                                        topk=None,
                                        tta=str(tta),
                                        smooth_k=int(smooth_k),
                                        use_minarea=use_minarea,
                                        min_area=int(min_area),
                                    ),
                                )
                            )
                        if use_otsu:
                            run_list.append(
                                (
                                    SweepCfg(ckpt_name=ckpt_name, checkpoint=ckpt_path),
                                    RunCfg(
                                        ckpt_name=ckpt_name,
                                        mode=mode,
                                        indices=(str(args.ensemble_indices) if mode != "none" else None),
                                        thr_mode="otsu",
                                        thr=None,
                                        topk=None,
                                        tta=str(tta),
                                        smooth_k=int(smooth_k),
                                        use_minarea=use_minarea,
                                        min_area=int(min_area),
                                    ),
                                )
                            )
                        for k in topks:
                            run_list.append(
                                (
                                    SweepCfg(ckpt_name=ckpt_name, checkpoint=ckpt_path),
                                    RunCfg(
                                        ckpt_name=ckpt_name,
                                        mode=mode,
                                        indices=(str(args.ensemble_indices) if mode != "none" else None),
                                        thr_mode="topk",
                                        thr=None,
                                        topk=float(k),
                                        tta=str(tta),
                                        smooth_k=int(smooth_k),
                                        use_minarea=use_minarea,
                                        min_area=int(min_area),
                                    ),
                                )
                            )

    for sweep_cfg, rc in run_list:
        ckpt_tag = _slug(sweep_cfg.ckpt_name)
        mode_tag = _slug(rc.mode)
        idx_tag = _slug(rc.indices or "auto")
        tta_tag = _slug(rc.tta)
        post_tag = f"s{int(rc.smooth_k)}" + (f"_a{int(rc.min_area)}" if rc.use_minarea else "_noarea")

        if rc.thr_mode == "fixed":
            thr_tag = f"fixed_{rc.thr:g}"
        elif rc.thr_mode == "topk":
            thr_tag = f"topk_{rc.topk:g}"
        else:
            thr_tag = "otsu"

        out_dir = base_out / ckpt_tag / f"{mode_tag}_{idx_tag}" / f"tta_{tta_tag}" / post_tag / thr_tag

        rep = _call_eval(
            exe=exe,
            out_dir=out_dir,
            checkpoint=sweep_cfg.checkpoint,
            data_root=str(args.data_root),
            calib_root=(str(args.calib_root) if str(args.calib_root).strip() else None),
            device=str(args.device),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            eval_crop=int(args.eval_crop),
            window=int(args.window),
            stride=int(args.stride),
            thr_mode=str(rc.thr_mode),
            thr=rc.thr,
            topk=rc.topk,
            tta=str(rc.tta),
            smooth_k=int(rc.smooth_k),
            use_minarea=bool(rc.use_minarea and int(rc.min_area) > 0),
            min_area=int(rc.min_area),
            ensemble_mode=str(rc.mode),
            ensemble_indices=(str(rc.indices) if rc.mode != "none" else None),
        )

        row = {
            "run": str(out_dir.relative_to(base_out)).replace("\\", "/"),
            "ckpt_name": sweep_cfg.ckpt_name,
            "checkpoint": str(sweep_cfg.checkpoint),
            "data_root": str(args.data_root),
            "calib_root": (str(args.calib_root) if str(args.calib_root).strip() else ""),
            "ensemble_mode": str(rc.mode),
            "ensemble_indices": (str(rc.indices) if rc.indices else ""),
            "tta": str(rc.tta),
            "thr_mode": str(rc.thr_mode),
            "thr": rc.thr,
            "topk": rc.topk,
            "smooth_k": int(rc.smooth_k),
            "use_minarea": bool(rc.use_minarea and int(rc.min_area) > 0),
            "min_area": int(rc.min_area),
            "precision": float(rep.get("precision", 0.0)),
            "recall": float(rep.get("recall", 0.0)),
            "f1": float(rep.get("f1", 0.0)),
            "iou": float(rep.get("iou", 0.0)),
        }
        rows.append(row)

    (base_out / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    # Best overall and best fixed0.5/tta=none subsets per checkpoint.
    def _best(rows_in: Iterable[Dict[str, object]]) -> Optional[Dict[str, object]]:
        best: Optional[Dict[str, object]] = None
        for r in rows_in:
            if best is None or float(r["f1"]) > float(best["f1"]):
                best = r
        return best

    best_overall: Dict[str, Dict[str, object]] = {}
    best_fixed: Dict[str, Dict[str, object]] = {}
    for name in ckpt_names:
        rs = [r for r in rows if str(r["ckpt_name"]) == str(name)]
        bo = _best(rs)
        if bo is not None:
            best_overall[str(name)] = bo
        rs_fixed = [
            r
            for r in rs
            if (str(r["thr_mode"]) == "fixed")
            and (abs(float(r["thr"]) - 0.5) < 1e-12)
            and (str(r["tta"]) == "none")
        ]
        bf = _best(rs_fixed)
        if bf is not None:
            best_fixed[str(name)] = bf

    out = {"best_overall": best_overall, "best_fixed_thr0.5_ttaNone": best_fixed}
    (base_out / "best.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n== Best overall ==")
    for k in sorted(best_overall.keys()):
        r = best_overall[k]
        print(f"{k}: F1={float(r['f1']):.4f} P={float(r['precision']):.4f} R={float(r['recall']):.4f} | {r['run']}")

    print("\n== Best fixed (thr=0.5, tta=none) ==")
    for k in sorted(best_fixed.keys()):
        r = best_fixed[k]
        print(f"{k}: F1={float(r['f1']):.4f} P={float(r['precision']):.4f} R={float(r['recall']):.4f} | {r['run']}")


if __name__ == "__main__":
    main()
