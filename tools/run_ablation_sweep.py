import argparse
import csv
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _recognized_bool(x: Any) -> bool:
    return isinstance(x, bool)


def _slugify(s: str, max_len: int = 120) -> str:
    s = s.strip()
    s = s.replace(os.sep, "-")
    s = re.sub(r"[^0-9a-zA-Z._=+-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    if len(s) > max_len:
        s = s[:max_len].rstrip("-")
    return s or "run"


def _ensure_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _product_grid(grid: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    keys = sorted(grid.keys())
    values_list = [_ensure_list(grid[k]) for k in keys]
    for vals in itertools.product(*values_list):
        yield dict(zip(keys, vals))


def _tee_run(cmd: List[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"[{_now_ts()}] CMD: {' '.join(cmd)}\n\n")
        f.flush()
        p = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)
        return p.wait()


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _maybe_load_best_f1_from_ckpt(ckpt_path: Path) -> Optional[float]:
    try:
        import torch
    except Exception:
        return None
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
    except Exception:
        return None
    if isinstance(ckpt, dict) and "best_f1" in ckpt:
        try:
            return float(ckpt["best_f1"])
        except Exception:
            return None
    return None


def _build_kv_args(params: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    for k, v in params.items():
        flag = k if k.startswith("--") else f"--{k}"
        if _recognized_bool(v):
            if v:
                args.append(flag)
            continue
        args.extend([flag, str(v)])
    return args


@dataclass
class SweepConfig:
    sweep_name: str
    out_root: Path
    train_script: str
    eval_script: str
    train_base_args: List[str]
    eval_args: List[str]
    grid: Dict[str, Any]
    skip_existing: bool = True
    copy_best: bool = False
    prune_nonbest_checkpoints: bool = False
    best_metric: str = "f1"


def _load_sweep_config(path: Path) -> SweepConfig:
    raw = _read_json(path)
    sweep_name = str(raw.get("sweep_name") or path.stem)
    out_root = Path(raw.get("out_root") or Path("outputs") / "sweeps" / sweep_name)
    train_script = str(raw.get("train_script") or "train_dino_head.py")
    eval_script = str(raw.get("eval_script") or "eval_dino_head.py")
    train_base_args = [str(x) for x in (raw.get("train_base_args") or [])]
    eval_args = [str(x) for x in (raw.get("eval_args") or [])]
    grid = raw.get("grid") or {}
    if not isinstance(grid, dict) or not grid:
        raise ValueError("config.grid must be a non-empty object")
    return SweepConfig(
        sweep_name=sweep_name,
        out_root=out_root,
        train_script=train_script,
        eval_script=eval_script,
        train_base_args=train_base_args,
        eval_args=eval_args,
        grid=grid,
        skip_existing=bool(raw.get("skip_existing", True)),
        copy_best=bool(raw.get("copy_best", False)),
        prune_nonbest_checkpoints=bool(raw.get("prune_nonbest_checkpoints", False)),
        best_metric=str(raw.get("best_metric", "f1")),
    )


def _run_id_from_params(params: Dict[str, Any]) -> str:
    items = [f"{k}={params[k]}" for k in sorted(params.keys())]
    return _slugify("_".join(items))


def _safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))


def _prune_checkpoints(run_dir: Path) -> None:
    for name in ("best.pt", "last.pt"):
        p = run_dir / name
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def main() -> int:
    p = argparse.ArgumentParser(description="Run ablation sweeps (train + eval) sequentially and keep best.")
    p.add_argument("--config", type=str, required=True, help="Path to sweep JSON config.")
    p.add_argument("--dry_run", action="store_true", help="Print commands without executing.")
    p.add_argument("--max_runs", type=int, default=0, help="Limit number of runs (0 means no limit).")
    args = p.parse_args()

    cfg = _load_sweep_config(Path(args.config))
    repo_root = Path(__file__).resolve().parents[1]
    out_root = (repo_root / cfg.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    runs_jsonl = out_root / "runs.jsonl"
    summary_csv = out_root / "summary.csv"
    best_json = out_root / "best.json"
    best_dir = out_root / "best"

    best_metric = cfg.best_metric
    best_score = float("-inf")
    best_run: Optional[Dict[str, Any]] = None
    best_run_dir: Optional[Path] = None
    summary_rows: List[Dict[str, Any]] = []

    run_count = 0
    for run_params in _product_grid(cfg.grid):
        run_id = _run_id_from_params(run_params)
        run_dir = out_root / run_id
        train_log = run_dir / "train.log"
        eval_log = run_dir / "eval.log"
        eval_dir = run_dir / "eval"
        eval_json_path = eval_dir / "eval_results.json"

        if cfg.skip_existing and eval_json_path.exists():
            record = {
                "time": _now_ts(),
                "status": "skipped_existing",
                "run_id": run_id,
                "run_dir": str(run_dir),
                "params": run_params,
                "eval_results_json": str(eval_json_path),
            }
            print(f"[{record['time']}] SKIP {run_id} (exists: {eval_json_path})")
            _append_jsonl(runs_jsonl, record)
            try:
                eval_res = _read_json(eval_json_path)
                f1 = eval_res.get(best_metric)
                if isinstance(f1, (int, float)):
                    summary_rows.append(
                        {
                            "run_id": run_id,
                            "status": "skipped_existing",
                            best_metric: float(f1),
                            "run_dir": str(run_dir),
                        }
                    )
            except Exception:
                pass
            continue

        run_dir.mkdir(parents=True, exist_ok=True)

        train_cmd = [sys.executable, str(repo_root / cfg.train_script)]
        train_cmd += cfg.train_base_args
        train_cmd += _build_kv_args(run_params)
        train_cmd += ["--out_dir", str(run_dir)]

        # Training
        t0 = time.time()
        print(f"\n[{_now_ts()}] ===== RUN {run_id} =====")
        if args.dry_run:
            print("TRAIN:", " ".join(train_cmd))
            train_rc = 0
        else:
            train_rc = _tee_run(train_cmd, cwd=repo_root, log_path=train_log)
        t1 = time.time()

        ckpt_path = run_dir / "best.pt"
        if not ckpt_path.exists():
            ckpt_path = run_dir / "last.pt"
        if train_rc != 0 or not ckpt_path.exists():
            record = {
                "time": _now_ts(),
                "status": "train_failed",
                "run_id": run_id,
                "run_dir": str(run_dir),
                "params": run_params,
                "train_rc": train_rc,
                "train_seconds": round(t1 - t0, 3),
            }
            print(f"[{record['time']}] TRAIN FAILED {run_id} (rc={train_rc})")
            _append_jsonl(runs_jsonl, record)
            summary_rows.append(
                {"run_id": run_id, "status": "train_failed", "run_dir": str(run_dir), best_metric: ""}
            )
            run_count += 1
            if args.max_runs and run_count >= args.max_runs:
                break
            continue

        val_best_f1 = _maybe_load_best_f1_from_ckpt(ckpt_path)

        # Eval
        eval_cmd = [sys.executable, str(repo_root / cfg.eval_script)]
        eval_cmd += ["--checkpoint", str(ckpt_path)]
        eval_cmd += ["--out_dir", str(eval_dir)]
        eval_cmd += cfg.eval_args

        if args.dry_run:
            print("EVAL :", " ".join(eval_cmd))
            eval_rc = 0
        else:
            eval_rc = _tee_run(eval_cmd, cwd=repo_root, log_path=eval_log)

        eval_res: Dict[str, Any] = {}
        if eval_rc == 0 and eval_json_path.exists():
            try:
                eval_res = _read_json(eval_json_path)
            except Exception:
                eval_res = {}

        score = eval_res.get(best_metric, None)
        score_f = float(score) if isinstance(score, (int, float)) else float("-inf")

        record = {
            "time": _now_ts(),
            "status": "ok" if eval_rc == 0 else "eval_failed",
            "run_id": run_id,
            "run_dir": str(run_dir),
            "params": run_params,
            "train_seconds": round(t1 - t0, 3),
            "train_rc": train_rc,
            "eval_rc": eval_rc,
            "ckpt": str(ckpt_path),
            "val_best_f1": val_best_f1,
            "eval_results_json": str(eval_json_path) if eval_json_path.exists() else None,
            "eval_metrics": eval_res,
        }
        _append_jsonl(runs_jsonl, record)

        summary_rows.append(
            {
                "run_id": run_id,
                "status": record["status"],
                best_metric: score if score is not None else "",
                "val_best_f1": val_best_f1 if val_best_f1 is not None else "",
                "run_dir": str(run_dir),
            }
        )

        if eval_rc == 0 and score_f > best_score:
            prev_best_dir = best_run_dir
            best_score = score_f
            best_run_dir = run_dir
            best_run = {
                "time": _now_ts(),
                "run_id": run_id,
                "run_dir": str(run_dir),
                "ckpt": str(ckpt_path),
                "val_best_f1": val_best_f1,
                "metric": best_metric,
                "score": best_score,
                "eval_results_json": str(eval_json_path) if eval_json_path.exists() else None,
                "params": run_params,
            }
            _write_json(best_json, best_run)
            print(f"[{best_run['time']}] NEW BEST: {best_metric}={best_score:.4f} @ {run_id}")
            if cfg.copy_best:
                best_dir.mkdir(parents=True, exist_ok=True)
                _safe_copy(ckpt_path, best_dir / ckpt_path.name)
                if eval_json_path.exists():
                    _safe_copy(eval_json_path, best_dir / "eval_results.json")
                _write_json(best_dir / "best_meta.json", best_run)
            if cfg.prune_nonbest_checkpoints and prev_best_dir is not None and prev_best_dir != best_run_dir:
                _prune_checkpoints(prev_best_dir)

        if cfg.prune_nonbest_checkpoints and best_run_dir is not None and run_dir != best_run_dir:
            _prune_checkpoints(run_dir)

        run_count += 1
        if args.max_runs and run_count >= args.max_runs:
            break

    _write_csv(summary_csv, summary_rows, fieldnames=["run_id", "status", best_metric, "val_best_f1", "run_dir"])
    if best_run is not None:
        print(f"\nBEST: {best_metric}={best_run['score']:.4f} @ {best_run['run_dir']}")
    print(f"Saved sweep logs: {runs_jsonl}")
    print(f"Saved sweep summary: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
