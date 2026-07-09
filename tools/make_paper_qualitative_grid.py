from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import LEVIRCDDataset, get_test_transforms_full  # noqa: E402
from dino_head_core import (  # noqa: E402
    HeadCfg,
    DinoSiameseHead,
    DinoFrozenA0Head,
    filter_small_cc,
    threshold_map,
    tta_inference_prob,
)

BIFA_ROOT = ROOT / "baselines" / "BiFA"
if str(BIFA_ROOT) not in sys.path:
    sys.path.insert(0, str(BIFA_ROOT))
# dino_head_core imports the project-level `models` package.  BiFA's evaluator
# also imports a package named `models`, so clear the cached project package
# before importing BiFA utilities.
sys.modules.pop("models", None)
from tools.tile_test_cd import (  # noqa: E402
    infer_prob_full as infer_baseline_prob_full,
    load_model as load_baseline_model,
    first_existing_dir,
    resolve_by_stem,
    split_root,
)


OUT_DIR = ROOT / "outputs" / "paper_qualitative_2026-07-09"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CELL_SIZE = 180

COLS = ["T1", "T2", "GT", "DLV-CD", "BiFA", "DMINet", "ACABFNet", "AANet", "SNUNet", "BIT"]
BASELINE_LABELS = ["BiFA", "DMINet", "ACABFNet", "AANet", "SNUNet", "BIT"]
BASELINE_MODEL_NAMES = {
    "BiFA": "bifa",
    "DMINet": "dminet",
    "ACABFNet": "acabfnet",
    "AANet": "aanet",
    "SNUNet": "snunet",
    "BIT": "bit",
}

DIRECTIONS = [
    {
        "direction": "LEVIR->WHU",
        "target_root": ROOT / "data" / "WHUCD",
        "sample": "0_10",
        "scene": "urban buildings / road texture",
        "dlv_ckpt": ROOT / "outputs" / "retrain_2026-02-06" / "levir2whu" / "best.pt",
        "dlv_thr": 0.5,
    },
    {
        "direction": "WHU->LEVIR",
        "target_root": ROOT / "data" / "LEVIR-CD",
        "sample": "test_10",
        "scene": "sparse rural buildings / curved roads",
        "dlv_ckpt": ROOT / "outputs" / "retrain_2026-02-06" / "whu2levir" / "best.pt",
        "dlv_thr": 0.5,
    },
    {
        "direction": "S2Looking->WHU",
        "target_root": ROOT / "data" / "WHUCD",
        "sample": "0_0",
        "scene": "roof texture / shadows / standard aerial target",
        "dlv_ckpt": ROOT / "outputs" / "task4_s2looking_train" / "run1" / "best.pt",
        "dlv_thr": 0.5,
    },
    {
        "direction": "S2Looking->LEVIR",
        "target_root": ROOT / "data" / "LEVIR-CD",
        "sample": "test_1",
        "scene": "rural vegetation / sparse small buildings",
        "dlv_ckpt": ROOT / "outputs" / "task4_s2looking_train" / "run1" / "best.pt",
        "dlv_thr": 0.5,
    },
    {
        "direction": "DSIFN->WHU",
        "target_root": ROOT / "data" / "WHUCD",
        "sample": "0_1",
        "scene": "road / vegetation / pseudo-change textures",
        "dlv_ckpt": ROOT / "outputs" / "fixed05_traincal_2026-07-06" / "dsifn_tv97_fp3_3ep" / "last.pt",
        "dlv_thr": 0.5,
    },
    {
        "direction": "DSIFN->LEVIR",
        "target_root": ROOT / "data" / "LEVIR-CD",
        "sample": "test_22",
        "scene": "hilly rural background / sparse small buildings / curved roads",
        "dlv_ckpt": ROOT / "outputs" / "fixed05_traincal_2026-07-08" / "dsifn_tv98_fp3_3ep" / "best.pt",
        "dlv_thr": 0.78,
    },
]


def baseline_result_json(label: str, direction: str) -> Path:
    src, tgt = direction.split("->")
    if label == "DMINet" and src == "S2Looking":
        return ROOT / "baselines" / "BiFA" / "experiments" / "tile_eval_retrain_260708" / "dminet" / f"{src}2{tgt}" / "tile_eval_results.json"
    if label == "AANet" and direction in {"WHU->LEVIR", "S2Looking->LEVIR"}:
        return ROOT / "baselines" / "BiFA" / "experiments" / "tile_eval_retrain_260708" / "aanet" / f"{src}2{tgt}" / "tile_eval_results.json"
    if label == "SNUNet" and direction == "S2Looking->LEVIR":
        return ROOT / "baselines" / "BiFA" / "experiments" / "tile_eval_retrain_260708" / "snunet" / f"{src}2{tgt}" / "tile_eval_results.json"
    if label in {"SNUNet", "BIT"}:
        return ROOT / "baselines" / "BiFA" / "experiments" / "tile_eval_extra_260708" / label.lower() / f"{src}2{tgt}" / "tile_eval_results.json"
    return ROOT / "baselines" / "BiFA" / "experiments" / "tile_eval_260707" / f"{label}_{src}2{tgt}_tile256" / "tile_eval_results.json"


def load_target_sample(root: Path, stem: str) -> Tuple[Image.Image, Image.Image, np.ndarray, str]:
    split = split_root(root, "test")
    a_dir = first_existing_dir(split, ("A", "Image1", "T1", "t1", "img1", "im1"))
    b_dir = first_existing_dir(split, ("B", "Image2", "T2", "t2", "img2", "im2"))
    l_dir = first_existing_dir(split, ("label", "Label", "GT", "gt", "mask", "masks", "labels"))
    a_path = resolve_by_stem(a_dir, stem)
    b_path = resolve_by_stem(b_dir, stem)
    l_path = resolve_by_stem(l_dir, stem)
    img_a = Image.open(a_path).convert("RGB")
    img_b = Image.open(b_path).convert("RGB")
    label = (np.asarray(Image.open(l_path).convert("L")) > 0).astype(np.uint8)
    return img_a, img_b, label, a_path.stem


def load_dlv_model(checkpoint: Path):
    ckpt = torch.load(checkpoint, map_location=DEVICE)
    load_cfg = ckpt.get("cfg") if isinstance(ckpt, dict) else None
    cfg = HeadCfg()
    if isinstance(load_cfg, dict):
        cfg.arch = load_cfg.get("arch", getattr(cfg, "arch", "dlv"))
        cfg.a0_layer = load_cfg.get("a0_layer", getattr(cfg, "a0_layer", 12))
        cfg.use_layer_ensemble = load_cfg.get("use_layer_ensemble", cfg.use_layer_ensemble)
        cfg.layer_head_ch = load_cfg.get("layer_head_ch", cfg.layer_head_ch)
        if load_cfg.get("selected_layers") is not None:
            cfg.selected_layers = tuple(int(x) for x in load_cfg["selected_layers"])
    arch = load_cfg.get("arch", getattr(cfg, "arch", "dlv")) if isinstance(load_cfg, dict) else "dlv"
    def _local_dino_name() -> str:
        name = load_cfg.get("dino_name", cfg.dino_name) if isinstance(load_cfg, dict) else cfg.dino_name
        if "dinov3-vitb16" in str(name).lower() or str(name).startswith("facebook/dinov3"):
            return str(ROOT / "dinov3-vitb16")
        return str(name)

    dino_name = _local_dino_name()

    if arch == "a0":
        if DinoFrozenA0Head is None:
            raise RuntimeError("DinoFrozenA0Head unavailable for A0 checkpoint")
        model = DinoFrozenA0Head(
            dino_name=dino_name,
            layer=load_cfg.get("a0_layer", cfg.a0_layer) if isinstance(load_cfg, dict) else cfg.a0_layer,
            use_whiten=load_cfg.get("use_whiten", cfg.use_whiten) if isinstance(load_cfg, dict) else cfg.use_whiten,
        ).to(DEVICE)
    else:
        model = DinoSiameseHead(
            dino_name=dino_name,
            selected_layers=cfg.selected_layers,
            use_whiten=load_cfg.get("use_whiten", cfg.use_whiten) if isinstance(load_cfg, dict) else cfg.use_whiten,
            use_domain_adv=load_cfg.get("use_domain_adv", cfg.use_domain_adv) if isinstance(load_cfg, dict) else cfg.use_domain_adv,
            domain_hidden=load_cfg.get("domain_hidden", cfg.domain_hidden) if isinstance(load_cfg, dict) else cfg.domain_hidden,
            domain_grl=load_cfg.get("domain_grl", cfg.domain_grl) if isinstance(load_cfg, dict) else cfg.domain_grl,
            use_style_norm=load_cfg.get("use_style_norm", cfg.use_style_norm) if isinstance(load_cfg, dict) else cfg.use_style_norm,
            proto_path=load_cfg.get("proto_path", cfg.proto_path) if isinstance(load_cfg, dict) else cfg.proto_path,
            proto_weight=load_cfg.get("proto_weight", cfg.proto_weight) if isinstance(load_cfg, dict) else cfg.proto_weight,
            boundary_dim=load_cfg.get("boundary_dim", cfg.boundary_dim) if isinstance(load_cfg, dict) else cfg.boundary_dim,
            use_layer_ensemble=load_cfg.get("use_layer_ensemble", cfg.use_layer_ensemble) if isinstance(load_cfg, dict) else cfg.use_layer_ensemble,
            layer_head_ch=load_cfg.get("layer_head_ch", cfg.layer_head_ch) if isinstance(load_cfg, dict) else cfg.layer_head_ch,
        ).to(DEVICE)
    model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
    model.eval()
    return model


@torch.no_grad()
def infer_dlv_mask(checkpoint: Path, data_root: Path, stem: str, thr: float) -> np.ndarray:
    model = load_dlv_model(checkpoint)
    ds = LEVIRCDDataset(data_root, split="test", transform=get_test_transforms_full(), crop_size=256)
    idx = ds.img_names.index(stem)
    item = ds[idx]
    img_a = item["img_a"].unsqueeze(0).to(DEVICE)
    img_b = item["img_b"].unsqueeze(0).to(DEVICE)
    ensemble_cfg = {"mode": "mean_logit", "indices": [3, 4]}
    prob = tta_inference_prob(
        model=model,
        img_a=img_a,
        img_b=img_b,
        device=str(DEVICE),
        window=256,
        stride=256,
        use_ensemble=True,
        ensemble_cfg=ensemble_cfg,
        tta_mode="none",
    )
    prob = F.avg_pool2d(prob, kernel_size=3, stride=1, padding=1)
    prob_np = prob[0, 0].detach().float().cpu().numpy()
    pred, _ = threshold_map(prob_np, "fixed", thr, 0.01)
    pred = filter_small_cc(pred, min_area=256)
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return pred.astype(np.uint8)


@torch.no_grad()
def infer_baseline_mask(label: str, direction: str, data_root: Path, stem: str) -> np.ndarray:
    result_path = baseline_result_json(label, direction)
    meta = json.loads(result_path.read_text(encoding="utf-8"))
    ckpt = Path(meta["checkpoint"])
    if not ckpt.is_absolute():
        ckpt = BIFA_ROOT / ckpt
    model, _, _, _ = load_baseline_model(BASELINE_MODEL_NAMES[label], ckpt, DEVICE)
    img_a, img_b, _, _ = load_target_sample(data_root, stem)
    prob = infer_baseline_prob_full(model, img_a, img_b, tile=256, stride=256, device=DEVICE)
    pred = (prob > 0.5).astype(np.uint8)
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return pred


def crop_box(label: np.ndarray, pad: int = 96) -> Tuple[int, int, int, int]:
    ys, xs = np.where(label > 0)
    h, w = label.shape
    if len(xs) == 0:
        return 0, 0, w, h
    x0, x1 = int(xs.min()), int(xs.max() + 1)
    y0, y1 = int(ys.min()), int(ys.max() + 1)
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    side = max(x1 - x0, y1 - y0) + 2 * pad
    side = max(256, min(side, max(w, h)))
    x0 = max(0, min(w - side, cx - side // 2))
    y0 = max(0, min(h - side, cy - side // 2))
    x1 = min(w, x0 + side)
    y1 = min(h, y0 + side)
    return int(x0), int(y0), int(x1), int(y1)


def resize_rgb(arr: np.ndarray, size: int) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8)).resize((size, size), Image.Resampling.BILINEAR)


def mask_to_bw(mask: np.ndarray, size: int) -> Image.Image:
    img = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    return img.resize((size, size), Image.Resampling.NEAREST).convert("RGB")


def error_map(pred: np.ndarray, gt: np.ndarray, size: int) -> Image.Image:
    pred = pred > 0
    gt = gt > 0
    out = np.zeros((*gt.shape, 3), dtype=np.uint8)
    out[np.logical_and(pred, gt)] = (0, 180, 70)      # TP green
    out[np.logical_and(pred, ~gt)] = (230, 45, 45)    # FP red
    out[np.logical_and(~pred, gt)] = (45, 95, 230)    # FN blue
    return Image.fromarray(out).resize((size, size), Image.Resampling.NEAREST)


def draw_grid(rows: list[dict], out_path: Path):
    cell = CELL_SIZE
    header_h = 52
    row_label_w = 150
    row_h = cell + 42
    w = row_label_w + len(COLS) * cell
    h = header_h + len(rows) * row_h + 36
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
        small_font = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
        small_font = font

    for ci, col in enumerate(COLS):
        x = row_label_w + ci * cell + cell // 2
        draw.text((x, 22), col, fill=(0, 0, 0), anchor="mm", font=font)

    for ri, row in enumerate(rows):
        y0 = header_h + ri * row_h
        draw.text((8, y0 + cell // 2), row["direction"], fill=(0, 0, 0), anchor="lm", font=font)
        draw.text((8, y0 + cell // 2 + 20), row["sample"], fill=(80, 80, 80), anchor="lm", font=small_font)
        for ci, col in enumerate(COLS):
            x0 = row_label_w + ci * cell
            canvas.paste(row["images"][col], (x0, y0))
            draw.rectangle((x0, y0, x0 + cell - 1, y0 + cell - 1), outline=(220, 220, 220))

    legend_y = header_h + len(rows) * row_h + 10
    legend = [("TP", (0, 180, 70)), ("FP", (230, 45, 45)), ("FN", (45, 95, 230))]
    x = row_label_w
    for name, color in legend:
        draw.rectangle((x, legend_y, x + 18, legend_y + 18), fill=color)
        draw.text((x + 24, legend_y + 9), name, fill=(0, 0, 0), anchor="lm", font=font)
        x += 78
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, dict] = {"device": str(DEVICE), "rows": []}
    rows = []
    for row_cfg in DIRECTIONS:
        direction = row_cfg["direction"]
        data_root = Path(row_cfg["target_root"])
        stem = row_cfg["sample"]
        print(f"[Row] {direction} sample={stem}")
        img_a, img_b, gt, resolved_stem = load_target_sample(data_root, stem)
        box = crop_box(gt)
        x0, y0, x1, y1 = box
        gt_c = gt[y0:y1, x0:x1]
        images = {
            "T1": resize_rgb(np.asarray(img_a)[y0:y1, x0:x1], CELL_SIZE),
            "T2": resize_rgb(np.asarray(img_b)[y0:y1, x0:x1], CELL_SIZE),
            "GT": mask_to_bw(gt_c, CELL_SIZE),
        }
        preds = {}
        dlv_pred = infer_dlv_mask(Path(row_cfg["dlv_ckpt"]), data_root, resolved_stem, float(row_cfg["dlv_thr"]))
        preds["DLV-CD"] = dlv_pred
        images["DLV-CD"] = error_map(dlv_pred[y0:y1, x0:x1], gt_c, CELL_SIZE)

        for label in BASELINE_LABELS:
            print(f"  - {label}")
            pred = infer_baseline_mask(label, direction, data_root, resolved_stem)
            preds[label] = pred
            images[label] = error_map(pred[y0:y1, x0:x1], gt_c, CELL_SIZE)

        rows.append({"direction": direction, "sample": resolved_stem, "images": images})
        manifest["rows"].append(
            {
                "direction": direction,
                "sample": resolved_stem,
                "scene": row_cfg["scene"],
                "crop_box_xyxy": list(box),
                "dlv_checkpoint": str(row_cfg["dlv_ckpt"]),
                "dlv_threshold": float(row_cfg["dlv_thr"]),
                "baseline_result_json": {label: str(baseline_result_json(label, direction)) for label in BASELINE_LABELS},
            }
        )

    draw_grid(rows, OUT_DIR / "qualitative_grid_tp_fp_fn.png")
    with open(OUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Saved: {OUT_DIR / 'qualitative_grid_tp_fp_fn.png'}")
    print(f"Saved: {OUT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
