# DINO Change Detection Head（跨域 / Zero-shot）

本项目的 `train_dino_head.py` / `eval_dino_head.py` 提供一套“冻结 DINO 特征 + 轻量变化检测 Head”的跨域变化检测方案，目标是在**不使用目标域标签**（也不强制使用目标域图像）的情况下，提升跨域泛化（例如 WHU↔LEVIR）。

当前这版（`a6d7d2e`：凸优化层间特征加权）以 **凸优化权重的多头集成** 为主线：训练出 `logits_all`（各层 head + fused），推理时在源域 val 上拟合最优融合权重，再迁移到目标域测试。

---

## 1. 代码结构（常用入口）

- 模型：`models/dinov2_head.py`（`DinoSiameseHead`）
- 训练/评估公共逻辑：`dino_head_core.py`
- 训练脚本：`train_dino_head.py`
- 评估脚本：`eval_dino_head.py`

---

## 2. Multi-Head 是怎么工作的？

启用 `--use_layer_ensemble` 后：

- 从 DINO 的多个 block 抽取特征（默认 4 层：`(3,6,9,12)`）。
- 每层做差分特征（`DifferenceModule`），各自通过 `LayerHead` 输出一个像素级 logit。
- 同时将多层差分特征融合，走一个 fused head 输出 fused logit。
- forward 返回：
  - `pred`：默认 fused logit（不做集成时就是它）。
  - `logits_all`：形状 `[K,B,1,H,W]`，`K=层数+1`，最后一个是 fused。

日志里 `head[0..K-2]` 是各层 head，`head[K-1]` 是 fused。

---

## 3. 推荐训练命令（稳定起点）

经验上 `layer_head_ch=128` 比 192 更稳（跨域波动更小）。下面以 **WHU 训练** 为例（PowerShell 续行）：
python train_dino_head.py --data_root data/WHUCD --out_dir outputs/dino_head_cd --device auto --epochs 150 --batch_size 8 --crop_size 256 --bce_weight 0.5 --dice_weight 0.5 --thr_mode fixed --thr 0.5 --use_layer_ensemble --layer_head_ch 128 --head_aux_weight 0.25 --head_cons_weight 0.05 --use_ensemble_pred
```powershell
python train_dino_head.py `
  --data_root data/WHUCD `
  --out_dir outputs/dino_head_cd `
  --device auto `
  --epochs 200 `
  --batch_size 8 `
  --crop_size 256 `
  --bce_weight 0.5 --dice_weight 0.5 `
  --thr_mode fixed --thr 0.5 `
  --use_layer_ensemble --layer_head_ch 128 `
  --head_aux_weight 0.25 `
  --head_cons_weight 0.05
  --use_ensemble_pred 
```

注意：命令行参数都必须带 `--`（例如 `--self_sup_weight 0.4`）。

---

## 4. 推荐评估命令（WHU→LEVIR 最稳主线）
python eval_dino_head.py --checkpoint outputs/dino_head_cd/best.pt --data_root data/LEVIR-CD --calib_root data/WHUCD --full_eval --use_ensemble_pred --ensemble_strategy cvx_nll --ensemble_indices 3,4 --cvx_lambda 1e-3 --cvx_steps 200 --cvx_lr 0.5 --cvx_max_pixels 400000 --cvx_pos_weight auto --thr_mode fixed --thr 0.5 --smooth_k 3 --use_minarea --min_area 256

python eval_dino_head.py --checkpoint outputs/dino_head_cd/best.pt --data_root data/WHUCD --calib_root data/LEVIR-CD --full_eval --use_ensemble_pred --ensemble_strategy topk --ensemble_topk 1 --thr_mode fixed --thr 0.5 --smooth_k 3 --use_minarea --min_area 256

python eval_dino_head.py --checkpoint outputs/dino_head_cd/best.pt --data_root data/WHUCD --calib_root data/LEVIR-CD --full_eval
--use_ensemble_pred --ensemble_strategy mean_logit --ensemble_indices 3,4 --thr_mode fixed --thr 0.5 --smooth_k 3 --use_minarea --min_area 256

python eval_dino_head.py --checkpoint outputs/dino_head_cd/best.pt --data_root data/WHUCD --calib_root data/LEVIR-CD --full_eval
--thr_mode fixed --thr 0.5 --smooth_k 3 --use_minarea --min_area 256

核心要点：
- `--calib_root data/WHUCD`：用源域（WHU）val 拟合 ensemble 权重（凸优化），更贴近“源域先验”。
- `--ensemble_indices 3,4`：只用 `{layer3, fused}`（你的实验里很稳的一组）。
- 后处理固定：`--smooth_k 3 --use_minarea --min_area 256`

### 4.1 主推：`cvx_nll`（凸优化权重）

```powershell
python eval_dino_head.py `
  --checkpoint outputs/dino_head_cd/best.pt `
  --data_root data/LEVIR-CD `
  --calib_root data/WHUCD `
  --full_eval `
  --use_ensemble_pred `
  --ensemble_strategy cvx_nll --ensemble_indices 3,4 `
  --cvx_lambda 1e-3 --cvx_steps 200 --cvx_lr 0.5 --cvx_max_pixels 400000 --cvx_pos_weight auto `
  --thr_mode fixed --thr 0.5 `
  --smooth_k 3 --use_minarea --min_area 256
```
### --ensemble_topk 2：会先按源域 val F1 选 Top-2 layer heads + fused 再做 cvx

### 4.2 对照：只用 fused（不集成）

```powershell
python eval_dino_head.py `
  --checkpoint outputs/dino_head_cd/best.pt `
  --data_root data/LEVIR-CD `
  --full_eval `
  --thr_mode fixed --thr 0.5 `
  --smooth_k 3 --use_minarea --min_area 256
```

---

#### 可视化（可选）
评估脚本默认不保存可视化；加上 `--vis` 才会导出图像面板（默认写到 `out_dir/vis_eval`，可用 `--vis_dir` 指定）。
跨域评估通常更关心目标域的 test，可以用 `--vis_split test`（默认是 val）。

## 5. 消融建议（够写论文的最小闭环）

建议你把实验做成一张表（同一数据划分、同一阈值/后处理设置）：

1) **Base（fused-only）**：不启用 `--use_ensemble_pred`，只用 fused 的 `pred`
2) **+ Multi-Head**：训练启用 `--use_layer_ensemble`
3) **+ Convex Weighting（cvx_nll）**：`--use_ensemble_pred --ensemble_strategy cvx_nll`

建议固定的后处理：`--smooth_k 3 --use_minarea --min_area 256`。

---

## 6. 其它探索（建议写为负向/讨论/附录）

你之前探索过 TGFC、ADSA、MemTTA、Pixel-wise MoE 等；它们在跨域上容易出现“源域更好但目标域失灵/波动大”的现象。论文里建议作为：

- 负向消融：说明跨域中更复杂的 test-time 机制可能引入不稳定
- 讨论：不确定性与域偏移的关系、为什么“简单但稳健”的推理策略更可靠

---

## 7. 自动跑消融（睡觉挂机）

项目里提供了一个顺序跑实验的 sweep 脚本：`tools/run_ablation_sweep.py`。它会对 `grid` 里所有组合：
- 先跑 `train_dino_head.py`（每个 run 单独 `out_dir`，训练过程本身会保存 `best.pt`）
- 再跑 `eval_dino_head.py`（把目标域测试结果写入 `run_dir/eval/eval_results.json`）
- 自动汇总到 `outputs/sweeps/<sweep>/summary.csv`，并写 `best.json` 指向当前最优 run
- 可选清理：在 config 里设置 `prune_nonbest_checkpoints=true`，会自动删除非最优 run 的 `best.pt/last.pt` 来省磁盘

示例配置：`tools/sweep_example_whu2levir.json`（WHU 训练 → LEVIR 测试，`cvx_nll` 作为推理集成主线）。

```powershell
python tools/run_ablation_sweep.py --config tools/sweep_example_whu2levir.json
```

想跑你自己的消融：
- 直接复制一份 `tools/sweep_example_whu2levir.json` 改 `grid`（比如不同 `head_aux_weight/head_cons_weight/seed`）
- 把 `train_base_args` / `eval_args` 换成你当前最想固定的那套命令参数

断点续跑：
- `skip_existing=true` 时，若某个 run 已经存在 `run_dir/eval/eval_results.json`，就会自动跳过。
