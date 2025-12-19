# DINO Change Detection Head（跨域 / Zero-shot）

这个目录下的 `train_dino_head.py` / `eval_dino_head.py` 是一个“冻结 DINO 特征 + 轻量 CD Head”的变化检测方案，核心目标是跨域（例如 WHU↔LEVIR）在不使用目标域标签/图像（可选）的情况下保持较强泛化。

## 1. 核心思路（你现在跑的这套）

- **多层特征**：从 DINO 的多个 block 取特征（默认 4 层），浅层偏纹理/边缘，深层偏语义。
- **差分模块**：对每层做 `DifferenceModule`（`|Fa-Fb|` 与 `Fa+Fb` 拼接 + 卷积 + 通道注意力）。
- **主干解码器（MLP/轻解码器）**：多尺度 `1×1` 投影对齐后做融合与卷积细化，输出主 logit（`pred`）。
- **Multi-Head Ensemble（可选）**：每一层差分特征各自一个 `LayerHead` 出 logit，同时再做一个 fused head；推理时可做 Top‑k / 加权集成 + 不确定性统计（方差等）。

相关实现位置：
- 模型：`models/dinov2_head.py`（`DinoSiameseHead`）
- 训练/评估与集成逻辑：`dino_head_core.py`、`train_dino_head.py`、`eval_dino_head.py`

## 2. Multi-Head 是怎么工作的？

当训练/推理启用 `--use_layer_ensemble` 时：

- 对每个选中的 DINO 层（默认 4 层）都会得到一个 `diff` feature；
- 每个 `diff` 过一个轻量 `LayerHead`，上采样到原图大小，得到 `head[i]` 的像素级 logit；
- 另外再把多层 `diff` 做融合 + `HeavyDecoder` 得到一个 **fused logit**；
- 最终返回：
  - `pred`：默认是 fused logit（不开集成时也是它）
  - `logits_all`：`[K,B,1,H,W]`，其中 `K = 层数 + 1`，最后一个是 fused

所以你日志里 `head[0..K-2]` 是各层 `LayerHead`，`head[K-1]` 是 fused。

## 3. 推荐训练参数（你目前最稳的基线风格）

你已经验证过：`layer_head_ch=128` 比 `192` 更稳（跨域波动更小）。

一个常用的“稳健起点”（以 WHU 训练为例）：

```bash
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
  --head_cons_weight 0.05 `
  --use_ensemble_pred
```

注意：命令行里所有参数都必须带 `--`，比如 `--self_sup_weight 0.4`（你之前报错就是少了 `--`）。

## 4. 推荐评估命令（Top‑k / 加权集成）

### 4.1 只用 fused（不开集成）

```bash
python eval_dino_head.py --checkpoint outputs/dino_head_cd/best.pt --data_root data/LEVIR-CD --full_eval
```

### 4.2 Top‑k（常用：Top‑1）

Top‑k 会在 **VAL** 上先给每个 head 单独算 F1，然后选 “fused + Top‑k layer head” 做集成（logit 平均）。

```bash
python eval_dino_head.py `
  --checkpoint outputs/dino_head_cd/best.pt `
  --data_root data/LEVIR-CD `
  --full_eval `
  --use_ensemble_pred `
  --ensemble_strategy topk `
  --ensemble_topk 1
```

### 4.3 加权（权重来自 VAL 的 F1）

```bash
python eval_dino_head.py `
  --checkpoint outputs/dino_head_cd/best.pt `
  --data_root data/LEVIR-CD `
  --full_eval `
  --use_ensemble_pred `
  --ensemble_strategy weighted_logit `
  --ensemble_topk 1 `
  --ensemble_weight_norm softmax `
  --ensemble_weight_temp 1.0
```

`eval_dino_head.py` 会把选择结果保存到：
- `outputs/.../ensemble_cfg.json`

### 4.4 凸优化权重（cvx_nll，推荐）

在 VAL 上解一个凸优化（BCEWithLogits + 单纯形约束 + L2），得到**唯一稳定**的最优权重，然后用 `weighted_logit` 推理：

```bash
python eval_dino_head.py `
  --checkpoint outputs/dino_head_cd/best.pt `
  --data_root data/LEVIR-CD `
  --full_eval `
  --use_ensemble_pred `
  --ensemble_strategy cvx_nll `
  --ensemble_topk 1 `
  --cvx_lambda 1e-3 --cvx_steps 200 --cvx_lr 0.5 --cvx_max_pixels 400000
```

输出会保存到 `out_dir/ensemble_cfg.json`（包含 weights 与求解参数）。

跨域 zero-shot 常用方式（源域拟合权重，目标域测试）：

```bash
python eval_dino_head.py `
  --checkpoint outputs/dino_head_cd/best.pt `
  --data_root data/LEVIR-CD `
  --calib_root data/WHUCD `
  --full_eval `
  --use_ensemble_pred `
  --ensemble_strategy cvx_nll `
  --ensemble_topk 1 `
  --cvx_lambda 1e-3 --cvx_steps 200 --cvx_lr 0.5 --cvx_max_pixels 400000
```

如果你已经知道目标域里最稳的 head 组合（例如经常是 `layer3 + fused`），可以固定入选头再在源域上求凸最优权重：

```bash
python eval_dino_head.py `
  --checkpoint outputs/dino_head_cd/best.pt `
  --data_root data/LEVIR-CD `
  --calib_root data/WHUCD `
  --full_eval `
  --use_ensemble_pred `
  --ensemble_strategy cvx_nll `
  --ensemble_indices 3,4 `
  --cvx_lambda 1e-3 --cvx_steps 200 --cvx_lr 0.5 --cvx_max_pixels 400000
```

## 5. TGFC（Temperature‑Guided Feature Calibration，想法记录）

你提的 TGFC 本质是 **按置信度分桶**，对不确定区域做 temperature 校准（偏 calibration 指标，未必提升 F1）。

当前分支没有默认集成 TGFC 的开关（因为我们实测跨域 F1 受 precision/recall trade-off 影响很大，TGFC 容易把预测变保守）。

如果你要写论文，可以把 TGFC 作为“校准尝试/负向消融/讨论”，或在另一个实验分支中实现。

## 6. 边界感知 / 因果推断（训练开关还是评估开关？）

- **边界感知（boundary）**：属于网络结构/分支（`--boundary_dim` + `--boundary_weight` 等），通常需要 **训练时打开**；评估时只要加载对应 checkpoint（脚本会读 ckpt 里的 cfg），不需要再额外开同名参数。
- **因果推断 / 域对齐相关**（`--use_domain_adv`、`--use_style_norm`、`--lambda_domain`、`--lambda_consis`、`--self_sup_weight`、style aug 一组）：这些决定训练目标与训练时的数据扰动；同样主要是 **训练阶段** 的事。评估阶段只要 checkpoint 里启用了对应结构（比如 domain head / style norm），脚本会按 cfg 复现。

如果你要做“只改 eval，不改训练”的稳定器，那就优先用：
- `ensemble_strategy topk/weighted_logit`
- （可选）`tgfc_mode heuristic/fit_nll`

## 7. ADSA（方案2：想法记录）

ADSA 属于结构改动（双向 cross-attention + uncertainty mixing），需要重新训练/调参；我们实测在跨域 F1 上不稳定、提升有限，所以当前主线不作为推荐方案。

## 8. Memory-Augmented Test-Time Adaptation（方案3）

这是一个**纯评估侧（test-time）**的实验开关：在测试时维护一个“记忆库”，把高置信样本的特征与预测存进去；新样本用特征检索相似历史样本，再把“记忆预测”与当前预测做融合。

实现说明（简化版、可落地）：
- 特征：使用模型输出 `out["feat"]` 做全局平均池化得到 image-level embedding
- 记忆内容：embedding + 下采样后的预测概率图（默认 `32×32`），避免存整图
- 更新条件：只有当“图像级置信度”足够高才写入（默认 `--mem_update_conf 0.90`）
- 融合权重：与检索相似度成正比，最大不超过 `--mem_alpha_max`

开启命令示例（WHU→LEVIR）：

```bash
python eval_dino_head.py `
  --checkpoint outputs/dino_head_cd/best.pt `
  --data_root data/LEVIR-CD `
  --full_eval `
  --use_ensemble_pred --ensemble_strategy topk --ensemble_topk 1 `
  --mem_tta --mem_size 512 --mem_topk 8 --mem_pred_size 32 --mem_alpha_max 0.35 --mem_update_conf 0.90
```

建议先固定其它设置（阈值/平滑/minarea），只扫：
- `--mem_alpha_max`：`0.15 / 0.25 / 0.35`
- `--mem_update_conf`：`0.90 / 0.92 / 0.95`
