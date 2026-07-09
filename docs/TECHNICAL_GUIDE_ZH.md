# DLV-CD / DINOv3 变化检测项目：技术文档（中文）

本文档面向**代码复现、实验排查、开源使用**，内容会比论文更“工程化”。默认你已经阅读论文方法部分（DLF + MHE），这里只讲项目如何跑通、怎么复现实验、以及哪些设置算严格 zero-shot。

---

## 1. 项目目标与核心思路

**目标**：在遥感双时相变化检测中，利用**冻结的 DINOv3**（foundation model）作为特征提取器，仅训练一个轻量变化检测头，实现：

- **域内**（in-domain）稳定性能；
- **跨域**（zero-shot transfer）更稳健：LEVIR ↔ WHU、以及向 DSIFN / S2Looking 等数据集迁移；
- 在**噪声 / 光照 / 压缩**等成像质量扰动下保持“优雅退化”。

**核心模块**

- **Frozen DINOv3 Siamese Encoder**：对两时相输入共享权重提取多层特征 \(\{A_a^l, A_b^l\}\)。
- **Adapter-Difference（差分特征构造）**：把两时相同层特征变为差分特征 \(\{D^l\}\)（见 `models/dinov2_head.py::DifferenceModule`）。
- **DLF Decoder（轻量多尺度融合解码器）**：对 \(\{D^l\}\) 做 1×1 投影 + resize 对齐 + concat + 轻量卷积融合，输出 fused logit（见 `models/dinov2_head.py::MultiScaleFusionDecoder`）。
- **MHE（多头层级预测）**：每层差分特征 \(D^l\) 经过一个 layer-wise head 输出 logits \(\{z_l\}\)，与 fused logit \(z_K\) 一起做推理时集成与不确定性估计（推理逻辑主要在 `eval_dino_head.py` / `dino_head_core.py`）。

---

## 2. 代码结构速览

建议先从这几个入口看起：

- `train_dino_head.py`：训练入口（源域训练）
- `eval_dino_head.py`：评估入口（域内/跨域/滑窗/TTA/后处理/集成策略/合成扰动）
- `dino_head_core.py`：训练与评估的公共实现（阈值、后处理、TTA、集成等）
- `models/dinov2_head.py`：`DinoSiameseHead` + 差分模块 + DLF + layer-wise heads
- 可视化/分析：
  - `vis_tsne_model_feats.py`：PCA/t-SNE 嵌入可视化（支持合成双子图 `pca_tsne.png`）
  - `vis_freq_energy_layers.py`：频域能量分布分析（用于解释不同层互补性）
  - `tools/robustness_curves_f1.py`：合成扰动鲁棒性曲线（Gaussian/BC/JPEG）

> 注：仓库里还保留了早期的 SAM 蒸馏相关内容（legacy），论文与当前 DLV-CD 主线无关；开源时建议只保留 DLV-CD 相关入口。

---

## 3. 环境安装与离线设置

### 3.1 依赖安装

```bash
pip install -r requirements.txt
```

### 3.2 离线/受限环境（强烈建议）

在 Windows、公司内网或评测机上，建议设置：

```bash
set ALBUMENTATIONS_DISABLE_VERSION_CHECK=1
set NO_ALBUMENTATIONS_UPDATE=1
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
```

否则会出现版本检查/Hub 请求阻塞或超时。

### 3.3 DINOv3 权重来源

`--dino_name` 支持：

- HuggingFace 模型名（在线下载）
- 本地目录（推荐离线）：例如 `dinov3-vitb16/`（包含 `config.json` + `model.safetensors` 等）

> 开源不发布权重时，应从 git 索引中移除 `dinov3-*/`，但本地仍可保留用于实验。

---

## 4. 数据集与目录规范

项目的数据集读取封装在 `dataset.py`，建议使用统一结构：

```
data/<DATASET>/
  train/A, train/B, train/label
  val/A,   val/B,   val/label
  test/A,  test/B,  test/label
```

其中：

- `A` / `B`：两时相影像（对齐）
- `label`：二值变化掩码（0/1）

常用数据集：

- `data/LEVIR-CD`
- `data/WHUCD`
- `data/DSIFN-Dataset`（或你实际路径）
- `data/S2Looking`（或你实际路径）

### 4.1 xBD 灾害域评估（source-only / target-free）

如果要把论文扩展到灾害场景，建议把 xBD 作为**探索性目标域评估集**，而不是训练集。推荐表述：

> We used a labeled subset from the official xBD training data as an exploratory disaster-domain evaluation set. No xBD images or labels were used for training or model selection.

工程上建议：

- 源域仍然只用 `LEVIR-CD / WHU-CD / S2Looking / DSIFN` 之一训练；
- xBD 只用于最终测试，不参与训练、微调、阈值搜索、head 选择；
- 若需要定量指标，应使用 **official training split 中带标注的子集**，因为 challenge test 通常没有公开 GT；
- 严格 zero-shot 时，不要在 xBD 上搜索 best threshold。优先使用：
  - `--thr_mode fixed --thr 0.5`
  - 或只在**源域验证集**上确定阈值/固定推理流程

xBD 原始目录通常不是本项目的 `A/B/label` 结构。仓库提供了预处理脚本：

```bash
python tools/prepare_xbd_cd.py ^
  --raw_root data/xBD ^
  --out_root data/xBD-CD
```

该脚本会生成：

```text
data/xBD-CD/
  test/A
  test/B
  test/label
```

默认标签定义为：`post_disaster.json` 中 `minor-damage / major-damage / destroyed` 的建筑区域视为“变化”像素；`no-damage` 视为负类，`un-classified` 默认忽略。

---

## 5. 训练流程（源域训练）

### 5.1 最小可跑训练命令（例：LEVIR 训练）

```bash
python train_dino_head.py ^
  --data_root data/LEVIR-CD ^
  --out_dir outputs/levir_train ^
  --device cuda --epochs 200 --batch_size 8 --num_workers 2 --crop_size 256 ^
  --use_layer_ensemble --layer_head_ch 128 ^
  --ft_mode frozen
```

要点：

- `--ft_mode frozen`：冻结 backbone，训练更稳定、也更符合论文设定。
- `--use_layer_ensemble`：开启 MHE 的 layer-wise heads（训练时也会产生 `logits_all`）。
- `--layer_head_ch`：layer-wise head 的通道宽度（工程上 128 往往更稳）。

### 5.2 训练输出

训练目录（`--out_dir`）通常包含：

- `best.pt` / `last.pt`：权重
- `config.json`：训练配置（可用于复现）
- `metrics.jsonl`：每个 epoch 的记录（如有）
- 可视化样例（如开启）

> `train_dino_head.py` 支持 `--config <json>`：把某个 `config.json` 当默认值加载，再用命令行覆盖差异参数（便于复现/续训）。

---

## 6. 评估流程（域内 / 跨域）

### 6.1 严格 zero-shot（推荐写论文的协议）

严格 zero-shot 的关键是：**不能用目标域的 GT 去选择阈值或调参**。

推荐固定：

- `--thr_mode fixed --thr 0.5`
- 固定的后处理与推理流程（是否 TTA、smooth、min_area、是否 MHE、集成策略等）

示例（LEVIR→WHU）：

```bash
python eval_dino_head.py ^
  --checkpoint outputs/levir_train/best.pt ^
  --data_root data/WHUCD ^
  --device cuda --batch_size 1 --num_workers 0 ^
  --full_eval --eval_crop 256 ^
  --thr_mode fixed --thr 0.5 ^
  --smooth_k 3 --use_minarea --min_area 256 ^
  --use_ensemble_pred --ensemble_strategy mean_logit --ensemble_indices 3,4
```

### 6.2 非严格协议（仅用于调试/上限参考，不建议写成 strict zero-shot）

以下做法会引入“目标域泄露”风险：

- `--thr_mode val_best`：在**某个 VAL** 上用 GT 搜索最佳阈值；如果这个 VAL 属于目标域，则不严格。
- 在目标域上网格搜索 `thr/topk/smooth/min_area/TTA/ensemble` 并挑 “best”。

如果一定要做（比如做 ablation 或 sanity check），建议明确写成：

- “oracle threshold / tuned on target” 或 “upper bound”

### 6.3 阈值策略说明（`--thr_mode`）

- `fixed`：固定阈值 `--thr`（最适合 strict zero-shot）
- `otsu`：Otsu 自适应（不需要 GT，但结果可能不稳定）
- `topk`：选择 top-k 像素为变化（不需要 GT，但隐含先验比例）
- `val_best`：用 VAL 的 GT 搜索最佳 fixed 阈值（若 VAL=目标域，则泄露）

### 6.4 后处理（推理阶段）

常用后处理开关：

- `--smooth_k K`：对概率图做均值平滑（`avg_pool2d`）
- `--use_minarea --min_area N`：连通域过滤（去除小区域噪声）
- 滑窗推理（大图/非裁剪）：
  - `--window` / `--stride`（`--full_eval` 时生效）

### 6.5 TTA（Test-Time Augmentation）

`--tta` 支持：

- `none`：不使用
- `flip`：水平翻转
- `d4`：旋转/翻转组合（更慢但有时更稳）

> 严格意义上，TTA 属于“推理时固定流程”的一部分：只要你事先规定、且不在目标域上调参挑选，就不算泄露。

---

## 7. MHE / 集成策略（推理阶段）

### 7.1 MHE on/off

- 不开 MHE：相当于只使用 fused 分支（或单头预测）
- 开 MHE：启用 layer-wise heads + fused head 的多头输出，在推理阶段做集成

在 CLI 中体现为：

- `--use_ensemble_pred`
- `--ensemble_strategy ...`
- `--ensemble_indices 3,4`（手动选用第 3/4 个 head；索引含义以代码输出为准）

### 7.2 常用集成策略（`--ensemble_strategy`）

常用且稳定的组合：

- `mean_logit`：logit 平均（论文 Eq.(8) 的常见实现）
- `weighted_logit`：基于校准集 F1 生成权重（注意校准集选取要避免目标域泄露）
- `topk`：自动挑选 top-k heads（同样依赖校准集/VAL 的度量）

本仓库还包含一些“自适应/不确定性相关”的推理策略（例如 `uwi/consis2/...`），用于研究或增强推理表达；是否用于论文需看你是否能保证协议严格与表述一致。

---

## 8. 合成扰动鲁棒性评估（Gaussian / BC / JPEG）

评估入口仍然是 `eval_dino_head.py`，通过 `--corrupt` 与对应参数控制：

- `--corrupt gaussian --gaussian_sigma 0.03`
- `--corrupt bc --bc_brightness 0.3 --bc_contrast 0.3`
- `--corrupt jpeg --jpeg_quality 55`

要做 sweep，建议用 `tools/robustness_curves_f1.py`（或你自己封装的 sweep 脚本）批量运行并画曲线。

推荐论文展示方式：

- 直接画 \(\Delta\)F1 vs severity（比堆表格更紧凑）
- 严格 zero-shot：阈值与推理流程固定（例如统一 `thr=0.5`）

---

## 9. 频域能量分布分析（解释层互补性）

`vis_freq_energy_layers.py` 用于计算不同层特征/差分特征的频域能量分布（FFT），常用于支撑这样的论点：

- 浅层：更多高频（边界/细节）
- 深层：更多低频（语义/稳定）
- 跨域/跨分辨率下，高频更易失配，因此浅层与 fused 的互补更重要

一般做法（工程建议）：

1. 对 LEVIR/WHU 各抽样若干对图像；
2. 提取层特征或差分特征（l=3/6/9/12）；
3. 对特征图做 2D FFT，径向平均得到能量谱；
4. 汇总成 8 张图：两数据集 × 若干层（或 feature vs diff 各一组）。

该分析通常**不需要重新训练**，使用现有 checkpoint 或 pretrained backbone 即可。

---

## 10. 嵌入可视化（PCA / t-SNE）

脚本：`vis_tsne_model_feats.py`

两种模式：

1) **pretrained 模式（推荐画 LEVIR vs WHU 的“基础表征差异”）**

```bash
python vis_tsne_model_feats.py ^
  --data_roots "data/LEVIR-CD,data/WHUCD" ^
  --data_names "LEVIR-CD,WHU-CD" ^
  --dino_name dinov3-vitb16 --layer 12 --which mix ^
  --split test --crop 256 --num_samples 300 --num_workers 0 ^
  --out_dir outputs/tsne_pretrained ^
  --tsne --pca_dim 50 --tsne_perplexity 30 --tsne_iter 1000
```

输出：

- `pca.png`
- `tsne.png`
- `pca_tsne.png`（左右双子图，带 (a)(b)，用于论文排版）

2) **checkpoint 模式（画 head 特征或 backbone 特征）**

```bash
python vis_tsne_model_feats.py ^
  --checkpoint outputs/<train_out>/best.pt ^
  --src_root data/LEVIR-CD --tgt_root data/WHUCD --split test ^
  --feature head --which both_avg ^
  --num_samples 300 --out_dir outputs/tsne_head ^
  --tsne
```

---

## 11. 推荐的“论文级复现实验清单”（不含泄露）

下面这套组合通常能覆盖论文结果与图表：

1. **域内**：LEVIR→LEVIR、WHU→WHU（固定 `thr=0.5`，固定后处理）
2. **跨域**：LEVIR→WHU、WHU→LEVIR（同样固定协议）
3. **MHE ablation**：开/不开 MHE；或固定使用 `mean_logit` + indices=3,4
4. **合成扰动**：Gaussian/BC/JPEG 的 \(\Delta\)F1 曲线（两迁移方向都画）
5. **解释性分析**：频域能量分布 + PCA/t-SNE 嵌入图（用于补强“为什么层级互补/为什么冻结 backbone 更稳”）

---

## 12. 常见坑与排查建议

- **跑很慢**：
  - `--full_eval` + 滑窗推理 + 多 TTA + 多阈值/多策略 sweep 会指数级变慢；
  - 建议先固定：`batch_size=1`、`num_workers=0`、只跑 1 个设置 sanity check。
- **zero-shot 泄露**：
  - 目标域上调 `thr` / `topk` / `min_area` / `smooth_k` / `TTA` 并挑 best，都属于“目标域调参”；
  - 写论文时建议只报告“固定协议”版本；其他版本最多放附录/上限参考并明确说明。
- **Windows 多进程 DataLoader 问题**：
  - 评估时 `--num_workers 0` 更稳；
  - 脚本里也尽量避免网络检查。

---

## 13. 开源发布建议（不发布权重/数据）

- `.gitignore` 已忽略 `data/`、`outputs/`、`dinov3-*`、`*.safetensors` 等。
- 但 **.gitignore 不会移除已被 git 跟踪的大文件**，推送前需从索引移除：

```bash
git rm --cached -r .idea dinov3-*
git rm --cached prototypes*.npy
```

如需发布权重，建议使用 Git LFS。
