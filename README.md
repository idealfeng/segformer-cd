# SAM蒸馏遥感图像分割项目

知识蒸馏SAM模型用于遥感图像分割，针对Potsdam数据集优化。

## 🎯 项目目标

- **教师网络**: SAM ViT-H (636M参数)
- **学生网络**: SegFormer-B1 (13.7M参数)
- **核心创新**: Boundary-aware Distillation Loss（针对遥感边界模糊问题）
- **目标期刊**: SCI低区 / EI会议

## 📁 项目结构

```
project/
├── config.py                    # 配置文件 (核心！)
├── split_dataset.py             # 【1】数据集划分脚本 (防止泄漏)
├── dataset.py                   # 【2】数据加载器 (含数据增强)
├── losses.py                    # 【3】损失函数 (蒸馏Loss + 分割Loss)
|
├── models/
│   ├── __init__.py             # (使其成为一个包)
│   └── segformer.py            # 【4】学生网络: SegFormer-B1
|
├── utils/
│   ├── __init__.py             # (使其成为一个包)
│   ├── metrics.py              # 评估指标 (mIoU, F1等)
│   └── logger.py               # (新增) 日志记录器，比print更专业
│
├── train.py                     # 【5】训练主脚本
├── eval.py                      # 【6】评估主脚本
|
├── eval_teacher.py              # (新增) 专门用于评估SAM-H教师网络性能的脚本
|
├── .gitignore                   # (重要) Git忽略文件
└── README.md                    # 项目说明文档

# ==========================================================
#         以下是“数据资产”，与上面的“代码”分离
# ==========================================================

├── data/                          # (新增) 所有数据的根目录
│   ├── Potsdam_processed/       # 【输入】预处理后的训练数据
│   │   ├── images/              # (2904张图像)
│   │   └── labels/              # (2904张标签)
│   │
│   ├── teacher_outputs/         # 【输入】教师网络特征
│   │   ├── features_block30/    # ✅ Block 30特征
│   │   └── features_encoder/    # ✅ Encoder最终输出
│   │
│   └── splits/                  # 【输入】数据集划分文件
│       ├── train.txt
│       ├── val.txt
│       └── test.txt
│
└── outputs/                     # 【输出】所有实验产出
    ├── checkpoints/             # 模型权重
    ├── logs/                    # 训练日志 (.log, tensorboard)
    └── results/                 # 评估结果 (JSON, CSV)
        ├── visualizations/      # (可选) 可视化图片
        └── predictions/         # (可选) 预测的掩码图

## 🚀 快速开始

### 阶段0：环境配置

```bash
# 创建conda环境
conda create -n sam_distill python=3.9
conda activate sam_distill

# 安装依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy opencv-python tqdm
pip install timm  # SegFormer依赖
pip install tensorboard  # 可视化（可选）
```

### 阶段1：数据集划分

```bash
# 第一次运行，划分数据集
python split_dataset.py

# 输出：
# splits1/train.txt (1680张, 70%)
# splits1/val.txt (360张, 15%)
# splits1/test.txt (360张, 15%)
```

### 阶段2：Baseline评估（5个对比方法）

```bash
# 评估所有baseline模型
python eval_baselines.py

# 对比方法：
# 1. SAM ViT-H (用已提取的masks)
# 2. MobileSAM
# 3. FastSAM
# 4. DeepLabV3+
# 5. SegFormer-B1 (无蒸馏)

# 输出：
# results/baseline_results.csv
# 包含6个指标：mIoU, F1, OA, Params, FLOPs, FPS
```

### 阶段3：训练学生网络

```bash
# Baseline训练（无蒸馏）
python train.py --exp_name baseline --no_distill

# 完整训练（带蒸馏）
python train.py --exp_name full_distill

# 消融实验
python train.py --exp_name ablation_logit --distill_logit_only
python train.py --exp_name ablation_feat --distill_feat_only
python train.py --exp_name ablation_no_boundary --no_boundary_loss

# 训练参数（可选）：
# --batch_size 8
# --epochs 100
# --lr 6e-5
# --gpu 0
```

### 阶段4：评估和可视化

```bash
# 评估最佳模型
python eval.py --checkpoint outputs/checkpoints/best_model.pth

# 生成可视化对比
python eval.py --checkpoint outputs/checkpoints/best_model.pth --visualize

# 输出：
# results/final_results.csv (6个指标)
# visualizations/ (定性对比图)
```

## 📊 实验设计

### 对比方法（5个Baseline）

| Method | Type | mIoU (预期) | Params | Speed |
|--------|------|-------------|--------|-------|
| SAM ViT-H | Teacher | ~85% | 636M | 5 FPS |
| MobileSAM | 通用蒸馏 | ~83-84% | 5.7M | 40 FPS |
| FastSAM | YOLO-based | ~82-83% | 68M | 30 FPS |
| DeepLabV3+ | 经典CNN | ~82% | 59M | 25 FPS |
| SegFormer-B1 | Baseline | ~82.5% | 13.7M | 42 FPS |
| **Ours** | 遥感特化 | ~84.5% | 15M | 38 FPS |

### 消融实验（4组）

| Components | mIoU | 说明 |
|-----------|------|------|
| Baseline | 82.5% | SegFormer-B1无蒸馏 |
| + Logit蒸馏 | 83.4% | 加入KD Loss |
| + Feature蒸馏 | 84.0% | 加入Feature Loss |
| + Boundary Loss | 84.5% | 加入边缘损失（创新） |

### 评估指标（6个）

1. **mIoU** - 平均交并比（主要精度指标）
2. **F1-Score** - 精确率和召回率的调和平均
3. **OA** - 整体准确率
4. **Params** - 参数量（M）
5. **FLOPs** - 计算量（G）
6. **FPS** - 推理速度

## 🔧 配置说明

所有配置在 `config.py` 中：

```python
# 主要配置
BATCH_SIZE = 8              # 根据显存调整
NUM_EPOCHS = 100            # 可以增加到150-200
LEARNING_RATE = 6e-5        # AdamW学习率
USE_AUGMENTATION = True     # 训练时数据增强

# 损失权重
LOSS_CE_WEIGHT = 1.0        # 交叉熵
LOSS_KD_WEIGHT = 0.5        # KD蒸馏
LOSS_FEAT_WEIGHT = 0.3      # 特征蒸馏
LOSS_BOUNDARY_WEIGHT = 0.2  # 边缘损失（创新）

# 数据增强（训练时）
AUG_HFLIP = True            # 水平翻转
AUG_VFLIP = True            # 垂直翻转
AUG_ROTATE = True           # 90度旋转
AUG_COLOR_JITTER = True     # 颜色抖动
```

## 📈 训练监控

使用TensorBoard查看训练过程：

```bash
tensorboard --logdir outputs/logs
```

监控指标：
- Loss曲线（total, ce, kd, feat, boundary）
- 验证集mIoU
- 学习率变化
- 可视化样本

## 🎨 可视化输出

`visualizations/` 目录包含：
- **comparison_*.png**: 各方法对比（Image | GT | SAM | Ours | Error）
- **ablation_*.png**: 消融实验对比
- **boundary_*.png**: 边缘细节对比（展示创新）

## 📝 论文撰写

### Table 1: 主对比实验

从 `results/baseline_results.csv` 和 `results/final_results.csv` 生成

### Table 2: 消融实验

从不同实验的 `results/` 汇总

### Figure 1: 方法框架图

需要手动绘制（PPT/Visio）

### Figure 2-4: 定性对比

直接使用 `visualizations/` 中的图像

## ⚙️ 硬件要求

- **GPU**: NVIDIA RTX 4060 8GB（最小）
- **内存**: 16GB以上
- **存储**: 60GB以上（数据+模型）
- **训练时间**: 约6-8小时（100 epochs）

## 🐛 常见问题

### Q1: CUDA Out of Memory

```python
# 减小batch size
BATCH_SIZE = 4  # 在config.py中

# 或使用梯度累积
python train.py --batch_size 4 --accumulate_grad 2  # 等效batch=8
```

### Q2: 数据加载慢

```python
# 增加worker数量
NUM_WORKERS = 8  # 在config.py中
```

### Q3: 训练不收敛

```python
# 调整学习率
LEARNING_RATE = 3e-5  # 降低学习率

# 或增加warmup
WARMUP_EPOCHS = 10
```

## 📚 参考文献

主要参考论文：
1. SAM - Kirillov et al. "Segment Anything" (ICCV 2023)
2. MobileSAM - Zhang et al. (arXiv 2023)
3. FastSAM - Zhao et al. (ICCV 2023)
4. SegFormer - Xie et al. "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers" (NeurIPS 2021)
5. Knowledge Distillation - Hinton et al. "Distilling the Knowledge in a Neural Network" (NIPS 2014)

## 📧 联系方式

有问题随时在issue中提问！

## 📄 许可证

本项目仅用于学术研究，不得用于商业用途。

---

**项目状态**：
- [x] 数据预处理完成
- [x] 教师特征提取完成  
- [ ] Baseline评估（进行中）
- [ ] 学生网络训练（待开始）
- [ ] 论文撰写（待开始）

**预计完成时间**：2个月（2025年12月底）

**目标**：SCI低区 / EI会议