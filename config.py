"""
配置文件 - SegFormer变化检测
"""
import os
from pathlib import Path
import torch


class Config:
    """变化检测训练和评估配置"""

    # ==================== 路径配置 ====================
    PROJECT_ROOT = Path(__file__).parent.resolve()

    # 数据路径 - LEVIR-CD数据集
    DATA_ROOT = PROJECT_ROOT / "data" / "LEVIR-CD"
    TRAIN_DIR = DATA_ROOT / "train"
    VAL_DIR = DATA_ROOT / "val"
    TEST_DIR = DATA_ROOT / "test"

    # 输出路径
    OUTPUT_ROOT = PROJECT_ROOT / "outputs"
    CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
    LOG_DIR = OUTPUT_ROOT / "logs"
    VIS_DIR = OUTPUT_ROOT / "visualizations"
    RESULTS_DIR = OUTPUT_ROOT / "results"

    # 创建必要的目录
    for dir_path in [OUTPUT_ROOT, CHECKPOINT_DIR, LOG_DIR, VIS_DIR, RESULTS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # ==================== 数据配置 ====================
    # 原始图像尺寸
    ORIGINAL_SIZE = 1024

    # 训练图像尺寸（裁剪后）
    IMAGE_SIZE = 256

    # 类别数（二值变化检测：1通道输出，用Sigmoid）
    NUM_CLASSES = 1

    # Ignore标签
    IGNORE_INDEX = 255

    # 类别名称
    CLASS_NAMES = ['unchanged', 'changed']

    # 类别颜色（用于可视化）
    CLASS_COLORS = [
        [0, 0, 0],       # 黑色 - 无变化
        [255, 255, 255]  # 白色 - 有变化
    ]

    # ==================== 模型配置 ====================
    # SegFormer变体
    MODEL_TYPE = "segformer_b1"  # b0, b1, b2, b3, b4, b5
    PRETRAINED = True  # 使用ImageNet预训练

    # 特征融合方式
    FUSION_TYPE = "diff"  # "diff", "concat", "attention"

    # 是否使用深度监督
    DEEP_SUPERVISION = True
    DEEP_SUPERVISION_WEIGHTS = [1.0, 0.4, 0.2, 0.1]  # 各尺度权重

    # 条带式感受野增强（实验结论：失败）
    # 实验结果：F1=90.11% vs baseline 90.42%（下降0.31%）
    # 结论：在diff feature上扩大感受野是错误方向
    USE_STRIP_ENHANCE = False        # 已禁用
    STRIP_SIZE = 11
    STRIP_RESIDUAL_WEIGHT = 0.3

    # 轻量级ASPP（Atrous Spatial Pyramid Pooling）
    # 设计：在decoder输出的fused feature上增强多尺度上下文
    # 位置：decoder → ASPP → classifier
    USE_ASPP = True                  # 启用ASPP
    ASPP_CHANNELS = 256              # ASPP输出通道数
    ASPP_DILATIONS = [1, 2, 4]       # 膨胀率（1×1, rate=2, rate=4）

    # ==================== 训练配置 ====================
    # 基础训练参数
    BATCH_SIZE = 8  # 4060 8G显存，256x256图像
    NUM_EPOCHS = 300  # 实验发现300 epochs优于200 epochs
    NUM_WORKERS = 4

    # 梯度累积（等效batch_size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS）
    GRADIENT_ACCUMULATION_STEPS = 2  # 等效batch_size=16

    # 优化器参数
    OPTIMIZER = "adamw"
    LEARNING_RATE = 5e-4  # 最佳学习率（调参实验得出）
    WEIGHT_DECAY = 0.01
    BETAS = (0.9, 0.999)

    # 学习率调度
    LR_SCHEDULER = "cosine"  # "polynomial", "cosine", "step"
    LR_POWER = 0.9
    WARMUP_EPOCHS = 10
    WARMUP_LR = 1e-6
    MIN_LR = 1e-7

    # 损失函数配置
    LOSS_TYPE = "combined"  # "bce", "dice", "focal", "combined"

    # 损失权重
    LOSS_WEIGHT_BCE = 0.5
    LOSS_WEIGHT_DICE = 0.5
    LOSS_WEIGHT_BOUNDARY = 0.1

    # Focal Loss参数（如果使用）
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0

    # 类别权重（处理类别不平衡）
    USE_CLASS_WEIGHTS = True
    # LEVIR-CD中变化像素约占5-10%，设置权重平衡
    CLASS_WEIGHTS = [1.0, 4.0]  # [unchanged, changed] - 最佳权重（调参实验得出）

    # 混合精度训练
    USE_AMP = True

    # ==================== 数据增强配置 ====================
    USE_AUGMENTATION = True

    # 随机裁剪（从1024裁剪到256）
    RANDOM_CROP = True
    CROP_SIZE = 256

    # 几何增强
    AUG_HFLIP = True
    AUG_VFLIP = True
    AUG_ROTATE = True
    AUG_HFLIP_PROB = 0.5
    AUG_VFLIP_PROB = 0.5
    AUG_ROTATE_PROB = 0.5

    # 颜色增强（对两张图像同时应用）
    AUG_COLOR_JITTER = True
    AUG_BRIGHTNESS = 0.2
    AUG_CONTRAST = 0.2
    AUG_SATURATION = 0.2
    AUG_HUE = 0.1

    # 高斯噪声
    AUG_GAUSSIAN_NOISE = True
    AUG_NOISE_VAR_LIMIT = (10.0, 50.0)

    # ==================== 评估配置 ====================
    # 保存频率
    SAVE_FREQ = 10  # 每10个epoch保存一次
    VAL_FREQ = 1    # 每个epoch验证一次

    # 早停
    EARLY_STOPPING = True
    PATIENCE = 30  # 30个epoch不提升则停止

    # 评估指标
    EVAL_METRICS = ['F1', 'IoU', 'OA', 'Precision', 'Recall', 'Kappa']

    # 最佳模型选择指标
    BEST_METRIC = 'F1'  # 基于F1分数选择最佳模型

    # ==================== 可视化配置 ====================
    VIS_NUM_SAMPLES = 10
    VIS_SAVE_INTERVAL = 20

    # ==================== 设备配置 ====================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42

    # ==================== 调试配置 ====================
    DEBUG = False
    DEBUG_SAMPLES = 50
    USE_TENSORBOARD = True

    # ==================== 测试时增强 ====================
    TTA = False  # Test Time Augmentation

    @classmethod
    def display(cls):
        """打印配置信息"""
        print("=" * 60)
        print("SegFormer Change Detection Configuration")
        print("=" * 60)
        print(f"Project Root: {cls.PROJECT_ROOT}")
        print(f"Data Root: {cls.DATA_ROOT}")
        print(f"Model: {cls.MODEL_TYPE}")
        print(f"Image Size: {cls.IMAGE_SIZE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Effective Batch Size: {cls.BATCH_SIZE * cls.GRADIENT_ACCUMULATION_STEPS}")
        print(f"Epochs: {cls.NUM_EPOCHS}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Device: {cls.DEVICE}")
        print(f"Deep Supervision: {cls.DEEP_SUPERVISION}")
        print(f"Fusion Type: {cls.FUSION_TYPE}")
        print("=" * 60)


# 创建全局配置实例
cfg = Config()

if __name__ == '__main__':
    cfg.display()
