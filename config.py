"""
配置文件 - 所有超参数和路径配置
"""
import os
from pathlib import Path


class Config:
    """训练和评估配置"""

    # ==================== 路径配置 ====================
    # 项目根目录（修改为你的实际路径）
    PROJECT_ROOT = Path("D:/Paper/project")

    # 数据路径
    DATA_ROOT = PROJECT_ROOT / "Potsdam_processed"
    IMAGE_DIR = DATA_ROOT / "images"
    LABEL_DIR = DATA_ROOT / "labels"

    # 教师网络输出路径
    TEACHER_ROOT = PROJECT_ROOT / "teacher_outputs"
    FEATURE_30_DIR = TEACHER_ROOT / "features_30"
    FEATURE_31_DIR = TEACHER_ROOT / "features_31"
    LOGITS_DIR = TEACHER_ROOT / "logits"
    MASKS_DIR = TEACHER_ROOT / "masks"

    # 输出路径
    OUTPUT_ROOT = PROJECT_ROOT / "outputs"
    CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
    LOG_DIR = OUTPUT_ROOT / "logs"
    VIS_DIR = OUTPUT_ROOT / "visualizations"
    RESULTS_DIR = OUTPUT_ROOT / "results"

    # 数据集划分文件
    SPLIT_DIR = PROJECT_ROOT / "splits"
    TRAIN_LIST = SPLIT_DIR / "train.txt"
    VAL_LIST = SPLIT_DIR / "val.txt"
    TEST_LIST = SPLIT_DIR / "test.txt"

    # 创建必要的目录
    for dir_path in [OUTPUT_ROOT, CHECKPOINT_DIR, LOG_DIR,
                     VIS_DIR, RESULTS_DIR, SPLIT_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # ==================== 数据配置 ====================
    # 数据集划分比例
    TRAIN_RATIO = 0.70  # 1680张
    VAL_RATIO = 0.15  # 360张
    TEST_RATIO = 0.15  # 360张

    # 图像尺寸
    IMAGE_SIZE = 1024

    # 类别数（根据Potsdam调整）
    NUM_CLASSES = 6  # Potsdam: 建筑、道路、树木、草地、车辆、背景

    # 类别名称
    CLASS_NAMES = [
        'Impervious surfaces',
        'Building',
        'Low vegetation',
        'Tree',
        'Car',
        'Background'
    ]

    # 类别颜色（用于可视化）
    CLASS_COLORS = [
        [255, 255, 255],  # 白色 - Impervious
        [0, 0, 255],  # 蓝色 - Building
        [0, 255, 255],  # 青色 - Low vegetation
        [0, 255, 0],  # 绿色 - Tree
        [255, 255, 0],  # 黄色 - Car
        [255, 0, 0]  # 红色 - Background
    ]

    # ==================== 模型配置 ====================
    # 学生网络
    STUDENT_MODEL = "segformer_b1"  # 或 b0, b2, b3
    STUDENT_PRETRAINED = True  # 使用ImageNet预训练

    # 教师网络特征维度
    TEACHER_FEAT_30_DIM = 1280  # Block 30
    TEACHER_FEAT_31_DIM = 256  # Block 31

    # ==================== 训练配置 ====================
    # 基础训练参数
    BATCH_SIZE = 8  # 4060 8G显存可以用8
    NUM_EPOCHS = 100  # 可以调整到150-200
    NUM_WORKERS = 4  # 数据加载线程数

    # 优化器参数
    OPTIMIZER = "adamw"
    LEARNING_RATE = 6e-5
    WEIGHT_DECAY = 0.01
    BETAS = (0.9, 0.999)

    # 学习率调度
    LR_SCHEDULER = "polynomial"  # 或 "cosine"
    LR_POWER = 0.9
    WARMUP_EPOCHS = 5
    WARMUP_LR = 1e-6

    # 损失函数权重
    LOSS_CE_WEIGHT = 1.0  # 交叉熵损失
    LOSS_KD_WEIGHT = 0.5  # KD蒸馏损失
    LOSS_FEAT_WEIGHT = 0.3  # 特征蒸馏损失
    LOSS_BOUNDARY_WEIGHT = 0.2  # 边缘损失（你的创新）

    # KD蒸馏温度
    KD_TEMPERATURE = 4.0

    # ==================== 数据增强配置 ====================
    # 训练时增强（只在训练集使用）
    USE_AUGMENTATION = True

    # 几何增强（需要同步）
    AUG_HFLIP = True  # 水平翻转
    AUG_VFLIP = True  # 垂直翻转
    AUG_ROTATE = True  # 90度旋转
    AUG_HFLIP_PROB = 0.5
    AUG_VFLIP_PROB = 0.5
    AUG_ROTATE_PROB = 0.25

    # 颜色增强（只对图像）
    AUG_COLOR_JITTER = True
    AUG_BRIGHTNESS = 0.2
    AUG_CONTRAST = 0.2
    AUG_SATURATION = 0.2
    AUG_HUE = 0.1

    # ==================== 评估配置 ====================
    # 保存频率
    SAVE_FREQ = 5  # 每5个epoch保存一次
    VAL_FREQ = 1  # 每1个epoch验证一次

    # 早停
    EARLY_STOPPING = True
    PATIENCE = 20  # 20个epoch不提升则停止

    # 评估指标
    EVAL_METRICS = ['mIoU', 'F1', 'OA', 'Params', 'FLOPs', 'FPS']

    # ==================== 可视化配置 ====================
    VIS_NUM_SAMPLES = 10  # 可视化样本数量
    VIS_SAVE_INTERVAL = 10  # 每10个epoch保存可视化

    # ==================== Baseline模型配置 ====================
    # 用于对比的baseline模型
    BASELINE_MODELS = {
        'SAM_ViT_H': {
            'type': 'sam',
            'checkpoint': str(PROJECT_ROOT / "数据预处理与教师网络输出" / "sam_vit_h_4b8939.pth"),
            'use_masks': True  # 直接用已提取的masks
        },
        'MobileSAM': {
            'type': 'mobile_sam',
            'checkpoint': None,  # 需要下载
            'repo': 'https://github.com/ChaoningZhang/MobileSAM'
        },
        'FastSAM': {
            'type': 'fast_sam',
            'checkpoint': None,  # 需要下载
            'repo': 'https://github.com/CASIA-IVA-Lab/FastSAM'
        },
        'DeepLabV3+': {
            'type': 'deeplabv3plus',
            'backbone': 'resnet101',
            'pretrained': True
        },
        'SegFormer_B1': {
            'type': 'segformer',
            'variant': 'b1',
            'pretrained': True
        }
    }

    # ==================== 设备配置 ====================
    DEVICE = "cuda"  # 或 "cpu"
    SEED = 42  # 随机种子

    # ==================== 调试配置 ====================
    DEBUG = False
    DEBUG_SAMPLES = 100  # debug模式下只用100个样本

    @classmethod
    def display(cls):
        """打印配置信息"""
        print("=" * 60)
        print("Configuration Settings")
        print("=" * 60)
        print(f"Project Root: {cls.PROJECT_ROOT}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Epochs: {cls.NUM_EPOCHS}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Device: {cls.DEVICE}")
        print(f"Image Size: {cls.IMAGE_SIZE}")
        print(f"Num Classes: {cls.NUM_CLASSES}")
        print("=" * 60)


# 创建全局配置实例
cfg = Config()

if __name__ == '__main__':
    # 测试配置
    cfg.display()

    # 检查路径是否存在
    print("\n检查路径...")
    print(f"图像目录: {cfg.IMAGE_DIR.exists()}")
    print(f"标签目录: {cfg.LABEL_DIR.exists()}")
    print(f"教师特征: {cfg.FEATURE_31_DIR.exists()}")