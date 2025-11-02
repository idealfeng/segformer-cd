"""
配置文件 - 所有超参数和路径配置
"""
import os
from pathlib import Path

import torch  # <--- 这才是正确的写法！


class Config:
    """训练和评估配置"""

    # ==================== 路径配置 ====================
    # 以当前文件所在目录为项目根目录（修改为你的实际路径）
    PROJECT_ROOT = Path(__file__).parent.resolve()

    # 定义一个所有数据的“根”，指向 "data" 文件夹
    DATA_PARENT_DIR = PROJECT_ROOT / "data"

    # 数据路径
    DATA_ROOT = DATA_PARENT_DIR / "Potsdam_processed"
    IMAGE_DIR = DATA_ROOT / "images"
    LABEL_DIR = DATA_ROOT / "labels"

    # 教师网络输出路径
    TEACHER_ROOT = DATA_PARENT_DIR / "teacher_outputs"
    FEATURE_BLOCK30_DIR = TEACHER_ROOT / "features_block30"
    FEATURE_ENCODER_DIR = TEACHER_ROOT / "features_encoder"


    # 输出路径
    OUTPUT_ROOT = PROJECT_ROOT / "outputs"
    CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
    LOG_DIR = OUTPUT_ROOT / "logs"
    VIS_DIR = OUTPUT_ROOT / "visualizations"
    RESULTS_DIR = OUTPUT_ROOT / "results"

    # 数据集划分文件
    SPLIT_DIR = DATA_PARENT_DIR / "splits"
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
    NUM_CLASSES = 1  # 二分类单通道输出
    NUM_CLASSES_SEMANTIC = 2  # 语义上是2类（背景/前景）
    NUM_CLASSES_DATA=6 # 原始数据集

    # Ignore标签
    IGNORE_LABEL = 255  # 用于标记无效区域

    # 类别名称
    CLASS_NAMES_DATA = [
        'Impervious surfaces',  # Class 0
        'Building',  # Class 1
        'Low vegetation',  # Class 2
        'Tree',  # Class 3
        'Car',  # Class 4
        'Clutter/Background'  # Class 5
    ]

    # 类别颜色（用于可视化）
    CLASS_COLORS_DATA = [
        [255, 255, 255],  # 白色 - Impervious
        [0, 0, 255],  # 蓝色 - Building
        [0, 255, 255],  # 青色 - Low vegetation
        [0, 255, 0],  # 绿色 - Tree
        [255, 255, 0],  # 黄色 - Car
        [255, 0, 0]  # 红色 - Background
    ]

    CLASS_NAMES = [
        'Background',  # Class 0
        'Foreground'  # Class 1（所有非背景区域）
    ]

    CLASS_COLORS = [
        [0, 0, 0],  # 黑色 - 背景
        [255, 255, 255]  # 白色 - 前景
    ]

    # ==================== 模型配置 ====================
    # 学生网络
    STUDENT_MODEL = "segformer_b1"  # 或 b0, b2, b3
    STUDENT_PRETRAINED = True  # 使用ImageNet预训练

    # 教师网络特征维度
    TEACHER_FEAT_BLOCK30_DIM = 1280  # Block 30
    TEACHER_FEAT_ENCODER_DIM = 256  # Block 31

    # ==================== 训练配置 ====================
    # 基础训练参数
    BATCH_SIZE = 4# 4060 8G显存可以用8
    NUM_EPOCHS = 100  # 可以调整到150-200
    NUM_WORKERS = 4  # 数据加载线程数
    USE_AMP = True

    # 可选：添加梯度累积（新增）
    GRADIENT_ACCUMULATION_STEPS = 2# 等效batch_size=4

    # 优化器参数
    OPTIMIZER = "adamw"
    LEARNING_RATE = 6e-5
    WEIGHT_DECAY = 0.01
    BETAS = (0.9, 0.999)

    # 学习率调度
    LR_SCHEDULER = "polynomial"  # 或 "cosine"，此处是多项式衰减
    LR_POWER = 0.9
    WARMUP_EPOCHS = 5       # 前5个epoch较小学习率
    WARMUP_LR = 1e-6        # 初始学习率

    # 损失函数权重
    # 阶段一：二分类分割蒸馏
    LOSS_SEG_WEIGHT = 1.0  # 分割损失 (例如 Dice/BCE Loss)
    LOSS_FEAT_B30_WEIGHT = 0.5  # Block 30 特征蒸馏损失
    LOSS_FEAT_ENC_WEIGHT = 0.5  # Encoder 特征蒸馏损失

    # 训练配置新增
    USE_AMP = True  # 使用 Automatic Mixed Precision 加速并节省 VRAM
    USE_GRADIENT_CHECKPOINTING =False  # 默认我们先关掉它
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
            'checkpoint': str(PROJECT_ROOT / "pretrained_weights/sam_vit_h_4b8939.pth"), # 假设路径
            'use_masks': True  # 直接用已提取的masks
        },
        'MobileSAM': {
            'type': 'mobile_sam',
            'checkpoint': str(PROJECT_ROOT / "checkpoints/mobile_sam.pt"),  # 需下载
            'repo': 'https://github.com/ChaoningZhang/MobileSAM'
        },
        'FastSAM': {
            'type': 'fast_sam',
            'checkpoint': str(PROJECT_ROOT / "checkpoints/FastSAM.pt"),  # 需下载
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
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    SEED = 42  # 随机种子

    # ==================== 调试配置 ====================
    DEBUG = False
    DEBUG_SAMPLES = 100  # debug模式下只用100个样本
    USE_TENSORBOARD = True

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
    cfg.setup_env()  # 在主脚本开头调用
    cfg.display()

