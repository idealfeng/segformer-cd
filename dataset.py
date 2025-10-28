"""
数据加载器 - 支持动态同步数据增强
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import random
from pathlib import Path
from config import cfg
from split_dataset import load_split


class DistillationDataset(Dataset):
    """
    知识蒸馏数据集

    同时加载：
    - 学生输入：图像
    - 监督信号：标签
    - 教师输出：特征、logits
    """

    def __init__(self, split='train', use_augmentation=True):
        """
        Args:
            split: 'train', 'val', or 'test'
            use_augmentation: 是否使用数据增强（只在train时True）
        """
        self.split = split
        self.use_augmentation = use_augmentation and (split == 'train')

        # 加载文件列表
        self.img_ids = load_split(split)

        print(f"加载 {split} 集: {len(self.img_ids)} 张图像")
        print(f"  数据增强: {self.use_augmentation}")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        """
        返回：
            image: (3, H, W) - 学生输入
            label: (H, W) - 标签
            teacher_feat: (C, h, w) - 教师特征
            teacher_logit: (num_classes, h, w) - 教师logits
            img_id: str - 图像ID（用于可视化）
        """
        img_id = self.img_ids[idx]

        # 1. 加载图像
        image = self.load_image(img_id)  # (H, W, 3)

        # 2. 加载标签
        label = self.load_label(img_id)  # (H, W)

        # 3. 加载教师特征
        teacher_feat = self.load_teacher_feature(img_id)  # (256, 64, 64)

        # 4. 加载教师logits
        teacher_logit = self.load_teacher_logit(img_id)  # (num_classes, 256, 256)

        # 5. 数据增强（同步）
        if self.use_augmentation:
            image, label, teacher_feat, teacher_logit = self.sync_transform(
                image, label, teacher_feat, teacher_logit
            )

        # 6. 转换为tensor
        image = self.to_tensor(image)  # (3, H, W)
        label = torch.from_numpy(label).long()  # (H, W)
        teacher_feat = torch.from_numpy(teacher_feat).float()
        teacher_logit = torch.from_numpy(teacher_logit).float()

        # 7. Normalize图像
        image = self.normalize(image)

        return {
            'image': image,
            'label': label,
            'teacher_feat': teacher_feat,
            'teacher_logit': teacher_logit,
            'img_id': img_id
        }

    def load_image(self, img_id):
        """加载图像"""
        img_path = cfg.IMAGE_DIR / f"{img_id}.png"
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image  # (H, W, 3), uint8

    def load_label(self, img_id):
        """加载标签"""
        label_path = cfg.LABEL_DIR / f"{img_id}.png"
        label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)

        # Potsdam标签映射
        # 原始值 -> 类别索引
        label_mapping = {
            255: 0,  # Impervious surfaces (白色) -> 0
            0: 1,    # Building (蓝色) -> 1
            1: 2,    # Low vegetation (青色) -> 2
            2: 3,    # Tree (绿色) -> 3
            3: 4,    # Car (黄色) -> 4
            4: 5,    # Clutter/background -> 5
        }

        # 如果是RGB格式，先转为索引
        if len(label.shape) == 3:
            label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

        # 创建映射后的标签
        mapped_label = np.zeros_like(label, dtype=np.uint8)
        for old_val, new_val in label_mapping.items():
            mapped_label[label == old_val] = new_val

        return mapped_label  # (H, W), uint8, 值域[0, 5]

    def load_teacher_feature(self, img_id):
        """加载教师特征（Feature 31）"""
        feat_path = cfg.FEATURE_31_DIR / f"{img_id}.npz"
        data = np.load(feat_path)
        features = data['features']  # (1, 256, 64, 64)
        return features[0]  # (256, 64, 64)

    def load_teacher_logit(self, img_id):
        """加载教师logits"""
        logit_path = cfg.LOGITS_DIR / f"{img_id}.npz"
        data = np.load(logit_path)
        logits = data['logits']  # (1, 1, 256, 256)

        # SAM输出是单通道mask，需要转换为多类别logits
        # 这里简化处理，实际可能需要根据Potsdam调整
        logits = logits[0]  # (1, 256, 256)

        # 如果需要多类别，可以复制或使用其他策略
        # 这里假设已经是合适的格式
        return logits  # (num_classes, 256, 256)

    def sync_transform(self, image, label, teacher_feat, teacher_logit):
        """
        同步数据增强

        几何变换需要同时应用到所有数据
        颜色变换只应用到图像
        """
        # 转换为numpy array以便处理
        image = np.array(image)
        label = np.array(label)

        # === 几何增强（同步）===

        # 1. 水平翻转
        if cfg.AUG_HFLIP and random.random() < cfg.AUG_HFLIP_PROB:
            image = np.flip(image, axis=1).copy()
            label = np.flip(label, axis=1).copy()
            teacher_feat = np.flip(teacher_feat, axis=2).copy()  # W维
            teacher_logit = np.flip(teacher_logit, axis=2).copy()

        # 2. 垂直翻转
        if cfg.AUG_VFLIP and random.random() < cfg.AUG_VFLIP_PROB:
            image = np.flip(image, axis=0).copy()
            label = np.flip(label, axis=0).copy()
            teacher_feat = np.flip(teacher_feat, axis=1).copy()  # H维
            teacher_logit = np.flip(teacher_logit, axis=1).copy()

        # 3. 90度旋转
        if cfg.AUG_ROTATE and random.random() < cfg.AUG_ROTATE_PROB:
            k = random.randint(1, 3)  # 1,2,3 对应 90,180,270度
            image = np.rot90(image, k, axes=(0, 1)).copy()
            label = np.rot90(label, k, axes=(0, 1)).copy()
            teacher_feat = np.rot90(teacher_feat, k, axes=(1, 2)).copy()
            teacher_logit = np.rot90(teacher_logit, k, axes=(1, 2)).copy()

        # === 颜色增强（只对图像）===
        if cfg.AUG_COLOR_JITTER and random.random() < 0.5:
            image = self.color_jitter(image)

        return image, label, teacher_feat, teacher_logit

    def color_jitter(self, image):
        """颜色抖动（只对图像）"""
        # 简单的亮度和对比度调整
        if random.random() < 0.5:
            # 亮度
            alpha = 1.0 + random.uniform(-cfg.AUG_BRIGHTNESS, cfg.AUG_BRIGHTNESS)
            image = np.clip(image * alpha, 0, 255).astype(np.uint8)

        if random.random() < 0.5:
            # 对比度
            mean = image.mean()
            alpha = 1.0 + random.uniform(-cfg.AUG_CONTRAST, cfg.AUG_CONTRAST)
            image = np.clip((image - mean) * alpha + mean, 0, 255).astype(np.uint8)

        return image

    def to_tensor(self, image):
        """将numpy图像转为tensor"""
        # (H, W, C) -> (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        return image

    def normalize(self, image):
        """
        归一化图像
        使用ImageNet统计量（SegFormer预训练用的）
        """
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        image = image / 255.0  # [0, 255] -> [0, 1]
        image = (image - mean) / std

        return image


def build_dataloader(split='train', batch_size=None, num_workers=None, shuffle=None):
    """
    构建数据加载器

    Args:
        split: 'train', 'val', or 'test'
        batch_size: batch大小（None则使用配置）
        num_workers: 数据加载线程数（None则使用配置）
        shuffle: 是否打乱（None则train打乱，val/test不打乱）

    Returns:
        DataLoader
    """
    # 使用配置中的默认值
    if batch_size is None:
        batch_size = cfg.BATCH_SIZE
    if num_workers is None:
        num_workers = cfg.NUM_WORKERS
    if shuffle is None:
        shuffle = (split == 'train')

    # 创建数据集
    use_aug = (split == 'train') and cfg.USE_AUGMENTATION
    dataset = DistillationDataset(split=split, use_augmentation=use_aug)

    # 创建dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')  # 训练时丢弃最后不完整的batch
    )

    return dataloader


if __name__ == '__main__':
    """测试数据加载器"""
    print("=" * 60)
    print("测试数据加载器")
    print("=" * 60)

    # 创建训练集dataloader
    train_loader = build_dataloader('train', batch_size=2)

    print(f"\n训练集:")
    print(f"  总batch数: {len(train_loader)}")
    print(f"  Batch size: {cfg.BATCH_SIZE}")

    # 测试加载一个batch
    batch = next(iter(train_loader))

    print(f"\nBatch内容:")
    print(f"  image: {batch['image'].shape} | {batch['image'].dtype}")
    print(f"  label: {batch['label'].shape} | {batch['label'].dtype}")
    print(f"  teacher_feat: {batch['teacher_feat'].shape} | {batch['teacher_feat'].dtype}")
    print(f"  teacher_logit: {batch['teacher_logit'].shape} | {batch['teacher_logit'].dtype}")
    print(f"  img_id: {batch['img_id']}")

    # 检查数值范围
    print(f"\n数值范围:")
    print(f"  image: [{batch['image'].min():.2f}, {batch['image'].max():.2f}]")
    print(f"  label: [{batch['label'].min()}, {batch['label'].max()}]")

    print("\n" + "=" * 60)
    print("✓ 数据加载器测试通过")
    print("=" * 60)