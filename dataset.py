"""
LEVIR-CD 变化检测数据集加载器
"""
import os
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

from config import cfg

EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]


def _find_existing(folder: Path, stem: str) -> Path:
    for ext in EXTS:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing file for stem={stem} under {folder}")


def _read_rgb_any(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    # 16-bit tif -> 8-bit
    if arr.dtype == np.uint16:
        arr = (arr >> 8).astype(np.uint8)

    # 保证3通道
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[2] > 3:
        arr = arr[:, :, :3]
    return arr


def _read_mask_any(path: Path) -> np.ndarray:
    m = np.array(Image.open(path))
    # 可能是 0/255 或 0/1 都统一成 0/1
    if m.dtype == np.uint16:
        m = (m >> 8).astype(np.uint8)
    m = (m > 0).astype(np.uint8)
    return m


class LEVIRCDDataset(Dataset):
    """LEVIR-CD变化检测数据集"""

    def __init__(
        self,
        root_dir: Path,
        split: str = 'train',
        transform: Optional[A.Compose] = None,
        crop_size: int = 256
    ):
        """
        Args:
            root_dir: 数据集根目录 (包含train/val/test子目录)
            split: 数据集划分 ('train', 'val', 'test')
            transform: albumentations变换
            crop_size: 裁剪尺寸
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.crop_size = crop_size

        # 设置路径
        split_dir = self.root_dir / split
        self.img_a_dir = split_dir / 'A'
        self.img_b_dir = split_dir / 'B'
        self.label_dir = split_dir / 'OUT'

        # 获取所有图像文件名
        self.img_names = self._get_image_names()

        print(f"[{split.upper()}] Loaded {len(self.img_names)} image pairs")


    def _get_image_names(self) -> List[str]:
        if not self.img_a_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.img_a_dir}")

        stems = []
        for ext in EXTS:
            stems += [f.stem for f in self.img_a_dir.glob(f"*{ext}")]
        stems = sorted(list(set(stems)))
        return stems

    def __len__(self) -> int:
        return len(self.img_names)

    def __getitem__(self, idx: int) -> dict:
        img_name = self.img_names[idx]

        img_a_path = _find_existing(self.img_a_dir, img_name)
        img_b_path = _find_existing(self.img_b_dir, img_name)
        label_path = _find_existing(self.label_dir, img_name)

        img_a = _read_rgb_any(img_a_path)
        img_b = _read_rgb_any(img_b_path)
        label = _read_mask_any(label_path)

        # 应用变换
        if self.transform:
            # 将两张图像拼接以保证同步变换
            # albumentations的additional_targets功能
            transformed = self.transform(
                image=img_a,
                image2=img_b,
                mask=label
            )
            img_a = transformed['image']
            img_b = transformed['image2']
            label = transformed['mask']
        else:
            # 默认转换为tensor
            img_a = torch.from_numpy(img_a).permute(2, 0, 1).float() / 255.0
            img_b = torch.from_numpy(img_b).permute(2, 0, 1).float() / 255.0
            label = torch.from_numpy(label).long()

        return {
            'img_a': img_a,
            'img_b': img_b,
            'label': label,
            'name': img_name
        }


def get_train_transforms(crop_size: int = 256) -> A.Compose:
    """训练集数据增强"""
    transforms_list = []

    # 随机裁剪（从1024裁剪到256）
    if cfg.RANDOM_CROP:
        transforms_list.append(
            A.RandomCrop(height=crop_size, width=crop_size, p=1.0)
        )

    # 几何变换
    if cfg.AUG_HFLIP:
        transforms_list.append(A.HorizontalFlip(p=cfg.AUG_HFLIP_PROB))

    if cfg.AUG_VFLIP:
        transforms_list.append(A.VerticalFlip(p=cfg.AUG_VFLIP_PROB))

    if cfg.AUG_ROTATE:
        transforms_list.append(A.RandomRotate90(p=cfg.AUG_ROTATE_PROB))

    # 颜色变换（对两张图同时应用）
    if cfg.AUG_COLOR_JITTER:
        transforms_list.append(
            A.ColorJitter(
                brightness=cfg.AUG_BRIGHTNESS,
                contrast=cfg.AUG_CONTRAST,
                saturation=cfg.AUG_SATURATION,
                hue=cfg.AUG_HUE,
                p=0.5
            )
        )

    # 高斯噪声
    if cfg.AUG_GAUSSIAN_NOISE:
        try:
            # 新版本albumentations
            transforms_list.append(
                A.GaussNoise(std_range=(0.01, 0.05), p=0.3)
            )
        except TypeError:
            # 旧版本albumentations
            transforms_list.append(
                A.GaussNoise(var_limit=cfg.AUG_NOISE_VAR_LIMIT, p=0.3)
            )

    # 归一化和转换为tensor
    transforms_list.extend([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

    return A.Compose(
        transforms_list,
        additional_targets={'image2': 'image'}  # 确保image2应用相同变换
    )


def get_val_transforms(crop_size: int = 256) -> A.Compose:
    """验证/测试集数据变换（无增强，只裁剪中心）"""
    transforms_list = []

    # 中心裁剪
    if crop_size < cfg.ORIGINAL_SIZE:
        transforms_list.append(
            A.CenterCrop(height=crop_size, width=crop_size)
        )

    # 归一化和转换为tensor
    transforms_list.extend([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

    return A.Compose(
        transforms_list,
        additional_targets={'image2': 'image'}
    )


def get_test_transforms_full() -> A.Compose:
    """测试集变换（完整1024x1024，用于最终评估）"""
    return A.Compose(
        [
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ],
        additional_targets={'image2': 'image'}
    )


def worker_init_fn(worker_id):
    """确保多进程数据加载的随机性"""
    seed = int(np.random.get_state()[1][0]) + worker_id  # 转换为Python int
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def create_dataloaders(
    batch_size: int = 8,
    num_workers: int = 4,
    crop_size: int = 256
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建训练、验证、测试数据加载器"""

    # 创建数据集
    train_dataset = LEVIRCDDataset(
        root_dir=cfg.DATA_ROOT,
        split='train',
        transform=get_train_transforms(crop_size),
        crop_size=crop_size
    )

    val_dataset = LEVIRCDDataset(
        root_dir=cfg.DATA_ROOT,
        split='val',
        transform=get_val_transforms(crop_size),
        crop_size=crop_size
    )

    test_dataset = LEVIRCDDataset(
        root_dir=cfg.DATA_ROOT,
        split='test',
        transform=get_val_transforms(crop_size),
        crop_size=crop_size
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init_fn
    )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # 测试数据加载
    print("Testing LEVIR-CD Dataset Loading...")
    cfg.display()

    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            batch_size=4,
            num_workers=0,  # 测试时用0
            crop_size=256
        )

        # 测试一个batch
        batch = next(iter(train_loader))
        print(f"\nBatch contents:")
        print(f"  img_a shape: {batch['img_a'].shape}")
        print(f"  img_b shape: {batch['img_b'].shape}")
        print(f"  label shape: {batch['label'].shape}")
        print(f"  label unique values: {torch.unique(batch['label'])}")
        print(f"  names: {batch['name']}")

        print(f"\nDataset sizes:")
        print(f"  Train: {len(train_loader.dataset)} samples")
        print(f"  Val: {len(val_loader.dataset)} samples")
        print(f"  Test: {len(test_loader.dataset)} samples")

        print("\nDataset loading test passed!")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure LEVIR-CD dataset is placed in data/LEVIR-CD/")
