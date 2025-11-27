# unified_dataset.py

import os
from pathlib import Path
from typing import Optional, List
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class UnifiedChangeDetectionDataset(Dataset):
    """统一的变化检测数据集 - 支持LEVIR-CD, S2Looking, WHUCD"""

    DATASET_CONFIG = {
        'levir': {
            'img_a': 'A',
            'img_b': 'B',
            'label': 'label',
            'img_ext': '.png',
            'label_ext': '.png',
            'name_pattern': 'same'  # A和B文件名相同
        },
        's2looking': {
            'img_a': 'Image1',
            'img_b': 'Image2',
            'label': 'label',
            'img_ext': '.png',
            'label_ext': '.png',
            'name_pattern': 'same'
        },
        'whucd': {
            'img_a': 'splited_images',  # 特殊：需要到2012子目录
            'img_b': 'splited_images',  # 特殊：需要到2016子目录
            'label': 'change_label',
            'img_ext': '.tif',
            'label_ext': '.png',
            'name_pattern': 'same',
            'year_a': '2012',
            'year_b': '2016'
        }
    }

    def __init__(
            self,
            root_dir: Path,
            dataset_name: str,
            split: str = 'train',
            transform: Optional[A.Compose] = None,
            crop_size: int = 256
    ):
        self.root_dir = Path(root_dir)
        self.dataset_name = dataset_name.lower()
        self.split = split
        self.transform = transform
        self.crop_size = crop_size

        # 获取数据集配置
        if self.dataset_name not in self.DATASET_CONFIG:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        self.config = self.DATASET_CONFIG[self.dataset_name]

        # 设置路径
        self._setup_paths()

        # 获取文件列表
        self.img_names = self._get_image_names()

        print(f"[{dataset_name.upper()}][{split.upper()}] Loaded {len(self.img_names)} pairs")

    def _setup_paths(self):
        """设置数据集路径"""
        split_dir = self.root_dir / self.split

        if self.dataset_name == 'whucd':
            # WHUCD特殊处理
            self.img_a_dir = split_dir / self.config['img_a'] / self.config['year_a']
            self.img_b_dir = split_dir / self.config['img_b'] / self.config['year_b']
        else:
            self.img_a_dir = split_dir / self.config['img_a']
            self.img_b_dir = split_dir / self.config['img_b']

        self.label_dir = split_dir / self.config['label']

    def _get_image_names(self) -> List[str]:
        """获取图像文件名列表"""
        if not self.img_a_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.img_a_dir}")

        # 获取文件扩展名
        img_ext = self.config['img_ext']

        # 获取所有图像文件
        img_names = sorted([
            f.stem for f in self.img_a_dir.glob(f'*{img_ext}')
        ])

        if len(img_names) == 0:
            raise FileNotFoundError(f"No images found in {self.img_a_dir}")

        return img_names

    def __len__(self) -> int:
        return len(self.img_names)

    def __getitem__(self, idx: int):
        img_name = self.img_names[idx]

        # 构建文件路径
        img_a_path = self.img_a_dir / f"{img_name}{self.config['img_ext']}"
        img_b_path = self.img_b_dir / f"{img_name}{self.config['img_ext']}"
        label_path = self.label_dir / f"{img_name}{self.config['label_ext']}"

        # 加载图像
        try:
            img_a = np.array(Image.open(img_a_path).convert('RGB'))
            img_b = np.array(Image.open(img_b_path).convert('RGB'))
            label = np.array(Image.open(label_path).convert('L'))
        except Exception as e:
            print(f"Error loading: {img_a_path}")
            raise e

        # 二值化标签
        label = (label > 127).astype(np.uint8)

        # 应用变换
        if self.transform:
            transformed = self.transform(
                image=img_a,
                image2=img_b,
                mask=label
            )
            img_a = transformed['image']
            img_b = transformed['image2']
            label = transformed['mask']
        else:
            img_a = torch.from_numpy(img_a).permute(2, 0, 1).float() / 255.0
            img_b = torch.from_numpy(img_b).permute(2, 0, 1).float() / 255.0
            label = torch.from_numpy(label).long()

        return {
            'img_a': img_a,
            'img_b': img_b,
            'label': label,
            'name': img_name
        }


def create_dataloaders_unified(
        dataset_name: str,
        data_root: Path,
        batch_size: int = 8,
        num_workers: int = 4,
        crop_size: int = 256
):
    """创建统一格式的数据加载器"""
    from dataset import get_train_transforms, get_val_transforms, worker_init_fn
    from torch.utils.data import DataLoader

    # 创建数据集
    train_dataset = UnifiedChangeDetectionDataset(
        root_dir=data_root,
        dataset_name=dataset_name,
        split='train',
        transform=get_train_transforms(crop_size)
    )

    val_dataset = UnifiedChangeDetectionDataset(
        root_dir=data_root,
        dataset_name=dataset_name,
        split='val',
        transform=get_val_transforms(crop_size)
    )

    test_dataset = UnifiedChangeDetectionDataset(
        root_dir=data_root,
        dataset_name=dataset_name,
        split='test',
        transform=get_val_transforms(crop_size)
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
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader