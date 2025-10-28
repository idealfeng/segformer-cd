"""
数据加载器 - 【v3.5 最终毕业版】
- ✅ 采纳顶级GPT建议，修复所有已知BUG和潜在风险
- ✅ 修复Dataloader调用语法错误
- ✅ 完善worker_init_fn，确保完全可复现
- ✅ 增加图像读取断言，提升健壮性
- ✅ 增加特征dtype兜底，兼容AMP
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from packaging import version
import random

from config import cfg
from split_dataset import load_split

IGNORE_LABEL = 255

class DistillationDataset(Dataset):
    def __init__(self, split='train', use_augmentation=True):
        self.split = split
        self.use_augmentation = use_augmentation and (split == 'train')
        self.img_ids = load_split(split)
        self.geo_transform, self.color_transform, self.post_transform = self.get_transforms()
        print(f"加载 {split} 集: {len(self.img_ids)} 张图像, 数据增强: {self.use_augmentation}")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        image = self.load_image(img_id)
        label = self.load_label(img_id)
        teacher_feat_b30, teacher_feat_enc = self.load_teacher_features(img_id)

        # 统一将特征转为(H, W, C)以适配Albumentations
        feat_b30_hwc = np.transpose(teacher_feat_b30, (1, 2, 0))
        feat_enc_hwc = np.transpose(teacher_feat_enc, (1, 2, 0))

        if self.use_augmentation:
            res = self.geo_transform(image=image, mask=label)
            image, label, replay = res['image'], res['mask'], res['replay']

            # 回放到特征
            feat_b30_aug_hwc = A.ReplayCompose.replay(replay, image=feat_b30_hwc)['image']
            feat_enc_aug_hwc = A.ReplayCompose.replay(replay, image=feat_enc_hwc)['image']

            image = self.color_transform(image=image)['image']

            # 将增强后的特征转回 (C, H, W)
            teacher_feat_b30 = np.transpose(feat_b30_aug_hwc, (2, 0, 1))
            teacher_feat_enc = np.transpose(feat_enc_aug_hwc, (2, 0, 1))

        post_res = self.post_transform(image=image, mask=label)
        image, label = post_res['image'], post_res['mask']

        label = label.long()
        # ✅ 修复4: 使用 .copy() 和 astype 确保内存连续性和正确的dtype
        teacher_feat_b30 = torch.from_numpy(teacher_feat_b30.copy().astype(np.float32)).float()
        teacher_feat_enc = torch.from_numpy(teacher_feat_enc.copy().astype(np.float32)).float()

        return {
            'image': image, 'label': label,
            'teacher_feat_b30': teacher_feat_b30, 'teacher_feat_enc': teacher_feat_enc,
            'img_id': img_id
        }

    def load_image(self, img_id):
        img_path = cfg.IMAGE_DIR / f"{img_id}.png"
        image = cv2.imread(str(img_path))
        # ✅ 修复2: 增加图像读取断言
        assert image is not None, f"致命错误: 无法读取图像文件 {img_path}"
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def load_label(self, img_id):
        label_path = cfg.LABEL_DIR / f"{img_id}.png"
        label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
        assert label is not None, f"致命错误: 无法读取标签文件 {label_path}"
        if label.ndim == 3:
            label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        label[label == 65] = IGNORE_LABEL
        return label.astype(np.uint8)

    def load_teacher_features(self, img_id):
        feat30_path = cfg.FEATURE_BLOCK30_DIR / f"{img_id}.npz"
        feat_enc_path = cfg.FEATURE_ENCODER_DIR / f"{img_id}.npz"
        # 已经是(C,H,W)格式
        feat30 = np.load(feat30_path)['features'][0]
        feat_enc = np.load(feat_enc_path)['features'][0]
        return feat30, feat_enc

    def get_transforms(self):
        geo_transforms = [
            A.HorizontalFlip(p=cfg.AUG_HFLIP_PROB),
            A.VerticalFlip(p=cfg.AUG_VFLIP_PROB),
            A.RandomRotate90(p=cfg.AUG_ROTATE_PROB),
        ]
        color_transforms = [A.ColorJitter(p=0.5)]
        post_transforms = [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(transpose_mask=True),
        ]

        if self.use_augmentation:
            return A.ReplayCompose(geo_transforms), A.Compose(color_transforms), A.Compose(post_transforms)
        else:
            return A.ReplayCompose([]), A.Compose([]), A.Compose(post_transforms)

def worker_init_fn(worker_id):
    """✅ 修复3: 完善的随机种子设置"""
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def build_dataloader(split='train', batch_size=None, num_workers=None, shuffle=None):
    if batch_size is None: batch_size = cfg.BATCH_SIZE
    if num_workers is None: num_workers = cfg.NUM_WORKERS
    if shuffle is None: shuffle = (split == 'train')

    dataset = DistillationDataset(split=split, use_augmentation=(split == 'train'))

    use_persistent = (num_workers > 0) and (version.parse(torch.__version__) >= version.parse("1.8.0"))
    pin = torch.cuda.is_available() and num_workers > 0

    # ✅ 修复1: 修正Dataloader调用语法
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=(split == 'train'),
        persistent_workers=use_persistent,
        worker_init_fn=worker_init_fn
    )
    return dataloader



if __name__ == '__main__':
    train_loader = build_dataloader('train', batch_size=2)
    batch = next(iter(train_loader))
    print(f"  image: {batch['image'].shape} | {batch['image'].dtype}")
    print(f"  label: {batch['label'].shape} | {batch['label'].dtype}")
    print(f"  teacher_feat_b30: {batch['teacher_feat_b30'].shape} | {batch['teacher_feat_b30'].dtype}")
    print(f"  teacher_feat_enc: {batch['teacher_feat_enc'].shape} | {batch['teacher_feat_enc'].dtype}")
    print(f"  img_id: {batch['img_id']}")
