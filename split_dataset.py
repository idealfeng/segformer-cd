"""
数据集划分脚本
将2400张图像按7:1.5:1.5划分为训练集、验证集、测试集
"""
import os
import random
from pathlib import Path
from config import cfg


def split_dataset(seed=42):
    """
    划分数据集为train/val/test

    Args:
        seed: 随机种子，保证可复现
    """
    print("=" * 60)
    print("数据集划分")
    print("=" * 60)

    # 设置随机种子
    random.seed(seed)

    # 获取所有图像文件
    image_dir = cfg.IMAGE_DIR
    all_images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

    print(f"总图像数: {len(all_images)}")

    # 打乱顺序
    random.shuffle(all_images)

    # 计算划分点
    total = len(all_images)
    train_end = int(total * cfg.TRAIN_RATIO)
    val_end = train_end + int(total * cfg.VAL_RATIO)

    # 划分数据集
    train_list = all_images[:train_end]
    val_list = all_images[train_end:val_end]
    test_list = all_images[val_end:]

    print(f"\n划分结果:")
    print(f"  训练集: {len(train_list)} ({len(train_list) / total * 100:.1f}%)")
    print(f"  验证集: {len(val_list)} ({len(val_list) / total * 100:.1f}%)")
    print(f"  测试集: {len(test_list)} ({len(test_list) / total * 100:.1f}%)")

    # 保存到文件
    def save_list(file_list, save_path):
        """保存文件列表"""
        with open(save_path, 'w') as f:
            for filename in file_list:
                # 只保存文件名（不含扩展名）
                img_id = filename.replace('.png', '')
                f.write(f"{img_id}\n")
        print(f"  已保存: {save_path}")

    save_list(train_list, cfg.TRAIN_LIST)
    save_list(val_list, cfg.VAL_LIST)
    save_list(test_list, cfg.TEST_LIST)

    print("\n" + "=" * 60)
    print("数据集划分完成！")
    print("=" * 60)

    # 验证：检查是否有重叠
    train_set = set(train_list)
    val_set = set(val_list)
    test_set = set(test_list)

    assert len(train_set & val_set) == 0, "训练集和验证集有重叠！"
    assert len(train_set & test_set) == 0, "训练集和测试集有重叠！"
    assert len(val_set & test_set) == 0, "验证集和测试集有重叠！"

    print("✓ 验证通过：数据集无重叠")

    return train_list, val_list, test_list


def load_split(split='train'):
    """
    加载数据集划分

    Args:
        split: 'train', 'val', or 'test'

    Returns:
        list: 图像ID列表
    """
    split_files = {
        'train': cfg.TRAIN_LIST,
        'val': cfg.VAL_LIST,
        'test': cfg.TEST_LIST
    }

    if split not in split_files:
        raise ValueError(f"Invalid split: {split}")

    split_file = split_files[split]

    if not split_file.exists():
        print(f"警告: {split_file} 不存在，正在创建数据集划分...")
        split_dataset()

    with open(split_file, 'r') as f:
        img_ids = [line.strip() for line in f.readlines()]

    return img_ids


def check_data_integrity():
    """
    检查数据完整性：确保所有图像都有对应的标签和教师特征
    """
    print("\n" + "=" * 60)
    print("数据完整性检查")
    print("=" * 60)

    splits = ['train', 'val', 'test']
    all_valid = True

    for split in splits:
        print(f"\n检查 {split} 集...")
        img_ids = load_split(split)

        missing_labels = []
        missing_features = []
        missing_logits = []

        for img_id in img_ids:
            # 检查标签
            label_path = cfg.LABEL_DIR / f"{img_id}.png"
            if not label_path.exists():
                missing_labels.append(img_id)

            # 检查特征
            feat_path = cfg.FEATURE_31_DIR / f"{img_id}.npz"
            if not feat_path.exists():
                missing_features.append(img_id)

            # 检查logits
            logit_path = cfg.LOGITS_DIR / f"{img_id}.npz"
            if not logit_path.exists():
                missing_logits.append(img_id)

        # 报告结果
        print(f"  总数: {len(img_ids)}")

        if missing_labels:
            print(f"  ✗ 缺少标签: {len(missing_labels)}个")
            all_valid = False
        else:
            print(f"  ✓ 标签完整")

        if missing_features:
            print(f"  ✗ 缺少特征: {len(missing_features)}个")
            all_valid = False
        else:
            print(f"  ✓ 特征完整")

        if missing_logits:
            print(f"  ✗ 缺少logits: {len(missing_logits)}个")
            all_valid = False
        else:
            print(f"  ✓ Logits完整")

    print("\n" + "=" * 60)
    if all_valid:
        print("✓ 数据完整性检查通过")
    else:
        print("✗ 数据完整性检查失败，请检查缺失文件")
    print("=" * 60)

    return all_valid


if __name__ == '__main__':
    # 创建数据集划分
    train, val, test = split_dataset(seed=42)

    # 检查数据完整性
    check_data_integrity()

    # 显示一些统计信息
    print("\n示例文件名:")
    print(f"  训练集前3个: {train[:3]}")
    print(f"  验证集前3个: {val[:3]}")
    print(f"  测试集前3个: {test[:3]}")