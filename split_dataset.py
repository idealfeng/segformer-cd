"""
数据集划分脚本
将2400张图像按7:1.5:1.5划分为训练集、验证集、测试集
"""
import os
import random
from pathlib import Path
from config import cfg
import numpy as np

ENABLE_SHAPE_CHECK = True  # 想关闭就改成 False
def split_dataset(seed=42):
    """
    划分数据集为train/val/test

    关键：按原图划分，避免数据泄漏！
    同一张原图的所有patch必须在同一个子集中

    Args:
        seed: 随机种子，保证可复现
    """
    print("=" * 60)
    print("数据集划分（按原图）")
    print("=" * 60)

    # 设置随机种子
    random.seed(seed)

    # 获取所有图像文件
    image_dir = cfg.IMAGE_DIR
    all_images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

    print(f"总patch数: {len(all_images)}")

    # 提取原图名称（去掉_cropXXXX后缀）
    # 例如：top_potsdam_2_10_RGB_crop0001.png -> top_potsdam_2_10_RGB
    orig_images = {}
    for img_file in all_images:
        # 提取原图名
        orig_name = '_'.join(img_file.split('_')[:-1])  # 去掉cropXXXX
        if orig_name not in orig_images:
            orig_images[orig_name] = []
        orig_images[orig_name].append(img_file)

    print(f"原图数量: {len(orig_images)}")
    print(f"平均每张原图的patch数: {len(all_images) / len(orig_images):.1f}")

    # 按原图划分
    orig_names = list(orig_images.keys())
    random.shuffle(orig_names)

    # 计算划分点
    total = len(orig_names)
    val_count = int(total * cfg.VAL_RATIO)
    test_count = int(total * cfg.TEST_RATIO)
    train_count = total - val_count - test_count  # ← 剩余的全给训练集

    # 划分原图
    train_orig = orig_names[:train_count]
    val_orig = orig_names[train_count:train_count + val_count]
    test_orig = orig_names[train_count + val_count:]

    # 收集所有patch
    train_list = []
    val_list = []
    test_list = []

    for orig in train_orig:
        train_list.extend(orig_images[orig])
    for orig in val_orig:
        val_list.extend(orig_images[orig])
    for orig in test_orig:
        test_list.extend(orig_images[orig])

    print(f"\n划分结果:")
    print(f"  训练集: {len(train_orig)}张原图 -> {len(train_list)}个patch ({len(train_list)/len(all_images)*100:.1f}%)")
    print(f"  验证集: {len(val_orig)}张原图 -> {len(val_list)}个patch ({len(val_list)/len(all_images)*100:.1f}%)")
    print(f"  测试集: {len(test_orig)}张原图 -> {len(test_list)}个patch ({len(test_list)/len(all_images)*100:.1f}%)")

    print(f"\n原图分布:")
    print(f"  训练原图: {train_orig[:3]}...")
    print(f"  验证原图: {val_orig}")
    print(f"  测试原图: {test_orig}")

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
    print("✓ 数据集划分完成（无数据泄漏）")
    print("=" * 60)

    # 验证：检查原图是否有重叠
    train_orig_set = set(train_orig)
    val_orig_set = set(val_orig)
    test_orig_set = set(test_orig)

    assert len(train_orig_set & val_orig_set) == 0, "训练集和验证集原图有重叠！"
    assert len(train_orig_set & test_orig_set) == 0, "训练集和测试集原图有重叠！"
    assert len(val_orig_set & test_orig_set) == 0, "验证集和测试集原图有重叠！"

    print("✓ 验证通过：原图无重叠，无数据泄漏")

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

def quick_probe_feat(npz_path, key='features'):
    with np.load(npz_path) as f:
        arr = f[key][0]  # 取第0条（你现在的存储方式）
    return arr.shape

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
        badshape_features = []  # 新增：形状不匹配的样本

        for img_id in img_ids:
            # 1) 标签存在性
            label_path = cfg.LABEL_DIR / f"{img_id}.png"
            if not label_path.exists():
                missing_labels.append(img_id)

            # 2) 特征存在性
            feat30_path = cfg.FEATURE_BLOCK30_DIR / f"{img_id}.npz"
            feat_enc_path = cfg.FEATURE_ENCODER_DIR / f"{img_id}.npz"
            has_feat30 = feat30_path.exists()
            has_featenc = feat_enc_path.exists()

            if not (has_feat30 and has_featenc):
                missing_features.append(img_id)
                continue  # 没文件就别做形状检查，避免双重计数

            # 3) （可选）形状检查：只在文件都存在时进行
            if ENABLE_SHAPE_CHECK:
                try:
                    shp30 = quick_probe_feat(feat30_path)  # 你目前导出是 (64,64,1280)
                    shpenc = quick_probe_feat(feat_enc_path)  # 你目前导出是 (256,64,64)

                    # 允许 B30 两种存储习惯：HWC 或 CHW（更稳妥）
                    ok30 = (shp30 == (64, 64, cfg.TEACHER_FEAT_BLOCK30_DIM)) or \
                           (shp30 == (cfg.TEACHER_FEAT_BLOCK30_DIM, 64, 64))

                    # Encoder 期望 (C, H, W)；如果你可能存成 HWC，也放宽一下
                    okenc = (shpenc == (cfg.TEACHER_FEAT_ENCODER_DIM, 64, 64)) or \
                            (shpenc == (64, 64, cfg.TEACHER_FEAT_ENCODER_DIM))

                    if not (ok30 and okenc):
                        badshape_features.append((img_id, shp30, shpenc))
                except Exception as e:
                    badshape_features.append((img_id, 'read_error', str(e)))

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
        if ENABLE_SHAPE_CHECK:
            if badshape_features:
                print(f"  ✗ 特征形状不匹配: {len(badshape_features)}，示例: {badshape_features[:5]}")
                all_valid = False
            else:
                print("  ✓ 特征形状匹配")

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