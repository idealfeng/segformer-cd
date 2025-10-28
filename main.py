"""
本地数据终极校验脚本 (2025-10-26 Final)

功能:
- 在本地电脑上，验证从AutoDL下载并解压的数据是否完整、正确。
- 检查预处理数据 (images + labels)
- 检查教师网络输出 (只有2层特征)
"""
import cv2
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

# ======================================================================
#                     【唯一需要你修改的地方】
# ======================================================================
# 请把下面的路径，修改成你解压后数据所在的【父目录】
# 例如，如果你的数据在 'D:\MyProjectData\Potsdam_processed'
# 那么 BASE_DIR 就应该是 'D:\MyProjectData'

# --- Windows 路径示例 ---
BASE_DIR = Path(r'D:\Paper\project')

# --- macOS / Linux 路径示例 ---
# BASE_DIR = Path('/Users/your_username/Documents/MyProjectData')
# ======================================================================


def verify_preprocessed_data(base_path):
    """验证预处理数据"""
    print("\n" + "=" * 60)
    print("1. 正在验证: 预处理数据 (Potsdam_processed)")
    print("=" * 60)

    preprocessed_dir = base_path / 'Potsdam_processed'
    img_dir = preprocessed_dir / 'images'
    label_dir = preprocessed_dir / 'labels'

    if not preprocessed_dir.exists():
        print(f"✗ 错误: 找不到目录 '{preprocessed_dir}'")
        return False

    img_files = sorted(list(img_dir.glob('*.png')))
    label_files = sorted(list(label_dir.glob('*.png')))

    print(f"  - 找到 Images: {len(img_files)} 个")
    print(f"  - 找到 Labels: {len(label_files)} 个")

    if not img_files or len(img_files) != len(label_files):
        print("✗ 错误: Images 和 Labels 数量不匹配或为空！")
        return False

    print("\n  - 抽样检查前3个文件...")
    for i in range(min(3, len(img_files))):
        label = cv2.imread(str(label_files[i]), cv2.IMREAD_GRAYSCALE)
        unique_vals = set(np.unique(label))
        valid_vals = {0, 1, 2, 3, 4, 5, 255}

        if not unique_vals.issubset(valid_vals):
            print(f"  ✗ 错误: 文件 '{label_files[i].name}' 包含非法标签值: {unique_vals - valid_vals}")
            return False

    print("  ✓ 文件数量匹配，抽样标签值正确。")
    print("✅ [通过] 预处理数据验证通过")
    return True


def verify_teacher_outputs(base_path):
    """验证教师网络输出（只有特征）"""
    print("\n" + "=" * 60)
    print("2. 正在验证: 教师网络输出 (teacher_outputs)")
    print("=" * 60)

    teacher_dir = base_path / 'teacher_outputs'
    block30_dir = teacher_dir / 'features_block30'
    encoder_dir = teacher_dir / 'features_encoder'

    if not teacher_dir.exists():
        print(f"✗ 错误: 找不到目录 '{teacher_dir}'")
        return False

    block30_files = sorted(list(block30_dir.glob('*.npz')))
    encoder_files = sorted(list(encoder_dir.glob('*.npz')))

    print(f"  - 找到 Block 30 特征: {len(block30_files)} 个")
    print(f"  - 找到 Encoder 特征: {len(encoder_files)} 个")

    if not block30_files or len(block30_files) != len(encoder_files):
        print("✗ 错误: 两层特征数量不匹配或为空！")
        return False

    print("\n  - 抽样检查前3个文件的Shape...")
    for i in range(min(3, len(block30_files))):
        block30_data = np.load(str(block30_files[i]))
        encoder_data = np.load(str(encoder_files[i]))

        feat_30 = block30_data['features']
        feat_enc = encoder_data['features']

        # 宽松检查，兼容 (B,H,W,C) 和 (B,C,H,W)
        if feat_30.shape[-1] != 1280 and feat_30.shape[1] != 1280:
             print(f"  ✗ 错误: 文件 '{block30_files[i].name}' Block 30 shape不对: {feat_30.shape}")
             return False
        if feat_enc.shape != (1, 256, 64, 64):
            print(f"  ✗ 错误: 文件 '{encoder_files[i].name}' Encoder shape不对: {feat_enc.shape}")
            return False

    print("  ✓ 文件数量匹配，抽样Shape正确。")
    print("✅ [通过] 教师网络输出验证通过")
    return True


def main():
    """主验证流程"""
    print("=" * 60)
    print("        本地数据终极校验脚本")
    print("=" * 60)

    if not BASE_DIR.exists():
        print(f"\n❌ 致命错误: 配置的基础路径不存在！")
        print(f"  请修改脚本中的 'BASE_DIR' 变量为您数据所在的正确路径。")
        print(f"  当前配置路径: {BASE_DIR}")
        return 1

    checks = [
        verify_preprocessed_data(BASE_DIR),
        verify_teacher_outputs(BASE_DIR)
    ]

    print("\n" + "=" * 60)
    print("                  总结")
    print("=" * 60)

    if all(checks):
        print("\n🎉🎉🎉 恭喜！所有数据在本地验证通过！🎉🎉🎉")
        print("\n你可以安心地关闭AutoDL实例，开始下一阶段的实验了。")
        return 0
    else:
        print("\n❌ 注意：部分数据验证失败，请检查上面的错误信息！")
        return 1


if __name__ == '__main__':
    sys.exit(main())