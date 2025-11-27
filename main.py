# check_whucd_structure.py

from pathlib import Path

root = Path(r"D:\Paper\project\data\Building change detection dataset_add")

print("="*60)
print("WHUCD Dataset Structure Analysis")
print("="*60)

# 检查2012 train图像
path_2012_train_img = root / "1. The two-period image data" / "2012" / "splited_images" / "train" / "image"
print(f"\n2012 Train Images:")
print(f"  Path exists: {path_2012_train_img.exists()}")
if path_2012_train_img.exists():
    files = list(path_2012_train_img.glob("*"))
    print(f"  Total files: {len(files)}")
    if files:
        print(f"  First 3 files:")
        for f in files[:3]:
            print(f"    - {f.name} ({f.suffix})")

# 检查2016 train图像
path_2016_train_img = root / "1. The two-period image data" / "2016" / "splited_images" / "train" / "image"
print(f"\n2016 Train Images:")
print(f"  Path exists: {path_2016_train_img.exists()}")
if path_2016_train_img.exists():
    files = list(path_2016_train_img.glob("*"))
    print(f"  Total files: {len(files)}")
    if files:
        print(f"  First 3 files:")
        for f in files[:3]:
            print(f"    - {f.name} ({f.suffix})")

# 检查change_label
path_label = root / "change_label" / "train"
print(f"\nChange Label (train):")
print(f"  Path exists: {path_label.exists()}")
if path_label.exists():
    files = list(path_label.glob("*"))
    print(f"  Total files: {len(files)}")
    for f in files:
        print(f"    - {f.name} ({f.suffix}, {f.stat().st_size / 1024 / 1024:.2f} MB)")

# 检查test集
path_2012_test_img = root / "1. The two-period image data" / "2012" / "splited_images" / "test" / "image"
print(f"\n2012 Test Images:")
print(f"  Path exists: {path_2012_test_img.exists()}")
if path_2012_test_img.exists():
    files = list(path_2012_test_img.glob("*"))
    print(f"  Total files: {len(files)}")

# 检查是否有val集
path_2012_val_img = root / "1. The two-period image data" / "2012" / "splited_images" / "val" / "image"
print(f"\n2012 Val Images:")
print(f"  Path exists: {path_2012_val_img.exists()}")

print("\n" + "="*60)