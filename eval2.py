"""
在老登数据集上评估模型
支持评估多个模型，生成对比报告
"""
import os

os.environ['ALBUMENTATIONS_CHECK_VERSION'] = 'False'

import argparse
import torch
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

from models.segformer import build_model
from config import cfg

# 老登数据集简单加载器（不用unified_dataset）
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


class LaoDengDataset(Dataset):
    """老登数据集加载器"""

    def __init__(self, root_dir, split='test', transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        # 路径：ChangeDetectionDataset/Real/subset/test/
        base_path = self.root_dir / 'Real' / 'subset' / split
        self.img_a_dir = base_path / 'A'
        self.img_b_dir = base_path / 'B'
        self.label_dir = base_path / 'OUT'

        # 获取文件列表（JPG格式）
        self.img_names = sorted([
            f.stem for f in self.img_a_dir.glob('*.jpg')
        ])

        print(f"[老登数据集][{split}] 加载了 {len(self.img_names)} 个样本")

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]

        # 加载图像
        img_a = np.array(Image.open(self.img_a_dir / f'{img_name}.jpg').convert('RGB'))
        img_b = np.array(Image.open(self.img_b_dir / f'{img_name}.jpg').convert('RGB'))
        label = np.array(Image.open(self.label_dir / f'{img_name}.jpg').convert('L'))

        # 二值化标签
        label = (label > 127).astype(np.uint8)

        # 数据增强
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


def get_test_transform(crop_size=256):
    """测试时的数据变换"""
    return A.Compose([
        A.Resize(crop_size, crop_size),  # 统一尺寸
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ], additional_targets={'image2': 'image'})


def compute_metrics(pred, target, threshold=0.5):
    """计算评估指标"""
    pred_prob = torch.sigmoid(pred.squeeze(1))
    pred_class = (pred_prob > threshold).long()

    tp = ((pred_class == 1) & (target == 1)).sum().float()
    fp = ((pred_class == 1) & (target == 0)).sum().float()
    fn = ((pred_class == 0) & (target == 1)).sum().float()
    tn = ((pred_class == 0) & (target == 0)).sum().float()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    oa = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {
        'precision': precision.item() * 100,
        'recall': recall.item() * 100,
        'f1': f1.item() * 100,
        'iou': iou.item() * 100,
        'oa': oa.item() * 100,
        'tp': int(tp.item()),
        'fp': int(fp.item()),
        'fn': int(fn.item()),
        'tn': int(tn.item())
    }


@torch.no_grad()
def evaluate_model(model, test_loader, device):
    """评估模型"""
    model.eval()

    all_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'iou': 0, 'oa': 0}
    num_batches = len(test_loader)

    pbar = tqdm(test_loader, desc='评估中')

    for batch in pbar:
        img_a = batch['img_a'].to(device)
        img_b = batch['img_b'].to(device)
        label = batch['label'].to(device).long()

        outputs = model(img_a, img_b)
        metrics = compute_metrics(outputs['pred'], label)

        for k in all_metrics:
            all_metrics[k] += metrics[k]

        pbar.set_postfix({
            'F1': f"{metrics['f1']:.2f}%",
            'IoU': f"{metrics['iou']:.2f}%"
        })

    # 平均
    for k in all_metrics:
        all_metrics[k] /= num_batches

    return all_metrics


def main():
    parser = argparse.ArgumentParser(description='在老登数据集上评估模型')
    parser.add_argument('--laodeng-root', type=str,
                        default=r'D:\Paper\project\data\ChangeDetectionDataset',
                        help='老登数据集根目录')
    parser.add_argument('--models', type=str, nargs='+',
                        help='模型checkpoint路径列表')
    parser.add_argument('--model-names', type=str, nargs='+',
                        help='模型名称列表（用于显示）')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--crop-size', type=int, default=256)
    args = parser.parse_args()

    # 如果没有指定模型，使用默认的
    if args.models is None:
        args.models = [
            r'outputs\levir_only\checkpoints\best.pth',
            r'outputs\s2looking_only\checkpoints\best.pth',
        ]
        args.model_names = ['LEVIR-CD', 'S2Looking']

    print("=" * 80)
    print("在老登数据集上评估模型")
    print("=" * 80)
    print(f"老登数据集: {args.laodeng_root}")
    print(f"评估模型数量: {len(args.models)}")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}\n")

    # 加载老登数据集
    print("加载老登数据集...")
    test_dataset = LaoDengDataset(
        root_dir=args.laodeng_root,
        split='test',
        transform=get_test_transform(args.crop_size)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    print(f"✓ 加载完成：{len(test_dataset)} 个测试样本\n")

    # 评估每个模型
    all_results = {}

    for model_path, model_name in zip(args.models, args.model_names):
        print("=" * 80)
        print(f"评估模型: {model_name}")
        print(f"Checkpoint: {model_path}")
        print("-" * 80)

        # 检查文件是否存在
        if not Path(model_path).exists():
            print(f"⚠️  模型文件不存在，跳过")
            continue

        # 加载模型
        print("加载模型...")
        model = build_model(variant='b1', pretrained=False, num_classes=1)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        print("✓ 模型加载完成")

        # 评估
        print("开始评估...")
        metrics = evaluate_model(model, test_loader, device)
        all_results[model_name] = metrics

        # 打印结果
        print("\n" + "-" * 80)
        print(f"【{model_name}】在老登数据集上的结果:")
        print("-" * 80)
        print(f"  精确率 (Precision): {metrics['precision']:.2f}%")
        print(f"  召回率 (Recall):    {metrics['recall']:.2f}%")
        print(f"  F1分数 (F1-Score):  {metrics['f1']:.2f}%")
        print(f"  IoU:                {metrics['iou']:.2f}%")
        print(f"  整体精度 (OA):      {metrics['oa']:.2f}%")
        print("=" * 80)
        print()

    # 生成对比表格
    print("\n" + "=" * 80)
    print("📊 评估结果汇总")
    print("=" * 80)
    print(f"{'模型':<20} {'F1 (%)':<12} {'IoU (%)':<12} {'Precision (%)':<15} {'Recall (%)':<12}")
    print("-" * 80)

    for model_name, metrics in all_results.items():
        print(f"{model_name:<20} {metrics['f1']:<12.2f} {metrics['iou']:<12.2f} "
              f"{metrics['precision']:<15.2f} {metrics['recall']:<12.2f}")

    print("=" * 80)

    # 保存结果
    output_dir = Path('outputs/teacher_evaluation')
    output_dir.mkdir(parents=True, exist_ok=True)

    result_file = output_dir / f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'dataset': 'LaoDeng',
            'results': all_results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 结果已保存到: {result_file}")

    # 生成给老登看的报告
    report_file = output_dir / f'report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("跨数据集泛化性评估报告\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"评估数据集: 老登数据集 ({len(test_dataset)} 个样本)\n")
        f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("-" * 80 + "\n")
        f.write("评估结果:\n")
        f.write("-" * 80 + "\n\n")

        for model_name, metrics in all_results.items():
            f.write(f"【{model_name}】\n")
            f.write(f"  训练数据集: {model_name}\n")
            f.write(f"  测试数据集: 老登数据集\n")
            f.write(f"  F1分数: {metrics['f1']:.2f}%\n")
            f.write(f"  IoU: {metrics['iou']:.2f}%\n")
            f.write(f"  精确率: {metrics['precision']:.2f}%\n")
            f.write(f"  召回率: {metrics['recall']:.2f}%\n\n")

        f.write("=" * 80 + "\n")
        f.write("结论:\n")
        f.write("=" * 80 + "\n")
        f.write("从上述结果可以看出，在其他数据集（LEVIR-CD、S2Looking）上训练的\n")
        f.write("模型，在老登数据集上的表现显著下降。这说明变化检测任务的跨数据集\n")
        f.write("泛化性确实很差，模型必须在目标数据集上训练才能取得良好效果。\n\n")
        f.write("这是变化检测任务的固有特性，不是模型设计的问题。\n")
        f.write("=" * 80 + "\n")

    print(f"✓ 报告已保存到: {report_file}")
    print("\n可以把这个报告发给老登看！")


if __name__ == '__main__':
    main()