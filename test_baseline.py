"""
测试Baseline - 【v2.0 最终版】
匹配用户的二分类任务和代码结构
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import kornia
import torch.nn.functional as F
from config import cfg # ✅ 正确的导入方式！
from dataset import build_dataloader
# 从 models 文件夹下的 segformer.py 文件中导入
from models.segformer import build_segformer_distillation

# losses.py 和 dataset.py 也在根目录，所以可以直接导入
from losses import DistillationLoss
from dataset import build_dataloader


def test_single_batch():
    """测试单个batch的完整流程"""
    print("=" * 60)
    print("测试单个Batch（二分类分割+特征蒸馏）")
    print("=" * 60)

    # 1. 创建模型
    print("\n[1/5] 创建模型...")
    model = build_segformer_distillation(
        variant='b1',
        pretrained=True
    )

    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # 2. 加载数据
    print("\n[2/5] 加载数据...")
    train_loader = build_dataloader('train', batch_size=1, num_workers=0)  # batch=1, workers=0
    batch = next(iter(train_loader))

    print("\n数据信息:")
    print(f"  image:            {batch['image'].shape} | {batch['image'].dtype}")
    print(f"  label:            {batch['label'].shape} | {batch['label'].dtype}")
    print(f"  teacher_feat_b30: {batch['teacher_feat_b30'].shape} | {batch['teacher_feat_b30'].dtype}")
    print(f"  teacher_feat_enc: {batch['teacher_feat_enc'].shape} | {batch['teacher_feat_enc'].dtype}")

    # 3. 检查标签
    print("\n[3/5] 检查标签...")
    labels = batch['label']
    unique_vals = labels.unique().tolist()
    print(f"  标签唯一值: {unique_vals}")

    # 统计分布
    for val in unique_vals:
        count = (labels == val).sum().item()
        ratio = count / labels.numel() * 100
        label_name = "ignore" if val == 255 else ("背景" if val == 0 else "前景")
        print(f"    {val} ({label_name}): {count} 像素 ({ratio:.2f}%)")

    # 4. 前向传播
    print("\n[4/5] 前向传播...")
    images = batch['image'].to(device)
    # images = F.interpolate(images, size=(512, 512), mode='bilinear', align_corners=False)
    with torch.no_grad():
        outputs = model(images)

    print(f"\n学生网络输出:")
    print(f"  pred:     {outputs['pred'].shape} - 二分类预测 (单通道)")
    print(f"  feat_b30: {outputs['feat_b30'].shape} - Block30特征（已对齐）")
    print(f"  feat_enc: {outputs['feat_enc'].shape} - Encoder特征（已对齐）")
    print(f"  logits范围: [{outputs['pred'].min():.2f}, {outputs['pred'].max():.2f}]")

    # 5. 计算loss
    print("\n[5/5] 计算损失...")
    # ✅ 正确的尺寸对应
    # ✅ 正确的尺寸对应
    targets = {
        'label': batch['label'].to(device),  # 不再 resize
        'teacher_feat_b30': batch['teacher_feat_b30'].to(device),  # 64×64
        'teacher_feat_enc': batch['teacher_feat_enc'].to(device)  # 64×64
    }

    loss_fn = DistillationLoss()
    total_loss, loss_dict = loss_fn(outputs, targets)

    print(f"\n损失值:")
    print(f"  Total Loss: {total_loss.item():.4f}")
    for key, value in loss_dict.items():
        if key != 'total':
            print(f"    {key:12s}: {value:.4f}")

    # 6. 维度对比
    print(f"\n特征维度对比:")
    print(f"  教师 Block30:  {batch['teacher_feat_b30'].shape}")
    print(f"  学生 Block30:  {outputs['feat_b30'].shape}")
    print(f"  → 通道匹配: {batch['teacher_feat_b30'].shape[1] == outputs['feat_b30'].shape[1]}")

    print(f"\n  教师 Encoder:  {batch['teacher_feat_enc'].shape}")
    print(f"  学生 Encoder:  {outputs['feat_enc'].shape}")
    print(f"  → 通道匹配: {batch['teacher_feat_enc'].shape[1] == outputs['feat_enc'].shape[1]}")

    print("\n" + "=" * 60)
    print("✓ 单batch测试通过")
    print("=" * 60)

    return outputs, targets, loss_dict


def test_metrics():
    """测试评估指标计算"""
    print("=" * 60)
    print("测试评估指标（二分类）")
    print("=" * 60)

    # 创建模型
    print("\n[1/3] 创建模型...")
    model = build_segformer_distillation(variant='b1', pretrained=True)
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # 创建数据加载器（用验证集）
    print("\n[2/3] 创建数据加载器...")
    val_loader = build_dataloader('val', batch_size=4, shuffle=False)
    print(f"  验证集batch数: {len(val_loader)}")

    # 评估（只跑几个batch测试）
    print("\n[3/3] 开始评估（测试模式：只跑前5个batch）...")

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, total=min(5, len(val_loader)), desc="评估")):
            if i >= 5:  # 只测试5个batch
                break

            images = batch['image'].to(device)
            labels = batch['label']  # CPU上

            # 前向传播
            outputs = model(images)
            pred_logits = outputs['pred']  # (B, 1, H, W)

            # 转为二值预测
            pred_binary = (torch.sigmoid(pred_logits) > 0.5).long().squeeze(1)  # (B, H, W)

            # 转为numpy
            pred_binary = pred_binary.cpu().numpy()
            labels_np = labels.numpy()

            # 过滤ignore区域
            mask = (labels_np != 255)
            pred_valid = pred_binary[mask]
            label_valid = labels_np[mask]

            # 二值化标签（>0即为前景）
            label_binary = (label_valid > 0).astype(np.int64)

            all_preds.append(pred_valid)
            all_labels.append(label_binary)

    # 合并所有结果
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # 计算指标
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    # 计算IoU
    intersection = np.logical_and(all_labels == 1, all_preds == 1).sum()
    union = np.logical_or(all_labels == 1, all_preds == 1).sum()
    iou = intersection / (union + 1e-10)

    print("\n" + "=" * 60)
    print("评估结果（前5个batch，仅供测试）")
    print("=" * 60)
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall:    {rec * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    print(f"IoU:       {iou * 100:.2f}%")
    print("=" * 60)

    print("\n注意：这是未训练模型的随机结果，正常情况下指标很低。")
    print("      训练后应该能达到 80%+ 的IoU。")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--metrics':
        # 测试评估指标
        test_metrics()
    else:
        # 默认：测试单个batch
        test_single_batch()

        print("\n提示:")
        print("  - 运行评估测试: python test_baseline.py --metrics")
        print("  - 如果以上测试通过，说明数据流和模型都正常！")
        print("  - 下一步：开始训练")