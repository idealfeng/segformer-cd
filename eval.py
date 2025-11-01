"""
评估脚本 - 修复版
功能：完整测试集评估，计算各项指标

运行：
    python eval.py --checkpoint outputs/checkpoints/best.pth
    python eval.py --checkpoint outputs/checkpoints/epoch_2.pth
"""
import argparse
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import time

from config import cfg
from dataset import build_dataloader
from models.segformer import build_segformer_distillation

# ✅ 修复1: 定义IGNORE_LABEL
IGNORE_LABEL = 255  # 背景忽略标签


class Evaluator:
    """评估器"""

    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 加载模型
        print("=" * 60)
        print("加载模型...")
        print("=" * 60)
        self.model = self.load_model(checkpoint_path)
        self.print_model_stats()

        # 数据加载器
        self.test_loader = build_dataloader('test', batch_size=1, shuffle=False)
        print(f"测试集: {len(self.test_loader.dataset)} 张")

    def print_model_stats(self):
        """打印模型统计信息"""
        # 1. 参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print("\n" + "=" * 60)
        print("模型统计信息")
        print("=" * 60)
        print(f"总参数量:     {total_params / 1e6:.2f}M")
        print(f"可训练参数:   {trainable_params / 1e6:.2f}M")

        # 2. FLOPs（需要安装thop）
        try:
            from thop import profile
            dummy_input = torch.randn(1, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE).to(self.device)
            flops, _ = profile(self.model, inputs=(dummy_input,), verbose=False)
            print(f"FLOPs:        {flops / 1e9:.2f}G")
        except ImportError:
            print("FLOPs:        需安装thop (pip install thop)")
        except Exception as e:
            print(f"FLOPs:        计算失败 ({e})")

        print("=" * 60)

    def load_model(self, checkpoint_path):
        """加载模型"""
        model = build_segformer_distillation(
            variant=cfg.STUDENT_MODEL.replace('segformer_', ''),
            pretrained=False  # 评估时不需要预训练权重
        )

        # 加载checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()

        print(f"✓ 加载checkpoint: {checkpoint_path}")
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if 'best_iou' in checkpoint:
            print(f"  Best IoU: {checkpoint['best_iou']:.4f}")

        return model

    @torch.no_grad()
    def evaluate(self, compute_fps=False):
        """完整评估"""
        print("\n" + "=" * 60)
        print("开始评估...")
        print("=" * 60)

        all_preds = []
        all_labels = []
        latency_list = []

        # ✅ 修复4: FPS统计增加warm-up
        warmup = 5 if compute_fps else 0

        pbar = tqdm(self.test_loader, desc="Evaluating")

        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            labels_binary = batch['label'].cpu().numpy()  # (B, H, W)

            # Warm-up阶段
            if batch_idx < warmup:
                _ = self.model(images)
                continue

            # 计时开始
            if compute_fps:
                torch.cuda.synchronize()
                start_time = time.time()

            # 前向传播
            outputs = self.model(images)
            logits = outputs['pred'] if isinstance(outputs, dict) else outputs

            # 计时结束
            if compute_fps:
                torch.cuda.synchronize()
                latency_list.append(time.time() - start_time)

            # ✅ 修复3: 单通道阈值化（当前代码已对）
            pred = torch.sigmoid(logits) > 0.5  # (B, 1, H, W)
            pred = pred.squeeze(1).cpu().numpy()  # (B, H, W)

            # ✅ 修复2: 改进维度处理
            for i in range(pred.shape[0]):
                label_i = labels_binary[i].flatten()  # (H*W,)
                pred_i = pred[i].flatten()  # (H*W,)

                # 过滤ignore区域
                mask = (label_i != IGNORE_LABEL)

                all_labels.extend(label_i[mask].tolist())
                all_preds.extend(pred_i[mask].tolist())

        # 转换为numpy
        all_preds = np.array(all_preds, dtype=np.uint8)
        all_labels = np.array(all_labels, dtype=np.uint8)

        # 计算指标
        metrics = self.compute_metrics(all_preds, all_labels)

        # 添加FPS
        if compute_fps and len(latency_list) > 0:
            avg_latency = np.mean(latency_list)
            fps = self.test_loader.batch_size / avg_latency
            metrics['FPS'] = float(fps)
            metrics['Latency_ms'] = float(avg_latency * 1000)

        return metrics

    def compute_metrics(self, preds, labels):
        """计算各项指标"""
        # Confusion Matrix
        cm = confusion_matrix(labels, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        # Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='binary', pos_label=1
        )

        # IoU (Intersection over Union)
        intersection = tp
        union = tp + fp + fn
        iou = intersection / (union + 1e-10)

        # Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        # Dice Coefficient
        dice = 2 * tp / (2 * tp + fp + fn + 1e-10)

        # ✅ 新增: Per-class IoU
        iou_bg = tn / (tn + fp + fn + 1e-10)  # 背景IoU
        iou_fg = tp / (tp + fp + fn + 1e-10)  # 前景IoU
        miou = (iou_bg + iou_fg) / 2  # mIoU

        metrics = {
            'IoU': float(iou),  # 前景IoU（主要指标）
            'mIoU': float(miou),  # 平均IoU
            'IoU_bg': float(iou_bg),  # 背景IoU
            'IoU_fg': float(iou_fg),  # 前景IoU（同IoU）
            'F1': float(f1),
            'Dice': float(dice),
            'Precision': float(precision),
            'Recall': float(recall),
            'Accuracy': float(accuracy),
            'TP': int(tp),
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn)
        }

        return metrics

    def print_metrics(self, metrics):
        """打印指标"""
        print("\n" + "=" * 60)
        print("评估结果")
        print("=" * 60)
        print(f"mIoU (Mean IoU):               {metrics['mIoU']:.4f}")
        print(f"IoU (Foreground):              {metrics['IoU']:.4f}")
        print(f"  - Background IoU:            {metrics['IoU_bg']:.4f}")
        print(f"  - Foreground IoU:            {metrics['IoU_fg']:.4f}")
        print(f"F1-Score:                      {metrics['F1']:.4f}")
        print(f"Dice Coefficient:              {metrics['Dice']:.4f}")
        print(f"Precision:                     {metrics['Precision']:.4f}")
        print(f"Recall:                        {metrics['Recall']:.4f}")
        print(f"Overall Accuracy:              {metrics['Accuracy']:.4f}")

        if 'FPS' in metrics:
            print(f"\n推理速度:")
            print(f"  FPS:                         {metrics['FPS']:.2f}")
            print(f"  Latency:                     {metrics['Latency_ms']:.2f} ms")

        print("\nConfusion Matrix:")
        print(f"  True Positive  (TP): {metrics['TP']:8d}")
        print(f"  True Negative  (TN): {metrics['TN']:8d}")
        print(f"  False Positive (FP): {metrics['FP']:8d}")
        print(f"  False Negative (FN): {metrics['FN']:8d}")
        print("=" * 60)

    def save_metrics(self, metrics, save_path):
        """保存指标到JSON"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        print(f"\n✓ 指标已保存到: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='评估SegFormer蒸馏模型')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='checkpoint路径')
    parser.add_argument('--save-dir', type=str, default='outputs/eval_results',
                        help='结果保存目录')
    parser.add_argument('--compute-fps', action='store_true',
                        help='是否计算FPS（需要warm-up）')
    args = parser.parse_args()

    # 创建评估器
    evaluator = Evaluator(args.checkpoint)

    # 评估
    metrics = evaluator.evaluate(compute_fps=args.compute_fps)

    # 打印结果
    evaluator.print_metrics(metrics)

    # 保存结果
    checkpoint_path = Path(args.checkpoint)
    checkpoint_name = checkpoint_path.stem  # 'epoch_2' or 'best'

    save_path = Path(args.save_dir) / f'metrics_{checkpoint_name}.json'
    evaluator.save_metrics(metrics, save_path)

    print("\n✓ 评估完成！")


if __name__ == '__main__':
    main()