"""
# 快速评估（不计算FPS）
python eval.py --checkpoint outputs/checkpoints/best.pth --batch-size 4

# 完整评估（包括FPS，使用warm-up）
python eval.py --checkpoint outputs/checkpoints/best.pth \
               --batch-size 4 \
               --compute-fps \
               --num-warmup 10 \
               --num-measure 100

# 对比不同batch size的速度
python eval.py --checkpoint best.pth --batch-size 1 --compute-fps
python eval.py --checkpoint best.pth --batch-size 4 --compute-fps
python eval.py --checkpoint best.pth --batch-size 8 --compute-fps
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
    """评估器 - 优化版"""

    def __init__(self, checkpoint_path, device='cuda', batch_size=4):  # ✅ 新增batch_size参数
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size  # ✅ 可配置batch size

        print("=" * 60)
        print("加载模型...")
        print("=" * 60)
        self.model = self.load_model(checkpoint_path)
        self.print_model_stats()

        # ✅ 优化1: 使用更大的batch_size
        self.test_loader = build_dataloader(
            'test',
            batch_size=self.batch_size,  # 原来是1，现在是4，此处只影响评估速度
            shuffle=False
        )
        print(f"测试集: {len(self.test_loader.dataset)} 张")
        print(f"Batch Size: {self.batch_size}")

    def print_model_stats(self):
        """打印模型统计信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print("\n" + "=" * 60)
        print("模型统计信息")
        print("=" * 60)
        print(f"总参数量:     {total_params / 1e6:.2f}M")
        print(f"可训练参数:   {trainable_params / 1e6:.2f}M")

        # FLOPs计算
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
            pretrained=False
        )

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
    def evaluate(self, compute_fps=False, num_warmup=10, num_measure=100):
        """
        完整评估

        Args:
            compute_fps: 是否计算FPS
            num_warmup: warm-up批次数（默认10）
            num_measure: 测量批次数（默认100）
        """
        print("\n" + "=" * 60)
        print("开始评估...")
        print("=" * 60)

        all_preds = []
        all_labels = []
        latency_list = []

        # ✅ 优化2: 改进FPS统计
        warmup_done = False
        measure_count = 0

        pbar = tqdm(self.test_loader, desc="Evaluating")

        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            labels_binary = batch['label'].cpu().numpy()  # (B, H, W)

            if compute_fps:
                if batch_idx < num_warmup:
                    _ = self.model(images)
                    continue
                if not warmup_done:
                    warmup_done = True
                    torch.cuda.synchronize()

            # 计时开始
            if compute_fps and measure_count < num_measure:
                torch.cuda.synchronize()
                start_time = time.time()

            # 前向传播
            outputs = self.model(images)
            logits = outputs['pred'] if isinstance(outputs, dict) else outputs

            # 计时结束
            if compute_fps and measure_count < num_measure:
                torch.cuda.synchronize()
                latency_list.append(time.time() - start_time)
                measure_count += 1

            # 阈值化
            pred = torch.sigmoid(logits) > 0.5  # (B, 1, H, W)
            pred = pred.squeeze(1).cpu().numpy()  # (B, H, W)

            # ✅ 批量处理（支持batch>1）
            for i in range(pred.shape[0]):
                label_i = labels_binary[i].flatten()
                pred_i = pred[i].flatten()

                # 过滤ignore区域
                mask = (label_i != IGNORE_LABEL)

                all_labels.extend(label_i[mask].tolist())
                all_preds.extend(pred_i[mask].tolist())

            # 更新进度条
            if compute_fps:
                avg_latency = np.mean(latency_list) if latency_list else 0
                fps = self.batch_size / avg_latency if avg_latency > 0 else 0
                pbar.set_postfix({
                    'FPS': f'{fps:.2f}',
                    'Latency': f'{avg_latency * 1000:.2f}ms'
                })

        # 转换为numpy
        all_preds = np.array(all_preds, dtype=np.uint8)
        all_labels = np.array(all_labels, dtype=np.uint8)

        # 计算指标
        metrics = self.compute_metrics(all_preds, all_labels)

        # ✅ 优化3: 更准确的FPS统计
        if compute_fps and len(latency_list) > 0:
            # 去掉最快和最慢的10%，取中间80%的平均值（更稳定）
            latency_sorted = sorted(latency_list)
            trim = len(latency_sorted) // 10
            latency_trimmed = latency_sorted[trim:-trim] if trim > 0 else latency_sorted

            avg_latency = np.mean(latency_trimmed)
            std_latency = np.std(latency_trimmed)
            fps = self.batch_size / avg_latency

            metrics['FPS'] = float(fps)
            metrics['Latency_ms'] = float(avg_latency * 1000)
            metrics['Latency_std_ms'] = float(std_latency * 1000)
            metrics['Throughput'] = float(fps)  # 样本/秒
            metrics['Num_measured_batches'] = len(latency_list)

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

        # IoU
        intersection = tp
        union = tp + fp + fn
        iou = intersection / (union + 1e-10)

        # Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        # Dice
        dice = 2 * tp / (2 * tp + fp + fn + 1e-10)

        # Per-class IoU
        iou_bg = tn / (tn + fp + fn + 1e-10)
        iou_fg = tp / (tp + fp + fn + 1e-10)
        miou = (iou_bg + iou_fg) / 2

        metrics = {
            'IoU': float(iou),
            'mIoU': float(miou),
            'IoU_bg': float(iou_bg),
            'IoU_fg': float(iou_fg),
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
            print(f"\n推理速度 (Batch={self.batch_size}):")
            print(f"  FPS:                         {metrics['FPS']:.2f} 样本/秒")
            print(f"  Throughput:                  {metrics['Throughput']:.2f} 样本/秒")
            print(f"  Latency (avg):               {metrics['Latency_ms']:.2f} ms")
            print(f"  Latency (std):               {metrics['Latency_std_ms']:.2f} ms")
            print(f"  Measured batches:            {metrics['Num_measured_batches']}")

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
    parser = argparse.ArgumentParser(description='评估SegFormer蒸馏模型 - 优化版')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='checkpoint路径')
    parser.add_argument('--save-dir', type=str, default='outputs/eval_results',
                        help='结果保存目录')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='评估batch size（默认4，可选1/4/8）')
    parser.add_argument('--compute-fps', action='store_true',
                        help='是否计算FPS（需要warm-up）')
    parser.add_argument('--num-warmup', type=int, default=10,
                        help='Warm-up批次数')
    parser.add_argument('--num-measure', type=int, default=100,
                        help='FPS测量批次数')
    args = parser.parse_args()

    # 创建评估器
    evaluator = Evaluator(
        args.checkpoint,
        batch_size=args.batch_size  # ✅ 使用命令行参数
    )

    # 评估
    metrics = evaluator.evaluate(
        compute_fps=args.compute_fps,
        num_warmup=args.num_warmup,
        num_measure=args.num_measure
    )

    # 打印结果
    evaluator.print_metrics(metrics)

    # 保存结果
    checkpoint_path = Path(args.checkpoint)
    checkpoint_name = checkpoint_path.stem

    save_path = Path(args.save_dir) / f'metrics_{checkpoint_name}.json'
    evaluator.save_metrics(metrics, save_path)

    print("\n✓ 评估完成！")


if __name__ == '__main__':
    main()