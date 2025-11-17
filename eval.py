"""
变化检测评估脚本

运行:
    python eval.py --checkpoint outputs/checkpoints/best_model.pth
    python eval.py --checkpoint best.pth --compute-fps
    python eval.py --checkpoint best.pth --visualize
"""
import argparse
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json
import time
from PIL import Image
from torch.utils.data import DataLoader

from config import cfg
from dataset import create_dataloaders
from models.segformer import build_model


class Evaluator:
    """变化检测评估器"""

    def __init__(self, checkpoint_path, device='cuda', batch_size=8):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size

        print("=" * 60)
        print("Loading model...")
        print("=" * 60)

        self.model = self.load_model(checkpoint_path)
        self.test_loader = self.build_test_loader()

        print(f"Test samples: {len(self.test_loader.dataset)}")
        print(f"Batch size: {batch_size}")
        print(f"Device: {self.device}")

    def load_model(self, checkpoint_path):
        """加载模型"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 从checkpoint获取配置
        config = checkpoint.get('config', {})
        model_type = config.get('model_type', cfg.MODEL_TYPE)
        num_classes = config.get('num_classes', cfg.NUM_CLASSES)

        # 构建模型
        variant = model_type.split('_')[-1]
        model = build_model(
            variant=variant,
            pretrained=False,  # 不需要预训练，直接加载权重
            num_classes=num_classes
        )

        # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()

        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Best metric: {checkpoint.get('best_metric', 'N/A'):.4f}")

        return model

    def build_test_loader(self):
        """构建测试数据加载器"""
        from dataset import LEVIRCDDataset, get_test_transforms_full

        # 测试集用完整1024x1024图像，不裁剪
        test_dataset = LEVIRCDDataset(
            root_dir=cfg.DATA_ROOT,
            split='test',
            transform=get_test_transforms_full(),  # 只归一化，不裁剪
            crop_size=cfg.ORIGINAL_SIZE  # 1024
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=1,  # 1024x1024图像用batch_size=1避免OOM
            shuffle=False,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=False
        )
        return test_loader

    def print_model_stats(self):
        """打印模型统计信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"\nModel Statistics:")
        print(f"  Total parameters: {total_params / 1e6:.2f}M")
        print(f"  Trainable parameters: {trainable_params / 1e6:.2f}M")

    def sliding_window_inference(self, img_a, img_b, window_size=256, stride=256):
        """
        滑动窗口推理，处理大尺寸图像

        Args:
            img_a: (1, 3, H, W)
            img_b: (1, 3, H, W)
            window_size: 窗口大小（与训练尺寸一致）
            stride: 滑动步长（stride < window_size 会有重叠）

        Returns:
            pred_prob: (1, H, W) 概率图
        """
        _, _, H, W = img_a.shape

        # 如果图像尺寸小于等于窗口，直接推理
        if H <= window_size and W <= window_size:
            outputs = self.model(img_a, img_b)
            return torch.sigmoid(outputs['pred'].squeeze(1))

        # 初始化输出和计数器（用于重叠区域平均）
        pred_sum = torch.zeros((1, H, W), device=self.device)
        count_map = torch.zeros((1, H, W), device=self.device)

        # 滑动窗口
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                # 计算窗口边界
                y_end = min(y + window_size, H)
                x_end = min(x + window_size, W)
                y_start = y_end - window_size
                x_start = x_end - window_size

                # 确保不越界
                y_start = max(0, y_start)
                x_start = max(0, x_start)

                # 提取patch
                patch_a = img_a[:, :, y_start:y_end, x_start:x_end]
                patch_b = img_b[:, :, y_start:y_end, x_start:x_end]

                # 推理
                outputs = self.model(patch_a, patch_b)
                patch_pred = torch.sigmoid(outputs['pred'].squeeze(1))  # (1, h, w)

                # 累加到输出
                pred_sum[:, y_start:y_end, x_start:x_end] += patch_pred
                count_map[:, y_start:y_end, x_start:x_end] += 1

        # 平均（处理重叠区域）
        pred_prob = pred_sum / count_map.clamp(min=1)

        return pred_prob

    @torch.no_grad()
    def evaluate(self):
        """评估模型性能"""
        print("\n" + "=" * 60)
        print("Evaluating on test set...")
        print(f"Using sliding window inference (window={cfg.IMAGE_SIZE}, stride={cfg.IMAGE_SIZE})")
        print("=" * 60)

        # 累积统计量
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_tn = 0

        pbar = tqdm(self.test_loader, desc='Evaluating')

        for batch in pbar:
            img_a = batch['img_a'].to(self.device)
            img_b = batch['img_b'].to(self.device)
            label = batch['label'].to(self.device).long()

            # 滑动窗口推理
            pred_prob = self.sliding_window_inference(
                img_a, img_b,
                window_size=cfg.IMAGE_SIZE,  # 256
                stride=cfg.IMAGE_SIZE  # 无重叠，可改为128有重叠
            )
            pred = (pred_prob > 0.5).long().squeeze(0)  # (H, W)

            # 累积混淆矩阵
            tp = ((pred == 1) & (label == 1)).sum().item()
            fp = ((pred == 1) & (label == 0)).sum().item()
            fn = ((pred == 0) & (label == 1)).sum().item()
            tn = ((pred == 0) & (label == 0)).sum().item()

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

        # 计算指标
        precision = total_tp / (total_tp + total_fp + 1e-8)
        recall = total_tp / (total_tp + total_fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        iou = total_tp / (total_tp + total_fp + total_fn + 1e-8)
        oa = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + 1e-8)

        # Kappa系数
        total = total_tp + total_fp + total_fn + total_tn
        pe = ((total_tp + total_fp) * (total_tp + total_fn) +
              (total_fn + total_tn) * (total_fp + total_tn)) / (total * total + 1e-8)
        kappa = (oa - pe) / (1 - pe + 1e-8)

        results = {
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'IoU': iou,
            'OA': oa,
            'Kappa': kappa,
            'TP': total_tp,
            'FP': total_fp,
            'FN': total_fn,
            'TN': total_tn
        }

        return results

    @torch.no_grad()
    def compute_fps(self, num_warmup=10, num_measure=100):
        """计算FPS"""
        print("\n" + "=" * 60)
        print("Computing FPS...")
        print("=" * 60)

        # 创建虚拟输入
        dummy_a = torch.randn(1, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE).to(self.device)
        dummy_b = torch.randn(1, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE).to(self.device)

        # Warmup
        print(f"Warming up ({num_warmup} iterations)...")
        for _ in range(num_warmup):
            _ = self.model(dummy_a, dummy_b)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        # 测量
        print(f"Measuring ({num_measure} iterations)...")
        timings = []

        for _ in tqdm(range(num_measure), desc='FPS Test'):
            start_time = time.time()
            _ = self.model(dummy_a, dummy_b)

            if self.device.type == 'cuda':
                torch.cuda.synchronize()

            elapsed = time.time() - start_time
            timings.append(elapsed)

        # 统计
        timings = np.array(timings)

        # 去除离群值（最快和最慢的10%）
        timings_sorted = np.sort(timings)
        trim_idx = int(len(timings_sorted) * 0.1)
        timings_trimmed = timings_sorted[trim_idx:-trim_idx] if trim_idx > 0 else timings_sorted

        avg_time = np.mean(timings_trimmed)
        std_time = np.std(timings_trimmed)
        fps = 1.0 / avg_time

        results = {
            'FPS': fps,
            'Avg Latency (ms)': avg_time * 1000,
            'Std Latency (ms)': std_time * 1000,
            'Image Size': cfg.IMAGE_SIZE
        }

        return results

    @torch.no_grad()
    def visualize(self, num_samples=10, save_dir=None):
        """可视化预测结果"""
        if save_dir is None:
            save_dir = cfg.VIS_DIR / 'predictions'
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving visualizations to {save_dir}...")

        count = 0
        for batch in tqdm(self.test_loader, desc='Visualizing'):
            if count >= num_samples:
                break

            img_a = batch['img_a'].to(self.device)
            img_b = batch['img_b'].to(self.device)
            label = batch['label'].to(self.device).long()
            names = batch['name']

            # 使用滑动窗口推理
            pred_prob = self.sliding_window_inference(
                img_a, img_b,
                window_size=cfg.IMAGE_SIZE,
                stride=cfg.IMAGE_SIZE
            )
            pred = (pred_prob > 0.5).long().squeeze(0)  # (H, W)

            # 保存每个样本（batch_size=1）
            name = names[0]

            # 转换为numpy
            pred_np = pred.cpu().numpy().astype(np.uint8) * 255
            label_np = label[0].cpu().numpy().astype(np.uint8) * 255

            # 保存预测和GT
            pred_img = Image.fromarray(pred_np)
            label_img = Image.fromarray(label_np)

            pred_img.save(save_dir / f'{name}_pred.png')
            label_img.save(save_dir / f'{name}_gt.png')

            # 计算误差图（红色：FP，蓝色：FN）
            error_map = np.zeros((pred_np.shape[0], pred_np.shape[1], 3), dtype=np.uint8)
            error_map[(pred_np > 0) & (label_np == 0)] = [255, 0, 0]  # FP: 红色
            error_map[(pred_np == 0) & (label_np > 0)] = [0, 0, 255]  # FN: 蓝色
            error_map[(pred_np > 0) & (label_np > 0)] = [0, 255, 0]   # TP: 绿色

            error_img = Image.fromarray(error_map)
            error_img.save(save_dir / f'{name}_error.png')

            count += 1

        print(f"Saved {count} visualizations")


def main():
    parser = argparse.ArgumentParser(description='Change Detection Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--compute-fps', action='store_true', help='Compute FPS')
    parser.add_argument('--visualize', action='store_true', help='Save visualizations')
    parser.add_argument('--num-vis', type=int, default=10, help='Number of visualizations')
    parser.add_argument('--num-warmup', type=int, default=10, help='FPS warmup iterations')
    parser.add_argument('--num-measure', type=int, default=100, help='FPS measurement iterations')

    args = parser.parse_args()

    # 创建评估器
    evaluator = Evaluator(
        checkpoint_path=args.checkpoint,
        device=cfg.DEVICE,
        batch_size=args.batch_size
    )

    # 打印模型统计
    evaluator.print_model_stats()

    # 评估性能
    results = evaluator.evaluate()

    print("\n" + "=" * 60)
    print("Evaluation Results:")
    print("=" * 60)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # 计算FPS
    if args.compute_fps:
        fps_results = evaluator.compute_fps(
            num_warmup=args.num_warmup,
            num_measure=args.num_measure
        )

        print("\nFPS Results:")
        for k, v in fps_results.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.2f}")
            else:
                print(f"  {k}: {v}")

        results.update(fps_results)

    # 可视化
    if args.visualize:
        evaluator.visualize(num_samples=args.num_vis)

    # 保存结果
    results_path = cfg.RESULTS_DIR / 'eval_results.json'
    with open(results_path, 'w') as f:
        # 转换numpy类型为python原生类型
        results_json = {k: float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v for k, v in results.items()}
        json.dump(results_json, f, indent=2)

    print(f"\nResults saved to {results_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
