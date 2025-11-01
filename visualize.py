"""
可视化脚本
功能：生成预测结果可视化图，用于论文

运行：
    python visualize.py --checkpoint outputs/checkpoints/best.pth --num-samples 10
    python visualize.py --checkpoint outputs/checkpoints/best.pth --save-all
"""
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import cv2

from config import cfg
from dataset import build_dataloader
from models.segformer import build_segformer_distillation


class Visualizer:
    """可视化器"""

    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 加载模型
        print("=" * 60)
        print("加载模型...")
        print("=" * 60)
        self.model = self.load_model(checkpoint_path)

        # 数据加载器
        self.test_loader = build_dataloader('test', batch_size=1, shuffle=False)

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
        return model

    @torch.no_grad()
    def visualize_samples(self, num_samples=10, save_dir='outputs/visualizations'):
        """可视化样本"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 60)
        print(f"生成可视化（{num_samples}张）...")
        print("=" * 60)

        count = 0
        pbar = tqdm(self.test_loader, total=num_samples)

        for batch_idx, batch in enumerate(pbar):
            if count >= num_samples:
                break

            images = batch['image'].to(self.device)  # (1, 3, H, W)
            labels = batch['label'].cpu().numpy()[0]  # (H, W)

            # 前向传播
            outputs = self.model(images)
            pred = torch.sigmoid(outputs['pred']) > 0.5
            pred = pred.squeeze().cpu().numpy()  # (H, W)

            # 反归一化图像
            image = self.denormalize_image(images[0])

            # 绘制对比图
            self.plot_comparison(
                image, labels, pred,
                save_path=save_dir / f'sample_{batch_idx:04d}.png'
            )

            count += 1

        print(f"\n✓ 可视化完成，保存到: {save_dir}")

    def denormalize_image(self, tensor):
        """反归一化"""
        # ImageNet mean/std
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)

        image = tensor * std + mean
        image = image.clamp(0, 1)
        image = image.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)

        return image

    def plot_comparison(self, image, label, pred, save_path):
        """绘制对比图"""
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # 原图
        axes[0].imshow(image)
        axes[0].set_title('Input Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # Ground Truth
        label_vis = self.mask_to_color(label)
        axes[1].imshow(label_vis)
        axes[1].set_title('Ground Truth', fontsize=14, fontweight='bold')
        axes[1].axis('off')

        # Prediction
        pred_vis = self.mask_to_color(pred)
        axes[2].imshow(pred_vis)
        axes[2].set_title('Prediction', fontsize=14, fontweight='bold')
        axes[2].axis('off')

        # Overlay
        overlay = self.create_overlay(image, pred)
        axes[3].imshow(overlay)
        axes[3].set_title('Overlay', fontsize=14, fontweight='bold')
        axes[3].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def mask_to_color(self, mask):
        """将mask转换为彩色图"""
        # 创建RGB图像
        color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)

        # 背景：黑色
        color_mask[mask == 0] = [0, 0, 0]

        # 前景：绿色
        color_mask[mask == 1] = [0, 255, 0]

        # Ignore区域：灰色
        color_mask[mask == 255] = [128, 128, 128]

        return color_mask

    def create_overlay(self, image, mask, alpha=0.5):
        """创建叠加图"""
        overlay = image.copy()
        mask_color = self.mask_to_color(mask) / 255.0

        # 只在前景区域叠加
        foreground = mask == 1
        overlay[foreground] = (1 - alpha) * overlay[foreground] + alpha * mask_color[foreground]

        return overlay

    @torch.no_grad()
    def compute_error_map(self, num_samples=5, save_dir='outputs/error_analysis'):
        """计算错误分析图"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 60)
        print(f"生成错误分析图（{num_samples}张）...")
        print("=" * 60)

        count = 0
        pbar = tqdm(self.test_loader, total=num_samples)

        for batch_idx, batch in enumerate(pbar):
            if count >= num_samples:
                break

            images = batch['image'].to(self.device)
            labels = batch['label'].cpu().numpy()[0]

            outputs = self.model(images)
            pred = torch.sigmoid(outputs['pred']) > 0.5
            pred = pred.squeeze().cpu().numpy()

            image = self.denormalize_image(images[0])

            # 错误分析
            self.plot_error_analysis(
                image, labels, pred,
                save_path=save_dir / f'error_{batch_idx:04d}.png'
            )

            count += 1

        print(f"\n✓ 错误分析完成，保存到: {save_dir}")

    def plot_error_analysis(self, image, label, pred, save_path):
        """绘制错误分析图"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 二值化标签
        label_binary = (label > 0).astype(np.uint8)
        mask = (label != 255)

        # 原图
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Input Image', fontsize=12)
        axes[0, 0].axis('off')

        # Ground Truth
        axes[0, 1].imshow(label_binary, cmap='gray')
        axes[0, 1].set_title('Ground Truth', fontsize=12)
        axes[0, 1].axis('off')

        # Prediction
        axes[0, 2].imshow(pred, cmap='gray')
        axes[0, 2].set_title('Prediction', fontsize=12)
        axes[0, 2].axis('off')

        # True Positive (白色)
        tp = (pred == 1) & (label_binary == 1) & mask
        axes[1, 0].imshow(tp, cmap='Greens')
        axes[1, 0].set_title(f'True Positive: {tp.sum()} px', fontsize=12)
        axes[1, 0].axis('off')

        # False Positive (红色)
        fp = (pred == 1) & (label_binary == 0) & mask
        axes[1, 1].imshow(fp, cmap='Reds')
        axes[1, 1].set_title(f'False Positive: {fp.sum()} px', fontsize=12)
        axes[1, 1].axis('off')

        # False Negative (蓝色)
        fn = (pred == 0) & (label_binary == 1) & mask
        axes[1, 2].imshow(fn, cmap='Blues')
        axes[1, 2].set_title(f'False Negative: {fn.sum()} px', fontsize=12)
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='可视化SegFormer蒸馏模型预测结果')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='checkpoint路径')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='可视化样本数量')
    parser.add_argument('--save-dir', type=str, default='outputs/visualizations',
                        help='保存目录')
    parser.add_argument('--error-analysis', action='store_true',
                        help='是否生成错误分析图')
    args = parser.parse_args()

    # 创建可视化器
    visualizer = Visualizer(args.checkpoint)

    # 生成可视化
    visualizer.visualize_samples(
        num_samples=args.num_samples,
        save_dir=args.save_dir
    )

    # 错误分析（可选）
    if args.error_analysis:
        visualizer.compute_error_map(
            num_samples=min(args.num_samples, 5),
            save_dir=Path(args.save_dir).parent / 'error_analysis'
        )

    print("\n✓ 可视化完成！")


if __name__ == '__main__':
    main()