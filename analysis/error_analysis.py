"""
深度错误分析工具
找出模型的弱点，指导后续优化
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from scipy import ndimage

from config import cfg
from dataset import build_dataloader
from models.segformer_tricks import build_segformer_tricks


class ErrorAnalyzer:
    """错误分析器"""

    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device)
        self.model = self.load_model(checkpoint_path)
        self.test_loader = build_dataloader('test', batch_size=1, shuffle=False)

        # 用于存储分析结果
        self.error_stats = {
            'by_size': [],
            'by_boundary': [],
            'by_density': [],
            'failure_cases': []
        }

    def load_model(self, checkpoint_path):
        """加载模型"""
        model = build_segformer_tricks(variant='b1', pretrained=False)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        return model

    @torch.no_grad()
    def analyze_all(self, save_dir='outputs/analysis'):
        """完整分析流程"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print("开始深度错误分析...")
        print("=" * 60)

        all_results = []

        for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="分析中")):
            images = batch['image'].to(self.device)
            labels = batch['label'].cpu().numpy()[0]
            img_id = batch['img_id'][0]

            # 预测
            outputs = self.model(images)
            pred = torch.sigmoid(outputs['pred']) > 0.5
            pred = pred.squeeze().cpu().numpy()

            # 各种分析
            result = {
                'img_id': img_id,
                'size_analysis': self.analyze_by_building_size(pred, labels),
                'boundary_analysis': self.analyze_by_boundary_distance(pred, labels),
                'density_analysis': self.analyze_by_building_density(pred, labels),
                'error_score': self.compute_error_score(pred, labels)
            }

            all_results.append(result)

        # 汇总统计
        self.summarize_results(all_results, save_dir)

        # 找出最差样本
        self.find_failure_cases(all_results, save_dir, top_k=50)

        print(f"\n✓ 分析完成！结果保存在: {save_dir}")

    def analyze_by_building_size(self, pred, label):
        """
        按建筑物大小分析

        发现哪种尺寸的建筑物最难分割
        """
        # 提取连通域
        label_binary = (label > 0).astype(np.uint8)
        pred_binary = pred.astype(np.uint8)

        # 标记连通域
        num_labels, labels_cc = cv2.connectedComponents(label_binary)

        size_stats = {
            'small': {'tp': 0, 'fp': 0, 'fn': 0},  # <100m² (约<400px)
            'medium': {'tp': 0, 'fp': 0, 'fn': 0},  # 100-1000m²
            'large': {'tp': 0, 'fp': 0, 'fn': 0}  # >1000m²
        }

        for i in range(1, num_labels):
            # 获取当前建筑物
            mask = (labels_cc == i)
            size = mask.sum()

            # 分类尺寸
            if size < 400:
                category = 'small'
            elif size < 4000:
                category = 'medium'
            else:
                category = 'large'

            # 计算IoU
            pred_masked = pred_binary[mask]
            label_masked = label_binary[mask]

            intersection = (pred_masked & label_masked).sum()
            union = (pred_masked | label_masked).sum()
            iou = intersection / (union + 1e-10)

            # 统计TP/FP/FN
            if iou > 0.5:
                size_stats[category]['tp'] += 1
            elif pred_masked.sum() > 0:
                size_stats[category]['fp'] += 1
            else:
                size_stats[category]['fn'] += 1

        # 计算每个尺寸的精度
        for cat in size_stats:
            tp = size_stats[cat]['tp']
            fp = size_stats[cat]['fp']
            fn = size_stats[cat]['fn']

            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)

            size_stats[cat].update({
                'precision': precision,
                'recall': recall,
                'f1': f1
            })

        return size_stats

    def analyze_by_boundary_distance(self, pred, label):
        """
        按距离边界的距离分析

        发现边界附近的精度如何
        """
        # 提取边界
        label_binary = (label > 0).astype(np.uint8)

        # 使用形态学操作提取边界
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(label_binary, kernel, iterations=1)
        boundary = label_binary - eroded

        # 计算到边界的距离
        dist_transform = ndimage.distance_transform_edt(1 - boundary)

        # 分区域统计
        distance_stats = {}
        ranges = [(0, 2), (2, 5), (5, 10), (10, float('inf'))]
        range_names = ['0-2px', '2-5px', '5-10px', '>10px']

        for (d_min, d_max), name in zip(ranges, range_names):
            mask = (dist_transform >= d_min) & (dist_transform < d_max)

            if mask.sum() == 0:
                continue

            pred_masked = pred[mask]
            label_masked = (label[mask] > 0)

            # 计算精度
            correct = (pred_masked == label_masked).sum()
            total = mask.sum()
            accuracy = correct / total

            # IoU
            intersection = (pred_masked & label_masked).sum()
            union = (pred_masked | label_masked).sum()
            iou = intersection / (union + 1e-10)

            distance_stats[name] = {
                'accuracy': float(accuracy),
                'iou': float(iou),
                'pixel_count': int(total)
            }

        return distance_stats

    def analyze_by_building_density(self, pred, label):
        """
        按建筑物密度分析

        发现密集区域 vs 稀疏区域的表现
        """
        # 计算局部密度（使用滑动窗口）
        label_binary = (label > 0).astype(np.float32)

        # 50x50窗口计算密度
        kernel = np.ones((50, 50)) / (50 * 50)
        density_map = cv2.filter2D(label_binary, -1, kernel)

        # 分区域
        density_stats = {}
        ranges = [(0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.0)]
        range_names = ['sparse', 'low', 'medium', 'dense']

        for (d_min, d_max), name in zip(ranges, range_names):
            mask = (density_map >= d_min) & (density_map < d_max)

            if mask.sum() == 0:
                continue

            pred_masked = pred[mask]
            label_masked = (label[mask] > 0)

            # 计算IoU
            intersection = (pred_masked & label_masked).sum()
            union = (pred_masked | label_masked).sum()
            iou = intersection / (union + 1e-10)

            density_stats[name] = {
                'iou': float(iou),
                'pixel_count': int(mask.sum())
            }

        return density_stats

    def compute_error_score(self, pred, label):
        """
        计算错误分数（用于排序最差样本）

        返回越大表示错误越严重
        """
        mask = (label != 255)
        label_binary = (label > 0)[mask]
        pred_masked = pred[mask]

        # 计算IoU
        intersection = (pred_masked & label_binary).sum()
        union = (pred_masked | label_binary).sum()
        iou = intersection / (union + 1e-10)

        # 错误分数 = 1 - IoU
        error_score = 1 - iou

        return float(error_score)

    def summarize_results(self, all_results, save_dir):
        """汇总所有结果，生成统计报告"""
        print("\n" + "=" * 60)
        print("统计分析报告")
        print("=" * 60)

        # 1. 按建筑物大小汇总
        size_summary = {'small': [], 'medium': [], 'large': []}
        for result in all_results:
            for size, stats in result['size_analysis'].items():
                size_summary[size].append(stats['f1'])

        print("\n【按建筑物大小分析】")
        for size, f1_list in size_summary.items():
            if len(f1_list) > 0:
                avg_f1 = np.mean(f1_list)
                print(f"  {size:8s}: F1 = {avg_f1:.4f}")

        # 2. 按边界距离汇总
        boundary_summary = {}
        for result in all_results:
            for dist, stats in result['boundary_analysis'].items():
                if dist not in boundary_summary:
                    boundary_summary[dist] = []
                boundary_summary[dist].append(stats['iou'])

        print("\n【按边界距离分析】")
        for dist in ['0-2px', '2-5px', '5-10px', '>10px']:
            if dist in boundary_summary and len(boundary_summary[dist]) > 0:
                avg_iou = np.mean(boundary_summary[dist])
                print(f"  {dist:8s}: IoU = {avg_iou:.4f}")

        # 3. 按密度汇总
        density_summary = {}
        for result in all_results:
            for dens, stats in result['density_analysis'].items():
                if dens not in density_summary:
                    density_summary[dens] = []
                density_summary[dens].append(stats['iou'])

        print("\n【按建筑物密度分析】")
        for dens in ['sparse', 'low', 'medium', 'dense']:
            if dens in density_summary and len(density_summary[dens]) > 0:
                avg_iou = np.mean(density_summary[dens])
                print(f"  {dens:8s}: IoU = {avg_iou:.4f}")

        print("=" * 60)

        # 保存详细结果
        df = pd.DataFrame([
            {
                'img_id': r['img_id'],
                'error_score': r['error_score'],
                'small_f1': r['size_analysis'].get('small', {}).get('f1', 0),
                'medium_f1': r['size_analysis'].get('medium', {}).get('f1', 0),
                'large_f1': r['size_analysis'].get('large', {}).get('f1', 0),
            }
            for r in all_results
        ])

        df.to_csv(save_dir / 'detailed_analysis.csv', index=False)
        print(f"\n✓ 详细结果保存至: {save_dir / 'detailed_analysis.csv'}")

    def find_failure_cases(self, all_results, save_dir, top_k=50):
        """找出最差的K个样本"""
        # 按error_score排序
        sorted_results = sorted(all_results, key=lambda x: x['error_score'], reverse=True)

        failure_cases = sorted_results[:top_k]

        print(f"\n【最差的{top_k}个样本】")
        for i, case in enumerate(failure_cases[:10]):  # 只打印前10个
            print(f"  {i + 1}. {case['img_id']}: error_score = {case['error_score']:.4f}")

        # 保存失败案例列表
        with open(save_dir / 'failure_cases.txt', 'w') as f:
            for case in failure_cases:
                f.write(f"{case['img_id']}\t{case['error_score']:.4f}\n")

        print(f"\n✓ 失败案例保存至: {save_dir / 'failure_cases.txt'}")


def main():
    """运行错误分析"""
    analyzer = ErrorAnalyzer(
        checkpoint_path='outputs/tricks/checkpoints/best.pth'
    )

    analyzer.analyze_all(save_dir='outputs/analysis')


if __name__ == '__main__':
    main()