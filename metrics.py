"""
评估指标计算
包含：mIoU, F1-Score, OA, Params, FLOPs, FPS
"""
import torch
import torch.nn as nn
import numpy as np
import time
from thop import profile, clever_format


class Evaluator:
    """评估指标计算器"""

    def __init__(self, num_classes):
        """
        Args:
            num_classes: 类别数量
        """
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """重置统计"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self, pred, target):
        """
        更新混淆矩阵

        Args:
            pred: (B, H, W) or (B, C, H, W) - 预测
            target: (B, H, W) - ground truth
        """
        # 如果pred是logits，转为类别索引
        if len(pred.shape) == 4:
            pred = torch.argmax(pred, dim=1)  # (B, H, W)

        # 转为numpy
        pred = pred.cpu().numpy().flatten()
        target = target.cpu().numpy().flatten()

        # 过滤ignore_index（如果有）
        mask = (target >= 0) & (target < self.num_classes)
        pred = pred[mask]
        target = target[mask]

        # 更新混淆矩阵
        for p, t in zip(pred, target):
            self.confusion_matrix[int(t), int(p)] += 1

    def get_miou(self):
        """
        计算 mean Intersection over Union

        Returns:
            miou: float
            iou_per_class: list
        """
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusion_matrix)
        union = (
                self.confusion_matrix.sum(axis=1) +  # GT
                self.confusion_matrix.sum(axis=0) -  # Pred
                intersection  # 减去重复计算的TP
        )

        iou_per_class = intersection / (union + 1e-10)
        miou = np.nanmean(iou_per_class)

        return miou, iou_per_class

    def get_f1_score(self):
        """
        计算 F1-Score
        F1 = 2 * (Precision * Recall) / (Precision + Recall)

        Returns:
            f1: float
            f1_per_class: list
        """
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - tp
        fn = self.confusion_matrix.sum(axis=1) - tp

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)

        f1_per_class = 2 * (precision * recall) / (precision + recall + 1e-10)
        f1 = np.nanmean(f1_per_class)

        return f1, f1_per_class

    def get_overall_accuracy(self):
        """
        计算 Overall Accuracy (像素级准确率)
        OA = correct_pixels / total_pixels

        Returns:
            oa: float
        """
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        oa = correct / (total + 1e-10)

        return oa

    def get_all_metrics(self):
        """
        获取所有指标

        Returns:
            dict: 包含所有指标
        """
        miou, iou_per_class = self.get_miou()
        f1, f1_per_class = self.get_f1_score()
        oa = self.get_overall_accuracy()

        return {
            'mIoU': miou * 100,  # 转为百分比
            'F1': f1 * 100,
            'OA': oa * 100,
            'IoU_per_class': iou_per_class * 100,
            'F1_per_class': f1_per_class * 100
        }


def count_parameters(model):
    """
    计算模型参数量

    Args:
        model: PyTorch模型

    Returns:
        params: 参数量（M）
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total_params / 1e6,  # 转为M
        'trainable': trainable_params / 1e6
    }


def count_flops(model, input_size=(1, 3, 1024, 1024), device='cuda'):
    """
    计算模型FLOPs

    Args:
        model: PyTorch模型
        input_size: 输入尺寸
        device: 设备

    Returns:
        flops: 计算量（G）
        params: 参数量（M）
    """
    model = model.to(device)
    dummy_input = torch.randn(*input_size).to(device)

    try:
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")

        # 转换为数值（G和M）
        flops_g = float(flops.replace('G', '')) if 'G' in flops else float(flops.replace('M', '')) / 1000
        params_m = float(params.replace('M', '')) if 'M' in params else float(params.replace('K', '')) / 1000

        return flops_g, params_m
    except Exception as e:
        print(f"FLOPs计算失败: {e}")
        return None, None


def measure_fps(model, input_size=(1, 3, 1024, 1024), device='cuda', num_runs=100, warmup=10):
    """
    测量模型推理速度（FPS）

    Args:
        model: PyTorch模型
        input_size: 输入尺寸
        device: 设备
        num_runs: 测试次数
        warmup: 预热次数

    Returns:
        fps: 每秒帧数
        latency: 延迟（ms）
    """
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(*input_size).to(device)

    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    # 测速
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)

    torch.cuda.synchronize()
    end_time = time.time()

    # 计算
    total_time = end_time - start_time
    latency_ms = (total_time / num_runs) * 1000
    fps = num_runs / total_time

    return fps, latency_ms


def evaluate_model_complete(model, dataloader, num_classes, device='cuda',
                            measure_speed=True, input_size=(1, 3, 1024, 1024)):
    """
    完整评估模型（6个指标）

    Args:
        model: PyTorch模型
        dataloader: 测试集dataloader
        num_classes: 类别数
        device: 设备
        measure_speed: 是否测速
        input_size: 输入尺寸（用于FLOPs和FPS计算）

    Returns:
        results: dict包含所有指标
    """
    model = model.to(device)
    model.eval()

    # 创建evaluator
    evaluator = Evaluator(num_classes)

    print("评估中...")
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            # 前向
            if isinstance(model, dict):
                # 如果是dict（包含pred和feat）
                outputs = model
                pred = outputs['pred']
            else:
                outputs = model(images)
                if isinstance(outputs, dict):
                    pred = outputs['pred']
                else:
                    pred = outputs

            # 更新指标
            evaluator.update(pred, labels)

    # 获取精度指标
    metrics = evaluator.get_all_metrics()

    # 计算参数量
    params_info = count_parameters(model)
    metrics['Params'] = params_info['total']

    # 计算FLOPs
    if measure_speed:
        flops, _ = count_flops(model, input_size, device)
        if flops is not None:
            metrics['FLOPs'] = flops

        # 测速
        fps, latency = measure_fps(model, input_size, device)
        metrics['FPS'] = fps
        metrics['Latency_ms'] = latency

    return metrics


def print_metrics(metrics, model_name='Model'):
    """
    打印指标

    Args:
        metrics: 指标dict
        model_name: 模型名称
    """
    print("=" * 60)
    print(f"{model_name} 评估结果")
    print("=" * 60)

    # 精度指标
    print("\n精度指标:")
    print(f"  mIoU:  {metrics['mIoU']:.2f}%")
    print(f"  F1:    {metrics['F1']:.2f}%")
    print(f"  OA:    {metrics['OA']:.2f}%")

    # 效率指标
    print("\n效率指标:")
    print(f"  Params: {metrics['Params']:.2f}M")
    if 'FLOPs' in metrics:
        print(f"  FLOPs:  {metrics['FLOPs']:.2f}G")
    if 'FPS' in metrics:
        print(f"  FPS:    {metrics['FPS']:.1f}")
        print(f"  Latency: {metrics['Latency_ms']:.2f}ms")

    # 每类IoU（可选）
    if 'IoU_per_class' in metrics:
        print("\n每类IoU:")
        for i, iou in enumerate(metrics['IoU_per_class']):
            print(f"  Class {i}: {iou:.2f}%")

    print("=" * 60)


def save_metrics_to_csv(metrics_dict, save_path):
    """
    保存多个模型的指标到CSV

    Args:
        metrics_dict: {model_name: metrics}
        save_path: 保存路径
    """
    import pandas as pd

    # 提取主要指标
    data = []
    for model_name, metrics in metrics_dict.items():
        row = {
            'Method': model_name,
            'mIoU': f"{metrics['mIoU']:.2f}",
            'F1': f"{metrics['F1']:.2f}",
            'OA': f"{metrics['OA']:.2f}",
            'Params': f"{metrics['Params']:.2f}M",
        }

        if 'FLOPs' in metrics:
            row['FLOPs'] = f"{metrics['FLOPs']:.2f}G"
        if 'FPS' in metrics:
            row['FPS'] = f"{metrics['FPS']:.1f}"

        data.append(row)

    # 保存
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"\n结果已保存到: {save_path}")


if __name__ == '__main__':
    """测试评估指标"""
    print("=" * 60)
    print("测试评估指标")
    print("=" * 60)

    # 创建虚拟数据
    num_classes = 6
    pred = torch.randint(0, num_classes, (4, 512, 512))
    target = torch.randint(0, num_classes, (4, 512, 512))

    # 测试Evaluator
    evaluator = Evaluator(num_classes)
    evaluator.update(pred, target)

    metrics = evaluator.get_all_metrics()
    print("\n测试指标:")
    print(f"  mIoU: {metrics['mIoU']:.2f}%")
    print(f"  F1:   {metrics['F1']:.2f}%")
    print(f"  OA:   {metrics['OA']:.2f}%")

    print("\n" + "=" * 60)
    print("✓ 评估指标测试通过")
    print("=" * 60)