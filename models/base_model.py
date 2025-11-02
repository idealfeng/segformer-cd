"""
基础模型类 - 统一接口
为Phase 3的6个对比模型提供统一的forward和evaluate接口
"""
import sys
from pathlib import Path

# ✅ 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent  # models/ → project/
sys.path.insert(0, str(project_root))
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import time
from pathlib import Path
import json

from config import cfg

IGNORE_LABEL = 255


class BaseModel(nn.Module):
    """
    所有对比模型的基类

    子类必须实现：
    - __init__: 初始化模型
    - forward: 前向传播
    - get_model_info: 返回模型信息
    """

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        """
        前向传播（子类必须实现）

        Args:
            x: (B, 3, H, W) 输入图像

        Returns:
            pred: (B, 1, H, W) 二分类logits
        """
        raise NotImplementedError("子类必须实现forward方法")

    def get_model_info(self):
        """
        获取模型信息（子类必须实现）

        Returns:
            dict: {
                'name': str,
                'params': float (M),
                'flops': float (G),
                'input_size': int
            }
        """
        raise NotImplementedError("子类必须实现get_model_info方法")

    @torch.no_grad()
    def predict(self, image):
        """
        单张图像预测

        Args:
            image: (3, H, W) Tensor or (H, W, 3) numpy

        Returns:
            pred: (H, W) numpy array, {0, 1}
        """
        # 预处理
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float()

        if image.ndim == 3:
            image = image.unsqueeze(0)  # (1, 3, H, W)

        image = image.to(self.device)

        # 前向传播
        logits = self.forward(image)  # (1, 1, H, W)

        # 阈值化
        pred = (torch.sigmoid(logits) > 0.5).squeeze().cpu().numpy()

        return pred.astype(np.uint8)

    @torch.no_grad()
    def evaluate(self, dataloader, compute_fps=False, num_warmup=10, num_measure=100):
        """
        统一评估接口

        Args:
            dataloader: PyTorch DataLoader
            compute_fps: 是否计算FPS
            num_warmup: warm-up批次数
            num_measure: 测量批次数

        Returns:
            metrics: dict with all evaluation metrics
        """
        self.eval()

        all_preds = []
        all_labels = []
        latency_list = []

        warmup_done = False
        measure_count = 0

        print(f"\n{'=' * 60}")
        print(f"评估模型: {self.model_name}")
        print(f"{'=' * 60}")

        pbar = tqdm(dataloader, desc=f"Evaluating {self.model_name}")

        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            labels = batch['label'].cpu().numpy()

            # Warm-up
            if compute_fps and batch_idx < num_warmup:
                _ = self.forward(images)
                pbar.set_postfix({'status': f'Warming up... {batch_idx + 1}/{num_warmup}'})
                continue

            if compute_fps and not warmup_done:
                warmup_done = True
                torch.cuda.synchronize()

            # 计时
            if compute_fps and measure_count < num_measure:
                torch.cuda.synchronize()
                start_time = time.time()

            # 前向传播
            logits = self.forward(images)

            # 计时结束
            if compute_fps and measure_count < num_measure:
                torch.cuda.synchronize()
                latency_list.append(time.time() - start_time)
                measure_count += 1

            # 阈值化
            pred = (torch.sigmoid(logits) > 0.5).squeeze(1).cpu().numpy()

            # 批量处理
            for i in range(pred.shape[0]):
                label_i = labels[i].flatten()
                pred_i = pred[i].flatten()

                # 过滤ignore
                mask = (label_i != IGNORE_LABEL)

                all_labels.extend(label_i[mask].tolist())
                all_preds.extend(pred_i[mask].tolist())

        # 计算指标
        all_preds = np.array(all_preds, dtype=np.uint8)
        all_labels = np.array(all_labels, dtype=np.uint8)

        metrics = self._compute_metrics(all_preds, all_labels)

        # FPS统计
        if compute_fps and len(latency_list) > 0:
            latency_sorted = sorted(latency_list)
            trim = len(latency_sorted) // 10
            latency_trimmed = latency_sorted[trim:-trim] if trim > 0 else latency_sorted

            avg_latency = np.mean(latency_trimmed)
            std_latency = np.std(latency_trimmed)
            batch_size = dataloader.batch_size
            fps = batch_size / avg_latency

            metrics['FPS'] = float(fps)
            metrics['Latency_ms'] = float(avg_latency * 1000)
            metrics['Latency_std_ms'] = float(std_latency * 1000)
            metrics['Throughput'] = float(fps)
            metrics['Batch_size'] = batch_size

        # 添加模型信息
        model_info = self.get_model_info()
        metrics.update(model_info)

        return metrics

    def _compute_metrics(self, preds, labels):
        """计算评估指标"""
        cm = confusion_matrix(labels, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='binary', pos_label=1
        )

        iou = tp / (tp + fp + fn + 1e-10)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        dice = 2 * tp / (2 * tp + fp + fn + 1e-10)

        iou_bg = tn / (tn + fp + fn + 1e-10)
        iou_fg = tp / (tp + fp + fn + 1e-10)
        miou = (iou_bg + iou_fg) / 2

        return {
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

    def print_metrics(self, metrics):
        """打印指标"""
        print(f"\n{'=' * 60}")
        print(f"模型: {self.model_name}")
        print(f"{'=' * 60}")

        # 模型信息
        if 'params' in metrics:
            print(f"参数量:                        {metrics['params']:.2f}M")
        if 'flops' in metrics:
            print(f"FLOPs:                         {metrics['flops']:.2f}G")

        # 精度指标
        print(f"\n精度指标:")
        print(f"  mIoU:                        {metrics['mIoU']:.4f}")
        print(f"  IoU (前景):                  {metrics['IoU']:.4f}")
        print(f"  IoU (背景):                  {metrics['IoU_bg']:.4f}")
        print(f"  F1-Score:                    {metrics['F1']:.4f}")
        print(f"  Dice:                        {metrics['Dice']:.4f}")
        print(f"  Precision:                   {metrics['Precision']:.4f}")
        print(f"  Recall:                      {metrics['Recall']:.4f}")
        print(f"  Accuracy:                    {metrics['Accuracy']:.4f}")

        # 速度指标
        if 'FPS' in metrics:
            print(f"\n速度指标 (Batch={metrics.get('Batch_size', 'N/A')}):")
            print(f"  FPS:                         {metrics['FPS']:.2f} 样本/秒")
            print(f"  Latency:                     {metrics['Latency_ms']:.2f} ± {metrics['Latency_std_ms']:.2f} ms")

        print(f"{'=' * 60}\n")

    def save_metrics(self, metrics, save_path):
        """保存指标"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        print(f"✓ 指标已保存: {save_path}")


# ============================================================
#                   示例：包装你的SegFormer
# ============================================================

class SegFormerWrapper(BaseModel):
    """SegFormer蒸馏模型的Wrapper"""

    def __init__(self, checkpoint_path):
        super().__init__(model_name='SegFormer-B1 (Ours)')

        from models.segformer import build_segformer_distillation

        self.model = build_segformer_distillation(
            variant='b1',
            pretrained=False
        )

        # 加载权重
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"✓ 加载模型: {checkpoint_path}")

    def forward(self, x):
        """前向传播"""
        outputs = self.model(x)
        return outputs['pred']  # (B, 1, H, W)

    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())

        try:
            from thop import profile
            # ✅ 用512×512避免OOM
            dummy = torch.randn(1, 3, 512, 512).to(self.device)
            flops, _ = profile(self, inputs=(dummy,), verbose=False)
            # 换算到1024×1024
            flops = flops * (cfg.IMAGE_SIZE / 512) ** 2
        except Exception as e:
            print(f"FLOPs计算失败: {e}")
            flops = 0

        return {
            'name': self.model_name,
            'params': total_params / 1e6,
            'flops': flops / 1e9,
            'input_size': cfg.IMAGE_SIZE
        }


if __name__ == '__main__':
    """测试BaseModel"""
    print("=" * 60)
    print("测试统一评估接口")
    print("=" * 60)

    # 示例：评估你的SegFormer
    from dataset import build_dataloader

    model = SegFormerWrapper('outputs/checkpoints/best.pth')
    test_loader = build_dataloader('test', batch_size=4, shuffle=False)

    metrics = model.evaluate(
        test_loader,
        compute_fps=True,
        num_warmup=10,
        num_measure=50
    )

    model.print_metrics(metrics)
    model.save_metrics(metrics, 'outputs/eval_results/segformer_unified.json')

    print("\n✓ 统一接口测试通过！")