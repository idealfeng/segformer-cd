# models/sam_wrapper.py
"""
SAM Teacher模型的Wrapper
用于Phase 3对比实验
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor
import time
from tqdm import tqdm
from models.base_model import BaseModel
from config import cfg


class SAMWrapper(BaseModel):
    """
    SAM (Segment Anything Model) Wrapper

    特点：
    - 使用全图Box prompt (0,0,W,H)
    - 输出二值化mask（前景/背景）
    - 统一接口，便于Phase 3对比
    """

    def __init__(self, checkpoint_path, model_type='vit_h'):
        super().__init__(model_name='SAM-ViT-H (Teacher)')

        print(f"加载SAM模型...")
        print(f"  模型: {model_type}")
        print(f"  权重: {checkpoint_path}")

        # 加载SAM
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam = self.sam.to(self.device)
        self.sam.eval()

        # 创建Predictor
        self.predictor = SamPredictor(self.sam)

        print("✓ SAM加载完成")

    # models/sam_wrapper.py
    # 修改forward函数

    def forward(self, x, labels=None):
        """
        前向传播（GT-guided版本）

        Args:
            x: (B, 3, H, W) 输入图像
            labels: (B, H, W) GT标签（可选，用于生成box）

        Returns:
            pred: (B, 1, H, W) 预测logits
        """
        B, C, H, W = x.shape

        # SAM预处理
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_denorm = x * std + mean
        x_rgb = (x_denorm * 255).clamp(0, 255).byte()

        outputs = []
        for i in range(B):
            img = x_rgb[i].permute(1, 2, 0).cpu().numpy()

            # ========== 关键修改 ==========
            # 使用GT生成更精确的box
            if labels is not None:
                label = labels[i].cpu().numpy()
                # 找出前景像素（label==1）的位置
                fg_coords = np.where(label == 1)

                if len(fg_coords[0]) > 0:
                    # 计算前景的bounding box
                    y_min, y_max = fg_coords[0].min(), fg_coords[0].max()
                    x_min, x_max = fg_coords[1].min(), fg_coords[1].max()

                    # 添加一些padding（5%）
                    pad_y = int((y_max - y_min) * 0.05)
                    pad_x = int((x_max - x_min) * 0.05)

                    y_min = max(0, y_min - pad_y)
                    y_max = min(H, y_max + pad_y)
                    x_min = max(0, x_min - pad_x)
                    x_max = min(W, x_max + pad_x)

                    box = np.array([x_min, y_min, x_max, y_max])
                else:
                    # 如果没有前景，使用全图box
                    box = np.array([0, 0, W, H])
            else:
                # 如果没有GT，使用全图box
                box = np.array([0, 0, W, H])
            # ========== 修改结束 ==========

            # SAM推理
            self.predictor.set_image(img)
            masks, scores, logits = self.predictor.predict(
                box=box,
                multimask_output=False
            )

            mask = torch.from_numpy(masks[0]).float()
            outputs.append(mask)

        outputs = torch.stack(outputs, dim=0).unsqueeze(1).to(x.device)

        # 转为logits
        logits = torch.where(outputs > 0.5,
                             torch.tensor(10.0, device=x.device),
                             torch.tensor(-10.0, device=x.device))

        return logits

    @torch.no_grad()
    def evaluate(self, dataloader, compute_fps=False, num_warmup=10, num_measure=100):
        """
        SAM专用evaluate - 强制传入labels
        覆盖BaseModel的方法
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

        from tqdm import tqdm
        pbar = tqdm(dataloader, desc=f"Evaluating {self.model_name}")

        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            labels_tensor = batch['label'].to(self.device)  # ← GT用于生成box
            labels = labels_tensor.cpu().numpy()

            # Warm-up
            if compute_fps and batch_idx < num_warmup:
                _ = self.forward(images, labels=labels_tensor)  # ← 强制传入
                pbar.set_postfix({'status': f'Warming up... {batch_idx + 1}/{num_warmup}'})
                continue

            if compute_fps and not warmup_done:
                warmup_done = True
                torch.cuda.synchronize()

            # 计时
            if compute_fps and measure_count < num_measure:
                torch.cuda.synchronize()
                start_time = time.time()

            # 前向传播（强制传入labels）
            logits = self.forward(images, labels=labels_tensor)  # ← 关键！

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
                mask = (label_i != 255)

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

    def get_model_info(self):
        """获取模型信息"""
        # SAM参数量
        total_params = sum(p.numel() for p in self.sam.parameters())

        # FLOPs（SAM非常大，不在线计算，直接用论文数据）
        # 论文: SAM-ViT-H ~400G FLOPs (1024x1024)
        flops = 400.0  # G

        return {
            'name': self.model_name,
            'params': total_params / 1e6,  # ~632M
            'flops': flops,
            'input_size': 1024
        }


if __name__ == '__main__':
    """测试SAM Wrapper"""
    print("=" * 60)
    print("测试SAM Wrapper")
    print("=" * 60)

    # 加载模型
    checkpoint_path = 'pretrained_weights/sam_vit_h_4b8939.pth'  # 你的SAM权重路径

    if not Path(checkpoint_path).exists():
        print(f"❌ SAM权重不存在: {checkpoint_path}")
        print("\n请从以下地址下载:")
        print("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        exit(1)

    model = SAMWrapper(checkpoint_path)

    # 测试前向传播
    print("\n测试前向传播...")
    dummy_input = torch.randn(1, 3, 1024, 1024).to(model.device)

    import time

    start = time.time()
    output = model.forward(dummy_input)
    elapsed = time.time() - start

    print(f"输入: {dummy_input.shape}")
    print(f"输出: {output.shape}")
    print(f"推理时间: {elapsed:.2f}秒")

    # 模型信息
    info = model.get_model_info()
    print(f"\n模型信息:")
    print(f"  参数量: {info['params']:.2f}M")
    print(f"  FLOPs: {info['flops']:.2f}G")

    print("\n✓ SAM Wrapper测试通过")