"""
SegFormer变化检测训练脚本 - 统一数据集版本

支持数据集: LEVIR-CD, S2Looking, WHUCD

运行示例:
    # LEVIR-CD
    python train_unified.py --dataset levir --data-root "data/LEVIR-CD" --exp-name levir_only --epochs 50

    # S2Looking
    python train_unified.py --dataset s2looking --data-root "data/S2Looking" --exp-name s2looking_only --epochs 50

    # WHUCD
    python train_unified.py --dataset whucd --data-root "data/Building change detection dataset_add" --exp-name whucd_only --epochs 50
"""
import os

os.environ['ALBUMENTATIONS_CHECK_VERSION'] = 'False'

import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import json
import random
import numpy as np
from datetime import datetime

from config import cfg
from unified_dataset import create_dataloaders_unified  # ← 修改这里
from models.segformer import build_model
from losses.losses import build_criterion

import logging

logging.getLogger("transformers").setLevel(logging.ERROR)


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_metrics(pred, target, threshold=0.5):
    """计算评估指标（二值版本）"""
    # 获取预测类别：sigmoid + threshold
    pred_prob = torch.sigmoid(pred.squeeze(1))  # (B, H, W)
    pred_class = (pred_prob > threshold).long()  # (B, H, W)

    # 计算TP, FP, FN, TN (只针对前景类/变化类)
    tp = ((pred_class == 1) & (target == 1)).sum().float()
    fp = ((pred_class == 1) & (target == 0)).sum().float()
    fn = ((pred_class == 0) & (target == 1)).sum().float()
    tn = ((pred_class == 0) & (target == 0)).sum().float()

    # 精度指标
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    oa = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
        'iou': iou.item(),
        'oa': oa.item()
    }


class Trainer:
    """变化检测训练器"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(cfg.DEVICE)

        # ← 新增：保存数据集信息
        self.dataset_name = args.dataset

        # 设置随机种子
        set_seed(cfg.SEED)

        # 创建输出目录（使用exp_name）
        self.setup_dirs()

        # 初始化组件
        print("=" * 60)
        print(f"Initializing Training for {args.dataset.upper()}...")
        print("=" * 60)

        self.model = self.build_model()
        self.train_loader, self.val_loader, self.test_loader = self.build_dataloaders()
        self.criterion = build_criterion().to(self.device)
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()

        # AMP
        self.scaler = GradScaler() if cfg.USE_AMP else None
        self.use_amp = cfg.USE_AMP

        # TensorBoard
        if cfg.USE_TENSORBOARD:
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
        else:
            self.writer = None

        # 训练状态
        self.start_epoch = 0
        self.best_metric = 0.0
        self.best_epoch = 0
        self.patience_counter = 0

        # 恢复训练
        if args.resume:
            self.load_checkpoint(args.resume)

        self.print_info()

    def setup_dirs(self):
        """创建输出目录（基于exp_name）"""
        # 使用实验名称创建独立目录
        self.exp_dir = Path('outputs') / self.args.exp_name
        self.checkpoint_dir = self.exp_dir / 'checkpoints'
        self.log_dir = self.exp_dir / 'logs'
        self.results_dir = self.exp_dir / 'results'

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        print(f"Experiment directory: {self.exp_dir}")

    def build_model(self):
        """构建模型"""
        variant = cfg.MODEL_TYPE.split('_')[-1]  # e.g., "segformer_b1" -> "b1"
        model = build_model(
            variant=variant,
            pretrained=cfg.PRETRAINED,
            num_classes=cfg.NUM_CLASSES
        )
        model = model.to(self.device)
        return model

    def build_dataloaders(self):
        """构建数据加载器（统一接口）"""
        # ← 修改：使用unified接口
        train_loader, val_loader, test_loader = create_dataloaders_unified(
            dataset_name=self.args.dataset,
            data_root=Path(self.args.data_root),
            batch_size=self.args.batch_size,
            num_workers=cfg.NUM_WORKERS,
            crop_size=cfg.CROP_SIZE
        )
        return train_loader, val_loader, test_loader

    def build_optimizer(self):
        """构建优化器"""
        if cfg.OPTIMIZER == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=cfg.WEIGHT_DECAY,
                betas=cfg.BETAS
            )
        elif cfg.OPTIMIZER == 'sgd':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.args.lr,
                momentum=0.9,
                weight_decay=cfg.WEIGHT_DECAY
            )
        else:
            raise ValueError(f"Unknown optimizer: {cfg.OPTIMIZER}")
        return optimizer

    def build_scheduler(self):
        """构建学习率调度器"""
        if cfg.LR_SCHEDULER == 'polynomial':
            # Polynomial decay with warmup
            def lr_lambda(epoch):
                if epoch < cfg.WARMUP_EPOCHS:
                    # Warmup
                    return cfg.WARMUP_LR / self.args.lr + \
                        (1 - cfg.WARMUP_LR / self.args.lr) * epoch / cfg.WARMUP_EPOCHS
                else:
                    # Polynomial decay
                    return (1 - (epoch - cfg.WARMUP_EPOCHS) / (self.args.epochs - cfg.WARMUP_EPOCHS)) ** cfg.LR_POWER

            scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        elif cfg.LR_SCHEDULER == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.args.epochs, eta_min=cfg.MIN_LR
            )
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        return scheduler

    def print_info(self):
        """打印训练信息"""
        print(f"\nTraining Configuration:")
        print(f"  Dataset: {self.dataset_name.upper()}")
        print(f"  Model: {cfg.MODEL_TYPE}")
        print(f"  Device: {self.device}")
        print(f"  Train samples: {len(self.train_loader.dataset)}")
        print(f"  Val samples: {len(self.val_loader.dataset)}")
        print(f"  Test samples: {len(self.test_loader.dataset)}")
        print(f"  Batch size: {self.args.batch_size}")
        print(f"  Gradient accumulation: {cfg.GRADIENT_ACCUMULATION_STEPS}")
        print(f"  Effective batch size: {self.args.batch_size * cfg.GRADIENT_ACCUMULATION_STEPS}")
        print(f"  Epochs: {self.args.epochs}")
        print(f"  Learning rate: {self.args.lr}")
        print(f"  AMP: {self.use_amp}")
        print("=" * 60)

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        self.criterion.train()

        total_loss = 0.0
        all_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'iou': 0, 'oa': 0}
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.args.epochs}')

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            # 获取数据
            img_a = batch['img_a'].to(self.device)
            img_b = batch['img_b'].to(self.device)
            label = batch['label'].to(self.device).long()

            # 前向传播
            if self.use_amp:
                with autocast():
                    outputs = self.model(img_a, img_b)
                    loss, loss_dict = self.criterion(outputs, label)
                    loss = loss / cfg.GRADIENT_ACCUMULATION_STEPS

                # 反向传播
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(img_a, img_b)
                loss, loss_dict = self.criterion(outputs, label)
                loss = loss / cfg.GRADIENT_ACCUMULATION_STEPS
                loss.backward()

            # 梯度累积
            if (batch_idx + 1) % cfg.GRADIENT_ACCUMULATION_STEPS == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            # 计算指标
            with torch.no_grad():
                metrics = compute_metrics(outputs['pred'], label)
                for k in all_metrics:
                    all_metrics[k] += metrics[k]

            total_loss += loss_dict['total']

            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'F1': f"{metrics['f1']:.4f}",
                'IoU': f"{metrics['iou']:.4f}"
            })

        # 平均指标
        avg_loss = total_loss / num_batches
        for k in all_metrics:
            all_metrics[k] /= num_batches

        return avg_loss, all_metrics

    @torch.no_grad()
    def validate(self, epoch):
        """验证"""
        self.model.eval()
        self.criterion.eval()

        total_loss = 0.0
        all_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'iou': 0, 'oa': 0}
        num_batches = len(self.val_loader)

        pbar = tqdm(self.val_loader, desc='Validating')

        for batch in pbar:
            img_a = batch['img_a'].to(self.device)
            img_b = batch['img_b'].to(self.device)
            label = batch['label'].to(self.device).long()

            if self.use_amp:
                with autocast():
                    outputs = self.model(img_a, img_b)
                    loss, loss_dict = self.criterion(outputs, label)
            else:
                outputs = self.model(img_a, img_b)
                loss, loss_dict = self.criterion(outputs, label)

            metrics = compute_metrics(outputs['pred'], label)

            total_loss += loss_dict['total']
            for k in all_metrics:
                all_metrics[k] += metrics[k]

            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'F1': f"{metrics['f1']:.4f}"
            })

        avg_loss = total_loss / num_batches
        for k in all_metrics:
            all_metrics[k] /= num_batches

        return avg_loss, all_metrics

    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'dataset': self.dataset_name,  # ← 保存数据集信息
            'config': {
                'model_type': cfg.MODEL_TYPE,
                'num_classes': cfg.NUM_CLASSES,
                'image_size': cfg.IMAGE_SIZE
            }
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        # 保存最新检查点
        checkpoint_path = self.checkpoint_dir / f'epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)

        # 保存最佳检查点
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved best model (F1: {self.best_metric:.4f})")

        # 清理旧检查点（保留最新5个）
        checkpoints = sorted(self.checkpoint_dir.glob('epoch_*.pth'))
        if len(checkpoints) > 5:
            for old_ckpt in checkpoints[:-5]:
                old_ckpt.unlink()

    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # ← 新增：根据参数决定是否重置
        if self.args.reset_epoch:
            self.start_epoch = 0
            print("  ✓ Epoch counter reset to 0")
        else:
            self.start_epoch = checkpoint['epoch'] + 1

        if self.args.reset_best:
            self.best_metric = 0.0
            self.best_epoch = 0
            print("  ✓ Best metric reset to 0.0")
        else:
            self.best_metric = checkpoint.get('best_metric', 0.0)
            self.best_epoch = checkpoint.get('best_epoch', 0)

        print(f"  Start epoch: {self.start_epoch}")
        print(f"  Best F1 threshold: {self.best_metric:.4f}")
    def train(self):
        """主训练循环"""
        print("\nStarting training...")

        for epoch in range(self.start_epoch, self.args.epochs):
            # 训练
            train_loss, train_metrics = self.train_epoch(epoch + 1)

            # 学习率调度
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            # 打印训练结果
            print(f"\nEpoch {epoch + 1}/{self.args.epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Train F1: {train_metrics['f1']:.4f}, IoU: {train_metrics['iou']:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")

            # 验证
            if (epoch + 1) % cfg.VAL_FREQ == 0:
                val_loss, val_metrics = self.validate(epoch + 1)

                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Val F1: {val_metrics['f1']:.4f}, IoU: {val_metrics['iou']:.4f}")
                print(f"  Val Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")

                # 检查是否是最佳模型
                current_metric = val_metrics['f1']  # 固定用F1
                is_best = current_metric > self.best_metric

                if is_best:
                    self.best_metric = current_metric
                    self.best_epoch = epoch + 1
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                # TensorBoard日志
                if self.writer:
                    self.writer.add_scalar('Loss/train', train_loss, epoch + 1)
                    self.writer.add_scalar('Loss/val', val_loss, epoch + 1)
                    self.writer.add_scalar('Metrics/train_f1', train_metrics['f1'], epoch + 1)
                    self.writer.add_scalar('Metrics/val_f1', val_metrics['f1'], epoch + 1)
                    self.writer.add_scalar('Metrics/val_iou', val_metrics['iou'], epoch + 1)
                    self.writer.add_scalar('LR', current_lr, epoch + 1)

                # 保存检查点
                if (epoch + 1) % cfg.SAVE_FREQ == 0 or is_best:
                    self.save_checkpoint(epoch + 1, is_best)

                # 早停检查
                if cfg.EARLY_STOPPING and self.patience_counter >= cfg.PATIENCE:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    print(f"Best F1: {self.best_metric:.4f} at epoch {self.best_epoch}")
                    break

        # 训练结束
        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Best F1: {self.best_metric:.4f} at epoch {self.best_epoch}")
        print("=" * 60)

        # 保存最终结果
        self.save_final_results()

        if self.writer:
            self.writer.close()

    def save_final_results(self):
        """保存最终结果"""
        results = {
            'dataset': self.dataset_name,
            'best_f1': self.best_metric,
            'best_epoch': self.best_epoch,
            'exp_name': self.args.exp_name
        }

        with open(self.results_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Unified Change Detection Training')

    # 必需参数
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['levir', 's2looking', 'whucd'],
                        help='Dataset name')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Path to dataset root directory')
    parser.add_argument('--exp-name', type=str, required=True,
                        help='Experiment name (for saving outputs)')

    # 可选参数
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate (default: 5e-4)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--reset-best', action='store_true',  # ← 新增
                       help='Reset best metric when resuming (for cross-dataset fine-tuning)')
    parser.add_argument('--reset-epoch', action='store_true',  # ← 新增
                       help='Reset epoch counter when resuming')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable AMP')

    args = parser.parse_args()

    # 更新配置（如果需要）
    if args.no_amp:
        cfg.USE_AMP = False

    # 显示配置
    print("=" * 60)
    print(f"Unified Change Detection Training")
    print("=" * 60)
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Data Root: {args.data_root}")
    print(f"Experiment: {args.exp_name}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print("=" * 60)

    # 创建训练器并开始训练
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()