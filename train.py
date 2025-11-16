"""
SegFormer变化检测训练脚本

运行:
    python train.py                     # 完整训练
    python train.py --epochs 50         # 自定义epoch
    python train.py --resume best.pth   # 恢复训练
    python train.py --batch-size 16     # 自定义batch size
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
from dataset import create_dataloaders
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

        # 设置随机种子
        set_seed(cfg.SEED)

        # 创建输出目录
        self.setup_dirs()

        # 初始化组件
        print("=" * 60)
        print("Initializing Change Detection Training...")
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
            self.writer = SummaryWriter(log_dir=str(cfg.LOG_DIR / datetime.now().strftime('%Y%m%d_%H%M%S')))
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
        """创建输出目录"""
        cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)
        cfg.VIS_DIR.mkdir(parents=True, exist_ok=True)
        cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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
        """构建数据加载器"""
        train_loader, val_loader, test_loader = create_dataloaders(
            batch_size=cfg.BATCH_SIZE,
            num_workers=cfg.NUM_WORKERS,
            crop_size=cfg.CROP_SIZE
        )
        return train_loader, val_loader, test_loader

    def build_optimizer(self):
        """构建优化器"""
        if cfg.OPTIMIZER == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=cfg.LEARNING_RATE,
                weight_decay=cfg.WEIGHT_DECAY,
                betas=cfg.BETAS
            )
        elif cfg.OPTIMIZER == 'sgd':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=cfg.LEARNING_RATE,
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
                    return cfg.WARMUP_LR / cfg.LEARNING_RATE + \
                           (1 - cfg.WARMUP_LR / cfg.LEARNING_RATE) * epoch / cfg.WARMUP_EPOCHS
                else:
                    # Polynomial decay
                    return (1 - (epoch - cfg.WARMUP_EPOCHS) / (cfg.NUM_EPOCHS - cfg.WARMUP_EPOCHS)) ** cfg.LR_POWER

            scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        elif cfg.LR_SCHEDULER == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg.NUM_EPOCHS, eta_min=cfg.MIN_LR
            )
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        return scheduler

    def print_info(self):
        """打印训练信息"""
        print(f"\nTraining Configuration:")
        print(f"  Model: {cfg.MODEL_TYPE}")
        print(f"  Device: {self.device}")
        print(f"  Train samples: {len(self.train_loader.dataset)}")
        print(f"  Val samples: {len(self.val_loader.dataset)}")
        print(f"  Test samples: {len(self.test_loader.dataset)}")
        print(f"  Batch size: {cfg.BATCH_SIZE}")
        print(f"  Gradient accumulation: {cfg.GRADIENT_ACCUMULATION_STEPS}")
        print(f"  Effective batch size: {cfg.BATCH_SIZE * cfg.GRADIENT_ACCUMULATION_STEPS}")
        print(f"  Epochs: {cfg.NUM_EPOCHS}")
        print(f"  Learning rate: {cfg.LEARNING_RATE}")
        print(f"  AMP: {self.use_amp}")
        print("=" * 60)

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        self.criterion.train()

        total_loss = 0.0
        all_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'iou': 0, 'oa': 0}
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{cfg.NUM_EPOCHS}')

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
            'config': {
                'model_type': cfg.MODEL_TYPE,
                'num_classes': cfg.NUM_CLASSES,
                'image_size': cfg.IMAGE_SIZE
            }
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        # 保存最新检查点
        checkpoint_path = cfg.CHECKPOINT_DIR / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)

        # 保存最佳检查点
        if is_best:
            best_path = cfg.CHECKPOINT_DIR / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"  Saved best model (F1: {self.best_metric:.4f})")

        # 清理旧检查点（保留最新5个）
        checkpoints = sorted(cfg.CHECKPOINT_DIR.glob('checkpoint_epoch_*.pth'))
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

        self.start_epoch = checkpoint['epoch'] + 1
        self.best_metric = checkpoint.get('best_metric', 0.0)
        self.best_epoch = checkpoint.get('best_epoch', 0)

        print(f"Resumed from epoch {self.start_epoch}, best F1: {self.best_metric:.4f}")

    def train(self):
        """主训练循环"""
        print("\nStarting training...")

        for epoch in range(self.start_epoch, cfg.NUM_EPOCHS):
            # 训练
            train_loss, train_metrics = self.train_epoch(epoch + 1)

            # 学习率调度
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            # 打印训练结果
            print(f"\nEpoch {epoch + 1}/{cfg.NUM_EPOCHS}")
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
                current_metric = val_metrics[cfg.BEST_METRIC.lower()]
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

        # 保存最终模型
        self.save_checkpoint(cfg.NUM_EPOCHS, is_best=False)

        if self.writer:
            self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='SegFormer Change Detection Training')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--no-amp', action='store_true', help='Disable AMP')

    args = parser.parse_args()

    # 更新配置
    if args.epochs:
        cfg.NUM_EPOCHS = args.epochs
    if args.batch_size:
        cfg.BATCH_SIZE = args.batch_size
    if args.lr:
        cfg.LEARNING_RATE = args.lr
    if args.no_amp:
        cfg.USE_AMP = False

    # 显示配置
    cfg.display()

    # 创建训练器并开始训练
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
