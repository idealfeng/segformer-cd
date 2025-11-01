"""
训练脚本 - 稳健版
功能：完整但不臃肿，易于调试

运行：
    python train.py                    # 完整训练
    python train.py --epochs 5         # 快速测试
    python train.py --resume best.pth  # 恢复训练
"""
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import json
from datetime import datetime

from config import cfg
from dataset import build_dataloader
from models.segformer import build_segformer_distillation
from losses import DistillationLoss


class Trainer:
    """训练器"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')

        # 创建输出目录
        self.setup_dirs()

        # 初始化模型
        print("=" * 60)
        print("初始化训练...")
        print("=" * 60)
        self.model = self.build_model()

        # 数据加载器
        actual_batch_size = cfg.BATCH_SIZE // cfg.GRADIENT_ACCUMULATION_STEPS
        self.train_loader = build_dataloader('train', batch_size=actual_batch_size, shuffle=True)
        self.val_loader = build_dataloader('val', batch_size=1, shuffle=False)

        print(f"训练集: {len(self.train_loader.dataset)} 张")
        print(f"验证集: {len(self.val_loader.dataset)} 张")
        print(f"Batch Size: {cfg.BATCH_SIZE}")
        print(f"设备: {self.device}")
        print(f"实际Batch Size: {actual_batch_size} (累积{cfg.GRADIENT_ACCUMULATION_STEPS}步 = 等效{cfg.BATCH_SIZE})")
        # 优化器和调度器
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()

        # 损失函数
        self.criterion = DistillationLoss()

        # AMP
        self.scaler = GradScaler() if cfg.USE_AMP else None
        if cfg.USE_AMP:
            print("✓ 使用混合精度训练(AMP)")

        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(cfg.LOG_DIR))

        # 训练状态
        self.start_epoch = 0
        self.best_iou = 0.0
        self.patience_counter = 0

        # 恢复训练
        if args.resume:
            self.load_checkpoint(args.resume)

        print("=" * 60)

    def setup_dirs(self):
        """创建必要的目录"""
        cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)

    def build_model(self):
        """构建模型"""
        model = build_segformer_distillation(
            variant=cfg.STUDENT_MODEL.replace('segformer_', ''),
            pretrained=cfg.STUDENT_PRETRAINED
        )
        model = model.to(self.device)
        return model

    def build_optimizer(self):
        """构建优化器"""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.LEARNING_RATE,
            weight_decay=cfg.WEIGHT_DECAY,
            betas=cfg.BETAS
        )
        return optimizer

    def build_scheduler(self):
        """构建学习率调度器"""
        total_steps = len(self.train_loader) * cfg.NUM_EPOCHS

        if cfg.LR_SCHEDULER == 'polynomial':
            from torch.optim.lr_scheduler import LambdaLR

            def poly_lr_lambda(step):
                if step < cfg.WARMUP_EPOCHS * len(self.train_loader):
                    # Warmup阶段
                    return step / (cfg.WARMUP_EPOCHS * len(self.train_loader))
                else:
                    # Polynomial decay
                    progress = (step - cfg.WARMUP_EPOCHS * len(self.train_loader)) / (
                                total_steps - cfg.WARMUP_EPOCHS * len(self.train_loader))
                    return (1 - progress) ** cfg.LR_POWER

            scheduler = LambdaLR(self.optimizer, lr_lambda=poly_lr_lambda)

        elif cfg.LR_SCHEDULER == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)

        else:
            raise ValueError(f"Unknown scheduler: {cfg.LR_SCHEDULER}")

        return scheduler

    def train_epoch(self, epoch):
        """训练一个epoch"""

        self.model.train()

        total_loss = 0.0
        loss_dict_sum = {}

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{cfg.NUM_EPOCHS}")

        self.optimizer.zero_grad()  # ✅ 移到循环外
        for batch_idx, batch in enumerate(pbar):
            # 准备数据
            images = batch['image'].to(self.device)
            targets = {
                'label': batch['label'].to(self.device),
                'teacher_feat_b30': batch['teacher_feat_b30'].to(self.device),
                'teacher_feat_enc': batch['teacher_feat_enc'].to(self.device)
            }

            if cfg.USE_AMP:
                with autocast():
                    outputs = self.model(images)
                    loss, loss_dict = self.criterion(outputs, targets)

                # ✅ 缩放loss
                loss = loss / cfg.GRADIENT_ACCUMULATION_STEPS

                # 反向传播
                self.scaler.scale(loss).backward()

                # ✅ 每N步更新一次参数
                if (batch_idx + 1) % cfg.GRADIENT_ACCUMULATION_STEPS == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                    # 学习率调度（移到这里）
                    self.scheduler.step()
            else:
                outputs = self.model(images)
                loss, loss_dict = self.criterion(outputs, targets)

                # ✅ 缩放loss
                loss = loss / cfg.GRADIENT_ACCUMULATION_STEPS

                # 反向传播
                loss.backward()

                # ✅ 每N步更新一次参数
                if (batch_idx + 1) % cfg.GRADIENT_ACCUMULATION_STEPS == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # 学习率调度（移到这里）
                    self.scheduler.step()
            # 累计损失
            total_loss += loss.item()
            for key, value in loss_dict.items():
                loss_dict_sum[key] = loss_dict_sum.get(key, 0) + value

            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })

            # 记录到TensorBoard（每50步）
            if (batch_idx + 1) % cfg.GRADIENT_ACCUMULATION_STEPS == 0 and (
                    batch_idx // cfg.GRADIENT_ACCUMULATION_STEPS) % 25 == 0:
                step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/loss_step', loss.item(), step)
                self.writer.add_scalar('Train/lr', self.optimizer.param_groups[0]['lr'], step)

        # 计算平均损失
        avg_loss = total_loss / len(self.train_loader)
        avg_loss_dict = {k: v / len(self.train_loader) for k, v in loss_dict_sum.items()}

        return avg_loss, avg_loss_dict

    @torch.no_grad()
    def validate(self, epoch):
        """验证"""
        self.model.eval()

        total_loss = 0.0

        # 用于计算IoU
        intersection = 0
        union = 0

        pbar = tqdm(self.val_loader, desc=f"Validation")

        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            targets = {
                'label': labels,
                'teacher_feat_b30': batch['teacher_feat_b30'].to(self.device),
                'teacher_feat_enc': batch['teacher_feat_enc'].to(self.device)
            }

            # 前向传播
            outputs = self.model(images)
            loss, _ = self.criterion(outputs, targets)

            total_loss += loss.item()

            # 计算IoU
            pred = torch.sigmoid(outputs['pred']) > 0.5  # (B, 1, H, W)
            pred = pred.squeeze(1).long()  # (B, H, W)

            # 二值化标签（过滤ignore）
            mask = (labels != 255)
            labels_binary = (labels > 0).long()

            # 只在有效区域计算
            pred_valid = pred[mask]
            labels_valid = labels_binary[mask]

            # IoU计算
            intersection += (pred_valid & labels_valid).sum().item()
            union += (pred_valid | labels_valid).sum().item()

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # 平均损失
        avg_loss = total_loss / len(self.val_loader)

        # IoU
        iou = intersection / (union + 1e-10)

        return avg_loss, iou

    def save_checkpoint(self, epoch, iou, is_best=False):
        """保存checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_iou': self.best_iou,
            'config': {
                'BATCH_SIZE': cfg.BATCH_SIZE,
                'LEARNING_RATE': cfg.LEARNING_RATE,
                'NUM_CLASSES': cfg.NUM_CLASSES
            }
        }

        if cfg.USE_AMP:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        # 保存最新
        save_path = cfg.CHECKPOINT_DIR / f'epoch_{epoch}.pth'
        torch.save(checkpoint, save_path)
        print(f"✓ 保存checkpoint: {save_path}")

        # 保存最优
        if is_best:
            best_path = cfg.CHECKPOINT_DIR / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ 保存最优模型: {best_path} (IoU: {iou:.4f})")

    def load_checkpoint(self, checkpoint_path):
        """加载checkpoint"""
        print(f"加载checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_iou = checkpoint['best_iou']

        if cfg.USE_AMP and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(f"✓ 恢复训练从epoch {self.start_epoch}")
        print(f"✓ 最优IoU: {self.best_iou:.4f}")

    def train(self):
        """主训练循环"""
        print("\n" + "=" * 60)
        print("开始训练")
        print("=" * 60)

        for epoch in range(self.start_epoch, cfg.NUM_EPOCHS):
            # 训练
            train_loss, train_loss_dict = self.train_epoch(epoch)

            # 记录训练损失
            self.writer.add_scalar('Train/loss_epoch', train_loss, epoch)
            for key, value in train_loss_dict.items():
                self.writer.add_scalar(f'Train/{key}', value, epoch)

            print(f"\nEpoch {epoch}/{cfg.NUM_EPOCHS}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"    seg: {train_loss_dict.get('seg', 0):.4f}, "
                  f"feat_b30: {train_loss_dict.get('feat_b30', 0):.4f}, "
                  f"feat_enc: {train_loss_dict.get('feat_enc', 0):.4f}")

            # 验证
            if (epoch + 1) % cfg.VAL_FREQ == 0 or epoch == cfg.NUM_EPOCHS - 1:
                val_loss, val_iou = self.validate(epoch)

                # 记录验证指标
                self.writer.add_scalar('Val/loss', val_loss, epoch)
                self.writer.add_scalar('Val/IoU', val_iou, epoch)

                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Val IoU: {val_iou:.4f}")

                # 检查是否最优
                is_best = val_iou > self.best_iou
                if is_best:
                    self.best_iou = val_iou
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                # 保存checkpoint
                if (epoch + 1) % cfg.SAVE_FREQ == 0 or is_best:
                    self.save_checkpoint(epoch, val_iou, is_best)

                # Early stopping
                if cfg.EARLY_STOPPING and self.patience_counter >= cfg.PATIENCE:
                    print(f"\n早停触发！{cfg.PATIENCE} epochs无提升")
                    break

            print("-" * 60)

        # 训练结束
        print("\n" + "=" * 60)
        print("训练完成！")
        print(f"最优IoU: {self.best_iou:.4f}")
        print("=" * 60)

        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='训练SegFormer蒸馏模型')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的checkpoint路径')
    args = parser.parse_args()

    # 覆盖epochs配置
    if args.epochs is not None:
        cfg.NUM_EPOCHS = args.epochs

    # 创建训练器
    trainer = Trainer(args)

    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()