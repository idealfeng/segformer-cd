"""
训练脚本 - Tricks版本
专注于边界感知等tricks，不使用蒸馏
"""
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from config import cfg
from dataset import build_dataloader
from models.segformer_tricks import build_segformer_tricks  # ✅ 新模型
from losses.losses import DistillationLossWithTricks  # ✅ 使用带tricks的损失
import random
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TricksTrainer:
    """Tricks训练器（简化版）"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 修改输出目录
        self.output_dir = Path('outputs/tricks')  # ✅ 独立目录
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.log_dir = self.output_dir / 'logs'

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print("Tricks Baseline训练")
        print("=" * 60)

        # 模型
        self.model = build_segformer_tricks(
            variant=cfg.STUDENT_MODEL.replace('segformer_', ''),
            pretrained=cfg.STUDENT_PRETRAINED
        )
        self.model = self.model.to(self.device)

        # 数据
        actual_batch_size = cfg.BATCH_SIZE // cfg.GRADIENT_ACCUMULATION_STEPS
        self.train_loader = build_dataloader('train', batch_size=actual_batch_size, shuffle=True)
        self.val_loader = build_dataloader('val', batch_size=1, shuffle=False)

        print(f"训练集: {len(self.train_loader.dataset)} 张")
        print(f"验证集: {len(self.val_loader.dataset)} 张")

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.LEARNING_RATE,
            weight_decay=cfg.WEIGHT_DECAY
        )

        # 损失（使用带tricks的）
        self.criterion = DistillationLossWithTricks()

        # AMP
        self.scaler = GradScaler() if cfg.USE_AMP else None

        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        self.best_iou = 0.0
        print("=" * 60)

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{cfg.NUM_EPOCHS}")
        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            # ✅ 构造targets（tricks的loss需要teacher特征，但实际不用）
            targets = {
                'label': labels,
                'teacher_feat_b30': batch['teacher_feat_b30'].to(self.device),
                'teacher_feat_enc': batch['teacher_feat_enc'].to(self.device)
            }

            if cfg.USE_AMP:
                with autocast():
                    outputs = self.model(images)
                    loss, loss_dict = self.criterion(outputs, targets)
                    loss = loss / cfg.GRADIENT_ACCUMULATION_STEPS

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % cfg.GRADIENT_ACCUMULATION_STEPS == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
            else:
                outputs = self.model(images)
                loss, loss_dict = self.criterion(outputs, targets)
                loss = loss / cfg.GRADIENT_ACCUMULATION_STEPS

                loss.backward()

                if (batch_idx + 1) % cfg.GRADIENT_ACCUMULATION_STEPS == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self, epoch):
        """验证"""
        self.model.eval()

        intersection = 0
        union = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            outputs = self.model(images)
            pred = torch.sigmoid(outputs['pred']) > 0.5
            pred = pred.squeeze(1).long()

            mask = (labels != 255)
            labels_binary = (labels > 0).long()

            pred_valid = pred[mask]
            labels_valid = labels_binary[mask]

            intersection += (pred_valid & labels_valid).sum().item()
            union += (pred_valid | labels_valid).sum().item()

        iou = intersection / (union + 1e-10)
        return iou

    def save_checkpoint(self, epoch, iou, is_best=False):
        """保存checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_iou': self.best_iou
        }

        save_path = self.checkpoint_dir / f'epoch_{epoch}.pth'
        torch.save(checkpoint, save_path)

        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ 保存最优模型: {best_path} (IoU: {iou:.4f})")

    def train(self):
        """主训练循环"""
        print("\n开始训练...")

        for epoch in range(cfg.NUM_EPOCHS):
            train_loss = self.train_epoch(epoch)

            print(f"\nEpoch {epoch}/{cfg.NUM_EPOCHS}")
            print(f"  Train Loss: {train_loss:.4f}")

            if (epoch + 1) % cfg.VAL_FREQ == 0:
                val_iou = self.validate(epoch)
                print(f"  Val IoU: {val_iou:.4f}")

                self.writer.add_scalar('Val/IoU', val_iou, epoch)

                is_best = val_iou > self.best_iou
                if is_best:
                    self.best_iou = val_iou

                if (epoch + 1) % cfg.SAVE_FREQ == 0 or is_best:
                    self.save_checkpoint(epoch, val_iou, is_best)

        print(f"\n训练完成！最优IoU: {self.best_iou:.4f}")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Tricks Baseline训练')
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()

    set_seed(cfg.SEED)

    if args.epochs is not None:
        cfg.NUM_EPOCHS = args.epochs

    trainer = TricksTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()