"""
训练脚本 - 稳健版
功能：完整但不臃肿，易于调试

运行：
    python train.py                    # 完整训练
    python train.py --epochs 5         # 快速测试
    python train.py --resume best.pth  # 恢复训练
    python train.py --no-distillation  # 不使用蒸馏（消融实验）
"""
import os
os.environ['ALBUMENTATIONS_CHECK_VERSION'] = 'False'
from losses.losses import DistillationLossWithTricks
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import json
import random
import numpy as np
from config import cfg
from dataset import build_dataloader
from models.segformer import build_segformer_distillation
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)


def compute_seg_loss_only(pred, labels, ignore_index=255):
    """
    计算纯分割损失（不含蒸馏），正确处理ignore区域

    Args:
        pred: (B, 1, H, W) - 模型预测logits
        labels: (B, H, W) - 标签（0=背景, 1=前景, 255=ignore）
        ignore_index: 忽略标签值

    Returns:
        loss: BCE损失（只在有效区域）
    """
    import torch.nn.functional as F

    # 过滤ignore区域
    mask = (labels != ignore_index).unsqueeze(1).float()  # (B, 1, H, W)
    labels_binary = (labels > 0).unsqueeze(1).float()  # (B, 1, H, W)

    # 只在有效区域计算BCE
    loss = F.binary_cross_entropy_with_logits(
        pred * mask,
        labels_binary * mask,
        reduction='sum'
    ) / (mask.sum() + 1e-6)

    return loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Trainer:
    """训练器"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')
        self.no_distillation = args.no_distillation

        # 创建输出目录
        self.setup_dirs()

        # 初始化模型
        print("=" * 60)
        print("初始化训练...")
        if self.no_distillation:
            print("⚠️ 消融实验模式：不使用蒸馏")
        print("=" * 60)
        self.model = self.build_model()

        # 数据加载器
        actual_batch_size = cfg.BATCH_SIZE // cfg.GRADIENT_ACCUMULATION_STEPS
        if actual_batch_size == 0:
            actual_batch_size = 1

        self.train_loader = build_dataloader('train', batch_size=actual_batch_size, shuffle=True)
        self.val_loader = build_dataloader('val', batch_size=1, shuffle=False)

        print(f"训练集: {len(self.train_loader.dataset)} 张")
        print(f"验证集: {len(self.val_loader.dataset)} 张")
        print(f"全局Batch Size: {cfg.BATCH_SIZE}")
        print(f"物理Batch Size (per step): {actual_batch_size}")
        print(f"梯度累积步数: {cfg.GRADIENT_ACCUMULATION_STEPS}")
        print(f"设备: {self.device}")

        # 优化器和调度器
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()

        # 损失函数
        self.criterion = DistillationLossWithTricks()

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
        optimizer_steps_per_epoch = len(self.train_loader) // cfg.GRADIENT_ACCUMULATION_STEPS
        total_steps = optimizer_steps_per_epoch * cfg.NUM_EPOCHS

        if cfg.LR_SCHEDULER == 'polynomial':
            from torch.optim.lr_scheduler import LambdaLR

            warmup_steps = cfg.WARMUP_EPOCHS * optimizer_steps_per_epoch

            def poly_lr_lambda(step):
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                else:
                    decay_steps = total_steps - warmup_steps
                    progress = float(step - warmup_steps) / float(max(1, decay_steps))
                    return (1.0 - progress) ** cfg.LR_POWER

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

        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(pbar):
            # 准备数据
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)  # ✅ 提前定义labels

            targets = {
                'label': labels,
                'teacher_feat_b30': batch['teacher_feat_b30'].to(self.device),
                'teacher_feat_enc': batch['teacher_feat_enc'].to(self.device)
            }

            if cfg.USE_AMP:
                with autocast():
                    outputs = self.model(images)

                    # ========== 蒸馏开关 ==========
                    if self.no_distillation:
                        pred = outputs['pred']
                        loss_seg = compute_seg_loss_only(
                            pred, labels,
                            ignore_index=cfg.IGNORE_INDEX
                        )

                        loss = loss_seg
                        loss_dict = {
                            'seg': loss_seg.item(),
                            'feat_b30': 0.0,
                            'feat_enc': 0.0
                        }
                    else:
                        loss, loss_dict = self.criterion(outputs, targets)
                    # ========== 修改结束 ==========

                # 缩放loss
                loss = loss / cfg.GRADIENT_ACCUMULATION_STEPS

                # 反向传播
                self.scaler.scale(loss).backward()

                # 每N步更新一次参数
                if (batch_idx + 1) % cfg.GRADIENT_ACCUMULATION_STEPS == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()

            else:
                outputs = self.model(images)

                # ========== 蒸馏开关 ==========
                if self.no_distillation:
                    pred = outputs['pred']
                    loss_seg = compute_seg_loss_only(
                        pred, labels,
                        ignore_index=cfg.IGNORE_INDEX
                    )

                    loss = loss_seg
                    loss_dict = {
                        'seg': loss_seg.item(),
                        'feat_b30': 0.0,
                        'feat_enc': 0.0
                    }
                else:
                    loss, loss_dict = self.criterion(outputs, targets)
                # ========== 修改结束 ==========

                # 缩放loss
                loss = loss / cfg.GRADIENT_ACCUMULATION_STEPS

                # 反向传播
                loss.backward()

                # 每N步更新一次参数
                if (batch_idx + 1) % cfg.GRADIENT_ACCUMULATION_STEPS == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
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

            # 记录到TensorBoard
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
            labels = batch['label'].to(self.device)  # ✅ 提前定义labels

            targets = {
                'label': labels,
                'teacher_feat_b30': batch['teacher_feat_b30'].to(self.device),
                'teacher_feat_enc': batch['teacher_feat_enc'].to(self.device)
            }

            # 前向传播
            outputs = self.model(images)

            # ========== 蒸馏开关 ==========
            if self.no_distillation:
                pred = outputs['pred']
                loss = compute_seg_loss_only(
                    pred, labels,
                    ignore_index=cfg.IGNORE_INDEX
                )
            else:
                loss, _ = self.criterion(outputs, targets)
            # ========== 修改结束 ==========

            total_loss += loss.item()

            # 计算IoU
            pred = torch.sigmoid(outputs['pred']) > 0.5
            pred = pred.squeeze(1).long()

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
            'no_distillation': self.no_distillation,  # ✅ 保存实验配置
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

        # 清理旧checkpoint
        max_keep = 5
        checkpoints = sorted(cfg.CHECKPOINT_DIR.glob('epoch_*.pth'), key=os.path.getmtime)
        if len(checkpoints) > max_keep:
            for old_ckpt in checkpoints[:-max_keep]:
                old_ckpt.unlink()
                print(f"✓ 清理旧checkpoint: {old_ckpt.name}")

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
    parser.add_argument('--no-distillation', action='store_true',
                        help='不使用蒸馏，仅训练分割任务（消融实验）')
    args = parser.parse_args()

    set_seed(cfg.SEED)

    # 覆盖epochs配置
    if args.epochs is not None:
        cfg.NUM_EPOCHS = args.epochs

    config_dict = {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in vars(cfg).items()
        if not k.startswith('__') and not callable(v)
    }

    # 在创建Trainer之前确保日志目录存在
    cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)

    with open(cfg.LOG_DIR / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=4)
    print(f"✓ 配置文件已保存到: {cfg.LOG_DIR / 'config.json'}")

    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()