"""
损失函数 - 【v2.0 最终毕业版】
- ✅ 彻底删除KDLoss，聚焦于特征蒸馏
- ✅ 简化FeatureLoss，只负责计算，不负责对齐
- ✅ 改造BoundaryLoss，使其适用于二分类任务
- ✅ 重写TotalLoss，使其与我们的最终方案完全匹配
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
import logging
import kornia  # 需要安装: pip install kornia
logger = logging.getLogger(__name__)
IGNORE_INDEX = getattr(cfg, 'IGNORE_INDEX', 255)

# ==========================================================
#                  核心损失函数组件
# ==========================================================

class ConfidenceWeightedLoss(nn.Module):
    """
    置信度加权损失

    对高置信度预测降低权重
    对低置信度预测增加权重
    """

    def __init__(self, temperature=2.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W) logits
            target: (B, H, W) labels
        """
        # 预测概率
        pred_prob = torch.sigmoid(pred)  # (B, 1, H, W)

        # 计算置信度（最大概率）
        confidence = torch.max(
            torch.stack([pred_prob, 1 - pred_prob], dim=0),
            dim=0
        )[0]  # (B, 1, H, W)

        # 置信度权重（置信度低的权重高）
        # weight = exp(-confidence / temperature)
        weight = torch.exp(-confidence / self.temperature)

        # BCE损失
        target_float = target.unsqueeze(1).float()
        bce = F.binary_cross_entropy_with_logits(
            pred, target_float, reduction='none'
        )

        # 加权
        weighted_bce = bce * weight

        return weighted_bce.mean()


class BoundaryAwareLoss(nn.Module):
    """
    边界感知损失

    使用Sobel算子检测边界，对边界区域增加权重
    """

    def __init__(self, boundary_weight=2.0):
        super().__init__()
        self.boundary_weight = boundary_weight

    def extract_boundary(self, mask):
        """
        使用Sobel算子提取边界

        Args:
            mask: (B, H, W) 标签mask

        Returns:
            boundary: (B, 1, H, W) 边界map
        """
        # 转换为float并添加channel维度
        mask_float = mask.unsqueeze(1).float()  # (B, 1, H, W)

        # Sobel边缘检测
        from kornia.filters import sobel
        edges = sobel(mask_float)  # (B, 1, H, W)

        # 二值化边界
        boundary = (edges > 0.1).float()

        return boundary

    def forward(self, pred, target):
        """
        计算边界感知损失

        Args:
            pred: (B, 1, H, W) 预测logits
            target: (B, H, W) GT标签

        Returns:
            loss: scalar
        """
        # 提取边界
        boundary = self.extract_boundary(target)  # (B, 1, H, W)

        # 计算BCE损失（逐像素）
        target_float = target.unsqueeze(1).float()
        bce = F.binary_cross_entropy_with_logits(
            pred, target_float, reduction='none'
        )  # (B, 1, H, W)

        # 边界权重map
        # 边界区域权重更高，非边界区域权重为1
        weight_map = 1.0 + self.boundary_weight * boundary

        # 加权BCE
        weighted_bce = bce * weight_map

        return weighted_bce.mean()


class DiceLoss(nn.Module):
    """Dice损失，适用于二分类分割"""
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target,mask=None):
        """
        Args:
            pred: (B, 1, H, W) - 预测的logits
            target: (B, H, W) - ground truth (0或1)
        """
        pred = torch.sigmoid(pred)
        if mask is not None:
            pred = pred * mask
            target = target * mask
        # 展平
        pred_flat = pred.flatten(1)
        target_flat = target.flatten(1)

        intersection = (pred_flat * target_flat).sum(1)
        union = pred_flat.sum(1) + target_flat.sum(1)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class FeatureLoss(nn.Module):
    """
    【简化版】特征蒸馏损失
    只负责计算MSE，不负责对齐
    """
    def __init__(self):
        super(FeatureLoss, self).__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, student_feat, teacher_feat):
        """
        假设输入的特征维度已经对齐
        """
        return self.loss_fn(student_feat, teacher_feat)

# ==========================================================
#                   总损失 (最终方案)
# ==========================================================

class DistillationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss  = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.feat_loss = FeatureLoss()

        # ✅ 正确
        self.w_seg = getattr(cfg, 'LOSS_SEG_WEIGHT', 1.0)
        self.w_feat_b30 = getattr(cfg, 'LOSS_FEAT_B30_WEIGHT', 0.5)
        self.w_feat_enc = getattr(cfg, 'LOSS_FEAT_ENC_WEIGHT', 0.5)

    def forward(self, outputs, targets):
        pred      = outputs['pred']                  # (B,1,H,W)
        device = pred.device
        gt_label = targets['label'].to(pred.device)  # ✅ 现在拿到的gt_label直接就是二值的了！              # (B,H,W)
        mask      = (gt_label != IGNORE_INDEX).to(pred.device)   # bool
        gt_binary = gt_label.float()  # ✅ 不再需要 (gt_label > 0) 的转换！

        # 1. Seg Loss
        mask_4d   = mask.unsqueeze(1).float()        # (B,1,H,W)
        loss_dice = self.dice_loss(pred, gt_binary.unsqueeze(1), mask_4d)

        valid = mask.bool()
        loss_bce = F.binary_cross_entropy_with_logits(
            pred.squeeze(1)[valid],
            gt_binary[valid],
            reduction='mean'
        )
        pos_weight = getattr(cfg, 'BCE_POS_WEIGHT', None)
        if pos_weight is not None:
            self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
            loss_bce = self.bce_loss(pred.squeeze(1)[valid], gt_binary[valid])

        loss_seg  = loss_bce + loss_dice

        # 2. Feature KD
        t_feat_b30 = targets['teacher_feat_b30'].to(device).detach()
        t_feat_enc = targets['teacher_feat_enc'].to(device).detach()
        loss_feat_b30 = self.feat_loss(outputs['feat_b30'], t_feat_b30.detach())
        loss_feat_enc = self.feat_loss(outputs['feat_enc'], t_feat_enc.detach())

        # 3. Total
        total_loss = (self.w_seg * loss_seg +
                      self.w_feat_b30 * loss_feat_b30 +
                      self.w_feat_enc * loss_feat_enc)

        with torch.no_grad():
            loss_dict = {
                'total': total_loss.item(),
                'seg': loss_seg.item(),
                'bce': loss_bce.item(),
                'dice': loss_dice.item(),
                'feat_b30': loss_feat_b30.item(),
                'feat_enc': loss_feat_enc.item()
            }
        return total_loss, loss_dict


class DistillationLossWithTricks(nn.Module):
    """
    蒸馏损失 + 边界感知
    组合你现有的蒸馏损失和新的边界感知
    """

    def __init__(self):
        super().__init__()

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.boundary_loss = BoundaryAwareLoss(boundary_weight=2.0)
        self.confidence_loss = ConfidenceWeightedLoss(temperature=2.0)  # 新增
        self.mse_loss = nn.MSELoss()

    def forward(self, outputs, targets):
        """
        Args:
            outputs: {
                'pred': (B, 1, H, W),
                'feat_b30': (B, C1, H1, W1),
                'feat_enc': (B, C2, H2, W2)
            }
            targets: {
                'label': (B, H, W),
                'teacher_feat_b30': (B, C1, H1, W1),
                'teacher_feat_enc': (B, C2, H2, W2)
            }
        """
        pred = outputs['pred']
        label = targets['label']

        # ========== 分割损失 ==========
        # 原来的BCE
        loss_bce = self.bce_loss(
            pred,
            label.unsqueeze(1).float()
        )

        # 原来的Dice
        loss_dice = self.dice_loss(
            torch.sigmoid(pred),
            label.unsqueeze(1).float()
        )

        # 新增：边界感知
        loss_boundary = self.boundary_loss(pred, label)

        # ========== 特征蒸馏损失（不变）==========
        loss_feat_b30 = self.mse_loss(
            outputs['feat_b30'],
            targets['teacher_feat_b30']
        )

        loss_feat_enc = self.mse_loss(
            outputs['feat_enc'],
            targets['teacher_feat_enc']
        )
        loss_confidence = self.confidence_loss(pred, label)
        # ========== 总损失 ==========
        # 组合所有损失
        from config import cfg

        total_loss = (
                cfg.LOSS_WEIGHT_SEG * (loss_bce + loss_dice) +
                cfg.LOSS_WEIGHT_BOUNDARY * loss_boundary +
                cfg.LOSS_WEIGHT_CONFIDENCE * loss_confidence +  # 新增
                cfg.LOSS_WEIGHT_FEAT_B30 * loss_feat_b30 +
                cfg.LOSS_WEIGHT_FEAT_ENC * loss_feat_enc
        )

        return total_loss, {
            'seg': (loss_bce + loss_dice).item(),
            'boundary': loss_boundary.item(),
            'confidence': loss_confidence.item(),  # 新增
            'feat_b30': loss_feat_b30.item(),
            'feat_enc': loss_feat_enc.item()
        }

if __name__ == '__main__':
    """测试最终的DistillationLoss"""
    print("=" * 60)
    print("测试最终版DistillationLoss")
    print("=" * 60)

    # 虚拟数据 (新增 ignore 测试)
    B, H, W = 2, 256, 256
    h, w = 64, 64

    # 学生输出
    outputs = {
        'pred': torch.randn(B, 1, H, W),
        'feat_b30': torch.randn(B, 1280, h, w),
        'feat_enc': torch.randn(B, 256, h, w)
    }

    # 监督信号 (新增 255 ignore 区域)
    # 先生成二值，再打一些 ignore
    label = torch.randint(0, 2, (B, H, W))  # {0,1}
    ignore_mask = torch.rand(B, H, W) < 0.1  # 10% 像素忽略（示例）
    label[ignore_mask] = IGNORE_INDEX  # 255

    targets = {
        'label': label,
        'teacher_feat_b30': torch.randn(B, 1280, h, w),
        'teacher_feat_enc': torch.randn(B, 256, h, w)
    }

    loss_fn = DistillationLoss()
    total_loss, loss_dict = loss_fn(outputs, targets)

    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"Loss Dict: {loss_dict}")

    assert 'seg' in loss_dict and 'feat_b30' in loss_dict and 'feat_enc' in loss_dict

    print("\n" + "=" * 60)
    print("✓ 最终版损失函数测试通过")
    print("=" * 60)