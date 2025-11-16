"""
变化检测损失函数
- BCE + Dice组合损失
- Focal Loss（处理类别不平衡）
- 边界感知损失
- 深度监督支持
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg


class DiceLoss(nn.Module):
    """Dice损失"""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) logits
            target: (B, H, W) labels
        """
        num_classes = pred.shape[1]

        if num_classes == 1:
            # 二分类单通道
            pred_prob = torch.sigmoid(pred).squeeze(1)
            target_float = target.float()
        else:
            # 多类别
            pred_prob = F.softmax(pred, dim=1)
            target_float = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

        # 计算Dice
        if num_classes == 1:
            intersection = (pred_prob * target_float).sum()
            union = pred_prob.sum() + target_float.sum()
            dice = (2 * intersection + self.smooth) / (union + self.smooth)
        else:
            # 只计算前景类（类别1）的Dice
            pred_fg = pred_prob[:, 1]
            target_fg = target_float[:, 1]
            intersection = (pred_fg * target_fg).sum()
            union = pred_fg.sum() + target_fg.sum()
            dice = (2 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice


class FocalLoss(nn.Module):
    """Focal Loss - 处理类别不平衡"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) logits
            target: (B, H, W) labels
        """
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class BoundaryLoss(nn.Module):
    """边界感知损失"""
    def __init__(self, boundary_weight=2.0):
        super().__init__()
        self.boundary_weight = boundary_weight

        # Sobel算子
        self.sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)

        self.sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)

    def extract_boundary(self, mask):
        """提取边界"""
        device = mask.device

        # 移动Sobel算子到正确设备
        sobel_x = self.sobel_x.to(device)
        sobel_y = self.sobel_y.to(device)

        # 转换为float并添加channel维度
        mask_float = mask.unsqueeze(1).float()

        # Sobel边缘检测
        edge_x = F.conv2d(mask_float, sobel_x, padding=1)
        edge_y = F.conv2d(mask_float, sobel_y, padding=1)
        edges = torch.sqrt(edge_x ** 2 + edge_y ** 2)

        # 二值化边界
        boundary = (edges > 0.1).float()
        return boundary

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) logits
            target: (B, H, W) labels
        """
        # 提取边界
        boundary = self.extract_boundary(target)

        # 计算CE损失（逐像素）
        ce_loss = F.cross_entropy(pred, target, reduction='none')

        # 边界权重map
        weight_map = 1.0 + self.boundary_weight * boundary.squeeze(1)

        # 加权CE
        weighted_ce = ce_loss * weight_map
        return weighted_ce.mean()


class ChangeLoss(nn.Module):
    """
    变化检测组合损失

    支持:
    - BCE/CE + Dice
    - Focal Loss
    - 边界感知
    - 类别权重
    - 深度监督
    """

    def __init__(self):
        super().__init__()

        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha=cfg.FOCAL_ALPHA, gamma=cfg.FOCAL_GAMMA)
        self.boundary_loss = BoundaryLoss(boundary_weight=2.0)

        # 类别权重
        if cfg.USE_CLASS_WEIGHTS:
            self.class_weights = torch.tensor(cfg.CLASS_WEIGHTS, dtype=torch.float32)
        else:
            self.class_weights = None

        # 损失权重
        self.w_bce = cfg.LOSS_WEIGHT_BCE
        self.w_dice = cfg.LOSS_WEIGHT_DICE
        self.w_boundary = cfg.LOSS_WEIGHT_BOUNDARY

        # 深度监督权重
        if cfg.DEEP_SUPERVISION:
            self.ds_weights = cfg.DEEP_SUPERVISION_WEIGHTS
        else:
            self.ds_weights = None

    def compute_loss(self, pred, target):
        """计算单个预测的损失"""
        device = pred.device

        # 移动类别权重到正确设备
        if self.class_weights is not None:
            weight = self.class_weights.to(device)
        else:
            weight = None

        # BCE/CE损失
        if cfg.LOSS_TYPE == "focal":
            loss_ce = self.focal_loss(pred, target)
        else:
            loss_ce = F.cross_entropy(pred, target, weight=weight, ignore_index=cfg.IGNORE_INDEX)

        # Dice损失
        loss_dice = self.dice_loss(pred, target)

        # 边界损失
        if self.w_boundary > 0:
            loss_boundary = self.boundary_loss(pred, target)
        else:
            loss_boundary = torch.tensor(0.0, device=device)

        # 组合
        total = self.w_bce * loss_ce + self.w_dice * loss_dice + self.w_boundary * loss_boundary

        return total, {
            'ce': loss_ce.item(),
            'dice': loss_dice.item(),
            'boundary': loss_boundary.item()
        }

    def forward(self, outputs, targets):
        """
        Args:
            outputs: {
                'pred': (B, C, H, W) 主预测
                'aux_preds': [(B, C, H, W), ...] 辅助预测（深度监督）
            }
            targets: (B, H, W) 标签

        Returns:
            total_loss: 总损失
            loss_dict: 各项损失详情
        """
        pred = outputs['pred']
        label = targets

        # 主损失
        main_loss, loss_dict = self.compute_loss(pred, label)

        total_loss = main_loss
        loss_dict['main'] = main_loss.item()

        # 深度监督损失
        if self.training and 'aux_preds' in outputs and self.ds_weights is not None:
            aux_losses = []
            for i, aux_pred in enumerate(outputs['aux_preds']):
                aux_loss, _ = self.compute_loss(aux_pred, label)
                aux_losses.append(aux_loss * self.ds_weights[i])
                loss_dict[f'aux_{i}'] = aux_loss.item()

            ds_loss = sum(aux_losses)
            total_loss = total_loss + ds_loss
            loss_dict['deep_supervision'] = ds_loss.item()

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict


def build_criterion():
    """构建损失函数"""
    return ChangeLoss()


if __name__ == '__main__':
    """测试损失函数"""
    print("=" * 60)
    print("Testing Change Detection Loss")
    print("=" * 60)

    # 虚拟数据
    B, H, W = 2, 256, 256

    # 模型输出
    outputs = {
        'pred': torch.randn(B, 2, H, W),  # 2类输出
        'aux_preds': [
            torch.randn(B, 2, H, W),
            torch.randn(B, 2, H, W),
            torch.randn(B, 2, H, W),
            torch.randn(B, 2, H, W)
        ]
    }

    # 标签
    targets = torch.randint(0, 2, (B, H, W))

    # 创建损失函数
    criterion = ChangeLoss()
    criterion.train()

    # 计算损失
    total_loss, loss_dict = criterion(outputs, targets)

    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"\nLoss Details:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")

    print("\n" + "=" * 60)
    print("Loss function test passed!")
    print("=" * 60)
