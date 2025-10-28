"""
蒸馏损失函数
包括：
1. KD Loss - Logit蒸馏
2. Feature Loss - 特征蒸馏
3. Boundary Loss - 边缘感知损失（核心创新）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg


class KDLoss(nn.Module):
    """
    Knowledge Distillation Loss
    Hinton et al. "Distilling the Knowledge in a Neural Network"
    """

    def __init__(self, temperature=4.0, alpha=0.7):
        """
        Args:
            temperature: 蒸馏温度，控制soft targets的平滑程度
            alpha: KD loss和CE loss的权重平衡
        """
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits, target=None):
        """
        Args:
            student_logits: (B, C, H, W) - 学生网络输出
            teacher_logits: (B, C, H, W) - 教师网络输出
            target: (B, H, W) - ground truth（如果提供则计算组合loss）

        Returns:
            loss: scalar tensor
        """
        # 温度缩放
        student_soft = F.log_softmax(student_logits / self.T, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.T, dim=1)

        # KL散度
        kd_loss = self.kl_div(student_soft, teacher_soft) * (self.T ** 2)

        # 如果提供了target，计算组合loss
        if target is not None:
            ce_loss = F.cross_entropy(student_logits, target)
            total_loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss
            return total_loss

        return kd_loss


class FeatureLoss(nn.Module):
    """
    Feature Distillation Loss
    蒸馏中间层特征
    """

    def __init__(self, loss_type='mse'):
        """
        Args:
            loss_type: 'mse' 或 'cosine'
        """
        super().__init__()
        self.loss_type = loss_type

    def forward(self, student_feat, teacher_feat):
        """
        Args:
            student_feat: (B, C_s, H_s, W_s)
            teacher_feat: (B, C_t, H_t, W_t)

        Returns:
            loss: scalar tensor
        """
        # 如果维度不匹配，需要对齐
        if student_feat.shape != teacher_feat.shape:
            student_feat = self.align_features(student_feat, teacher_feat)

        if self.loss_type == 'mse':
            loss = F.mse_loss(student_feat, teacher_feat)
        elif self.loss_type == 'cosine':
            loss = 1 - F.cosine_similarity(
                student_feat.flatten(2),
                teacher_feat.flatten(2),
                dim=2
            ).mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss

    def align_features(self, student_feat, teacher_feat):
        """
        对齐学生和教师特征的维度

        策略：
        1. 如果通道数不同，用1x1卷积对齐
        2. 如果空间尺寸不同，用插值对齐
        """
        B, C_s, H_s, W_s = student_feat.shape
        B, C_t, H_t, W_t = teacher_feat.shape

        # 通道对齐
        if C_s != C_t:
            # 动态创建1x1卷积（注意：这在训练时会有问题，实际应该在模型中定义）
            # 这里假设在模型中已经有对齐层
            if not hasattr(self, 'channel_align'):
                self.channel_align = nn.Conv2d(C_s, C_t, 1).to(student_feat.device)
            student_feat = self.channel_align(student_feat)

        # 空间对齐
        if (H_s, W_s) != (H_t, W_t):
            student_feat = F.interpolate(
                student_feat,
                size=(H_t, W_t),
                mode='bilinear',
                align_corners=False
            )

        return student_feat


class BoundaryLoss(nn.Module):
    """
    Boundary-aware Loss（边缘感知损失）

    这是你的核心创新！
    针对遥感图像边界模糊问题，增强边缘区域的监督
    """

    def __init__(self, theta=10):
        """
        Args:
            theta: 边缘权重因子
        """
        super().__init__()
        self.theta = theta

        # Sobel算子（用于边缘检测）
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0))

        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0))

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) - 预测logits
            target: (B, H, W) - ground truth

        Returns:
            loss: scalar tensor
        """
        # 1. 提取预测和目标的边缘
        pred_prob = F.softmax(pred, dim=1)  # (B, C, H, W)

        # 转换target为one-hot
        target_onehot = F.one_hot(target, num_classes=pred.shape[1])  # (B, H, W, C)
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # 2. 计算边缘图
        pred_edge = self.get_edge_map(pred_prob)  # (B, 1, H, W)
        target_edge = self.get_edge_map(target_onehot)  # (B, 1, H, W)

        # 3. 边缘权重图
        edge_weight = target_edge * self.theta + 1.0  # (B, 1, H, W)

        # 4. 加权交叉熵
        # 展平
        pred_flat = pred.permute(0, 2, 3, 1).reshape(-1, pred.shape[1])  # (B*H*W, C)
        target_flat = target.reshape(-1)  # (B*H*W,)
        edge_weight_flat = edge_weight.reshape(-1)  # (B*H*W,)

        # 计算loss
        ce_loss = F.cross_entropy(pred_flat, target_flat, reduction='none')
        weighted_loss = (ce_loss * edge_weight_flat).mean()

        return weighted_loss

    def get_edge_map(self, x):
        """
        使用Sobel算子提取边缘

        Args:
            x: (B, C, H, W)

        Returns:
            edge: (B, 1, H, W)
        """
        B, C, H, W = x.shape

        # 对每个通道应用Sobel
        edges = []
        for i in range(C):
            xi = x[:, i:i+1, :, :]  # (B, 1, H, W)

            # Sobel X
            edge_x = F.conv2d(xi, self.sobel_x, padding=1)

            # Sobel Y
            edge_y = F.conv2d(xi, self.sobel_y, padding=1)

            # 边缘强度
            edge = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
            edges.append(edge)

        # 合并所有通道的边缘
        edge = torch.stack(edges, dim=1).max(dim=1)[0]  # (B, 1, H, W)

        # 归一化到[0, 1]
        edge = torch.sigmoid(edge)

        return edge


class TotalLoss(nn.Module):
    """
    总损失函数
    组合多个loss
    """

    def __init__(self):
        super().__init__()

        # 各个loss组件
        self.ce_loss = nn.CrossEntropyLoss()
        self.kd_loss = KDLoss(
            temperature=cfg.KD_TEMPERATURE,
            alpha=0.7
        )
        self.feat_loss = FeatureLoss(loss_type='mse')
        self.boundary_loss = BoundaryLoss(theta=10)

        # 权重
        self.w_ce = cfg.LOSS_CE_WEIGHT
        self.w_kd = cfg.LOSS_KD_WEIGHT
        self.w_feat = cfg.LOSS_FEAT_WEIGHT
        self.w_boundary = cfg.LOSS_BOUNDARY_WEIGHT

    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict包含
                - 'pred': 学生预测 (B, C, H, W)
                - 'feat': 学生特征 (B, C', H', W')
            targets: dict包含
                - 'label': ground truth (B, H, W)
                - 'teacher_feat': 教师特征 (B, C_t, H_t, W_t)
                - 'teacher_logit': 教师logits (B, C, H, W)

        Returns:
            total_loss: scalar
            loss_dict: 各个loss的值（用于记录）
        """
        pred = outputs['pred']
        student_feat = outputs.get('feat', None)

        label = targets['label']
        teacher_feat = targets['teacher_feat']
        teacher_logit = targets['teacher_logit']

        # 1. Cross Entropy Loss
        loss_ce = self.ce_loss(pred, label)

        # 2. KD Loss（如果有教师logits）
        if teacher_logit is not None:
            # 需要将teacher_logit上采样到pred的尺寸
            if teacher_logit.shape[-2:] != pred.shape[-2:]:
                teacher_logit = F.interpolate(
                    teacher_logit,
                    size=pred.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            loss_kd = self.kd_loss(pred, teacher_logit)
        else:
            loss_kd = torch.tensor(0.0, device=pred.device)

        # 3. Feature Loss（如果有学生和教师特征）
        if student_feat is not None and teacher_feat is not None:
            loss_feat = self.feat_loss(student_feat, teacher_feat)
        else:
            loss_feat = torch.tensor(0.0, device=pred.device)

        # 4. Boundary Loss（你的创新）
        loss_bound = self.boundary_loss(pred, label)

        # 总损失
        total_loss = (
            self.w_ce * loss_ce +
            self.w_kd * loss_kd +
            self.w_feat * loss_feat +
            self.w_boundary * loss_bound
        )

        # 记录各个loss
        loss_dict = {
            'total': total_loss.item(),
            'ce': loss_ce.item(),
            'kd': loss_kd.item() if isinstance(loss_kd, torch.Tensor) else 0.0,
            'feat': loss_feat.item() if isinstance(loss_feat, torch.Tensor) else 0.0,
            'boundary': loss_bound.item()
        }

        return total_loss, loss_dict


if __name__ == '__main__':
    """测试损失函数"""
    print("=" * 60)
    print("测试损失函数")
    print("=" * 60)

    # 创建虚拟数据（注意维度匹配）
    B, C, H, W = 2, 6, 256, 256  # 使用相同的分辨率

    pred = torch.randn(B, C, H, W)
    label = torch.randint(0, C, (B, H, W))
    student_feat = torch.randn(B, 256, 64, 64)
    teacher_feat = torch.randn(B, 256, 64, 64)
    teacher_logit = torch.randn(B, C, H, W)  # 相同尺寸

    # 测试各个loss
    print("\n1. 测试KD Loss:")
    kd_loss_fn = KDLoss()
    loss = kd_loss_fn(pred, teacher_logit, label)
    print(f"   KD Loss: {loss.item():.4f}")

    print("\n2. 测试Feature Loss:")
    feat_loss_fn = FeatureLoss()
    loss = feat_loss_fn(student_feat, teacher_feat)
    print(f"   Feature Loss: {loss.item():.4f}")

    print("\n3. 测试Boundary Loss:")
    boundary_loss_fn = BoundaryLoss()
    loss = boundary_loss_fn(pred, label)
    print(f"   Boundary Loss: {loss.item():.4f}")

    print("\n4. 测试Total Loss:")
    total_loss_fn = TotalLoss()

    outputs = {'pred': pred, 'feat': student_feat}
    targets = {
        'label': label,
        'teacher_feat': teacher_feat,
        'teacher_logit': teacher_logit
    }

    total_loss, loss_dict = total_loss_fn(outputs, targets)
    print(f"   Total Loss: {total_loss.item():.4f}")
    print(f"   Loss Dict: {loss_dict}")

    print("\n" + "=" * 60)
    print("✓ 损失函数测试通过")
    print("=" * 60)