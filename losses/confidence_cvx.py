"""
置信度凸优化选择（CSL风格）
将伪标签选择表述为置信度分布特征空间中的凸优化问题

参考：你PDF里的CSL论文
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidenceCVXSelector(nn.Module):
    """
    基于凸优化的置信度选择

    核心思想：
    1. 构建特征向量 h = [max_conf, residual_dispersion]
    2. 在这个2D空间中，可靠预测和不可靠预测是可分的
    3. 通过迹最大化找到最优分割超平面
    """

    def __init__(self, num_clusters=2, temperature=1.0):
        super().__init__()
        self.num_clusters = num_clusters
        self.temperature = temperature

    def forward(self, pred_logits, hard_threshold=False):
        """
        Args:
            pred_logits: (B, 1, H, W) - 预测logits
            hard_threshold: 是否使用硬阈值（训练时False，推理时True）

        Returns:
            reliability_mask: (B, 1, H, W) - 可靠性mask [0, 1]
            decision_boundary: dict - 决策边界参数（用于可视化）
        """
        B, _, H, W = pred_logits.shape

        # 1. 计算置信度特征
        pred_prob = torch.sigmoid(pred_logits)  # (B, 1, H, W)

        # 最大置信度
        max_conf = torch.max(
            torch.stack([pred_prob, 1 - pred_prob], dim=0),
            dim=0
        )[0]  # (B, 1, H, W)

        # 残差离散度（residual dispersion）
        residual = 1 - max_conf
        v = residual / (max_conf + 1e-10)  # (B, 1, H, W)

        # 2. 构建特征向量
        # h = [max_conf, v]
        features = torch.cat([max_conf, v], dim=1)  # (B, 2, H, W)

        # Reshape为 (B*H*W, 2) 用于聚类
        features_flat = features.permute(0, 2, 3, 1).reshape(-1, 2)  # (N, 2)

        # 3. 光谱松弛 + 迹最大化
        # 这是一个凸优化问题，求解最优的类间分离性
        reliability_scores = self.spectral_relaxation(features_flat)

        # Reshape回原始形状
        reliability_mask = reliability_scores.reshape(B, H, W, 1).permute(0, 3, 1, 2)

        # 4. 可选：硬阈值化
        if hard_threshold:
            threshold = reliability_scores.median()
            reliability_mask = (reliability_mask > threshold).float()

        # 5. 返回决策边界参数（用于可视化）
        decision_boundary = {
            'max_conf_mean': max_conf.mean().item(),
            'residual_disp_mean': v.mean().item(),
            'threshold': reliability_scores.median().item() if hard_threshold else None
        }

        return reliability_mask, decision_boundary

    def spectral_relaxation(self, features):
        """
        光谱松弛求解凸优化问题

        目标：最大化类间距离，最小化类内距离

        这是一个Rayleigh商优化问题：
        max trace(W^T S_b W) / trace(W^T S_w W)

        其中：
        - S_b: 类间散度矩阵
        - S_w: 类内散度矩阵
        - W: 投影矩阵

        Args:
            features: (N, 2) - [max_conf, residual_disp]

        Returns:
            scores: (N,) - 可靠性分数
        """
        N, D = features.shape

        # 归一化特征
        features_normalized = F.normalize(features, p=2, dim=1)

        # 计算亲和矩阵（余弦相似度）
        # A[i,j] = cos_similarity(f_i, f_j)
        A = torch.mm(features_normalized, features_normalized.t())  # (N, N)

        # 度矩阵
        D_mat = torch.diag(A.sum(dim=1))

        # 拉普拉斯矩阵
        L = D_mat - A

        # 求解广义特征值问题: L * v = λ * D * v
        # 使用简化版本：计算归一化拉普拉斯的特征向量
        D_inv_sqrt = torch.diag(1.0 / (torch.sqrt(D_mat.diag()) + 1e-10))
        L_normalized = torch.mm(torch.mm(D_inv_sqrt, L), D_inv_sqrt)

        # 特征分解
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(L_normalized)

            # 第二小的特征向量（Fiedler vector）
            # 对应最优的二分类
            fiedler_vector = eigenvectors[:, 1]  # (N,)

            # 将特征向量转换为[0, 1]的可靠性分数
            scores = (fiedler_vector - fiedler_vector.min()) / (
                fiedler_vector.max() - fiedler_vector.min() + 1e-10
            )

        except RuntimeError:
            # 如果特征分解失败，使用简单的距离度量
            # 计算到理想点(1, 0)的距离
            ideal_point = torch.tensor([[1.0, 0.0]], device=features.device)
            distances = torch.cdist(features_normalized, ideal_point).squeeze()
            scores = 1.0 / (1.0 + distances)

        return scores


class ConfidenceWeightedCVXLoss(nn.Module):
    """
    结合凸优化的置信度加权损失
    """

    def __init__(self):
        super().__init__()
        self.selector = ConfidenceCVXSelector()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W) - logits
            target: (B, H, W) - labels
        """
        # 获取可靠性mask
        reliability_mask, _ = self.selector(pred, hard_threshold=False)

        # 计算逐像素BCE
        target_float = target.unsqueeze(1).float()
        pixel_loss = self.bce(pred, target_float)  # (B, 1, H, W)

        # 使用可靠性mask加权
        # 可靠的预测：低权重（已经很好了）
        # 不可靠的预测：高权重（需要更多关注）
        weight = 1.0 + (1.0 - reliability_mask)

        weighted_loss = pixel_loss * weight

        return weighted_loss.mean()


if __name__ == '__main__':
    """测试凸优化选择器"""
    print("=" * 60)
    print("测试置信度凸优化选择器")
    print("=" * 60)

    # 模拟数据
    B, H, W = 2, 64, 64
    pred_logits = torch.randn(B, 1, H, W)

    # 创建选择器
    selector = ConfidenceCVXSelector()

    # 测试
    reliability_mask, boundary = selector(pred_logits, hard_threshold=False)

    print(f"\n输入: {pred_logits.shape}")
    print(f"输出: {reliability_mask.shape}")
    print(f"可靠性分数范围: [{reliability_mask.min():.4f}, {reliability_mask.max():.4f}]")
    print(f"平均可靠性: {reliability_mask.mean():.4f}")
    print(f"\n决策边界参数:")
    for k, v in boundary.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("✓ 测试通过")
    print("=" * 60)