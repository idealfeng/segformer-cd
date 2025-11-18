"""
多方向差分模块 (Multi-Direction Difference Module)

灵感来源：RS-Mamba的全向扫描
核心思想：遥感图像的变化可能沿任意方向，需要多方向捕获
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiDirectionDiffModule(nn.Module):
    """
    多方向差分模块

    在4个主要方向计算差异特征：
    1. 横向 (Horizontal)
    2. 纵向 (Vertical)
    3. 对角 (Diagonal ↘)
    4. 反对角 (Anti-diagonal ↙)

    每个方向的差异特征都包含独特的空间信息
    """
    def __init__(self, channels):
        super().__init__()

        # 每个方向独立的卷积权重
        self.dir_convs = nn.ModuleList([
            # 横向
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ),
            # 纵向
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ),
            # 对角
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ),
            # 反对角
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ),
        ])

        # 方向注意力：学习每个方向的重要性
        self.direction_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 4, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 4, 1),  # 4个方向的权重
            nn.Softmax(dim=1)  # 归一化权重
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat_a, feat_b, guide=None):
        """
        Args:
            feat_a: 时刻A特征 (B, C, H, W)
            feat_b: 时刻B特征 (B, C, H, W)
            guide: 语义引导置信图 (忽略，保持接口兼容)
        Returns:
            fused_diff: 融合的多方向差异特征 (B, C, H, W)
        """
        B, C, H, W = feat_a.shape

        # 1. 横向差异（原始）
        diff_h = torch.abs(feat_a - feat_b)
        diff_h = self.dir_convs[0](diff_h)

        # 2. 纵向差异（转置）
        # 交换H和W维度，计算差异，再转回来
        feat_a_t = feat_a.transpose(-1, -2)  # (B, C, W, H)
        feat_b_t = feat_b.transpose(-1, -2)
        diff_v = torch.abs(feat_a_t - feat_b_t).transpose(-1, -2)  # 转回 (B, C, H, W)
        diff_v = self.dir_convs[1](diff_v)

        # 3. 对角差异（↘ 方向）
        # 使用shift近似对角采样
        diff_d1 = torch.abs(
            self._shift(feat_a, 1, 1) - self._shift(feat_b, 1, 1)
        )
        diff_d1 = self.dir_convs[2](diff_d1)

        # 4. 反对角差异（↙ 方向）
        diff_d2 = torch.abs(
            self._shift(feat_a, 1, -1) - self._shift(feat_b, 1, -1)
        )
        diff_d2 = self.dir_convs[3](diff_d2)

        # 拼接所有方向
        all_diffs = torch.cat([diff_h, diff_v, diff_d1, diff_d2], dim=1)  # (B, 4C, H, W)

        # 计算方向注意力权重
        dir_weights = self.direction_attention(all_diffs)  # (B, 4, 1, 1)

        # 加权融合（可选）
        # 注释掉加权，直接拼接融合效果可能更好
        # weighted_diffs = []
        # for i in range(4):
        #     weighted_diffs.append(dir_weights[:, i:i+1] * [diff_h, diff_v, diff_d1, diff_d2][i])
        # all_diffs = torch.cat(weighted_diffs, dim=1)

        # 融合
        fused_diff = self.fusion(all_diffs)

        return fused_diff

    def _shift(self, x, shift_h, shift_w):
        """
        平移特征图（用于近似对角采样）

        Args:
            x: 输入特征 (B, C, H, W)
            shift_h: 垂直平移量
            shift_w: 水平平移量
        Returns:
            shifted: 平移后的特征
        """
        return torch.roll(x, shifts=(shift_h, shift_w), dims=(-2, -1))


class SimplifiedMultiDirectionDiff(nn.Module):
    """
    简化版多方向差分（更轻量，适合快速实验）

    只计算差异，不用独立卷积，参数量更少
    """
    def __init__(self, channels):
        super().__init__()

        # 直接融合4个方向的差异
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat_a, feat_b, guide=None):
        """
        Args:
            feat_a: 时刻A特征
            feat_b: 时刻B特征
            guide: 语义引导置信图 (忽略，保持接口兼容)
        """
        # 4个方向的差异
        diff_h = torch.abs(feat_a - feat_b)
        diff_v = torch.abs(feat_a.transpose(-1, -2) - feat_b.transpose(-1, -2)).transpose(-1, -2)
        diff_d1 = torch.abs(self._shift(feat_a, 1, 1) - self._shift(feat_b, 1, 1))
        diff_d2 = torch.abs(self._shift(feat_a, 1, -1) - self._shift(feat_b, 1, -1))

        # 直接拼接融合
        all_diffs = torch.cat([diff_h, diff_v, diff_d1, diff_d2], dim=1)
        fused_diff = self.fusion(all_diffs)

        return fused_diff

    def _shift(self, x, shift_h, shift_w):
        return torch.roll(x, shifts=(shift_h, shift_w), dims=(-2, -1))
