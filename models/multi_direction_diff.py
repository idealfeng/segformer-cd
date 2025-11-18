"""
条带式感受野模块 (Strip Context Module)

核心思想：
遥感图像中的变化往往呈现线性结构（道路、长条建筑、场地边界、成排厂房）
使用条带式卷积（strip pooling）在横向和纵向获得长距离感受野

设计原则：
1. 横向strip (1×k): 捕获水平方向的长条结构
2. 纵向strip (k×1): 捕获垂直方向的长条结构
3. 局部conv (3×3): 保留局部上下文作为基线
4. 极度轻量：depthwise实现，参数开销<0.03%
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class StripContextModule(nn.Module):
    """
    条带式感受野模块

    使用depthwise strip convolution在不同方向扩大感受野：
    - 1×k depthwise conv: 横向长距离感受野（捕获水平道路、建筑边界等）
    - k×1 depthwise conv: 纵向长距离感受野（捕获垂直道路、建筑边界等）
    - 3×3 depthwise conv: 局部上下文基线

    参数量极小，效果显著
    """
    def __init__(self, channels, strip_size=11):
        """
        Args:
            channels: 特征通道数
            strip_size: 条带卷积核大小（推荐7, 11, 15）
        """
        super().__init__()
        self.strip_size = strip_size

        # 分支1: 局部上下文 (3×3 depthwise)
        self.local_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # 分支2: 横向条带 (1×k depthwise) - 捕获水平结构
        self.horizontal_strip = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, strip_size),
                     padding=(0, strip_size // 2), groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # 分支3: 纵向条带 (k×1 depthwise) - 捕获垂直结构
        self.vertical_strip = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(strip_size, 1),
                     padding=(strip_size // 2, 0), groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # 融合层：3个分支拼接后压缩回原通道数
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
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
            fused_diff: 融合的条带式上下文差异特征 (B, C, H, W)
        """
        # 首先计算差异特征
        diff = torch.abs(feat_a - feat_b)

        # 三个分支分别处理差异特征
        local_feat = self.local_conv(diff)          # 局部 3×3
        h_strip_feat = self.horizontal_strip(diff)  # 横向 1×k
        v_strip_feat = self.vertical_strip(diff)    # 纵向 k×1

        # 拼接三个分支
        concat_feat = torch.cat([local_feat, h_strip_feat, v_strip_feat], dim=1)

        # 1×1 融合
        fused_diff = self.fusion(concat_feat)

        return fused_diff


class StripContextModuleV2(nn.Module):
    """
    增强版条带式感受野模块（备选方案）

    额外加入全局池化分支，提供全局统计信息
    如果V1版本效果不够好，可以尝试这个版本
    """
    def __init__(self, channels, strip_size=11):
        super().__init__()
        self.strip_size = strip_size

        # 分支1: 局部上下文 (3×3 depthwise)
        self.local_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # 分支2: 横向条带 (1×k depthwise)
        self.horizontal_strip = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, strip_size),
                     padding=(0, strip_size // 2), groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # 分支3: 纵向条带 (k×1 depthwise)
        self.vertical_strip = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(strip_size, 1),
                     padding=(strip_size // 2, 0), groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # 分支4: 全局上下文 (Global Average Pooling)
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # 融合层：4个分支拼接后压缩
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat_a, feat_b, guide=None):
        """
        Args:
            feat_a: 时刻A特征
            feat_b: 时刻B特征
            guide: 语义引导置信图 (忽略)
        """
        B, C, H, W = feat_a.shape
        diff = torch.abs(feat_a - feat_b)

        # 四个分支
        local_feat = self.local_conv(diff)
        h_strip_feat = self.horizontal_strip(diff)
        v_strip_feat = self.vertical_strip(diff)
        global_feat = self.global_context(diff)

        # 全局特征上采样回原尺寸
        global_feat = global_feat.expand(-1, -1, H, W)

        # 拼接并融合
        concat_feat = torch.cat([local_feat, h_strip_feat, v_strip_feat, global_feat], dim=1)
        fused_diff = self.fusion(concat_feat)

        return fused_diff


# ===== 保留旧版本以便对比实验 =====

class MultiDirectionDiffModule(nn.Module):
    """
    [已废弃] 多方向差分模块（基于torch.roll的版本）

    实验结果：感受野太小（只shift 1像素），效果不佳
    保留此代码仅供对比实验用
    """
    def __init__(self, channels):
        super().__init__()

        # 每个方向独立的卷积权重
        self.dir_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ) for _ in range(4)
        ])

        # 方向注意力
        self.direction_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 4, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 4, 1),
            nn.Softmax(dim=1)
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
        diff_h = torch.abs(feat_a - feat_b)
        diff_h = self.dir_convs[0](diff_h)

        feat_a_t = feat_a.transpose(-1, -2)
        feat_b_t = feat_b.transpose(-1, -2)
        diff_v = torch.abs(feat_a_t - feat_b_t).transpose(-1, -2)
        diff_v = self.dir_convs[1](diff_v)

        diff_d1 = torch.abs(self._shift(feat_a, 1, 1) - self._shift(feat_b, 1, 1))
        diff_d1 = self.dir_convs[2](diff_d1)

        diff_d2 = torch.abs(self._shift(feat_a, 1, -1) - self._shift(feat_b, 1, -1))
        diff_d2 = self.dir_convs[3](diff_d2)

        all_diffs = torch.cat([diff_h, diff_v, diff_d1, diff_d2], dim=1)
        fused_diff = self.fusion(all_diffs)

        return fused_diff

    def _shift(self, x, shift_h, shift_w):
        return torch.roll(x, shifts=(shift_h, shift_w), dims=(-2, -1))


class SimplifiedMultiDirectionDiff(nn.Module):
    """
    [已废弃] 简化版多方向差分

    现在使用 StripContextModule 替代
    """
    def __init__(self, channels):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat_a, feat_b, guide=None):
        diff_h = torch.abs(feat_a - feat_b)
        diff_v = torch.abs(feat_a.transpose(-1, -2) - feat_b.transpose(-1, -2)).transpose(-1, -2)
        diff_d1 = torch.abs(self._shift(feat_a, 1, 1) - self._shift(feat_b, 1, 1))
        diff_d2 = torch.abs(self._shift(feat_a, 1, -1) - self._shift(feat_b, 1, -1))

        all_diffs = torch.cat([diff_h, diff_v, diff_d1, diff_d2], dim=1)
        fused_diff = self.fusion(all_diffs)

        return fused_diff

    def _shift(self, x, shift_h, shift_w):
        return torch.roll(x, shifts=(shift_h, shift_w), dims=(-2, -1))
