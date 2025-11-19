"""
轻量级ASPP模块 (Lightweight Atrous Spatial Pyramid Pooling)

设计思路：
1. 在decoder输出的fused feature上增强多尺度上下文
2. 不破坏已验证的diff feature计算
3. 使用depthwise separable conv降低参数量

参考文献：
- DeepLabv3+: Encoder-Decoder with Atrous Separable Convolution
- 变化检测中的应用：捕获不同尺度的变化区域（小建筑、大片区域等）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv2d(nn.Module):
    """深度可分离卷积（Depthwise Separable Convolution）"""
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        # Depthwise conv: 每个通道独立卷积
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=in_channels, bias=bias
        )
        # Pointwise conv: 1×1卷积融合通道
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class LightweightASPP(nn.Module):
    """
    轻量级空洞空间金字塔池化

    设计：
    - 分支1: 1×1 conv（保持原始特征）
    - 分支2: 3×3 conv dilation=2（中等感受野）
    - 分支3: 3×3 conv dilation=4（大感受野）
    - 分支4: Global Average Pooling（全局上下文）

    每个分支输出64 channels，concat后256 channels，再融合到out_channels
    """
    def __init__(self, in_channels=256, out_channels=256, dilations=[1, 2, 4]):
        """
        Args:
            in_channels: 输入通道数（decoder输出）
            out_channels: 输出通道数
            dilations: 膨胀率列表
        """
        super().__init__()

        self.dilations = dilations
        branch_channels = 64  # 每个分支的输出通道数

        # 分支1: 1×1 conv（保持原始）
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        # 分支2-N: 3×3 separable conv with different dilations
        self.atrous_convs = nn.ModuleList()
        for dilation in dilations[1:]:  # 跳过dilation=1（已在branch1）
            self.atrous_convs.append(
                SeparableConv2d(
                    in_channels, branch_channels,
                    kernel_size=3,
                    padding=dilation,  # padding = dilation保持尺寸
                    dilation=dilation
                )
            )

        # 分支4: Global Average Pooling
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, branch_channels, 1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        # 融合所有分支
        # 总通道数 = branch1(64) + atrous_convs(64*2) + global(64) = 256
        total_channels = branch_channels * (len(dilations) + 1)
        self.fusion = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)  # 轻微dropout防止过拟合
        )

    def forward(self, x):
        """
        Args:
            x: 输入特征 (B, C, H, W)
        Returns:
            out: ASPP增强后的特征 (B, C, H, W)
        """
        B, C, H, W = x.shape

        # 分支1: 1×1
        branch1_out = self.branch1(x)

        # 分支2-N: atrous convs
        atrous_outs = [conv(x) for conv in self.atrous_convs]

        # 分支4: global pooling
        global_out = self.global_pool(x)
        # 上采样回原尺寸
        global_out = F.interpolate(global_out, size=(H, W),
                                   mode='bilinear', align_corners=False)

        # 拼接所有分支
        all_branches = [branch1_out] + atrous_outs + [global_out]
        concat = torch.cat(all_branches, dim=1)

        # 融合
        out = self.fusion(concat)

        return out


class SimplifiedASPP(nn.Module):
    """
    简化版ASPP（更轻量，备选方案）

    只使用3个分支：1×1, 3×3 dilation=2, global pooling
    参数量更少，适合快速实验
    """
    def __init__(self, in_channels=256, out_channels=256):
        super().__init__()

        branch_channels = 64

        # 分支1: 1×1
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        # 分支2: 3×3 dilation=2
        self.branch2 = SeparableConv2d(
            in_channels, branch_channels,
            kernel_size=3, padding=2, dilation=2
        )

        # 分支3: Global pooling
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, branch_channels, 1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        # 融合：3×64 = 192 channels
        self.fusion = nn.Sequential(
            nn.Conv2d(branch_channels * 3, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.global_pool(x)
        b3 = F.interpolate(b3, size=(H, W), mode='bilinear', align_corners=False)

        concat = torch.cat([b1, b2, b3], dim=1)
        out = self.fusion(concat)

        return out


# 测试参数量
if __name__ == '__main__':
    # 测试LightweightASPP
    model = LightweightASPP(in_channels=256, out_channels=256)
    x = torch.randn(2, 256, 64, 64)
    y = model(x)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"LightweightASPP:")
    print(f"  Input: {x.shape}")
    print(f"  Output: {y.shape}")
    print(f"  Parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    # 测试SimplifiedASPP
    model2 = SimplifiedASPP(in_channels=256, out_channels=256)
    y2 = model2(x)

    total_params2 = sum(p.numel() for p in model2.parameters())
    print(f"\nSimplifiedASPP:")
    print(f"  Input: {x.shape}")
    print(f"  Output: {y2.shape}")
    print(f"  Parameters: {total_params2:,} ({total_params2/1e6:.2f}M)")
