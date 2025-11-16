"""
SegFormer变化检测模型 - Siamese架构 + 多尺度差异融合
核心创新点：
1. 共享权重的Siamese编码器
2. 多尺度时序差异模块 (MTDM)
3. 通道注意力增强的差异特征
4. 深度监督辅助损失
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerModel
from config import cfg


class ChannelAttention(nn.Module):
    """通道注意力模块 (SE-style)"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class DifferenceModule(nn.Module):
    """差异计算模块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 差异特征融合
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 通道注意力
        self.ca = ChannelAttention(out_channels)

    def forward(self, feat_a, feat_b):
        # 计算差异特征
        diff = torch.abs(feat_a - feat_b)
        # 拼接原始特征和差异特征
        concat = torch.cat([diff, feat_a + feat_b], dim=1)
        # 融合
        out = self.conv(concat)
        # 注意力增强
        out = self.ca(out)
        return out


class MLPDecoder(nn.Module):
    """轻量级MLP解码器"""
    def __init__(self, in_channels_list, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim

        # 每个尺度的MLP
        self.linear_layers = nn.ModuleList([
            nn.Conv2d(in_ch, embed_dim, 1) for in_ch in in_channels_list
        ])

        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels_list), embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        """
        Args:
            features: list of tensors at different scales
        """
        # 获取目标尺寸（最大的特征图尺寸）
        target_size = features[0].shape[2:]

        # 统一尺寸并映射到embed_dim
        aligned_features = []
        for i, feat in enumerate(features):
            feat = self.linear_layers[i](feat)
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            aligned_features.append(feat)

        # 拼接
        fused = torch.cat(aligned_features, dim=1)
        # 融合
        out = self.fusion(fused)
        return out


class SegFormerCD(nn.Module):
    """
    SegFormer变化检测模型

    架构:
    Image A ─→ [SegFormer Encoder] ─→ 4级特征 {F1_a, F2_a, F3_a, F4_a}
                                              ↓
                                    [Multi-scale Difference]
                                              ↑
    Image B ─→ [SegFormer Encoder] ─→ 4级特征 {F1_b, F2_b, F3_b, F4_b}
                                              ↓
                                      [MLP Decoder]
                                              ↓
                                        Change Map
    """

    def __init__(self, variant='b1', pretrained=True, num_classes=2):
        super().__init__()

        self.num_classes = num_classes

        # 加载预训练的SegFormer骨干
        model_names = {f'b{i}': f'nvidia/mit-b{i}' for i in range(6)}
        model_name = model_names[variant]

        # 检查本地权重
        local_weights_path = os.path.join("pretrained_weights", f"segformer_{variant}")
        if os.path.exists(local_weights_path):
            print(f"Loading local weights from '{local_weights_path}'...")
            model_name = local_weights_path
        else:
            print(f"Downloading from Hugging Face Hub '{model_name}'...")

        print(f"Building SegFormer-{variant.upper()} for Change Detection...")

        # Siamese编码器（共享权重）
        self.encoder = SegformerModel.from_pretrained(model_name)

        # 获取各stage通道数
        self.channels = self.encoder.config.hidden_sizes  # e.g., [64, 128, 320, 512] for B1

        # 多尺度差异模块
        self.diff_modules = nn.ModuleList([
            DifferenceModule(ch, ch) for ch in self.channels
        ])

        # MLP解码器
        self.decoder = MLPDecoder(self.channels, embed_dim=256)

        # 最终分类头
        self.classifier = nn.Conv2d(256, num_classes, 1)

        # 深度监督头（可选）
        if cfg.DEEP_SUPERVISION:
            self.aux_heads = nn.ModuleList([
                nn.Conv2d(ch, num_classes, 1) for ch in self.channels
            ])
        else:
            self.aux_heads = None

        # 只初始化新增模块，不要动预训练的encoder！
        self.diff_modules.apply(self._init_weights)
        self.decoder.apply(self._init_weights)
        self.classifier.apply(self._init_weights)
        if self.aux_heads is not None:
            self.aux_heads.apply(self._init_weights)

        print(f"Model built successfully!")
        print(f"  Encoder channels: {self.channels}")
        print(f"  Deep supervision: {cfg.DEEP_SUPERVISION}")
        print(f"  Output classes: {num_classes}")

    def _init_weights(self, m):
        """初始化新增层的权重（只对新模块调用）"""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def _extract_features(self, x):
        """提取多尺度特征"""
        B, C, H, W = x.shape

        # 通过编码器
        outputs = self.encoder(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True
        )

        # 获取所有hidden states
        # hidden_states = [embedding输出] + [每层Transformer block输出]
        # 例如B1的depths=[2,2,2,2]，hidden_states长度=1+2+2+2+2=9
        all_hidden = outputs.hidden_states
        depths = self.encoder.config.depths  # e.g., [2, 2, 2, 2] for B1

        # 计算每个stage最后一层在hidden_states里的索引
        # 跳过index 0（embedding输出），取每个stage的最后一层
        stage_indices = []
        acc = 0
        for d in depths:
            acc += d
            stage_indices.append(acc)  # index 2, 4, 6, 8 for B1

        # 计算每个stage的特征图尺寸
        h_sizes = [H // 4, H // 8, H // 16, H // 32]
        w_sizes = [W // 4, W // 8, W // 16, W // 32]

        # 只取4个stage的最后输出，reshape为4D特征图
        features = []
        for i, idx in enumerate(stage_indices):
            hidden_state = all_hidden[idx]

            # 处理不同格式的hidden_state
            if hidden_state.ndim == 3:
                # (B, N, C) -> (B, C, H, W)
                B_, N, C_ = hidden_state.shape
                feat = hidden_state.permute(0, 2, 1).reshape(B_, C_, h_sizes[i], w_sizes[i])
            else:
                # 已经是 (B, C, H, W) 格式
                feat = hidden_state

            features.append(feat)

        return features  # 长度4，对应4个尺度

    def forward(self, img_a, img_b):
        """
        Args:
            img_a: 时刻1图像 (B, 3, H, W)
            img_b: 时刻2图像 (B, 3, H, W)

        Returns:
            dict: {
                'pred': 主输出 (B, num_classes, H, W)
                'aux_preds': 辅助输出列表（深度监督时）
            }
        """
        B, _, H, W = img_a.shape

        # 提取两个时刻的特征
        feats_a = self._extract_features(img_a)
        feats_b = self._extract_features(img_b)

        # 计算多尺度差异特征
        diff_feats = []
        for i in range(len(self.channels)):
            diff_feat = self.diff_modules[i](feats_a[i], feats_b[i])
            diff_feats.append(diff_feat)

        # 解码
        decoded = self.decoder(diff_feats)

        # 分类
        logits = self.classifier(decoded)

        # 上采样到原始尺寸
        pred = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)

        result = {'pred': pred}

        # 深度监督
        if self.training and self.aux_heads is not None:
            aux_preds = []
            for i, aux_head in enumerate(self.aux_heads):
                aux_logit = aux_head(diff_feats[i])
                aux_pred = F.interpolate(aux_logit, size=(H, W), mode='bilinear', align_corners=False)
                aux_preds.append(aux_pred)
            result['aux_preds'] = aux_preds

        return result

    def get_params_info(self):
        """获取参数信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total': total_params / 1e6,
            'trainable': trainable_params / 1e6
        }


def build_model(variant='b1', pretrained=True, num_classes=2):
    """
    构建变化检测模型

    Args:
        variant: SegFormer变体 (b0-b5)
        pretrained: 是否使用预训练权重
        num_classes: 类别数

    Returns:
        model: SegFormerCD模型
    """
    model = SegFormerCD(
        variant=variant,
        pretrained=pretrained,
        num_classes=num_classes
    )

    # 打印参数信息
    params_info = model.get_params_info()
    print(f"\nModel Parameters:")
    print(f"  Total: {params_info['total']:.2f}M")
    print(f"  Trainable: {params_info['trainable']:.2f}M")

    return model


if __name__ == '__main__':
    """测试模型"""
    print("=" * 60)
    print("Testing SegFormer Change Detection Model")
    print("=" * 60)

    # 临时禁用深度监督测试
    original_ds = cfg.DEEP_SUPERVISION
    cfg.DEEP_SUPERVISION = True

    # 创建模型（二值变化检测用1通道输出）
    model = build_model(variant='b1', pretrained=True, num_classes=1)

    # 测试前向传播
    print("\nTesting forward pass...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()  # 测试训练模式（包含深度监督）

    # 创建虚拟输入（256x256，8G显存友好）
    img_a = torch.randn(2, 3, 256, 256).to(device)
    img_b = torch.randn(2, 3, 256, 256).to(device)

    with torch.no_grad():
        outputs = model(img_a, img_b)

    print(f"\nInput shapes:")
    print(f"  img_a: {img_a.shape}")
    print(f"  img_b: {img_b.shape}")

    print(f"\nOutput shapes:")
    print(f"  pred: {outputs['pred'].shape}")
    if 'aux_preds' in outputs:
        for i, aux in enumerate(outputs['aux_preds']):
            print(f"  aux_pred_{i}: {aux.shape}")

    # 验证输出（1通道二值变化检测）
    assert outputs['pred'].shape == (2, 1, 256, 256), f"Expected (2,1,256,256), got {outputs['pred'].shape}"
    print("\n" + "=" * 60)
    print("Model test passed!")
    print("=" * 60)

    # 恢复配置
    cfg.DEEP_SUPERVISION = original_ds
