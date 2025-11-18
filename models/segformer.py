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


class SemanticGuidanceModule(nn.Module):
    """
    语义引导模块 - 生成变化区域置信图

    核心思想（正交设计）：
    1. 不改变原始特征，保持差异计算纯净
    2. 融合双时相语义特征，生成"哪里可能变化"的置信图
    3. 用置信图去调制差异特征的响应

    这才是真正的"语义引导 + 空间定位"：
    - 语义融合 → 确定大致变化区域（guide mask）
    - 空间差异 → 精确定位变化地物（|Fa - Fb|）
    """
    def __init__(self, channels):
        super().__init__()
        # 语义融合网络：产生变化置信图
        self.guidance_net = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, 1),  # 输出1通道置信图
            nn.Sigmoid()  # (0, 1) 表示变化概率
        )

    def forward(self, feat_a, feat_b):
        """
        Args:
            feat_a: 时刻A特征 (B, C, H, W)
            feat_b: 时刻B特征 (B, C, H, W)
        Returns:
            guide: 变化置信图 (B, 1, H, W) ∈ (0, 1)
        """
        # 拼接双时相语义特征
        concat = torch.cat([feat_a, feat_b], dim=1)
        # 生成变化置信图
        guide = self.guidance_net(concat)  # (B, 1, H, W)
        return guide


class DifferenceModule(nn.Module):
    """
    差异计算模块（支持语义引导）

    改进：接受可选的guide mask来调制差异响应
    """
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

    def forward(self, feat_a, feat_b, guide=None):
        """
        Args:
            feat_a: 时刻A特征 (B, C, H, W)
            feat_b: 时刻B特征 (B, C, H, W)
            guide: 语义引导置信图 (B, 1, H, W) [可选]
        Returns:
            out: 差异特征 (B, C, H, W)
        """
        # 计算差异特征（保持纯净，不被语义交换污染）
        diff = torch.abs(feat_a - feat_b)
        # 拼接原始特征和差异特征
        concat = torch.cat([diff, feat_a + feat_b], dim=1)
        # 融合
        out = self.conv(concat)

        # 语义引导：用置信图调制差异响应
        if guide is not None:
            # guide ∈ (0,1) 表示"该位置变化的概率"
            # 方案选择（config可配置）
            guide_mode = getattr(cfg, 'GUIDE_MODULATION_MODE', 'soft')

            if guide_mode == 'hard':
                # 原始方案：直接相乘（可能过aggressive）
                out = out * guide
            elif guide_mode == 'soft':
                # 软调制：避免完全抑制 (推荐)
                out = out * (0.5 + 0.5 * guide)
            elif guide_mode == 'residual':
                # 残差式：guide作为加权残差
                out = out + out * (guide - 0.5)
            elif guide_mode == 'attention':
                # 注意力式：增强对比度
                guide_enhanced = torch.sigmoid(2 * (guide - 0.5))
                out = out * guide_enhanced

        # 通道注意力增强
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

        # 语义引导模块（仅高层，生成置信图）
        self.use_semantic_guidance = getattr(cfg, 'USE_SEMANTIC_GUIDANCE', True)
        self.guidance_high_level_only = getattr(cfg, 'GUIDANCE_HIGH_LEVEL_ONLY', True)

        if self.use_semantic_guidance:
            if self.guidance_high_level_only:
                # 只在高层（Stage 3-4）生成语义引导
                self.semantic_guides = nn.ModuleList([
                    None,  # Stage 1: 不需要语义引导
                    None,  # Stage 2: 不需要语义引导
                    SemanticGuidanceModule(self.channels[2]),  # Stage 3: 320 channels
                    SemanticGuidanceModule(self.channels[3])   # Stage 4: 512 channels
                ])
                print(f"  Semantic guidance: Enabled (high-level only, Stage 3-4)")
            else:
                # 所有层都生成语义引导
                self.semantic_guides = nn.ModuleList([
                    SemanticGuidanceModule(ch) for ch in self.channels
                ])
                print(f"  Semantic guidance: Enabled (all stages)")
        else:
            self.semantic_guides = None
            print(f"  Semantic guidance: Disabled")

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
        if self.semantic_guides is not None:
            self.semantic_guides.apply(self._init_weights)
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
        all_hidden = outputs.hidden_states
        depths = self.encoder.config.depths  # e.g., [2, 2, 2, 2] for B1

        # 调试信息（首次运行时打印）
        if not hasattr(self, '_debug_printed'):
            print(f"\n[DEBUG] Hidden states info:")
            print(f"  Number of hidden states: {len(all_hidden)}")
            print(f"  Depths: {depths}")
            for i, hs in enumerate(all_hidden):
                print(f"  hidden_state[{i}] shape: {hs.shape}")
            self._debug_printed = True

        # SegFormer的hidden_states包含每个stage的输出（共4个）
        # 直接取最后4个（或者就是4个stage的输出）
        num_stages = len(depths)

        # 如果hidden_states正好是4个，直接用
        if len(all_hidden) == num_stages:
            stage_outputs = all_hidden
        else:
            # 否则取每个stage的最后一层
            stage_indices = []
            acc = 0
            for d in depths:
                acc += d
                stage_indices.append(acc - 1)
            stage_outputs = [all_hidden[idx] for idx in stage_indices]

        # 计算每个stage的特征图尺寸
        h_sizes = [H // 4, H // 8, H // 16, H // 32]
        w_sizes = [W // 4, W // 8, W // 16, W // 32]

        # Reshape为4D特征图
        features = []
        for i, hidden_state in enumerate(stage_outputs):
            # 处理不同格式的hidden_state
            # 初始化feat以避免未定义错误
            feat = None

            if hidden_state.ndim == 3:
                # (B, N, C) -> (B, C, H, W)
                B_, N, C_ = hidden_state.shape
                feat = hidden_state.permute(0, 2, 1).reshape(B_, C_, h_sizes[i], w_sizes[i])
            elif hidden_state.ndim == 4:
                # 已经是 (B, C, H, W) 格式
                feat = hidden_state
            else:
                raise ValueError(f"Unexpected hidden_state ndim: {hidden_state.ndim}, shape: {hidden_state.shape}")

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

        # 提取两个时刻的特征（保持纯净，不被语义引导改变）
        feats_a = self._extract_features(img_a)
        feats_b = self._extract_features(img_b)

        # 计算多尺度差异特征（语义引导调制）
        diff_feats = []
        for i in range(len(self.channels)):
            # 生成语义引导置信图（仅高层）
            guide = None
            if self.semantic_guides is not None and self.semantic_guides[i] is not None:
                guide = self.semantic_guides[i](feats_a[i], feats_b[i])  # (B, 1, H, W)

            # 计算差异特征，并用guide调制
            diff_feat = self.diff_modules[i](feats_a[i], feats_b[i], guide=guide)
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
