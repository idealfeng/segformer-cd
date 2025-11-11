"""
SegFormer Baseline（不带蒸馏，只做tricks）
用于保底方案：边界感知 + 其他tricks
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
from config import cfg

OUT_CHANNELS = 1  # 二分类


class SegFormerTricks(nn.Module):
    """
    简化版SegFormer，去掉蒸馏特征
    专注于分割任务 + tricks
    """

    def __init__(self, variant='b1', pretrained=True):
        super().__init__()

        model_names = {f'b{i}': f'nvidia/mit-b{i}' for i in range(6)}
        model_name = model_names[variant]

        print(f"加载SegFormer-{variant.upper()}...")
        print(f"  模式: Baseline + Tricks（无蒸馏）")

        # 加载完整模型
        full_model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=cfg.NUM_CLASSES,
            ignore_mismatched_sizes=True
        )

        self.backbone = full_model.segformer
        self.decode_head = full_model.decode_head

        # 修改输出层为单通道
        self.decode_head.classifier = nn.Conv2d(
            self.decode_head.classifier.in_channels,
            OUT_CHANNELS,
            kernel_size=1
        )

        print("✓ 模型加载完成（纯分割，无蒸馏）")

    def forward(self, x):
        """
        简单的前向传播

        Returns:
            只返回pred，不返回特征
        """
        B, C, H, W = x.shape

        # Backbone提取特征
        outputs = self.backbone(
            pixel_values=x,
            output_hidden_states=True
        )

        hidden_states = outputs.hidden_states

        # Reshape特征
        h_s = [H // 4, H // 8, H // 16, H // 32]
        w_s = [W // 4, W // 8, W // 16, W // 32]

        all_stage_features = []
        for i, hidden_state in enumerate(hidden_states):
            if hidden_state.ndim == 3:
                B_h, N, C_h = hidden_state.shape
                feature = hidden_state.permute(0, 2, 1).reshape(B_h, C_h, h_s[i], w_s[i])
            else:
                feature = hidden_state
            all_stage_features.append(feature)

        # 解码
        logits = self.decode_head(all_stage_features)

        # 上采样到原始尺寸
        logits_upsampled = F.interpolate(
            logits,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )

        return {'pred': logits_upsampled}

    def get_params_info(self):
        """获取参数信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total': total_params / 1e6,
            'trainable': trainable_params / 1e6
        }


def build_segformer_tricks(variant='b1', pretrained=True):
    """构建tricks版本的SegFormer"""
    model = SegFormerTricks(variant=variant, pretrained=pretrained)

    params_info = model.get_params_info()
    print(f"\n模型信息:")
    print(f"  总参数: {params_info['total']:.2f}M")
    print(f"  可训练参数: {params_info['trainable']:.2f}M")

    return model