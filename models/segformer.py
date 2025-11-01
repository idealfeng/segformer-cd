"""
SegFormer模型 - 【v2.0 最终版】
- ✅ 单通道输出用于二分类分割
- ✅ 返回两层特征用于与SAM蒸馏
- ✅ 内置特征对齐层，自动匹配SAM的维度
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerModel, SegformerConfig
from config import cfg # ✅ 正确的导入方式！
import torch.utils.checkpoint as checkpoint
from transformers.models.segformer.modeling_segformer import SegformerMLP

# ✅ 修复1: 明确定义模型输出通道数，与任务解耦
OUT_CHANNELS = 1 # 我们使用单通道输出 + BCEWithLogitsLoss

class FeatureAlignLayer(nn.Module):
    """
    特征对齐层
    用于将学生特征的通道数和空间分辨率对齐到教师特征
    """

    def __init__(self, in_channels, out_channels, scale_factor=1):
        """
        Args:
            in_channels: 学生特征通道数
            out_channels: 教师特征通道数
            scale_factor: 空间尺度变化（1=不变，2=上采样2倍）
        """
        super().__init__()
        self.scale_factor = scale_factor

        # 通道对齐：1x1卷积
        """
        self.conv = nn.Conv2d(...): 这是一个1x1的卷积。它的唯一作用就是“升维”或“降维”，改变特征图的通道数（in_channels -> out_channels），而不改变其空间尺寸（H, W）。
        self.bn = nn.BatchNorm2d(...): 在卷积后进行归一化，让训练过程更稳定。
        self.relu = nn.ReLU(...): 引入非线性，增强这个“翻译官”的学习能力。
        F.interpolate(...): 如果需要，它会像“放大镜”一样，使用双线性插值，将特征图的空间尺寸（H, W）放大scale_factor倍。
        """
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')


    def forward(self, x):
        """
        Args:
            x: (B, in_channels, H, W)

        Returns:
            aligned: (B, out_channels, H*scale_factor, W*scale_factor)
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        # 空间对齐
        if self.scale_factor != 1:
            x = F.interpolate(
                x,
                scale_factor=self.scale_factor,
                mode='bilinear',
                align_corners=False
            )

        return x


class SegFormerDistillation(nn.Module):
    """
    full_model = SegformerForSemanticSegmentation.from_pretrained(...): 这是整个模型最精髓的一步。我们没有直接加载纯粹的SegformerModel，而是加载了完整的SegformerForSemanticSegmentation。
    self.backbone = full_model.segformer: 我们**“偷”来了这个完整模型内部的segformer骨干。为什么这么做？ 因为这个.segformer骨干的forward函数，有一个非常方便的功能，就是会返回已经帮你reshape好的4D特征图**，我们就不需要自己去处理恼人的(B, N, C)序列了！
    student_dims = self.backbone.config.hidden_sizes: 我们从加载好的模型配置中，动态地、安全地获取了每个stage的输出通道数，非常健壮。
    self.decode_head = SegformerMLP(...): 我们直接使用了HuggingFace官方实现的SegformerMLP解码头。这是保证模型性能的关键。我们还正确地传入了input_dim，告诉它我们将要输入所有stage特征的总通道数。
    self.classifier = nn.Conv2d(...): 在MLP解码头之后，我们接上我们自己的、简单的1x1卷积分类器。它的输出通道数被硬编码为1 (OUT_CHANNELS)，这保证了我们的模型最终只会输出单通道的二分类Logits。
    self.align_... = FeatureAlignLayer(...): 我们创建了两个“翻译官”实例，并为它们配置了正确的输入/输出通道数和缩放因子。
    align_block30: scale_factor=1，因为SegFormer的Stage3输出就是64x64，与SAM特征尺寸一致。
    align_encoder: scale_factor=2，因为Stage4输出是32x32，需要上采样2倍才能对齐SAM的64x64。
    """
    def __init__(self, variant='b1', pretrained=True,
                 teacher_feat_b30_dim=1280,
                 teacher_feat_enc_dim=256):
        super().__init__()

        model_names = {f'b{i}': f'nvidia/mit-b{i}' for i in range(6)}
        model_name = model_names[variant]

        print(f"加载SegFormer-{variant.upper()}...")
        print(f"  任务: 二分类分割 (前景/背景)")

        # ✅ 修复1: 【最重要】加载端到端模型，然后只取其backbone
        # 这样做可以白嫖到它内部从序列reshape到4D图像特征的功能
        full_model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=cfg.NUM_CLASSES,  # 从config读取，确保一致性
            ignore_mismatched_sizes=True
        )
        self.backbone = full_model.segformer
        # if cfg.USE_GRADIENT_CHECKPOINTING and self.training:
        #     self.backbone.gradient_checkpointing_enable()
        #     print("✓ 官方梯度检查点 (Gradient Checkpointing) 已启用")
        # 从加载的模型配置中动态获取维度，更健壮
        student_dims = self.backbone.config.hidden_sizes
        c1, c2, c3, c4 = student_dims

        # ✅ 修复3: 【最重要】正确地使用官方MLP解码头
        # ✅ 正确：直接用加载的decode_head
        self.decode_head = full_model.decode_head

        # ✅ 修改classifier为单通道输出
        self.decode_head.classifier = nn.Conv2d(
            self.decode_head.classifier.in_channels,
            OUT_CHANNELS, # 输出1通道
            kernel_size=1
        )
        # 原来两路都写了 scale_factor=2
        student_dims = self.backbone.config.hidden_sizes
        c3, c4 = student_dims[2], student_dims[3]

        self.align_block30 = FeatureAlignLayer(in_channels=c3, out_channels=teacher_feat_b30_dim, scale_factor=1)
        self.align_encoder = FeatureAlignLayer(in_channels=c4, out_channels=teacher_feat_enc_dim, scale_factor=2)

        print("✓ 模型加载完成, 特征对齐层和官方解码头创建完成")

    def _reshape_hidden_state(self, hidden_state, H, W):
        """✅ 修复: 支持3D和4D输入"""
        if hidden_state.ndim == 3:
            # (B, N, C) → (B, C, H, W)
            B, N, C = hidden_state.shape
            return hidden_state.permute(0, 2, 1).reshape(B, C, H, W)
        else:
            # 已经是(B, C, H, W)，直接返回
            return hidden_state

    def forward(self, x):
        """
        1.  **`outputs = self.backbone(...)`**: 图像首先通过`backbone`，进行特征提取。我们让它返回`hidden_states`。
        2.  **`all_stage_features = outputs.all_hidden_states`**: 我们从`backbone`的输出中，直接拿到了一个包含**4个、已经是4D格式**的特征图的列表。
        3.  **`feat_stage3`, `feat_stage4`**: 我们从中取出第3和第4个stage的特征，备用。
        4.  **【解码路径】**:
            *   **`decoder_output = self.decode_head(all_stage_features)`**: **最关键的一步！** 我们将**所有4个**stage的特征，作为一个列表，整体喂给官方的`MLP`解码头。它会在内部进行上采样、拼接(concat)、融合，最终输出一个融合了多尺度信息的、更强大的特征图。
            *   **`logits = self.classifier(decoder_output)`**: 融合后的特征，通过我们自己的分类器，被压缩成**单通道**的Logits。
            *   **`logits_upsampled = F.interpolate(...)`**: 最后，将这个小尺寸的Logits图，通过插值，放大回原始输入图像的尺寸` (H, W)`。
        5.  **【蒸馏路径】**:
            *   **`feat_b30_aligned = self.align_block30(feat_stage3)`**: 第3个stage的特征，被送入我们准备好的第一个“翻译官”，翻译成与SAM Block30维度一致的特征。
            *   **`feat_enc_aligned = self.align_encoder(feat_stage4)`**: 第4个stage的特征，被送入第二个“翻译官”，翻译成与SAM Encoder最终输出维度一致的特征。
        6.  **`return {...}`**: 最后，将“解码路径”的最终结果 (`pred`) 和“蒸馏路径”的两个结果 (`feat_b30`, `feat_enc`)，打包成一个字典返回。
        """
        B, C, H, W = x.shape

        # ✅ 用checkpoint包装backbone（节省显存）
        def backbone_forward(pixel_values):
            return self.backbone(
                pixel_values=pixel_values,
                output_hidden_states=True
            )

        if self.training and cfg.USE_GRADIENT_CHECKPOINTING:
            outputs = checkpoint.checkpoint(
                backbone_forward,
                x,
                use_reentrant=False
            )
        else:
            outputs = backbone_forward(x)

        hidden_states = outputs.hidden_states

        # 计算每个stage的特征图尺寸
        h_s = [H // 4, H // 8, H // 16, H // 32]
        w_s = [W // 4, W // 8, W // 16, W // 32]

        all_stage_features = []
        for i, hidden_state in enumerate(hidden_states):
            all_stage_features.append(self._reshape_hidden_state(hidden_state, h_s[i], w_s[i]))

        feat_stage3 = all_stage_features[2]
        feat_stage4 = all_stage_features[3]

        # ✅ 【核心修正】: 正确地调用官方解码头
        # 它期望输入一个包含4个stage特征的列表
        logits = self.decode_head(all_stage_features)  # 输出已经是 (B, 1, H/4, W/4)

        logits_upsampled = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)

        feat_b30_aligned = self.align_block30(feat_stage3)
        feat_enc_aligned = self.align_encoder(feat_stage4)

        return {
            'pred': logits_upsampled,
            'feat_b30': feat_b30_aligned,
            'feat_enc': feat_enc_aligned
        }

    def get_params_info(self):
        """获取参数信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total': total_params / 1e6,
            'trainable': trainable_params / 1e6
        }


def build_segformer_distillation(
        variant='b1',
        pretrained=True,
        teacher_feat_b30_dim=None,
        teacher_feat_enc_dim=None
):
    """
    构建SegFormer蒸馏模型

    Args:
        variant: 模型变体
        pretrained: 是否预训练
        teacher_feat_b30_dim: 教师Block30特征维度（从config读取）
        teacher_feat_enc_dim: 教师Encoder特征维度（从config读取）

    Returns:
        model: SegFormer蒸馏模型
    """
    # 从config读取教师特征维度
    if teacher_feat_b30_dim is None:
        teacher_feat_b30_dim = cfg.TEACHER_FEAT_BLOCK30_DIM # 假设config里是这个名字
    if teacher_feat_enc_dim is None:
        teacher_feat_enc_dim = cfg.TEACHER_FEAT_ENCODER_DIM # 假设config里是这个名字

    model = SegFormerDistillation(
        variant=variant,
        pretrained=pretrained,
        teacher_feat_b30_dim=teacher_feat_b30_dim,
        teacher_feat_enc_dim=teacher_feat_enc_dim
    )

    # 打印参数信息
    params_info = model.get_params_info()
    print(f"\n模型信息:")
    print(f"  总参数: {params_info['total']:.2f}M")
    print(f"  可训练参数: {params_info['trainable']:.2f}M")

    return model


if __name__ == '__main__':
    """测试模型"""
    print("=" * 60)
    print("测试SegFormer蒸馏模型（二分类+特征对齐）")
    print("=" * 60)

    # 创建模型
    model = build_segformer_distillation(
        variant='b1',
        pretrained=True,
        teacher_feat_b30_dim=1280,  # SAM Block30
        teacher_feat_enc_dim=256  # SAM Encoder
    )

    # 测试前向传播
    print("\n测试前向传播...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # 创建虚拟输入
    dummy_input = torch.randn(2, 3, 1024, 1024).to(device)

    with torch.no_grad():
        outputs = model(dummy_input)

    print(f"\n输入: {dummy_input.shape}")
    print(f"\n输出:")
    print(f"  pred:     {outputs['pred'].shape} - 二分类预测")
    print(f"  feat_b30: {outputs['feat_b30'].shape} - 对齐Block30特征")
    print(f"  feat_enc: {outputs['feat_enc'].shape} - 对齐Encoder特征")

    # 最终验证
    assert outputs['pred'].shape == (2, OUT_CHANNELS, 1024, 1024)
    assert outputs['feat_b30'].shape[-2:] == (64, 64), f"Block30对齐后尺寸错误! 期望(64,64), 得到{outputs['feat_b30'].shape[-2:]}"
    assert outputs['feat_enc'].shape[-2:] == (64, 64), f"Encoder对齐后尺寸错误! 期望(64,64), 得到{outputs['feat_enc'].shape[-2:]}"
    print("\n" + "=" * 60)
    print("✓ 模型测试通过")
    print("=" * 60)