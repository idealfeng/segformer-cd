"""

Simple Baseline Model for Change Detection

使用ResNet18作为骨干网络，最简单的Siamese架构

"""

import torch

import torch.nn as nn

import torch.nn.functional as F

from torchvision.models import resnet18, resnet34, ResNet18_Weights, ResNet34_Weights


class SimpleDecoder(nn.Module):
    """简单的上采样解码器"""

    def __init__(self, in_channels_list, num_classes=1):

        super().__init__()

        # 总输入通道数

        total_channels = sum(in_channels_list)

        # 简单的融合和上采样

        self.fusion = nn.Sequential(

            nn.Conv2d(total_channels, 256, 3, padding=1, bias=False),

            nn.BatchNorm2d(256),

            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, 3, padding=1, bias=False),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, 3, padding=1, bias=False),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

        )

        # 分类头

        self.classifier = nn.Conv2d(64, num_classes, 1)

    def forward(self, features):

        """

        Args:

            features: list of [f1, f2, f3, f4]，不同尺度的特征

        """

        # 将所有特征上采样到最大尺度

        target_size = features[0].shape[2:]

        upsampled = []

        for feat in features:

            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)

            upsampled.append(feat)

        # 拼接所有特征

        fused = torch.cat(upsampled, dim=1)

        # 融合

        out = self.fusion(fused)

        # 分类

        logits = self.classifier(out)

        return logits


class SimpleSiameseCD(nn.Module):
    """

    最简单的Siamese变化检测模型



    架构：

    - 骨干：ResNet18 (ImageNet预训练)

    - 融合：简单差异 |Fa - Fb|

    - 解码器：卷积上采样



    目的：作为baseline，证明跨数据集泛化问题不是模型复杂度导致的

    """

    def __init__(self, backbone='resnet18', pretrained=True, num_classes=1):

        super().__init__()

        self.num_classes = num_classes

        # 加载预训练ResNet

        if backbone == 'resnet18':

            if pretrained:

                weights = ResNet18_Weights.IMAGENET1K_V1

                self.encoder = resnet18(weights=weights)

            else:

                self.encoder = resnet18(weights=None)

            channels = [64, 128, 256, 512]  # ResNet18各stage通道数

        elif backbone == 'resnet34':

            if pretrained:

                weights = ResNet34_Weights.IMAGENET1K_V1

                self.encoder = resnet34(weights=weights)

            else:

                self.encoder = resnet34(weights=None)

            channels = [64, 128, 256, 512]  # ResNet34各stage通道数

        else:

            raise ValueError(f"Unsupported backbone: {backbone}")

        self.channels = channels

        print(f"Building Simple Siamese CD with {backbone.upper()}...")

        print(f"  Pretrained: {pretrained}")

        print(f"  Encoder channels: {channels}")

        # 提取ResNet的各个stage

        self.conv1 = self.encoder.conv1

        self.bn1 = self.encoder.bn1

        self.relu = self.encoder.relu

        self.maxpool = self.encoder.maxpool

        self.layer1 = self.encoder.layer1  # 64 channels

        self.layer2 = self.encoder.layer2  # 128 channels

        self.layer3 = self.encoder.layer3  # 256 channels

        self.layer4 = self.encoder.layer4  # 512 channels

        # 简单解码器（接收差异特征）

        self.decoder = SimpleDecoder(channels, num_classes)

        # 辅助头（深度监督，可选）

        self.aux_heads = nn.ModuleList([

            nn.Conv2d(ch, num_classes, 1) for ch in channels

        ])

        print(f"Model built successfully!")

        print(f"  Output classes: {num_classes}")

    def _extract_features(self, x):

        """提取ResNet多尺度特征"""

        features = []

        # Stage 0: stem

        x = self.conv1(x)

        x = self.bn1(x)

        x = self.relu(x)

        x = self.maxpool(x)

        # Stage 1: 64 channels, 1/4

        x = self.layer1(x)

        features.append(x)

        # Stage 2: 128 channels, 1/8

        x = self.layer2(x)

        features.append(x)

        # Stage 3: 256 channels, 1/16

        x = self.layer3(x)

        features.append(x)

        # Stage 4: 512 channels, 1/32

        x = self.layer4(x)

        features.append(x)

        return features

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

        # 计算简单差异 |Fa - Fb|

        diff_feats = [

            torch.abs(fa - fb) for fa, fb in zip(feats_a, feats_b)

        ]

        # 解码

        logits = self.decoder(diff_feats)

        # 上采样到原始尺寸

        pred = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)

        result = {'pred': pred}

        # 深度监督（训练时）

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


def build_simple_model(backbone='resnet18', pretrained=True, num_classes=1):
    """

    构建简单baseline模型



    Args:

        backbone: 骨干网络 ('resnet18', 'resnet34')

        pretrained: 是否使用ImageNet预训练

        num_classes: 类别数



    Returns:

        model: SimpleSiameseCD模型

    """

    model = SimpleSiameseCD(

        backbone=backbone,

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

    print("Testing Simple Siamese CD Model")

    print("=" * 60)

    # 创建模型

    model = build_simple_model(backbone='resnet18', pretrained=True, num_classes=1)

    # 测试前向传播

    print("\nTesting forward pass...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    model.train()

    # 创建虚拟输入

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

    print("\n" + "=" * 60)

    print("Model test passed!")

    print("=" * 60)