import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


# -----------------------------
# Normalization helper
# -----------------------------
def norm2d(norm: str, ch: int) -> nn.Module:
    """Small-batch friendly norm. Default: GroupNorm."""
    norm = (norm or "gn").lower()
    if norm == "bn":
        return nn.BatchNorm2d(ch)
    elif norm == "gn":
        # 32 groups is typical; fallback if ch < 32 or not divisible.
        g = 32
        while g > 1 and (ch % g != 0):
            g //= 2
        return nn.GroupNorm(g, ch)
    else:
        raise ValueError(f"Unknown norm: {norm}")


class ChannelAttention(nn.Module):
    """通道注意力（SE-style）"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        r = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(r, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class DifferenceModule(nn.Module):
    """|Fa-Fb| + (Fa+Fb) + 通道注意力"""

    def __init__(self, in_channels: int, out_channels: int, norm: str = "gn"):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 3, padding=1, bias=False),
            norm2d(norm, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            norm2d(norm, out_channels),
            nn.GELU(),
        )
        self.ca = ChannelAttention(out_channels)

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(feat_a - feat_b)
        semantic = feat_a + feat_b
        out = torch.cat([diff, semantic], dim=1)
        out = self.conv(out)
        return self.ca(out)


class MultiScaleFusionDecoder(nn.Module):
    """多尺度融合解码器（类似 SegFormer MLPDecoder）"""

    def __init__(
        self, in_channels_list: List[int], embed_dim: int = 256, norm: str = "gn"
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.linear_layers = nn.ModuleList(
            [nn.Conv2d(in_ch, embed_dim, 1) for in_ch in in_channels_list]
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels_list), embed_dim, 1, bias=False),
            norm2d(norm, embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False),
            norm2d(norm, embed_dim),
            nn.GELU(),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        target_size = features[0].shape[2:]
        aligned = []
        for i, feat in enumerate(features):
            feat = self.linear_layers[i](feat)
            if feat.shape[2:] != target_size:
                feat = F.interpolate(
                    feat, size=target_size, mode="bilinear", align_corners=False
                )
            aligned.append(feat)
        fused = torch.cat(aligned, dim=1)
        return self.fusion(fused)


class DinoSiameseHead(nn.Module):
    """
    改进版 DINOv2 头（稳定跨域版）：
    - 修复 *_reg 模型的 register tokens
    - 增加 1x1 adapter 降维（默认 192）避免 decoder 过重/过拟合
    - 默认用 GroupNorm（小 batch 更稳），可切回 BN
    - forward 返回 dict: {"pred": logits} 兼容你现有 evaluator
    """

    def __init__(
        self,
        dino_name: str = "dinov2_vits14_reg",
        patch: int = 14,
        use_whiten: bool = False,
        embed_dim: int = 256,
        head_channels: int = 256,
        head_depth: int = 3,
        dropout: float = 0.10,
        selected_layers: Tuple[int, ...] = (3, 6, 9, 12),
        adapter_dim: int = 192,
        norm: str = "gn",  # "gn" (recommended) or "bn"
    ):
        super().__init__()
        self.patch = int(patch)
        self.use_whiten = bool(use_whiten)
        self.selected_layers = list(selected_layers)
        self.adapter_dim = int(adapter_dim)
        self.norm = norm

        self.backbone = torch.hub.load("facebookresearch/dinov2", dino_name)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # number of register tokens for *_reg backbones (0 for non-reg)
        self.num_reg = int(getattr(self.backbone, "num_register_tokens", 0))

        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            out = self.backbone.forward_features(dummy)
            self.feat_dim = out["x_norm_patchtokens"].shape[-1]

        # adapters: feat_dim -> adapter_dim (per selected layer)
        self.adapters = nn.ModuleList(
            [
                nn.Conv2d(self.feat_dim, self.adapter_dim, 1, bias=False)
                for _ in self.selected_layers
            ]
        )

        self.diff_modules = nn.ModuleList(
            [
                DifferenceModule(self.adapter_dim, self.adapter_dim, norm=self.norm)
                for _ in self.selected_layers
            ]
        )

        self.decoder = MultiScaleFusionDecoder(
            in_channels_list=[self.adapter_dim] * len(self.selected_layers),
            embed_dim=embed_dim,
            norm=self.norm,
        )

        # lightweight refinement head
        head_layers = []
        in_ch = embed_dim
        for i in range(head_depth):
            head_layers += [
                nn.Conv2d(in_ch, head_channels, 3, padding=1, bias=False),
                norm2d(self.norm, head_channels),
                nn.GELU(),
            ]
            if dropout > 0 and i < head_depth - 1:
                head_layers.append(nn.Dropout2d(dropout))
            in_ch = head_channels

        self.head = nn.Sequential(*head_layers)
        self.classifier = nn.Conv2d(head_channels, 1, 1)

    @torch.no_grad()
    def _pad_to_patch_multiple(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        B, C, H, W = x.shape
        H2 = ((H + self.patch - 1) // self.patch) * self.patch
        W2 = ((W + self.patch - 1) // self.patch) * self.patch
        pad_h, pad_w = H2 - H, W2 - W
        if pad_h == 0 and pad_w == 0:
            return x, (H, W)
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, (H, W)

    @torch.no_grad()
    def _extract_multi_layer_features(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], Tuple[int, int], Tuple[int, int]]:
        """
        Return: list of [B,C,h,w] for selected layers (same spatial resolution),
                plus padded spatial size and original size for unpad.
        """
        x, (H0, W0) = self._pad_to_patch_multiple(x)
        B, _, Hp, Wp = x.shape
        h, w = Hp // self.patch, Wp // self.patch

        feats_raw: List[torch.Tensor] = []

        def hook_fn(_, __, output):
            feats_raw.append(output)

        hooks = []
        # register hooks on blocks (1-indexed in selected_layers)
        for idx in self.selected_layers:
            hooks.append(self.backbone.blocks[idx - 1].register_forward_hook(hook_fn))

        _ = self.backbone.forward_features(x)

        for hk in hooks:
            hk.remove()

        if len(feats_raw) != len(self.selected_layers):
            raise RuntimeError(
                f"Expected {len(self.selected_layers)} hooks, got {len(feats_raw)}"
            )

        feats: List[torch.Tensor] = []
        for feat in feats_raw:
            # feat: [B, 1+num_reg+h*w, C] typically
            if feat.ndim != 3:
                raise ValueError(f"Unexpected hooked feat shape: {feat.shape}")
            # drop cls + reg tokens
            tokens = feat[:, 1 + self.num_reg :, :]  # [B, Npatch, C]
            # safety: keep last h*w tokens if mismatch (robust to weird packing)
            if tokens.shape[1] != h * w:
                tokens = tokens[:, -h * w :, :]
            feat_map = tokens.transpose(1, 2).reshape(B, tokens.shape[-1], h, w)
            feat_map = F.normalize(feat_map, dim=1)

            if self.use_whiten:
                mu = feat_map.mean(dim=(2, 3), keepdim=True)
                sd = feat_map.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
                feat_map = (feat_map - mu) / sd

            feats.append(feat_map)

        return feats, (Hp, Wp), (H0, W0)

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor):
        B, _, H, W = img_a.shape

        with torch.no_grad():
            fa_list, (Hp, Wp), (H0, W0) = self._extract_multi_layer_features(img_a)
            fb_list, (Hp2, Wp2), _ = self._extract_multi_layer_features(img_b)
            if (Hp, Wp) != (Hp2, Wp2):
                raise RuntimeError("Backbone outputs mismatch.")

        diff_feats = []
        for i, (fa, fb, dm) in enumerate(zip(fa_list, fb_list, self.diff_modules)):
            fa = self.adapters[i](fa)
            fb = self.adapters[i](fb)
            diff_feats.append(dm(fa, fb))

        fused = self.decoder(diff_feats)  # [B, embed_dim, h, w]
        x = self.head(fused)  # [B, head_channels, h, w]
        logit_patch = self.classifier(x)  # [B, 1, h, w]

        # upsample back to padded spatial size, then unpad, then ensure match original
        logit_pad = F.interpolate(
            logit_patch, size=(Hp, Wp), mode="bilinear", align_corners=False
        )
        logit = logit_pad[..., :H0, :W0]
        if logit.shape[-2:] != (H, W):
            logit = F.interpolate(
                logit, size=(H, W), mode="bilinear", align_corners=False
            )

        return {"pred": logit}


__all__ = [
    "DinoSiameseHead",
    "ChannelAttention",
    "DifferenceModule",
    "MultiScaleFusionDecoder",
]
