import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import List, Tuple
import numpy as np

try:
    from transformers import AutoModel
except Exception:
    AutoModel = None  # only needed for DINOv3


# -----------------------------
# Normalization helper
# -----------------------------
def norm2d(norm: str, ch: int) -> nn.Module:
    """Small-batch friendly norm. Default: GroupNorm."""
    norm = (norm or "gn").lower()
    if norm == "bn":
        return nn.BatchNorm2d(ch)
    elif norm == "gn":
        g = 32
        while g > 1 and (ch % g != 0):
            g //= 2
        return nn.GroupNorm(g, ch)
    else:
        raise ValueError(f"Unknown norm: {norm}")

class GradReverse(Function):
    """Gradient reversal layer."""

    @staticmethod
    def forward(ctx, x, lambd=1.0):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


class DomainDiscriminator(nn.Module):
    """Lightweight domain discriminator over fused features."""

    def __init__(self, in_channels: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            norm2d("gn", in_channels),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):  # x: [B,C,h,w]
        return self.net(x)


class ConvBlock(nn.Module):
    """Two conv layers with norm and GELU; optional stride for downsampling."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, norm: str = "gn"):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            norm2d(norm, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            norm2d(norm, out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling with a global context branch."""

    def __init__(self, in_ch: int, out_ch: int, dilations=(1, 3, 6, 9), norm: str = "gn"):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=d, dilation=d, bias=False),
                    norm2d(norm, out_ch),
                    nn.GELU(),
                )
                for d in dilations
            ]
        )
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GELU(),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(out_ch * (len(dilations) + 1), out_ch, kernel_size=1, bias=False),
            norm2d(norm, out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        feats = [b(x) for b in self.branches]
        g = self.global_branch(x)
        g = F.interpolate(g, size=(H, W), mode="bilinear", align_corners=False)
        feats.append(g)
        x = torch.cat(feats, dim=1)
        return self.proj(x)


class UpBlock(nn.Module):
    """Upsample + skip concat + ConvBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, norm: str = "gn"):
        super().__init__()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch, stride=1, norm=norm)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class HeavyDecoder(nn.Module):
    """
    U-Net style decoder with ASPP context.
    Input: fused [B, C, H, W]; Output: refined [B, base_ch, H, W].
    """

    def __init__(self, in_ch: int, base_ch: int = 256, norm: str = "gn"):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base_ch, stride=1, norm=norm)
        self.enc2 = ConvBlock(base_ch, base_ch, stride=2, norm=norm)
        self.enc3 = ConvBlock(base_ch, base_ch, stride=2, norm=norm)
        self.aspp = ASPP(base_ch, base_ch, dilations=(1, 3, 6, 9), norm=norm)
        self.dec2 = UpBlock(base_ch, base_ch, base_ch, norm=norm)
        self.dec1 = UpBlock(base_ch, base_ch, base_ch, norm=norm)
        self.refine = ConvBlock(base_ch, base_ch, stride=1, norm=norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x3 = self.aspp(x3)
        u2 = self.dec2(x3, x2)
        u1 = self.dec1(u2, x1)
        return self.refine(u1)


class LayerHead(nn.Module):
    """Per-layer prediction head for ensemble logits."""

    def __init__(self, in_ch: int, base_ch: int = 64, norm: str = "gn"):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1, bias=False),
            norm2d(norm, base_ch),
            nn.GELU(),
            nn.Conv2d(base_ch, base_ch, 3, padding=1, bias=False),
            norm2d(norm, base_ch),
            nn.GELU(),
        )
        self.classifier = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x: torch.Tensor, out_size: Tuple[int, int]) -> torch.Tensor:
        x = self.conv(x)
        x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return self.classifier(x)


class ChannelAttention(nn.Module):
    """Channel attention (SE-style)."""

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
    """|Fa-Fb| + (Fa+Fb) + channel attention."""

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
    """Multi-scale fusion decoder (similar to SegFormer MLPDecoder)."""

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


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable conv block with norm + GELU."""

    def __init__(self, in_ch: int, out_ch: int, norm: str = "gn"):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            norm2d(norm, out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BoundaryDecoder(nn.Module):
    """
    Boundary-aware branch (SegFormer-style):
    - Project multi-scale diff features to shared dim
    - Upsample to highest resolution and fuse
    - Light depthwise refinement + 1x1 prediction
    """

    def __init__(
        self, in_channels_list: List[int], embed_dim: int = 192, norm: str = "gn"
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_ch, embed_dim, 1, bias=False),
                    norm2d(norm, embed_dim),
                    nn.GELU(),
                )
                for in_ch in in_channels_list
            ]
        )
        fuse_in = embed_dim * len(in_channels_list)
        self.fuse = nn.Sequential(
            DepthwiseSeparableConv(fuse_in, embed_dim, norm=norm),
            DepthwiseSeparableConv(embed_dim, embed_dim, norm=norm),
        )
        self.refine = DepthwiseSeparableConv(embed_dim, embed_dim, norm=norm)
        self.out_conv = nn.Conv2d(embed_dim, 1, 1)
        self.out_channels = embed_dim

    def forward(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        target_size = features[0].shape[2:]
        aligned = []
        for i, feat in enumerate(features):
            feat = self.proj[i](feat)
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
            aligned.append(feat)
        fused = torch.cat(aligned, dim=1)
        fused = self.fuse(fused)
        refined = self.refine(fused)
        logit = self.out_conv(refined)
        return logit, refined


class EnhancedMLPDecoder(nn.Module):
    """
    Lightweight SegFormer-style decoder with BN fusion and shallow refinement.
    Projects each scale to a shared dim, fuses, refines, and returns feature map.
    """

    def __init__(
        self,
        in_channels_list: List[int],
        embedding_dim: int = 512,
        decoder_depth: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.decoder_depth = decoder_depth
        self.dropout = dropout

        self.linear_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_ch, embedding_dim, 1, bias=False),
                    nn.BatchNorm2d(embedding_dim),
                    nn.ReLU(inplace=True),
                )
                for in_ch in in_channels_list
            ]
        )

        fusion_in_dim = embedding_dim * len(in_channels_list)
        self.fusion_layers = nn.Sequential(
            nn.Conv2d(fusion_in_dim, embedding_dim, 1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
        )

        refinement = []
        in_ch = embedding_dim
        out_ch = embedding_dim // 2
        for _ in range(decoder_depth):
            refinement.extend(
                [
                    nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                ]
            )
            if dropout > 0:
                refinement.append(nn.Dropout2d(dropout))
            in_ch = out_ch
            out_ch = max(64, out_ch // 2)
        self.refinement = nn.Sequential(*refinement)
        self.out_channels = in_ch

    def forward(self, diff_features: List[torch.Tensor]) -> torch.Tensor:
        # assume diff_features have same spatial size
        B, _, H, W = diff_features[0].shape
        mlp_feats = []
        for i, feat in enumerate(diff_features):
            feat = self.linear_layers[i](feat)
            if feat.shape[2:] != (H, W):
                feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)
            mlp_feats.append(feat)
        fused = torch.cat(mlp_feats, dim=1)
        fused = self.fusion_layers(fused)
        refined = self.refinement(fused)
        return refined


class PrototypeChangeHead(nn.Module):
    """
    Prototype-based change scoring: fixed prototypes (e.g., from offline k-means).
    Score = 1 - sum_k min(p_a[k], p_b[k]) where p_* are softmax over prototypes.
    """

    def __init__(self, prototypes: np.ndarray, proto_weight: float = 0.5):
        super().__init__()
        if prototypes.ndim != 2:
            raise ValueError("prototypes must be 2D [K, C]")
        proto = torch.from_numpy(prototypes.astype(np.float32))
        self.register_buffer("proto", proto)  # [K, C]
        self.weight = float(proto_weight)

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        # feat_*: [B, C, H, W]
        proto = F.normalize(self.proto, dim=1)  # [K, C]
        feat_a_n = F.normalize(feat_a, dim=1)
        feat_b_n = F.normalize(feat_b, dim=1)
        sim_a = torch.einsum("bchw,kc->bkhw", feat_a_n, proto)
        sim_b = torch.einsum("bchw,kc->bkhw", feat_b_n, proto)
        p_a = F.softmax(sim_a, dim=1)
        p_b = F.softmax(sim_b, dim=1)
        overlap = torch.sum(torch.minimum(p_a, p_b), dim=1, keepdim=True)  # [B,1,H,W]
        change = 1.0 - overlap
        change = change.clamp(1e-4, 1 - 1e-4)
        proto_logit = torch.logit(change)
        return self.weight * proto_logit  # [B,1,H,W]


class DinoSiameseHead(nn.Module):
    """
    DINOv2 / DINOv3 siamese head for change detection.
    - Supports DINOv2 (torch.hub) and DINOv3 (HF AutoModel).
    - Adapter to reduce dim, GroupNorm by default.
    - forward returns dict {"pred": logits}.
    - Boundary-aware branch (SegFormer-style) sharpens edges for cross-domain generalization.
    """

    def __init__(
        self,
        dino_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        patch: int = 14,
        use_whiten: bool = False,
        embed_dim: int = 256,
        head_channels: int = 256,
        head_depth: int = 3,
        dropout: float = 0.10,
        selected_layers: Tuple[int, ...] = (3, 6, 9, 12),
        adapter_dim: int = 192,
        norm: str = "gn",  # "gn" (recommended) or "bn"
        feat_smooth: bool = True,
        feat_smooth_kernel: int = 3,
        feat_smooth_tau: float = 1.0,
        use_domain_adv: bool = False,
        domain_hidden: int = 256,
        domain_grl: float = 1.0,
        use_style_norm: bool = False,
        proto_path: str = None,
        proto_weight: float = 0.0,
        boundary_dim: int = 0,
        use_layer_ensemble: bool = False,
        layer_head_ch: int = 128,
    ):
        super().__init__()
        self.patch = int(patch)
        self.use_whiten = bool(use_whiten)
        self.selected_layers = list(selected_layers)
        self.adapter_dim = int(adapter_dim)
        self.norm = norm
        self.feat_smooth = bool(feat_smooth)
        self.feat_smooth_kernel = int(max(1, feat_smooth_kernel))
        self.feat_smooth_tau = float(max(1e-6, feat_smooth_tau))
        self.use_style_norm = bool(use_style_norm)
        self.use_domain_adv = bool(use_domain_adv)
        self.domain_grl = float(domain_grl)
        self.proto_path = proto_path
        self.proto_weight = float(proto_weight)
        self.boundary_dim = int(boundary_dim)
        self.use_layer_ensemble = bool(use_layer_ensemble)
        self.layer_head_ch = int(layer_head_ch)

        self.use_hf = "dinov3" in dino_name.lower() or dino_name.startswith(
            "facebook/dinov3"
        )
        if self.use_hf:
            if AutoModel is None:
                raise ImportError(
                    "transformers is required for DINOv3. Please install transformers>=4.56.0."
                )
            self.backbone = AutoModel.from_pretrained(
                dino_name, trust_remote_code=True
            )
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.num_reg = int(getattr(self.backbone.config, "num_register_tokens", 0) or 0)
            self.patch = int(getattr(self.backbone.config, "patch_size", self.patch))
            self.feat_dim = int(getattr(self.backbone.config, "hidden_size", 768))
        else:
            self.backbone = torch.hub.load("facebookresearch/dinov2", dino_name)
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False
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
        self.mlp_decoder = EnhancedMLPDecoder(
            in_channels_list=[self.adapter_dim] * len(self.selected_layers),
            embedding_dim=embed_dim,
            decoder_depth=head_depth,
            dropout=dropout,
        )

        self.domain_head = DomainDiscriminator(embed_dim, domain_hidden) if self.use_domain_adv else None

        self.classifier = nn.Conv2d(self.mlp_decoder.out_channels, 1, 1)

        self.layer_heads = None
        self.fused_decoder = None
        self.fused_head = None
        self.fused_classifier = None
        if self.use_layer_ensemble:
            self.layer_heads = nn.ModuleList(
                [
                    LayerHead(
                        self.adapter_dim,
                        base_ch=max(16, self.layer_head_ch // 2),
                        norm=self.norm,
                    )
                    for _ in self.selected_layers
                ]
            )
            self.fused_decoder = MultiScaleFusionDecoder(
                in_channels_list=[self.adapter_dim] * len(self.selected_layers),
                embed_dim=self.layer_head_ch,
                norm=self.norm,
            )
            self.fused_head = HeavyDecoder(
                in_ch=self.layer_head_ch, base_ch=self.layer_head_ch, norm=self.norm
            )
            self.fused_classifier = nn.Conv2d(self.layer_head_ch, 1, 1)

        self.boundary_head = None
        self.boundary_refine = None
        if self.boundary_dim > 0:
            self.boundary_head = BoundaryDecoder(
                in_channels_list=[self.adapter_dim] * len(self.selected_layers),
                embed_dim=self.boundary_dim,
                norm=self.norm,
            )
            self.boundary_refine = DepthwiseSeparableConv(
                self.mlp_decoder.out_channels + self.boundary_head.out_channels,
                self.mlp_decoder.out_channels,
                norm=self.norm,
            )

        self.proto_head = None
        if proto_path:
            try:
                proto_arr = np.load(proto_path)
                self.proto_head = PrototypeChangeHead(proto_arr, proto_weight=self.proto_weight)
            except Exception as e:
                print(f"Failed to load prototypes from {proto_path}: {e}")
                self.proto_head = None

    def _smooth_feat(self, x: torch.Tensor) -> torch.Tensor:
        """
        Neighborhood consistency smoothing to suppress outlier patches.
        """
        if not self.feat_smooth or self.feat_smooth_kernel <= 1:
            return x
        k = self.feat_smooth_kernel
        mean = F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
        dev = (x - mean).abs().mean(dim=1, keepdim=True)  # [B,1,h,w]
        w = torch.exp(-dev / self.feat_smooth_tau).clamp(0.0, 1.0)
        return w * x + (1.0 - w) * mean

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
    def _register_hooks_once(self):
        if self.use_hf:
            return
        if hasattr(self, "_hooks") and self._hooks:
            return
        self._hook_feats: List[torch.Tensor] = []

        def _make_hook():
            def _hook(_, __, output):
                self._hook_feats.append(output)

            return _hook

        self._hooks = []
        for idx in self.selected_layers:
            self._hooks.append(
                self.backbone.blocks[idx - 1].register_forward_hook(_make_hook())
            )

    @torch.no_grad()
    def _extract_pair_features(
        self, img_a: torch.Tensor, img_b: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], Tuple[int, int], Tuple[int, int]]:
        """
        Extract features for img_a/img_b in one forward (batch concat) to save time.
        Return two lists of [B,C,h,w], padded size, and original size.
        """
        x = torch.cat([img_a, img_b], dim=0)
        x, (H0, W0) = self._pad_to_patch_multiple(x)
        B2, _, Hp, Wp = x.shape
        assert B2 % 2 == 0, "Batch should be even when concatenating A/B"
        B = B2 // 2
        h, w = Hp // self.patch, Wp // self.patch

        feats_raw: List[torch.Tensor] = []

        if self.use_hf:
            out = self.backbone(
                pixel_values=x, output_hidden_states=True, return_dict=True
            )
            hidden_states = out.hidden_states
            if hidden_states is None:
                hidden_states = [out.last_hidden_state]
            for idx in self.selected_layers:
                real_idx = min(idx, len(hidden_states) - 1)
                feats_raw.append(hidden_states[real_idx])
        else:
            self._hook_feats = []
            self._register_hooks_once()
            _ = self.backbone.forward_features(x)
            feats_raw = self._hook_feats
            if len(feats_raw) != len(self.selected_layers):
                raise RuntimeError(
                    f"Expected {len(self.selected_layers)} hooks, got {len(feats_raw)}"
                )

        feats_a: List[torch.Tensor] = []
        feats_b: List[torch.Tensor] = []
        for feat in feats_raw:
            if feat.ndim != 3:
                raise ValueError(f"Unexpected hooked feat shape: {feat.shape}")
            tokens = feat[:, 1 + self.num_reg :, :]  # [2B, Npatch, C]
            if tokens.shape[1] != h * w:
                tokens = tokens[:, -h * w :, :]
            feat_map = tokens.transpose(1, 2).reshape(B2, tokens.shape[-1], h, w)
            feat_map = F.normalize(feat_map, dim=1)

            if self.use_whiten:
                mu = feat_map.mean(dim=(2, 3), keepdim=True)
                sd = feat_map.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
                feat_map = (feat_map - mu) / sd

            feats_a.append(feat_map[:B])
            feats_b.append(feat_map[B:])

        return feats_a, feats_b, (Hp, Wp), (H0, W0)

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor):
        B, _, H, W = img_a.shape

        with torch.no_grad():
            fa_list, fb_list, (Hp, Wp), (H0, W0) = self._extract_pair_features(
                img_a, img_b
            )

        diff_feats = []
        logits_list = []
        deep_fa = None
        deep_fb = None
        for i, (fa, fb, dm) in enumerate(zip(fa_list, fb_list, self.diff_modules)):
            fa = self.adapters[i](fa)
            fb = self.adapters[i](fb)
            fa = self._smooth_feat(fa)
            fb = self._smooth_feat(fb)
            if i == len(self.diff_modules) - 1:
                deep_fa = fa
                deep_fb = fb
            diff = dm(fa, fb)
            diff_feats.append(diff)
            if self.layer_heads is not None:
                logits_list.append(self.layer_heads[i](diff, out_size=(H, W)))

        fused = self.decoder(diff_feats)  # [B, embed_dim, h, w]
        if self.use_style_norm:
            fused = fused - fused.mean(dim=(2, 3), keepdim=True)
            fused = fused / fused.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)

        domain_logit = None
        if self.domain_head is not None:
            dom_feat = grad_reverse(fused, self.domain_grl)
            domain_logit = self.domain_head(dom_feat)

        refined = self.mlp_decoder(diff_feats)  # [B, C', h, w]

        boundary_logit = None
        if self.boundary_head is not None:
            boundary_logit, boundary_feat = self.boundary_head(diff_feats)
            boundary_feat = F.interpolate(
                boundary_feat, size=refined.shape[-2:], mode="bilinear", align_corners=False
            )
            refined = torch.cat([refined, boundary_feat], dim=1)
            refined = self.boundary_refine(refined)
            boundary_att = torch.sigmoid(
                F.interpolate(boundary_logit, size=refined.shape[-2:], mode="bilinear", align_corners=False)
            )
            refined = refined * (1.0 + boundary_att)

        logit_patch = self.classifier(refined)  # [B, 1, h, w]

        proto_logit = None
        if self.proto_head is not None and deep_fa is not None and deep_fb is not None:
            proto_logit = self.proto_head(deep_fa, deep_fb)
            logit_patch = logit_patch + proto_logit

        logit_pad = F.interpolate(
            logit_patch, size=(Hp, Wp), mode="bilinear", align_corners=False
        )
        logit = logit_pad[..., :H0, :W0]
        if logit.shape[-2:] != (H, W):
            logit = F.interpolate(
                logit, size=(H, W), mode="bilinear", align_corners=False
            )

        proto_up = None
        if proto_logit is not None:
            proto_up = F.interpolate(
                proto_logit, size=(Hp, Wp), mode="bilinear", align_corners=False
            )
            proto_up = proto_up[..., :H0, :W0]
            if proto_up.shape[-2:] != (H, W):
                proto_up = F.interpolate(
                    proto_up, size=(H, W), mode="bilinear", align_corners=False
                )

        logits_all = None
        pred_logit = logit
        if self.layer_heads is not None:
            fused = self.fused_decoder(diff_feats)
            fused = self.fused_head(fused)
            fused_logit = self.fused_classifier(fused)
            fused_logit = F.interpolate(
                fused_logit, size=(Hp, Wp), mode="bilinear", align_corners=False
            )
            fused_logit = fused_logit[..., :H0, :W0]
            if fused_logit.shape[-2:] != (H, W):
                fused_logit = F.interpolate(
                    fused_logit, size=(H, W), mode="bilinear", align_corners=False
                )
            if proto_up is not None:
                fused_logit = fused_logit + proto_up
            logits_all = torch.stack(logits_list + [fused_logit], dim=0)
            pred_logit = fused_logit

        boundary_up = None
        if boundary_logit is not None:
            boundary_up = F.interpolate(
                boundary_logit, size=(Hp, Wp), mode="bilinear", align_corners=False
            )
            boundary_up = boundary_up[..., :H0, :W0]
            if boundary_up.shape[-2:] != (H, W):
                boundary_up = F.interpolate(
                    boundary_up, size=(H, W), mode="bilinear", align_corners=False
                )

        return {
            "pred": pred_logit,
            "feat": refined,
            "domain_logit": domain_logit,
            "proto_logit": proto_logit,
            "boundary": boundary_up,
            "logits_all": logits_all,
        }


class DinoFrozenA0Head(nn.Module):
    """
    A0 baseline:
      Frozen DINOv3/DINOv2 backbone + single 1×1 conv head (no DLF decoder, no MHE multi-head).

    - Extract one layer's patch-token features for T1/T2 (default: layer=12)
    - Build diff feature: |Fa - Fb|
    - Predict change logit with a single 1×1 conv
    """

    def __init__(
        self,
        dino_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        layer: int = 12,
        patch: int = 14,
        use_whiten: bool = False,
    ):
        super().__init__()
        self.patch = int(patch)
        self.use_whiten = bool(use_whiten)
        self.layer = int(layer)

        self.use_hf = "dinov3" in dino_name.lower() or dino_name.startswith("facebook/dinov3")
        if self.use_hf:
            if AutoModel is None:
                raise ImportError("transformers is required for DINOv3. Please install transformers>=4.56.0.")
            self.backbone = AutoModel.from_pretrained(dino_name, trust_remote_code=True)
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.num_reg = int(getattr(self.backbone.config, "num_register_tokens", 0) or 0)
            self.patch = int(getattr(self.backbone.config, "patch_size", self.patch))
            self.feat_dim = int(getattr(self.backbone.config, "hidden_size", 768))
        else:
            self.backbone = torch.hub.load("facebookresearch/dinov2", dino_name)
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.num_reg = int(getattr(self.backbone, "num_register_tokens", 0))
            with torch.no_grad():
                dummy = torch.randn(1, 3, 224, 224)
                out = self.backbone.forward_features(dummy)
                self.feat_dim = out["x_norm_patchtokens"].shape[-1]

        self.classifier = nn.Conv2d(self.feat_dim, 1, kernel_size=1, bias=True)

    @torch.no_grad()
    def _pad_to_patch_multiple(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        _, _, H, W = x.shape
        H2 = ((H + self.patch - 1) // self.patch) * self.patch
        W2 = ((W + self.patch - 1) // self.patch) * self.patch
        pad_h, pad_w = H2 - H, W2 - W
        if pad_h == 0 and pad_w == 0:
            return x, (H, W)
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, (H, W)

    @torch.no_grad()
    def _register_hooks_once(self):
        if self.use_hf:
            return
        if hasattr(self, "_hooks") and self._hooks:
            return
        self._hook_feats: List[torch.Tensor] = []

        def _hook(_, __, output):
            self._hook_feats.append(output)

        self._hooks = [self.backbone.blocks[self.layer - 1].register_forward_hook(_hook)]

    @torch.no_grad()
    def _extract_pair_features(self, img_a: torch.Tensor, img_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([img_a, img_b], dim=0)
        x, _ = self._pad_to_patch_multiple(x)
        B2, _, Hp, Wp = x.shape
        assert B2 % 2 == 0, "Batch should be even when concatenating A/B"
        B = B2 // 2
        h, w = Hp // self.patch, Wp // self.patch

        if self.use_hf:
            out = self.backbone(pixel_values=x, output_hidden_states=True, return_dict=True)
            hidden_states = out.hidden_states
            if hidden_states is None:
                hidden_states = [out.last_hidden_state]
            real_idx = min(self.layer, len(hidden_states) - 1)
            feat = hidden_states[real_idx]
        else:
            self._hook_feats = []
            self._register_hooks_once()
            _ = self.backbone.forward_features(x)
            if len(self._hook_feats) != 1:
                raise RuntimeError(f"Expected 1 hook feat, got {len(self._hook_feats)}")
            feat = self._hook_feats[0]

        if feat.ndim != 3:
            raise ValueError(f"Unexpected feat shape: {feat.shape}")
        tokens = feat[:, 1 + self.num_reg :, :]  # [2B, Npatch, C]
        if tokens.shape[1] != h * w:
            tokens = tokens[:, -h * w :, :]
        feat_map = tokens.transpose(1, 2).reshape(B2, tokens.shape[-1], h, w)
        feat_map = F.normalize(feat_map, dim=1)
        if self.use_whiten:
            mu = feat_map.mean(dim=(2, 3), keepdim=True)
            sd = feat_map.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            feat_map = (feat_map - mu) / sd

        return feat_map[:B], feat_map[B:]

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor):
        _, _, H, W = img_a.shape
        with torch.no_grad():
            fa, fb = self._extract_pair_features(img_a, img_b)
        diff = torch.abs(fa - fb)
        logit_patch = self.classifier(diff)
        logit = F.interpolate(logit_patch, size=(H, W), mode="bilinear", align_corners=False)
        return {"pred": logit}


__all__ = [
    "DinoSiameseHead",
    "DinoFrozenA0Head",
    "ChannelAttention",
    "DifferenceModule",
    "MultiScaleFusionDecoder",
    "HeavyDecoder",
    "LayerHead",
    "EnhancedMLPDecoder",
    "PrototypeChangeHead",
    "BoundaryDecoder",
    "DepthwiseSeparableConv",
]
