import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from .extractors.fph import FPH

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math

from ForensicHub.core.base_model import BaseModel
from ForensicHub.registry import register_model
from IMDLBenCo.modules.extractors.high_frequency_feature_extraction import (
    FFTExtractor,
    DCTExtractor
)
from IMDLBenCo.modules.extractors.sobel import SobelFilter
from IMDLBenCo.modules.extractors.bayar_conv import BayerConv


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class MixVisionTransformer(BaseModel):
    def __init__(self, input_head=None, output_type='label', num_channels=0, pretrain_path='', image_size=512,
                 patch_size=4, in_chans=3,
                 embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
                 qk_scale=None, drop_rate=0.0,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.depths = depths
        self.output_type = output_type

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=image_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])

        self.patch_embed2 = OverlapPatchEmbed(img_size=image_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=image_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=image_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        original_first_layer = self.patch_embed1.proj

        self.load_state_dict(torch.load(pretrain_path),
                             strict=False)
        if input_head != None:
            self.input_head = input_head
            new_first_layer = nn.Conv2d(num_channels + 3, original_first_layer.out_channels,
                                        kernel_size=original_first_layer.kernel_size,
                                        stride=original_first_layer.stride,
                                        padding=original_first_layer.padding, bias=False)
            new_first_layer.weight.data[:, :3, :, :] = original_first_layer.weight.data.clone()[:, :3, :, :]

            if num_channels > 0:
                new_first_layer.weight.data[:, 3:, :, :] = torch.nn.init.kaiming_normal_(
                    new_first_layer.weight[:, 3:, :, :])
            self.patch_embed1.proj = new_first_layer
        else:
            self.input_head = None

        if output_type == 'label':
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(512, 1)
            )
        elif output_type == 'mask':
            out_channels = 512
            self.head = nn.Sequential(
                nn.Conv2d(out_channels, out_channels // 2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 2, out_channels // 4, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 4, 1, kernel_size=1),
                nn.Upsample(size=(image_size, image_size), mode='bilinear', align_corners=False)
            )
        else:
            raise ValueError(f"Unsupported output_type: {output_type}")

        self.head.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, image, *args, **kwargs):
        if self.input_head != None:
            feature = self.input_head(image)
            input = torch.cat([image, feature], dim=1)
        else:
            input = image
        B = input.shape[0]

        # stage 1
        x, H, W = self.patch_embed1(input)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = self.head(x)

        if self.output_type == 'label':
            if len(out.shape) == 2:
                out = out.squeeze(dim=1)
            loss = F.binary_cross_entropy_with_logits(out, kwargs['label'].float())
            pred = out.sigmoid()
        else:
            loss = F.binary_cross_entropy_with_logits(out, kwargs['mask'].float())
            pred = out.sigmoid()
        out_dict = {
            "backward_loss": loss,
            f"pred_{self.output_type}": pred,
            "visual_loss": {
                "combined_loss": loss
            }
        }

        return out_dict


@register_model("Segformerb2")
class Segformerb2(MixVisionTransformer):
    def __init__(self, output_type='label', pretrain_path='', image_size=256):
        super().__init__(output_type=output_type, pretrain_path=pretrain_path, image_size=image_size,
                         patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
                         qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3],
                         sr_ratios=[8, 4, 2, 1],
                         drop_rate=0.0, drop_path_rate=0.1)


@register_model("Segformerb3")
class Segformerb3(MixVisionTransformer):
    def __init__(self, output_type='label', pretrain_path='', image_size=256):
        super().__init__(output_type=output_type, pretrain_path=pretrain_path, image_size=image_size,
                         patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
                         qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3],
                         sr_ratios=[8, 4, 2, 1],
                         drop_rate=0.0, drop_path_rate=0.1)


@register_model("SobelSegformerb3")
class SobelSegformerb3(MixVisionTransformer):
    def __init__(self, output_type='label', pretrain_path='', image_size=256):
        super().__init__(input_head=SobelFilter(), num_channels=1, output_type=output_type, pretrain_path=pretrain_path,
                         image_size=image_size,
                         patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
                         qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3],
                         sr_ratios=[8, 4, 2, 1],
                         drop_rate=0.0, drop_path_rate=0.1)


@register_model("BayerSegformerb3")
class BayerSegformerb3(MixVisionTransformer):
    def __init__(self, output_type='label', pretrain_path='', image_size=256):
        super().__init__(input_head=BayerConv(), num_channels=3, output_type=output_type, pretrain_path=pretrain_path,
                         image_size=image_size,
                         patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
                         qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3],
                         sr_ratios=[8, 4, 2, 1],
                         drop_rate=0.0, drop_path_rate=0.1)


@register_model("FFTSegformerb3")
class FFTSegformerb3(MixVisionTransformer):
    def __init__(self, output_type='label', pretrain_path='', image_size=256):
        super().__init__(input_head=FFTExtractor(), num_channels=3, output_type=output_type,
                         pretrain_path=pretrain_path,
                         image_size=image_size,
                         patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
                         qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3],
                         sr_ratios=[8, 4, 2, 1],
                         drop_rate=0.0, drop_path_rate=0.1)


@register_model("DCTSegformerb3")
class DCTSegformerb3(MixVisionTransformer):
    def __init__(self, output_type='label', pretrain_path='', image_size=256):
        super().__init__(input_head=DCTExtractor(), num_channels=3, output_type=output_type,
                         pretrain_path=pretrain_path,
                         image_size=image_size,
                         patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
                         qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3],
                         sr_ratios=[8, 4, 2, 1],
                         drop_rate=0.0, drop_path_rate=0.1)


@register_model("QtSegformerb3")
class QtSegformerb3(MixVisionTransformer):
    def __init__(self, output_type='label', pretrain_path='', image_size=256):
        super().__init__(input_head=None, output_type=output_type,
                         pretrain_path=pretrain_path, image_size=image_size,
                         patch_size=4, embed_dims=[64, 128, 320, 512],
                         num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
                         qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                         depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
                         drop_rate=0.0, drop_path_rate=0.1)

        self.fph = FPH()

        # 分解Segformer结构
        self.patch_embed1_proj = self.patch_embed1.proj
        self.patch_embed1_norm = self.patch_embed1.norm

        # 融合层 (Segformer第一层输出通道是64)
        self.fusion_conv = nn.Conv2d(64 + 256, 64, kernel_size=1)

        # 重建head确保维度匹配
        out_channels = 512  # SegformerB3最终特征维度
        if output_type == 'label':
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(out_channels, 1)
            )
        else:
            self.head = nn.Sequential(
                nn.Conv2d(out_channels, out_channels // 2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 2, out_channels // 4, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 4, 1, kernel_size=1),
                nn.Upsample(size=(image_size, image_size), mode='bilinear', align_corners=False)
            )

    def forward(self, image, dct, qt, *args, **kwargs):
        dct = dct.squeeze(1).long()  # [B,1,H,W] -> [B,H,W]
        # FPH特征提取 (B, 256, H/8, W/8)
        x_aux = self.fph(dct, qt)

        # 主干初始部分
        B = image.shape[0]
        x = self.patch_embed1_proj(image)  # [B, 64, H/4, W/4]
        _, _, H, W = x.shape

        # 尺寸对齐
        if x.shape[-2:] != x_aux.shape[-2:]:
            x_aux = F.interpolate(x_aux, size=(H, W), mode='bilinear', align_corners=False)

        # 拼接并融合
        x = torch.cat([x, x_aux], dim=1)
        x = self.fusion_conv(x)

        # 继续Segformer流程
        x = x.flatten(2).transpose(1, 2)
        x = self.patch_embed1_norm(x)

        # stage 1
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = self.head(x)

        if self.output_type == 'label':
            if len(out.shape) == 2:
                out = out.squeeze(dim=1)
            loss = F.binary_cross_entropy_with_logits(out, kwargs['label'].float())
            pred = out.sigmoid()
        else:
            loss = F.binary_cross_entropy_with_logits(out, kwargs['mask'].float())
            pred = out.sigmoid()

        return {
            "backward_loss": loss,
            f"pred_{self.output_type}": pred,
            "visual_loss": {"combined_loss": loss}
        }
