import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models as models
import math
import numpy as np

from einops import rearrange
from timm.layers import to_2tuple


class ConvModule(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 inplace=True,
                 with_spectral_norm=False,
                 padding_mode='zeros',
                 order=('conv', 'norm', 'act')):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace)

    def forward(self, input):
        out = self.conv(input)
        out = self.norm(out)
        out = self.act(out)
        return out


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class _NonLocalNd(nn.Module):
    """Basic Non-local module.

    This module is proposed in
    "Non-local Neural Networks"
    Paper reference: https://arxiv.org/abs/1711.07971
    Code reference: https://github.com/AlexHex7/Non-local_pytorch

    Args:
        in_channels (int): Channels of the input feature map.
        reduction (int): Channel reduction ratio. Default: 2.
        use_scale (bool): Whether to scale pairwise_weight by
            `1/sqrt(inter_channels)` when the mode is `embedded_gaussian`.
            Default: True.
        conv_cfg (None | dict): The config dict for convolution layers.
            If not specified, it will use `nn.Conv2d` for convolution layers.
            Default: None.
        norm_cfg (None | dict): The config dict for normalization layers.
            Default: None. (This parameter is only applicable to conv_out.)
        mode (str): Options are `gaussian`, `concatenation`,
            `embedded_gaussian` and `dot_product`. Default: embedded_gaussian.
    """

    def __init__(self,
                 in_channels,
                 reduction=2,
                 use_scale=True,
                 conv_cfg=None,
                 norm_cfg=None,
                 mode='embedded_gaussian',
                 **kwargs):
        super(_NonLocalNd, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.use_scale = use_scale
        self.inter_channels = max(in_channels // reduction, 1)
        self.mode = mode

        if mode not in [
            'gaussian', 'embedded_gaussian', 'dot_product', 'concatenation'
        ]:
            raise ValueError("Mode should be in 'gaussian', 'concatenation', "
                             f"'embedded_gaussian' or 'dot_product', but got "
                             f'{mode} instead.')

        # g, theta, phi are defaulted as `nn.ConvNd`.
        # Here we use ConvModule for potential usage.
        self.g = ConvModule(
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            act_cfg=None)
        self.conv_out = ConvModule(
            self.inter_channels,
            self.in_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        if self.mode != 'gaussian':
            self.theta = ConvModule(
                self.in_channels,
                self.inter_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                act_cfg=None)
            self.phi = ConvModule(
                self.in_channels,
                self.inter_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                act_cfg=None)

        if self.mode == 'concatenation':
            self.concat_project = ConvModule(
                self.inter_channels * 2,
                1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                act_cfg=dict(type='ReLU'))

        # self.init_weights(**kwargs)

    # def init_weights(self, std=0.01, zeros_init=True):
    #     if self.mode != 'gaussian':
    #         for m in [self.g, self.theta, self.phi]:
    #             normal_init(m.conv, std=std)
    #     else:
    #         normal_init(self.g.conv, std=std)
    #     if zeros_init:
    #         if self.conv_out.norm_cfg is None:
    #             constant_init(self.conv_out.conv, 0)
    #         else:
    #             constant_init(self.conv_out.norm, 0)
    #     else:
    #         if self.conv_out.norm_cfg is None:
    #             normal_init(self.conv_out.conv, std=std)
    #         else:
    #             normal_init(self.conv_out.norm, std=std)

    def gaussian(self, theta_x, phi_x):
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def embedded_gaussian(self, theta_x, phi_x):
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        if self.use_scale:
            # theta_x.shape[-1] is `self.inter_channels`
            pairwise_weight /= theta_x.shape[-1] ** 0.5
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def dot_product(self, theta_x, phi_x):
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight /= pairwise_weight.shape[-1]
        return pairwise_weight

    def concatenation(self, theta_x, phi_x):
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        h = theta_x.size(2)
        w = phi_x.size(3)
        theta_x = theta_x.repeat(1, 1, 1, w)
        phi_x = phi_x.repeat(1, 1, h, 1)

        concat_feature = torch.cat([theta_x, phi_x], dim=1)
        pairwise_weight = self.concat_project(concat_feature)
        n, _, h, w = pairwise_weight.size()
        pairwise_weight = pairwise_weight.view(n, h, w)
        pairwise_weight /= pairwise_weight.shape[-1]

        return pairwise_weight

    def forward(self, x):
        # Assume `reduction = 1`, then `inter_channels = C`
        # or `inter_channels = C` when `mode="gaussian"`

        # NonLocal1d x: [N, C, H]
        # NonLocal2d x: [N, C, H, W]
        # NonLocal3d x: [N, C, T, H, W]
        n = x.size(0)

        # NonLocal1d g_x: [N, H, C]
        # NonLocal2d g_x: [N, HxW, C]
        # NonLocal3d g_x: [N, TxHxW, C]
        g_x = self.g(x).view(n, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # NonLocal1d theta_x: [N, H, C], phi_x: [N, C, H]
        # NonLocal2d theta_x: [N, HxW, C], phi_x: [N, C, HxW]
        # NonLocal3d theta_x: [N, TxHxW, C], phi_x: [N, C, TxHxW]
        if self.mode == 'gaussian':
            theta_x = x.view(n, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            if self.sub_sample:
                phi_x = self.phi(x).view(n, self.in_channels, -1)
            else:
                phi_x = x.view(n, self.in_channels, -1)
        elif self.mode == 'concatenation':
            theta_x = self.theta(x).view(n, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(n, self.inter_channels, 1, -1)
        else:
            theta_x = self.theta(x).view(n, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi(x).view(n, self.inter_channels, -1)

        pairwise_func = getattr(self, self.mode)
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        pairwise_weight = pairwise_func(theta_x, phi_x)

        # NonLocal1d y: [N, H, C]
        # NonLocal2d y: [N, HxW, C]
        # NonLocal3d y: [N, TxHxW, C]
        y = torch.matmul(pairwise_weight, g_x)
        # NonLocal1d y: [N, C, H]
        # NonLocal2d y: [N, C, H, W]
        # NonLocal3d y: [N, C, T, H, W]
        y = y.permute(0, 2, 1).contiguous().reshape(n, self.inter_channels,
                                                    *x.size()[2:])

        output = x + self.conv_out(y)

        return output


class NonLocal2d(_NonLocalNd):
    """2D Non-local module.

    Args:
        in_channels (int): Same as `NonLocalND`.
        sub_sample (bool): Whether to apply max pooling after pairwise
            function (Note that the `sub_sample` is applied on spatial only).
            Default: False.
        conv_cfg (None | dict): Same as `NonLocalND`.
            Default: dict(type='Conv2d').
    """

    _abbr_ = 'nonlocal_block'

    def __init__(self,
                 in_channels,
                 sub_sample=False,
                 conv_cfg=dict(type='Conv2d'),
                 **kwargs):
        super(NonLocal2d, self).__init__(
            in_channels, conv_cfg=conv_cfg, **kwargs)

        self.sub_sample = sub_sample

        if sub_sample:
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            self.g = nn.Sequential(self.g, max_pool_layer)
            if self.mode != 'gaussian':
                self.phi = nn.Sequential(self.phi, max_pool_layer)
            else:
                self.phi = max_pool_layer


class MLP(nn.Module):
    """
    Linear Embedding: github.com/NVlabs/SegFormer
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class PatchEmbed(nn.Module):
    """
    Patch Embedding: github.com/SwinTransformer/
    """

    def __init__(self, proj_type='pool', patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj_type = proj_type
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if proj_type == 'conv':
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size,
                                  groups=patch_size * patch_size)
        elif proj_type == 'pool':
            self.proj = nn.ModuleList([nn.MaxPool2d(kernel_size=patch_size, stride=patch_size),
                                       nn.AvgPool2d(kernel_size=patch_size, stride=patch_size)])
        else:
            raise NotImplementedError(f'{proj_type} is not currently supported.')

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        if self.proj_type == 'conv':
            x = self.proj(x)  # B C Wh Ww
        else:
            x = 0.5 * (self.proj[0](x) + self.proj[1](x))

        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        return x


class LawinAttn(NonLocal2d):
    def __init__(self, *arg, head=1,
                 patch_size=None, **kwargs):
        super().__init__(*arg, **kwargs)
        self.head = head
        self.patch_size = patch_size

        if self.head != 1:
            self.position_mixing = nn.ModuleList(
                [nn.Linear(patch_size * patch_size, patch_size * patch_size) for _ in range(self.head)])

    def forward(self, query, context):
        # x: [N, C, H, W]

        n = context.size(0)
        n, c, h, w = context.shape

        if self.head != 1:
            context = context.reshape(n, c, -1)
            context_mlp = []
            for hd in range(self.head):
                context_crt = context[:, (c // self.head) * (hd):(c // self.head) * (hd + 1), :]
                context_mlp.append(self.position_mixing[hd](context_crt))

            context_mlp = torch.cat(context_mlp, dim=1)
            context = context + context_mlp
            context = context.reshape(n, c, h, w)

        # g_x: [N, HxW, C]
        g_x = self.g(context).view(n, self.inter_channels, -1)
        g_x = rearrange(g_x, 'b (h dim) n -> (b h) dim n', h=self.head)
        g_x = g_x.permute(0, 2, 1)

        # theta_x: [N, HxW, C], phi_x: [N, C, HxW]
        if self.mode == 'gaussian':
            theta_x = query.view(n, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            if self.sub_sample:
                phi_x = self.phi(context).view(n, self.in_channels, -1)
            else:
                phi_x = context.view(n, self.in_channels, -1)
        elif self.mode == 'concatenation':
            theta_x = self.theta(query).view(n, self.inter_channels, -1, 1)
            phi_x = self.phi(context).view(n, self.inter_channels, 1, -1)
        else:
            theta_x = self.theta(query).view(n, self.inter_channels, -1)
            theta_x = rearrange(theta_x, 'b (h dim) n -> (b h) dim n', h=self.head)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi(context).view(n, self.inter_channels, -1)
            phi_x = rearrange(phi_x, 'b (h dim) n -> (b h) dim n', h=self.head)

        pairwise_func = getattr(self, self.mode)
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = pairwise_func(theta_x, phi_x)

        # y: [N, HxW, C]
        y = torch.matmul(pairwise_weight, g_x)
        y = rearrange(y, '(b h) n dim -> b n (h dim)', h=self.head)
        # y: [N, C, H, W]
        y = y.permute(0, 2, 1).contiguous().reshape(n, self.inter_channels,
                                                    *query.size()[2:])

        output = query + self.conv_out(y)

        return output


class LawinHead(nn.Module):
    def __init__(self, embed_dim=768, use_scale=True, reduction=2, num_classes=2, **kwargs):
        super(LawinHead, self).__init__()
        self.conv_cfg = None
        self.norm_cfg = None
        self.act_cfg = None
        self.in_channels = [64, 128, 256, 512]
        self.lawin_8 = LawinAttn(in_channels=512, reduction=reduction, use_scale=use_scale, conv_cfg=self.conv_cfg,
                                 norm_cfg=self.norm_cfg, mode='embedded_gaussian', head=64, patch_size=8)
        self.lawin_4 = LawinAttn(in_channels=512, reduction=reduction, use_scale=use_scale, conv_cfg=self.conv_cfg,
                                 norm_cfg=self.norm_cfg, mode='embedded_gaussian', head=16, patch_size=8)
        self.lawin_2 = LawinAttn(in_channels=512, reduction=reduction, use_scale=use_scale, conv_cfg=self.conv_cfg,
                                 norm_cfg=self.norm_cfg, mode='embedded_gaussian', head=4, patch_size=8)

        self.image_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                        ConvModule(512, 512, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg,
                                                   act_cfg=self.act_cfg))
        self.linear_c4 = MLP(input_dim=self.in_channels[-1], embed_dim=embed_dim)
        self.linear_c3 = MLP(input_dim=self.in_channels[2], embed_dim=embed_dim)
        self.linear_c2 = MLP(input_dim=self.in_channels[1], embed_dim=embed_dim)
        self.linear_c1 = MLP(input_dim=self.in_channels[0], embed_dim=48)

        self.linear_fuse = ConvModule(
            in_channels=embed_dim * 3,
            out_channels=512,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.short_path = ConvModule(
            in_channels=512,
            out_channels=512,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.cat = ConvModule(
            in_channels=512 * 5,
            out_channels=512,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.low_level_fuse = ConvModule(
            in_channels=560,
            out_channels=512,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.ds_8 = PatchEmbed(proj_type='pool', patch_size=8, in_chans=512, embed_dim=512, norm_layer=nn.LayerNorm)
        self.ds_4 = PatchEmbed(proj_type='pool', patch_size=4, in_chans=512, embed_dim=512, norm_layer=nn.LayerNorm)
        self.ds_2 = PatchEmbed(proj_type='pool', patch_size=2, in_chans=512, embed_dim=512, norm_layer=nn.LayerNorm)

        self.conv_seg = nn.Conv2d(512, num_classes, kernel_size=1)

    def get_context(self, x, patch_size):
        n, _, h, w = x.shape
        context = []
        for i, r in enumerate([8, 4, 2]):
            _context = F.unfold(x, kernel_size=patch_size * r, stride=patch_size, padding=int((r - 1) / 2 * patch_size))
            _context = rearrange(_context, 'b (c ph pw) (nh nw) -> (b nh nw) c ph pw', ph=patch_size * r,
                                 pw=patch_size * r, nh=h // patch_size, nw=w // patch_size)
            context.append(getattr(self, f'ds_{r}')(_context))

        return context

    def cls_seg(self, feat):
        """Classify each pixel."""

        output = self.conv_seg(feat)
        return output
# class LawinHead(nn.Module):
#     def __init__(self, embed_dim=384, use_scale=True, reduction=4, num_classes=2, **kwargs):  # 关键参数调整
#         super(LawinHead, self).__init__()
#         self.conv_cfg = None
#         self.norm_cfg = None
#         self.act_cfg = None
        
#         # 1. 调整输入通道数
#         self.in_channels = [64, 96, 192, 384]  # 原[64, 128, 320, 512]
        
#         # 2. 降低注意力模块通道数
#         self.lawin_8 = LawinAttn(in_channels=384, reduction=reduction, use_scale=use_scale,  # 原512
#                                  conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, 
#                                  mode='embedded_gaussian', head=32, patch_size=8)  # head减半
#         self.lawin_4 = LawinAttn(in_channels=384, reduction=reduction, use_scale=use_scale,
#                                  conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg,
#                                  mode='embedded_gaussian', head=8, patch_size=8)
#         self.lawin_2 = LawinAttn(in_channels=384, reduction=reduction, use_scale=use_scale,
#                                  conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg,
#                                  mode='embedded_gaussian', head=2, patch_size=8)

#         # 3. 压缩图像池化通道
#         self.image_pool = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             ConvModule(384, 192, 1,  # 原512->512
#                      conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg,
#                      act_cfg=self.act_cfg)
#         )

#         # 4. 调整MLP维度
#         self.linear_c4 = MLP(input_dim=384, embed_dim=embed_dim)  # 原512
#         self.linear_c3 = MLP(input_dim=192, embed_dim=embed_dim)  # 原320
#         self.linear_c2 = MLP(input_dim=96, embed_dim=embed_dim)   # 原128
#         self.linear_c1 = MLP(input_dim=64, embed_dim=32)          # 原48

#         # 5. 降低特征融合通道
#         self.linear_fuse = ConvModule(
#             in_channels=embed_dim * 3,
#             out_channels=256,  # 原512
#             kernel_size=1,
#             norm_cfg=dict(type='SyncBN', requires_grad=True)
#         )

#         # 6. 压缩shortcut路径
#         self.short_path = ConvModule(
#             in_channels=256,  # 原512
#             out_channels=256,
#             kernel_size=1,
#             norm_cfg=dict(type='SyncBN', requires_grad=True)
#         )

#         # 7. 调整特征拼接
#         self.cat = ConvModule(
#             in_channels=256 * 5,  # 原512*5
#             out_channels=256,
#             kernel_size=1,
#             norm_cfg=dict(type='SyncBN', requires_grad=True)
#         )

#         # 8. 简化低层融合
#         self.low_level_fuse = ConvModule(
#             in_channels=256 + 32,  # 原512+48=560
#             out_channels=256,
#             kernel_size=1,
#             norm_cfg=dict(type='SyncBN', requires_grad=True)
#         )

#         # 9. 调整上下文生成维度
#         self.ds_8 = PatchEmbed(proj_type='pool', patch_size=8, 
#                              in_chans=256, embed_dim=128,  # 原512->512
#                              norm_layer=nn.LayerNorm)
#         self.ds_4 = PatchEmbed(proj_type='pool', patch_size=4,
#                              in_chans=256, embed_dim=128,
#                              norm_layer=nn.LayerNorm)
#         self.ds_2 = PatchEmbed(proj_type='pool', patch_size=2,
#                              in_chans=256, embed_dim=128,
#                              norm_layer=nn.LayerNorm)

#         # 10. 最终输出层调整
#         self.conv_seg = nn.Conv2d(256, num_classes, kernel_size=1)  # 原512

    # forward方法保持完全不变

    def forward(self, inputs):

        # inputs = self._transform_inputs(inputs)
        c1, c2, c3, c4 = inputs

        ############### MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])  # (n, c, 32, 32)
        _c4 = F.interpolate(_c4, size=c2.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c2.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2], dim=1))  # (n, c, 128, 128)
        n, _, h, w = _c.shape

        ############### Lawin attention spatial pyramid pooling ###########
        patch_size = 8
        context = self.get_context(_c, patch_size)
        query = F.unfold(_c, kernel_size=patch_size, stride=patch_size)
        query = rearrange(query, 'b (c ph pw) (nh nw) -> (b nh nw) c ph pw', ph=patch_size, pw=patch_size,
                          nh=h // patch_size, nw=w // patch_size)

        output = []
        output.append(self.short_path(_c))
        output.append(F.interpolate(self.image_pool(_c),
                                    size=(h, w),
                                    mode='bilinear',
                                    align_corners=False))

        for i, r in enumerate([8, 4, 2]):
            _output = getattr(self, f'lawin_{r}')(query, context[i])
            _output = rearrange(_output, '(b nh nw) c ph pw -> b c (nh ph) (nw pw)', ph=patch_size, pw=patch_size,
                                nh=h // patch_size, nw=w // patch_size)
            _output = F.interpolate(_output,
                                    size=(h, w),
                                    mode='bilinear',
                                    align_corners=False)
            output.append(_output)

        output = self.cat(torch.cat(output, dim=1))

        ############### Low-level feature enhancement ###########
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        output = F.interpolate(output, size=c1.size()[2:], mode='bilinear', align_corners=False)
        output = self.low_level_fuse(torch.cat([output, _c1], dim=1))

        output = self.cls_seg(output)

        return output


class RowAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(RowAttention, self).__init__()
        # 初始化 Query, Key, Value 卷积操作
        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // reduction_ratio, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // reduction_ratio, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # 可学习的缩放参数

    def forward(self, x):
        B, C, H, W = x.size()

        # 1. 计算 Query, Key, Value 特征
        query = self.query_conv(x)  # (B, C//8, H, W)
        key = self.key_conv(x)  # (B, C//8, H, W)
        value = self.value_conv(x)  # (B, C, H, W)

        # 2. 调整 Query 和 Key 的形状，计算每一行的全局列关系
        query_row = query.permute(0, 2, 1, 3).contiguous().view(B * H, W, -1)  # (B*H, W, C//8)
        key_row = key.permute(0, 2, 3, 1).contiguous().view(B * H, -1, W)  # (B*H, C//8, W)

        # 3. 计算行全局注意力权重
        energy_row = torch.bmm(query_row, key_row)  # (B*H, W, W)
        attention_row = F.softmax(energy_row, dim=-1)  # 行注意力权重

        # 4. 加权求和 Value
        value_row = value.permute(0, 2, 3, 1).contiguous().view(B * H, W, -1)  # (B*H, W, C)
        out_row = torch.bmm(attention_row, value_row)  # (B*H, W, C)
        out_row = out_row.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        # 5. 将注意力特征与原输入特征相加
        out = self.gamma * out_row + x
        return out


class ColumnAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(ColumnAttention, self).__init__()
        # 初始化 Query, Key, Value 卷积操作
        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // reduction_ratio, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // reduction_ratio, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # 可学习的缩放参数

    def forward(self, x):
        B, C, H, W = x.size()

        # 1. 计算 Query, Key, Value 特征
        query = self.query_conv(x)  # (B, C//8, H, W)
        key = self.key_conv(x)  # (B, C//8, H, W)
        value = self.value_conv(x)  # (B, C, H, W)

        # 2. 调整 Query 和 Key 的形状，计算每一列的全局行关系
        query_col = query.permute(0, 3, 1, 2).contiguous().view(B * W, H, -1)  # (B*W, H, C//8)
        key_col = key.permute(0, 3, 2, 1).contiguous().view(B * W, -1, H)  # (B*W, C//8, H)

        # 3. 计算列全局注意力权重
        energy_col = torch.bmm(query_col, key_col)  # (B*W, H, H)
        attention_col = F.softmax(energy_col, dim=-1)  # 列注意力权重

        # 4. 加权求和 Value
        value_col = value.permute(0, 3, 2, 1).contiguous().view(B * W, H, -1)  # (B*W, H, C)
        out_col = torch.bmm(attention_col, value_col)  # (B*W, H, C)
        out_col = out_col.view(B, W, H, C).permute(0, 3, 2, 1)  # (B, C, H, W)

        # 5. 将注意力特征与原输入特征相加
        out = self.gamma * out_col + x
        return out


class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        # 行注意力和列注意力模块
        self.row_attention = RowAttention(in_channels=in_channels)
        self.column_attention = ColumnAttention(in_channels=in_channels)
        self.total_weight = nn.Parameter(torch.tensor(0.5))  # 融合权重

    def forward(self, x):
        # 行注意力
        out_row = self.row_attention(x)
        # 列注意力
        out_col = self.column_attention(out_row)
        # 融合行列注意力
        out = self.total_weight * out_col + x
        return out


class ChannelAttention(nn.Module):

    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.in_channels = in_channels

        # 初始化 gamma 参数
        self.gamma = nn.Parameter(torch.zeros(1))

        # 定义 Softmax 激活函数
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()

        # 1. 将 x 展平为 (B, C, H*W)
        proj_query = x.reshape(m_batchsize, C, -1)  # (B, C, H*W)
        proj_key = x.reshape(m_batchsize, C, -1).permute(0, 2, 1)  # (B, H*W, C)

        # 2. 计算相似度矩阵
        energy = torch.bmm(proj_query, proj_key)  # (B, C, C)

        # 3. 计算注意力权重
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy  # 对称注意力
        attention = self.softmax(energy_new)  # (B, C, C)

        # 4. 投影 Value
        proj_value = x.reshape(m_batchsize, C, -1)  # (B, C, H*W)
        out = torch.bmm(attention, proj_value)  # (B, C, H*W)

        # 5. 恢复为原始形状
        out = out.reshape(m_batchsize, C, height, width)

        # 6. 将注意力特征与输入特征相加
        out = self.gamma * out + x
        return out


class SE_block2(nn.Module):
    def __init__(self, inchannel, SEratio):
        super(SE_block2, self).__init__()
        self.liner_s = nn.Conv2d(inchannel * 4, inchannel * 2, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(inchannel * 2, inchannel, kernel_size=1)
        self.fuseblock = fuseblock2(inchannel)
        self.ca = ChannelAttention(inchannel * 2)

    def forward(self, t, rgb):
        f = self.fuseblock(t, rgb)
        fuse = torch.cat([t, rgb, f], dim=1)
        fuse = self.relu(self.liner_s(fuse))
        fuse = self.ca(fuse)
        fuse = self.relu(self.conv(fuse))
        fuse = fuse + rgb
        return fuse


class fuseblock2(nn.Module):
    def __init__(self, inchannel):
        super(fuseblock2, self).__init__()
        self.sa = SpatialAttention(inchannel)
        self.sig = nn.Sigmoid()

    def forward(self, rgb, t):
        fuse = torch.mul(rgb, t)
        # sa = self.sa(fuse)
        sa = fuse
        out = torch.mul(fuse, sa)
        out = self.sig(out)
        t = torch.mul(t, out)
        rgb = torch.mul(rgb, out)
        out = torch.cat([rgb, t], dim=1)
        return out


class BGALayer(nn.Module):
    # def __init__(self, inc, guide_inc=64):
    #     super(BGALayer, self).__init__()
    #     self.left1 = nn.Sequential(
    #         nn.Conv2d(
    #             inc, inc//4, kernel_size=3, stride=1,
    #             padding=1, bias=False),
    #         nn.BatchNorm2d(inc//4),
    #         nn.Conv2d(
    #             inc//4, inc // 4, kernel_size=1, stride=1,
    #             padding=0, bias=False),
    #     )
    #     self.right1 = nn.Sequential(
    #         nn.Conv2d(
    #             guide_inc, inc//4, kernel_size=3, stride=1,
    #             padding=1, bias=False),
    #         nn.BatchNorm2d(inc//4),
    #     )
    #     self.conv = nn.Sequential(
    #         nn.Conv2d(
    #             inc // 4, inc // 2, kernel_size=3, stride=1,
    #             padding=1, bias=False),
    #         nn.BatchNorm2d(inc//2),
    #         nn.ReLU(inplace=True),  # not shown in paper
    #         nn.Conv2d(
    #             inc // 2, inc, kernel_size=3, stride=1,
    #             padding=1, bias=False),
    #         nn.BatchNorm2d(inc),
    #         nn.ReLU(inplace=True),  # not shown in paper
    #     ) 存于2/base_spa_guide
    # def __init__(self, inc, guide_inc=64):
    #     super(BGALayer, self).__init__()
    #     self.left1 = nn.Sequential(
    #         nn.Conv2d(
    #             inc, inc//4, kernel_size=3, stride=1,
    #             padding=1, bias=False),
    #         nn.BatchNorm2d(inc//4),
    #         nn.Conv2d(
    #             inc//4, inc // 4, kernel_size=1, stride=1,
    #             padding=0, bias=False),
    #     )
    #     self.right1 = nn.Sequential(
    #         nn.Conv2d(
    #             guide_inc, inc//4, kernel_size=3, stride=1,
    #             padding=1, bias=False),
    #         nn.BatchNorm2d(inc//4),
    #         nn.Conv2d(
    #             inc//4, inc // 4, kernel_size=1, stride=1,
    #             padding=0, bias=False),
    #     )
    #     self.conv = nn.Sequential(
    #         nn.Conv2d(
    #             inc // 4, inc, kernel_size=3, stride=1,
    #             padding=1, bias=False),
    #         nn.BatchNorm2d(inc//2),
    #         nn.ReLU(inplace=True),  # not shown in paper
    #         nn.Conv2d(
    #             inc, inc, kernel_size=1, stride=1,
    #             padding=0, bias=False),
    #         nn.BatchNorm2d(inc),
    #         nn.ReLU(inplace=True),  # not shown in paper
    #     ) 在上述改的还没跑
    def __init__(self, inc, guide_inc=64):
        super(BGALayer, self).__init__()
        self.left1 = nn.Sequential(
            nn.Conv2d(
                inc, inc, kernel_size=3, stride=1,
                padding=1, groups=inc, bias=False),
            nn.BatchNorm2d(inc),
            nn.Conv2d(
                inc, guide_inc, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        self.right1 = nn.Sequential(
            nn.Conv2d(
                guide_inc, guide_inc, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(guide_inc),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(
                guide_inc, inc, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(inc),
            nn.ReLU(inplace=True),  # not shown in paper
        ) #原版本的，也没跑


    def forward(self, x_d, x_s):
        # x_d:rgb_before 
        dsize = x_d.size()[2:]

        left1 = self.left1(x_d)
        right1 = self.right1(x_s)

        right1 = F.interpolate(
            right1, size=dsize, mode='bilinear', align_corners=True)
        left = left1 * torch.sigmoid(right1)
        right = right1 * torch.sigmoid(left1)
        right = F.interpolate(
            right, size=dsize, mode='bilinear', align_corners=True)
        out = self.conv(left + right)
        #         明天试试加个残差
        return out

class Guide(nn.Module):
    def __init__(self, rgb_inc, rgb_inc_before, dop_inc, reduction_ratio=4):
        super(Guide, self).__init__()

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 行方向
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 列方向

        mip = max(8, rgb_inc_before // reduction_ratio)

        self.conv1 = nn.Conv2d(rgb_inc_before, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()

        self.conv_h = nn.Conv2d(mip, rgb_inc, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, rgb_inc, kernel_size=1, stride=1, padding=0)

        # self.self_SA_Enhance = SA_Enhance()

        self.BGA = BGALayer(rgb_inc_before)

        self.weight_h = nn.Parameter(torch.ones(1, rgb_inc, 1, 1))  # 形状适配通道维度
        self.weight_w = nn.Parameter(torch.ones(1, rgb_inc, 1, 1))

    def forward(self, rgb, rgb_before, dop):
        # print(rgb.shape,rgb_before.shape)
        depth = self.BGA(rgb_before, dop)
        depth = F.interpolate(depth, size=rgb.shape[-2:], mode='bilinear', align_corners=True)
        # print(depth.shape,rgb.shape)
        # 1. 对深度特征进行行和列方向的池化（深度特征用于约束）
        depth_h = self.pool_h(depth)  # (n, c, h, 1)
        depth_w = self.pool_w(depth)  # (n, c, 1, w)
        # print(depth_w.shape,'w')
        # 2. 行方向约束
        depth_h = self.conv1(depth_h)  # (n, mip, h, 1)
        depth_h = self.bn1(depth_h)
        depth_h = self.act(depth_h)  # 激活
        a_h = self.conv_h(depth_h).sigmoid()  # (n, oup, h, 1)
        # 120 1

        # 3. 列方向约束
        depth_w = self.conv1(depth_w)  # (n, mip, 1, w)
        depth_w = self.bn1(depth_w)
        depth_w = self.act(depth_w)  # 激活
        a_w = self.conv_w(depth_w).sigmoid()  # (n, oup, 1, w)
        # print(a_h.shape,a_w.shape)

        # 4. 用深度特征的行和列注意力分别约束 RGB 特征
        # out = (rgb * a_h) * self.weight_h + (rgb * a_w) * self.weight_w + rgb
        # out = rgb * a_h * a_w + rgb #base_guide
        out = rgb * a_h + rgb * a_w + rgb #base_guide_1

        # 5. 空间增强
        # out_sa = self.self_SA_Enhance(out_ca)

        return out


class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        self.conv2048 = ConvModule(2048, 1024, 3, 1, 1)
        self.conv1024 = ConvModule(1024, 512, 3, 1, 1)
        self.conv512 = ConvModule(512, 256, 3, 1, 1)
        self.conv256 = ConvModule(256, 128, 3, 1, 1)
        self.conv128 = ConvModule(128, 64, 3, 1, 1)
        self.conv64 = ConvModule(64, 64, 3, 1, 1)
        self.conv2 = ConvModule(64, 2, 3, 1, 1)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, l5, l4, l3, l2, l1):
        l5 = self.up2(self.conv512(l5))
        l4 = self.up2(self.conv256(l5 + l4))
        l3 = self.up2(self.conv128(l4 + l3))
        l2 = self.up2(self.conv64(l3 + l2))
        l = l2 + l1
        out = self.up2(self.conv2(l2 + l1))
        return out, l


class SGFNet(nn.Module):
    def __init__(self, n_classes=1):
        super(SGFNet, self).__init__()
        resnet_raw_model1 = models.resnet18(pretrained=True)
        resnet_raw_model2 = models.resnet18(pretrained=True)
        ########  Thermal ENCODER  ########

        self.encoder_thermal_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder_thermal_conv1.weight.data = torch.unsqueeze(torch.mean(resnet_raw_model1.conv1.weight.data, dim=1),
                                                                 dim=1)
        self.encoder_thermal_bn1 = resnet_raw_model1.bn1
        self.encoder_thermal_relu = resnet_raw_model1.relu
        self.encoder_thermal_maxpool = resnet_raw_model1.maxpool
        self.encoder_thermal_layer1 = resnet_raw_model1.layer1
        self.encoder_thermal_layer2 = resnet_raw_model1.layer2
        self.encoder_thermal_layer3 = resnet_raw_model1.layer3
        self.encoder_thermal_layer4 = resnet_raw_model1.layer4

        ########  RGB ENCODER  ########

        self.encoder_rgb_conv1 = resnet_raw_model2.conv1
        self.encoder_rgb_bn1 = resnet_raw_model2.bn1
        self.encoder_rgb_relu = resnet_raw_model2.relu
        self.encoder_rgb_maxpool = resnet_raw_model2.maxpool
        self.encoder_rgb_layer1 = resnet_raw_model2.layer1
        self.encoder_rgb_layer2 = resnet_raw_model2.layer2
        self.encoder_rgb_layer3 = resnet_raw_model2.layer3
        self.encoder_rgb_layer4 = resnet_raw_model2.layer4

        del resnet_raw_model1, resnet_raw_model2

        self.pol_guide1 = Guide(64, 64, 64)  # 输入：RGB和引导 输入：和RGB相同
        self.pol_guide2 = Guide(64, 64, 96)
        self.pol_guide3 = Guide(128, 64, 64)
        self.pol_guide4 = Guide(256, 128, 64)
        self.pol_guide5 = Guide(512, 256, 64)

        self.SE_block1 = SE_block2(64, 16)
        self.SE_block2 = SE_block2(64, 16)
        self.SE_block3 = SE_block2(128, 16)
        self.SE_block4 = SE_block2(256, 16)
        self.SE_block5 = SE_block2(512, 16)

        ## Edge-aware Lawin ASPP ##
        self.lowconv = nn.Conv2d(64 + 64, 64, 1)
        # self.compressconv1 = nn.Conv2d(128, 96, 1)
        # self.compressconv2 = nn.Conv2d(256, 192, 1)
        # self.compressconv3 = nn.Conv2d(512, 384, 1)
        self.lawin = LawinHead(num_classes=n_classes)
        self.edgeout = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.t_semout = FPN()
        self.binaryout = nn.Conv2d(2, 1, kernel_size=1)

    def forward(self, rgbinput, tin):
        rgb = rgbinput

        thermal = tin[:, :1, ...]

        # encoder
        rgb = self.encoder_rgb_conv1(rgb)
        rgb = self.encoder_rgb_bn1(rgb)
        rgb = self.encoder_rgb_relu(rgb)
        thermal = self.encoder_thermal_conv1(thermal)
        thermal = self.encoder_thermal_bn1(thermal)
        thermal = self.encoder_thermal_relu(thermal)
        t1 = thermal

        thermal = self.encoder_thermal_maxpool(thermal)
        thermal = self.encoder_thermal_layer1(thermal)
        t2 = thermal

        thermal = self.encoder_thermal_layer2(thermal)
        t3 = thermal

        thermal = self.encoder_thermal_layer3(thermal)
        t4 = thermal

        thermal = self.encoder_thermal_layer4(thermal)
        t5 = thermal
        t_sem, content_guide = self.t_semout(t5, t4, t3, t2, t1)

        binarymap = self.binaryout(t_sem)

        rgb = self.SE_block1(t1, rgb)  # C = 64 t1:B 64 320 240
        # rgb = self.pol_guide1(rgb, rgb, content_guide)
        rgb1 = rgb
        rgb = self.encoder_rgb_maxpool(rgb)

        ######################################################################
        rgb = self.encoder_rgb_layer1(rgb)
        rgb = self.SE_block2(t2, rgb)  # C = 64 t2:B 64 160 120
        rgb = self.pol_guide2(rgb, rgb1, content_guide)
        rgb2 = rgb

        ######################################################################
        rgb = self.encoder_rgb_layer2(rgb)
        # print(t3.shape,'t3.shape')
        rgb = self.SE_block3(t3, rgb)  # C = 128 t3:B 128 80 60
        rgb = self.pol_guide3(rgb, rgb2, content_guide)
        rgb3 = rgb

        ######################################################################
        rgb = self.encoder_rgb_layer3(rgb)
        rgb = self.SE_block4(t4, rgb)  # C = 256 t4:B 256 40 30
        rgb = self.pol_guide4(rgb, rgb3, content_guide)

        rgb4 = rgb
        ######################################################################
        fuse = self.encoder_rgb_layer4(rgb)
        fuse = self.SE_block5(t5, fuse)   # C = 512 t3:B 256 20 15
        fuse = self.pol_guide5(fuse, rgb4, content_guide)

        ######################################################################
        rgb1 = F.interpolate(rgb1, rgb2.size()[2:], mode='bilinear', align_corners=False)
        low = torch.cat([rgb1, rgb2], dim=1)
        low = self.lowconv(low)
        edgeout = self.edgeout(low)
        # rgb3 = self.compressconv1(rgb3)
        # rgb4 = self.compressconv2(rgb4)
        # fuse = self.compressconv3(fuse)
        input = [low, rgb3, rgb4, fuse]  # 这里在初始化定义过了 输入通道 64 128 320 512
        segout = self.lawin(input)
        segout = F.interpolate(segout, scale_factor=4, mode='bilinear', align_corners=False)
        edgeout = F.interpolate(edgeout, scale_factor=4, mode='bilinear', align_corners=False)

        t_sem = F.interpolate(t_sem, rgbinput.size()[2:], mode='bilinear', align_corners=False)

        return segout, edgeout, t_sem
        # return segout


if __name__ == '__main__':
    i = torch.randn(2, 3, 480, 640)
    i2 = torch.randn(2, 1, 480, 640)
    t = torch.randn(2, 64, 50, 50)
    rgb = torch.randn(2, 64, 50, 50)

    ca = SGFNet(9)
    out = ca(i, i2)
    for o in out:
        print(o.shape)
