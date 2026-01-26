import torch
import math
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from archs.transformer_util import (single_head_full_attention, single_head_split_window_attention,
                        single_head_full_attention_1d, single_head_split_window_attention_1d,
                        generate_shift_window_attn_mask, generate_shift_window_attn_mask_1d)


def coords_grid(b, h, w, homogeneous=False, device=None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]

    stacks = [x, y]

    if homogeneous:
        ones = torch.ones_like(x)  # [H, W]
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]

    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    if device is not None:
        grid = grid.to(device)

    return grid

def generate_window_grid(h_min, h_max, w_min, w_max, len_h, len_w, device=None):
    assert device is not None

    x, y = torch.meshgrid([torch.linspace(w_min, w_max, len_w, device=device),
                           torch.linspace(h_min, h_max, len_h, device=device)],
                          )
    grid = torch.stack((x, y), -1).transpose(0, 1).float()  # [H, W, 2]

    return grid

def normalize_coords(coords, h, w):
    # coords: [B, H, W, 2]
    c = torch.Tensor([(w - 1) / 2., (h - 1) / 2.]).float().to(coords.device)
    return (coords - c) / c  # [-1, 1]

def upsample_corres_with_mask(corres, up_mask, upsample_factor,
                            multiple_value=True):
    # convex upsampling following raft
    mask = up_mask
    b, flow_channel, h, w = corres.shape
    mask = mask.view(b, 1, 9, upsample_factor, upsample_factor, h, w)  # [B, 1, 9, K, K, H, W]
    mask = torch.softmax(mask, dim=2)

    multiplier = upsample_factor if multiple_value else 1
    up_corres = F.unfold(multiplier * corres, [3, 3], padding=1)
    up_corres = up_corres.view(b, flow_channel, 9, 1, 1, h, w)  # [B, 2, 9, 1, 1, H, W]

    up_corres = torch.sum(mask * up_corres, dim=2)  # [B, 2, K, K, H, W]
    up_corres = up_corres.permute(0, 1, 4, 2, 5, 3)  # [B, 2, K, H, K, W]
    up_corres = up_corres.reshape(b, flow_channel, upsample_factor * h,
                              upsample_factor * w)  # [B, 2, K*H, K*W]

    return up_corres

def global_correlation_softmax(feature0, feature1):
    b, c, h, w = feature0.shape
    feature0 = feature0.view(b, c, -1).permute(0, 2, 1)  # [B, H*W, C]
    feature1 = feature1.view(b, c, -1)  # [B, C, H*W]
    

    
    correlation = torch.matmul(feature0, feature1).view(b, h, w, h, w) / (c ** 0.5)  # [B, H, W, H, W]

    # flow from softmax
    init_grid = coords_grid(b, h, w).to(correlation.device)  # [B, 2, H, W]
    grid = init_grid.view(b, 2, -1).permute(0, 2, 1)  # [B, H*W, 2]

    correlation = correlation.view(b, h * w, h * w)  # [B, H*W, H*W]

    prob = F.softmax(correlation, dim=-1)  # [B, H*W, H*W]

    correspondence = torch.matmul(prob, grid).view(b, h, w, 2).permute(0, 3, 1, 2)  # [B, 2, H, W]

    return correspondence, prob

def local_correlation_with_flow(feature0, feature1,
                                flow,
                                local_radius,
                                padding_mode='zeros',
                                dilation=1,
                                ):
    b, c, h, w = feature0.size()
    coords_init = coords_grid(b, h, w).to(feature0.device)  # [B, 2, H, W]
    coords = coords_init.view(b, 2, -1).permute(0, 2, 1)  # [B, H*W, 2]

    local_h = 2 * local_radius + 1
    local_w = 2 * local_radius + 1

    window_grid = generate_window_grid(-local_radius, local_radius,
                                       -local_radius, local_radius,
                                       local_h, local_w, device=feature0.device)  # [2R+1, 2R+1, 2]
    window_grid = window_grid.reshape(-1, 2).repeat(b, 1, 1, 1)  # [B, 1, (2R+1)^2, 2]
    sample_coords = coords.unsqueeze(-2) + window_grid * dilation  # [B, H*W, (2R+1)^2, 2]

    # flow can be zero when using features after transformer
    if not isinstance(flow, float):
        sample_coords = sample_coords + flow.view(
            b, 2, -1).permute(0, 2, 1).unsqueeze(-2)  # [B, H*W, (2R+1)^2, 2]
    else:
        assert flow == 0.

    # normalize coordinates to [-1, 1]
    sample_coords_norm = normalize_coords(sample_coords, h, w)  # [-1, 1]
    window_feature = F.grid_sample(feature1, sample_coords_norm,
                                   padding_mode=padding_mode, align_corners=True
                                   ).permute(0, 2, 1, 3)  # [B, H*W, C, (2R+1)^2]
    feature0_view = feature0.permute(0, 2, 3, 1).view(b, h * w, 1, c)  # [B, H*W, 1, C]

    correlation = torch.matmul(feature0_view, window_feature).view(b, h * w, -1) / (c ** 0.5)  # [B, H*W, (2R+1)^2]

    correlation = correlation.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()  # [B, (2R+1)^2, H, W]

    return correlation


def local_correlation_with_corres(feature0, feature1,
                                corres,
                                local_radius,
                                padding_mode='zeros',
                                dilation=1,
                                ):
    # corres: [B, 2, H, W]
    # transform correspondence to flow
    b, c, h, w = feature0.size()
    init_grid = coords_grid(b, h, w).to(corres.device)  # [B, 2, H, W]
    flow = corres - init_grid
    correlation = local_correlation_with_flow(feature0, feature1,flow,local_radius,padding_mode,dilation)

    return correlation


class MultiScaleTridentConv(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            strides=1,
            paddings=0,
            dilations=1,
            dilation=1,
            groups=1,
            num_branch=1,
            test_branch_idx=-1,
            bias=False,
            norm=None,
            activation=None,
    ):
        super(MultiScaleTridentConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.num_branch = num_branch
        self.stride = _pair(stride)
        self.groups = groups
        self.with_bias = bias
        self.dilation = dilation
        if isinstance(paddings, int):
            paddings = [paddings] * self.num_branch
        if isinstance(dilations, int):
            dilations = [dilations] * self.num_branch
        if isinstance(strides, int):
            strides = [strides] * self.num_branch
        self.paddings = [_pair(padding) for padding in paddings]
        self.dilations = [_pair(dilation) for dilation in dilations]
        self.strides = [_pair(stride) for stride in strides]
        self.test_branch_idx = test_branch_idx
        self.norm = norm
        self.activation = activation

        assert len({self.num_branch, len(self.paddings), len(self.strides)}) == 1

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None

        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, inputs):
        num_branch = self.num_branch if self.training or self.test_branch_idx == -1 else 1
        assert len(inputs) == num_branch

        if self.training or self.test_branch_idx == -1:
            outputs = [
                F.conv2d(input, self.weight, self.bias, stride, padding, self.dilation, self.groups)
                for input, stride, padding in zip(inputs, self.strides, self.paddings)
            ]
        else:
            outputs = [
                F.conv2d(
                    inputs[0],
                    self.weight,
                    self.bias,
                    self.strides[self.test_branch_idx] if self.test_branch_idx == -1 else self.strides[-1],
                    self.paddings[self.test_branch_idx] if self.test_branch_idx == -1 else self.paddings[-1],
                    self.dilation,
                    self.groups,
                )
            ]

        if self.norm is not None:
            outputs = [self.norm(x) for x in outputs]
        if self.activation is not None:
            outputs = [self.activation(x) for x in outputs]
        return outputs
    
class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_layer=nn.InstanceNorm2d, stride=1, dilation=1,
                 ):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               dilation=dilation, padding=dilation, stride=stride, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               dilation=dilation, padding=dilation, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.norm1 = norm_layer(planes)
        self.norm2 = norm_layer(planes)
        if not stride == 1 or in_planes != planes:
            self.norm3 = norm_layer(planes)

        if stride == 1 and in_planes == planes:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class CNNEncoder(nn.Module):
    def __init__(self, 
                 input_dim=3,
                 output_dim=128,
                 norm_layer=nn.InstanceNorm2d,
                 num_output_scales=1,
                 **kwargs,
                 ):
        super(CNNEncoder, self).__init__()
        self.num_branch = num_output_scales

        feature_dims = [64, 96, 128]

        self.conv1 = nn.Conv2d(input_dim, feature_dims[0], kernel_size=7, stride=2, padding=3, bias=False)  # 1/2
        self.norm1 = norm_layer(feature_dims[0])
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = feature_dims[0]
        self.layer1 = self._make_layer(feature_dims[0], stride=1, norm_layer=norm_layer)  # 1/2
        self.layer2 = self._make_layer(feature_dims[1], stride=2, norm_layer=norm_layer)  # 1/4

        # highest resolution 1/4 or 1/8
        stride = 2 if num_output_scales == 1 else 1
        self.layer3 = self._make_layer(feature_dims[2], stride=stride,
                                       norm_layer=norm_layer,
                                       )  # 1/4 or 1/8

        self.conv2 = nn.Conv2d(feature_dims[2], output_dim, 1, 1, 0)

        if self.num_branch > 1:
            if self.num_branch == 4:
                strides = (1, 2, 4, 8)
            elif self.num_branch == 3:
                strides = (1, 2, 4)
            elif self.num_branch == 2:
                strides = (1, 2)
            else:
                raise ValueError

            self.trident_conv = MultiScaleTridentConv(output_dim, output_dim,
                                                      kernel_size=3,
                                                      strides=strides,
                                                      paddings=1,
                                                      num_branch=self.num_branch,
                                                      )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1, dilation=1, norm_layer=nn.InstanceNorm2d):
        layer1 = ResidualBlock(self.in_planes, dim, norm_layer=norm_layer, stride=stride, dilation=dilation)
        layer2 = ResidualBlock(dim, dim, norm_layer=norm_layer, stride=1, dilation=dilation)

        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)  # 1/2
        x = self.layer2(x)  # 1/4
        x = self.layer3(x)  # 1/8 or 1/4

        x = self.conv2(x)

        if self.num_branch > 1:
            out = self.trident_conv([x] * self.num_branch)  # high to low res
        else:
            out = [x]

        return out


class TransformerLayer(nn.Module):
    def __init__(self,
                 d_model=128,
                 nhead=1,
                 no_ffn=False,
                 ffn_dim_expansion=4,
                 ):
        super(TransformerLayer, self).__init__()

        self.dim = d_model
        self.nhead = nhead
        self.no_ffn = no_ffn

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.merge = nn.Linear(d_model, d_model, bias=False)

        self.norm1 = nn.LayerNorm(d_model)

        # no ffn after self-attn, with ffn after cross-attn
        if not self.no_ffn:
            in_channels = d_model * 2
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, in_channels * ffn_dim_expansion, bias=False),
                nn.GELU(),
                nn.Linear(in_channels * ffn_dim_expansion, d_model, bias=False),
            )

            self.norm2 = nn.LayerNorm(d_model)

    def forward(self, source, target,
                height=None,
                width=None,
                shifted_window_attn_mask=None,
                shifted_window_attn_mask_1d=None,
                attn_type='swin',
                with_shift=False,
                attn_num_splits=None,
                ):
        # source, target: [B, L, C]
        query, key, value = source, target, target

        # for stereo: 2d attn in self-attn, 1d attn in cross-attn
        is_self_attn = (query - key).abs().max() < 1e-6

        # single-head attention
        query = self.q_proj(query)  # [B, L, C]
        key = self.k_proj(key)  # [B, L, C]
        value = self.v_proj(value)  # [B, L, C]

        if attn_type == 'swin' and attn_num_splits > 1:  # self, cross-attn: both swin 2d
            if self.nhead > 1:
                # we observe that multihead attention slows down the speed and increases the memory consumption
                # without bringing obvious performance gains and thus the implementation is removed
                raise NotImplementedError
            else:
                message = single_head_split_window_attention(query, key, value,
                                                             num_splits=attn_num_splits,
                                                             with_shift=with_shift,
                                                             h=height,
                                                             w=width,
                                                             attn_mask=shifted_window_attn_mask,
                                                             )

        elif attn_type == 'self_swin2d_cross_1d':  # self-attn: swin 2d, cross-attn: full 1d
            if self.nhead > 1:
                raise NotImplementedError
            else:
                if is_self_attn:
                    if attn_num_splits > 1:
                        message = single_head_split_window_attention(query, key, value,
                                                                     num_splits=attn_num_splits,
                                                                     with_shift=with_shift,
                                                                     h=height,
                                                                     w=width,
                                                                     attn_mask=shifted_window_attn_mask,
                                                                     )
                    else:
                        # full 2d attn
                        message = single_head_full_attention(query, key, value)  # [N, L, C]

                else:
                    # cross attn 1d
                    message = single_head_full_attention_1d(query, key, value,
                                                            h=height,
                                                            w=width,
                                                            )

        elif attn_type == 'self_swin2d_cross_swin1d':  # self-attn: swin 2d, cross-attn: swin 1d
            if self.nhead > 1:
                raise NotImplementedError
            else:
                if is_self_attn:
                    if attn_num_splits > 1:
                        # self attn shift window
                        message = single_head_split_window_attention(query, key, value,
                                                                     num_splits=attn_num_splits,
                                                                     with_shift=with_shift,
                                                                     h=height,
                                                                     w=width,
                                                                     attn_mask=shifted_window_attn_mask,
                                                                     )
                    else:
                        # full 2d attn
                        message = single_head_full_attention(query, key, value)  # [N, L, C]
                else:
                    if attn_num_splits > 1:
                        assert shifted_window_attn_mask_1d is not None
                        # cross attn 1d shift
                        message = single_head_split_window_attention_1d(query, key, value,
                                                                        num_splits=attn_num_splits,
                                                                        with_shift=with_shift,
                                                                        h=height,
                                                                        w=width,
                                                                        attn_mask=shifted_window_attn_mask_1d,
                                                                        )
                    else:
                        message = single_head_full_attention_1d(query, key, value,
                                                                h=height,
                                                                w=width,
                                                                )

        else:
            message = single_head_full_attention(query, key, value)  # [B, L, C]

        message = self.merge(message)  # [B, L, C]
        message = self.norm1(message)

        if not self.no_ffn:
            message = self.mlp(torch.cat([source, message], dim=-1))
            message = self.norm2(message)

        return source + message


class TransformerBlock(nn.Module):
    """self attention + cross attention + FFN"""

    def __init__(self,
                 d_model=128,
                 nhead=1,
                 ffn_dim_expansion=4,
                 ):
        super(TransformerBlock, self).__init__()

        self.self_attn = TransformerLayer(d_model=d_model,
                                          nhead=nhead,
                                          no_ffn=True,
                                          ffn_dim_expansion=ffn_dim_expansion,
                                          )

        self.cross_attn_ffn = TransformerLayer(d_model=d_model,
                                               nhead=nhead,
                                               ffn_dim_expansion=ffn_dim_expansion,
                                               )

    def forward(self, source, target,
                height=None,
                width=None,
                shifted_window_attn_mask=None,
                shifted_window_attn_mask_1d=None,
                attn_type='swin',
                with_shift=False,
                attn_num_splits=None,
                ):
        # source, target: [B, L, C]

        # self attention
        source = self.self_attn(source, source,
                                height=height,
                                width=width,
                                shifted_window_attn_mask=shifted_window_attn_mask,
                                attn_type=attn_type,
                                with_shift=with_shift,
                                attn_num_splits=attn_num_splits,
                                )

        # cross attention and ffn
        source = self.cross_attn_ffn(source, target,
                                     height=height,
                                     width=width,
                                     shifted_window_attn_mask=shifted_window_attn_mask,
                                     shifted_window_attn_mask_1d=shifted_window_attn_mask_1d,
                                     attn_type=attn_type,
                                     with_shift=with_shift,
                                     attn_num_splits=attn_num_splits,
                                     )

        return source


class FeatureTransformer(nn.Module):
    def __init__(self,
                 num_layers=6,
                 d_model=128,
                 nhead=1,
                 ffn_dim_expansion=4,
                 ):
        super(FeatureTransformer, self).__init__()

        self.d_model = d_model
        self.nhead = nhead

        self.layers = nn.ModuleList([
            TransformerBlock(d_model=d_model,
                             nhead=nhead,
                             ffn_dim_expansion=ffn_dim_expansion,
                             )
            for i in range(num_layers)])

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feature0, feature1,
                attn_type='swin',
                attn_num_splits=None,
                **kwargs,
                ):

        b, c, h, w = feature0.shape
        assert self.d_model == c

        feature0 = feature0.flatten(-2).permute(0, 2, 1)  # [B, H*W, C]
        feature1 = feature1.flatten(-2).permute(0, 2, 1)  # [B, H*W, C]

        # 2d attention
        if 'swin' in attn_type and attn_num_splits > 1:
            # global and refine use different number of splits
            window_size_h = h // attn_num_splits
            window_size_w = w // attn_num_splits

            # compute attn mask once
            shifted_window_attn_mask = generate_shift_window_attn_mask(
                input_resolution=(h, w),
                window_size_h=window_size_h,
                window_size_w=window_size_w,
                shift_size_h=window_size_h // 2,
                shift_size_w=window_size_w // 2,
                device=feature0.device,
            )  # [K*K, H/K*W/K, H/K*W/K]
        else:
            shifted_window_attn_mask = None

        # 1d attention
        if 'swin1d' in attn_type and attn_num_splits > 1:
            window_size_w = w // attn_num_splits

            # compute attn mask once
            shifted_window_attn_mask_1d = generate_shift_window_attn_mask_1d(
                input_w=w,
                window_size_w=window_size_w,
                shift_size_w=window_size_w // 2,
                device=feature0.device,
            )  # [K, W/K, W/K]
        else:
            shifted_window_attn_mask_1d = None

        # concat feature0 and feature1 in batch dimension to compute in parallel
        concat0 = torch.cat((feature0, feature1), dim=0)  # [2B, H*W, C]
        concat1 = torch.cat((feature1, feature0), dim=0)  # [2B, H*W, C]

        for i, layer in enumerate(self.layers):
            concat0 = layer(concat0, concat1,
                            height=h,
                            width=w,
                            attn_type=attn_type,
                            with_shift='swin' in attn_type and attn_num_splits > 1 and i % 2 == 1,
                            attn_num_splits=attn_num_splits,
                            shifted_window_attn_mask=shifted_window_attn_mask,
                            shifted_window_attn_mask_1d=shifted_window_attn_mask_1d,
                            )

            # update feature1
            concat1 = torch.cat(concat0.chunk(chunks=2, dim=0)[::-1], dim=0)

        feature0, feature1 = concat0.chunk(chunks=2, dim=0)  # [B, H*W, C]

        # reshape back
        feature0 = feature0.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        feature1 = feature1.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

        return feature0, feature1



class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256,
                 out_dim=2,
                 ):
        super(FlowHead, self).__init__()

        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, out_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv2(self.relu(self.conv1(x)))

        return out


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128,
                 kernel_size=5,
                 ):
        padding = (kernel_size - 1) // 2

        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, kernel_size), padding=(0, padding))
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, kernel_size), padding=(0, padding))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, kernel_size), padding=(0, padding))

        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (kernel_size, 1), padding=(padding, 0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (kernel_size, 1), padding=(padding, 0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (kernel_size, 1), padding=(padding, 0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


class BasicMotionEncoder(nn.Module):
    def __init__(self, corr_channels=324,
                 flow_channels=2,
                 ):
        super(BasicMotionEncoder, self).__init__()

        self.convc1 = nn.Conv2d(corr_channels, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(flow_channels, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 192, 128 - flow_channels, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class BasicUpdateBlock(nn.Module):
    def __init__(self, corr_channels=324,
                 hidden_dim=128,
                 context_dim=128,
                 downsample_factor=8,
                 flow_dim=2,
                 bilinear_up=False,
                 ):
        super(BasicUpdateBlock, self).__init__()

        self.encoder = BasicMotionEncoder(corr_channels=corr_channels,
                                          flow_channels=flow_dim,
                                          )

        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=context_dim + hidden_dim)

        self.flow_head = FlowHead(hidden_dim, hidden_dim=256,
                                  out_dim=flow_dim,
                                  )

        if bilinear_up:
            self.mask = None
        else:
            self.mask = nn.Sequential(
                nn.Conv2d(hidden_dim, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, downsample_factor ** 2 * 9, 1, padding=0))

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)

        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        if self.mask is not None:
            mask = self.mask(net)
        else:
            mask = None

        return net, mask, delta_flow
