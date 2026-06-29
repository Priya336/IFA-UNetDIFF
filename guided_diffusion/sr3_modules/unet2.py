import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding

# Feature-wise affine modulation
class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super().__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))

class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

# Basic convolutional block
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)

# Resnet Block with noise embedding
class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)

# Memory-efficient cross-attention for decoder
class CrossAttention(nn.Module):
    def __init__(self, dim, n_head=1):
        super().__init__()
        self.n_head = n_head
        self.dim = dim
        self.head_dim = dim // n_head

        self.to_q = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_k = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_v = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x, context):
        b, c, h, w = x.shape
        q = self.to_q(x).view(b, self.n_head, self.head_dim, h*w)
        k = self.to_k(context).view(b, self.n_head, self.head_dim, -1)
        v = self.to_v(context).view(b, self.n_head, self.head_dim, -1)

        attn = torch.einsum('bnhp,bnhq->bnpq', q, k) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn, dim=-1)

        out = torch.einsum('bnpq,bnhq->bnhp', attn, v)
        out = out.contiguous().view(b, c, h, w)
        return self.to_out(out) + x

# Resnet block with optional attention
class ResnetBlockWithAttn(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, norm_groups=32, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(dim, dim_out, noise_level_emb_dim, dropout=dropout, norm_groups=norm_groups)
        if with_attn:
            self.cross = CrossAttention(dim_out)

    def forward(self, x, time_emb=None, context=None):
        x = self.res_block(x, time_emb)
        if self.with_attn and context is not None:
            x = self.cross(x, context)
        return x

def Reverse(lst):
    return [ele for ele in reversed(lst)]

# Full UNet with cross-attention in decoder
class UNet(nn.Module):
    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=(8,),
        res_blocks=3,
        dropout=0,
        with_noise_level_emb=True,
        image_size=128
    ):
        super().__init__()

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel*4),
                Swish(),
                nn.Linear(inner_channel*4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size

        self.init_conv = nn.Conv2d(in_channel, inner_channel, 3, padding=1)

        downs = []
        for ind in range(num_mults):
            is_last = (ind == num_mults-1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(res_blocks):
                downs.append(ResnetBlockWithAttn(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups, dropout=dropout, with_attn=False))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res // 2
        self.downs = nn.ModuleList(downs)

        # Middle
        self.mid = nn.ModuleList([
            ResnetBlockWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups, dropout=dropout, with_attn=True),
            ResnetBlockWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups, dropout=dropout, with_attn=False)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(res_blocks+1):
                ups.append(ResnetBlockWithAttn(
                    pre_channel + feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res*2
        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

    def forward(self, x, time, feat_need=False):
        t = self.noise_level_mlp(time) if exists(self.noise_level_mlp) else None
        x = self.init_conv(x)

        feats = [x]
        for layer in self.downs:
            if isinstance(layer, ResnetBlockWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        if feat_need:
            fe = feats.copy()

        for layer in self.mid:
            if isinstance(layer, ResnetBlockWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)

        if feat_need:
            fd = []

        for layer in self.ups:
            if isinstance(layer, ResnetBlockWithAttn):
                context = feats[-1] if layer.with_attn else None
                x = layer(torch.cat((x, feats.pop()), dim=1), t, context=context)
                if feat_need:
                    fd.append(x)
            else:
                x = layer(x)

        x = self.final_conv(x)

        if feat_need:
            return fe, Reverse(fd)
        else:
            return x
