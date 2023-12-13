import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt
from einops import rearrange
from utils.hparams import hparams
# from .utils import up_or_down_sampling

Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def override(self, attrs):
        if isinstance(attrs, dict):
            self.__dict__.update(**attrs)
        elif isinstance(attrs, (list, tuple, set)):
            for attr in attrs:
                self.override(attr)
        elif attrs is not None:
            raise NotImplementedError
        return self

class BaseModule(torch.nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()

    @property
    def nparams(self):
        """
        Returns number of trainable parameters of the module.
        """
        num_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                num_params += np.prod(param.detach().cpu().numpy().shape)
        return num_params


    def relocate_input(self, x: list):
        """
        Relocates provided tensors to the same device set for the module.
        """
        device = next(self.parameters()).device
        for i in range(len(x)):
            if isinstance(x[i], torch.Tensor) and x[i].device != device:
                x[i] = x[i].to(device)
        return x

class Mish(BaseModule):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


class UpsampleConv(BaseModule):
    def __init__(self, dim):
        super(UpsampleConv, self).__init__()
        self.conv = torch.nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class DownsampleConv(BaseModule):
    # 后两维变为1/2
    def __init__(self, dim):
        super(DownsampleConv, self).__init__()
        self.conv = torch.nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample(BaseModule):
  def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False,
               fir_kernel=(1, 3, 3, 1)):
    super().__init__()
    out_ch = out_ch if out_ch else in_ch
    if not fir:
      if with_conv:
        self.Conv_0 = conv3x3(in_ch, out_ch)
    else:
      if with_conv:
        self.Conv2d_0 = up_or_down_sampling.Conv2d(in_ch, out_ch,
                                                 kernel=3, up=True,
                                                 resample_kernel=fir_kernel,
                                                 use_bias=True,
                                                 )
    self.fir = fir
    self.with_conv = with_conv
    self.fir_kernel = fir_kernel
    self.out_ch = out_ch

  def forward(self, x):
    B, C, H, W = x.shape
    if not self.fir:
      h = F.interpolate(x, (H * 2, W * 2), 'nearest')
      if self.with_conv:
        h = self.Conv_0(h)
    else:
      if not self.with_conv:
        h = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
      else:
        h = self.Conv2d_0(x)

    return h


class Downsample(BaseModule):
  def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False,
               fir_kernel=(1, 3, 3, 1)):
    super().__init__()
    out_ch = out_ch if out_ch else in_ch
    if not fir:
      if with_conv:
        self.Conv_0 = conv3x3(in_ch, out_ch, stride=2, padding=0)
    else:
      if with_conv:
        self.Conv2d_0 = up_or_down_sampling.Conv2d(in_ch, out_ch,
                                                 kernel=3, down=True,
                                                 resample_kernel=fir_kernel,
                                                 use_bias=True,
                                                 )
    self.fir = fir
    self.fir_kernel = fir_kernel
    self.with_conv = with_conv
    self.out_ch = out_ch

  def forward(self, x):
    B, C, H, W = x.shape
    if not self.fir:
      if self.with_conv:
        x = F.pad(x, (0, 1, 0, 1))
        x = self.Conv_0(x)
      else:
        x = F.avg_pool2d(x, 2, stride=2)
    else:
      if not self.with_conv:
        x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
      else:
        x = self.Conv2d_0(x)

    return x


class Rezero(BaseModule):
    def __init__(self, fn):
        super(Rezero, self).__init__()
        self.fn = fn
        self.g = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g


class Block(BaseModule):
    def __init__(self, dim, dim_out, groups=8):
        super(Block, self).__init__()
        # 后两维不变
        w_kernel_size = 3
        w_padding = int((w_kernel_size - 1) // 2)
        self.block = torch.nn.Sequential(torch.nn.Conv2d(dim, dim_out, kernel_size=(3, w_kernel_size), 
                                         padding=(1, w_padding)), torch.nn.GroupNorm(
                                         groups, dim_out), Mish())

    def forward(self, x, mask):
        # x: [B, C, M, T]
        output = self.block(x * mask)
        return output * mask


class ResnetBlock(BaseModule):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super(ResnetBlock, self).__init__()
        self.t_mlp = torch.nn.Sequential(Mish(), torch.nn.Linear(time_emb_dim, 
                                                               dim_out))
        self.c_mlp = torch.nn.Sequential(Mish(), torch.nn.Conv2d(1, dim_out, 1))

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.block3 = Block(dim_out, dim_out, groups=groups)
        if dim != dim_out:
            self.res_conv = torch.nn.Conv2d(dim, dim_out, 1)
        else:
            self.res_conv = torch.nn.Identity()

    def forward(self, x, mask, time_emb, cond):
        h = self.block1(x, mask)
        h = h + self.t_mlp(time_emb).unsqueeze(-1).unsqueeze(-1) # add time info
        h = self.block2(h, mask)
        h = h + self.c_mlp(cond) # add cond info
        h = self.block3(h, mask)
        output = h + self.res_conv(x * mask) * (1 / torch.sqrt(torch.tensor(2.)))
        return output

class UncondResnetBlock(BaseModule):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super(UncondResnetBlock, self).__init__()
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        if dim != dim_out:
            self.res_conv = torch.nn.Conv2d(dim, dim_out, 1)
        else:
            self.res_conv = torch.nn.Identity()

    def forward(self, x, mask):
        h = self.block1(x, mask)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask) * (1 / torch.sqrt(torch.tensor(2.)))
        return output


class LinearAttention(BaseModule):
    def __init__(self, dim, heads=4, dim_head=32):
        super(LinearAttention, self).__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = torch.nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = torch.nn.Conv2d(hidden_dim, dim, 1)            

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', 
                            heads = self.heads, qkv=3)            
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', 
                        heads=self.heads, h=h, w=w)
        return self.to_out(out)


class Residual(BaseModule):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        output = self.fn(x, *args, **kwargs) + x
        return output

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):
    def __init__(self, encoder_hidden, residual_channels, dilation):
        super().__init__()
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = Linear(residual_channels, residual_channels)
        self.conditioner_projection = Conv1d(encoder_hidden, 2 * residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, condition, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(condition)
        y = x + diffusion_step # [B, RC, T]

        y = self.dilated_conv(y) + conditioner # [B, 2RC, T]

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter) # [B, RC, T]

        y = self.output_projection(y) # [B, 2RC, T]
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip # [B, RC, T]

class AdaCondResBlock(nn.Module):
    def __init__(self, encoder_hidden, residual_channels, dilation):
        super().__init__()
        self.dilated_conv = nn.Sequential(
                                Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation),
                                )
        self.g_norm = nn.GroupNorm(32, residual_channels, affine=False)
        self.diffusion_projection = nn.Sequential(
                                    nn.SiLU(),
                                    Linear(residual_channels, residual_channels),
                                    )
        self.ada_conditioner = nn.Sequential(
                                nn.SiLU(),
                                Conv1d(encoder_hidden, 2 * residual_channels, 1),
                                )

        self.output_projection = nn.Sequential(
                                Conv1d(residual_channels, 2 * residual_channels, 1),
                                )

    def forward(self, x, condition, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        shift, scale = self.ada_conditioner(condition + diffusion_step).chunk(2, dim=1)

        y = self.g_norm(x) * (1 + scale) + shift
        y = self.dilated_conv(y) # [B, 2RC, T]

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter) # [B, RC, T]

        y = self.output_projection(y) # [B, 2RC, T]
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip # [B, RC, T]

class UncondResidualBlock(nn.Module):
    def __init__(self, encoder_hidden, residual_channels, dilation):
        super().__init__()
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x):
        y = x  # [B,RC,T]

        y = self.dilated_conv(y) # [B,2RC,T]

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter) # [B,RC,T]

        y = self.output_projection(y) # [B,2RC,T]
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip # [B,RC,T]

class GaussianFourierProjection(nn.Module):
  """Gaussian Fourier embeddings for noise levels."""

  def __init__(self, embedding_size=256, scale=1.0):
    super().__init__()
    self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class LayerNorm(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.gama = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor(0.), requires_grad=True)
    def forward(self, x, eps = 1e-5):
        numerator = x - x.mean([1,2])[:, None, None]
        denominator = x.std([1,2])[:, None, None] + eps
        return ( numerator / denominator) * self.gama + self.beta

class CondLayer(nn.Module):
    def __init__(self, encoder_hidden, residual_channels):
        super().__init__()
        self.conv = Conv1d(encoder_hidden, residual_channels, 1)
        self.factor = nn.Parameter(torch.tensor(0.), requires_grad=True)
    def forward(self, x, cond):
        return x + self.factor * self.conv(cond)

class WaveNet(BaseModule):
    def __init__(self, in_dims=80):
        super().__init__()
        self.params = params = AttrDict(
            # Model params
            encoder_hidden=hparams['hidden_size'],
            residual_layers=hparams['residual_layers'],
            residual_channels=hparams['residual_channels'],
            dilation_cycle_length=hparams['dilation_cycle_length'],
        )
        self.layernorm = LayerNorm()
        self.activate_fn = Mish()
        self.input_projection = Conv1d(in_dims, params.residual_channels, 1)
        self.diffusion_embedding = SinusoidalPosEmb(params.residual_channels)
        dim = params.residual_channels
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim)
        )
        self.cond_layers = nn.ModuleList([
            CondLayer(params.encoder_hidden, params.residual_channels)
            for i in range(params.residual_layers)
        ])
        self.residual_layers = nn.ModuleList([
            ResidualBlock(in_dims, params.residual_channels, 2 ** (i % params.dilation_cycle_length))
            for i in range(params.residual_layers)
        ])
        self.skip_projection = Conv1d(params.residual_channels, params.residual_channels, 1)
        self.output_projection = Conv1d(params.residual_channels, in_dims, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond, mask):
        """

        :param spec: [B, 1, M, T] ,M: 梅尔通道数， T: 帧数
        :param mask: [B, 1, T]
        :param diffusion_step: [B, 1]
        :param mu: [B, 1, M, T]
        :param cond: [B, H, T]
        :return:
        """
        mu, cond = cond
        assert spec.shape == mu.shape
        x = spec[:, 0] #[B, M, T]
        mu = mu[:, 0]
        x = self.input_projection(x)  # x [B, residual_channel, T]

        x = self.activate_fn(x)
        x = x * mask
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)
        skip = []
        for layer_id, layer in enumerate(self.residual_layers):
            x = self.cond_layers[layer_id](x, cond)
            x, skip_connection = layer(x, mu, diffusion_step)
            x = x * mask
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = self.activate_fn(x)
        x = self.output_projection(x)  # [B, M, T]
        x = x * mask
        return x[:, None, :, :] # [B,1,M,T]

class UncondWaveNet(BaseModule):
    def __init__(self, in_dims=80):
        super().__init__()
        self.params = params = AttrDict(
            # Model params
            encoder_hidden=hparams['hidden_size'],
            residual_layers=hparams['residual_layers'],
            residual_channels=hparams['residual_channels'],
            dilation_cycle_length=hparams['dilation_cycle_length'],
        )
        self.layernorm = LayerNorm()
        self.activate_fn = Mish()
        dim = params.residual_channels
        self.residual_layers = nn.ModuleList([
            UncondResidualBlock(in_dims, params.residual_channels, 2 ** (i % params.dilation_cycle_length))
            for i in range(params.residual_layers)
        ])
        self.skip_projection = Conv1d(params.residual_channels, params.residual_channels, 1)

    def forward(self, spec, mask):
        """
        :param spec: [B, 1, H, T] ,M: 梅尔通道数， T: 帧数
        :param mask: [B, 1, T]
        :return: [B, 1, H, T]
        """
        x = spec[:, 0] # [B, H, T]
        skip = []
        for layer_id, layer in enumerate(self.residual_layers):
            x, skip_connection = layer(x)
            x = x * mask
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = self.activate_fn(x)
        x = x * mask
        return x[:, None, :, :] # [B, 1, H, T]

class UNET(BaseModule):
    def __init__(self, channel, hidden_size, channel_mults=(1, 2, 4)):
        super(UNET, self).__init__()
        self.channel = channel
        self.channel_mults = channel_mults
        self.time_pos_emb = SinusoidalPosEmb(hidden_size)
        self.tmlp = torch.nn.Sequential(torch.nn.Linear(dim, dim * 2), Mish(),
                                       torch.nn.Linear(dim * 2, dim))
        self.cmlp = torch.nn.Sequential(torch.nn.Linear(hidden_size, hidden_size * 2), Mish(),
                                       torch.nn.Linear(hidden_size * 2, 80))


        channels = [2, *map(lambda m: channel * m, channel_mults)] # 通道数为2，因为cat(cond, x, dim=1)
        in_out = list(zip(channels[:-1], channels[1:])) # [(2, channel), (channel, 2*channel), (2*channel, 4*channel)]
        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        num_resolutions = len(in_out) # 3

        for ind, (channel_in, channel_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(torch.nn.ModuleList([
                       ResnetBlock(channel_in, channel_out, time_emb_dim=hidden_size), # [B, channel_out, *, *]
                       ResnetBlock(channel_out, channel_out, time_emb_dim=hidden_size), # [B, channel_out, *, *]
                       Residual(Rezero(LinearAttention(channel_out))), # [B, channel_out, *, *]
                       DownsampleConv(channel_out) if not is_last else torch.nn.Identity(),
                       DownsampleConv(1) if not is_last else torch.nn.Identity()])) # [B, channel_out, */2, */2]

        mid_channel = channels[-1] # 4 * dim
        self.mid_block1 = ResnetBlock(mid_channel, mid_channel, time_emb_dim=hidden_size)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_channel)))
        self.mid_block2 = ResnetBlock(mid_channel, mid_channel, time_emb_dim=hidden_size)

        for ind, (channel_in, channel_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(torch.nn.ModuleList([
                     ResnetBlock(channel_out * 2, channel_in, time_emb_dim=hidden_size),
                     ResnetBlock(channel_in, channel_in, time_emb_dim=hidden_size),
                     Residual(Rezero(LinearAttention(channel_in))),
                     UpsampleConv(channel_in), UpsampleConv(1)]))
        self.final_block = Block(channel, channel)
        self.final_conv = torch.nn.Conv2d(channel, 1, 1)

    def forward(self, x, t, cond, mask, spk=None):
        """
        :param x: [B, 1, M, T], M: 梅尔通道数, T: 帧数
        :param mask: [B, 1, T]
        :param t: [B, 1]
        :param cond: [B, H, T]
        :return:
           noise: [B, 1, M, T]
        """
        
        t = self.time_pos_emb(t) # [B, H]
        t = self.tmlp(t)
        cond = self.cmlp(cond.transpose(1, 2)).transpose(1, 2) 
        cond = cond.unsqueeze(1) # [B, 1, M, T]
        x = torch.cat([cond, x], 1)
        mask = mask.unsqueeze(1)
        hiddens = []
        masks = [mask]
        for resnet1, resnet2, attn, downsample_conv, downsample_cond in self.downs:
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t, cond)
            x = resnet2(x, mask_down, t, cond)
            x = attn(x)
            hiddens.append(x)
            x = downsample_conv(x * mask_down)
            cond = downsample_cond(cond * mask_down)
            masks.append(mask_down[:, :, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]
        x = self.mid_block1(x, mask_mid, t, cond)
        x = self.mid_attn(x)
        x = self.mid_block2(x, mask_mid, t, cond)

        for resnet1, resnet2, attn, upsample_conv, upsample_cond in self.ups:
            mask_up = masks.pop()
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = resnet1(x, mask_up, t, cond)
            x = resnet2(x, mask_up, t, cond)
            x = attn(x)
            x = upsample_conv(x * mask_up)
            cond = upsample_cond(cond * mask_up)

        x = self.final_block(x, mask)
        output = self.final_conv(x * mask)

        return output * mask

class UncondUNET(BaseModule):
    def __init__(self, channel, hidden_size, channel_mults=(1, 2, 4)):
        super(UncondUNET, self).__init__()
        self.channel = channel
        self.channel_mults = channel_mults

        channels = [1, *map(lambda m: channel * m, channel_mults)] # 通道数为2，因为cat(cond, x, dim=1)
        in_out = list(zip(channels[:-1], channels[1:])) # [(2, channel), (channel, 2*channel), (2*channel, 4*channel)]
        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        num_resolutions = len(in_out) # 3

        for ind, (channel_in, channel_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(torch.nn.ModuleList([
                       UncondResnetBlock(channel_in, channel_out, time_emb_dim=hidden_size), # [B, channel_out, *, *]
                       UncondResnetBlock(channel_out, channel_out, time_emb_dim=hidden_size), # [B, channel_out, *, *]
                       Residual(Rezero(LinearAttention(channel_out))), # [B, channel_out, *, *]
                       DownsampleConv(channel_out) if not is_last else torch.nn.Identity(),
                       ])) # [B, channel_out, */2, */2]

        mid_channel = channels[-1] # 4 * dim
        self.mid_block1 = UncondResnetBlock(mid_channel, mid_channel, time_emb_dim=hidden_size)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_channel)))
        self.mid_block2 = UncondResnetBlock(mid_channel, mid_channel, time_emb_dim=hidden_size)

        for ind, (channel_in, channel_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(torch.nn.ModuleList([
                     UncondResnetBlock(channel_out * 2, channel_in, time_emb_dim=hidden_size),
                     UncondResnetBlock(channel_in, channel_in, time_emb_dim=hidden_size),
                     Residual(Rezero(LinearAttention(channel_in))),
                     UpsampleConv(channel_in)]))
        self.final_block = Block(channel, channel)
        self.final_conv = torch.nn.Conv2d(channel, 1, 1)

    def forward(self, x, mask):
        """
        :param x: [B, 1, H, T], M: 梅尔通道数, T: 帧数
        :param mask: [B, 1, T]
        :param t: [B, 1]
        :param cond: [B, H, T]
        :return:
           noise: [B, 1, H, T]
        """
        
        mask = mask.unsqueeze(1)
        hiddens = []
        masks = [mask]
        for resnet1, resnet2, attn, downsample_conv in self.downs:
            mask_down = masks[-1]
            x = resnet1(x, mask_down)
            x = resnet2(x, mask_down)
            x = attn(x)
            hiddens.append(x)
            x = downsample_conv(x * mask_down)
            masks.append(mask_down[:, :, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]
        x = self.mid_block1(x, mask_mid)
        x = self.mid_attn(x)
        x = self.mid_block2(x, mask_mid)

        for resnet1, resnet2, attn, upsample_conv in self.ups:
            mask_up = masks.pop()
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = resnet1(x, mask_up)
            x = resnet2(x, mask_up)
            x = attn(x)
            x = upsample_conv(x * mask_up)

        x = self.final_block(x, mask)
        output = self.final_conv(x * mask)

        return output * mask

class DiffNet(nn.Module):
    def __init__(self, in_dims=80):
        super().__init__()
        self.params = params = AttrDict(
            # Model params
            encoder_hidden=hparams['hidden_size'],
            residual_layers=hparams['residual_layers'],
            residual_channels=hparams['residual_channels'],
            dilation_cycle_length=hparams['dilation_cycle_length'],
        )
        self.input_projection = Conv1d(in_dims, params.residual_channels, 1)
        self.diffusion_embedding = SinusoidalPosEmb(params.residual_channels)
        dim = params.residual_channels
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.Dropout(0.1),
            Mish(),
            nn.Linear(dim * 4, dim)
        )
        self.residual_layers = nn.ModuleList([
            ResidualBlock(params.encoder_hidden, params.residual_channels, 2 ** (i % params.dilation_cycle_length))
            for i in range(params.residual_layers)
        ])
        self.skip_projection = Conv1d(params.residual_channels, params.residual_channels, 1)
        self.output_projection = Conv1d(params.residual_channels, in_dims, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond, mask):
        """
        :param spec: [B, 1, M, T]
        :param mask: [B, 1, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, H, T]
        :return:
            score: [B, 1, M, T]
        """
        x = spec[:, 0]
        x = self.input_projection(x)  # x [B, residual_channel, T]

        x = F.relu(x)
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step) # [B, H]
        skip = []
        for layer_id, layer in enumerate(self.residual_layers):
            x, skip_connection = layer(x, cond, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)  # [B, 80, T]
        x = x * mask
        return x[:, None, :, :]

class AdaDiffNet(nn.Module):
    def __init__(self, in_dims=80):
        super().__init__()
        self.params = params = AttrDict(
            # Model params
            encoder_hidden=hparams['hidden_size'],
            residual_layers=hparams['residual_layers'],
            residual_channels=hparams['residual_channels'],
            dilation_cycle_length=hparams['dilation_cycle_length'],
        )
        self.input_projection = Conv1d(in_dims, params.residual_channels, 1)
        self.diffusion_embedding = SinusoidalPosEmb(params.residual_channels)
        dim = params.residual_channels
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.Dropout(0.1),
            Mish(),
            nn.Linear(dim * 4, dim)
        )
        self.residual_layers = nn.ModuleList([
            AdaCondResBlock(params.encoder_hidden, params.residual_channels, 2 ** (i % params.dilation_cycle_length))
            for i in range(params.residual_layers)
        ])
        self.skip_projection = Conv1d(params.residual_channels, params.residual_channels, 1)
        self.output_projection = Conv1d(params.residual_channels, in_dims, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond, mask):
        """
        :param spec: [B, 1, M, T]
        :param mask: [B, 1, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, H, T]
        :return:
            score: [B, 1, M, T]
        """
        x = spec[:, 0]
        x = self.input_projection(x)  # x [B, residual_channel, T]

        x = F.relu(x)
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step) # [B, H]
        skip = []
        for layer_id, layer in enumerate(self.residual_layers):
            x, skip_connection = layer(x, cond, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)  # [B, 80, T]
        x = x * mask
        return x[:, None, :, :]