import torch
import torch.nn as nn
from torch.nn import functional as F
import math

from modules.commons.espnet_positional_embedding import RelPositionalEncoding
from modules.commons.common_layers import SinusoidalPositionalEmbedding, Linear, EncSALayer, DecSALayer, BatchNorm1dTBC, MultiheadAttention
from modules.commons.common_layers import Embedding, TransformerFFNLayer
from modules.fastspeech.tts_modules import TransformerEncoderLayer
from utils.hparams import hparams
from conformer.encoder import ConformerBlock

DEFAULT_MAX_SOURCE_POSITIONS = 2000
DEFAULT_MAX_TARGET_POSITIONS = 6000


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

class TransformerDecoderLayer(nn.Module):
    def __init__(self, hidden_size, dropout, kernel_size=None, num_heads=2, norm='ln'):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_heads = num_heads
        self.op = DecSALayer(
            hidden_size, num_heads, dropout=dropout,
            attention_dropout=0.1, relu_dropout=dropout,
            kernel_size=kernel_size
            if kernel_size is not None else hparams['enc_ffn_kernel_size'], act=hparams['ffn_act'])

    def forward(self, x, encoder_out, encoder_padding_mask, **kwargs):
        return self.op(x, encoder_out, encoder_padding_mask, **kwargs)

class SADecoder(nn.Module):
    def __init__(self, hidden_size, num_layers, ffn_kernel_size=9, dropout=None, num_heads=2,
                 use_pos_embed=True, use_last_norm=True, norm='ln', use_pos_embed_alpha=True):
        super().__init__()
        self.num_layers = num_layers
        embed_dim = self.hidden_size = hidden_size
        self.dropout = dropout if dropout is not None else hparams['dropout']
        self.use_pos_embed = use_pos_embed
        self.use_last_norm = use_last_norm
        if use_pos_embed:
            self.max_source_positions = DEFAULT_MAX_TARGET_POSITIONS
            self.padding_idx = 0
            self.pos_embed_alpha = nn.Parameter(torch.Tensor([1])) if use_pos_embed_alpha else 1
            self.embed_positions = SinusoidalPositionalEmbedding(
                embed_dim, self.padding_idx, init_size=DEFAULT_MAX_TARGET_POSITIONS,
            )

        self.layers = nn.ModuleList([])
        self.in_proj = nn.Linear(80, hidden_size)
        self.out_proj = nn.Linear(hidden_size, 80)
        self.label_embed = SinusoidalPosEmb(embed_dim)

        self.layers.extend([
            TransformerEncoderLayer(self.hidden_size, self.dropout,
                                    kernel_size=ffn_kernel_size, num_heads=num_heads)
            for _ in range(self.num_layers)
        ])
        if self.use_last_norm:
            if norm == 'ln':
                self.layer_norm = nn.LayerNorm(embed_dim)
            elif norm == 'bn':
                self.layer_norm = BatchNorm1dTBC(embed_dim)
        else:
            self.layer_norm = None

    def forward(self, x, labels=None, cond=None, mask=None, attn_mask=None, return_hiddens=False):
        """
        :param x: [B, 1, M, T]
        :param mask: [B, 1, T]
        :param labels: [B]
        :param cond: [B, C, T]
        :return: [B, 1, M, T]
        """
        x = x.squeeze(1).transpose(1, 2) # [B, T, M]
        x = self.in_proj(x) # [B, T, C]
        x = x + self.label_embed(labels)[:, None, :] + cond.transpose(1, 2)
        padding_mask = x.abs().sum(-1).eq(0).data
        nonpadding_mask_TB = 1 - padding_mask.transpose(0, 1).float()[:, :, None]  # [T, B, 1]

        if self.use_pos_embed:
            positions = self.pos_embed_alpha * self.embed_positions(x[..., 0])
            x = x + positions
            x = F.dropout(x, p=self.dropout, training=self.training)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1) * nonpadding_mask_TB
        hiddens = []
        for layer in self.layers:
            x = layer(x, encoder_padding_mask=padding_mask) 
            x = x * nonpadding_mask_TB
            hiddens.append(x)
        if self.use_last_norm:
            x = self.layer_norm(x) * nonpadding_mask_TB
        if return_hiddens:
            x = torch.stack(hiddens, 0)  # [L, T, B, C]
            x = x.transpose(1, 2)  # [L, B, T, C]
        else:
            x = x.transpose(0, 1)  # [B, T, C]
        return self.out_proj(x).unsqueeze(1).transpose(2, 3) * mask[:, None, :, :]

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class AdaLNSALayer(nn.Module):
    def __init__(self, c, num_heads, dropout, attention_dropout=0.1,
                 relu_dropout=0.1, kernel_size=9, padding='SAME', norm='ln', act='gelu'):
        super().__init__()
        self.c = c
        self.dropout = dropout
        self.num_heads = num_heads
        if num_heads > 0:
            if norm == 'ln':
                self.layer_norm1 = nn.LayerNorm(c)
            elif norm == 'bn':
                self.layer_norm1 = BatchNorm1dTBC(c)
            self.self_attn = MultiheadAttention(
                self.c, num_heads, self_attention=True, dropout=attention_dropout, bias=False,
            )
        if norm == 'ln':
            self.layer_norm2 = nn.LayerNorm(c)
        elif norm == 'bn':
            self.layer_norm2 = BatchNorm1dTBC(c)
        self.ffn = TransformerFFNLayer(
            c, 4 * c, kernel_size=kernel_size, dropout=relu_dropout, padding=padding, act=act)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(c, c, bias=True),
            nn.Dropout(0.1),
            nn.Linear(c, 6 * c, bias=True)
        )

    def forward(self, x, cond, encoder_padding_mask=None, **kwargs):
        '''
        :param x: [T, B, C]
        :param cond: [T, B, C]
        :param encoder_padding_mask: [B, T]
        '''
        layer_norm_training = kwargs.get('layer_norm_training', None)
        if layer_norm_training is not None:
            self.layer_norm1.training = layer_norm_training
            self.layer_norm2.training = layer_norm_training
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(6, dim=2)
        if self.num_heads > 0:
            residual = x
            x = modulate(self.layer_norm1(x), shift_msa, scale_msa)
            x, _, = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask
            )
            x = x * gate_msa
            x = F.dropout(x, self.dropout, training=self.training)
            x = residual + x
            x = x * (1 - encoder_padding_mask.float()).transpose(0, 1)[..., None]

        residual = x
        x = modulate(self.layer_norm2(x), shift_mlp, scale_mlp)
        x = self.ffn(x)
        x = x * gate_mlp
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        x = x * (1 - encoder_padding_mask.float()).transpose(0, 1)[..., None]
        return x

class AdaLNDecoder(nn.Module):
    def __init__(self, hidden_size, num_layers, ffn_kernel_size=9, dropout=None, num_heads=2,
                 use_pos_embed=True, use_last_norm=True, norm='ln', use_pos_embed_alpha=True):
        super().__init__()
        self.num_layers = num_layers
        embed_dim = self.hidden_size = hidden_size
        self.dropout = dropout if dropout is not None else hparams['dropout']
        self.use_pos_embed = use_pos_embed
        self.use_last_norm = use_last_norm
        if use_pos_embed:
            self.max_source_positions = DEFAULT_MAX_TARGET_POSITIONS
            self.padding_idx = 0
            self.pos_embed_alpha = nn.Parameter(torch.Tensor([1])) if use_pos_embed_alpha else 1
            self.embed_positions = SinusoidalPositionalEmbedding(
                embed_dim, self.padding_idx, init_size=DEFAULT_MAX_TARGET_POSITIONS,
            )
        self.in_proj = nn.Linear(80, hidden_size)
        self.out_proj = nn.Linear(hidden_size, 80)
        self.label_embed = SinusoidalPosEmb(embed_dim)
        self.label_proj = nn.Sequential(
        					nn.Dropout(0.1),
        					nn.Linear(embed_dim, embed_dim),
        					nn.SiLU()
        					)
        if self.use_last_norm:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size, bias=True),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, 2 * hidden_size, bias=True)
            )

        self.layers = nn.ModuleList([])
        self.layers.extend([
            AdaLNSALayer(self.hidden_size, num_heads=num_heads, dropout=self.dropout,
                            kernel_size=ffn_kernel_size)
            for _ in range(self.num_layers)
        ])
        if self.use_last_norm:
            if norm == 'ln':
                self.layer_norm = nn.LayerNorm(embed_dim)
            elif norm == 'bn':
                self.layer_norm = BatchNorm1dTBC(embed_dim)
        else:
            self.layer_norm = None
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Zero-out adaLN modulation layers in DiT blocks:
        for layer in self.layers:
            nn.init.constant_(layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(layer.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        if self.use_last_norm:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.out_proj.weight, 0)
        nn.init.constant_(self.out_proj.bias, 0)


    def forward(self, x, labels, cond, mask=None, attn_mask=None, return_hiddens=False):
        """
        :param x: [B, 1, M, T]
        :param mask: [B, 1, T]
        :param labels: [B]
        :param cond: [B, C, T]
        :return: [B, 1, M, T]
        """
        
        x = x.squeeze(1).transpose(1, 2) # [B, T, M]
        x = self.in_proj(x) # [B, T, C]
        cond = cond + self.label_proj(self.label_embed(labels))[:, :, None]
        cond = cond.permute(2, 0, 1) # [T, B, C]
        padding_mask = x.abs().sum(-1).eq(0).data
        nonpadding_mask_TB = 1 - padding_mask.transpose(0, 1).float()[:, :, None]  # [T, B, 1]
        if self.use_pos_embed:
            positions = self.pos_embed_alpha * self.embed_positions(x[..., 0])
            x = x + positions
            x = F.dropout(x, p=self.dropout, training=self.training)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1) * nonpadding_mask_TB
        hiddens = []
        for layer in self.layers:
            x = layer(x, cond, encoder_padding_mask=padding_mask, attn_mask=attn_mask) * nonpadding_mask_TB
            hiddens.append(x)
        if self.use_last_norm:
        	shift, scale = self.adaLN_modulation(cond).chunk(2, dim=2)
        	x = modulate(self.layer_norm(x), shift, scale) * nonpadding_mask_TB
        if return_hiddens:
            x = torch.stack(hiddens, 0)  # [L, T, B, C]
            x = x.transpose(1, 2)  # [L, B, T, C]
        else:
            x = x.transpose(0, 1)  # [B, T, C]
        return self.out_proj(x).unsqueeze(1).transpose(2, 3) * mask[:, None, :, :]