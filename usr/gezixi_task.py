import torch
import utils
import random
from utils.hparams import hparams
from vocoders.base_vocoder import get_vocoder_cls, BaseVocoder
from modules.fastspeech.pe import PitchExtractor
from modules.fastspeech.fs2 import FastSpeech2
# from modules.diffsinger_midi.fs2 import FastSpeech2MIDI
from modules.fastspeech.tts_modules import mel2ph_to_dur, FastspeechEncoder, PitchPredictor, FastspeechDecoder, SinusoidalPosEmb
from modules.commons.common_layers import *
from conformer.encoder import ConformerBlock
from usr.diff.net import UncondUNET, UncondWaveNet

from utils import sde_lib
from utils.pitch_utils import denorm_f0, norm_interp_f0, midi_to_f0, f0_to_coarse, norm_f0
from tasks.tts.fs2_utils import FastSpeechDataset
from tasks.tts.fs2 import FastSpeech2Task
# from usr.phone_rec_task import PHNREC, LabelSmoothingCrossEntropy

import numpy as np
import os
import torch.nn.functional as F
from taming.modules.discriminator.model import NLayerDiscriminator, weights_init
from modules.commons.patchgan import PatchGANDiscriminator, PatchGAN1dDiscriminator
from taming.modules.losses.lpips import LPIPS
from modules.commons.lpips import LPIPSv2
# from taming.modules.losses.vqperceptual import hinge_d_loss, vanilla_d_loss

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss

def lsgan_loss(logits_real, logits_fake):
    ls_loss = 0.5 * (
        torch.mean((logits_real - 1.) ** 2) + 
        torch.mean((logits_fake + 0.) ** 2)
        )
    return ls_loss

class Focalloss(nn.Module):
    def __init__(self, gamma=2., alpha=1., class_num=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.class_num = class_num

    def forward(self, logits, target):
        ''' 
        Args:
            logits: logits prediction of model output [B, class_num, T] or [B, class_num]
            target: ground truth of sampler [B, T] or [B]
        '''
        assert logits.dim() in [2, 3], f"dim {logits.dim()} not supported"
        logprobs = F.log_softmax(logits, dim = 1)   # softmax + log
        target_one_hot = F.one_hot(target, self.class_num)  # 转换成one-hot, [B, T, class_num] or [B, class_num]

        if logits.dim() == 3:
            B, class_num, T = logits.shape
            mask = target > 0
            mask = mask.unsqueeze(2).expand(B, T, class_num)
            mask = mask.to(logits.device)
            logprobs = logprobs.transpose(1, 2) # [B, T, class_num]
        elif logits.dim() == 2:
            mask = torch.ones_like(logits)
        else:
            raise NotImplementedError
        
        score = target_one_hot * logprobs
        sub_pt = 1. - torch.exp(score)
        fl = -self.alpha * (sub_pt)**self.gamma * score
        loss = torch.sum(fl * mask, dim = -1)

        return loss.mean()

class FastspeechMIDIEncoder(FastspeechEncoder):
    def forward_embedding(self, txt_tokens, midi_dur_embedding, slur_embedding, uv_embedding):
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(txt_tokens)
        x = x + midi_dur_embedding + slur_embedding + uv_embedding
        if hparams['use_pos_embed']:
            if hparams.get('rel_pos') is not None and hparams['rel_pos']:
                x = self.embed_positions(x)
            else:
                positions = self.embed_positions(txt_tokens) # 包含padding的position信息
                x = x + positions
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, txt_tokens, midi_dur_embedding, slur_embedding, uv_embedding):
        """

        :param txt_tokens: [B, T]
        :return: {
            'encoder_out': [T x B x C]
        }
        """
        encoder_padding_mask = txt_tokens.eq(self.padding_idx).data
        x = self.forward_embedding(txt_tokens, midi_dur_embedding, slur_embedding, uv_embedding)  # [B, T, H]
        x = super(FastspeechEncoder, self).forward(x, encoder_padding_mask) # cal forward in parent of FastspeechEncoder
        return x

class SinusoidalPosEmb3d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = emb[None, :].repeat(x.shape[0], 1)
        emb = x[:, :, None] * emb[:, None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class PitchEncoder(FastspeechEncoder):
    def forward_embedding(self, midi_tokens, x, slur_embedding, uv_embedding):
        x = x * self.embed_scale + slur_embedding + uv_embedding
        if hparams['use_pos_embed']:
            if hparams.get('rel_pos') is not None and hparams['rel_pos']:
                x = self.embed_positions(x)
            else:
                positions = self.embed_positions(midi_tokens) # 包含padding的position信息
                x = x + positions
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, midi_tokens, midi_embedding, slur_embedding, uv_embedding):
        """

        :param txt_tokens: [B, T]
        :return: {
            'encoder_out': [T x B x C]
        }
        """
        encoder_padding_mask = midi_tokens.eq(self.padding_idx).data
        x = self.forward_embedding(midi_tokens, midi_embedding, slur_embedding, uv_embedding)  # [B, T, H]
        x = super(FastspeechEncoder, self).forward(x, encoder_padding_mask) # cal forward in parent of FastspeechEncoder
        return x

class ConformerDecoder(nn.Module):
    def __init__(self, hidden_size=None, num_layers=None, kernel_size=None, num_heads=None):
        num_heads = hparams['num_heads'] if num_heads is None else num_heads
        hidden_size = hparams['hidden_size'] if hidden_size is None else hidden_size
        kernel_size = hparams['dec_ffn_kernel_size'] if kernel_size is None else kernel_size
        num_layers = hparams['dec_layers'] if num_layers is None else num_layers
        super().__init__()
        self.conformers = nn.ModuleList([ConformerBlock(encoder_dim=hidden_size,
            num_attention_heads=num_heads, conv_kernel_size=kernel_size,) for _ in range(num_layers)
            ])

    def forward(self, x):
        for conformer in self.conformers:
            x = conformer(x)
        return x # [B, T, C]


FS_ENCODERS = {
    'fft': lambda hp, embed_tokens, d: FastspeechMIDIEncoder(
        embed_tokens, hp['hidden_size'], hp['enc_layers'], hp['enc_ffn_kernel_size'],
        num_heads=hp['num_heads']),
}

PES = {
    'fft': lambda hp, embed_tokens, d: PitchEncoder(
        embed_tokens, hp['hidden_size'], hp['enc_layers'], hp['enc_ffn_kernel_size'],
        num_heads=hp['num_heads']),
}

FS_DECODERS = {
    'fft': lambda hp: FastspeechDecoder(
        hp['hidden_size'], hp['dec_layers'], hp['dec_ffn_kernel_size'], hp['num_heads']),
    'conformer': lambda hp: ConformerDecoder(
        hp['hidden_size'], hp['dec_layers'], hp['dec_conv_kernel_size'], hp['num_heads']),
    'wavenet': lambda hp: UncondWaveNet(hp['audio_num_mel_bins']),
    'unet': lambda hp: UncondUNET(channel = hp['unet_channel_0'], hidden_size=hp['hidden_size'], channel_mults=eval(hp['unet_dim_mults']))
}


class FastSpeech2MIDI(FastSpeech2):
    def __init__(self, dictionary, out_dims=None):
        super().__init__(dictionary, out_dims)
        del self.encoder
        del self.decoder
        del self.dur_predictor
        del self.length_regulator

        # encoder_embed_tokens: token embedding
        self.encoder = FS_ENCODERS[hparams['encoder_type']](hparams, self.encoder_embed_tokens, self.dictionary)
        self.decoder = FS_DECODERS[hparams['decoder_type']](hparams)
        if hparams['use_midi']:
            self.pitchencoder = PES['fft'](hparams, self.encoder_embed_tokens, self.dictionary)
            self.midi_embed = Embedding(hparams['pitch_num'], self.hidden_size, self.padding_idx)
            self.midi_dur_layer = Linear(1, self.hidden_size)
            self.film = nn.Sequential(
                            nn.LayerNorm(self.hidden_size),
                            Linear(self.hidden_size, self.hidden_size, bias=False),
                            nn.SiLU(),
                            Linear(self.hidden_size, 2 * self.hidden_size, bias=False),
                            nn.Tanh(),
                            )
        self.is_slur_embed = Embedding(2, self.hidden_size)
        self.uv_embed = Embedding(2, self.hidden_size)
        

    def forward(self, txt_tokens, mel2ph=None, spk_embed=None,
                ref_mels=None, f0=None, uv=None, energy=None, skip_decoder=False,
                spk_embed_dur_id=None, spk_embed_f0_id=None, infer=False, **kwargs):
        ret = {}

        midi_dur_embedding, slur_embedding, uv_embedding = 0, 0, 0
        if kwargs.get('midi_dur') is not None:
            midi_dur_embedding = self.midi_dur_layer(kwargs['midi_dur'][:, :, None])  # [B, T_t, 1] -> [B, T_t, C]
        if kwargs.get('is_slur') is not None:
            slur_embedding = self.is_slur_embed(kwargs['is_slur'])
        if kwargs.get('uv_shengmu') is not None:
            uv_embedding = self.uv_embed(kwargs['uv_shengmu'])
        encoder_out = self.encoder(txt_tokens, midi_dur_embedding, slur_embedding, uv_embedding)  # [B, T_t, C]
        ret['encoder_out'] = encoder_out
        src_nonpadding = (txt_tokens > 0).float()[:, :, None] # [B, T_t, 1]

        if hparams['use_midi']:
            pitch_midi = kwargs['pitch_midi'] # [B, T_t]
            midi_embedding = self.midi_embed(pitch_midi)
            midi_embedding = self.pitchencoder(pitch_midi, midi_embedding, slur_embedding, uv_embedding) * src_nonpadding # [B, T_t, C]
            shift, scale = self.film(midi_embedding).chunk(2, dim=-1)
            decoder_inp_origin = encoder_out * (1. + scale) + shift
            decoder_inp_origin = decoder_inp_origin * src_nonpadding # add midi info
            ret['pitch_embedding'] = midi_embedding
        else:
            decoder_inp_origin = encoder_out
        
        if hparams['use_spk_embed']:
            spk_embed_dur = spk_embed_f0 = spk_embed = self.spk_embed_proj(spk_embed)[:, None, :]
        elif hparams['use_spk_id']:
            spk_embed_id = spk_embed
            if spk_embed_dur_id is None:
                spk_embed_dur_id = spk_embed_id
            if spk_embed_f0_id is None:
                spk_embed_f0_id = spk_embed_id
            spk_embed = self.spk_embed_proj(spk_embed_id)[:, None, :]
            spk_embed_dur = spk_embed_f0 = spk_embed
            if hparams['use_split_spk_id']:
                spk_embed_dur = self.spk_embed_dur(spk_embed_dur_id)[:, None, :]
                spk_embed_f0 = self.spk_embed_f0(spk_embed_f0_id)[:, None, :]
        else:
            spk_embed_dur = spk_embed_f0 = spk_embed = 0

        # # add dur
        # dur_inp = (encoder_out + spk_embed_dur) * src_nonpadding
        # mel2ph = self.add_dur(dur_inp, mel2ph, txt_tokens, ret)

        if hparams['use_midi']:
            # one_pad = pitch_midi == 1
            # resume_midi = pitch_midi + hparams['min_midi'] - 2 # when binarizing the data, we shift midi to remove the unused midis
            # resume_midi[one_pad] = 1
            resume_midi = pitch_midi
            pitch_midi_pad = F.pad(resume_midi, [1, 0])
            reg_midi = torch.gather(pitch_midi_pad, 1, mel2ph)
            ret['reg_midi'] = reg_midi # [B, T]

        decoder_inp = F.pad(encoder_out, [0, 0, 1, 0])
        decoder_inp_origin = F.pad(decoder_inp_origin, [0, 0, 1, 0])
        mel2ph_ = mel2ph[..., None].repeat([1, 1, encoder_out.shape[-1]])
        decoder_inp = torch.gather(decoder_inp, 1, mel2ph_)  # [B, T, C] # w/o midi info
        decoder_inp_origin = torch.gather(decoder_inp_origin, 1, mel2ph_) # w/ midi info

        if hparams['use_midi'] and not hparams['use_pitch_embed']:
            decoder_inp = decoder_inp_origin

        tgt_nonpadding = (mel2ph > 0).float()[:, :, None] # [B, T, 1]

        # add pitch and energy embed
        pitch_inp = (decoder_inp_origin + spk_embed_f0) * tgt_nonpadding
        pitch_inp_ph = (encoder_out + spk_embed_f0) * src_nonpadding

        if hparams['use_pitch_embed']:
            decoder_inp = decoder_inp + self.add_pitch(pitch_inp, f0, uv, mel2ph, ret, encoder_out=pitch_inp_ph)
        if hparams['use_energy_embed']:
            energy_emb = self.add_energy(pitch_inp, energy, ret)
            decoder_inp = decoder_inp + energy_emb

        ret['decoder_inp'] = decoder_inp = (decoder_inp + spk_embed) * tgt_nonpadding # [B, T, C]

        if skip_decoder:
            return ret
        if hparams['decoder_type'] in ['wavenet', 'unet']:
            dec_inp = decoder_inp.unsqueeze(1).transpose(2, 3) # [B, 1, C, T]
            dec_out = self.decoder(dec_inp, tgt_nonpadding.transpose(1, 2)).squeeze(1).transpose(1, 2) # [B, T, C]
            ret['mel_out'] = self.mel_out(dec_out) * tgt_nonpadding
        else:
            ret['mel_out'] = self.run_decoder(decoder_inp, tgt_nonpadding, ret, infer=infer, **kwargs)

        return ret

    def _add_pitch(self, decoder_inp, f0, uv, mel2ph, ret, midi, encoder_out=None):
        decoder_inp = decoder_inp.detach() + hparams['predictor_grad'] * (decoder_inp - decoder_inp.detach())

        pitch_padding = mel2ph == 0

        ret['pitch_pred'] = pitch_pred = self.pitch_predictor(decoder_inp)
        if f0 is None:
            f0_bias = pitch_pred[:, :, 0]
            f0_pred = midi_to_f0(midi) + f0_bias
        if hparams['use_uv'] and uv is None:
            uv = pitch_pred[:, :, 1] > 0
        if f0 is None:
            ret['f0_denorm'] = f0_denorm = f0_pred
        else:
            ret['f0_denorm'] = f0_denorm = denorm_f0(f0, uv, hparams, pitch_padding=pitch_padding)

        pitch = f0_to_coarse(f0_denorm)  # start from 0m, rescaling to [0, 255]
        pitch_embed = self.pitch_embed(pitch)
        return pitch_embed

class GradientReverseFunction(torch.autograd.Function):
        """
        GRF
        """
        @staticmethod
        def forward(ctx, input, coeff = 1.):
            ctx.coeff = coeff
            output = input * 1.0
            return output

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output.neg() * ctx.coeff, None

class GRL_Layer(nn.Module):
    def __init__(self):
        super(GRL_Layer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)

class GRL3dClassfier(nn.Module):
    def __init__(self, hidden_size, out_dims=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.out_dims = out_dims
        self.grl = GRL_Layer()
        self.classfier = nn.Sequential(
                            nn.LayerNorm(self.hidden_size),
                            Transpose((1, 2)), # [B, C, T]
                            nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1),
                            nn.SiLU(),
                            nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1),
                            nn.SiLU(),
                            Transpose((1, 2)), # [B, T, C]
                            nn.Linear(self.hidden_size, self.out_dims), # [B, T, out_dims]
                            )
        
    def forward(self, x):
        '''
        :param x: [B, T, C]
        '''
        x = self.grl(x) # reverse the gradient
        return self.classfier(x)

class GRL2dClassfier(nn.Module):
    def __init__(self, hidden_size, out_dims=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.out_dims = out_dims
        self.grl = GRL_Layer()
        self.classfier = nn.Sequential(
                            nn.LayerNorm(self.hidden_size),
                            Transpose((1, 2)), # [B, C, T_in]
                            nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, stride=2),
                            nn.SiLU(),
                            nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, stride=2),
                            nn.SiLU(),
                            Transpose((1, 2)), # [B, T_out, C]
                            nn.AdaptiveAvgPool2d((1, None)), # [B, 1, C]
                            nn.Linear(self.hidden_size, self.out_dims), # [B, 1, out_dims]
                            )
        
    def forward(self, x):
        '''
        :param x: [B, T, C]
        '''
        x = self.grl(x) # reverse the gradient
        return self.classfier(x).squeeze(1) # [B, out_dims]

class LinearClassfier(nn.Module):
    def __init__(self, hidden_size, out_dims=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.out_dims = out_dims
        self.classfier = nn.Sequential(
                            nn.LayerNorm(self.hidden_size),
                            nn.Linear(self.hidden_size, self.hidden_size),
                            nn.SiLU(),
                            nn.Linear(self.hidden_size, self.out_dims), # [B, T, out_dims]
                            )
        
    def forward(self, x):
        '''
        :param x: [B, T, C]
        '''
        return self.classfier(x) # [B, T, out_dims]

class FastSpeech2Perceptual(nn.Module):
    def __init__(self, dictionary, out_dims=None, disc_conditional=True):
        super().__init__()
        self.n_discs = hparams['n_discs']
        self.predict_pitch = hparams['use_pitch_embed']
        assert self.n_discs in [1, 3], f"discriminator num {hparams['n_discs']} not implemented yet!"
        self.fs2 = FastSpeech2MIDI(dictionary, out_dims)
        # self.perceptual_loss = LPIPS().eval()
        self.perceptual_loss = LPIPSv2(ckpt_path=hparams['speccls_path'], 
                            conv_reduce=False, conv1d=False).eval()
        self.classfiers = nn.ModuleDict({
                                # 'pitch_classfier': LinearClassfier(hparams['hidden_size'], out_dims=hparams['pitch_num']),
                                # 'phone_classfier': LinearClassfier(hparams['hidden_size'], out_dims=len(dictionary)),
                                'pitch_spk_classfier': GRL2dClassfier(hparams['hidden_size'], out_dims=hparams['num_spk']),
                                # 'pitch_phone_classfier': GRL3dClassfier(hparams['hidden_size'], out_dims=len(dictionary)),
                                'phone_spk_classfier': GRL2dClassfier(hparams['hidden_size'], out_dims=hparams['num_spk']),
                                # 'phone_pitch_classfier': GRL3dClassfier(hparams['hidden_size'], out_dims=hparams['pitch_num']),
                                })
        self.sub_discriminators = nn.ModuleList([PatchGANDiscriminator(input_nc=1,
                                                 n_layers=hparams['sub_disc_layers'],
                                                 use_actnorm=True
                                                 ).apply(weights_init) for _ in range(hparams['n_discs'])])
        self.ml_discriminators = nn.ModuleList([PatchGANDiscriminator(input_nc=1,
                                                 n_layers=hparams['ml_disc_layers'],
                                                 use_actnorm=True
                                                 ).apply(weights_init) for _ in range(hparams['n_discs'])])
        if self.predict_pitch:
            self.pitch_cond_mlp = nn.Sequential(
                                nn.LayerNorm(hparams['hidden_size']),
                                nn.Linear(hparams['hidden_size'], hparams['hidden_size']),
                                nn.SiLU(),
                                nn.Linear(hparams['hidden_size'], 1)
                                )
            self.pitch_discriminator = PatchGAN1dDiscriminator(input_nc=2,
                                                     n_layers=2,
                                                     use_actnorm=False
                                                     ).apply(weights_init)
        if disc_conditional:
            self.cond_mlp = nn.Sequential(
                                nn.LayerNorm(hparams['hidden_size']),
                                nn.Linear(hparams['hidden_size'], hparams['audio_num_mel_bins']),
                                nn.SiLU(),
                                nn.Linear(hparams['audio_num_mel_bins'], hparams['audio_num_mel_bins'])
                                )


    def forward(self, txt_tokens, mel2ph=None, spk_embed=None, mel_lengths=None,
                ref_mels=None, f0=None, uv=None, energy=None, skip_decoder=False,
                spk_embed_dur_id=None, spk_embed_f0_id=None, infer=False, **kwargs):
        output = self.fs2(txt_tokens, mel2ph, spk_embed, ref_mels, f0, uv, energy,
                       skip_decoder=False, infer=infer, **kwargs)

        return output

    def out2mel(self, out):
        return out

class GeZiXiSliceDataset(FastSpeechDataset):
    def get_slice(self, slice_len, txt_len, mel2ph):
        mel2ph = torch.from_numpy(mel2ph)
        seg_len = len(mel2ph)
        if seg_len <= slice_len:
            return 0, seg_len, 0, txt_len

        durs = torch.zeros(txt_len).scatter_add(0, mel2ph - 1, torch.ones(seg_len).float())
        durs = F.pad(durs, [1, 0]).long()
        cum_durs = torch.cumsum(durs, 0)
        start = torch.round((torch.rand(1) * seg_len)).long().item()
        end = start + slice_len
        while start + slice_len > seg_len - 1:
            start = torch.round((torch.rand(1) * seg_len)).long().item()

        start_found = False
        end_found = False
        txt_start = 0
        txt_end = 0

        for idx, mark in enumerate(cum_durs):
            if not start_found:
                if mark > start:
                    start = cum_durs[idx - 1]
                    txt_start = idx - 1
                    end = start + slice_len
                    start_found = True
                    if mark > end:
                        txt_end = idx
                        end_found = True
                else:
                    continue
            elif not end_found:
                if mark >= end:
                    txt_end = idx
                    end_found = True
                else:
                    continue
            else:
                break
        assert start_found and end_found, f'{len(cum_durs)}, dur{cum_durs}, start{start}'
        return start, end, txt_start, txt_end

    def num_tokens(self, index):
        return hparams['slice_len']

    def __getitem__(self, index):
        hparams = self.hparams
        item = self._get_item(index)
        max_frames = hparams['max_frames']
        slice_len = hparams['slice_len']
        start, end, txt_start, txt_end = self.get_slice(slice_len, len(item['phone']), item['mel2ph'])

        spec = torch.Tensor(item['mel'])[start:end]
        energy = (spec.exp() ** 2).sum(-1).sqrt() # 均方能量
        mel2ph = torch.LongTensor(item['mel2ph'])[start:end] if 'mel2ph' in item else None # 梅尔谱的每一帧对应的音素序列的序号
        mel2ph = mel2ph - mel2ph[0] + 1
        f0, uv = norm_interp_f0(item["f0"][start:end], hparams) # 标准化f0, floattensor
        phone = torch.LongTensor(item['phone'][txt_start:txt_end])
        pitch = torch.LongTensor(item.get("pitch"))[start:end]
        pitch_midi = torch.LongTensor(item['pitch_midi'])[txt_start:txt_end]
        midi_dur = torch.FloatTensor(item['midi_dur'])[txt_start:txt_end]
        is_slur = torch.LongTensor(item['is_slur'])[txt_start:txt_end]
        word_boundary = torch.LongTensor(item['word_boundary'])[txt_start:txt_end]
        uv_shengmu = torch.LongTensor(item['uv_shengmu'])[txt_start:txt_end]

        sample = {
            "id": index,
            "item_name": item['item_name'],
            "text": item['txt'],
            "txt_token": phone,
            "mel": spec,
            "pitch": pitch, # 梅尔音高,rescaled to [0, 255]
            "energy": energy,
            "f0": f0, # 线性音高
            "uv": uv, # f0等于零的地方
            "mel2ph": mel2ph,
            "mel_nonpadding": spec.abs().sum(-1) > 0,
            "pitch_midi": pitch_midi,
            "midi_dur": midi_dur,
            "is_slur": is_slur,
            "word_boundary": word_boundary,
            "uv_shengmu": uv_shengmu,
        }
        if self.hparams['use_spk_embed']:
            sample["spk_embed"] = torch.Tensor(item['spk_embed'])
        if self.hparams['use_spk_id']:
            sample["spk_id"] = item['spk_id']
            # sample['spk_id'] = 0
            # for key in self.name2spk_id.keys():
            #     if key in item['item_name']:
            #         sample['spk_id'] = self.name2spk_id[key]
            #         break
        if self.hparams['pitch_type'] == 'cwt':
            cwt_spec = torch.Tensor(item['cwt_spec'])[start:end]
            f0_mean = item.get('f0_mean', item.get('cwt_mean'))
            f0_std = item.get('f0_std', item.get('cwt_std'))
            sample.update({"cwt_spec": cwt_spec, "f0_mean": f0_mean, "f0_std": f0_std})
        elif self.hparams['pitch_type'] == 'ph':
            f0_phlevel_sum = torch.zeros_like(phone).float().scatter_add(0, mel2ph - 1, f0)
            f0_phlevel_num = torch.zeros_like(phone).float().scatter_add(
                0, mel2ph - 1, torch.ones_like(f0)).clamp_min(1)
            sample["f0_ph"] = f0_phlevel_sum / f0_phlevel_num
        return sample

    def collater(self, samples):
        batch = super(GeZiXiSliceDataset, self).collater(samples)
        batch['pitch_midi'] = utils.collate_1d([s['pitch_midi'] for s in samples], 0)
        batch['midi_dur'] = utils.collate_1d([s['midi_dur'] for s in samples], 0)
        batch['is_slur'] = utils.collate_1d([s['is_slur'] for s in samples], 0)
        batch['word_boundary'] = utils.collate_1d([s['word_boundary'] for s in samples], 0)
        batch['uv_shengmu'] = utils.collate_1d([s['uv_shengmu'] for s in samples], 0)
        return batch

class GeZiXiDataset(FastSpeechDataset):
    def __getitem__(self, index):
        sample = super(GeZiXiDataset, self).__getitem__(index)
        item = self._get_item(index)
        sample['pitch_midi'] = torch.LongTensor(item['pitch_midi'])[:hparams['max_frames']]
        sample['midi_dur'] = torch.FloatTensor(item['midi_dur'])[:hparams['max_frames']]
        sample['is_slur'] = torch.LongTensor(item['is_slur'])[:hparams['max_frames']]
        sample['word_boundary'] = torch.LongTensor(item['word_boundary'])[:hparams['max_frames']]
        sample['uv_shengmu'] = torch.LongTensor(item['uv_shengmu'])[:hparams['max_frames']]
        sample["spk_id"] = item['spk_id']
        return sample

    def collater(self, samples):
        batch = super(GeZiXiDataset, self).collater(samples)
        batch['pitch_midi'] = utils.collate_1d([s['pitch_midi'] for s in samples], 0)
        batch['midi_dur'] = utils.collate_1d([s['midi_dur'] for s in samples], 0)
        batch['is_slur'] = utils.collate_1d([s['is_slur'] for s in samples], 0)
        batch['word_boundary'] = utils.collate_1d([s['word_boundary'] for s in samples], 0)
        batch['uv_shengmu'] = utils.collate_1d([s['uv_shengmu'] for s in samples], 0)
        spk_ids = torch.LongTensor([s['spk_id'] for s in samples])
        batch['spk_ids'] = spk_ids
        return batch

class FastSpeech2TestTask(FastSpeech2Task):
    def __init__(self):
        super(FastSpeech2TestTask, self).__init__()
        # self.dataset_cls = FastSpeechDataset
        self.dataset_cls = GeZiXiDataset
        # self.lsce = LabelSmoothingCrossEntropy(label_smooth=0.1, class_num=self.phone_encoder.vocab_size)

    def build_tts_model(self):
        out_dims = hparams['audio_num_mel_bins']
        self.model = FastSpeech2MIDI(self.phone_encoder, out_dims)
        # self.model = FastSpeech2PD(self.phone_encoder, out_dims)


    def build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, hparams['decay_steps'], gamma=hparams['step_lr_gamma'])

    def build_optimizer(self, model):
        self.optimizer = optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            weight_decay=hparams['weight_decay'])
        return optimizer

    def run_model(self, model, sample, return_output=False, infer=False):
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        target = sample['mels']  # [B, T_s, 80]
        mel2ph = sample['mel2ph']
        f0 = sample['f0'] # 训练的时候一定要给gt f0，这样decoder才能训练好
        uv = sample['uv']

        energy = sample['energy']
        mel_lengths = sample['mel_lengths']

        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        if hparams['pitch_type'] == 'cwt':
            cwt_spec = sample[f'cwt_spec']
            f0_mean = sample['f0_mean']
            f0_std = sample['f0_std']
            sample['f0_cwt'] = f0 = model.cwt2f0_norm(cwt_spec, f0_mean, f0_std, mel2ph)

        if hparams['use_midi']:
            output = model(txt_tokens, mel2ph=mel2ph, spk_embed=spk_embed,
                           ref_mels=target, f0=f0, uv=uv, energy=energy, infer=infer, pitch_midi=sample['pitch_midi'],
                           midi_dur=sample.get('midi_dur'), is_slur=sample.get('is_slur'), uv_shengmu=sample.get('uv_shengmu'))
        else:
            output = model(txt_tokens, mel2ph=mel2ph, spk_embed=spk_embed, mel_lengths=mel_lengths,
                           ref_mels=target, f0=f0, uv=uv, energy=energy, infer=False)

        losses = {}
        if 'diff_loss' in output:
            losses['mel'] = output['diff_loss']
        self.add_mel_loss(output['mel_out'], target, losses)
        #self.add_dur_loss(output['dur'], mel2ph, txt_tokens, sample['word_boundary'], losses=losses)
        if hparams['use_pitch_embed'] or hparams['use_pe_assist']:
            self.add_pitch_loss(output, sample, losses)
            nonpadding = (mel2ph != 0).float()
            self.add_f0_deriv_loss(output['pitch_pred'], sample, losses, nonpadding)
        if hparams['use_energy_embed']:
            self.add_energy_loss(output['energy_pred'], energy, losses)
        if not return_output:
            return losses
        else:
            return losses, output

    def add_ce_loss(self, txt_tokens, mel2ph, log_prob, losses):
        '''
        txt_tokens: [B,T]
        mel2ph: [B, N]
        log_prob: [N,B,vocab_size]
        '''
        txt_tokens = F.pad(txt_tokens, [1, 0])
        targets = torch.gather(input=txt_tokens, dim=1, index=mel2ph) # [B, N]
        log_prob = log_prob.permute(1, 2, 0)
        losses['ce_loss'] = self.lsce(log_prob, targets) * hparams['lambda_phnrec_ce']

    def add_f0_deriv_loss(self, p_pred, sample, losses, nonpadding):
        f0 = sample['f0']
        uv = sample['uv']
        assert p_pred[..., 0].shape == f0.shape

        f0_pred = p_pred[:, :, 0]
        f0_next = torch.zeros_like(f0).to(f0)
        f0_pred_next = torch.zeros_like(f0_pred).to(f0_pred)
        f0_next[:-1] = f0[1:]
        f0_pred_next[:-1] = f0_pred[1:]
        f0_deriv = (f0 - f0_next).clamp(max=0.1)
        f0_pred_deriv = (f0_pred - f0_pred_next).clamp(max=0.1)

        if hparams['pitch_loss'] in ['l1', 'l2']:
            pitch_deriv_loss_fn = F.l1_loss if hparams['pitch_loss'] == 'l1' else F.mse_loss
            losses['f0_deriv'] = (pitch_deriv_loss_fn(f0_pred_deriv, f0_deriv, reduction='none') * nonpadding).sum() \
                           / nonpadding.sum() * hparams['lambda_f0_deriv']


    def add_dur_loss(self, dur_pred, mel2ph, txt_tokens, wdb, losses=None):
        """
        :param dur_pred: [B, T], float, log scale
        :param mel2ph: [B, T]
        :param txt_tokens: [B, T]
        :param losses:
        :return:
        """
        B, T = txt_tokens.shape
        nonpadding = (txt_tokens != 0).float()
        dur_gt = mel2ph_to_dur(mel2ph, T).float() * nonpadding
        is_sil = torch.zeros_like(txt_tokens).bool()
        is_sil = is_sil | (txt_tokens == self.phone_encoder.encode('sp')[0])
        is_sil = is_sil.float()  # [B, T_txt]

        # phone duration loss
        if hparams['dur_loss'] == 'mse':
            losses['pdur'] = F.mse_loss(dur_pred, (dur_gt + 1).log(), reduction='none')
            losses['pdur'] = (losses['pdur'] * nonpadding).sum() / nonpadding.sum()
            losses['pdur'] = losses['pdur'] * hparams['lambda_ph_dur']
            dur_pred = (dur_pred.exp() - 1).clamp(min=0)
        else:
            raise NotImplementedError

        # use linear scale for sent and word duration
        if hparams['lambda_word_dur'] > 0:
            idx = F.pad(wdb.cumsum(axis=1), (1, 0))[:, :-1]
            # word_dur_g = dur_gt.new_zeros([B, idx.max() + 1]).scatter_(1, idx, midi_dur)  # midi_dur can be implied by add gt-ph_dur
            word_dur_p = dur_pred.new_zeros([B, idx.max() + 1]).scatter_add(1, idx, dur_pred)
            word_dur_g = dur_gt.new_zeros([B, idx.max() + 1]).scatter_add(1, idx, dur_gt)
            wdur_loss = F.mse_loss((word_dur_p + 1).log(), (word_dur_g + 1).log(), reduction='none')
            word_nonpadding = (word_dur_g > 0).float()
            wdur_loss = (wdur_loss * word_nonpadding).sum() / word_nonpadding.sum()
            losses['wdur'] = wdur_loss * hparams['lambda_word_dur']
        if hparams['lambda_sent_dur'] > 0:
            sent_dur_p = dur_pred.sum(-1)
            sent_dur_g = dur_gt.sum(-1)
            sdur_loss = F.mse_loss((sent_dur_p + 1).log(), (sent_dur_g + 1).log(), reduction='mean')
            losses['sdur'] = sdur_loss.mean() * hparams['lambda_sent_dur']

    def validation_step(self, sample, batch_idx):
        outputs = {}
        txt_tokens = sample['txt_tokens']  # [B, T_t]

        target = sample['mels']  # [B, T_s, 80]
        energy = sample['energy']
        # fs2_mel = sample['fs2_mels']
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        mel2ph = sample['mel2ph']
        mel_lengths = sample['mel_lengths']
        f0, uv = None, None
        if hparams['use_gt_f0']:
            f0 = sample['f0']
            uv = sample['uv']

        outputs['losses'] = {}

        outputs['losses'], model_out = self.run_model(self.model, sample, return_output=True, infer=False)

        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        outputs = utils.tensors_to_scalars(outputs)
        if batch_idx < hparams['num_valid_plots']:
            if hparams['use_midi']:
                model_out = self.model(
                    txt_tokens, spk_embed=spk_embed, mel2ph=mel2ph, f0=f0, uv=uv, energy=energy, ref_mels=None, infer=True,
                    pitch_midi=sample['pitch_midi'], midi_dur=sample.get('midi_dur'), is_slur=sample.get('is_slur'), uv_shengmu=sample.get('uv_shengmu'))
            else:
                model_out = self.model(txt_tokens, mel2ph=mel2ph, spk_embed=spk_embed, mel_lengths=mel_lengths, 
                           ref_mels=target, f0=f0, uv=uv, energy=energy, infer=True)
            # self.plot_wav(batch_idx, sample['mels'], model_out['mel_out'], is_mel=True, gt_f0=gt_f0, f0=pred_f0)
            self.plot_mel(batch_idx, sample['mels'], model_out['mel_out'], name=f'fs2mel_{batch_idx}')
            # self.plot_mel(batch_idx, sample['mels'], model_out['fs2_mel'], name=f'fs2mel_{batch_idx}')
            if hparams['use_pitch_embed']:
                self.plot_pitch(batch_idx, sample, model_out)
        return outputs

    def test_step(self, sample, batch_idx):
        # 需要真实的f0给vocoder使用
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        txt_tokens = sample['txt_tokens']
        mel2ph, uv, f0 = None, None, None
        ref_mels = None
        energy = sample['energy']
        mel_lengths = sample['mel_lengths']
        if hparams['profile_infer']:
            pass
        else:
            if hparams['use_gt_dur']:
                mel2ph = sample['mel2ph']
            if hparams['use_gt_f0']:
                f0 = sample['f0']
                uv = sample['uv']
                print('Here using gt f0!!')
            if hparams.get('use_midi') is not None and hparams['use_midi']:
                outputs = self.model(
                    txt_tokens, spk_embed=spk_embed, mel2ph=mel2ph, f0=f0, uv=uv, energy=energy, ref_mels=ref_mels, infer=True,
                    pitch_midi=sample['pitch_midi'], midi_dur=sample.get('midi_dur'), is_slur=sample.get('is_slur'), uv_shengmu=sample.get('uv_shengmu'))
            else:
                outputs = self.model(txt_tokens, mel2ph=mel2ph, spk_embed=spk_embed, mel_lengths=mel_lengths, 
                           ref_mels=None, f0=f0, uv=uv, energy=energy, infer=True)
            if hparams['use_midi']:
                sample['outputs'] = self.model.out2mel(outputs['mel_out'])
            else:
                sample['outputs'] = self.model.out2mel(outputs['mel_out'])
            sample['mel2ph_pred'] = sample['mel2ph']
            if hparams.get('pe_enable') is not None and hparams['pe_enable']:
                sample['f0'] = self.pe(sample['mels'])['f0_denorm_pred']  # pe predict from GT mel
                sample['f0_pred'] = self.pe(sample['outputs'])['f0_denorm_pred']  # pe predict from Pred mel
            else:
                sample['f0'] = denorm_f0(sample['f0'], sample['uv'], hparams)
                sample['f0_pred'] = outputs.get('f0_denorm')
            return self.after_infer(sample)

class FastSpeech2PerceptualTask(FastSpeech2TestTask):
    def __init__(self):
        super(FastSpeech2PerceptualTask, self).__init__()
        self.dataset_cls = GeZiXiDataset
        self.spk_loss_func = Focalloss(class_num=hparams['num_spk'])
        self.pitch_cls_loss_func = Focalloss(class_num=hparams['pitch_num'])
        self.phone_cls_loss_func = Focalloss(class_num=len(self.phone_encoder))

        self.discriminator_iter_start = hparams['discriminator_iter_start']
        self.feature_matching_iter_start = hparams['feature_matching_iter_start']
        if hparams['disc_loss'] == "hinge":
            self.disc_loss = hinge_d_loss
        elif hparams['disc_loss'] == "vanilla":
            self.disc_loss = vanilla_d_loss
        elif hparams['disc_loss'] == "lsgan":
            self.disc_loss = lsgan_loss
        else:
            raise NotImplementedError
        self.disc_conditional = hparams['disc_conditional']
        self.predict_pitch = hparams['predict_pitch']

        self.fm_factor = hparams['fm_factor']
        self.perceptual_weight = hparams['lambda_perceptual']
        self.disc_factor = hparams['disc_factor']
        self.discriminator_weight = hparams['lambda_disc']
        self.disc_weights = hparams['disc_weights']
        self.pitch_cls_weight = hparams['pitch_cls_weight']
        self.phone_cls_weight = hparams['phone_cls_weight']
        self.disent_weight = hparams['disent_weight']
        self.spk_cls_weight = hparams['spk_cls_weight']

    def build_model(self):
        self.build_tts_model()
        utils.print_arch(self.model)
        print("#" * 40)
        utils.print_arch(self.model.perceptual_loss, model_name='perceptual_loss')
        print("#" * 40)
        utils.print_arch(self.model.sub_discriminators, model_name='discriminator')
        print("#" * 40)
        utils.print_arch(self.model.fs2, model_name='fs2')
        return self.model

    def build_tts_model(self):
        out_dims = hparams['audio_num_mel_bins']
        self.model = FastSpeech2Perceptual(self.phone_encoder, out_dims, self.disc_conditional)

    def build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, hparams['decay_steps'], gamma=hparams['step_lr_gamma'])

    def build_optimizers(self, model):
        # discriminator params
        if self.disc_conditional:
            disc_params = list(filter(lambda p: p.requires_grad, model.cond_mlp.parameters()))
        else:
            disc_params = []
        for discriminator in model.sub_discriminators:
            disc_params = disc_params + list(filter(lambda p: p.requires_grad, discriminator.parameters()))
        for discriminator in model.ml_discriminators:
            disc_params = disc_params + list(filter(lambda p: p.requires_grad, discriminator.parameters()))
        if self.predict_pitch:
            disc_params = disc_params + list(filter(lambda p: p.requires_grad, model.pitch_discriminator.parameters()))
            disc_params = disc_params + list(filter(lambda p: p.requires_grad, model.pitch_cond_mlp.parameters()))

        # generator params
        gener_params = list(filter(lambda p: p.requires_grad, model.fs2.parameters()))
        for classfier in model.classfiers.values():
            gener_params = gener_params + list(filter(lambda p: p.requires_grad, classfier.parameters())) 
        
        disc_parameters = sum([np.prod(p.size()) for p in disc_params]) / 1_000_000
        gener_parameters = sum([np.prod(p.size()) for p in gener_params]) / 1_000_000
        print(f"Classfiers {model.classfiers.keys()}")
        print(f"| Optimize discriminator params {disc_parameters:.3f}M, generator params {gener_parameters:.3f}M")

        disc_optimizer = torch.optim.AdamW(
            disc_params,
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            weight_decay=hparams['weight_decay'])

        gener_optimizer = torch.optim.AdamW(
            gener_params,
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            weight_decay=hparams['weight_decay'])
        return disc_optimizer, gener_optimizer

    def configure_optimizers(self):
        d_optm, g_optm = self.build_optimizers(self.model)
        self.d_scheduler = self.build_scheduler(d_optm)
        self.g_scheduler = self.build_scheduler(g_optm)
        return [g_optm, d_optm]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.step()
        optimizer.zero_grad()
        self.d_scheduler.step(self.global_step // hparams['accumulate_grad_batches'])
        self.g_scheduler.step(self.global_step // hparams['accumulate_grad_batches'])

    def training_step(self, sample, batch_idx, optimizer_idx=-1):
        loss_ret = self._training_step(sample, batch_idx, optimizer_idx)
        self.opt_idx = optimizer_idx
        if loss_ret is None:
            return {'loss': None}
        total_loss, log_outputs = loss_ret
        log_outputs = utils.tensors_to_scalars(log_outputs)
        for k, v in log_outputs.items():
            if k not in self.training_losses_meter:
                self.training_losses_meter[k] = utils.AvgrageMeter()
            if not np.isnan(v):
                self.training_losses_meter[k].update(v)
        self.training_losses_meter['total_loss'].update(total_loss.item())

        try:
            log_outputs['lr'] = self.d_scheduler.get_lr()[0]
        except:
            pass

        # log_outputs['all_loss'] = total_loss.item()
        progress_bar_log = log_outputs
        tb_log = {f'tr/{k}': v for k, v in log_outputs.items()}
        return {
            'loss': total_loss,
            'progress_bar': progress_bar_log,
            'log': tb_log
        }

    def _training_step(self, sample, batch_idx, optimizer_idx):
        loss_output = self.run_model(self.model, sample, optimizer_idx)
        total_loss = sum([v for v in loss_output.values() if isinstance(v, torch.Tensor) and v.requires_grad])
        loss_output['batch_size'] = sample['txt_tokens'].size()[0]
        return total_loss, loss_output

    def run_sub_discriminators(self, discs, input):
        def split_sub_band(input):
            low = input[..., 0:40]
            mid = input[..., 20:60]
            high = input[..., 40:80]
            return (low, mid, high)
        if isinstance(discs[0], PatchGAN1dDiscriminator):
            input = input.squeeze(1) # [B, T, M]
        if len(discs) == 1:
            sl = (input)
        elif len(discs) == 3:
            sl = split_sub_band(input)
        else:
            raise NotImplementedError
        logits, features = [], []
        for idx, disc in enumerate(discs):
            logit, feature = disc(sl[idx])
            logits.append(logit)
            features.append(feature)
        return logits, features, sl

    def run_ml_discriminators(self, discs, input):
        '''
        :params discs: the discriminators
        :params discs: the disc input [B, 1, T, M]
        '''
        # def random_split(input):
        #     lengths = [100, 200]
        #     len_input = input.size(-2)
        #     L = input[:]
        #     if len_input < lengths[0]:
        #         S = input[:]
        #         M = input[:]
        #     elif lengths[0] < len_input < lengths[1]:
        #         s_idx = random.randint(0, len_input - lengths[0])
        #         S = input[..., s_idx: s_idx + lengths[0], :]
        #         M = input[:]
        #     else:
        #         s_idx = random.randint(0, len_input - lengths[0])
        #         m_idx = random.randint(0, len_input - lengths[1])
        #         S = input[..., s_idx: s_idx + lengths[0], :]
        #         M = input[..., m_idx: m_idx + lengths[1], :]
        #     return (S, M, L)

        def curate_input(input):
            b, c, t, m = input.shape
            lengths = [20, 50, 100]
            len_input = input.size(-2)
            redun = len_input % lengths[-1]
            res_input = input[:, :, :-redun, :]
            S = res_input.contiguous().view(-1, c, lengths[0], m)
            M = res_input.contiguous().view(-1, c, lengths[1], m)
            L = res_input.contiguous().view(-1, c, lengths[2], m)
            return (S, M, L)

        if isinstance(discs[0], PatchGAN1dDiscriminator):
            input = input.squeeze(1) # [B, T, M]
        if len(discs) == 1:
            sl = (input)
        elif len(discs) == 3:
            sl = curate_input(input)
        else:
            raise NotImplementedError
        logits, features = [], []
        for idx, disc in enumerate(discs):
            logit, feature = disc(sl[idx])
            logits.append(logit)
            features.append(feature)
        return logits, features, sl

    def run_model(self, model, sample, optimizer_idx=0, return_output=False, infer=False):
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        target = sample['mels']  # [B, T, M]
        mel2ph = sample['mel2ph']
        f0 = sample['f0'] # 训练的时候一定要给gt f0，这样decoder才能训练好
        uv = sample['uv']

        energy = sample['energy']
        mel_lengths = sample['mel_lengths']

        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        if hparams['pitch_type'] == 'cwt':
            cwt_spec = sample[f'cwt_spec']
            f0_mean = sample['f0_mean']
            f0_std = sample['f0_std']
            sample['f0_cwt'] = f0 = model.fs2.cwt2f0_norm(cwt_spec, f0_mean, f0_std, mel2ph)

        if hparams['use_midi']:
            output = model.fs2(txt_tokens, mel2ph=mel2ph, spk_embed=spk_embed,
                           ref_mels=target, f0=f0, uv=uv, energy=energy, infer=infer, pitch_midi=sample['pitch_midi'],
                           midi_dur=sample.get('midi_dur'), is_slur=sample.get('is_slur'), uv_shengmu=sample.get('uv_shengmu'))
        else:
            output = model.fs2(txt_tokens, mel2ph=mel2ph, spk_embed=spk_embed, mel_lengths=mel_lengths,
                           ref_mels=target, f0=f0, uv=uv, energy=energy, infer=False)

        losses = {}
        inputs = target[:, None, :, :]
        reconstructions = output['mel_out'][:, None, :, :] # [B, 1, T, M]

        # now the GAN part
        if optimizer_idx == 0:
            # self.add_dur_loss(output['dur'], mel2ph, txt_tokens, sample['word_boundary'], losses=losses)
            self.add_cls_losses(model, sample, output['pitch_embedding'], output['encoder_out'], losses)
            if hparams['use_energy_embed']:
                self.add_energy_loss(output['energy_pred'], energy, losses)

            losses['l1'] = l1_loss = self.l1_loss(output['mel_out'], target)
            if self.perceptual_weight > 0:
                p_loss = model.perceptual_loss(target.contiguous(), output['mel_out'].contiguous())
                losses['per'] = p_loss = self.perceptual_weight * torch.mean(p_loss)
                nll_loss = p_loss + l1_loss
            else:
                nll_loss = l1_loss
            # generator update
            if self.disc_conditional:
                cond = model.cond_mlp(output['decoder_inp'])[:, None, :, :] # [B, 1, T, M]
            else:
                cond = torch.tensor([0.], device=inputs.device).detach()

            # compute sub_discriminator losses
            sub_logits_fake, sub_features_f, sub_sl_f = self.run_sub_discriminators(model.sub_discriminators, reconstructions.contiguous() + cond)
            sub_logits_real, sub_features_r, sub_sl_r = self.run_sub_discriminators(model.sub_discriminators, inputs.contiguous() + cond)
            sub_g_loss = -sum([torch.mean(logits) * self.disc_weights[i] for i, logits in enumerate(sub_logits_fake)])
            sub_fm_loss = sum([model.sub_discriminators[i].feature_matching_loss(sub_features_r[i], sub_features_f[i]) * self.disc_weights[i] \
                            for i in range(len(model.sub_discriminators))])

            # compute ml_discriminator losses
            ml_logits_fake, ml_features_f, ml_sl_f = self.run_ml_discriminators(model.ml_discriminators, reconstructions.contiguous())
            ml_logits_real, ml_features_r, ml_sl_r = self.run_ml_discriminators(model.ml_discriminators, inputs.contiguous())
            ml_g_loss = -sum([torch.mean(logits) / 3 for i, logits in enumerate(ml_logits_fake)])
            ml_fm_loss = sum([model.ml_discriminators[i].feature_matching_loss(ml_features_r[i], ml_features_f[i]) / 3\
                            for i in range(len(model.ml_discriminators))])

            if self.disc_factor > 0.0:
                try:
                    last_layer = self.get_last_layer()
                    sub_d_weight = self.calculate_adaptive_weight(nll_loss, sub_g_loss, last_layer=last_layer)
                    ml_d_weight = self.calculate_adaptive_weight(nll_loss, ml_g_loss, last_layer=last_layer)
                    # sub_d_weight = torch.tensor(1.0)
                    # ml_d_weight = torch.tensor(1.0)
                except RuntimeError:
                    assert not self.training
                    sub_d_weight = torch.tensor(0.0)
                    ml_d_weight = torch.tensor(0.0)
            else:
                sub_d_weight = torch.tensor(0.0)
                ml_d_weight = torch.tensor(0.0)

            ml_disc_factor = utils.adopt_weight(self.disc_factor, self.global_step, threshold=self.discriminator_iter_start)
            losses['ml_g'] = ml_d_loss = ml_d_weight * ml_disc_factor * ml_g_loss

            sub_disc_factor = utils.adopt_weight(self.disc_factor, self.global_step, threshold=self.discriminator_iter_start)
            losses['sub_g'] = sub_d_loss = sub_d_weight * sub_disc_factor * sub_g_loss

            losses['fm'] = 0.5 * (ml_fm_loss + sub_fm_loss) * self.fm_factor

            if self.predict_pitch:
                p_cond = model.pitch_cond_mlp(output['decoder_inp']) # [B, T, 1]
                # losses['pil'] = p_l = self.add_f0_loss(output['pitch_pred'], sample)
                p_l, losses['uv_l']= self.add_f0_loss(output['pitch_pred'], sample)
                losses['pil'] = p_l
                pitch_logits, f_f = model.pitch_discriminator(torch.cat([output['pitch_pred'][:, :, 0][:, :, None], p_cond], dim = -1))
                # p_gen_l = -torch.mean(pitch_logits)
                p_gen_l = 0.5 * torch.mean(pitch_logits ** 2)
                pitch_p_last_layer = self.get_pitch_p_last_layer()

                # params of generator do not needs grad when validating, so autograd do not work
                try:
                    # p_weight = self.calculate_adaptive_weight(p_l, p_gen_l, last_layer=pitch_p_last_layer)
                    p_weight = torch.tensor(1.0)
                except RuntimeError:
                    assert not self.training
                    p_weight = torch.tensor(0.0)
                losses['p_gen'] = p_loss = p_weight * p_gen_l

        if optimizer_idx == 1:
            # second pass for discriminator update
            if self.disc_conditional:
                cond = model.cond_mlp(output['decoder_inp'])[:, None, :, :]
            else:
                cond = torch.tensor([0.], device=inputs.device).detach()
                
            # sub_discriminator
            sub_logits_fake, sub_features_f, sub_sl_f = self.run_sub_discriminators(model.sub_discriminators, reconstructions.contiguous().detach() + cond)
            sub_logits_real, sub_features_r, sub_sl_r = self.run_sub_discriminators(model.sub_discriminators, inputs.contiguous().detach() + cond)
            sub_disc_factor = utils.adopt_weight(self.disc_factor, self.global_step, threshold=self.discriminator_iter_start)
            sub_d_loss_ = sum([self.disc_loss(sub_logits_real[i], sub_logits_fake[i]) * self.disc_weights[i] for i in range(len(sub_logits_real))])
            losses['sub_disc'] = sub_d_loss = sub_disc_factor * sub_d_loss_

            # ml_discriminator
            ml_logits_fake, ml_features_f, ml_sl_f = self.run_ml_discriminators(model.ml_discriminators, reconstructions.contiguous().detach())
            ml_logits_real, ml_features_r, ml_sl_r = self.run_ml_discriminators(model.ml_discriminators, inputs.contiguous().detach())
            ml_disc_factor = utils.adopt_weight(self.disc_factor, self.global_step, threshold=self.discriminator_iter_start)
            ml_d_loss_ = sum([self.disc_loss(ml_logits_real[i], ml_logits_fake[i]) / 3 for i in range(len(ml_logits_real))])
            losses['ml_disc'] = ml_d_loss = ml_disc_factor * ml_d_loss_

            if self.predict_pitch:
                p_cond = model.pitch_cond_mlp(output['decoder_inp']) # [B, T, 1]
                f_input = torch.cat([output['pitch_pred'][:, :, 0][:, :, None], p_cond], dim=-1)
                r_input = torch.cat([sample['f0'][:, :, None], p_cond], dim=-1)
                pitch_f_logits, f_f = model.pitch_discriminator(f_input)
                pitch_r_logits, f_r = model.pitch_discriminator(r_input)
                losses['p_disc'] = self.disc_loss(pitch_r_logits, pitch_f_logits) + \
                                    self.gradient_penalty(model.pitch_discriminator, xr=r_input, xf=f_input)

        if not return_output:
            return losses
        else:
            return losses, output

    def test_step(self, sample, batch_idx):
        # 需要真实的f0给vocoder使用
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        txt_tokens = sample['txt_tokens']
        mel2ph, uv, f0 = None, None, None
        ref_mels = None
        energy = sample['energy']
        mel_lengths = sample['mel_lengths']
        if hparams['profile_infer']:
            pass
        else:
            if hparams['use_gt_dur']:
                mel2ph = sample['mel2ph']
            if hparams['use_gt_f0']:
                f0 = sample['f0']
                uv = sample['uv']
                print('Here using gt f0!!')
            if hparams.get('use_midi') is not None and hparams['use_midi']:
                outputs = self.model.fs2(
                    txt_tokens, spk_embed=spk_embed, mel2ph=mel2ph, f0=f0, uv=uv, energy=energy, ref_mels=ref_mels, infer=True,
                    pitch_midi=sample['pitch_midi'], midi_dur=sample.get('midi_dur'), is_slur=sample.get('is_slur'), uv_shengmu=sample.get('uv_shengmu'))
            else:
                outputs = self.model(txt_tokens, mel2ph=mel2ph, spk_embed=spk_embed, mel_lengths=mel_lengths, 
                           ref_mels=None, f0=f0, uv=uv, energy=energy, infer=True)
            if hparams['use_midi']:
                sample['outputs'] = self.model.out2mel(outputs['mel_out'])
            else:
                sample['outputs'] = self.model.fs2.out2mel(outputs['mel_out'])
            sample['mel2ph_pred'] = sample['mel2ph']
            if hparams.get('pe_enable') is not None and hparams['pe_enable']:
                sample['f0'] = self.pe(sample['mels'])['f0_denorm_pred']  # pe predict from GT mel
                sample['f0_pred'] = self.pe(sample['outputs'])['f0_denorm_pred']  # pe predict from Pred mel
            else:
                sample['f0'] = denorm_f0(sample['f0'], sample['uv'], hparams)
                sample['f0_pred'] = outputs.get('f0_denorm')

            if hparams['use_midi']:
                reg_midi = outputs.get('reg_midi')
                midi_f0 = midi_to_f0(reg_midi)
                sample['midi_gt'] = midi_f0
            return self.after_infer(sample)

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e3).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def get_last_layer(self):
        return self.model.fs2.mel_out.weight

    def get_pitch_p_last_layer(self):
        return self.model.fs2.pitch_predictor.linear.weight

    def add_pitch_metric_loss(self, pitch_tokens, pitch_embedding):
        # only compute the neighbor distance, applicable to situations where bxt is less than 1000
        b, t, c = pitch_embedding.shape 
        len_reg = torch.tensor(c).float()
        pm_loss = torch.tensor(0.).to(pitch_embedding)
        for i in range(b):
            for j in range(t):
                if j == 0:
                    r_post = 2. ** ((pitch_tokens[i, j] - pitch_tokens[i, j + 1]) / 12.)
                    pm_loss = pm_loss + torch.norm(pitch_embedding[i, j] - r_post * pitch_embedding[i, j + 1]) / torch.sqrt(len_reg)
                elif j == t - 1 or pitch_tokens[i, j + 1] == 0:
                    r_pre = 2. ** ((pitch_tokens[i, j] - pitch_tokens[i, j - 1]) / 12.)
                    pm_loss = pm_loss + torch.norm(pitch_embedding[i, j] - r_pre * pitch_embedding[i, j - 1]) / torch.sqrt(len_reg)
                else:
                    r_pre = 2. ** ((pitch_tokens[i, j] - pitch_tokens[i, j - 1]) / 12.)
                    r_post = 2. ** ((pitch_tokens[i, j] - pitch_tokens[i, j + 1]) / 12.)
                    pm_loss = pm_loss + torch.norm(pitch_embedding[i, j] - r_pre * pitch_embedding[i, j - 1]) / torch.sqrt(len_reg) + \
                                        torch.norm(pitch_embedding[i, j] - r_post * pitch_embedding[i, j + 1]) / torch.sqrt(len_reg)
        return pm_loss / (4 * torch.sum(pitch_tokens > 0))

    def add_f0_loss(self, p_pred, sample):
        f0 = sample['f0']
        uv = sample['uv']
        nonpadding = (sample['mel2ph'] != 0).float()
        assert p_pred[..., 0].shape == f0.shape

        f0_pred = p_pred[:, :, 0]
        if hparams['pitch_loss'] in ['l1', 'l2']:
            pitch_deriv_loss_fn = F.l1_loss if hparams['pitch_loss'] == 'l1' else F.mse_loss
            f0_loss = (pitch_deriv_loss_fn(f0_pred, f0, reduction='none') * nonpadding).sum() / nonpadding.sum() * hparams['lambda_f0']

        if hparams['use_uv']:
            assert p_pred[..., 1].shape == uv.shape
            uv_loss = (F.binary_cross_entropy_with_logits(
                p_pred[:, :, 1], uv, reduction='none') * nonpadding).sum() / nonpadding.sum() * hparams['lambda_uv']
        return f0_loss, uv_loss


    def add_cls_losses(self, model, sample, pitch_embedding, phone_embedding, losses):
        for classfier_name, classfier in model.classfiers.items():
            if classfier_name == 'pitch_classfier':
                logits = classfier(pitch_embedding)
                losses['pic'] = self.pitch_cls_weight * self.pitch_cls_loss_func(logits.transpose(1, 2), sample['pitch_midi'])
            elif classfier_name == 'phone_classfier':
                logits = classfier(phone_embedding)
                losses['phc'] = self.phone_cls_weight * self.phone_cls_loss_func(logits.transpose(1, 2), sample['txt_tokens'])
            elif classfier_name == 'pitch_spk_classfier':
                logits = classfier(pitch_embedding)
                losses['pikc'] = self.spk_cls_weight * self.spk_loss_func(logits, sample['spk_ids'])
            elif classfier_name == 'pitch_phone_classfier':
                logits = classfier(pitch_embedding)
                losses['pihc'] = self.disent_weight * self.phone_cls_loss_func(logits.transpose(1, 2), sample['txt_tokens'])
            elif classfier_name == 'phone_spk_classfier':
                logits = classfier(phone_embedding)
                losses['phkc'] = self.spk_cls_weight * self.spk_loss_func(logits, sample['spk_ids'])
            elif classfier_name == 'phone_pitch_classfier':
                logits = classfier(phone_embedding)
                losses['phic'] = self.disent_weight * self.pitch_cls_loss_func(logits.transpose(1, 2), sample['pitch_midi'])
            else:
                raise NotImplementedError

    def gradient_penalty(self, D, xr, xf):
        """

        :param D: discriminator
        :param xr: real sample, [B, C, H, W]
        :param xf: fake sample, [B, C, H, W]
        :return: gradient penalty, scalar
        """
        LAMBDA = hparams['lambda_gp']
        seq_len = xr.size(-2)

        # only constrait for Discriminator
        xf = xf.detach()
        xr = xr.detach()

        alpha = torch.rand(1).to(xf)

        interpolates = alpha * xr + ((1. - alpha) * xf)
        interpolates.requires_grad_()

        disc_interpolates, f_interpolates = D(interpolates)
        # shape like interpolates
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones_like(disc_interpolates),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.contiguous().view(gradients.size(0), -1)
        gp = ((gradients.norm(2, dim=1) - 1.) ** 2).mean() * LAMBDA

        return gp

    def compute_gp(self, discs, sl_r, sl_f):
        n_discs = len(discs)
        gps = []
        for idx, disc in enumerate(discs):
            gps.append(self.gradient_penalty(disc, sl_r[idx], sl_f[idx]))
        return sum(gps) / n_discs
            
     