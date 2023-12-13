import math
import random
from functools import partial
from inspect import isfunction
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from einops import rearrange

from modules.fastspeech.fs2 import FastSpeech2
# from modules.diffsinger_midi.fs2 import FastSpeech2MIDI
from usr.gezixi_task import FastSpeech2MIDI
from utils.hparams import hparams
from utils.sampling import get_sampling_fn
from utils import mutils
from usr.diff import net
from usr.diff.utils.random_utils import get_generator, dev
from usr.diff.karras_diffusion import karras_sample

class SDEDiffusion(nn.Module):
    def __init__(self, phone_encoder, out_dims, score_model,
                 spec_min=None, spec_max=None):
        super().__init__()
        # score_fn为噪声估计函数，核心
        self.score_model = score_model
        # 复用fastspeech2的encoder
        if hparams.get('use_midi') is not None and hparams['use_midi']:
            self.fs2 = FastSpeech2MIDI(phone_encoder, out_dims)
            del self.fs2.decoder
        else:
            self.fs2 = FastSpeech2(phone_encoder, out_dims)
        # freeze encoder
        if not hparams['train_fs2']:
            for p in self.fs2.parameters():
                p.requires_grad = False
        self.mel_bins = out_dims

        self.register_buffer('spec_min', torch.FloatTensor([-10.])[None, None, :hparams['keep_bins']])
        self.register_buffer('spec_max', torch.FloatTensor([1.5])[None, None, :hparams['keep_bins']])

        to_torch = partial(torch.tensor, dtype=torch.float32)
        

    def compute_loss(self, batch, sde, t, cond, continuous):
        _, mask = cond
        noise =  torch.randn_like(batch)
        score_fn = mutils.get_score_fn(sde, self.score_model, train=True, continuous=continuous)

        if not continuous:
            discrete_t = t * sde.N
            discrete_t = discrete_t.long()
            discrete_t = discrete_t / sde.N
            mean, std = sde.marginal_prob(batch, discrete_t)
        else:
            mean, std = sde.marginal_prob(batch, t)
        perturbed_data = mean + std[:, None, None, None] * noise
        score = score_fn(perturbed_data, t, cond) # [B, 1, M, T]
        score = score * std[:, None, None, None]

        mask = mask[:, None, :, :].repeat(1, 1, self.mel_bins, 1) # [B, 1, M, T]
        mat_loss = mask * ((score + noise)**2)
        loss =  torch.sum(mat_loss) / (torch.sum(mask) + 1e-8)
        if torch.isnan(loss):
            nan_pos = score.isnan()
            torch.set_printoptions(threshold=50000)
            print(f"mel {batch[nan_pos]}, score {score}, mel nan {batch.isnan()}, mel num nan {batch.isnan().sum()}")
            raise ValueError

        return loss

    def forward(self, txt_tokens, sde, mel2ph=None, spk_embed=None,
                ref_mels=None, f0=None, uv=None, energy=None, infer=False, eps=1e-3, continuous=False, **kwargs):
        b, *_, device = *txt_tokens.shape, txt_tokens.device
        ret = self.fs2(txt_tokens, mel2ph, spk_embed, ref_mels, f0, uv, energy,
                       skip_decoder=True, infer=infer, **kwargs)
        cond = ret['decoder_inp'].transpose(1, 2) # [B, H, T_mel]
        
        if not infer:
            # 采样t
            t = torch.rand((b,), device=device).clamp(eps,sde.T) # 设定最小值eps，为了训练稳定
            x = ref_mels[0]
            x = self.norm_spec(x)
            x = x.transpose(1, 2)[:, None, :, :]  # [B, 1, M, T]
            mask = (mel2ph != 0).unsqueeze(1).float() # [B, 1, T_mel]
            mask = mask.to(device)
            # mask = torch.ones_like(mask,)
            condition = [cond, mask]
            ret['diff_loss'] = self.compute_loss(x, sde, t, condition, continuous)
        else:
            shape = (cond.shape[0], 1, self.mel_bins, mel2ph.shape[1]) # [B, 1, M, T_mel]
            mask = torch.ones([cond.shape[0], 1, cond.shape[2]], device = device) # [B ,1, T]
            condition = [cond,mask]
            # denorm_fn = lambda x : x
            sampling_fn = get_sampling_fn(hparams, sde, shape, self.denorm_spec, eps = eps)
            sample, n = sampling_fn(self.score_model, condition) # [B, T, M]
            ret['mel_out'] = sample
        return ret
    # 放缩到[-1,1]
    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min

    def cwt2f0_norm(self, cwt_spec, mean, std, mel2ph):
        return self.fs2.cwt2f0_norm(cwt_spec, mean, std, mel2ph)

    def out2mel(self, x):
        return x

class EDMDiffusion(nn.Module):
    def __init__(self, phone_encoder, out_dims, score_model,
                 spec_min=None, spec_max=None):
        super().__init__()
        # score_fn为噪声估计函数，核心
        self.score_model = score_model
        # 复用fastspeech2的encoder
        if hparams.get('use_midi') is not None and hparams['use_midi']:
            self.fs2 = FastSpeech2MIDI(phone_encoder, out_dims)
            del self.fs2.decoder
        else:
            self.fs2 = FastSpeech2(phone_encoder, out_dims)
            del self.fs2.decoder
        # freeze encoder
        if not hparams['train_fs2']:
            for p in self.fs2.parameters():
                p.requires_grad = False
        self.mel_bins = out_dims

        self.register_buffer('spec_min', torch.FloatTensor([-10.])[None, None, :hparams['keep_bins']])
        self.register_buffer('spec_max', torch.FloatTensor([1.5])[None, None, :hparams['keep_bins']])

        to_torch = partial(torch.tensor, dtype=torch.float32)
        

    def compute_loss(self, batch, sde, t, cond, continuous):
        _, mask = cond
        noise =  torch.randn_like(batch)
        score_fn = mutils.get_score_fn(sde, self.score_model, train=True, continuous=continuous)

        if not continuous:
            discrete_t = t * sde.N
            discrete_t = discrete_t.long()
            discrete_t = discrete_t / sde.N
            mean, std = sde.marginal_prob(batch, discrete_t)
        else:
            mean, std = sde.marginal_prob(batch, t)
        perturbed_data = mean + std[:, None, None, None] * noise
        score = score_fn(perturbed_data, t, cond) # [B, 1, M, T]
        score = score * std[:, None, None, None]

        mask = mask[:, None, :, :].repeat(1, 1, self.mel_bins, 1) # [B, 1, M, T]
        mat_loss = mask * ((score + noise)**2)
        loss =  torch.sum(mat_loss) / (torch.sum(mask) + 1e-8)
        if torch.isnan(loss):
            nan_pos = score.isnan()
            torch.set_printoptions(threshold=50000)
            print(f"mel {batch[nan_pos]}, score {score}, mel nan {batch.isnan()}, mel num nan {batch.isnan().sum()}")
            raise ValueError

        return loss

    def forward(self, txt_tokens, diffusion, schedule_sampler, mel2ph=None, spk_embed=None,
                ref_mels=None, f0=None, uv=None, energy=None, infer=False, **kwargs):
        b, *_, device = *txt_tokens.shape, txt_tokens.device
        ret = self.fs2(txt_tokens, mel2ph, spk_embed, ref_mels, f0, uv, energy,
                       skip_decoder=True, infer=infer, **kwargs)
        cond = ret['decoder_inp'].transpose(1, 2) # [B, H, T_mel]
        
        if not infer:
            mask = (mel2ph != 0).unsqueeze(1).float() # [B, 1, T]
            mask = mask.to(dev())
            mels = self.norm_spec(ref_mels[0])
            mels = mels.transpose(1, 2)[:, None, :, :]  # [B, 1, M, T]
            model_kwargs = {
                            'cond': cond, 
                            'mask': mask
                            }
            t, weights = schedule_sampler.sample(mels.shape[0], dev())

            compute_losses = partial(
                diffusion.training_losses,
                self.score_model,
                mels,
                t,
                model_kwargs=model_kwargs,
            )
            ret['diff_loss'] = compute_losses().mean()
        else:
            shape = (mel2ph.shape[0], 1, self.mel_bins, mel2ph.shape[1]) # [B, 1, M, T_mel]
            mask = torch.ones([mel2ph.shape[0], 1, mel2ph.shape[1]], device = device) # [B, 1, T]
            model_kwargs = {
                            'cond': cond, 
                            'mask': mask
                            }
            # denorm_fn = lambda x : x
            generator = get_generator('determ-indiv', 50000, hparams['seed'])
            sample = karras_sample(
                    diffusion,
                    self.score_model,
                    shape,
                    steps=hparams['num_steps'],
                    model_kwargs=model_kwargs,
                    device=dev(),
                    clip_denoised=True,
                    sampler='dpm',
                    sigma_min=0.002,
                    sigma_max=80.0,
                    s_churn=40,
                    s_tmin=0.05,
                    s_tmax=50,
                    s_noise=1.003,
                    generator=generator,
                    ts=None,
                )
            # sample = karras_sample(
            #         diffusion,
            #         self.score_model,
            #         shape,
            #         steps=hparams['num_steps'],
            #         model_kwargs=model_kwargs,
            #         device=dev(),
            #         clip_denoised=True,
            #         sampler='heun',
            #         sigma_min=0.002,
            #         sigma_max=80.0,
            #         generator=generator,
            #         ts=None,
            #     )
            sample = sample[:, 0].transpose(1, 2)
            ret['mel_out'] = self.denorm_spec(sample)
        return ret
    # 放缩到[-1,1]
    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min

    def cwt2f0_norm(self, cwt_spec, mean, std, mel2ph):
        return self.fs2.cwt2f0_norm(cwt_spec, mean, std, mel2ph)

    def out2mel(self, x):
        return x
