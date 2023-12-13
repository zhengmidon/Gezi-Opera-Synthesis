import torch
import utils
from utils.hparams import hparams
from .diff.net import WaveNet, UNET, DiffNet, AdaDiffNet
from .diff.diffusion import SDEDiffusion, EDMDiffusion
from .diffspeech_task import DiffSpeechTask
# from modules.diffsinger_midi.fs2 import FastSpeech2MIDI
from usr.gezixi_task import FastSpeech2MIDI, GeZiXiDataset
from modules.fastspeech.tts_modules import mel2ph_to_dur
from usr.diff.openaiunet import UNetModel
from usr.diff.transformer_dec import SADecoder, AdaLNDecoder

from utils import sde_lib
from utils.pitch_utils import denorm_f0
from tasks.tts.fs2_utils import FastSpeechDataset
from usr.diff.karras_diffusion import KarrasDenoiser
from usr.diff.utils.resample import create_named_schedule_sampler

import numpy as np
import os
import torch.nn.functional as F


#  todu
DIFF_DECODERS = {
    'wavenet': lambda hp: WaveNet(hp['audio_num_mel_bins']),
    'unet': lambda hp: UNET(channel = hp['unet_channel_0'], hidden_size=hp['hidden_size'], channel_mults=eval(hp['unet_dim_mults'])),
    'diffnet': lambda hp: DiffNet(hp['audio_num_mel_bins']),
    'adadiffnet': lambda hp: AdaDiffNet(hp['audio_num_mel_bins']),
    'sa_trans': lambda hp: SADecoder(hp['hidden_size'], hp['diff_decoder_layers'], 
                                        hp['diff_decoder_conv_kernel_size'], num_heads=hp['num_heads']),
    'adaln_sa_trans': lambda hp: AdaLNDecoder(hp['hidden_size'], hp['diff_decoder_layers'], 
                                        hp['diff_decoder_conv_kernel_size'], num_heads=hp['num_heads']),
    'openaiunet': lambda hp: UNetModel(
        in_channels=1,
        model_channels=hp['unet_channel_0'],
        out_channels=1,
        num_res_blocks=2,
        attention_resolutions=[1, 2, 4],
        dropout=0.1,
        channel_mult=eval(hp['unet_dim_mults']),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=True,
        use_fp16=False,
        num_heads=hp['num_heads'],
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=True, # true
        transformer_depth=1,
        context_dim=hp['hidden_size'],
        )
}

class EDMTask(DiffSpeechTask):
    def __init__(self):
        super(EDMTask, self).__init__()
        self.dataset_cls = GeZiXiDataset
        # set up SDE
        self.diffusion = KarrasDenoiser(
                        sigma_data=0.5,
                        sigma_max=80.0,
                        sigma_min=0.002,
                        distillation=False,
                        weight_schedule='karras',
                        )
        self.schedule_sampler = create_named_schedule_sampler('lognormal', self.diffusion)

    # contruct model
    def build_tts_model(self):
        mel_bins = hparams['audio_num_mel_bins']
        self.model = EDMDiffusion(
            phone_encoder=self.phone_encoder, # 音素编码
            out_dims=mel_bins, score_model=DIFF_DECODERS[hparams['diff_decoder_type']](hparams),
            spec_min=hparams['spec_min'], spec_max=hparams['spec_max'],
        )
        if hparams['fs2_ckpt'] != '':
            utils.load_ckpt(self.model.fs2, hparams['fs2_ckpt'], 'model', strict=True)
            # self.model.fs2.decoder = None
            for k, v in self.model.fs2.named_parameters():
                v.requires_grad = False

    # forward
    def run_model(self, model, sample, return_output=False, infer=False):

        txt_tokens = sample['txt_tokens']  # [B, T_t]
        target = sample['mels']  # [B, T_s, 80]
        mel2ph = sample['mel2ph']  # [B, T_s]
        f0 = sample['f0']
        uv = sample['uv']
        energy = sample['energy']
        fs2_mel = None #sample['fs2_mels']
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        if hparams['pitch_type'] == 'cwt':
            cwt_spec = sample[f'cwt_spec']
            f0_mean = sample['f0_mean']
            f0_std = sample['f0_std']
            sample['f0_cwt'] = f0 = model.cwt2f0_norm(cwt_spec, f0_mean, f0_std, mel2ph)
        
        output = model(txt_tokens, diffusion=self.diffusion, schedule_sampler=self.schedule_sampler, 
                        mel2ph=mel2ph, spk_embed=spk_embed, ref_mels=[target, fs2_mel], 
                        f0=f0, uv=uv, energy=energy, infer=infer,  
                        is_slur = sample['is_slur'], pitch_midi=sample['pitch_midi'],
                        midi_dur=sample.get('midi_dur'), uv_shengmu=sample.get('uv_shengmu'))

        losses = {}
        if 'diff_loss' in output:
            losses['diff'] = output['diff_loss']
        # self.add_dur_loss(output['dur'], mel2ph, txt_tokens, losses=losses)
        # if hparams['use_pitch_embed']:
        #     self.add_pitch_loss(output, sample, losses)
        # if hparams['train_fs2']:
        #     self.add_mel_loss(output['mel_out'], target, losses)
        if hparams['use_energy_embed']:
            self.add_energy_loss(output['energy_pred'], energy, losses)

        if not return_output:
            return losses
        else:
            return losses, output

    def validation_step(self, sample, batch_idx):
        outputs = {}
        txt_tokens = sample['txt_tokens']  # [B, T_t]

        target = sample['mels']  # [B, T_s, 80]
        energy = sample['energy']
        # fs2_mel = sample['fs2_mels']
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        mel2ph = sample['mel2ph']
        f0 = sample['f0']
        uv = sample['uv']

        outputs['losses'] = {}

        outputs['losses'], model_out = self.run_model(self.model, sample, return_output=True, infer=False)

        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        outputs = utils.tensors_to_scalars(outputs)
        if batch_idx < hparams['num_valid_plots']:
            fs2_mel = None
            model_out = self.model(
                txt_tokens, diffusion=self.diffusion, schedule_sampler=self.schedule_sampler, 
                spk_embed=spk_embed, mel2ph=mel2ph, f0=f0, uv=uv, energy=energy,
                ref_mels=[None, fs2_mel], infer=True, 
                pitch_midi=sample['pitch_midi'], midi_dur=sample.get('midi_dur'), uv_shengmu=sample.get('uv_shengmu'))

            #self.plot_wav(batch_idx, sample['mels'], model_out['mel_out'], is_mel=True, gt_f0=gt_f0, f0=pred_f0)
            self.plot_mel(batch_idx, sample['mels'], model_out['mel_out'], name=f'diffmel_{batch_idx}')
            #self.plot_mel(batch_idx, sample['mels'], fs2_mel, name=f'fs2mel_{batch_idx}')
        return outputs

    def test_step(self, sample, batch_idx):
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        txt_tokens = sample['txt_tokens']
        energy = sample['energy']
        if hparams['profile_infer']:
            pass
        else:
            mel2ph, uv, f0 = None, None, None
            if hparams['use_gt_dur']:
                mel2ph = sample['mel2ph']
            if hparams['use_gt_f0']:
                f0 = sample['f0']
                uv = sample['uv']
            fs2_mel = None
            outputs = self.model(
                txt_tokens, diffusion=self.diffusion, schedule_sampler=self.schedule_sampler,
                spk_embed=spk_embed, mel2ph=mel2ph, f0=f0, uv=uv, ref_mels=[None, fs2_mel], energy=energy,
                infer=True, pitch_midi=sample['pitch_midi'], midi_dur=sample.get('midi_dur'), uv_shengmu=sample.get('uv_shengmu'))

            sample['outputs'] = self.model.out2mel(outputs['mel_out'])
            sample['mel2ph_pred'] = outputs['mel2ph']

            if hparams.get('pe_enable') is not None and hparams['pe_enable']:
                sample['f0'] = self.pe(sample['mels'])['f0_denorm_pred']  # pe predict from GT mel
                sample['f0_pred'] = self.pe(sample['outputs'])['f0_denorm_pred']  # pe predict from Pred mel
            else:
                sample['f0'] = denorm_f0(sample['f0'], sample['uv'], hparams)
                sample['f0_pred'] = outputs.get('f0_denorm')
            return self.after_infer(sample)