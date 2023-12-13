import torch
import torch.nn as nn
import utils
from utils.hparams import hparams
from .diff.net import WaveNet, UNET
from .diffspeech_task import DiffSpeechTask
from vocoders.base_vocoder import get_vocoder_cls, BaseVocoder
from modules.fastspeech.pe import PitchExtractor
from modules.fastspeech.fs2 import FastSpeech2
# from modules.diffsinger_midi.fs2 import FastSpeech2MIDI
from modules.fastspeech.tts_modules import mel2ph_to_dur, FastspeechEncoder
from modules.commons.common_layers import *
from data_gen.tts.data_gen_utils import build_phone_encoder

from utils.pitch_utils import denorm_f0, f0_to_coarse
from tasks.tts.fs2_utils import FastSpeechDataset
from tasks.tts.fs2 import FastSpeech2Task

import numpy as np
import os
import torch.nn.functional as F

def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool() # pad之处为0
    return mask # [B,T]

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, label_smooth=None, class_num=None):
        super().__init__()
        self.label_smooth = label_smooth
        self.class_num = class_num

    def forward(self, logprobs, target):
        ''' 
        Args:
            logprobs: log softmaxed prediction of model output [B,vocab_size,T]
            target: ground truth of sampler [B,T]
        '''
        eps = 1e-12
        
        if self.label_smooth is not None:
            # cross entropy loss with label smoothing, in word scale
            B,vocab_size,T = logprobs.shape
            mask = target > 0
            mask = mask.unsqueeze(2).expand(B, T, vocab_size)
            mask = mask.to(logprobs.device)

            # logprobs = F.log_softmax(pred, dim = 1)   # softmax + log
            logprobs = logprobs.transpose(1, 2)
            target = F.one_hot(target, self.class_num)  # 转换成one-hot,[B,T,vocab_size]
            
            target = torch.clamp(target.float(), min = self.label_smooth / (self.class_num - 1), max = 1.0 - self.label_smooth) # float必需
            loss = -1 * torch.sum(target * logprobs * mask, dim = 2)
        
        else:
            raise ModuleNotFoundError

        return loss.mean()

class Prenet(nn.Module):
    def __init__(self, in_dim, prenet_dim, prenet_k_sizes, prenet_d_sizes, strides, pool_k_sizes):
        super(Prenet, self).__init__()
        assert len(prenet_k_sizes) == len(prenet_d_sizes)
        layers = []
        for i in range(len(prenet_k_sizes)):
            se = nn.Sequential(
                Block(in_channels=1, out_channels=prenet_dim, \
                kernel_size=prenet_k_sizes[i], stride=strides[i], \
                padding=(0, int(prenet_d_sizes[i][1] * (prenet_k_sizes[i][1] - 1) / 2)), 
                dilation=prenet_d_sizes[i], groups=8),
                nn.MaxPool2d(kernel_size=pool_k_sizes[i], stride=1, padding=(0,1), dilation=1)
                )
            layers.append(se)
        self.convs = nn.ModuleList(layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        y = self.convs[0](x)
        for layer in self.convs[1:]:
            y = y + layer(x)
        y = F.dropout(F.relu(y), p=0.2, training=True)
        y = y.squeeze(2).transpose(1,2)
        return y # [B,N,hidden_size]

class MelEncoder(nn.Module):
    def __init__(self, hparams):
        super(MelEncoder, self).__init__()
        self.hparams = hparams
        self.mel_bins = hparams['audio_num_mel_bins']
        self.hidden_size = hparams['hidden_size']
        self.num_layers = 2
        self.prenet_k_sizes = [(40,1), (40,1)]
        self.prenet_d_sizes = [(1,1), (1,1)]
        self.strides = [(10,1), (20,1)]
        self.pool_k_sizes = [(5,3), (3,3)]

        self.prenet = Prenet(
            self.mel_bins,
            self.hidden_size, self.prenet_k_sizes, 
            self.prenet_d_sizes, self.strides, self.pool_k_sizes)

        self.lstm = nn.LSTM(self.hidden_size,
                            int(self.hidden_size // 2), num_layers = self.num_layers,
                            batch_first=True, bidirectional=True)

        self.layer_norm = nn.LayerNorm(self.hidden_size)

    def forward(self, x, input_lengths):
        x = self.prenet(x) # [B, N, hidden_size]
        #x = x.transpose(1, 2)

        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True, enforce_sorted=False) #去除PAD，压紧成一维，从而消除PAD对RNN的影响

        self.lstm.flatten_parameters() # 把参数存放成连续块
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)
        outputs = self.layer_norm(outputs)
        
        return outputs  # [B,N,hidden_size]

class CTCDecoder(nn.Module):
    def __init__(self, hparams, vocab_size):
        super(CTCDecoder, self).__init__()
        self.hidden_size = hparams['hidden_size']
        self.num_layers = 2
        self.vocab_size = vocab_size

        self.ctcdecoder_lstm = nn.LSTM(self.hidden_size,
            int(self.hidden_size // 2), num_layers = self.num_layers,
                             bidirectional=True)

        self.ctc_linear = LinearNorm(self.hidden_size, vocab_size, bias=False,
                                       w_init_gain='tanh')

    def forward(self, x, input_lengths):
        '''
        x: [N, B, 2C]
        '''
        tgt_mask = get_mask_from_lengths(input_lengths) # [B,N]
        tgt_mask = tgt_mask.transpose(0,1).unsqueeze(2)
        input_lengths = input_lengths.cpu().numpy() # [B, N]
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, enforce_sorted=False) #去除PAD，压紧成一维，从而消除PAD对RNN的影响

        self.ctcdecoder_lstm.flatten_parameters() # 把参数存放成连续块
        x, _ = self.ctcdecoder_lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            x)

        outputs = F.dropout(F.relu(x), 0.2, self.training)
        logits = self.ctc_linear(outputs)
        log_prob = logits.log_softmax(2)
        return x, log_prob * tgt_mask# [N, B, hidden_size], [N, B, vocab_size]

class MELDecoder(nn.Module):
    def __init__(self, hparams):
        super(MELDecoder, self).__init__()
        self.hidden_size = hparams['hidden_size']
        self.num_layers = 2
        self.mel_bins = hparams['audio_num_mel_bins']

        self.meldecoder_lstm = nn.LSTM(self.hidden_size,
            int(self.hidden_size // 2), num_layers = self.num_layers,
                             bidirectional=True)

        self.mel_linear = LinearNorm(self.hidden_size, self.mel_bins, bias=False,
                                       w_init_gain='tanh')

    def forward(self, x, input_lengths):
        '''
        x: [N, B, hidden_size]
        '''
        input_lengths = input_lengths.cpu().numpy() # [B, N]
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, enforce_sorted=False) #去除PAD，压紧成一维，从而消除PAD对RNN的影响
        self.meldecoder_lstm.flatten_parameters() # 把参数存放成连续块
        x, _ = self.meldecoder_lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            x)

        outputs = F.dropout(F.relu(x), 0.2, self.training)
        outputs = self.mel_linear(x)
        return outputs  # [N, B, audio_num_mel_bins]

class PHNREC(nn.Module):
    def __init__(self, hparams):
        super(PHNREC, self).__init__()
        self.hparams = hparams
        self.hidden_size = hparams['hidden_size']
        self.text_encoder = build_phone_encoder(hparams['binary_data_dir'])
        self.melencoder = MelEncoder(hparams)
        self.ctcdecoder = CTCDecoder(hparams, self.text_encoder.vocab_size)
        self.meldecoder = MELDecoder(hparams)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.bottleneck_linear = LinearNorm(self.hidden_size * 2, \
                                    self.hidden_size, bias=False,
                                       w_init_gain='tanh')
        self.pitch_embed = Embedding(300, self.hidden_size, self.text_encoder.pad())
        self.spk_embed_proj = Linear(256, self.hidden_size, bias=True)

        self.register_buffer('spec_min', torch.FloatTensor([-12.0]))
        self.register_buffer('spec_max', torch.FloatTensor([0.0]))

    # 放缩到[-1,1]
    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min

    def forward(self, mels, mel_lengths, spk_embed, f0, uv):
        '''
        mels: [B, N, MB]
        mel_lengths: [B]
        spk_embed: [B, 256]
        f0: [B, N]
        '''
        mel_lengths = mel_lengths.data
        mels = mels.transpose(1, 2)
        mels = self.norm_spec(mels)

        f0_denorm = denorm_f0(f0, uv, self.hparams)
        pitch = f0_to_coarse(f0_denorm)  # start from 0, rescaling to [0, 255]
        pitch_embed = self.pitch_embed(pitch) # [B, N, hidden_size]
        spk_embed = self.spk_embed_proj(spk_embed)[:, None, :] # [B, 1, hidden_size]
        
        mel_hidden = self.melencoder(mels, mel_lengths) # [B,N,hidden_size]
        decoder_input = mel_hidden + spk_embed

        decoder_input = decoder_input.transpose(0,1)

        ctc_hidden, log_prob = self.ctcdecoder(decoder_input, mel_lengths) # [N,B,hidden_size], [N,B,vocab_size]
        bottleneck_hidden = torch.cat([decoder_input, ctc_hidden], dim = 2)
        bottleneck_hidden = self.bottleneck_linear(bottleneck_hidden)
        bottleneck_hidden = self.layer_norm(bottleneck_hidden)
        mel_output = self.meldecoder(bottleneck_hidden, mel_lengths) # [N,B,audio_num_mel_bins]

        output = {}
        output['mel_out'] = self.denorm_spec(mel_output).transpose(0, 1) # [B, N, audio_num_mel_bins]
        output['log_prob'] = log_prob # [N,B,vocab_size]

        return output 

class PhnRecTask(FastSpeech2Task):
    def __init__(self):
        super(PhnRecTask, self).__init__()
        # self.dataset_cls = MIDIDataset
        self.dataset_cls = FastSpeechDataset
        self._text_encoder = build_phone_encoder(hparams['binary_data_dir'])
        self.lsce = LabelSmoothingCrossEntropy(label_smooth=0.1, class_num=self._text_encoder.vocab_size)

    def build_tts_model(self):
        self.model = PHNREC(hparams)

    def build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, hparams['decay_steps'], gamma=0.6)

    def build_optimizer(self, model):
        self.optimizer = optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            weight_decay=hparams['weight_decay'])
        return optimizer

    def run_model(self, model, sample, return_output=False, infer=False):
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        txt_lengths = sample['txt_lengths']
        mels = sample['mels']  # [B, T_s, 80]
        mel_lengths = sample['mel_lengths']
        mel2ph = sample['mel2ph'] # [B, T_s]
        f0 = sample['f0'] # [B, T_s]
        uv = sample['uv'] # [B, T_s]

        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')

        output = model(mels=mels, mel_lengths=mel_lengths, spk_embed=spk_embed, f0=f0, uv=uv)

        losses = {}
        self.add_mel_loss(output['mel_out'], mels, losses)
        # todo
        self.add_ce_loss(txt_tokens, mel2ph, output['log_prob'], losses)

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
        losses['ce_loss'] = self.lsce(log_prob, targets)

    def validation_step(self, sample, batch_idx):
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        mel2ph = sample['mel2ph'] # [B, T_s]
        txt_tokens = F.pad(txt_tokens, [1, 0])
        targets = torch.gather(input=txt_tokens, dim=1, index=mel2ph) # [B, N]
        mask = targets > 0

        outputs = {}

        outputs['losses'] = {}
        outputs['losses'], model_out = self.run_model(self.model, sample, return_output=True, infer=False)
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']

        prediction = model_out['log_prob'].max(dim=2)[1]
        prediction = prediction.transpose(0, 1)
        bingo = prediction == targets
        accuracy = torch.sum(bingo * mask).true_divide(torch.sum(mask))
        outputs['accuracy'] = accuracy * sample['nsamples']

        outputs = utils.tensors_to_scalars(outputs)

        return outputs

    def _validation_end(self, outputs):
        all_losses_meter = {
            'total_loss': utils.AvgrageMeter(),
            'accuracy': utils.AvgrageMeter(),
        }
        for output in outputs:
            n = output['nsamples']
            for k, v in output['losses'].items():
                if k not in all_losses_meter:
                    all_losses_meter[k] = utils.AvgrageMeter()
                all_losses_meter[k].update(v, n)
            all_losses_meter['total_loss'].update(output['total_loss'], n)
            all_losses_meter['accuracy'].update(output['accuracy'], n)
        return {k: round(v.avg, 4) for k, v in all_losses_meter.items()}

    def test_step(self, sample, batch_idx):
        # 需要真实的f0给vocoder使用
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        txt_tokens = sample['txt_tokens']
        mel2ph, uv, f0 = None, None, None
        ref_mels = None
        energy = sample['energy']
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
                outputs = self.model(
                    txt_tokens, spk_embed=spk_embed, mel2ph=mel2ph, f0=f0, uv=uv, ref_mels=ref_mels, infer=True)
            sample['outputs'] = self.model.out2mel(outputs['mel_out'])
            sample['mel2ph_pred'] = outputs['mel2ph']
            if hparams.get('pe_enable') is not None and hparams['pe_enable']:
                sample['f0'] = self.pe(sample['mels'])['f0_denorm_pred']  # pe predict from GT mel
                sample['f0_pred'] = self.pe(sample['outputs'])['f0_denorm_pred']  # pe predict from Pred mel
            else:
                sample['f0'] = denorm_f0(sample['f0'], sample['uv'], hparams)
                sample['f0_pred'] = outputs.get('f0_denorm')
            return self.after_infer(sample)