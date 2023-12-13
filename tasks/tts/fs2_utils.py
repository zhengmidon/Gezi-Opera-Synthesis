import matplotlib

matplotlib.use('Agg')

import glob
import importlib
from utils.cwt import get_lf0_cwt
import os
import torch.optim
import torch.utils.data
from utils.indexed_datasets import IndexedDataset
from utils.pitch_utils import norm_interp_f0
import numpy as np
from tasks.base_task import BaseDataset
import torch
import torch.optim
import torch.utils.data
import utils
import torch.distributions
from utils.hparams import hparams
import random
from data_gen.tts.data_gen_utils import build_phone_encoder
import json


class FastSpeechDataset(BaseDataset):
    def __init__(self, prefix, shuffle=False):
        super().__init__(shuffle)
        self.data_dir = hparams['binary_data_dir']
        self.prefix = prefix
        self.hparams = hparams
        self.sizes = np.load(f'{self.data_dir}/{self.prefix}_lengths.npy')
        self.indexed_ds = None
        # self.name2spk_id={}

        # pitch stats
        f0_stats_fn = f'{self.data_dir}/train_f0s_mean_std.npy'
        if os.path.exists(f0_stats_fn):
            hparams['f0_mean'], hparams['f0_std'] = self.f0_mean, self.f0_std = np.load(f0_stats_fn)
            hparams['f0_mean'] = float(hparams['f0_mean'])
            hparams['f0_std'] = float(hparams['f0_std'])
        else:
            hparams['f0_mean'], hparams['f0_std'] = self.f0_mean, self.f0_std = None, None

        if prefix == 'test':
            if hparams['test_input_dir'] != '':
                self.indexed_ds, self.sizes = self.load_test_inputs(hparams['test_input_dir'])
            else:
                if hparams['num_test_samples'] > 0:
                    self.avail_idxs = list(range(hparams['num_test_samples'])) + hparams['test_ids']
                    self.sizes = [self.sizes[i] for i in self.avail_idxs]

        if hparams['pitch_type'] == 'cwt':
            _, hparams['cwt_scales'] = get_lf0_cwt(np.ones(10))

    def _get_item(self, index):
        if hasattr(self, 'avail_idxs') and self.avail_idxs is not None:
            index = self.avail_idxs[index]
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f'{self.data_dir}/{self.prefix}') # 从.data和.idx中读取数据
        return self.indexed_ds[index]

    def __getitem__(self, index):
        hparams = self.hparams
        item = self._get_item(index)
        max_frames = hparams['max_frames']
        spec = torch.Tensor(item['mel'])[:max_frames]
        energy = (spec.exp() ** 2).sum(-1).sqrt() # 均方能量
        mel2ph = torch.LongTensor(item['mel2ph'])[:max_frames] if 'mel2ph' in item else None # 梅尔谱的每一帧对应的音素序列的序号
        f0, uv = norm_interp_f0(item["f0"][:max_frames], hparams) # 标准化f0, floattensor
        phone = torch.LongTensor(item['phone'][:hparams['max_input_tokens']])
        pitch = torch.LongTensor(item.get("pitch"))[:max_frames]
        # print(item.keys(), item['mel'].shape, spec.shape)
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
            cwt_spec = torch.Tensor(item['cwt_spec'])[:max_frames]
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
        if len(samples) == 0:
            return {}
        if self.hparams['use_data_aug']:
            samples = self.random_replace_data_augment(samples, self.hparams['re_prob'], self.hparams['re_rate'])

        id = torch.LongTensor([s['id'] for s in samples])
        item_names = [s['item_name'] for s in samples]
        text = [s['text'] for s in samples]
        txt_tokens = utils.collate_1d([s['txt_token'] for s in samples], 0)
        f0 = utils.collate_1d([s['f0'] for s in samples], 0.0)
        pitch = utils.collate_1d([s['pitch'] for s in samples])
        uv = utils.collate_1d([s['uv'] for s in samples])
        energy = utils.collate_1d([s['energy'] for s in samples], 0.0)
        mel2ph = utils.collate_1d([s['mel2ph'] for s in samples], 0.0) \
            if samples[0]['mel2ph'] is not None else None
        mels = utils.collate_2d([s['mel'] for s in samples], 0.0)
        txt_lengths = torch.LongTensor([s['txt_token'].numel() for s in samples])
        mel_lengths = torch.LongTensor([s['mel'].shape[0] for s in samples])

        batch = {
            'id': id,
            'item_name': item_names,
            'nsamples': len(samples),
            'text': text,
            'txt_tokens': txt_tokens, # encoded phonemes, [B, T]
            'txt_lengths': txt_lengths,
            'mels': mels, # [B, N, MB]
            'mel_lengths': mel_lengths,
            'mel2ph': mel2ph,
            'energy': energy, #[B, N]
            'pitch': pitch,
            'f0': f0,
            'uv': uv,
        }

        if self.hparams['use_spk_embed']:
            spk_embed = torch.stack([s['spk_embed'] for s in samples])
            batch['spk_embed'] = spk_embed
        if self.hparams['use_spk_id']:
            spk_ids = torch.LongTensor([s['spk_id'] for s in samples])
            batch['spk_ids'] = spk_ids
        if self.hparams['pitch_type'] == 'cwt':
            cwt_spec = utils.collate_2d([s['cwt_spec'] for s in samples])
            f0_mean = torch.Tensor([s['f0_mean'] for s in samples])
            f0_std = torch.Tensor([s['f0_std'] for s in samples])
            batch.update({'cwt_spec': cwt_spec, 'f0_mean': f0_mean, 'f0_std': f0_std})
        elif self.hparams['pitch_type'] == 'ph':
            batch['f0'] = utils.collate_1d([s['f0_ph'] for s in samples])

        return batch

    def load_test_inputs(self, test_input_dir, spk_id=0):
        inp_wav_paths = glob.glob(f'{test_input_dir}/*.wav') + glob.glob(f'{test_input_dir}/*.mp3')
        sizes = []
        items = []

        binarizer_cls = hparams.get("binarizer_cls", 'data_gen.tts.base_binarizerr.BaseBinarizer')
        pkg = ".".join(binarizer_cls.split(".")[:-1])
        cls_name = binarizer_cls.split(".")[-1]
        binarizer_cls = getattr(importlib.import_module(pkg), cls_name)
        binarization_args = hparams['binarization_args']

        for wav_fn in inp_wav_paths:
            item_name = os.path.basename(wav_fn)
            ph = txt = tg_fn = ''
            wav_fn = wav_fn
            encoder = None
            item = binarizer_cls.process_item(item_name, ph, txt, tg_fn, wav_fn, spk_id, encoder, binarization_args)
            items.append(item)
            sizes.append(item['len'])
        return items, sizes

    def random_replace_data_augment(self, samples, re_prob=0.5, re_rate=0.3, smooth=10):
        """
        samples: list of sample dicts
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
        }
        """
        # torch.set_printoptions(threshold=3000)

        SHENGMU = ['pʰ', 'm', 'b', 't', 'n', 'l', 'k', 'g', 'ŋ', 'j', 'p', 'tʰ', 'kʰ', 'ts', 'tsʰ', 's', 'h']
        random.seed(hparams['seed'])
        phoneme_encoder = build_phone_encoder(hparams['binary_data_dir'])
        with open(os.path.join(self.data_dir,"phoneme_frequency.json"), encoding="utf-8") as f:
            phoneme_fre_dict = json.load(f)
        
        for sample in samples:
            if random.random() > re_prob:
                token_list = sample['txt_token'].tolist()
                # m2p = sample['mel2ph']
                # print(f'BEFORE tokens {phoneme_encoder.decode(token_list)}, mel2ph {m2p}, frames {m2p.shape[0]}')
                re_num_tokens = int(len(token_list) * re_rate + 0.5)
                # print(f"replacing {re_num_tokens} positions in {phoneme_encoder.decode(token_list)}")
                done_tokens = 0
                while done_tokens < re_num_tokens:
                    cand_token_idx = random.choice(range(len(token_list)))
                    cand_token = phoneme_encoder.decode([token_list[cand_token_idx]])
                    # print(f"candidate token {cand_token}")

                    if cand_token == 'sp':
                        done_tokens += 1
                        continue
                    elif cand_token in SHENGMU:
                        found_target = False 
                        while not found_target:
                            t_sample = random.choice(samples)
                            t_token_list = t_sample['txt_token'].tolist()
                            # t_token_idx = random.choice(range(len(t_token_list)))
                            # t_token = phoneme_encoder.decode([t_token_list[t_token_idx]])
                            t_phoneme_tokens = phoneme_encoder.decode(t_token_list).split(' ')
                            t_phoneme_weights = [1 / (phoneme_fre_dict[token] + smooth) if token in phoneme_fre_dict.keys() \
                                                else 1 / (1 + smooth) for token in t_phoneme_tokens]
                            t_token = random.choices(t_phoneme_tokens, weights=t_phoneme_weights, k=1)[0] # key step
                            t_token_idx = t_phoneme_tokens.index(t_token)
                            if t_token == 'sp' or t_token not in SHENGMU:
                                continue
                            else:
                                # print(f"found a target token {t_token} to replace {cand_token} in {phoneme_encoder.decode(t_token_list)}")
                                # 关键：通过mel2ph来找到目标音素对应的帧长度, former和latter是要保留的帧, target是要替换的帧
                                cand_former_frames, cand_target_frames, cand_latter_frames = \
                                sample['mel2ph'] < (cand_token_idx + 1), sample['mel2ph'] == (cand_token_idx + 1), sample['mel2ph'] > (cand_token_idx + 1)
                                t_target_frames = t_sample['mel2ph'] == (t_token_idx + 1)

                                sample['txt_token'][cand_token_idx] = t_token_list[t_token_idx]

                                sample['mel'] = torch.cat([sample['mel'][cand_former_frames], \
                                    t_sample['mel'][t_target_frames], sample['mel'][cand_latter_frames]], dim = 0)
                                sample['pitch'] = torch.cat([sample['pitch'][cand_former_frames], \
                                    t_sample['pitch'][t_target_frames], sample['pitch'][cand_latter_frames]], dim = 0)
                                sample['f0'] = torch.cat([sample['f0'][cand_former_frames], \
                                    t_sample['f0'][t_target_frames], sample['f0'][cand_latter_frames]], dim = 0)
                                sample['uv'] = torch.cat([sample['uv'][cand_former_frames], \
                                    t_sample['uv'][t_target_frames], sample['uv'][cand_latter_frames]], dim = 0)

                                t_mel2ph_seg = torch.zeros_like(t_sample['mel2ph'][t_target_frames]).fill_(cand_token_idx + 1)
                                sample['mel2ph'] = torch.cat([sample['mel2ph'][cand_former_frames], \
                                    t_mel2ph_seg, sample['mel2ph'][cand_latter_frames]], dim = 0)
                                
                                assert sample['mel'].shape[0] == sample['pitch'].shape[0] == sample['f0'].shape[0] == \
                                sample['uv'].shape[0] == sample['mel2ph'].shape[0]

                                found_target = True
                                done_tokens += 1
                    else:
                        found_target = False 
                        while not found_target:
                            t_sample = random.choice(samples)
                            t_token_list = t_sample['txt_token'].tolist()
                            t_token_idx = random.choice(range(len(t_token_list)))
                            t_token = phoneme_encoder.decode([t_token_list[t_token_idx]])
                            if t_token == 'sp' or t_token in SHENGMU:
                                continue
                            else:
                                # print(f"found a target token {t_token} to replace {cand_token} in {phoneme_encoder.decode(t_token_list)}")
                                sample['txt_token'][cand_token_idx] = t_token_list[t_token_idx]
                                cand_former_frames, cand_target_frames, cand_latter_frames = \
                                sample['mel2ph'] < (cand_token_idx + 1), sample['mel2ph'] == (cand_token_idx + 1), sample['mel2ph'] > (cand_token_idx + 1)
                                t_target_frames = t_sample['mel2ph'] == (t_token_idx + 1)

                                sample['mel'] = torch.cat([sample['mel'][cand_former_frames], \
                                    t_sample['mel'][t_target_frames], sample['mel'][cand_latter_frames]], dim = 0)
                                sample['pitch'] = torch.cat([sample['pitch'][cand_former_frames], \
                                    t_sample['pitch'][t_target_frames], sample['pitch'][cand_latter_frames]], dim = 0)
                                sample['f0'] = torch.cat([sample['f0'][cand_former_frames], \
                                    t_sample['f0'][t_target_frames], sample['f0'][cand_latter_frames]], dim = 0)
                                sample['uv'] = torch.cat([sample['uv'][cand_former_frames], \
                                    t_sample['uv'][t_target_frames], sample['uv'][cand_latter_frames]], dim = 0)
                                t_mel2ph_seg = torch.zeros_like(t_sample['mel2ph'][t_target_frames]).fill_(cand_token_idx + 1)
                                sample['mel2ph'] = torch.cat([sample['mel2ph'][cand_former_frames], \
                                    t_mel2ph_seg, sample['mel2ph'][cand_latter_frames]], dim = 0)
                                found_target = True
                                done_tokens += 1
                sample['energy'] = (sample['mel'].exp() ** 2).sum(-1).sqrt() # 均方能量
                # end_token_list = sample['txt_token'].tolist()
                # end_m2p = sample['mel2ph']
                # print(f'AFTER tokens {phoneme_encoder.decode(end_token_list)}, mel2ph {end_m2p}, frames {end_m2p.shape[0]}')
            else:
                continue
        return samples

