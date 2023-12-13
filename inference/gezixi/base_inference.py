import os
import sys
sys.path.append('../../')

import torch
import numpy as np
from modules.hifigan.hifigan import HifiGanGenerator
from vocoders.hifigan import HifiGAN
from inference.svs.opencpop.map import cpop_pinyin2ph_func

from utils import load_ckpt
from utils.hparams import set_hparams, hparams
from utils.text_encoder import TokenTextEncoder
from pypinyin import pinyin, lazy_pinyin, Style
import librosa
import glob
import re
import json
from inference.gezixi.infer_utils import curate_text



class BaseSVSInfer:
    def __init__(self, hparams, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hparams = hparams
        self.device = device

        phone_list = ['sp', 'i', 'a', 'u', 's', 'l', 'oo', 'k', 'ts', 'ng', 'h', 'e', 't',
                     'ing', 'ong', 'b', 'p', 'ai', 'an', 'ah', 'tsh', 'o', 'in', 'ua', 'g', 
                     'kh', 'ut', 'it', 'iu', 'm', 'n', 'au', 'ok', 'iong', 'th', 'un', 'ui', 
                     'im', 'ann', 'ian', 'eh', 'ia', 'ue', 'uan', 'ang', 'iann', 'ik', 'inn', 
                     'iau', 'ioo', 'uann', 'ph', 'am', 'at', 'iat', 'iok', 'iunn', 'io', 'ak', 
                     'iam', 'ip', 'ap', 'uat', 'iah', 'oh', 'ih', 'uah', 'ioh', 'uai', 'iang', 
                     'iap', 'ainn', 'uainn', 'enn', 'uih', 'ueh', 'aih', 'uang', 'iannh']

        self.ph_encoder = self.build_phone_encoder(hparams['binary_data_dir'])
        self.pinyin2phs = cpop_pinyin2ph_func()
        self.spk_map = {'姚琼男': 0, '张蓉鑫': 1, '曾宝珠': 2, '林姗姗': 3, '郭少鹏':4}
        self.UV = ['p', 'th', 'kh', 'ts', 'tsh', 's', 'h']

        self.model = self.build_model()
        self.model.eval()
        self.model.to(self.device)
        self.vocoder = self.build_vocoder()
        self.vocoder.eval()
        self.vocoder.to(self.device)

    def build_model(self):
        raise NotImplementedError

    def forward_model(self, inp):
        raise NotImplementedError

    def build_phone_encoder(self, data_dir):
        phone_list_file = os.path.join(data_dir, 'phone_set.json')

        phone_list = json.load(open(phone_list_file))
        return TokenTextEncoder(None, vocab_list=phone_list, replace_oov=',')

    def build_vocoder(self):
        base_dir = hparams['vocoder_ckpt']
        config_path = f'{base_dir}/config.yaml'
        ckpt = sorted(glob.glob(f'{base_dir}/model_ckpt_steps_*.ckpt'), key=
        lambda x: int(re.findall(f'{base_dir}/model_ckpt_steps_(\d+).ckpt', x)[0]))[-1]
        print('| load HifiGAN: ', ckpt)
        ckpt_dict = torch.load(ckpt, map_location="cpu")
        config = set_hparams(config_path, global_hparams=False)
        state = ckpt_dict["state_dict"]["model_gen"]
        vocoder = HifiGanGenerator(config)
        vocoder.load_state_dict(state, strict=True)
        vocoder.remove_weight_norm()
        vocoder = vocoder.eval().to(self.device)
        return vocoder

    def run_vocoder(self, c, **kwargs):
        c = c.transpose(2, 1)  # [B, 80, T]
        f0 = kwargs.get('f0')  # [B, T]
        if f0 is not None and hparams.get('use_nsf'):
            # f0 = torch.FloatTensor(f0).to(self.device)
            y = self.vocoder(c, f0).view(-1)
        else:
            y = self.vocoder(c).view(-1)
            # [T]
        return y[None]

    def preprocess_word_level_input(self, inp):
        # Text
        text_raw = inp['text'].strip().lower().split(' ')
        text_raw_list = [i for i in text_raw if i != ''] # list like ['sp', '此', '去', '-', '-', '云', '南', 'sp', '路', '-', '千', '-', '-', '-', '里', '-', 'sp']
        # convert simplified Chinese to tl and extends the slur(marked by '-') by yunmu
        ph_per_word_lst = curate_text(text_raw_list) # list like ['sp', 'tsh|u', 'kh|i', 'i', 'i', 'h|un', 'l|am', 'sp', 'l|oo', 'oo', 'tsh|ian', 'an', 'an', 'an', 'l|i', 'i', 'sp']
        assert len(text_raw_list) == len(ph_per_word_lst)

        # Note
        note_per_word_lst = [x.strip() for x in inp['notes'].split(' ') if x.strip() != '']
        # Duration
        mididur_per_word_lst = [x.strip() for x in inp['notes_duration'].split(' ') if x.strip() != '']

        if len(note_per_word_lst) == len(ph_per_word_lst) == len(mididur_per_word_lst):
            print('Pass word-notes check.')
        else:
            print('The number of words does\'t match the number of notes\' windows. ',
                  )
            print(ph_per_word_lst, note_per_word_lst, mididur_per_word_lst)
            print(len(ph_per_word_lst), len(note_per_word_lst), len(mididur_per_word_lst))
            return None

        note_lst = []
        ph_lst = []
        midi_dur_lst = []
        is_slur = []
        uv_shengmu = []
        mel2ph = []
        ph_idx = 1
        shengmu_mean_dur_dict = json.load(open('inference/gezixi/shengmu_mean_dur.json')) # we use statistical mean shengmu duration but the predicted duration
        for idx, ph_per_word in enumerate(ph_per_word_lst):
            p_dur = float(mididur_per_word_lst[idx])
            shengmu_dur = 0.
            pys = ph_per_word.split('|')
            for py in pys:
                ph_lst.append(py)
                note_lst.append(note_per_word_lst[idx])
                if text_raw_list[idx] == '-':
                    is_slur.append(1)
                else:
                    is_slur.append(0)
                if py in self.UV:
                    uv_shengmu.append(1)
                else:
                    uv_shengmu.append(0)
                if py in shengmu_mean_dur_dict.keys():
                    shengmu_dur = shengmu_mean_dur_dict[py]
                    if shengmu_dur >= p_dur:
                        shengmu_dur = p_dur / 2
                    midi_dur_lst.append(str(shengmu_dur))
                    mel2ph.extend([ph_idx] * int(shengmu_dur * hparams['audio_sample_rate'] / hparams['hop_size'] + 0.5))
                    ph_idx += 1
                else:
                    yunmu_dur = p_dur - shengmu_dur
                    assert yunmu_dur > 0
                    midi_dur_lst.append(str(yunmu_dur))
                    mel2ph.extend([ph_idx] * int(yunmu_dur * hparams['audio_sample_rate'] / hparams['hop_size'] + 0.5))
                    ph_idx += 1

        ph_seq = ' '.join(ph_lst)
        print(f"audio length: {len(mel2ph)}")

        if len(ph_lst) == len(note_lst) == len(midi_dur_lst) == len(uv_shengmu):
            print('Pass word-notes check.')
        else:
            print('The number of words does\'t match the number of notes\' windows. ',
                  'You should split the note(s) for each word by | mark.')
            return None
        return ph_seq, note_lst, midi_dur_lst, is_slur, uv_shengmu, mel2ph


    def preprocess_input(self, inp, input_type='word'):
        """
        :param inp: {'text': str, 'notes': str, 'notes_duration': str, 'item_name': (str, optional), 'spk_name': (str, optional)}
        :return:
        """
        spk_name = inp.get('spk_name', '姚琼男')

        # single spk
        spk_id = [self.spk_map[spk_name]]

        # get ph seq, note lst, midi dur lst, is slur lst.
        if input_type == 'word':
            ret = self.preprocess_word_level_input(inp)
        else:
            print('Invalid input type.')
            return None

        if ret:
            ph_seq, note_lst, midi_dur_lst, is_slur, uv_shengmu, mel2ph = ret
        else:
            print('==========> Preprocess_word_level or phone_level input wrong.')
            return None

        # convert note lst to midi id; convert note dur lst to midi duration
        try:
            midis = [librosa.note_to_midi(x) if x != 'sp' else 1
                     for x in note_lst]
            midi_dur_lst = [float(x) for x in midi_dur_lst]
        except Exception as e:
            print(e)
            print('Invalid Input Type.')
            return None

        ph_token = self.ph_encoder.encode(ph_seq)
        item = {'item_name': "inference item", 'text': inp['text'], 'ph': ph_seq, 'spk_id': spk_id,
                'ph_token': ph_token, 'pitch_midi': np.asarray(midis), 'midi_dur': np.asarray(midi_dur_lst),
                'is_slur': np.asarray(is_slur), 'uv_shengmu': np.asarray(uv_shengmu), 'mel2ph': np.asarray(mel2ph)}
        item['ph_len'] = len(item['ph_token'])
        return item

    def input_to_batch(self, item):
        item_names = [item['item_name']] 
        text = [item['text']]
        ph = [item['ph']]
        txt_tokens = torch.LongTensor(item['ph_token'])[None, :].to(self.device)
        txt_lengths = torch.LongTensor([txt_tokens.shape[1]]).to(self.device)
        spk_ids = torch.LongTensor(item['spk_id']).to(self.device)

        hparams['max_frames'] = 12000
        pitch_midi = torch.LongTensor(item['pitch_midi'])[None, :hparams['max_frames']].to(self.device)
        midi_dur = torch.FloatTensor(item['midi_dur'])[None, :hparams['max_frames']].to(self.device)
        is_slur = torch.LongTensor(item['is_slur'])[None, :hparams['max_frames']].to(self.device)
        uv_shengmu = torch.LongTensor(item['uv_shengmu'])[None, :hparams['max_frames']].to(self.device)
        mel2ph = torch.LongTensor(item['mel2ph'])[None, :hparams['max_frames']].to(self.device)

        batch = {
            'item_name': item_names,
            'text': text,
            'ph': ph,
            'txt_tokens': txt_tokens,
            'txt_lengths': txt_lengths,
            'spk_ids': spk_ids,
            'pitch_midi': pitch_midi,
            'midi_dur': midi_dur,
            'is_slur': is_slur,
            'uv_shengmu': uv_shengmu,
            'mel2ph': mel2ph,
        }
        return batch

    def postprocess_output(self, output):
        return output

    def infer_once(self, inp):
        inp = self.preprocess_input(inp, input_type=inp['input_type'] if inp.get('input_type') else 'word')
        np.set_printoptions(threshold=50000)
        print(f"Lyrics: {inp['text']}\nTLs: {inp['ph']} \nSingerid: {inp['spk_id']}")
        output = self.forward_model(inp)
        output = self.postprocess_output(output)
        return output

    @classmethod
    def example_run(cls, inp):
        from utils.audio import save_wav
        set_hparams(print_hparams=False)
        infer_ins = cls(hparams)
        out = infer_ins.infer_once(inp)
        os.makedirs('infer_out', exist_ok=True)
        save_wav(out, f'infer_out/example_out.wav', hparams['audio_sample_rate'])

if __name__ == '__main__':
    # debug
    set_hparams()
    a = BaseSVSInfer(hparams)
    item = a.preprocess_input({'text': '你 说 你 不  懂 sp 为 何 在 这 时 牵 手 - - sp',
                        'notes': 'D#4 D#4 D#4 D#4 sp D#4 D4 D4 D4 D#4 F4 D#4 D4 C4 B3 sp',
                        'notes_duration': '0.113740 0.329060 0.287950 0.133480 0.150900 0.484730 0.242010 0.180820 0.343570 0.152050 0.266720 0.280310 0.633300 0.21 0.13 0.444590'
                        })
    print(item)
    # b = {
    #     'text': '小酒窝长睫毛AP是你最美的记号',
    #     'notes': 'C#4/Db4 | F#4/Gb4 | G#4/Ab4 | A#4/Bb4 F#4/Gb4 | F#4/Gb4 C#4/Db4 | C#4/Db4 | rest | C#4/Db4 | A#4/Bb4 | G#4/Ab4 | A#4/Bb4 | G#4/Ab4 | F4 | C#4/Db4',
    #     'notes_duration': '0.407140 | 0.376190 | 0.242180 | 0.509550 0.183420 | 0.315400 0.235020 | 0.361660 | 0.223070 | 0.377270 | 0.340550 | 0.299620 | 0.344510 | 0.283770 | 0.323390 | 0.360340'
    # }
    # c = {
    #     'text': '小酒窝长睫毛AP是你最美的记号',
    #     'ph_seq': 'x iao j iu w o ch ang ang j ie ie m ao AP sh i n i z ui m ei d e j i h ao',
    #     'note_seq': 'C#4/Db4 C#4/Db4 F#4/Gb4 F#4/Gb4 G#4/Ab4 G#4/Ab4 A#4/Bb4 A#4/Bb4 F#4/Gb4 F#4/Gb4 F#4/Gb4 C#4/Db4 C#4/Db4 C#4/Db4 rest C#4/Db4 C#4/Db4 A#4/Bb4 A#4/Bb4 G#4/Ab4 G#4/Ab4 A#4/Bb4 A#4/Bb4 G#4/Ab4 G#4/Ab4 F4 F4 C#4/Db4 C#4/Db4',
    #     'note_dur_seq': '0.407140 0.407140 0.376190 0.376190 0.242180 0.242180 0.509550 0.509550 0.183420 0.315400 0.315400 0.235020 0.361660 0.361660 0.223070 0.377270 0.377270 0.340550 0.340550 0.299620 0.299620 0.344510 0.344510 0.283770 0.283770 0.323390 0.323390 0.360340 0.360340',
    #     'is_slur_seq': '0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0'
    # }  # input like Opencpop dataset.
    # a.preprocess_input(b)
    # a.preprocess_input(c, input_type='phoneme')