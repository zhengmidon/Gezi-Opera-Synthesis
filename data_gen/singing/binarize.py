import os
import random
from copy import deepcopy
import pandas as pd
import logging
from tqdm import tqdm
import json
import glob
import re
from resemblyzer import VoiceEncoder
import traceback
import numpy as np
import pretty_midi
import librosa
from scipy.interpolate import interp1d
import torch
from textgrid import TextGrid

from utils.hparams import hparams
from data_gen.tts.data_gen_utils import build_phone_encoder, get_pitch
from utils.pitch_utils import f0_to_coarse
from data_gen.tts.base_binarizer import BaseBinarizer, BinarizationError
from data_gen.tts.binarizer_zh import ZhBinarizer
from data_gen.tts.txt_processors.zh_g2pM import ALL_YUNMU
from vocoders.base_vocoder import VOCODERS
from data_gen.singing.phn_map import PHN_MAP, make_phn_map_dict


class SingingBinarizer(BaseBinarizer):
    def __init__(self, processed_data_dir=None):
        if processed_data_dir is None:
            processed_data_dir = hparams['processed_data_dir']
        self.processed_data_dirs = processed_data_dir.split(",")
        self.binarization_args = hparams['binarization_args']
        self.pre_align_args = hparams['pre_align_args']
        self.item2txt = {}
        self.item2ph = {}
        self.item2wavfn = {}
        self.item2f0fn = {}
        self.item2tgfn = {}
        self.item2spk = {}

    def split_train_test_set(self, item_names):
        item_names = deepcopy(item_names)
        test_item_names = [x for x in item_names if any([ts in x for ts in hparams['test_prefixes']])]
        train_item_names = [x for x in item_names if x not in set(test_item_names)]
        logging.info("train {}".format(len(train_item_names)))
        logging.info("test {}".format(len(test_item_names)))
        return train_item_names, test_item_names
        
    # 加载数据到字典
    def load_meta_data(self):
        for ds_id, processed_data_dir in enumerate(self.processed_data_dirs):
            wav_suffix = '_wf0.wav'
            txt_suffix = '.txt'
            ph_suffix = '_ph.txt'
            tg_suffix = '.TextGrid'
            all_wav_pieces = glob.glob(f'{processed_data_dir}/*/*{wav_suffix}')

            for piece_path in all_wav_pieces:
                item_name = raw_item_name = piece_path[len(processed_data_dir)+1:].replace('/', '-')[:-len(wav_suffix)]
                if len(self.processed_data_dirs) > 1:
                    item_name = f'ds{ds_id}_{item_name}'
                self.item2txt[item_name] = open(f'{piece_path.replace(wav_suffix, txt_suffix)}').readline()
                self.item2ph[item_name] = open(f'{piece_path.replace(wav_suffix, ph_suffix)}').readline()
                self.item2wavfn[item_name] = piece_path

                self.item2spk[item_name] = re.split('-|#', piece_path.split('/')[-2])[0] # popcs
                if len(self.processed_data_dirs) > 1:
                    self.item2spk[item_name] = f"ds{ds_id}_{self.item2spk[item_name]}"
                self.item2tgfn[item_name] = piece_path.replace(wav_suffix, tg_suffix)
        print('spkers: ', set(self.item2spk.values()))
        self.item_names = sorted(list(self.item2txt.keys()))
        if self.binarization_args['shuffle']:
            random.seed(1234)
            random.shuffle(self.item_names)
        self._train_item_names, self._test_item_names = self.split_train_test_set(self.item_names)

    @property
    def train_item_names(self):
        return self._train_item_names

    @property
    def valid_item_names(self):
        return self._test_item_names

    @property
    def test_item_names(self):
        return self._test_item_names
    # 核心函数
    def process(self):
        self.load_meta_data()
        os.makedirs(hparams['binary_data_dir'], exist_ok=True)
        self.spk_map = self.build_spk_map()
        print("| spk_map: ", self.spk_map)
        spk_map_fn = f"{hparams['binary_data_dir']}/spk_map.json" # 写入spk_map.json
        json.dump(self.spk_map, open(spk_map_fn, 'w'))

        self.phone_encoder = self._phone_encoder() # 把音素编码成数字
        self.process_data('valid')
        self.process_data('test')
        self.process_data('train')

    def _phone_encoder(self):
        ph_set_fn = f"{hparams['binary_data_dir']}/phone_set.json" # 写入phone_set.json
        ph_set = []
        if hparams['reset_phone_dict'] or not os.path.exists(ph_set_fn):
            for ph_sent in self.item2ph.values():
                ph_set += ph_sent.split(' ')
            ph_set = sorted(set(ph_set)) # set函数去除重复元素
            json.dump(ph_set, open(ph_set_fn, 'w'))
            print("| Build phone set: ", ph_set)
        else:
            ph_set = json.load(open(ph_set_fn, 'r'))
            print("| Load phone set: ", ph_set)
        return build_phone_encoder(hparams['binary_data_dir'])

    # @staticmethod
    # def get_pitch(wav_fn, spec, res):
    #     wav_suffix = '_wf0.wav'
    #     f0_suffix = '_f0.npy'
    #     f0fn = wav_fn.replace(wav_suffix, f0_suffix)
    #     pitch_info = np.load(f0fn)
    #     f0 = [x[1] for x in pitch_info]
    #     spec_x_coor = np.arange(0, 1, 1 / len(spec))[:len(spec)]
    #     f0_x_coor = np.arange(0, 1, 1 / len(f0))[:len(f0)]
    #     f0 = interp1d(f0_x_coor, f0, 'nearest', fill_value='extrapolate')(spec_x_coor)[:len(spec)]
    #     # f0_x_coor = np.arange(0, 1, 1 / len(f0))
    #     # f0_x_coor[-1] = 1
    #     # f0 = interp1d(f0_x_coor, f0, 'nearest')(spec_x_coor)[:len(spec)]
    #     if sum(f0) == 0:
    #         raise BinarizationError("Empty f0")
    #     assert len(f0) == len(spec), (len(f0), len(spec))
    #     pitch_coarse = f0_to_coarse(f0)
    #
    #     # vis f0
    #     # import matplotlib.pyplot as plt
    #     # from textgrid import TextGrid
    #     # tg_fn = wav_fn.replace(wav_suffix, '.TextGrid')
    #     # fig = plt.figure(figsize=(12, 6))
    #     # plt.pcolor(spec.T, vmin=-5, vmax=0)
    #     # ax = plt.gca()
    #     # ax2 = ax.twinx()
    #     # ax2.plot(f0, color='red')
    #     # ax2.set_ylim(0, 800)
    #     # itvs = TextGrid.fromFile(tg_fn)[0]
    #     # for itv in itvs:
    #     #     x = itv.maxTime * hparams['audio_sample_rate'] / hparams['hop_size']
    #     #     plt.vlines(x=x, ymin=0, ymax=80, color='black')
    #     #     plt.text(x=x, y=20, s=itv.mark, color='black')
    #     # plt.savefig('tmp/20211229_singing_plots_test.png')
    #
    #     res['f0'] = f0
    #     res['pitch'] = pitch_coarse

    # 核心函数
    @classmethod
    def process_item(cls, item_name, ph, txt, tg_fn, wav_fn, spk_id, encoder, binarization_args):
        if hparams['vocoder'] in VOCODERS:
            # 获取音频数组和梅尔谱图
            wav, mel = VOCODERS[hparams['vocoder']].wav2spec(wav_fn) # wav2spec是staticmethod
        else:
            wav, mel = VOCODERS[hparams['vocoder'].split('.')[-1]].wav2spec(wav_fn)
        res = {
            'item_name': item_name, 'txt': txt, 'ph': ph, 'mel': mel, 'wav': wav, 'wav_fn': wav_fn,
            'sec': len(wav) / hparams['audio_sample_rate'], 'len': mel.shape[0], 'spk_id': spk_id
        }
        try:
            if binarization_args['with_f0']:
                # cls.get_pitch(wav_fn, mel, res)
                cls.get_pitch(wav, mel, res)
            if binarization_args['with_txt']:
                try:
                    # print(ph)
                    phone_encoded = res['phone'] = encoder.encode(ph) # 对音素进行编码
                except:
                    traceback.print_exc()
                    raise BinarizationError(f"Empty phoneme")
                if binarization_args['with_align']:
                    cls.get_align(tg_fn, ph, mel, phone_encoded, res)
        except BinarizationError as e:
            print(f"| Skip item ({e}). item_name: {item_name}, wav_fn: {wav_fn}")
            return None
        return res


class MidiSingingBinarizer(SingingBinarizer):
    item2midi = {}
    item2midi_dur = {}
    item2is_slur = {}
    item2ph_durs = {}
    item2wdb = {}

    def load_meta_data(self):
        for ds_id, processed_data_dir in enumerate(self.processed_data_dirs):
            meta_midi = json.load(open(os.path.join(processed_data_dir, 'meta.json')))   # [list of dict]

            for song_item in meta_midi:
                item_name = raw_item_name = song_item['item_name']
                if len(self.processed_data_dirs) > 1:
                    item_name = f'ds{ds_id}_{item_name}'
                self.item2wavfn[item_name] = song_item['wav_fn']
                self.item2txt[item_name] = song_item['txt']

                self.item2ph[item_name] = ' '.join(song_item['phs'])
                self.item2wdb[item_name] = [1 if x in ALL_YUNMU + ['AP', 'SP', '<SIL>'] else 0 for x in song_item['phs']]
                self.item2ph_durs[item_name] = song_item['ph_dur']

                self.item2midi[item_name] = song_item['notes']
                self.item2midi_dur[item_name] = song_item['notes_dur']
                self.item2is_slur[item_name] = song_item['is_slur']
                self.item2spk[item_name] = 'pop-cs'
                if len(self.processed_data_dirs) > 1:
                    self.item2spk[item_name] = f"ds{ds_id}_{self.item2spk[item_name]}"

        print('spkers: ', set(self.item2spk.values()))
        self.item_names = sorted(list(self.item2txt.keys()))
        if self.binarization_args['shuffle']:
            random.seed(1234)
            random.shuffle(self.item_names)
        self._train_item_names, self._test_item_names = self.split_train_test_set(self.item_names)

    @staticmethod
    def get_pitch(wav_fn, wav, spec, ph, res):
        wav_suffix = '.wav'
        # midi_suffix = '.mid'
        wav_dir = 'wavs'
        f0_dir = 'f0'

        item_name = '/'.join(os.path.splitext(wav_fn)[0].split('/')[-2:]).replace('_wf0', '')
        res['pitch_midi'] = np.asarray(MidiSingingBinarizer.item2midi[item_name])
        res['midi_dur'] = np.asarray(MidiSingingBinarizer.item2midi_dur[item_name])
        res['is_slur'] = np.asarray(MidiSingingBinarizer.item2is_slur[item_name])
        res['word_boundary'] = np.asarray(MidiSingingBinarizer.item2wdb[item_name])
        assert res['pitch_midi'].shape == res['midi_dur'].shape == res['is_slur'].shape, (
        res['pitch_midi'].shape, res['midi_dur'].shape, res['is_slur'].shape)

        # gt f0.
        gt_f0, gt_pitch_coarse = get_pitch(wav, spec, hparams)
        if sum(gt_f0) == 0:
            raise BinarizationError("Empty **gt** f0")
        res['f0'] = gt_f0
        res['pitch'] = gt_pitch_coarse

    @staticmethod
    def get_align(ph_durs, mel, phone_encoded, res, hop_size=hparams['hop_size'], audio_sample_rate=hparams['audio_sample_rate']):
        mel2ph = np.zeros([mel.shape[0]], int)
        startTime = 0

        for i_ph in range(len(ph_durs)):
            start_frame = int(startTime * audio_sample_rate / hop_size + 0.5)
            end_frame = int((startTime + ph_durs[i_ph]) * audio_sample_rate / hop_size + 0.5)
            mel2ph[start_frame:end_frame] = i_ph + 1
            startTime = startTime + ph_durs[i_ph]
        if any(mel2ph==0):
            zero_pos = mel2ph == 0
            zero_num = sum(zero_pos)
            mel2ph[-zero_num:] = mel2ph[-zero_num - 1]
        res['mel2ph'] = mel2ph

    @classmethod
    def process_item(cls, item_name, ph, txt, tg_fn, wav_fn, spk_id, encoder, binarization_args):
        if hparams['vocoder'] in VOCODERS:
            wav, mel = VOCODERS[hparams['vocoder']].wav2spec(wav_fn)
        else:
            wav, mel = VOCODERS[hparams['vocoder'].split('.')[-1]].wav2spec(wav_fn)
        res = {
            'item_name': item_name, 'txt': txt, 'ph': ph, 'mel': mel, 'wav': wav, 'wav_fn': wav_fn,
            'sec': len(wav) / hparams['audio_sample_rate'], 'len': mel.shape[0], 'spk_id': spk_id
        }
        try:
            if binarization_args['with_f0']:
                cls.get_pitch(wav_fn, wav, mel, ph, res)
            if binarization_args['with_txt']:
                try:
                    phone_encoded = res['phone'] = encoder.encode(ph)
                except:
                    traceback.print_exc()
                    raise BinarizationError(f"Empty phoneme")
                if binarization_args['with_align']:
                    cls.get_align(MidiSingingBinarizer.item2ph_durs[item_name], mel, phone_encoded, res)
        except BinarizationError as e:
            print(f"| Skip item ({e}). item_name: {item_name}, wav_fn: {wav_fn}")
            return None
        return res


class ZhSingingBinarizer(ZhBinarizer, SingingBinarizer):
    pass


class OpencpopBinarizer(MidiSingingBinarizer):
    item2midi = {}
    item2midi_dur = {}
    item2is_slur = {}
    item2ph_durs = {}
    item2wdb = {}

    def split_train_test_set(self, item_names):
        item_names = deepcopy(item_names)
        test_item_names = [x for x in item_names if any([x.startswith(ts) for ts in hparams['test_prefixes']])]
        train_item_names = [x for x in item_names if x not in set(test_item_names)]
        logging.info("train {}".format(len(train_item_names)))
        logging.info("test {}".format(len(test_item_names)))
        return train_item_names, test_item_names

    def load_meta_data(self):
        raw_data_dir = hparams['raw_data_dir']
        # meta_midi = json.load(open(os.path.join(raw_data_dir, 'meta.json')))   # [list of dict]
        utterance_labels = open(os.path.join(raw_data_dir, 'transcriptions.txt')).readlines()

        for utterance_label in utterance_labels:
            song_info = utterance_label.split('|')
            item_name = raw_item_name = song_info[0]
            self.item2wavfn[item_name] = f'{raw_data_dir}/wavs/{item_name}.wav'
            self.item2txt[item_name] = song_info[1] # 字符串

            self.item2ph[item_name] = song_info[2] # 带空格字符
            # self.item2wdb[item_name] = list(np.nonzero([1 if x in ALL_YUNMU + ['AP', 'SP'] else 0 for x in song_info[2].split()])[0])
            self.item2wdb[item_name] = [1 if x in ALL_YUNMU + ['AP', 'SP'] else 0 for x in song_info[2].split()] # 数字01列表
            self.item2ph_durs[item_name] = [float(x) for x in song_info[5].split(" ")] # 浮点数列表

            self.item2midi[item_name] = [librosa.note_to_midi(x.split("/")[0]) if x != 'rest' else 0
                                   for x in song_info[3].split(" ")] # 整数列表
            self.item2midi_dur[item_name] = [float(x) for x in song_info[4].split(" ")] # 浮点数列表
            self.item2is_slur[item_name] = [int(x) for x in song_info[6].split(" ")] # 数字01列表
            self.item2spk[item_name] = 'opencpop'

        print('spkers: ', set(self.item2spk.values()))
        self.item_names = sorted(list(self.item2txt.keys()))
        if self.binarization_args['shuffle']:
            random.seed(1234)
            random.shuffle(self.item_names)
        self._train_item_names, self._test_item_names = self.split_train_test_set(self.item_names)

    @staticmethod
    def get_pitch(wav_fn, wav, spec, ph, res):
        wav_suffix = '.wav'
        # midi_suffix = '.mid'
        wav_dir = 'wavs'
        f0_dir = 'text_f0_align'

        item_name = os.path.splitext(os.path.basename(wav_fn))[0]
        res['pitch_midi'] = np.asarray(OpencpopBinarizer.item2midi[item_name])
        res['midi_dur'] = np.asarray(OpencpopBinarizer.item2midi_dur[item_name])
        res['is_slur'] = np.asarray(OpencpopBinarizer.item2is_slur[item_name])
        res['word_boundary'] = np.asarray(OpencpopBinarizer.item2wdb[item_name])
        assert res['pitch_midi'].shape == res['midi_dur'].shape == res['is_slur'].shape, (res['pitch_midi'].shape, res['midi_dur'].shape, res['is_slur'].shape)

        # gt f0.
        # f0 = None
        # f0_suffix = '_f0.npy'
        # f0fn = wav_fn.replace(wav_suffix, f0_suffix).replace(wav_dir, f0_dir)
        # pitch_info = np.load(f0fn)
        # f0 = [x[1] for x in pitch_info]
        # spec_x_coor = np.arange(0, 1, 1 / len(spec))[:len(spec)]
        #
        # f0_x_coor = np.arange(0, 1, 1 / len(f0))[:len(f0)]
        # f0 = interp1d(f0_x_coor, f0, 'nearest', fill_value='extrapolate')(spec_x_coor)[:len(spec)]
        # if sum(f0) == 0:
        #     raise BinarizationError("Empty **gt** f0")
        #
        # pitch_coarse = f0_to_coarse(f0)
        # res['f0'] = f0
        # res['pitch'] = pitch_coarse

        # gt f0.
        gt_f0, gt_pitch_coarse = get_pitch(wav, spec, hparams)
        if sum(gt_f0) == 0:
            raise BinarizationError("Empty **gt** f0")
        res['f0'] = gt_f0
        res['pitch'] = gt_pitch_coarse

    @classmethod
    def process_item(cls, item_name, ph, txt, tg_fn, wav_fn, spk_id, encoder, binarization_args):
        if hparams['vocoder'] in VOCODERS:
            wav, mel = VOCODERS[hparams['vocoder']].wav2spec(wav_fn)
        else:
            wav, mel = VOCODERS[hparams['vocoder'].split('.')[-1]].wav2spec(wav_fn)
        res = {
            'item_name': item_name, 'txt': txt, 'ph': ph, 'mel': mel, 'wav': wav, 'wav_fn': wav_fn,
            'sec': len(wav) / hparams['audio_sample_rate'], 'len': mel.shape[0], 'spk_id': spk_id
        }
        try:
            if binarization_args['with_f0']:
                cls.get_pitch(wav_fn, wav, mel, ph, res)
            if binarization_args['with_txt']:
                try:
                    phone_encoded = res['phone'] = encoder.encode(ph)
                except:
                    traceback.print_exc()
                    raise BinarizationError(f"Empty phoneme")
                if binarization_args['with_align']:
                    cls.get_align(OpencpopBinarizer.item2ph_durs[item_name], mel, phone_encoded, res)
        except BinarizationError as e:
            print(f"| Skip item ({e}). item_name: {item_name}, wav_fn: {wav_fn}")
            return None
        return res

class OperaBinarizer(SingingBinarizer):

    #todo
    ph_items_list = {}

    def split_train_test_set(self, item_names):
        item_names = deepcopy(item_names)
        test_item_names = [x for x in item_names if any([x.startswith(ts) for ts in hparams['test_prefixes']])]
        train_item_names = [x for x in item_names if x not in set(test_item_names)]
        assert len(test_item_names) != 0 , "test items do not match any"
        logging.info("train {}".format(len(train_item_names)))
        logging.info("test {}".format(len(test_item_names)))
        return train_item_names, test_item_names

    def load_meta_data(self):
        for ds_id, processed_data_dir in enumerate(self.processed_data_dirs):
            wav_suffix = 'audio.wav'
            txt_suffix = 'char.txt'
            ph_suffix = 'phoneme.txt'
            all_wav_pieces = glob.glob(f'{processed_data_dir}/*/segment_*/{wav_suffix}')

            for piece_path in all_wav_pieces:
                path_segs = piece_path.split(os.sep) #list
                item_name = path_segs[-3] + '_' + path_segs[-2]
                self.item2txt[item_name] = open(f'{piece_path.replace(wav_suffix, txt_suffix)}').readline()
                with open(f'{piece_path.replace(wav_suffix, ph_suffix)}', 'r', encoding = 'utf-8') as f:
                    ph_items = f.readlines()
                    self.ph_items_list[item_name] = ph_items_list = [item.strip().split('\t') for item in ph_items]
                    self.item2ph[item_name] = ' '.join([item[2] for item in ph_items_list])
                self.item2wavfn[item_name] = piece_path
                self.item2spk[item_name] = 'opera'
        print('spkers: ', set(self.item2spk.values()))
        self.item_names = sorted(list(self.item2txt.keys()))
        if self.binarization_args['shuffle']:
            random.seed(1234)
            random.shuffle(self.item_names)
        self._train_item_names, self._test_item_names = self.split_train_test_set(self.item_names)

    @staticmethod
    def get_pitch(wav_fn, wav, spec, ph, res):
        # gt f0.
        gt_f0, gt_pitch_coarse = get_pitch(wav, spec, hparams)
        if sum(gt_f0) == 0:
            raise BinarizationError("Empty **gt** f0")
        res['f0'] = gt_f0
        res['pitch'] = gt_pitch_coarse

    @staticmethod
    def get_slur(ph_items,res):
        is_slur = []
        ph_items_pre = ph_items[:-1]
        ph_items_pre.insert(0,['0','0','sil'])
        ph_items_pre_pre = ph_items_pre[:-1]
        ph_items_pre_pre.insert(0,['0','0','sil'])
        for idx, ph_item in enumerate(ph_items):
            if ph_item[2] == ph_items_pre[idx][2] and ph_item[2] != 'sil':
                is_slur.append(1)
            elif ph_item[2] == ph_items_pre_pre[idx][2] and (ph_items_pre[idx][2] in ['sil','?']):
                is_slur.append(1)
            else:
                is_slur.append(0)
        assert len(is_slur) == len(ph_items)
        res['is_slur'] = np.asarray(is_slur)

    @staticmethod
    def get_align(ph_items, mel, phone_encoded, res, hop_size=hparams['hop_size'], audio_sample_rate=hparams['audio_sample_rate']):
        mel2ph = np.zeros([mel.shape[0]], int)

        for idx, ph_item in enumerate(ph_items):
            startTime = float(ph_item[0])
            endTime = float(ph_item[1])
            start_frame = int(startTime * audio_sample_rate / hop_size + 0.5)
            end_frame = int(endTime * audio_sample_rate / hop_size + 0.5)
            mel2ph[start_frame:end_frame] = idx + 1

        res['mel2ph'] = mel2ph

    @classmethod
    def process_item(cls, item_name, ph, txt, tg_fn, wav_fn, spk_id, encoder, binarization_args):
        if hparams['vocoder'] in VOCODERS:
            wav, mel = VOCODERS[hparams['vocoder']].wav2spec(wav_fn)
        else:
            wav, mel = VOCODERS[hparams['vocoder'].split('.')[-1]].wav2spec(wav_fn)
        res = {
            'item_name': item_name, 'txt': txt, 'ph': ph, 'mel': mel, 'wav': wav, 'wav_fn': wav_fn,
            'sec': len(wav) / hparams['audio_sample_rate'], 'len': mel.shape[0], 'spk_id': spk_id
        }
        try:
            if binarization_args['with_f0']:
                cls.get_pitch(wav_fn, wav, mel, ph, res)
            if binarization_args['with_slur']:
                cls.get_slur(OperaBinarizer.ph_items_list[item_name], res)
            if binarization_args['with_txt']:
                try:
                    phone_encoded = res['phone'] = encoder.encode(ph)
                except:
                    traceback.print_exc()
                    raise BinarizationError(f"Empty phoneme")
                if binarization_args['with_align']:
                    cls.get_align(OperaBinarizer.ph_items_list[item_name], mel, phone_encoded, res)
        except BinarizationError as e:
            print(f"| Skip item ({e}). item_name: {item_name}, wav_fn: {wav_fn}")
            return None
        return res

class IPABinarizer(SingingBinarizer):

    #todo
    ph_items_list = {}

    def split_train_test_set(self, item_names):
        item_names = deepcopy(item_names)
        test_item_names = [x for x in item_names if any([x.startswith(ts) for ts in hparams['test_prefixes']])]
        train_item_names = [x for x in item_names if x not in set(test_item_names)]
        assert len(test_item_names) != 0 , "test items do not match any"
        logging.info("train {}".format(len(train_item_names)))
        logging.info("test {}".format(len(test_item_names)))
        return train_item_names, test_item_names

    def load_meta_data(self):
        for ds_id, processed_data_dir in enumerate(self.processed_data_dirs):
            wav_suffix = '.wav'
            txt_suffix = '.lab'
            tg_suffix = '.TextGrid'
            all_wav_pieces = glob.glob(f'{processed_data_dir}/*.wav')

            for piece_path in all_wav_pieces:
                try:
                    tg_data = TextGrid.fromFile(piece_path.replace(wav_suffix, tg_suffix))
                except:
                    continue
                song_txt = open(f'{piece_path.replace(wav_suffix, txt_suffix)}').readline()
                wav_fn = piece_path.split(os.sep)[-1] #list
                item_name = wav_fn[:-4]
                self.item2txt[item_name] = ''.join(song_txt.split(' '))
                
                self.ph_items_list[item_name] = tg_data[-1]
                phs = [item.mark if item.mark != '' else 'sil' for item in tg_data[-1]]
                self.item2ph[item_name] = ' '.join(phs)
                self.item2wavfn[item_name] = piece_path
                self.item2spk[item_name] = 'ipa'
        print('spkers: ', set(self.item2spk.values()))
        self.item_names = sorted(list(self.item2txt.keys()))
        if self.binarization_args['shuffle']:
            random.seed(1234)
            random.shuffle(self.item_names)
        self._train_item_names, self._test_item_names = self.split_train_test_set(self.item_names)

    @staticmethod
    def get_pitch(wav_fn, wav, spec, ph, res):
        # gt f0.
        gt_f0, gt_pitch_coarse = get_pitch(wav, spec, hparams)
        if sum(gt_f0) == 0:
            raise BinarizationError("Empty **gt** f0")
        res['f0'] = gt_f0
        res['pitch'] = gt_pitch_coarse

    @staticmethod
    def get_slur(ph_items,res):
        is_slur = []
        ph_items_pre = ph_items[:-1]
        ph_items_pre.insert(0,['0','0','sil'])
        ph_items_pre_pre = ph_items_pre[:-1]
        ph_items_pre_pre.insert(0,['0','0','sil'])
        for idx, ph_item in enumerate(ph_items):
            if ph_item[2] == ph_items_pre[idx][2] and ph_item[2] != 'sil':
                is_slur.append(1)
            elif ph_item[2] == ph_items_pre_pre[idx][2] and (ph_items_pre[idx][2] in ['sil','?']):
                is_slur.append(1)
            else:
                is_slur.append(0)
        assert len(is_slur) == len(ph_items)
        res['is_slur'] = np.asarray(is_slur)

    @staticmethod
    def get_align(ph_items, mel, phone_encoded, res, hop_size=hparams['hop_size'], audio_sample_rate=hparams['audio_sample_rate']):
        mel2ph = np.zeros([mel.shape[0]], int)

        for idx, ph_item in enumerate(ph_items):
            startTime = float(ph_item.minTime)
            endTime = float(ph_item.maxTime)
            start_frame = int(startTime * audio_sample_rate / hop_size + 0.5)
            end_frame = int(endTime * audio_sample_rate / hop_size + 0.5)
            mel2ph[start_frame:end_frame] = idx + 1

        res['mel2ph'] = mel2ph

    @classmethod
    def process_item(cls, item_name, ph, txt, tg_fn, wav_fn, spk_id, encoder, binarization_args):
        if hparams['vocoder'] in VOCODERS:
            wav, mel = VOCODERS[hparams['vocoder']].wav2spec(wav_fn)
        else:
            wav, mel = VOCODERS[hparams['vocoder'].split('.')[-1]].wav2spec(wav_fn)
        res = {
            'item_name': item_name, 'txt': txt, 'ph': ph, 'mel': mel, 'wav': wav, 'wav_fn': wav_fn,
            'sec': len(wav) / hparams['audio_sample_rate'], 'len': mel.shape[0], 'spk_id': spk_id
        }
        try:
            if binarization_args['with_f0']:
                cls.get_pitch(wav_fn, wav, mel, ph, res)
            if binarization_args['with_slur']:
                pass
                #cls.get_slur(IPABinarizer.ph_items_list[item_name], res)
            if binarization_args['with_txt']:
                try:
                    phone_encoded = res['phone'] = encoder.encode(ph)
                except:
                    traceback.print_exc()
                    raise BinarizationError(f"Empty phoneme")
                if binarization_args['with_align']:
                    cls.get_align(IPABinarizer.ph_items_list[item_name], mel, phone_encoded, res)
        except BinarizationError as e:
            print(f"| Skip item ({e}). item_name: {item_name}, wav_fn: {wav_fn}")
            return None
        return res

class GeZiBinarizer(MidiSingingBinarizer):
    
    item2uv = {}
    def split_train_test_set(self, item_names):
        item_names = deepcopy(item_names)
        test_item_names = [x for x in item_names if any([x.startswith(ts) for ts in hparams['test_prefixes']])]
        train_item_names = [x for x in item_names if x not in set(test_item_names)]
        assert len(test_item_names) != 0 , "test items do not match any"
        logging.info("train {}".format(len(train_item_names)))
        logging.info("test {}".format(len(test_item_names)))
        return train_item_names, test_item_names

    def load_meta_data(self):
        UV = ['p', 'th', 'kh', 'ts', 'tsh', 's', 'h']
        SHENGMU = ['ph', 'm', 'b', 't', 'n', 'l', 'k', 'g', 'ng', 'j', 'p', 'th', 'kh', 'ts', 'tsh', 's', 'h']
        PHN_REDUCE_DICT = make_phn_map_dict(PHN_MAP)
        for ds_id, processed_data_dir in enumerate(self.processed_data_dirs):
            all_wav_pieces = glob.glob(f'{processed_data_dir}/*/*.wav')
            for piece_path in all_wav_pieces:
                ph_list = []
                *_, speaker, wav_piece = piece_path.strip().split(os.sep)
                item_name = wav_piece[:-4]

                with open(piece_path.replace('wav', 'info'), 'r', encoding = 'utf-8') as f:
                    infos = f.readlines()
                self.item2wavfn[item_name] = piece_path
                self.item2txt[item_name] = item_name # 字符串
                # for ph in infos[0].strip().split(' '):
                #     if ph in PHN_REDUCE_DICT.keys():
                #         ph_list.append(PHN_REDUCE_DICT[ph])
                #     else:
                #         ph_list.append(ph)
                #     self.item2ph[item_name] = ' '.join(ph_list) # 空格分隔字符列表
                self.item2ph[item_name] = infos[0].strip() # 空格分隔字符列表
                try:
                    self.item2wdb[item_name] = [int(i) for i in infos[4].strip().split(' ')] # 01列表
                except:
                    print(piece_path, item_name, infos[4])
                self.item2ph_durs[item_name] = [float(x) for x in infos[1].strip().split(" ")] # 浮点数列表
                self.item2midi[item_name] = [librosa.note_to_midi(x) if x != 'sp' else 1
                                   for x in infos[2].strip().split(" ")] # 整数列表
                # midi_dur = [float(x) for x in infos[1].strip().split(" ")]
                # phs = infos[0].strip().split(' ')
                # for idx, ph in enumerate(phs):
                #     if ph in SHENGMU and idx < len(phs) - 1 and phs[idx + 1] not in SHENGMU:
                #         midi_dur[idx] = midi_dur[idx] + midi_dur[idx + 1]
                #         midi_dur[idx + 1] = midi_dur[idx]
                # self.item2midi_dur[item_name] = midi_dur
                self.item2midi_dur[item_name] = [float(x) for x in infos[1].strip().split(" ")]
                self.item2is_slur[item_name] = [int(i) for i in infos[3].strip().split(' ')] # 01列表
                self.item2spk[item_name] = speaker
                self.item2uv[item_name] = [1 if i in UV else 0 for i in infos[0].strip().split(' ')]

        print('spkers: ', set(self.item2spk.values()))
        self.item_names = sorted(list(self.item2txt.keys()))
        if self.binarization_args['shuffle']:
            random.seed(1234)
            random.shuffle(self.item_names)
        self._train_item_names, self._test_item_names = self.split_train_test_set(self.item_names)

    @staticmethod
    def get_pitch(wav_fn, wav, spec, ph, res):
        item_name = wav_fn.strip().split(os.sep)[-1][:-4]
        res['pitch_midi'] = np.asarray(GeZiBinarizer.item2midi[item_name])
        res['midi_dur'] = np.asarray(GeZiBinarizer.item2midi_dur[item_name])
        res['is_slur'] = np.asarray(GeZiBinarizer.item2is_slur[item_name])
        res['word_boundary'] = np.asarray(GeZiBinarizer.item2wdb[item_name])
        res['uv_shengmu'] = np.asarray(GeZiBinarizer.item2uv[item_name])
        assert res['pitch_midi'].shape == res['midi_dur'].shape == res['is_slur'].shape, (
        res['pitch_midi'].shape, res['midi_dur'].shape, res['is_slur'].shape)

        # gt f0.
        gt_f0, gt_pitch_coarse = get_pitch(wav, spec, hparams)
        if sum(gt_f0) == 0:
            raise BinarizationError("Empty **gt** f0")
        res['f0'] = gt_f0
        res['pitch'] = gt_pitch_coarse # 放缩到[0-255]的音高

    @classmethod
    def process_item(cls, item_name, ph, txt, tg_fn, wav_fn, spk_id, encoder, binarization_args):
        if hparams['vocoder'] in VOCODERS:
            wav, mel = VOCODERS[hparams['vocoder']].wav2spec(wav_fn)
        else:
            wav, mel = VOCODERS[hparams['vocoder'].split('.')[-1]].wav2spec(wav_fn)
        if len(mel) % 4 != 0:
            retain = 4 * (len(mel) // 4)
            mel = mel[:retain, :]
        res = {
            'item_name': item_name, 'txt': txt, 'ph': ph, 'mel': mel, 'wav': wav, 'wav_fn': wav_fn,
            'sec': len(wav) / hparams['audio_sample_rate'], 'len': mel.shape[0], 'spk_id': spk_id
        }
        try:
            if binarization_args['with_f0']:
                cls.get_pitch(wav_fn, wav, mel, ph, res)
            if binarization_args['with_txt']:
                try:
                    phone_encoded = res['phone'] = encoder.encode(ph)
                except:
                    traceback.print_exc()
                    raise BinarizationError(f"Empty phoneme")
                if binarization_args['with_align']:
                    cls.get_align(GeZiBinarizer.item2ph_durs[item_name], mel, phone_encoded, res)
        except BinarizationError as e:
            print(f"| Skip item ({e}). item_name: {item_name}, wav_fn: {wav_fn}")
            return None
        return res

class HokkienBinarizer(GeZiBinarizer):
    
    def load_meta_data(self):
        for ds_id, processed_data_dir in enumerate(self.processed_data_dirs):
            all_wav_pieces = glob.glob(f'{processed_data_dir}/*/*.wav')
            for piece_path in all_wav_pieces:
                try:
                    with open(piece_path.replace('wav', 'info'), 'r', encoding = 'utf-8') as f:
                        infos = f.readlines()
                except:
                    continue
                *_, spk, wav_piece = piece_path.strip().split(os.sep)
                item_name = spk + '-' + wav_piece[:-4]
                self.item2wavfn[item_name] = piece_path
                self.item2txt[item_name] = infos[0].strip() # 字符串
                self.item2ph[item_name] = infos[0].strip() # 空格分隔字符列表
                self.item2ph_durs[item_name] = [float(x) for x in infos[1].strip().split(" ")] # 浮点数列表
                self.item2spk[item_name] = spk

        print('spkers: ', set(self.item2spk.values()))
        self.item_names = sorted(list(self.item2txt.keys()))
        if self.binarization_args['shuffle']:
            random.seed(1234)
            random.shuffle(self.item_names)
        self._train_item_names, self._test_item_names = self.split_train_test_set(self.item_names)

    @staticmethod
    def get_pitch(wav_fn, wav, spec, ph, res):
        # gt f0.
        gt_f0, gt_pitch_coarse = get_pitch(wav, spec, hparams)
        if sum(gt_f0) == 0:
            raise BinarizationError("Empty **gt** f0")
        res['f0'] = gt_f0
        res['pitch'] = gt_pitch_coarse # 放缩到[0-255]的音高

    @classmethod
    def process_item(cls, item_name, ph, txt, tg_fn, wav_fn, spk_id, encoder, binarization_args):
        if hparams['vocoder'] in VOCODERS:
            wav, mel = VOCODERS[hparams['vocoder']].wav2spec(wav_fn)
        else:
            wav, mel = VOCODERS[hparams['vocoder'].split('.')[-1]].wav2spec(wav_fn)
            
        if len(mel) % 4 != 0:
            retain = 4 * (len(mel) // 4)
            mel = mel[:retain, :]

        res = {
            'item_name': item_name, 'txt': txt, 'ph': ph, 'mel': mel, 'wav': wav, 'wav_fn': wav_fn,
            'sec': len(wav) / hparams['audio_sample_rate'], 'len': mel.shape[0], 'spk_id': spk_id
        }
        try:
            if binarization_args['with_f0']:
                cls.get_pitch(wav_fn, wav, mel, ph, res)
            if binarization_args['with_txt']:
                try:
                    phone_encoded = res['phone'] = encoder.encode(ph)
                except:
                    traceback.print_exc()
                    raise BinarizationError(f"Empty phoneme")
                if binarization_args['with_align']:
                    cls.get_align(HokkienBinarizer.item2ph_durs[item_name], mel, phone_encoded, res)
        except BinarizationError as e:
            print(f"| Skip item ({e}). item_name: {item_name}, wav_fn: {wav_fn}")
            return None
        return res

if __name__ == "__main__":
    SingingBinarizer().process()
