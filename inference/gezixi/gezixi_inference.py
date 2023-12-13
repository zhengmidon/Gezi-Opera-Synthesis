import torch
from inference.gezixi.base_inference import BaseSVSInfer
from utils import load_ckpt
from utils.hparams import hparams
from usr.gezixi_task import FastSpeech2Perceptual
from modules.fastspeech.pe import PitchExtractor
import utils


class GeZiXiInfer(BaseSVSInfer):
    def build_model(self):
        out_dims = hparams['audio_num_mel_bins']
        disc_conditional = hparams['disc_conditional']
        model = FastSpeech2Perceptual(self.ph_encoder, out_dims, disc_conditional)
        model.eval()
        load_ckpt(model, hparams['work_dir'], 'model')

        if hparams.get('pe_enable') is not None and hparams['pe_enable']:
            self.pe = PitchExtractor().to(self.device)
            utils.load_ckpt(self.pe, hparams['pe_ckpt'], 'model', strict=True)
            self.pe.eval()
        return model

    def forward_model(self, inp):
        sample = self.input_to_batch(inp)
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        spk_id = sample.get('spk_ids')
        mel2ph = sample.get('mel2ph')
        print(f"Mel length {mel2ph.shape[1] * hparams['hop_size'] / hparams['audio_sample_rate']}s")
        with torch.no_grad():
            output = self.model(txt_tokens, mel2ph=mel2ph, spk_embed=spk_id, ref_mels=None, infer=True,
                                pitch_midi=sample['pitch_midi'], midi_dur=sample['midi_dur'],
                                is_slur=sample['is_slur'], uv_shengmu=sample['uv_shengmu'])
            mel_out = output['mel_out']  # [B, T, 80]
            if hparams.get('pe_enable') is not None and hparams['pe_enable']:
                f0_pred = self.pe(mel_out)['f0_denorm_pred']  # pe predict from Pred mel
            else:
                f0_pred = output['f0_denorm']
            wav_out = self.run_vocoder(mel_out, f0=f0_pred)
        wav_out = wav_out.cpu().numpy()
        return wav_out[0]

if __name__ == '__main__':
    inp = {
            'text': '你 说 你 不 懂 sp 为 何 在 这 时 牵 手 - - sp',
            'notes': 'D#4 D#4 D#4 D#4 sp D#4 D4 D4 D4 D#4 F4 D#4 D4 C4 B3 sp',
            'notes_duration': '0.113740 0.329060 0.287950 0.133480 0.150900 0.484730 0.242010 0.180820 0.343570 0.152050 0.266720 0.280310 0.633300 0.21 0.13 0.444590',
            'input_type': 'word',
            'spk_name': '姚琼男',
    }  # user input: Chinese characters
    c = {
        'text': 'sp 阿 兄 - 勤 - - - 劳 - sp 又 - - 善 - - - 良 - sp',
        'notes': 'sp D4 E4 F♯4 A♯3 A3 B3 D4 G♯3 A3 sp E4 F♯4 A4 F4 D♯4 C♯4 B3 D4 E4 sp',
        'notes_duration': '0.415 0.81 0.602 0.36 0.328 0.217 0.278 0.202 0.322 0.386 0.3 0.448 0.25 0.142 0.577 0.138 0.148 0.211 0.563 0.959 0.039',
    }  # input like Opencpop dataset.
    d = {
        'text': '夫 唱 妇 - - - 随 - sp 相 敬 如 - 宾 sp',
        'notes': 'C5 E4 G4 A4 F4 D4 A3 C4 sp C4 A3 C4 D4 E4 sp',
        'notes_duration': '0.545 0.666 0.485 0.244 0.247 0.23 0.784 1.142 0.541 0.64 1.245 0.323 0.339 1.534 0.316',
    }  # input like Opencpop dataset.
    e = {
        'text': '小 酒 窝 长 - 睫 - 毛 SP 是 你 最 美 的 记 号',
        'notes': 'C#4 F#4 G#4 A#4 F#4 F#4 C#4 C#4 sp C#4 A#4 G#4 A#4 G#4 F4 C#4',
        'notes_duration': '0.407 0.376 0.242 0.509 0.183 0.315 0.235 0.361 0.223 0.377 0.340 0.299 0.344 0.283 0.323 0.360',
    }  # input like Opencpop dataset.

    GeZiXiInfer.example_run(d)


# python inference/svs/ds_e2e.py --config usr/configs/midi/e2e/opencpop/ds100_adj_rel.yaml --exp_name 0228_opencpop_ds100_rel