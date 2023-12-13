import torch
import torch.nn as nn
import utils
from utils.hparams import hparams
from modules.commons.common_layers import *
from tasks.base_task import BaseDataset
import glob
import os
from utils.pl_utils import data_loader

from tasks.tts.fs2 import FastSpeech2Task

import numpy as np
import os
import torch.nn.functional as F

class SpecClsDataset(torch.utils.data.Dataset):
    def __init__(self, prefix, base_dir):
        super().__init__()
        path = os.path.join(base_dir, prefix)
        self.item_paths = glob.glob(f'{path}/*')
        self.item_dict = {p.split(os.sep)[-1]:p for p in self.item_paths}
        self.items = [k for k,v in self.item_dict.items()]
        print(f"{base_dir}/{prefix}: {len(self)} items")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        spec = np.load(self.item_dict[self.items[index]]) # [N, MB]
        spec = torch.from_numpy(spec)
        label = 1 if self.items[index].startswith('g') else 0

        sample = {
            "id": index,
            "label": label,
            "mel": spec,
        }
        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}

        id = torch.LongTensor([s['id'] for s in samples])
        mels = utils.collate_2d([s['mel'] for s in samples], 0.0)
        labels = torch.FloatTensor([s['label'] for s in samples])
        mel_lengths = torch.LongTensor([s['mel'].shape[0] for s in samples])

        batch = {
            'id': id,
            'labels': labels,
            'mels': mels, # [B, N, MB]
            'mel_lengths': mel_lengths,
            'nsamples': len(samples)
        }

        return batch

class SpecClsModel(nn.Module):
    def __init__(self):
        super(SpecClsModel, self).__init__()

        self.register_buffer('spec_min', torch.FloatTensor([-10.0]))
        self.register_buffer('spec_max', torch.FloatTensor([1.5]))

        self.conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1]),
                nn.GroupNorm(2, 32, eps=1e-05, affine=True),
                nn.ELU(alpha=1.0)),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=3, padding=0, dilation=1, ceil_mode=False),
                nn.Dropout(p=0.1),
                nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1]),
                nn.GroupNorm(2, 64, eps=1e-05, affine=True),
                nn.ELU(alpha=1.0)),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False),
                nn.Dropout(p=0.1),
                nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1]),
                nn.GroupNorm(2, 64, eps=1e-05, affine=True),
                nn.ELU(alpha=1.0)),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False),
                nn.Dropout(p=0.1)
            )
        ])
        self.lstm = nn.LSTM(64, 64, num_layers=2, bidirectional=True, batch_first=True)
        self.linear = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Linear(in_features=64, out_features=1, bias=True),
            nn.Sigmoid()
            )

    # 放缩到[-1,1]
    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min

    def forward(self, mels, features_only=True):
        '''
        mels: [B, N, MB]

        '''
        output = {}
        mels = mels.transpose(1, 2)
        mels = self.norm_spec(mels)
        
        x = mels.unsqueeze(1) # [B, 1, MB, N]
        features = {}
        for idx, net in enumerate(self.conv):
            if idx < 3:
                x = net(x)
                features[f"conv_{idx}"] = x # [B, C_i, MB_i, N_i]
            else:
                x_hidden = net(x) # [B, 64, 1, N_i]

        x_hidden = x_hidden.reshape(x_hidden.shape[0], x_hidden.shape[1], -1).transpose(1, 2)
        x_pack, hidden = self.lstm(x_hidden)

        h_n = hidden[0][-1] # [B, C]
        features[f"lstm"] = h_n
        output['features'] = features
        if features_only:
            return output

        output['prob'] = self.linear(h_n)[:, 0] # [B]
        
        return output 

class SpecClsModel1d(nn.Module):
    def __init__(self):
        super(SpecClsModel1d, self).__init__()

        self.register_buffer('spec_min', torch.FloatTensor([-10.0]))
        self.register_buffer('spec_max', torch.FloatTensor([1.5]))

        self.conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(80, 32, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(2, 32, eps=1e-05, affine=True),
                nn.ELU(alpha=1.0)),
            nn.Sequential(
                nn.MaxPool1d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
                nn.Dropout(p=0.1),
                nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(2, 64, eps=1e-05, affine=True),
                nn.ELU(alpha=1.0)),
            nn.Sequential(
                nn.MaxPool1d(kernel_size=4, stride=3, padding=0, dilation=1, ceil_mode=False),
                nn.Dropout(p=0.1),
                nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(2, 64, eps=1e-05, affine=True),
                nn.ELU(alpha=1.0)),
            nn.Sequential(
                nn.MaxPool1d(kernel_size=4, stride=3, padding=0, dilation=1, ceil_mode=False),
                nn.Dropout(p=0.1)
            )
        ])
        self.lstm = nn.LSTM(64, 64, num_layers=2, bidirectional=True, batch_first=True)
        self.linear = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Linear(in_features=64, out_features=1, bias=True),
            nn.Sigmoid()
            )

    # 放缩到[-1,1]
    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min

    def forward(self, mels, features_only=True):
        '''
        mels: [B, N, MB]

        '''
        output = {}
        mels = mels.transpose(1, 2)
        mels = self.norm_spec(mels)
        
        x = mels # [B, MB, N]
        features = {}
        for idx, net in enumerate(self.conv):
            if idx < 3:
                x = net(x)
                features[f"conv_{idx}"] = x # [B, C_i, N_i]
            else:
                x_hidden = net(x) # [B, 64, N_i]

        x_hidden = x_hidden.transpose(1, 2) # [B, N_i, 64]
        x_pack, hidden = self.lstm(x_hidden)

        h_n = hidden[0][-1] # [B, C]
        features[f"lstm"] = h_n
        output['features'] = features
        if features_only:
            return output

        output['prob'] = self.linear(h_n)[:, 0] # [B]
        
        return output 

class SpecClsTask(FastSpeech2Task):
    def __init__(self):
        super(SpecClsTask, self).__init__()
        self.dataset_cls = SpecClsDataset
        self.bce_loss = nn.BCELoss()

    def build_tts_model(self):
        self.model = SpecClsModel1d()

    def build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, hparams['decay_steps'], gamma=0.6)

    def build_optimizer(self, model):
        self.optimizer = optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            weight_decay=hparams['weight_decay'])
        return optimizer

    def _training_step(self, sample, batch_idx, _):
        loss_output = self.run_model(self.model, sample)
        total_loss = sum([v for v in loss_output.values() if isinstance(v, torch.Tensor) and v.requires_grad])
        loss_output['batch_size'] = sample['mels'].size()[0]
        return total_loss, loss_output

    @data_loader
    def train_dataloader(self):
        train_dataset = self.dataset_cls(prefix='train', base_dir='data/processed/gezi_speech_mel')
        return torch.utils.data.DataLoader(train_dataset,
                                           collate_fn=train_dataset.collater,
                                           batch_size=8, shuffle=True,
                                           num_workers=hparams['ds_workers'],
                                           pin_memory=False)

    @data_loader
    def val_dataloader(self):
        valid_dataset = self.dataset_cls(prefix='valid', base_dir='data/processed/gezi_speech_mel')
        return torch.utils.data.DataLoader(valid_dataset,
                                           collate_fn=valid_dataset.collater,
                                           batch_size=8, shuffle=True,
                                           num_workers=hparams['ds_workers'],
                                           pin_memory=False)

    def run_model(self, model, sample, return_output=False, infer=False):
        labels = sample['labels'] # [B]
        mels = sample['mels']  # [B, T_s, 80]
        mel_lengths = sample['mel_lengths']

        output = model(mels=mels, features_only=False)

        losses = {}
        
        losses["bce"] = self.bce_loss(output['prob'], labels)

        if not return_output:
            return losses
        else:
            return losses, output

    def validation_step(self, sample, batch_idx):
        # print(f'Validating {batch_idx}')
        outputs = {}
        labels = sample['labels']

        outputs['losses'] = {}
        outputs['losses'], model_out = self.run_model(self.model, sample, return_output=True, infer=False)
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']

        prediction = torch.zeros_like(model_out['prob'])
        for idx, prob in enumerate(model_out['prob']):
            if prob > 0.99:
                prediction[idx] = 1. 
            elif prob < 0.01:
                prediction[idx] = 0. 
            else:
                prediction[idx] = 0.5

        bingo = prediction == labels
        
        accuracy = torch.sum(bingo).true_divide(sample['nsamples'])
        outputs['accuracy'] = accuracy
        # print(f"prediction, {prediction}, labels {labels}, bingo {bingo}, acc {accuracy}")

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
