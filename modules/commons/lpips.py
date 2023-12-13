"""Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models"""

import torch
import torch.nn as nn
from collections import namedtuple
from usr.spec_cls_task import SpecClsModel, SpecClsModel1d
import utils


class LPIPSv2(nn.Module):
    # Learned perceptual metric
    def __init__(self, ckpt_path, conv_reduce=True, use_dropout=True, conv1d=True):
        super().__init__()
        self.ckpt_path = ckpt_path 
        self.conv_reduce = conv_reduce
        self.chns = [32, 64, 64]  # vg16 features
        self.net = SpecClsModel1d() if conv1d else SpecClsModel()
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self):
        utils.load_ckpt(self.net, self.ckpt_path, 'model', strict=True)
        print("loaded pretrained SpecClsModel loss from {}".format(self.ckpt_path))


    def forward(self, input, target):
        outs0, outs1 = self.net(input), self.net(target)
        conv_feat_0 = [v for k, v in outs0['features'].items() if k.startswith('conv')]
        conv_feat_1 = [v for k, v in outs1['features'].items() if k.startswith('conv')]
        lstm_feat_0 = [v for k, v in outs0['features'].items() if k.startswith('lstm')]
        lstm_feat_1 = [v for k, v in outs1['features'].items() if k.startswith('lstm')]

        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(conv_feat_0[kk]), normalize_tensor(conv_feat_1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        if self.conv_reduce:
            res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        else:
            res = [spatial_average(diffs[kk].sum(dim=1,keepdim=True), keepdim=True) for kk in range(len(self.chns))]
        val = res[0].mean()
        for l in range(1, len(self.chns)):
            val = val + res[l].mean()
        lstm_diff = (normalize_tensor(lstm_feat_0[0]) - normalize_tensor(lstm_feat_1[0])) ** 2
        val = val + lstm_diff.mean()
        return val


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


def normalize_tensor(x,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x, keepdim=True):
    return x.mean([-2, -1],keepdim=keepdim)

