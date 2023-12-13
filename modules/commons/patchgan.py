import functools
import torch.nn as nn
import torch

class ActNorm(nn.Module):
    def __init__(self, num_features, logdet=False, affine=True,
                 allow_reverse_init=False):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height*width*torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            else:
                self.initialize(output)
                self.initialized.fill_(1)

        if len(output.shape) == 2:
            output = output[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h

def normalize_tensor(x,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2,dim=1,keepdim=True))
    return x/(norm_factor+eps)


def spatial_average(x, keepdim=True):
    return x.mean([-2, -1],keepdim=keepdim)

class PatchGANDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(PatchGANDiscriminator, self).__init__()
        self.L = n_layers
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 5
        padw = 2
        self.in_conv = nn.Sequential(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True))
        nf_mult = 1
        nf_mult_prev = 1
        module_list = []
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            module_list.append(
                nn.Sequential(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
                )
            )

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        module_list.append(
            nn.Sequential(
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
            )
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw),
            )  # output 1 channel prediction map
        self.layers = nn.ModuleList(module_list)

    def forward(self, input):
        """Standard forward."""
        x = self.in_conv(input)
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        out = self.out_conv(x)
        return out, features

    def feature_matching_loss(self, features_real, features_fake):
        feats0, feats1, diffs = {}, {}, {}
        for kk in range(self.L):
            feats0[kk], feats1[kk] = normalize_tensor(features_real[kk]), normalize_tensor(features_fake[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(diffs[kk].sum(dim=1,keepdim=True), keepdim=True) for kk in range(self.L)]
        val = res[0].mean()
        for l in range(1, self.L):
            val = val + res[l].mean()
        return val


class PatchGAN1dDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=80, ndf=64, n_layers=4, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(PatchGAN1dDiscriminator, self).__init__()
        self.L = n_layers

        kw = 5
        padw = 2
        use_bias = False
        self.in_conv = nn.Sequential(nn.Conv1d(input_nc, ndf, kernel_size=kw, stride=3, padding=padw), nn.LeakyReLU(0.2, True))
        nf_mult = 1
        nf_mult_prev = 1
        module_list = []
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            module_list.append(
                nn.Sequential(
                nn.Conv1d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=3, padding=padw, bias=use_bias),
                nn.GroupNorm(16, ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
                )
            )

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        module_list.append(
            nn.Sequential(
            nn.Conv1d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            nn.GroupNorm(16, ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
            )
        )

        self.out_conv = nn.Sequential(
            nn.Conv1d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw),
            )  # output 1 channel prediction map
        self.gsl = GSL_Layer()
        self.layers = nn.ModuleList(module_list)

    def forward(self, input):
        """Standard forward
        :param input: [B, T, M]
        output:
            out: [B, 1, T_o]
        """
        input = input.transpose(1, 2)
        x = self.in_conv(input)
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x) # [B, C_i, T_i]
        out = self.out_conv(x)
        out = self.gsl(out)
        return out, features

    def feature_matching_loss(self, features_real, features_fake):
        feats0, feats1, diffs = {}, {}, {}
        for kk in range(self.L):
            feats0[kk], feats1[kk] = normalize_tensor(features_real[kk]), normalize_tensor(features_fake[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(diffs[kk].sum(dim=1, keepdim=True), keepdim=True) for kk in range(self.L)]
        val = res[0].mean()
        for l in range(1, self.L):
            val = val + res[l].mean()
        return val

class GradientScalingFunction(torch.autograd.Function):
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
            return grad_output * ctx.coeff, None

class GSL_Layer(nn.Module):
    def __init__(self):
        super(GSL_Layer, self).__init__()

    def forward(self, *input):
        return GradientScalingFunction.apply(*input)
