"""
Noiseprint model implementation based on:

D. Cozzolino and L. Verdoliva,
"Noiseprint: A CNN-Based Camera Model Fingerprint",
IEEE Transactions on Information Forensics and Security, vol. 15, pp. 144–159, 2020.
DOI: 10.1109/TIFS.2019.2916364

Original implementation (Matlab/TensorFlow) © 2019 GRIP-UNINA.
Adapted for research and educational use in Python/PyTorch by Jakub Teichman, 2025.
"""

import torch
import torch.nn as nn

class AddBias(nn.Module):
    """Warstwa dodająca bias (niestandardowa implementacja z Noiseprinta)"""
    def __init__(self, num_features):
        super(AddBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        return x + self.bias.view(1, -1, 1, 1).expand_as(x)


class CustomBatchNorm(nn.Module):
    """Zamiennik BatchNorm – używa predefiniowanych wartości gamma i wariancji"""
    def __init__(self, num_features, bnorm_init_gamma, bnorm_init_var, bnorm_decay, bnorm_epsilon):
        super(CustomBatchNorm, self).__init__()
        self.num_features = num_features
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.moving_mean = nn.Parameter(torch.ones(num_features))
        self.moving_variance = nn.Parameter(torch.ones(num_features))
        self.bnorm_epsilon = bnorm_epsilon

    def forward(self, x):
        _, C, _, _ = x.shape
        x = (x - self.moving_mean.reshape((1, C, 1, 1))) / torch.sqrt(
            self.moving_variance.reshape((1, C, 1, 1)) + self.bnorm_epsilon
        )
        return self.gamma.reshape((1, C, 1, 1)) * x


class FullConvNet(nn.Module):
    """Pełna sieć konwolucyjna Noiseprint (17 poziomów)"""
    def __init__(self, bnorm_decay=0.9, flag_train=False, num_levels=17):
        super(FullConvNet, self).__init__()
        self._num_levels = num_levels
        self._actfun = [nn.ReLU()] * (num_levels - 1) + [nn.Identity()]
        self._f_size = [3] * num_levels
        self._f_num_in = [1] + [64] * (num_levels - 1)
        self._f_num_out = [64] * (num_levels - 1) + [1]
        self._f_stride = [1] * num_levels
        self._bnorm = [False] + [True] * (num_levels - 2) + [False]
        self.conv_bias = [True] + [False] * (num_levels - 2) + [True]
        self._bnorm_init_gamma = torch.sqrt(torch.tensor(2.0 / (9.0 * 64.0)))
        self._bnorm_init_var = 1e-4
        self._bnorm_epsilon = 1e-5
        self._bnorm_decay = bnorm_decay

        self.conv_layers = nn.ModuleList([
            self._conv_layer(self._f_size[i], self._f_num_in[i], self._f_num_out[i],
                             self._f_stride[i], self._bnorm[i], self.conv_bias[i], self._actfun[i])
            for i in range(self._num_levels)
        ])

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x

    def _batch_norm(self, out_filters):
        return CustomBatchNorm(out_filters, self._bnorm_init_gamma, self._bnorm_init_var,
                               self._bnorm_decay, self._bnorm_epsilon)

    def _conv_layer(self, filter_size, in_filters, out_filters, stride, apply_bnorm, conv_bias, actfun):
        layers = [nn.Conv2d(in_filters, out_filters, filter_size, stride=stride, padding="same", bias=conv_bias)]
        if apply_bnorm:
            layers.append(self._batch_norm(out_filters))
        if not conv_bias:
            layers.append(AddBias(out_filters))
        layers.append(actfun)
        return nn.Sequential(*layers)
