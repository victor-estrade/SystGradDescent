# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import torch
import torch.nn as nn

from . import layers


def softmax_cat(x_, x):
    x_ = torch.softmax(x_, 1)
    x = torch.cat((x, x_), 1)
    return x


class ResidualBlock(nn.Module):
    __constants__ = ['n_in', 'n_middle', 'bias', 'activation']

    def __init__(self, n_in, n_middle, bias=False, activation=torch.relu):
        super().__init__()
        self.n_in = n_in
        self.n_middle = n_middle
        self.bias = bias
        self.fc_in = nn.Linear(n_in, n_middle, bias)
        self.fc_out = nn.Linear(n_middle, n_in, bias)
        self.activation = activation

    def forward(self, x):
        x_ = self.fc_in(x)
        x_ = self.activation(x_)
        x_ = self.fc_out(x_)
        x  = x + x_
        return x

    def reset_parameters(self):
        self.fc_in.reset_parameters()
        self.fc_out.reset_parameters()


class ResidualAverageBlock(nn.Module):
    __constants__ = ['n_in', 'n_middle', 'bias', 'activation']

    def __init__(self, n_in, n_middle, bias=False, activation=torch.relu):
        super().__init__()
        self.n_in = n_in
        self.n_middle = n_middle
        self.avg_in = layers.Average(n_in, n_middle, bias)
        self.avg_out = layers.Average(n_middle, n_in, bias)
        self.activation = activation

    def forward(self, x, w, p=None):
        x_ = self.avg_in(x, w)
        x_ = self.activation(x_)
        x_ = self.avg_out(x_, w)
        x  = x + x_
        return x

    def reset_parameters(self):
        self.fc_in.reset_parameters()
        self.fc_out.reset_parameters()


class ResidualAverageExtraBlock(nn.Module):
    __constants__ = ['n_in', 'n_middle', 'n_extra', 'bias', 'activation']

    def __init__(self, n_in, n_middle, n_extra,  bias=False, activation=torch.relu):
        super().__init__()
        self.n_in = n_in
        self.n_middle = n_middle
        self.n_extra = n_extra
        self.avg_in = layers.AverageExtra(n_in, n_middle, n_extra, bias)
        self.avg_out = layers.Average(n_middle, n_in, bias)
        self.activation = activation

    def forward(self, x, w, p):
        x_ = self.avg_in(x, w, p)
        x_ = self.activation(x_)
        x_ = self.avg_out(x_, w)
        x  = x + x_
        return x

    def reset_parameters(self):
        self.fc_in.reset_parameters()
        self.fc_out.reset_parameters()

