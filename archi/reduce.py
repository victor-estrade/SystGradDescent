# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import torch
import torch.nn as nn


def torch_weighted_mean(input, weight, dim, keepdim=False, dtype=None):
    z = torch.sum(weight)
    out = torch.sum(input * weight, dim, keepdim=keepdim) / z
    return out

def torch_weighted_sum(input, weight, dim, keepdim=False, dtype=None):
    out = torch.sum(input * weight, dim, keepdim=keepdim)
    return out

# Pytorch example to add a module
# https://pytorch.org/docs/stable/notes/extending.html#adding-a-module

class MeanBloc(nn.Module):
    def __init__(self, n_in, n_out, n_extra=0):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_out)
        self.fc2 = nn.Linear(n_in+n_extra, n_out)

    def forward(self, x, w, extra=None):
        x_out = self.fc1(x)
        x_mean = torch_weighted_mean(x, w, 0, keepdim=True)
        if extra is not None :
            x_mean = torch.cat((x_mean, extra), 1)
        x_mean = self.fc2(x_mean)
        out = x_mean + x_out
        return out

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()


class SumBloc(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_out)
        self.fc2 = nn.Linear(n_in, n_out)

    def forward(self, x, w):
        x_out = self.fc1(x)
        x_sum = self.fc2(torch_weighted_sum(x, w, 0, keepdim=True))
        out = x_sum + x_out
        return out

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
