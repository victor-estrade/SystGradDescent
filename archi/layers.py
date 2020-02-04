# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def torch_weighted_mean(input, weight, dim, keepdim=False, dtype=None):
    z = torch.sum(weight)
    out = torch.sum(input * weight, dim, keepdim=keepdim) / z
    return out

def torch_weighted_sum(input, weight, dim, keepdim=False, dtype=None):
    out = torch.sum(input * weight, dim, keepdim=keepdim)
    return out

# Pytorch example to add a module
# https://pytorch.org/docs/stable/notes/extending.html#adding-a-module
# Pytorch Linear module implementation
# https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L33

class AverageExtra(nn.Module):
    def __init__(self, n_in, n_out, n_extra, bias=True):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_extra = n_extra
        self.bias = bias
        self.fc1 = nn.Linear(n_in, n_out, bias=False)
        self.fc2 = nn.Linear(n_in+n_extra, n_out, bias=bias)
        self.reset_parameters()

    def forward(self, x, w, extra):
        x_out = self.fc1(x)
        x_mean = torch_weighted_mean(x, w, 0, keepdim=True)
        x_mean = torch.cat((x_mean, extra), 1)
        x_mean = self.fc2(x_mean)
        out = x_mean + x_out
        return out

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def extra_repr(self):
        return 'in_features={}, out_features={}, extra_summaries={}, bias={}'.format(
            self.n_in, self.n_out, self.n_extra, self.bias
        )


class Average(nn.Module):
    def __init__(self, n_in, n_out, bias=True):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.bias = bias
        self.fc1 = nn.Linear(n_in, n_out, bias=False)
        self.fc2 = nn.Linear(n_in, n_out)
        self.reset_parameters()

    def forward(self, x, w):
        x_out = self.fc1(x)
        x_mean = torch_weighted_mean(x, w, 0, keepdim=True)
        x_mean = self.fc2(x_mean)
        out = x_mean + x_out
        return out

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.n_in, self.n_out, self.bias
        )


class SumExtra(nn.Module):
    def __init__(self, n_in, n_out, n_extra, bias=True):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_extra = n_extra
        self.bias = bias
        self.fc1 = nn.Linear(n_in, n_out, bias=False)
        self.fc2 = nn.Linear(n_in+n_extra, n_out)
        self.reset_parameters()

    def forward(self, x, w, extra):
        x_out = self.fc1(x)
        x_sum = self.fc2(torch_weighted_sum(x, w, 0, keepdim=True))
        x_sum = torch.cat((x_sum, extra), 1)
        out = x_sum + x_out
        return out

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def extra_repr(self):
        return 'in_features={}, out_features={}, extra_summaries={}, bias={}'.format(
            self.n_in, self.n_out, self.n_extra, self.bias
        )


class Sum(nn.Module):
    def __init__(self, n_in, n_out, bias=True):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.bias = bias
        self.fc1 = nn.Linear(n_in, n_out, bias=False)
        self.fc2 = nn.Linear(n_in, n_out)
        self.reset_parameters()

    def forward(self, x, w):
        x_out = self.fc1(x)
        x_sum = self.fc2(torch_weighted_sum(x, w, 0, keepdim=True))
        out = x_sum + x_out
        return out

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.n_in, self.n_out, self.bias
        )


class Average2(nn.Module):
# Shamelessly copy and adapted from 
# Pytorch Linear module implementation
# https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L33
# Or permalink
# https://github.com/pytorch/pytorch/blob/b894dc06de3e0750d9db8bd20b92429f6d873fa1/torch/nn/modules/linear.py#L33
    __constants__ = ['in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.feature_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.summary_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.feature_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.summary_weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.summary_weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, weight):
        # x = F.linear(input, self.feature_weight, None)
        x = input.matmul(self.feature_weight.t())
        x_mean = torch_weighted_mean(input, weight, 0, keepdim=True)
        x_summary = F.linear(x_mean, self.summary_weight, self.bias)
        output = x + x_summary
        return output


    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )