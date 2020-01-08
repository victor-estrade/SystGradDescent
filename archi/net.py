# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import torch
import torch.nn as nn

from .reduce import torch_weighted_mean
from .reduce import torch_weighted_sum
from .reduce import MeanBloc
from .reduce import SumBloc


class RegNet(nn.Module):
    def __init__(self, n_in=1, n_out=1, n_extra=0):
        super().__init__()
        N_UNITS = 80
        self.bloc1 = MeanBloc(n_in, N_UNITS, n_extra=n_extra)
        self.bloc2 = MeanBloc(N_UNITS, N_UNITS)
        self.bloc3 = MeanBloc(N_UNITS, N_UNITS)

        self.fc1 = nn.Linear(N_UNITS, N_UNITS)
        self.fc2 = nn.Linear(N_UNITS, N_UNITS)
        self.fc3 = nn.Linear(N_UNITS*2, N_UNITS)
        self.fc_out = nn.Linear(N_UNITS, n_out)
        
    def forward(self, x, w, p=None):
        x = self.bloc1(x, w, p)
        x = torch.relu(x)
        # x = self.fc1(x)
        # x = torch.softmax(x, 1)
        x = self.bloc2(x, w)
        x = torch.relu(x)

        x_ = self.fc2(x)
        x_ = torch.softmax(x_, 1)
        x_ = self.bloc3(x_, w)
        x_ = torch.relu(x_)
        x = torch.cat((x, x_), 1)

        x = torch_weighted_mean(x, w, 0, keepdim=False)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc_out(x)
        return x

    def reset_parameters(self):
        self.bloc1.reset_parameters()
        self.bloc2.reset_parameters()
        self.bloc3.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.fc_out.reset_parameters()
