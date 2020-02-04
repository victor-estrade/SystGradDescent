# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import torch
import torch.nn as nn

from . import layers

class RegNet(nn.Module):
    def __init__(self, n_in=1, n_out=1):
        super().__init__()
        N_UNITS = 80
        self.bloc1 = layers.Average(n_in, N_UNITS)
        self.bloc2 = layers.Average(N_UNITS, N_UNITS)
        self.bloc3 = layers.Average(N_UNITS, N_UNITS)

        self.fc1 = nn.Linear(N_UNITS, N_UNITS)
        self.fc2 = nn.Linear(N_UNITS, N_UNITS)
        self.fc3 = nn.Linear(N_UNITS*2, N_UNITS)
        self.fc_out = nn.Linear(N_UNITS, n_out)

    def forward(self, x, w, p=None):
        x = self.bloc1(x, w)
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

        x = layers.torch_weighted_mean(x, w, 0, keepdim=False)
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


class RegNetExtra(nn.Module):
    def __init__(self, n_in=1, n_out=1, n_extra=0):
        super().__init__()
        N_UNITS = 80
        self.bloc1 = layers.AverageExtra(n_in, N_UNITS, n_extra)
        self.bloc2 = layers.Average(N_UNITS, N_UNITS)
        self.bloc3 = layers.Average(N_UNITS, N_UNITS)

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

        x = layers.torch_weighted_mean(x, w, 0, keepdim=False)
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


class FullyClassifier_1x3(nn.Module):
    def __init__(self, n_in=1, n_out=1, n_unit=80):
        super().__init__()
        self.fc_in = nn.Linear(n_in, n_unit)
        self.fc_out = nn.Linear(n_unit, n_out)


class FullyDenseClassifier_1x3(nn.Module):
    def __init__(self, n_in=1, n_out=1, n_unit=80):
        super().__init__()
        self.fc_in = nn.Linear(n_in, n_unit)
        self.fc_out = nn.Linear(n_unit, n_out)


class AverageClassifier_1x3(nn.Module):
    def __init__(self, n_in=1, n_out=1, n_unit=80):
        super().__init__()
        self.avg_in = layers.Average(n_in, n_unit)
        self.fc_out = nn.Linear(n_unit, n_out)


class AverageDenseClassifier_1x3(nn.Module):
    def __init__(self, n_in=1, n_out=1, n_unit=80):
        super().__init__()
        self.avg_in = layers.Average(n_in, n_unit)
        self.fc_out = nn.Linear(n_unit, n_out)




class FullyReducer_1x3(nn.Module):
    def __init__(self, n_in=1, n_out=1, n_unit=80):
        super().__init__()
        self.fc_in = nn.Linear(n_in, n_unit)
        self.fc_out = nn.Linear(n_unit, n_out)


class FullyDenseReducer_1x3(nn.Module):
    def __init__(self, n_in=1, n_out=1, n_unit=80):
        super().__init__()
        self.fc_in = nn.Linear(n_in, n_unit)
        self.fc_out = nn.Linear(n_unit, n_out)


class AverageReducer_1x3(nn.Module):
    def __init__(self, n_in=1, n_out=1, n_unit=80):
        super().__init__()
        self.avg_in = layers.Average(n_in, n_unit)
        self.fc_out = nn.Linear(n_unit, n_out)


class AverageDenseReducer_1x3(nn.Module):
    def __init__(self, n_in=1, n_out=1, n_unit=80):
        super().__init__()
        self.avg_in = layers.Average(n_in, n_unit)
        self.fc_out = nn.Linear(n_unit, n_out)


