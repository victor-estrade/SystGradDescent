# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import torch
import torch.nn as nn

from . import layers


class Fully_3(nn.Module):
    __constants__ = ['n_unit']

    def __init__(self, n_unit):
        super().__init__()
        self.n_unit = n_unit
        self.fc1 = nn.Linear(n_unit, n_unit)
        self.fc2 = nn.Linear(n_unit, n_unit)
        self.fc3 = nn.Linear(n_unit, n_unit)

    def forward(self, x, w, p=None):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()


class FullyDense_3(nn.Module):
    __constants__ = ['n_unit']

    def __init__(self, n_unit):
        super().__init__()
        self.n_unit = n_unit
        self.fc1 = nn.Linear(n_unit, n_unit)
        self.fc2 = nn.Linear(n_unit, n_unit)
        self.fc3 = nn.Linear(n_unit, n_unit)

    def forward(self, x, w, p=None):
    	x_0 = x
        
        x_1 = self.fc1(x)
        x = x_0 + x_1
        x = torch.relu(x)

        x_2 = self.fc2(x)
        x = x_0 + x_1 + x_2
        x = torch.relu(x)

        x_3 = self.fc3(x)
        x = x_0 + x_1 + x_2 + x_3
        return x

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()


class Average_3(nn.Module):
    __constants__ = ['n_unit']

    def __init__(self, n_unit):
        super().__init__()
        self.n_unit = n_unit
        self.avg1 = layers.Average(n_unit, n_unit)
        self.avg2 = layers.Average(n_unit, n_unit)
        self.avg3 = layers.Average(n_unit, n_unit)

    def forward(self, x, w, p=None):
        x = self.avg1(x)
        x = torch.relu(x)
        x = self.avg2(x)
        x = torch.relu(x)
        x = self.avg3(x)
        return x

    def reset_parameters(self):
        self.avg1.reset_parameters()
        self.avg2.reset_parameters()
        self.avg3.reset_parameters()


class AverageDense_3(nn.Module):
    __constants__ = ['n_unit']

    def __init__(self, n_unit):
        super().__init__()
        self.n_unit = n_unit
        self.avg1 = layers.Average(n_unit, n_unit)
        self.avg2 = layers.Average(n_unit, n_unit)
        self.avg3 = layers.Average(n_unit, n_unit)

    def forward(self, x, w, p=None):
    	x_0 = x
        
        x_1 = self.avg1(x)
        x = x_0 + x_1
        x = torch.relu(x)

        x_2 = self.avg2(x)
        x = x_0 + x_1 + x_2
        x = torch.relu(x)

        x_3 = self.avg3(x)
        x = x_0 + x_1 + x_2 + x_3
        return x

    def reset_parameters(self):
        self.avg1.reset_parameters()
        self.avg2.reset_parameters()
        self.avg3.reset_parameters()


