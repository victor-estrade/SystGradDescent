# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import torch
import torch.nn as nn

from . import layers
from .blocks import ResidualAverageBlock
from .blocks import ResidualBlock
from .blocks import softmax_cat

from .base import BaseArchi


"""
Fix residual net initialization
https://openreview.net/forum?id=H1gsz30cKX

Or use batch norm ?

---

Note :

E = Extra input
L = Linear
A = Average layer
S = Sum layer
M = Mean operation
R = Residual block
AR = Average Residual block


"""

class L1A2L1(BaseArchi):
    def __init__(self, n_in=1, n_out=1, n_unit=80):
        super().__init__(n_unit)
        self.fc_in  = nn.Linear(n_in, n_unit)
        self.avg1 = layers.Average(n_in, n_unit)
        self.avg2 = layers.Average(n_in, n_unit)
        self.fc_out = nn.Linear(n_unit, n_out)

    def forward(self, x):
        x = self.fc_in(x)
        x = torch.relu(x)
        x = self.avg1(x)
        x = torch.relu(x)
        x = self.avg2(x)
        x = torch.relu(x)
        x = self.fc_out(x)
        return x

    def reset_parameters(self):
        self.fc_in.reset_parameters()
        self.avg1.reset_parameters()
        self.avg2.reset_parameters()
        self.fc_out.reset_parameters()



class A1AR8L1(BaseArchi):
    def __init__(self, n_in=1, n_out=1, n_unit=80):
        super().__init__(n_unit)
        self.avg_in = layers.Average(n_in, n_unit)
        self.avg1   = ResidualAverageBlock(n_unit, n_unit//2)
        self.avg2   = ResidualAverageBlock(n_unit, n_unit//2)
        self.avg3   = ResidualAverageBlock(n_unit, n_unit//2)
        self.avg4   = ResidualAverageBlock(n_unit, n_unit//2)
        self.fc_out = nn.Linear(n_unit, n_out)

    def forward(self, x, w):
        x = self.avg_in(x, w)
        x = self.avg1(x, w)
        x = self.avg2(x, w)
        x = self.avg3(x)
        x = self.avg4(x)
        x = layers.relu6_tanh(x)
        x = self.fc_out(x)
        return x

    def reset_parameters(self):
        self.fc_in.reset_parameters()
        self.avg1.reset_parameters()
        self.avg2.reset_parameters()
        self.avg3.reset_parameters()
        self.avg4.reset_parameters()
        self.fc_out.reset_parameters()


class A1AR20L1(BaseArchi):
    def __init__(self, n_in=1, n_out=1, n_unit=80):
        super().__init__(n_unit)
        self.avg_in = layers.Average(n_in, n_unit)
        self.avg1   = ResidualAverageBlock(n_unit, n_unit//2)
        self.avg2   = ResidualAverageBlock(n_unit, n_unit//2)
        self.avg3   = ResidualAverageBlock(n_unit, n_unit//2)
        self.avg4   = ResidualAverageBlock(n_unit, n_unit//2)
        self.avg5   = ResidualAverageBlock(n_unit, n_unit//2)
        self.avg6   = ResidualAverageBlock(n_unit, n_unit//2)
        self.avg7   = ResidualAverageBlock(n_unit, n_unit//2)
        self.avg8   = ResidualAverageBlock(n_unit, n_unit//2)
        self.avg9   = ResidualAverageBlock(n_unit, n_unit//2)
        self.avg10  = ResidualAverageBlock(n_unit, n_unit//2)
        self.fc_out = nn.Linear(n_unit, n_out)

    def forward(self, x, w):
        x = self.avg_in(x, w)
        x = self.avg1(x, w)
        x = self.avg2(x, w)
        x = self.avg3(x, w)
        x = self.avg4(x, w)
        x = self.avg5(x, w)
        x = self.avg6(x, w)
        x = self.avg7(x, w)
        x = self.avg8(x, w)
        x = self.avg9(x, w)
        x = self.avg10(x, w)
        x = layers.relu6_tanh(x)
        x = self.fc_out(x)
        return x

    def reset_parameters(self):
        self.fc_in.reset_parameters()
        self.avg1.reset_parameters()
        self.avg2.reset_parameters()
        self.avg3.reset_parameters()
        self.avg4.reset_parameters()
        self.avg5.reset_parameters()
        self.avg6.reset_parameters()
        self.avg7.reset_parameters()
        self.avg8.reset_parameters()
        self.avg9.reset_parameters()
        self.avg10.reset_parameters()
        self.fc_out.reset_parameters()

