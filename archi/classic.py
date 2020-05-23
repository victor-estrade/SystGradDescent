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

class L4(BaseArchi):
    def __init__(self, n_in=1, n_out=1, n_unit=80):
        super().__init__(n_unit)
        self.fc_in  = nn.Linear(n_in, n_unit)
        self.fc1    = nn.Linear(n_unit, n_unit)
        self.fc2    = nn.Linear(n_unit, n_unit)
        self.fc_out = nn.Linear(n_unit, n_out)

    def forward(self, x):
        x = self.fc_in(x)
        x = torch.relu(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc_out(x)
        return x

    def reset_parameters(self):
        self.fc_in.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc_out.reset_parameters()



class L1R8L1(BaseArchi):
    def __init__(self, n_in=1, n_out=1, n_unit=80):
        super().__init__(n_unit)
        self.fc_in  = nn.Linear(n_in, n_unit)
        self.res1   = ResidualBlock(n_unit, n_unit//2)
        self.res2   = ResidualBlock(n_unit, n_unit//2)
        self.res3   = ResidualBlock(n_unit, n_unit//2)
        self.res4   = ResidualBlock(n_unit, n_unit//2)
        self.fc_out = nn.Linear(n_unit, n_out)

    def forward(self, x):
        x = self.fc_in(x)
        x = torch.relu(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = layers.relu6_tanh(x)
        x = self.fc_out(x)
        return x

    def reset_parameters(self):
        self.fc_in.reset_parameters()
        self.res1.reset_parameters()
        self.res2.reset_parameters()
        self.res3.reset_parameters()
        self.res4.reset_parameters()
        self.fc_out.reset_parameters()



class L1R20L1(BaseArchi):
    def __init__(self, n_in=1, n_out=1, n_unit=80):
        super().__init__(n_unit)
        self.fc_in  = nn.Linear(n_in, n_unit)
        self.res1   = ResidualBlock(n_unit, n_unit//2)
        self.res2   = ResidualBlock(n_unit, n_unit//2)
        self.res3   = ResidualBlock(n_unit, n_unit//2)
        self.res4   = ResidualBlock(n_unit, n_unit//2)
        self.res5   = ResidualBlock(n_unit, n_unit//2)
        self.res6   = ResidualBlock(n_unit, n_unit//2)
        self.res7   = ResidualBlock(n_unit, n_unit//2)
        self.res8   = ResidualBlock(n_unit, n_unit//2)
        self.res9   = ResidualBlock(n_unit, n_unit//2)
        self.res10  = ResidualBlock(n_unit, n_unit//2)
        self.fc_out = nn.Linear(n_unit, n_out)

    def forward(self, x):
        x = self.fc_in(x)
        x = torch.relu(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.res7(x)
        x = self.res8(x)
        x = self.res9(x)
        x = self.res10(x)
        x = layers.relu6_tanh(x)
        x = self.fc_out(x)
        return x

    def reset_parameters(self):
        self.fc_in.reset_parameters()
        self.res1.reset_parameters()
        self.res2.reset_parameters()
        self.res3.reset_parameters()
        self.res4.reset_parameters()
        self.res5.reset_parameters()
        self.res6.reset_parameters()
        self.res7.reset_parameters()
        self.res8.reset_parameters()
        self.res9.reset_parameters()
        self.res10.reset_parameters()
        self.fc_out.reset_parameters()



