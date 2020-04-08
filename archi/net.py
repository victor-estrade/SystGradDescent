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

class BaseArchi(nn.Module):
    def __init__(self, n_unit=80):
        super().__init__()
        self.name = "{}x{:d}".format(self.__class__.__name__, n_unit)


class F6(BaseArchi):
    def __init__(self, n_in=1, n_out=1, n_unit=80):
        super().__init__(n_unit)
        self.fc_in  = nn.Linear(n_in, n_unit)
        self.fc1    = nn.Linear(n_unit, n_unit)
        self.fc2    = nn.Linear(n_unit, n_unit)
        self.fc3    = nn.Linear(n_unit, n_unit)
        self.fc4    = nn.Linear(n_unit, n_unit)
        self.fc_out = nn.Linear(n_unit, n_out)

    def forward(self, x):
        x = self.fc_in(x)
        x = torch.relu(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)

        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        x = torch.relu(x)
        x = self.fc_out(x)
        return x

    def reset_parameters(self):
        self.fc_in.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.fc4.reset_parameters()
        self.fc_out.reset_parameters()


class RegNet(BaseArchi):
    def __init__(self, n_in=1, n_out=1):
        N_UNITS = 80
        super().__init__(N_UNITS)
        self.avg1 = layers.Average(n_in, N_UNITS)
        self.avg2 = layers.Average(N_UNITS, N_UNITS)
        self.avg3 = layers.Average(N_UNITS, N_UNITS)

        self.fc1 = nn.Linear(N_UNITS, N_UNITS)
        self.fc2 = nn.Linear(N_UNITS, N_UNITS)
        self.fc3 = nn.Linear(N_UNITS*2, N_UNITS)
        self.fc_out = nn.Linear(N_UNITS, n_out)

    def forward(self, x, w, p=None):
        x = self.avg1(x, w)
        x = torch.relu(x)
        # x = self.fc1(x)
        # x = torch.softmax(x, 1)
        x = self.avg2(x, w)
        x = torch.relu(x)

        x_ = self.fc2(x)
        x_ = torch.softmax(x_, 1)
        x_ = self.avg3(x_, w)
        x_ = torch.relu(x_)
        x = torch.cat((x, x_), 1)

        x = layers.torch_weighted_mean(x, w, 0, keepdim=False)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc_out(x)
        return x

    def reset_parameters(self):
        self.avg1.reset_parameters()
        self.avg2.reset_parameters()
        self.avg3.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.fc_out.reset_parameters()


class RegNetExtra(BaseArchi):
    def __init__(self, n_in=1, n_out=1, n_extra=0):
        N_UNITS = 80
        super().__init__(N_UNITS)
        self.avg1 = layers.AverageExtra(n_in, N_UNITS, n_extra)
        self.avg2 = layers.Average(N_UNITS, N_UNITS)
        self.avg3 = layers.Average(N_UNITS, N_UNITS)

        self.fc1 = nn.Linear(N_UNITS, N_UNITS)
        self.fc2 = nn.Linear(N_UNITS, N_UNITS)
        self.fc3 = nn.Linear(N_UNITS*2, N_UNITS)
        self.fc_out = nn.Linear(N_UNITS, n_out)
        
    def forward(self, x, w, p):
        x = self.avg1(x, w, p)
        x = torch.relu(x)
        # x = self.fc1(x)
        # x = torch.softmax(x, 1)
        x = self.avg2(x, w)
        x = torch.relu(x)

        x_ = self.fc2(x)
        x_ = torch.softmax(x_, 1)
        x_ = self.avg3(x_, w)
        x_ = torch.relu(x_)
        x = torch.cat((x, x_), 1)

        x = layers.torch_weighted_mean(x, w, 0, keepdim=False)
        x = self.fc3(x)
        x = torch.relu(x)

        x = self.fc_out(x)
        return x

    def reset_parameters(self):
        self.avg1.reset_parameters()
        self.avg2.reset_parameters()
        self.avg3.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.fc_out.reset_parameters()


class AR9R1(BaseArchi):
    def __init__(self, n_in=1, n_out=1, n_unit=80):
        super().__init__(n_unit)
        self.avg_in  = layers.Average(n_in, n_unit)
        self.avg1   = ResidualAverageBlock(n_unit, n_unit//2)
        self.avg2   = ResidualAverageBlock(n_unit, n_unit//2)
        self.avg3   = ResidualAverageBlock(n_unit, n_unit//2)
        self.avg4   = ResidualAverageBlock(n_unit, n_unit//2)
        self.fc_out = nn.Linear(n_unit, n_out)

    def forward(self, x, w, p=None):
        x = self.avg_in(x, w)
        x = self.avg1(x, w)
        x = self.avg2(x, w)
        x = self.avg3(x, w)
        x = self.avg4(x, w)

        x = layers.torch_weighted_mean(x, w, 0, keepdim=False)
        x = self.fc_out(x)
        return x

    def reset_parameters(self):
        self.avg_in.reset_parameters()
        self.avg1.reset_parameters()
        self.avg2.reset_parameters()
        self.avg3.reset_parameters()
        self.avg4.reset_parameters()
        self.fc_out.reset_parameters()


class AR5R5(BaseArchi):
    def __init__(self, n_in=1, n_out=1, n_unit=80):
        super().__init__(n_unit)
        self.avg_in = layers.Average(n_in, n_unit)
        self.avg1   = ResidualAverageBlock(n_unit, n_unit//2)
        self.avg2   = ResidualAverageBlock(n_unit, n_unit//2)
        self.res3   = ResidualBlock       (n_unit, n_unit//2)
        self.res4   = ResidualBlock       (n_unit, n_unit//2)
        self.fc_out = nn.Linear(n_unit, n_out)

    def forward(self, x, w, p=None):
        x = self.avg_in(x, w)
        x = self.avg1(x, w)
        x = self.avg2(x, w)

        x = layers.torch_weighted_mean(x, w, 0, keepdim=False)
        x = self.res3(x)
        x = self.res4(x)
        x = layers.relu_tanh(x)
        x = self.fc_out(x)
        return x

    def reset_parameters(self):
        self.avg_in.reset_parameters()
        self.avg1.reset_parameters()
        self.avg2.reset_parameters()
        self.res3.reset_parameters()
        self.res4.reset_parameters()
        self.fc_out.reset_parameters()


class AR5R5E(BaseArchi):
    def __init__(self, n_in=1, n_out=1, n_extra=0, n_unit=80):
        super().__init__(n_unit)
        self.avg_in = layers.AverageExtra(n_in, n_unit, n_extra)
        self.avg1   = ResidualAverageBlock(n_unit, n_unit//2)
        self.avg2   = ResidualAverageBlock(n_unit, n_unit//2)
        self.res3   = ResidualBlock       (n_unit, n_unit//2)
        self.res4   = ResidualBlock       (n_unit, n_unit//2)
        self.fc_out = nn.Linear(n_unit, n_out)

    def forward(self, x, w, p):
        x = self.avg_in(x, w, p)
        x = self.avg1(x, w)
        x = self.avg2(x, w)

        x = layers.torch_weighted_mean(x, w, 0, keepdim=False)
        x = self.res3(x)
        x = self.res4(x)
        x = layers.relu_tanh(x)
        x = self.fc_out(x)
        return x

    def reset_parameters(self):
        self.avg_in.reset_parameters()
        self.avg1.reset_parameters()
        self.avg2.reset_parameters()
        self.res3.reset_parameters()
        self.res4.reset_parameters()
        self.fc_out.reset_parameters()

class AR9R9E(BaseArchi):
    def __init__(self, n_in=1, n_out=1, n_extra=0, n_unit=80):
        super().__init__(n_unit)
        self.avg_in = layers.AverageExtra(n_in, n_unit, n_extra)
        self.avg1   = ResidualAverageBlock(n_unit, n_unit//2)
        self.avg2   = ResidualAverageBlock(n_unit, n_unit//2)
        self.avg3   = ResidualAverageBlock(n_unit, n_unit//2)
        self.avg4   = ResidualAverageBlock(n_unit, n_unit//2)
        self.res5   = ResidualBlock       (n_unit, n_unit//2)
        self.res6   = ResidualBlock       (n_unit, n_unit//2)
        self.res7   = ResidualBlock       (n_unit, n_unit//2)
        self.res8   = ResidualBlock       (n_unit, n_unit//2)
        self.fc_out = nn.Linear(n_unit, n_out)

    def forward(self, x, w, p):
        x = self.avg_in(x, w, p)
        x = self.avg1(x, w)
        x = self.avg2(x, w)
        x = self.avg3(x, w)
        x = self.avg4(x, w)

        x = layers.torch_weighted_mean(x, w, 0, keepdim=False)
        x = self.res5(x)
        x = self.res6(x)
        x = self.res7(x)
        x = self.res8(x)
        x = layers.relu_tanh(x)
        x = self.fc_out(x)
        return x

    def reset_parameters(self):
        self.avg_in.reset_parameters()
        self.avg1.reset_parameters()
        self.avg2.reset_parameters()
        self.avg3.reset_parameters()
        self.avg4.reset_parameters()
        self.res5.reset_parameters()
        self.res6.reset_parameters()
        self.res7.reset_parameters()
        self.res8.reset_parameters()
        self.fc_out.reset_parameters()

class AR9R9(BaseArchi):
    def __init__(self, n_in=1, n_out=1, n_extra=0, n_unit=80):
        super().__init__(n_unit)
        self.avg_in = layers.Average(n_in, n_unit)
        self.avg1   = ResidualAverageBlock(n_unit, n_unit//2)
        self.avg2   = ResidualAverageBlock(n_unit, n_unit//2)
        self.avg3   = ResidualAverageBlock(n_unit, n_unit//2)
        self.avg4   = ResidualAverageBlock(n_unit, n_unit//2)
        self.res5   = ResidualBlock       (n_unit, n_unit//2)
        self.res6   = ResidualBlock       (n_unit, n_unit//2)
        self.res7   = ResidualBlock       (n_unit, n_unit//2)
        self.res8   = ResidualBlock       (n_unit, n_unit//2)
        self.fc_out = nn.Linear(n_unit, n_out)

    def forward(self, x, w, p):
        x = self.avg_in(x, w)
        x = self.avg1(x, w)
        x = self.avg2(x, w)
        x = self.avg3(x, w)
        x = self.avg4(x, w)

        x = layers.torch_weighted_mean(x, w, 0, keepdim=False)
        x = self.res5(x)
        x = self.res6(x)
        x = self.res7(x)
        x = self.res8(x)
        x = layers.relu_tanh(x)
        x = self.fc_out(x)
        return x

    def reset_parameters(self):
        self.avg_in.reset_parameters()
        self.avg1.reset_parameters()
        self.avg2.reset_parameters()
        self.avg3.reset_parameters()
        self.avg4.reset_parameters()
        self.res5.reset_parameters()
        self.res6.reset_parameters()
        self.res7.reset_parameters()
        self.res8.reset_parameters()
        self.fc_out.reset_parameters()


class AR19R5E(BaseArchi):
    def __init__(self, n_in=1, n_out=1, n_extra=0, n_unit=80):
        super().__init__(n_unit)
        self.avg_in = layers.AverageExtra(n_in, n_unit, n_extra)
        self.avg1   = ResidualAverageBlock(n_unit, n_unit//2)
        self.avg2   = ResidualAverageBlock(n_unit, n_unit//2)
        self.avg3   = ResidualAverageBlock(n_unit, n_unit//2)
        self.avg4   = ResidualAverageBlock(n_unit, n_unit//2)
        self.avg5   = ResidualAverageBlock(n_unit, n_unit//2)
        self.avg6   = ResidualAverageBlock(n_unit, n_unit//2)
        self.avg7   = ResidualAverageBlock(n_unit, n_unit//2)
        self.avg8   = ResidualAverageBlock(n_unit, n_unit//2)
        self.avg9   = ResidualAverageBlock(n_unit, n_unit//2)
        self.res10  = ResidualBlock       (n_unit, n_unit//2)
        self.res11  = ResidualBlock       (n_unit, n_unit//2)
        self.fc_out = nn.Linear(n_unit, n_out)

    def forward(self, x, w, p):
        x = self.avg_in(x, w, p)
        x = self.avg1(x, w)
        x = self.avg2(x, w)
        x = self.avg3(x, w)
        x = self.avg4(x, w)
        x = self.avg5(x, w)
        x = self.avg6(x, w)
        x = self.avg7(x, w)
        x = self.avg8(x, w)
        x = self.avg9(x, w)

        x = layers.torch_weighted_mean(x, w, 0, keepdim=False)
        x = self.res10(x)
        x = self.res11(x)
        x = layers.relu_tanh(x)
        x = self.fc_out(x)
        return x

    def reset_parameters(self):
        self.avg_in.reset_parameters()
        self.avg1.reset_parameters()
        self.avg2.reset_parameters()
        self.avg3.reset_parameters()
        self.avg4.reset_parameters()
        self.avg5.reset_parameters()
        self.avg6.reset_parameters()
        self.avg7.reset_parameters()
        self.avg8.reset_parameters()
        self.avg9.reset_parameters()
        self.res10.reset_parameters()
        self.res11.reset_parameters()
        self.fc_out.reset_parameters()


class AR19R5(BaseArchi):
    def __init__(self, n_in=1, n_out=1, n_extra=0, n_unit=80):
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
        self.res10  = ResidualBlock       (n_unit, n_unit//2)
        self.res11  = ResidualBlock       (n_unit, n_unit//2)
        self.fc_out = nn.Linear(n_unit, n_out)

    def forward(self, x, w, p):
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

        x = layers.torch_weighted_mean(x, w, 0, keepdim=False)
        x = self.res10(x)
        x = self.res11(x)
        x = layers.relu_tanh(x)
        x = self.fc_out(x)
        return x

    def reset_parameters(self):
        self.avg_in.reset_parameters()
        self.avg1.reset_parameters()
        self.avg2.reset_parameters()
        self.avg3.reset_parameters()
        self.avg4.reset_parameters()
        self.avg5.reset_parameters()
        self.avg6.reset_parameters()
        self.avg7.reset_parameters()
        self.avg8.reset_parameters()
        self.avg9.reset_parameters()
        self.res10.reset_parameters()
        self.res11.reset_parameters()
        self.fc_out.reset_parameters()



class AF3R3(BaseArchi):
    def __init__(self, n_in=1, n_out=1, n_unit=80):
        super().__init__(n_unit)
        self.avg_in = layers.Average(n_in,   n_unit)
        self.avg1   = layers.Average(n_unit, n_unit)
        self.avg2   = layers.Average(n_unit, n_unit)
        self.fc3    = nn.Linear(n_unit, n_unit)
        self.fc4    = nn.Linear(n_unit, n_unit)
        self.fc_out = nn.Linear(n_unit, n_out)

    def forward(self, x, w, p=None):
        x = self.avg_in(x, w)
        x = torch.relu(x)
        x = self.avg1(x, w)
        x = torch.relu(x)
        x = self.avg2(x, w)

        x = layers.torch_weighted_mean(x, w, 0, keepdim=False)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        x = torch.relu(x)
        x = self.fc_out(x)
        return x

    def reset_parameters(self):
        self.avg_in.reset_parameters()
        self.avg1.reset_parameters()
        self.avg2.reset_parameters()
        self.fc3.reset_parameters()
        self.fc4.reset_parameters()
        self.fc_out.reset_parameters()


class AF3R3E(BaseArchi):
    def __init__(self, n_in=1, n_out=1, n_extra=0, n_unit=80):
        super().__init__(n_unit)
        self.avg_in = layers.AverageExtra(n_in, n_unit, n_extra)
        self.avg1   = layers.Average(n_unit, n_unit)
        self.avg2   = layers.Average(n_unit, n_unit)
        self.fc3    = nn.Linear(n_unit, n_unit)
        self.fc4    = nn.Linear(n_unit, n_unit)
        self.fc_out = nn.Linear(n_unit, n_out)

    def forward(self, x, w, p):
        x = self.avg_in(x, w, p)
        x = torch.relu(x)
        x = self.avg1(x, w)
        x = torch.relu(x)
        x = self.avg2(x, w)

        x = layers.torch_weighted_mean(x, w, 0, keepdim=False)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        x = torch.relu(x)
        x = self.fc_out(x)
        return x

    def reset_parameters(self):
        self.avg_in.reset_parameters()
        self.avg1.reset_parameters()
        self.avg2.reset_parameters()
        self.fc3.reset_parameters()
        self.fc4.reset_parameters()
        self.fc_out.reset_parameters()


class F3R3(BaseArchi):
    def __init__(self, n_in=1, n_out=1, n_unit=80):
        super().__init__(n_unit)
        self.fc_in  = nn.Linear(n_in, n_unit)
        self.fc1    = nn.Linear(n_unit, n_unit)
        self.fc2    = nn.Linear(n_unit, n_unit)
        self.fc3    = nn.Linear(n_unit, n_unit)
        self.fc4    = nn.Linear(n_unit, n_unit)
        self.fc_out = nn.Linear(n_unit, n_out)

    def forward(self, x, w, p=None):
        x = self.fc_in(x)
        x = torch.relu(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)

        x = layers.torch_weighted_mean(x, w, 0, keepdim=False)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        x = torch.relu(x)
        x = self.fc_out(x)
        return x

    def reset_parameters(self):
        self.fc_in.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.fc4.reset_parameters()
        self.fc_out.reset_parameters()


class F3R3E(BaseArchi):
    def __init__(self, n_in=1, n_out=1, n_unit=80):
        super().__init__(n_unit)
        self.fc_in  = nn.Linear(n_in, n_unit)
        self.fc1    = nn.Linear(n_unit, n_unit)
        self.fc2    = nn.Linear(n_unit, n_unit)
        self.fc3    = nn.Linear(n_unit, n_unit)
        self.fc4    = nn.Linear(n_unit, n_unit)
        self.fc_out = nn.Linear(n_unit, n_out)

    def forward(self, x, w, p):
        x = self.fc_in(x)
        x = torch.relu(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)

        x = layers.torch_weighted_mean(x, w, 0, keepdim=False)
        x = torch.cat((x, p), 1)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        x = torch.relu(x)
        x = self.fc_out(x)
        return x

    def reset_parameters(self):
        self.fc_in.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.fc4.reset_parameters()
        self.fc_out.reset_parameters()



class AR5S2S2R1(BaseArchi):
    def __init__(self, n_in=1, n_out=1, n_unit=80):
        super().__init__(n_unit)
        self.avg_in = layers.Average(n_in, n_unit)
        self.avg1   = ResidualAverageBlock(n_unit, n_unit//2)
        self.avg2   = ResidualAverageBlock(n_unit, n_unit//2)
        self.avg3   = ResidualAverageBlock(n_unit, n_unit//2)
        self.avg4   = ResidualAverageBlock(n_unit+n_unit, n_unit//2)
        self.fc_out = nn.Linear(n_unit+n_unit, n_out)

    def forward(self, x, w, p=None):
        x = self.avg_in(x, w)
        x = self.avg1(x, w)
        x = self.avg2(x, w)

        x_ = self.avg3(x, w)
        x = softmax_cat(x_, x)
        x = self.avg4(x, w)
        
        x = layers.torch_weighted_mean(x, w, 0, keepdim=False)
        x = self.fc_out(x)
        return x

    def reset_parameters(self):
        self.avg_in.reset_parameters()
        self.avg1.reset_parameters()
        self.avg2.reset_parameters()
        self.avg3.reset_parameters()
        self.avg4.reset_parameters()
        self.fc_out.reset_parameters()


class AR3S2S2R3(BaseArchi):
    def __init__(self, n_in=1, n_out=1, n_unit=200):
        super().__init__(n_unit)
        self.avg_in = layers.Average(n_in, n_unit)
        self.avg1   = ResidualAverageBlock(n_unit, n_unit//2)
        self.avg2   = ResidualAverageBlock(n_unit, n_unit//2)
        self.avg3   = ResidualAverageBlock(n_unit+n_unit, n_unit//2)
        self.res4   = ResidualBlock       (n_unit+n_unit, n_unit//2)
        self.fc_out = nn.Linear(n_unit+n_unit, n_out)

    def forward(self, x, w, p=None):
        x = self.avg_in(x)
        x = self.avg1(x, w)

        x_ = self.avg2(x, w)
        x = softmax_cat(x_, x)
        
        x = self.avg3(x, w)
        
        x = layers.torch_weighted_mean(x, w, 0, keepdim=False)
        x = self.res4(x)
        x = self.fc_out(x)
        return x

    def reset_parameters(self):
        self.avg_in.reset_parameters()
        self.avg1.reset_parameters()
        self.avg2.reset_parameters()
        self.avg3.reset_parameters()
        self.avg4.reset_parameters()
        self.fc_out.reset_parameters()
