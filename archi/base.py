# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import torch
import torch.nn as nn

class BaseArchi(nn.Module):
    def __init__(self, n_unit=80):
        super().__init__()
        self.name = "{}x{:d}".format(self.__class__.__name__, n_unit)
