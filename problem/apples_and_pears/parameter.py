# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from collections import namedtuple


class Parameter(namedtuple('Parameter', ['apple_ratio'])):
    @property
    def nuisance_parameters(self):
        return None

    @property
    def interest_parameters(self):
        return self[-1]

    @property
    def parameter_names(self):
        return self._fields

    @property
    def nuisance_parameters_names(self):
        return None

    @property
    def interest_parameters_names(self):
        return self._fields[-1]

