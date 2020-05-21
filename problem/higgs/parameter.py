# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

from collections import namedtuple


class Parameter(namedtuple('Parameter', ['tes', 'jes', 'les', 'nasty_bkg', 'sigma_soft', 'mu'])):
    @property
    def nuisance_parameters(self):
        return self[:-1]

    @property
    def interest_parameters(self):
        return self[-1]

    @property
    def parameter_names(self):
        return self._fields

    @property
    def nuisance_parameters_names(self):
        return self._fields[:-1]

    @property
    def interest_parameters_names(self):
        return self._fields[-1]

