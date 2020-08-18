# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from dataclasses import dataclass
from dataclasses import astuple
from dataclasses import asdict

# from collections import namedtuple


# class Parameter(namedtuple('Parameter', ['r', 'lam', 'mu'])):
#     @property
#     def nuisance_parameters(self):
#         return self[:-1]

#     @property
#     def interest_parameters(self):
#         return self[-1]

#     @property
#     def parameter_names(self):
#         return self._fields

#     @property
#     def nuisance_parameters_names(self):
#         return self._fields[:-1]

#     @property
#     def interest_parameters_names(self):
#         return self._fields[-1]

@dataclass(frozen=True)
class Parameter:
    r : float
    lam : float
    mu : float
    
    @property
    def nuisance_parameters(self):
        return (self.r, self.lam)

    @property
    def interest_parameters(self):
        return self.mu

    @property
    def parameter_names(self):
        return ("r", "lam", "mu")

    @property
    def nuisance_parameters_names(self):
        return ("r", "lam",)

    @property
    def interest_parameters_names(self):
        return "mu"

    def __iter__(self):
        return iter(astuple(self))

    def clone_with(self, r=None, lam=None, mu=None):
        r = self.r if r is None else r
        lam = self.lam if lam is None else lam
        mu = self.mu if mu is None else mu
        new_parameter = Parameter(r, lam, mu)
        return new_parameter

    def __getitem__(self, key): 
        return asdict(self)[key]

