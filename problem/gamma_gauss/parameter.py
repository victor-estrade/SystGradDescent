# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from dataclasses import dataclass
from dataclasses import astuple
from dataclasses import asdict

# from collections import namedtuple

# class Parameter(namedtuple('Parameter', ['rescale', 'mix'])):
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
    rescale : float
    mix : float
    
    @property
    def nuisance_parameters(self):
        return (self.rescale,)

    @property
    def interest_parameters(self):
        return self.mix

    @property
    def parameter_names(self):
        return ("rescale", "mix")

    @property
    def nuisance_parameters_names(self):
        return ("rescale",)

    @property
    def interest_parameters_names(self):
        return "mix"

    def __iter__(self):
        return iter(astuple(self))

    def __add__(self, other):
        rescale = self.rescale + other.rescale
        mix = self.mix + other.mix
        return Parameter(rescale, mix)

    def clone_with(self, rescale=None, mix=None):
        rescale = self.rescale if rescale is None else rescale
        mix = self.mix if mix is None else mix
        new_parameter = Parameter(rescale, mix)
        return new_parameter

    def __getitem__(self, key): 
        return asdict(self)[key]

    def items(self):
        return asdict(self).items()

    def to_dict(self, prefix='', suffix=''):
        if not prefix and not suffix:
            return asdict(self)
        else:
            return {prefix+key+suffix : value for key, value in self.items()}

    def to_tuple(self):
        return astuple(self)
