# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

from dataclasses import dataclass
from dataclasses import astuple
from dataclasses import asdict

# from collections import namedtuple


# class Parameter(namedtuple('Parameter', ['tes', 'jes', 'les', 'mu'])):
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



# class FuturParameter(namedtuple('Parameter', ['tes', 'jes', 'les', 'nasty_bkg', 'sigma_soft', 'mu'])):
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
    tes : float
    jes : float
    les : float
    mu : float
    
    @property
    def nuisance_parameters(self):
        return (self.tes, self.jes, self.les)

    @property
    def interest_parameters(self):
        return self.mu

    @property
    def parameter_names(self):
        return ('tes', 'jes', 'les', 'mu')

    @property
    def nuisance_parameters_names(self):
        return ('tes', 'jes', 'les')

    @property
    def interest_parameters_names(self):
        return "mu"

    def __iter__(self):
        return iter(astuple(self))

    def clone_with(self, tes=None, jes=None, les=None, mu=None):
        tes = self.tes if tes is None else tes
        jes = self.jes if jes is None else jes
        les = self.les if les is None else les
        mu  = self.mu if mu is None else mu
        new_parameter = Parameter(tes, jes, les, mu)
        return new_parameter

    def __getitem__(self, key): 
        return asdict(self)[key]


@dataclass(frozen=True)
class FuturParameter:
    tes : float
    jes : float
    les : float
    nasty_bkg : float
    sigma_soft : float
    mu : float
    
    @property
    def nuisance_parameters(self):
        return (self.tes, self.jes, self.les, self.nasty_bkg, self.sigma_soft)

    @property
    def interest_parameters(self):
        return self.mu

    @property
    def parameter_names(self):
        return ('tes', 'jes', 'les', 'nasty_bkg', 'sigma_soft', 'mu')

    @property
    def nuisance_parameters_names(self):
        return ('tes', 'jes', 'les', 'nasty_bkg', 'sigma_soft')

    @property
    def interest_parameters_names(self):
        return "mu"

    def __iter__(self):
        return iter(astuple(self))

    def clone_with(self, tes=None, jes=None, les=None, nasty_bkg=None, sigma_soft=None, mu=None):
        tes = self.tes if tes is None else tes
        jes = self.jes if jes is None else jes
        les = self.les if les is None else les
        nasty_bkg = self.nasty_bkg if nasty_bkg is None else nasty_bkg
        sigma_soft = self.sigma_soft if sigma_soft is None else sigma_soft
        mu  = self.mu if mu is None else mu
        new_parameter = Parameter(tes, jes, les, nasty_bkg, sigma_soft, mu)
        return new_parameter

    def __getitem__(self, key): 
        return asdict(self)[key]

