# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import dataclasses as dc

from dataclasses import dataclass
from dataclasses import astuple
from dataclasses import asdict

from dataclasses import make_dataclass


class BaseParameter:
    @property
    def nuisance_parameters(self):
        return dc.astuple(self)[:-1]

    @property
    def interest_parameters(self):
        return self.mu

    @property
    def parameter_names(self):
        return [field.name for field in dc.fields(self)]

    @property
    def nuisance_parameters_names(self):
        return [field.name for field in dc.fields(self)[:-1]]

    @property
    def interest_parameters_names(self):
        return "mu"

    def __iter__(self):
        return iter(dc.astuple(self))

    def clone_with(self, **changes):
        new_parameter = dc.replace(self, **changes)
        return new_parameter

    def __getitem__(self, key):
        return dc.asdict(self)[key]

    def items(self):
        return dc.asdict(self).items()

    def to_dict(self, prefix='', suffix=''):
        if not prefix and not suffix:
            return dc.asdict(self)
        else:
            return {prefix+key+suffix : value for key, value in self.items()}

    def to_tuple(self):
        return dc.astuple(self)



@dataclass(frozen=True)
class ParameterTes(BaseParameter):
    tes : float
    mu : float


@dataclass(frozen=True)
class ParameterJes(BaseParameter):
    jes : float
    mu : float


@dataclass(frozen=True)
class ParameterLes(BaseParameter):
    les : float
    mu : float


@dataclass(frozen=True)
class ParameterTesJes(BaseParameter):
    tes : float
    jes : float
    mu : float


@dataclass(frozen=True)
class ParameterTesLes(BaseParameter):
    tes : float
    les : float
    mu : float


@dataclass(frozen=True)
class ParameterTesJesLes(BaseParameter):
    tes : float
    jes : float
    les : float
    mu : float


# For backward compatibility
@dataclass(frozen=True)
class Parameter(BaseParameter):
    tes : float
    jes : float
    les : float
    mu : float


@dataclass(frozen=True)
class FuturParameter(BaseParameter):
    tes : float
    jes : float
    les : float
    nasty_bkg : float
    sigma_soft : float
    mu : float



ALL_PARAMETER_DICT = {
    'Tes' : ParameterTes,
    'Jes' : ParameterJes,
    'Les' : ParameterLes,
    'TesJes' : ParameterTesJes,
    'TesLes' : ParameterTesLes,
    'TesJesLes' : ParameterTesJesLes,
}


def get_parameter_class(tes=True, jes=False, les=False):
    key = ''
    if tes : key += 'Tes'
    if jes : key += 'Jes'
    if les : key += 'Les'

    if key in ALL_PARAMETER_DICT :
        return ALL_PARAMETER_DICT[key]
    else:
        raise ValueError(f"Nuisance parameter combination not implemented yet tes={tes}, jes={jes}, les={les}")
