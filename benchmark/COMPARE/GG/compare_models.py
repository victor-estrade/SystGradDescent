#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

from .hyper_parameters import DA_HP
from .hyper_parameters import GB_HP
from .hyper_parameters import INF_HP
from .hyper_parameters import NN_HP
from .hyper_parameters import PIVOT_HP
from .hyper_parameters import REG_HP
from .hyper_parameters import REG_M_HP
from .hyper_parameters import TP_HP

from ..loader import DALoader
from ..loader import GBLoader
from ..loader import INFLoader
from ..loader import NNLoader
from ..loader import PIVOTLoader
from ..loader import REGLoader
from ..loader import TPLoader

from .visual.common import hp_kwargs_generator

def main():
    print("hello")

    ALL_HP = [DA_HP, GB_HP, INF_HP, NN_HP, PIVOT_HP, REG_HP, TP_HP]
    ALL_LOADER = [DALoader, GBLoader, INFLoader, NNLoader, PIVOTLoader, REGLoader, TPLoader]
    data_name = 'GG'
    
    benchmark_name = 'GG-calib'
    for hp_args, TheLoader in zip(ALL_HP, ALL_LOADER):
        all_loader = [TheLoader(data_name, benchmark_name, **kwargs) for kwargs in hp_kwargs_generator(hp_args)]
        all_evaluation = [loader.load_evaluation_config() for loader in all_loader]


    benchmark_name = 'GG-prior'
    for hp_args, TheLoader in zip(ALL_HP, ALL_LOADER):
        all_loader = [TheLoader(data_name, benchmark_name, **kwargs) for kwargs in hp_kwargs_generator(hp_args)]
        all_evaluation = [loader.load_evaluation_config() for loader in all_loader]
            




if __name__ == '__main__':
    main()
