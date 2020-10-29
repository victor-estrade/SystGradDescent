#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals


from .visual.common import make_common_plots
from .visual.common import make_hp_table
from ..loader import INFLoader
from .hyper_parameters import INF_HP


def main():
    print("hello")
    data_name = 'HIGGS'
    benchmark_name = 'HIGGS-prior'
    make_hp_table(data_name, benchmark_name, INF_HP, INFLoader)
    make_common_plots(data_name, benchmark_name, INF_HP, INFLoader)

    data_name = 'HIGGSTES'
    benchmark_name = 'HIGGSTES-prior'
    make_hp_table(data_name, benchmark_name, INF_HP, INFLoader)
    make_common_plots(data_name, benchmark_name, INF_HP, INFLoader)






if __name__ == '__main__':
    main()
