#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

# Command line
# python -m benchmark.COMPARE.HIGGS.GB

from .visual.common import make_common_plots
from .visual.common import make_hp_table
from ..loader import GBLoader
from .hyper_parameters import GB_HP


def main():
    print("hello")
    # data_name = 'HIGGS'
    # benchmark_name = 'HIGGS-prior'
    # make_hp_table(data_name, benchmark_name, GB_HP, GBLoader)
    # make_common_plots(data_name, benchmark_name, GB_HP, GBLoader)

    data_name = 'HIGGSTES'
    benchmark_name = 'HIGGSTES-prior'
    make_hp_table(data_name, benchmark_name, GB_HP, GBLoader)
    make_common_plots(data_name, benchmark_name, GB_HP, GBLoader)

    data_name = 'BALANCEDHIGGSTES'
    benchmark_name = 'BALANCEDHIGGSTES-prior'
    make_hp_table(data_name, benchmark_name, GB_HP, GBLoader)
    make_common_plots(data_name, benchmark_name, GB_HP, GBLoader)

    data_name = 'EASYHIGGSTES'
    benchmark_name = 'EASYHIGGSTES-prior'
    make_hp_table(data_name, benchmark_name, GB_HP, GBLoader)
    make_common_plots(data_name, benchmark_name, GB_HP, GBLoader)

    data_name = 'HIGGSTES'
    benchmark_name = 'HIGGSTES-calib'
    make_hp_table(data_name, benchmark_name, GB_HP, GBLoader)
    make_common_plots(data_name, benchmark_name, GB_HP, GBLoader)

    data_name = 'BALANCEDHIGGSTES'
    benchmark_name = 'BALANCEDHIGGSTES-calib'
    make_hp_table(data_name, benchmark_name, GB_HP, GBLoader)
    make_common_plots(data_name, benchmark_name, GB_HP, GBLoader)

    data_name = 'EASYHIGGSTES'
    benchmark_name = 'EASYHIGGSTES-calib'
    make_hp_table(data_name, benchmark_name, GB_HP, GBLoader)
    make_common_plots(data_name, benchmark_name, GB_HP, GBLoader)





if __name__ == '__main__':
    main()
