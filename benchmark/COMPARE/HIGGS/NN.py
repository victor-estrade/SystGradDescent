#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

# Command line
# python -m benchmark.COMPARE.HIGGS.NN

from .visual.common import make_common_plots
from .visual.common import make_hp_table
from ..loader import NNLoader
from .hyper_parameters import NN_HP


def main():
    print("hello")
    # data_name = 'HIGGS'
    # benchmark_name = 'HIGGS-prior'
    # make_hp_table(data_name, benchmark_name, NN_HP, NNLoader)
    # make_common_plots(data_name, benchmark_name, NN_HP, NNLoader)

    data_name = 'HIGGSTES'
    benchmark_name = 'HIGGSTES-prior'
    make_hp_table(data_name, benchmark_name, NN_HP, NNLoader)
    make_common_plots(data_name, benchmark_name, NN_HP, NNLoader)

    # data_name = 'BALANCEDHIGGSTES'
    # benchmark_name = 'BALANCEDHIGGSTES-prior'
    # make_hp_table(data_name, benchmark_name, NN_HP, NNLoader)
    # make_common_plots(data_name, benchmark_name, NN_HP, NNLoader)
    #
    # data_name = 'EASYHIGGSTES'
    # benchmark_name = 'EASYHIGGSTES-prior'
    # make_hp_table(data_name, benchmark_name, NN_HP, NNLoader)
    # make_common_plots(data_name, benchmark_name, NN_HP, NNLoader)
    #
    # data_name = 'HIGGSTES'
    # benchmark_name = 'HIGGSTES-calib'
    # make_hp_table(data_name, benchmark_name, NN_HP, NNLoader)
    # make_common_plots(data_name, benchmark_name, NN_HP, NNLoader)
    #
    # data_name = 'BALANCEDHIGGSTES'
    # benchmark_name = 'BALANCEDHIGGSTES-calib'
    # make_hp_table(data_name, benchmark_name, NN_HP, NNLoader)
    # make_common_plots(data_name, benchmark_name, NN_HP, NNLoader)
    #
    # data_name = 'EASYHIGGSTES'
    # benchmark_name = 'EASYHIGGSTES-calib'
    # make_hp_table(data_name, benchmark_name, NN_HP, NNLoader)
    # make_common_plots(data_name, benchmark_name, NN_HP, NNLoader)




if __name__ == '__main__':
    main()
