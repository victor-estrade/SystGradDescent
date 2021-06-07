#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

# Command line :
# python -m benchmark.COMPARE.S3D2.DA

from .visual.common import make_common_plots
from .visual.common import make_hp_table
from ..loader import FFLoader
from .hyper_parameters import FF_HP


def main():
    print("hello")
    data_name = 'S3D2'
    benchmark_name = 'S3D2-prior'
    make_hp_table(data_name, benchmark_name, FF_HP, FFLoader)
    make_common_plots(data_name, benchmark_name, FF_HP, FFLoader)

    data_name = 'S3D2'
    benchmark_name = 'S3D2-calib'
    make_hp_table(data_name, benchmark_name, FF_HP, FFLoader)
    make_common_plots(data_name, benchmark_name, FF_HP, FFLoader)





if __name__ == '__main__':
    main()
