#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals


from .visual.common import make_common_plots
from ..loader import TPLoader


def main():
    print("hello")

    args = dict(
                archi_name=["L4"]
                , n_steps=[2000, 5000]
                , n_units=[50, 100, 200, 500]
                , trade_off=[1, 0.1, 1e-2, 1e-3]
                )
    data_name = 'GG'
    benchmark_name = 'GG-calib'
    make_common_plots(data_name, benchmark_name, args, TPLoader)






if __name__ == '__main__':
    main()
