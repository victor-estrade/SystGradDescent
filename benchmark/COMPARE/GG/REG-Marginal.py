#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals


from .visual.common import make_common_plots
from ..loader import REGLoader


def main():
    print("hello")

    args = dict(
                archi_name=["A1AR8MR8L1"]
                , n_steps=[2000, 5000]
                , n_units=[50, 100, 200, 500]
                )
    data_name = 'GG'
    benchmark_name = 'GG-marginal'
    make_common_plots(data_name, benchmark_name, args, REGLoader)






if __name__ == '__main__':
    main()
