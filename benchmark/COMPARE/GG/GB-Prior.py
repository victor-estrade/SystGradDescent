#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals


from .visual.common import make_common_plots
from ..loader import GBLoader


def main():
    print("hello")

    args = dict(
                max_depth=[3, 5, 10]
                , n_estimators=[100, 300, 1000]
                , learning_rate=[0.1, 0.05, 0.01]
                )
    data_name = 'GG'
    benchmark_name = 'GG-prior'
    make_common_plots(data_name, benchmark_name, args, GBLoader)






if __name__ == '__main__':
    main()
