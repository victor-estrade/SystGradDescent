#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals


from .visual.common import make_common_plots
from ..loader import TPLoader
from .hyper_parameters import TP_HP


def main():
    print("hello")
    data_name = 'GG'
    benchmark_name = 'GG-prior'
    make_common_plots(data_name, benchmark_name, TP_HP, TPLoader)






if __name__ == '__main__':
    main()
