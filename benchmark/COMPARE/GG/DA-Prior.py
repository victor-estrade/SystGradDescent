#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals


from .visual.common import make_common_plots
from ..loader import DALoader
from.hyper_parameters import DA_HP


def main():
    print("hello")
    data_name = 'GG'
    benchmark_name = 'GG-prior'
    make_common_plots(data_name, benchmark_name, DA_HP, DALoader)






if __name__ == '__main__':
    main()
