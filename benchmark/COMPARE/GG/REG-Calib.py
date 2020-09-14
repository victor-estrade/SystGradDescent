#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals


from .visual.common import make_common_plots
from ..loader import REGLoader
from .hyper_parameters import REG_HP

def main():
    print("hello")
    data_name = 'GG'
    benchmark_name = 'GG-calib'
    make_common_plots(data_name, benchmark_name, REG_HP, REGLoader)






if __name__ == '__main__':
    main()
