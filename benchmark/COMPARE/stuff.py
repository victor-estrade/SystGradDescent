#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import logging


from visual.misc import set_plot_config
set_plot_config()


from .loader import GBLoader
from .loader import NNLoader

def main():
    print("hello !")
    my_loader = GBLoader('GG', 'GG-prior')
    my_loader = NNLoader('GG', 'GG-prior', "L4")
    evaluation = my_loader.load_evaluation()
    print(evaluation)


if __name__ == '__main__':
    main()