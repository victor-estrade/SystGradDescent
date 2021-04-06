# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import logging

import matplotlib.pyplot as plt

from config import DEFAULT_DIR
from .misc import now_str

def plot_infer(values, probas, expected_value=None, true_value=None, std=None,
                title="posterior marginal proba", directory=DEFAULT_DIR, fname='marginal.png',
                name=''):
    logger = logging.getLogger()

    try:
        if true_value is not None:
            plt.axvline(true_value, c="orange", label=f"true {name}")
            if std is not None:
                plt.axvline(true_value-std, c="orange", label=f"true {name} - std({name})")
                plt.axvline(true_value+std, c="orange", label=f"true {name} + std({name})")
        if expected_value is not None:
            plt.axvline(expected_value, c="green", label=f"E[{name}|x]")
        plt.plot(values, probas, label="posterior")
        plt.xlabel(f"{name} values")
        plt.ylabel("proba density")
        plt.title(now_str()+title)
        plt.legend()
        plt.savefig(os.path.join(directory, fname))
        plt.clf()
    except Exception as e:
        plt.clf()
        logger.warning(f'Plot infer {name} failed')
        logger.warning(str(e))
