# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import logging

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from config import DEFAULT_DIR


def plot_losses(losses, log=False, title='no title', directory=DEFAULT_DIR, fname='losses.png'):
    logger = logging.getLogger()
    try:
        for name, values in losses.items():
            plt.plot(values, label=name)
        plt.title(title)
        plt.xlabel('# iter')
        plt.ylabel('Loss/MSE')
        if log:
            plt.yscale('log')
        plt.legend()
        plt.savefig(os.path.join(directory, fname))
        plt.clf()
    except Exception as e:
        plt.clf()
        logger.warning('Plot REG losses failed')
        logger.warning(str(e))


def plot_REG_log_mse(mse_losses, title='no title', directory=DEFAULT_DIR, fname='log_mse_loss.png'):
    logger = logging.getLogger()
    try:
        plt.plot(mse_losses, label='mse')
        plt.title(title)
        plt.xlabel('# iter')
        plt.ylabel('Loss/MSE')
        plt.yscale('log')
        plt.legend()
        plt.savefig(os.path.join(directory, fname))
        plt.clf()
    except Exception as e:
        plt.clf()
        logger.warning('Plot REG log losses failed')
        logger.warning(str(e))

