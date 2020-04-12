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


def plot_REG_losses(model):
    logger = logging.getLogger()
    losses = model.losses
    mse_losses = model.mse_losses
    try:
        plt.plot(mse_losses, label='mse')
        plt.plot(losses, label='loss')
        plt.title(model.full_name)
        plt.xlabel('# iter')
        plt.ylabel('Loss/MSE')
        plt.legend()
        plt.savefig(os.path.join(model.path, 'losses.png'))
        plt.clf()
    except Exception as e:
        logger.warning('Plot REG losses failed')
        logger.warning(str(e))


def plot_INFERNO_losses(model):
    logger = logging.getLogger()
    losses = model.loss_hook.losses
    try:
        plt.plot(losses, label='loss')
        plt.title(model.full_name)
        plt.xlabel('# iter')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(model.path, 'losses.png'))
        plt.clf()
    except Exception as e:
        logger.warning('Plot INFERNO losses failed')
        logger.warning(str(e))



def plot_REG_log_mse(model):
    logger = logging.getLogger()
    mse_losses = model.mse_losses
    try:
        plt.plot(mse_losses, label='mse')
        plt.title(model.full_name)
        plt.xlabel('# iter')
        plt.ylabel('Loss/MSE')
        plt.yscale('log')
        plt.legend()
        plt.savefig(os.path.join(model.path, 'log_mse_loss.png'))
        plt.clf()
    except Exception as e:
        logger.warning('Plot REG log losses failed')
        logger.warning(str(e))

