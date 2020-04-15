# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import logging

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from config import DEFAULT_DIR

from config import _ERROR
from config import _TRUTH


def set_plot_config():
    sns.set()
    sns.set_style("whitegrid")
    sns.set_context("poster")

    mpl.rcParams['figure.figsize'] = [8.0, 6.0]
    mpl.rcParams['figure.dpi'] = 80
    mpl.rcParams['savefig.dpi'] = 100

    mpl.rcParams['font.size'] = 10
    mpl.rcParams['axes.labelsize'] = 10
    mpl.rcParams['axes.titlesize'] = 17
    mpl.rcParams['ytick.labelsize'] = 10
    mpl.rcParams['xtick.labelsize'] = 10
    mpl.rcParams['legend.fontsize'] = 'large'
    mpl.rcParams['figure.titlesize'] = 'medium'
    mpl.rcParams['lines.markersize'] = np.sqrt(30)



def plot_params(param_name, result_table, title='no title', directory=DEFAULT_DIR):
    logger = logging.getLogger()
    values = result_table[param_name]
    errors = result_table[param_name+_ERROR]
    truths = result_table[param_name+_TRUTH]
    xx = np.arange(len(values))
    if 'is_valid' in result_table:
        valid_values = values[result_table['is_valid']]
        valid_errors = errors[result_table['is_valid']]
        valid_x = xx[result_table['is_valid']]
        logger.debug("Plot_params valid lenght = {}, {}, {}".format(len(valid_x), len(valid_values), len(valid_errors)))
        values =  values[result_table['is_valid'] == False]
        errors =  errors[result_table['is_valid'] == False]
        x = xx[result_table['is_valid'] == False]
        logger.debug('Plot_params invalid lenght = {}, {}, {}'.format(len(x), len(values), len(errors)))
    try:
        if 'is_valid' in result_table:
            plt.errorbar(valid_x, valid_values, yerr=valid_errors, fmt='o', capsize=15, capthick=2, label='valid_infer')
            plt.errorbar(x, values, yerr=errors, fmt='o', capsize=15, capthick=2, label='invalid_infer')
        else:
            plt.errorbar(xx, values, yerr=errors, fmt='o', capsize=15, capthick=2, label='infer')
        plt.scatter(xx, truths, c='red', label='truth')
        plt.xticks(xx)
        # plt.xticks(xx, map(lambda x: round(x, 3), truths))
        plt.xlabel('#iter')
        plt.ylabel(param_name)
        plt.title(title)
        plt.legend()
        plt.savefig(os.path.join(directory, 'estimate_{}.png'.format(param_name)))
        plt.clf()
    except Exception as e:
        logger.warning('Plot params failed')
        logger.warning(str(e))
