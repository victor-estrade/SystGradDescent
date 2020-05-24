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



def plot_test_distrib(y_proba, y_test, title="no title", 
                      directory=DEFAULT_DIR, fname='test_distrib.png', classes=('b', 's')):
    logger = logging.getLogger()
    # logger.info( 'Test accuracy = {} %'.format(100 * model.score(X_test, y_test)) )
    try:
        sns.distplot(y_proba[y_test==0, 1], label=classes[0])
        sns.distplot(y_proba[y_test==1, 1], label=classes[1])
        plt.title(title)
        plt.legend()
        plt.savefig(os.path.join(directory, fname))
        plt.clf()
    except Exception as e:
        plt.clf()
        logger.warning('Plot test distrib failed')
        logger.warning(str(e))


def plot_ROC(fpr, tpr, title="no title", directory=DEFAULT_DIR, fname='roc.png'):
    from sklearn.metrics import auc
    logger = logging.getLogger()
    try:
        plt.plot(fpr, tpr, label='AUC {}'.format(auc(fpr, tpr)))
        plt.title(title)
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.legend()
        plt.savefig(os.path.join(directory, fname))
        plt.clf()
    except Exception as e:
        plt.clf()
        logger.warning('Plot ROC failed')
        logger.warning(str(e))

