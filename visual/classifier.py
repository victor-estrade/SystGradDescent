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


def old_plot_test_distrib(model, X_test, y_test):
    logger = logging.getLogger()
    logger.info( 'Test accuracy = {} %'.format(100 * model.score(X_test, y_test)) )
    proba = model.predict_proba(X_test)
    try:
        sns.distplot(proba[y_test==0, 1], label='b')
        sns.distplot(proba[y_test==1, 1], label='s')
        plt.title(model.full_name)
        plt.legend()
        plt.savefig(os.path.join(model.path, 'test_distrib.png'))
        plt.clf()
    except Exception as e:
        logger.warning('Plot test distrib failed')
        logger.warning(str(e))


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
        logger.warning('Plot test distrib failed')
        logger.warning(str(e))


def plot_valid_distrib(model, X, y, classes=('b', 's')):
    logger = logging.getLogger()
    logger.info( 'Valid accuracy = {} %'.format(100 * model.score(X, y)) )
    proba = model.predict_proba(X)
    try:
        sns.distplot(proba[y==0, 1], label=classes[0])
        sns.distplot(proba[y==1, 1], label=classes[1])
        plt.xlabel('classifier score')
        plt.ylabel('density')
        plt.title(model.full_name)
        plt.legend()
        plt.savefig(os.path.join(model.path, 'valid_distrib.png'))
        plt.clf()
    except Exception as e:
        logger.warning('Plot valid distrib failed')
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
        logger.warning('Plot ROC failed')
        logger.warning(str(e))

