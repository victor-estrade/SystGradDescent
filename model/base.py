# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import json
import config

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from .summaries import ClassifierSummaryComputer
from .summaries import DEFAULT_N_BINS

# TODO : Maybe the sklearn dependancy is useless.
#       For now set_param(), get_param() or score() methods are never used

class ModelInfo(object):
    """Gather all basic external information of the model
    like the number of cross validation or the path where to save the model.
    """
    old_path = None
    path = None
    directory = None
    full_name = None
    i_cv = None
    benchmark_name = None

    def get_name(self):
        raise NotImplementedError("Should be implemented in child class")

    def save(self, save_directory):
        info = dict(path=self.path,
                    directory=self.directory,
                    full_name=self.full_name,
                    i_cv=self.i_cv,
                    benchmark_name=self.benchmark_name
                    )
        info_path = os.path.join(save_directory, 'info.json')
        with open(info_path, 'w') as f:
            json.dump(info, f)
        return self

    def load(self, save_directory):
        info_path = os.path.join(save_directory, 'info.json')
        with open(info_path, 'r') as f:
            info = json.load(f)
        self.path = save_directory
        self.old_path = info['path']
        self.directory = info['directory']
        self.full_name = info['full_name']
        self.i_cv = info['i_cv']
        self.benchmark_name = info['benchmark_name']
        return self

    def _set_full_name(self, i_cv):
        self.full_name = '{}{}{}'.format(self.get_name(), os.sep, i_cv)

    def _set_path(self, benchmark_name, i_cv):
        model_class = type(self).__name__
        name = self.get_name()
        cv_id = "{:d}".format(i_cv)
        self.directory = os.path.join(config.SAVING_DIR, benchmark_name, 
                                        model_class, name)
        self.path = os.path.join(self.directory, cv_id)

    def set_info(self, benchmark_name, i_cv):
        self.benchmark_name = benchmark_name
        self.i_cv = i_cv
        self._set_path(benchmark_name, i_cv)
        self._set_full_name(i_cv)


class BaseModel(ModelInfo, BaseEstimator):
    """ Gather all basic methods and utils"""
    pass


class BaseClassifierModel(BaseModel, ClassifierMixin):
    """ More specific than BaseModel for classifiers
    """
    def summary_computer(self, n_bins=DEFAULT_N_BINS):
        return ClassifierSummaryComputer(self, n_bins=n_bins)

    def compute_summaries(self, X, W, n_bins=DEFAULT_N_BINS):
        proba = self.predict_proba(X)
        count, _ = np.histogram(proba[:, 1], range=(0., 1.), weights=W, bins=n_bins)
        return count

