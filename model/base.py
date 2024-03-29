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
from .summaries import DEFAULT_N_BINS

# TODO : Maybe the sklearn dependancy is useless.
#       For now set_param(), get_param() or score() methods are never used

class ModelInfo(object):
    """Gather all basic external information of the model
    like the number of cross validation or the path where to save the model.
    """
    old_path = None
    model_path = None
    model_directory = None
    full_name = None
    i_cv = None
    benchmark_name = None

    def __init__(self):
        self.base_name = type(self).__name__

    def get_name(self):
        raise NotImplementedError("Should be implemented in child class")

    @property
    def name(self):
        return self.get_name()

    def save(self, save_directory):
        info = dict(model_path=self.model_path,
                    model_directory=self.model_directory,
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
        self.model_path = save_directory
        try:
            self.old_path = info['path']
            self.model_directory = info['model_directory']
        except KeyError:
            pass
        self.full_name = info['full_name']
        self.i_cv = info['i_cv']
        self.benchmark_name = info['benchmark_name']
        return self

    def _set_full_name(self, i_cv):
        self.full_name = '{}{}{}'.format(self.get_name(), os.sep, i_cv)

    def _set_results_path(self, benchmark_name, i_cv):
        name = self.get_name()
        cv_id = "cv_{:d}".format(i_cv)
        self.results_directory = os.path.join(config.SAVING_DIR, benchmark_name,
                                        self.base_name, name)
        self.results_path = os.path.join(self.results_directory, cv_id)

    def _set_model_path(self, data_name, i_cv):
        name = self.get_name()
        cv_id = "cv_{:d}".format(i_cv)
        self.model_directory = os.path.join(config.MODEL_SAVING_DIR, data_name,
                                        self.base_name, name)
        self.model_path = os.path.join(self.model_directory, cv_id)

    def set_info(self, data_name, benchmark_name, i_cv):
        self.benchmark_name = benchmark_name
        self.i_cv = i_cv
        self._set_results_path(benchmark_name, i_cv)
        self._set_model_path(data_name, i_cv)
        self._set_full_name(i_cv)


class BaseModel(ModelInfo, BaseEstimator):
    """ Gather all basic methods and utils"""
    pass


class BaseClassifierModel(BaseModel, ClassifierMixin):
    """ More specific than BaseModel for classifiers
    """
    def summary_computer(self, n_bins=DEFAULT_N_BINS):
        return lambda X, w : self.compute_summaries(X, w, n_bins=n_bins)

    def compute_summaries(self, X, W, n_bins=DEFAULT_N_BINS):
        proba = self.predict_proba(X)
        decision = proba[:, 1]
        if np.isnan(decision).any():
            print("[WARNING] : NaN detected in predicted decision/proba ")
        count, _ = np.histogram(decision, range=(0., 1.), weights=W, bins=n_bins)
        return count



class BaseNeuralNet():
    optimizer = None

    def get_adam_name(self):
        lr = self.optimizer.defaults['lr']
        beta1, beta2 = self.optimizer.defaults['betas']
        name = "Adam-{lr}-({beta1}-{beta2})".format(**locals())
        return name

    def get_sgd_name(self):
        lr = self.optimizer.defaults['lr']
        weight_decay = self.optimizer.defaults['weight_decay']
        name = "SGD-{lr}-({weight_decay})".format(**locals())
        return name

    def get_optimizer_name(self):
        import torch.optim as optim
        if isinstance(self.optimizer, optim.Adam):
            return self.get_adam_name()
        if isinstance(self.optimizer, optim.SGD):
            return self.get_sgd_name()

    def set_optimizer_name(self):
        self.optimizer_name = self.get_optimizer_name()

    def to_double(self):
        self.net = self.net.double()
