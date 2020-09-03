# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import logging

import pandas as pd

from config import SAVING_DIR


from visual.misc import set_plot_config
set_plot_config()




class Loader(object):
    """docstring for Loader"""
    def __init__(self, benchmark_name, base_name, model_full_name):
        self.benchmark_name = benchmark_name
        self.base_name = base_name
        self.model_full_name = model_full_name

    def _get_result_directory(self):
        return os.path.join(SAVING_DIR, self.benchmark_name, self.base_name, self.model_full_name)

    def _get_result_directory_cv(self, i_cv):
        return os.path.join(self._get_result_directory(), "cv_{:d}".format(i_cv))

    def _get_result_directory_iter(self, i_cv, i_iter):
        return os.path.join(self._get_result_directory_cv(i_cv), "iter_{:d}".format(i_iter))

    def _load_from_global(self, fname):
        path = os.path.join(self._get_result_directory(), fname)
        df = pd.read_csv(path)
        return df

    def _load_from_cv(self, fname, i_cv):
        path = os.path.join(self._get_result_directory_cv(i_cv), fname)
        df = pd.read_csv(path)
        return df

    def _load_from_iter(self, fname, i_cv, i_iter):
        path = os.path.join(self._get_result_directory_iter(i_cv, i_iter), fname)
        df = pd.read_csv(path)
        return df

    def load_estimations(self):
        df = self._load_from_global("estimations.csv")
        return df

    def load_config_table(self):
        df = self._load_from_global("config_table.csv")
        return df

    def load_evaluation(self):
        df = self._load_from_global("evaluation.csv")
        return df

    def load_conditional_estimations(self):
        df = self._load_from_global("conditional_estimations.csv")
        return df






