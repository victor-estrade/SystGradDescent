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


from model.gradient_boost import GradientBoostingModel
# from model.neural_network import NeuralNetClassifier
# from model.tangent_prop import TangentPropClassifier
# from model.pivot import PivotClassifier
# from model.inferno import Inferno
# from model.regressor import Regressor



class Loader(object):
    """docstring for Loader"""
    def __init__(self, data_name, benchmark_name, base_name, model_full_name, tolerance=None):
        self.data_name = data_name
        self.benchmark_name = benchmark_name
        self.base_name = base_name
        self.model_full_name = model_full_name
        self.tolerance = tolerance

    def _get_var_result_directory(self):
        benchmark_name = 'VAR-'+self.data_name
        return os.path.join(SAVING_DIR, benchmark_name, self.base_name, self.model_full_name)

    def _get_result_directory(self):
        if self.tolerance is None:
            return os.path.join(SAVING_DIR, self.benchmark_name, self.base_name, self.model_full_name)
        else:
            return os.path.join(SAVING_DIR, f"{self.benchmark_name}-{self.tolerance}", self.base_name, self.model_full_name)

    def _get_result_directory_cv(self, i_cv):
        return os.path.join(self._get_result_directory(), "cv_{:d}".format(i_cv))

    def _get_result_directory_iter(self, i_cv, i_iter):
        return os.path.join(self._get_result_directory_cv(i_cv), "iter_{:d}".format(i_iter))

    def _load_from_global(self, fname):
        path = os.path.join(self._get_result_directory(), fname)
        df = pd.read_csv(path, index_col=0)
        return df

    def _load_from_cv(self, fname, i_cv):
        path = os.path.join(self._get_result_directory_cv(i_cv), fname)
        df = pd.read_csv(path, index_col=0)
        return df

    def _load_from_iter(self, fname, i_cv, i_iter):
        path = os.path.join(self._get_result_directory_iter(i_cv, i_iter), fname)
        df = pd.read_csv(path, index_col=0)
        return df

    def load_estimations(self):
        df = self._load_from_global("estimations.csv")
        return df

    def load_config_table(self):
        df = self._load_from_global("config_table.csv")
        df['model_full_name'] = self.model_full_name
        df['benchmark_name'] = self.benchmark_name
        df['base_name'] = self.base_name
        return df

    def load_evaluation(self):
        df = self._load_from_global("evaluation.csv")
        return df

    def load_estimation_evaluation(self):
        df = self._load_from_global("estimation_evaluation.csv")
        return df

    def load_conditional_evaluation(self):
        df = self._load_from_global("conditional_evaluation.csv")
        return df

    def load_conditional_estimations(self):
        df = self._load_from_global("conditional_estimations.csv")
        return df

    def load_evaluation_config(self):
        config_table = self.load_config_table()
        evaluation = self.load_evaluation()
        evaluation = evaluation.join(config_table, rsuffix='_')
        evaluation.rename(columns={'true_mix': "true_mu"}, inplace=True)
        return evaluation

    def load_fisher(self):
        path = os.path.join(self._get_var_result_directory(), "fisher.csv")
        fisher = pd.read_csv(path, index_col=0)
        fisher.rename(columns={'true_mix': "true_mu"}, inplace=True)
        return fisher

    def load_threshold(self):
        path = os.path.join(self._get_var_result_directory(), "threshold.csv")
        fisher = pd.read_csv(path, index_col=0)
        fisher.rename(columns={'true_mix': "true_mu"}, inplace=True)
        return fisher

    def hyper_parameter_code(self):
        try :
            code = '-'.join([ f"{key}={value}" for key, value in self.args.items()])
        except AttributeError as e:
            raise e
        return code


class LikelihoodLoader(Loader):
    """docstring for LikelihoodLoader"""
    def __init__(self, data_name, benchmark_name, dummy=None, tolerance=None):
        base_name = "Likelihood"
        model_full_name = f"{base_name}"
        super().__init__(data_name, data_name, base_name, model_full_name, tolerance=tolerance)
        self.args = dict(dummy=dummy)

    def _get_result_directory(self):
        if self.tolerance is None:
            return os.path.join(SAVING_DIR, self.benchmark_name, self.base_name)
        else:
            return os.path.join(SAVING_DIR, f"{self.benchmark_name}-{self.tolerance}", self.base_name)



class FFLoader(Loader):
    """docstring for FFLoader"""
    def __init__(self, data_name, benchmark_name, feature_id=0, tolerance=None):
        base_name = "FeatureModel"
        model_full_name = f"{base_name}-{feature_id}"
        super().__init__(data_name, benchmark_name, base_name, model_full_name, tolerance=tolerance)
        self.args = dict(feature_id=feature_id)


class GBLoader(Loader):
    """docstring for GBLoader"""
    def __init__(self, data_name, benchmark_name, n_estimators=100, max_depth=3, learning_rate=0.1, tolerance=None):
        model = GradientBoostingModel(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
        model.set_info(data_name, benchmark_name, 0)
        model_full_name = model.get_name()
        base_name = model.base_name
        super().__init__(data_name, benchmark_name, base_name, model_full_name, tolerance=tolerance)
        self.args = dict(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)


class NNLoader(Loader):
    """docstring for NNLoader"""
    def __init__(self, data_name, benchmark_name, archi_name, n_steps=2000, n_units=100,
                batch_size=20, learning_rate=1e-3, beta1=0.9, beta2=0.999, optimizer_name="Adam", tolerance=None):
        if optimizer_name == "Adam":
            optimizer_name = f"Adam-{learning_rate}-({beta1}-{beta2})"
        else:
            optimizer_name = f"SGD-{learning_rate}"
        base_name = "NeuralNetClassifier"
        archi_name = archi_name+f"x{n_units:d}"
        model_full_name = f"{base_name}-{archi_name}-{optimizer_name}-{n_steps}-{batch_size}"
        super().__init__(data_name, benchmark_name, base_name, model_full_name, tolerance=tolerance)
        self.args = dict(archi_name=archi_name, n_steps=n_steps, n_units=n_units, batch_size=batch_size,
                learning_rate=learning_rate, beta1=beta1, beta2=beta2, optimizer_name=optimizer_name)



class DALoader(Loader):
    """docstring for DALoader"""
    def __init__(self, data_name, benchmark_name, archi_name, n_steps=2000, n_units=100,
                batch_size=20, learning_rate=1e-3, beta1=0.9, beta2=0.999, optimizer_name="Adam", tolerance=None):
        if optimizer_name == "Adam":
            optimizer_name = f"Adam-{learning_rate}-({beta1}-{beta2})"
        else:
            optimizer_name = f"SGD-{learning_rate}"
        base_name = "DataAugmentation"
        archi_name = archi_name+f"x{n_units:d}"
        model_full_name = f"{base_name}-{archi_name}-{optimizer_name}-{n_steps}-{batch_size}"
        super().__init__(data_name, benchmark_name, base_name, model_full_name, tolerance=tolerance)
        self.args = dict(archi_name=archi_name, n_steps=n_steps, n_units=n_units, batch_size=batch_size,
                learning_rate=learning_rate, beta1=beta1, beta2=beta2, optimizer_name=optimizer_name)


class TPLoader(Loader):
    """docstring for TPLoader"""
    def __init__(self, data_name, benchmark_name, archi_name, n_steps=2000, n_units=100,
                batch_size=1000, learning_rate=1e-3, beta1=0.9, beta2=0.999, optimizer_name="Adam", trade_off=0, tolerance=None):
        if optimizer_name == "Adam":
            optimizer_name = f"Adam-{learning_rate}-({beta1}-{beta2})"
        else:
            optimizer_name = f"SGD-{learning_rate}"
        base_name = "TangentPropClassifier"
        archi_name = archi_name+f"x{n_units:d}"
        model_full_name = f"{base_name}-{archi_name}-{optimizer_name}-{n_steps}-{batch_size}-{trade_off}"
        super().__init__(data_name, benchmark_name, base_name, model_full_name, tolerance=tolerance)
        self.args = dict(archi_name=archi_name, n_steps=n_steps, n_units=n_units, batch_size=batch_size,
                learning_rate=learning_rate, beta1=beta1, beta2=beta2, optimizer_name=optimizer_name, trade_off=trade_off)


class PIVOTLoader(Loader):
    """docstring for PIVOTLoader"""
    def __init__(self, data_name, benchmark_name, archi_name, n_steps=2000, n_units=100,
                batch_size=1000, learning_rate=1e-3, beta1=0.9, beta2=0.999, optimizer_name="Adam", trade_off=0, tolerance=None):
        if optimizer_name == "Adam":
            optimizer_name = f"Adam-{learning_rate}-({beta1}-{beta2})"
        else:
            optimizer_name = f"SGD-{learning_rate}"
        base_name = "PivotClassifier"
        archi_name = archi_name+f"x{n_units:d}"
        model_full_name = f"{base_name}-{archi_name}_{archi_name}-{optimizer_name}-{n_steps}-{batch_size}-{trade_off}"
        super().__init__(data_name, benchmark_name, base_name, model_full_name, tolerance=tolerance)
        self.args = dict(archi_name=archi_name, n_steps=n_steps, n_units=n_units, batch_size=batch_size,
                learning_rate=learning_rate, beta1=beta1, beta2=beta2, optimizer_name=optimizer_name, trade_off=trade_off)



class INFLoader(Loader):
    """docstring for INFLoader"""
    def __init__(self, data_name, benchmark_name, archi_name, n_steps=2000, n_units=100,
                sample_size=1000, learning_rate=1e-3, beta1=0.5, beta2=0.9, optimizer_name="Adam",
                temperature=1.0, tolerance=None):
        if optimizer_name == "Adam":
            optimizer_name = f"Adam-{learning_rate}-({beta1}-{beta2})"
        else:
            optimizer_name = f"SGD-{learning_rate}"
        base_name = "Inferno"
        archi_name = archi_name+f"x{n_units:d}"
        model_full_name = f"{base_name}-{archi_name}-{optimizer_name}-{n_steps}-{sample_size}-{temperature}"
        super().__init__(data_name, benchmark_name, base_name, model_full_name, tolerance=tolerance)
        self.args = dict(archi_name=archi_name, n_steps=n_steps, n_units=n_units, sample_size=sample_size,
                learning_rate=learning_rate, beta1=beta1, beta2=beta2, optimizer_name=optimizer_name, temperature=temperature)


class REGLoader(Loader):
    """docstring for REGLoader"""
    def __init__(self, data_name, benchmark_name, archi_name, n_steps=2000, n_units=100,
                batch_size=20, sample_size=1000, learning_rate=1e-4, beta1=0.5, beta2=0.9, optimizer_name="Adam"):
        if optimizer_name == "Adam":
            optimizer_name = f"Adam-{learning_rate}-({beta1}-{beta2})"
        else:
            optimizer_name = f"SGD-{learning_rate}"
        base_name = "Regressor"
        archi_name = archi_name+f"x{n_units:d}"
        model_full_name = f"{base_name}-{archi_name}-{optimizer_name}-{n_steps}-{batch_size}-{sample_size}"
        super().__init__(data_name, benchmark_name, base_name, model_full_name)
        self.args = dict(archi_name=archi_name, n_steps=n_steps, n_units=n_units, batch_size=batch_size,
                learning_rate=learning_rate, beta1=beta1, beta2=beta2, optimizer_name=optimizer_name)


class FREGLoader(Loader):
    """docstring for FREGLoader"""
    def __init__(self, data_name, benchmark_name, archi_name, n_steps=2000, n_units=100,
                batch_size=20, sample_size=1000, threshold=0.85, learning_rate=1e-4, beta1=0.5, beta2=0.9, optimizer_name="Adam"):
        if optimizer_name == "Adam":
            optimizer_name = f"Adam-{learning_rate}-({beta1}-{beta2})"
        else:
            optimizer_name = f"SGD-{learning_rate}"
        base_name = "FilterRegressor"
        archi_name = archi_name+f"x{n_units:d}"
        model_full_name = f"{base_name}-{archi_name}-{optimizer_name}-{n_steps}-{batch_size}-{sample_size}"
        super().__init__(data_name, benchmark_name, base_name, model_full_name)
        self.args = dict(archi_name=archi_name, n_steps=n_steps, n_units=n_units, batch_size=batch_size,
                learning_rate=learning_rate, beta1=beta1, beta2=beta2, optimizer_name=optimizer_name)
