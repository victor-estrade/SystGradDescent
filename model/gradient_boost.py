# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
import joblib

from .base import BaseClassifierModel
from .base import BaseModel
from .utils import to_numpy
from .utils import classwise_balance_weight


class GradientBoostingModel(BaseClassifierModel):
    def __init__(self, learning_rate=0.1, n_estimators=1000, max_depth=3,):
        super().__init__()
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.clf = GradientBoostingClassifier(learning_rate=learning_rate,
                                 n_estimators=n_estimators,
                                 max_depth=max_depth,
                                 )

    def fit(self, X, y, sample_weight=None):
        X = to_numpy(X)
        y = to_numpy(y)
        sample_weight = to_numpy(sample_weight)
        w = classwise_balance_weight(sample_weight, y)
        self.clf.fit(X, y, sample_weight=w)

    def predict(self, X):
        X = to_numpy(X)
        y_pred = self.clf.predict(X)
        return y_pred

    def predict_proba(self, X):
        X = to_numpy(X)
        proba = self.clf.predict_proba(X)
        return proba

    def save(self, save_directory):
        """Save the model in the given directory"""
        super().save(save_directory)
        path = os.path.join(save_directory, 'GradientBoosting.joblib')
        joblib.dump(self.clf, path)
        return self

    def load(self, save_directory):
        """Load the model of the i-th CV from the given directory"""
        super().load(save_directory)
        path = os.path.join(save_directory, 'GradientBoosting.joblib')
        self.clf = joblib.load(path)
        return self

    def get_name(self):
        name = "{base_name}-{learning_rate}-{n_estimators}-{max_depth}".format(**self.__dict__)
        return name


class BlindGradientBoostingModel(GradientBoostingModel):
    def __init__(self, learning_rate=0.1, n_estimators=1000, max_depth=3,):
        super().__init__()
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth

        self.skewed_idx = [0, 1, 8, 9, 10, 12, 18, 19]
        # ['DER_mass_transverse_met_lep', 'DER_mass_vis', 'DER_sum_pt',
        #  'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality', 'PRI_tau_pt',
        #  'PRI_met', 'PRI_met_phi']

        self.clf = GradientBoostingClassifier(learning_rate=learning_rate,
                                 n_estimators=n_estimators,
                                 max_depth=max_depth,
                                 )

    def fit(self, X, y, sample_weight=None):
        X = to_numpy(X)
        y = to_numpy(y)
        sample_weight = to_numpy(sample_weight)
        X = np.delete(X, self.skewed_idx, axis=1)
        w = classwise_balance_weight(sample_weight, y)
        self.clf.fit(X, y, sample_weight=w)

    def predict(self, X):
        X = to_numpy(X)
        X = np.delete(X, self.skewed_idx, axis=1)        
        y_pred = self.clf.predict(X)
        return y_pred

    def predict_proba(self, X):
        X = to_numpy(X)
        X = np.delete(X, self.skewed_idx, axis=1)        
        proba = self.clf.predict_proba(X)
        return proba

    def save(self, save_directory):
        """Save the model in the given directory"""
        super(BaseModel, self).save(save_directory)
        path = os.path.join(save_directory, 'GradientBoosting.joblib')
        joblib.dump(self.clf, path)
        return self

    def load(self, save_directory):
        """Load the model of the i-th CV from the given directory"""
        super(BaseModel, self).load(save_directory)
        path = os.path.join(save_directory, 'GradientBoosting.joblib')
        self.clf = joblib.load(path)
        return self
