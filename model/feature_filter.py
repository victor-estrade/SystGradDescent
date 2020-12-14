# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import numpy as np

import json

from .base import BaseModel
from .summaries import DEFAULT_N_BINS
from .utils import to_numpy


class FeatureModel(BaseModel):
    def __init__(self, feature_id=0):
        super().__init__()
        self.feature_id = feature_id
        self.max_ = None
        self.min_ = None
        self.threshold_ = None
        self.invert_ = False

    def fit(self, X, y, sample_weight=None):
        X = to_numpy(X)
        y = to_numpy(y)
        feature = self._extract_feature(X)
        self.max_ = np.max(feature)
        self.min_ = np.min(feature)
        sort_idx = np.argsort(feature)
        n_samples = feature.shape[0]
        half_idx = n_samples//2
        self.threshold_ = feature[sort_idx][half_idx]
        y_sorted = y[sort_idx]
        self.invert_ = np.mean(y_sorted[:half_idx]) > 0.5
        return self

    def _rescale(self, x):
        new_x = (x - self.min_) / (self.max_ - self.min_)
        return new_x

    def _extract_feature(self, X):
        assert X.ndim == 2, "X should be a 2D numpy array like object"
        assert X.shape[1] >= self.feature_id, f"X.shape[1] = {X.shape[1]} which is less than required feature_id {self.feature_id}"
        feature = X[:, self.feature_id]
        return feature

    def transform(self, X):
        feature = self._extract_feature(X)
        new_x = self._rescale(feature)
        return new_x

    def predict(self, X):
        X = to_numpy(X)
        feature = self._extract_feature(X)
        y_pred = feature < self.threshold_
        return y_pred

    def predict_proba(self, X):
        X = to_numpy(X)
        rescaled_feature = self.transform(X)
        if self.invert_:
            proba = np.vstack([rescaled_feature, 1-rescaled_feature]).T
        else:
            proba = np.vstack([1-rescaled_feature, rescaled_feature]).T
        return proba

    def compute_summaries(self, X, W, n_bins=DEFAULT_N_BINS):
        rescaled_feature = self.transform(X)
        count, _ = np.histogram(rescaled_feature, weights=W, bins=n_bins)
        return count

    def summary_computer(self, n_bins=DEFAULT_N_BINS):
        return lambda X, w : self.compute_summaries(X, w, n_bins=n_bins)

    def save(self, save_directory):
        """Save the model in the given directory"""
        super().save(save_directory)
        feature_stats = {
                "max": float(self.max_),
                "min": float(self.min_),
                "threshold": float(self.threshold_),
                "invert": bool(self.invert_),
                }
        print(feature_stats)
        path = os.path.join(save_directory, 'feature_stats.json')
        with open(path, 'w') as f:
            json.dump(feature_stats, f)
        return self

    def load(self, save_directory):
        """Load the model of the i-th CV from the given directory"""
        super().load(save_directory)
        path = os.path.join(save_directory, 'feature_stats.json')
        with open(path, 'r') as f:
            feature_stats = json.load(f)
        self.max_ = feature_stats['max']
        self.min_ = feature_stats['min']
        self.threshold_ = feature_stats['threshold']
        self.invert_ = feature_stats['invert']
        return self

    def get_name(self):
        name = "{base_name}-{feature_id}".format(**self.__dict__)
        return name
