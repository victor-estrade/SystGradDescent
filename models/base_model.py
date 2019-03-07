# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin


class BaseClassifierModel(BaseEstimator, ClassifierMixin):
    def __init__(self):
        super().__init__()

    def compute_summaries(self, X, W, n_bins=10):
        proba = self.clf.predict_proba(X)
        count, _ = np.histogram(proba[:, 1], weights=W, bins=n_bins)
        return count

