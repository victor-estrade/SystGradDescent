# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np

DEFAULT_N_BINS = 10

def compute_summaries(clf, X, W, n_bins=DEFAULT_N_BINS):
    proba = clf.predict_proba(X)
    count, _ = np.histogram(proba[:, 1], range=(0., 1.), weights=W, bins=n_bins)
    return count


class ClassifierSummaryComputer():
    def __init__(self, clf, n_bins=DEFAULT_N_BINS):
        self.clf = clf
        self.n_bins = n_bins

    def __call__(self, X, W):
        proba = self.clf.predict_proba(X)
        count, _ = np.histogram(proba[:, 1], range=(0., 1.), weights=W, bins=self.n_bins)
        return count
