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



class HistogramSummaryComputer():
    def __init__(self, n_bins=DEFAULT_N_BINS):
        self.n_bins = n_bins

    def fit(self, X):
        self.edges_list = []
        for i in range(X.shape[1]):
            x = X[:, i]
            maximum = np.max(x)
            minimum = np.min(x)
            diff = maximum - minimum
            maximum = maximum + diff / self.n_bins  # be a bit more inclusive
            minimum = minimum - diff / self.n_bins  # be a bit more inclusive
            count, bin_edges = np.histogram(x, range=(minimum, maximum), bins=self.n_bins)
            self.edges_list.append(bin_edges)
        return self

    def predict(self, X, W):
        counts = []        
        for i, bin_edges in enumerate(self.edges_list):
            x = X[:, i]
            count, _ = np.histogram(x, bins=bin_edges, weights=W)
            counts.extend(count)
        return counts

    def __call__(self, X, W):
        counts = self.predict(X, W)
        return np.array(counts)
