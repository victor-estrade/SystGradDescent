# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np

def compute_summaries(clf, X, W, n_bins=10):
    proba = clf.predict_proba(X)
    count, _ = np.histogram(proba[:, 1], weights=W, bins=n_bins)
    return count
