# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import pandas as pd

def load_estimations(directory, start_cv=0, end_cv=30):
    all_estimations = [pd.read_csv(os.path.join(directory, f"cv_{i_cv}", "estimations.csv")) for i_cv in range(start_cv, end_cv)]
    estimations = pd.concat(all_estimations, ignore_index=True)
    return estimations


def load_conditional_estimations(directory, start_cv=0, end_cv=30):
    all_estimations = [pd.read_csv(os.path.join(directory, f"cv_{i_cv}", "conditional_estimations.csv")) for i_cv in range(start_cv, end_cv)]
    conditional_estimations = pd.concat(all_estimations, ignore_index=True)
    return conditional_estimations
