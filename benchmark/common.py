# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import glob
import pandas as pd

def load_estimations(directory, start_cv=0, end_cv=30):
    all_estimations = [pd.read_csv(os.path.join(directory, f"cv_{i_cv}", "estimations.csv")) for i_cv in range(start_cv, end_cv)]
    estimations = pd.concat(all_estimations, ignore_index=True)
    return estimations


def load_conditional_estimations(directory, start_cv=0, end_cv=30):
    try:
        all_estimations = [pd.read_csv(os.path.join(directory, f"cv_{i_cv}", "conditional_estimations.csv")) for i_cv in range(start_cv, end_cv)]
    except FileNotFoundError:
        try:
            generate_conditional_estimations(directory, start_cv=start_cv, end_cv=end_cv)
            all_estimations = [pd.read_csv(os.path.join(directory, f"cv_{i_cv}", "conditional_estimations.csv")) for i_cv in range(start_cv, end_cv)]
        except FileNotFoundError as e:
            raise e
    conditional_estimations = pd.concat(all_estimations, ignore_index=True)
    return conditional_estimations



def generate_conditional_estimations(directory, start_cv=0, end_cv=30):
    for i_cv in range(start_cv, end_cv):
        fname = os.path.join(directory, f"cv_{i_cv}", "iter_*", "no_nuisance.csv")
        print("looking for subfiles")
        all_files = glob.glob(fname)
        print(all_files)
        conditional_estimations = pd.concat([pd.read_csv(f) for f in all_files])
        conditional_estimations['i_cv'] = i_cv
        fname = os.path.join(directory, f"cv_{i_cv}", "conditional_estimations.csv")
        conditional_estimations.to_csv(fname)

