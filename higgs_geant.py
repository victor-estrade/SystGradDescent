# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

from datawarehouse import load_higgs

def load_data():
    data = load_higgs()  # Load the full dataset and return it as a pandas.DataFrame
    data = handle_missing_values(data)
    data = clean_columns(data)
    return data


def handle_missing_values(data, missing_value=-999.0, dummy_variables=False):
    """
    Find missing values.
    Replace missing value (-999.9) with 0.
    If dummy_variables is created then :
        for each feature having missing values, add a 'feature_is_missing' boolean feature.
        'feature_is_missing' = 1 if the 'feature' is missing, 0 otherwise.
    
    Args
    ----
        data : (pandas.DataFrame) the data with missing values.
        missing_value : (default=-999.0) the value that should be considered missing.
        dummy_variables : (bool, default=False), if True will add boolean feature columns 
            indicating that values are missing.
    Returns
    -------
        filled_data : (pandas.DataFrame) the data with handled missing values.
    """
    is_missing = (data == missing_value)
    filled_data = data[~is_missing].fillna(0.)  # Magik
    if dummy_variables :
        missing_col = [c for c in is_missing.columns if np.any(is_missing[c])]
        new_cols = {c: c+"_is_missing" for c in missing_col}  # new columns names
        bool_df = is_missing[missing_col]  # new boolean columns
        bool_df = bool_df.rename(columns=new_cols)
        filled_data = filled_data.join(bool_df)  # append the new boolean columns to the data
    return filled_data


def clean_columns(data):
    """
    Removes : EventId, KaggleSet, KaggleWeight
    Cast labels to float.
    """
    data = data.drop(["EventId", "KaggleSet", "KaggleWeight",], axis=1)
    label_to_float(data)  # Works inplace
    return data


def label_to_float(data):
    """
    Transform the string labels to float values.
    s -> 1.0
    b -> 0.0

    Works inplace on the given data !

    Args
    ----
        data: the dataset should be a pandas.DataFrame like object.
            This function will modify the given data inplace.
    """
    if data['Label'].dtype == object:
        #copy entry in human usable form
        data["Label"] = (data["Label"] == 's').astype("float")
    else:
        pass


def normalize_weight(W, y, background_luminosity=410999.84732187376, signal_luminosity=691.9886077135781):
    """Normalize the given weight to assert that the luminosity is the same as the nominal.
    Returns the normalized weight vector/Series
    """
    background_weight_sum = W[y==0].sum()
    signal_weight_sum = W[y==1].sum()
    W_new = W.copy()
    W_new[y==0] = W[y==0] * ( background_luminosity / background_weight_sum )
    W_new[y==1] = W[y==1] * ( signal_luminosity / signal_weight_sum )
    return W_new


def split_data_label_weights(data):
    X = data.drop(["Weight", "Label"], axis=1)
    X = X.drop(["origWeight", "detailLabel"], axis=1, errors="ignore")
    y = data["Label"]
    W = data["Weight"]
    return X, y, W


def split_train_test(data, idx_train, idx_test):
    n_samples = data.shape[0]
    n_train = idx_train.shape[0]
    n_test = n_samples - n_train
    if n_test < 0:
        raise ValueError('The number of train samples ({}) exceed the total number of samples ({})'.format(n_train, n_samples))
    train_data = data.iloc[idx_train]
    test_data = data.iloc[idx_test]
    return train_data, test_data




# def skew(data, z=1.0, missing_value=0., remove_mass_MMC=True):
#     data_skewed = data.copy()
#     if not "DER_mass_MMC" in data_skewed.columns:
#         data_skewed["DER_mass_MMC"] =  np.zeros(data.shape[0]) # Add dummy column

#     tau_energy_scale(data_skewed, z, missing_value=missing_value)  # Modify data inplace
#     data_skewed = data_skewed.drop(["ORIG_mass_MMC", "ORIG_sum_pt"], axis=1)

#     if remove_mass_MMC and "DER_mass_MMC" in data_skewed.columns:
#         data_skewed = data_skewed.drop( ["DER_mass_MMC"], axis=1 )
#     return data_skewed
