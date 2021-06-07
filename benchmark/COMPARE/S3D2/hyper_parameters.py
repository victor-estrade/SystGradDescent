# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals


DA_HP    = dict(
                archi_name=["L4"]
                , n_steps=[5000]
                , n_units=[200, 500]
                , batch_size=[1000]
                )

FF_HP    = dict(
                feature_id=list(range(1))
                )

GB_HP    = dict(
                max_depth=[3, 5, 10]
                , n_estimators=[100, 300, 1000]
                , learning_rate=[0.1, 0.05, 0.01]
                )

INF_HP   = dict(
                archi_name=["L4"]
                , n_steps=[5000]
                , n_units=[200, 500]
                )

NN_HP    = dict(
                archi_name=["L4"]
                , n_steps=[5000]
                , n_units=[200, 500]
                , batch_size=[20, 1000]
                )

PIVOT_HP = dict(
                archi_name=["L4"]
                , n_steps=[5000]
                , n_units=[200, 500]
                , trade_off=[1.0, 0.1, 1e-2, 1e-3]
                , batch_size=[1000]
                )

REG_HP   = dict(
                archi_name=["EA3ML3"]
                , n_steps=[5000]
                , n_units=[200, 500]
                )

REG_M_HP = dict(
                archi_name=["A1AR8MR8L1"]
                , n_steps=[5000]
                , n_units=[200, 500]
                )

TP_HP    = dict(
                archi_name=["L4"]
                , n_steps=[5000]
                , n_units=[200, 500]
                , trade_off=[1.0, 0.1, 1e-2, 1e-3]
                , batch_size=[1000]
                )

Likelihood_HP    = dict(dummy=[None])
