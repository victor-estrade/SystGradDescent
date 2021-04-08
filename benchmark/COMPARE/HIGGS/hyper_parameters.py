# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals


DA_HP    = dict(
                archi_name=["L4"]
                , n_steps=[15000, 25000]
                , n_units=[500]
                , batch_size=[10000]
                , tolerance=[10]
                )

FF_HP    = dict(
                feature_id=list(range(28))
                , tolerance=[10]
                )

GB_HP    = dict(
                max_depth=[3, 5, 10]
                , n_estimators=[100, 300, 1000]
                , learning_rate=[0.1, 0.05, 0.01]
                , tolerance=[10]
                )

INF_HP   = dict(
                archi_name=["L4"]
                , n_steps=[15000, 25000]
                , n_units=[500]
                , sample_size=[10000]
                , tolerance=[10]
                )

NN_HP    = dict(
                archi_name=["L4"]
                , n_steps=[15000, 25000]
                , n_units=[200, 500]
                , batch_size=[10000]
                , tolerance=[10]
                )

PIVOT_HP = dict(
                archi_name=["L4"]
                , n_steps=[15000, 25000]
                , n_units=[100, 200, 500]
                , trade_off=[1.0, 0.1, 1e-2, 1e-3]
                , batch_size=[10000]
                , tolerance=[10]
                )

REG_HP   = dict(
                archi_name=["EA3ML3"]
                , n_steps=[15000, 25000]
                , n_units=[100, 200, 500]
                , sample_size=[10000]
                )

FREG_HP  = dict(
                archi_name=["EA3ML3"]
                , n_steps=[15000, 25000]
                , n_units=[100, 200, 500]
                , sample_size=[10000]
                )

REG_M_HP = dict(
                archi_name=["A3ML3"]
                , n_steps=[15000, 25000]
                , n_units=[100, 200, 500]
                , sample_size=[10000]
                )

TP_HP    = dict(
                archi_name=["L4"]
                , n_steps=[15000, 25000]
                , n_units=[100, 200, 500]
                , trade_off=[1.0, 0.1, 1e-2, 1e-3]
                , batch_size=[10000]
                , tolerance=[10]
                )
