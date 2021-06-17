# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals


DA_HP    = dict(
                archi_name=["L4"]
                , n_steps=[15000, 25000]
                , n_units=[200, 500]
                , batch_size=[10000]
                , tolerance=[100.0]
                )

FF_HP    = dict(
                feature_id=list(range(28))
                , tolerance=[100.0]
                )

GB_HP    = dict(
                max_depth=[3, 6]
                , n_estimators=[300, 800]
                , learning_rate=[0.1, 0.01]
                , tolerance=[100.0]
                )

INF_HP   = dict(
                archi_name=["L4"]
                , n_steps=[2000]
                , n_units=[200, 500]
                , sample_size=[10000]
                , tolerance=[100.0]
                )

NN_HP    = dict(
                archi_name=["L4"]
                , n_steps=[15000, 25000]
                , n_units=[200, 500]
                , batch_size=[200, 10000]
                , tolerance=[100.0]
                )

PIVOT_HP = dict(
                archi_name=["L4"]
                , n_steps=[15000, 25000]
                , n_units=[200, 500]
                , trade_off=[1.0, 0.1]
                , batch_size=[1000, 10000]
                , tolerance=[100.0]
                )

REG_HP   = dict(
                archi_name=["EA3ML3"]
                , n_steps=[15000, 25000]
                , n_units=[200, 500]
                , sample_size=[10000, 50000]
                )

FREG_HP  = dict(
                archi_name=["EA3ML3"]
                , n_steps=[15000, 25000]
                , n_units=[200, 500]
                , sample_size=[10000, 50000]
                )

REG_M_HP = dict(
                archi_name=["A3ML3"]
                , n_steps=[15000, 25000]
                , n_units=[100, 200, 500]
                , sample_size=[10000, 50000]
                )

TP_HP    = dict(
                archi_name=["L4"]
                , n_steps=[15000, 25000]
                , n_units=[200, 500]
                , trade_off=[1.0, 0.1]
                , batch_size=[200]
                , tolerance=[100.0]
                )
