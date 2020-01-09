# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import json
import config

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

# TODO : Maybe the sklearn dependancy is useless.
#       For now set_param(), get_param() or score() methods are never used

class ModelInfo(object):
    """Gather all basic external information of the model
    like the number of cross validation or the path where to save the model.
    """
    old_model_path = None
    model_path = None
    model_id = None
    i_cv = None
    benchmark_name = None

    def get_name(self):
        raise NotImplementedError("Should be implemented in child class")

    def save(self, dir_path):
        info = dict(model_path=self.model_path,
                    model_id=self.model_id,
                    i_cv=self.i_cv,
                    benchmark_name=self.benchmark_name
                    )
        info_path = os.path.join(dir_path, 'info.json')
        with open(info_path, 'w') as f:
            json.dump(info, f)
        return self

    def load(self, dir_path):
        info_path = os.path.join(dir_path, 'info.json')
        with open(info_path, 'r') as f:
            info = json.load(f)
        self.model_path = dir_path
        self.old_model_path = info['model_path']
        self.model_id = info['model_id']
        return self

    def _set_id(self, i_cv):
        self.model_id = '{}{}{}'.format(self.get_name(), os.sep, i_cv)

    def _set_path(self, benchmark_name, i_cv):
        model_class = type(self).__name__
        model_name = self.get_name()
        cv_id = "{:d}".format(i_cv)
        self.model_path = os.path.join(config.SAVING_DIR, benchmark_name, 
                                        model_class, model_name, cv_id)

    def set_info(self, benchmark_name, i_cv):
        self.benchmark_name = benchmark_name
        self.i_cv = i_cv
        self._set_path(benchmark_name, i_cv)
        self._set_id(i_cv)


class BaseModel(ModelInfo, BaseEstimator):
    """ Gather all basic methods and utils"""
    pass


class BaseClassifierModel(BaseModel, ClassifierMixin):
    """ More specific than BaseModel for classifiers
    """
    # TODO : put compute summary here ?
    pass

