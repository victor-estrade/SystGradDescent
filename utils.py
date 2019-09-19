# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import logging
import inspect


def set_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    logger.info('Hello')
    return logger


def flush(logger):
	logger.handlers[0].flush()


def extract_model_args(args, model_class):
    sig = inspect.signature(model_class)
    args_dict = vars(args)
    model_args = { k: args_dict[k] for k in sig.parameters.keys() if k in args_dict }
    return model_args

def get_model(args, model_class):
    logger = logging.getLogger()
    logger.info('Building model ...')
    model_args = extract_model_args(args, model_class)
    logger.info( 'model_args :{}'.format(model_args) )
    model = model_class(**model_args)
    flush(logger)
    return model


def print_params(param, params_truth):
    print('name = truth vs  value  +/-  error'.format(**locals()))
    for p, truth in zip(param, params_truth):
        name  = p['name']
        value = p['value']
        error = p['error']
        print('{name:4} = {truth} vs {value} +/- {error}'.format(**locals()))