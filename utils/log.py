# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import logging

def set_logger(lvl=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(lvl)
    formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(lvl)
    logger.addHandler(stream_handler)
    logger.info('Hello')
    return logger


def flush(logger):
	logger.handlers[0].flush()


def print_line(symbol='='):
    logger = logging.getLogger()
    logger.info(symbol*105)


def print_params(param, params_truth):
    logger = logging.getLogger()
    logger.info('[param_name] = [truth] vs  [value]  +/-  [error]')
    for p, truth in zip(param, params_truth):
        name  = p['name']
        value = p['value']
        error = p['error']
        logger.info('{name:4} = {truth} vs {value} +/- {error}'.format(**locals()))
