# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import logging
import inspect

import warnings

def get_class_name(obj):
    class_name = type(obj).__name__
    return class_name

def get_model_id(model, i_cv):
    warnings.warn("Model info are now attributes of the model itself."
    "Initialize the info with model.set_info(bench_name, i_cv) "
    "and use model.full_name attribute instead", DeprecationWarning)
    model_id = '{}{}{}'.format(model.get_name(), os.sep, i_cv)
    return model_id


def get_model_path(benchmark_name, model, i_cv=None):
    warnings.warn("Model info are now attributes of the model itself."
    "Initialize the info with model.set_info(bench_name, i_cv) "
    "and use model.path or model.directory attributes instead", DeprecationWarning)
    import config
    model_name = model.get_name()
    model_class = get_class_name(model)
    model_path = os.path.join(config.SAVING_DIR, benchmark_name, model_class, model_name)
    if i_cv is not None:
        cv_id = "{:d}".format(i_cv)
        model_path = os.path.join(model_path, cv_id)
    return model_path


def save_model(model):
    if model.path is None:
        raise ValueError("model's info should be initialized first." 
            "Use model.set_info(bench_name, i_cv)")
    logger = logging.getLogger()
    logger.info("Saving in {}".format(model.path))
    os.makedirs(model.path, exist_ok=True)
    model.save(model.path)

def extract_class_args(args, model_class):
    sig = inspect.signature(model_class)
    args_dict = vars(args)
    model_args = { k: args_dict[k] for k in sig.parameters.keys() if k in args_dict }
    return model_args

def get_model(args, model_class, quiet=True):
    model_args = extract_class_args(args, model_class)
    model = model_class(**model_args)
    if not quiet:
        logger = logging.getLogger()
        logger.info('Building model ...')
        logger.info( 'model_args :{}'.format(model_args) )
    return model


def get_optimizer(args):
    import torch.optim as optim
    all_optims = dict(
        sgd  = optim.SGD,
        SGD  = optim.SGD,
        adam = optim.Adam,
        Adam = optim.Adam,
        ADAM = optim.Adam,
        )
    logger = logging.getLogger()
    optim_class = all_optims[args.optimizer_name]

    args.lr = args.learning_rate
    args.betas = (args.beta1, args.beta2)
    kwargs = extract_class_args(args, optim_class)
    
    net = args.net
    optimizer =  optim_class(net.parameters(), **kwargs)
    logger.info( '{} args :{}'.format(args.optimizer_name, kwargs) )
    return optimizer


def train_or_load_classifier(model, train_generator, parameters, n_samples, retrain=True):
    logger = logging.getLogger()
    if not retrain:
        try:
            logger.info('loading from {}'.format(model.path))
            model.load(model.path)
        except Exception as e:
            logger.warning(e)
            retrain = True
    if retrain:
        logger.info('Generate training data')
        logger.info('Training {}'.format(model.get_name()))
        X_train, y_train, w_train = train_generator.generate(*parameters, n_samples=n_samples)
        model.fit(X_train, y_train, w_train)
        logger.info('Training DONE')

        # SAVE MODEL
        save_model(model)


def train_or_load_neural_net(model, train_generator, retrain=True):
    logger = logging.getLogger()
    if not retrain:
        try:
            logger.info('loading from {}'.format(model.path))
            model.load(model.path)
        except Exception as e:
            logger.warning(e)
            retrain = True
    if retrain:
        logger.info('Training {}'.format(model.get_name()))
        model.fit(train_generator)
        logger.info('Training DONE')

        # SAVE MODEL
        save_model(model)


def train_or_load_inferno(model, train_generator, retrain=True):
    logger = logging.getLogger()
    if not retrain:
        try:
            logger.info('loading from {}'.format(model.path))
            model.load(model.path)
        except Exception as e:
            logger.warning(e)
            retrain = True
    if retrain:
        logger.info('Training {}'.format(model.get_name()))
        model.fit(train_generator)
        logger.info('Training DONE')

        # SAVE MODEL
        save_model(model)
