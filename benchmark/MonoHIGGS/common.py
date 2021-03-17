# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

from utils.model import get_model
from utils.model import get_optimizer

N_BINS = 30

class GeneratorCPU:
    def __init__(self, data_generator):
        self.data_generator = data_generator
        self.n_samples = data_generator.size

    def generate(self, *params, n_samples=None, no_grad=False):
            X, y, w = self.data_generator.generate(*params, n_samples=n_samples, no_grad=no_grad)
            X = X.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            w = w.detach().cpu().numpy()
            return X, y, w

    def reset(self):
        self.data_generator.reset()




def load_calib_tes(DATA_NAME, BENCHMARK_NAME):
    from model.regressor import Regressor
    from archi.reducer import A3ML3 as CALIB_ARCHI
    args = lambda : None
    args.n_unit     = 300
    args.optimizer_name  = "adam"
    args.beta1      = 0.5
    args.beta2      = 0.9
    args.learning_rate = 1e-4
    args.sample_size  = 10000
    args.n_steps    = 5000
    args.batch_size = 20

    args.net = CALIB_ARCHI(n_in=29, n_out=2, n_unit=args.n_unit)
    args.optimizer = get_optimizer(args)
    model = get_model(args, Regressor)
    model.base_name = "Calib_tes"
    model.set_info(DATA_NAME, BENCHMARK_NAME, 0)
    model.load(model.model_path)
    return model


def load_calib_jes(DATA_NAME, BENCHMARK_NAME):
    from model.regressor import Regressor
    from archi.reducer import A3ML3 as CALIB_ARCHI
    args = lambda : None
    args.n_unit     = 300
    args.optimizer_name  = "adam"
    args.beta1      = 0.5
    args.beta2      = 0.9
    args.learning_rate = 1e-4
    args.sample_size  = 10000
    args.n_steps    = 5000
    args.batch_size = 20

    args.net = CALIB_ARCHI(n_in=29, n_out=2, n_unit=args.n_unit)
    args.optimizer = get_optimizer(args)
    model = get_model(args, Regressor)
    model.base_name = "Calib_jes"
    model.set_info(DATA_NAME, BENCHMARK_NAME, 0)
    model.load(model.model_path)
    return model


def load_calib_les(DATA_NAME, BENCHMARK_NAME):
    from model.regressor import Regressor
    from archi.reducer import A3ML3 as CALIB_ARCHI
    args = lambda : None
    args.n_unit     = 300
    args.optimizer_name  = "adam"
    args.beta1      = 0.5
    args.beta2      = 0.9
    args.learning_rate = 1e-4
    args.sample_size  = 10000
    args.n_steps    = 5000
    args.batch_size = 20

    args.net = CALIB_ARCHI(n_in=29, n_out=2, n_unit=args.n_unit)
    args.optimizer = get_optimizer(args)
    model = get_model(args, Regressor)
    model.base_name = "Calib_les"
    model.set_info(DATA_NAME, BENCHMARK_NAME, 0)
    model.load(model.model_path)
    return model
