# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

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
