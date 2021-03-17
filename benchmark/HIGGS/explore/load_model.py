# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals


def load_some_GB():
    from model.gradient_boost import GradientBoostingModel
    i_cv = 0
    model = GradientBoostingModel(learning_rate=0.1, n_estimators=300, max_depth=3)
    model.set_info(DATA_NAME, BENCHMARK_NAME, i_cv)
    print(f"loading {model.model_path}")
    model.load(model.model_path)
    return model

def load_some_NN():
    from model.neural_network import NeuralNetClassifier
    from archi.classic import L4 as ARCHI
    import torch.optim as optim
    i_cv = 0
    n_unit = 200
    net = ARCHI(n_in=29, n_out=2, n_unit=n_unit)
    optimizer = optim.Adam(lr=1e-4)
    model = NeuralNetClassifier(net, optimizer, n_steps=5000, batch_size=20, cuda=False)
    model.set_info(DATA_NAME, BENCHMARK_NAME, i_cv)
    print(f"loading {model.model_path}")
    model.load(model.model_path)
