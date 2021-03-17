# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

DATA_NAME = 'HIGGSTES'
BENCHMARK_NAME = DATA_NAME+'-prior'

def load_some_GB(i_cv=0):
    from model.gradient_boost import GradientBoostingModel
    model = GradientBoostingModel(learning_rate=0.1, n_estimators=300, max_depth=3)
    model.set_info(DATA_NAME, BENCHMARK_NAME, i_cv)
    print(f"loading {model.model_path}")
    model.load(model.model_path)
    return model


def load_some_NN(i_cv=0, cuda=False):
    from model.neural_network import NeuralNetClassifier
    from archi.classic import L4 as ARCHI
    import torch.optim as optim
    n_unit = 200
    net = ARCHI(n_in=29, n_out=2, n_unit=n_unit)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    model = NeuralNetClassifier(net, optimizer, n_steps=5000, batch_size=10000, cuda=cuda)
    model.set_info(DATA_NAME, BENCHMARK_NAME, i_cv)
    print(f"loading {model.model_path}")
    model.load(model.model_path)
    return model
