# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import numpy as np

import torch
import torch.nn.functional as F

from .base import BaseClassifierModel
from .base import BaseNeuralNet
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

from .minibatch import EpochShuffle
from .minibatch import OneEpoch

from .utils import to_torch
from .utils import to_numpy

class PivotModel(BaseClassifierModel):
    def __init__(self, perturbator, n_clf_pre_training_steps=10, n_adv_pre_training_steps=10, n_steps=1000,
                 n_recovery_steps=10, batch_size=20, classifier_learning_rate=1e-3, adversarial_learning_rate=1e-3,
                 trade_off=1, width=1, cuda=False, verbose=0):
        super().__init__()
        self.n_clf_pre_training_steps = n_clf_pre_training_steps
        self.n_adv_pre_training_steps = n_adv_pre_training_steps
        self.n_steps = n_steps
        self.n_recovery_steps = n_recovery_steps
        self.batch_size = batch_size
        self.classifier_learning_rate = classifier_learning_rate
        self.adversarial_learning_rate = adversarial_learning_rate
        self.trade_off = trade_off
        self.width = width
        self.cuda = cuda
        self.verbose = verbose

        self.dnet = Net()
        self.net = self.dnet  # alias
        self.rnet = RNet(n_out=3)

        self.doptimizer = optim.Adam(self.dnet.parameters(), lr=classifier_learning_rate)
        self.dcriterion = WeightedCrossEntropyLoss()
        self.dloss_hook = LightLossMonitorHook()
        self.dcriterion.register_forward_hook(self.dloss_hook)
        self.classifier = NeuralNetClassifier(self.dnet, self.dcriterion, self.doptimizer,
                                              n_steps=n_clf_pre_training_steps, batch_size=batch_size, cuda=self.cuda)
        self.clf = self.classifier  # alias
        
        self.roptimizer = optim.Adam(self.rnet.parameters(), lr=adversarial_learning_rate)
        self.rcriterion = WeightedMSELoss()
        self.rloss_hook = LightLossMonitorHook()
        self.rcriterion.register_forward_hook(self.rloss_hook)
        self.adversarial = NeuralNetRegressor(self.rnet, self.rcriterion, self.roptimizer,
                                              n_steps=n_adv_pre_training_steps, batch_size=batch_size, cuda=self.cuda)

        self.droptimizer = optim.Adam(list(self.dnet.parameters()) + list(self.rnet.parameters()), lr=adversarial_learning_rate)
        self.pivot = PivotTrainer(self.classifier, self.adversarial, self.droptimizer,
                                  n_steps=self.n_steps, n_recovery_steps=n_recovery_steps, batch_size=batch_size,
                                  trade_off=trade_off, cuda=self.cuda)

        self.perturbator = perturbator
        self.scaler = StandardScaler()

    def fit(self, X, y, sample_weight=None):
        X, y, sample_weight, z = self.perturbator(X, y, sample_weight)
        z = (z - 1) / self.width
        X = to_numpy(X)
        y = to_numpy(y)
        sample_weight = to_numpy(sample_weight).reshape(-1, 1)
        X = self.scaler.fit_transform(X)
        self.dloss_hook.reset()
        self.rloss_hook.reset()
        W = balance_training_weight(sample_weight, y) * y.shape[0] / 2
        self.classifier.fit(X, y, sample_weight=W.reshape(-1, 1))  # pre-training
        proba_pred = self.classifier.predict_proba(X)
        self.adversarial.fit(proba_pred, z, sample_weight=W)  # pre-training
        self.pivot.partial_fit(X, y, z, sample_weight=W)
        return self

    def predict(self, X):
        X = to_numpy(X)
        X = self.scaler.transform(X)
        y_pred = self.classifier.predict(X)
        return y_pred

    def predict_proba(self, X):
        X = to_numpy(X)
        X = self.scaler.transform(X)
        proba = self.classifier.predict_proba(X)
        return proba

    def save(self, dir_path):
        path = os.path.join(dir_path, 'dnet_weights.pth')
        torch.save(self.dnet.state_dict(), path)

        path = os.path.join(dir_path, 'rnet_weights.pth')
        torch.save(self.rnet.state_dict(), path)

        path = os.path.join(dir_path, 'Scaler.pkl')
        joblib.dump(self.scaler, path)

        path = os.path.join(dir_path, 'dlosses.json')
        self.dloss_hook.save_state(path)
        path = os.path.join(dir_path, 'rlosses.json')
        self.rloss_hook.save_state(path)
        return self

    def load(self, dir_path):
        path = os.path.join(dir_path, 'dnet_weights.pth')
        if self.cuda:
            self.dnet.load_state_dict(torch.load(path))
        else:
            self.dnet.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

        path = os.path.join(dir_path, 'rnet_weights.pth')
        if self.cuda:
            self.rnet.load_state_dict(torch.load(path))
        else:
            self.rnet.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

        path = os.path.join(dir_path, 'Scaler.pkl')
        self.scaler = joblib.load(path)

        path = os.path.join(dir_path, 'dlosses.json')
        self.dloss_hook.load_state(path)
        path = os.path.join(dir_path, 'rlosses.json')
        self.rloss_hook.load_state(path)
        return self

    def describe(self):
        return dict(name='pivot', n_clf_pre_training_steps=self.n_clf_pre_training_steps,
                    n_adv_pre_training_steps=self.n_adv_pre_training_steps, n_steps=self.n_steps,
                    n_recovery_steps=self.n_recovery_steps, classifier_learning_rate=self.classifier_learning_rate,
                    batch_size=self.batch_size,
                    adversarial_learning_rate=self.adversarial_learning_rate, trade_off=self.trade_off, width=self.width,
                    )

    def get_name(self):
        name = "PivotModel-{}-{}-{}-{}-{}-{}-{}-{}-{}".format(self.n_clf_pre_training_steps, self.n_adv_pre_training_steps, 
                    self.n_steps, self.n_recovery_steps, self.classifier_learning_rate, self.batch_size,
                    self.adversarial_learning_rate, self.trade_off, self.width,
                    )
        return name
