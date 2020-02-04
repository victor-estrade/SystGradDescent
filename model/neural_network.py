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

from .criterion import WeightedCrossEntropyLoss
from .monitor import LightLossMonitorHook

from itertools import islice
from .minibatch import EpochShuffle
from .minibatch import OneEpoch

from .utils import to_torch
from .utils import to_numpy
from .utils import classwise_balance_weight


class NeuralNetClassifier(BaseClassifierModel, BaseNeuralNet):
    def __init__(self, net, optimizer, n_steps=5000, batch_size=20, learning_rate=1e-3, cuda=False, verbose=0):
        super().__init__()
        self.basic_name = "NeuralNetClassifier"
        self.n_steps    = n_steps
        self.batch_size = batch_size
        self.cuda_flag  = cuda
        self.verbose    = verbose

        self.scaler        = StandardScaler()
        self.net           = net
        self.archi_name    = net.__class__.__name__
        self.learning_rate = learning_rate
        self.optimizer     = optimizer
        self.set_optimizer_name()
        self.criterion     = WeightedCrossEntropyLoss()

        self.loss_hook = LightLossMonitorHook()
        self.criterion.register_forward_hook(self.loss_hook)
        if cuda:
            self.cuda()

    def cuda(self, device=None):
        self.net = self.net.cuda(device=device)
        self.criterion = self.criterion.cuda(device=device)

    def cpu(self):
        self.net = self.net.cpu()
        self.criterion = self.criterion.cpu()

    def fit(self, X, y, sample_weight=None):
        # To numpy arrays
        X = to_numpy(X)
        y = to_numpy(y)
        if sample_weight is None:
            sample_weight = np.ones_like(y, dtype=np.float64)
        else:
            sample_weight = to_numpy(sample_weight)
        # Preprocessing
        X = self.scaler.fit_transform(X)
        w = classwise_balance_weight(sample_weight, y)
        # to cuda friendly types
        X = X.astype(np.float32)
        w = w.astype(np.float32)
        y = y.astype(np.int64)
        # Reset model
        self.loss_hook.reset()
        self.net.reset_parameters()
        # Train
        self._fit(X, y, w)
        return self

    def _fit(self, X, y, w):
        """Training loop. Asumes that preprocessing is done."""
        batch_size = self.batch_size
        n_steps = self.n_steps
        batch_gen = EpochShuffle(X, y, w, batch_size=batch_size)
        self.net.train()  # train mode
        for i, (X_batch, y_batch, w_batch) in enumerate(islice(batch_gen, n_steps)):
            X_batch = to_torch(X_batch, cuda=self.cuda_flag)
            w_batch = to_torch(w_batch, cuda=self.cuda_flag)
            y_batch = to_torch(y_batch, cuda=self.cuda_flag)
            self.optimizer.zero_grad()  # zero-out the gradients because they accumulate by default
            y_pred = self.net.forward(X_batch)
            loss = self.criterion(y_pred, y_batch, w_batch)
            loss.backward()  # compute gradients
            self.optimizer.step()  # update params
        return self

    def predict(self, X):
        proba = self.predict_proba(X)
        y_pred = np.argmax(proba, axis=1)
        return y_pred

    def predict_proba(self, X):
        X = to_numpy(X)
        X = self.scaler.transform(X)
        proba = self._predict_proba(X)
        return proba

    def _predict_proba(self, X):
        batch_gen = OneEpoch(X, batch_size=self.batch_size)
        y_proba = []
        self.net.eval()  # evaluation mode
        for X_batch in batch_gen:
            X_batch = X_batch.astype(np.float32)
            with torch.no_grad():
                X_batch = to_torch(X_batch, cuda=self.cuda_flag)
                proba_batch = F.softmax(self.net.forward(X_batch), dim=1).cpu().data.numpy()
            y_proba.extend(proba_batch)
        y_proba = np.array(y_proba)
        return y_proba

    def save(self, save_directory):
        super().save(save_directory)
        path = os.path.join(save_directory, 'weights.pth')
        torch.save(self.net.state_dict(), path)

        path = os.path.join(save_directory, 'Scaler.pkl')
        joblib.dump(self.scaler, path)

        path = os.path.join(save_directory, 'losses.json')
        self.loss_hook.save_state(path)
        return self

    def load(self, save_directory):
        super().load(save_directory)
        path = os.path.join(save_directory, 'weights.pth')
        if self.cuda_flag:
            self.net.load_state_dict(torch.load(path))
        else:
            self.net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

        path = os.path.join(save_directory, 'Scaler.pkl')
        self.scaler = joblib.load(path)

        path = os.path.join(save_directory, 'losses.json')
        self.loss_hook.load_state(path)
        return self

    def describe(self):
        return dict(name=self.basic_name, learning_rate=self.learning_rate,
                    n_steps=self.n_steps, batch_size=self.batch_size)

    def get_name(self):
        name = "{basic_name}-{archi_name}-{optimizer_name}-{n_steps}-{batch_size}".format(**self.__dict__)
        return name


class AugmentedNeuralNetModel(NeuralNetClassifier):
    def __init__(self, net, augmenter, n_steps=5000, batch_size=20, learning_rate=1e-3, width=1, n_augment=2,
                 cuda=False, verbose=0):
        super().__init__(net, n_steps=n_steps, batch_size=batch_size, 
                        learning_rate=learning_rate, cuda=cuda, verbose=verbose)
        self.basic_name = "AugmentedNeuralNetClf"
        self.width = width
        self.n_augment = n_augment
        self.augmenter = augmenter

    def fit(self, X, y, sample_weight=None):
        X, y, sample_weight, z = self.augmenter(X, y, sample_weight)
        super().fit(X, y, sample_weight)
        return self

    def describe(self):
        return dict(name='augmented_neural_net', learning_rate=self.learning_rate,
                    n_steps=self.n_steps, batch_size=self.batch_size, width=self.width, n_augment=self.n_augment)

    def get_name(self):
        name = "{basic_name}-{n_steps}-{batch_size}-{learning_rate}-{width}-{n_augment}".format(**self.__dict__)
        return name


class BlindNeuralNetModel(NeuralNetClassifier):
    def __init__(self, net, n_steps=5000, batch_size=20, learning_rate=1e-3, cuda=False, verbose=0):
        super().__init__(net, n_steps=n_steps, batch_size=batch_size, 
                        learning_rate=learning_rate, cuda=cuda, verbose=verbose)
        self.basic_name = "BlindNeuralNetClf"
        self.skewed_idx = [0, 1, 8, 9, 10, 12, 18, 19]
        # ['DER_mass_transverse_met_lep', 'DER_mass_vis', 'DER_sum_pt',
        #  'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality', 'PRI_tau_pt',
        #  'PRI_met', 'PRI_met_phi']

    def fit(self, X, y, sample_weight=None):
        X = to_numpy(X)
        X = np.delete(X, self.skewed_idx, axis=1)        
        super().fit(X, y, sample_weight)
        return self

    def predict(self, X):
        proba = self.predict_proba(X)
        y_pred = np.argmax(proba, axis=1)
        return y_pred

    def predict_proba(self, X):
        X = to_numpy(X)
        X = np.delete(X, self.skewed_idx, axis=1)        
        X = self.scaler.transform(X)
        proba = self._predict_proba(X)
        return proba

    def describe(self):
        return dict(name='blind_neural_net', learning_rate=self.learning_rate,
                    n_steps=self.n_steps, batch_size=self.batch_size)
