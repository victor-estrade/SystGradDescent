# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import json
import numpy as np

import torch
import torch.nn.functional as F

from itertools import islice

from .base import BaseClassifierModel
from .base import BaseNeuralNet
from sklearn.preprocessing import StandardScaler
import joblib

from .minibatch import EpochShuffle
from .minibatch import OneEpoch

from .utils import to_torch
from .utils import to_numpy
from .utils import classwise_balance_weight

class Pivot(BaseClassifierModel, BaseNeuralNet):
    def __init__(self, net, adv_net, 
                net_criterion, adv_criterion, trade_off,
                net_optimizer, adv_optimizer,
                n_net_pre_training_steps=10, n_adv_pre_training_steps=10,
                n_steps=1000, n_recovery_steps=10,
                batch_size=20, rescale=True, cuda=False, verbose=0):
        super().__init__()
        self.net = net
        self.adv_net = adv_net
        self.archi_name    = f"{net.name}_{adv_net.name}"
        self.net_criterion = net_criterion
        self.adv_criterion = adv_criterion
        self.trade_off = trade_off
        self.net_optimizer = net_optimizer
        self.adv_optimizer = adv_optimizer
        self.optimizer = net_optimizer  # monkey patching for name convention only 
        self.set_optimizer_name()
        self.n_net_pre_training_steps = n_net_pre_training_steps
        self.n_adv_pre_training_steps = n_adv_pre_training_steps
        self.n_steps = n_steps
        self.n_recovery_steps = n_recovery_steps
        self.batch_size = batch_size
        self.verbose = verbose
        if rescale:
            self.scaler = StandardScaler()
        else:
            self.scaler = None
        self._reset_losses()
        self.cuda_flag = cuda
        if cuda:
            self.cuda()

    def cuda(self, device=None):
        self.net = self.net.cuda(device=device)
        self.adv_net = self.adv_net.cuda(device=device)
        self.net_criterion = self.net_criterion.cuda(device=device)
        self.adv_criterion = self.adv_criterion.cuda(device=device)

    def cpu(self):
        self.net = self.net.cpu()
        self.adv_net = self.adv_net.cpu()
        self.net_criterion = self.net_criterion.cpu()
        self.adv_criterion = self.adv_criterion.cpu()

    def get_losses(self):
        losses = dict(net_loss=self.net_loss
                    , adv_loss=self.adv_loss
                    , comb_loss=self.comb_loss
                    , recov_loss=self.recov_loss
                    )
        return losses

    def reset(self):
        self._reset_losses()
        if self.scaler:
            self.scaler = StandardScaler()

    def _reset_losses(self):
        self.net_loss = []
        self.adv_loss = []
        self.comb_loss = []
        self.recov_loss = []

    def fit(self, X, y, z, sample_weight=None):
        X, y, z, w = self._prepare(X, y, z, sample_weight=sample_weight)
        # Pre-training classifier
        net_generator = EpochShuffle(X, y, w, batch_size=self.batch_size)
        self._fit_net(net_generator, self.n_net_pre_training_steps)  # pre-training

        # Pre-training adversarial
        adv_generator = EpochShuffle(X, z, w, batch_size=self.batch_size)
        self._fit_adv_net(adv_generator, self.n_adv_pre_training_steps)  # pre-training
        
        # Training
        comb_generator = EpochShuffle(X, y, z, w, batch_size=self.batch_size)
        self._fit_combined(comb_generator, comb_generator, self.n_steps)
        return self

    def _prepare(self, X, y, z, sample_weight=None):
        X = to_numpy(X)
        y = to_numpy(y)
        z = to_numpy(z)
        if sample_weight is None:
            sample_weight = np.ones_like(y, dtype=np.float64)
        else:
            sample_weight = to_numpy(sample_weight)
        # FIXME now Pivot canot be a regressor ... Breaks inheritage !
        w = classwise_balance_weight(sample_weight, y)
        # Preprocessing
        if self.scaler is not None:
            X = self.scaler.fit_transform(X)
        # to cuda friendly types
        X = X.astype(np.float32)
        z = z.astype(np.float32)
        y = y.astype(np.int64)
        w = w.astype(np.float32)
        return X, y, z, w


    def _fit_net(self, generator, n_steps):
        self.net.train()  # train mode
        for i, (X_batch, y_batch, w_batch) in enumerate(islice(generator, n_steps)):
            X_batch = to_torch(X_batch, cuda=self.cuda_flag)
            y_batch = to_torch(y_batch, cuda=self.cuda_flag)
            w_batch = to_torch(w_batch, cuda=self.cuda_flag)
            self.net_optimizer.zero_grad()  # zero-out the gradients because they accumulate by default
            y_pred = self.net.forward(X_batch)
            loss = self.net_criterion(y_pred, y_batch, w_batch)
            self.net_loss.append(loss.item())
            loss.backward()  # compute gradients
            self.net_optimizer.step()  # update params
        return self

    def _fit_adv_net(self, generator, n_steps):
        self.adv_net.train()  # train mode
        for i, (X_batch, z_batch, w_batch) in enumerate(islice(generator, n_steps)):
            X_batch = to_torch(X_batch, cuda=self.cuda_flag)
            z_batch = to_torch(z_batch, cuda=self.cuda_flag)
            w_batch = to_torch(w_batch, cuda=self.cuda_flag)
            self.adv_optimizer.zero_grad()  # zero-out the gradients because they accumulate by default
            y_pred = self.net.forward(X_batch)
            z_pred = self.adv_net.forward(y_pred)
            loss = self.adv_criterion(z_pred, z_batch, w_batch)
            self.adv_loss.append(loss.item())
            loss.backward()  # compute gradients
            self.adv_optimizer.step()  # update params
        return self

    def _fit_recovery(self, generator, n_steps):
        self.adv_net.train()  # train mode
        for i, (X_batch, y_batch, z_batch, w_batch) in enumerate(islice(generator, n_steps)):
            X_batch = to_torch(X_batch, cuda=self.cuda_flag)
            y_batch = to_torch(y_batch, cuda=self.cuda_flag)
            z_batch = to_torch(z_batch, cuda=self.cuda_flag)
            w_batch = to_torch(w_batch, cuda=self.cuda_flag)
            self.adv_optimizer.zero_grad()  # zero-out the gradients because they accumulate by default
            y_pred = self.net.forward(X_batch)
            z_pred = self.adv_net.forward(y_pred)
            # net_loss = self.net_criterion(y_pred, y_batch)
            adv_loss = self.adv_criterion(z_pred, z_batch, w_batch)
            # loss = (self.trade_off * adv_loss) #- net_loss
            loss = adv_loss
            self.recov_loss.append(loss.item())
            loss.backward()  # compute gradients
            self.adv_optimizer.step()  # update params
        return self

    def _fit_combined(self, generator, recovery_generator, n_steps):
        self.net.train()  # train mode
        self.adv_net.train()  # train mode
        for i, (X_batch, y_batch, z_batch, w_batch) in enumerate(islice(generator, n_steps)):
            X_batch = to_torch(X_batch, cuda=self.cuda_flag)
            y_batch = to_torch(y_batch, cuda=self.cuda_flag)
            z_batch = to_torch(z_batch, cuda=self.cuda_flag)
            w_batch = to_torch(w_batch, cuda=self.cuda_flag)
            self.net_optimizer.zero_grad()  # zero-out the gradients because they accumulate by default
            y_pred = self.net.forward(X_batch)
            z_pred = self.adv_net.forward(y_pred)
            net_loss = self.net_criterion(y_pred, y_batch, w_batch)
            adv_loss = self.adv_criterion(z_pred, z_batch, w_batch)
            loss = net_loss - (self.trade_off * adv_loss)
            self.adv_loss.append(adv_loss.item())
            self.net_loss.append(net_loss.item())
            self.comb_loss.append(loss.item())
            loss.backward()  # compute gradients
            self.net_optimizer.step()  # update params
            # Adversarial recovery
            self._fit_recovery(recovery_generator, self.n_recovery_steps)
        return self

    def save(self, save_directory):
        path = os.path.join(save_directory, 'net_weights.pth')
        torch.save(self.net.state_dict(), path)

        path = os.path.join(save_directory, 'adv_net_weights.pth')
        torch.save(self.adv_net.state_dict(), path)

        path = os.path.join(save_directory, 'Scaler.joblib')
        joblib.dump(self.scaler, path)

        path = os.path.join(save_directory, 'losses.json')
        with open(path, 'w') as f:
            json.dump(self.get_losses(), f)
        return self

    def load(self, save_directory):
        path = os.path.join(save_directory, 'net_weights.pth')
        if self.cuda:
            self.net.load_state_dict(torch.load(path))
        else:
            self.net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

        path = os.path.join(save_directory, 'adv_net_weights.pth')
        if self.cuda:
            self.adv_net.load_state_dict(torch.load(path))
        else:
            self.adv_net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

        path = os.path.join(save_directory, 'Scaler.joblib')
        self.scaler = joblib.load(path)

        path = os.path.join(save_directory, 'losses.json')
        with open(path, 'r') as f:
            losses_to_load = json.load(f)
        self.net_loss = losses_to_load['net_loss']
        self.adv_loss = losses_to_load['adv_loss']
        self.comb_loss = losses_to_load['comb_loss']
        self.recov_loss = losses_to_load['recov_loss']
        return self

    def get_name(self):
        name = "{base_name}-{archi_name}-{optimizer_name}-{n_steps}-{batch_size}-{trade_off}".format(**self.__dict__)
        return name


class PivotBinaryClassifier(Pivot):

    def _prepare(self, X, y, z):
        X = to_numpy(X)
        y = to_numpy(y)
        z = to_numpy(z)
        # Preprocessing
        if self.scaler is not None:
            X = self.scaler.fit_transform(X)
        # to cuda friendly types
        X = X.astype(np.float32)
        z = z.astype(np.float32).reshape(-1, 1)
        y = y.astype(np.float32).reshape(-1, 1)
        return X, y, z


    def predict(self, X):
        proba = self.predict_proba(X)
        y_pred = np.argmax(proba, axis=1)
        return y_pred

    def predict_proba(self, X):
        X = to_numpy(X)
        if self.scaler is not None :
            X = self.scaler.transform(X)
        proba_s = self._predict_proba(X)
        proba_b = 1-proba_s
        proba = np.concatenate((proba_b, proba_s), axis=1)
        return proba

    def _predict_proba(self, X):
        y_proba = []
        self.net.eval()  # evaluation mode
        for X_batch in OneEpoch(X, batch_size=self.batch_size):
            X_batch = X_batch.astype(np.float32)
            with torch.no_grad():
                X_batch = to_torch(X_batch, cuda=self.cuda_flag)
                proba_batch = torch.sigmoid(self.net.forward(X_batch)).cpu().data.numpy()
            y_proba.extend(proba_batch)
        y_proba = np.array(y_proba)
        return y_proba


class PivotClassifier(Pivot):

    def predict(self, X):
        proba = self.predict_proba(X)
        y_pred = np.argmax(proba, axis=1)
        return y_pred

    def predict_proba(self, X):
        X = to_numpy(X)
        if self.scaler is not None :
            X = self.scaler.transform(X)
        proba = self._predict_proba(X)
        return proba

    def _predict_proba(self, X):
        y_proba = []
        self.net.eval()  # evaluation mode
        for X_batch in OneEpoch(X, batch_size=self.batch_size):
            X_batch = X_batch.astype(np.float32)
            with torch.no_grad():
                X_batch = to_torch(X_batch, cuda=self.cuda_flag)
                proba_batch = F.softmax(self.net.forward(X_batch), dim=1).cpu().data.numpy()
            y_proba.extend(proba_batch)
        y_proba = np.array(y_proba)
        return y_proba




class PivotRegressor(Pivot):
    # TODO

    def predict(self, X):
        X = to_numpy(X)
        if self.scaler is not None :
            X = self.scaler.transform(X)
        # TODO : extract regressed
        return None
