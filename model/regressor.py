# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import numpy as np

import torch
import torch.optim as optim

from sklearn.base import BaseEstimator

from .monitor import LightLossMonitorHook

from .utils import to_torch
# from .utils import to_numpy

from archi.losses import RegressorLoss


class Regressor(BaseEstimator):
    def __init__(self, net, n_steps=5000, batch_size=2000, learning_rate=1e-3, cuda=False, verbose=0):
        super().__init__()
        self.n_steps    = n_steps
        self.batch_size = batch_size
        self.cuda_flag  = cuda
        self.verbose    = verbose

        self.net           = net
        self.learning_rate = learning_rate
        self.optimizer     = optim.Adam(self.net.parameters(), lr=learning_rate)
        self.criterion     = RegressorLoss()

        # self.loss_hook = LightLossMonitorHook()
        # self.criterion.register_forward_hook(self.loss_hook)
        self.losses = []
        self.mse_losses = []
        if cuda:
            self.cuda()

    def cuda(self, device=None):
        self.net = self.net.cuda(device=device)
        self.criterion = self.criterion.cuda(device=device)

    def cpu(self):
        self.net = self.net.cpu()
        self.criterion = self.criterion.cpu()

    def fit(self, generator):
        for i in range(self.n_steps):
            params = self.param_generator()
            X, y, w = generator.generate(*params, n_samples=self.batch_size)
            target = np.sum(w[y==0]) / np.sum(w)
            
            target = target.astype(np.float32)
            target = to_torch(target.reshape(-1), cuda=self.cuda_flag)
            X = X.astype(np.float32)
            w = w.astype(np.float32).reshape(-1, 1)
            X_torch = to_torch(X, cuda=self.cuda_flag)
            w_torch = to_torch(w, cuda=self.cuda_flag)
            X_out = self.net.forward(X_torch, w_torch)
            mu, logsigma = torch.split(X_out, 1, dim=0)
            loss, mse = self.criterion(mu, target, logsigma)
            self.losses.append(loss.item())
            self.mse_losses.append(mse.item())

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def fit_batch(self, generator):
        for i in range(self.n_steps):
            losses = []
            mse_losses = []
            for j in range(15):
                params = self.param_generator()
                X, y, w = generator.generate(*params, n_samples=self.batch_size)
                target = np.sum(w[y==0]) / np.sum(w)
                
                target = target.astype(np.float32)
                target = to_torch(target.reshape(-1), cuda=self.cuda_flag)
                X = X.astype(np.float32)
                w = w.astype(np.float32).reshape(-1, 1)
                X_torch = to_torch(X, cuda=self.cuda_flag)
                w_torch = to_torch(w, cuda=self.cuda_flag)
                X_out = self.net.forward(X_torch, w_torch)
                mu, logsigma = torch.split(X_out, 1, dim=0)
                loss, mse = self.criterion(mu, target, logsigma)
                losses.append(loss.view(1, 1))
                mse_losses.append(mse.view(1, 1))
            loss = torch.mean( torch.cat(losses), 0 )
            mse  = torch.mean( torch.cat(mse_losses), 0 )
            self.losses.append(loss.item())
            self.mse_losses.append(mse.item())

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict(self, X, w):
        X = X.astype(np.float32)
        w = w.astype(np.float32).reshape(-1, 1)
        X_torch = to_torch(X, cuda=self.cuda_flag)
        w_torch = to_torch(w, cuda=self.cuda_flag)
        X_out = self.net.forward(X_torch, w_torch)
        mu, logsigma = torch.split(X_out, 1, dim=0)
        mu = mu.item()
        sigma = np.exp(logsigma.item())
        return mu, sigma


    def save(self, dir_path):
        path = os.path.join(dir_path, 'weights.pth')
        torch.save(self.net.state_dict(), path)

        # path = os.path.join(dir_path, 'losses.json')
        # self.loss_hook.save_state(path)
        return self

    def load(self, dir_path):
        path = os.path.join(dir_path, 'weights.pth')
        if self.cuda_flag:
            self.net.load_state_dict(torch.load(path))
        else:
            self.net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

        # path = os.path.join(dir_path, 'losses.json')
        # self.loss_hook.load_state(path)
        return self

    def describe(self):
        return dict(name='regressor', learning_rate=self.learning_rate,
                    n_steps=self.n_steps, batch_size=self.batch_size)

    def get_name(self):
        name = "Regressor-{}-{}-{}".format(self.n_steps, self.batch_size, self.learning_rate)
        return name

