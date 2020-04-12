# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import json
import numpy as np

import torch

from .base import BaseModel
from .base import BaseNeuralNet

# from .monitor import LightLossMonitorHook

from .utils import to_torch
# from .utils import to_numpy

from .criterion import GaussNLLLoss


class Regressor(BaseModel, BaseNeuralNet):
    def __init__(self, net, optimizer, n_steps=5000, batch_size=20, sample_size=1000, 
                cuda=False, verbose=0):
        super().__init__()
        self.n_steps     = n_steps
        self.batch_size  = batch_size
        self.sample_size = sample_size
        self.cuda_flag   = cuda
        self.verbose     = verbose

        self.net           = net
        self.archi_name    = net.name
        self.optimizer     = optimizer
        self.set_optimizer_name()
        self.scheduler     = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.6)
        self.criterion     = GaussNLLLoss()

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

    def get_losses(self):
        losses = dict(loss=self.losses, mse_loss=self.mse_losses)
        return losses

    def fit(self, generator):
        if self.batch_size > 1:
            return self._fit_batch(generator)
        else:
            return self._fit(generator)

    def _fit(self, generator):
        for i in range(self.n_steps):
            loss, mse = self._forward(generator)

            self.losses.append(loss.item())
            self.mse_losses.append(mse.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        return self

    def _fit_batch(self, generator):
        for i in range(self.n_steps):
            losses = []
            mse_losses = []
            for j in range(self.batch_size):
                loss, mse = self._forward(generator)
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
            self.scheduler.step()

    def _forward(self, generator):
        X, y, w, p = generator.generate(n_samples=self.sample_size)
        
        X = X.astype(np.float32)
        w = w.astype(np.float32).reshape(-1, 1)
        y = np.array(y).astype(np.float32)
        p = np.array(p).astype(np.float32).reshape(1, -1) if p is not None else None

        X_torch = to_torch(X, cuda=self.cuda_flag)
        w_torch = to_torch(w, cuda=self.cuda_flag)
        y_torch = to_torch(y.reshape(-1), cuda=self.cuda_flag)
        p_torch = to_torch(p, cuda=self.cuda_flag) if p is not None else None

        X_out = self.net.forward(X_torch, w_torch, p_torch)
        mu, logsigma = torch.split(X_out, 1, dim=0)
        loss, mse = self.criterion(mu, y_torch, logsigma)
        return loss, mse


    def predict(self, X, w, p=None):
        X = X.astype(np.float32)
        w = w.astype(np.float32).reshape(-1, 1)
        p = p.astype(np.float32).reshape(1, -1) if p is not None else None
        X_torch = to_torch(X, cuda=self.cuda_flag)
        w_torch = to_torch(w, cuda=self.cuda_flag)
        p_torch = to_torch(p, cuda=self.cuda_flag) if p is not None else None
        X_out = self.net.forward(X_torch, w_torch, p_torch)
        mu, logsigma = torch.split(X_out, 1, dim=0)
        mu = mu.item()
        sigma = np.exp(logsigma.item())
        return mu, sigma

    def save(self, save_directory):
        super(BaseModel, self).save(save_directory)
        path = os.path.join(save_directory, 'weights.pth')
        torch.save(self.net.state_dict(), path)

        path = os.path.join(save_directory, 'losses.json')
        losses_to_save = dict(losses=self.losses, mse_losses=self.mse_losses)
        with open(path, 'w') as f:
            json.dump(losses_to_save, f)
        return self

    def load(self, save_directory):
        super(BaseModel, self).load(save_directory)
        path = os.path.join(save_directory, 'weights.pth')
        if self.cuda_flag:
            self.net.load_state_dict(torch.load(path))
        else:
            self.net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

        path = os.path.join(save_directory, 'losses.json')
        with open(path, 'r') as f:
            losses_to_load = json.load(f)
        self.losses = losses_to_load['losses']
        self.mse_losses = losses_to_load['mse_losses']
        return self

    def describe(self):
        return dict(name=self.basic_name, learning_rate=self.learning_rate,
                    n_steps=self.n_steps, batch_size=self.batch_size)

    def get_name(self):
        name = "{base_name}-{archi_name}-{optimizer_name}-{n_steps}-{batch_size}-{sample_size}".format(**self.__dict__)
        return name

