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
        self.base_name   = "Regressor"
        self.n_steps     = n_steps
        self.batch_size  = batch_size
        self.sample_size = sample_size
        self.cuda_flag   = cuda
        self.verbose     = verbose

        self.net           = net
        self.archi_name    = net.name
        self.optimizer     = optimizer
        self.set_optimizer_name()
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

    def fit(self, generator):
        for i in range(self.n_steps):
            loss, mse = self._forward(generator)

            self.losses.append(loss.item())
            self.mse_losses.append(mse.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return self

    def fit_batch(self, generator):
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

    def _forward(self, generator):
        params = self.param_generator()
        X, y, w = generator.generate(*params, n_samples=self.sample_size)
        target = params[-1]
        
        X = X.astype(np.float32)
        w = w.astype(np.float32).reshape(-1, 1)
        target = np.array(target).astype(np.float32)
        p = np.array(params[:-1]).astype(np.float32).reshape(1, -1)

        X_torch = to_torch(X, cuda=self.cuda_flag)
        w_torch = to_torch(w, cuda=self.cuda_flag)
        target = to_torch(target.reshape(-1), cuda=self.cuda_flag)
        p_torch = to_torch(p, cuda=self.cuda_flag)

        X_out = self.net.forward(X_torch, w_torch, p_torch)
        mu, logsigma = torch.split(X_out, 1, dim=0)
        loss, mse = self.criterion(mu, target, logsigma)
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

    def many_predict(self, X, w, param_generator, ncall=100):
        all_pred = []
        all_sigma = []
        all_nuisance_params = []
        for _ in range(ncall):
            params = param_generator()
            nuisance_params = np.array(params[:-1])
            pred, sigma = self.predict(X, w, nuisance_params)
            all_pred.append(pred)
            all_sigma.append(sigma)
            all_nuisance_params.append(nuisance_params)
        
        return all_pred, all_sigma, all_nuisance_params


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

