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

from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.ensemble import GradientBoostingClassifier

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
        self.scaler = None
        self._reset_losses()
        if cuda:
            self.cuda()

    def cuda(self, device=None):
        self.net = self.net.cuda(device=device)
        self.criterion = self.criterion.cuda(device=device)

    def cpu(self):
        self.net = self.net.cpu()
        self.criterion = self.criterion.cpu()

    def get_losses(self):
        losses = dict(loss=self.losses, mse_loss=self.mse_losses, msre_sigma=self.msre_sigma_losses)
        return losses

    def reset(self):
        self._reset_losses()
        if self.scaler:
            self.scaler = StandardScaler()

    def _reset_losses(self):
        self.losses = []
        self.mse_losses = []
        self.msre_sigma_losses = []

    def fit(self, generator):
        X, _, _, _ = generator.generate(n_samples=None)
        self.scaler = StandardScaler().fit(X)
        if self.batch_size > 1:
            return self._fit_batch(generator)
        else:
            return self._fit(generator)

    def _fit(self, generator):
        self.net.train()  # train mode
        for i in range(self.n_steps):
            loss, mse, msre_sigma = self._forward(generator)

            self.losses.append(loss.item())
            self.mse_losses.append(mse.item())
            self.msre_sigma_losses.append(msre_sigma.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        return self

    def _fit_batch(self, generator):
        self.net.train()  # train mode
        self.hasnan = False
        self.optimizer.zero_grad()
        for i in range(self.n_steps):
            losses = []
            mse_losses = []
            msre_sigma_losses = []
            for j in range(self.batch_size):
                loss, mse, msre_sigma = self._forward(generator)
                losses.append(loss.item())
                mse_losses.append(mse.item())
                msre_sigma_losses.append(msre_sigma.item())
                loss.backward()
                # mse.backward()

            loss = np.mean( losses )
            mse  = np.mean( mse_losses )
            msre_sigma  = np.mean( msre_sigma_losses )
            self.losses.append(loss)
            self.mse_losses.append(mse)
            self.msre_sigma_losses.append(msre_sigma)

            if self.verbose:
                print(f"---- {i:5d} loss={loss}, mse={mse}, msre_sigma={msre_sigma} ")
            # Backward
            if np.isnan(loss.item()):
                print("="*50)
                print('/!\\ NaN detected at iter /!\\ ', i)
                print("mse = ", mse.item())
                print("msre_sigma = ", msre_sigma.item())
                print("="*50)
                self.hasnan = True
                break
            else:
                grad_rescale = 1.0/self.batch_size
                for p in self.net.parameters():
                    p.grad *= grad_rescale
                # torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1, norm_type=2)
                # self.chocolat = [p.grad for p in self.net.parameters()]
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

    def _forward(self, generator):
        X, y, w, p = generator.generate(n_samples=self.sample_size)
        X = self.scaler.transform(X)

        X = X.astype(np.float32)
        w = w.astype(np.float32).reshape(-1, 1)
        y = np.array(y).astype(np.float32)
        p = np.array(p).astype(np.float32).reshape(1, -1) if p is not None else None

        X_torch = to_torch(X, cuda=self.cuda_flag)
        w_torch = to_torch(w, cuda=self.cuda_flag)
        y_torch = to_torch(y.reshape(-1), cuda=self.cuda_flag)
        p_torch = to_torch(p, cuda=self.cuda_flag) if p is not None else None

        X_out = self.net.forward(X_torch, w_torch, p_torch)
        target, logsigma = torch.split(X_out, 1, dim=0)
        loss, mse, msre_sigma = self.criterion(target, y_torch, logsigma)
        print("X.isnan().any() = ", torch.isnan(X_torch).byte().any() == 1 )
        print("X.mean(), X.std() = ", X.mean(), X.std() )
        print("X_out = ", X_out, "=?=", "target = ", y)
        if self.verbose and (self.verbose > 1 or np.abs(target.item()) > 5) :
            print(f"logsigma={logsigma.item()}  loss={loss.item()} ")
            print(f"target={y.item()}  predict={target.item()}   mse={mse.item()}")
            print(f"mean={X.mean(axis=0)}  std={X.std(axis=0)}  max={X.max(axis=0)}  min={X.min(axis=0)}  ")
        return loss, mse, msre_sigma


    def predict(self, X, w, p=None):
        X = self.scaler.transform(X)
        X = X.astype(np.float32)
        w = w.astype(np.float32).reshape(-1, 1)
        p = p.astype(np.float32).reshape(1, -1) if p is not None else None

        self.net.eval()  # evaluation mode
        with torch.no_grad():
            X_torch = to_torch(X, cuda=self.cuda_flag)
            w_torch = to_torch(w, cuda=self.cuda_flag)
            p_torch = to_torch(p, cuda=self.cuda_flag) if p is not None else None
            X_out = self.net.forward(X_torch, w_torch, p_torch)
            target, logsigma = torch.split(X_out, 1, dim=0)
        target = target.item()
        sigma = np.exp(logsigma.item())
        return target, sigma

    def save(self, save_directory):
        super(BaseModel, self).save(save_directory)
        path = os.path.join(save_directory, 'weights.pth')
        torch.save(self.net.state_dict(), path)

        path = os.path.join(save_directory, 'Scaler.joblib')
        joblib.dump(self.scaler, path)

        path = os.path.join(save_directory, 'losses.json')
        with open(path, 'w') as f:
            json.dump(self.get_losses(), f)
        return self

    def load(self, save_directory):
        super(BaseModel, self).load(save_directory)
        path = os.path.join(save_directory, 'weights.pth')
        if self.cuda_flag:
            self.net.load_state_dict(torch.load(path))
        else:
            self.net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

        path = os.path.join(save_directory, 'Scaler.joblib')
        self.scaler = joblib.load(path)

        path = os.path.join(save_directory, 'losses.json')
        with open(path, 'r') as f:
            losses_to_load = json.load(f)
        self.losses = losses_to_load['loss']
        self.mse_losses = losses_to_load['mse_loss']
        self.msre_sigma_losses = losses_to_load['msre_sigma']
        return self

    def get_name(self):
        name = "{base_name}-{archi_name}-{optimizer_name}-{n_steps}-{batch_size}-{sample_size}".format(**self.__dict__)
        return name




class ClfRegressor(Regressor):
    def __init__(self, net, optimizer, n_steps=5000, batch_size=20, sample_size=1000, 
                cuda=False, verbose=0):
        super().__init__(net, optimizer, n_steps=n_steps, batch_size=batch_size, sample_size=sample_size, 
                cuda=cuda, verbose=verbose)
        self.clf = GradientBoostingClassifier()


    def fit(self, generator):
        X, y, w = generator.clf_generate(n_samples=None)
        # self.clf.fit(X, y)
        self.scaler = StandardScaler().fit(X)
        if self.batch_size > 1:
            return self._fit_batch(generator)
        else:
            return self._fit(generator)


    def _forward(self, generator):
        X, y, w, p, labels = generator.generate(n_samples=self.sample_size)
        # proba = self.clf.predict_proba(X)
        # X = self.scaler.transform(X)
        # X = np.concatenate([X, proba, y.reshape(-1, 1)], axis=1)
        # X = np.concatenate([X, y.reshape(-1, 1)], axis=1)
        X = labels.reshape(-1, 1)

        X = X.astype(np.float32)
        w = w.astype(np.float32).reshape(-1, 1)
        y = np.array(y).astype(np.float32)
        p = np.array(p).astype(np.float32).reshape(1, -1) if p is not None else None

        X_torch = to_torch(X, cuda=self.cuda_flag)
        w_torch = to_torch(w, cuda=self.cuda_flag)
        y_torch = to_torch(y.reshape(-1), cuda=self.cuda_flag)
        p_torch = to_torch(p, cuda=self.cuda_flag) if p is not None else None

        X_out = self.net.forward(X_torch, w_torch, p_torch)
        target, logsigma = torch.split(X_out, 1, dim=0)
        loss, mse, msre_sigma = self.criterion(target, y_torch, logsigma)
        if self.verbose and (self.verbose > 1 or np.abs(target.item()) > 5) :
            print(f"logsigma={logsigma.item()}  loss={loss.item()} ")
            print(f"target={y.item()}  predict={target.item()}   mse={mse.item()}")
            print(f"mean={X.mean(axis=0)}  std={X.std(axis=0)}  max={X.max(axis=0)}  min={X.min(axis=0)}  ")
        return loss, mse, msre_sigma


    def predict(self, X, w, p=None):
        # proba = self.clf.predict_proba(X)
        # X = self.scaler.transform(X)
        # X = np.concatenate([X, proba], axis=1)
        X = X.reshape(-1, 1)
        X = X.astype(np.float32)
        w = w.astype(np.float32).reshape(-1, 1)
        p = p.astype(np.float32).reshape(1, -1) if p is not None else None

        with torch.no_grad():
            X_torch = to_torch(X, cuda=self.cuda_flag)
            w_torch = to_torch(w, cuda=self.cuda_flag)
            p_torch = to_torch(p, cuda=self.cuda_flag) if p is not None else None
            X_out = self.net.forward(X_torch, w_torch, p_torch)
            target, logsigma = torch.split(X_out, 1, dim=0)
        target = target.item()
        sigma = np.exp(logsigma.item())
        return target, sigma

    def save(self, save_directory):
        """Save the model in the given directory"""
        super().save(save_directory)
        path = os.path.join(save_directory, 'GradientBoosting.joblib')
        joblib.dump(self.clf, path)
        return self

    def load(self, save_directory):
        """Load the model of the i-th CV from the given directory"""
        super().load(save_directory)
        path = os.path.join(save_directory, 'GradientBoosting.joblib')
        self.clf = joblib.load(path)
        return self
