import torch
# import torch.nn as nn

import os
import json
import numpy as np

from collections import OrderedDict
from .base import BaseModel
from .base import BaseNeuralNet
from .utils import to_torch
from .criterion import WeightedCrossEntropyLoss
from .criterion import WeightedTPLoss

# from hessian import hessian


class TangentPropClassifier(BaseModel, BaseNeuralNet):
    def __init__(self, net, trade_off, optimizer, n_steps=5000, sample_size=1500,
                temperature=1.0, cuda=False, verbose=0):
        super().__init__()
        self.n_steps    = n_steps
        self.trade_off = trade_off
        self.sample_size = sample_size
        self.temperature= temperature
        self.cuda_flag  = cuda
        self.verbose    = verbose

        self.net           = net
        self.archi_name    = net.name
        self.optimizer     = optimizer
        self.set_optimizer_name()
        self.jac_criterion  = WeightedTPLoss()
        self.cross_entropy = WeightedCrossEntropyLoss()
        self._reset_losses()
        if cuda:
            self.cuda()

    def cuda(self, device=None):
        self.net = self.net.cuda(device=device)
        self.jac_criterion = self.jac_criterion.cuda(device=device)
        self.cross_entropy = self.cross_entropy.cuda(device=device)

    def cpu(self):
        self.net = self.net.cpu()
        self.jac_criterion = self.jac_criterion.cpu()
        self.cross_entropy = self.cross_entropy.cpu()

    def get_losses(self):
        losses = dict(loss=self.loss
                    ,cross_entropy_loss = self.cross_entropy_loss
                    ,jac_loss = self.jac_loss
                    )
        return losses

    def _reset_losses(self):
        self.loss = []
        self.cross_entropy_loss = []
        self.jac_loss = []
        
    def fit(self, generator):
        for i in range(self.n_steps):
            s, w_s, b, w_b, y_batch = generator.generate(self.sample_size)
            X_batch = torch.cat([s, b], axis=0)
            w_batch = torch.cat([w_s, w_b], axis=0)
            self.optimizer.zero_grad()  # zero-out the gradients because they accumulate by default

            logits = self.net(X_batch)
            cross_entropy_loss = self.cross_entropy(logits, y_batch, w_batch)
            jac_loss = self.jac_criterion(logits, w_batch, generator.nuisance_params)
            loss = cross_entropy_loss + self.trade_off * jac_loss
            self.loss.append(loss.item())
            self.jac_loss.append(jac_loss.item())
            self.cross_entropy_loss.append(cross_entropy_loss.item())
            # loss.backward(retain_graph=True)
            loss.backward()
            self.optimizer.step()  # update params

    def predict(self, X):
        proba = self.predict_proba(X)
        y_pred = np.argmax(proba, axis=1)
        return y_pred

    def predict_proba(self, X):
        X = X.astype(np.float32)
        with torch.no_grad():
            X_torch = to_torch(X, cuda=self.cuda_flag)
            logits = self.net(X_torch)
            probas = torch.softmax(logits / self.temperature, 1)
        y_proba = np.array(probas.cpu())
        return y_proba
    
    def compute_summaries(self, X, W, n_bins=None):
        proba = self.predict_proba(X)
        weighted_counts = np.sum(proba*W.reshape(-1,1), 0)
        return weighted_counts

    def save(self, save_directory):
        super().save(save_directory)
        path = os.path.join(save_directory, 'weights.pth')
        torch.save(self.net.state_dict(), path)

        path = os.path.join(save_directory, 'losses.json')
        with open(path, 'w') as f:
            json.dump(self.get_losses(), f)
        return self

    def load(self, save_directory):
        super().load(save_directory)
        path = os.path.join(save_directory, 'weights.pth')
        if self.cuda_flag:
            self.net.load_state_dict(torch.load(path))
        else:
            self.net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

        path = os.path.join(save_directory, 'losses.json')
        with open(path, 'r') as f:
            losses_to_load = json.load(f)
        self.loss = losses_to_load['loss']
        self.cross_entropy_loss = losses_to_load['cross_entropy_loss']
        self.jac_loss = losses_to_load['jac_loss']
        return self


    def get_name(self):
        name = "{base_name}-{archi_name}-{optimizer_name}-{n_steps}-{sample_size}-{temperature}".format(**self.__dict__)
        return name


