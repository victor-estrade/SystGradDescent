import torch
# import torch.nn as nn

import os
import copy
import json
import numpy as np

from collections import OrderedDict
from .base import BaseClassifierModel
from .base import BaseNeuralNet
from .utils import to_torch
# from .monitor import LightLossMonitorHook

# from hessian import hessian


class Inferno(BaseClassifierModel, BaseNeuralNet):
    def __init__(self, net, criterion, optimizer, n_steps=5000, sample_size=1500,
                temperature=1.0, cuda=False, verbose=0):
        super().__init__()
        self.n_steps    = n_steps
        self.sample_size = sample_size
        self.temperature= temperature
        self.cuda_flag  = cuda
        self.verbose    = verbose

        self.net           = net
        self.archi_name    = net.name
        self.optimizer     = optimizer
        self.set_optimizer_name()
        self.criterion     = criterion

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
        losses = dict(loss=self.losses)
        return losses
        
    def reset(self):
        self._reset_losses()

    def _reset_losses(self):
        self.losses = []

    def fit(self, generator):
        checkpoint = copy.deepcopy(self.net.state_dict())
        mu = torch.tensor(1.0, requires_grad=True, device="cuda" if self.cuda_flag else 'cpu')
        mu_prime = mu.detach()
        params = OrderedDict([('mu', mu)])
        params.update(generator.nuisance_params)

        for i in range(self.n_steps):
            if (i % 100) == 0:
                checkpoint = copy.deepcopy(self.net.state_dict())
            s, w_s, b, w_b, y = generator.generate(self.sample_size)
            s_prime, b_prime = s.detach(), b.detach()
            self.optimizer.zero_grad()  # zero-out the gradients because they accumulate by default

            s_counts = self.forward(s, w_s)
            b_counts = self.forward(b, w_b)
            s_prime_counts = self.forward(s_prime, w_s.detach())
            b_prime_counts = self.forward(b_prime, w_b.detach())

            total_count = mu * s_counts + b_counts # if NaN then should be mu s + b + epsilon ?
            asimov = mu_prime * s_prime_counts + b_prime_counts # if NaN then should be mu s + b + epsilon ?
            loss = self.criterion(total_count, asimov, params)
            self.losses.append(loss.item())
            
            if self._is_bad_training(i, total_count, loss):
                print('NaN detected at ', i)
                print('output', total_count)
                print('loss', loss)
                print('-'*40)
                self.net.load_state_dict(checkpoint)
                continue
            else:
                # loss.backward(retain_graph=True)
                loss.backward()
                is_all_finite_grad = True
                for name, param in self.net.named_parameters():
                    if not torch.isfinite(param.grad).all():
                        print("i = ", i, "found non finite gradients in", name)
                        is_all_finite_grad = False
                        break
                if is_all_finite_grad : 
                    self.optimizer.step()  # update params

    def _is_bad_training(self, i, total_count, loss):
        loss_flag = torch.isnan(loss).byte().any() 
        output_flag = torch.isnan(total_count).byte().any()
        flag = loss_flag or output_flag
        return flag

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
    
    def forward(self, x, w):
        logits = self.net(x)
        probas = torch.softmax(logits / self.temperature, 1)
        counts = torch.sum(probas * w, 0, keepdim=False)
        return counts

    def compute_summaries(self, X, W, n_bins=None):
        proba = self.predict_proba(X)
        weighted_counts = np.sum(proba*W.reshape(-1,1), 0)
        return weighted_counts

    def save(self, save_directory):
        super(BaseModel, self).save(save_directory)
        path = os.path.join(save_directory, 'weights.pth')
        torch.save(self.net.state_dict(), path)

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

        path = os.path.join(save_directory, 'losses.json')
        with open(path, 'r') as f:
            losses_to_load = json.load(f)
        self.losses = losses_to_load['loss']
        return self


    def get_name(self):
        name = "{base_name}-{archi_name}-{optimizer_name}-{n_steps}-{sample_size}-{temperature}".format(**self.__dict__)
        return name


