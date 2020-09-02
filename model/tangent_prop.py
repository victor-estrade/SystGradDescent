import torch
# import torch.nn as nn

import os
import numpy as np

from collections import OrderedDict
from .base import BaseModel
from .base import BaseNeuralNet
from .utils import to_torch
from .monitor import LightLossMonitorHook

# from hessian import hessian


class TangentPropClassifier(BaseModel, BaseNeuralNet):
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

    def get_losses(self):
        losses = dict(loss=self.loss_hook.losses)
        return losses
        
    def fit(self, generator):
        mu = torch.tensor(1.0, requires_grad=True)
        mu_prime = mu.detach()
        params = OrderedDict([('mu', mu)])
        params.update(generator.nuisance_params)

        for i in range(self.n_steps):
            s, w_s, b, w_b = generator.generate(self.sample_size)
            s_prime, b_prime = s.detach(), b.detach()
            self.optimizer.zero_grad()  # zero-out the gradients because they accumulate by default

            s_counts = self.forward(s, w_s)
            b_counts = self.forward(b, w_b)
            s_prime_counts = self.forward(s_prime, w_s.detach())
            b_prime_counts = self.forward(b_prime, w_b.detach())

            total_count = mu * s_counts + b_counts # should be mu s + b + epsilon
            asimov = mu_prime * s_prime_counts + b_prime_counts # should be mu s + b + epsilon
            loss = self.criterion(total_count, asimov, params)
            
            if np.isnan(loss.item()):
                print('NaN detected at ', i)
                print('output', total_count)
                print('loss', loss)
                break
            else:
                loss.backward(retain_graph=True)
                # loss.backward()
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
        y_proba = np.array(probas)
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
        self.loss_hook.save_state(path)
        return self

    def load(self, save_directory):
        super(BaseModel, self).load(save_directory)
        path = os.path.join(save_directory, 'weights.pth')
        if self.cuda_flag:
            self.net.load_state_dict(torch.load(path))
        else:
            self.net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

        path = os.path.join(save_directory, 'losses.json')
        self.loss_hook.load_state(path)
        return self


    def get_name(self):
        name = "{base_name}-{archi_name}-{optimizer_name}-{n_steps}-{sample_size}-{temperature}".format(**self.__dict__)
        return name


