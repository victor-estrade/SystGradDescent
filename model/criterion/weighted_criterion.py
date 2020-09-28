#!/usr/bin/env python
# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

from hessian.gradient import jacobian

__doc__ = """
Sample wise weighted criterions.
"""
__version__ = "0.1"
__author__ = "Victor Estrade"



class WeightedCrossEntropyLoss(nn.Module):
    def forward(self, input, target, weight):
        element_loss = F.cross_entropy(input, target, reduction='none')
        loss = torch.mean(element_loss * weight)
        return loss


class WeightedBinaryCrossEntropyLoss(nn.Module):
    def forward(self, input, target, weight):
        input = input.view(target.size())
        log_sigm = F.logsigmoid(input) * target + (1 - target) * (-input - F.softplus(-input))
        element_loss = -log_sigm * weight
        loss = torch.mean(element_loss)
        return loss


class WeightedMSELoss(nn.Module):
    def forward(self, input, target, weight):
        loss = (input - target)**2
        element_loss = loss * weight
        loss = torch.mean(element_loss)
        return loss


class WeightedGaussEntropyLoss(nn.Module):
    def forward(self, input, target, weight):
        prediction, logsigma = torch.split(input, 1, dim=1)
        error = (prediction - target)
        error_sigma = error / torch.exp(logsigma)
        loss = logsigma + 0.5 * (error_sigma * error_sigma)
        element_loss = loss * weight
        loss = torch.mean(element_loss)
        return loss


class WeightedL2Loss(nn.Module):
    def forward(self, input, weight):
        loss = torch.sum( input * input, 1) * weight / input.size(1)
        loss = torch.mean(loss)
        return loss


class WeightedL1Loss(nn.Module):
    def forward(self, input, weight):
        loss = torch.sum( torch.abs(input), 1) * weight / input.size(1)
        loss = torch.mean(loss)
        return loss


class WeightedTPLoss(nn.Module):
    def forward(self, logits, weight, nuisance_params):
        # weight = torch.sqrt(weight)
        w_sum = torch.sum(weight)
        # jac = jacobian(torch.sum(logits * weight / w_sum, 0), nuisance_params.values())
        jac = jacobian(logits, nuisance_params.values())
        loss = torch.sum( jac * jac, 1) / jac.size(1)
        loss = torch.sum(loss * weight) / w_sum
        loss = torch.mean(loss)
        # print(loss)
        return loss
