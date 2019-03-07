#!/usr/bin/env python
# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


__doc__ = """
TODO : Add code
"""
__version__ = "0.1"
__author__ = "Victor Estrade"


def to_one_hot(y, n_class=2):
    oh = np.zeros((y.shape[0], n_class), np.float32)
    oh[np.arange(y.shape[0]), y] = 1
    return oh


class WeightedCrossEntropyLoss(nn.Module):
    def forward(self, input, target, weight):
        element_loss = F.cross_entropy(input, target, reduce=False)
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
