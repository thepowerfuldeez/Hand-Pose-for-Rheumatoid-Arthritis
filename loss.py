import numpy as np
import torch as t
import torch.nn as nn
import time


class Modified_SmoothL1Loss(nn.Module):

    def __init__(self):
        super(Modified_SmoothL1Loss, self).__init__()

    def forward(self, x, y):
        total_loss = 0
        assert x.shape == y.shape
        z = (x - y).float()
        mse = (t.abs(z) < 0.01).float() * z
        l1 = (t.abs(z) >= 0.01).float() * z
        total_loss += self._calculate_MSE(mse).sum()
        total_loss += self._calculate_L1(l1).sum()

        return total_loss / z.shape[0]

    def _calculate_MSE(self, z):
        return 0.5 * (t.pow(z, 2))

    def _calculate_L1(self, z):
        return 0.01 * (t.abs(z) - 0.005)
