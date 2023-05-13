import torch
import torch.nn as nn

from math import pi as PI

import foldingdiff_pytorch.util as util


class WrappedSmoothL1Loss(nn.Module):
    def __init__(self, beta=0.1 * PI):
        super().__init__()
        self.beta = beta

    def forward(self, input, target):
        d = util.wrap(target - input)
        cond = d.abs() < self.beta

        return torch.where(cond, 0.5 * d.square() / self.beta, d.abs() - 0.5 * self.beta).mean()
