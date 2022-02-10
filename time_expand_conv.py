import os
import math
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class TxpCnn(nn.Module):
    def __init__(self, input_seq, output_seq):
        super(TxpCnn, self).__init__()

        self.tpcnns = nn.ModuleList()
        self.prelus = nn.ModuleList()

        self.tpcnns.append(nn.Conv2d(input_seq, (output_seq + input_seq) // 2, 3, padding=1))
        self.prelus.append(nn.PReLU())

        self.tpcnns.append(nn.Conv2d((output_seq + input_seq) // 2, (output_seq + input_seq) // 2,
                                     3, padding=1))
        self.prelus.append(nn.PReLU())

        self.tpcnns.append(nn.Conv2d((output_seq + input_seq) // 2, output_seq,
                                     3, padding=1))
        self.prelus.append(nn.PReLU())

        self.tpcnn_ouput = nn.Conv2d(output_seq, output_seq, 3, padding=1)

    def forward(self, v):
        # input size: [batch, channel, h, w] = [batch, seq, nodes, features]
        v = self.prelus[0](self.tpcnns[0](v))
        v = self.prelus[1](self.tpcnns[1](v))
        v = self.prelus[2](self.tpcnns[2](v))
        v = self.tpcnn_ouput(v)
        # v = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])

        return v


if __name__ == '__main__':
    x = torch.rand(1, 8, 4, 100)
    tpcnn = TxpCnn(x.shape[1], 12)
    y = tpcnn(x)
    print(y.shape)
