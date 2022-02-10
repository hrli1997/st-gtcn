import torch
from torch import nn


class Discriminator_net(nn.Module):

    def __init__(self, GRU_input_size=2, GRU_hidden_size=40, GRU_num_layers=1):
        super(Discriminator_net, self).__init__()
        self.GRU = nn.GRU(input_size=GRU_input_size, hidden_size=GRU_hidden_size,
                           num_layers=GRU_num_layers)
        self.FC = nn.Linear(GRU_hidden_size*GRU_num_layers, 1)
        self.sigm = nn.Sigmoid()

    def forward(self, GRU_input, GRU_h0):
        """
        :param GRU_input:[seq, batch, fratures] such as [12, n, 2]
        :param GRU_h0: [num_layers, batch, hiddensize], such as [1, n, 40]
        :return:[batch, True/False], such as [n, 2]
        """
        GRU_output, h_n = self.GRU(GRU_input, GRU_h0)
        # print('h_n', h_n.shape)
        h_n = h_n.permute(1, 0, 2)
        h_n = h_n.view(GRU_input.shape[1], -1)
        # print("h_n", h_n.shape)
        FC_out = self.FC(h_n)
        net_out = self.sigm(FC_out)
        return net_out
