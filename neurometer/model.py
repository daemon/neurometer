import torch
import torch.nn as nn


class IndexingLayer(nn.Module):

    def __init__(self, salient_channels, final_size):
        super().__init__()
        self.register_buffer("salient_channels", torch.LongTensor(salient_channels))
        self.final_size = final_size

    def forward(self, x):
        template = torch.zeros(x.size(0), self.final_size, *x.size()[2:]).to(x.device)
        template[:, self.salient_channels] = x
        return template