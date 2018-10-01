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


def capture_config(model, is_cuda=False):
    captures = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            config_dict = dict(in_channels=m.in_channels, out_channels=m.out_channels, kernel_size=m.kernel_size[0], 
                padding=m.padding[0], stride=m.stride[0], is_cuda=is_cuda)
            captures.append(config_dict)
        elif isinstance(m, nn.Linear):
            config_dict = dict(in_features=m.in_features, out_features=m.out_features, is_cuda=is_cuda)
            captures.append(config_dict)
    return captures
