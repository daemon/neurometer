import gc
import json
import os
import time

from easydict import EasyDict as edict
from neurometer import LatencyWatch
from neurometer.model import capture_config
from tqdm import tqdm
import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class LeNet5(nn.Module):

    def __init__(self, config):
        super().__init__()
        convs = [nn.Conv2d(1, config.conv1.out, 5),
                 nn.ReLU(), nn.MaxPool2d(2),
                 nn.Conv2d(config.conv1.out, config.conv2.out, 5),
                 nn.ReLU(), nn.MaxPool2d(2)]
        self.convs = nn.Sequential(*convs)
        fcs = [nn.Linear(config.conv2.out * 16, config.lin1.out), nn.ReLU(),
               nn.Linear(config.lin1.out, 10)]
        self.fcs = nn.Sequential(*fcs)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        return self.fcs(x)


def make_config(conv1, conv2, lin1):
    return edict(dict(conv1=conv1, conv2=conv2, lin1=lin1))


class LeNet5Benchmark(object):

    def run(self, cuda=False, n_trials=10, burn_in=100, clear_cache=False, main=True, write_measurements=False):
        x = torch.zeros(1, 1, 28, 28)
        conv1 = dict(out=20)
        conv2 = dict(out=50)
        lin1 = dict(out=500)
        model = LeNet5(make_config(conv1, conv2, lin1))
        model.eval()
        print(json.dumps(capture_config(model)))
        if cuda:
            x = x.cuda()
            model.cuda()
        watch = LatencyWatch()
        for _ in tqdm(range(burn_in)):
            model(x)
            if cuda:
                torch.cuda.synchronize()
            x = x.detach()
        while len(watch.measurements) < 1000 or watch.std / (np.sqrt(len(watch.measurements)) * watch.mean) > 0.01:
            gc.disable()
            with watch:
                # model.forward(x)
                for _ in range(500000):
                    pass
        if main:
            watch.write()
        df = pd.DataFrame(data=dict(measurements=watch.measurements))
        np.save("measurements.npy", df)
        df.hist(bins=500)
        plt.show()
        return watch.mean, watch.std

    def run_sequential(self, begin_features=1, end_features=1000, step=1):
        for feat in tqdm(range(begin_features, end_features + 1, step), position=0):
            measure, std = self.run(feat, 64, 64, 64, main=False)
            print(f"{feat},{measure},{std}")

    def build_table(self, begin_in=1, end_in=1, begin_out=1, end_out=20, height=28, width=28, kernel_size=5, cuda=False, padding=1, stride=1, step=50):
        with open(config_filename(height, width, kernel_size, padding, stride, cuda), "w") as f:
            print("in,out,latency", file=f)
            for feat_in in tqdm(range(begin_in, end_in + 1 + step, step), position=0):
                for feat_out in tqdm(range(begin_out, end_out + 1 + step, step), position=1):
                    measure, std = self.run(feat_in, feat_out, height, width, kernel_size, padding, stride, main=False, cuda=cuda, clear_cache=0)
                    print(f"{feat_in},{feat_out},{measure}", file=f)


if __name__ == "__main__":
    fire.Fire(LeNet5Benchmark)