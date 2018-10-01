import gc
import random

from easydict import EasyDict as edict
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from tqdm import tqdm
import pandas as pd
import fire
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

from neurometer import LatencyWatch, GridSearch


class LeNet5(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.convs = [nn.Conv2d(1, config.conv1_out, 5),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(config.conv1_out, config.conv2_out, 5),
            nn.ReLU(), nn.MaxPool2d(2)]
        self._convs = nn.Sequential(*self.convs)
        self.fcs = [nn.Linear(config.conv2_out * 16, config.lin1_out), nn.ReLU(),
            nn.Linear(config.lin1_out, 10)]
        self._fcs = nn.Sequential(*self.fcs)
        self.watch = LatencyWatch()

    def dummy_input(self):
        return torch.zeros(1, 1, 28, 28)

    def forward(self, x):
        with self.watch:
            for conv in self.convs:
                x = conv(x)
            x = x.view(x.size(0), -1)
            for fc in self.fcs:
                x = fc(x)
        return x


class LeNet5Conv1(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.convs = [nn.Conv2d(1, config.conv1_out, 5),
                 nn.ReLU(), nn.MaxPool2d(2)]
        self._convs = nn.Sequential(*self.convs)
        self.watch = LatencyWatch()

    def dummy_input(self):
        return torch.zeros(1, 1, 28, 28)

    def forward(self, x):
        with self.watch:
            for conv in self.convs:
                x = conv(x)
        return x


class LeNet5Conv2(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.convs = [nn.Conv2d(config.conv1_out, config.conv2_out, 5),
            nn.ReLU(), nn.MaxPool2d(2)]
        self._convs = nn.Sequential(*self.convs)
        self.watch = LatencyWatch()
        self.conv1_out = config.conv1_out
        self.conv2_out = config.conv2_out

    def dummy_input(self):
        return torch.zeros(1, self.conv1_out, 12, 12)

    def forward(self, x):
        with self.watch:
            for conv in self.convs:
                x = conv(x)
        return x


class LeNet5Fc1(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.fcs = [nn.Linear(config.conv2_out * 16, config.lin1_out), nn.ReLU(),
            nn.Linear(config.lin1_out, 10)]
        self._fcs = nn.Sequential(*self.fcs)
        self.lin1_out = config.lin1_out
        self.conv2_out = config.conv2_out
        self.watch = LatencyWatch()

    def dummy_input(self):
        return torch.zeros(1, self.conv2_out * 16)

    def forward(self, x):
        with self.watch:
            for fc in self.fcs:
                x = fc(x)
        return x


class MeasureComponentBenchmark(object):

    def run(self, component_name, cuda=False, n_trials=100, burn_in=10, clear_cache=True, main=True, input_size=tuple(), **component_kwargs):
        torch.set_grad_enabled(False)
        model = components[component_name](edict(component_kwargs))
        model.eval()
        x = model.dummy_input()
        if cuda:
            x = x.cuda()
            model.cuda()
        for _ in tqdm(range(burn_in)):
            model(x)
            if cuda:
                torch.cuda.synchronize()
        model.watch.measurements = []
        for _ in tqdm(range(n_trials), position=0):
            model(x)
            if cuda:
                torch.cuda.synchronize()
            if clear_cache:
                x = model.dummy_input()
                if cuda:
                    x = x.cuda()
        if main:
            model.watch.write()
        else:
            return model.watch.measurements

    def build_table(self, component_name, method="random", cuda=False, ranges={}, n_samples=1000, n_trials=20, input_size=tuple(), 
            seed=0, output_file="output.csv", **component_kwargs):
        if method == "random":
            rand = random.Random(seed)
        elif method == "grid":
            grid_keys = list(ranges.keys())
            grid_iter = GridSearch([list(range(*range_args)) for range_args in ranges.values()])
        frames = []
        gc.disable()
        if method == "random":
            for idx in tqdm(range(n_samples), position=1):
                sample = {}
                cols = {}
                for key, range_ in ranges.items():
                    sample[key] = rand.randint(*range_)
                    cols[key] = [sample[key]] * n_trials
                cols["measurements"] = self.run(component_name, cuda=cuda, n_trials=n_trials + 20, main=False, input_size=input_size, **sample)[20:]
                frames.append(pd.DataFrame(cols))
                if idx % 100 == 0:
                    gc.collect()
        elif method == "grid":
            pbar = tqdm(total=len(grid_iter), position=1)
            for idx, args in enumerate(grid_iter):
                comp_args = {k: v for k, v in zip(grid_keys, args)}
                cols = comp_args.copy()
                cols["measurements"] = self.run(component_name, cuda=cuda, n_trials=n_trials + 20, main=False, input_size=input_size, **comp_args)[20:]
                frames.append(pd.DataFrame(cols))
                if idx % 100 == 0:
                    gc.collect()
                pbar.update(1)
            pbar.close()
        pd.concat(frames).to_csv(output_file, index_label="idx")

    def plot_scatter(self, filename):
        df = pd.read_csv(filename)
        sns.violinplot(x=df["conv1_out"], y=df["measurements"])
        plt.show()

    def plot3d(self, *filenames, title="", legend_names=[]):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        colors = ["red", "blue", "green", "orange", "purple", "black"]
        for idx, filename in enumerate(filenames):
            df = pd.read_csv(filename)
            df50 = df.groupby(["conv1_out", "conv2_out"]).quantile(0.75).reset_index()
            x, y = df50["conv1_out"], df50["conv2_out"]
            ax.scatter(x, y, df50["measurements"], color=colors[idx % len(colors)])
        if title:
            plt.title(title)
        if legend_names:
            legend_elements = []
            for idx, name in enumerate(legend_names):
                legend_elements.append(Line2D([0], [0], color=colors[idx % len(colors)], lw=4, label=name))
            ax.legend(handles=legend_elements)
        plt.show()


components = dict(lenet5_conv1=LeNet5Conv1, lenet5_conv2=LeNet5Conv2, lenet5_fc1=LeNet5Fc1, lenet5=LeNet5)

if __name__ == "__main__":
    fire.Fire(MeasureComponentBenchmark)
