from neurometer import LatencyWatch
from tqdm import tqdm
import fire
import torch
import torch.nn as nn


class Conv2dBenchmark(object):

    def run(self, features_in, features_out, height, width, kernel_size=3, padding=1, stride=1,
            cuda=True, n_trials=300, burn_in=100, cache_clear_size=10000000, main=True):
        ones = torch.ones(1, features_in, height, width)
        conv2d = nn.Conv2d(features_in, features_out, kernel_size, padding=padding, stride=stride)
        if cuda:
            ones = ones.cuda()
            conv2d.cuda()
        watch = LatencyWatch()
        for _ in tqdm(range(burn_in)):
            conv2d(ones)
            if cuda:
                torch.cuda.synchronize()
            ones = ones.detach()
        for _ in tqdm(range(n_trials)):
            with watch:
                conv2d(ones)
                if cuda:
                    torch.cuda.synchronize()
            ones = ones.detach()
            c = torch.zeros(cache_clear_size)
            if cuda:
                c.cuda()
        if main:
            watch.write()
        return watch.mean

    def run_sequential(self, begin_features=1, end_features=1000, step=5):
        for feat in tqdm(range(begin_features, end_features + 1, step), position=0):
            measure = self.run(feat, 64, 64, 64, main=False)
            print(f"{feat},{measure}")


if __name__ == "__main__":
    fire.Fire(Conv2dBenchmark)