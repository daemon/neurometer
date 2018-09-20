from neurometer import LatencyWatch
from tqdm import tqdm
import fire
import torch
import torch.nn as nn


def config_filename(height, width, kernel_size, padding, stride, cuda):
    str_ = f"{height}-{width}-{kernel_size}-{padding}-{stride}-{cuda}.dat"
    return str_


class Conv2dBenchmark(object):

    def run(self, features_in, features_out, height, width, kernel_size=3, padding=1, stride=1,
            cuda=False, n_trials=1000, burn_in=100, clear_cache_size=10000000, main=True):
        ones = torch.ones(1, features_in, height, width)
        conv2d = nn.Conv2d(features_in, features_out, kernel_size, padding=padding, stride=stride, bias=False)
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
            c = torch.zeros(clear_cache_size)
            if cuda:
                c.cuda()
        if main:
            watch.write()
        return watch.mean, watch.std

    def run_sequential(self, begin_features=1, end_features=1000, step=1):
        for feat in tqdm(range(begin_features, end_features + 1, step), position=0):
            measure, std = self.run(feat, 64, 64, 64, main=False)
            print(f"{feat},{measure},{std}")

    def build_table(self, begin_in=1, end_in=1, begin_out=1, end_out=20, height=28, width=28, kernel_size=5, cuda=False, padding=1, stride=1, step=10):
        with open(config_filename(height, width, kernel_size, padding, stride, cuda), "w") as f:
            for feat_in in tqdm(range(begin_in, end_in + 1 + step, step), position=0):
                for feat_out in tqdm(range(begin_out, end_out + 1 + step, step), position=1):
                    measure, std = self.run(feat_in, feat_out, height, width, kernel_size, padding, stride, main=False, cuda=cuda, clear_cache_size=0)
                    print(f"{feat_in},{feat_out},{measure}", file=f)


if __name__ == "__main__":
    fire.Fire(Conv2dBenchmark)