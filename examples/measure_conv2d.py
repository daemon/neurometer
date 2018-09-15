from neurometer import LatencyWatch
from tqdm import tqdm
import fire
import torch
import torch.nn as nn


class Conv2dBenchmark(object):

    def run(self, features_in, features_out, height, width, step=5, kernel_size=3, padding=1, stride=1,
            cuda=True, n_trials=300):
        ones = torch.ones(1, features_in, height, width)
        conv2d = nn.Conv2d(features_in, features_out, kernel_size, padding=padding, stride=stride)
        if cuda:
            ones = ones.cuda()
            conv2d.cuda()
        watch = LatencyWatch()
        for _ in tqdm(range(n_trials)):
            with watch:
                conv2d(ones)
                if cuda:
                    torch.cuda.synchronize()
            ones = ones.detach()
        watch.write()


if __name__ == "__main__":
    fire.Fire(Conv2dBenchmark)