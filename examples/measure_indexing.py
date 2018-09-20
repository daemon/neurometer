import random

from neurometer import LatencyWatch
from neurometer.model import IndexingLayer
from tqdm import tqdm
import torch
import fire


class IndexingBenchmark(object):

    def run(self, features, height, width, burn_in=100, cuda=False, clear_cache_size=10000000, 
            main=True, zero_prob=0.5, n_trials=1000):
        indices_ = list(range(features))
        indices = []
        for idx in indices_:
            if random.random() > zero_prob:
                indices.append(idx)
        layer = IndexingLayer(indices, features)
        if cuda:
            layer.cuda()
        x = torch.ones(1, len(indices), height, width)
        if cuda:
            x = x.cuda()
        for _ in tqdm(range(burn_in)):
            layer(x)
        watch = LatencyWatch()
        for _ in tqdm(range(n_trials)):
            with watch:
                layer(x)
                if cuda:
                    torch.cuda.synchronize()
            x = x.detach()
            c = torch.zeros(clear_cache_size)
            if cuda:
                c.cuda()
        if main:
            watch.write()
        return watch.mean, watch.std


if __name__ == "__main__":
    fire.Fire(IndexingBenchmark)