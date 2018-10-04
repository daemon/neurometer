from __future__ import print_function

from neurometer import LatencyWatch
from tqdm import tqdm
import fire
import torch
import torch.nn as nn


class LinearBenchmark(object):

    def run(self, features_in, features_out, cuda=False, n_trials=300, burn_in=100, 
            clear_cache_size=10000000, main=True, bias=True, write_measurements=False):
        ones = torch.ones(1, features_in)
        linear = nn.Linear(features_in, features_out, bias=bias)
        if cuda:
            ones = ones.cuda()
            linear.cuda()
        watch = LatencyWatch()
        for _ in tqdm(range(burn_in)):
            linear(ones)
            if cuda:
                torch.cuda.synchronize()
            ones = ones.detach()
        for _ in tqdm(range(n_trials)):
            with watch:
                linear(ones)
                if cuda:
                    torch.cuda.synchronize()
            ones = ones.detach()
            c = torch.zeros(clear_cache_size)
            if cuda:
                c.cuda()
        if main and write_measurements:
            for measurement in watch.measurements:
                #print(f"{features_in},{features_out},{measurement}")
                print(','.join([str(features_in), str(features_out), str(measurements)]))
            else:
                watch.write()
        return watch.mean

    def run_sequential(self, begin_features=1, end_features=1000, step=5):
        pbar = tqdm(range(begin_features, end_features + 1, step), position=0)
        for feat in pbar:
            measure = self.run(feat, 1000, main=False)
            #print(f"{feat},{measure}")
            print(','.join([str(feat),str(measure)]))
            pbar.set_postfix(dict(measure=measure * 1000))

    def build_table(self, begin_in=1, end_in=500, begin_out=10, end_out=10):
        for feat_in in tqdm(range(begin_in, end_in + 1), position=0):
            for feat_out in tqdm(range(begin_out, end_out + 1), position=1):
                measure = self.run(feat_in, feat_out, main=False, clear_cache_size=0)
                #print(f"{feat_in},{feat_out},{measure}")
                print(','.join([str(feat_in), str(feat_out), str(measure)]))


if __name__ == "__main__":
    #fire.Fire(LinearBenchmark)
    pass

