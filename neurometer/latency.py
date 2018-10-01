from sys import stderr
import time

import numpy as np


class LatencyWatch(object):

    def __init__(self):
        self.measurements = []

    def __enter__(self):
        self.time_a = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.time_b = time.perf_counter()
        self.measurements.append(self.time_b - self.time_a)

    @property
    def mean(self):
        return np.mean(self.measurements)

    @property
    def std(self):
        return np.sqrt(np.var(self.measurements) / len(self.measurements))

    def write(self, file=stderr):
        mean = self.mean
        std = self.std
        n = len(self.measurements)
        print(f"Latency stats: {mean * 1000} +/- {std * 1000}ms (n={n})")


class GridSearch(object):

    def __init__(self, params):
        self.params = params
        self.param_lengths = [len(param) for param in self.params]
        self.indices = [1] * len(params)

    def _update(self, carry_idx):
        if carry_idx >= len(self.params):
            return True
        if self.indices[carry_idx] < self.param_lengths[carry_idx]:
            self.indices[carry_idx] += 1
            return False
        else:
            self.indices[carry_idx] = 1
            return False or self._update(carry_idx + 1)

    def __iter__(self):
        self.stop_next = False
        self.indices = [1] * len(self.params)
        return self

    def __len__(self):
        return np.prod(self.param_lengths)

    def __next__(self):
        if self.stop_next:
            raise StopIteration
        result = [param[idx - 1] for param, idx in zip(self.params, self.indices)]
        self.indices[0] += 1
        if self.indices[0] == self.param_lengths[0] + 1:
            self.indices[0] = 1
            self.stop_next = self._update(1)
        return result
