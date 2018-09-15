from sys import stderr
import time

import numpy as np


class LatencyWatch(object):

    def __init__(self):
        self.measurements = []

    def __enter__(self):
        self.time_a = time.time()
        return self

    def __exit__(self, *args):
        self.time_b = time.time()
        self.measurements.append(self.time_b - self.time_a)

    @property
    def mean(self):
        return np.mean(self.measurements)

    @property
    def std(self):
        return np.sqrt(np.std(self.measurements) / len(self.measurements))

    def write(self, file=stderr):
        mean = self.mean
        std = self.std
        n = len(self.measurements)
        print(f"Latency stats: {mean * 1000} +/- {std * 1000}ms (n={n})")
