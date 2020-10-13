# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

# Command line : 
# python -m benchmark.HIGGS.speedy

import time

from problem.higgs import Generator
from problem.higgs import GeneratorTorch
from problem.higgs import param_generator
from problem.higgs.higgs_geant import load_data


def time_to_str(t):
    return f"{t: 2.5f} sec"

def measure_time(func, repeat=3):
    start_time = time.time()
    for i in range(repeat):
        func()
    end_time = time.time()
    total_time = end_time - start_time
    mean_time = total_time / repeat
    return mean_time

def main():
    print("hello")
    print("Loading data ...", end='', flush=True)
    data = load_data()
    print(".. data loaded", flush=True)

    N_SAMPLES = 100_000
    # N_SAMPLES = None


    generator = Generator(data, seed=42)
    def cpu_generator():
        param = param_generator()
        X, y, w = generator.generate(*param, n_samples=N_SAMPLES)

    # mean_time = measure_time(cpu_generator)
    # print(time_to_str(mean_time))


    generator = GeneratorTorch(data, seed=42)
    def gpu_generator():
        param = param_generator()
        X, y, w = generator.generate(*param, n_samples=N_SAMPLES)

    mean_time = measure_time(gpu_generator)
    print(time_to_str(mean_time))


    print("Done")


if __name__ == '__main__':
    main()
