# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

# Command line : 
# python -m benchmark.HIGGS.speedy

import argparse
import time
import numpy as np

from problem.higgs import Generator
from problem.higgs import GeneratorTorch
from problem.higgs import param_generator
from problem.higgs.higgs_geant import load_data

def parse_args(main_description="assert GPU and CPU generator do the same computation"):
    parser = argparse.ArgumentParser(description=main_description)

    parser.add_argument("--verbose", "-v", type=int, choices=[0, 1, 2],
                        default=0, help="increase output verbosity")

    parser.add_argument('--no-cuda', '--no-gpu', help='flag to use or not the gpu',
                        action='store_false', dest='cuda')

    args, others = parser.parse_known_args()
    return args

def time_to_str(t, std):
    return f"{t:2.5f} +/- {std:.5f} sec"

def measure_time(func, repeat=3):
    iter_times = []
    for i in range(repeat):
        start_time = time.time()
        func()
        end_time = time.time()
        exec_time = end_time - start_time
        iter_times.append(exec_time)
    total_time = sum(iter_times)
    mean_time = total_time / len(iter_times)
    std_time = np.std(iter_times)
    return mean_time, std_time

def main():
    print("hello")
    args = parse_args()
    print("Loading data ...", end='', flush=True)
    data = load_data()
    print(".. data loaded", flush=True)

    N_SAMPLES = 100_000
    # N_SAMPLES = None


    cpu_generator = Generator(data, seed=42)
    def get_cpu_func(n_samples):
        def func():
            param = param_generator()
            X, y, w = cpu_generator.generate(*param, n_samples=n_samples)
        return func

    mean_time = measure_time(get_cpu_func(N_SAMPLES))
    print(f"{time_to_str(mean_time)} for CPU")


    gpu_generator = GeneratorTorch(data, seed=42, cuda=args.cuda)
    def get_gpu_func(n_samples):
        def func():
            param = param_generator()
            X, y, w = gpu_generator.generate(*param, n_samples=n_samples)
            X = X.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            w = w.detach().cpu().numpy()
        return func

    mean_time = measure_time(get_gpu_func(N_SAMPLES))
    print(f"{time_to_str(mean_time)} for GPU")

    N_LIST = [1000, 5000, 10_000, 50_000, 100_000, 200_000]
    cpu_times = [f"{n_samples:6d}" + time_to_str(measure_time(get_cpu_func(n_samples))) for n_samples in N_LIST]
    gpu_times = [f"{n_samples:6d}" + time_to_str(measure_time(get_gpu_func(n_samples))) for n_samples in N_LIST]
    print("CPU")
    print('\n'.join(cpu_times))
    print("="*10)
    print("GPU")
    print('\n'.join(gpu_times))

    print("Done")


if __name__ == '__main__':
    main()
