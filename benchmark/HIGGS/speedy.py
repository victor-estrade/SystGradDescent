# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

# Command line : 
# python -m benchmark.HIGGS.speedy

import argparse
import time

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
    args = parse_args()
    print("Loading data ...", end='', flush=True)
    data = load_data()
    print(".. data loaded", flush=True)

    N_SAMPLES = 100_000
    # N_SAMPLES = None


    generator = Generator(data, seed=42)
    def cpu_generator():
        param = param_generator()
        X, y, w = generator.generate(*param, n_samples=N_SAMPLES)

    mean_time = measure_time(cpu_generator)
    print(f"{time_to_str(mean_time)} for CPU")


    generator = GeneratorTorch(data, seed=42, cuda=args.cuda)
    def gpu_generator():
        param = param_generator()
        X, y, w = generator.generate(*param, n_samples=N_SAMPLES)
        X = X.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        w = w.detach().cpu().numpy()

    mean_time = measure_time(gpu_generator)
    print(f"{time_to_str(mean_time)} for GPU")


    print("Done")


if __name__ == '__main__':
    main()
