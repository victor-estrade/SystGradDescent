# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

# Command line : 
# python -m benchmark.HIGGS.assert_torch_generator

import time
import numpy as np

from problem.higgs import Generator
from problem.higgs import GeneratorTorch
from problem.higgs import param_generator
from problem.higgs.higgs_geant import load_data

import argparse


def parse_args(main_description="assert GPU and CPU generator do the same computation"):
    parser = argparse.ArgumentParser(description=main_description)

    parser.add_argument("--verbose", "-v", type=int, choices=[0, 1, 2],
                        default=0, help="increase output verbosity")

    parser.add_argument('--no-cuda', '--no-gpu', help='flag to use or not the gpu',
                        action='store_false', dest='cuda')

    args = parser.parse_args()
    return args

def main():
    print("hello")
    args = parse_args()
    print("Loading data ...", end='', flush=True)
    data = load_data()
    print(".. data loaded", flush=True)
    feature_names = data.columns[:-2] if len(data.columns) == 31 else data.columns[:-3]
    dtypes = {name : "float32" for name in feature_names}
    dtypes.update({"Label": "int32", "Weight": "float32"})
    data = data.astype(dtypes)

    R_TOL = 1e-3  # Original data is precise up to 3 decimals anyway
    A_TOL = 1e-8
    N_SAMPLES = None
    param = param_generator()
    param = param_generator()
    # param = param.clone_with(1., 1., 1., 1.)
    print(param)

    print('Generating Ref cpu data')
    cpu_generator = Generator(data, seed=42)
    X_ref, y_ref, w_ref = cpu_generator.generate(*param, n_samples=N_SAMPLES)


    print('Generating gpu data')
    gpu_generator = GeneratorTorch(data, seed=42, cuda=args.cuda)
    X, y, w = gpu_generator.generate(*param, n_samples=N_SAMPLES)


    print(X_ref.shape, X.size(), len(feature_names))
    # print(np.allclose(X_ref, X.detach().cpu().numpy(), rtol=R_TOL, atol=A_TOL))

    for i, col in enumerate(data.columns):
        diff = cpu_generator.data[col] - gpu_generator.data_dict[col].cpu().numpy()
        diff = data[col] - gpu_generator.data_dict[col].cpu().numpy()
        pp = ( np.abs(diff) > (A_TOL + R_TOL * np.abs(data[col]) ) ).mean() * 100
        print(f"{pp:.3f} % [{diff.min()}, {diff.max()}]", col)

    print("--"*20)
    X_diff = X_ref - X.detach().cpu().numpy()
    for i, col in enumerate(feature_names):
        pp = ( np.abs(X_diff[:, i]) > (A_TOL + R_TOL * np.abs(X_ref[:, i]) ) ).mean() * 100
        print(f"{pp:.3f} % [{X_diff[:, i].min()}, {X_diff[:, i].max()}] {col}")

    w_diff = w_ref - w.cpu().numpy()
    pp = ( np.abs(w_diff) > (A_TOL + R_TOL * np.abs(w_ref) ) ).mean() * 100
    print(f"{pp:.3f} % [{w_diff.min()}, {w_diff.max()}] Weight")

    y_diff = y_ref - y.cpu().numpy()
    pp = ( np.abs(y_diff) > (A_TOL + R_TOL * np.abs(y_ref) ) ).mean() * 100
    print(f"{pp:.3f} % [{y_diff.min()}, {y_diff.max()}] Label")


    # ============== 2nd round ====================
    print('=='*8, "2nd round", "=="*8)
    param = param_generator()
    print(param)
    X_ref, y_ref, w_ref = cpu_generator.generate(*param, n_samples=N_SAMPLES)
    X, y, w = gpu_generator.generate(*param, n_samples=N_SAMPLES)

    print("--"*20)
    X_diff = X_ref - X.detach().cpu().numpy()
    for i, col in enumerate(feature_names):
        pp = ( np.abs(X_diff[:, i]) > (A_TOL + R_TOL * np.abs(X_ref[:, i]) ) ).mean() * 100
        print(f"{pp:.3f} % [{X_diff[:, i].min()}, {X_diff[:, i].max()}] {col}")

    w_diff = w_ref - w.cpu().numpy()
    pp = ( np.abs(w_diff) > (A_TOL + R_TOL * np.abs(w_ref) ) ).mean() * 100
    print(f"{pp:.3f} % [{w_diff.min()}, {w_diff.max()}] Weight")

    y_diff = y_ref - y.cpu().numpy()
    pp = ( np.abs(y_diff) > (A_TOL + R_TOL * np.abs(y_ref) ) ).mean() * 100
    print(f"{pp:.3f} % [{y_diff.min()}, {y_diff.max()}] Label")


    print("Done")


if __name__ == '__main__':
    main()
