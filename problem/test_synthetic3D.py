# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

from synthetic3D import Synthetic3D
from synthetic3D import Synthetic3DGenerator


def test_generator_reproducibility():
    r = 0
    lam = 3
    mu = 50/1050
    seed = 42
    gen = Synthetic3DGenerator(seed=seed)
    D1 = gen.generate(r, lam, mu)
    gen.reset()
    D2 = gen.generate(r, lam, mu)
    assert np.alltrue(D1.values==D2.values)


def test_train_reproducibility():
    r = 0
    lam = 3
    mu = 50/1050
    seed = 42
    gen = Synthetic3D(seed=seed)
    D1 = gen.train_sample(r, lam, mu)
    gen = Synthetic3D(seed=seed)
    D2 = gen.train_sample(r, lam, mu)
    assert np.alltrue(D1.values==D2.values)
    

def test_test_reproducibility():
    r = 0
    lam = 3
    mu = 50/1050
    seed = 42
    gen = Synthetic3D(seed=seed)
    D1 = gen.test_sample(r, lam, mu)
    gen = Synthetic3D(seed=seed)
    D2 = gen.test_sample(r, lam, mu)
    assert np.alltrue(D1.values==D2.values)
    