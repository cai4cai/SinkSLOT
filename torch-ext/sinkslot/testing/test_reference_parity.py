"""SROT and Spar-Sink ports in reference_solvers against the authors' code.

The authors' files are loaded from local clones (neither repository has a licence,
so they are not vendored); the tests skip when the variables are unset:

    git clone https://github.com/khainb/SROT                 # tested at 626a3c5
    git clone https://github.com/Mengyu8042/Spar-Sink        # tested at 7e5511e
    SROT_REPO=path/to/SROT SPARSINK_REPO=path/to/Spar-Sink \
        pytest torch-ext/sinkslot/testing/test_reference_parity.py

Both ports run in float64 on CPU here; the comparisons use the same projection
directions (SROT) and the same sampled kernel entries (Spar-Sink).
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from sinkslot.bench.reference_solvers import (
    _sparsink_sinkhorn, _srot_sinkhorn, build_sot_plan, build_sparse_kernel,
)
from sinkslot.solver import get_random_projections

DT = torch.float64


def _load(env, relpath, name):
    root = os.environ.get(env)
    if not root:
        pytest.skip(f"{env} is not set (path to a clone of the authors' repository)")
    path = Path(root) / relpath
    if not path.is_file():
        pytest.skip(f"{path} not found")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def srot():
    pytest.importorskip("ot")
    return _load("SROT_REPO", "lib/sinkhorn_methods.py", "srot_sinkhorn_methods")


@pytest.fixture(scope="module")
def sparsink():
    pytest.importorskip("sklearn")
    pytest.importorskip("scipy")
    return _load("SPARSINK_REPO", "code/all_funcs.py", "sparsink_all_funcs")


def _problem(n=60, m=50, d=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g, dtype=DT)
    y = torch.randn(m, d, generator=g, dtype=DT) * 0.8 + 0.5
    a = torch.rand(n, generator=g, dtype=DT) + 0.2
    b = torch.rand(m, generator=g, dtype=DT) + 0.2
    return x, y, a / a.sum(), b / b.sum(), torch.cdist(x, y) ** 2


class _FixedDirections:
    """Stands in for the numpy Generator their build_sot_plan draws directions from."""

    def __init__(self, thetas):
        self.thetas = thetas

    def standard_normal(self, shape):
        assert shape == self.thetas.shape
        return self.thetas.copy()


@pytest.mark.parametrize("delta", [0.0, 1e-8])
def test_srot_reference_plan_matches_authors(srot, delta):
    x, y, a, b, _ = _problem()
    L, seed = 16, 3
    ours = build_sot_plan(x, y, a, b, slices=L, delta=delta, seed=seed)
    thetas = get_random_projections(x.shape[1], L, seed).numpy()
    theirs = srot.build_sot_plan(x.numpy(), y.numpy(), a.numpy(), b.numpy(), L=L, delta=delta,
                                 rng=_FixedDirections(thetas))
    np.testing.assert_allclose(ours.numpy(), theirs, rtol=0, atol=1e-15)


def test_srot_sinkhorn_iterates_match_authors(srot):
    """Same starting scalings (u = v = 1, i.e. f = g = 0), so every iterate agrees."""
    x, y, a, b, C = _problem()
    eps, iters = 0.5, 300
    pi = build_sot_plan(x, y, a, b, slices=16, seed=0)
    f, g, *_ = _srot_sinkhorn(C, pi.log(), a.log(), b.log(), eps, iters)
    ours = (pi.log() + (f[:, None] + g[None, :] - C) / eps).exp().numpy()
    log = srot.sinkhorn_sot(a.numpy(), b.numpy(), C.numpy(), pi.numpy(), eps,
                            max_iter=iters, tol=-1.0, log_freq=iters)
    theirs = log[-1][2]
    np.testing.assert_allclose(ours, theirs, rtol=1e-10, atol=1e-16)


@pytest.mark.parametrize("method, sample_size, iters", [
    ("spar_sink", 2000, 2000), ("rand_sink", 2000, 2000),
    ("spar_sink", 150, 40),  # empty lines: potentials drift, and their 1e-50 floor on u, v would end the parity
])
def test_sparsink_kernel_and_plan_match_authors(sparsink, monkeypatch, method, sample_size, iters):
    """Their spar_sinkhorn on our sampled mask: same rescaled kernel, same dropped
    lines, and the same plan after the same number of iterations (their start
    v = 1/m differs from ours, v = 1, by a constant that cancels in the plan)."""
    x, y, a, b, C = _problem()
    eps = 1.0
    rows, cols, lv = build_sparse_kernel(C, a, b, eps, method=method, sample_size=sample_size, seed=1)
    mask = torch.zeros(C.shape, dtype=DT)
    mask[rows, cols] = 1.0

    K = (-C / eps).exp()
    assert bool((K > 0).all())  # their sampling probability is zero where K underflows
    monkeypatch.setattr(torch, "bernoulli", lambda prob: mask.clone())
    _, nnz, G = sparsink.spar_sinkhorn(a.numpy(), b.numpy(), C.numpy(), K.numpy(), sample_size,
                                       method=method.replace("_", "-"), plan=True,
                                       numItermax=iters, stopThr=0.0)
    assert nnz == len(rows)

    K_spar = np.zeros(C.shape)
    K_spar[rows.numpy(), cols.numpy()] = lv.exp().numpy()
    prob = (np.sqrt(np.outer(a.numpy(), b.numpy())) if method == "spar_sink" else np.ones(C.shape))
    prob = np.minimum(sample_size * prob / prob.sum(), 1.0)
    np.testing.assert_allclose(K_spar[mask.numpy() != 0], (K.numpy() / prob)[mask.numpy() != 0], rtol=1e-13)

    f, g, empty, *_ = _sparsink_sinkhorn(rows, cols, lv, a.log(), b.log(), eps, iters)
    ours = np.zeros(C.shape)
    ours[rows.numpy(), cols.numpy()] = (lv + (f[rows] + g[cols]) / eps).exp().numpy()
    kept_rows, kept_cols = G.sum(1) > 0, G.sum(0) > 0
    assert empty == int((~kept_rows).sum() + (~kept_cols).sum())
    if sample_size == 150:
        assert empty > 0
    np.testing.assert_allclose(ours, G, rtol=1e-8, atol=1e-14)
