"""Every fused backend against the pure-torch path on the same device.

Run once per fused backend (Triton on CUDA, Metal on MPS) through the `fused`
fixture, skipped where the backend can't run. Same device on both sides on
purpose: the sliced-plan builder is plain torch and device-dependent in its
last bits, so comparing against CPU would mix support differences into what
is meant to be a kernel comparison.
"""

import pytest
import torch

from sinkslot import seg_lse_online, sinkslot_solve, sparse_sqeuclidean_cost
from sinkslot._backends import resolve_backend
from sinkslot.sinkhorn_solvers import _seg_lse_coo


def _problem(n=300, m=250, d=3, seed=0, device="cpu"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    y = torch.randn(m, d, generator=g) + 1.0
    a = torch.rand(n, generator=g) + 0.1
    b = torch.rand(m, generator=g) + 0.1
    return (t.to(device) for t in (x, y, a / a.sum(), b / b.sum()))


def test_auto_picks_the_fused_backend(fused):
    device, backend = fused
    assert resolve_backend("auto", device) == backend


@pytest.mark.parametrize("m", [200, 40_000], ids=["int16-idx", "int32-idx"])
def test_seg_lse_online_matches_coo_reference(fused, m):
    """Rows longer than a threadgroup, a row shorter than a simdgroup, an empty
    row (-inf), both index widths `to_csr` emits, with and without `base`."""
    device, _ = fused
    g = torch.Generator(device="cpu").manual_seed(0)
    lengths = torch.tensor([0, 1, 7, 31, 32, 33, 64, 65, 300, 2000, 5])
    n = lengths.numel()
    rows = torch.repeat_interleave(torch.arange(n), lengths)
    cols = torch.randint(0, m, (rows.numel(),), generator=g)
    lam = 5 * torch.randn(rows.numel(), generator=g)
    phi, psi, base = torch.randn(n, generator=g), torch.randn(m, generator=g), torch.randn(n, generator=g)
    want = _seg_lse_coo(lam + phi[rows] + psi[cols], rows, n)

    indptr = torch.zeros(n + 1, dtype=torch.int32)
    indptr[1:] = lengths.cumsum(0)
    idx = cols.to(torch.int16 if m < 2**15 else torch.int32)
    on = lambda t: t.to(device)  # noqa: E731
    got = seg_lse_online(on(indptr), on(idx), on(lam), on(phi), on(psi), n).cpu()
    sub = seg_lse_online(on(indptr), on(idx), on(lam), on(phi), on(psi), n, base=on(base)).cpu()

    assert got[0] == float("-inf") and sub[0] == float("inf")
    torch.testing.assert_close(got[1:], want[1:], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(sub[1:], (base - want)[1:], rtol=1e-5, atol=1e-5)


def test_sparse_sqeuclidean_cost_matches_torch(fused):
    device, backend = fused
    x, y, *_ = _problem(device=device, d=5)
    g = torch.Generator(device="cpu").manual_seed(1)
    rows = torch.randint(0, x.shape[0], (5000,), generator=g).to(device)
    cols = torch.randint(0, y.shape[0], (5000,), generator=g).to(device)
    got = sparse_sqeuclidean_cost(x, y, rows, cols, backend=backend)
    want = sparse_sqeuclidean_cost(x, y, rows, cols, backend="torch")
    assert got.dtype == torch.float32
    torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("variant", ["alternating", "symmetric"])
@pytest.mark.parametrize("stop_mode", ["fixed", "marginal", "potential"])
def test_sinkslot_solve_fused_matches_torch(fused, variant, stop_mode):
    """The acceptance criterion: both schemes, all three stop modes."""
    device, backend = fused
    x, y, a, b = _problem(device=device)
    kw = dict(eps=0.05, L=40, seed=0, n_iters=300, stop_mode=stop_mode,
              stop_max_iter=20000, stop_tol=1e-6 if stop_mode == "marginal" else 1e-5,
              stop_check_every=5, variant=variant)
    got = sinkslot_solve(x, y, a, b, backend=backend, **kw)
    want = sinkslot_solve(x, y, a, b, backend="torch", **kw)

    assert torch.equal(got.rows, want.rows) and torch.equal(got.cols, want.cols)
    if stop_mode != "fixed":
        assert got.converged and want.converged
        # Float32 rounding can move the stopping check by an interval or two.
        assert abs(got.iters_run - want.iters_run) <= 3 * kw["stop_check_every"]
    dphi = float((got.phi - want.phi).abs().max())
    dpsi = float((got.psi - want.psi).abs().max())
    assert dphi < 1e-3 and dpsi < 1e-3, f"{backend} vs torch: {dphi:.2e}, {dpsi:.2e}"
