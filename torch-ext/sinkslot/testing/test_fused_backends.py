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


def _csr(lengths, m, seed=0):
    """A random CSR problem with the given row lengths, as CPU tensors."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = len(lengths)
    lengths = torch.tensor(lengths)
    rows = torch.repeat_interleave(torch.arange(n), lengths)
    cols = torch.randint(0, m, (rows.numel(),), generator=g)
    indptr = torch.zeros(n + 1, dtype=torch.int32)
    indptr[1:] = lengths.cumsum(0)
    lam = torch.randn(rows.numel(), generator=g)
    return indptr, rows, cols, lam, torch.randn(n, generator=g), torch.randn(m, generator=g)


def test_seg_lse_online_matches_torch_on_nan_rows(fused):
    """A row containing a NaN gets the torch path's answer (-inf), not a finite
    LSE over the remaining entries, which would let a diverged solve look fine."""
    device, _ = fused
    indptr, rows, cols, lam, phi, psi = _csr([5, 300, 40], m=50)
    lam[2] = float("nan")      # row 0, short row
    lam[5 + 123] = float("nan")  # row 1, spans several simdgroups
    want = _seg_lse_coo(lam + phi[rows] + psi[cols], rows, 3)
    on = lambda t: t.to(device)  # noqa: E731
    got = seg_lse_online(on(indptr), on(cols.to(torch.int16)), on(lam), on(phi), on(psi), 3).cpu()
    assert (want[:2] == float("-inf")).all(), want
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_seg_lse_online_leaves_the_tail_of_a_longer_out_alone(fused):
    """Only rows 0..n-1 are computed, even when the reused `out` is longer."""
    device, _ = fused
    indptr, rows, cols, lam, phi, psi = _csr([3, 7, 2], m=20)
    want = _seg_lse_coo(lam + phi[rows] + psi[cols], rows, 3)
    out = torch.full((8,), 7.0, device=device)
    on = lambda t: t.to(device)  # noqa: E731
    seg_lse_online(on(indptr), on(cols.to(torch.int16)), on(lam), on(phi), on(psi), 3, out=out)
    torch.testing.assert_close(out[:3].cpu(), want, rtol=1e-5, atol=1e-5)
    assert (out[3:] == 7.0).all()


def test_metal_seg_lse_rejects_dtypes_it_would_misread(fused):
    """The shader reads raw int32/float32 buffers, so int64 indptr or float16
    values must raise rather than be reinterpreted."""
    device, backend = fused
    if backend != "metal":
        pytest.skip("Triton's loads are dtype-generic")
    indptr, rows, cols, lam, phi, psi = _csr([3, 4], m=10)
    on = lambda t: t.to(device)  # noqa: E731
    idx = on(cols.to(torch.int16))
    with pytest.raises(ValueError, match="indptr must be int32"):
        seg_lse_online(on(indptr.long()), idx, on(lam), on(phi), on(psi), 2)
    with pytest.raises(ValueError, match="float32-only"):
        seg_lse_online(on(indptr), idx, on(lam).half(), on(phi), on(psi), 2)


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
    # The potential-mode tolerance has to sit above float32 resolution: the check
    # is eps * max|dphi| < tol (alpha * tol for symmetric), and with |phi| ~ 300
    # one ulp is already ~3e-6 of that. At 1e-5 the stop point is decided by
    # rounding noise and Metal and torch drift tens of iterations apart; at 1e-4
    # they stop on the same iteration.
    kw = dict(eps=0.05, L=40, seed=0, n_iters=300, stop_mode=stop_mode,
              stop_max_iter=20000, stop_tol=1e-6 if stop_mode == "marginal" else 1e-4,
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
