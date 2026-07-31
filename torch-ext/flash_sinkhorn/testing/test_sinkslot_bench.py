"""Tests for the SinkSLOT bench layer (`sinkslot.solver`).

Three equivalences the module otherwise only asserts in prose:

* the batched 1-D OT builder against an independent north-west-corner reference,
* the naive plan builder against the CUDA one, which must agree on the transport
  while remaining separate implementations,
* the fused Triton Sinkhorn loop against a plain-torch segmented LSE.

The middle one is regression cover. The naive builder is the SinkSLOT baseline
and the CUDA one is the treatment; if an edit or a merge ever makes their bodies
the same, the reported SinkSLOT-CUDA setup speedup silently becomes a
measurement of one path against itself, and every other test still passes.
"""

import pytest
import torch

triton = pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from sinkslot.solver import (  # noqa: E402
    _ot_1d_coo_batched,
    _ot_1d_coo_batched_cuda,
    _run_v5,
    sot_plan_coo,
    sparse_sqeuclidean_cost,
    to_csr,
)


def _problem(n=512, m=384, d=4, seed=0, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n, d, generator=g).to(device)
    y = (torch.randn(m, d, generator=g) + 1.5).to(device)
    a = (torch.rand(n, generator=g) + 0.1).to(device)
    b = (torch.rand(m, generator=g) + 0.1).to(device)
    return x, y, a / a.sum(), b / b.sum()


def _dense(rows, cols, vals, n, m):
    P = torch.zeros(n, m, device=vals.device, dtype=torch.float64)
    P.index_put_((rows, cols), vals.double(), accumulate=True)
    return P


def _ot_1d_reference(px, py, a, b):
    """North-west corner on the sorted order, in fp64. Independent of the module.

    The two cumulative-weight vectors cut [0, 1] into segments; each segment is a
    single (i, j) pair carrying its own length as mass.
    """
    ix, iy = torch.argsort(px), torch.argsort(py)
    ca = torch.cumsum(a[ix].double(), 0)
    cb = torch.cumsum(b[iy].double(), 0)
    bounds = torch.cat([ca, cb]).sort().values
    prev = torch.cat([bounds.new_zeros(1), bounds[:-1]])
    mass = bounds - prev
    keep = mass > 0
    mass, mid = mass[keep], (0.5 * (prev + bounds))[keep]
    i = torch.searchsorted(ca.contiguous(), mid).clamp_(max=ca.numel() - 1)
    j = torch.searchsorted(cb.contiguous(), mid).clamp_(max=cb.numel() - 1)
    return ix[i], iy[j], mass


def test_batched_1d_builder_matches_reference():
    """`_ot_1d_coo_batched` vectorises C independent 1-D OT solves."""
    n, m, C = 256, 192, 8
    _, _, a, b = _problem(n, m, d=2)
    g = torch.Generator(device="cpu").manual_seed(1)
    PX = torch.randn(n, C, generator=g).cuda()
    PY = torch.randn(m, C, generator=g).cuda()

    rows, cols, vals = _ot_1d_coo_batched(PX, PY, a, b)
    batched = _dense(rows, cols, vals, n, m)

    ref = torch.zeros(n, m, device="cuda", dtype=torch.float64)
    for c in range(C):
        r, o, v = _ot_1d_reference(PX[:, c], PY[:, c], a, b)
        ref += _dense(r, o, v, n, m)

    assert torch.allclose(batched, ref, atol=1e-5), \
        f"max |dP| = {(batched - ref).abs().max():.3e}"


def test_naive_and_cuda_builders_are_distinct_implementations():
    """The two plan builders must not collapse into the same code.

    Compares source rather than behaviour because behaviour is *supposed* to
    agree -- that is the next test -- and agreement cannot distinguish "two paths
    that agree" from "one path called twice".
    """
    import inspect

    def code(fn):
        body = inspect.getsource(fn).split('"""')[2]
        return [ln.strip() for ln in body.splitlines()
                if ln.strip() and not ln.strip().startswith("#")]

    naive, cuda = code(_ot_1d_coo_batched), code(_ot_1d_coo_batched_cuda)
    assert naive != cuda, (
        "the naive and CUDA sliced-plan builders have identical bodies -- the "
        "SinkSLOT-CUDA setup speedup would be measuring a path against itself"
    )
    # The distinguishing feature is where the accuracy/layout work lives.
    assert "double()" in "".join(cuda), "CUDA path lost its fp64 cumsum"
    assert "double()" not in "".join(naive), \
        "naive path gained the fp64 cumsum -- it is the untouched baseline"


def test_naive_and_cuda_builders_agree_on_the_transport():
    """Different supports, same transport.

    The fp32 and fp64 scans disagree on which breakpoints survive `mass > 0`, so
    the supports differ slightly, but the plan they define must match and both
    must lie in Gamma(a, b) -- the property the solver relies on for no row or
    column to be empty.
    """
    n, m = 512, 384
    x, y, a, b = _problem(n, m, d=4)

    r0, c0, v0 = sot_plan_coo(x, y, a, b, L=32, seed=0, ot1d=_ot_1d_coo_batched)
    r1, c1, v1 = sot_plan_coo(x, y, a, b, L=32, seed=0, ot1d=_ot_1d_coo_batched_cuda)

    P0, P1 = _dense(r0, c0, v0, n, m), _dense(r1, c1, v1, n, m)
    for P in (P0, P1):
        assert torch.allclose(P.sum(1).float(), a, atol=1e-5)
        assert torch.allclose(P.sum(0).float(), b, atol=1e-5)
    assert torch.allclose(P0, P1, atol=1e-5), f"max |dP| = {(P0 - P1).abs().max():.3e}"


def test_run_v5_matches_plain_torch_segmented_lse():
    """The fused Triton loop against a straightforward scatter-based Sinkhorn."""
    n, m, eps, L, iters = 512, 384, 0.05, 32, 40
    x, y, a, b = _problem(n, m, d=4)

    rows, cols, S = sot_plan_coo(x, y, a, b, L=L, seed=0)
    cost = sparse_sqeuclidean_cost(x, y, rows, cols)
    lam = S.clamp_min(torch.finfo(S.dtype).tiny).log() - cost / eps
    log_a, log_b = a.log(), b.log()

    r_ptr, r_idx, r_lam, _ = to_csr(rows, cols, lam, n)
    c_ptr, c_idx, c_lam, _ = to_csr(cols, rows, lam, m)
    phi, psi, *_ = _run_v5(r_ptr, r_idx, r_lam, c_ptr, c_idx, c_lam,
                           log_a, log_b, n, m, iters)

    def seg_lse(vals, idx, size):
        mx = vals.new_full((size,), -1e30).scatter_reduce(
            0, idx, vals, reduce="amax", include_self=True)
        acc = vals.new_zeros(size).index_add_(0, idx, (vals - mx[idx]).exp())
        return mx + acc.clamp_min(torch.finfo(vals.dtype).tiny).log()

    # Same absorbed convention as _run_v5: phi = f/eps, psi = g/eps.
    p = torch.zeros(n, device=x.device)
    q = torch.zeros(m, device=x.device)
    for _ in range(iters):
        p = log_a - seg_lse(lam + q[cols], rows, n)
        q = log_b - seg_lse(lam + p[rows], cols, m)

    # Potentials are only determined up to the usual (+c, -c) shift.
    assert torch.allclose(phi - phi.mean(), p - p.mean(), atol=2e-3), \
        f"max |dphi| = {((phi - phi.mean()) - (p - p.mean())).abs().max():.3e}"
    assert torch.allclose(psi - psi.mean(), q - q.mean(), atol=2e-3), \
        f"max |dpsi| = {((psi - psi.mean()) - (q - q.mean())).abs().max():.3e}"
