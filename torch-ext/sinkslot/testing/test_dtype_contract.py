"""What dtypes SinkSLOT accepts, and what it refuses.

The contract is not the same on both devices, which is worth stating plainly
because everything else about the two paths is deliberately identical:

    float32   accepted everywhere
    float64   accepted on the pure-torch path; refused on the Triton path,
              whose cost kernel has a fixed float32 accumulator
    float16   refused everywhere
    bfloat16  refused everywhere

Half precision is refused rather than merely discouraged. It used to be
accepted on the pure-torch path and returned numbers that looked reasonable
and were not: measured against a float64 reference at n=2048, bfloat16 comes
back as inf at every eps tried, and float16 lands several percent out. The
cost expression is genuinely dtype-agnostic -- the failure is downstream, in
the log-domain iteration, whose exp() overflows a narrow exponent. Nothing
raised, so the only signal was the wrong answer.

The float64/Triton asymmetry is a real restriction, not an oversight, so
there is a test asserting it stays an explicit error rather than decaying
into a silent downcast.
"""

import pytest
import torch

from sinkslot import SamplesLoss, sinkslot_solve

HALF = [torch.float16, torch.bfloat16]
FULL = [torch.float32, torch.float64]


def _pair(dtype, device="cpu", n=48, m=40, d=3):
    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(n, d, generator=g).to(device=device, dtype=dtype)
    y = (torch.randn(m, d, generator=g) + 1.0).to(device=device, dtype=dtype)
    return x, y


@pytest.mark.parametrize("dtype", HALF)
def test_half_precision_is_refused_on_cpu(dtype):
    x, y = _pair(dtype)
    with pytest.raises(ValueError, match="half precision"):
        sinkslot_solve(x, y, eps=0.1, L=16, seed=0, n_iters=10)


@pytest.mark.parametrize("dtype", HALF)
def test_half_precision_is_refused_through_samples_loss_too(dtype):
    """The guard has to sit where every entry point passes through it."""
    x, y = _pair(dtype)
    with pytest.raises(ValueError, match="half precision"):
        SamplesLoss(eps=0.1, L=16, seed=0, n_iters=10)(x, y)


@pytest.mark.parametrize("dtype", FULL)
def test_full_precision_runs_and_keeps_its_dtype_on_cpu(dtype):
    x, y = _pair(dtype)
    res = sinkslot_solve(x, y, eps=0.1, L=16, seed=0, n_iters=10)
    assert res.phi.dtype == dtype
    assert torch.isfinite(res.phi).all()


def test_float32_and_float64_agree_on_cpu():
    """Guards the accuracy claim behind allowing float64 at all."""
    loss = lambda dt: float(  # noqa: E731
        SamplesLoss(eps=0.1, L=16, seed=0, n_iters=50)(*_pair(dt))
    )
    assert loss(torch.float32) == pytest.approx(loss(torch.float64), rel=1e-3)


@pytest.mark.parametrize("dtype", HALF)
def test_half_precision_is_refused_on_cuda_too(dtype):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    x, y = _pair(dtype, device="cuda")
    with pytest.raises(ValueError):
        sinkslot_solve(x, y, eps=0.1, L=16, seed=0, n_iters=10)


def test_float64_is_an_explicit_error_on_the_triton_path():
    """The one place the two devices genuinely disagree.

    float64 is fine on CPU and cannot work on the Triton path, whose cost
    kernel accumulates in a fixed float32. That has to stay a loud error: a
    silent downcast would hand back float32-quality numbers wearing a float64
    dtype, which is precisely the failure mode this file exists to prevent.
    """
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    x, y = _pair(torch.float64, device="cuda")
    with pytest.raises(ValueError, match="float32"):
        sinkslot_solve(x, y, eps=0.1, L=16, seed=0, n_iters=10, backend="triton")
