<p align="center">
  <img src="assets/overview.png" alt="Overview of the SinkSLOT pipeline" width="100%">
</p>

# SinkSLOT

SinkSLOT computes entropic optimal transport (EOT) using a sparse sliced
lifted plan as the reference measure, instead of the usual independent
product `a ⊗ b`. Restricting each Sinkhorn iteration to that plan's support
reduces the per-iteration cost from `O(N²)` to `O(LN)`, where `L` is the
number of random slicing directions.

The resulting objective, SLOT, is a divergence: it needs no debiasing, and
SLOT(x, x) = 0.

By default SinkSLOT runs on fused Triton/CUDA kernels. It's also fully
usable with plain PyTorch, on CPU or on CUDA without Triton installed,
through an automatic pure-torch fallback: same algorithm, just without the
fused-kernel throughput.

## Install

This repo isn't published to PyPI; install from source:

```bash
git clone https://github.com/cai4cai/SinkSLOT
cd SinkSLOT
pip install -e .
```

**Requirements:** PyTorch >= 2.5. Triton >= 3.1 is optional
(`pip install -e ".[triton]"`) for the fused Triton/CUDA kernels -- without
it, SinkSLOT runs the pure-torch fallback automatically.

## Basic Usage

The quickest way to call SinkSLOT is `sinkslot.SamplesLoss`, a
[GeomLoss](https://www.kernel-operations.io/geomloss/)-style callable loss
module (same calling convention as `geomloss.SamplesLoss` /
`flash_sinkhorn.SamplesLoss`):

```python
import torch
from sinkslot import SamplesLoss

x = torch.randn(10000, 2, requires_grad=True, device="cuda")
y = torch.randn(10000, 2, device="cuda")

loss = SamplesLoss(eps=0.05, L=64, n_iters=200)
cost = loss(x, y)                          # achieved SLOT_eps cost, <T, C>
grad_x, = torch.autograd.grad(cost, [x])   # analytic gradient, no backprop through Sinkhorn
```

This picks the fused Triton kernels automatically, since Triton is installed
and the tensors are on CUDA. Without Triton, or on CPU, the exact same code
runs through the pure-torch fallback instead, no changes needed:

```python
x = torch.randn(10000, 2, requires_grad=True)   # CPU, no Triton needed
y = torch.randn(10000, 2)

loss = SamplesLoss(eps=0.05, L=64, n_iters=200)
cost = loss(x, y)
grad_x, = torch.autograd.grad(cost, [x])
```

### Gradient Flow

For an explicit gradient-descent loop, `slot_grad` gives you the same
analytic gradient directly, without going through `autograd`:

```python
from sinkslot import slot_grad

x = torch.randn(1000, 2, device="cuda")
y = torch.randn(1000, 2, device="cuda") + 3.0
a = torch.full((1000,), 1.0 / 1000, device="cuda")

lr = 0.1
for step in range(200):
    grad = slot_grad(x, y, a, eps=0.01, L=100, seed=0, n_iters=200)
    x = x - lr * grad
```

### Potentials and the Transport Plan

`potentials=True` returns the converged dual potentials `(phi, psi)`
instead of the scalar cost:

```python
phi, psi = loss(x, y, potentials=True)
```

For the transport plan itself, call `sinkslot_solve` directly. It returns
the potentials together with the sparse support (`rows`, `cols`) they were
solved on, from which the plan's nonzero values follow directly:

```python
from sinkslot import sinkslot_solve

a = torch.full((10000,), 1.0 / 10000)
eps = 0.05
phi, psi, rows, cols, S, iters_run, converged, final_viol = sinkslot_solve(
    x, y, a, a, eps=eps, L=64, seed=0, n_iters=200,
)

cost = (x[rows] - y[cols]).square().sum(1)
log_S = S.clamp_min(torch.finfo(S.dtype).tiny).log()
plan_vals = (phi[rows] + psi[cols] + log_S - cost / eps).exp()
```

`(rows[k], cols[k], plan_vals[k])` is the transport plan in sparse COO form:
`plan_vals[k]` is the mass moved between `x[rows[k]]` and `y[cols[k]]`. Most
`(i, j)` pairs never appear at all -- that sparsity is the whole point.

## API Reference

### `SamplesLoss`

```python
SamplesLoss(
    loss="sinkhorn",   # only "sinkhorn" is implemented
    eps=0.05,          # entropic regularisation strength
    L=64,               # number of random slicing directions
    seed=0,             # RNG seed for the slicing directions
    n_iters=200,        # fixed Sinkhorn iteration count (no early stopping)
    backend="auto",     # "auto" | "triton" | "torch"
    symmetric=False,    # False: alternating (Gauss-Seidel) update; True: symmetric (Jacobi)
    alpha=0.5,          # Jacobi blend weight, only used when symmetric=True
)
```

### `sinkslot_solve`

Early stopping isn't exposed on `SamplesLoss`; for that, call the
lower-level `sinkslot_solve` directly and pass a `stop` object (any object
with these four attributes -- a small `dataclass` is the easiest way):

```python
from dataclasses import dataclass
from sinkslot import sinkslot_solve

@dataclass
class Stop:
    mode: str = "fixed"     # "fixed" | "marginal" | "potential" | "potential_linf"
    max_iter: int = 20000   # iteration cap when mode != "fixed"
    tol: float = 1e-6       # convergence threshold
    check_every: int = 5    # check convergence every N iterations

phi, psi, rows, cols, S, iters_run, converged, final_viol = sinkslot_solve(
    x, y, a, b, eps=0.05, L=64, seed=0, n_iters=200, stop=Stop(mode="marginal"),
)
```

Unlike `SamplesLoss`, `sinkslot_solve` takes independent source/target
weights `a`/`b`.

`stop.mode`:

- `"fixed"` (default when `stop=None`): ignore the rest of `stop` and run
  exactly `n_iters` iterations.
- `"marginal"` / `"potential"`: run up to `stop.max_iter`, checking every
  `stop.check_every` iterations, and stop once the max (L-infinity) marginal
  violation drops below `stop.tol`.
- `"potential_linf"`: stop once the dual potentials themselves stop moving,
  `max(|Δphi|, |Δpsi|) < stop.tol` (in the potentials' physical scale, not
  the absorbed `phi = f/eps` form used internally).

## Reproducing the paper

| Result | Config | Grid |
|---|---|---|
| Table 1, half-moons / 8-Gaussians / two-rings | `configs/speedup.py` | N=M=10,000, d=2, 8-point log grid for ε in [0.001, 0.1], L 32 to 4096 |
| Table 1, Gaussian d=3 | `configs/speedup_gaussian_d3.py` | same policy, d=3 |
| Table 1, Gaussian d=64 | `configs/speedup_gaussian_d64.py` | same policy, d=64, ε in [0.1, 1], L 64 to 8192 |
| Figure 2, scalability | `configs/scalability.py` via `scripts/scalability.py` | N in {5k,10k,20k,30k,50k} at d in {3,64}; d in {4..1024} at N=10,000; 5 seeds |
| Figure 3, gradient flow | `gradient_flow/config.py` via `gradient_flow/run.py` | N=1000, ε=0.01, L=100, 50 steps; SOT/EOT/SROT/SinkSLOT |
| Appendix, gradient term split | `gradient_flow/term_norms.py` | same problem; splits the complete gradient into the envelope term and the residual |
| Appendix, dropped-term finite difference | `gradient_flow/finite_diff.py` | same problem, float64, 6 random directions per point |
| Gradient accuracy under early stopping | `gradient_flow/appendix_checks/stopping.py` | same problem, inner iterations 1 to 1000 against a 5000-iteration reference |

```bash
python run.py --config speedup --execute   # a published sweep
python -m gradient_flow.run                # the gradient-flow figure
python scripts/scalability.py              # print every scalability command
python scripts/scalability.py --execute    # run them
```

All three benchmark configs share one solver policy: marginal stopping on
`max(‖P1 - a‖∞, ‖Pᵀ1 - b‖∞) < 1e-6`, float32, TF32 off, and the exact LP
reference plan from POT's network simplex (`ot.emd`, CPU, float64). `max_iter`
is 10,000 for `speedup.py` and 20,000 for the other two; each file's docstring
says why.

`configs/scalability.py` sweeps L independently of N and d per method, an axis
`run.py`'s single-`BenchConfig` sweep cannot express, so
`scripts/scalability.py` generates its commands directly. Spar-Sink is absent
from that experiment: it cannot reach N=50,000 without an int32 index overflow
in `torch.nonzero()` during sampling. In `configs/speedup*.py` its sample budget
follows the authors' formula, `s = k·s₀(n)` with `s₀(n) = 1e-3·n·log⁴(n)` and
`k` in {5,10,15,20}.

## Baselines

Built on [FlashSinkhorn](https://github.com/ot-triton-lab/flash-sinkhorn), and
compared against [SROT](https://github.com/khainb/SROT) (Nguyen),
[Spar-Sink](https://github.com/Mengyu8042/Spar-Sink) (Li, Yu, Li, Meng).

SROT and Spar-Sink are adapted from the authors' released implementations and
validated against them. Changes are confined to this harness: GPU, precision,
stopping rule and timing instrumentation are fixed across every method, so the
algorithm is the only difference. The update equations, reference coupling and
sampling scheme are the authors' own.

## Citation

The arXiv paper isn't public yet. A citation will be added here once it is.
