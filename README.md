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
module.

### SamplesLoss

```python
import torch
from sinkslot import SamplesLoss

x = torch.randn(10000, 2, requires_grad=True, device="cuda")  # also works on CPU, drop device="cuda"
y = torch.randn(10000, 2, device="cuda")

loss = SamplesLoss(eps=0.05, L=64, n_iters=200)  # backend="auto": automatically detects whether Triton is installed
cost = loss(x, y)                          # achieved SLOT_eps cost, <T, C>
grad_x, = torch.autograd.grad(cost, [x])   # analytic gradient, no backprop through Sinkhorn
```

### Potentials

```python
phi, psi = loss(x, y, potentials=True)
```

### Sparse Transport Plan and Barycentric Map

```python
from sinkslot import sparse_transport_plan, sparse_barycentric_map

P = sparse_transport_plan(x, y, eps=0.05, L=64, seed=0, n_iters=200)  # torch.sparse_coo_tensor (10000, 10000): P[i, j] = mass moved x[i] -> y[j]
rows, cols = P.indices()
Tx, Ty = sparse_barycentric_map(P.values(), rows, cols, x, y)
```

### Gradient Flow (slot_grad, same gradients without autograd)

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

## API Reference

### SamplesLoss

```python
SamplesLoss(
    loss="sinkhorn",   # only "sinkhorn" is implemented
    eps=0.05,          # entropic regularisation strength
    L=64,               # number of random slicing directions
    seed=0,             # RNG seed for the slicing directions
    n_iters=200,        # Sinkhorn iteration cap (exact count unless stop overrides it)
    backend="auto",     # "auto" | "triton" | "torch"
    symmetric=False,    # False: alternating (Gauss-Seidel) update; True: symmetric (Jacobi)
    alpha=0.5,          # Jacobi blend weight, only used when symmetric=True
    stop=None,          # early stopping config, see sinkslot_solve below
)

loss(x, y, a=None, b=None, potentials=False)
# a/b: source/target marginal weights, independent, uniform if omitted
# potentials=True returns (phi, psi) instead of the scalar cost
```

### sparse_transport_plan

```python
sparse_transport_plan(
    x, y, a=None, b=None,   # a/b: source/target marginal weights, uniform if omitted
    eps=0.05, L=64, seed=0, n_iters=200, stop=None,
    backend="auto", variant="alternating", alpha=0.5,
)
```

Same arguments as `sinkslot_solve`, but returns the plan itself: a
`torch.sparse_coo_tensor` of shape `(n, m)`. Non-differentiable, like
`potentials=True`.

### sinkslot_solve

`stop` (on `SamplesLoss` and `sparse_transport_plan` above, or passed
directly here) is any object with these four attributes -- a small
`dataclass` is the easiest way:

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

Benchmarked against [FlashSinkhorn](https://github.com/ot-triton-lab/flash-sinkhorn),
[SROT](https://github.com/khainb/SROT), and
[Spar-Sink](https://github.com/Mengyu8042/Spar-Sink).

## Citation

The arXiv paper isn't public yet. A citation will be added here once it is.
