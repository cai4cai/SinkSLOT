<p align="center">
  <img src="assets/overview.png" alt="Overview of the SinkSLOT pipeline" width="100%">
</p>

# SinkSLOT: Sinkhorn via Sparse Lifted Optimal Transport

[![arXiv](https://img.shields.io/badge/arXiv-2608.28262-b31b1b.svg)](https://arxiv.org/abs/2608.28262)

SinkSLOT computes entropic optimal transport (EOT) using a sparse sliced
lifted plan as the reference measure, instead of the usual independent
product `a ⊗ b`. Restricting each Sinkhorn iteration to that plan's support
reduces the per-iteration cost from `O(N²)` to `O(LN)`, where `L` is the
number of random slicing directions.

The resulting objective, SLOT, is a divergence: SLOT(x, x) = 0, so it
needs no debiasing.

On CUDA, `backend="auto"` (the default) uses the fused Triton kernels when
Triton is installed; otherwise, and always on CPU, SinkSLOT automatically
falls back to plain PyTorch, running the same algorithm just without the
fused-kernel throughput.

## Install

This repo isn't published to PyPI; install from source:

```bash
git clone https://github.com/cai4cai/SinkSLOT
cd SinkSLOT
pip install -e .
```

**Requirements:** PyTorch >= 2.5. Triton >= 3.1 is optional
(`pip install -e ".[triton]"`) for the fused Triton/CUDA kernels. Without
it, SinkSLOT runs the pure-torch fallback automatically.

## Tensor Contract

- `x`: `(N, d)` floating source point cloud.
- `y`: `(M, d)` floating target point cloud.
- `a`: `(N,)` optional nonnegative source weights, uniform if omitted.
- `b`: `(M,)` optional nonnegative target weights, uniform if omitted.

`x`, `y`, `a`, `b` must all share the same dtype and device. `a`/`b` are
used exactly as given, not normalized; `sum(a)` must equal `sum(b)`, which
is checked and raises `ValueError` on mismatch. Zero entries are legal.

Ground cost is squared Euclidean, $C_{ij} = \|x_i - y_j\|^2$, which sets
the scale `eps` is measured against.

`seed` controls the random slicing directions: the same `(x.shape[-1], L,
seed)` always gives the same directions, from a generator local to that
call, independent of global PyTorch RNG state or any prior call.

| | Supported |
|---|---|
| `N != M` | Yes |
| `float32` | Yes, on every backend |
| `float64` | Yes on `backend="torch"`. No on `backend="triton"`, raises `ValueError`: the Triton kernels are float32-only |
| CPU | Yes. `backend="auto"` (the default) resolves to `torch`; Triton requires CUDA |
| CUDA | Yes. `backend="auto"` (the default) picks `triton` when installed, `torch` otherwise |

## Basic Usage

The quickest way to call SinkSLOT is `sinkslot.SamplesLoss`, a
[GeomLoss](https://www.kernel-operations.io/geomloss/)-style callable loss
module.

### SamplesLoss

```python
import torch
from sinkslot import SamplesLoss

device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(10000, 2, requires_grad=True, device=device)
y = torch.randn(10000, 2, device=device)

loss = SamplesLoss(eps=0.05, L=64, n_iters=200)
cost = loss(x, y)                          # achieved SLOT_eps divergence (primal value)
grad_x, = torch.autograd.grad(cost, [x])   # analytic gradient, no backprop through Sinkhorn
phi, psi = loss(x, y, potentials=True)     # dual potentials
```

### Gradient Flow

An explicit gradient-descent loop works either through `autograd`
(`SamplesLoss`) or directly via `slot_grad`: same analytic gradient either
way, the loops differ only in how `grad` gets computed.

<table>
<tr><th>autograd</th><th>slot_grad</th></tr>
<tr>
<td>

```diff
 device = "cuda" if torch.cuda.is_available() else "cpu"
 n = 1000
 x = torch.randn(n, 2, device=device)
 y = torch.randn(n, 2, device=device) + 3.0
 a = torch.full((n,), 1.0 / n, device=device)
 b = torch.full((n,), 1.0 / n, device=device)
 loss = SamplesLoss(eps=0.01, L=100, n_iters=200)

 lr = 0.1
 for step in range(200):
     x = x.detach().requires_grad_(True)
     cost = loss(x, y, a, b)
+    grad, = torch.autograd.grad(cost, [x])
     x = x - lr * grad
```

</td>
<td>

```diff
 from sinkslot import slot_grad

 device = "cuda" if torch.cuda.is_available() else "cpu"
 n = 1000
 x = torch.randn(n, 2, device=device)
 y = torch.randn(n, 2, device=device) + 3.0
 a = torch.full((n,), 1.0 / n, device=device)
 b = torch.full((n,), 1.0 / n, device=device)

 lr = 0.1
 for step in range(200):
+    grad = slot_grad(
+        x, y, a, b, eps=0.01,
+        L=100, seed=0, n_iters=200)
     x = x - lr * grad
```

</td>
</tr>
</table>

### Sparse Transport Plan and Barycentric Map

```python
from sinkslot import sparse_transport_plan, sparse_barycentric_map

device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(10000, 2, device=device)
y = torch.randn(10000, 2, device=device)

P = sparse_transport_plan(x, y, eps=0.05, L=64, seed=0, n_iters=200)  # torch.sparse_coo_tensor (10000, 10000): P[i, j] = mass moved x[i] -> y[j]
rows, cols = P.indices()
mass = P.values()          # mass[k]: transported from x[rows[k]] to y[cols[k]]

Tx, Ty = sparse_barycentric_map(P, x, y)
# Tx[i]: barycentric image of source point x[i] in the target cloud
# Ty[j]: barycentric image of target point y[j] in the source cloud
```

## API Reference

### SamplesLoss

```python
SamplesLoss(
    eps=0.05,                # entropic regularisation strength
    L=64,                    # number of random slicing directions
    seed=0,                  # RNG seed for the slicing directions
    n_iters=200,             # exact iteration count when stop_mode="fixed" (the default)
    backend="auto",          # "auto" | "triton" | "torch"
    variant="alternating",   # "alternating" (Gauss-Seidel) | "symmetric" (Jacobi)
    alpha=0.5,                # Jacobi blend weight, only used when variant="symmetric"
    stop_mode="fixed",       # "fixed" | "marginal" | "potential"
    stop_max_iter=20000,     # iteration cap when stop_mode != "fixed" (n_iters is then unused)
    stop_tol=1e-6,           # convergence threshold
    stop_check_every=5,      # check convergence every N iterations
)

loss(x, y, a=None, b=None, potentials=False)
# a/b: source/target marginal weights, independent, uniform if omitted
# potentials=True returns (phi, psi) instead of the SLOT_eps divergence
```

`stop_mode` (shared by `SamplesLoss`, `sparse_transport_plan`, and
`sinkslot_solve` below):

- `"fixed"`: runs exactly `n_iters`, no convergence check.
- `"marginal"`: stop once $\|(P\mathbf{1} - a,\, P^\top \mathbf{1} - b)\|_\infty \le$ `stop_tol` (both marginals' max deviation).
- `"potential"`: stop once $\varepsilon \max(|\Delta\phi|, |\Delta\psi|) \lt$
  `stop_tol` (change in the dual potentials since the last check).

### sparse_transport_plan

```python
sparse_transport_plan(
    x, y, a=None, b=None,      # a/b: source/target marginal weights, uniform if omitted
    eps=0.05, L=64, seed=0, n_iters=200,
    stop_mode="fixed", stop_max_iter=20000, stop_tol=1e-6, stop_check_every=5,
    backend="auto", variant="alternating", alpha=0.5,
)
```

Same arguments as the low-level solver below, but returns the plan itself: a
`torch.sparse_coo_tensor` of shape `(n, m)`. Non-differentiable, like
`potentials=True`.

## Low-Level Solver API

`sinkslot_solve` is what `SamplesLoss` and `sparse_transport_plan` are both
built on. Same arguments, but it returns the raw solve state as a
`SinkslotSolveResult` NamedTuple instead of a scalar cost or a plan tensor
(unpacks/indexes like a plain tuple, plus named-field access).

```python
from sinkslot import sinkslot_solve

result = sinkslot_solve(
    x, y, a, b, eps=0.05, L=64, seed=0, n_iters=200,
    stop_mode="marginal", stop_max_iter=20000, stop_tol=1e-6, stop_check_every=5,
)
# also unpacks/indexes like a plain tuple:
# phi, psi, rows, cols, S, iters_run, converged, final_viol = result
```

- `result.phi`/`result.psi`: final dual potentials, absorbed (`phi = f/eps`)
  -- not necessarily converged under `stop_mode="fixed"`, which always runs
  exactly `n_iters` with no convergence check.
- `result.rows`/`result.cols`/`result.S`: the sliced-OT support and
  reference plan. `S[k]` is the reference mass on `(rows[k], cols[k])`;
  the achieved plan's own value there is
  `(phi[rows] + psi[cols] + S.log() - cost/eps).exp()`, what
  `sparse_transport_plan` returns.
- `result.iters_run`/`result.converged`/`result.final_viol`:
  `None`/`None`/`None` under `stop_mode="fixed"` (ran exactly `n_iters`);
  otherwise the actual iteration count, whether it converged within
  `stop_max_iter`, and the final convergence-check value.

## Reproducing the paper

Every result in the paper comes from one of the configs or commands below. Synthetic benchmarks run
through `run.py` (sharded with `--num-shards K --shard-idx k`, resumable, merged with `--merge`);
colour transfer runs through `color_transfer/`. Each config's docstring says which result it
produces. Scripts that turn the result CSVs and JSONs into the paper's tables and figures are not
part of the repository.

### Main text

| Result | How |
|---|---|
| Table 1, time to a cost gap (synthetic benchmark) | `configs/speedup.py`, plus `configs/speedup_multiscale_grid.py` (GeomLoss multiscale, grid) and `configs/speedup_sparsink_k.py` (Spar-Sink k = 256, 512). N = M = 10,000; half-moons, 8-Gaussians and two-rings at d = 2, Gaussian at d = 64; 10 eps values per dataset; 5 seeds; potential-change stop at 1e-5 median(C), cap 5e5. Prepare once with `scripts/speedup_prepare.py` |
| Figure 2, scalability | `configs/scalability_figure2.py`: Gaussian data at eps = 0.1, N from 5,000 to 50,000 at d = 64 and d from 4 to 1024 at N = 10,000; 5 seeds |
| Table 2, colour transfer | `python -m color_transfer.main --output_dir OUT --eps 0.01 --sinkslot_L 100 --max_iter 500000 --sym_alpha 0.9 --geomloss_truncate 1 --methods sinkslot sinkslot_sym flashsinkhorn_alt flashsinkhorn_alt_fp32 geomloss_online geomloss_multiscale`, then the K-means row with `--methods geomloss_multiscale --geomloss_points_per_cluster 250` into another `--output_dir`. All 132 ordered pairs of the 12 paintings; `--num_shards`/`--shard_idx` split them |
| Figure 3, colour transfer images | `python -m color_transfer.export_qualitative --output_dir OUT --eps 0.01 --sym_alpha 0.9 --geomloss_truncate 1 --geomloss_points_per_cluster 250 --pair_idx 11 5 3 4 --methods flashsinkhorn geomloss_online geomloss_multiscale sinkslot` (water lilies to haystacks, bridge to garden) |

### Appendix

| Result | How |
|---|---|
| Gradient flow, blob to crescent | `python -m gradient_flow.run` (config in `gradient_flow/config.py`): N = 1000, eps = 0.01, L = 100, 50 steps; SOT, EOT, SROT, SinkSLOT |
| Dropped-term finite-difference check (table) | `python -m gradient_flow.finite_diff` |
| Gradient term norms along the flow (figure) | `python -m gradient_flow.term_norms` |
| Precision table (FlashSinkhorn FP32 vs TF32) | the FlashSinkhorn runs of `configs/speedup.py` |
| Damping table | alpha = 0.9 rows and GeomLoss multiscale (K-means) from `configs/speedup.py`, grid from `configs/speedup_multiscale_grid.py`; alpha = 0.3, 0.5, 0.7 and 0.95 from `configs/speedup_damping.py` with `DAMPING_ALPHA` set to each value |
| Truncation table (truncate 1 vs 5) | truncate 1 from `configs/speedup.py` (K-means) and `configs/speedup_multiscale_grid.py` (grid); truncate 5 from `configs/speedup_multiscale_truncate5.py` with `MULTISCALE_CLUSTERING=kmeans` and `=grid` |
| Convergence and memory tables, Pareto figure | the same runs as Table 1 |
| Memory figure | the same runs as Figure 2 (`configs/scalability_figure2.py`) |

`configs/scalability.py`, `configs/speedup_cap5e4.py`, `configs/speedup_gauss3_long.py`,
`configs/speedup_tf32.py` and `configs/speedup_rounded.py` are not needed to reproduce the paper:
the first is the base of `configs/scalability_figure2.py`, the next three reran subsets of an earlier
version of `configs/speedup.py` at higher iteration caps, and the last computes rounded-plan cost gaps.

```bash
python scripts/speedup_prepare.py                                          # once: medians of the cost, caches
python run.py --config speedup --count                                     # unit counts
python run.py --config speedup --execute --num-shards 210 --shard-idx 0    # one shard
python run.py --config speedup --merge                                     # shard CSVs into forward_all.csv
DAMPING_ALPHA=0.5 python run.py --config speedup_damping --execute         # one alpha of the damping table
python run.py --config scalability_figure2 --execute --num-shards 20 --shard-idx 0
python -m gradient_flow.run                                                # the gradient-flow figure
```

Benchmarked against [FlashSinkhorn](https://github.com/ot-triton-lab/flash-sinkhorn),
[SROT](https://github.com/khainb/SROT), and
[Spar-Sink](https://github.com/Mengyu8042/Spar-Sink).

## Citation

If you use SinkSLOT, please cite:

Hsieh, I., Kundu, S. S., Vercauteren, T. and Dorent, R. (2026). [SinkSLOT: Sinkhorn via Sparse Lifted Optimal Transport](https://arxiv.org/abs/2608.28262). *arXiv preprint arXiv:2608.28262*.

```bibtex
@article{hsieh2026sinkslot,
  title={{SinkSLOT}: {Sinkhorn} via Sparse Lifted Optimal Transport},
  author={Hsieh, Ian and Kundu, Soumya Snigdha and Vercauteren, Tom and Dorent, Reuben},
  journal={arXiv preprint arXiv:2608.28262},
  year={2026},
  url={https://arxiv.org/abs/2608.28262}
}
```

## License

Apache-2.0
