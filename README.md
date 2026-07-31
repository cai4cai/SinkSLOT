# SinkSLOT

Fused-Triton sparse Sinkhorn on the unsmoothed sliced OT plan — our contribution.
**SinkSLOT-CUDA** is the same method with a CUDA-optimised setup path (2.1–3.1×
faster plan build), benchmarked as a peer method so the speedup is visible in the
tables.

Built on top of [FlashSinkhorn](https://github.com/ot-triton-lab/flash-sinkhorn)
and the sliced / sparse OT baselines it is compared against:
[SROT](https://github.com/khainb/SROT) (Nguyen),
[Spar-Sink](https://github.com/Mengyu8042/Spar-Sink) (Li, Yu, Li, Meng), and
GeomLoss / OTT-JAX.

## Where things are

### The package — `torch-ext/flash_sinkhorn/`

| File | What |
|---|---|
| `samples_loss.py` | `SamplesLoss` — the GeomLoss-compatible entry point |
| `sinkhorn_solvers.py` | Sinkhorn solver drivers |
| `kernels/` | fused Triton kernels (forward, gradient, apply-plan, c-transform) |
| `_autograd.py`, `implicit_grad.py` | analytic gradients (no backprop through iterations) |
| `hvp.py`, `cg.py` | Hessian-vector products via streaming CG |
| `c_transform.py` | streaming hard-argmin c-transform / semi-dual OT |

### The benchmark — `torch-ext/flash_sinkhorn/bench/`

| File | What |
|---|---|
| `bench_forward.py` | forward sweep + every baseline adapter and its RMAE reference: FlashSinkhorn, GeomLoss/KeOps, OTT-JAX, SROT, Spar-Sink, Rand-Sink, SinkSLOT, SinkSLOT-CUDA |
| `bench_backward.py` | gradient-evaluation sweep |
| `sinkslot.py` | SinkSLOT (fused-Triton γ=0 sparse SROT): sliced-plan builder + Triton kernels + v5 loop. Also the SinkSLOT-CUDA setup path: `_ot_1d_coo_batched_cuda` (fp64/transposed plan build), `sparse_sqeuclidean_cost` (fused cost kernel), int32-key `to_csr` |

### The harness — repo root

| File | What |
|---|---|
| `configs/base.py` | `BenchConfig` + a quick-iteration sweep (tiny grid, runs in minutes; not used for any reported number) |
| `configs/speedup*.py`, `configs/scalability.py` | the published sweeps — see [Which config produced which result](#which-config-produced-which-result) |
| `run.py` | sweep driver for `configs/speedup*.py`: one subprocess per measurement, results appended to one CSV |
| `scripts/scalability.py` | command generator for `configs/scalability.py` (see below for why this one isn't a plain `run.py` sweep) |
| `view.py` | results viewer (textual TUI; `--print` for static tables) |

### The gradient-flow figure — `gradient_flow/`

| File | What |
|---|---|
| `config.py` | fixed experiment constants (N, steps, learning rate, ε, L — no sweep, so a plain module rather than a `BenchConfig`) |
| `vendor/sinkhorn_methods.py` | dense, differentiable SOT/EOT/SROT baselines (autograd through a plain-torch Sinkhorn loop), ported from a sibling repo since this one has no such loss of its own — `torch-ext/flash_sinkhorn/bench` is benchmark-timing code only |
| `solver.py` | the SinkSLOT arm's gradient, built from the native v5 solver (`torch-ext/flash_sinkhorn/bench/sinkslot.py`) plus the envelope-theorem projection `grad_X SLOT_eps(X,Y) = 2*diag(a)*(X - T_eps(X))` |
| `run.py` | runs all 4 methods and writes the compiled figure |
| `data/` | the two density images (blob → crescent) sampled into point clouds |

```bash
python -m gradient_flow.run    # needs a CUDA GPU: the SinkSLOT arm's kernels are Triton
```

## Run a sweep

```bash
python run.py --dry-run                        # quick grid from configs/base.py
python run.py --execute                        # run it -> output/quick
python run.py --config speedup --execute       # a published sweep
python view.py --output-dir output/quick

python scripts/scalability.py                   # print every scalability command (dry run)
python scripts/scalability.py --execute         # actually run them
```

## Which config produced which result

| Paper | Config | Grid |
|---|---|---|
| Table 1 (speedup at each cost-gap threshold), half-moons / 8-Gaussians / two-rings | `configs/speedup.py` | N=M=10,000, d=2, ε 8-point log grid over [0.001, 0.1], L 32–4096 |
| Table 1, Gaussian d=3 | `configs/speedup_gaussian_d3.py` | same policy, d=3 |
| Table 1, Gaussian d=64 | `configs/speedup_gaussian_d64.py` | same policy, d=64, wider ε range [0.1, 1] |
| Figure 2 (scalability in N and d) | `configs/scalability.py` + `scripts/scalability.py` | N ∈ {5k,10k,20k,30k,50k} at d∈{3,64}; d ∈ {4…1024} at N=10,000; 5 seeds; SinkSLOT-CUDA/SROT/FlashSinkhorn-alternating |
| Figure 3 (gradient flow, blob → crescent) | `gradient_flow/config.py` + `gradient_flow/run.py` | N=1000, ε=0.01, L=100, 50 gradient steps; SOT/EOT/SROT/SinkSLOT |

The three benchmark configs share one solver policy: marginal stopping on
`max(‖P1−a‖∞, ‖Pᵀ1−b‖∞) < 1e-6`, float32, TF32 off, and the exact LP reference
plan from POT's network simplex (`ot.emd`, CPU, float64). `max_iter` is 10000
for `speedup.py` and 20000 for the other two (see each file's docstring for why).

`configs/scalability.py` sweeps SinkSLOT-CUDA, SROT and FlashSinkhorn-alternating
over 5 seeds each, with L swept independently of N/d per method — an axis
`run.py`'s single-`BenchConfig` sweep can't express, which is why
`scripts/scalability.py` builds its commands directly instead (see that config's
docstring for the full reasoning, including why Spar-Sink was dropped from this
particular experiment: it can't reach N=50,000 without hitting a `torch.nonzero()`
int32 index-overflow in its sampling step). Spar-Sink's sample budget in
`configs/speedup*.py` follows the authors' own formula, `s = k*s0(n)` with
`s0(n) = 1e-3*n*log(n)^4`, `k in {2,4,8,16}`.

SROT and Spar-Sink are adapted from the authors' released implementations
([SROT](https://github.com/khainb/SROT),
[Spar-Sink](https://github.com/Mengyu8042/Spar-Sink)) and validated against
them. The changes are confined to this harness — GPU, precision, stopping rule
and timing instrumentation are held fixed across every method, so the algorithm
is the only difference. The update equations, reference coupling and sampling
scheme are the authors' own.
