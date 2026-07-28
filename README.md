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
| `run.py` | sweep driver: one subprocess per measurement, results appended to one CSV |
| `view.py` | results viewer (textual TUI; `--print` for static tables) |

## Run a sweep

```bash
python run.py --dry-run                        # quick grid from configs/base.py
python run.py --execute                        # run it -> output/quick
python run.py --config speedup --execute       # a published sweep
python view.py --output-dir output/quick
```

## Which config produced which result

| Paper | Config | Grid |
|---|---|---|
| Table 1 (speedup at each cost-gap threshold), half-moons / 8-Gaussians / two-rings | `configs/speedup.py` | N=M=10,000, d=2, ε 8-point log grid over [0.001, 0.1], L 32–4096 |
| Table 1, Gaussian d=3 | `configs/speedup_gaussian_d3.py` | same policy, d=3 |
| Table 1, Gaussian d=64 | `configs/speedup_gaussian_d64.py` | same policy, d=64, wider ε range [0.1, 1] |
| Figure 2 (scalability in N and d) | `configs/scalability.py` | N ∈ {5k,10k,20k,30k,50k}; d ∈ {4…1024} at N=10,000 |

All four share one solver policy: marginal stopping on
`max(‖P1−a‖∞, ‖Pᵀ1−b‖∞) < 1e-6`, `max_iter=10000`, float32, TF32 off, and the
exact LP reference plan from POT's network simplex (`ot.emd`, CPU, float64).

`configs/scalability.py` holds the constants only; its per-unit commands are built
by a cluster launcher that is not part of this artifact. Spar-Sink's sample budget
there is density-based, so the absolute value is recomputed per N.

SROT and Spar-Sink are adapted from the authors' released implementations
([SROT](https://github.com/khainb/SROT),
[Spar-Sink](https://github.com/Mengyu8042/Spar-Sink)) and validated against
them. The changes are confined to this harness — GPU, precision, stopping rule
and timing instrumentation are held fixed across every method, so the algorithm
is the only difference. The update equations, reference coupling and sampling
scheme are the authors' own.
