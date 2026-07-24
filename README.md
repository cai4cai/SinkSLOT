# SinkSLOT

Fused-Triton sparse Sinkhorn on the unsmoothed sliced OT plan — our contribution.

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
| `bench_forward.py` | forward sweep + every baseline adapter and its RMAE reference: FlashSinkhorn, GeomLoss/KeOps, OTT-JAX, SROT, Spar-Sink, Rand-Sink, SinkSLOT |
| `bench_backward.py` | gradient-evaluation sweep |
| `sinkslot.py` | SinkSLOT (fused-Triton γ=0 sparse SROT): sliced-plan builder + Triton kernels + v5 loop |

### The harness — repo root

| File | What |
|---|---|
| `config.py` | `BenchConfig` — the sweep definition (sizes, dims, eps, methods, per-method params) |
| `run.py` | sweep driver: one subprocess per measurement, results appended to one CSV |
| `view.py` | results viewer (textual TUI; `--print` for static tables) |
| `analysis.md` | working notes on every methodological choice (local, untracked) |

## Run a sweep

```bash
python run.py --dry-run     # list the jobs config.py defines
python run.py --execute     # run them
python view.py --output-dir output/paper_benchmarks
```

Baselines ported from other repos (SROT, Spar-Sink, SinkSLOT) are written from
their papers/code and validated against upstream; provenance and every
convention choice are documented in `analysis.md`.
