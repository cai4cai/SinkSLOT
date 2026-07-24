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
| `config.py` | `BenchConfig` + a quick-iteration sweep (tiny grid, runs in minutes) |
| `config_paper.py` | the same schema at full publication scale — structurally identical to `config.py`, only the values differ |
| `run.py` | sweep driver: one subprocess per measurement, results appended to one CSV |
| `view.py` | results viewer (textual TUI; `--print` for static tables) |
| `analysis.md` | working notes on every methodological choice (local, untracked) |

## Run a sweep

```bash
python run.py --dry-run                      # quick grid from config.py
python run.py --execute                       # run it -> output/quick
python run.py --config config_paper --execute # full publication sweep -> output/paper
python view.py --output-dir output/quick
```

Baselines ported from other repos (SROT, Spar-Sink, SinkSLOT) are written from
their papers/code and validated against upstream; provenance and every
convention choice are documented in `analysis.md`.
