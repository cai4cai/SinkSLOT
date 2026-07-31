# SinkSLOT

**Fused-Triton sparse Sinkhorn on the unsmoothed sliced-OT plan.**

Entropic OT restricted to the support of a sliced-OT reference plan, solved with
fused Triton kernels. **SinkSLOT-CUDA** is the same method with a CUDA-optimised
setup path — a 2.1–3.1× faster plan build — benchmarked as a peer method so the
speedup shows up in the tables rather than as a footnote.

Requires a CUDA GPU: the solver is Triton-only, with no CPU fallback.

## Quick start

```bash
python run.py --dry-run                    # tiny grid from configs/base.py
python run.py --execute                    # run it -> output/quick
python view.py --output-dir output/quick   # results viewer (TUI; --print for tables)

python -m gradient_flow.run                # the gradient-flow figure
```

## Reproducing the paper

| Result | Config | Grid |
|---|---|---|
| Table 1 — half-moons / 8-Gaussians / two-rings | `configs/speedup.py` | N=M=10,000, d=2, ε over an 8-point log grid in [0.001, 0.1], L 32–4096 |
| Table 1 — Gaussian d=3 | `configs/speedup_gaussian_d3.py` | same policy, d=3 |
| Table 1 — Gaussian d=64 | `configs/speedup_gaussian_d64.py` | same policy, d=64, ε in [0.1, 1] |
| Figure 2 — scalability in N and d | `configs/scalability.py` via `scripts/scalability.py` | N ∈ {5k,10k,20k,30k,50k} at d ∈ {3,64}; d ∈ {4…1024} at N=10,000; 5 seeds |
| Figure 3 — gradient flow, blob → crescent | `gradient_flow/config.py` via `gradient_flow/run.py` | N=1000, ε=0.01, L=100, 50 steps; SOT/EOT/SROT/SinkSLOT |
| Gradient accuracy under early stopping | `gradient_flow/stopping.py` | same problem, inner iterations swept 1–1000 against a 5000-iteration reference |

```bash
python run.py --config speedup --execute   # a published sweep
python scripts/scalability.py              # print every scalability command
python scripts/scalability.py --execute    # actually run them
```

All three benchmark configs share one solver policy: marginal stopping on
`max(‖P1−a‖∞, ‖Pᵀ1−b‖∞) < 1e-6`, float32, TF32 off, and the exact LP reference
plan from POT's network simplex (`ot.emd`, CPU, float64). `max_iter` is 10,000
for `speedup.py` and 20,000 for the other two; each file's docstring says why.

`configs/scalability.py` sweeps L independently of N and d per method — an axis
`run.py`'s single-`BenchConfig` sweep cannot express, which is why
`scripts/scalability.py` generates its commands directly. Spar-Sink is absent
from that experiment specifically: it cannot reach N=50,000 without hitting an
int32 index overflow in `torch.nonzero()` during sampling. In `configs/speedup*.py`
its sample budget follows the authors' own formula, `s = k·s₀(n)` with
`s₀(n) = 1e-3·n·log(n)⁴` and `k ∈ {2,4,8,16}`.

## Repository map

### `torch-ext/sinkslot/` — the method

Kept in its own package rather than nested under `flash_sinkhorn/`: it is a
different algorithm (sparse Sinkhorn over a sliced-OT reference plan, not a
dense fused kernel), and it is a real solver used outside benchmarking — see
`gradient_flow/solver.py`.

`solver.py` holds the sliced-plan builder, the Triton kernels and the v5 loop,
plus the SinkSLOT-CUDA setup path: `_ot_1d_coo_batched_cuda` (fp64 transposed
plan build), `sparse_sqeuclidean_cost` (fused cost kernel) and int32-key
`to_csr`.

### `gradient_flow/` — Figure 3 and the gradient study

| File | What |
|---|---|
| `run.py` | runs all four methods and writes the figure |
| `config.py` | fixed constants (N, steps, learning rate, ε, L) — no sweep, so a plain module |
| `solver.py` | the SinkSLOT arm's gradient: the native v5 solver plus the envelope projection `∇ₓ SLOT_ε(X,Y) = 2·diag(a)·(X − T_ε(X))` |
| `vendor/sinkhorn_methods.py` | dense differentiable SOT/EOT/SROT baselines, autograd through a plain-torch Sinkhorn loop |
| `data/` | the two density images (blob → crescent) sampled into point clouds |

Beyond the figure, this package is where the gradient itself is investigated —
whether the envelope gradient is the right one, and what the alternatives cost:

```bash
python -m gradient_flow.stopping           # how early the inner solve can stop
python -m gradient_flow.term_norms         # split the gradient into its two terms
python -m gradient_flow.finite_diff        # is the dropped term really zero?
python -m gradient_flow.closed_form_check  # is the closed form an artefact?
```

`stopping.py` asks what a fixed `MAX_ITER=1000` buys. The envelope gradient is
exact only *at* the optimum, so stopping early biases it in a way the formula
cannot see; the gradient norm — the quantity you can measure at run time —
settles before the direction does, so it reads as converged while still turning.

`term_norms.py` splits the complete gradient into the envelope term and the
residual the envelope theorem drops, and `finite_diff.py` measures that dropped
term by central differences with the reference plan rebuilt at each perturbed
point. Each module's docstring carries its own results table.

### `torch-ext/flash_sinkhorn/` — the underlying package

| File | What |
|---|---|
| `samples_loss.py` | `SamplesLoss` — the GeomLoss-compatible entry point |
| `sinkhorn_solvers.py` | Sinkhorn solver drivers |
| `kernels/` | fused Triton kernels (forward, gradient, apply-plan, c-transform) |
| `_autograd.py`, `implicit_grad.py` | analytic gradients — no backprop through iterations |
| `hvp.py`, `cg.py` | Hessian-vector products via streaming CG |
| `c_transform.py` | streaming hard-argmin c-transform / semi-dual OT |

### `torch-ext/flash_sinkhorn/bench/` — the benchmark

`bench_forward.py` holds the forward sweep and every baseline adapter with its
RMAE reference; `bench_backward.py` the gradient-evaluation sweep. Both compare
all methods rather than any one of them, so nothing here is really specific to
the enclosing package — it stays nested for now because moving it is a larger
change than pulling the solver out was.

### Harness — repo root

| File | What |
|---|---|
| `configs/base.py` | `BenchConfig` plus a quick-iteration grid (minutes; not used for any reported number) |
| `run.py` | sweep driver for `configs/speedup*.py` — one subprocess per measurement, appended to one CSV |
| `scripts/scalability.py` | command generator for `configs/scalability.py` |
| `view.py` | results viewer |

## Baselines and attribution

Built on top of [FlashSinkhorn](https://github.com/ot-triton-lab/flash-sinkhorn),
and compared against [SROT](https://github.com/khainb/SROT) (Nguyen),
[Spar-Sink](https://github.com/Mengyu8042/Spar-Sink) (Li, Yu, Li, Meng),
GeomLoss and OTT-JAX.

SROT and Spar-Sink are adapted from the authors' released implementations and
validated against them. Changes are confined to this harness — GPU, precision,
stopping rule and timing instrumentation are held fixed across every method, so
the algorithm is the only thing that differs. The update equations, reference
coupling and sampling scheme are the authors' own.
