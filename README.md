# SinkSLOT

Fused-Triton sparse Sinkhorn on the unsmoothed sliced-OT plan. Entropic OT is
restricted to the support of a sliced-OT reference plan and solved with fused
Triton kernels. SinkSLOT-CUDA is the same method with a CUDA-optimised setup
path (2.1 to 3.1× faster plan build), benchmarked as a peer method.

The benchmarked numbers below are all from the fused Triton kernels on a CUDA
GPU. `sinkslot.sinkslot_solve` also runs on CPU (or CUDA without Triton
installed) via a pure-torch fallback -- same algorithm, no fused kernels, so
noticeably slower, not a substitute for the reported throughput.

## Quick start

```bash
python run.py --dry-run                    # tiny grid from configs/base.py
python run.py --execute                    # run it -> output/quick
python -m gradient_flow.run                # the gradient-flow figure
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

## Layout

`torch-ext/sinkslot/solver.py` is the method: sliced-plan builder, Triton
kernels, v5 loop, the SinkSLOT-CUDA setup path (`_ot_1d_coo_batched_cuda`,
`sparse_sqeuclidean_cost`, int32-key `to_csr`), and the envelope-theorem
gradient (`slot_grad`, `plan_barycentric_sparse`). It sits in its own package
because it is a different algorithm from the dense fused kernel it builds on,
and is used outside benchmarking, by `gradient_flow/` below.

`gradient_flow/` produces Figure 3 and the two gradient-decomposition results
quoted in the appendix (the `fig:gradient_terms` figure and the `tab:fd_check`
table):

```bash
python -m gradient_flow.run          # Figure 3: blob -> crescent, SOT/EOT/SROT/SinkSLOT
python -m gradient_flow.term_norms   # appendix figure: split the gradient into two terms
python -m gradient_flow.finite_diff  # appendix table: is the dropped term really zero?
```

`along_flow.py`, `estimators.py`, `sweep_along_flow.py` and `config.py` are
shared infrastructure underneath those three (the trajectory/`three_gradients`
helpers), not standalone results.

`gradient_flow/appendix_checks/` holds the controls behind specific appendix
claims that don't have a directly-embedded figure of their own -- each answers
one worry about the result above (is the closed form an artefact? is the
cosine decay a discretisation artefact rather than the flow converging? does
the small-`L` wiggle in the sweep wash out as sampling noise? does the gap
actually change the flow you'd see?):

```bash
python -m gradient_flow.appendix_checks.stopping           # how early the inner solve can stop
python -m gradient_flow.appendix_checks.closed_form_check  # is the closed form an artefact?
python -m gradient_flow.appendix_checks.step_size          # is the cosine decay a discretisation artefact?
python -m gradient_flow.appendix_checks.projection_noise   # is the small-L wiggle sampling noise?
python -m gradient_flow.appendix_checks.flow_qualitative   # does the gap change the flow you see?
python -m gradient_flow.appendix_checks.figure             # 6-panel summary of the sweep above
```

Each module's docstring carries its own results table. `torch-ext/sinkslot/
solver.py` holds the envelope projection `∇_X SLOT_ε(X,Y) = 2 diag(a)(X - T_ε(X))`
(`slot_grad`), `vendor/sinkhorn_methods.py` the dense differentiable SOT/EOT/SROT
baselines, and `data/` the two densities (blob to crescent) sampled into point
clouds.

`torch-ext/flash_sinkhorn/` is the underlying package: `samples_loss.py`
(GeomLoss-compatible entry point), `sinkhorn_solvers.py`, `kernels/` (fused
Triton kernels), `_autograd.py` and `implicit_grad.py` (analytic gradients, no
backprop through iterations), `hvp.py` and `cg.py` (Hessian-vector products via
streaming CG), `c_transform.py`. Its `bench/` holds `bench_forward.py` (forward
sweep, every baseline adapter and its RMAE reference).

At the repo root: `configs/base.py` (`BenchConfig` plus a quick grid, not used
for any reported number), `run.py` (sweep driver, one subprocess per
measurement, appended to one CSV), `scripts/scalability.py`.

`scripts/memory_audit/` is a closed audit, not a reproduction path: it checked
whether the sweep's `gpu_memory_mb` column (used for `tab:accuracy_memory`) is
comparable across methods, and confirmed it is (`memory.md`'s status note has
the full reasoning). It doesn't produce anything the paper cites and isn't run
as part of reproducing any result above.

## Baselines

Built on [FlashSinkhorn](https://github.com/ot-triton-lab/flash-sinkhorn), and
compared against [SROT](https://github.com/khainb/SROT) (Nguyen),
[Spar-Sink](https://github.com/Mengyu8042/Spar-Sink) (Li, Yu, Li, Meng).

SROT and Spar-Sink are adapted from the authors' released implementations and
validated against them. Changes are confined to this harness: GPU, precision,
stopping rule and timing instrumentation are fixed across every method, so the
algorithm is the only difference. The update equations, reference coupling and
sampling scheme are the authors' own.
