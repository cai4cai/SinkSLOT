"""Exp1: matched-policy comparison at a single controlled scale.

Reuses the ``BenchConfig`` schema from ``config.py``. Unlike ``config_paper.py``
(a scaling sweep across sizes, with dense O(n^2) baselines dropping out above
max_dense_size), this fixes n at a single size and sweeps eps and L finely, with
every method -- including SROT/Spar-Sink/Rand-Sink -- sharing the exact same
stopping rule, check frequency, threshold, and precision. Run with:

    python run.py --config config_exp1 --execute

Design:

- ``gaussian`` only (isotropic normal), d in {3, 64} -- a low- and high-dimensional
  case.
- n=10,000, fixed across every method/eps/L. ``max_dense_size`` is set to 10,000 so
  the dense O(n^2) baselines (SROT, Spar-Sink, Rand-Sink) are NOT excluded here --
  every method gets a row at this one size, unlike config_paper's scaling story.
- 5 seeds -- a real repeat of the full grid per seed (data only: x, y, a, b).
  Method-internal randomness (SROT/SinkSLOT's slice projections, Spar-Sink's kernel
  sampling) is untouched by this, so it stays independently/deterministically seeded
  regardless of which data seed is active -- only the underlying problem instance
  varies across seeds, not each method's own algorithmic randomness.
- eps: a 10-point log sweep from 1e-3 to 1e-1 (same range/style as the earlier
  3-point sweep in config_paper.py, just finer).
- L: a 10-point sweep, 8 -> 4096 (doubling each step), applied to SROT, SinkSLOT and
  SinkSLOT-CUDA. Spar-Sink/Rand-Sink's own knob (s, not L) is left at config_paper's
  values since a 10-point sweep wasn't requested for it.
- Stopping: "potential_linf" everywhere, not "fixed" -- max(|df|, |dg|) < stop_tol
  since the last check, capped at max_iter=10000. This is FlashSinkhorn's own native
  rule, ported verbatim to srot/sinkslot/sinkslotcuda/spar_sink/rand_sink (see the
  feat/potential-linf-stop-parity branch), so every retained method shares the
  identical stopping rule, not just a similar one.
- GeomLoss and OTT-JAX are DROPPED (no_geomloss=True, no_ott=True): neither has an
  early-stopping hook to plug "potential_linf" into (GeomLoss calls upstream
  geomloss.sinkhorn_loop directly, which has no threshold parameter at all; OTT is
  off by default anyway). Re-add them once/if that gap is closed.
- warmup=0 (none), tf32=False (strict fp32 everywhere, no TF32 -- so timing isn't a
  mix of two different arithmetic modes across methods).

Caveat carried over from verifying potential_linf (see the branch's PR description):
Spar-Sink/Rand-Sink's importance-sampled sparse kernel can have weakly-connected
components with a local contraction rate near 1, so a check_every that's too small
relative to that support's mixing timescale can satisfy stop_tol while still
meaningfully far from the true fixed point. check_every=10 here is the codebase's
existing default, carried over untuned for this specific grid -- after a first run,
check Spar-Sink/Rand-Sink's `converged`/`iters_run` columns: many rows hitting
max_iter suggests stop_tol is too strict; suspiciously fast convergence at very
small iters_run is worth cross-checking against a deep/fixed-iteration run before
trusting it.

Not yet wired in: cost_gap / barycentric_sym against a true unregularized exact-OT
reference (as opposed to rmae_pct against a converged entropic reference at the same
eps). This config only changes the sweep grid and stopping policy; the accuracy
metric is still whatever bench_forward.py currently reports.
"""

from config import BenchConfig

CONFIG = BenchConfig(
    which="forward",

    sizes=[10000],
    dims=[3, 64],

    eps_values=[
        0.001, 0.0016681, 0.00278256, 0.00464159, 0.00774264,
        0.0129155, 0.0215443, 0.0359381, 0.0599484, 0.1,
    ],
    n_iters=10000,  # vestigial under potential_linf (max_iter governs the loop instead)

    stop_mode="potential_linf",
    max_iter=10000,
    stop_tol=1e-4,
    potential_tol=1e-6,
    mass_tol=1e-6,
    check_every=10,  # see the Spar-Sink caveat above -- untuned for this grid yet

    warmup=0,
    rep=50,
    tf32=False,  # strict fp32 everywhere -- no method gets a TF32 speed advantage

    seeds=[0, 1, 2, 3, 4],
    datasets=["gaussian"],

    no_srot=False,
    srot_slices=[8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
    srot_delta=1e-8,

    no_sinkslot=False,
    sinkslot_slices=[8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],

    no_sinkslotcuda=False,
    sinkslotcuda_slices=[8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],

    no_sparsink=False,
    sparsink_s=[2000, 8000, 32000],  # config_paper's values; not part of the requested L sweep
    sparsink_replicates=50,

    no_ott=True,           # no early-stopping hook -- dropped
    no_rmae_check=False,
    no_geomloss=True,      # no early-stopping hook -- dropped
    no_flash_symmetric=False,
    no_flash_alternating=False,

    isolate=True,
    tensorized=False,
    max_dense_size=10000,  # >= n so SROT/Spar-Sink/Rand-Sink aren't excluded at this size

    output_dir="output/exp1",
    dry_run=True,           # review the job list first; pass --execute to run
)
