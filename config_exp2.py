"""Exp2: exp1's grid, fixed iterations instead of early stopping, GeomLoss back in.

Identical to config_exp1.py in every axis EXCEPT the stopping policy: n, d, eps,
L, s, seeds, precision, warmup, rep are all unchanged. Run with:

    python run.py --config config_exp2 --execute

This is exp2's leg for srot/sinkslot/sinkslotcuda/spar_sink/rand_sink (flash and
geomloss are OFF here, same reason as exp1: no L/s-style sweep knob of their own,
so they can't reach the same 200-row count without their OWN eps sweep at a
different point count). Covered separately by ``config_exp2_flash.py`` -- run
BOTH for the complete exp2 grid, then concatenate CSVs (separate output_dir's on
purpose, same reasoning as exp1 -- see config_exp1.py's docstring).

Why fixed iterations instead of potential_linf:

- Resolves the stopping-criterion mismatch that's still open for exp1 (see that
  PR's "Known issue"): SROT/SinkSLOT never ran under potential-change stopping in
  the working SLOT repo, only marginal violation, and empirically don't converge
  well under potential-change at n=10,000. Fixed iterations sidesteps the question
  entirely -- nobody stops early, so there's no criterion-mismatch to resolve.
- This is also literally the ORIGINAL "fixed" protocol from config_paper.py:
  "exactly n_iters per method, for a fair per-iteration throughput comparison."
  exp1 deliberately moved away from it (that's the whole point of the
  potential-linf branch); exp2 deliberately moves back, to have both readings
  side by side.
- GeomLoss can rejoin the comparison for exactly this reason: it can't be given
  early stopping at all (see below), so it was dropped from exp1, but it was
  ALWAYS fixed-iteration-only regardless of stop_mode -- under exp2's fixed-
  everywhere policy, that's no longer a special case, it's what everyone does.
  Needed real work to re-include properly, not just a config flag: GeomLoss
  never had cost_gap_pct/barycentric_sym/mass/marg_viol wired in (only rmae_pct,
  the metric exp1/exp2 replaced), and bench_geomloss_online was actually BROKEN
  on the cluster (imported from geomloss.sinkhorn_divergence/sinkhorn_samples,
  which don't exist in the installed geomloss 0.3.1 -- moved to
  geomloss._legacy.* in that version). Both fixed on this branch and verified on
  an A100 (see the PR description).
- Why GeomLoss can't get early stopping even in principle (not just "not done
  yet"): bench_geomloss_online calls geomloss's own sinkhorn_loop directly --
  upstream, unmodified library code, not something this codebase owns the way it
  owns _srot_sinkhorn/_run_v5/_sparsink_sinkhorn. sinkhorn_loop takes a fixed-
  length eps_list and loops over it with no threshold/check_every parameter
  anywhere in its signature -- there's no hook to extend. Adding early stopping
  would mean not calling sinkhorn_loop at all: copying its per-iteration update
  logic out and reimplementing a modified version with a convergence check added,
  i.e. forking a piece of the upstream library rather than extending an existing
  extension point (which is exactly what potential_linf did for SROT/SinkSLOT/
  Spar-Sink, since those loops already live in this codebase).

n_iters=2000: matches n_iters/max_iter's rough order of magnitude in exp1
(max_iter=10,000, but most methods there hit that ceiling without converging --
2,000 fixed iterations is a deliberately chosen, affordable budget for a first
pass, not an attempt to match exp1's iteration count exactly). Applies
identically to every method: FlashSinkhorn, SROT/SinkSLOT/SinkSLOT-CUDA/
Spar-Sink/Rand-Sink under stop_mode="fixed", and GeomLoss (which only ever runs
n_iters regardless of stop_mode). iters_run == n_iters and converged/
hit_max_iters == None for every row here -- there is no stopping check to report
on.

cost_gap_pct/barycentric_sym/mass/marg_viol are computed exactly as in exp1 (same
exact-OT reference, same formulas, same achieved-marginals fix) -- comparable
across exp1 and exp2, not just within exp2.
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
    n_iters=2000,

    stop_mode="fixed",  # no early stopping -- exactly n_iters for every method

    warmup=0,
    rep=5,
    tf32=False,  # strict fp32 everywhere -- no method gets a TF32 speed advantage

    seeds=[0],  # matches config_exp1.py's current scope
    datasets=["gaussian"],

    no_srot=False,
    srot_slices=[8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
    srot_delta=1e-8,

    no_sinkslot=False,
    sinkslot_slices=[8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],

    no_sinkslotcuda=False,
    sinkslotcuda_slices=[8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],

    no_sparsink=False,
    sparsink_s=[2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000, 1024000],
    sparsink_replicates=50,

    no_ott=True,                # OTT still dropped -- separate RNG, not comparable
    no_rmae_check=False,
    no_geomloss=True,           # moved to config_exp2_flash.py -- 100-pt eps sweep
    no_flash_symmetric=True,    # to match everyone else's row count; run that config too
    no_flash_alternating=True,

    isolate=True,
    tensorized=False,
    max_dense_size=10000,  # >= n so SROT/Spar-Sink/Rand-Sink aren't excluded at this size

    output_dir="output/exp2",
    dry_run=True,           # review the job list first; pass --execute to run
)
