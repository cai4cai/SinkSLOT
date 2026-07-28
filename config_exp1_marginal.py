"""Exp1-marginal: same grid as config_exp1.py, but stop_mode="marginal" instead of
"potential_linf".

Run with:

    python run.py --config config_exp1_marginal --execute

Why this config exists: config_exp1.py ported FlashSinkhorn's own native
potential_linf rule (max(|df|,|dg|) < stop_tol) verbatim to srot/sinkslot/
sinkslotcuda/spar_sink/rand_sink, at stop_tol=1e-6 (SLOT's own default). That
run showed near-universal non-convergence -- most SROT/SinkSLOT/SinkSLOT-CUDA
units hit max_iter=10000 without satisfying the tolerance (100% at d=64), and
Flash showed cost gaps swinging from -12% to +267% depending on eps, including
physically-impossible negative gaps at small eps (see the PR's investigation).
Root cause: SLOT never actually ran SROT/SinkSLOT under potential-change
stopping -- only marginal-violation stopping (stop_mode="marginal": total-
variation violation <= stop_tol AND |mass-1| <= mass_tol). potential_linf was
untested for those methods at this n and tol; marginal is the criterion
they're actually proven to converge well under.

This config swaps ONLY the stopping mode, keeping every other axis (eps, L,
seeds, rep, precision) identical to config_exp1.py, so the two runs are a
direct, controlled comparison of stopping criterion alone.

Two real bugs were found and fixed to make this config meaningful, not just a
config flag flip:

1. FlashSinkhorn itself had no "marginal" stopping option at all -- only its own
   native potential_linf check. Implemented natively in sinkhorn_solvers.py for
   both flash_symmetric and flash_alternating backends (same TV/mass_tol
   convention as SROT/SinkSLOT/Spar-Sink), so Flash is now directly comparable
   under this config, not left out.

2. flash_alternating's early stopping was never wired into the actual timed
   computation path at all: _autograd.py's _SinkhornCostFn.forward, for
   backend="alternating", never read config.threshold/config.inner_iterations
   before calling the solver -- so it always ran its full fixed iteration
   budget regardless of stop_mode, silently. This explains why exp1's
   flash_alternating rows showed almost-constant runtime across every eps
   tested. Fixed; verified on-cluster that stop_mode now measurably changes
   flash_alternating's runtime (up to ~7x faster at eps where marginal mode
   can actually trigger).

Also fixed (not directly required for this config, but discovered alongside
the above and worth having as row counts are re-checked this run): the
Spar-Sink/Rand-Sink CSV dispatch bug where a job launched with `--only
spar_sink` silently also ran every `rand_sink` row too (SPARSINK_METHODS was
always iterated in full regardless of --only). Not wrong data -- every row was
correctly tagged -- but it doubled compute and confused row-count accounting
in exp1/exp2. Now `--only spar_sink` produces ONLY spar_sink rows, so
launching this config's cluster jobs needs an actual, separate rand_sink job
(the previous "free ride" from spar_sink's job no longer happens).

See config_exp1.py's docstring for the full design rationale (grid, dataset,
n, L-sweep, seeds) -- unchanged here except stop_mode/stop_tol's meaning.
stop_tol=1e-6 is unchanged in VALUE from config_exp1.py, but means something
different now: a total-variation marginal-violation bound (range [0,4]), not a
potential-change bound -- this is the exact tolerance SLOT itself used
successfully in "marginal" mode, so it's the tolerance actually being tested
here, not an untested one reused by coincidence.
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
    n_iters=10000,  # vestigial under marginal (max_iter governs the loop instead)

    stop_mode="marginal",
    max_iter=10000,
    stop_tol=1e-6,      # TV marginal-violation bound now, not potential-change
    potential_tol=1e-6,
    mass_tol=1e-6,
    check_every=10,

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

    no_ott=True,                # no early-stopping hook -- dropped
    no_rmae_check=False,
    no_geomloss=True,           # no early-stopping hook -- dropped
    no_flash_symmetric=True,    # moved to config_exp1_marginal_flash.py -- 100-pt eps
    no_flash_alternating=True,  # sweep to match everyone else's row count; run it too

    isolate=True,
    tensorized=False,
    max_dense_size=10000,  # >= n so SROT/Spar-Sink/Rand-Sink aren't excluded at this size

    output_dir="output/exp1_marginal",
    dry_run=True,           # review the job list first; pass --execute to run
)
