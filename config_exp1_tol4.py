"""Exp1-tol4: same grid as config_exp1_marginal.py, but stop_tol=1e-4 (looser
than the 1e-6 used so far) and the L sweep extended one point further, to
8192 (2x the previous max of 4096).

Run with:

    python run.py --config config_exp1_tol4 --execute

Why: config_exp1_marginal.py already fixed the sum-vs-max bug in marginal-
violation stopping and confirmed it converges well under tol=1e-6 (SROT
~1,410 iters, SinkSLOT/-CUDA ~900-1000 iters at a representative eps=0.01,
n=10,000). This config asks the complementary question: at a looser,
1e-4 tolerance -- and with SinkSLOT/-CUDA/SROT allowed to sweep one L step
further than before -- how much does convergence speed and feasibility
change? tol=1e-4 also matches the independent post-hoc feasibility
threshold already used throughout the reporting (marg_viol < 1e-4), so
"converged" and "feasible" should track much more closely here than they
did at tol=1e-6 (see the "converged vs feasible" table in the report --
at tol=1e-6 many d=64 runs were feasible long before they were "converged").

Spar-Sink/Rand-Sink's own s sweep is left unchanged (2,000 -> 1,024,000);
only the L axis (SROT/SinkSLOT/SinkSLOT-CUDA) was asked to double.

Flash is covered by config_exp1_tol4_flash.py (same row-count-matching
reasoning as every other exp1 variant this session).

Same eps grid, seeds, rep, precision as config_exp1_marginal.py -- only
stop_tol and the L sweep's upper bound changed, so this is directly
comparable to that run.
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
    stop_tol=1e-4,      # looser than exp1_marginal's 1e-6
    potential_tol=1e-6,
    mass_tol=1e-6,      # unused for the stopping decision (dropped this session); harmless
    check_every=10,

    warmup=0,
    rep=5,
    tf32=False,

    seeds=[0],
    datasets=["gaussian"],

    no_srot=False,
    srot_slices=[8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
    srot_delta=1e-8,

    no_sinkslot=False,
    sinkslot_slices=[8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],

    no_sinkslotcuda=False,
    sinkslotcuda_slices=[8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],

    no_sparsink=False,
    sparsink_s=[2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000, 1024000],
    sparsink_replicates=50,

    no_ott=True,
    no_rmae_check=False,
    no_geomloss=True,
    no_flash_symmetric=True,    # moved to config_exp1_tol4_flash.py
    no_flash_alternating=True,

    isolate=True,
    tensorized=False,
    max_dense_size=10000,

    output_dir="output/exp1_tol4",
    dry_run=True,
)
