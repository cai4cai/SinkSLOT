"""ExpB-marginal: FlashSinkhorn (Flash-only companion is config_expB_marginal_flash.py)
and SinkSLOT/-CUDA only, N=10,000, marginal stopping, stop_tol=1e-6 (max-based
marg_viol -- confirmed via _marg_viol/viol in sinkhorn_solvers.py/sinkslot.py,
both use .abs().max(), not .abs().sum()), max_iter=20000 (2x the earlier
exp1_marginal's 10000). L sweep widened to 16..8192 (10 points, was 8..4096).

Run with:

    python run.py --config config_expB_marginal --execute

Companion configs: config_expB_potential.py (same grid, potential_linf mode),
config_expB_marginal_flash.py / config_expB_potential_flash.py (Flash's 100-pt
eps sweep), config_expB_nscale_marginal.py / config_expB_nscale_potential.py
(N-scaling leg, no exact-OT reference).
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
    n_iters=20000,  # vestigial under marginal (max_iter governs the loop instead)

    stop_mode="marginal",
    max_iter=20000,
    stop_tol=1e-6,
    potential_tol=1e-6,
    mass_tol=1e-6,      # unused for the stopping decision; harmless, kept for the diagnostic column
    check_every=10,

    warmup=0,
    rep=5,
    tf32=False,

    seeds=[0],
    datasets=["gaussian"],

    no_srot=True,
    no_sparsink=True,

    no_sinkslot=False,
    sinkslot_slices=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],

    no_sinkslotcuda=False,
    sinkslotcuda_slices=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],

    no_ott=True,
    no_rmae_check=False,
    no_geomloss=True,
    no_flash_symmetric=True,    # moved to config_expB_marginal_flash.py
    no_flash_alternating=True,

    isolate=True,
    tensorized=False,
    max_dense_size=10000,

    output_dir="output/expB_marginal",
    dry_run=True,
)
