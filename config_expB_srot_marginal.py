"""ExpB-SROT-marginal: SROT only, matching config_expB_marginal.py's grid
exactly so it's directly comparable in the expB report -- marginal stopping,
stop_tol=1e-6, max_iter=20000 (doubled from the earlier config_exp1_marginal's
10000), N=10,000, d=3/64, the same 10-point eps sweep, L swept 16..8192
(10 points, same range as SinkSLOT/-CUDA in expB).

Run with:

    python run.py --config config_expB_srot_marginal --execute
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
    n_iters=20000,

    stop_mode="marginal",
    max_iter=20000,
    stop_tol=1e-6,
    potential_tol=1e-6,
    mass_tol=1e-6,
    check_every=10,

    warmup=0,
    rep=5,
    tf32=False,

    seeds=[0],
    datasets=["gaussian"],

    no_srot=False,
    srot_slices=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
    srot_delta=1e-8,

    no_sinkslot=True,
    no_sinkslotcuda=True,
    no_sparsink=True,

    no_ott=True,
    no_rmae_check=False,
    no_geomloss=True,
    no_flash_symmetric=True,
    no_flash_alternating=True,

    isolate=True,
    tensorized=False,
    max_dense_size=10000,

    output_dir="output/expB_srot_marginal",
    dry_run=True,
)
