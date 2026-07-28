"""ExpB-d64wide-marginal: d=64 only, re-run with a WIDER eps range [0.01, 1]
(10x the original [0.001, 0.1] range -- same 100x log-ratio, so this is
literally the old 10-point eps sequence times 10). SinkSLOT/-CUDA/SROT
(Flash's companion is config_expB_d64wide_marginal_flash.py, 100-pt eps
sweep over the same range). Same marginal stopping, stop_tol=1e-6,
max_iter=20000, L=16..8192 (10 pts) as config_expB_marginal.py -- only d
(fixed at 64, not swept) and the eps range differ.

Run with:

    python run.py --config config_expB_d64wide_marginal --execute
"""

from config import BenchConfig

CONFIG = BenchConfig(
    which="forward",

    sizes=[10000],
    dims=[64],

    eps_values=[
        0.01, 0.016681, 0.027826, 0.046416, 0.077426,
        0.129155, 0.215443, 0.359381, 0.599484, 1.0,
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

    no_sinkslot=False,
    sinkslot_slices=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],

    no_sinkslotcuda=False,
    sinkslotcuda_slices=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],

    no_sparsink=True,

    no_ott=True,
    no_rmae_check=False,
    no_geomloss=True,
    no_flash_symmetric=True,    # moved to config_expB_d64wide_marginal_flash.py
    no_flash_alternating=True,

    isolate=True,
    tensorized=False,
    max_dense_size=10000,

    output_dir="output/expB_d64wide_marginal",
    dry_run=True,
)
