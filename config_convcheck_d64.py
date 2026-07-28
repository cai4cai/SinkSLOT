"""Convcheck-d64: quick convergence probe. Gaussian, N=M=10,000, d=64,
marginal stopping, stop_tol=1e-6, max_iter=20000. 10-point eps sweep over
[0.01, 1] (same wide range/ratio as config_expB_d64wide_marginal.py) x 3
representative L values (128, 1024, 4096, matching the frontier figure's
current convention) for SinkSLOT-CUDA and SROT. Sized to finish within the
qos_gpu_h100-dev 2-hour cap.

Purpose: how many of these (eps, L) combinations actually converge within
20k iterations, at d=64 -- not a full report, just a converged/hit_max_iters
count per method.

Run with:

    python run.py --config config_convcheck_d64 --execute
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
    n_iters=20000,  # vestigial under marginal (max_iter governs the loop instead)

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
    srot_slices=[128, 1024, 4096],
    srot_delta=1e-8,

    no_sinkslot=True,

    no_sinkslotcuda=False,
    sinkslotcuda_slices=[128, 1024, 4096],

    no_sparsink=True,

    no_ott=True,
    no_rmae_check=False,
    no_geomloss=True,
    no_flash_symmetric=True,
    no_flash_alternating=True,

    isolate=True,
    tensorized=False,
    max_dense_size=10000,

    output_dir="output/convcheck_d64",
    dry_run=True,
)
