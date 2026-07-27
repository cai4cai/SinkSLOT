"""ExpB-nscale-potential: same as config_expB_nscale_marginal.py, but
stop_mode=potential_linf. See that module's docstring for the full rationale
(no exact-OT reference at any N, full L=16..8192 sweep at every N, iters_run
tracking decoupled from rmae_check).

Run with:

    python run.py --config config_expB_nscale_potential --execute
"""

from config import BenchConfig

CONFIG = BenchConfig(
    which="forward",

    sizes=[1000, 2000, 5000, 20000, 50000, 100000],
    dims=[3, 64],

    eps_values=[0.1],
    n_iters=20000,

    stop_mode="potential_linf",
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

    no_srot=True,
    no_sparsink=True,

    no_sinkslot=False,
    sinkslot_slices=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],

    no_sinkslotcuda=False,
    sinkslotcuda_slices=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],

    no_ott=True,
    no_rmae_check=True,   # no exact-OT reference at any N -- see marginal sibling's docstring
    no_geomloss=True,
    no_flash_symmetric=False,
    no_flash_alternating=False,

    isolate=True,
    tensorized=False,
    max_dense_size=100000,

    output_dir="output/expB_nscale_potential",
    dry_run=True,
)
