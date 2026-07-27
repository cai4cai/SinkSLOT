"""ExpB-dscale-marginal: dimension-scaling probe. N=10,000 fixed, eps=0.1
fixed, d swept 4->1024 (doubling: 4,8,16,32,64,128,256,512,1024 -- 9 points),
L swept 16..8192 (10 pts) for SinkSLOT/-CUDA/SROT. Flash (both backends)
included directly in this same config since eps isn't being swept broadly
here (single value), unlike the main expB leg's row-count-driven split into
_flash companions -- no such split needed when there's only one eps.

Same marginal stopping, stop_tol=1e-6, max_iter=20000 as config_expB_marginal.py.
d=1024 matches the FlashSinkhorn paper's own reported memory benchmark point,
motivating this sweep.

Run with:

    python run.py --config config_expB_dscale_marginal --execute
"""

from config import BenchConfig

CONFIG = BenchConfig(
    which="forward",

    sizes=[10000],
    dims=[4, 8, 16, 32, 64, 128, 256, 512, 1024],

    eps_values=[0.1],
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
    no_flash_symmetric=False,
    no_flash_alternating=False,

    isolate=True,
    tensorized=False,
    max_dense_size=10000,

    output_dir="output/expB_dscale_marginal",
    dry_run=True,
)
