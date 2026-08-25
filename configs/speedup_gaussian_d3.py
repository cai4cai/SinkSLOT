"""Speedup benchmark, part 2 of 3 -- Gaussian dataset at d=3, same grid/stop
policy as configs/speedup.py (eps/L/S all 8-point, max_iter=10000, marginal,
stop_tol=1e-6). See configs/speedup.py's docstring for how the 3 speedup files
relate -- this one exists only because a single BenchConfig can't mix dims
across datasets (dims applies uniformly to every dataset in cfg.datasets), so
this Gaussian-at-d=3 slice has to be its own config.

Its commands are appended into the SAME per-method output directories as
configs/speedup.py's 3 non-Gaussian datasets (both configs point --output-dir
at the same path per method), so each method's forward_all.csv ends up
covering all 4 dataset slices together.

Run with:

    python run.py --config speedup_gaussian_d3 --execute
"""

import math

from configs.base import BenchConfig

# Spar-Sink's own recipe (Li, Yu, Li, Meng, "Importance Sparsification for
# Sinkhorn Algorithm", JMLR): s = k * s0(n), s0(n) = 1e-3 * n * log(n)^4.
# At n=10,000, s0(n)~=71,962, giving s = [143924, 287848, 575695, 1151391] --
# comfortably inside the max_dense_size=10000 dense-kernel budget below.
_n = 10000
_s0 = 1e-3 * _n * (math.log(_n) ** 4)
_sparsink_s = [int(round(k * _s0)) for k in [5, 10, 15, 20]]

CONFIG = BenchConfig(

    sizes=[10000],
    dims=[3],

    eps_values=[
        0.001, 0.0019307, 0.0037276, 0.0071969,
        0.013895, 0.026827, 0.0517947, 0.1,
    ],
    n_iters=20000,

    stop_mode="marginal",
    max_iter=10000,
    stop_tol=1e-6,
    potential_tol=1e-6,
    check_every=10,

    warmup=0,
    rep=5,
    tf32=False,

    seeds=[0],
    datasets=["gaussian"],

    no_srot=False,
    srot_slices=[32, 64, 128, 256, 512, 1024, 2048, 4096],
    srot_delta=1e-8,

    no_sinkslot=True,

    no_sinkslotcuda=False,
    sinkslotcuda_slices=[32, 64, 128, 256, 512, 1024, 2048, 4096],

    no_sparsink=False,
    no_randsink=True,  # Rand-Sink dropped, no longer of interest
    sparsink_s=_sparsink_s,
    sparsink_replicates=50,

    no_ott=True,
    no_rmae_check=False,
    no_geomloss=True,
    no_flash_symmetric=True,
    no_flash_alternating=False,

    isolate=True,
    tensorized=False,
    max_dense_size=10000,

    output_dir="output/table1",
    dry_run=True,
)
