"""ExpC-potential: same grid as config_expC_marginal.py, but stop_mode=potential_linf
(Spar-Sink/Rand-Sink have no native "potential_linf" rule of their own, so this
falls back to the same max(|df|,|dg|) rule FlashSinkhorn/SROT/SinkSLOT use --
see bench_sparsink's stop_mode handling in bench_forward.py).

Run with:

    python run.py --config config_expC_potential --execute
"""

from config import BenchConfig

_NM = 10000 * 10000
_densities = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.80]
_sparsink_s = [int(round(d * _NM)) for d in _densities]

CONFIG = BenchConfig(
    which="forward",

    sizes=[10000],
    dims=[3, 64],

    eps_values=[
        0.001, 0.0016681, 0.00278256, 0.00464159, 0.00774264,
        0.0129155, 0.0215443, 0.0359381, 0.0599484, 0.1,
    ],
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
    no_sinkslot=True,
    no_sinkslotcuda=True,

    no_sparsink=False,
    sparsink_s=_sparsink_s,
    sparsink_replicates=50,

    no_ott=True,
    no_rmae_check=False,
    no_geomloss=True,
    no_flash_symmetric=True,
    no_flash_alternating=True,

    isolate=True,
    tensorized=False,
    max_dense_size=10000,

    output_dir="output/expC_potential",
    dry_run=True,
)
