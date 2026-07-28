"""Exp1-tol4-flash: Flash-only leg of config_exp1_tol4.py, 100-point eps sweep
(same range/point-count reasoning as every other exp1 flash companion this
session -- Flash has no L/s knob to widen instead).

Run with:

    python run.py --config config_exp1_tol4_flash --execute

Only difference from config_exp1_marginal_flash.py: stop_tol=1e-4 instead of
1e-6, matching config_exp1_tol4.py's main leg.
"""

from config import BenchConfig

_EPS_MIN, _EPS_MAX, _EPS_N = 0.001, 0.1, 100
_eps_values = [
    _EPS_MIN * (_EPS_MAX / _EPS_MIN) ** (i / (_EPS_N - 1))
    for i in range(_EPS_N)
]

CONFIG = BenchConfig(
    which="forward",

    sizes=[10000],
    dims=[3, 64],

    eps_values=_eps_values,
    n_iters=10000,

    stop_mode="marginal",
    max_iter=10000,
    stop_tol=1e-4,
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
    no_sparsink=True,

    no_ott=True,
    no_rmae_check=False,
    no_geomloss=True,
    no_flash_symmetric=False,
    no_flash_alternating=False,

    isolate=True,
    tensorized=False,
    max_dense_size=10000,

    output_dir="output/exp1_tol4_flash",
    dry_run=True,
)
