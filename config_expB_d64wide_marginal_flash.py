"""ExpB-d64wide-marginal-flash: Flash-only leg of config_expB_d64wide_marginal.py,
100-point eps sweep over the same [0.01, 1] range.

Run with:

    python run.py --config config_expB_d64wide_marginal_flash --execute
"""

from config import BenchConfig

_EPS_MIN, _EPS_MAX, _EPS_N = 0.01, 1.0, 100
_eps_values = [
    _EPS_MIN * (_EPS_MAX / _EPS_MIN) ** (i / (_EPS_N - 1))
    for i in range(_EPS_N)
]

CONFIG = BenchConfig(
    which="forward",

    sizes=[10000],
    dims=[64],

    eps_values=_eps_values,
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

    output_dir="output/expB_d64wide_marginal_flash",
    dry_run=True,
)
