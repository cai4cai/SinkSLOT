"""Exp1-flash: Flash-only leg of exp1, with a 100-point eps sweep instead of the
main config's 10-point one, so Flash's row count matches the other 6 methods'.

Flash has no L/s-style sweep parameter of its own (unlike SROT/SinkSLOT/-CUDA's L
or Spar-Sink/Rand-Sink's s), so there's no other axis to bump to reach the same
200 rows/method/dim-pair as everyone else in config_exp1.py (10 eps x 10 L-or-s
= 100 combos/dim x 2 dims = 200). Widening Flash's own eps axis 10x (100 eps x
2 dims = 200) is the only way to match row count without inventing a parameter
Flash doesn't have. Same eps range as config_exp1.py (1e-3 to 1e-1) -- only the
point count differs.

Run with:

    python run.py --config config_exp1_flash --execute

Separate output_dir from config_exp1.py's ("output/exp1"): run.py clears its
output_dir's CSVs at the start of every sweep, so running this after
config_exp1 (or vice versa) into the SAME dir would silently wipe the other's
results. Merge the two CSVs afterward if a single combined table is wanted --
every row is self-describing (method, dataset, tf32, etc. columns), so a
straight concatenation is safe.
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
    n_iters=10000,  # vestigial under potential_linf (max_iter governs the loop instead)

    stop_mode="potential_linf",
    max_iter=10000,
    stop_tol=1e-6,
    potential_tol=1e-6,
    mass_tol=1e-6,
    check_every=10,

    warmup=0,
    rep=5,
    tf32=False,  # strict fp32 everywhere -- no method gets a TF32 speed advantage

    seeds=[0],  # matches config_exp1.py's current scope
    datasets=["gaussian"],

    no_srot=True,
    no_sinkslot=True,
    no_sinkslotcuda=True,
    no_sparsink=True,

    no_ott=True,           # no early-stopping hook -- dropped
    no_rmae_check=False,
    no_geomloss=True,      # no early-stopping hook -- dropped
    no_flash_symmetric=False,
    no_flash_alternating=False,

    isolate=True,
    tensorized=False,
    max_dense_size=10000,

    output_dir="output/exp1_flash",
    dry_run=True,           # review the job list first; pass --execute to run
)
