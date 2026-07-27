"""Exp1-marginal-flash: Flash-only leg of config_exp1_marginal.py, with a
100-point eps sweep instead of the main config's 10-point one, so Flash's row
count matches the other 5 methods' (same reasoning as config_exp1_flash.py --
Flash has no L/s-style sweep knob of its own to widen instead).

Run with:

    python run.py --config config_exp1_marginal_flash --execute

Only real difference from config_exp1_flash.py: stop_mode="marginal" instead
of "potential_linf". This is meaningful now (not just a flag flip) because
FlashSinkhorn previously had no marginal-violation stopping option at all --
implemented natively for both backends in sinkhorn_solvers.py as part of this
config's PR, alongside a real bug fix (flash_alternating's early stopping was
never wired into the timed computation path -- see config_exp1_marginal.py's
docstring for the full story). Both flash_symmetric and flash_alternating are
expected to behave meaningfully differently here than under config_exp1_flash,
not just relabeled.

Same eps range as config_exp1_flash.py (1e-3 to 1e-1, 100 points) -- only the
point count/stop_mode differ from config_exp1_marginal.py's 10-point sweep.

Separate output_dir from config_exp1_marginal.py's ("output/exp1_marginal"):
run.py clears its output_dir's CSVs at the start of every sweep, so running
this into the same dir would silently wipe the other's results. Concatenate
the two CSVs afterward for the full grid -- every row is self-describing.
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
    n_iters=10000,  # vestigial under marginal (max_iter governs the loop instead)

    stop_mode="marginal",
    max_iter=10000,
    stop_tol=1e-6,      # TV marginal-violation bound now, not potential-change
    potential_tol=1e-6,
    mass_tol=1e-6,
    check_every=10,

    warmup=0,
    rep=5,
    tf32=False,  # strict fp32 everywhere -- no method gets a TF32 speed advantage

    seeds=[0],  # matches config_exp1_marginal.py's current scope
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

    output_dir="output/exp1_marginal_flash",
    dry_run=True,           # review the job list first; pass --execute to run
)
