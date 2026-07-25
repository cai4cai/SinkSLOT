"""Exp2-flash: the no-L/s-knob leg of exp2 (Flash x2 + GeomLoss), 100-pt eps sweep.

Companion to config_exp2.py, same relationship as config_exp1_flash.py is to
config_exp1.py: FlashSinkhorn (both backends) and GeomLoss have no L/s-style
sweep parameter of their own, so they can't reach the same 200-rows-per-method
count as SROT/SinkSLOT/-CUDA (10 eps x 10 L) or Spar-Sink/Rand-Sink (10 eps x
10 s) without widening their own eps axis instead. Same eps range as
config_exp2.py (1e-3 to 1e-1), 100 points instead of 10 -- and the same range/
point-count as config_exp1_flash.py, so exp1's and exp2's Flash/GeomLoss rows
sit at identical eps values too (every 11th of these 100 lands on the 10-point
sweep exp2.py's other 5 methods use, exactly as verified for exp1).

Run with:

    python run.py --config config_exp2_flash --execute

Separate output_dir from config_exp2.py's ("output/exp2") for the same reason
as exp1's split: run.py clears its target dir's CSVs at the start of every
sweep, so pointing both at the same dir would let the second run silently wipe
the first's results. Concatenate the two CSVs afterward (every row is self-
describing by ``method``).

stop_mode="fixed", n_iters=2000: applies identically to all three methods here.
GeomLoss always ran fixed-iteration only (see config_exp2.py's docstring for why
it can't get early stopping at all); Flash's native potential-change stopping is
simply turned off here, matching exp2's whole point (no early stopping anywhere,
sidestepping the SROT/SinkSLOT criterion-mismatch question that's still open for
exp1).
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
    n_iters=2000,

    stop_mode="fixed",  # no early stopping -- exactly n_iters for every method

    warmup=0,
    rep=5,
    tf32=False,  # strict fp32 everywhere -- no method gets a TF32 speed advantage

    seeds=[0],  # matches config_exp2.py's current scope
    datasets=["gaussian"],

    no_srot=True,
    no_sinkslot=True,
    no_sinkslotcuda=True,
    no_sparsink=True,

    no_ott=True,             # still dropped -- separate RNG, not comparable
    no_rmae_check=False,
    no_geomloss=False,       # back in, unlike exp1_flash -- fixed mode is what let it return
    no_flash_symmetric=False,
    no_flash_alternating=False,

    isolate=True,
    tensorized=False,
    max_dense_size=10000,

    output_dir="output/exp2_flash",
    dry_run=True,             # review the job list first; pass --execute to run
)
