"""N-scaling probe: Flash (symmetric/alternating) and SinkSLOT (/-CUDA) only,
at a single eps=0.1, across N = 1k/2k/5k/20k/50k/100k. SROT and Spar-Sink/
Rand-Sink are NOT included (dense O(n*m) setup -- SROT's pi_SOT build and
Spar-Sink's dense probability matrix would dominate wall-clock at N=100k in a
way that's a setup-cost question, not the solve-speed-vs-N question this
config is asking).

Run with:

    python run.py --config config_nscale --execute

Single eps=0.1 for every N (not swept -- this config asks "how does solve
time/accuracy move with N," not "how does it move with eps," which the other
exp1 configs already cover at N=10,000 fixed). SinkSLOT/-CUDA's L is fixed at
256 (not swept) for the same reason -- one representative sparsity level,
matching the old SLOT-repo preliminary report's Experiment 2 convention
("L nearest to 256").

Same stop_mode=marginal, stop_tol=1e-4 as config_exp1_tol4.py -- launched
alongside it, so directly comparable.

max_dense_size raised to 100,000 (>= the largest N here) even though neither
Flash nor SinkSLOT/-CUDA are gated by it (only SROT/Spar-Sink/dense-GeomLoss
are, and those are off here) -- set generously just in case.
"""

from config import BenchConfig

CONFIG = BenchConfig(
    which="forward",

    sizes=[1000, 2000, 5000, 20000, 50000, 100000],
    dims=[3, 64],

    eps_values=[0.1],
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
    no_sparsink=True,

    no_sinkslot=False,
    sinkslot_slices=[256],

    no_sinkslotcuda=False,
    sinkslotcuda_slices=[256],

    no_ott=True,
    no_rmae_check=False,
    no_geomloss=True,
    no_flash_symmetric=False,
    no_flash_alternating=False,

    isolate=True,
    tensorized=False,
    max_dense_size=100000,

    output_dir="output/nscale",
    dry_run=True,
)
