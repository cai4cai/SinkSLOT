"""ExpB-nscale-marginal: N-scaling leg. Flash (symmetric/alternating) and
SinkSLOT/-CUDA only, marginal stopping, stop_tol=1e-6, max_iter=20000, eps
fixed at 0.1, across N = 1k/2k/5k/20k/50k/100k. SinkSLOT/-CUDA sweep the full
L=16..8192 grid (10 points) at every N -- unlike the earlier config_nscale.py,
which fixed L=256.

no_rmae_check=True: NO exact-OT reference computed at all, at any N (not even
1k/2k/5k, which would fit under _EXACT_OT_MAX_N=10000) -- this leg is purely a
speed/iteration-count-vs-N probe, not an accuracy one (that question is already
covered by the N=10,000 config_expB_marginal.py leg). This also sidesteps the
CPU-side dense (n,m) cost-matrix OOM in _cached_exact_ot_reference that hit
every method uniformly at N>=50,000 in the original nscale run.

iters_run/converged/hit_max_iters still populate for every method regardless of
no_rmae_check: SinkSLOT/-CUDA already track them unconditionally in their timed
solve, and Flash's untimed potentials-only call (added this session, decoupled
from the cost_gap/rmae_check gate specifically so this leg wouldn't lose
iteration tracking) is O(n), not O(n*m), so it's safe even at N=100,000.

Run with:

    python run.py --config config_expB_nscale_marginal --execute
"""

from config import BenchConfig

CONFIG = BenchConfig(
    which="forward",

    sizes=[1000, 2000, 5000, 20000, 50000, 100000],
    dims=[3, 64],

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

    no_srot=True,
    no_sparsink=True,

    no_sinkslot=False,
    sinkslot_slices=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],

    no_sinkslotcuda=False,
    sinkslotcuda_slices=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],

    no_ott=True,
    no_rmae_check=True,   # no exact-OT reference at any N -- see module docstring
    no_geomloss=True,
    no_flash_symmetric=False,
    no_flash_alternating=False,

    isolate=True,
    tensorized=False,
    max_dense_size=100000,

    output_dir="output/expB_nscale_marginal",
    dry_run=True,
)
