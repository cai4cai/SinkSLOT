"""Speedup benchmark, part 3 of 3 -- Gaussian only, d=64 only, "easy" high-eps
regime. N=M=10,000, marginal stopping, stop_tol=1e-6, max_iter=20000 (up from
configs/speedup.py's 10000 -- this sweep sits entirely in the eps range the
earlier convcheck_d64 probe showed reliably converges, so the higher budget
is just headroom, not expected to be needed). See configs/speedup.py's
docstring for how the 3 speedup files relate.

eps: 8-point log grid spanning exactly one decade, [0.1, 1] (both endpoints
exact -- lo=hi/10 over 7 steps), chosen so 0.1 and 1 are themselves grid
points rather than falling between them.

SROT and Flash-alternating already ran and completed against the ORIGINAL
sinkslotcuda_slices=[32..4096] grid (results in output/table1_gaussian_d64_h100/).
sinkslotcuda_slices was then doubled to [64..8192] and SinkSLOT-CUDA re-run
against just this new grid, after the sinkslot-vs-flash-alternating comparison
showed SinkSLOT-CUDA never reached a <=1% cost gap anywhere in this eps range --
more slices was the natural first thing to try. SROT was not re-run (this
change only concerns the SinkSLOT-CUDA vs Flash-alternating comparison), so
srot_slices below is left at its original value, documenting what SROT's
completed run actually used.

Same S (Spar-Sink density) grid as configs/speedup.py.

Run with:

    python run.py --config speedup_gaussian_d64 --execute
"""

from configs.base import BenchConfig

_NM = 10000 * 10000
_s_densities = [0.001, 0.0017487, 0.0030579, 0.0053472, 0.0093506, 0.0163512, 0.028593, 0.05]
_sparsink_s = [int(round(d * _NM)) for d in _s_densities]

CONFIG = BenchConfig(
    which="forward",

    sizes=[10000],
    dims=[64],

    eps_values=[
        0.1, 0.13895, 0.19307, 0.26827,
        0.372759, 0.517947, 0.719686, 1.0,
    ],
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
    srot_slices=[32, 64, 128, 256, 512, 1024, 2048, 4096],
    srot_delta=1e-8,

    no_sinkslot=True,

    no_sinkslotcuda=False,
    sinkslotcuda_slices=[64, 128, 256, 512, 1024, 2048, 4096, 8192],

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

    output_dir="output/table1_gaussian_d64",
    dry_run=True,
)
