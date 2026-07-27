"""ExpC-marginal: Spar-Sink/Rand-Sink only, N=10,000, marginal stopping,
stop_tol=1e-6, max_iter=20000. 10-point eps sweep (same grid as every other
config this session) x a NEW 10-point sample-size sweep, this time
density-parameterized (S = density * N * M) rather than the old arbitrary
doubling sequence [2000..1024000] (which topped out at just ~1.02% density,
and whose bottom half [2000..64000] -- below ~0.128% density -- turned out to
be entirely useless: every replicate hit a structurally empty row/column, so
cost_gap_pct/iters_run never populated there; see the exp1_tol4 report
discussion).

Density grid: 0.1%, 0.2%, 0.5%, 1%, 2%, 5%, 10%, 20%, 40%, 80% of N*M
(=100,000,000 at N=M=10,000) -> S = 100k, 200k, 500k, 1M, 2M, 5M, 10M, 20M,
40M, 80M. Max chosen to MATCH SinkSLOT's own density at its max L=8192
(measured from existing data: nnz/(N*M) = 68.8% at d=3, 77.0% at d=64) --
80% brackets both, making this a directly comparable "same support density"
head-to-head rather than an arbitrary cutoff. The bottom of this grid
(0.1%-1%) revisits the transition zone at finer resolution than the old
sweep's coarse doubling did.

Run with:

    python run.py --config config_expC_marginal --execute

Companion: config_expC_potential.py (same grid, potential_linf mode).
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

    output_dir="output/expC_marginal",
    dry_run=True,
)
