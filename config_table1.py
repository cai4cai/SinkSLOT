"""Table1: non-Gaussian benchmark. N=M=10,000, d=2 (native, no padding), marginal
stopping, stop_tol=1e-6, max_iter=20000 -- same stop policy as config_expB_marginal.py.

Datasets: half_moon, 8gaussians, two_rings (all natively 2D; see
sample_point_cloud() in bench_forward.py).

Methods, one SLURM job each (see table1_launch/gen_scripts.py):
  - SinkSLOT-CUDA only (no plain SinkSLOT) and SROT: 10-point eps sweep x
    10-point L sweep (16..8192), same grid as config_expB_marginal.py.
  - FlashSinkhorn-alternating only (no symmetric): the SAME 10-point eps
    sweep (not the expanded 100-point grid used in the Flash companion
    configs elsewhere this session -- there's no L to match here).
  - Spar-Sink and Rand-Sink: the same 10-point eps sweep x the same
    10-point density-parameterized sample-size sweep as config_expC_marginal.py
    (density = 0.1%..80% of N*M, sparsink_replicates=50).

Run with:

    python run.py --config config_table1 --execute
"""

from config import BenchConfig

_NM = 10000 * 10000
_densities = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.80]
_sparsink_s = [int(round(d * _NM)) for d in _densities]

CONFIG = BenchConfig(
    which="forward",

    sizes=[10000],
    dims=[2],

    eps_values=[
        0.001, 0.0016681, 0.00278256, 0.00464159, 0.00774264,
        0.0129155, 0.0215443, 0.0359381, 0.0599484, 0.1,
    ],
    n_iters=20000,  # vestigial under marginal (max_iter governs the loop instead)

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
    datasets=["half_moon", "8gaussians", "two_rings"],

    no_srot=False,
    srot_slices=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
    srot_delta=1e-8,

    no_sinkslot=True,

    no_sinkslotcuda=False,
    sinkslotcuda_slices=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],

    no_sparsink=False,
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

    output_dir="output/table1",
    dry_run=True,
)
