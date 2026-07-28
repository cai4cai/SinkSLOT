"""Speedup benchmark, part 1 of 3 -- non-Gaussian datasets. N=M=10,000, d=2
(native, no padding), marginal stopping, stop_tol=1e-6, max_iter=10000 (down
from the first pass's 20000, to fit the slow methods within a 2h dev-queue
budget once split by dataset).

THIS FILE IS ONE OF THREE that together define the complete speedup
benchmark -- each is a separate BenchConfig only because a single BenchConfig
can't mix `dims` across datasets, not because the experiment itself is
different. Read all three together for the full picture:
  - configs/speedup.py            (this file) -- half_moon, 8gaussians,
    two_rings, all d=2.
  - configs/speedup_gaussian_d3.py -- Gaussian, d=3, same eps/L/S grid.
  - configs/speedup_gaussian_d64.py -- Gaussian, d=64, wider eps range [0.1, 1].
All three share the same solver policy (marginal, stop_tol=1e-6, same L/S
grids) and the same 5 methods below. Execution was additionally split into
many per-dataset/per-method SLURM jobs purely for cluster scheduling (2h dev
queue vs 20h t3 queue) -- that job splitting has no bearing on what the
experiment IS, which is exactly what these three CONFIGs specify.

Datasets in this file: half_moon, 8gaussians, two_rings (all natively 2D; see
sample_point_cloud() in bench_forward.py).

Grids cut from 10 to 8 points on every axis (eps, L, S) vs the first pass, to
reduce per-job unit count now that the slow methods (SROT/Spar-Sink) are
split one job per dataset:
  - eps: 8-point log grid over [0.001, 0.1] (was 10-point).
  - L (SROT, SinkSLOT-CUDA): 8-point log2 grid 32..4096 (was 16..8192).
  - S (Spar-Sink density): 8-point log grid over 0.1%..5% of N*M (was
    0.1%..80% -- capped lower because real timing showed density-driven
    kernel-construction cost, not iteration count, dominates runtime here,
    and grows steeply enough that higher densities risked hours per unit).

Methods, one SLURM job each (see table1_launch/gen_scripts.py):
  - SinkSLOT-CUDA only (no plain SinkSLOT) and SROT: eps x L grid.
  - FlashSinkhorn-alternating only (no symmetric): eps sweep only.
  - Spar-Sink: eps x S grid, sparsink_replicates=50. Rand-Sink dropped
    (no_randsink=True) -- no longer of interest.

Run with:

    python run.py --config speedup --execute
"""

from configs.base import BenchConfig

_NM = 10000 * 10000
_s_densities = [0.001, 0.0017487, 0.0030579, 0.0053472, 0.0093506, 0.0163512, 0.028593, 0.05]
_sparsink_s = [int(round(d * _NM)) for d in _s_densities]

CONFIG = BenchConfig(
    which="forward",

    sizes=[10000],
    dims=[2],

    eps_values=[
        0.001, 0.0019307, 0.0037276, 0.0071969,
        0.013895, 0.026827, 0.0517947, 0.1,
    ],
    n_iters=20000,  # vestigial under marginal (max_iter governs the loop instead)

    stop_mode="marginal",
    max_iter=10000,
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
    srot_slices=[32, 64, 128, 256, 512, 1024, 2048, 4096],
    srot_delta=1e-8,

    no_sinkslot=True,

    no_sinkslotcuda=False,
    sinkslotcuda_slices=[32, 64, 128, 256, 512, 1024, 2048, 4096],

    no_sparsink=False,
    no_randsink=True,
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
