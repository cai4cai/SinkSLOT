"""Publication sweep configuration.

The full grid for the paper's tables and figures. Reuses the ``BenchConfig``
schema from ``config.py`` (see its docstring for every field); this file only
pins the values. Run with:

    python run.py --config config_paper --execute

This is a large, cluster-scale sweep (thousands of units, hours on one GPU), not
a smoke test. Every run is a separate subprocess and results append to one CSV,
so it is resumable-friendly and survives individual failures.

Design of the grid:

- ``sizes`` span two regimes. Up to ``max_dense_size`` (4096) every method runs,
  for the accuracy comparison. Above it the dense baselines (SROT, Spar-Sink,
  Rand-Sink, tensorized GeomLoss) drop out and only the streaming/sparse methods
  (FlashSinkhorn, GeomLoss online, SinkSLOT) remain, for the scaling and memory
  story where the O(N^2) baselines OOM.
- ``sinkslot_slices`` follow the SLOT paper's memory sweep (L = 32/100/256/512),
  so the O(L(N+M)) growth is visible.
- ``tensorized`` is on so the dense O(N^2) GeomLoss baseline is present as the
  memory contrast (it OOMs beyond max_dense_size, which is the point).
- ``n_iters`` is high enough that the dense methods are near-converged, so RMAE
  reflects the regularisation/approximation gap rather than truncation.

Caveat: ``sparsink_s`` is absolute here. The Spar-Sink paper scales it as
s = {2,4,8,16} * s0(n), s0(n) = 1e-3 * n * log^4(n), which the flat schema cannot
express per-n; the values below bracket that range at the mid sizes. (Our
Spar-Sink port also currently disagrees with the upstream reference by ~10x on
RMAE -- see analysis.md -- so treat those rows as provisional.)
"""

from config import BenchConfig

CONFIG = BenchConfig(
    which="forward",

    sizes=[512, 1024, 2048, 4096, 8192, 16384],
    dims=[8, 64],

    eps_values=[0.1, 0.01, 0.001],
    n_iters=100,
    warmup=10,
    rep=50,
    tf32=True,

    datasets=["gaussian", "8gaussians"],

    no_srot=False,
    srot_slices=[10, 50, 100],
    srot_delta=1e-8,

    no_sparsink=False,
    sparsink_s=[2000, 8000, 32000],
    sparsink_replicates=50,

    no_ott=True,          # flip to False on a host with JAX/OTT installed
    no_rmae_check=False,
    no_geomloss=False,
    no_flash_symmetric=False,
    no_flash_alternating=False,

    no_sinkslot=False,
    sinkslot_slices=[32, 100, 256, 512],

    isolate=True,
    tensorized=True,
    max_dense_size=4096,

    output_dir="output/paper",
    dry_run=True,         # review the job list first; pass --execute to run
)
