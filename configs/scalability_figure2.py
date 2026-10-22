"""The paper's scalability figure: every method at one fixed eps = 0.1 (Gaussian data), on the
N-sweep at d = 64 (N in N_SWEEP) and the d-sweep at N = 10,000 (d in D_SWEEP), 5 seeds.

Paper: main-text Figure 2 (runtime) and the appendix memory figure (peak memory of the same runs).
GeomLoss online is run but not plotted.

Methods and parameters, as in the figure:
  SinkSLOT-CUDA alternating at L = 0.1 N and L = 1000; SROT at L = 0.1 N; Spar-Sink at k = 512
  (s = k * s0(N)); FlashSinkhorn alternating with TF32 matrix products; GeomLoss online; GeomLoss
  multiscale with K-means (about 250 points per cluster, truncate 1). Symmetric variants, OTT and
  Rand-Sink are off. Stop rule, damping (0.9), warmup and repetitions as configs/scalability.py,
  with max_iter 5e5. The d-sweep also computes the exact-OT cost gap (N = 10,000).

Run with run.py like the other configs:

    python run.py --config scalability_figure2 --count
    python run.py --config scalability_figure2 --execute --num-shards K --shard-idx k
    python run.py --config scalability_figure2 --merge
"""

import dataclasses
import math
import os
from typing import List

from configs import scalability as S
from configs.base import BenchConfig

EPS = 0.1
SPARSINK_K = 512


def _figure_config(n: int, d: int, dsweep: bool) -> BenchConfig:
    s0 = 1e-3 * n * math.log(n) ** 4
    L_frac = int(round(0.1 * n))
    return dataclasses.replace(
        S._config(n, d), problems=[("gaussian", d, [EPS])],
        tf32=False, flash_tf32=[True], no_flash_symmetric=True, no_flash_alternating=False,
        no_sinkslot=True, no_sinkslotcuda=False, no_sinkslotcuda_symmetric=True,
        sinkslotcuda_slices=sorted({L_frac, 1000}),
        no_srot=False, srot_slices=[L_frac],
        no_sparsink=False, no_randsink=True, sparsink_s=[int(round(SPARSINK_K * s0))],
        no_geomloss=False, no_geomloss_multiscale=False, geomloss_clustering="kmeans",
        geomloss_points_per_cluster=250, geomloss_truncate=1.0,
        no_rmae_check=not dsweep,
        output_dir=os.environ.get("SCAL_OUTPUT_DIR") or "output/scalability_figure2")


def build_configs() -> List[BenchConfig]:
    return ([_figure_config(n, 64, dsweep=False) for n in S.N_SWEEP]
            + [_figure_config(S.N_DSWEEP, d, dsweep=True) for d in S.D_SWEEP if d != 64])


def __getattr__(name):
    if name == "CONFIGS":
        return build_configs()
    raise AttributeError(name)
