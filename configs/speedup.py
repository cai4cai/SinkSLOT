"""Speedup benchmark: time to a potential-change tolerance at N=M=10,000.

Stop rule: max(|df|, |dg|) < 1e-5 * median(C) between checkpoints (tol/2 for the
alpha=0.5 damped methods).

One config covers all five problem slices: half_moon, 8gaussians and two_rings
at d=2, and gaussian at d=3 and d=64. Each slice gets its own eps grid, spanning
the 1% and 10% cost-gap crossings (EPS_CROSSINGS), shared by every method.
median(C) is the lower median of the squared Euclidean cost on the seed-0
instance (MEDIAN_C, printed by scripts/speedup_prepare.py) and sets the tolerance.

Methods: SROT, SinkSLOT-CUDA and SinkSLOT-CUDA-symmetric over L; FlashSinkhorn
alternating and symmetric, each in strict FP32 and TF32; GeomLoss online;
Spar-Sink (one kernel draw) over s = k * s0(n), s0(n) = 1e-3 * n * ln(n)^4,
k in {1, 2, 4, ..., 128}. Every other method runs in strict FP32.

Run with:

    python run.py --config speedup --dry-run
    python run.py --config speedup --execute --num-shards 210 --shard-idx 0
"""

import math
from typing import Dict, Optional, Tuple

import numpy as np

from configs.base import BenchConfig

N = 10000

# (dataset, d) -> lower median of the squared Euclidean cost on the seed-0
# instance, from scripts/speedup_prepare.py (fp32, H100).
MEDIAN_C: Dict[Tuple[str, int], Optional[float]] = {
    ("half_moon", 2): 1.9413249492645264,
    ("8gaussians", 2): 7.766776084899902,
    ("two_rings", 2): 4.999881267547607,
    ("gaussian", 3): 4.734315872192383,
    ("gaussian", 64): 126.7506103515625,
}

# (dataset, d) -> (eps where FlashSinkhorn-alternating fp32 reaches a 1% cost gap,
# eps where it reaches 10%), interpolated from scripts/speedup_calibrate_eps.py
# (seed 0). Each slice sweeps 10 log-spaced eps from eps_1% / 2 to 2 * eps_10%, the
# same grid for every method.
EPS_CROSSINGS: Dict[Tuple[str, int], Tuple[float, float]] = {
    ("half_moon", 2): (0.02325, 0.4669),
    ("8gaussians", 2): (0.003289, 0.03157),
    ("two_rings", 2): (0.01537, 0.1917),
    ("gaussian", 3): (0.002494, 0.008904),
    ("gaussian", 64): (1.477, 3.774),
}
EPS_POINTS = 10
# Stop tolerance relative to the cost scale (as OTT's scale_cost="median"): the
# potentials grow with C, and an absolute 1e-6 falls below one fp32 ulp of them
# once |f| exceeds ~8, so the rule could never fire on the larger-cost slices.
REL_TOL = 1e-5
L_VALUES = [25, 50, 100, 250, 500, 1000, 2500, 5000]

_s0 = 1e-3 * N * (math.log(N) ** 4)
SPARSINK_S = [int(round(k * _s0)) for k in (1, 2, 4, 8, 16, 32, 64, 128)]


def eps_grid(eps_1pct: float, eps_10pct: float):
    return [float(f"{e:.4g}") for e in np.geomspace(eps_1pct / 2, 2 * eps_10pct, EPS_POINTS)]


def build_config(median_c: Dict[Tuple[str, int], Optional[float]] = MEDIAN_C) -> BenchConfig:
    missing = [key for key, value in median_c.items() if value is None]
    if missing:
        raise ValueError(
            f"configs/speedup.py: MEDIAN_C has no value for {missing}. Run "
            "scripts/speedup_prepare.py on a GPU and paste the dict it prints.")
    return BenchConfig(
        sizes=[N],
        dims=sorted({d for _, d in median_c}),
        problems=[(dataset, d, eps_grid(*EPS_CROSSINGS[(dataset, d)])) for (dataset, d) in median_c],
        n_iters=20000,

        stop_mode="potential",
        max_iter=20000,
        stop_tol=REL_TOL,
        stop_tol_by_problem={key: REL_TOL * median for key, median in median_c.items()},
        check_every=5,

        warmup=1,
        warmup_iters=10,
        rep=5,
        tf32=False,
        flash_tf32=[False, True],

        seeds=[0, 1, 2, 3, 4],

        no_srot=False,
        srot_slices=L_VALUES,
        srot_delta=1e-8,

        no_sinkslot=True,

        no_sinkslotcuda=False,
        no_sinkslotcuda_symmetric=False,
        sinkslotcuda_slices=L_VALUES,

        no_sparsink=False,
        no_randsink=True,
        sparsink_s=SPARSINK_S,
        sparsink_replicates=1,

        no_ott=True,
        no_rmae_check=False,
        no_geomloss=False,
        no_flash_symmetric=False,
        no_flash_alternating=False,

        isolate=True,
        tensorized=False,
        max_dense_size=10000,

        output_dir="output/speedup_potential_final",
        dry_run=True,
    )


def __getattr__(name):
    # CONFIG is built on access so the module imports even while MEDIAN_C is unset.
    if name == "CONFIG":
        return build_config()
    raise AttributeError(name)
