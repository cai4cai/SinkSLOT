"""Scalability: runtime to convergence against N and against d, Gaussian data.

Experiment 1: d=3,  N in N_SWEEP.
Experiment 2: d=64, N in N_SWEEP.
Experiment 3: N=10,000, d in D_SWEEP.

Every method of configs/speedup.py runs at each point, 5 seeds: FlashSinkhorn
alternating and symmetric (fp32 and TF32), GeomLoss online, SROT, SinkSLOT-CUDA
and SinkSLOT-CUDA-symmetric at L in L_VALUES, and Spar-Sink at
s = k * s0(N), k in SPARSINK_K, s0(N) = 1e-3 * N * ln(N)^4.

eps at each d is the value where FlashSinkhorn-alternating fp32 reaches a ~5%
cost gap at N=10,000 (scripts/speedup_calibrate_eps.py), so every point sits at
roughly the same accuracy. The stop rule is the speedup benchmark's:
max(|df|, |dg|) < 1e-5 * median(C) between checkpoints (tol/2 for the damped
methods), check_every=5, max_iter=20000. No exact-OT reference (infeasible at
N=50,000), so cost_gap_pct is N/A; runtime and memory are the outputs.

The points are several BenchConfigs (CONFIGS) sharing one output directory,
because s depends on N. SCAL_PART=small (N <= 20,000, including the d-sweep) or
SCAL_PART=large (N >= 30,000) restricts CONFIGS to one part, with its own output
directory, so the two parts can run on different queues at the same time.
SCAL_OUTPUT_DIR overrides the output directory. Run
with run.py like the speedup config:

    python run.py --config scalability --count
    python run.py --config scalability --execute --num-shards K --shard-idx k
    python run.py --config scalability --merge
"""

import math
import os
from typing import Dict, List, Tuple

from configs.base import BenchConfig
from configs.speedup import REL_TOL

N_SWEEP = [5000, 10000, 20000, 30000, 50000]
D_SWEEP = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
N_DSWEEP = 10000
L_VALUES = [100, 1000, 5000]
SPARSINK_K = [4, 16, 64]
SEEDS = [0, 1, 2, 3, 4]
PART = os.environ.get("SCAL_PART", "")  # "", "small" (N <= 20,000) or "large" (N >= 30,000)

# d -> (median(C) at N=10,000 seed 0, eps at a ~5% cost gap), from
# scripts/speedup_calibrate_eps.py.
CALIBRATION: Dict[int, Tuple[float, float]] = {
    3: (4.734315872192383, 0.005918),
    4: (6.690551280975342, 0.01952),
    8: (14.674134254455566, 0.1599),
    16: (30.619638442993164, 0.5804),
    32: (62.67027282714844, 1.416),
    64: (126.7506103515625, 2.827),
    128: (254.42835998535156, 5.117),
    256: (510.0793151855469, 9.053),
    512: (1021.3765869140625, 15.78),
    1024: (2045.8070068359375, 27.92),
}


def s_values(n: int) -> List[int]:
    s0 = 1e-3 * n * math.log(n) ** 4
    return [int(round(k * s0)) for k in SPARSINK_K]


def _config(n: int, d: int) -> BenchConfig:
    median, eps = CALIBRATION[d]
    return BenchConfig(
        sizes=[n],
        dims=[d],
        problems=[("gaussian", d, [eps])],
        n_iters=20000,

        stop_mode="potential",
        max_iter=20000,
        stop_tol=REL_TOL,
        stop_tol_by_problem={("gaussian", d): REL_TOL * median},
        check_every=5,

        warmup=1,
        warmup_iters=10,
        rep=5,
        tf32=False,
        flash_tf32=[False, True],

        seeds=SEEDS,

        no_srot=False,
        srot_slices=L_VALUES,
        srot_delta=1e-8,

        no_sinkslot=True,

        no_sinkslotcuda=False,
        no_sinkslotcuda_symmetric=False,
        sinkslotcuda_slices=L_VALUES,

        no_sparsink=False,
        no_randsink=True,
        sparsink_s=s_values(n),
        sparsink_replicates=1,

        no_ott=True,
        no_rmae_check=True,
        no_geomloss=False,
        no_flash_symmetric=False,
        no_flash_alternating=False,

        isolate=True,
        tensorized=False,
        max_dense_size=max(N_SWEEP),

        output_dir=os.environ.get("SCAL_OUTPUT_DIR") or "output/scalability_potential" + (f"_{PART}" if PART else ""),
        dry_run=True,
    )


def build_configs() -> List[BenchConfig]:
    missing = sorted({3, 64, *D_SWEEP} - set(CALIBRATION))
    if missing:
        raise ValueError(f"configs/scalability.py: CALIBRATION has no entry for d={missing}.")
    points = [(n, d) for d in (3, 64) for n in N_SWEEP]
    points += [(N_DSWEEP, d) for d in D_SWEEP if (N_DSWEEP, d) not in points]
    if PART == "small":
        points = [(n, d) for n, d in points if n <= 20000]
    elif PART == "large":
        points = [(n, d) for n, d in points if n > 20000]
    elif PART:
        raise ValueError(f"SCAL_PART must be '', 'small' or 'large', got {PART!r}")
    return [_config(n, d) for n, d in points]


def __getattr__(name):
    # CONFIGS is built on access so the module imports while CALIBRATION is incomplete.
    if name == "CONFIGS":
        return build_configs()
    raise AttributeError(name)
