"""Spar-Sink only, at larger kernel budgets s = k * s0 with k in {256, 512}, on the
slices where k <= 128 does not reach the 1% or 10% cost gap (8-Gaussians,
Gaussian d=3 and d=64). Otherwise configs/speedup.py: same eps grids, stop rule,
seeds and timing protocol, with max_iter=50,000.

SPARSINK_K_OUTPUT_DIR overrides the output directory; SPARSINK_K_MAX_ITER overrides max_iter.

    python run.py --config speedup_sparsink_k --count
"""

import dataclasses
import os

from configs import speedup

K_VALUES = (256, 512)
SLICES = {("8gaussians", 2), ("gaussian", 3), ("gaussian", 64)}
MAX_ITER = int(os.environ.get("SPARSINK_K_MAX_ITER", "50000"))


def build_config():
    base = speedup.build_config()
    return dataclasses.replace(
        base, problems=[p for p in base.problems if (p[0], p[1]) in SLICES],
        sparsink_s=[int(round(k * speedup._s0)) for k in K_VALUES],
        no_srot=True, no_sinkslot=True, no_sinkslotcuda=True, no_sinkslotcuda_symmetric=True,
        no_geomloss=True, no_flash_symmetric=True, no_flash_alternating=True,
        no_sparsink=False, no_randsink=True,
        n_iters=MAX_ITER, max_iter=MAX_ITER,
        output_dir=os.environ.get("SPARSINK_K_OUTPUT_DIR") or "output/speedup_sparsink_k")


def __getattr__(name):
    if name == "CONFIG":
        return build_config()
    raise AttributeError(name)
