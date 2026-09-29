"""The Gaussian d=3 slice of configs/speedup.py with max_iter=500,000 instead of
20,000. Same eps grid, stop rule, seeds and timing protocol (warmup, rep).

GAUSS3_LONG_OUTPUT_DIR overrides the output directory.

    python run.py --config speedup_gauss3_long --count
"""

import dataclasses
import os

from configs import speedup

MAX_ITER = 500_000


def build_config():
    base = speedup.build_config()
    return dataclasses.replace(
        base, problems=[p for p in base.problems if (p[0], p[1]) == ("gaussian", 3)],
        n_iters=MAX_ITER, max_iter=MAX_ITER,
        output_dir=os.environ.get("GAUSS3_LONG_OUTPUT_DIR") or "output/speedup_gauss3_long")


def __getattr__(name):
    if name == "CONFIG":
        return build_config()
    raise AttributeError(name)
