"""The Gaussian d=3 slice of configs/speedup.py with max_iter=500,000 instead of
20,000. Same eps grid, stop rule, seeds and timing protocol (warmup, rep).

GAUSS3_LONG_OUTPUT_DIR overrides the output directory. GAUSS3_LONG_SKIP ("+"-separated
methods, e.g. "srot+spar_sink") leaves those methods out.

    python run.py --config speedup_gauss3_long --count
"""

import dataclasses
import os

from configs import speedup

MAX_ITER = 500_000


def build_config():
    base = speedup.build_config()
    skip = set(filter(None, os.environ.get("GAUSS3_LONG_SKIP", "").split("+")))
    unknown = skip - {"srot", "spar_sink"}
    if unknown:
        raise ValueError(f"GAUSS3_LONG_SKIP: unsupported {sorted(unknown)}")
    return dataclasses.replace(
        base, no_srot=base.no_srot or "srot" in skip,
        no_sparsink=base.no_sparsink or "spar_sink" in skip,
        problems=[p for p in base.problems if (p[0], p[1]) == ("gaussian", 3)],
        n_iters=MAX_ITER, max_iter=MAX_ITER,
        output_dir=os.environ.get("GAUSS3_LONG_OUTPUT_DIR") or "output/speedup_gauss3_long")


def __getattr__(name):
    if name == "CONFIG":
        return build_config()
    raise AttributeError(name)
