"""Paper, appendix (damping table): FlashSinkhorn symmetric (strict FP32) and GeomLoss online at one
damping alpha on the problems of configs/speedup.py.

alpha comes from DAMPING_ALPHA: 0.3, 0.5 and 0.7 run on every problem, 0.95 on Gaussian d=64 only.
The alpha = 0.9 rows of the table come from configs/speedup.py. Result keys do not include alpha, so
each alpha writes to its own output directory, output/speedup_damping_<alpha> (DAMPING_OUTPUT_DIR
overrides the prefix; the alpha suffix is kept). Everything else (eps grids, stop rule, iteration cap, seeds, timing) as configs/speedup.py.

    for a in 0.3 0.5 0.7 0.95; do DAMPING_ALPHA=$a python run.py --config speedup_damping --count; done
    DAMPING_ALPHA=0.3 python run.py --config speedup_damping --execute --num-shards K --shard-idx k
"""

import dataclasses
import os

from configs import speedup


def build_config():
    alpha = float(os.environ.get("DAMPING_ALPHA", "0.3"))
    base = speedup.build_config()
    problems = base.problems if alpha != 0.95 else [p for p in base.problems if p[:2] == ("gaussian", 64)]
    return dataclasses.replace(
        base, problems=problems, sym_alpha=alpha, flash_tf32=[False],
        no_srot=True, no_sinkslot=True, no_sinkslotcuda=True, no_sinkslotcuda_symmetric=True,
        no_geomloss=False, no_geomloss_multiscale=True, no_flash_symmetric=False, no_flash_alternating=True,
        no_sparsink=True, no_randsink=True,
        output_dir=f"{os.environ.get('DAMPING_OUTPUT_DIR') or 'output/speedup_damping'}_{alpha:g}")


def __getattr__(name):
    if name == "CONFIG":
        return build_config()
    raise AttributeError(name)
