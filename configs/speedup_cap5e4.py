"""configs/speedup.py with max_iter=50,000 instead of 20,000, for rerunning the
runs that hit the 20,000 cap. Same eps grids, stop rule, seeds and timing
protocol. SPEEDUP_CAP_SLICES (dataset:d entries separated by "+" or ",", e.g.
"half_moon:2+gaussian:64"; "+" survives sbatch --export) restricts the slices; CAP5E4_OUTPUT_DIR overrides the
output directory.

    python run.py --config speedup_cap5e4 --count
"""

import dataclasses
import os

from configs import speedup

MAX_ITER = 50_000


def build_config():
    base = speedup.build_config()
    problems = base.problems
    slices = os.environ.get("SPEEDUP_CAP_SLICES")
    if slices:
        keep = {(s.split(":")[0], int(s.split(":")[1])) for s in slices.replace("+", ",").split(",")}
        problems = [p for p in problems if (p[0], p[1]) in keep]
    return dataclasses.replace(
        base, problems=problems, n_iters=MAX_ITER, max_iter=MAX_ITER,
        output_dir=os.environ.get("CAP5E4_OUTPUT_DIR") or "output/speedup_cap5e4")


def __getattr__(name):
    if name == "CONFIG":
        return build_config()
    raise AttributeError(name)
