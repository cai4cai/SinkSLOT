"""GeomLoss multiscale with GeomLoss's grid clustering (d <= 3 only) on the problems of configs/speedup.py.

Paper: the GeomLoss (multiscale, grid) row of main-text Table 1, and the same row in the appendix
damping, convergence and memory tables and the Pareto figure. Everything else as configs/speedup.py."""

import dataclasses
import os

from configs import speedup


def build_config():
    base = speedup.build_config()
    return dataclasses.replace(
        base, problems=[p for p in base.problems if p[1] <= 3], geomloss_clustering="grid",
        no_srot=True, no_sinkslot=True, no_sinkslotcuda=True, no_sinkslotcuda_symmetric=True,
        no_geomloss=True, no_flash_symmetric=True, no_flash_alternating=True, no_sparsink=True, no_randsink=True,
        output_dir=os.environ.get("SPEEDUP_OUTPUT_DIR") or "output/speedup_multiscale_grid")


def __getattr__(name):
    if name == "CONFIG":
        return build_config()
    raise AttributeError(name)
