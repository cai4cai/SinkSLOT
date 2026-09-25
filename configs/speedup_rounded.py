"""Rounded-plan pass over the speedup benchmark: every unit of configs/speedup.py
solved once, untimed (warmup=0, rep=1), recording rounded_cost_gap_pct, the cost
gap of the plan rounded onto the transport polytope (Altschuler, Weed, Rigollet
2017, Algorithm 2). Runtimes for the rounded table come from the timed speedup run.

SPEEDUP_ROUNDED_OUTPUT_DIR overrides the output directory.

    python run.py --config speedup_rounded --count
"""

import dataclasses
import os

from configs import speedup


def build_config():
    return dataclasses.replace(
        speedup.build_config(), warmup=0, rep=1,
        output_dir=os.environ.get("SPEEDUP_ROUNDED_OUTPUT_DIR") or "output/speedup_rounded")


def __getattr__(name):
    if name == "CONFIG":
        return build_config()
    raise AttributeError(name)
