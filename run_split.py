"""Split configs/speedup_gaussian_d3's sweep into two parallel jobs: SROT (slow,
plain-PyTorch eager, no Triton compilation) alone, and the other 6 methods
together. Each writes to its own output_dir to avoid a write race if both run
at once -- run.py's run_sweep clears its output CSVs once at the start of the
call, so two runs sharing one output_dir would stomp on each other.

Merge the two resulting output dirs back into configs/speedup_gaussian_d3's own
output_dir (output/table1) with merge_split.py once both finish.

Usage:
    python run_split.py srot
    python run_split.py rest
"""
import dataclasses
import sys

from configs.speedup_gaussian_d3 import CONFIG
from run import run_sweep

GROUP = sys.argv[1] if len(sys.argv) > 1 else None
if GROUP not in ("srot", "rest"):
    print("Usage: python run_split.py {srot|rest}")
    sys.exit(1)

if GROUP == "srot":
    cfg = dataclasses.replace(
        CONFIG,
        no_srot=False,
        no_sinkslotcuda=True,
        no_flash_symmetric=True,
        no_flash_alternating=True,
        no_geomloss=True,
        no_sparsink=True,
        output_dir="output/table1_split_srot",
    )
else:
    cfg = dataclasses.replace(
        CONFIG,
        no_srot=True,
        output_dir="output/table1_split_rest",
    )

run_sweep(cfg, cfg.output_dir, dry_run=False, label=f"[{GROUP}]")
