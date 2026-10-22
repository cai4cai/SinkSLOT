"""Paper, appendix (truncate table): GeomLoss multiscale at truncate = 5, GeomLoss's default, on the
problems of configs/speedup.py, next to the truncate = 1 rows of the main runs.

MULTISCALE_CLUSTERING selects the clustering: "kmeans" (about 250 points per cluster, every problem;
the truncate = 1 counterpart is in configs/speedup.py) or "grid" (d <= 3 only; the truncate = 1
counterpart is configs/speedup_multiscale_grid.py). Each writes to its own output directory,
output/speedup_multiscale_<clustering>_truncate5. Everything else as configs/speedup.py.

    MULTISCALE_CLUSTERING=kmeans python run.py --config speedup_multiscale_truncate5 --count
    MULTISCALE_CLUSTERING=grid python run.py --config speedup_multiscale_truncate5 --execute --num-shards K --shard-idx k
"""

import dataclasses
import os

from configs import speedup


def build_config():
    clustering = os.environ.get("MULTISCALE_CLUSTERING", "kmeans")
    if clustering not in ("kmeans", "grid"):
        raise ValueError(f"MULTISCALE_CLUSTERING must be 'kmeans' or 'grid', got {clustering!r}")
    base = speedup.build_config()
    problems = base.problems if clustering == "kmeans" else [p for p in base.problems if p[1] <= 3]
    return dataclasses.replace(
        base, problems=problems, geomloss_clustering=clustering, geomloss_truncate=5.0,
        no_srot=True, no_sinkslot=True, no_sinkslotcuda=True, no_sinkslotcuda_symmetric=True,
        no_geomloss=True, no_geomloss_multiscale=False, no_flash_symmetric=True, no_flash_alternating=True,
        no_sparsink=True, no_randsink=True,
        output_dir=f"output/speedup_multiscale_{clustering}_truncate5")


def __getattr__(name):
    if name == "CONFIG":
        return build_config()
    raise AttributeError(name)
