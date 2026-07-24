"""Benchmark run configuration.

Edit CONFIG below to change what run.py executes.

The sweep is the full cross product of datasets x eps_values x methods x sizes x dims.
With `isolate` on (the default) each of those is a separate subprocess, so the run count
is len(datasets) * len(eps_values) * n_methods * len(sizes) * len(dims) -- currently
2 * 3 * 3 * 2 * 2 = 72. Every run appends into one forward_all.csv / backward_all.csv,
with dataset, tf32, eps, d and n as ordinary columns.

Sizes here are deliberately small (a quick-turnaround grid). Note that the memory column
cannot show anything at this scale: gpu_memory_mb is whole-device usage, ~640MB of which
is fixed CUDA context, and the problem data (a few MB) fits inside blocks the caching
allocator has already reserved. Demonstrating the O(nd)-vs-O(n^2) claim needs n in the
tens of thousands plus `tensorized=True` for a dense baseline -- see analysis.md.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BenchConfig:
    which: str = "forward"  # "forward" or "backward"

    sizes: List[int] = field(default_factory=lambda: [256, 512])
    dims: List[int] = field(default_factory=lambda: [8, 16])

    eps_values: List[float] = field(default_factory=lambda: [0.1, 0.01, 0.001])
    n_iters: int = 15
    warmup: int = 10
    rep: int = 30
    tf32: bool = True

    datasets: List[str] = field(default_factory=lambda: ["gaussian", "8gaussians"])
    no_ott: bool = True  # skip OTT-JAX (JAX/OTT often not installed locally)
    # RMAE reference: a converged Sinkhorn solve on GPU, one per (dataset, n, d, eps),
    # disk-cached across runs. Dominates sweep time at large n (~50s at n=4096).
    no_rmae_check: bool = False
    no_geomloss: bool = False
    no_flash_symmetric: bool = False
    no_flash_alternating: bool = False
    only: Optional[str] = None  # "flash_symmetric" | "flash_alternating" | "flash" | "geomloss" | "ott"

    # One subprocess per (dataset, eps, method, n, d). Required for gpu_memory_mb to be
    # attributable: the reported figure is whole-device usage, and PyTorch never returns
    # pooled memory to the driver, so measurements sharing a process inherit each other's
    # footprint. Costs ~5s of CUDA/JIT startup per row.
    isolate: bool = True

    tensorized: bool = False
    max_dense_size: int = 512

    verify: bool = False
    quiet: bool = False

    output_dir: str = "output/paper_benchmarks"

    dry_run: bool = True  # print the constructed command instead of running it
    # target should be added. 


CONFIG = BenchConfig()

# Look into any and all convergence difference so w can scope the experiemtns a bti better. 

