"""Benchmark run configuration.

Edit ``CONFIG`` below to change what ``run.py`` executes, then
``python run.py --dry-run`` to see the exact command list before committing to it.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BenchConfig:
    """Settings for one benchmark sweep.

    The sweep is the cross product of ``datasets`` x ``eps_values`` x methods x
    ``sizes`` x ``dims``. Two baselines carry a swept parameter of their own -- SROT's
    number of projections and Spar-Sink/Rand-Sink's subsample size -- which expands only
    their own rows, since no other method has such a knob. With ``isolate`` on, each
    combination is a separate subprocess; the defaults here are a deliberately minimal
    grid (one size, one dim, one value of each per-method parameter) that comes to
    36 runs -- enough to exercise every method at every eps, quickly.

    Every run appends into a single ``forward_all.csv`` (or ``backward_all.csv``) plus
    one speedup table, with dataset, tf32, eps, d and n as ordinary columns.

    Attributes:
        which: Which benchmark to run, ``"forward"`` or ``"backward"``.

        sizes: Point-cloud sizes n to benchmark. Deliberately small here, for
            quick turnaround. Note the memory column cannot show anything at this
            scale: ``gpu_memory_mb`` is whole-device usage, ~640MB of which is fixed
            CUDA context, against a few MB of problem data. Demonstrating the
            O(nd)-vs-O(n^2) claim needs n in the tens of thousands together with
            ``tensorized=True`` for a dense baseline. See analysis.md.
        dims: Feature dimensions d to benchmark.

        eps_values: Entropic regularization strengths. One run per value; every
            method is measured at each.
        n_iters: Sinkhorn iterations. Fixed for every method, with early stopping
            disabled throughout, so the timing column compares equal work.
        warmup: Untimed iterations before measurement.
        rep: Timed repetitions; the reported figure is their mean.
        tf32: Allow TF32 matmuls (10-bit mantissa) rather than strict FP32. Tracked
            per row, so a TF32 run and an FP32 run of the same configuration are
            distinct rows rather than one overwriting the other.

        datasets: Synthetic point-cloud distributions. ``"gaussian"`` is an isotropic
            normal; ``"8gaussians"`` is 8 clusters on a radius-2 ring.

        no_srot: Skip the SROT (Sliced-Regularized OT) baseline.
        srot_slices: L values to sweep -- the number of random 1-D projections
            averaged into SROT's reference plan. Dense O(n*m), so SROT also respects
            ``max_dense_size``.
        srot_delta: Weight of the independent coupling mixed into SROT's plan. Keeps
            every entry strictly positive so the kernel has full support; delta=1
            recovers standard entropic Sinkhorn.

        no_sinkslot: Skip the SinkSLOT (fused-Triton gamma=0 sparse SROT) baseline.
        sinkslot_slices: L values to sweep for SinkSLOT -- number of 1-D projections.
            Sparse O(L(N+M)), so unlike SROT it is not gated by max_dense_size.

        no_sparsink: Skip the Spar-Sink and Rand-Sink baselines.
        sparsink_s: Expected kernel subsample sizes s to sweep. Their paper uses
            s = {2,4,8,16} * s0(n) with s0(n) = 1e-3 * n * log^4(n), i.e. s0(256) ~ 242
            and s0(512) ~ 775. Small s can leave a row or column unsampled, which makes
            the problem infeasible for that row; such draws are dropped and counted in
            the ``empty_lines`` column.
        sparsink_replicates: Independent kernel draws averaged per row. Sampling is
            stochastic, so a single draw would report sampling noise as method quality;
            their paper averages 100.

        no_ott: Skip OTT-JAX (JAX/OTT is often not installed locally).
        no_rmae_check: Skip the accuracy metric. The reference is a converged Sinkhorn
            solve on GPU, one per (dataset, n, d, eps), disk-cached across runs; it
            dominates sweep time at large n (~50s at n=4096).
        no_geomloss: Skip the GeomLoss/KeOps baseline.
        no_flash_symmetric: Skip FlashSinkhorn's symmetric backend.
        no_flash_alternating: Skip FlashSinkhorn's alternating backend.
        only: Restrict the sweep to a single method: ``"flash_symmetric"``,
            ``"flash_alternating"``, ``"flash"`` (both backends), ``"geomloss"``,
            ``"ott"``, ``"srot"``, ``"spar_sink"`` or ``"rand_sink"``.

        isolate: Give every measurement its own subprocess. Required for
            ``gpu_memory_mb`` to be attributable to a method: the figure is
            whole-device usage, and compiled kernels and CUDA context are not released
            within a process, so measurements sharing one inherit each other's
            footprint. Costs ~5s of CUDA/JIT startup per row.

        tensorized: Include the dense O(n^2) GeomLoss baseline.
        max_dense_size: Largest n for which dense methods run -- the tensorized
            baseline, SROT, and Spar-Sink/Rand-Sink's probability build.

        verify: Run correctness checks instead of benchmarking.
        quiet: Suppress per-measurement output.

        output_dir: Where the CSVs are written. Cleared at the start of each sweep.
        dry_run: Print the constructed commands instead of running them.
    """

    which: str = "forward"

    sizes: List[int] = field(default_factory=lambda: [256])
    dims: List[int] = field(default_factory=lambda: [8])

    eps_values: List[float] = field(default_factory=lambda: [0.1, 0.01, 0.001])
    n_iters: int = 15
    warmup: int = 3
    rep: int = 5
    tf32: bool = True

    datasets: List[str] = field(default_factory=lambda: ["gaussian", "8gaussians"])

    no_srot: bool = False
    srot_slices: List[int] = field(default_factory=lambda: [5])
    srot_delta: float = 1e-8

    no_sinkslot: bool = False
    sinkslot_slices: List[int] = field(default_factory=lambda: [10])

    no_sparsink: bool = False
    sparsink_s: List[int] = field(default_factory=lambda: [8000])
    sparsink_replicates: int = 3

    no_ott: bool = True
    no_rmae_check: bool = False
    no_geomloss: bool = False
    no_flash_symmetric: bool = False
    no_flash_alternating: bool = False
    only: Optional[str] = None

    isolate: bool = True

    tensorized: bool = False
    max_dense_size: int = 512

    verify: bool = False
    quiet: bool = False

    output_dir: str = "output/paper_benchmarks"
    dry_run: bool = True


CONFIG = BenchConfig()
