"""Benchmark run configuration.

Edit ``CONFIG`` below to change what ``run.py`` executes, then
``python run.py --dry-run`` to see the exact command list before committing to it.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class BenchConfig:
    """Settings for one benchmark sweep.

    The sweep is the cross product of ``datasets`` x ``eps_values`` x methods x
    ``sizes`` x ``dims`` (or, when ``problems`` is set, of those explicit
    (dataset, d, eps list) triples x methods x ``sizes``). Two baselines carry a swept parameter of their own -- SROT's
    number of projections and Spar-Sink/Rand-Sink's subsample size -- which expands only
    their own rows, since no other method has such a knob. With ``isolate`` on, each
    combination is a separate subprocess; the defaults here are a deliberately minimal
    grid (one size, one dim, one value of each per-method parameter) that comes to
    36 runs -- enough to exercise every method at every eps, quickly.

    Every run appends into a single ``forward_all.csv`` plus
    one speedup table, with dataset, tf32, eps, d and n as ordinary columns.

    Attributes:
        sizes: Point-cloud sizes n to benchmark. Deliberately small here, for
            quick turnaround. Note the memory column cannot show anything at this
            scale: ``gpu_memory_mb`` is whole-device usage, ~640MB of which is fixed
            CUDA context, against a few MB of problem data. Demonstrating the
            O(nd)-vs-O(n^2) claim needs n in the tens of thousands together with
            ``tensorized=True`` for a dense baseline. See analysis.md.
        dims: Feature dimensions d to benchmark.
        problems: Optional explicit list of (dataset, d, eps_values) triples. When
            set it replaces the ``datasets`` x ``dims`` x ``eps_values`` product, so
            one config can mix dimensions and give each (dataset, d) its own eps grid.

        eps_values: Entropic regularization strengths. One run per value; every
            method is measured at each.
        n_iters: Sinkhorn iterations. Fixed for every method, with early stopping
            disabled throughout, so the timing column compares equal work.
        warmup: Untimed warmup calls before measurement.
        warmup_iters: Iteration cap for each warmup call.
        rep: Timed repetitions; the reported figure is their mean.
        tf32: Allow TF32 matmuls (10-bit mantissa) rather than strict FP32, for every
            method except FlashSinkhorn (see ``flash_tf32``). Tracked per row, so a
            TF32 run and an FP32 run of the same configuration are distinct rows
            rather than one overwriting the other.
        flash_tf32: TF32 settings to run FlashSinkhorn at. Each Flash unit is
            repeated once per value.
        seeds: Data-generation seeds (x, y, a, b only). The full grid is repeated
            once per seed. Method-internal randomness (slice projections, sparse-
            kernel sampling) is not affected -- it stays at its own independent
            default so varying seeds only changes the underlying problem instance.

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

        no_sinkslotcuda: Skip SinkSLOT-CUDA: the same method and solve kernels as
            SinkSLOT with a CUDA-optimised setup path (fused Triton cost, CUDA 1-D OT
            scan in fp32, int32 CSC key). It keeps its own RMAE reference.
        sinkslotcuda_slices: L values to sweep for SinkSLOT-CUDA and
            SinkSLOT-CUDA-symmetric. Kept separate from sinkslot_slices so the two
            can be compared at matched L or swept apart.
        no_sinkslotcuda_symmetric: Skip SinkSLOT-CUDA with symmetric (damped)
            updates. Skipped by default.

        no_sparsink: Skip the Spar-Sink and Rand-Sink baselines.
        no_randsink: Skip just Rand-Sink (the uniform-sampling variant), while
            still running Spar-Sink (importance sampling). Independent of
            no_sparsink, which skips both.
        sparsink_s: Expected kernel subsample sizes s to sweep, in units of
            s0(n) = 1e-3 * n * ln(n)^4 (the authors' simu_ot.py uses s = {2,4,8,16} * s0
            with 50 replications). Small s can leave a row or column unsampled; as in
            the authors' code, those lines are dropped and the rest solved, and the
            count is reported in the ``empty_lines`` column.
        sparsink_replicates: Independent kernel draws averaged per row.

        no_ott: Skip OTT-JAX (JAX/OTT is often not installed locally).
        no_rmae_check: Skip the accuracy metric. The reference is a converged Sinkhorn
            solve on GPU, one per (dataset, n, d, eps), disk-cached across runs; it
            dominates sweep time at large n (~50s at n=4096).
        no_geomloss: Skip the GeomLoss/KeOps baseline.
        no_flash_symmetric: Skip FlashSinkhorn's symmetric backend.
        no_flash_alternating: Skip FlashSinkhorn's alternating backend.
        only: Restrict the sweep to a single method: ``"flash_symmetric"``,
            ``"flash_alternating"``, ``"flash"`` (both backends), ``"geomloss"``,
            ``"ott"``, ``"srot"``, ``"spar_sink"``, ``"rand_sink"``, ``"sinkslot"``,
            ``"sinkslotcuda"`` or ``"sinkslotcuda_symmetric"``.

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

        output_dir: Where the CSVs are written. Cleared at the start of each
            non-sharded sweep; sharded runs resume instead (see run.py).
        dry_run: Print the constructed commands instead of running them.
    """

    sizes: List[int] = field(default_factory=lambda: [256, 512])
    dims: List[int] = field(default_factory=lambda: [8, 16])

    eps_values: List[float] = field(default_factory=lambda: [0.1, 0.01, 0.001])
    n_iters: int = 50

    # Early stopping. stop_mode "fixed" (default) runs exactly n_iters, the
    # FlashSinkhorn protocol for per-iteration throughput. The other modes run up
    # to max_iter and check every check_every iterations:
    #   "marginal":  max(max|P1-a|, max|P^T1-b|) <= stop_tol.
    #   "potential": max(|df|, |dg|) <= stop_tol between consecutive checkpoints.
    #   "scaling":   Spar-Sink's own rule on its u/v scalings, <= scaling_tol
    #                (spar_sink/rand_sink only).
    stop_mode: str = "fixed"     # "fixed" | "marginal" | "potential" | "scaling"
    max_iter: int = 10000        # cap in non-fixed modes (n_iters is the count in "fixed")
    stop_tol: float = 1e-4       # marginal/potential threshold
    stop_tol_by_problem: Optional[Dict[Tuple[str, int], float]] = None  # per-(dataset, d) override of stop_tol
    scaling_tol: float = 1e-6    # Spar-Sink u/v threshold ("scaling" mode)
    check_every: int = 10        # iterations between convergence checks
    warmup: int = 5
    warmup_iters: int = 10
    rep: int = 15
    tf32: bool = True
    flash_tf32: List[bool] = field(default_factory=lambda: [False])

    # Data-generation seeds only (x, y, a, b) -- method-internal randomness (slice
    # projections, sparse-kernel sampling) stays independently seeded regardless.
    # One full sweep is repeated per seed, letting results be aggregated afterward.
    seeds: List[int] = field(default_factory=lambda: [0])

    datasets: List[str] = field(default_factory=lambda: ["gaussian", "8gaussians"])
    problems: Optional[List[Tuple[str, int, List[float]]]] = None

    no_srot: bool = False
    srot_slices: List[int] = field(default_factory=lambda: [10])
    srot_delta: float = 1e-8

    no_sinkslot: bool = False
    sinkslot_slices: List[int] = field(default_factory=lambda: [10])

    no_sinkslotcuda: bool = False
    sinkslotcuda_slices: List[int] = field(default_factory=lambda: [10])
    no_sinkslotcuda_symmetric: bool = True

    no_sparsink: bool = False
    no_randsink: bool = False
    sparsink_s: List[int] = field(default_factory=lambda: [8000])
    sparsink_replicates: int = 5

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

    # Deprecated alias for scaling_tol, kept so older configs still load.
    potential_tol: Optional[float] = None

    def __post_init__(self) -> None:
        if self.potential_tol is not None:
            self.scaling_tol = self.potential_tol
            self.potential_tol = None


# Quick-iteration sweep. Structurally identical ("ditto") to the published
# sweeps in this package (speedup.py, scalability.py) -- same fields, same
# methods (including both SinkSLOT and SinkSLOT-CUDA), same early-stopping/
# convergence block -- but a deliberately tiny grid so the whole thing runs in
# minutes while wiring is being changed. The published configs are the same shape
# with the paper's values; this one is not used for any reported number.
CONFIG = BenchConfig(
    sizes=[512, 2048],
    dims=[8],

    eps_values=[0.1, 0.01],
    n_iters=50,

    # Convergence / early stopping (see the class docstring). The published
    # protocol is stop_mode="potential" (time-to-accuracy, stop_tol=1e-6); see
    # configs/speedup.py. "fixed" here runs exactly n_iters for every method, for
    # a quick per-iteration throughput check while wiring is being changed.
    stop_mode="fixed",
    max_iter=10000,
    stop_tol=1e-4,
    scaling_tol=1e-6,
    check_every=10,

    warmup=5,
    rep=10,
    tf32=True,
    flash_tf32=[True],

    datasets=["gaussian"],

    no_srot=False,
    srot_slices=[10],
    srot_delta=1e-8,

    no_sinkslot=False,
    sinkslot_slices=[64],

    no_sinkslotcuda=False,
    sinkslotcuda_slices=[64],   # matched to sinkslot_slices for a head-to-head

    no_sparsink=False,
    sparsink_s=[8000],
    sparsink_replicates=5,

    no_ott=True,
    no_rmae_check=False,
    no_geomloss=False,
    no_flash_symmetric=False,
    no_flash_alternating=False,

    isolate=True,
    tensorized=False,
    max_dense_size=2048,

    output_dir="output/quick",
    dry_run=True,
)
