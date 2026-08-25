"""
Forward pass benchmark for FlashSinkhorn paper.

USE CASE: Inference / Evaluation
    - Computing OT distance without gradient flow
    - Model evaluation, metric computation
    - Scenarios where `torch.no_grad()` is used
    - Example: `with torch.no_grad(): dist = sinkhorn(x, y)`

Compares forward pass timing across dimensions:
- FlashSinkhorn (Triton fused kernels with autotuning)
- GeomLoss Online (KeOps) / Tensorized (dense)
- OTT-JAX Online / Dense

IMPORTANT:
- TF32 is enabled by default for ~2x speedup on A100/H100 GPUs (uses Tensor Cores)
- All kernels use bucketed autotune cache keys (CACHE_KEY = n // 32), so nearby
  sizes share configs and cross-size cache pollution is minimal (~5% variance).
  Subprocess isolation is no longer needed for most use cases.
- Sizes are still run large->small as a best practice
- Autotuning finds optimal block sizes; first call per (n,d) has ~2-3s overhead
- Memory overhead note: First run at each config incurs ~256MB Triton compilation
  overhead. Subsequent runs use cached configs (~4MB steady-state). This explains
  why the largest size (run first) may show higher memory than expected.

Cost convention (all methods use full squared Euclidean):
- FlashSinkhorn: C(x,y) = ||x-y||² (half_cost=False default)
- GeomLoss SqDist: C(x,y) = ||x-y||²
- OTT-JAX PointCloud: C(x,y) = ||x-y||² (default)
- Loss values should match within numerical tolerance (~1e-4 relative error)

Timing methodology:
- FlashSinkhorn/GeomLoss: CUDA events (precise GPU timing)
- OTT-JAX: Wall-clock time with block_until_ready() sync
  (JAX lacks CUDA event API; wall-clock includes minor Python overhead ~1-5%)

Usage:
    # Default: d=3,8,64, TF32 enabled, online methods only (in-process)
    python -m sinkslot.bench.bench_forward

    # Strict FP32 (slower but higher precision)
    python -m sinkslot.bench.bench_forward --no-tf32

    # Include tensorized/dense methods (small sizes only)
    python -m sinkslot.bench.bench_forward --tensorized --max-dense-size 20000

    # Verify loss parity first, then benchmark
    python -m sinkslot.bench.bench_forward --verify

    # Single dimension
    python -m sinkslot.bench.bench_forward --dims 64

    # Subprocess mode: still available for maximum isolation if needed
    python -m sinkslot.bench.bench_forward --subprocess --dims 512
"""

from __future__ import annotations

import os

# Must set CUDA_VISIBLE_DEVICES before importing torch/KeOps
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import csv
import ctypes
import gc
import json
import math
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch


def _preload_cuda_libs() -> None:
    """Preload CUDA libraries for KeOps.

    Discovers CUDA_HOME from environment or PyTorch, then loads nvrtc/cudart
    so that KeOps can find them at JIT compile time.
    """
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home is None:
        try:
            from torch.utils.cpp_extension import CUDA_HOME as _ch
            cuda_home = _ch
        except Exception:
            pass
    if cuda_home is None:
        return
    os.environ.setdefault("CUDA_HOME", cuda_home)
    os.environ.setdefault("CUDA_PATH", cuda_home)
    lib_dir = Path(cuda_home) / "targets" / "x86_64-linux" / "lib"
    if not lib_dir.is_dir():
        lib_dir = Path(cuda_home) / "lib64"
    for pattern in ("libnvrtc.so*", "libnvrtc-builtins.so*", "libcudart.so*"):
        for lib in sorted(lib_dir.glob(pattern)):
            try:
                ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
            except Exception:
                pass


def _set_tf32(enabled: bool) -> None:
    """Set TF32 mode."""
    torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
    torch.backends.cudnn.allow_tf32 = bool(enabled)
    if not enabled:
        try:
            torch.set_float32_matmul_precision("highest")
        except Exception:
            pass


def _nvtx_available() -> bool:
    try:
        torch.cuda.nvtx.range_push("nvtx_check")
        torch.cuda.nvtx.range_pop()
        return True
    except Exception:
        return False


@contextmanager
def _nvtx_range(message: str, *, enabled: bool) -> None:
    """Emit an NVTX range (useful for Nsight Systems)."""
    if enabled:
        try:
            torch.cuda.nvtx.range_push(message)
        except Exception:
            enabled = False
    try:
        yield
    finally:
        if enabled:
            try:
                torch.cuda.nvtx.range_pop()
            except Exception:
                pass


@dataclass
class TimingResult:
    """Timing measurement result."""
    method: str
    n: int
    m: int
    d: int
    eps: float
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    median_ms: float
    gpu_memory_mb: float  # whole-device GPU memory in use, as nvidia-smi reports it
    oom: bool
    n_iters: int = 0
    rmae_pct: Optional[float] = None  # legacy: |loss - ref| / ref * 100, vs converged entropic OT at
                                       # the same eps. Superseded by cost_gap_pct/barycentric_sym below
                                       # for srot/sinkslot/sinkslotcuda/spar_sink/rand_sink/flash; kept
                                       # in the schema but no longer populated by those methods.
    cost_gap_pct: Optional[float] = None  # 100*(cost - ref.cost)/|ref.cost|, vs EXACT (unregularized) OT
    barycentric_sym: Optional[float] = None  # weighted RMS deviation of both barycentric projections
                                              # from the exact plan's, vs the same exact-OT reference
    dataset: str = "gaussian"  # point-cloud distribution; see sample_point_cloud()
    tf32: bool = False  # TF32 matmuls enabled (10-bit mantissa) vs strict FP32
    iters_run: Optional[int] = None    # iterations actually executed (= n_iters in fixed mode)
    converged: Optional[bool] = None   # reached the stop threshold before max_iter (None in fixed)
    final_viol: Optional[float] = None # max marginal violation at stop (diagnostic)
    srot_slices: Optional[int] = None  # SROT only: number of random 1-D projections (L)
    setup_ms: Optional[float] = None  # per-method setup excluded from mean_ms (SROT plan, sparsink sampling)
    sample_size: Optional[int] = None  # Spar-Sink/Rand-Sink only: requested subsample size s
    nnz: Optional[int] = None  # Spar-Sink/Rand-Sink only: mean entries actually drawn
    empty_lines: Optional[int] = None  # Spar-Sink/Rand-Sink only: rows+cols with no sampled entry
    rmae_std: Optional[float] = None  # std of rmae_pct across replicates (sampling methods)
    valid_replicates: Optional[int] = None  # draws that produced a finite loss (no empty rows/cols)
    seed: int = 0  # data-generation seed (x, y, a, b) only -- not method-internal randomness

    # Feasibility diagnostics on the solved plan itself, computed unconditionally
    # for every row (not gated by stop mode) -- matching the SLOT repo's own
    # convention (bench/run.py): mass and marg_viol measure DIFFERENT failures and
    # must not be conflated. mass < 1 is STRUCTURAL (the solver's support cannot
    # carry the marginals, e.g. a sparse method with an empty row/column) and no
    # amount of iteration fixes it; mass ~= 1 with large marg_viol is ordinary
    # non-convergence.
    mass: Optional[float] = None  # T.sum() -- achieved total transported mass (target: 1.0)
    marg_viol: Optional[float] = None  # max(||r-a||_inf, ||c-b||_inf), achieved vs target marginals
    plan_empty_rows: Optional[int] = None  # rows of T with zero achieved mass
    plan_empty_cols: Optional[int] = None  # cols of T with zero achieved mass
    hit_max_iters: Optional[bool] = None  # iters_run >= stop.max_iter (None under "fixed" mode)


@dataclass
class JITOverheadResult:
    """JIT compilation overhead measurement result."""
    method: str
    n: int
    d: int
    eps: float
    cold_start_ms: float  # First call (includes JIT compilation)
    warm_ms: float        # Steady-state (average of subsequent calls)
    jit_overhead_ms: float  # cold_start - warm
    overhead_ratio: float   # cold_start / warm


def timing_result_to_json(r: TimingResult) -> str:
    """Serialize TimingResult to a JSON string (one line)."""
    return json.dumps({
        "method": r.method, "n": r.n, "m": r.m, "d": r.d, "eps": r.eps,
        "mean_ms": r.mean_ms, "std_ms": r.std_ms, "min_ms": r.min_ms,
        "max_ms": r.max_ms, "median_ms": r.median_ms,
        "gpu_memory_mb": r.gpu_memory_mb, "oom": r.oom,
        "n_iters": r.n_iters, "rmae_pct": r.rmae_pct, "dataset": r.dataset,
        "tf32": r.tf32, "srot_slices": r.srot_slices, "setup_ms": r.setup_ms,
        "iters_run": r.iters_run, "converged": r.converged, "final_viol": r.final_viol,
        "sample_size": r.sample_size, "nnz": r.nnz,
        "empty_lines": r.empty_lines, "rmae_std": r.rmae_std,
        "valid_replicates": r.valid_replicates, "seed": r.seed,
        "cost_gap_pct": r.cost_gap_pct, "barycentric_sym": r.barycentric_sym,
        "mass": r.mass, "marg_viol": r.marg_viol,
        "plan_empty_rows": r.plan_empty_rows, "plan_empty_cols": r.plan_empty_cols,
        "hit_max_iters": r.hit_max_iters,
    })


def timing_result_from_json(line: str) -> TimingResult:
    """Deserialize TimingResult from a JSON string."""
    d = json.loads(line)
    return TimingResult(**d)


def compute_entropic_ot_reference_pot(
    x: torch.Tensor, y: torch.Tensor, a: torch.Tensor, b: torch.Tensor, eps: float,
    *, max_iter: int = 20000, tol: float = 1e-12,
) -> float:
    """Same reference as compute_entropic_ot_reference(), but via POT on CPU.

    Independent third-party implementation, used only to cross-check the GPU solver
    (see compute_entropic_ot_reference). Far too slow for the benchmark itself: dense single-threaded
    float64 NumPy costs ~41 ms/iteration at n=1024 and scales as n^2, i.e. hours per
    solve at n=4096.

    Uses method="sinkhorn_log"; the standard kernel form underflows to zero at eps=0.001.
    """
    import numpy as np
    import ot

    x_np = x.detach().cpu().numpy().astype("float64")
    y_np = y.detach().cpu().numpy().astype("float64")
    a_np = a.detach().cpu().numpy().astype("float64")
    b_np = b.detach().cpu().numpy().astype("float64")
    cost_matrix = ot.dist(x_np, y_np, metric="sqeuclidean")

    _, log = ot.sinkhorn(
        a_np, b_np, cost_matrix, eps,
        method="sinkhorn_log", log=True, numItermax=max_iter, stopThr=tol,
    )
    f = eps * np.asarray(log["log_u"], dtype="float64").ravel()
    g = eps * np.asarray(log["log_v"], dtype="float64").ravel()
    return float(a_np @ f + b_np @ g)


def eps_scaled(
    eps: float, other: torch.Tensor, cost: torch.Tensor, log_w: torch.Tensor, *, dim: int,
) -> torch.Tensor:
    """eps * logsumexp over `dim` of ((other - cost)/eps + log_w), broadcast on `dim`."""
    if dim == 1:
        shifted = (other.unsqueeze(0) - cost) / eps + log_w.unsqueeze(0)
    else:
        shifted = (other.unsqueeze(1) - cost) / eps + log_w.unsqueeze(1)
    return eps * torch.logsumexp(shifted, dim=dim)


def compute_entropic_ot_reference(
    x: torch.Tensor, y: torch.Tensor, a: torch.Tensor, b: torch.Tensor, eps: float,
    *, max_iter: int = 20000, tol: float = 1e-6, check_every: int = 10,
    verbose: bool = False,
) -> float:
    """Converged entropic-OT dual objective at regularization `eps`, on GPU in float64.

    Reference for the RMAE metric used by Spar-Sink ("Importance Sparsification for
    Sinkhorn Algorithm", Li/Yu/Li/Meng): each method is compared against a *converged*
    Sinkhorn solve at the SAME eps, which isolates solver/iteration error from the
    entropic bias inherent to regularizing at all.

    Textbook dense log-domain Sinkhorn -- it materializes the full n x m cost matrix and
    uses torch.logsumexp, sharing no code with the FlashSinkhorn kernels or GeomLoss's
    KeOps path, so it remains an independent check on what those streaming kernels
    should reproduce. Cross-checked against POT at small n (compute_entropic_ot_reference_pot).

    Returns the dual objective <a, f> + <b, g>, NOT the primal transport cost <P, C>
    that POT's `ot.sinkhorn2` reports. The benchmarked methods -- FlashSinkhorn's
    SamplesLoss and GeomLoss's sinkhorn_cost(debias=False) -- both report the dual, and
    the two differ by the entropic term: at n=64, eps=0.1 the dual is 0.938 against
    <P, C> = 1.424. Comparing against sinkhorn2 would report a large "error" that is
    purely a difference of functional.

    Convergence is on max marginal violation |P.sum(dim=1) - a|, checked every
    `check_every` iterations. Prints a warning if `max_iter` is hit without reaching
    `tol`, since an unconverged reference would silently corrupt every rmae_pct.

    Accuracy achieved in practice (n=256, d=8, float64): ~2e-10 marginal error at
    eps >= 0.01, agreeing with POT to ~2e-8 relative. At eps=1e-3 it reaches ~6e-5,
    where POT is *less* converged than this solver (our dual is higher, and the dual is
    maximized), so rmae_pct at eps=1e-3 carries a reference uncertainty around 5e-4
    relative -- still three orders below the percent-level signal being measured.
    """
    xd = x.detach().double()
    yd = y.detach().double()
    ad = a.detach().double()
    bd = b.detach().double()

    cost = torch.cdist(xd, yd, p=2) ** 2  # ||x - y||^2, matching half_cost=False
    log_a = ad.log()
    log_b = bd.log()

    f = torch.zeros_like(ad)
    g = torch.zeros_like(bd)

    def sweep(stage_eps: float, iters: int) -> Tuple[float, int]:
        """Alternating log-domain updates at `stage_eps`; returns (marginal err, iters used)."""
        nonlocal f, g
        stage_err = float("inf")
        used = 0
        for used in range(1, iters + 1):
            f = -eps_scaled(stage_eps, g, cost, log_b, dim=1)
            g = -eps_scaled(stage_eps, f, cost, log_a, dim=0)
            if used % check_every == 0 or used == iters:
                log_plan = (
                    (f.unsqueeze(1) + g.unsqueeze(0) - cost) / stage_eps
                    + log_a.unsqueeze(1) + log_b.unsqueeze(0)
                )
                row_marginal = torch.logsumexp(log_plan, dim=1).exp()
                stage_err = (row_marginal - ad).abs().max().item()
                if stage_err < tol:
                    break
        return stage_err, used

    # Epsilon annealing: plain Sinkhorn at small eps converges far too slowly (marginal
    # error stalls near 2.5e-4 at eps=1e-3 even after 20k iterations). Solving at a large
    # eps and walking down, warm-starting (f, g) each stage, reaches a much tighter
    # solution at the target eps in a fraction of the iterations.
    schedule = []
    stage_eps = max(eps, 1.0)
    while stage_eps > eps * 1.001:
        schedule.append(stage_eps)
        stage_eps *= 0.5
    schedule.append(eps)

    it = 0
    for stage_eps in schedule[:-1]:
        _, used = sweep(stage_eps, max(check_every, 200))
        it += used
    err, used = sweep(eps, max_iter)
    it += used

    if err >= tol:
        print(
            f"  [warn] reference Sinkhorn hit max_iter={max_iter} at eps={eps:g} "
            f"with marginal error {err:.3e} > tol={tol:g}; rmae_pct will be unreliable"
        )
    elif verbose:
        print(f"  reference converged in {it} iters (marginal err {err:.3e})")

    return float((ad * f).sum() + (bd * g).sum())


_ref_cost_cache: Dict[str, float] = {}

_REF_COST_CACHE_PATH = Path.home() / ".cache" / "sinkslot" / "entropic_ot_reference.json"

# Bump whenever the benchmark's (x, y, a, b) generation changes -- seed, distribution,
# or draw order -- or when the reference functional itself changes. Cached values from
# an older scheme would silently produce wrong rmae_pct values, so a mismatch discards
# the whole file.
_REF_COST_CACHE_VERSION = 1


def _load_ref_cost_cache() -> None:
    """Populate the in-process cache from disk, ignoring stale or unreadable files."""
    try:
        blob = json.loads(_REF_COST_CACHE_PATH.read_text())
    except (OSError, ValueError):
        return
    if blob.get("version") == _REF_COST_CACHE_VERSION:
        _ref_cost_cache.update(blob.get("costs", {}))


def _cached_entropic_ot_reference(
    n: int, m: int, d: int, eps: float,
    x: torch.Tensor, y: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
    *, dataset: str = "gaussian",
) -> float:
    """Memoized entropic-OT reference, keyed by (dataset, n, m, d, eps), cached on disk.

    All PyTorch-side benchmark functions seed with torch.manual_seed(0) before drawing
    (x, y, a, b), so every method at a given (dataset, n, m, d) shares one point cloud
    and can share one reference solve. Unlike the old exact-OT cost this DOES depend on
    eps, so the sweep pays one CPU solve per (dataset, n, m, d, eps) rather than one per
    point cloud. Cached across processes so re-runs and --compare-tf32 are free.

    Keys are strings because JSON cannot key on tuples.
    """
    key = f"{dataset},{n},{m},{d},{eps:g}"
    if key in _ref_cost_cache:
        return _ref_cost_cache[key]

    if not _ref_cost_cache:
        _load_ref_cost_cache()
        if key in _ref_cost_cache:
            return _ref_cost_cache[key]

    _ref_cost_cache[key] = compute_entropic_ot_reference(x, y, a, b, eps)
    try:
        _REF_COST_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _REF_COST_CACHE_PATH.write_text(json.dumps(
            {"version": _REF_COST_CACHE_VERSION, "costs": _ref_cost_cache}
        ))
    except OSError as e:
        # A read-only or full cache dir shouldn't fail the benchmark -- just recompute.
        print(f"  [warn] could not write reference cache to {_REF_COST_CACHE_PATH}: {e}")
    return _ref_cost_cache[key]


# =============================================================================
# Exact (unregularized) OT reference -- cost_gap / barycentric_sym metrics
# =============================================================================
# Unlike compute_entropic_ot_reference (converged entropic dual AT THE SAME eps),
# this is literal exact OT: solved once per (dataset, n, m, d, seed) via POT's
# ot.emd (network simplex, CPU, float64) and shared across every eps, since exact
# OT doesn't depend on eps at all. Empirically ~O(n^2), not the O(n^3) worst case
# folklore suggests -- ~21.5s at n=10,000 on this data (see PR discussion), far
# cheaper than the annealed entropic reference above despite being the exact
# problem, not an approximation of it.

from dataclasses import dataclass as _dataclass


@_dataclass
class ExactRef:
    """Exact (unregularized) OT reference for one (dataset, n, m, d, seed) instance."""
    cost: float               # <T*, C>, the exact optimal transport cost
    bary_x: "np.ndarray"      # [n, d]: bary_x[i] = sum_j T*_ij y_j / a_i
    bary_y: "np.ndarray"      # [m, d]: bary_y[j] = sum_i T*_ij x_i / b_j


def compute_exact_ot_reference(
    x: torch.Tensor, y: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
) -> ExactRef:
    """Solve exact OT via POT's ot.emd (network simplex) on CPU in float64."""
    import numpy as np
    import ot

    x_np = x.detach().cpu().numpy().astype("float64")
    y_np = y.detach().cpu().numpy().astype("float64")
    a_np = a.detach().cpu().numpy().astype("float64")
    b_np = b.detach().cpu().numpy().astype("float64")
    cost_matrix = ot.dist(x_np, y_np, metric="sqeuclidean")

    plan = ot.emd(a_np, b_np, cost_matrix, numItermax=int(1e8))
    cost = float((plan * cost_matrix).sum())
    bary_x = plan @ y_np / np.clip(a_np, 1e-300, None)[:, None]
    bary_y = plan.T @ x_np / np.clip(b_np, 1e-300, None)[:, None]
    return ExactRef(cost=cost, bary_x=bary_x, bary_y=bary_y)


_exact_ref_cache: Dict[str, ExactRef] = {}

_EXACT_REF_CACHE_DIR = Path.home() / ".cache" / "sinkslot" / "exact_ot_reference"

# Bump whenever (x, y, a, b) generation changes (seed, distribution, draw order) --
# a stale cache would silently corrupt every cost_gap/barycentric_sym.
_EXACT_REF_CACHE_VERSION = 1

# compute_exact_ot_reference is inherently dense regardless of which method calls
# it: ot.dist(x_np, y_np) builds a full (n, m) float64 cost matrix on CPU for
# ot.emd's network simplex, even for methods (SinkSLOT/-CUDA) whose own solve is
# sparse. At n=m=50,000 that's 20GB; at 100,000 it's 80GB -- confirmed on-cluster
# to OOM-kill every method uniformly (flash_symmetric/alternating, sinkslot,
# sinkslotcuda all failed identically at n>=50,000 in the N-scaling run,
# all traced to this shared call, not to any one method's own solver). Every
# rmae_check gate below also checks n <= this bound, so cost_gap_pct/
# barycentric_sym/mass/marg_viol are simply not computed above it -- timing and
# memory are still recorded normally.
_EXACT_OT_MAX_N = 10000


def _cached_exact_ot_reference(
    n: int, m: int, d: int, seed: int,
    x: torch.Tensor, y: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
    *, dataset: str = "gaussian",
) -> ExactRef:
    """Memoized exact-OT reference, keyed by (dataset, n, m, d, seed) -- NOT eps.

    Cached as .npz (unlike the scalar entropic reference's JSON cache) since this
    also needs to persist the bary_x/bary_y arrays, not just a float.
    """
    import numpy as np

    key = f"{dataset}_n{n}_m{m}_d{d}_seed{seed}_v{_EXACT_REF_CACHE_VERSION}"
    if key in _exact_ref_cache:
        return _exact_ref_cache[key]

    cache_path = _EXACT_REF_CACHE_DIR / f"{key}.npz"
    if cache_path.exists():
        try:
            blob = np.load(cache_path)
            ref = ExactRef(cost=float(blob["cost"]), bary_x=blob["bary_x"], bary_y=blob["bary_y"])
            _exact_ref_cache[key] = ref
            return ref
        except (OSError, ValueError, KeyError):
            pass  # corrupt/partial cache file -- recompute below

    ref = compute_exact_ot_reference(x, y, a, b)
    _exact_ref_cache[key] = ref
    try:
        _EXACT_REF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, cost=ref.cost, bary_x=ref.bary_x, bary_y=ref.bary_y)
    except OSError as e:
        print(f"  [warn] could not write exact-OT reference cache to {cache_path}: {e}")
    return ref


def cost_gap(cost: float, ref: ExactRef) -> float:
    """Relative excess transport cost, in percent.

    Invariant to the cost convention: scaling C by lambda scales both terms.
    """
    return 100.0 * (cost - ref.cost) / abs(ref.cost)


def barycentric_sym(
    Tx: torch.Tensor, Ty: torch.Tensor, ref: ExactRef, a: torch.Tensor, b: torch.Tensor,
) -> float:
    """Weighted squared deviation of both barycentric projections from the exact ones."""
    rx = torch.as_tensor(ref.bary_x, dtype=Tx.dtype, device=Tx.device)
    ry = torch.as_tensor(ref.bary_y, dtype=Ty.dtype, device=Ty.device)
    ex = float((a * ((Tx - rx) ** 2).sum(1)).sum())
    ey = float((b * ((Ty - ry) ** 2).sum(1)).sum())
    return ex + ey


def plan_barycentric_dense(
    T: torch.Tensor, x: torch.Tensor, y: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Barycentric projections (Tx, Ty) of a dense plan T [n, m], plus its achieved
    marginals (r, c) -- returned rather than discarded so callers can record
    mass/marg_viol (see plan_feasibility) instead of just consuming the normalization.

    Normalizes by T's own ACHIEVED marginals (row/col sums), not the target a, b --
    the identity Tx_i = sum_j T_ij y_j / r_i stays exact even when the solve has not
    fully converged (r != a). Dividing by the target a instead would silently corrupt
    the projection by exactly the row's convergence error, and did in an earlier
    version of this function -- see the SLOT repo's PotentialResult.barycentric()
    docstring for the same fix.
    """
    tiny = torch.finfo(T.dtype).tiny
    r = T.sum(1)
    c = T.sum(0)
    Tx = (T @ y) / r.clamp_min(tiny).unsqueeze(1)
    Ty = (T.t() @ x) / c.clamp_min(tiny).unsqueeze(1)
    return Tx, Ty, r, c


def plan_barycentric_sparse(
    T_vals: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor,
    x: torch.Tensor, y: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Barycentric projections (Tx, Ty) of a sparse plan given as (rows, cols, T_vals),
    plus its achieved marginals (r, c) -- see plan_barycentric_dense's docstring.

    Normalizes by the plan's own ACHIEVED marginals (scatter-summed from T_vals), not
    the target a, b -- see plan_barycentric_dense's docstring for why.
    """
    n, d = x.shape
    m = y.shape[0]
    tiny = torch.finfo(T_vals.dtype).tiny
    r = torch.zeros(n, device=x.device, dtype=T_vals.dtype).index_add_(0, rows, T_vals)
    c = torch.zeros(m, device=y.device, dtype=T_vals.dtype).index_add_(0, cols, T_vals)
    Tx = torch.zeros(n, d, device=x.device, dtype=T_vals.dtype).index_add_(
        0, rows, T_vals.unsqueeze(1) * y[cols])
    Ty = torch.zeros(m, d, device=y.device, dtype=T_vals.dtype).index_add_(
        0, cols, T_vals.unsqueeze(1) * x[rows])
    Tx = Tx / r.clamp_min(tiny).unsqueeze(1)
    Ty = Ty / c.clamp_min(tiny).unsqueeze(1)
    return Tx, Ty, r, c


def plan_feasibility(r: torch.Tensor, c: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> dict:
    """mass, marg_viol, plan_empty_rows/cols from a plan's achieved marginals (r, c)
    vs its targets (a, b). Matches the SLOT repo's bench/run.py convention exactly
    (mass = r.sum(); marg_viol = max(||r-a||_inf, ||c-b||_inf)) -- computed the same
    way regardless of stop mode, unlike final_viol which is mode-dependent.
    """
    return {
        "mass": float(r.sum()),
        "marg_viol": max(float((r - a).abs().max()), float((c - b).abs().max())),
        "plan_empty_rows": int((r == 0).sum()),
        "plan_empty_cols": int((c == 0).sum()),
    }


@_dataclass
class StopCfg:
    """Early-stopping configuration threaded into the solver loops.

    mode="fixed" runs exactly n_iters. mode="marginal" runs up to max_iter,
    stopping when the max (L-infinity) row/col marginal violation <= tol --
    srot/sinkslot/sinkslotcuda/spar_sink/rand_sink's proven default (see SLOT
    repo), and now also implemented natively in FlashSinkhorn's own solvers
    (sinkhorn_solvers.py) for flash_symmetric/flash_alternating, so it's
    directly comparable across every method. Not gated on mass_tol -- SLOT's
    own working rule doesn't check mass separately either. mode="potential" stops on
    ||du||_1+||dv||_1 <= tol (Spar-Sink's rule; no FlashSinkhorn equivalent --
    Flash falls back to potential_linf for this mode). mode="potential_linf" stops
    once the dual potentials themselves stop moving, max(|Δf|, |Δg|) < tol since
    the last check -- FlashSinkhorn's own native rule (see sinkhorn_solvers.py),
    reproduced verbatim for srot/sinkslot/sinkslotcuda/spar_sink/rand_sink so every
    method can share the identical stopping rule, check frequency and threshold.
    Checked every `check_every` iterations. `fixed(n)` is the no-stopping default.

    Known asymmetry: the two FlashSinkhorn backends aren't equally cheap to check
    under mode="marginal". flash_alternating's Gauss-Seidel structure satisfies one
    marginal exactly by construction after each update (only the other needs a
    fresh check, one extra reduction call); flash_symmetric's damped Jacobi
    averaging leaves both marginals inexact after every update, so both need a
    fresh check. Both are still cheaper than a full extra Sinkhorn iteration, and
    both are only paid at check_every intervals, not every iteration.
    """
    mode: str = "fixed"
    max_iter: int = 10000
    tol: float = 1e-4
    potential_tol: float = 1e-6
    mass_tol: float = 1e-6
    check_every: int = 10

    @staticmethod
    def fixed() -> "StopCfg":
        return StopCfg(mode="fixed")


def _marginal_tv(row_marg: torch.Tensor, col_marg: torch.Tensor,
                 a: torch.Tensor, b: torch.Tensor) -> float:
    """Total-variation marginal violation sum|P1 - a| + |P^T1 - b|.

    L1/TV rather than max: it lives in [0, 4] regardless of n, so a fixed
    threshold means the same relative accuracy at every problem size (an absolute
    max threshold loosens as n grows, inverting cross-n timing).
    """
    return float((row_marg - a).abs().sum() + (col_marg - b).abs().sum())


def _rmae_pct(loss_value: float, reference: float) -> float:
    """Relative mean absolute error, in percent, against the converged entropic solve."""
    return abs(loss_value - reference) / max(abs(reference), 1e-12) * 100.0


def gpu_memory_used_mb(device: torch.device) -> float:
    """GPU memory in use on the device, in MB -- the number nvidia-smi / nvitop shows.

    `torch.cuda.mem_get_info()` returns (free, total) straight from the CUDA driver, so
    total - free is the whole-device figure: CUDA context, the caching allocator's
    reserved pool, compiled Triton/KeOps kernels, cuBLAS workspaces -- everything, not
    just live PyTorch tensors. It therefore also captures KeOps, which allocates outside
    PyTorch's allocator and is invisible to torch.cuda.max_memory_allocated().

    Device-level, not per-process: it includes any other process on the same GPU. Runs
    are pinned to a dedicated device (CUDA_VISIBLE_DEVICES), so in practice this is our
    process.

    Calls empty_cache() first. PyTorch's caching allocator keeps freed blocks rather than
    returning them to the driver, so without this the figure is dominated by allocator
    hysteresis -- Triton autotune alone leaves ~270MB pooled-but-unused, which swamps the
    actual data at any size this benchmark runs. Releasing the unused portion leaves CUDA
    context + live tensors + compiled kernels: the footprint a method actually needs.
    This reports memory still held at the end of the run, not the peak reached.

    Still run each measurement in its own process (BenchConfig.isolate): compiled kernels
    and context are not freed by empty_cache(), so they accumulate across methods.
    """
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(device)
    return (total - free) / 1e6


DATASET_CHOICES = ("gaussian", "8gaussians", "half_moon", "two_rings")

_EIGHT_GAUSSIANS_RADIUS = 2.0
_EIGHT_GAUSSIANS_STD = 0.18
_HALF_MOON_NOISE = 0.06
_TWO_RINGS_RADIUS_SOURCE = 1.0
_TWO_RINGS_RADIUS_TARGET = 2.0
_TWO_RINGS_NOISE = 0.04
_EXTRA_DIM_NOISE = 0.03  # i.i.d. filler for dims beyond the 2D generative structure


def sample_point_cloud(
    n: int, d: int, device: torch.device, *, dataset: str = "gaussian", target: bool = False,
) -> torch.Tensor:
    """Draw an (n, d) synthetic point cloud for benchmarking.

    dataset="gaussian" (default): isotropic standard normal, N(0, I_d).

    dataset="8gaussians": 8 clusters on a radius-2 ring, 45 degrees apart, with
    std=0.18 isotropic noise per cluster (matches khainb/SROT's setup). The
    source ring (target=False) starts at angle 0; the target ring (target=True)
    is offset by 22.5 degrees so the two rings don't trivially line up. Only
    the first 2 dims carry ring structure; any remaining dims are i.i.d. noise.

    dataset="half_moon": two noisy interleaving half-moons. The source
    (target=False) is the upper moon x=[cos(t), sin(t)] + noise, t~U(0, pi);
    the target (target=True) is the lower, offset moon
    y=[1-cos(t), 0.5-sin(t)] + noise. Only the first 2 dims carry moon
    structure; any remaining dims are i.i.d. noise.

    dataset="two_rings": concentric noisy rings sharing a center, source
    radius 1.0 (target=False) vs target radius 2.0 (target=True). Only the
    first 2 dims carry ring structure; any remaining dims are i.i.d. noise.
    """
    if dataset == "gaussian":
        return torch.randn(n, d, device=device, dtype=torch.float32)

    if dataset not in DATASET_CHOICES:
        raise ValueError(f"Unknown dataset: {dataset!r}. Choices: {DATASET_CHOICES}")
    if d < 2:
        raise ValueError(f"dataset={dataset!r} requires d >= 2")

    if dataset == "8gaussians":
        offset = math.pi / 8 if target else 0.0
        angles = offset + torch.arange(8, device=device, dtype=torch.float32) * (math.pi / 4)
        centers = _EIGHT_GAUSSIANS_RADIUS * torch.stack([angles.cos(), angles.sin()], dim=1)  # (8, 2)

        cluster_idx = torch.randint(0, 8, (n,), device=device)
        points = torch.randn(n, d, device=device, dtype=torch.float32) * _EIGHT_GAUSSIANS_STD
        points[:, :2] = points[:, :2] + centers[cluster_idx]
        return points

    if dataset == "half_moon":
        t = torch.rand(n, device=device, dtype=torch.float32) * math.pi
        if not target:
            xy = torch.stack([t.cos(), t.sin()], dim=1)
        else:
            xy = torch.stack([1.0 - t.cos(), 0.5 - t.sin()], dim=1)
        xy = xy + _HALF_MOON_NOISE * torch.randn(n, 2, device=device, dtype=torch.float32)
    else:  # "two_rings"
        t = torch.rand(n, device=device, dtype=torch.float32) * (2 * math.pi)
        radius = _TWO_RINGS_RADIUS_TARGET if target else _TWO_RINGS_RADIUS_SOURCE
        xy = radius * torch.stack([t.cos(), t.sin()], dim=1)
        xy = xy + _TWO_RINGS_NOISE * torch.randn(n, 2, device=device, dtype=torch.float32)

    if d == 2:
        return xy

    points = torch.empty(n, d, device=device, dtype=torch.float32)
    points[:, :2] = xy
    points[:, 2:] = _EXTRA_DIM_NOISE * torch.randn(n, d - 2, device=device, dtype=torch.float32)
    return points


# =============================================================================
# SROT: Sliced-Regularized Optimal Transport (baseline)
# =============================================================================
# Nguyen, "Sliced-Regularized Optimal Transport", arXiv:2604.23944
# Reference implementation: https://github.com/khainb/SROT
#
# Written here from the algorithm as described rather than vendored. SROT replaces
# the entropic regularizer's reference measure: standard Sinkhorn penalizes
# KL(pi || a (x) b) and so uses the kernel a (x) b * exp(-C/eps), whereas SROT
# penalizes KL(pi || pi_SOT) with pi_SOT the uniform average of L one-dimensional
# OT plans taken on random projections. It is therefore a different optimum, not a
# faster route to the same one -- see compute_srot_reference().


def build_sot_plan(
    x: torch.Tensor, y: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
    *, slices: int, delta: float = 1e-8, seed: int = 0,
) -> torch.Tensor:
    """Uniform-average sliced-OT reference plan from `slices` random 1-D projections.

    Adapted from the authors' released SROT code (github.com/khainb/SROT) onto
    this harness; the reference coupling below is theirs.

    Returns (1 - delta) * pi_SOT + delta * (a (x) b). The delta mix keeps every entry
    strictly positive when a, b > 0, so the Sinkhorn kernel has full support.

    Each slice projects both clouds onto a random unit direction and solves the 1-D OT
    problem, which for a convex ground cost is exactly the north-west corner rule on the
    sorted marginals: with cumulative masses ca and cb, the plan entry is the overlap
    max(0, min(ca_i, cb_j) - max(ca_{i-1}, cb_{j-1})). That is computed densely here --
    the plan is O(n*m) by construction, which is why this baseline is gated behind
    --max-dense-size.

    The projection RNG is seeded explicitly so that a benchmarked run and its converged
    reference share the same pi_SOT; otherwise rmae_pct would be measuring a difference
    of random directions rather than iteration error.

    Computed in float64 -- cumulative marginals are exactly where float32 accumulates
    error over n terms -- then returned in `x`'s dtype. This is one-off setup, not the
    timed solve loop, so the precision costs nothing in the comparison.
    """
    n, d = x.shape
    m = y.shape[0]
    device = x.device
    xd, yd = x.double(), y.double()
    ad, bd = a.double(), b.double()

    generator = torch.Generator(device=device).manual_seed(seed)
    thetas = torch.randn(slices, d, generator=generator, device=device, dtype=torch.float64)
    thetas = thetas / thetas.norm(dim=1, keepdim=True).clamp_min(1e-300)

    px_all = xd @ thetas.T  # (n, L)
    py_all = yd @ thetas.T  # (m, L)

    pi_sot = torch.zeros(n, m, device=device, dtype=torch.float64)
    for ell in range(slices):
        order_x = torch.argsort(px_all[:, ell])
        order_y = torch.argsort(py_all[:, ell])

        ca = torch.cumsum(ad[order_x], dim=0)
        cb = torch.cumsum(bd[order_y], dim=0)
        ca_prev = torch.cat([ca.new_zeros(1), ca[:-1]])
        cb_prev = torch.cat([cb.new_zeros(1), cb[:-1]])

        upper = torch.minimum(ca.unsqueeze(1), cb.unsqueeze(0))
        lower = torch.maximum(ca_prev.unsqueeze(1), cb_prev.unsqueeze(0))
        overlap = (upper - lower).clamp_min(0.0)

        pi_sot[order_x.unsqueeze(1), order_y.unsqueeze(0)] += overlap

    pi_sot /= slices
    if delta > 0.0:
        pi_sot = (1.0 - delta) * pi_sot + delta * torch.outer(ad, bd)
    return pi_sot.to(x.dtype)


def _srot_sinkhorn(
    cost: torch.Tensor, log_pi: torch.Tensor, log_a: torch.Tensor, log_b: torch.Tensor,
    eps: float, n_iters: int, stop: "StopCfg" = None,
):
    """`n_iters` log-domain Sinkhorn sweeps against the pi_SOT reference plan.

    Fixed point is pi = pi_SOT * exp((f (+) g - C)/eps) with marginals a, b, giving

        f_i = eps * [log a_i - logsumexp_j(log pi_ij + (g_j - C_ij)/eps)]
        g_j = eps * [log b_j - logsumexp_i(log pi_ij + (f_i - C_ij)/eps)]

    which reduces to the standard updates when pi_SOT = a (x) b.

    Returns (f, g, iters_run, converged, final_viol). stop None / "fixed" runs
    n_iters. "marginal"/"potential" run to stop.max_iter, stopping on the max
    (L-infinity) marginal violation (row marginal = exp(f/eps + LSE_row(g));
    col is exactly b) -- matches the SLOT repo's actual working "marg_viol"
    rule, not a total-variation sum (which is unreachable at n=10,000
    regardless of convergence).
    "potential_linf" reproduces FlashSinkhorn's own native rule exactly: stop once
    the dual potentials themselves stop moving, max(|Δf|, |Δg|) < stop.tol, measured
    since the last check (not the last iteration) -- see the identical check in
    sinkhorn_solvers.py. f, g here are already the standard (non-absorbed) potentials,
    same scale as FlashSinkhorn's unshifted f, g, so stop.tol means the same thing.
    """
    f = torch.zeros_like(log_a)
    g = torch.zeros_like(log_b)

    def _row_lse(gv):
        return torch.logsumexp(log_pi + (gv.unsqueeze(0) - cost) / eps, dim=1)

    def _col_lse(fv):
        return torch.logsumexp(log_pi + (fv.unsqueeze(1) - cost) / eps, dim=0)

    mode = getattr(stop, "mode", "fixed") if stop is not None else "fixed"

    if mode == "fixed":
        for _ in range(n_iters):
            f = eps * (log_a - _row_lse(g))
            g = eps * (log_b - _col_lse(f))
        return f, g, n_iters, None, None

    if mode == "potential_linf":
        prev_f, prev_g = f, g
        it = 0
        converged = False
        change = float("inf")
        while it < stop.max_iter:
            f = eps * (log_a - _row_lse(g))
            g = eps * (log_b - _col_lse(f))
            it += 1
            if it % stop.check_every == 0:
                change = max((f - prev_f).abs().max().item(), (g - prev_g).abs().max().item())
                if change < stop.tol:
                    converged = True
                    break
                prev_f = f
                prev_g = g
        return f, g, it, converged, change

    a = log_a.exp()
    b = log_b.exp()
    f_old, g_old = f, g
    it = 0
    converged = False
    viol = float("inf")
    while it < stop.max_iter:
        f_old = f
        g_old = g
        f = eps * (log_a - _row_lse(g))
        g = eps * (log_b - _col_lse(f))
        it += 1
        if it % stop.check_every == 0 or it == stop.max_iter:
            # Check both row and column marginal violation, without an extra
            # _row_lse call.
            row_marg = a * ((f_old - f) / eps).exp()
            col_marg = b * ((g_old - g) / eps).exp()
            # max (L-infinity), not sum: matches the SLOT repo's actual working
            # "marg_viol" rule (bench/solvers/sinkslot.py's _violation/_run_v5).
            # A sum over n terms against a fixed absolute tol is unreachable at
            # n=10,000 regardless of convergence -- SLOT's own ConvergenceCfg
            # documents max as the n-invariant criterion. Not gated on mass_tol
            # either, matching SLOT exactly.
            viol = float(torch.maximum((row_marg - a).abs().max(), (col_marg - b).abs().max()))
            if viol <= stop.tol:
                converged = True
                break
    return f, g, it, converged, viol


_srot_ref_cache: Dict[str, float] = {}

_SROT_REF_CACHE_PATH = Path.home() / ".cache" / "sinkslot" / "srot_reference.json"
_SROT_REF_CACHE_VERSION = 1


def compute_srot_reference(
    cost: torch.Tensor, log_pi: torch.Tensor, log_a: torch.Tensor, log_b: torch.Tensor,
    a: torch.Tensor, b: torch.Tensor, eps: float,
    *, max_iter: int = 20000, tol: float = 1e-6, check_every: int = 10,
) -> float:
    """Converged SROT dual objective -- the RMAE reference for SROT rows.

    SROT cannot share the entropic reference used by the flash/GeomLoss rows: it
    minimizes <pi, C> + eps*KL(pi || pi_SOT) rather than <pi, C> + eps*KL(pi || a (x) b),
    so it converges somewhere else by design. Measuring it against the entropic optimum
    would report that design difference as if it were solver error, and it would be
    nonzero even for a perfectly converged run. Giving SROT its own converged reference
    keeps rmae_pct meaning the same thing in every row: distance from the optimum of the
    problem this method actually solves.

    Also keyed by L, since pi_SOT -- and therefore the optimum -- changes with it.

    Same eps annealing and marginal-violation stopping rule as
    compute_entropic_ot_reference(); see that docstring for why annealing is needed.
    """
    f = torch.zeros_like(log_a)
    g = torch.zeros_like(log_b)

    def sweep(stage_eps: float, iters: int) -> Tuple[float, int]:
        nonlocal f, g
        stage_err = float("inf")
        used = 0
        for used in range(1, iters + 1):
            f = stage_eps * (log_a - torch.logsumexp(log_pi + (g.unsqueeze(0) - cost) / stage_eps, dim=1))
            g = stage_eps * (log_b - torch.logsumexp(log_pi + (f.unsqueeze(1) - cost) / stage_eps, dim=0))
            if used % check_every == 0 or used == iters:
                log_plan = log_pi + (f.unsqueeze(1) + g.unsqueeze(0) - cost) / stage_eps
                row = torch.logsumexp(log_plan, dim=1).exp()
                stage_err = (row - a).abs().max().item()
                if stage_err < tol:
                    break
        return stage_err, used

    schedule = []
    stage_eps = max(eps, 1.0)
    while stage_eps > eps * 1.001:
        schedule.append(stage_eps)
        stage_eps *= 0.5
    for stage_eps in schedule:
        sweep(stage_eps, max(check_every, 200))
    err, _ = sweep(eps, max_iter)

    if err >= tol:
        print(
            f"  [warn] SROT reference hit max_iter={max_iter} at eps={eps:g} with marginal "
            f"error {err:.3e} > tol={tol:g}; rmae_pct will be unreliable"
        )
    return float((a * f).sum() + (b * g).sum())


def _cached_srot_reference(
    n: int, m: int, d: int, eps: float, slices: int,
    cost: torch.Tensor, log_pi: torch.Tensor, log_a: torch.Tensor, log_b: torch.Tensor,
    a: torch.Tensor, b: torch.Tensor, *, dataset: str = "gaussian",
) -> float:
    """Memoized SROT reference, keyed by (dataset, n, m, d, eps, slices), cached on disk."""
    key = f"{dataset},{n},{m},{d},{eps:g},{slices}"
    if key in _srot_ref_cache:
        return _srot_ref_cache[key]

    if not _srot_ref_cache:
        try:
            blob = json.loads(_SROT_REF_CACHE_PATH.read_text())
            if blob.get("version") == _SROT_REF_CACHE_VERSION:
                _srot_ref_cache.update(blob.get("costs", {}))
        except (OSError, ValueError):
            pass
        if key in _srot_ref_cache:
            return _srot_ref_cache[key]

    _srot_ref_cache[key] = compute_srot_reference(cost, log_pi, log_a, log_b, a, b, eps)
    try:
        _SROT_REF_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SROT_REF_CACHE_PATH.write_text(json.dumps(
            {"version": _SROT_REF_CACHE_VERSION, "costs": _srot_ref_cache}
        ))
    except OSError as e:
        print(f"  [warn] could not write SROT reference cache to {_SROT_REF_CACHE_PATH}: {e}")
    return _srot_ref_cache[key]


def bench_with_stats(
    fn: Callable[[], None],
    warmup: int = 10,
    rep: int = 50,
    *,
    nvtx: bool = False,
    nvtx_label: Optional[str] = None,
) -> Tuple[float, float, float, float, float]:
    """Benchmark with full statistics.

    Returns: (mean_ms, std_ms, min_ms, max_ms, median_ms)
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    with _nvtx_range(
        f"{nvtx_label}/timed" if nvtx_label else "timed",
        enabled=bool(nvtx and nvtx_label),
    ):
        times = []
        for _ in range(rep):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

    times_t = torch.tensor(times)
    return (
        times_t.mean().item(),
        times_t.std().item(),
        times_t.min().item(),
        times_t.max().item(),
        times_t.median().item(),
    )


# =============================================================================
# FlashSinkhorn Benchmarks
# =============================================================================

def bench_flashsinkhorn(
    n: int, m: int, d: int, eps: float, n_iters: int,
    device: torch.device, warmup: int, rep: int,
    *,
    nvtx: bool = False,
    backend: str = "symmetric",
    allow_tf32: bool = False,
    rmae_check: bool = True,
    dataset: str = "gaussian",
    stop: "StopCfg" = None,
    seed: int = 0,
) -> TimingResult:
    """Benchmark FlashSinkhorn with fixed iterations.

    Args:
        backend: "symmetric" (GeomLoss-style) or "alternating" (OTT-JAX-style)
        allow_tf32: Enable TF32 for ~2x speedup (default: False for strict fp32)
        dataset: "gaussian" (default), "8gaussians", "half_moon", or "two_rings"; see sample_point_cloud().
        seed: data-generation seed only (x, y, a, b). Does not affect any
            method-internal randomness (e.g. SROT/SinkSLOT's slice projections,
            Spar-Sink's kernel sampling), which stay independently seeded.

    Uses full squared Euclidean cost C(x,y) = ||x-y||² (half_cost=False default).
    Autotuning is enabled for best Triton kernel performance (~2-3s first call overhead).
    """
    from flash_sinkhorn import SamplesLoss

    torch.manual_seed(seed)
    x = sample_point_cloud(n, d, device, dataset=dataset, target=False)
    y = sample_point_cloud(m, d, device, dataset=dataset, target=True)
    a = torch.rand(n, device=device, dtype=torch.float32) + 0.1
    b = torch.rand(m, device=device, dtype=torch.float32) + 0.1
    a = a / a.sum()
    b = b / b.sum()

    _stop = stop or StopCfg.fixed()
    if _stop.mode == "fixed":
        _fs_kwargs = {}
    else:
        # FlashSinkhorn's own SamplesLoss only knows "potential_linf" (its native
        # rule) and "marginal" (added to match SROT/SinkSLOT/Spar-Sink's proven
        # convention). Spar-Sink's own "potential" mode (max change in the scaling
        # variable, a different quantity) has no Flash equivalent -- fall back to
        # potential_linf for it, preserving this function's pre-existing behavior
        # for that mode (it never distinguished "potential" from "potential_linf").
        _fs_stop_mode = "marginal" if _stop.mode == "marginal" else "potential_linf"
        _fs_kwargs = {
            "threshold": _stop.tol, "inner_iterations": _stop.check_every,
            "stop_mode": _fs_stop_mode, "mass_tol": _stop.mass_tol,
        }
    _fs_iters = n_iters if _stop.mode == "fixed" else _stop.max_iter
    loss_fn = SamplesLoss(
        "sinkhorn",
        backend=backend,
        use_epsilon_scaling=False,
        eps=eps,
        n_iters=_fs_iters,
        debias=False,
        potentials=False,
        normalize=False,
        autotune=True,  # Enable Triton kernel tuning (~2-3s first call overhead)
        last_extrapolation=False,  # Match GeomLoss benchmark setting
        allow_tf32=allow_tf32,
        **_fs_kwargs,
    )

    method_name = f"flash_{backend}"

    def run():
        _ = loss_fn(a, x, b, y)

    try:
        # Trigger Triton JIT compilation + autotuning here, outside the peak-memory
        # window, so the one-time compile overhead doesn't get counted as steady-state
        # memory (matches the GeomLoss/KeOps benchmark's pre-JIT warmup below).
        loss_fn(a, x, b, y)
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        return TimingResult(method_name, n, m, d, eps, float("inf"), 0, 0, 0, 0, 0, oom=True, n_iters=n_iters)

    cost_gap_pct = None
    bary = None
    feas: dict = {}
    iters_run = n_iters if _stop.mode == "fixed" else None
    converged = None
    hit_max_iters = None
    f = g = None

    # Iteration tracking is decoupled from the cost_gap machinery below: the
    # timed run() above goes through _SinkhornCostFn.apply (an autograd.Function,
    # tensors-only return), so return_n_iters has no side channel there -- but
    # this untimed potentials call is O(n) (no dense n*m materialization), so
    # it's safe to run even at N where the dense plan reconstruction below would
    # OOM. Always get it when early stopping is in play, independent of rmae_check.
    if _stop.mode != "fixed":
        potentials_fn = SamplesLoss(
            "sinkhorn", backend=backend, use_epsilon_scaling=False, eps=eps,
            n_iters=_fs_iters, debias=False, potentials=True, normalize=False,
            autotune=True, last_extrapolation=False, allow_tf32=allow_tf32,
            return_n_iters=True,
            **_fs_kwargs,
        )
        f, g, iters_run = potentials_fn(a, x, b, y)
        converged = iters_run < _stop.max_iter
        hit_max_iters = iters_run >= _stop.max_iter

    if rmae_check and n <= _EXACT_OT_MAX_N:  # kept as the enable/disable flag name; now gates cost_gap/barycentric_sym
        # Flash never materializes a plan (the point of its streaming kernels), so
        # cost_gap/barycentric_sym need (f, g) from an UNTIMED potentials call, then
        # a dense O(n*m) plan materialized post-hoc, outside the timed region.
        # Reuse the potentials already computed above if we have them (deterministic
        # solve, so identical to a fresh call); only fixed mode needs one here.
        if f is None:
            potentials_fn = SamplesLoss(
                "sinkhorn", backend=backend, use_epsilon_scaling=False, eps=eps,
                n_iters=_fs_iters, debias=False, potentials=True, normalize=False,
                autotune=True, last_extrapolation=False, allow_tf32=allow_tf32,
                **_fs_kwargs,
            )
            f, g = potentials_fn(a, x, b, y)
        cost = torch.cdist(x, y, p=2) ** 2
        T = a.unsqueeze(1) * b.unsqueeze(0) * ((f.unsqueeze(1) + g.unsqueeze(0) - cost) / eps).exp()
        plan_cost = float((T * cost).sum())
        Tx, Ty, r, c = plan_barycentric_dense(T, x, y)
        ref = _cached_exact_ot_reference(n, m, d, seed, x, y, a, b, dataset=dataset)
        cost_gap_pct = cost_gap(plan_cost, ref)
        bary = barycentric_sym(Tx, Ty, ref, a, b)
        feas = plan_feasibility(r, c, a, b)

    try:
        # Measure peak memory during benchmark
        # Memory is reported as the whole-device figure nvidia-smi/nvitop would show
        # (see gpu_memory_mb), read after the timed loop. No allocator bookkeeping.
        mean, std, min_t, max_t, median = bench_with_stats(
            run,
            warmup,
            rep,
            nvtx=nvtx,
            nvtx_label=f"{method_name} n={n} d={d} eps={eps} iters={n_iters}",
        )
        gpu_memory_mb = gpu_memory_used_mb(device)
        return TimingResult(
            method_name, n, m, d, eps, mean, std, min_t, max_t, median, gpu_memory_mb, oom=False,
            n_iters=n_iters, cost_gap_pct=cost_gap_pct, barycentric_sym=bary,
            iters_run=iters_run, converged=converged, hit_max_iters=hit_max_iters, **feas,
        )
    except torch.cuda.OutOfMemoryError:
        return TimingResult(method_name, n, m, d, eps, float("inf"), 0, 0, 0, 0, 0, oom=True, n_iters=n_iters)


# =============================================================================
# GeomLoss Benchmarks
# =============================================================================

def bench_geomloss_online(
    n: int, m: int, d: int, eps: float, n_iters: int,
    device: torch.device, warmup: int, rep: int,
    *,
    nvtx: bool = False,
    rmae_check: bool = True,
    dataset: str = "gaussian",
    seed: int = 0,
) -> TimingResult:
    """Benchmark GeomLoss online (KeOps) with fixed iterations.

    Uses low-level `sinkhorn_loop` with `eps_list=[eps]*n_iters` to force exactly
    `n_iters` iterations (matching FlashSinkhorn / OTT-JAX settings).

    Cost convention: SqDist(X,Y) = ||x-y||² (full squared Euclidean, matches FlashSinkhorn).
    dataset: "gaussian" (default), "8gaussians", "half_moon", or "two_rings"; see sample_point_cloud().
    seed: data-generation seed only (x, y, a, b).
    """
    try:
        from pykeops.torch import generic_logsumexp  # noqa: F401 - needed by lse_genred
    except ImportError:
        return TimingResult("geomloss_online", n, m, d, eps, float("inf"), 0, 0, 0, 0, 0, oom=True, n_iters=n_iters)

    from geomloss._legacy.sinkhorn_divergence import log_weights, sinkhorn_cost, sinkhorn_loop
    from geomloss._legacy.sinkhorn_samples import lse_genred, softmin_online

    torch.manual_seed(seed)
    x = sample_point_cloud(n, d, device, dataset=dataset, target=False)
    y = sample_point_cloud(m, d, device, dataset=dataset, target=True)
    a = torch.rand(n, device=device, dtype=torch.float32) + 0.1
    b = torch.rand(m, device=device, dtype=torch.float32) + 0.1
    a = a / a.sum()
    b = b / b.sum()

    eps_list = [eps] * n_iters

    a_log = log_weights(a)
    b_log = log_weights(b)
    # SqDist(X,Y) = ||x-y||² (full squared Euclidean, matches FlashSinkhorn)
    my_lse = lse_genred("SqDist(X,Y)", d)
    softmin = partial(softmin_online, log_conv=my_lse)
    C_xy = (x, y.detach())
    C_yx = (y, x.detach())

    try:
        _, _, g_ab, f_ba = sinkhorn_loop(
            softmin, a_log, b_log, None, None,
            C_xy, C_yx, eps_list,
            rho=None, debias=False, last_extrapolation=False,
        )
        torch.cuda.synchronize()
    except Exception:
        return TimingResult("geomloss_online", n, m, d, eps, float("inf"), 0, 0, 0, 0, 0, oom=True, n_iters=n_iters)

    cost_gap_pct = None
    bary = None
    feas: dict = {}
    if rmae_check and n <= _EXACT_OT_MAX_N:  # kept as the enable/disable flag name; now gates cost_gap/barycentric_sym
        # sinkhorn_loop already returned dual potentials (g_ab, f_ba) from the SAME solve
        # used for timing -- no separate untimed call needed, unlike Flash. f_ba is the
        # source-side potential (on a's points), g_ab the target-side (on b's) -- same
        # GeomLoss convention bench_flashsinkhorn's "both report the dual" note refers to.
        # Both come back with a leading batch dim (1, n)/(1, m) even for this non-batched
        # call -- squeeze it before treating them as plain (n,)/(m,) potentials.
        f_ba_flat = f_ba.squeeze(0)
        g_ab_flat = g_ab.squeeze(0)
        cost = torch.cdist(x, y, p=2) ** 2
        T = a.unsqueeze(1) * b.unsqueeze(0) * ((f_ba_flat.unsqueeze(1) + g_ab_flat.unsqueeze(0) - cost) / eps).exp()
        plan_cost = float((T * cost).sum())
        Tx, Ty, r, c = plan_barycentric_dense(T, x, y)
        ref = _cached_exact_ot_reference(n, m, d, seed, x, y, a, b, dataset=dataset)
        cost_gap_pct = cost_gap(plan_cost, ref)
        bary = barycentric_sym(Tx, Ty, ref, a, b)
        feas = plan_feasibility(r, c, a, b)

    def run():
        sinkhorn_loop(
            softmin, a_log, b_log, None, None,
            C_xy, C_yx, eps_list,
            rho=None, debias=False, last_extrapolation=False,
        )

    try:
        # Measure peak memory during benchmark
        # Memory is reported as the whole-device figure nvidia-smi/nvitop would show
        # (see gpu_memory_mb), read after the timed loop. No allocator bookkeeping.
        mean, std, min_t, max_t, median = bench_with_stats(
            run,
            warmup,
            rep,
            nvtx=nvtx,
            nvtx_label=f"geomloss_online n={n} d={d} eps={eps} iters={n_iters}",
        )
        gpu_memory_mb = gpu_memory_used_mb(device)
        return TimingResult(
            "geomloss_online", n, m, d, eps, mean, std, min_t, max_t, median, gpu_memory_mb, oom=False,
            n_iters=n_iters, cost_gap_pct=cost_gap_pct, barycentric_sym=bary,
            iters_run=n_iters, converged=None, hit_max_iters=None,  # always fixed-iteration, no stopping check
            seed=seed, **feas,
        )
    except torch.cuda.OutOfMemoryError:
        return TimingResult("geomloss_online", n, m, d, eps, float("inf"), 0, 0, 0, 0, 0, oom=True,
                            n_iters=n_iters, seed=seed)


def bench_geomloss_tensorized(
    n: int, m: int, d: int, eps: float, n_iters: int,
    device: torch.device, warmup: int, rep: int,
    *,
    nvtx: bool = False,
    rmae_check: bool = True,
    dataset: str = "gaussian",
    seed: int = 0,
) -> TimingResult:
    """Benchmark GeomLoss tensorized (dense) with fixed iterations.

    Materializes O(n²) cost matrix in GPU memory.
    Cost convention: ||x-y||² (full squared Euclidean, matches FlashSinkhorn).
    dataset: "gaussian" (default), "8gaussians", "half_moon", or "two_rings"; see sample_point_cloud().
    seed: data-generation seed only (x, y, a, b).
    """
    from geomloss._legacy.sinkhorn_divergence import log_weights, sinkhorn_cost, sinkhorn_loop
    from geomloss._legacy.sinkhorn_samples import softmin_tensorized

    torch.manual_seed(seed)
    x = sample_point_cloud(n, d, device, dataset=dataset, target=False)
    y = sample_point_cloud(m, d, device, dataset=dataset, target=True)
    a = torch.rand(n, device=device, dtype=torch.float32) + 0.1
    b = torch.rand(m, device=device, dtype=torch.float32) + 0.1
    a = a / a.sum()
    b = b / b.sum()

    def _sqdist_cost(x_t: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        """Compute ||x-y||² (full squared Euclidean, matches FlashSinkhorn)."""
        x2 = (x_t * x_t).sum(dim=-1, keepdim=True)
        y2 = (y_t * y_t).sum(dim=-1, keepdim=True).transpose(-2, -1)
        return x2 + y2 - 2.0 * torch.matmul(x_t, y_t.transpose(-2, -1))

    eps_list = [eps] * n_iters

    C_xy = _sqdist_cost(x.unsqueeze(0), y.unsqueeze(0))
    C_yx = C_xy.transpose(-1, -2)

    a_log = log_weights(a).unsqueeze(0)
    b_log = log_weights(b).unsqueeze(0)

    softmin = partial(softmin_tensorized)

    try:
        _, _, g_ab, f_ba = sinkhorn_loop(
            softmin, a_log, b_log, None, None,
            C_xy, C_yx, eps_list,
            rho=None, debias=False, last_extrapolation=False,
        )
        loss_value = sinkhorn_cost(
            eps, None, a.unsqueeze(0), b.unsqueeze(0), None, None, g_ab, f_ba,
            batch=True, debias=False, potentials=False,
        ).item()
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        return TimingResult("geomloss_tensorized", n, m, d, eps, float("inf"), 0, 0, 0, 0, 0, oom=True, n_iters=n_iters)

    rmae_pct = None
    if rmae_check:
        reference = _cached_entropic_ot_reference(n, m, d, eps, x, y, a, b, dataset=dataset)
        rmae_pct = _rmae_pct(loss_value, reference)

    def run():
        sinkhorn_loop(
            softmin, a_log, b_log, None, None,
            C_xy, C_yx, eps_list,
            rho=None, debias=False, last_extrapolation=False,
        )

    try:
        # Measure peak memory during benchmark
        # Memory is reported as the whole-device figure nvidia-smi/nvitop would show
        # (see gpu_memory_mb), read after the timed loop. No allocator bookkeeping.
        mean, std, min_t, max_t, median = bench_with_stats(
            run,
            warmup,
            rep,
            nvtx=nvtx,
            nvtx_label=f"geomloss_tensorized n={n} d={d} eps={eps} iters={n_iters}",
        )
        gpu_memory_mb = gpu_memory_used_mb(device)
        return TimingResult(
            "geomloss_tensorized", n, m, d, eps, mean, std, min_t, max_t, median, gpu_memory_mb, oom=False,
            n_iters=n_iters, rmae_pct=rmae_pct,
        )
    except torch.cuda.OutOfMemoryError:
        return TimingResult("geomloss_tensorized", n, m, d, eps, float("inf"), 0, 0, 0, 0, 0, oom=True, n_iters=n_iters)


def bench_srot(
    n: int, m: int, d: int, eps: float, n_iters: int,
    device: torch.device, warmup: int, rep: int,
    *,
    nvtx: bool = False,
    allow_tf32: bool = False,
    dataset: str = "gaussian",
    rmae_check: bool = True,
    slices: int = 50,
    delta: float = 1e-8,
    stop: "StopCfg" = None,
    seed: int = 0,
) -> TimingResult:
    """Benchmark SROT with fixed iterations.

    Dense O(n*m): materializes both the cost matrix and pi_SOT, so this is gated behind
    --max-dense-size like the GeomLoss tensorized baseline.

    Timing is split. `setup_ms` covers building pi_SOT (L projections, sorts and 1-D OT
    solves), which no other method has -- flash and GeomLoss derive their kernel
    implicitly from a, b and the streamed coordinates, with no setup at all. `mean_ms`
    then covers the solve loop alone, so it stays directly comparable to the other rows.
    Total cost of the method is setup_ms + mean_ms. Keeping them apart matters because
    setup_ms is the term that scales with L, which is the point of sweeping it.

    Cost convention: ||x-y||^2 (full squared Euclidean, matches FlashSinkhorn).
    seed: data-generation seed only (x, y, a, b). build_sot_plan's own projection
    RNG (thetas) stays at its independent default seed=0, unaffected by this.
    """
    _set_tf32(allow_tf32)

    torch.manual_seed(seed)
    x = sample_point_cloud(n, d, device, dataset=dataset, target=False)
    y = sample_point_cloud(m, d, device, dataset=dataset, target=True)
    a = torch.rand(n, device=device, dtype=torch.float32) + 0.1
    b = torch.rand(m, device=device, dtype=torch.float32) + 0.1
    a = a / a.sum()
    b = b / b.sum()

    try:
        # pi_SOT depends only on (x, y, a, b, L, delta), so it is built once and reused
        # across the timed repetitions -- it is setup, not per-iteration work.
        #
        # Build it twice and time the second. The first call absorbs one-time CUDA/kernel
        # initialisation, which otherwise lands entirely in setup_ms for whichever L runs
        # first in a process -- making setup_ms decrease with L instead of increasing with
        # it. bench_with_stats already warms up the solve loop for the same reason.
        build_sot_plan(x, y, a, b, slices=slices, delta=delta)
        torch.cuda.synchronize()
        setup_start = time.perf_counter()
        pi_sot = build_sot_plan(x, y, a, b, slices=slices, delta=delta)
        torch.cuda.synchronize()
        setup_ms = (time.perf_counter() - setup_start) * 1e3

        # float32 to match the other benchmarked methods: flash and GeomLoss both run
        # fp32 with TF32 matmuls, and fp64 on a consumer GPU is 1/64 rate, so timing an
        # fp64 SROT against them would compare different arithmetic. It would also give
        # SROT ~16 digits against their ~3 in rmae_pct, and make the tf32 column
        # meaningless for these rows (TF32 only affects fp32 matmuls).
        # The converged reference below stays fp64 -- references should be exact.
        cost = torch.cdist(x, y, p=2) ** 2
        log_pi = pi_sot.clamp_min(torch.finfo(pi_sot.dtype).tiny).log()
        log_a = a.log()
        log_b = b.log()
    except torch.cuda.OutOfMemoryError:
        return TimingResult(
            "srot", n, m, d, eps, float("inf"), 0, 0, 0, 0, 0, oom=True,
            n_iters=n_iters, dataset=dataset, tf32=allow_tf32, srot_slices=slices,
        )

    _stop = stop or StopCfg.fixed()

    def run():
        _srot_sinkhorn(cost, log_pi, log_a, log_b, eps, n_iters, _stop)

    try:
        f, g, iters_run, converged, final_viol = _srot_sinkhorn(
            cost, log_pi, log_a, log_b, eps, n_iters, _stop)
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        return TimingResult(
            "srot", n, m, d, eps, float("inf"), 0, 0, 0, 0, 0, oom=True,
            n_iters=n_iters, dataset=dataset, tf32=allow_tf32, srot_slices=slices, seed=seed,
        )

    cost_gap_pct = None
    bary = None
    if rmae_check and n <= _EXACT_OT_MAX_N:  # kept as the enable/disable flag name; now gates cost_gap/barycentric_sym
        # SROT's own (possibly entropic-biased) plan, vs. the exact (unregularized) OT
        # reference -- shared across every eps for this (dataset, n, d, seed), unlike the
        # old per-eps entropic reference this replaces.
        T = pi_sot * ((f.unsqueeze(1) + g.unsqueeze(0) - cost) / eps).exp()
        plan_cost = float((T * cost).sum())
        Tx, Ty, r, c = plan_barycentric_dense(T, x, y)
        ref = _cached_exact_ot_reference(n, m, d, seed, x, y, a, b, dataset=dataset)
        cost_gap_pct = cost_gap(plan_cost, ref)
        bary = barycentric_sym(Tx, Ty, ref, a, b)
        feas = plan_feasibility(r, c, a, b)
    else:
        feas = {}

    hit_max_iters = (iters_run >= _stop.max_iter) if _stop.mode != "fixed" else None

    try:
        mean, std, min_t, max_t, median = bench_with_stats(
            run, warmup, rep, nvtx=nvtx,
            nvtx_label=f"srot n={n} d={d} eps={eps} iters={n_iters} L={slices}",
        )
        gpu_memory_mb = gpu_memory_used_mb(device)
        return TimingResult(
            "srot", n, m, d, eps, mean, std, min_t, max_t, median, gpu_memory_mb, oom=False,
            n_iters=n_iters, cost_gap_pct=cost_gap_pct, barycentric_sym=bary,
            dataset=dataset, tf32=allow_tf32,
            srot_slices=slices, setup_ms=setup_ms,
            iters_run=iters_run, converged=converged, final_viol=final_viol, seed=seed,
            hit_max_iters=hit_max_iters, **feas,
        )
    except torch.cuda.OutOfMemoryError:
        return TimingResult(
            "srot", n, m, d, eps, float("inf"), 0, 0, 0, 0, 0, oom=True,
            n_iters=n_iters, dataset=dataset, tf32=allow_tf32, srot_slices=slices, seed=seed,
        )


# =============================================================================
# Spar-Sink / Rand-Sink: importance-sparsified Sinkhorn (baselines)
# =============================================================================
# Li, Yu, Li, Meng, "Importance Sparsification for Sinkhorn Algorithm", JMLR
# (arXiv:2306.06581). Reference implementation:
# https://github.com/Mengyu8042/Spar-Sink
#
# Both methods sparsify the Sinkhorn kernel and iterate on the survivors; they
# differ only in the sampling distribution. Unlike SROT they approximate the SAME
# entropic problem, so they share the standard entropic reference for rmae_pct --
# which makes rmae_pct exactly the RMAE their paper reports.
#
# Deviation from their code: we iterate in the log domain. Theirs is linear
# (u = a / (K v)), which underflows to exactly zero for eps <= 0.01 at our costs
# (exp(-16/0.01) = 0), so two of our three eps values would be unrunnable. The
# sampling scheme, the K/q rescaling and the sparse iteration structure are theirs
# -- their code also switches to sparse CSR above 200 columns.

SPARSINK_METHODS = ("spar_sink", "rand_sink")


def build_sparse_kernel(
    cost: torch.Tensor, a: torch.Tensor, b: torch.Tensor, eps: float,
    *, method: str, sample_size: int, seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Poisson-sample the Sinkhorn kernel; return (rows, cols, log_values).

    Adapted from the authors' released Spar-Sink code (github.com/Mengyu8042/
    Spar-Sink) onto this harness; the sampling scheme below is theirs.
    Their eq. (7) and (9): with inclusion probability q_ij = min(1, s*p_ij),
    keep entry (i, j) with probability q_ij and rescale it to K_ij / q_ij, which is
    unbiased for K. s bounds the *expected* nnz, so the realised count varies.

        spar_sink: p_ij ∝ sqrt(a_i b_j)   -- their importance probability
        rand_sink: p_ij ∝ 1               -- uniform over all entries

    Note p_ij carries no dependence on K, so under uniform marginals sqrt(a_i b_j) is
    constant and the two methods coincide exactly. They differ only to the extent the
    marginals are non-uniform.

    Values are returned in log space: log(K_ij / q_ij) = -C_ij/eps - log q_ij.

    Note their kernel is K = exp(-C/eps) with no a (x) b factor -- the "entropy"
    convention of their eq. (6), OT_eps = <T,C> - eps*H(T). Ours (FlashSinkhorn,
    GeomLoss, and our reference solver) regularizes by KL(T || a (x) b). The two duals
    differ by exactly eps*(H(a) + H(b)). bench_sparsink() sidesteps this by reporting the
    plan's KL-convention entropic value <T, C> + eps*KL(T || a (x) b) rather than the
    sparsified problem's dual, so these rows are comparable with the rest of the table.
    """
    if method not in SPARSINK_METHODS:
        raise ValueError(f"Unknown method: {method!r}. Choices: {SPARSINK_METHODS}")

    n, m = cost.shape
    if method == "spar_sink":
        weights = torch.outer(a.sqrt(), b.sqrt())
    else:
        weights = torch.ones(n, m, device=cost.device, dtype=cost.dtype)
    probs = weights / weights.sum()
    q = (sample_size * probs).clamp_max(1.0)

    generator = torch.Generator(device=cost.device).manual_seed(seed)

    # torch.nonzero() rejects any tensor with more than INT_MAX elements, regardless
    # of how many are actually nonzero -- at n=m=50000, n*m=2.5e9 already exceeds that
    # bound before sample_size even enters the picture, so this hits unconditionally
    # at that scale (not a real memory ceiling: weights/probs/q above are already the
    # same n x m footprint and fit fine). Chunk over rows so each torch.rand/nonzero
    # call stays under the limit; RNG order is unaffected since a given generator
    # advances the same way whether one n x m call or several row-chunked calls
    # consume it, so this reproduces the same sample as the unchunked path exactly.
    int32_max = 2**31 - 1
    if n * m <= int32_max:
        keep = torch.rand(n, m, generator=generator, device=cost.device, dtype=cost.dtype) < q
        rows, cols = keep.nonzero(as_tuple=True)
    else:
        chunk_rows = max(1, int32_max // m)
        row_chunks, col_chunks = [], []
        for start in range(0, n, chunk_rows):
            end = min(start + chunk_rows, n)
            keep_chunk = torch.rand(end - start, m, generator=generator, device=cost.device,
                                     dtype=cost.dtype) < q[start:end]
            r, c = keep_chunk.nonzero(as_tuple=True)
            row_chunks.append(r + start)
            col_chunks.append(c)
        rows = torch.cat(row_chunks)
        cols = torch.cat(col_chunks)

    log_values = -cost[rows, cols] / eps - q[rows, cols].log()
    return rows, cols, log_values


def _sparsink_sinkhorn(
    rows: torch.Tensor, cols: torch.Tensor, log_values: torch.Tensor,
    log_a: torch.Tensor, log_b: torch.Tensor, eps: float, n_iters: int,
    stop: "StopCfg" = None,
):
    """`n_iters` log-domain sweeps over the sampled support only -- O(nnz) per sweep.

    Each half-update is a segmented logsumexp over the kept entries, grouped by row
    (then by column), computed in two passes: a max-reduce for stability, then an
    exp-sum. This is the log-domain form of their u = a/(Kv), v = b/(K^T u).

    Returns (f, g, empty). A row (or column) with no sampled entry cannot transport its
    mass anywhere: its logsumexp is -inf and the potential diverges. Their linear-domain
    formulation hides this -- the row simply gets zero mass in T and contributes nothing
    to <T,C> -- but the marginal constraint is violated either way. We surface it instead:
    `empty` counts such rows plus columns, and the caller reports N/A for rmae_pct when it
    is nonzero. This is not rare at their published subsample sizes; see analysis.md.
    """
    n = log_a.shape[0]
    m = log_b.shape[0]
    f = torch.zeros_like(log_a)
    g = torch.zeros_like(log_b)
    neg_inf = torch.finfo(log_values.dtype).min

    def segmented_lse(z: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
        mx = torch.full((size,), neg_inf, device=z.device, dtype=z.dtype)
        mx = mx.scatter_reduce(0, index, z, reduce="amax", include_self=True)
        acc = torch.zeros(size, device=z.device, dtype=z.dtype)
        acc = acc.index_add(0, index, (z - mx[index]).exp())
        return mx + acc.clamp_min(torch.finfo(z.dtype).tiny).log()

    empty = int(n - rows.unique().numel() + m - cols.unique().numel())

    def _row_lse(gv):
        return segmented_lse(log_values + gv[cols] / eps, rows, n)

    def _col_lse(fv):
        return segmented_lse(log_values + fv[rows] / eps, cols, m)

    mode = getattr(stop, "mode", "fixed") if stop is not None else "fixed"

    if mode == "fixed":
        for _ in range(n_iters):
            f = eps * (log_a - _row_lse(g))
            g = eps * (log_b - _col_lse(f))
        return f, g, empty, n_iters, None, None

    if mode == "potential_linf":
        # FlashSinkhorn's own rule, verbatim: max(|Δf|, |Δg|) < stop.tol since the
        # last check. f, g are already standard-scale here (f = eps*(...)), matching
        # Flash's unshifted potentials, so stop.tol needs no rescaling. Distinct from
        # Spar-Sink's own "potential" mode below, which uses max change in the scaling
        # vectors u=exp(f/eps), checked every stop.check_every iterations like everyone else.
        #
        # Caveat (found while verifying this against a deep-converged reference):
        # importance-sampled sparse supports can have weakly-connected components
        # with a local contraction rate near 1, so "iterate barely moved since the
        # last check" can be satisfied while still meaningfully far from the true
        # fixed point -- worse the smaller check_every is, since a short window
        # only sees a thin slice of a slow drift. Not a bug in this check (it's the
        # same rule FlashSinkhorn uses natively) -- just don't assume tol alone
        # bounds solution error here the way it more safely does for SROT/SinkSLOT's
        # denser supports. check_every should span the support's mixing timescale.
        prev_f, prev_g = f, g
        it = 0
        converged = False
        change = float("inf")
        while it < stop.max_iter:
            f = eps * (log_a - _row_lse(g))
            g = eps * (log_b - _col_lse(f))
            it += 1
            if it % stop.check_every == 0:
                change = max((f - prev_f).abs().max().item(), (g - prev_g).abs().max().item())
                if change < stop.tol:
                    converged = True
                    break
                prev_f = f
                prev_g = g
        return f, g, empty, it, converged, change

    a = log_a.exp()
    b = log_b.exp()
    it = 0
    converged = False
    viol = float("inf")
    while it < stop.max_iter:
        f_old, g_old = f, g
        f = eps * (log_a - _row_lse(g))
        g = eps * (log_b - _col_lse(f))
        it += 1
        if it % stop.check_every == 0 or it == stop.max_iter:
            # Check both row and column marginal violation, without an extra
            # _row_lse call.
            row_marg = a * ((f_old - f) / eps).exp()
            col_marg = b * ((g_old - g) / eps).exp()
            # max (L-infinity), not sum -- see _srot_sinkhorn's comment: a sum
            # over n terms against a fixed absolute tol is unreachable at
            # n=10,000 regardless of convergence. Matches SLOT's actual
            # working "marg_viol" rule. Not gated on mass_tol either, matching
            # SLOT exactly.
            viol = float(torch.maximum((row_marg - a).abs().max(), (col_marg - b).abs().max()))
            if stop.mode == "potential":
                # Spar-Sink's rule: max(||du||_inf, ||dv||_inf) on the scaling
                # vectors u=exp(f/eps). Also switched from sum to max: same
                # n-invariance reasoning as marg_viol above -- SLOT doesn't
                # implement this mode at all (its own Spar-Sink never early-stops),
                # so there's no reference to match, but a sum over n+m terms
                # against a fixed tol has the identical unreachable-at-scale flaw.
                du = float(((f / eps).exp() - (f_old / eps).exp()).abs().max())
                dv = float(((g / eps).exp() - (g_old / eps).exp()).abs().max())
                if max(du, dv) <= stop.potential_tol:
                    converged = True
                    break
            elif viol <= stop.tol:
                converged = True
                break
    return f, g, empty, it, converged, viol


def bench_sparsink(
    n: int, m: int, d: int, eps: float, n_iters: int,
    device: torch.device, warmup: int, rep: int,
    *,
    nvtx: bool = False,
    allow_tf32: bool = False,
    dataset: str = "gaussian",
    rmae_check: bool = True,
    method: str = "spar_sink",
    sample_size: int = 2000,
    replicates: int = 10,
    stop: "StopCfg" = None,
    seed: int = 0,
) -> TimingResult:
    """Benchmark Spar-Sink / Rand-Sink with fixed iterations.

    Sampling is stochastic, so a single draw reports sampling noise as method quality;
    their paper averages 100 replications. We draw `replicates` independent kernels,
    seeded per replicate, and report mean RMAE (with rmae_std) and mean timing.

    Setup (sampling and building the sparse kernel) is timed into setup_ms; mean_ms
    covers the solve loop alone, so it stays comparable to the other rows. The
    probability matrix is built densely -- setup only, O(n*m) -- while the iterations
    are O(nnz), as in their implementation, which also switches to sparse above 200
    columns.

    seed: data-generation seed only (x, y, a, b). The per-replicate kernel draws
    below (seed=0 for warmup/reference, seed=r for r in range(replicates)) stay
    independent of this, unaffected.
    """
    _set_tf32(allow_tf32)

    torch.manual_seed(seed)
    x = sample_point_cloud(n, d, device, dataset=dataset, target=False)
    y = sample_point_cloud(m, d, device, dataset=dataset, target=True)
    a = torch.rand(n, device=device, dtype=torch.float32) + 0.1
    b = torch.rand(m, device=device, dtype=torch.float32) + 0.1
    a = a / a.sum()
    b = b / b.sum()

    try:
        cost = torch.cdist(x, y, p=2) ** 2
        log_a, log_b = a.log(), b.log()
    except torch.cuda.OutOfMemoryError:
        return TimingResult(
            method, n, m, d, eps, float("inf"), 0, 0, 0, 0, 0, oom=True,
            n_iters=n_iters, dataset=dataset, tf32=allow_tf32, sample_size=sample_size, seed=seed,
        )


    ref = None
    if rmae_check and n <= _EXACT_OT_MAX_N:  # kept as the enable/disable flag name; now gates cost_gap/barycentric_sym
        ref = _cached_exact_ot_reference(n, m, d, seed, x, y, a, b, dataset=dataset)

    # Warm up the sampling path: the first call in a process absorbs one-time CUDA
    # initialisation, which would otherwise land in setup_ms (~40ms against ~0.1ms).
    build_sparse_kernel(cost, a, b, eps, method=method, sample_size=sample_size, seed=0)
    torch.cuda.synchronize()

    _stop = stop or StopCfg.fixed()
    plan_costs, barys, nnzs, empties = [], [], [], 0
    build_ms = []
    iters_list, conv_list, viol_list = [], [], []
    feas_list = []
    for rep_i in range(replicates):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        rows, cols, log_values = build_sparse_kernel(
            cost, a, b, eps, method=method, sample_size=sample_size, seed=rep_i,
        )
        torch.cuda.synchronize()
        build_ms.append((time.perf_counter() - t0) * 1e3)

        f, g, empty, iters_r, conv_r, viol_r = _sparsink_sinkhorn(
            rows, cols, log_values, log_a, log_b, eps, n_iters, _stop)
        nnzs.append(rows.numel())
        empties += empty
        if empty == 0:
            iters_list.append(iters_r)
            if conv_r is not None:
                conv_list.append(conv_r)
            viol_list.append(viol_r)
        if empty == 0 and ref is not None:
            # T = diag(u) K_tilde diag(v) on the sampled support: the plan Spar-Sink/
            # Rand-Sink actually produced. <T,C> (not the KL-regularized objective) is
            # what cost_gap compares against exact OT's cost.
            log_T = log_values + (f[rows] + g[cols]) / eps
            T = log_T.exp()
            plan_cost = float((T * cost[rows, cols]).sum())
            Tx, Ty, r_marg, c_marg = plan_barycentric_sparse(T, rows, cols, x, y)
            plan_costs.append(cost_gap(plan_cost, ref))
            barys.append(barycentric_sym(Tx, Ty, ref, a, b))
            feas_list.append(plan_feasibility(r_marg, c_marg, a, b))

    cost_gap_pct = None
    bary = None
    feas = {}
    if plan_costs:
        cost_gap_pct = float(torch.tensor(plan_costs).mean())
        bary = float(torch.tensor(barys).mean())
        feas = {
            "mass": sum(f["mass"] for f in feas_list) / len(feas_list),
            "marg_viol": sum(f["marg_viol"] for f in feas_list) / len(feas_list),
            "plan_empty_rows": sum(f["plan_empty_rows"] for f in feas_list) / len(feas_list),
            "plan_empty_cols": sum(f["plan_empty_cols"] for f in feas_list) / len(feas_list),
        }

    # Time one representative draw (the last), warmed up like every other method.
    rows, cols, log_values = build_sparse_kernel(
        cost, a, b, eps, method=method, sample_size=sample_size, seed=0,
    )

    def run():
        _sparsink_sinkhorn(rows, cols, log_values, log_a, log_b, eps, n_iters, _stop)

    try:
        run()
        torch.cuda.synchronize()
        mean, std, min_t, max_t, median = bench_with_stats(
            run, warmup, rep, nvtx=nvtx,
            nvtx_label=f"{method} n={n} d={d} eps={eps} iters={n_iters} s={sample_size}",
        )
        gpu_memory_mb = gpu_memory_used_mb(device)
        return TimingResult(
            method, n, m, d, eps, mean, std, min_t, max_t, median, gpu_memory_mb, oom=False,
            n_iters=n_iters, cost_gap_pct=cost_gap_pct, barycentric_sym=bary, dataset=dataset,
            tf32=allow_tf32, sample_size=sample_size,
            nnz=int(sum(nnzs) / len(nnzs)), empty_lines=empties,
            iters_run=(int(sum(_it) / len(_it)) if (_it := [v for v in iters_list if v is not None]) else None),
            converged=(all(_cv) if (_cv := [v for v in conv_list if v is not None]) else None),
            final_viol=(sum(_vl) / len(_vl) if (_vl := [v for v in viol_list if v is not None]) else None),
            hit_max_iters=(any(it >= _stop.max_iter for it in iters_list)
                           if (_stop.mode != "fixed" and iters_list) else None),
            valid_replicates=len(plan_costs),
            setup_ms=float(sum(build_ms) / len(build_ms)), seed=seed, **feas,
        )
    except torch.cuda.OutOfMemoryError:
        return TimingResult(
            method, n, m, d, eps, float("inf"), 0, 0, 0, 0, 0, oom=True,
            n_iters=n_iters, dataset=dataset, tf32=allow_tf32, sample_size=sample_size, seed=seed,
        )


def compute_sinkslot_reference(
    rows: torch.Tensor, cols: torch.Tensor, log_S: torch.Tensor, cost: torch.Tensor,
    log_a: torch.Tensor, log_b: torch.Tensor, a: torch.Tensor, b: torch.Tensor, eps: float,
    *, max_iter: int = 20000, tol: float = 1e-6, check_every: int = 10,
) -> float:
    """Converged gamma=0 SROT dual on the sparse support -- the RMAE reference.

    SinkSLOT converges to its own optimum (the gamma=0 SROT plan), distinct from
    both entropic OT and gamma>0 SROT, so it needs its own reference. KL forces
    supp(P) subset of supp(P^SOT), so that optimum lives on the same sparse support
    the measured run uses -- the reference stays O(nnz). Warm-started eps annealing,
    same schedule as compute_entropic_ot_reference(); the potentials (f, g) persist
    across sweeps and stages, and lam is rebuilt from (log_S, cost) at each stage.
    """
    n, m = a.numel(), b.numel()
    tiny = torch.finfo(log_S.dtype).tiny
    f = torch.zeros_like(log_a)
    g = torch.zeros_like(log_b)

    def seg_lse(vals, idx, size):
        mx = vals.new_full((size,), -1e30).scatter_reduce(0, idx, vals, reduce="amax", include_self=True)
        acc = vals.new_zeros(size).index_add_(0, idx, (vals - mx[idx]).exp())
        return mx + acc.clamp_min(tiny).log()

    def sweep(stage_eps, iters):
        nonlocal f, g
        lam = log_S - cost / stage_eps
        err = float("inf")
        for used in range(1, iters + 1):
            f = stage_eps * (log_a - seg_lse(lam + g[cols] / stage_eps, rows, n))
            g = stage_eps * (log_b - seg_lse(lam + f[rows] / stage_eps, cols, m))
            if used % check_every == 0 or used == iters:
                z = lam + (f[rows] + g[cols]) / stage_eps
                r = f.new_zeros(n).index_add_(0, rows, z.exp())
                err = float((r - a).abs().max())
                if err < tol:
                    break
        return err

    schedule = []
    se = max(eps, 1.0)
    while se > eps * 1.001:
        schedule.append(se); se *= 0.5
    for se in schedule:
        sweep(se, max(check_every, 200))
    err = sweep(eps, max_iter)
    if err >= tol:
        print(f"  [warn] SinkSLOT reference hit max_iter={max_iter} at eps={eps:g} "
              f"with marginal error {err:.3e} > tol={tol:g}; rmae_pct unreliable")
    return float((a * f).sum() + (b * g).sum())


def bench_sinkslot(
    n: int, m: int, d: int, eps: float, n_iters: int,
    device: torch.device, warmup: int, rep: int,
    *,
    nvtx: bool = False,
    allow_tf32: bool = False,
    dataset: str = "gaussian",
    rmae_check: bool = True,
    slices: int = 50,
    stop: "StopCfg" = None,
    seed: int = 0,
) -> TimingResult:
    """Benchmark SinkSLOT (fused-Triton, gamma=0 sparse SROT).

    Sparse O(L(N+M)): the sliced support and its CSR/CSC layouts are built once in
    setup_ms; the timed loop is the fused-Triton alternating half-steps. fp32 (no
    matmul, so TF32 does not apply). Reference is the converged gamma=0 optimum on
    the same support. See torch-ext/sinkslot/solver.py.

    seed: data-generation seed only (x, y, a, b). sot_plan_coo's own projection
    RNG stays at its independent default seed=0, unaffected by this.
    """
    from sinkslot.solver import (
        sot_plan_coo, to_csr,
    )
    from sinkslot.sinkhorn_solvers import sinkslot_alternating_triton
    _set_tf32(allow_tf32)

    torch.manual_seed(seed)
    x = sample_point_cloud(n, d, device, dataset=dataset, target=False)
    y = sample_point_cloud(m, d, device, dataset=dataset, target=True)
    a = torch.rand(n, device=device, dtype=torch.float32) + 0.1
    b = torch.rand(m, device=device, dtype=torch.float32) + 0.1
    a = a / a.sum(); b = b / b.sum()
    log_a, log_b = a.log(), b.log()

    try:
        # Setup: sliced support + CSR/CSC layouts, timed once (warmed).
        rows, cols, S = sot_plan_coo(x, y, a, b, L=slices, seed=0)
        cost = (x[rows] - y[cols]).square().sum(1)
        log_S = S.clamp_min(torch.finfo(S.dtype).tiny).log()
        lam = log_S - cost / eps
        r_ptr, r_idx, r_lam, _ = to_csr(rows, cols, lam, n)
        c_ptr, c_idx, c_lam, _ = to_csr(cols, rows, lam, m)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        rows, cols, S = sot_plan_coo(x, y, a, b, L=slices, seed=0)
        cost = (x[rows] - y[cols]).square().sum(1)
        log_S = S.clamp_min(torch.finfo(S.dtype).tiny).log()
        lam = log_S - cost / eps
        r_ptr, r_idx, r_lam, _ = to_csr(rows, cols, lam, n)
        c_ptr, c_idx, c_lam, _ = to_csr(cols, rows, lam, m)
        torch.cuda.synchronize()
        setup_ms = (time.perf_counter() - t0) * 1e3
    except torch.cuda.OutOfMemoryError:
        return TimingResult("sinkslot", n, m, d, eps, float("inf"), 0, 0, 0, 0, 0, oom=True,
                            n_iters=n_iters, dataset=dataset, tf32=allow_tf32, srot_slices=slices,
                            seed=seed)

    _stop = stop or StopCfg.fixed()
    # sinkslot's own _STOP_MODES dropped "potential" (it was byte-for-byte
    # identical to "marginal") and renamed "potential_linf" -> "potential"
    # (see torch-ext/sinkslot/sinkhorn_solvers.py, issue #47) -- this CLI's
    # --stop-mode vocabulary is unchanged (still shared with srot/spar_sink/
    # rand_sink, where the old distinction still matters), so translate here
    # rather than propagate the old names into sinkslot's own solve loop.
    _sinkslot_stop_mode = {"potential": "marginal", "potential_linf": "potential"}.get(
        _stop.mode, _stop.mode)
    _sinkslot_stop = StopCfg(mode=_sinkslot_stop_mode, max_iter=_stop.max_iter,
                              tol=_stop.tol, potential_tol=_stop.potential_tol,
                              mass_tol=_stop.mass_tol, check_every=_stop.check_every)

    def run():
        sinkslot_alternating_triton(r_ptr, r_idx, r_lam, c_ptr, c_idx, c_lam, log_a, log_b, n, m, n_iters, _sinkslot_stop, eps=eps)

    phi, psi, iters_run, converged, final_viol = sinkslot_alternating_triton(
        r_ptr, r_idx, r_lam, c_ptr, c_idx, c_lam, log_a, log_b, n, m, n_iters, _sinkslot_stop, eps=eps)
    cost_gap_pct = None
    bary = None
    if rmae_check and n <= _EXACT_OT_MAX_N:  # kept as the enable/disable flag name; now gates cost_gap/barycentric_sym
        # T_ij = exp(phi_i + psi_j + lam_ij) on the sliced support -- SinkSLOT's own plan,
        # vs. the exact (unregularized) OT reference, shared across every eps for this
        # (dataset, n, d, seed).
        T_vals = (phi[rows] + psi[cols] + lam).exp()
        plan_cost = float((T_vals * cost).sum())
        Tx, Ty, r, c = plan_barycentric_sparse(T_vals, rows, cols, x, y)
        ref = _cached_exact_ot_reference(n, m, d, seed, x, y, a, b, dataset=dataset)
        cost_gap_pct = cost_gap(plan_cost, ref)
        bary = barycentric_sym(Tx, Ty, ref, a, b)
        feas = plan_feasibility(r, c, a, b)
    else:
        feas = {}

    hit_max_iters = (iters_run >= _stop.max_iter) if _stop.mode != "fixed" else None

    try:
        run(); torch.cuda.synchronize()
        mean, std, min_t, max_t, median = bench_with_stats(
            run, warmup, rep, nvtx=nvtx,
            nvtx_label=f"sinkslot n={n} d={d} eps={eps} iters={n_iters} L={slices}")
        gpu_memory_mb = gpu_memory_used_mb(device)
        return TimingResult("sinkslot", n, m, d, eps, mean, std, min_t, max_t, median,
                            gpu_memory_mb, oom=False, n_iters=n_iters,
                            cost_gap_pct=cost_gap_pct, barycentric_sym=bary,
                            dataset=dataset, tf32=allow_tf32, srot_slices=slices,
                            nnz=int(rows.numel()), setup_ms=setup_ms,
                            iters_run=iters_run, converged=converged, final_viol=final_viol,
                            hit_max_iters=hit_max_iters, seed=seed, **feas)
    except torch.cuda.OutOfMemoryError:
        return TimingResult("sinkslot", n, m, d, eps, float("inf"), 0, 0, 0, 0, 0, oom=True,
                            n_iters=n_iters, dataset=dataset, tf32=allow_tf32, srot_slices=slices,
                            seed=seed)


def bench_sinkslotcuda(
    n: int, m: int, d: int, eps: float, n_iters: int,
    device: torch.device, warmup: int, rep: int,
    *,
    nvtx: bool = False,
    allow_tf32: bool = False,
    dataset: str = "gaussian",
    rmae_check: bool = True,
    slices: int = 50,
    stop: "StopCfg" = None,
    seed: int = 0,
) -> TimingResult:
    """Benchmark SinkSLOT-CUDA: SinkSLOT with the CUDA-optimised setup path.

    Same method and same solve kernels as ``bench_sinkslot`` -- the only difference
    is the plan-build/setup, which is 2.1-3.1x faster end to end:

    * ``sparse_sqeuclidean_cost`` -- fused Triton cost kernel (9-10x on the cost
      stage; no (nnz, d) temporaries).
    * ``_ot_1d_coo_batched_cuda`` -- transposed (C, n) layout with fp64 cumsum
      accumulation (49.5x on the dominant stage AND a strictly more accurate plan).
    * ``to_csr(..., narrow_key=True)`` -- int32 CSC sort key (4.96 -> 2.24 ms).

    Because the fp64 scan yields a *different* (more accurate) sliced support than
    the baseline's fp32 scan, SinkSLOT-CUDA carries its own converged-reference
    cache. Its RMAE is therefore measured against its own plan's optimum, exactly
    as SinkSLOT is against the baseline plan. See sinkslot/solver.py.

    seed: data-generation seed only (x, y, a, b). sot_plan_coo's own projection
    RNG stays at its independent default seed=0, unaffected by this.
    """
    from sinkslot.solver import (
        sot_plan_coo, to_csr, sparse_sqeuclidean_cost, _ot_1d_coo_batched_cuda,
    )
    from sinkslot.sinkhorn_solvers import sinkslot_alternating_triton
    _set_tf32(allow_tf32)

    torch.manual_seed(seed)
    x = sample_point_cloud(n, d, device, dataset=dataset, target=False)
    y = sample_point_cloud(m, d, device, dataset=dataset, target=True)
    a = torch.rand(n, device=device, dtype=torch.float32) + 0.1
    b = torch.rand(m, device=device, dtype=torch.float32) + 0.1
    a = a / a.sum(); b = b / b.sum()
    log_a, log_b = a.log(), b.log()

    def _setup():
        rows, cols, S = sot_plan_coo(x, y, a, b, L=slices, seed=0, ot1d=_ot_1d_coo_batched_cuda)
        cost = sparse_sqeuclidean_cost(x, y, rows, cols)
        log_S = S.clamp_min(torch.finfo(S.dtype).tiny).log()
        lam = log_S - cost / eps
        r_ptr, r_idx, r_lam, _ = to_csr(rows, cols, lam, n, narrow_key=True)
        c_ptr, c_idx, c_lam, _ = to_csr(cols, rows, lam, m, narrow_key=True)
        return rows, cols, S, cost, log_S, r_ptr, r_idx, r_lam, c_ptr, c_idx, c_lam

    try:
        # Setup: sliced support + CSR/CSC layouts, timed once (warmed).
        _setup()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        (rows, cols, S, cost, log_S,
         r_ptr, r_idx, r_lam, c_ptr, c_idx, c_lam) = _setup()
        torch.cuda.synchronize()
        setup_ms = (time.perf_counter() - t0) * 1e3
    except torch.cuda.OutOfMemoryError:
        return TimingResult("sinkslotcuda", n, m, d, eps, float("inf"), 0, 0, 0, 0, 0, oom=True,
                            n_iters=n_iters, dataset=dataset, tf32=allow_tf32, srot_slices=slices,
                            seed=seed)

    _stop = stop or StopCfg.fixed()
    # See bench_sinkslot's matching comment: translates this CLI's stop-mode
    # vocabulary to sinkslot's own (now 3-value) _STOP_MODES.
    _sinkslot_stop_mode = {"potential": "marginal", "potential_linf": "potential"}.get(
        _stop.mode, _stop.mode)
    _sinkslot_stop = StopCfg(mode=_sinkslot_stop_mode, max_iter=_stop.max_iter,
                              tol=_stop.tol, potential_tol=_stop.potential_tol,
                              mass_tol=_stop.mass_tol, check_every=_stop.check_every)

    def run():
        sinkslot_alternating_triton(r_ptr, r_idx, r_lam, c_ptr, c_idx, c_lam, log_a, log_b, n, m, n_iters, _sinkslot_stop, eps=eps)

    phi, psi, iters_run, converged, final_viol = sinkslot_alternating_triton(
        r_ptr, r_idx, r_lam, c_ptr, c_idx, c_lam, log_a, log_b, n, m, n_iters, _sinkslot_stop, eps=eps)
    cost_gap_pct = None
    bary = None
    if rmae_check and n <= _EXACT_OT_MAX_N:  # kept as the enable/disable flag name; now gates cost_gap/barycentric_sym
        # lam is local to _setup() and never returned -- recompute from log_S/cost,
        # both of which ARE returned (fixes a NameError that crashed every unit).
        lam = log_S - cost / eps
        T_vals = (phi[rows] + psi[cols] + lam).exp()
        plan_cost = float((T_vals * cost).sum())
        Tx, Ty, r, c = plan_barycentric_sparse(T_vals, rows, cols, x, y)
        ref = _cached_exact_ot_reference(n, m, d, seed, x, y, a, b, dataset=dataset)
        cost_gap_pct = cost_gap(plan_cost, ref)
        bary = barycentric_sym(Tx, Ty, ref, a, b)
        feas = plan_feasibility(r, c, a, b)
    else:
        feas = {}

    hit_max_iters = (iters_run >= _stop.max_iter) if _stop.mode != "fixed" else None

    try:
        run(); torch.cuda.synchronize()
        mean, std, min_t, max_t, median = bench_with_stats(
            run, warmup, rep, nvtx=nvtx,
            nvtx_label=f"sinkslotcuda n={n} d={d} eps={eps} iters={n_iters} L={slices}")
        gpu_memory_mb = gpu_memory_used_mb(device)
        return TimingResult("sinkslotcuda", n, m, d, eps, mean, std, min_t, max_t, median,
                            gpu_memory_mb, oom=False, n_iters=n_iters,
                            cost_gap_pct=cost_gap_pct, barycentric_sym=bary,
                            dataset=dataset, tf32=allow_tf32, srot_slices=slices,
                            nnz=int(rows.numel()), setup_ms=setup_ms,
                            iters_run=iters_run, converged=converged, final_viol=final_viol,
                            hit_max_iters=hit_max_iters, seed=seed, **feas)
    except torch.cuda.OutOfMemoryError:
        return TimingResult("sinkslotcuda", n, m, d, eps, float("inf"), 0, 0, 0, 0, 0, oom=True,
                            n_iters=n_iters, dataset=dataset, tf32=allow_tf32, srot_slices=slices,
                            seed=seed)


_sinkslot_ref_cache: Dict[str, float] = {}
_SINKSLOT_REF_CACHE_PATH = Path.home() / ".cache" / "sinkslot" / "sinkslot_reference.json"
_SINKSLOT_REF_CACHE_VERSION = 1


def _cached_sinkslot_reference(n, m, d, eps, slices, rows, cols, log_S, cost,
                               log_a, log_b, a, b, *, dataset="gaussian"):
    """Memoized converged SinkSLOT reference, keyed by (dataset, n, m, d, eps, L)."""
    key = f"{dataset},{n},{m},{d},{eps:g},{slices}"
    if key in _sinkslot_ref_cache:
        return _sinkslot_ref_cache[key]
    if not _sinkslot_ref_cache:
        try:
            blob = json.loads(_SINKSLOT_REF_CACHE_PATH.read_text())
            if blob.get("version") == _SINKSLOT_REF_CACHE_VERSION:
                _sinkslot_ref_cache.update(blob.get("costs", {}))
        except (OSError, ValueError):
            pass
        if key in _sinkslot_ref_cache:
            return _sinkslot_ref_cache[key]
    _sinkslot_ref_cache[key] = compute_sinkslot_reference(
        rows, cols, log_S, cost, log_a, log_b, a, b, eps)
    try:
        _SINKSLOT_REF_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SINKSLOT_REF_CACHE_PATH.write_text(json.dumps(
            {"version": _SINKSLOT_REF_CACHE_VERSION, "costs": _sinkslot_ref_cache}))
    except OSError as e:
        print(f"  [warn] could not write SinkSLOT reference cache: {e}")
    return _sinkslot_ref_cache[key]


_sinkslot_cuda_ref_cache: Dict[str, float] = {}
_SINKSLOT_CUDA_REF_CACHE_PATH = Path.home() / ".cache" / "sinkslot" / "sinkslotcuda_reference.json"
# Separate namespace from the SinkSLOT cache: the CUDA path's fp64 cumsum yields a
# different (more accurate) sliced support than the baseline's fp32 scan -- the two
# disagreed on up to 3.8% of the support -- so each converges to its own optimum.
_SINKSLOT_CUDA_REF_CACHE_VERSION = 1


def _cached_sinkslotcuda_reference(n, m, d, eps, slices, rows, cols, log_S, cost,
                                   log_a, log_b, a, b, *, dataset="gaussian"):
    """Memoized converged SinkSLOT-CUDA reference, keyed by (dataset, n, m, d, eps, L)."""
    key = f"{dataset},{n},{m},{d},{eps:g},{slices}"
    if key in _sinkslot_cuda_ref_cache:
        return _sinkslot_cuda_ref_cache[key]
    if not _sinkslot_cuda_ref_cache:
        try:
            blob = json.loads(_SINKSLOT_CUDA_REF_CACHE_PATH.read_text())
            if blob.get("version") == _SINKSLOT_CUDA_REF_CACHE_VERSION:
                _sinkslot_cuda_ref_cache.update(blob.get("costs", {}))
        except (OSError, ValueError):
            pass
        if key in _sinkslot_cuda_ref_cache:
            return _sinkslot_cuda_ref_cache[key]
    _sinkslot_cuda_ref_cache[key] = compute_sinkslot_reference(
        rows, cols, log_S, cost, log_a, log_b, a, b, eps)
    try:
        _SINKSLOT_CUDA_REF_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SINKSLOT_CUDA_REF_CACHE_PATH.write_text(json.dumps(
            {"version": _SINKSLOT_CUDA_REF_CACHE_VERSION, "costs": _sinkslot_cuda_ref_cache}))
    except OSError as e:
        print(f"  [warn] could not write SinkSLOT-CUDA reference cache: {e}")
    return _sinkslot_cuda_ref_cache[key]


# =============================================================================
# OTT-JAX Benchmarks
# =============================================================================

def bench_ott_jax_online(
    n: int, m: int, d: int, eps: float, n_iters: int,
    device: torch.device, warmup: int, rep: int,
    batch_size: int = 256,
    *,
    nvtx: bool = False,
    allow_tf32: bool = False,
) -> Optional[TimingResult]:
    """Benchmark OTT-JAX online mode with native Sinkhorn solver.

    Uses native Sinkhorn solver with min_iterations=max_iterations=n_iters
    and threshold=-1 to force exactly n_iters iterations (no early stopping).
    This is ~23% faster than custom fori_loop due to internal optimizations.

    Cost convention: PointCloud default = ||x-y||² (matches FlashSinkhorn).

    Timing methodology: Wall-clock time with block_until_ready() synchronization.
    JAX lacks a CUDA event API, so wall-clock includes minor Python/dispatch overhead (~1-5%).

    Memory: JAX doesn't expose easy peak memory tracking like PyTorch, so we report 0.

    RMAE: not computed here. This function draws its own point cloud with JAX's
    RNG (jax.random.PRNGKey), which does not match the PyTorch-seeded (x, y, a, b) used
    by the flash/GeomLoss benchmarks, so there's no shared exact-OT reference to compare
    against without duplicating a separate POT solve for this data.
    """
    try:
        import jax
        import jax.numpy as jnp
        from jax import config as jax_config
        from ott.geometry import pointcloud
        from ott.problems.linear import linear_problem
        from ott.solvers.linear import sinkhorn
    except ImportError:
        return None

    # Match PyTorch TF32 setting for fair comparison
    jax_precision = "default" if allow_tf32 else "highest"
    jax_config.update("jax_default_matmul_precision", jax_precision)

    key = jax.random.PRNGKey(0)
    key1, key2, key3, key4 = jax.random.split(key, 4)

    x_jax = jax.random.normal(key1, (n, d), dtype=jnp.float32)
    y_jax = jax.random.normal(key2, (m, d), dtype=jnp.float32)
    a_jax = jax.random.uniform(key3, (n,), dtype=jnp.float32) + 0.1
    b_jax = jax.random.uniform(key4, (m,), dtype=jnp.float32) + 0.1
    a_jax = a_jax / a_jax.sum()
    b_jax = b_jax / b_jax.sum()

    # Native solver with fixed iterations (no early stopping)
    solver = sinkhorn.Sinkhorn(
        threshold=-1.0,  # Never converge early
        max_iterations=n_iters,
        min_iterations=n_iters,  # Force exactly n_iters iterations
    )

    # PointCloud default cost = ||x-y||² (squared Euclidean, matches FlashSinkhorn)
    geom = pointcloud.PointCloud(x_jax, y_jax, epsilon=eps, batch_size=batch_size)
    prob = linear_problem.LinearProblem(geom, a=a_jax, b=b_jax)

    @jax.jit
    def solve():
        return solver(prob)

    def run():
        out = solve()
        return jax.block_until_ready(out.f)

    try:
        for _ in range(warmup):
            run()
    except Exception:
        return TimingResult("ott_jax_online", n, m, d, eps, float("inf"), 0, 0, 0, 0, 0, oom=True, n_iters=n_iters)

    # Note: Using wall-clock time because JAX lacks CUDA event API.
    # block_until_ready() ensures GPU sync, but includes Python overhead.
    with _nvtx_range(
        f"ott_jax_online n={n} d={d} eps={eps} iters={n_iters}/timed",
        enabled=bool(nvtx),
    ):
        times = []
        for _ in range(rep):
            start = time.perf_counter()
            run()
            times.append((time.perf_counter() - start) * 1000)

    times_t = torch.tensor(times)
    # JAX doesn't expose easy peak memory tracking; report 0
    return TimingResult(
        "ott_jax_online", n, m, d, eps,
        times_t.mean().item(),
        times_t.std().item(),
        times_t.min().item(),
        times_t.max().item(),
        times_t.median().item(),
        0,  # gpu_memory_mb not measured for JAX
        oom=False,
        n_iters=n_iters,
    )


def bench_ott_jax_dense(
    n: int, m: int, d: int, eps: float, n_iters: int,
    device: torch.device, warmup: int, rep: int,
    *,
    nvtx: bool = False,
    allow_tf32: bool = False,
) -> Optional[TimingResult]:
    """Benchmark OTT-JAX dense mode with native Sinkhorn solver.

    Uses native Sinkhorn solver with min_iterations=max_iterations=n_iters
    and threshold=-1 to force exactly n_iters iterations (no early stopping).
    Dense mode: no batch_size parameter (materializes full O(n²) cost matrix).

    Cost convention: PointCloud default = ||x-y||² (matches FlashSinkhorn).

    Timing/Memory: Same limitations as online mode (wall-clock, no memory tracking).

    RMAE: not computed here; see bench_ott_jax_online for why (different RNG
    than the PyTorch-seeded flash/GeomLoss point cloud).
    """
    try:
        import jax
        import jax.numpy as jnp
        from jax import config as jax_config
        from ott.geometry import pointcloud
        from ott.problems.linear import linear_problem
        from ott.solvers.linear import sinkhorn
    except ImportError:
        return None

    # Match PyTorch TF32 setting for fair comparison
    jax_precision = "default" if allow_tf32 else "highest"
    jax_config.update("jax_default_matmul_precision", jax_precision)

    key = jax.random.PRNGKey(0)
    key1, key2, key3, key4 = jax.random.split(key, 4)

    x_jax = jax.random.normal(key1, (n, d), dtype=jnp.float32)
    y_jax = jax.random.normal(key2, (m, d), dtype=jnp.float32)
    a_jax = jax.random.uniform(key3, (n,), dtype=jnp.float32) + 0.1
    b_jax = jax.random.uniform(key4, (m,), dtype=jnp.float32) + 0.1
    a_jax = a_jax / a_jax.sum()
    b_jax = b_jax / b_jax.sum()

    # Native solver with fixed iterations (no early stopping)
    solver = sinkhorn.Sinkhorn(
        threshold=-1.0,  # Never converge early
        max_iterations=n_iters,
        min_iterations=n_iters,  # Force exactly n_iters iterations
    )

    # Dense mode: no batch_size (materializes O(n²) cost matrix)
    # PointCloud default cost = ||x-y||² (squared Euclidean, matches FlashSinkhorn)
    geom = pointcloud.PointCloud(x_jax, y_jax, epsilon=eps)
    prob = linear_problem.LinearProblem(geom, a=a_jax, b=b_jax)

    @jax.jit
    def solve():
        return solver(prob)

    def run():
        out = solve()
        return jax.block_until_ready(out.f)

    try:
        for _ in range(warmup):
            run()
    except Exception:
        return TimingResult("ott_jax_dense", n, m, d, eps, float("inf"), 0, 0, 0, 0, 0, oom=True, n_iters=n_iters)

    # Note: Using wall-clock time because JAX lacks CUDA event API.
    with _nvtx_range(
        f"ott_jax_dense n={n} d={d} eps={eps} iters={n_iters}/timed",
        enabled=bool(nvtx),
    ):
        times = []
        for _ in range(rep):
            start = time.perf_counter()
            run()
            times.append((time.perf_counter() - start) * 1000)

    times_t = torch.tensor(times)
    # JAX doesn't expose easy peak memory tracking; report 0
    return TimingResult(
        "ott_jax_dense", n, m, d, eps,
        times_t.mean().item(),
        times_t.std().item(),
        times_t.min().item(),
        times_t.max().item(),
        times_t.median().item(),
        0,  # gpu_memory_mb not measured for JAX
        oom=False,
        n_iters=n_iters,
    )


# =============================================================================
# JIT Overhead Measurement
# =============================================================================

def measure_jit_overhead(
    n: int,
    d: int,
    eps: float,
    n_iters: int,
    device: torch.device,
    warm_reps: int = 10,
    include_flash_symmetric: bool = True,
    include_flash_alternating: bool = True,
    include_geomloss: bool = True,
    include_ott: bool = True,
    verbose: bool = True,
    allow_tf32: bool = False,
) -> List[JITOverheadResult]:
    """Measure JIT compilation overhead for each method.

    For each method:
    1. Clear all caches (Triton, KeOps, JAX)
    2. Measure first call time (cold start, includes JIT compilation)
    3. Measure average of subsequent calls (warm, steady-state)
    4. Compute overhead = cold - warm

    Args:
        n: Number of points
        d: Feature dimension
        eps: Regularization epsilon
        n_iters: Number of Sinkhorn iterations
        device: CUDA device
        warm_reps: Number of warm repetitions for averaging
        include_*: Which methods to measure
        verbose: Print progress

    Returns:
        List of JITOverheadResult for each method
    """
    results = []

    # Setup data (same for all methods)
    torch.manual_seed(0)
    x = torch.randn(n, d, device=device, dtype=torch.float32)
    y = torch.randn(n, d, device=device, dtype=torch.float32)
    a = torch.rand(n, device=device, dtype=torch.float32) + 0.1
    b = torch.rand(n, device=device, dtype=torch.float32) + 0.1
    a = a / a.sum()
    b = b / b.sum()

    if verbose:
        print(f"\n{'='*70}")
        print(f"JIT OVERHEAD MEASUREMENT (n={n}, d={d}, eps={eps}, iters={n_iters})")
        print(f"{'='*70}")

    # -------------------------------------------------------------------------
    # FlashSinkhorn (symmetric backend)
    # -------------------------------------------------------------------------
    if include_flash_symmetric:
        try:
            from flash_sinkhorn import SamplesLoss

            # Clear Triton cache for this config (create fresh instance)
            loss_fn = SamplesLoss(
                "sinkhorn", backend="symmetric", use_epsilon_scaling=False,
                eps=eps, n_iters=n_iters, debias=False, normalize=False,
                allow_tf32=False, autotune=True,
            )

            # Cold start (first call - triggers Triton JIT + autotuning)
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = loss_fn(a, x, b, y)
            torch.cuda.synchronize()
            cold_ms = (time.perf_counter() - start) * 1000

            # Warm calls (subsequent - uses cached kernels)
            warm_times = []
            for _ in range(warm_reps):
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = loss_fn(a, x, b, y)
                torch.cuda.synchronize()
                warm_times.append((time.perf_counter() - start) * 1000)
            warm_ms = sum(warm_times) / len(warm_times)

            overhead_ms = cold_ms - warm_ms
            ratio = cold_ms / warm_ms if warm_ms > 0 else float('inf')

            results.append(JITOverheadResult(
                "flash_symmetric", n, d, eps, cold_ms, warm_ms, overhead_ms, ratio
            ))
            if verbose:
                print(f"  Flash (symmetric): cold={cold_ms:8.1f}ms  warm={warm_ms:6.1f}ms  "
                      f"overhead={overhead_ms:8.1f}ms  ratio={ratio:5.1f}x")
        except Exception as e:
            if verbose:
                print(f"  Flash (symmetric): FAILED ({e})")

    # -------------------------------------------------------------------------
    # FlashSinkhorn (alternating backend)
    # -------------------------------------------------------------------------
    if include_flash_alternating:
        try:
            from flash_sinkhorn import SamplesLoss

            loss_fn = SamplesLoss(
                "sinkhorn", backend="alternating", use_epsilon_scaling=False,
                eps=eps, n_iters=n_iters, debias=False, normalize=False,
                allow_tf32=False, autotune=True,
            )

            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = loss_fn(a, x, b, y)
            torch.cuda.synchronize()
            cold_ms = (time.perf_counter() - start) * 1000

            warm_times = []
            for _ in range(warm_reps):
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = loss_fn(a, x, b, y)
                torch.cuda.synchronize()
                warm_times.append((time.perf_counter() - start) * 1000)
            warm_ms = sum(warm_times) / len(warm_times)

            overhead_ms = cold_ms - warm_ms
            ratio = cold_ms / warm_ms if warm_ms > 0 else float('inf')

            results.append(JITOverheadResult(
                "flash_alternating", n, d, eps, cold_ms, warm_ms, overhead_ms, ratio
            ))
            if verbose:
                print(f"  Flash (altern.):   cold={cold_ms:8.1f}ms  warm={warm_ms:6.1f}ms  "
                      f"overhead={overhead_ms:8.1f}ms  ratio={ratio:5.1f}x")
        except Exception as e:
            if verbose:
                print(f"  Flash (altern.):   FAILED ({e})")

    # -------------------------------------------------------------------------
    # GeomLoss KeOps
    # -------------------------------------------------------------------------
    if include_geomloss:
        try:
            # Force KeOps to recompile by using a fresh import context
            # Note: KeOps caches compiled kernels on disk, so true cold start
            # requires clearing ~/.cache/keops* (not done here for safety)
            from geomloss._legacy.sinkhorn_divergence import log_weights, sinkhorn_loop
            from geomloss._legacy.sinkhorn_samples import lse_genred, softmin_online

            eps_list = [eps] * n_iters
            a_log = log_weights(a)
            b_log = log_weights(b)
            my_lse = lse_genred("SqDist(X,Y)", d)
            softmin = partial(softmin_online, log_conv=my_lse)
            C_xy = (x, y.detach())
            C_yx = (y, x.detach())

            # Cold start
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = sinkhorn_loop(
                softmin, a_log, b_log, None, None, C_xy, C_yx, eps_list,
                rho=None, debias=False, last_extrapolation=True,
            )
            torch.cuda.synchronize()
            cold_ms = (time.perf_counter() - start) * 1000

            # Warm calls
            warm_times = []
            for _ in range(warm_reps):
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = sinkhorn_loop(
                    softmin, a_log, b_log, None, None, C_xy, C_yx, eps_list,
                    rho=None, debias=False, last_extrapolation=True,
                )
                torch.cuda.synchronize()
                warm_times.append((time.perf_counter() - start) * 1000)
            warm_ms = sum(warm_times) / len(warm_times)

            overhead_ms = cold_ms - warm_ms
            ratio = cold_ms / warm_ms if warm_ms > 0 else float('inf')

            results.append(JITOverheadResult(
                "geomloss_keops", n, d, eps, cold_ms, warm_ms, overhead_ms, ratio
            ))
            if verbose:
                print(f"  GeomLoss KeOps:    cold={cold_ms:8.1f}ms  warm={warm_ms:6.1f}ms  "
                      f"overhead={overhead_ms:8.1f}ms  ratio={ratio:5.1f}x")
        except Exception as e:
            if verbose:
                print(f"  GeomLoss KeOps:    FAILED ({e})")

    # -------------------------------------------------------------------------
    # OTT-JAX
    # -------------------------------------------------------------------------
    if include_ott:
        try:
            import jax
            import jax.numpy as jnp
            from jax import config as jax_config
            from ott.geometry import pointcloud
            from ott.problems.linear import linear_problem
            from ott.solvers.linear import sinkhorn

            # Match PyTorch TF32 setting for fair comparison
            jax_precision = "default" if allow_tf32 else "highest"
            jax_config.update("jax_default_matmul_precision", jax_precision)

            x_jax = jnp.array(x.cpu().numpy())
            y_jax = jnp.array(y.cpu().numpy())
            a_jax = jnp.array(a.cpu().numpy())
            b_jax = jnp.array(b.cpu().numpy())

            solver = sinkhorn.Sinkhorn(
                threshold=-1.0,
                max_iterations=n_iters,
                min_iterations=n_iters,
            )

            geom = pointcloud.PointCloud(x_jax, y_jax, epsilon=eps, batch_size=256)
            prob = linear_problem.LinearProblem(geom, a=a_jax, b=b_jax)

            @jax.jit
            def solve():
                return solver(prob)

            # Cold start (includes XLA JIT compilation)
            start = time.perf_counter()
            out = solve()
            _ = jax.block_until_ready(out.f)
            cold_ms = (time.perf_counter() - start) * 1000

            # Warm calls
            warm_times = []
            for _ in range(warm_reps):
                start = time.perf_counter()
                out = solve()
                _ = jax.block_until_ready(out.f)
                warm_times.append((time.perf_counter() - start) * 1000)
            warm_ms = sum(warm_times) / len(warm_times)

            overhead_ms = cold_ms - warm_ms
            ratio = cold_ms / warm_ms if warm_ms > 0 else float('inf')

            results.append(JITOverheadResult(
                "ott_jax", n, d, eps, cold_ms, warm_ms, overhead_ms, ratio
            ))
            if verbose:
                print(f"  OTT-JAX:           cold={cold_ms:8.1f}ms  warm={warm_ms:6.1f}ms  "
                      f"overhead={overhead_ms:8.1f}ms  ratio={ratio:5.1f}x")
        except Exception as e:
            if verbose:
                print(f"  OTT-JAX:           FAILED ({e})")

    return results


def print_jit_overhead_summary(results: List[JITOverheadResult]) -> None:
    """Print a summary table of JIT overhead results."""
    if not results:
        print("No JIT overhead results to display.")
        return

    print(f"\n{'='*70}")
    print("JIT OVERHEAD SUMMARY")
    print(f"{'='*70}")
    print(f"{'Method':<18} {'Cold (ms)':>10} {'Warm (ms)':>10} {'Overhead (ms)':>14} {'Ratio':>8}")
    print("-" * 70)

    for r in results:
        print(f"{r.method:<18} {r.cold_start_ms:>10.1f} {r.warm_ms:>10.1f} "
              f"{r.jit_overhead_ms:>14.1f} {r.overhead_ratio:>7.1f}x")

    # Find best/worst
    if len(results) >= 2:
        sorted_by_overhead = sorted(results, key=lambda r: r.jit_overhead_ms)
        best = sorted_by_overhead[0]
        worst = sorted_by_overhead[-1]
        print("-" * 70)
        print(f"Lowest overhead:  {best.method} ({best.jit_overhead_ms:.1f}ms)")
        print(f"Highest overhead: {worst.method} ({worst.jit_overhead_ms:.1f}ms)")
        if best.jit_overhead_ms > 0:
            print(f"Ratio: {worst.method} has {worst.jit_overhead_ms/best.jit_overhead_ms:.1f}x "
                  f"more JIT overhead than {best.method}")


def save_jit_overhead_csv(results: List[JITOverheadResult], output_path: Path) -> None:
    """Save JIT overhead results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method", "n", "d", "eps",
            "cold_start_ms", "warm_ms", "jit_overhead_ms", "overhead_ratio"
        ])
        for r in results:
            writer.writerow([
                r.method, r.n, r.d, r.eps,
                f"{r.cold_start_ms:.2f}", f"{r.warm_ms:.2f}",
                f"{r.jit_overhead_ms:.2f}", f"{r.overhead_ratio:.2f}"
            ])

    print(f"Saved JIT overhead results to {output_path}")


# =============================================================================
# Main Benchmark Runner
# =============================================================================

def run_forward_benchmark(
    sizes: List[int],
    dims: List[int],
    eps: float,
    n_iters: int,
    device: torch.device,
    warmup: int,
    rep: int,
    include_flash_symmetric: bool = True,
    include_flash_alternating: bool = True,
    include_ott: bool = True,
    include_geomloss: bool = True,
    include_tensorized: bool = False,
    max_dense_size: int = 8192,
    verbose: bool = True,
    nvtx: bool = False,
    allow_tf32: bool = False,
    rmae_check: bool = True,
    dataset: str = "gaussian",
    include_srot: bool = False,
    srot_slices: Optional[List[int]] = None,
    srot_delta: float = 1e-8,
    include_sinkslot: bool = False,
    sinkslot_slices: Optional[List[int]] = None,
    include_sinkslotcuda: bool = False,
    sinkslotcuda_slices: Optional[List[int]] = None,
    include_sparsink: bool = False,
    sparsink_s: Optional[List[int]] = None,
    sparsink_replicates: int = 10,
    only_sparsink_method: Optional[str] = None,
    stop: "StopCfg" = None,
    seed: int = 0,
) -> List[TimingResult]:
    """Run forward pass benchmark.

    Sizes are run large->small as best practice. With bucketed autotune cache
    keys (CACHE_KEY = n // 32), cross-size cache pollution is minimal.

    FlashSinkhorn backends:
    - flash_symmetric: GeomLoss-style symmetric updates (compare with GeomLoss)
    - flash_alternating: OTT-JAX-style alternating updates (compare with OTT-JAX)

    rmae_check: if True, also run a converged Sinkhorn solve via POT (CPU, once per
    (dataset, n, d, eps), cached) and record each PyTorch-side method's relative error
    against it -- the RMAE metric from the Spar-Sink paper. Not computed for OTT-JAX
    (different RNG; see bench_ott_jax_online).

    dataset: "gaussian" (default), "8gaussians", "half_moon", or "two_rings"; see sample_point_cloud(). Not
    applied to OTT-JAX, which draws its own point cloud via JAX's RNG.

    only_sparsink_method: restrict the Spar-Sink/Rand-Sink block to just this one
    method ("spar_sink" or "rand_sink"). Without this, include_sparsink=True always
    runs BOTH SPARSINK_METHODS regardless of which single method was actually
    requested (e.g. via --only spar_sink) -- meaning a job launched for "spar_sink"
    silently produces a full set of "rand_sink" rows too. Not wrong data (each row
    is correctly tagged by its own method), but doubles the row count and wastes
    compute when the caller only wanted one. Pass the actual requested method name
    here to get exactly the rows asked for.

    seed: data-generation seed only (x, y, a, b), passed through to every method.
    Method-internal randomness (SROT/SinkSLOT's slice projections, Spar-Sink's
    kernel sampling) stays at its own independent default, unaffected by this --
    varying seed changes only the underlying problem instance. Not applied to
    OTT-JAX, which draws its own point cloud via JAX's RNG.
    """
    results = []

    for d in dims:
        if verbose:
            print(f"\n{'#'*70}")
            print(f"# Dimension d={d}")
            print(f"{'#'*70}")

        sizes_sorted = sorted(sizes, reverse=True)

        for n in sizes_sorted:
            if verbose:
                print(f"\n{'='*60}")
                print(f"Benchmarking n={n}, d={d}, eps={eps}, iters={n_iters}")
                print(f"{'='*60}")

            # FlashSinkhorn (symmetric backend - GeomLoss-style Jacobi updates)
            if include_flash_symmetric:
                res = bench_flashsinkhorn(
                    n, n, d, eps, n_iters, device, warmup, rep, nvtx=nvtx, backend="symmetric",
                    allow_tf32=allow_tf32, rmae_check=rmae_check, dataset=dataset, stop=stop,
                    seed=seed,
                )
                res.dataset = dataset
                res.tf32 = allow_tf32
                res.seed = seed
                results.append(res)
                if verbose:
                    status = "OOM" if res.oom else f"{res.mean_ms:.3f} +/- {res.std_ms:.3f} ms"
                    print(f"  Flash (symmetric):     {status}")

            # FlashSinkhorn (alternating backend - OTT-JAX-style Gauss-Seidel updates)
            if include_flash_alternating:
                res = bench_flashsinkhorn(
                    n, n, d, eps, n_iters, device, warmup, rep, nvtx=nvtx, backend="alternating",
                    allow_tf32=allow_tf32, rmae_check=rmae_check, dataset=dataset, stop=stop,
                    seed=seed,
                )
                res.dataset = dataset
                res.tf32 = allow_tf32
                res.seed = seed
                results.append(res)
                if verbose:
                    status = "OOM" if res.oom else f"{res.mean_ms:.3f} +/- {res.std_ms:.3f} ms"
                    print(f"  Flash (alternating):   {status}")

            # GeomLoss online (KeOps)
            if include_geomloss:
                res = bench_geomloss_online(
                    n, n, d, eps, n_iters, device, warmup, rep, nvtx=nvtx, rmae_check=rmae_check,
                    dataset=dataset, seed=seed,
                )
                res.dataset = dataset
                res.tf32 = allow_tf32
                res.seed = seed
                results.append(res)
                if verbose:
                    status = "OOM" if res.oom else f"{res.mean_ms:.3f} +/- {res.std_ms:.3f} ms"
                    print(f"  GeomLoss KeOps:        {status}")

            # GeomLoss tensorized (dense, small sizes only)
            if include_geomloss and include_tensorized and n <= max_dense_size:
                res = bench_geomloss_tensorized(
                    n, n, d, eps, n_iters, device, warmup, rep, nvtx=nvtx, rmae_check=rmae_check,
                    dataset=dataset, seed=seed,
                )
                res.dataset = dataset
                res.tf32 = allow_tf32
                res.seed = seed
                results.append(res)
                if verbose:
                    status = "OOM" if res.oom else f"{res.mean_ms:.3f} +/- {res.std_ms:.3f} ms"
                    print(f"  GeomLoss Tensorized:   {status}")

            # SROT (dense, small sizes only). One row per L: pi_SOT and therefore the
            # optimum both change with it, so the rows are not interchangeable.
            if include_srot and n <= max_dense_size:
                for slices in (srot_slices or [50]):
                    res = bench_srot(
                        n, n, d, eps, n_iters, device, warmup, rep, nvtx=nvtx,
                        allow_tf32=allow_tf32, dataset=dataset, rmae_check=rmae_check,
                        slices=slices, delta=srot_delta, stop=stop, seed=seed,
                    )
                    res.seed = seed
                    results.append(res)
                    if verbose:
                        status = "OOM" if res.oom else f"{res.mean_ms:.3f} +/- {res.std_ms:.3f} ms"
                        plan = "" if res.setup_ms is None else f" (plan {res.setup_ms:.1f} ms)"
                        print(f"  SROT L={slices}: {status}{plan}")
            elif verbose and include_srot and n > max_dense_size:
                print(f"  SROT:                  SKIPPED (n > max_dense_size={max_dense_size})")

            # SinkSLOT (fused-Triton, sparse O(L(N+M)) -- not gated on max_dense_size,
            # since the support is never densified). One row per L.
            if include_sinkslot:
                for slices in (sinkslot_slices or [50]):
                    res = bench_sinkslot(
                        n, n, d, eps, n_iters, device, warmup, rep, nvtx=nvtx,
                        allow_tf32=allow_tf32, dataset=dataset, rmae_check=rmae_check,
                        slices=slices, stop=stop, seed=seed,
                    )
                    res.seed = seed
                    results.append(res)
                    if verbose:
                        status = "OOM" if res.oom else f"{res.mean_ms:.3f} +/- {res.std_ms:.3f} ms"
                        extra = "" if res.nnz is None else f" (nnz {res.nnz}, setup {res.setup_ms:.1f} ms)"
                        print(f"  SinkSLOT L={slices}: {status}{extra}")

            if include_sinkslotcuda:
                for slices in (sinkslotcuda_slices or [50]):
                    res = bench_sinkslotcuda(
                        n, n, d, eps, n_iters, device, warmup, rep, nvtx=nvtx,
                        allow_tf32=allow_tf32, dataset=dataset, rmae_check=rmae_check,
                        slices=slices, stop=stop, seed=seed,
                    )
                    res.seed = seed
                    results.append(res)
                    if verbose:
                        status = "OOM" if res.oom else f"{res.mean_ms:.3f} +/- {res.std_ms:.3f} ms"
                        extra = "" if res.nnz is None else f" (nnz {res.nnz}, setup {res.setup_ms:.1f} ms)"
                        print(f"  SinkSLOT-CUDA L={slices}: {status}{extra}")

            # Spar-Sink / Rand-Sink (dense probability build, so gated like the other
            # O(n*m)-setup baselines). One row per (method, s). only_sparsink_method
            # restricts this to a single method instead of always running both.
            if include_sparsink and n <= max_dense_size:
                _sparsink_methods = (
                    (only_sparsink_method,) if only_sparsink_method is not None
                    else SPARSINK_METHODS
                )
                for sparsink_method in _sparsink_methods:
                    for s_size in (sparsink_s or [2000]):
                        res = bench_sparsink(
                            n, n, d, eps, n_iters, device, warmup, rep, nvtx=nvtx,
                            allow_tf32=allow_tf32, dataset=dataset, rmae_check=rmae_check,
                            method=sparsink_method, sample_size=s_size,
                            replicates=sparsink_replicates, stop=stop, seed=seed,
                        )
                        res.seed = seed
                        results.append(res)
                        if verbose:
                            status = "OOM" if res.oom else f"{res.mean_ms:.3f} +/- {res.std_ms:.3f} ms"
                            extra = "" if res.nnz is None else f" (nnz {res.nnz}, setup {res.setup_ms:.1f} ms)"
                            warn = "" if not res.empty_lines else f" [{res.empty_lines} empty rows/cols]"
                            print(f"  {sparsink_method} s={s_size}: {status}{extra}{warn}")
            elif verbose and include_sparsink and n > max_dense_size:
                print(f"  Spar-Sink/Rand-Sink:   SKIPPED (n > max_dense_size={max_dense_size})")

            # OTT-JAX online
            if include_ott:
                res = bench_ott_jax_online(
                    n, n, d, eps, n_iters, device, warmup, rep, nvtx=nvtx, allow_tf32=allow_tf32
                )
                if res:
                    res.tf32 = allow_tf32
                    results.append(res)
                    if verbose:
                        status = "OOM" if res.oom else f"{res.mean_ms:.3f} +/- {res.std_ms:.3f} ms"
                        print(f"  OTT-JAX Online:        {status}")
                elif verbose:
                    print(f"  OTT-JAX Online:        NOT AVAILABLE")

            # OTT-JAX dense (small sizes only)
            if include_tensorized and n <= max_dense_size and include_ott:
                res = bench_ott_jax_dense(
                    n, n, d, eps, n_iters, device, warmup, rep, nvtx=nvtx, allow_tf32=allow_tf32
                )
                if res:
                    res.tf32 = allow_tf32
                    results.append(res)
                    if verbose:
                        status = "OOM" if res.oom else f"{res.mean_ms:.3f} +/- {res.std_ms:.3f} ms"
                        print(f"  OTT-JAX Dense:         {status}")

            gc.collect()
            torch.cuda.empty_cache()

    return results


FORWARD_CSV_COLUMNS = [
    "dataset", "tf32", "method", "n", "m", "d", "eps", "seed",
    "mean_ms", "std_ms", "min_ms", "max_ms", "median_ms", "gpu_memory_mb",
    "oom", "n_iters", "iters_run", "converged", "final_viol", "hit_max_iters",
    "cost_gap_pct", "barycentric_sym", "mass", "marg_viol",
    "plan_empty_rows", "plan_empty_cols",
    "rmae_pct", "rmae_std", "srot_slices", "sample_size",
    "nnz", "empty_lines", "valid_replicates", "setup_ms",
]


def _forward_row(r: TimingResult) -> dict:
    """Render one result as a CSV row dict."""
    if r.oom or r.mean_ms != r.mean_ms:
        timings = {
            "mean_ms": "OOM", "std_ms": "", "min_ms": "", "max_ms": "",
            "median_ms": "", "gpu_memory_mb": "", "oom": True, "rmae_pct": "",
        }
    else:
        timings = {
            "mean_ms": f"{r.mean_ms:.4f}", "std_ms": f"{r.std_ms:.4f}",
            "min_ms": f"{r.min_ms:.4f}", "max_ms": f"{r.max_ms:.4f}",
            "median_ms": f"{r.median_ms:.4f}", "gpu_memory_mb": f"{r.gpu_memory_mb:.1f}",
            "oom": False,
            "rmae_pct": f"{r.rmae_pct:.4f}" if r.rmae_pct is not None else "N/A",
        }
    return {
        "dataset": r.dataset, "tf32": r.tf32, "method": r.method,
        "n": r.n, "m": r.m, "d": r.d, "seed": r.seed,
        "eps": r.eps, "n_iters": r.n_iters,
        "iters_run": r.iters_run if r.iters_run is not None else "N/A",
        "converged": r.converged if r.converged is not None else "N/A",
        "final_viol": f"{r.final_viol:.3e}" if r.final_viol is not None else "N/A",
        "hit_max_iters": r.hit_max_iters if r.hit_max_iters is not None else "N/A",
        "cost_gap_pct": f"{r.cost_gap_pct:.4f}" if r.cost_gap_pct is not None else "N/A",
        "barycentric_sym": f"{r.barycentric_sym:.6e}" if r.barycentric_sym is not None else "N/A",
        "mass": f"{r.mass:.6f}" if r.mass is not None else "N/A",
        "marg_viol": f"{r.marg_viol:.3e}" if r.marg_viol is not None else "N/A",
        "plan_empty_rows": r.plan_empty_rows if r.plan_empty_rows is not None else "N/A",
        "plan_empty_cols": r.plan_empty_cols if r.plan_empty_cols is not None else "N/A",
        "srot_slices": r.srot_slices if r.srot_slices is not None else "N/A",
        "sample_size": r.sample_size if r.sample_size is not None else "N/A",
        "nnz": r.nnz if r.nnz is not None else "N/A",
        "empty_lines": r.empty_lines if r.empty_lines is not None else "N/A",
        "valid_replicates": r.valid_replicates if r.valid_replicates is not None else "N/A",
        "rmae_std": f"{r.rmae_std:.4f}" if r.rmae_std is not None else "N/A",
        "setup_ms": f"{r.setup_ms:.4f}" if r.setup_ms is not None else "N/A",
        **timings,
    }


def _forward_key(row: dict) -> tuple:
    """Unique row identity.

    Includes tf32 so a strict-FP32 run and a TF32 run of the same configuration are
    distinct rows rather than one silently overwriting the other, and srot_slices so
    SROT rows at different L do not collide (it is "N/A" for every other method).
    """
    return (
        row["dataset"], str(row["tf32"]), row["method"], str(row["n"]), str(row["m"]),
        str(row["d"]), str(row["eps"]), str(row["n_iters"]), str(row["srot_slices"]),
        str(row["sample_size"]),
    )


def save_results_csv(results: List[TimingResult], output_path: Path) -> None:
    """Save all results into a single CSV, merging with any existing rows.

    Every run of the benchmark -- across datasets, eps values and dims -- appends
    into one table, with dataset/eps/d/n as ordinary columns. Rows are keyed by
    _forward_key(), so re-running a configuration overwrites its own row rather
    than duplicating it. Delete the file to start clean.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    merged: Dict[tuple, dict] = {}
    if output_path.exists():
        with open(output_path, newline="") as f:
            for row in csv.DictReader(f):
                if all(col in row for col in FORWARD_CSV_COLUMNS):
                    merged[_forward_key(row)] = row
        print(f"  Loaded {len(merged)} existing rows from {output_path}")

    for r in results:
        row = _forward_row(r)
        merged[_forward_key(row)] = row

    def sort_key(row: dict) -> tuple:
        return (
            row["dataset"], str(row["tf32"]), int(row["d"]), float(row["eps"]),
            int(row["n"]), row["method"],
        )

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FORWARD_CSV_COLUMNS)
        writer.writeheader()
        for row in sorted(merged.values(), key=sort_key):
            writer.writerow(row)

    print(f"\nSaved {len(merged)} results to {output_path}")


SPEEDUP_CSV_COLUMNS = [
    "dataset", "tf32", "d", "eps", "n",
    "flash_symmetric_ms", "flash_alternating_ms", "keops_ms", "ott_jax_ms",
    "online_vs_keops", "ott_vs_ott_jax",
]


def save_speedup_csv(results: List[TimingResult], output_path: Path) -> None:
    """Save one speedup table covering every dataset/dim/eps/size, merging with existing rows.

    dataset, tf32, d and eps are columns rather than separate files, so the whole
    sweep lands in a single table. Rows are keyed by (dataset, tf32, d, eps, n).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    merged: Dict[tuple, dict] = {}
    if output_path.exists():
        with open(output_path, newline="") as f:
            for row in csv.DictReader(f):
                if all(col in row for col in SPEEDUP_CSV_COLUMNS):
                    merged[(row["dataset"], row["tf32"], row["d"], row["eps"], row["n"])] = row

    def pick(subset: List[TimingResult], method: str) -> Optional[TimingResult]:
        matches = [r for r in subset if r.method == method]
        return matches[0] if matches else None

    def ms_of(res: Optional[TimingResult]) -> Optional[float]:
        return None if res is None or res.oom else res.mean_ms

    def fmt_ms(res: Optional[TimingResult], ms: Optional[float]) -> str:
        if res is None:
            return "N/A"
        if res.oom:
            return "OOM"
        return f"{ms:.3f}" if ms is not None else "OOM"

    def speedup(baseline_ms: Optional[float], our_ms: Optional[float]) -> str:
        if baseline_ms is None or our_ms is None:
            return "N/A"
        return f"{baseline_ms / our_ms:.2f}x"

    groups = sorted({(r.dataset, str(r.tf32), r.d, r.eps, r.n) for r in results})
    for dataset, tf32, d, eps, n in groups:
        subset = [
            r for r in results
            if r.dataset == dataset and str(r.tf32) == tf32
            and r.d == d and r.eps == eps and r.n == n
        ]

        flash_symmetric_res = pick(subset, "flash_symmetric")
        flash_alternating_res = pick(subset, "flash_alternating")
        gl_res = pick(subset, "geomloss_online")
        ott_res = pick(subset, "ott_jax_online")

        flash_symmetric_ms = ms_of(flash_symmetric_res)
        flash_alternating_ms = ms_of(flash_alternating_res)
        gl_ms = ms_of(gl_res)
        ott_ms = ms_of(ott_res)

        merged[(dataset, tf32, str(d), str(eps), str(n))] = {
            "dataset": dataset, "tf32": tf32, "d": d, "eps": eps, "n": n,
            "flash_symmetric_ms": fmt_ms(flash_symmetric_res, flash_symmetric_ms),
            "flash_alternating_ms": fmt_ms(flash_alternating_res, flash_alternating_ms),
            "keops_ms": fmt_ms(gl_res, gl_ms),
            "ott_jax_ms": fmt_ms(ott_res, ott_ms),
            "online_vs_keops": speedup(gl_ms, flash_symmetric_ms),
            "ott_vs_ott_jax": speedup(ott_ms, flash_alternating_ms),
        }

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SPEEDUP_CSV_COLUMNS)
        writer.writeheader()
        for key in sorted(merged, key=lambda k: (k[0], k[1], int(k[2]), float(k[3]), int(k[4]))):
            writer.writerow(merged[key])

    print(f"Saved speedup table to {output_path}")


def print_summary(results: List[TimingResult]) -> None:
    """Print summary table per dimension.

    Shows both FlashSinkhorn backends and their speedups vs references:
    - flash_symmetric vs GeomLoss (KeOps)
    - flash_alternating vs OTT-JAX
    """
    dims = sorted(set(r.d for r in results))

    for d in dims:
        subset = [r for r in results if r.d == d]
        sizes = sorted(set(r.n for r in subset))

        print(f"\n{'='*120}")
        print(f"FORWARD PASS SUMMARY: d={d}")
        print(f"{'='*120}")

        header = f"{'n':>8s}  {'F.symm':>10s}  {'F.alt':>10s}  {'KeOps':>10s}  {'OTT-JAX':>10s}  {'symm/KeOps':>12s}  {'alt/OTT-JAX':>12s}"
        print(header)
        print("-" * 120)

        for n in sizes:
            flash_symmetric = [r for r in subset if r.n == n and r.method == "flash_symmetric"]
            flash_alternating = [r for r in subset if r.n == n and r.method == "flash_alternating"]
            gl = [r for r in subset if r.n == n and r.method == "geomloss_online"]
            ott = [r for r in subset if r.n == n and r.method == "ott_jax_online"]

            flash_symmetric_res = flash_symmetric[0] if flash_symmetric else None
            flash_alternating_res = flash_alternating[0] if flash_alternating else None
            gl_res = gl[0] if gl else None
            ott_res = ott[0] if ott else None

            flash_symmetric_ms = None if flash_symmetric_res is None or flash_symmetric_res.oom else flash_symmetric_res.mean_ms
            flash_alternating_ms = None if flash_alternating_res is None or flash_alternating_res.oom else flash_alternating_res.mean_ms
            gl_ms = None if gl_res is None or gl_res.oom else gl_res.mean_ms
            ott_ms = None if ott_res is None or ott_res.oom else ott_res.mean_ms

            def fmt_cell(res: Optional[TimingResult], ms: Optional[float]) -> str:
                if res is None:
                    return "N/A"
                if res.oom:
                    return "OOM"
                return f"{ms:0.2f}" if ms is not None else "OOM"

            print(f"{n:>8d}", end="")
            print(f"  {fmt_cell(flash_symmetric_res, flash_symmetric_ms):>10s}", end="")
            print(f"  {fmt_cell(flash_alternating_res, flash_alternating_ms):>10s}", end="")
            print(f"  {fmt_cell(gl_res, gl_ms):>10s}", end="")
            print(f"  {fmt_cell(ott_res, ott_ms):>10s}", end="")

            # Speedup: flash_symmetric vs KeOps
            if flash_symmetric_ms is not None and gl_ms is not None:
                print(f"  {gl_ms/flash_symmetric_ms:>11.1f}x", end="")
            else:
                print(f"  {'N/A':>12s}", end="")

            # Speedup: flash_alternating vs OTT-JAX
            if flash_alternating_ms is not None and ott_ms is not None:
                print(f"  {ott_ms/flash_alternating_ms:>11.1f}x")
            else:
                print(f"  {'N/A':>12s}")

        print("=" * 120)


def verify_loss_parity(
    n: int = 1000,
    d: int = 64,
    eps: float = 0.1,
    n_iters: int = 10,
    device: torch.device = None,
) -> bool:
    """Verify loss parity between FlashSinkhorn backends and their references.

    Tests both backends:
    - flash_symmetric vs GeomLoss (should match, both use symmetric updates)
    - flash_alternating vs OTT-JAX (should match, both use alternating updates)

    Returns True if both comparisons pass (relative error < 1%).
    """
    if device is None:
        device = torch.device("cuda")

    print("\n" + "=" * 70)
    print("LOSS PARITY VERIFICATION")
    print("=" * 70)
    print(f"  n={n}, d={d}, eps={eps}, n_iters={n_iters}")

    # Setup data
    torch.manual_seed(42)
    x = torch.randn(n, d, device=device, dtype=torch.float32)
    y = torch.randn(n, d, device=device, dtype=torch.float32)
    a = torch.rand(n, device=device, dtype=torch.float32) + 0.1
    b = torch.rand(n, device=device, dtype=torch.float32) + 0.1
    a = a / a.sum()
    b = b / b.sum()

    results = {}
    all_passed = True

    # =========================================================================
    # Test 1: flash_symmetric vs GeomLoss (symmetric updates)
    # =========================================================================
    print("\n  --- Test 1: flash_symmetric vs GeomLoss ---")

    # FlashSinkhorn online loss
    try:
        from flash_sinkhorn import SamplesLoss
        flash_symmetric_fn = SamplesLoss(
            "sinkhorn", backend="symmetric", use_epsilon_scaling=False,
            eps=eps, n_iters=n_iters, debias=False, normalize=False,
            last_extrapolation=False, allow_tf32=False,
        )
        results["flash_symmetric"] = flash_symmetric_fn(a, x, b, y).item()
        print(f"    Flash (online): loss={results['flash_symmetric']:.6f}")
    except Exception as e:
        print(f"    Flash (online): FAILED ({e})")
        all_passed = False

    # GeomLoss loss
    try:
        from geomloss._legacy.sinkhorn_divergence import log_weights, sinkhorn_cost, sinkhorn_loop
        from geomloss._legacy.sinkhorn_samples import lse_genred, softmin_online

        eps_list = [eps] * n_iters
        a_log = log_weights(a)
        b_log = log_weights(b)
        my_lse = lse_genred("SqDist(X,Y)", d)
        softmin = partial(softmin_online, log_conv=my_lse)

        C_xy = (x, y.detach())
        C_yx = (y, x.detach())
        # Match FlashSinkhorn: last_extrapolation=False for fair comparison
        _, _, g_ab, f_ba = sinkhorn_loop(
            softmin, a_log, b_log, None, None, C_xy, C_yx, eps_list,
            rho=None, debias=False, last_extrapolation=False,
        )
        results["geomloss"] = sinkhorn_cost(
            eps, None, a, b, None, None, g_ab, f_ba,
            batch=False, debias=False, potentials=False,
        ).item()
        print(f"    GeomLoss:       loss={results['geomloss']:.6f}")
    except Exception as e:
        print(f"    GeomLoss: FAILED ({e})")
        all_passed = False

    # Check online vs GeomLoss parity
    if "flash_symmetric" in results and "geomloss" in results:
        rel_diff = abs(results["flash_symmetric"] - results["geomloss"]) / max(
            abs(results["flash_symmetric"]), 1e-8
        )
        passed = rel_diff < 0.01
        symbol = "✓" if passed else "✗"
        print(f"    {symbol} Relative diff: {rel_diff:.2e} ({'PASS' if passed else 'FAIL'})")
        all_passed = all_passed and passed

    # =========================================================================
    # Test 2: flash_alternating vs OTT-JAX (alternating updates)
    # =========================================================================
    print("\n  --- Test 2: flash_alternating vs OTT-JAX ---")

    # FlashSinkhorn ott loss
    try:
        flash_alternating_fn = SamplesLoss(
            "sinkhorn", backend="alternating", use_epsilon_scaling=False,
            eps=eps, n_iters=n_iters, debias=False, normalize=False,
            allow_tf32=False,
        )
        results["flash_alternating"] = flash_alternating_fn(a, x, b, y).item()
        print(f"    Flash (ott):    loss={results['flash_alternating']:.6f}")
    except Exception as e:
        print(f"    Flash (ott): FAILED ({e})")
        all_passed = False

    # OTT-JAX loss
    try:
        import jax
        import jax.numpy as jnp
        from jax import config as jax_config
        from ott.geometry import pointcloud
        from ott.problems.linear import linear_problem
        from ott.solvers.linear import sinkhorn

        jax_config.update("jax_default_matmul_precision", "highest")

        # Convert to JAX arrays (use same seed for fair comparison)
        x_jax = jnp.array(x.cpu().numpy())
        y_jax = jnp.array(y.cpu().numpy())
        a_jax = jnp.array(a.cpu().numpy())
        b_jax = jnp.array(b.cpu().numpy())

        solver = sinkhorn.Sinkhorn(
            threshold=-1.0,
            max_iterations=n_iters,
            min_iterations=n_iters,
        )
        geom = pointcloud.PointCloud(x_jax, y_jax, epsilon=eps, batch_size=256)
        prob = linear_problem.LinearProblem(geom, a=a_jax, b=b_jax)
        out = solver(prob)
        # Use dual objective <a, f> + <b, g> for fair comparison with FlashSinkhorn
        # (NOT reg_ot_cost which includes additional entropy terms)
        results["ott_jax"] = float(jnp.sum(a_jax * out.f) + jnp.sum(b_jax * out.g))
        print(f"    OTT-JAX:        loss={results['ott_jax']:.6f}")
    except Exception as e:
        print(f"    OTT-JAX: FAILED ({e})")

    # Check ott vs OTT-JAX parity
    if "flash_alternating" in results and "ott_jax" in results:
        rel_diff = abs(results["flash_alternating"] - results["ott_jax"]) / max(
            abs(results["flash_alternating"]), 1e-8
        )
        passed = rel_diff < 0.01
        symbol = "✓" if passed else "✗"
        print(f"    {symbol} Relative diff: {rel_diff:.2e} ({'PASS' if passed else 'FAIL'})")
        all_passed = all_passed and passed

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n  --- Summary ---")
    for name, loss in results.items():
        print(f"    {name:15s}: {loss:.6f}")

    if all_passed:
        print(f"\n  ✓ ALL PARITY TESTS PASSED")
    else:
        print(f"\n  ✗ SOME PARITY TESTS FAILED")

    print("=" * 70)
    return all_passed


def run_forward_benchmark_subprocess(
    sizes: List[int],
    dims: List[int],
    args: argparse.Namespace,
) -> List[TimingResult]:
    """Run forward benchmark with each (n, d) in a separate subprocess.

    With bucketed autotune cache keys, in-process mode is accurate for most
    use cases. Subprocess mode is still available for maximum isolation when
    exact reproducibility across runs is critical (e.g., paper figures).

    Args:
        sizes: Problem sizes to benchmark (will be sorted large->small).
        dims: Feature dimensions to benchmark.
        args: Parsed CLI args (forwarded to worker subprocess).

    Returns:
        Collected TimingResult list from all subprocesses.
    """
    results: List[TimingResult] = []
    sizes_sorted = sorted(sizes, reverse=True)

    # Build the list of (d, n) pairs: dims outer, sizes inner (large->small)
    pairs = [(d, n) for d in dims for n in sizes_sorted]
    total = len(pairs)

    for idx, (d, n) in enumerate(pairs, 1):
        print(f"  [{idx}/{total}] n={n:>7d}, d={d:>3d} ...", end="", flush=True, file=sys.stderr)

        # Build subprocess command, forwarding relevant flags
        cmd = [
            sys.executable, "-m", "sinkslot.bench.bench_forward",
            "--single-size", str(n),
            "--single-dim", str(d),
            "--eps", str(args.eps),
            "--n-iters", str(args.n_iters),
            "--warmup", str(args.warmup),
            "--rep", str(args.rep),
        ]
        if not args.tf32:
            cmd.append("--no-tf32")
        if args.no_flash_symmetric:
            cmd.append("--no-flash-symmetric")
        if args.no_flash_alternating:
            cmd.append("--no-flash-alternating")
        if args.no_sinkslot:
            cmd.append("--no-sinkslot")
        else:
            cmd.extend(["--sinkslot-slices", args.sinkslot_slices])
        if args.no_sinkslotcuda:
            cmd.append("--no-sinkslotcuda")
        else:
            cmd.extend(["--sinkslotcuda-slices", args.sinkslotcuda_slices])
        if args.no_sparsink:
            cmd.append("--no-sparsink")
        else:
            cmd.extend(["--sparsink-s", args.sparsink_s,
                        "--sparsink-replicates", str(args.sparsink_replicates)])
        if args.no_srot:
            cmd.append("--no-srot")
        else:
            cmd.extend(["--srot-slices", args.srot_slices, "--srot-delta", str(args.srot_delta)])
        if args.no_geomloss:
            cmd.append("--no-geomloss")
        if args.no_ott:
            cmd.append("--no-ott")
        if args.no_rmae_check:
            cmd.append("--no-rmae-check")
        if args.dataset != "gaussian":
            cmd.extend(["--dataset", args.dataset])
        if args.tensorized:
            cmd.extend(["--tensorized", "--max-dense-size", str(args.max_dense_size)])
        if args.only is not None:
            cmd.extend(["--only", args.only])

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if proc.returncode != 0:
            print(f" FAILED (exit code {proc.returncode})", file=sys.stderr)
            if proc.stderr.strip():
                # Show last few lines of stderr for debugging
                for line in proc.stderr.strip().splitlines()[-5:]:
                    print(f"    {line}", file=sys.stderr)
            continue

        # Parse JSON lines from stdout
        sub_results = []
        for line in proc.stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                sub_results.append(timing_result_from_json(line))
            except (json.JSONDecodeError, TypeError) as e:
                print(f" (parse error: {e})", end="", file=sys.stderr)

        results.extend(sub_results)

        # Print summary of timing for this size
        parts = []
        for r in sub_results:
            if r.oom:
                parts.append(f"{r.method}: OOM")
            else:
                parts.append(f"{r.method}: {r.mean_ms:.1f} ms")
        summary = ", ".join(parts) if parts else "no results"
        print(f" done ({summary})", file=sys.stderr)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified forward pass benchmark for FlashSinkhorn paper."
    )
    parser.add_argument(
        "--sizes", type=str,
        default="200000,100000,50000,20000,10000,8192,4096,2048,1024",
        help="Comma-separated sizes (sorted large->small internally)."
    )
    parser.add_argument(
        "--dims", type=str, default="3,8,64",
        help="Comma-separated dimensions to test."
    )
    parser.add_argument("--eps", type=float, default=0.1, help="Regularization epsilon.")
    parser.add_argument("--n-iters", type=int, default=10, help="Sinkhorn iterations.")
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Data-generation seed (x, y, a, b) only. Does not affect method-internal "
             "randomness (e.g. SROT/SinkSLOT's slice projections, Spar-Sink's kernel "
             "sampling), which stays independently seeded. Not applied to OTT-JAX.",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations.")
    parser.add_argument("--rep", type=int, default=50, help="Timed repetitions.")
    parser.add_argument("--no-ott", action="store_true", help="Skip OTT-JAX benchmarks.")
    parser.add_argument(
        "--no-rmae-check", action="store_true",
        help="Skip the RMAE reference solve (POT). Saves CPU time; rmae_pct will be N/A in the CSV.",
    )
    parser.add_argument(
        "--dataset", choices=DATASET_CHOICES, default="gaussian",
        help="Synthetic point cloud to benchmark against. Not applied to OTT-JAX "
             "(draws its own point cloud via JAX's RNG). Default: gaussian.",
    )
    parser.add_argument("--no-srot", action="store_true", help="Skip SROT benchmarks.")
    parser.add_argument("--no-sinkslot", action="store_true", help="Skip SinkSLOT benchmarks.")
    parser.add_argument("--stop-mode", choices=("fixed", "marginal", "potential", "potential_linf"),
                        default="fixed",
                        help="Early stopping: 'fixed' runs n_iters; 'marginal'/'potential' run to "
                             "convergence; 'potential_linf' reproduces FlashSinkhorn's own native rule "
                             "(max L_inf change in the dual potentials) for srot/sinkslot/sinkslotcuda/"
                             "spar_sink/rand_sink too.")
    parser.add_argument("--max-iter", type=int, default=10000, help="Iteration cap in non-fixed stop modes.")
    parser.add_argument("--stop-tol", type=float, default=1e-4, help="Max (L-infinity) marginal-violation threshold.")
    parser.add_argument("--potential-tol", type=float, default=1e-6, help="Spar-Sink ||du||+||dv|| threshold.")
    parser.add_argument("--mass-tol", type=float, default=1e-6, help="|sum(P) - 1| threshold.")
    parser.add_argument("--check-every", type=int, default=10, help="Iterations between convergence checks.")
    parser.add_argument(
        "--sinkslot-slices", type=str, default="50",
        help="Comma-separated L values (number of 1-D projections) for SinkSLOT.",
    )
    parser.add_argument("--no-sinkslotcuda", action="store_true", help="Skip SinkSLOT-CUDA benchmarks.")
    parser.add_argument(
        "--sinkslotcuda-slices", type=str, default="50",
        help="Comma-separated L values (number of 1-D projections) for SinkSLOT-CUDA.",
    )
    parser.add_argument("--no-sparsink", action="store_true", help="Skip Spar-Sink/Rand-Sink benchmarks.")
    parser.add_argument(
        "--sparsink-s", type=str, default="2000",
        help="Comma-separated expected subsample sizes s for Spar-Sink/Rand-Sink.",
    )
    parser.add_argument(
        "--sparsink-replicates", type=int, default=10,
        help="Independent kernel draws averaged for Spar-Sink/Rand-Sink (sampling is stochastic).",
    )
    parser.add_argument(
        "--srot-slices", type=str, default="50",
        help="Comma-separated L values (number of random 1-D projections) for SROT.",
    )
    parser.add_argument(
        "--srot-delta", type=float, default=1e-8,
        help="SROT: weight of the independent coupling mixed into pi_SOT.",
    )
    parser.add_argument("--no-geomloss", action="store_true", help="Skip GeomLoss benchmarks.")
    parser.add_argument("--no-flash-symmetric", action="store_true", help="Skip FlashSinkhorn symmetric backend.")
    parser.add_argument("--no-flash-alternating", action="store_true", help="Skip FlashSinkhorn alternating backend.")
    parser.add_argument(
        "--only",
        choices=("flash_symmetric", "flash_alternating", "flash", "geomloss", "ott", "srot",
                 "spar_sink", "rand_sink", "sinkslot", "sinkslotcuda"),
        default=None,
        help="Run only one method (useful for Nsight Systems profiling). 'flash' runs both FlashSinkhorn backends.",
    )
    parser.add_argument("--tensorized", action="store_true", help="Include tensorized/dense benchmarks.")
    parser.add_argument("--max-dense-size", type=int, default=20000,
                        help="Max size for tensorized/dense methods (to avoid OOM). Default: 20000.")
    parser.add_argument("--tf32", action="store_true", default=True,
                        help="Enable TF32 for ~2x speedup (default: enabled).")
    parser.add_argument("--no-tf32", dest="tf32", action="store_false",
                        help="Disable TF32 for strict FP32 (slower but higher precision).")
    parser.add_argument(
        "--nvtx",
        action="store_true",
        help="Emit NVTX ranges around timed regions (for Nsight Systems).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output/paper_benchmarks/forward",
        help="Output directory."
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output.")
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify loss parity between FlashSinkhorn and GeomLoss before benchmarking."
    )
    parser.add_argument(
        "--measure-jit-overhead", action="store_true",
        help="Measure JIT compilation overhead (cold start vs warm performance)."
    )
    parser.add_argument(
        "--jit-size", type=int, default=10000,
        help="Problem size for JIT overhead measurement (default: 10000)."
    )
    parser.add_argument(
        "--jit-dim", type=int, default=64,
        help="Feature dimension for JIT overhead measurement (default: 64)."
    )
    parser.add_argument(
        "--subprocess", action="store_true",
        help="Run each size in a separate subprocess (rarely needed with bucketed cache keys)."
    )
    parser.add_argument(
        "--single-size", type=int, default=None,
        help=argparse.SUPPRESS,  # Hidden: used internally by --subprocess mode
    )
    parser.add_argument(
        "--single-dim", type=int, default=None,
        help=argparse.SUPPRESS,  # Hidden: used with --single-size
    )
    args = parser.parse_args()
    _stop_cfg = StopCfg(mode=args.stop_mode, max_iter=args.max_iter, tol=args.stop_tol,
                        potential_tol=args.potential_tol, mass_tol=args.mass_tol,
                        check_every=args.check_every)

    srot_slices = [int(v) for v in str(args.srot_slices).split(",") if v.strip()]
    sparsink_s = [int(v) for v in str(args.sparsink_s).split(",") if v.strip()]
    sinkslot_slices = [int(v) for v in str(args.sinkslot_slices).split(",") if v.strip()]
    sinkslotcuda_slices = [int(v) for v in str(args.sinkslotcuda_slices).split(",") if v.strip()]

    # =====================================================================
    # Worker mode: benchmark a single (n, d) and emit JSON to stdout
    # =====================================================================
    if args.single_size is not None:
        if args.single_dim is None:
            parser.error("--single-dim is required with --single-size")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for this benchmark.")

        _preload_cuda_libs()
        _set_tf32(args.tf32)
        device = torch.device("cuda")

        # Determine which methods to benchmark (same logic as normal mode)
        include_flash_symmetric = not args.no_flash_symmetric
        include_flash_alternating = not args.no_flash_alternating
        include_geomloss = not args.no_geomloss
        include_ott = not args.no_ott
        include_srot = not args.no_srot
        include_sparsink = not args.no_sparsink
        include_sinkslot = not args.no_sinkslot
        include_sinkslotcuda = not args.no_sinkslotcuda
        include_tensorized = bool(args.tensorized)

        if args.only is not None:
            include_flash_symmetric = args.only in ("flash_symmetric", "flash")
            include_flash_alternating = args.only in ("flash_alternating", "flash")
            include_geomloss = args.only == "geomloss"
            include_ott = args.only == "ott"
            include_srot = args.only == "srot"
            include_sparsink = args.only in SPARSINK_METHODS
            include_sinkslot = args.only == "sinkslot"
            include_sinkslotcuda = args.only == "sinkslotcuda"
            include_tensorized = False

        results = run_forward_benchmark(
            sizes=[args.single_size],
            dims=[args.single_dim],
            eps=args.eps,
            n_iters=args.n_iters,
            device=device,
            warmup=args.warmup,
            rep=args.rep,
            include_flash_symmetric=include_flash_symmetric,
            include_flash_alternating=include_flash_alternating,
            include_ott=include_ott,
            include_geomloss=include_geomloss,
            include_tensorized=include_tensorized,
            max_dense_size=args.max_dense_size,
            verbose=False,
            nvtx=False,
            allow_tf32=args.tf32,
            rmae_check=not args.no_rmae_check,
            dataset=args.dataset,
            include_srot=include_srot,
            srot_slices=srot_slices,
            srot_delta=args.srot_delta,
            include_sparsink=include_sparsink,
            sparsink_s=sparsink_s,
            sparsink_replicates=args.sparsink_replicates,
            only_sparsink_method=args.only if args.only in SPARSINK_METHODS else None,
            include_sinkslot=include_sinkslot,
            sinkslot_slices=sinkslot_slices,
            include_sinkslotcuda=include_sinkslotcuda,
            sinkslotcuda_slices=sinkslotcuda_slices,
            stop=_stop_cfg,
            seed=args.seed,
        )

        # Emit JSON lines to stdout for the orchestrator to parse
        for r in results:
            print(timing_result_to_json(r), flush=True)
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this benchmark.")

    _preload_cuda_libs()
    _set_tf32(args.tf32)

    device = torch.device("cuda")
    nvtx_enabled = bool(args.nvtx)
    if nvtx_enabled and not _nvtx_available():
        print("Warning: NVTX unavailable; disabling --nvtx.")
        nvtx_enabled = False

    # Run verification if requested
    if args.verify:
        passed = verify_loss_parity(
            n=1000, d=64, eps=args.eps, n_iters=args.n_iters, device=device
        )
        if not passed:
            print("\nLoss parity verification failed. Check cost conventions.")
            return
        print("\nProceeding with benchmark...\n")

    # Run JIT overhead measurement if requested
    if args.measure_jit_overhead:
        include_flash_symmetric = not args.no_flash_symmetric
        include_flash_alternating = not args.no_flash_alternating
        include_geomloss = not args.no_geomloss
        include_ott = not args.no_ott

        jit_results = measure_jit_overhead(
            n=args.jit_size,
            d=args.jit_dim,
            eps=args.eps,
            n_iters=args.n_iters,
            device=device,
            warm_reps=10,
            include_flash_symmetric=include_flash_symmetric,
            include_flash_alternating=include_flash_alternating,
            include_geomloss=include_geomloss,
            include_ott=include_ott,
            verbose=not args.quiet,
            allow_tf32=args.tf32,
        )

        # Print and save summary
        print_jit_overhead_summary(jit_results)

        output_dir = Path(args.output_dir)
        save_jit_overhead_csv(
            jit_results,
            output_dir / f"jit_overhead_n{args.jit_size}_d{args.jit_dim}.csv"
        )

        print("\nJIT overhead measurement complete.")
        return

    sizes = [int(s) for s in args.sizes.split(",")]
    dims = [int(d) for d in args.dims.split(",")]

    # Determine which methods to benchmark
    include_flash_symmetric = not args.no_flash_symmetric
    include_flash_alternating = not args.no_flash_alternating
    include_geomloss = not args.no_geomloss
    include_ott = not args.no_ott
    include_srot = not args.no_srot
    include_sparsink = not args.no_sparsink
    include_sinkslot = not args.no_sinkslot
    include_sinkslotcuda = not args.no_sinkslotcuda
    include_tensorized = bool(args.tensorized)

    if args.only is not None:
        include_flash_symmetric = args.only in ("flash_symmetric", "flash")
        include_flash_alternating = args.only in ("flash_alternating", "flash")
        include_geomloss = args.only == "geomloss"
        include_ott = args.only == "ott"
        include_srot = args.only == "srot"
        include_sparsink = args.only in SPARSINK_METHODS
        include_sinkslot = args.only == "sinkslot"
        include_sinkslotcuda = args.only == "sinkslotcuda"
        if include_tensorized:
            print("Warning: Ignoring --tensorized because --only is set.")
            include_tensorized = False

    mode_label = "Subprocess Mode" if args.subprocess else "In-Process (bucketed cache keys)"
    print(f"Forward Pass Benchmark ({mode_label})")
    if args.subprocess:
        print(f"  Running each size in a separate subprocess for maximum isolation...")
    print(f"  Sizes: {sorted(sizes, reverse=True)} (large->small)")
    print(f"  Dimensions: {dims}")
    print(f"  Epsilon: {args.eps}")
    print(f"  Iterations: {args.n_iters}")
    print(f"  Warmup: {args.warmup}, Reps: {args.rep}")
    print(f"  Precision: {'TF32' if args.tf32 else 'FP32 (strict)'}")
    print(f"  FlashSinkhorn backends: symmetric={include_flash_symmetric}, alternating={include_flash_alternating}")
    print(f"  References: GeomLoss={include_geomloss}, OTT-JAX={include_ott}")
    print(f"  RMAE check (converged entropic OT via POT): {not args.no_rmae_check}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Seed: {args.seed} (data only)")
    print(f"  Stop: {args.stop_mode}" + ("" if args.stop_mode=="fixed" else f" (tol={args.stop_tol:g}, max_iter={args.max_iter}, every={args.check_every})"))
    print(f"  Include tensorized: {args.tensorized} (max size: {args.max_dense_size})")
    if args.only is not None:
        print(f"  Only: {args.only}")
    if not args.subprocess:
        print(f"  NVTX ranges: {nvtx_enabled}")
    print(f"  GPU: {torch.cuda.get_device_name()}")

    if args.subprocess:
        results = run_forward_benchmark_subprocess(
            sizes=sizes,
            dims=dims,
            args=args,
        )
    else:
        results = run_forward_benchmark(
            sizes=sizes,
            dims=dims,
            eps=args.eps,
            n_iters=args.n_iters,
            device=device,
            warmup=args.warmup,
            rep=args.rep,
            include_flash_symmetric=include_flash_symmetric,
            include_flash_alternating=include_flash_alternating,
            include_ott=include_ott,
            include_geomloss=include_geomloss,
            include_tensorized=include_tensorized,
            max_dense_size=args.max_dense_size,
            verbose=not args.quiet,
            nvtx=nvtx_enabled,
            allow_tf32=args.tf32,
            rmae_check=not args.no_rmae_check,
            dataset=args.dataset,
            include_srot=include_srot,
            srot_slices=srot_slices,
            srot_delta=args.srot_delta,
            include_sparsink=include_sparsink,
            sparsink_s=sparsink_s,
            sparsink_replicates=args.sparsink_replicates,
            only_sparsink_method=args.only if args.only in SPARSINK_METHODS else None,
            include_sinkslot=include_sinkslot,
            sinkslot_slices=sinkslot_slices,
            include_sinkslotcuda=include_sinkslotcuda,
            sinkslotcuda_slices=sinkslotcuda_slices,
            stop=_stop_cfg,
            seed=args.seed,
        )

    output_dir = Path(args.output_dir)
    save_results_csv(results, output_dir / "forward_all.csv")
    save_speedup_csv(results, output_dir / "forward_speedup.csv")

    print_summary(results)


if __name__ == "__main__":
    main()
