#!/usr/bin/env python3
"""Decompose each method's GPU memory into context, kernels, and working set.

The forward benchmark's ``gpu_memory_mb`` column is ``total - free`` from
``cudaMemGetInfo``: the whole-device figure, which bundles the CUDA context, every
compiled Triton/KeOps module the process has loaded, cuBLAS workspaces and the
caching allocator's reserved pool into one number. That is the right quantity for
"will this fit on the card", but it is dominated by a per-method *fixed* floor --
on H100 the exp1 sweep reports 1,786.8 MiB for Flash at every single eps, and
782.3 MiB for SinkSLOT at L=8 -- so it cannot answer "how much memory does the
algorithm need", and it is not what FlashSinkhorn's paper (Figure 3) plots.

This script splits that one number into four:

    context     CUDA context + driver reservation, measured before any kernel or
                tensor exists. Scales with the GPU (SM count, driver version), not
                with the problem. Identical work on an A1000 vs an H100 differs
                here by several hundred MB.

    modules     device memory held after the method has run that the PyTorch
                allocator does not account for: compiled Triton/CUDA modules and
                their per-thread local-memory reservation, cuBLAS/cuDNN
                workspaces, and KeOps' own allocations. Grows with the number of
                distinct kernels a method JITs -- which is why autotuning methods
                sit higher than non-autotuning ones for reasons unrelated to the
                transport problem.

    reserved    the caching allocator's pool, after ``empty_cache()``.

    peak        ``max_memory_allocated()`` across the phase: live tensor bytes at
                the high-water mark. This is the algorithmic footprint, the
                O(N*d)-vs-O(N*M) quantity, and the apples-to-apples counterpart to
                Figure 3. Reported separately for setup (support construction) and
                solve (the iteration loop), because for the sliced baselines those
                are very different numbers.

and reports one derived figure:

    total       ``context + modules + peak``: what the method needs on an
                otherwise-empty card. Note this is NOT ``device``, and can be far
                larger -- ``device`` is read after the run, so every transient the
                method allocated and freed at its high-water mark is already gone
                from it. SinkSLOT at n=4096, L=512 reads 209 MB of ``device``
                having momentarily held 2,935 MB. ``device`` answers "what is
                still held", ``total`` answers "will this fit".

Each method runs in its own subprocess (``--child``): compiled modules and the
CUDA context are never released within a process, so two methods sharing one
would each inherit the other's floor -- the same reason ``BenchConfig.isolate``
exists.

Usage:

    python scripts/memory_audit/memory.py                          # default grid, all methods
    python scripts/memory_audit/memory.py --n 100000 --d 64 --slices 2048
    python scripts/memory_audit/memory.py --methods flash_symmetric,sinkslot --json out/mem.json

Sizes default low enough to fit a 4GB laptop card. SROT and Spar-Sink/Rand-Sink
materialise an N x M matrix, so they are skipped above ``--max-dense-size``; that
skip is reported rather than silently dropped.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "torch-ext"))

import torch


# ---------------------------------------------------------------------------
# Measurement primitives
# ---------------------------------------------------------------------------

def _device_mb(device: torch.device) -> float:
    """Whole-device memory in use, in MB -- what nvidia-smi shows.

    ``empty_cache()`` first so the allocator's freed-but-pooled blocks are handed
    back to the driver; otherwise the figure is dominated by allocator hysteresis
    rather than anything a method actually needs. Matches
    ``bench_forward.gpu_memory_used_mb`` so the numbers here are directly
    comparable to the sweep's ``gpu_memory_mb`` column.
    """
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(device)
    return (total - free) / 1e6


def _alloc_mb() -> float:
    return torch.cuda.memory_allocated() / 1e6


def _peak_mb() -> float:
    return torch.cuda.max_memory_allocated() / 1e6


def _reserved_mb() -> float:
    return torch.cuda.memory_reserved() / 1e6


class DeviceHighWater:
    """Sample whole-device memory in a background thread; keep the maximum.

    ``max_memory_allocated`` counts PyTorch tensor bytes and nothing else -- not
    the CUDA context, not compiled modules, not KeOps. Adding those back after the
    fact assumes they are constant during the run, which is exactly what a
    JIT-compiling method violates. So poll the driver instead: this is the same
    ``cudaMemGetInfo`` figure nvidia-smi reports, read while the work is running,
    which is the *real* high-water mark and the honest answer to "will this fit".

    Deliberately does NOT call ``empty_cache()`` -- that is a synchronising,
    allocator-perturbing call, and inside the sampling loop it would change the
    thing being measured. It therefore includes the caching allocator's pool, as
    nvidia-smi would.

    Two caveats, both reported rather than hidden:

    * It is a sampler. A spike shorter than ``interval`` between two polls is
      missed, so the figure is a lower bound on the true maximum. The allocator's
      ``max_memory_allocated`` is exact for tensors and is kept alongside as a
      cross-check: if the sampled figure ever falls below (tensors + context) the
      sampling was too coarse for that row.
    * Polling needs the GIL. CUDA launches release it, but a long GIL-holding
      stretch on the main thread can stall the sampler; ``sys.setswitchinterval``
      is lowered while sampling to keep that short.
    """

    def __init__(self, device: torch.device, interval: float = 0.002):
        self.device = device
        self.interval = interval
        self.peak_mb = 0.0
        self.samples = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._switch_interval: Optional[float] = None

    def _poll(self) -> None:
        while not self._stop.is_set():
            free, total = torch.cuda.mem_get_info(self.device)
            self.peak_mb = max(self.peak_mb, (total - free) / 1e6)
            self.samples += 1
            self._stop.wait(self.interval)

    def __enter__(self) -> "DeviceHighWater":
        self._switch_interval = sys.getswitchinterval()
        sys.setswitchinterval(self.interval / 4)
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        # Sync first: without it the sampler stops while kernels are still in
        # flight and the tail of the run goes unmeasured.
        torch.cuda.synchronize(self.device)
        free, total = torch.cuda.mem_get_info(self.device)
        self.peak_mb = max(self.peak_mb, (total - free) / 1e6)
        self._stop.set()
        self._thread.join(timeout=5.0)
        if self._switch_interval is not None:
            sys.setswitchinterval(self._switch_interval)


# ---------------------------------------------------------------------------
# Per-method probes
# ---------------------------------------------------------------------------
#
# A probe is (import_fn, setup_fn, solve_fn). They are kept separate so the three
# things we want to attribute -- module loading, support construction, and the
# iteration loop -- can each be measured on their own. Every probe mirrors the
# corresponding bench_forward.bench_* function: same cost convention, same dtypes,
# same call sequence. It does not reproduce their timing protocol (no repetitions,
# no rmae check) -- this measures memory, not speed.


def _inputs(n: int, m: int, d: int, device: torch.device, dataset: str):
    from flash_sinkhorn.bench.bench_forward import sample_point_cloud

    torch.manual_seed(0)
    x = sample_point_cloud(n, d, device, dataset=dataset, target=False)
    y = sample_point_cloud(m, d, device, dataset=dataset, target=True)
    a = torch.rand(n, device=device, dtype=torch.float32) + 0.1
    b = torch.rand(m, device=device, dtype=torch.float32) + 0.1
    return x, y, a / a.sum(), b / b.sum()


def _probe_flash(backend: str, args) -> Dict[str, Callable]:
    def imports():
        from flash_sinkhorn import SamplesLoss  # noqa: F401

    def setup(state):
        from flash_sinkhorn import SamplesLoss

        # autotune=True matches the sweep. It is also the single biggest lever on
        # the `modules` column: each autotuned config is a separately compiled
        # kernel, and every one stays resident for the life of the process.
        state["loss_fn"] = SamplesLoss(
            "sinkhorn",
            backend=backend,
            use_epsilon_scaling=False,
            eps=args.eps,
            n_iters=args.iters,
            debias=False,
            potentials=False,
            normalize=False,
            autotune=not args.no_autotune,
            last_extrapolation=False,
            allow_tf32=args.tf32,
        )

    def solve(state):
        x, y, a, b = state["xyab"]
        state["loss_fn"](a, x, b, y).item()

    return {"imports": imports, "setup": setup, "solve": solve}


def _probe_srot(args) -> Dict[str, Callable]:
    def imports():
        from flash_sinkhorn.bench.bench_forward import build_sot_plan  # noqa: F401

    def setup(state):
        from flash_sinkhorn.bench.bench_forward import build_sot_plan

        x, y, a, b = state["xyab"]
        # build_sot_plan is fp64 throughout and allocates several N x M fp64
        # temporaries per slice on top of the fp64 accumulator, so its peak is
        # well above the size of the plan it returns. That is the number this
        # column is for.
        pi_sot = build_sot_plan(x, y, a, b, slices=args.slices, delta=args.srot_delta)
        state["cost"] = torch.cdist(x, y, p=2) ** 2
        state["log_pi"] = pi_sot.clamp_min(torch.finfo(pi_sot.dtype).tiny).log()
        del pi_sot
        state["log_a"], state["log_b"] = a.log(), b.log()

    def solve(state):
        from flash_sinkhorn.bench.bench_forward import StopCfg, _srot_sinkhorn

        _srot_sinkhorn(state["cost"], state["log_pi"], state["log_a"], state["log_b"],
                       args.eps, args.iters, StopCfg.fixed())

    return {"imports": imports, "setup": setup, "solve": solve}


def _probe_sinkslot(cuda_path: bool, args) -> Dict[str, Callable]:
    def imports():
        from sinkslot import solver as sinkslot  # noqa: F401

    def setup(state):
        from sinkslot.solver import (
            _ot_1d_coo_batched, _ot_1d_coo_batched_cuda, sot_coo,
            sparse_sqeuclidean_cost, to_csr,
        )

        x, y, a, b = state["xyab"]
        n, m = x.shape[0], y.shape[0]
        ot1d = _ot_1d_coo_batched_cuda if cuda_path else _ot_1d_coo_batched
        rows, cols, S = sot_coo(x, y, a, b, L=args.slices, seed=0, ot1d=ot1d)
        if cuda_path:
            cost = sparse_sqeuclidean_cost(x, y, rows, cols)
        else:
            cost = (x[rows] - y[cols]).square().sum(1)
        lam = S.clamp_min(torch.finfo(S.dtype).tiny).log() - cost / args.eps
        r = to_csr(rows, cols, lam, n, narrow_key=cuda_path)
        c = to_csr(cols, rows, lam, m, narrow_key=cuda_path)
        state["csr"] = (r[0], r[1], r[2], c[0], c[1], c[2])
        state["nm"] = (n, m)
        state["nnz"] = int(rows.numel())
        state["log_ab"] = (a.log(), b.log())

    def solve(state):
        from flash_sinkhorn.bench.bench_forward import StopCfg
        from sinkslot.sinkhorn_solvers import sinkslot_alternating_triton

        n, m = state["nm"]
        log_a, log_b = state["log_ab"]
        sinkslot_alternating_triton(*state["csr"], log_a, log_b, n, m, args.iters, StopCfg.fixed())

    return {"imports": imports, "setup": setup, "solve": solve}


def _probe_sparsink(method: str, args) -> Dict[str, Callable]:
    def imports():
        from flash_sinkhorn.bench.bench_forward import build_sparse_kernel  # noqa: F401

    def setup(state):
        from flash_sinkhorn.bench.bench_forward import build_sparse_kernel

        x, y, a, b = state["xyab"]
        # Dense N x M cost, then a dense N x M probability matrix inside
        # build_sparse_kernel -- the iterations are O(nnz) but the build is not,
        # which is exactly the split this script is meant to expose.
        state["cost"] = torch.cdist(x, y, p=2) ** 2
        state["log_ab"] = (a.log(), b.log())
        rows, cols, log_values = build_sparse_kernel(
            state["cost"], a, b, args.eps,
            method=method, sample_size=args.sample_size, seed=0,
        )
        state["sparse"] = (rows, cols, log_values)
        state["nnz"] = int(rows.numel())

    def solve(state):
        from flash_sinkhorn.bench.bench_forward import StopCfg, _sparsink_sinkhorn

        rows, cols, log_values = state["sparse"]
        log_a, log_b = state["log_ab"]
        _sparsink_sinkhorn(rows, cols, log_values, log_a, log_b,
                           args.eps, args.iters, StopCfg.fixed())

    return {"imports": imports, "setup": setup, "solve": solve}


def _probe_geomloss(args) -> Dict[str, Callable]:
    """Mirrors bench_geomloss_online: low-level sinkhorn_loop, eps_list=[eps]*iters.

    Deliberately NOT the high-level ``geomloss.SamplesLoss`` -- that turns on
    epsilon-scaling and would run a different iteration count against a different
    KeOps kernel than the sweep does.

    KeOps allocates outside the PyTorch allocator, so its `peak` (and hence
    `TOTAL`) understates: the LSE reduction's own workspace is invisible to
    ``max_memory_allocated`` and lands in `modules` instead. This is the one row
    where `device` carries information the allocator columns do not.
    """
    def imports():
        from pykeops.torch import generic_logsumexp  # noqa: F401

    def setup(state):
        from functools import partial

        from geomloss.sinkhorn_divergence import log_weights
        from geomloss.sinkhorn_samples import lse_genred, softmin_online

        x, y, a, b = state["xyab"]
        state["logs"] = (log_weights(a), log_weights(b))
        # SqDist(X,Y) = ||x-y||^2, matching FlashSinkhorn's cost convention.
        state["softmin"] = partial(softmin_online, log_conv=lse_genred("SqDist(X,Y)", args.d))
        state["C"] = ((x, y.detach()), (y, x.detach()))
        state["eps_list"] = [args.eps] * args.iters

    def solve(state):
        from geomloss.sinkhorn_divergence import sinkhorn_loop

        a_log, b_log = state["logs"]
        C_xy, C_yx = state["C"]
        sinkhorn_loop(
            state["softmin"], a_log, b_log, None, None,
            C_xy, C_yx, state["eps_list"],
            rho=None, debias=False, last_extrapolation=False,
        )

    return {"imports": imports, "setup": setup, "solve": solve}


# Dense methods materialise an N x M matrix and are gated on --max-dense-size.
DENSE_METHODS = ("srot", "spar_sink", "rand_sink")

PROBES: Dict[str, Callable] = {
    "flash_symmetric": lambda a: _probe_flash("symmetric", a),
    "flash_alternating": lambda a: _probe_flash("alternating", a),
    "srot": _probe_srot,
    "sinkslot": lambda a: _probe_sinkslot(False, a),
    "sinkslot_cuda": lambda a: _probe_sinkslot(True, a),
    "spar_sink": lambda a: _probe_sparsink("spar_sink", a),
    "rand_sink": lambda a: _probe_sparsink("rand_sink", a),
    "geomloss_online": _probe_geomloss,
}

DEFAULT_METHODS = (
    "flash_symmetric,flash_alternating,srot,sinkslot,sinkslot_cuda,spar_sink,"
    "rand_sink,geomloss_online"
)


# ---------------------------------------------------------------------------
# Child: measure one method
# ---------------------------------------------------------------------------

def measure(args) -> dict:
    """Measure one method in this process. Assumes a virgin CUDA state."""
    device = torch.device(args.device)
    out: dict = {
        "method": args.method, "n": args.n, "m": args.m, "d": args.d,
        "eps": args.eps, "iters": args.iters, "slices": args.slices,
        "sample_size": args.sample_size, "autotune": not args.no_autotune,
        "gpu": torch.cuda.get_device_name(device),
    }

    if args.n > args.max_dense_size and args.method in DENSE_METHODS:
        out["skipped"] = f"dense method above --max-dense-size ({args.max_dense_size})"
        return out

    # --- context: create it explicitly, before any kernel or user tensor ---
    torch.zeros(1, device=device)
    torch.cuda.synchronize()
    out["context_mb"] = _device_mb(device)

    probe = PROBES[args.method](args)

    # Sampler spanning everything from here on -- imports, inputs, JIT/autotune,
    # warm-up and both measured phases. The per-phase samplers below deliberately
    # exclude the warm-up, which is where Triton autotuning peaks; for Flash on an
    # A1000 that transient is 3.4x the steady-state figure, so a process that has
    # to compile before it can run needs this number, not the warm one.
    lifetime = DeviceHighWater(device, args.sample_interval)
    lifetime.__enter__()

    # --- module import: Python-side only; should be ~0 on the device ---
    probe["imports"]()
    torch.cuda.synchronize()
    out["after_import_mb"] = _device_mb(device)

    state: dict = {}
    try:
        # --- inputs: the point clouds and marginals every method shares ---
        torch.cuda.reset_peak_memory_stats(device)
        state["xyab"] = _inputs(args.n, args.m, args.d, device, args.dataset)
        torch.cuda.synchronize()
        out["inputs_mb"] = _alloc_mb()

        # --- setup: support construction (no-op for Flash/GeomLoss) ---
        # Run it once untimed first so one-time CUDA/JIT initialisation lands
        # outside the measured peak, then reset and measure the second call --
        # the same warm-then-measure discipline bench_forward uses for setup_ms.
        probe["setup"](state)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        with DeviceHighWater(device, args.sample_interval) as hw:
            probe["setup"](state)
        out["setup_tensors_mb"] = _peak_mb()
        out["setup_device_hw_mb"] = hw.peak_mb
        out["after_setup_mb"] = _alloc_mb()

        # --- warm-up solve: JIT + autotune, so compiled modules are resident ---
        probe["solve"](state)
        torch.cuda.synchronize()
        out["after_warmup_device_mb"] = _device_mb(device)

        # --- solve: the iteration loop, on a warm process ---
        torch.cuda.reset_peak_memory_stats(device)
        with DeviceHighWater(device, args.sample_interval) as hw:
            probe["solve"](state)
        out["solve_tensors_mb"] = _peak_mb()
        out["solve_device_hw_mb"] = hw.peak_mb
        out["hw_samples"] = hw.samples
    except torch.cuda.OutOfMemoryError:
        lifetime.__exit__(None, None, None)
        out["oom"] = True
        out["lifetime_mb"] = lifetime.peak_mb
        return out

    lifetime.__exit__(None, None, None)
    out["lifetime_mb"] = lifetime.peak_mb
    out["nnz"] = state.get("nnz")
    out["reserved_mb"] = _reserved_mb()
    out["device_final_mb"] = _device_mb(device)
    # Device memory the allocator does not account for: compiled modules and their
    # local-memory reservation, cuBLAS workspaces, KeOps. Clamped at 0 -- the two
    # figures come from different sources (driver vs allocator) and can cross by a
    # few MB of rounding on an otherwise-idle device.
    out["modules_mb"] = max(
        0.0, out["device_final_mb"] - out["context_mb"] - out["reserved_mb"]
    )
    # Tensor bytes only -- NOT a total. Excludes context and compiled modules, so
    # it is always smaller than what the card must supply. This is the
    # O(N*d)-vs-O(N*M) quantity and the counterpart to FlashSinkhorn Figure 3.
    out["tensors_mb"] = max(out["setup_tensors_mb"], out["solve_tensors_mb"])
    # WARM: whole-device high-water mark over setup + solve on an already-compiled
    # process. What a warm/persistent server needs per call.
    out["warm_mb"] = max(out["setup_device_hw_mb"], out["solve_device_hw_mb"])
    # TOTAL: the same thing over the entire process lifetime, so JIT compilation
    # and Triton autotuning are inside it. What a cold process needs, and what
    # nvidia-smi actually peaks at. Both are measured, not context + modules +
    # tensors -- that sum assumes the non-allocator terms hold still during the
    # run, which is precisely what compilation violates.
    out["total_mb"] = max(out["warm_mb"], out["lifetime_mb"])
    # Cross-check: TOTAL cannot legitimately be below what the allocator already
    # proved was live. If it is, the sampler was too coarse for this row.
    out["undersampled"] = out["total_mb"] < out["tensors_mb"] + out["context_mb"] - 1.0
    return out


# ---------------------------------------------------------------------------
# Parent: fan out over methods, one subprocess each
# ---------------------------------------------------------------------------

def run_child(args, method: str) -> dict:
    cmd = [
        sys.executable, str(Path(__file__).resolve()), "--child",
        "--method", method,
        "--n", str(args.n), "--m", str(args.m), "--d", str(args.d),
        "--eps", repr(args.eps), "--iters", str(args.iters),
        "--slices", str(args.slices), "--sample-size", str(args.sample_size),
        "--srot-delta", repr(args.srot_delta),
        "--sample-interval", repr(args.sample_interval),
        "--max-dense-size", str(args.max_dense_size),
        "--dataset", args.dataset, "--device", args.device,
    ]
    if args.no_autotune:
        cmd.append("--no-autotune")
    if args.tf32:
        cmd.append("--tf32")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if line.startswith("{"):
            return json.loads(line)
    return {
        "method": method,
        "error": (proc.stderr.strip().splitlines() or ["no output"])[-1],
    }


_COLUMNS = [
    ("method", "method", 18, "s"),
    ("device_final_mb", "device", 9, ".1f"),
    ("context_mb", "context", 9, ".1f"),
    ("modules_mb", "modules", 9, ".1f"),
    ("reserved_mb", "reserved", 9, ".1f"),
    ("setup_tensors_mb", "tens:setup", 11, ".1f"),
    ("solve_tensors_mb", "tens:solve", 11, ".1f"),
    ("tensors_mb", "TENSORS", 9, ".1f"),
    ("warm_mb", "warm", 9, ".1f"),
    ("total_mb", "TOTAL", 10, ".1f"),
]


def print_table(rows: List[dict]) -> None:
    header = "".join(f"{label:>{w}}" if fmt != "s" else f"{label:<{w}}"
                     for _, label, w, fmt in _COLUMNS)
    print()
    print(header)
    print("-" * len(header))
    for r in rows:
        if r.get("error") or r.get("oom") or r.get("skipped"):
            note = r.get("error") or ("OOM" if r.get("oom") else r["skipped"])
            print(f"{r['method']:<18}{note}")
            continue
        print("".join(
            f"{r[key]:<{w}}" if fmt == "s" else f"{r.get(key, float('nan')):>{w}{fmt}}"
            for key, _, w, fmt in _COLUMNS
        ))
    print()
    print("All figures MB.")
    print("  device  = the sweep's gpu_memory_mb, read AFTER the run (residual).")
    print("  TENSORS = max_memory_allocated: PyTorch tensor bytes only. NOT a total --")
    print("            excludes context and modules. The O(Nd)-vs-O(NM) quantity and")
    print("            the counterpart to FlashSinkhorn Figure 3.")
    print("  warm    = whole-device high-water over setup+solve on an already-")
    print("            compiled process: what a warm server needs per call.")
    print("  TOTAL   = whole-device high-water over the WHOLE process, so JIT and")
    print("            Triton autotuning are inside it. What nvidia-smi actually")
    print("            peaks at, and what a cold process needs. Measured, not summed.")
    if any(r.get("undersampled") for r in rows):
        print("  [!] rows marked undersampled: TOTAL fell below tensors + context --")
        print("      the sampler missed the spike. Lower --sample-interval and rerun.")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Decompose per-method GPU memory into context / modules / working set.",
    )
    p.add_argument("--methods", default=DEFAULT_METHODS,
                   help=f"comma-separated; choices: {','.join(PROBES)}")
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--m", type=int, default=None, help="defaults to --n")
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--eps", type=float, default=0.01)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--slices", type=int, default=512, help="L, for SROT/SinkSLOT")
    p.add_argument("--sample-size", type=int, default=2000,
                   help="s, for Spar-Sink/Rand-Sink")
    p.add_argument("--srot-delta", type=float, default=1e-8)
    p.add_argument("--max-dense-size", type=int, default=8192,
                   help="largest n for which the dense methods run")
    p.add_argument("--dataset", default="gaussian", choices=("gaussian", "8gaussians"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-autotune", action="store_true",
                   help="disable Flash Triton autotuning (isolates its module cost)")
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--sample-interval", type=float, default=0.002,
                   help="driver-polling interval in seconds for the TOTAL high-water "
                        "mark; lower catches shorter spikes at more overhead")
    p.add_argument("--json", type=Path, default=None, help="write raw results here")
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--method", default=None, help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.m is None:
        args.m = args.n

    if not torch.cuda.is_available():
        sys.exit("no CUDA device available")

    if args.child:
        print(json.dumps(measure(args)))
        return

    methods = [s.strip() for s in args.methods.split(",") if s.strip()]
    unknown = [s for s in methods if s not in PROBES]
    if unknown:
        sys.exit(f"unknown method(s): {', '.join(unknown)}. choices: {','.join(PROBES)}")

    print(f"n={args.n} m={args.m} d={args.d} eps={args.eps} iters={args.iters} "
          f"L={args.slices} s={args.sample_size} autotune={not args.no_autotune}")

    rows = [run_child(args, method) for method in methods]
    print_table(rows)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
