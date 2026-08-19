#!/usr/bin/env python3
"""Profile GPU memory allocation at the operator level using PyTorch's profiler.

The aggregate figures from scripts/memory_audit/memory.py answer "how much memory did the
entire phase need" but not "where did it come from". This script uses PyTorch's
built-in profilers to attribute GPU memory to specific kernels and operations:

  1. torch.profiler.profile with profile_memory=True and record_shapes=True:
     Reports top-N operators by self_cuda_memory_usage (or self_device_memory_usage
     on newer PyTorch), showing which kernels allocated the most. Printed after
     each phase as a table sorted by memory.

  2. torch.cuda.memory._record_memory_history and _dump_snapshot:
     Records every allocation/free event into a detailed timeline, saved as
     .pickle files per phase. These can be loaded into https://pytorch.org/memory_viz
     for interactive visualization of the memory timeline.

Each phase (setup and solve) is profiled separately on a warm process so JIT
compilation does not dominate the attribution.

Usage:

    python scripts/memory_audit/memory_profile.py --method sinkslot --n 4096 --d 64 --slices 512 --output-dir /tmp/profile
    python scripts/memory_audit/memory_profile.py --method flash_symmetric --n 4096 --d 64 --output-dir /tmp/profile
"""

from __future__ import annotations

import argparse
import pickle
import sys
import torch
from pathlib import Path
from typing import Callable, Dict, Optional

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "torch-ext"))

# Import the probes and helper functions from memory.py
from memory import (
    PROBES, _inputs, DeviceHighWater, _device_mb, _alloc_mb, _peak_mb,
)


def _get_memory_attribute() -> str:
    """Detect which memory attribute is available in this PyTorch version.

    Newer versions use self_device_memory_usage; older versions use
    self_cuda_memory_usage. Fail loudly if neither is available.
    """
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        profile_memory=True,
    ) as prof:
        pass

    key_avg = prof.key_averages()
    if hasattr(key_avg[0], "self_device_memory_usage"):
        return "self_device_memory_usage"
    elif hasattr(key_avg[0], "self_cuda_memory_usage"):
        return "self_cuda_memory_usage"
    else:
        raise RuntimeError(
            "PyTorch profiler memory attribute not found. This PyTorch version "
            "does not support profile_memory=True or has an incompatible API."
        )


def _print_profiler_table(prof: torch.profiler.profile, phase: str, k: int = 20) -> None:
    """Print top-k operators by memory usage."""
    memory_attr = _get_memory_attribute()

    print(f"\n{phase.upper()} PHASE - Top {k} operators by {memory_attr}:")
    print("-" * 100)

    key_avg = prof.key_averages(group_by_stack_n=5)
    key_avg.sort(key=lambda x: getattr(x, memory_attr, 0), reverse=True)

    # Print header
    print(f"{'Operator':<50} {'Memory (MB)':>15} {'Count':>10}")
    print("-" * 100)

    total_memory = 0.0
    for i, stat in enumerate(key_avg[:k], 1):
        memory_mb = getattr(stat, memory_attr, 0) / 1e6
        total_memory += memory_mb
        count = stat.count
        name = stat.key[:48]  # Truncate long names
        print(f"{name:<50} {memory_mb:>15.2f} {count:>10}")

    print("-" * 100)
    print(f"{'Top ' + str(k) + ' total':<50} {total_memory:>15.2f}")
    print()


def profile_phase(
    probe: Dict[str, Callable],
    phase_name: str,
    state: Dict,
    output_dir: Path,
    device: torch.device,
) -> None:
    """Profile one phase (setup or solve) with both profiler and memory history.

    Args:
        probe: dict with 'setup' and 'solve' callables from PROBES
        phase_name: 'setup' or 'solve'
        state: shared state dict passed to probe functions
        output_dir: directory to write .pickle snapshots
        device: torch device
    """
    phase_fn = probe[phase_name]

    memory_attr = _get_memory_attribute()

    # Run once warm on this process so JIT/autotune doesn't appear in the profile
    phase_fn(state)
    torch.cuda.synchronize(device)

    # Reset allocator tracking and record history
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.memory._record_memory_history(max_entries=100000)

    # Profile the phase
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        profile_memory=True,
        record_shapes=True,
        with_stack=True,
    ) as prof:
        phase_fn(state)

    torch.cuda.synchronize(device)

    # Dump memory history snapshot
    snapshot_path = output_dir / f"{phase_name}_snapshot.pickle"
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        torch.cuda.memory._dump_snapshot(str(snapshot_path))
        snapshot_size = snapshot_path.stat().st_size
        print(f"Wrote memory snapshot: {snapshot_path} ({snapshot_size / 1e6:.2f} MB)")
    except Exception as e:
        print(f"Warning: Could not write snapshot: {e}")
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)

    # Print profiler table
    _print_profiler_table(prof, phase_name, k=20)

    # Print peak allocation for this phase
    peak_mb = _peak_mb()
    print(f"{phase_name.upper()} phase peak allocated: {peak_mb:.2f} MB")
    print()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Profile per-method GPU memory at the operator level.",
    )
    p.add_argument(
        "--method",
        required=True,
        help=f"method to profile; choices: {','.join(PROBES)}",
    )
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--m", type=int, default=None, help="defaults to --n")
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--eps", type=float, default=0.01)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--slices", type=int, default=512, help="L, for SROT/SinkSLOT")
    p.add_argument("--sample-size", type=int, default=2000,
                   help="s, for Spar-Sink/Rand-Sink")
    p.add_argument("--dataset", default="gaussian", choices=("gaussian", "8gaussians"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", type=Path, default="./memory_profiles",
                   help="directory to write .pickle snapshots")
    p.add_argument("--no-autotune", action="store_true",
                   help="disable Flash Triton autotuning")
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--srot-delta", type=float, default=1e-8)
    p.add_argument("--sample-interval", type=float, default=0.002)
    p.add_argument("--max-dense-size", type=int, default=8192)
    args = p.parse_args()

    if args.m is None:
        args.m = args.n

    if not torch.cuda.is_available():
        sys.exit("no CUDA device available")

    if args.method not in PROBES:
        sys.exit(f"unknown method: {args.method}. choices: {','.join(PROBES)}")

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Profiling {args.method}")
    print(f"n={args.n} m={args.m} d={args.d} eps={args.eps} iters={args.iters}")
    print(f"L={args.slices} s={args.sample_size}")
    print(f"Output: {output_dir}")
    print()

    # Initialize probe for this method
    probe = PROBES[args.method](args)

    # Warm up the process with imports
    probe["imports"]()
    torch.cuda.synchronize(device)

    # Prepare inputs
    state: dict = {}
    try:
        state["xyab"] = _inputs(args.n, args.m, args.d, device, args.dataset)
        torch.cuda.synchronize(device)

        print(f"Inputs allocated: {_alloc_mb():.2f} MB\n")

        # Profile setup phase
        print("=" * 100)
        print("SETUP PHASE")
        print("=" * 100)
        profile_phase(probe, "setup", state, output_dir, device)

        # Profile solve phase
        print("=" * 100)
        print("SOLVE PHASE")
        print("=" * 100)
        profile_phase(probe, "solve", state, output_dir, device)

        print("=" * 100)
        print(f"Profiling complete. Snapshots written to {output_dir}")
        print(f"Load .pickle files at: https://pytorch.org/memory_viz")

    except torch.cuda.OutOfMemoryError as e:
        print(f"\nOOM Error during profiling of {args.method} (n={args.n}, d={args.d})")
        print(f"Try reducing --n or --d")
        sys.exit(1)


if __name__ == "__main__":
    main()
