"""Build and (optionally) execute a SinkSLOT-vs-baselines benchmark from config.py.

Sweeps every (dataset, eps) combination from CONFIG across every enabled method --
SinkSLOT/SinkSLOT-CUDA, SROT, Spar-Sink/Rand-Sink, and FlashSinkhorn. All runs
append into a single `<output_dir>/forward_all.csv` plus one speedup table, with
dataset, eps, d and n as ordinary columns. Existing CSVs are deleted at the start
of a sweep, so each invocation produces a clean table.

Usage:
    python run.py                  # honors CONFIG.dry_run
    python run.py --dry-run        # force dry run (just print the command)
    python run.py --execute        # force real execution
    python run.py --compare-tf32    # run once with TF32 on and once off, then diff timings
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import importlib

from configs.base import BenchConfig


def build_command(
    cfg: BenchConfig, dataset: str, eps: float, output_dir: str,
    *, method: Optional[str] = None, size: Optional[int] = None, dim: Optional[int] = None,
    slices: Optional[int] = None, seed: Optional[int] = None,
) -> List[str]:
    """Build the benchmark command for a single unit of work.

    The underlying bench module takes one --dataset/--eps per invocation, so both are
    passed explicitly rather than read off cfg. When method/size/dim are given the run is
    narrowed to that single measurement (isolation mode); otherwise the full grid runs in
    one process.

    seed: data-generation seed for this unit (defaults to cfg.seeds[0] if not given).
    Only affects (x, y, a, b); method-internal randomness stays independently seeded.
    """
    module = "sinkslot.bench.bench_forward"
    cmd = [
        sys.executable, "-m", module,
        "--sizes", str(size) if size is not None else ",".join(str(s) for s in cfg.sizes),
        "--dims", str(dim) if dim is not None else ",".join(str(d) for d in cfg.dims),
        "--eps", str(eps),
        "--n-iters", str(cfg.n_iters),
        "--warmup", str(cfg.warmup),
        "--rep", str(cfg.rep),
        "--max-dense-size", str(cfg.max_dense_size),
        "--output-dir", output_dir,
        "--seed", str(seed if seed is not None else cfg.seeds[0]),
    ]

    if not cfg.tf32:
        cmd.append("--no-tf32")
    if dataset != "gaussian":
        cmd += ["--dataset", dataset]
    if cfg.no_ott:
        cmd.append("--no-ott")
    if cfg.no_rmae_check:
        cmd.append("--no-rmae-check")
    if cfg.no_geomloss:
        cmd.append("--no-geomloss")
    if cfg.no_flash_symmetric:
        cmd.append("--no-flash-symmetric")
    if cfg.no_flash_alternating:
        cmd.append("--no-flash-alternating")
    only = method or cfg.only
    if only:
        cmd += ["--only", only]
    # SROT flags only matter to an SROT run; omit them elsewhere to keep commands readable.
    if cfg.no_srot:
        cmd.append("--no-srot")
    elif method in (None, "srot"):
        srot_values = [slices] if slices is not None else cfg.srot_slices
        cmd += ["--srot-slices", ",".join(str(v) for v in srot_values),
                "--srot-delta", str(cfg.srot_delta)]

    if cfg.no_sinkslot:
        cmd.append("--no-sinkslot")
    elif method in (None, "sinkslot"):
        ss_values = [slices] if slices is not None and method == "sinkslot" else cfg.sinkslot_slices
        cmd += ["--sinkslot-slices", ",".join(str(v) for v in ss_values)]

    if cfg.no_sinkslotcuda:
        cmd.append("--no-sinkslotcuda")
    elif method in (None, "sinkslotcuda"):
        sc_values = [slices] if slices is not None and method == "sinkslotcuda" else cfg.sinkslotcuda_slices
        cmd += ["--sinkslotcuda-slices", ",".join(str(v) for v in sc_values)]

    if cfg.stop_mode != "fixed":
        cmd += ["--stop-mode", cfg.stop_mode,
                "--max-iter", str(cfg.max_iter),
                "--stop-tol", str(cfg.stop_tol),
                "--potential-tol", str(cfg.potential_tol),
                "--mass-tol", str(cfg.mass_tol),
                "--check-every", str(cfg.check_every)]

    if cfg.no_sparsink:
        cmd.append("--no-sparsink")
    elif method in (None, "spar_sink", "rand_sink"):
        s_values = [slices] if slices is not None else cfg.sparsink_s
        cmd += ["--sparsink-s", ",".join(str(v) for v in s_values),
                "--sparsink-replicates", str(cfg.sparsink_replicates)]
    if cfg.tensorized:
        cmd.append("--tensorized")
    if cfg.verify:
        cmd.append("--verify")
    if cfg.quiet:
        cmd.append("--quiet")

    return cmd


def _results_csv(cfg: BenchConfig, output_dir: str) -> Path:
    return Path(output_dir) / "forward_all.csv"


def _speedup_csv(cfg: BenchConfig, output_dir: str) -> Path:
    return Path(output_dir) / "forward_speedup.csv"


def _clear_csvs(cfg: BenchConfig, output_dir: str) -> None:
    """Delete this run's CSVs so the sweep starts from an empty table.

    The bench modules merge into their CSVs by row key, which is what lets the
    (dataset, eps) sweep accumulate into one file. That same merging would also
    preserve rows from a previous, unrelated sweep, so we clear first.
    """
    for path in (_results_csv(cfg, output_dir), _speedup_csv(cfg, output_dir)):
        if path.exists():
            path.unlink()
            print(f"Removed stale {path}")


def _methods(cfg: BenchConfig) -> List[str]:
    """Method names to isolate into separate processes, matching the bench --only values."""
    names = []
    if not cfg.no_flash_symmetric:
        names.append("flash_symmetric")
    if not cfg.no_flash_alternating:
        names.append("flash_alternating")
    if not cfg.no_geomloss:
        names.append("geomloss")
    if not cfg.no_srot:
        names.append("srot")
    if not cfg.no_sinkslot:
        names.append("sinkslot")
    if not cfg.no_sinkslotcuda:
        names.append("sinkslotcuda")
    if not cfg.no_sparsink:
        names.append("spar_sink")
        if not cfg.no_randsink:
            names.append("rand_sink")
    return names


def _units(cfg: BenchConfig):
    """Yield one (dataset, eps, method, size, dim, slices, seed) unit of work per subprocess.

    With cfg.isolate the sweep is fully unrolled so every measured row gets a fresh
    process -- necessary because gpu_memory_mb reports whole-device usage, which is
    cumulative within a process. Without it, one process per (dataset, eps, seed).

    seed is the outermost loop: it varies only the underlying problem instance
    (x, y, a, b), so every (dataset, eps, method, size, dim, slices) combination gets
    repeated once per seed, letting results be aggregated across seeds afterward.
    """
    for seed in cfg.seeds:
        for dataset in cfg.datasets:
            for eps in cfg.eps_values:
                if not cfg.isolate:
                    yield dataset, eps, None, None, None, None, seed
                    continue
                for method in _methods(cfg):
                    for size in cfg.sizes:
                        for dim in cfg.dims:
                            if method in ("spar_sink", "rand_sink"):
                                # s expands these two only, like L for SROT.
                                for s_size in cfg.sparsink_s:
                                    yield dataset, eps, method, size, dim, s_size, seed
                            elif method in ("srot", "sinkslot", "sinkslotcuda"):
                                # L expands the sliced methods only -- for every other method
                                # it is meaningless, and sweeping it at the top level would
                                # just duplicate identical rows. Each method carries its own
                                # L list (srot_slices / sinkslot_slices / sinkslotcuda_slices).
                                l_values = {
                                    "srot": cfg.srot_slices,
                                    "sinkslot": cfg.sinkslot_slices,
                                    "sinkslotcuda": cfg.sinkslotcuda_slices,
                                }[method]
                                for slices in l_values:
                                    yield dataset, eps, method, size, dim, slices, seed
                            else:
                                yield dataset, eps, method, size, dim, None, seed


def run_sweep(cfg: BenchConfig, base_dir: str, *, dry_run: bool, label: str = "") -> None:
    """Run the benchmark for every unit of work, all into one output directory."""
    prefix = f"{label} " if label else ""
    if not dry_run:
        _clear_csvs(cfg, base_dir)
    units = list(_units(cfg))
    failures: List[Tuple[str, int]] = []
    for i, (dataset, eps, method, size, dim, slices, seed) in enumerate(units, 1):
        cmd = build_command(
            cfg, dataset, eps, base_dir, method=method, size=size, dim=dim, slices=slices,
            seed=seed,
        )
        printable = " ".join(cmd)
        tag = f"dataset={dataset} eps={eps} seed={seed}"
        if method is not None:
            tag += f" {method} n={size} d={dim}"
        if slices is not None:
            tag += (f" s={slices}" if method in ("spar_sink", "rand_sink") else f" L={slices}")
        if dry_run:
            print(f"[dry-run] {prefix}{tag} would execute:")
            print(f"  {printable}")
            continue
        print(f"\n[{i}/{len(units)}] {prefix}{tag}: {printable}", flush=True)
        # Don't abort the sweep on one bad unit. A single subprocess can die for reasons
        # unrelated to the rest of the grid (a transient CUDA/driver fault, an OOM at one
        # size), and check=True would then throw away every remaining unit. Record it and
        # carry on; the run ends with a summary of what failed.
        result = subprocess.run(cmd)
        if result.returncode != 0:
            failures.append((tag, result.returncode))
            print(f"  [FAILED rc={result.returncode}] {tag}", flush=True)

    if failures:
        print(f"\n{len(failures)}/{len(units)} units failed:")
        for tag, rc in failures:
            print(f"  rc={rc}  {tag}")
    elif not dry_run:
        print(f"\nAll {len(units)} units completed.")


def _timing_column(cfg: BenchConfig) -> str:
    return "mean_ms"


SweepKey = Tuple[str, float, str, int, int]  # (dataset, eps, method, n, d)


def _load_timings(csv_path: Path, timing_col: str) -> Dict[SweepKey, float]:
    """Load one sweep CSV, keyed by (dataset, eps, method, n, d)."""
    timings: Dict[SweepKey, float] = {}
    if not csv_path.exists():
        return timings
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("oom") in ("True", "true"):
                continue
            try:
                key = (
                    row.get("dataset", "gaussian"), float(row["eps"]),
                    row["method"], int(row["n"]), int(row["d"]),
                )
                timings[key] = float(row[timing_col])
            except (KeyError, ValueError):
                continue
    return timings


def run_tf32_comparison(base_cfg: BenchConfig, *, dry_run: bool) -> None:
    """Run the full eps sweep with TF32 on and again with it off, then print a diff table."""
    on_dir = f"{base_cfg.output_dir}/tf32_on"
    off_dir = f"{base_cfg.output_dir}/tf32_off"
    on_cfg = dataclasses.replace(base_cfg, tf32=True)
    off_cfg = dataclasses.replace(base_cfg, tf32=False)

    run_sweep(on_cfg, on_dir, dry_run=dry_run, label="TF32 ON")
    run_sweep(off_cfg, off_dir, dry_run=dry_run, label="TF32 OFF")

    if dry_run:
        return

    timing_col = _timing_column(base_cfg)
    on_timings = _load_timings(_results_csv(on_cfg, on_dir), timing_col)
    off_timings = _load_timings(_results_csv(off_cfg, off_dir), timing_col)

    keys = sorted(set(on_timings) | set(off_timings))
    header = (
        f"\n{'dataset':<12} {'eps':>8} {'method':<20} {'n':>8} {'d':>5} "
        f"{'tf32_on_ms':>12} {'tf32_off_ms':>12} {'off/on':>8}"
    )
    print(header)
    print("-" * 95)
    for key in keys:
        dataset, eps, method, n, d = key
        on_ms = on_timings.get(key)
        off_ms = off_timings.get(key)
        on_str = f"{on_ms:.3f}" if on_ms is not None else "N/A"
        off_str = f"{off_ms:.3f}" if off_ms is not None else "N/A"
        ratio = f"{off_ms / on_ms:.2f}x" if on_ms and off_ms else "N/A"
        print(f"{dataset:<12} {eps:>8g} {method:<20} {n:>8} {d:>5} {on_str:>12} {off_str:>12} {ratio:>8}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a SinkSLOT-vs-baselines benchmark using settings from a configs/ module"
    )
    parser.add_argument("--config", default="base",
                        help="Module in configs/ to load CONFIG from (default: base; e.g. speedup).")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running it.")
    parser.add_argument("--execute", action="store_true", help="Force execution even if CONFIG.dry_run is True.")
    parser.add_argument(
        "--compare-tf32", action="store_true",
        help="Run the benchmark once with TF32 on and once off, then print a timing diff.",
    )
    args = parser.parse_args()
    # Accept either the bare module name ("speedup") or a dotted path
    # ("configs.speedup"); the bare form is what the docs and --help use.
    _name = args.config if "." in args.config else f"configs.{args.config}"
    CONFIG = importlib.import_module(_name).CONFIG

    dry_run = CONFIG.dry_run
    if args.dry_run:
        dry_run = True
    if args.execute:
        dry_run = False

    if args.compare_tf32:
        run_tf32_comparison(CONFIG, dry_run=dry_run)
        return

    run_sweep(CONFIG, CONFIG.output_dir, dry_run=dry_run)


if __name__ == "__main__":
    main()
