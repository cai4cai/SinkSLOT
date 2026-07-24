"""Build and (optionally) execute a flash_sinkhorn benchmark from config.py.

Sweeps every (dataset, eps) combination from CONFIG. All runs append into a single
`<output_dir>/forward_all.csv` (or backward_all.csv) plus one speedup table, with
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
from typing import Dict, List, Tuple

from config import CONFIG, BenchConfig


def build_command(cfg: BenchConfig, dataset: str, eps: float, output_dir: str) -> List[str]:
    """Build the benchmark command for a single (dataset, eps) point.

    The underlying bench module takes one --dataset/--eps per invocation, so both
    are passed explicitly rather than read off cfg.
    """
    if cfg.which not in ("forward", "backward"):
        raise ValueError(f"Unknown benchmark kind: {cfg.which!r}")

    module = f"flash_sinkhorn.bench.bench_{cfg.which}"
    cmd = [
        sys.executable, "-m", module,
        "--sizes", ",".join(str(s) for s in cfg.sizes),
        "--dims", ",".join(str(d) for d in cfg.dims),
        "--eps", str(eps),
        "--n-iters", str(cfg.n_iters),
        "--warmup", str(cfg.warmup),
        "--rep", str(cfg.rep),
        "--max-dense-size", str(cfg.max_dense_size),
        "--output-dir", output_dir,
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
    if cfg.only:
        cmd += ["--only", cfg.only]
    if cfg.tensorized:
        cmd.append("--tensorized")
    if cfg.verify:
        cmd.append("--verify")
    if cfg.quiet:
        cmd.append("--quiet")

    return cmd


def _results_csv(cfg: BenchConfig, output_dir: str) -> Path:
    name = "forward_all.csv" if cfg.which == "forward" else "backward_all.csv"
    return Path(output_dir) / name


def _speedup_csv(cfg: BenchConfig, output_dir: str) -> Path:
    name = "forward_speedup.csv" if cfg.which == "forward" else "backward_speedup.csv"
    return Path(output_dir) / name


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


def run_sweep(cfg: BenchConfig, base_dir: str, *, dry_run: bool, label: str = "") -> None:
    """Run the benchmark for every (dataset, eps) pair, all into one output directory."""
    prefix = f"{label} " if label else ""
    if not dry_run:
        _clear_csvs(cfg, base_dir)
    for dataset in cfg.datasets:
        for eps in cfg.eps_values:
            cmd = build_command(cfg, dataset, eps, base_dir)
            printable = " ".join(cmd)
            if dry_run:
                print(f"[dry-run] {prefix}dataset={dataset} eps={eps} would execute:")
                print(f"  {printable}")
                continue
            print(f"\nRunning {prefix}dataset={dataset} eps={eps}: {printable}", flush=True)
            subprocess.run(cmd, check=True)


def _timing_column(cfg: BenchConfig) -> str:
    return "mean_ms" if cfg.which == "forward" else "total_ms"


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
        description="Run a flash_sinkhorn benchmark using settings from config.py"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running it.")
    parser.add_argument("--execute", action="store_true", help="Force execution even if CONFIG.dry_run is True.")
    parser.add_argument(
        "--compare-tf32", action="store_true",
        help="Run the benchmark once with TF32 on and once off, then print a timing diff.",
    )
    args = parser.parse_args()

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
