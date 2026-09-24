"""Build and (optionally) execute a SinkSLOT-vs-baselines benchmark from a configs/ module.

Sweeps every (dataset, d, eps) problem from CONFIG across every enabled method:
SinkSLOT/SinkSLOT-CUDA, SROT, Spar-Sink/Rand-Sink, GeomLoss and FlashSinkhorn. All
runs append into a single `<output_dir>/forward_all.csv` plus one speedup table,
with dataset, eps, d and n as ordinary columns. A plain run deletes existing CSVs
first, so each invocation produces a clean table.

Sharded runs split the unit list round-robin: shard k of K runs units[k::K] and
writes to `<output_dir>/shards/shard{k:04d}/`. Sharding implies resume: CSVs are
never cleared, and any unit that already has a row (including an OOM row) in
`<output_dir>/forward_all.csv` or any shard CSV is skipped. `--merge` then
concatenates the shard CSVs into `<output_dir>/forward_all.csv`.

Usage:
    python run.py                  # honors CONFIG.dry_run
    python run.py --dry-run        # force dry run (just print the command)
    python run.py --execute        # force real execution
    python run.py --compare-tf32   # run once with TF32 on and once off, then diff timings
    python run.py --config speedup --execute --num-shards 210 --shard-idx 7
    python run.py --config speedup --merge
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, NamedTuple, Optional, Set, Tuple

import importlib

from configs.base import BenchConfig

FLASH_METHODS = ("flash_symmetric", "flash_alternating")
SPARSINK_METHODS = ("spar_sink", "rand_sink")
SLICED_METHODS = ("srot", "sinkslot", "sinkslotcuda", "sinkslotcuda_symmetric")

# --only value -> the method string bench_forward writes in the CSV.
CSV_METHOD = {"geomloss": "geomloss_online"}

EPS_REL_TOL = 1e-9


class Unit(NamedTuple):
    """One subprocess worth of work. method/n/d/slices are None when not isolated."""
    dataset: str
    eps: float
    method: Optional[str]
    n: Optional[int]
    d: Optional[int]
    slices: Optional[int]
    seed: int
    tf32: bool


def build_command(
    cfg: BenchConfig, dataset: str, eps: float, output_dir: str,
    *, method: Optional[str] = None, size: Optional[int] = None, dim: Optional[int] = None,
    slices: Optional[int] = None, seed: Optional[int] = None, tf32: Optional[bool] = None,
) -> List[str]:
    """Build the benchmark command for a single unit of work.

    The underlying bench module takes one --dataset/--eps per invocation, so both are
    passed explicitly rather than read off cfg. When method/size/dim are given the run is
    narrowed to that single measurement (isolation mode); otherwise the full grid runs in
    one process.

    seed: data-generation seed for this unit (defaults to cfg.seeds[0] if not given).
    Only affects (x, y, a, b); method-internal randomness stays independently seeded.
    tf32: TF32 setting for this unit (defaults to cfg.tf32).
    """
    module = "sinkslot.bench.bench_forward"
    cmd = [
        sys.executable, "-m", module,
        "--sizes", str(size) if size is not None else ",".join(str(s) for s in cfg.sizes),
        "--dims", str(dim) if dim is not None else ",".join(str(d) for d in cfg.dims),
        "--eps", str(eps),
        "--n-iters", str(cfg.n_iters),
        "--warmup", str(cfg.warmup),
        "--warmup-iters", str(cfg.warmup_iters),
        "--rep", str(cfg.rep),
        "--max-dense-size", str(cfg.max_dense_size),
        "--output-dir", output_dir,
        "--seed", str(seed if seed is not None else cfg.seeds[0]),
    ]

    cmd.append("--tf32" if (cfg.tf32 if tf32 is None else tf32) else "--no-tf32")
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

    # SinkSLOT-CUDA and its symmetric variant share --sinkslotcuda-slices.
    if cfg.no_sinkslotcuda:
        cmd.append("--no-sinkslotcuda")
    if cfg.no_sinkslotcuda_symmetric:
        cmd.append("--no-sinkslotcuda-symmetric")
    if (method in (None, "sinkslotcuda", "sinkslotcuda_symmetric")
            and not (cfg.no_sinkslotcuda and cfg.no_sinkslotcuda_symmetric)):
        sc_values = [slices] if slices is not None and method is not None else cfg.sinkslotcuda_slices
        cmd += ["--sinkslotcuda-slices", ",".join(str(v) for v in sc_values)]

    if cfg.stop_mode != "fixed":
        cmd += ["--stop-mode", cfg.stop_mode,
                "--max-iter", str(cfg.max_iter),
                "--stop-tol", str(_stop_tol(cfg, dataset, dim)),
                "--scaling-tol", str(cfg.scaling_tol),
                "--check-every", str(cfg.check_every)]

    if cfg.no_sparsink:
        cmd.append("--no-sparsink")
    elif method in (None,) + SPARSINK_METHODS:
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


def _stop_tol(cfg: BenchConfig, dataset: str, dim: Optional[int]) -> float:
    """cfg.stop_tol_by_problem[(dataset, dim)] when set, else cfg.stop_tol."""
    by_problem = cfg.stop_tol_by_problem or {}
    return by_problem.get((dataset, dim), cfg.stop_tol)


def _results_csv(cfg: BenchConfig, output_dir: str) -> Path:
    return Path(output_dir) / "forward_all.csv"


def _speedup_csv(cfg: BenchConfig, output_dir: str) -> Path:
    return Path(output_dir) / "forward_speedup.csv"


def _shard_dir(output_dir: str, shard_idx: int) -> str:
    return str(Path(output_dir) / "shards" / f"shard{shard_idx:04d}")


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
    if not cfg.no_sinkslotcuda_symmetric:
        names.append("sinkslotcuda_symmetric")
    if not cfg.no_sparsink:
        names.append("spar_sink")
        if not cfg.no_randsink:
            names.append("rand_sink")
    return names


def _problems(cfg: BenchConfig) -> Iterator[Tuple[str, Optional[int], List[float]]]:
    """(dataset, d, eps_values) triples. d is None when all cfg.dims share one process."""
    if cfg.problems is not None:
        for dataset, d, eps_values in cfg.problems:
            yield dataset, d, list(eps_values)
    elif cfg.isolate:
        for dataset in cfg.datasets:
            for d in cfg.dims:
                yield dataset, d, list(cfg.eps_values)
    else:
        for dataset in cfg.datasets:
            yield dataset, None, list(cfg.eps_values)


def _units(cfg: BenchConfig) -> Iterator[Unit]:
    """Yield one Unit per subprocess.

    With cfg.isolate the sweep is fully unrolled so every measured row gets a fresh
    process -- necessary because gpu_memory_mb reports whole-device usage, which is
    cumulative within a process. Without it, one process per (dataset, eps, seed).

    seed is the outermost loop: it varies only the underlying problem instance
    (x, y, a, b), so every other combination gets repeated once per seed, letting
    results be aggregated across seeds afterward. FlashSinkhorn units are repeated
    over cfg.flash_tf32; every other unit uses cfg.tf32.
    """
    for seed in cfg.seeds:
        for dataset, d, eps_values in _problems(cfg):
            for eps in eps_values:
                if not cfg.isolate:
                    yield Unit(dataset, eps, None, None, d, None, seed, cfg.tf32)
                    continue
                for method in _methods(cfg):
                    tf32_values = cfg.flash_tf32 if method in FLASH_METHODS else [cfg.tf32]
                    if method in SPARSINK_METHODS:
                        # s expands these two only, like L for the sliced methods.
                        param_values: List[Optional[int]] = list(cfg.sparsink_s)
                    elif method in SLICED_METHODS:
                        # L expands the sliced methods only; for every other method it
                        # is meaningless and would just duplicate identical rows.
                        param_values = list({
                            "srot": cfg.srot_slices,
                            "sinkslot": cfg.sinkslot_slices,
                            "sinkslotcuda": cfg.sinkslotcuda_slices,
                            "sinkslotcuda_symmetric": cfg.sinkslotcuda_slices,
                        }[method])
                    else:
                        param_values = [None]
                    for size in cfg.sizes:
                        for slices in param_values:
                            for tf32 in tf32_values:
                                yield Unit(dataset, eps, method, size, d, slices, seed, tf32)


def _unit_tag(unit: Unit) -> str:
    tag = f"dataset={unit.dataset} eps={unit.eps} seed={unit.seed}"
    if unit.method is not None:
        tag += f" {unit.method} n={unit.n} d={unit.d}"
    if unit.slices is not None:
        tag += (f" s={unit.slices}" if unit.method in SPARSINK_METHODS else f" L={unit.slices}")
    return tag + f" tf32={unit.tf32}"


# ---------------------------------------------------------------------------
# Resume: which units already have a row.

_DoneKey = Tuple[str, bool, str, int, int, str, str, int]  # all but eps
_DoneIndex = Dict[_DoneKey, List[float]]


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in ("true", "1")


def _norm_param(value) -> str:
    return "N/A" if value in (None, "", "N/A") else str(int(float(value)))


def _read_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _existing_result_csvs(output_dir: str) -> List[Path]:
    base = Path(output_dir)
    return [base / "forward_all.csv"] + sorted((base / "shards").glob("*/forward_all.csv"))


def _load_done(paths: Iterable[Path]) -> _DoneIndex:
    """Index every row (OOM rows included) by its unit identity, eps kept as a float list."""
    done: _DoneIndex = defaultdict(list)
    for path in paths:
        for row in _read_rows(path):
            try:
                key = (
                    row.get("dataset", "gaussian"), _as_bool(row["tf32"]), row["method"],
                    int(row["n"]), int(row["d"]),
                    _norm_param(row.get("srot_slices")), _norm_param(row.get("sample_size")),
                    int(row.get("seed") or 0),
                )
                done[key].append(float(row["eps"]))
            except (KeyError, ValueError):
                continue
    return done


def _unit_done_key(unit: Unit) -> _DoneKey:
    L = unit.slices if unit.method in SLICED_METHODS else None
    s = unit.slices if unit.method in SPARSINK_METHODS else None
    return (
        unit.dataset, bool(unit.tf32), CSV_METHOD.get(unit.method, unit.method),
        int(unit.n), int(unit.d), _norm_param(L), _norm_param(s), int(unit.seed),
    )


def _is_done(unit: Unit, done: _DoneIndex) -> bool:
    if unit.method is None or unit.n is None or unit.d is None:
        return False  # a non-isolated unit covers many rows; never skipped
    return any(
        math.isclose(eps, unit.eps, rel_tol=EPS_REL_TOL, abs_tol=0.0)
        for eps in done.get(_unit_done_key(unit), ())
    )


# ---------------------------------------------------------------------------
# Sweep.

def count_units(cfg: BenchConfig, base_dir: str, *, num_shards: int = 1, shard_idx: int = 0) -> None:
    """Print total/done/remaining unit counts, overall and per method."""
    all_units = list(_units(cfg))
    units = all_units[shard_idx::num_shards] if num_shards > 1 else all_units
    done = _load_done(_existing_result_csvs(base_dir))
    n_done = sum(_is_done(u, done) for u in units)
    per_method: Dict[str, int] = defaultdict(int)
    for u in units:
        per_method[str(u.method)] += 1
    scope = f"shard {shard_idx}/{num_shards}: " if num_shards > 1 else ""
    print(f"{scope}{len(all_units)} units total, {len(units)} in scope, "
          f"{n_done} done, {len(units) - n_done} remaining")
    for method, count in per_method.items():
        print(f"  {method}: {count}")



def run_sweep(
    cfg: BenchConfig, base_dir: str, *, dry_run: bool, label: str = "",
    num_shards: int = 1, shard_idx: int = 0, resume: bool = False,
) -> None:
    """Run the benchmark for every unit of work (or one shard of them)."""
    prefix = f"{label} " if label else ""
    sharded = num_shards > 1
    resume = resume or sharded
    all_units = list(_units(cfg))
    units = all_units[shard_idx::num_shards] if sharded else all_units
    out_dir = _shard_dir(base_dir, shard_idx) if sharded else base_dir

    done: _DoneIndex = _load_done(_existing_result_csvs(base_dir)) if resume else {}
    todo = [u for u in units if not _is_done(u, done)]
    scope = f"shard {shard_idx}/{num_shards}: " if sharded else ""
    print(f"{prefix}{scope}{len(all_units)} units total, {len(units)} in scope, "
          f"{len(units) - len(todo)} done, {len(todo)} remaining", flush=True)

    if not dry_run and not resume:
        _clear_csvs(cfg, out_dir)
    failures: List[Tuple[str, int]] = []
    for i, unit in enumerate(todo, 1):
        cmd = build_command(
            cfg, unit.dataset, unit.eps, out_dir, method=unit.method, size=unit.n,
            dim=unit.d, slices=unit.slices, seed=unit.seed, tf32=unit.tf32,
        )
        printable = " ".join(cmd)
        tag = _unit_tag(unit)
        if dry_run:
            print(f"[dry-run] {prefix}{tag} would execute:")
            print(f"  {printable}")
            continue
        print(f"\n[{i}/{len(todo)}] {prefix}{tag}: {printable}", flush=True)
        # Don't abort the sweep on one bad unit. A single subprocess can die for reasons
        # unrelated to the rest of the grid (a transient CUDA/driver fault, an OOM at one
        # size), and check=True would then throw away every remaining unit. Record it and
        # carry on; the run ends with a summary of what failed.
        result = subprocess.run(cmd)
        if result.returncode != 0:
            failures.append((tag, result.returncode))
            print(f"  [FAILED rc={result.returncode}] {tag}", flush=True)

    if failures:
        print(f"\n{len(failures)}/{len(todo)} units failed:")
        for tag, rc in failures:
            print(f"  rc={rc}  {tag}")
    elif not dry_run:
        print(f"\nAll {len(todo)} units completed.")


# ---------------------------------------------------------------------------
# Merge.

def _row_key(row: dict) -> tuple:
    """bench_forward's row key: (dataset, tf32, method, n, m, d, eps, n_iters,
    srot_slices, sample_size, seed), with eps compared as a float."""
    return (
        row.get("dataset", "gaussian"), _as_bool(row.get("tf32", "")), row.get("method", ""),
        str(row.get("n", "")), str(row.get("m", "")), str(row.get("d", "")),
        float(row["eps"]), str(row.get("n_iters", "")),
        _norm_param(row.get("srot_slices")), _norm_param(row.get("sample_size")),
        str(row.get("seed", "")),
    )


def merge_shards(output_dir: str) -> Path:
    """Concatenate every shard's forward_all.csv (and any existing top-level one) into
    output_dir/forward_all.csv, one row per key. Shard rows win over top-level rows."""
    paths = _existing_result_csvs(output_dir)
    merged: Dict[tuple, dict] = {}
    columns: List[str] = []
    n_read = 0
    for path in paths:
        if not path.exists():
            continue
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for col in reader.fieldnames or []:
                if col not in columns:
                    columns.append(col)
            for row in reader:
                n_read += 1
                try:
                    merged[_row_key(row)] = row
                except (KeyError, ValueError):
                    continue
    out = Path(output_dir) / "forward_all.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(merged.values(), key=lambda r: tuple(str(v) for v in _row_key(r)))
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)
    n_shards = sum(1 for p in paths[1:] if p.exists())
    print(f"Merged {n_read} rows from {n_shards} shard CSVs into {len(rows)} unique rows: {out}")
    return out


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
    on_cfg = dataclasses.replace(base_cfg, tf32=True, flash_tf32=[True])
    off_cfg = dataclasses.replace(base_cfg, tf32=False, flash_tf32=[False])

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
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Split the units round-robin into this many shards (implies --resume).")
    parser.add_argument("--shard-idx", type=int, default=0,
                        help="Which shard to run, in [0, --num-shards).")
    parser.add_argument("--resume", action="store_true",
                        help="Keep existing CSVs and skip units that already have a row.")
    parser.add_argument("--merge", action="store_true",
                        help="Merge output_dir/shards/*/forward_all.csv into output_dir/forward_all.csv and exit.")
    parser.add_argument("--count", action="store_true",
                        help="Print the unit counts without printing or running any command.")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_idx < args.num_shards:
        parser.error(f"--shard-idx must be in [0, {args.num_shards}), got {args.shard_idx}")
    # Accept either the bare module name ("speedup") or a dotted path
    # ("configs.speedup"); the bare form is what the docs and --help use.
    _name = args.config if "." in args.config else f"configs.{args.config}"
    CONFIG = importlib.import_module(_name).CONFIG

    if args.merge:
        merge_shards(CONFIG.output_dir)
        return

    if args.count:
        count_units(CONFIG, CONFIG.output_dir, num_shards=args.num_shards,
                    shard_idx=args.shard_idx)
        return

    dry_run = CONFIG.dry_run
    if args.dry_run:
        dry_run = True
    if args.execute:
        dry_run = False

    if args.compare_tf32:
        run_tf32_comparison(CONFIG, dry_run=dry_run)
        return

    run_sweep(CONFIG, CONFIG.output_dir, dry_run=dry_run, num_shards=args.num_shards,
              shard_idx=args.shard_idx, resume=args.resume)


if __name__ == "__main__":
    main()
