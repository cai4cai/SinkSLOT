"""Render flash_sinkhorn benchmark results (output/**/*.csv) as rich tables.

Usage:
    python view.py                  # render every run found under output/
    python view.py --output-dir output/quick_preview
    python view.py --raw-only       # skip the *_speedup.csv summary tables
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _fmt_ms(value: str) -> str:
    try:
        return f"{float(value):.3f}"
    except ValueError:
        return value


def _fmt_rmae(value: str) -> str:
    if not value or value == "N/A":
        return "[dim]N/A[/dim]"
    try:
        pct = float(value)
    except ValueError:
        return value
    color = "green" if pct < 5 else ("yellow" if pct < 15 else "red")
    return f"[{color}]{pct:.3f}%[/{color}]"


def render_forward_all(path: Path, root: Path) -> Table:
    rows = _read_csv(path)
    has_dataset = bool(rows) and "dataset" in rows[0]
    has_tf32 = bool(rows) and "tf32" in rows[0]
    has_n_iters = bool(rows) and "n_iters" in rows[0]
    has_rmae = bool(rows) and "rmae_pct" in rows[0]
    has_srot = bool(rows) and "srot_slices" in rows[0]

    table = Table(
        title=path.relative_to(root).as_posix(),
        box=box.SIMPLE_HEAVY,
        header_style="bold cyan",
        title_style="bold white",
    )
    columns = ["Method", "n", "m", "d", "eps", "Mean (ms)", "Std (ms)", "Min (ms)", "Max (ms)", "GPU Mem (MB)", "OOM"]
    if has_tf32:
        columns.insert(0, "TF32")
    if has_dataset:
        columns.insert(0, "Dataset")
    if has_n_iters:
        columns.append("Iters")
    if has_srot:
        columns += ["L", "Plan (ms)"]
    if has_rmae:
        columns.append("RMAE vs own optimum")
    for col in columns:
        justify = "left" if col in ("Dataset", "TF32", "Method", "OOM") else "right"
        table.add_column(col, justify=justify)

    groups: Dict[tuple, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row.get("dataset", ""), row.get("tf32", ""), int(row["d"]),
               float(row["eps"]), int(row["n"]))
        groups[key].append(row)

    for key in sorted(groups):
        group_rows = sorted(groups[key], key=lambda r: float(r["mean_ms"]))
        fastest_mean = float(group_rows[0]["mean_ms"]) if group_rows else None
        for row in group_rows:
            is_oom = row.get("oom") in ("True", "true")
            is_fastest = not is_oom and float(row["mean_ms"]) == fastest_mean
            style = "red" if is_oom else ("bold green" if is_fastest else None)
            cells = []
            if has_dataset:
                cells.append(row.get("dataset", ""))
            if has_tf32:
                cells.append(row.get("tf32", ""))
            cells += [
                row["method"],
                row["n"],
                row["m"],
                row["d"],
                row["eps"],
                _fmt_ms(row["mean_ms"]),
                _fmt_ms(row["std_ms"]),
                _fmt_ms(row["min_ms"]),
                _fmt_ms(row["max_ms"]),
                row["gpu_memory_mb"],
                "OOM" if is_oom else "",
            ]
            if has_n_iters:
                cells.append(row.get("n_iters", ""))
            if has_srot:
                cells.append(row.get("srot_slices", ""))
                cells.append(_fmt_ms(row.get("plan_ms", "")))
            if has_rmae:
                cells.append(_fmt_rmae(row.get("rmae_pct", "")))
            table.add_row(*cells, style=style)
        table.add_section()

    return table


def render_speedup(path: Path, root: Path) -> Table:
    rows = _read_csv(path)
    if not rows:
        return Table(title=f"{path.name} (empty)")

    columns = list(rows[0].keys())
    table = Table(
        title=path.relative_to(root).as_posix(),
        box=box.SIMPLE_HEAVY,
        header_style="bold cyan",
        title_style="bold white",
    )
    for col in columns:
        table.add_column(col.replace("_", " "), justify="left" if col == "n" else "right")

    for row in rows:
        cells = []
        for col in columns:
            value = row[col]
            if value == "N/A":
                cells.append("[dim]N/A[/dim]")
            elif "vs" in col:
                cells.append(f"[bold green]{value}[/bold green]")
            elif col.endswith("_ms"):
                cells.append(_fmt_ms(value))
            else:
                cells.append(value)
        table.add_row(*cells)

    return table


def find_runs(output_dir: Path) -> Dict[str, List[Path]]:
    """Group CSVs by their parent directory (a single benchmark run)."""
    runs: Dict[str, List[Path]] = defaultdict(list)
    for csv_path in sorted(output_dir.rglob("*.csv")):
        run_label = csv_path.parent.relative_to(output_dir).as_posix()
        runs[run_label].append(csv_path)
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Render benchmark CSVs in output/ with rich tables.")
    parser.add_argument("--output-dir", default="output", help="Root directory to scan for CSVs (default: output)")
    parser.add_argument("--raw-only", action="store_true", help="Only render forward_all.csv tables, skip speedup summaries.")
    parser.add_argument("--speedup-only", action="store_true", help="Only render *_speedup.csv tables, skip raw results.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        console.print(f"[bold red]No such directory:[/bold red] {output_dir}")
        raise SystemExit(1)

    runs = find_runs(output_dir)
    if not runs:
        console.print(f"[yellow]No CSV files found under {output_dir}[/yellow]")
        return

    for run_label, csv_paths in runs.items():
        raw = [p for p in csv_paths if p.name == "forward_all.csv" or p.name == "backward_all.csv"]
        speedup = [p for p in csv_paths if p.name.endswith("_speedup.csv")]

        renderables = []
        if raw and not args.speedup_only:
            renderables += [render_forward_all(p, output_dir) for p in raw]
        if speedup and not args.raw_only:
            renderables += [render_speedup(p, output_dir) for p in speedup]

        if not renderables:
            continue

        console.print(Panel(Group(*renderables), title=f"[bold]{run_label}[/bold]", border_style="blue"))
        console.print()


if __name__ == "__main__":
    main()
