"""Browse flash_sinkhorn benchmark results (output/**/*.csv) in a scrollable TUI.

Usage:
    python view.py                  # launch the TUI over every run under output/
    python view.py --output-dir output/quick_preview
    python view.py --raw-only       # skip the *_speedup.csv summary tables
    python view.py --print          # non-interactive: dump rich tables to stdout

Keys inside the TUI:
    up/down, j/k     scroll the table
    left/right       switch CSV tabs when the tab bar is focused
    tab              cycle focus: tabs -> table
    /                filter rows (substring match over the whole row)
    r                reload the CSVs from disk
    q                quit
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

RAW_NAMES = ("forward_all.csv", "backward_all.csv")


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


def raw_grid(path: Path) -> Tuple[List[str], List[str], List[List[str]], List[str | None]]:
    """Return (columns, justifications, rows-of-markup, per-row style) for a *_all.csv."""
    rows = _read_csv(path)
    has_dataset = bool(rows) and "dataset" in rows[0]
    has_tf32 = bool(rows) and "tf32" in rows[0]
    has_n_iters = bool(rows) and "n_iters" in rows[0]
    has_rmae = bool(rows) and "rmae_pct" in rows[0]
    has_srot = bool(rows) and "srot_slices" in rows[0]

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
    justify = ["left" if c in ("Dataset", "TF32", "Method", "OOM") else "right" for c in columns]

    groups: Dict[tuple, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row.get("dataset", ""), row.get("tf32", ""), int(row["d"]),
               float(row["eps"]), int(row["n"]))
        groups[key].append(row)

    cells_out: List[List[str]] = []
    styles_out: List[str | None] = []
    for key in sorted(groups):
        group_rows = sorted(groups[key], key=lambda r: float(r["mean_ms"]))
        fastest_mean = float(group_rows[0]["mean_ms"]) if group_rows else None
        for row in group_rows:
            is_oom = row.get("oom") in ("True", "true")
            is_fastest = not is_oom and float(row["mean_ms"]) == fastest_mean
            styles_out.append("red" if is_oom else ("bold green" if is_fastest else None))
            cells: List[str] = []
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
            cells_out.append(cells)

    return columns, justify, cells_out, styles_out


def speedup_grid(path: Path) -> Tuple[List[str], List[str], List[List[str]], List[str | None]]:
    rows = _read_csv(path)
    if not rows:
        return [], [], [], []

    keys = list(rows[0].keys())
    columns = [k.replace("_", " ") for k in keys]
    justify = ["left" if k == "n" else "right" for k in keys]

    cells_out: List[List[str]] = []
    for row in rows:
        cells = []
        for col in keys:
            value = row[col]
            if value == "N/A":
                cells.append("[dim]N/A[/dim]")
            elif "vs" in col:
                cells.append(f"[bold green]{value}[/bold green]")
            elif col.endswith("_ms"):
                cells.append(_fmt_ms(value))
            else:
                cells.append(value)
        cells_out.append(cells)

    return columns, justify, cells_out, [None] * len(cells_out)


def grid_for(path: Path) -> Tuple[List[str], List[str], List[List[str]], List[str | None]]:
    return raw_grid(path) if path.name in RAW_NAMES else speedup_grid(path)


def find_runs(output_dir: Path) -> Dict[str, List[Path]]:
    """Group CSVs by their parent directory (a single benchmark run)."""
    runs: Dict[str, List[Path]] = defaultdict(list)
    for csv_path in sorted(output_dir.rglob("*.csv")):
        run_label = csv_path.parent.relative_to(output_dir).as_posix()
        runs[run_label].append(csv_path)
    return runs


def select_paths(output_dir: Path, raw_only: bool, speedup_only: bool) -> Dict[str, List[Path]]:
    """Runs -> the CSVs that pass the --raw-only / --speedup-only filters."""
    selected: Dict[str, List[Path]] = {}
    for run_label, csv_paths in find_runs(output_dir).items():
        raw = [p for p in csv_paths if p.name in RAW_NAMES]
        speedup = [p for p in csv_paths if p.name.endswith("_speedup.csv")]
        paths: List[Path] = []
        if not speedup_only:
            paths += raw
        if not raw_only:
            paths += speedup
        if paths:
            selected[run_label] = paths
    return selected


# --------------------------------------------------------------------------- #
# static (non-interactive) rendering
# --------------------------------------------------------------------------- #

def _rich_table(path: Path, root: Path) -> Table:
    columns, justify, rows, styles = grid_for(path)
    table = Table(
        title=path.relative_to(root).as_posix(),
        box=box.SIMPLE_HEAVY,
        header_style="bold cyan",
        title_style="bold white",
    )
    for col, just in zip(columns, justify):
        table.add_column(col, justify=just)
    for cells, style in zip(rows, styles):
        table.add_row(*cells, style=style)
    return table


def print_all(output_dir: Path, raw_only: bool, speedup_only: bool) -> None:
    selected = select_paths(output_dir, raw_only, speedup_only)
    if not selected:
        console.print(f"[yellow]No CSV files found under {output_dir}[/yellow]")
        return
    for run_label, paths in selected.items():
        renderables = [_rich_table(p, output_dir) for p in paths]
        console.print(Panel(Group(*renderables), title=f"[bold]{run_label}[/bold]", border_style="blue"))
        console.print()


# --------------------------------------------------------------------------- #
# TUI
# --------------------------------------------------------------------------- #

def build_app(output_dir: Path, raw_only: bool, speedup_only: bool):
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Vertical
    from textual.widgets import DataTable, Footer, Header, Input, Static, Tab, Tabs

    class BenchViewer(App):
        CSS = """
        #body { height: 1fr; }
        #title { padding: 0 1; background: $boost; color: $text; height: 1; }
        #filter { display: none; }
        #filter.visible { display: block; }
        DataTable { height: 1fr; }
        """

        BINDINGS = [
            Binding("q", "quit", "Quit"),
            Binding("r", "reload", "Reload"),
            Binding("slash", "focus_filter", "Filter", key_display="/"),
            Binding("escape", "clear_filter", "Clear filter", show=False),
        ]

        def __init__(self) -> None:
            super().__init__()
            self.output_dir = output_dir
            self.current_path: Path | None = None
            self.filter_text = ""
            self.tab_paths: Dict[str, Path] = {}

        def compose(self) -> ComposeResult:
            yield Header()
            with Vertical(id="body"):
                yield Tabs(id="tabs")
                yield Static("", id="title")
                yield Input(placeholder="filter rows…", id="filter")
                yield DataTable(id="table", zebra_stripes=True, cursor_type="row")
            yield Footer()

        async def on_mount(self) -> None:
            self.title = "flash-sinkhorn benchmarks"
            self.sub_title = str(self.output_dir)
            await self.populate_tabs()

        # -- data ---------------------------------------------------------- #

        async def populate_tabs(self) -> None:
            """One tab per CSV across every run; activating a tab shows it."""
            runs = select_paths(self.output_dir, raw_only, speedup_only)
            paths = [path for run_paths in runs.values() for path in run_paths]
            self.tab_paths = {f"csv{i}": path for i, path in enumerate(paths)}

            tabs = self.query_one("#tabs", Tabs)
            await tabs.clear()  # awaited so the old tab ids are free to reuse
            for tab_id, path in self.tab_paths.items():
                label = path.name if len(runs) == 1 else path.relative_to(self.output_dir).as_posix()
                tabs.add_tab(Tab(label, id=tab_id))

            if not paths:
                self.query_one("#table", DataTable).clear(columns=True)
                self.query_one("#title", Static).update(f"No CSV files found under {self.output_dir}")

        def show_path(self, path: Path) -> None:
            self.current_path = path
            table = self.query_one("#table", DataTable)
            table.clear(columns=True)
            try:
                columns, justify, rows, styles = grid_for(path)
            except Exception as exc:  # a malformed CSV shouldn't kill the app
                self.query_one("#title", Static).update(f"[red]{path}: {exc}[/red]")
                return

            for col, just in zip(columns, justify):
                table.add_column(Text(col, style="bold cyan", justify=just), key=col)

            needle = self.filter_text.lower()
            shown = 0
            for cells, style in zip(rows, styles):
                plain = [Text.from_markup(c) for c in cells]
                if needle and needle not in " ".join(t.plain for t in plain).lower():
                    continue
                if style:
                    for t in plain:
                        t.stylize(style)
                for t, just in zip(plain, justify):
                    t.justify = just
                table.add_row(*plain)
                shown += 1

            label = path.relative_to(self.output_dir).as_posix()
            suffix = f"  [dim]({shown}/{len(rows)} rows, filter: {self.filter_text!r})[/dim]" if needle \
                else f"  [dim]({shown} rows)[/dim]"
            self.query_one("#title", Static).update(f"[bold]{label}[/bold]{suffix}")

        # -- events -------------------------------------------------------- #

        def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
            path = self.tab_paths.get(event.tab.id or "")
            if path is not None:
                self.show_path(path)

        def on_input_changed(self, event: Input.Changed) -> None:
            self.filter_text = event.value
            if self.current_path is not None:
                self.show_path(self.current_path)

        def on_input_submitted(self, event: Input.Submitted) -> None:
            self.query_one("#table", DataTable).focus()

        # -- actions ------------------------------------------------------- #

        def action_focus_filter(self) -> None:
            self.query_one("#filter", Input).add_class("visible")
            self.query_one("#filter", Input).focus()

        def action_clear_filter(self) -> None:
            filter_input = self.query_one("#filter", Input)
            filter_input.value = ""
            filter_input.remove_class("visible")
            self.query_one("#table", DataTable).focus()

        async def action_reload(self) -> None:
            await self.populate_tabs()
            self.notify("Reloaded from disk")

    return BenchViewer()


def main() -> None:
    parser = argparse.ArgumentParser(description="Browse benchmark CSVs in output/ as a TUI.")
    parser.add_argument("--output-dir", default="output", help="Root directory to scan for CSVs (default: output)")
    parser.add_argument("--raw-only", action="store_true", help="Only show forward_all.csv / backward_all.csv tables.")
    parser.add_argument("--speedup-only", action="store_true", help="Only show *_speedup.csv tables.")
    parser.add_argument("--print", dest="print_only", action="store_true",
                        help="Dump static rich tables to stdout instead of launching the TUI.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        console.print(f"[bold red]No such directory:[/bold red] {output_dir}")
        raise SystemExit(1)

    if args.print_only:
        print_all(output_dir, args.raw_only, args.speedup_only)
        return

    try:
        build_app(output_dir, args.raw_only, args.speedup_only).run()
    except ImportError:
        console.print("[yellow]textual is not installed; falling back to static output "
                      "(pip install textual)[/yellow]")
        print_all(output_dir, args.raw_only, args.speedup_only)


if __name__ == "__main__":
    main()
