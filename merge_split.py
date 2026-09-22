"""Merge output/table1_split_srot and output/table1_split_rest (from
run_split.py) back into configs/speedup_gaussian_d3's own output_dir
(output/table1), keyed and sorted the same way save_results_csv does.

Usage:
    python merge_split.py
"""
import csv
from pathlib import Path

from sinkslot.bench.bench_forward import FORWARD_CSV_COLUMNS, _forward_key

SOURCES = ["output/table1_split_srot", "output/table1_split_rest"]
DEST = Path("output/table1")


def _load(path: Path) -> dict:
    rows = {}
    if not path.exists():
        print(f"  (missing, skipped: {path})")
        return rows
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if all(col in row for col in FORWARD_CSV_COLUMNS):
                rows[_forward_key(row)] = row
    print(f"  {len(rows)} rows from {path}")
    return rows


def _sort_key(row: dict) -> tuple:
    return (
        row["dataset"], str(row["tf32"]), int(row["d"]), float(row["eps"]),
        int(row["n"]), row["method"],
    )


def main():
    merged: dict = {}
    for src in SOURCES:
        for name in ("forward_all.csv",):
            merged.update(_load(Path(src) / name))

    DEST.mkdir(parents=True, exist_ok=True)
    dest_path = DEST / "forward_all.csv"
    existing = _load(dest_path)
    existing.update(merged)  # split results win over anything stale already in DEST

    with open(dest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FORWARD_CSV_COLUMNS)
        writer.writeheader()
        for row in sorted(existing.values(), key=_sort_key):
            writer.writerow(row)
    print(f"Wrote {len(existing)} total rows to {dest_path}")


if __name__ == "__main__":
    main()
