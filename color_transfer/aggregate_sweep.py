"""Aggregate main_pixel.py's full-sweep pixel_records_*.json files into a
mean +/- standard-error table across all pairs, matching the format of the
ICLR paper's Table 3 (Cost, Time, Peak memory, Iterations, Nonzero entries %).

    python -m color_transfer.aggregate_sweep --input_dir DIR --eps 0.1

Only "ok" (non-OOM) rows are included. Reports how many of the 132 pairs
were skipped (OOM) or did not converge separately, rather than silently
averaging them in or dropping them without comment.
"""

import argparse
import json
import math
import os

_METHOD_FILES = [
    ("sinkslot", "pixel_records_sinkslot_marginal.json"),
    ("flashsinkhorn", "pixel_records_flashsinkhorn_marginal.json"),
    ("flashsinkhorn_symmetric", "pixel_records_flashsinkhorn_symmetric_marginal.json"),
    ("geomloss_online", "pixel_records_geomloss_online_marginal.json"),
    ("geomloss_multiscale", "pixel_records_geomloss_multiscale_marginal.json"),
    ("flashsinkhorn_fast", "pixel_records_flashsinkhorn_fast_marginal.json"),
    ("flashsinkhorn_symmetric_fast", "pixel_records_flashsinkhorn_symmetric_fast_marginal.json"),
]

_DISPLAY_NAME = {
    "sinkslot": "SinkSLOT",
    "flashsinkhorn": "FlashSinkhorn (alternating)",
    "flashsinkhorn_symmetric": "FlashSinkhorn (symmetric)",
    "geomloss_online": "GeomLoss (online)",
    "geomloss_multiscale": "GeomLoss (multiscale)",
    "flashsinkhorn_fast": "FlashSinkhorn (alternating, fast)",
    "flashsinkhorn_symmetric_fast": "FlashSinkhorn (symmetric, fast)",
}


def mean_se(xs):
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    mean = sum(xs) / n
    if n < 2:
        return mean, float("nan")
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return mean, math.sqrt(var / n)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--eps", type=float, required=True)
    p.add_argument("--num_pairs", type=int, default=132,
                    help="Expected number of pairs (12 images -> 132 ordered pairs); "
                         "used only to report how many are missing so far.")
    return p.parse_args()


def main():
    args = parse_args()
    eps_key = str(args.eps)

    rows = []
    for stem, fname in _METHOD_FILES:
        path = os.path.join(args.input_dir, fname)
        if not os.path.exists(path):
            print(f"skipping {stem}: no {fname} in {args.input_dir}")
            continue
        with open(path) as f:
            data = json.load(f)
        records = data.get("records", {}).get(eps_key, [])
        n_total = len(records)
        n_ok = sum(1 for r in records if r["status"] == "ok")
        n_oom = sum(1 for r in records if r["status"] == "OOM")
        ok_records = [r for r in records if r["status"] == "ok"]
        n_not_converged = sum(1 for r in ok_records if r["converged"] is False)
        n_unknown_converged = sum(1 for r in ok_records if r["converged"] is None)

        costs = [r["cost"] for r in ok_records]
        times = [r["time"] for r in ok_records]
        peak_gb = [r["peak_mb"] / 1024 for r in ok_records]
        iters = [r["iters"] for r in ok_records]
        nonzero_pct = [100.0 * r["support_size"] / (r["n"] * r["m"]) for r in ok_records]

        rows.append({
            "stem": stem,
            "n_total": n_total, "n_ok": n_ok, "n_oom": n_oom,
            "n_not_converged": n_not_converged, "n_unknown_converged": n_unknown_converged,
            "cost": mean_se(costs), "time": mean_se(times),
            "peak_gb": mean_se(peak_gb), "iters": mean_se(iters),
            "nonzero_pct": mean_se(nonzero_pct),
        })

    if not rows:
        raise SystemExit(f"no pixel_records_*.json files found in {args.input_dir}")

    print(f"\n{'Method':<32} {'Pairs (ok/oom/total)':<22} {'Not converged':<15}")
    for r in rows:
        print(f"{_DISPLAY_NAME[r['stem']]:<32} "
              f"{r['n_ok']}/{r['n_oom']}/{r['n_total']:<18} "
              f"{r['n_not_converged']} (+{r['n_unknown_converged']} unknown)")
        if r["n_total"] < args.num_pairs:
            print(f"  WARNING: only {r['n_total']}/{args.num_pairs} pairs recorded so far -- sweep incomplete.")

    print("\nTable (mean +/- SE over converged 'ok' pairs):\n")
    print("| Method | Cost | Time (s) | Peak memory (GB) | Iterations | Nonzero entries (%) |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        def fmt(key, prec=4):
            m, se = r[key]
            if math.isnan(se):
                return f"{m:.{prec}f}"
            return f"{m:.{prec}f}$\\pm${se:.{prec}f}"
        print(f"| {_DISPLAY_NAME[r['stem']]} | {fmt('cost', 4)} | {fmt('time', 3)} | "
              f"{fmt('peak_gb', 2)} | {fmt('iters', 1)} | {fmt('nonzero_pct', 3)} |")

    out_path = os.path.join(args.input_dir, f"table3_eps{args.eps:g}.json")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
