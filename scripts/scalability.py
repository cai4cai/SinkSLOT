"""Build (and optionally run) the scalability benchmark from configs/scalability.py.

Mirrors run.py's build_command() flag ordering, but isn't expressed as a plain
run.py BenchConfig sweep -- see configs/scalability.py's docstring for why
(L needs to vary independently of N/d, and every unit needs its own per-seed
output directory). Each (experiment, method, seed) combination gets its own
output directory; save_results_csv's merge key doesn't include seed, so
sharing one directory across seeds would silently overwrite all but the last.

Usage:
    python scripts/scalability.py                # print every command (dry run)
    python scripts/scalability.py --execute       # actually run them, in order
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.scalability import (  # noqa: E402
    CHECK_EVERY, D_EXP1, D_EXP2, D_SWEEP, EPS_EXP1, EPS_EXP2, EPS_EXP3,
    L_VALUES, MAX_DENSE_SIZE_DSCALE, MAX_DENSE_SIZE_NSCALE, MAX_ITER,
    METHODS, N_EXP3, N_SWEEP, POTENTIAL_TOL, SEEDS, SROT_DELTA, STOP_MODE, STOP_TOL,
)


def build(d, eps, n, output_dir, *, method, seed, max_dense_size, slices=None,
          extra_flags=None):
    cmd = [
        sys.executable, "-m", "sinkslot.bench.bench_forward",
        "--sizes", str(n), "--dims", str(d), "--eps", str(eps),
        "--n-iters", str(MAX_ITER), "--warmup", "0", "--rep", "5",
        "--max-dense-size", str(max_dense_size), "--output-dir", output_dir,
        "--seed", str(seed), "--no-tf32",
        "--no-ott", "--no-geomloss", "--no-flash-symmetric", "--only", method,
    ]
    if method == "srot":
        cmd += ["--srot-slices", str(slices), "--srot-delta", str(SROT_DELTA)]
    else:
        cmd.append("--no-srot")
    cmd.append("--no-sinkslot")
    if method == "sinkslotcuda":
        cmd += ["--sinkslotcuda-slices", str(slices)]
    else:
        cmd.append("--no-sinkslotcuda")
    cmd += ["--stop-mode", STOP_MODE, "--max-iter", str(MAX_ITER),
            "--stop-tol", str(STOP_TOL), "--potential-tol", str(POTENTIAL_TOL),
            "--check-every", str(CHECK_EVERY)]
    cmd.append("--no-sparsink")
    if extra_flags:
        cmd += extra_flags
    return cmd


def gen_units():
    """Yield (tag, [commands]) for every (experiment, method, seed) group."""
    nscale_experiments = [
        ("exp1_gaussian_d3", D_EXP1, EPS_EXP1),
        ("exp2_gaussian_d64", D_EXP2, EPS_EXP2),
    ]
    for exp_name, d, eps in nscale_experiments:
        for method in METHODS:
            for seed in SEEDS:
                cmds = []
                out = f"output/scalability/{exp_name}_{method}_seed{seed}"
                for n in N_SWEEP:
                    if method in ("sinkslotcuda", "srot"):
                        for L in L_VALUES:
                            cmds.append(build(d, eps, n, out, method=method, seed=seed,
                                               max_dense_size=MAX_DENSE_SIZE_NSCALE, slices=L))
                    else:
                        cmds.append(build(d, eps, n, out, method=method, seed=seed,
                                           max_dense_size=MAX_DENSE_SIZE_NSCALE))
                yield f"{exp_name}_{method}_seed{seed}", cmds

    for method in METHODS:
        for seed in SEEDS:
            cmds = []
            out = f"output/scalability/exp3_dscale_{method}_seed{seed}"
            extra = ["--no-rmae-check"] if method == "sinkslotcuda" else None
            for d in D_SWEEP:
                if method in ("sinkslotcuda", "srot"):
                    for L in L_VALUES:
                        cmds.append(build(d, EPS_EXP3, N_EXP3, out, method=method, seed=seed,
                                           max_dense_size=MAX_DENSE_SIZE_DSCALE, slices=L,
                                           extra_flags=extra))
                else:
                    cmds.append(build(d, EPS_EXP3, N_EXP3, out, method=method, seed=seed,
                                       max_dense_size=MAX_DENSE_SIZE_DSCALE))
            yield f"exp3_dscale_{method}_seed{seed}", cmds


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true", help="actually run the commands")
    args = ap.parse_args()

    total = 0
    for tag, cmds in gen_units():
        print(f"=== {tag}: {len(cmds)} units ===")
        for cmd in cmds:
            total += 1
            if args.execute:
                subprocess.run(cmd, check=False)
            else:
                print(" ".join(cmd))
    print(f"\n{total} total units.")


if __name__ == "__main__":
    main()
