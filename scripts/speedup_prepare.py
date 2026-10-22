"""One-off preparation for configs/speedup.py, on one CUDA GPU.

Three steps, each skippable:

  1. median: for every (dataset, d) slice, the lower median of the full squared
     Euclidean cost on the seed-0 instance (N=M=10,000, drawn exactly as
     bench_forward draws it). Prints a MEDIAN_C dict to paste into
     configs/speedup.py.
  2. exact-OT: fills bench_forward's exact-OT reference cache for every slice
     and seed, so array tasks read it instead of each solving ot.emd.
  3. warm: runs every GPU method once per d (a few iterations, no accuracy
     reference) so the Triton and KeOps compile caches are populated. Uses
     the medians from step 1 when MEDIAN_C is still unset.

Usage:
    python scripts/speedup_prepare.py                     # all three steps
    python scripts/speedup_prepare.py --skip-exact-ot --skip-warm
    python scripts/speedup_prepare.py --skip-median --seeds 0,1 --workers 4
"""
from __future__ import annotations

import argparse
import dataclasses
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from configs import speedup  # noqa: E402
from sinkslot.bench.bench_forward import (  # noqa: E402
    _cached_exact_ot_reference, sample_point_cloud,
)

N = speedup.N
SLICES = list(speedup.MEDIAN_C)


def sample_instance(dataset: str, d: int, seed: int, device: torch.device):
    """(x, y, a, b) in bench_forward's own draw order."""
    torch.manual_seed(seed)
    x = sample_point_cloud(N, d, device, dataset=dataset, target=False)
    y = sample_point_cloud(N, d, device, dataset=dataset, target=True)
    a = torch.rand(N, device=device, dtype=torch.float32) + 0.1
    b = torch.rand(N, device=device, dtype=torch.float32) + 0.1
    return x, y, a / a.sum(), b / b.sum()


def lower_median_sq_cost(x: torch.Tensor, y: torch.Tensor, block: int = 256) -> float:
    """Lower median of ||x_i - y_j||^2 over all (i, j), in x's dtype (fp32)."""
    cost = torch.empty(x.shape[0], y.shape[0], dtype=x.dtype, device=x.device)
    for start in range(0, x.shape[0], block):
        stop = min(start + block, x.shape[0])
        cost[start:stop] = (x[start:stop, None, :] - y[None, :, :]).square().sum(-1)
    flat = cost.view(-1)
    return float(flat.kthvalue((flat.numel() + 1) // 2).values)


def step_median(device: torch.device) -> dict:
    print("== median(C), seed 0 ==", flush=True)
    medians = {}
    for dataset, d in SLICES:
        x, y, _, _ = sample_instance(dataset, d, 0, device)
        medians[(dataset, d)] = lower_median_sq_cost(x, y)
        print(f"  {dataset} d={d}: {medians[(dataset, d)]!r}", flush=True)
        del x, y
        torch.cuda.empty_cache()
    print("\nPaste into configs/speedup.py:\n")
    print("MEDIAN_C: Dict[Tuple[str, int], Optional[float]] = {")
    for (dataset, d), value in medians.items():
        print(f"    ({dataset!r}, {d}): {value!r},")
    print("}\n", flush=True)
    return medians


def step_exact_ot(device: torch.device, seeds, workers: int) -> None:
    print(f"== exact-OT reference cache, seeds {seeds}, {workers} workers ==", flush=True)
    jobs = []
    for dataset, d in SLICES:
        for seed in seeds:
            x, y, a, b = (t.cpu() for t in sample_instance(dataset, d, seed, device))
            jobs.append((dataset, d, seed, x, y, a, b))

    def solve(job):
        dataset, d, seed, x, y, a, b = job
        t0 = time.perf_counter()
        ref = _cached_exact_ot_reference(N, N, d, seed, x, y, a, b, dataset=dataset)
        print(f"  {dataset} d={d} seed={seed}: cost={ref.cost:.6g} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(solve, jobs))


def step_warm(out_dir: str, medians=None) -> None:
    """Run each GPU method for a few iterations at every d, at N=10,000."""
    import run

    print("== warm Triton/KeOps caches ==", flush=True)
    if all(v is not None for v in speedup.MEDIAN_C.values()):
        cfg = speedup.CONFIG
    elif medians is not None:
        cfg = speedup.build_config(medians)
    else:
        sys.exit("MEDIAN_C is unset in configs/speedup.py; run without --skip-median.")
    one_per_d = {}
    for dataset, d, eps_values in cfg.problems:
        one_per_d.setdefault(d, (dataset, d, [eps_values[len(eps_values) // 2]]))
    warm_cfg = dataclasses.replace(
        cfg, problems=list(one_per_d.values()), seeds=[0],
        n_iters=10, max_iter=10, warmup=0, warmup_iters=1, rep=1,
        no_rmae_check=True, no_srot=True, no_sparsink=True,
        sinkslotcuda_slices=cfg.sinkslotcuda_slices[:1], output_dir=out_dir,
    )
    units = list(run._units(warm_cfg))
    failures = 0
    for i, unit in enumerate(units, 1):
        cmd = run.build_command(
            warm_cfg, unit.dataset, unit.eps, out_dir, method=unit.method, size=unit.n,
            dim=unit.d, slices=unit.slices, seed=unit.seed, tf32=unit.tf32)
        print(f"  [{i}/{len(units)}] {run._unit_tag(unit)}", flush=True)
        t0 = time.perf_counter()
        rc = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.DEVNULL).returncode
        print(f"    rc={rc} ({time.perf_counter() - t0:.0f}s)", flush=True)
        failures += rc != 0
    if failures:
        print(f"  {failures}/{len(units)} warm-up runs failed", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-median", action="store_true", help="Skip step 1.")
    ap.add_argument("--skip-exact-ot", action="store_true", help="Skip step 2.")
    ap.add_argument("--skip-warm", action="store_true", help="Skip step 3.")
    ap.add_argument("--seeds", default="0,1,2,3,4", help="Seeds for step 2 (default: 0,1,2,3,4).")
    ap.add_argument("--workers", type=int, default=4, help="Concurrent ot.emd solves in step 2.")
    ap.add_argument("--warm-dir", default="output/speedup_potential_warmup",
                    help="Scratch output directory for step 3's CSVs.")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("speedup_prepare.py needs a CUDA GPU (bench_forward samples on cuda).")
    device = torch.device("cuda")

    medians = None if args.skip_median else step_median(device)
    if not args.skip_exact_ot:
        step_exact_ot(device, [int(s) for s in args.seeds.split(",")], args.workers)
    if not args.skip_warm:
        step_warm(args.warm_dir, medians)


if __name__ == "__main__":
    main()
