# Memory

**Status: audit closed.** This started as a worry that the sweep's `gpu_memory_mb`
column understates FlashSinkhorn (its `TOTAL` here, which includes the transient
Triton-autotune allocator pool, is far above its `device` reading). It doesn't:
`gpu_memory_used_mb()` (`torch-ext/flash_sinkhorn/bench/bench_forward.py`) calls
`empty_cache()` before reading specifically to strip that ~270MB of pooled-but-unused
autotune scratch, and says so in its own docstring — the sweep's number was always
meant to be steady-state footprint, not instantaneous peak, and that choice is applied
identically to every method (one shared function, `isolate=True` giving each
measurement its own subprocess). So `TOTAL` and `device` disagreeing here isn't a bug
in the sweep; they're deliberately different quantities. The published
`tab:accuracy_memory` numbers stand. Kept for the operator-level attribution and the
CUDA-path memory analysis below, which are still accurate and may be useful again, but
not maintained against future changes to `sinkslot`'s CUDA setup path.

The sweep's `gpu_memory_mb` is `total - free` from `cudaMemGetInfo` — whole-device. It bundles four things into one number, and it is read after the run so it misses transients. `scripts/memory_audit/memory.py` splits it and samples the true peak.

n=4096, d=64, L=512, RTX A1000, all MB:

**Autotune on** (`python scripts/memory_audit/memory.py --n 4096 --d 64 --slices 512`):

```
method               device  context  modules reserved tens:setup tens:solve  TENSORS     warm     TOTAL
flash_symmetric       112.5    106.2      2.1      4.2        2.1        3.2      3.2    112.5     381.0
flash_alternating     112.5    106.2      2.1      4.2        2.1        3.2      3.2    112.5     381.0
srot                  410.3    106.2      0.0    306.2     1122.7      291.8   1122.7   1253.4    1253.4
sinkslot              209.0    106.2     10.5     92.3     2934.8       54.6   2934.8   3428.1    3428.1
sinkslot_cuda         209.0    106.2     10.5     92.3      490.9       54.6    490.9    666.2     666.2
spar_sink             209.0    106.2     10.5     92.3      363.0       77.9    363.0    494.2     494.2
rand_sink             209.0    106.2     10.5     92.3      363.0       77.9    363.0    494.2     494.2
geomloss_online       112.5    106.2      2.1      4.2        2.2        2.3      2.3    112.5     112.5
```

**Autotune off** (same, plus `--no-autotune`):

```
method               device  context  modules reserved tens:setup tens:solve  TENSORS     warm     TOTAL
flash_symmetric       110.4    106.2      0.0      4.2        2.1        3.2      3.2    110.4     110.4
flash_alternating     110.4    106.2      0.0      4.2        2.1        3.2      3.2    110.4     110.4
srot                  410.3    106.2      0.0    306.2     1122.7      291.8   1122.7   1253.4    1253.4
sinkslot              209.0    106.2     10.5     92.3     2934.8       54.6   2934.8   3428.1    3428.1
sinkslot_cuda         209.0    106.2     10.5     92.3      490.9       54.6    490.9    666.2     666.2
spar_sink             209.0    106.2     10.5     92.3      363.0       77.9    363.0    494.2     494.2
rand_sink             209.0    106.2     10.5     92.3      363.0       77.9    363.0    494.2     494.2
geomloss_online       112.5    106.2      2.1      4.2        2.2        2.3      2.3    112.5     112.5
```

Only the Flash rows move: `--no-autotune` is wired into the Flash probe alone, since the other methods' Triton kernels use fixed launch configs and have nothing to tune. Flash's `TOTAL` drops **381.0 → 110.4 MB (3.5x)** while `TENSORS` is unchanged at 3.2 — the entire difference is compilation transient, not the algorithm. Everything else is byte-identical across the two tables, which is the control that says the change is real and not run-to-run noise.

Validated against `nvidia-smi` itself, polled at 20 ms during the Flash run: **364 MiB = 381.7 MB** with autotune against the script's 381.0 MB (0.2% agreement), and **106 MiB** without.

- `context` — CUDA context. **Identical for every method.** A per-GPU constant.
- `modules` — compiled Triton/CUDA modules, cuBLAS workspaces, KeOps.
- `reserved` — caching-allocator pool.
- `TENSORS` — `max_memory_allocated()`. **PyTorch tensor bytes only — not a total.** Excludes context and modules, so it is always well below what the card must supply. This is the O(Nd)-vs-O(NM) quantity and the counterpart to FlashSinkhorn Figure 3.
- `warm` — whole-device high-water over setup+solve on an already-compiled process, **sampled from the driver every 2 ms**. What a warm/persistent server needs per call.
- `TOTAL` — the same, over the **whole process lifetime**, so JIT compilation and Triton autotuning are inside it. What nvidia-smi actually peaks at, and what a cold process needs. Measured, not summed.

So Flash's `TENSORS` of 3.2 against a `TOTAL` of 381.0 is not a contradiction. Three different questions: 3.2 MB is all the transport problem itself needs, 112.5 MB is that plus the CUDA context the method never asked for, and 381.0 MB is what the card must actually have free because Triton autotuning transiently allocates 258 MiB while compiling.

`TOTAL` is also not `device`. `device` is read *after* the run, so every transient at the high-water mark has already been freed out of it — SinkSLOT reads 209 MB of `device` having genuinely peaked at 3,428 MB. `device` answers "what is still held"; `TOTAL` answers "will this fit".

`TOTAL` is a sampler, so it is a lower bound: a spike shorter than the polling interval is missed. The script cross-checks every row against `TENSORS + context` (which the allocator proves was live) and flags any row where the sampler was too coarse; none of the rows above are flagged. Lower `--sample-interval` if a row ever is.

**Caveat on the geomloss row.** KeOps allocates outside the PyTorch allocator, so its LSE workspace is invisible to `max_memory_allocated` and lands in `modules` instead. Its `TENSORS` is therefore an understatement; `TOTAL` still catches it, since that is driver-level. Everything else is pure PyTorch and fully accounted for in both. The probe drives the low-level `sinkhorn_loop` with `eps_list=[eps]*iters`, matching `bench_geomloss_online`; the high-level `SamplesLoss` would turn ε-scaling on and run a different iteration count.

## Three consequences

**1. Report `TENSORS` and `TOTAL`, not `device`.** `device` is a post-hoc residual: it misses transients entirely and is otherwise dominated by a fixed floor unrelated to the transport problem. `TENSORS` is the algorithmic quantity; `TOTAL` is the deployment one. Flash's 112.5 MB `TOTAL` reproduces FlashSinkhorn Figure 3's "barely above 100 MB" at (10k, 1024).

**2. The current table has the ordering backwards.** exp1 reports Flash at 1,786.8 MiB against SinkSLOT's 782.3. On `TENSORS` it is Flash 3.2 against SinkSLOT-CUDA's 490.9, and on `TOTAL` 381.0 against 666.2. **We do not have a memory advantage over Flash and should not claim one.** Our advantage is O(L(N+M)) support vs dense O(NM) — a different axis, and one that only holds against SROT, not against Flash. See below.

**3. Triton autotuning is the prime suspect for the H100 spread.** With autotune on, Flash's `TOTAL` is 381.0 MB against a 112.5 MB `warm`; with `--no-autotune` the nvidia-smi peak collapses from 364 MiB to 106 MiB. So 258 MiB of the A1000 figure is compilation transient, and an H100 admits more autotune configs across far more SMs. This supersedes an earlier reading of mine that dismissed autotune because `--no-autotune` only moved `modules` from 2.1 to 0.0 — that is the residual, which is not where autotuning shows up. Run `scripts/memory_audit/memory.py` on Jean Zay and compare `warm` against `TOTAL` to confirm.

## Why SinkSLOT-CUDA is heavier than Flash

Not the naive-path gather — this is the optimised path, and it is fundamental. SinkSLOT stores an explicit sparse kernel; Flash stores none, recomputing cost from coordinates inside the kernel. Persistent solve state at n=m=4096, d=64, L=512:

```
SinkSLOT-CUDA (CSR + CSC)          FlashSinkhorn
  r_idx  int16      7.3 MB           x  fp32   1.0 MB
  r_lam  fp32      14.6 MB           y  fp32   1.0 MB
  c_idx  int16      7.3 MB           a,b       0.0 MB
  c_lam  fp32      14.6 MB
  TOTAL           43.9 MB           TOTAL      2.1 MB     -> 21x
```

nnz = 3,655,051 (87% of L(N+M) after coalescing) at 12 bytes/nnz — indices and values kept twice, row-major for the CSR half-step and column-major for the CSC one. **SinkSLOT is O(L(N+M)) and independent of d; Flash is O(Nd) and independent of L.** Measured persistent MB: L=64 -> 6.2, L=512 -> 43.9, L=2048 -> 122.3; and d=3 -> 40.8 against d=64 -> 43.9 at fixed L. The ratio is roughly 2.6·L/d, so SinkSLOT is heavier whenever L/d is above ~0.4 — nearly the whole grid. Accuracy needs L to grow and Flash's memory does not grow with L at all.

The 12 bytes/nnz is not reducible by engineering: storing values once plus a `perm` is byte-neutral, and `lam` cannot be narrowed (`to_csr` explains why — fp16 puts ~5e-3 absolute error on a log-domain value of order 10, against a 1e-6 marginal threshold).

**It worsens at N=1e5.** `to_csr` picks int16 only when `idx.max() < 2**15`, so above 32,768 points the indices silently widen to int32 and 12 -> 16 bytes/nnz. At N=1e5, L=9k that puts persistent state near 25 GB before any transient, against ~51 MB for Flash. Chunking (below) controls the transient but cannot touch that floor; the levers there are capping L or pruning the support, both method changes.

## Open levers on the CUDA path

The build transient is 10–14x the persistent state — `sot_coo` materialises all L(N+M) raw entries before coalescing. `chunk` (already a parameter, defaulted to one-shot) caps it. Measured on the full setup path, prototyped and then reverted since the campaign ran without it:

```
   L    chunk   narrow_idx   setup ms   peak MB
 512     None        False       80.2      447.0
 512     None         True       64.6      447.0
 512      256         True       62.6      279.2     1.6x
2048     None        False      273.0     1756.0
2048     None         True      247.8     1756.0
2048      256         True      253.3      565.5     3.1x
```

Verified safe: identical support set, identical mass, `max |dS| = 0`. The gain scales with L — 1.6x at L=512 against 3.1x at L=2048 — and it is roughly time-neutral, not the ~8% penalty a narrower measurement suggested.

Narrowing the returned `rows`/`cols` from int64 to int32 (both fit whenever n, m < 2^31; the flat key `row*m+col` must stay int64) buys **no peak memory at all** — 447.0 and 1756.0 are unchanged with and without. The 8 bytes/nnz is real but those intermediates are not live at the high-water mark, which sits inside the build before they exist. What it does buy is speed: 80.2 -> 64.6 ms at L=512, since `to_csr` then sorts and gathers half-width arrays.

Neither is applied. End to end on the table, both together moved SinkSLOT-CUDA's `tens:setup` 490.9 -> 324.1 and `TOTAL` 666.2 -> 536.2 at L=512, leaving `tens:solve` and every other method untouched. Neither touches the 12 bytes/nnz persistent floor or the O(L(N+M)) vs O(Nd) gap against Flash.

## What the naive-vs-CUDA gap is made of

`sinkslot`'s 2,935 MB against `sinkslot_cuda`'s 491 MB is not a defect — the non-CUDA path is the untouched naive baseline (483aa68), and this gap is what the CUDA setup path exists to close. Recording it because it is the memory half of the same speedup, which the sweep only ever reported as time.

`scripts/memory_audit/memory_profile.py` attributes it operator by operator. Top of the SETUP phase, n=4096, d=64, L=512:

```
sinkslot                            sinkslot_cuda
aten::index    1843.82 MB   (x10)   aten::index     173.76 MB   (x8)
aten::sub       967.09 MB   (x3)    aten::sub        31.40 MB   (x2)
aten::pow       935.69 MB   (x1)    aten::pow          --
```

Those are the three intermediates of the baseline's `cost = (x[rows] - y[cols]).square().sum(1)` — gather, subtract, square — live at once, O(nnz*d) each. The CUDA path's blocked `sparse_sqeuclidean_cost` never materialises them and its `aten::pow` disappears entirely: 2,807 MB of transient becomes 14.7 MB.

## Reproduce

```
python scripts/memory_audit/memory.py                                  # defaults above
python scripts/memory_audit/memory.py --n 100000 --d 64 --slices 2048
python scripts/memory_audit/memory.py --no-autotune                    # second table above
```

One subprocess per method: context and compiled modules are never released within a process, so methods sharing one inherit each other's floor.

For operator-level attribution rather than aggregates, `scripts/memory_audit/memory_profile.py` wraps the same probes in `torch.profiler` and writes `torch.cuda.memory._dump_snapshot` pickles you can load at https://pytorch.org/memory_viz :

```
python scripts/memory_audit/memory_profile.py --method sinkslot --n 4096 --d 64 --slices 512 --output-dir out/prof
```
