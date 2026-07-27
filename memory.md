# Memory

The sweep's `gpu_memory_mb` is `total - free` from `cudaMemGetInfo` — whole-device. It bundles four things into one number, and it is read after the run so it misses transients. `scripts/memory.py` splits it and samples the true peak.

n=4096, d=64, L=512, RTX A1000, all MB:

**Autotune on** (`python scripts/memory.py --n 4096 --d 64 --slices 512`):

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

**2. The current table has the ordering backwards.** exp1 reports Flash at 1,786.8 MiB against SinkSLOT's 782.3. On `TENSORS` it is Flash 3.2 against SinkSLOT 490–2,935, and on `TOTAL` 381.0 against 666–3,428. **We do not have a memory advantage over Flash and should not claim one.** Our advantage is O(L(N+M)) support vs dense O(NM) — a different axis.

**3. Triton autotuning is the prime suspect for the H100 spread.** With autotune on, Flash's `TOTAL` is 381.0 MB against a 112.5 MB `warm`; with `--no-autotune` the nvidia-smi peak collapses from 364 MiB to 106 MiB. So 258 MiB of the A1000 figure is compilation transient, and an H100 admits more autotune configs across far more SMs. This supersedes an earlier reading of mine that dismissed autotune because `--no-autotune` only moved `modules` from 2.1 to 0.0 — that is the residual, which is not where autotuning shows up. Run `scripts/memory.py` on Jean Zay and compare `warm` against `TOTAL` to confirm.

## Open bug

`bench_sinkslot` builds the sparse cost by gathering (`bench_forward.py:1512`):

```python
cost = (x[rows] - y[cols]).square().sum(1)   # 2807 MB = 3 x nnz*d*4B
```

`bench_sinkslotcuda` already uses the blocked `sparse_sqeuclidean_cost` — **14.7 MB**, 190x less. Scales as nnz*d, so it is what OOMs at N=1e5 / L~9k, and it will read as SinkSLOT failing to scale. Not yet fixed.

`scripts/memory_profile.py` confirms this operator by operator. Top of the SETUP phase, n=4096, d=64, L=512:

```
sinkslot                            sinkslot_cuda
aten::index    1843.82 MB   (x10)   aten::index     173.76 MB   (x8)
aten::sub       967.09 MB   (x3)    aten::sub        31.40 MB   (x2)
aten::pow       935.69 MB   (x1)    aten::pow          --
```

Those are exactly the three intermediates of `(x[rows] - y[cols]).square()` — the gather, the subtraction, the squaring — live simultaneously. The CUDA path's blocked kernel never materialises them, and its `aten::pow` disappears entirely.

## Reproduce

```
python scripts/memory.py                                  # defaults above
python scripts/memory.py --n 100000 --d 64 --slices 2048
python scripts/memory.py --no-autotune                    # second table above
```

One subprocess per method: context and compiled modules are never released within a process, so methods sharing one inherit each other's floor.

For operator-level attribution rather than aggregates, `scripts/memory_profile.py` wraps the same probes in `torch.profiler` and writes `torch.cuda.memory._dump_snapshot` pickles you can load at https://pytorch.org/memory_viz :

```
python scripts/memory_profile.py --method sinkslot --n 4096 --d 64 --slices 512 --output-dir out/prof
```
