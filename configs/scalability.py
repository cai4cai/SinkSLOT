"""Scalability: 3 N-scaling/d-scaling experiments, 5 seeds each, SinkSLOT-CUDA
vs SROT vs FlashSinkhorn-alternating. Same marginal stopping policy as
configs/speedup.py: stop_tol=1e-6, max_iter=20000.

Spar-Sink data was also collected for this experiment (same s = k*s0(n)
recipe as configs/speedup.py, run over 5 seeds) but is not part of the final
scalability figure: it fails to reach N=50,000 in experiments 1/2 (its
sampling step builds a dense N x M mask, which exceeds torch.nonzero()'s
int32 index limit at N=M=50,000 -- see build_sparse_kernel's docstring in
bench_forward.py) and was dropped from the plotted comparison as a result.
Spar-Sink is intentionally excluded from METHODS below; re-add it there if
it's ever needed again.

Not expressed as a plain BenchConfig / run.py sweep: SinkSLOT-CUDA and SROT
both need L swept independently of N/d (a per-method axis run.py's
single-BenchConfig sweep can't express alongside the N-scaling/d-scaling
axis), and the whole experiment is repeated over 5 seeds with the commands
routed to per-seed output directories (save_results_csv's merge key omits
seed, so seeds sharing one output directory silently overwrite each other --
learned the hard way once this session). scripts/scalability.py builds the
actual per-unit commands directly (mirroring run.py's build_command() flag
ordering) rather than going through run.py's dry-run path; run it with:

    python scripts/scalability.py

Experiment 1 -- N-scaling, Gaussian d=3, eps=0.01, N in {5k,10k,20k,30k,50k}.
Experiment 2 -- N-scaling, Gaussian d=64, eps=0.1, same N sweep as experiment 1.
Experiment 3 -- d-scaling, Gaussian, N=10,000 fixed, eps=0.1, d in
  {4,8,16,32,64,128,256,512,1024}.

SinkSLOT-CUDA and SROT are swept over the same 3 L values in every
experiment. Experiment 3 (d-scaling) additionally needs --no-rmae-check on
SinkSLOT-CUDA: at large d/L its cost_gap_pct/barycentric_sym diagnostics
build a dense exact-OT reference that OOMs (confirmed at d>=256, L=4096);
Flash-alternating and SROT never hit this and keep full diagnostics.
"""

import math

SEEDS = [0, 1, 2, 3, 4]

L_VALUES = [128, 1024, 4096]

N_SWEEP = [5000, 10000, 20000, 30000, 50000]  # experiments 1 and 2
D_SWEEP = [4, 8, 16, 32, 64, 128, 256, 512, 1024]  # experiment 3

EPS_EXP1 = 0.01
EPS_EXP2 = 0.1
EPS_EXP3 = 0.1
D_EXP1 = 3
D_EXP2 = 64
N_EXP3 = 10000

METHODS = ["sinkslotcuda", "flash_alternating", "srot"]

# shared solver policy, same as configs/speedup.py
STOP_MODE = "marginal"
MAX_ITER = 20000
STOP_TOL = 1e-6
POTENTIAL_TOL = 1e-6
CHECK_EVERY = 10
SROT_DELTA = 1e-8

# Both SROT and Spar-Sink build a dense N x M intermediate (SROT's own pi_SOT
# reference plan; Spar-Sink's sampling mask), gated by bench_forward.py's
# --max-dense-size (default 10,000). N-scaling needs this raised to cover the
# full N sweep; d-scaling's N is fixed at 10,000, so the default is enough.
MAX_DENSE_SIZE_NSCALE = 50000
MAX_DENSE_SIZE_DSCALE = 10000


def s_for(n: int) -> list[int]:
    """Spar-Sink sample sizes for a given N, matching configs/speedup.py's
    formula: s = k * s0(n), s0(n) = 1e-3 * n * log(n)^4, k in {5, 10, 15, 20}.
    Kept here (unused by METHODS above) only so anyone re-adding Spar-Sink
    to this experiment starts from the same recipe as configs/speedup.py,
    not the retired density-based grid this file used to carry.
    """
    s0 = 1e-3 * n * (math.log(n) ** 4)
    return [int(round(k * s0)) for k in [5, 10, 15, 20]]
