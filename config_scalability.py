"""Figure2: 3 N-scaling/d-scaling experiments, all methods except SROT and
Rand-Sink (SinkSLOT-CUDA, Flash-alternating, Spar-Sink). Rand-Sink was dropped
-- no longer of interest. Same marginal stopping policy as config_speedup.py:
stop_tol=1e-6, max_iter=20000.

Not expressed as a plain BenchConfig / run.py sweep: Spar-Sink's S here is
DENSITY-based (density * N * M), so the absolute S value must be recomputed
for every N in the N-scaling experiments (1 and 2) -- something a single
static BenchConfig.sparsink_s list can't express, since it's one fixed list
applied uniformly across every N in cfg.sizes. These constants are the single
source of truth; the actual per-unit commands are built directly by
nscale_dscale_launch/gen_units.py (mirroring run.py's build_command() flag
ordering), not through run.py's dry-run path.

Experiment 1 -- N-scaling, 4 small-d datasets (half_moon d=2, 8gaussians d=2,
two_rings d=2, gaussian d=3), eps=0.01 fixed, N in {5k,10k,20k,30k,50k}.

Experiment 2 -- N-scaling, Gaussian d=64, eps=0.1 fixed, same N sweep as
experiment 1.

Experiment 3 -- d-scaling, Gaussian, N=10,000 fixed, eps=0.1 fixed, d in the
9-point grid {4,8,16,32,64,128,256,512,1024} (matches config_expB_dscale_marginal.py's
d grid).

Every experiment uses the same L_VALUES for SinkSLOT-CUDA and the same
S_DENSITIES for Spar-Sink (3 points each, not the 8-point grids used in
config_speedup.py -- these are smaller, exploratory sweeps).

Known cost risk: at N=50,000 (the top of the N sweep in experiments 1 and 2),
the largest density (5%) gives S=125,000,000 -- over 10x the largest S value
actually measured this session (S=10M took ~27min/unit under the old,
absolute-S table1 grid). Those specific units may take a very long time; see
nscale_dscale_launch/gen_scripts.py for the dev/t3 QOS split that routes
Spar-Sink's N-scaling jobs (experiments 1 and 2) to the 20h t3 queue for
exactly this reason, while everything else runs on the 2h dev queue.
"""

N = 10000  # fixed N for experiment 3; M = N throughout (source/target same size)

L_VALUES = [128, 1024, 4096]
S_DENSITIES = [0.001, 0.007071, 0.05]  # 3-point log grid, 0.1%..5% of N*M

N_SWEEP = [5000, 10000, 20000, 30000, 50000]  # experiments 1 and 2
D_SWEEP = [4, 8, 16, 32, 64, 128, 256, 512, 1024]  # experiment 3

DATASETS_SMALL_D = [("half_moon", 2), ("8gaussians", 2), ("two_rings", 2), ("gaussian", 3)]

EPS_EXP1 = 0.01
EPS_EXP2 = 0.1
EPS_EXP3 = 0.1

METHODS = ["sinkslotcuda", "flash_alternating", "spar_sink"]

# shared solver policy, same as config_speedup.py
STOP_MODE = "marginal"
MAX_ITER = 20000
STOP_TOL = 1e-6
POTENTIAL_TOL = 1e-6
MASS_TOL = 1e-6
CHECK_EVERY = 10
SPARSINK_REPLICATES = 50


def s_for(n: int) -> list[int]:
    """Absolute Spar-Sink sample sizes for a given N, holding density
    (S_DENSITIES) constant -- i.e. S scales up as N grows."""
    nm = n * n
    return [int(round(dens * nm)) for dens in S_DENSITIES]
