"""Gradient flow experiment parameters: blob -> crescent, N=1000, eps=0.01.

The single fixed-parameter run this repo needs (no sweep), so a plain module of
constants rather than a configs/-style BenchConfig -- see configs/base.py for
why that dataclass exists (it sweeps datasets/eps/methods for run.py; this
experiment doesn't sweep anything).
"""

N = 1000
STEPS = [0, 5, 10, 20, 30, 50]
N_STEPS = 50
LR = 0.05
EPS_VALUES = [0.01]
MAX_ITER = 1000  # gradient steps per method, and SinkSLOT-arm inner Sinkhorn iterations
L = 100  # random projections (SOT, SROT, SinkSLOT all share this slice count)
DELTA_SROT = 1e-8
DATA_SCALE = 2.0

METHOD_NAMES = ["SOT", "EOT", "SROT", "SinkSLOT"]
ROW_LABELS = {
    "SOT": "SOT\n(Bonneel et al. 2015)",
    "EOT": "EOT\n(Feydy et al. 2019)",
    "SROT": "SROT\n(Nguyen 2026)",
    "SinkSLOT": "SinkSLOT\n(ours)",
}
