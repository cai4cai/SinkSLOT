"""Gradient flow experiment parameters: blob -> crescent, N = M = 1000, one run per eps in EPS_VALUES."""

N = 1000
STEPS = [0, 5, 10, 20, 30, 50]
N_STEPS = 50
LR = 0.05
EPS_VALUES = [0.1, 0.01]
MAX_ITER = 1000  # Sinkhorn iterations per gradient step (EOT, SROT, SinkSLOT)
L = 100  # slice count shared by SOT, SROT and SinkSLOT
DELTA_SROT = 1e-8
DATA_SCALE = 2.0

METHOD_NAMES = ["SOT", "EOT", "SROT", "SinkSLOT"]
ROW_LABELS = {
    "SOT": "SOT\n(Bonneel et al. 2015)",
    "EOT": "EOT\n(Feydy et al. 2019)",
    "SROT": "SROT\n(Nguyen 2026)",
    "SinkSLOT": "SinkSLOT\n(ours)",
}
