"""SinkSLOT: fused-Triton sparse Sinkhorn on the unsmoothed sliced OT plan.

This is the paper's own contribution -- a distinct algorithm from FlashSinkhorn
(sparse, built on a sliced-OT reference plan, rather than FlashSinkhorn's dense
fused kernel over the full cost matrix). It lives in its own package rather
than nested inside flash_sinkhorn/bench/ because it's a real solver used
outside benchmarking too (see slot_grad, used by gradient_flow/), not
benchmark-only code.

Package name: sinkslot
"""

# _run_v5 and _run_v5_torch are deliberately not exported here (fixes #14):
# they're the underscore-prefixed internal solve loops sinkslot_solve already
# wraps and dispatches between, not something a caller should reach for
# directly. Re-exporting an internal name at the package's top level was the
# actual bug -- `sinkslot.solver._run_v5` still works for anyone who genuinely
# needs it (the benchmark harness and gradient_flow/stopping.py do, since they
# need the raw CSR/CSC-based loop, not the whole build-plan-then-solve
# pipeline sinkslot_solve wraps it in), it just isn't advertised as the public
# API.
from .solver import (
    sot_directions,
    sot_plan_coo,
    to_csr,
    sparse_sqeuclidean_cost,
    seg_lse_online,
    launch_cfg,
    sinkslot_solve,
    slot_grad,
    plan_barycentric_sparse,
)

__all__ = [
    "sot_directions",
    "sot_plan_coo",
    "to_csr",
    "sparse_sqeuclidean_cost",
    "seg_lse_online",
    "launch_cfg",
    "sinkslot_solve",
    "slot_grad",
    "plan_barycentric_sparse",
]
