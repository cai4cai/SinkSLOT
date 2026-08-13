"""SinkSLOT: fused-Triton sparse Sinkhorn on the unsmoothed sliced OT plan.

This is the paper's own contribution -- a distinct algorithm from FlashSinkhorn
(sparse, built on a sliced-OT reference plan, rather than FlashSinkhorn's dense
fused kernel over the full cost matrix). It lives in its own package rather
than nested inside flash_sinkhorn/bench/ because it's a real solver used
outside benchmarking too (see gradient_flow/solver.py), not benchmark-only code.

Package name: sinkslot
"""

from .solver import (
    sot_directions,
    sot_plan_coo,
    to_csr,
    sparse_sqeuclidean_cost,
    seg_lse_online,
    launch_cfg,
    slot_hvp,
    _run_v5,
)

__all__ = [
    "sot_directions",
    "sot_plan_coo",
    "to_csr",
    "sparse_sqeuclidean_cost",
    "seg_lse_online",
    "launch_cfg",
    "slot_hvp",
    "_run_v5",
]
