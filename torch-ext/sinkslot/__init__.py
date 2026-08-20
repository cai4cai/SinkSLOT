"""SinkSLOT: fused-Triton sparse Sinkhorn on the unsmoothed sliced OT plan.

This is the paper's own contribution -- a distinct algorithm from FlashSinkhorn
(sparse, built on a sliced-OT reference plan, rather than FlashSinkhorn's dense
fused kernel over the full cost matrix). It lives in its own package rather
than nested inside flash_sinkhorn/bench/ because it's a real solver used
outside benchmarking too (see slot_grad, used by gradient_flow/), not
benchmark-only code.

Layout mirrors flash_sinkhorn's own {sinkhorn_solvers,implicit_grad,hvp,
samples_loss}.py split: solver.py holds the sliced-OT support construction
with no FlashSinkhorn equivalent; sinkhorn_solvers.py the Sinkhorn iteration
loops and device-agnostic entry point; gradient.py the envelope-theorem
gradient (see that file's own docstring for why it isn't named
implicit_grad.py); hvp.py the Hessian-vector product; samples_loss.py the
GeomLoss-style `SamplesLoss` callable -- unlike flash_sinkhorn's own, with no
`_autograd.py`-style Function family behind it, just one
`torch.autograd.Function` reusing `slot_grad`'s formula.

Package name: sinkslot
"""

# sinkslot_alternating_triton/_torch are deliberately not exported here:
# they're the low-level solve loops sinkslot_solve already wraps and
# dispatches between, not something a caller should reach for directly.
# `sinkslot.sinkhorn_solvers.sinkslot_alternating_triton` still works for
# anyone who genuinely needs it (the benchmark harness and
# gradient_flow/appendix_checks/stopping.py do, since they need the raw
# CSR/CSC-based loop, not the whole build-plan-then-solve pipeline
# sinkslot_solve wraps it in),
# it just isn't advertised as the public API. Matches flash_sinkhorn's own
# convention: its sinkhorn_flashstyle_alternating/_symmetric aren't
# underscore-prefixed either, but aren't re-exported from
# flash_sinkhorn/__init__.py -- a descriptive name plus staying out of
# __init__.py's import list is how both packages mark "internal, but not
# hidden behind Python's underscore convention."
from .solver import (
    get_random_projections,
    sot_plan_coo,
    to_csr,
    sparse_sqeuclidean_cost,
)
from .sinkhorn_solvers import (
    seg_lse_online,
    launch_cfg,
    sinkslot_solve,
    sparse_transport_plan,
)
from .gradient import (
    slot_grad,
    plan_barycentric_sparse,
)
from .hvp import (
    hvp_x_sqeuclid,
    hvp_x_sqeuclid_from_potentials,
)
from .samples_loss import SamplesLoss

__all__ = [
    "get_random_projections",
    "sot_plan_coo",
    "to_csr",
    "sparse_sqeuclidean_cost",
    "seg_lse_online",
    "launch_cfg",
    "sinkslot_solve",
    "sparse_transport_plan",
    "slot_grad",
    "plan_barycentric_sparse",
    "hvp_x_sqeuclid",
    "hvp_x_sqeuclid_from_potentials",
    "SamplesLoss",
]
