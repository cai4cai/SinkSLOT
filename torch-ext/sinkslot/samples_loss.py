"""GeomLoss-compatible SamplesLoss for SinkSLOT.

    from sinkslot import SamplesLoss
    loss = SamplesLoss(eps=0.05, L=64, n_iters=200)
    L = loss(x, y)
    g_x, = torch.autograd.grad(L, [x])

Mirrors geomloss.SamplesLoss / flash_sinkhorn.SamplesLoss's calling convention
(construct once with hyperparameters, call as `loss(x, y)`) without either's
epsilon-scheduling, debiasing, unbalanced-OT, or double-backward/HVP support
-- `eps`, `L` (slice count), and `n_iters` are SinkSLOT's own primary knobs,
not a `blur`/scaling schedule, and there is no `_autograd.py`-style Function
family here: a single `torch.autograd.Function` below reuses the already
-solved plan from forward directly in backward via `slot_grad`'s own
envelope-theorem formula (Feydy et al. 2019), so there's no second Sinkhorn
solve and no unrolled backprop through the loop -- see
`gradient_flow/estimators.py`'s own comparison for why unrolling loses.

Only a single weight vector `a`, shared by both marginals, is supported --
not separate `a`/`b` the way `sinkslot_solve` itself allows -- because
`slot_grad`'s envelope-theorem formula (which this reuses for backward) is
only implemented and validated for that uniform-a case, matching every
existing caller in this repo (see `slot_grad`'s own docstring). Extending it
to independent `a`/`b` would need a fresh derivation, not attempted here.
"""

from __future__ import annotations

import torch

from .solver import sparse_sqeuclidean_cost
from .sinkhorn_solvers import sinkslot_solve
from .gradient import plan_barycentric_sparse


class _SLOTCostFn(torch.autograd.Function):
    """Forward: solve + the achieved entropic cost <T, C>. Backward: the
    envelope-theorem gradient, reusing forward's already-solved plan (no
    second solve) -- this IS `slot_grad`'s own formula, inlined so it can
    share `T_vals`/`rows`/`cols` with forward instead of recomputing them.

    The envelope theorem holds at any (approximate) fixed point of the
    Sinkhorn iteration regardless of which scheme reached it -- it's a
    property of the converged potentials, not of the alternating vs.
    symmetric update rule -- so `backward` needs no `variant`/`alpha`
    awareness at all; only `forward`'s call into `sinkslot_solve` does.
    """

    @staticmethod
    def forward(ctx, X, Y, a, eps, L, seed, n_iters, backend, variant, alpha):
        phi, psi, rows, cols, S, it, converged, viol = sinkslot_solve(
            X, Y, a, a, eps, L, seed, n_iters, backend=backend,
            variant=variant, alpha=alpha)
        cost = sparse_sqeuclidean_cost(
            X, Y, rows, cols,
            use_triton=(backend == "triton") if backend != "auto" else None,
        )
        log_S = S.clamp_min(torch.finfo(S.dtype).tiny).log()
        lam = log_S - cost / eps
        T_vals = (phi[rows] + psi[cols] + lam).exp()
        loss = (T_vals * cost).sum()
        ctx.save_for_backward(X, Y, a, T_vals, rows, cols)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        X, Y, a, T_vals, rows, cols = ctx.saved_tensors
        Tx, _ = plan_barycentric_sparse(T_vals, rows, cols, X, Y)
        grad_x = 2.0 * a[:, None] * (X - Tx)
        return (grad_output * grad_x,
                None, None, None, None, None, None, None, None, None)


class SamplesLoss(torch.nn.Module):
    """SLOT_eps(x, y) as a GeomLoss-style callable loss module.

    `loss`: only "sinkhorn" is accepted (kept as a constructor argument for
    call-site parity with geomloss/flash_sinkhorn, which support several
    other loss families this package doesn't implement at all).

    `eps`: entropic regularisation (SinkSLOT's own `eps`, not GeomLoss's
    `blur` -- there is no `blur**p = eps` conversion here; pass `eps`
    directly). `L`: number of slicing directions. `seed`: `expected_sliced_plan`'s
    projection RNG seed. `n_iters`: fixed Sinkhorn iteration count (no
    early-stopping `stop` config exposed here; use `sinkslot_solve` directly
    for that). `backend`: "auto" / "triton" / "torch", see `sinkslot_solve`.

    `symmetric` (#34): False (default) uses the alternating (Gauss-Seidel)
    solve loop, same as every other SinkSLOT entry point so far. True uses
    the symmetric (Jacobi) loop instead (`sinkslot_solve`'s `variant=
    "symmetric"`) -- see `sinkhorn_solvers.sinkslot_symmetric_triton`'s own
    docstring for the update rule. `alpha` is that update's blend weight
    (`f_new = (1-alpha)*f_old + alpha*f_cand`); unused when
    `symmetric=False`.

    Forward returns the achieved SLOT_eps cost `<T, C>` as a scalar tensor,
    differentiable w.r.t. `x` via the envelope-theorem gradient (not `y`,
    `a`, or `eps`/`L` -- matching `slot_grad`'s own scope exactly, since
    that's the formula backward reuses). The envelope theorem's validity
    doesn't depend on which solve loop produced the potentials, so this
    holds for `symmetric=True` exactly as it does for the default.
    """

    def __init__(self, loss: str = "sinkhorn", *, eps: float = 0.05, L: int = 64,
                 seed: int = 0, n_iters: int = 200, backend: str = "auto",
                 symmetric: bool = False, alpha: float = 0.5):
        super().__init__()
        if loss != "sinkhorn":
            raise ValueError(
                f"SamplesLoss only supports loss='sinkhorn' (SinkSLOT implements no "
                f"other loss family), got {loss!r}"
            )
        self.eps = eps
        self.L = L
        self.seed = seed
        self.n_iters = n_iters
        self.backend = backend
        self.symmetric = symmetric
        self.alpha = alpha

    def forward(self, x: torch.Tensor, y: torch.Tensor, a: torch.Tensor = None,
                potentials: bool = False):
        n = x.shape[0]
        if a is None:
            a = torch.full((n,), 1.0 / n, device=x.device, dtype=x.dtype)
        variant = "symmetric" if self.symmetric else "alternating"

        if potentials:
            phi, psi, *_ = sinkslot_solve(
                x, y, a, a, self.eps, self.L, self.seed, self.n_iters,
                backend=self.backend, variant=variant, alpha=self.alpha)
            return phi, psi

        return _SLOTCostFn.apply(x, y, a, self.eps, self.L, self.seed,
                                  self.n_iters, self.backend, variant, self.alpha)
