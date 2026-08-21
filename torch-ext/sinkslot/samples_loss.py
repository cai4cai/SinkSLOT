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

Independent `a`/`b` marginal weights (and `x.shape[0] != y.shape[0]`) are
supported: the envelope-theorem gradient below is unchanged by `b` -- only
the plan it's built from is -- verified by finite difference (~1e-10
relative error, float64, independent random a/b, n != m). `b` defaults to
`a` when omitted, matching this module's original shared-weight behavior.
"""

from __future__ import annotations

import torch

from .solver import sparse_sqeuclidean_cost
from .sinkhorn_solvers import sinkslot_solve
from .gradient import sparse_barycentric_map


class _SLOTCostFn(torch.autograd.Function):
    """Forward: solve + the achieved transport cost <T, C> (not the full
    SLOT_eps divergence, which also adds eps * KL(T, P^SOT)). Backward: the
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
    def forward(ctx, X, Y, a, b, eps, L, seed, n_iters, backend, variant, alpha,
                stop_mode, stop_max_iter, stop_tol, stop_check_every):
        phi, psi, rows, cols, S, it, converged, viol = sinkslot_solve(
            X, Y, a, b, eps, L, seed, n_iters, stop_mode=stop_mode,
            stop_max_iter=stop_max_iter, stop_tol=stop_tol,
            stop_check_every=stop_check_every, backend=backend,
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
        Tx, _ = sparse_barycentric_map(T_vals, rows, cols, X, Y)
        grad_x = 2.0 * a[:, None] * (X - Tx)
        return (grad_output * grad_x,
                None, None, None, None, None, None, None, None, None, None,
                None, None, None, None)


class SamplesLoss(torch.nn.Module):
    """SLOT_eps(x, y) as a GeomLoss-style callable loss module.

    `loss`: only "sinkhorn" is accepted (kept as a constructor argument for
    call-site parity with geomloss/flash_sinkhorn, which support several
    other loss families this package doesn't implement at all).

    `eps`: entropic regularisation (SinkSLOT's own `eps`, not GeomLoss's
    `blur` -- there is no `blur**p = eps` conversion here; pass `eps`
    directly). `L`: number of slicing directions. `seed`: `sot_plan_coo`'s
    projection RNG seed. `n_iters`: exact iteration count when
    `stop_mode == "fixed"` (the default); ignored otherwise, in favor of
    `stop_max_iter`. `stop_mode`/`stop_max_iter`/`stop_tol`/`stop_check_every`:
    forwarded to `sinkslot_solve` unchanged, see its own docstring for what
    each `stop_mode` value checks. Early stopping only changes how many
    iterations `forward` runs -- `backward` reuses whatever potentials it
    converged to either way, since the envelope theorem holds at any fixed
    point regardless of how it was reached. `backend`: "auto" / "triton" /
    "torch", see `sinkslot_solve`.

    `symmetric`: False (default) uses the alternating (Gauss-Seidel)
    solve loop, same as every other SinkSLOT entry point so far. True uses
    the symmetric (Jacobi) loop instead (`sinkslot_solve`'s `variant=
    "symmetric"`) -- see `sinkhorn_solvers.sinkslot_symmetric_triton`'s own
    docstring for the update rule. `alpha` is that update's blend weight
    (`f_new = (1-alpha)*f_old + alpha*f_cand`); unused when
    `symmetric=False`.

    Forward returns the achieved transport cost `<T, C>` as a scalar tensor
    (not the full SLOT_eps divergence, which also adds `eps * KL(T, P^SOT)`),
    differentiable w.r.t. `x` via the envelope-theorem gradient (not `y`,
    `a`/`b`, or `eps`/`L` -- matching `slot_grad`'s own scope exactly, since
    that's the formula backward reuses). The envelope theorem's validity
    doesn't depend on which solve loop produced the potentials, so this
    holds for `symmetric=True` exactly as it does for the default.

    `forward(x, y, a=None, b=None, potentials=False)`: `a`/`b` are the
    source/target marginal weights, each independently uniform over its own
    `x.shape[0]`/`y.shape[0]` when omitted -- matching `sparse_transport_plan`'s
    convention, not tied to each other. `potentials=True` returns `(phi, psi)`
    instead of the scalar cost.
    """

    def __init__(self, loss: str = "sinkhorn", *, eps: float = 0.05, L: int = 64,
                 seed: int = 0, n_iters: int = 200, backend: str = "auto",
                 symmetric: bool = False, alpha: float = 0.5,
                 stop_mode: str = "fixed", stop_max_iter: int = 20000,
                 stop_tol: float = 1e-6, stop_check_every: int = 5):
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
        self.stop_mode = stop_mode
        self.stop_max_iter = stop_max_iter
        self.stop_tol = stop_tol
        self.stop_check_every = stop_check_every

    def forward(self, x: torch.Tensor, y: torch.Tensor, a: torch.Tensor = None,
                b: torch.Tensor = None, potentials: bool = False):
        n, m = x.shape[0], y.shape[0]
        if a is None:
            a = torch.full((n,), 1.0 / n, device=x.device, dtype=x.dtype)
        if b is None:
            b = torch.full((m,), 1.0 / m, device=y.device, dtype=y.dtype)
        variant = "symmetric" if self.symmetric else "alternating"

        if potentials:
            phi, psi, *_ = sinkslot_solve(
                x, y, a, b, self.eps, self.L, self.seed, self.n_iters,
                stop_mode=self.stop_mode, stop_max_iter=self.stop_max_iter,
                stop_tol=self.stop_tol, stop_check_every=self.stop_check_every,
                backend=self.backend, variant=variant, alpha=self.alpha)
            return phi, psi

        return _SLOTCostFn.apply(x, y, a, b, self.eps, self.L, self.seed,
                                  self.n_iters, self.backend, variant, self.alpha,
                                  self.stop_mode, self.stop_max_iter, self.stop_tol,
                                  self.stop_check_every)
