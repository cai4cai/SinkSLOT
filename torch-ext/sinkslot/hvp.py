"""Hessian-vector product via implicit differentiation of the sliced-OT
Sinkhorn fixed point.

Split out from solver.py, and named to match flash_sinkhorn's own hvp.py:
`hvp_x_sqeuclid`/`hvp_x_sqeuclid_from_potentials` there solve the analogous
implicit-differentiation linear system (dense Schur complement, streaming
Triton application) that this module solves sparsely -- same underlying
method (implicit function theorem on the Sinkhorn optimality conditions),
different sparsity structure, so the same names apply honestly.

No `HvpInfo` dataclass here (flash_sinkhorn's `hvp_x_sqeuclid_from_potentials`
returns one, alongside cg_converged/cg_iters/cg_residual): the linear solve
below goes through `torchsparsegradutils.utils.minres`, which runs a fixed
`max_cg_iterations` with no early-stopping check and returns only the
solution tensor -- there is no convergence/residual data available to
surface. Fabricating an always-"converged" info object would misrepresent
what the solver actually reports. A real `HvpInfo` here would need computing
the residual `||M @ sol - rhs||` explicitly after the solve; left as a
follow-up.
"""

from __future__ import annotations

import functools

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped

from .solver import _HAS_TRITON, sot_plan_coo, _ot_1d_coo_batched_cuda, to_csr, sparse_sqeuclidean_cost
from .sinkhorn_solvers import sinkslot_alternating_triton, sinkslot_alternating_torch


def hvp_x_sqeuclid_from_potentials(X, Y, a, v, rows, cols, T_vals, eps, *,
                                    tau2=3e-7, solve_tol=1e-11, max_cg_iter=8000):
    """H(X) @ v given an already-solved sparse plan (rows, cols, T_vals) --
    the linear-algebra half of `hvp_x_sqeuclid`, with no Sinkhorn solve of its
    own. `T_vals` is `(phi[rows] + psi[cols] + lam).exp()` at the converged
    potentials -- the materialised plan itself, not phi/psi directly, since
    that's what this function's linear system actually needs and what a
    caller who already ran a solve (e.g. `slot_grad`) already has on hand
    without recomputing it.

    Derivation. At the converged potentials, the row/column marginal
    constraints hold identically along any curve X(t) = X + t*v with the
    potentials implicitly defined by those same constraints. Differentiating
    both constraints at t=0 gives a linear system in u := dphi/dt,
    w := dpsi/dt (n and m entries respectively):

        [ diag(a)      P    ] [u]   [rhs_u]
        [   P^T    diag(a)  ] [w] = [rhs_w]

    where P is the sparse plan (rows, cols, T_vals) and rhs_{u,w} come from
    scattering -P_ij * d(lam_ij)/dt (a per-entry quantity that's explicit in
    v, no implicit solve needed for it) by row and by column. This matrix is
    symmetric PSD -- for any (u, w), (u,w)^T M (u,w) = sum_ij P_ij*(u_i+w_j)^2
    >= 0, using P's row/col sums being a itself -- with a 1-D null space (the
    usual (+c,-c) potential-shift ambiguity), so `tau2` regularizes it, the
    same role FlashSinkhorn's own dense HVP (hvp.py) has its `tau2` play on
    its analogous Schur complement -- though NOT the same value: FlashSinkhorn's
    default (1e-5) is tuned for its dense Schur complement, a different matrix
    with different conditioning, and reused verbatim here gave ~23% error
    against a finite-difference check (see below) where this module's default
    (3e-7) gives ~2%. Given u, w, the HVP follows from differentiating
    T_eps(X) = diag(1/a) @ P @ Y (dr/dt = 0 along the constraint curve, so only
    dP/dt contributes):

        H(X)v = 2*diag(a)*v - 2*diag(1/a)-weighted scatter of
                P_ij*(u_i + w_j + d(lam_ij)/dt) * Y_j

    Solved via `torchsparsegradutils.sparse_generic_solve` on the sparse (n+m)
    system directly (MINRES, since M is symmetric but only PSD, not PD),
    rather than materialising the dense n x n Schur complement
    P @ diag(1/a) @ P^T the way the dense case can afford to.
    """
    try:
        import torchsparsegradutils as tsgu
        from torchsparsegradutils.utils import minres, MINRESSettings
    except ImportError as e:
        raise ImportError(
            "hvp_x_sqeuclid needs torchsparsegradutils (pip install torchsparsegradutils)"
        ) from e

    n, d = X.shape
    m = Y.shape[0]

    # Explicit part: d(lam_ij)/dt for this v, only X depends on t.
    dlam_dt = -(2.0 / eps) * ((X[rows] - Y[cols]) * v[rows]).sum(1)

    rhs_u = torch.zeros(n, device=X.device, dtype=T_vals.dtype).index_add_(
        0, rows, -(T_vals * dlam_dt))
    rhs_w = torch.zeros(m, device=X.device, dtype=T_vals.dtype).index_add_(
        0, cols, -(T_vals * dlam_dt))

    idx_u = torch.arange(n, device=X.device)
    idx_w = torch.arange(m, device=X.device) + n
    row_off, col_off = rows, cols + n
    I = torch.cat([idx_u, idx_w, row_off, col_off])
    J = torch.cat([idx_u, idx_w, col_off, row_off])
    V = torch.cat([a + tau2, a + tau2, T_vals, T_vals])
    Mmat = torch.sparse_coo_tensor(torch.stack([I, J]), V, (n + m, n + m)).coalesce()
    rhs = torch.cat([rhs_u, rhs_w]).unsqueeze(1)

    solve = functools.partial(
        minres, settings=MINRESSettings(minres_tolerance=solve_tol,
                                         max_cg_iterations=max_cg_iter))
    sol = tsgu.sparse_generic_solve(Mmat, rhs, solve=solve, transpose_solve=solve).squeeze(1)
    u, w = sol[:n], sol[n:]

    weight = T_vals * (u[rows] + w[cols] + dlam_dt)
    dT = torch.zeros(n, d, device=X.device, dtype=X.dtype).index_add_(
        0, rows, weight.unsqueeze(1) * Y[cols])
    return 2.0 * a[:, None] * v - 2.0 * dT


@jaxtyped(typechecker=beartype)
def hvp_x_sqeuclid(X: Float[torch.Tensor, "n d"], Y: Float[torch.Tensor, "m d"],
             a: Float[torch.Tensor, "n"], eps, L, seed, n_iters,
             v: Float[torch.Tensor, "n d"], tau2=3e-7, solve_tol=1e-11,
             max_cg_iter=8000, backend="auto"):
    """End-to-end Hessian-vector product H(X) @ v of grad_X SLOT_eps: solve
    Sinkhorn on the (frozen) sliced-OT support, then call
    `hvp_x_sqeuclid_from_potentials` for the linear-algebra half -- the
    second-order counterpart to `slot_grad`'s envelope-theorem gradient.

    This is new code, grounded in standard implicit-differentiation-of-Sinkhorn
    theory (same structure as OTT-JAX's and FlashSinkhorn's own Hessians), not
    the paper's own derivation -- cross-check the notation there before relying
    on this for anything reported. Validated empirically instead: against a
    FROZEN-SUPPORT central finite difference of `slot_grad` (testing/test_hvp.py)
    -- frozen support, not a naive slot_grad(X+h*v) vs slot_grad(X-h*v), because
    each such call rebuilds the sliced-OT support from scratch, and
    gradient_flow/finite_diff.py already shows (empirically, in this repo) that
    doing so at any h small enough to resolve second-order structure picks up
    O(1/h) rank-flip jump artifacts unrelated to the real signal. On a frozen
    support, matches to ~2% relative error at the tuned defaults, stable across
    3 orders of magnitude in h (i.e. not a finite-difference truncation
    artifact). n_iters affects accuracy the same way it affects
    slot_grad's: too few inner Sinkhorn iterations leaves phi, psi (and hence
    the linear system's own P, rhs) short of the true fixed point.

    X, Y, a, v expected fp32 (same Triton-kernel constraint as slot_grad).
    `a` is used as both marginals, matching slot_grad's own
    `sot_plan_coo(X, Y, a, a, ...)` call (X, Y assumed equal-size, uniform-a,
    as in every caller in this repo).

    `backend`: "auto" (default) / "triton" / "torch", same meaning and same
    dispatch rule as `sinkslot_solve`'s -- picks Triton
    (`sinkslot_alternating_triton`) when it's importable and X is CUDA,
    `sinkslot_alternating_torch` otherwise; "triton" forces it (raising if
    unavailable), "torch" forces the pure-torch solve regardless of device.
    Unlike `sinkslot_solve`, the sliced-support builder here stays
    `_ot_1d_coo_batched_cuda` for every backend, not device-switched to the
    naive builder on CPU: it's plain torch underneath (no Triton, no `.cuda()`
    calls), so it runs anywhere, and its internal fp64 cumsum is exactly the
    extra precision the existing validation below relies on -- switching to
    the naive builder would risk a genuinely different support (see its own
    docstring), not just different rounding. So `backend="torch"` changes
    only the solve stage's numerics, not the support.

    Accuracy caveat: the ~2% figure above and the `tau2=3e-7` default were
    both obtained through the Triton path (`sinkslot_alternating_triton`) on
    two small synthetic problems (testing/test_hvp.py's `_problem()`,
    n<=300, d=3) -- not on any of the paper's five real datasets at their
    actual n=10000 scale, and not on `sinkslot_alternating_torch` at all.
    `backend="torch"` is offered for portability the same way
    `sinkslot_solve`'s is, not because the 2%/3e-7 pairing has been
    re-confirmed on that path; matrix conditioning depends on the actual data
    (via `a`, `T_vals`), so a real distribution shift (dataset or scale)
    could call for a different `tau2` independent of backend too.
    """
    if backend not in ("auto", "triton", "torch"):
        raise ValueError(f"backend must be 'auto', 'triton', or 'torch', got {backend!r}")
    if backend == "triton":
        if not (_HAS_TRITON and X.is_cuda):
            raise ValueError("backend='triton' requires Triton installed and X on CUDA")
        use_triton = True
    elif backend == "torch":
        use_triton = False
    else:
        use_triton = _HAS_TRITON and X.is_cuda

    n, d = X.shape
    m = Y.shape[0]
    rows, cols, S = sot_plan_coo(X, Y, a, a, L=L, seed=seed, ot1d=_ot_1d_coo_batched_cuda)
    cost = sparse_sqeuclidean_cost(
        X, Y, rows, cols,
        use_triton=(backend == "triton") if backend != "auto" else None,
    )
    log_S = S.clamp_min(torch.finfo(S.dtype).tiny).log()
    lam = log_S - cost / eps
    log_a = a.log()

    if use_triton:
        r_ptr, r_idx, r_lam, _ = to_csr(rows, cols, lam, n, narrow_key=True)
        c_ptr, c_idx, c_lam, _ = to_csr(cols, rows, lam, m, narrow_key=True)
        phi, psi, _, _, _ = sinkslot_alternating_triton(r_ptr, r_idx, r_lam, c_ptr, c_idx, c_lam,
                                     log_a, log_a, n, m, n_iters)
    else:
        phi, psi, _, _, _ = sinkslot_alternating_torch(rows, cols, lam, log_a, log_a, n, m, n_iters)
    T_vals = (phi[rows] + psi[cols] + lam).exp()

    return hvp_x_sqeuclid_from_potentials(
        X, Y, a, v, rows, cols, T_vals, eps,
        tau2=tau2, solve_tol=solve_tol, max_cg_iter=max_cg_iter,
    )
