"""Which gradient estimator survives early stopping: envelope, detach, or unrolled?

    python -m gradient_flow.estimators


All three differentiate <P, C> wrt X. They differ only in what they do with the
solve:

  envelope : P treated as optimal, grad = 2*diag(a)*(X - T(X)), normalised by the
             plan's achieved row mass (this is sinkslot.gradient.slot_grad)
  detach   : P detached, autograd through C only -- the same envelope gradient
             WITHOUT that normalisation (verified: envelope == (a/r) * detach)
  unrolled : no detach; autograd runs back through all k Sinkhorn iterations, so
             it keeps the <dP/dX, C> term the envelope theorem discards

The support (rows, cols, log_S) is held fixed for all three. It is a discrete
object -- piecewise constant in X, so its derivative is zero a.e. and carries no
signal -- and freezing it is what makes the three comparable: the only remaining
difference is the treatment of the solve.

Result: the envelope form wins at every truncation, and unrolled loses badly --
3.6e-02 relative error at k=200 against the envelope's 1.3e-04, a factor of 280.
That is worth stating plainly because it inverts the obvious intuition. Keeping
the <dP/dX, C> term sounds strictly more correct, but the unrolled gradient is
the exact derivative of a *different* function: the k-iteration algorithm's
output, not SLOT_eps. Its extra term measures the sensitivity of an unfinished
solve, which is large precisely when the solve is unfinished, and it decays only
as the iteration map's contraction lets it. The envelope form discards that term
outright and is left with an error that decays with the plan itself.

So the ranking under early stopping is envelope < detach < unrolled, and the
cheapest estimator is also the most accurate one -- unrolled additionally costs
O(k) stored activations, which is the reason not to reach for it anyway.

Caveat on the floor: the reference is this module's own plain-torch fp32 loop at
3000 iterations, which settles at |a/r - 1| ~ 1e-5 rather than the ~1e-8 the
Triton path reaches. The envelope's last point (1.3e-04 at k=200) is therefore
within about an order of magnitude of the reference's own accuracy and should be
read as an upper bound. The separation between the three curves is 2-3 orders of
magnitude and is unaffected.
"""
import torch

from sinkslot.solver import (
    _ot_1d_coo_batched, _ot_1d_coo_batched_cuda, sot_coo, sparse_sqeuclidean_cost,
)
from gradient_flow.config import L, N
from gradient_flow.run import DATA_DIR, DEVICE, draw_samples

EPS, SEED = 0.01, 0


def seg_lse(vals, idx, size):
    """Differentiable segmented log-sum-exp. Max is detached (it cancels)."""
    mx = vals.new_full((size,), -1e30).scatter_reduce(
        0, idx, vals.detach(), reduce="amax", include_self=True)
    acc = vals.new_zeros(size).index_add_(0, idx, (vals - mx[idx]).exp())
    return mx + acc.clamp_min(torch.finfo(vals.dtype).tiny).log()


def solve(lam, rows, cols, log_a, n, m, k):
    """k plain-torch Sinkhorn iterations, absorbed potentials, matching sinkslot_alternating_triton."""
    phi = torch.zeros(n, device=lam.device, dtype=lam.dtype)
    psi = torch.zeros(m, device=lam.device, dtype=lam.dtype)
    for _ in range(k):
        phi = log_a - seg_lse(lam + psi[cols], rows, n)
        psi = log_a - seg_lse(lam + phi[rows], cols, m)
    return phi, psi


def three_gradients(k, X, Y, a, rows, cols, log_S, n, m, unroll=True, eps=EPS):
    log_a = a.log()

    def build(Xv):
        cost = ((Xv[rows] - Y[cols]) ** 2).sum(1)
        return cost, log_S - cost / eps

    # ---- unrolled: differentiate through everything -------------------------
    g_unrolled = None
    if unroll:
        Xu = X.clone().requires_grad_(True)
        cost_u, lam_u = build(Xu)
        phi, psi = solve(lam_u, rows, cols, log_a, n, m, k)
        T_u = (phi[rows] + psi[cols] + lam_u).exp()
        (g_unrolled,) = torch.autograd.grad((T_u * cost_u).sum(), [Xu])

    # ---- envelope + detach: same solve, no graph -----------------------------
    with torch.no_grad():
        cost, lam = build(X)
        phi, psi = solve(lam, rows, cols, log_a, n, m, k)
        T = (phi[rows] + psi[cols] + lam).exp()
        r = torch.zeros(n, device=X.device, dtype=T.dtype).index_add_(0, rows, T)
        Tx = torch.zeros(n, 2, device=X.device, dtype=T.dtype).index_add_(
            0, rows, T.unsqueeze(1) * Y[cols])
        Tx = Tx / r.clamp_min(torch.finfo(T.dtype).tiny).unsqueeze(1)
        g_env = 2.0 * a[:, None] * (X - Tx)
        transport = float((T * cost).sum())

    Xd = X.clone().requires_grad_(True)
    cost_d, _ = build(Xd)
    (g_det,) = torch.autograd.grad((T.detach() * cost_d).sum(), [Xd])

    return g_env, g_det, g_unrolled, float((a / r - 1).abs().max()), transport


def main():
    if DEVICE != "cuda":
        print("gradient_flow/estimators.py: no CUDA GPU found, running on CPU "
              "(pure torch throughout -- much slower, same algorithm).")
    rng = torch.Generator(device="cpu").manual_seed(1)
    X = draw_samples(DATA_DIR / "density_a.png", N, rng, device=DEVICE).float()
    Y = draw_samples(DATA_DIR / "density_b.png", N, rng, device=DEVICE).float()
    a = torch.full((N,), 1.0 / N, dtype=torch.float32, device=DEVICE)

    ot1d = _ot_1d_coo_batched_cuda if DEVICE == "cuda" else _ot_1d_coo_batched
    rows, cols, S = sot_coo(X, Y, a, a, L=L, seed=SEED, ot1d=ot1d)
    log_S = S.clamp_min(torch.finfo(S.dtype).tiny).log()
    print(f"support nnz={rows.numel()}  N={N}  L={L}  eps={EPS}")

    # Reference: envelope at deep convergence (all three agree there).
    g_ref, _, _, viol_ref, _ = three_gradients(3000, X, Y, a, rows, cols, log_S, N, N, unroll=False)
    ref = float(g_ref.norm())
    print(f"reference @3000 iters: |a/r-1|max={viol_ref:.2e}\n")

    grid = [1, 2, 5, 10, 20, 50, 100, 200]
    curves = {"envelope": [], "detach": [], "unrolled": []}
    print(f"{'k':>5} {'envelope':>11} {'detach':>11} {'unrolled':>11}   {'best':>9}")
    for k in grid:
        ge, gd, gu, _, _ = three_gradients(k, X, Y, a, rows, cols, log_S, N, N)
        e = [float((g - g_ref).norm()) / ref for g in (ge, gd, gu)]
        for name, v in zip(curves, e):
            curves[name].append(v)
        best = ["envelope", "detach", "unrolled"][int(torch.tensor(e).argmin())]
        print(f"{k:>5} {e[0]:>11.3e} {e[1]:>11.3e} {e[2]:>11.3e}   {best:>9}")
        del ge, gd, gu
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    _plot(grid, curves)


def _plot(grid, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from gradient_flow.run import OUT_DIR

    style = {"envelope": ("o-", "#1f77b4"), "detach": ("s--", "#ff7f0e"),
             "unrolled": ("^-.", "#d62728")}
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for name, vals in curves.items():
        mk, col = style[name]
        ax.plot(grid, vals, mk, ms=5, color=col, label=name)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("inner Sinkhorn iterations")
    ax.set_ylabel("relative error vs converged gradient")
    ax.set_title(f"Gradient estimators under early stopping\n"
                 f"$N={N}$, $L={L}$, $\\epsilon={EPS}$")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"gradient_estimators_eps_{EPS:g}.pdf"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
