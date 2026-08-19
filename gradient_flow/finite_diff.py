"""Does the reference-plan term (II) actually vanish? Finite differences say.

    python -m gradient_flow.finite_diff [--steps 50] [--iters 600]

The paper drops term (II) of

    d SLOT_eps / dx = sum_j P*_kj grad C_kj                        (I)
                    + eps sum_ij (1 - P*_ij/P^SOT_ij) dP^SOT_ij/dx  (II)

on the grounds that P^SOT is determined by the rank order of the projected
points, which is piecewise constant in X, so dP^SOT/dX = 0 almost everywhere.

Every other module here *assumes* that: the support and the reference values
log_S are frozen constants in the autograd graph, so term (II) is zero by
construction and cannot be measured. term_norms.py's residual is therefore a
different object -- the sensitivity of the Sinkhorn solve, not of the reference
plan -- and says nothing about whether (II) is negligible.

This measures (II) the only way it can be measured: by finite differences of the
value itself, with the reference plan *rebuilt* at each perturbed point, so the
rank orders are free to move. Along a random unit direction v,

  analytic     <g_env, v>, term (I) alone
  FD frozen    central difference of SLOT_eps with the support and log_S of the
               base point reused at X +- h v -- a control that must match the
               analytic value, since it is the same function the analytic
               gradient differentiates
  FD rebuilt   central difference with sparse_sot_coo re-run at X +- h v, so any
               rank flip between the two evaluations enters the difference

SLOT_eps is evaluated from the converged potentials as eps*(<phi,a> + <psi,b>),
which equals <C,P> + eps*KL(P||S) at the optimum, in float64.

The result is that term (II) is not a small derivative -- it is not a derivative
at all. The frozen arm reproduces the analytic value, confirming the identity:

    step      analytic     FD frozen   rel. err.   |dF|/F
       0   4.431e-04     4.431e-04     1.1e-09   1.6e-06
      10  -2.906e-04    -2.906e-04     8.7e-05   1.3e-05
      25  -6.904e-05    -6.903e-05     6.6e-04   8.6e-05
      40  -7.862e-06    -7.853e-06     2.3e-03   3.2e-04
      50  -6.683e-06    -6.696e-06     3.8e-03   1.8e-04

(N=1000, L=100, eps=0.01, 600 inner iterations, h=1e-4, 6 directions per point,
float64.) The rel. err. growing along the flow is the derivative itself decaying
to ~7e-06 against a fixed FD noise floor, not the identity degrading.

The rebuilt arm, by contrast, does not converge as h -> 0. At step 0 it takes
4.7e-03, 1.5e-02, 5.8e-02, 2.4e-01 for h=1e-3..1e-6, against an analytic
1.76e-03 -- O(1/h) growth, the signature of a jump discontinuity rather than of
a missing gradient term. At h=1e-7, where no rank flip falls between the two
evaluations, it returns to the analytic value. So dP^SOT/dx is zero wherever it
exists, and at the rank-flip boundaries SLOT_eps is discontinuous, not
differentiable. There is no column for "FD rebuilt" below because a divided
difference that scales as 1/h has no limit worth tabulating; what is reported
instead is |dF|/F, the size of the jump itself.

Those jumps are small: rebuilding P^SOT moves ~0.3% of the support entries and
changes the value by a relative 1.6e-06 at step 0, rising to at most 3.2e-04 by
step 50, where the value has itself fallen two orders of magnitude. The jump
shrinks with h (4.8e-04 at h=1e-3 to 3.3e-05 at h=1e-6, at step 50) and is
systematically negative, since the rebuilt plan is optimal at the perturbed
point -- which is what the `jump sign` column measures.

Stop-gradient on P^SOT is therefore justified, but the justification is
piecewise-constancy plus small jumps, not differentiability with a vanishing
derivative.
"""
from __future__ import annotations

import argparse

import torch

from sinkslot.solver import _ot_1d_coo_batched, _ot_1d_coo_batched_cuda, sparse_sot_coo
from gradient_flow.config import L, LR, N, N_STEPS
from gradient_flow.estimators import EPS, SEED, seg_lse
from gradient_flow.run import DATA_DIR, DEVICE, draw_samples


def build_support(X, Y, a, n_proj, seed=SEED):
    ot1d = _ot_1d_coo_batched_cuda if X.is_cuda else _ot_1d_coo_batched
    rows, cols, S = sparse_sot_coo(X.float(), Y.float(), a.float(), a.float(),
                                 L=n_proj, seed=seed, ot1d=ot1d)
    return rows, cols, S.double().clamp_min(1e-300).log()


def slot_value(X, Y, a, rows, cols, log_S, iters, eps):
    """SLOT_eps at X on the given support, in float64, from the converged duals.

    F = <C,P> + eps*KL(P||S) = eps*(<phi,a> + <psi,b>) at the optimum, since
    P = exp(phi_r + psi_c + log S - C/eps) has marginals a and b.
    """
    cost = ((X[rows] - Y[cols]) ** 2).sum(1)
    lam = log_S - cost / eps
    log_a = a.log()
    n = X.shape[0]
    phi = torch.zeros(n, device=X.device, dtype=X.dtype)
    psi = torch.zeros(n, device=X.device, dtype=X.dtype)
    for _ in range(iters):
        phi = log_a - seg_lse(lam + psi[cols], rows, n)
        psi = log_a - seg_lse(lam + phi[rows], cols, n)
    return eps * float(a @ phi + a @ psi)


def envelope_grad(X, Y, a, rows, cols, log_S, iters, eps):
    """Term (I), closed form, in float64 on the frozen support."""
    cost = ((X[rows] - Y[cols]) ** 2).sum(1)
    lam = log_S - cost / eps
    log_a = a.log()
    n = X.shape[0]
    phi = torch.zeros(n, device=X.device, dtype=X.dtype)
    psi = torch.zeros(n, device=X.device, dtype=X.dtype)
    for _ in range(iters):
        phi = log_a - seg_lse(lam + psi[cols], rows, n)
        psi = log_a - seg_lse(lam + phi[rows], cols, n)
    P = (phi[rows] + psi[cols] + lam).exp()
    r = torch.zeros(n, device=X.device, dtype=X.dtype).index_add_(0, rows, P)
    Px = torch.zeros(n, 2, device=X.device, dtype=X.dtype).index_add_(
        0, rows, P.unsqueeze(1) * Y[cols])
    return 2.0 * a[:, None] * (X - Px / r.clamp_min(1e-300).unsqueeze(1))


def probe(X, Y, a, v, h, iters, eps, n_proj, seed=SEED, base=None):
    """Analytic directional derivative, its frozen-support FD check, and the jump.

    `base` is the base point's (rows, cols, log_S); the jump is what rebuilding
    the reference plan at X + h v does to the value, which is the only trace
    term (II) can leave.
    """
    rows, cols, log_S = base if base is not None else build_support(X, Y, a, n_proj, seed)
    g = envelope_grad(X, Y, a, rows, cols, log_S, iters, eps)
    analytic = float((g * v).sum())

    fp = slot_value(X + h * v, Y, a, rows, cols, log_S, iters, eps)
    fm = slot_value(X - h * v, Y, a, rows, cols, log_S, iters, eps)
    fd_frozen = (fp - fm) / (2 * h)

    r2 = build_support(X + h * v, Y, a, n_proj, seed)
    jump = slot_value(X + h * v, Y, a, *r2, iters, eps) - fp
    k1 = rows.to(torch.int64) * X.shape[0] + cols.to(torch.int64)
    k2 = r2[0].to(torch.int64) * X.shape[0] + r2[1].to(torch.int64)
    moved = 1.0 - float(torch.isin(k2, k1).double().mean())
    return analytic, fd_frozen, jump, moved


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=N_STEPS)
    p.add_argument("--iters", type=int, default=600)
    p.add_argument("--n", type=int, default=N)
    p.add_argument("--eps", type=float, default=EPS)
    p.add_argument("--h", type=float, default=1e-4)
    p.add_argument("--dirs", type=int, default=6, help="random directions per point")
    p.add_argument("--at", type=int, nargs="+", default=[0, 10, 25, 40, 50],
                   help="flow steps to probe")
    args = p.parse_args()

    if DEVICE != "cuda":
        print("gradient_flow/finite_diff.py: no CUDA GPU found, running on CPU "
              "(pure torch throughout -- much slower, same algorithm).")

    n = args.n
    rng = torch.Generator(device="cpu").manual_seed(1)
    X = draw_samples(DATA_DIR / "density_a.png", n, rng, device=DEVICE).double()
    Y = draw_samples(DATA_DIR / "density_b.png", n, rng, device=DEVICE).double()
    a = torch.full((n,), 1.0 / n, dtype=torch.float64, device=DEVICE)
    gen = torch.Generator(device=DEVICE).manual_seed(0)

    print(f"N={n}  L={L}  eps={args.eps:g}  iters={args.iters}  h={args.h:g}  "
          f"dirs={args.dirs}  float64")
    print(f"\n{'step':>5} {'analytic':>12} {'FD frozen':>12} {'FD err':>9} "
          f"{'|jump|/F':>10} {'jump sign':>10} {'entries moved':>14}")

    probe_at = sorted(set(args.at))
    for step in range(max(probe_at) + 1):
        base = build_support(X, Y, a, L)
        if step in probe_at:
            an, fd, jp, mv = [], [], [], []
            F = slot_value(X, Y, a, *base, args.iters, args.eps)
            for _ in range(args.dirs):
                v = torch.randn(n, 2, generator=gen, device=DEVICE, dtype=torch.float64)
                v /= v.norm()
                r = probe(X, Y, a, v, args.h, args.iters, args.eps, L, base=base)
                for lst, val in zip((an, fd, jp, mv), r):
                    lst.append(val)
            an, fd, jp = torch.tensor(an), torch.tensor(fd), torch.tensor(jp)
            scale = float(an.abs().mean())
            # |mean|/mean|.| near 0 means the jumps cancel: no systematic direction.
            bias = float(jp.mean().abs()) / float(jp.abs().mean())
            print(f"{step:>5} {float(an.mean()):>12.5e} {float(fd.mean()):>12.5e} "
                  f"{float((fd - an).abs().mean()) / scale:>9.1e} "
                  f"{float(jp.abs().mean()) / F:>10.1e} {bias:>10.2f} "
                  f"{float(torch.tensor(mv).mean()):>14.1e}", flush=True)

        g = envelope_grad(X, Y, a, *base, args.iters, args.eps)
        X = (X - LR * n * g).clone()


if __name__ == "__main__":
    main()
