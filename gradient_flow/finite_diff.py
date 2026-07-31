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
  FD rebuilt   central difference with sot_plan_coo re-run at X +- h v, so any
               rank flip between the two evaluations enters the difference

The gap between the two FD columns is term (II) projected on v: same value,
same eps, same projection directions (the seed is fixed, so the *directions*
never change -- only the orders they induce).

SLOT_eps is evaluated from the converged potentials as eps*(<phi,a> + <psi,b>),
which equals <C,P> + eps*KL(P||S) at the optimum, in float64.

At N=1000, L=100, eps=0.01, 600 inner iterations, h=1e-4, over 8 directions:

    step   analytic     FD frozen    FD rebuilt   (frozen-an)/|an|  (rebuilt-an)/|an|
    (filled in from the run below)
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from flash_sinkhorn.bench.sinkslot import _ot_1d_coo_batched_cuda, sot_plan_coo
from gradient_flow.config import L, LR, N, N_STEPS
from gradient_flow.estimators import EPS, SEED, seg_lse
from gradient_flow.run import DATA_DIR, draw_samples


def build_support(X, Y, a, n_proj, seed=SEED):
    rows, cols, S = sot_plan_coo(X.float(), Y.float(), a.float(), a.float(),
                                 L=n_proj, seed=seed, ot1d=_ot_1d_coo_batched_cuda)
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

    if not torch.cuda.is_available():
        raise RuntimeError("needs a CUDA GPU: the support builder is Triton-only")

    n = args.n
    rng = np.random.default_rng(1)
    X = draw_samples(DATA_DIR / "density_a.png", n, rng, device="cuda").double()
    Y = draw_samples(DATA_DIR / "density_b.png", n, rng, device="cuda").double()
    a = torch.full((n,), 1.0 / n, dtype=torch.float64, device="cuda")
    gen = torch.Generator(device="cuda").manual_seed(0)

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
                v = torch.randn(n, 2, generator=gen, device="cuda", dtype=torch.float64)
                v /= v.norm()
                r = probe(X, Y, a, v, args.h, args.iters, args.eps, L, base=base)
                for lst, val in zip((an, fd, jp, mv), r):
                    lst.append(val)
            an, fd, jp = np.array(an), np.array(fd), np.array(jp)
            scale = np.abs(an).mean()
            # |mean|/mean|.| near 0 means the jumps cancel: no systematic direction.
            bias = abs(jp.mean()) / np.abs(jp).mean()
            print(f"{step:>5} {an.mean():>12.5e} {fd.mean():>12.5e} "
                  f"{np.abs(fd - an).mean() / scale:>9.1e} "
                  f"{np.abs(jp).mean() / F:>10.1e} {bias:>10.2f} "
                  f"{np.mean(mv):>14.1e}", flush=True)

        g = envelope_grad(X, Y, a, *base, args.iters, args.eps)
        X = (X - LR * n * g).clone()


if __name__ == "__main__":
    main()
