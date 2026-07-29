"""Envelope vs full gradient, measured at every step of the flow.

    python -m gradient_flow.along_flow [--steps 50] [--iters 200] [--n 1000]

estimators.py compares the estimators at one fixed X (the flow's starting
configuration). That leaves open whether what it measured is a property of that
particular point: X_0 is a blob far from the target, the plan there is diffuse,
and a diffuse plan looks like the regime where the <dP/dX, C> term the envelope
theorem discards should be largest. The expectation is then that the two
gradients agree better and better as the flow converges.

Measured, they do the opposite. This script walks the actual descent trajectory
and, at each step, reports:

  |g_env|   norm of the analytical (envelope-theorem) gradient,
            2*diag(a)*(X - T_eps(X)) -- the direction the flow actually takes
  |g_full|  norm of the complete gradient: autograd through all `iters`
            Sinkhorn iterations, keeping the <dP/dX, C> term
  cos       cosine similarity between the two, over the flattened (N,2) fields

Both are computed at the same step, on the same support, from the same solve,
so they differ only in whether the solve's sensitivity is kept.

At N=1000, L=100, eps=0.01, 200 inner iterations, over 50 steps:

    step     |g_env|    |g_full|      cos    ratio
       0   6.344e-02   6.365e-02   0.9994   1.0032
      10   2.221e-02   2.223e-02   0.9993   1.0007
      20   7.776e-03   7.790e-03   0.9954   1.0018
      30   2.768e-03   2.836e-03   0.9679   1.0245
      40   1.083e-03   1.254e-03   0.8273   1.1585
      50   5.844e-04   8.740e-04   0.6053   1.4955

The two agree to 6e-04 in cosine at the start and have fallen to 0.61 by step
50 -- they disagree *most* where the flow is closest to its fixed point. The
norms tell the same story: |g_env| decays by a factor of 109 over the run while
|g_full| decays by only 73, so the gap is not a small correction that shrinks
along with the signal.

It is also not a truncation artifact. Re-solving the step-50 configuration at
k = 50, 100, 200, 400, 800 moves the cosine only from 0.6185 to 0.6244 and
flattens there, while the max marginal violation falls 5.2e-03 -> 9.9e-05. The
disagreement is what the two estimators converge *to*, not how far from
converged they are.

The cause is that they are gradients of two different objectives. The unrolled
arm differentiates <P, C>, but the envelope identity
grad_X SLOT_eps = 2*diag(a)*(X - T_eps(X)) is the derivative of the *regularized*
objective <P, C> + eps*KL(P||S), whose optimality conditions are what make the
potential terms drop. Unrolling that full objective instead (--regularized,
which is eps*(sum P*(phi_r + psi_c) - sum P) up to an X-independent constant)
recovers the envelope gradient at step 50 to cosine 0.999992 and 0.4% relative
norm. So the entire gap is the entropic term, not envelope-theorem bias, and
the envelope gradient is the correct one for SLOT_eps at every step.

Why it grows: eps is fixed while the transport cost collapses as X approaches
Y, so the entropic contribution goes from negligible to comparable. The
practical reading is that unrolling gets *less* defensible the closer the flow
gets, which is the reverse of the usual intuition about early-stopping bias.

The flow is driven by g_env -- it is the method under study, and letting the
trajectory depend on which gradient is being measured would confound the two.

Note on `iters`: the full gradient stores O(iters) activations over the whole
support, so the default here (200) is below run.py's MAX_ITER=1000. 200 is
already past the point where estimators.py shows the envelope gradient
converged (1.3e-04 relative error), so the analytical arm is unaffected; the
unrolled arm is reported at the same truncation for a like-for-like comparison,
and the k-sweep above shows the conclusion does not depend on that choice.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from flash_sinkhorn.bench.sinkslot import _ot_1d_coo_batched_cuda, sot_plan_coo
from gradient_flow.config import L, LR, N, N_STEPS
from gradient_flow.estimators import EPS, SEED, solve, three_gradients
from gradient_flow.run import DATA_DIR, draw_samples


def regularized_unrolled(k, X, Y, a, rows, cols, log_S, n):
    """Unrolled gradient of the objective the envelope identity actually applies to.

    F = <P,C> + eps*KL(P||S), which for P = exp(phi_r + psi_c + lam) with
    lam = log_S - C/eps reduces to eps*(sum P*(phi_r + psi_c) - sum P) plus a
    term in S alone, which does not depend on X.
    """
    Xu = X.clone().requires_grad_(True)
    cost = ((Xu[rows] - Y[cols]) ** 2).sum(1)
    lam = log_S - cost / EPS
    phi, psi = solve(lam, rows, cols, a.log(), n, n, k)
    P = (phi[rows] + psi[cols] + lam).exp()
    F = EPS * ((P * (phi[rows] + psi[cols])).sum() - P.sum())
    (g,) = torch.autograd.grad(F, [Xu])
    return g


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=N_STEPS)
    p.add_argument("--iters", type=int, default=200,
                   help="inner Sinkhorn iterations, shared by both estimators")
    p.add_argument("--n", type=int, default=N)
    p.add_argument("--regularized", action="store_true",
                   help="also unroll <P,C> + eps*KL(P||S), which should match g_env")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("needs a CUDA GPU: the support builder is Triton-only")

    n = args.n
    rng = np.random.default_rng(1)
    X = draw_samples(DATA_DIR / "density_a.png", n, rng, device="cuda").float()
    Y = draw_samples(DATA_DIR / "density_b.png", n, rng, device="cuda").float()
    a = torch.full((n,), 1.0 / n, dtype=torch.float32, device="cuda")

    print(f"N={n}  L={L}  eps={EPS}  inner iters={args.iters}  lr={LR}  steps={args.steps}")
    extra = f" {'cos(env,reg)':>13}" if args.regularized else ""
    print(f"\n{'step':>5} {'|g_env|':>12} {'|g_full|':>12} {'cos':>10} "
          f"{'|g_full|/|g_env|':>17}{extra} {'nnz':>8}")

    for step in range(args.steps + 1):
        # Support is rebuilt from the current X, as the flow itself does; both
        # estimators then share it, so the step's comparison is like-for-like.
        rows, cols, S = sot_plan_coo(X, Y, a, a, L=L, seed=SEED, ot1d=_ot_1d_coo_batched_cuda)
        log_S = S.clamp_min(torch.finfo(S.dtype).tiny).log()

        g_env, _, g_full, _ = three_gradients(
            args.iters, X, Y, a, rows, cols, log_S, n, n)

        cosine = lambda u, v: float(torch.nn.functional.cosine_similarity(
            u.flatten(), v.flatten(), dim=0))
        ne, nf = float(g_env.norm()), float(g_full.norm())
        extra = ""
        if args.regularized:
            g_reg = regularized_unrolled(args.iters, X, Y, a, rows, cols, log_S, n)
            extra = f" {cosine(g_env, g_reg):>13.6f}"
            del g_reg
        print(f"{step:>5} {ne:>12.4e} {nf:>12.4e} {cosine(g_env, g_full):>10.6f} "
              f"{nf / ne:>17.4f}{extra} {rows.numel():>8}")

        if step == args.steps:
            break
        X = (X - LR * n * g_env).detach().clone()
        del g_env, g_full, rows, cols, S, log_S
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
