"""EOT baseline for the gradient-flow experiment, via GeomLoss's own kernels.

Previously held a dense SOT/EOT/SROT baseline vendored from a sibling
research repository. Both call sites in gradient_flow/run.py have since moved
off it -- EOT to GeomLoss (below), SROT to sinkslot's own sparse solve
machinery (see sinkslot_smoothed_divergence in run.py) after the vendored
SROT implementation turned numerically unstable at eps=0.01 once an
unjustified floor on its Gibbs kernel was removed. The vendored functions
are gone with it: dead code with no remaining callers.
"""
import torch


def geomloss_sinkhorn_divergence_fixed_iters(X_t, Y_t, eps, n_iters):
    """Debiased Sinkhorn divergence via GeomLoss's own kernels, at a fixed
    iteration count instead of GeomLoss's default epsilon-annealing schedule.

    GeomLoss's SamplesLoss anneals eps from the point cloud's diameter down to
    the target blur, converging in ~8-30 steps for typical blur values -- not
    comparable to this experiment's other three arms, which all run a fixed
    n_iters regardless of when they'd otherwise stop. This calls the same
    softmin_tensorized/sinkhorn_loop primitives SamplesLoss uses internally,
    but with eps_list held constant at the target eps for the full n_iters,
    so every arm in the comparison spends the same fixed compute per step.

    GeomLoss's own p=2 cost is 0.5*||x-y||^2 (half of the ||x-y||^2 convention
    used by the other three arms and by sinkslot itself), so eps is halved
    before being fed to GeomLoss's kernel and the returned divergence is
    doubled at the end -- both the Gibbs kernel and the resulting gradient
    then match the other arms' convention exactly, not just up to a constant.
    """
    from geomloss._legacy.sinkhorn_divergence import log_weights, sinkhorn_cost, sinkhorn_loop
    from geomloss._legacy.sinkhorn_samples import cost_routines, softmin_tensorized

    n, m = X_t.shape[0], Y_t.shape[0]
    Xb, Yb = X_t.unsqueeze(0), Y_t.unsqueeze(0)
    a = torch.full((1, n), 1.0 / n, dtype=X_t.dtype, device=X_t.device)
    b = torch.full((1, m), 1.0 / m, dtype=Y_t.dtype, device=Y_t.device)

    cost = cost_routines[2]
    C_xy, C_yx = cost(Xb, Yb.detach()), cost(Yb, Xb.detach())
    C_xx, C_yy = cost(Xb, Xb.detach()), cost(Yb, Yb.detach())

    eps_geomloss = eps / 2.0
    eps_list = [eps_geomloss] * n_iters
    f_aa, g_bb, g_ab, f_ba = sinkhorn_loop(
        softmin_tensorized, log_weights(a), log_weights(b),
        C_xx, C_yy, C_xy, C_yx, eps_list, rho=None, debias=True,
    )
    div = sinkhorn_cost(eps_geomloss, None, a, b, f_aa, g_bb, g_ab, f_ba,
                         batch=True, debias=True, potentials=False)
    return 2.0 * div.squeeze(0)
