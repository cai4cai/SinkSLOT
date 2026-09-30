"""LCN-Sinkhorn (Gasteiger, Lienen, Günnemann, ICML 2021), ported from
github.com/gasteigerjo/lcn (lcn/lcn_sinkhorn.py, arg_log_lcn_sinkhorn) for one
problem instance, with the benchmark's weighted marginals and potential-change stop
rule added.

The kernel K = exp(-C/eps) is replaced by the locally corrected Nyström (LCN)
approximation
    K ~= U V + S,   U = K_xz,  V = pinv(K_zz) K_zy,
    S_ij = K_ij - (U V)_ij on a sparse set of pairs (i, j), 0 elsewhere,
with m landmarks z. Differences from the reference code:
  - weights: u = log a - LSE(...), v = log b - LSE(...); the reference code has
    no marginal terms (every point has mass 1);
  - landmarks: k-means++ seeding on the union of x and y (the reference config's
    "sampling_kmeanspp"), without Lloyd iterations;
  - sparse pairs: the k exact nearest neighbours of every x_i among y and of every
    y_j among x (the reference config uses LSH buckets from hierarchical k-means);
  - stop rule: max(|df|, |dg|) < tol between checkpoints (f = eps u, g = eps v),
    as for every other benchmarked method; the reference code runs a fixed niter;
  - no batching and no TorchScript; float64 throughout.

Sums of the approximate kernel can be negative (V has mixed signs). A row or
column whose sum is not positive makes the log undefined; the solve then stops
and reports failed=True.
"""

from typing import NamedTuple, Optional

import torch


def _signed_lse(logabs: torch.Tensor, sign: torch.Tensor, dim: int):
    """log|sum(sign * exp(logabs))| and its sign, along dim."""
    mx = logabs.amax(dim=dim, keepdim=True)
    mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
    s = (sign * torch.exp(logabs - mx)).sum(dim=dim)
    return torch.log(s.abs()) + mx.squeeze(dim), torch.sign(s)


def _sqdist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x.square().sum(1)[:, None] + y.square().sum(1)[None, :] - 2 * x @ y.T).clamp_min(0)


def kmeanspp_landmarks(pts: torch.Tensor, m: int, generator: torch.Generator) -> torch.Tensor:
    """k-means++ seeding: m points of pts, each drawn with probability proportional to
    its squared distance to the nearest point already chosen."""
    idx = [int(torch.randint(pts.shape[0], (1,), generator=generator, device=pts.device))]
    d2 = _sqdist(pts, pts[idx[0]][None]).squeeze(1)
    for _ in range(m - 1):
        i = int(torch.multinomial(d2 / d2.sum(), 1, generator=generator))
        idx.append(i)
        d2 = torch.minimum(d2, _sqdist(pts, pts[i][None]).squeeze(1))
    return pts[idx]


def knn_pairs(x: torch.Tensor, y: torch.Tensor, k: int, chunk: int = 2048):
    """(rows, cols) of the union of each x_i's k nearest y and each y_j's k nearest x."""
    n, m = x.shape[0], y.shape[0]
    keys = []
    for s in range(0, n, chunk):
        nn = _sqdist(x[s:s + chunk], y).topk(k, dim=1, largest=False).indices
        keys.append((torch.arange(s, s + nn.shape[0], device=x.device)[:, None] * m + nn).reshape(-1))
    for s in range(0, m, chunk):
        nn = _sqdist(y[s:s + chunk], x).topk(k, dim=1, largest=False).indices
        keys.append((nn * m + torch.arange(s, s + nn.shape[0], device=x.device)[:, None]).reshape(-1))
    keys = torch.unique(torch.cat(keys))
    return keys // m, keys % m


class LCNFactors(NamedTuple):
    log_u: torch.Tensor      # (n, L)  log U, U > 0
    log_v: torch.Tensor      # (L, m)  log |V|
    sign_v: torch.Tensor     # (L, m)
    rows: torch.Tensor       # (P,)    sparse correction pairs
    cols: torch.Tensor       # (P,)
    log_s: torch.Tensor      # (P,)    log |S_ij|
    sign_s: torch.Tensor     # (P,)


def lcn_factors(x, y, eps: float, landmarks: int, neighbors: int, seed: int = 0) -> LCNFactors:
    """LCN approximation of K = exp(-||x - y||^2 / eps), in float64."""
    x, y = x.double(), y.double()
    g = torch.Generator(device=x.device).manual_seed(seed)
    z = kmeanspp_landmarks(torch.cat([x, y]), landmarks, g)
    log_u = -_sqdist(x, z) / eps
    kzz = torch.exp(-_sqdist(z, z) / eps)
    pinv = torch.linalg.pinv(kzz, hermitian=True)
    s_zy = -_sqdist(z, y) / eps
    shift = s_zy.amax(dim=0, keepdim=True)          # per column, keeps exp() in range
    inner = pinv @ torch.exp(s_zy - shift)
    log_v = torch.log(inner.abs()) + shift
    sign_v = torch.sign(inner)
    rows, cols = knn_pairs(x, y, neighbors)
    log_exact = -((x[rows] - y[cols]).square().sum(1)) / eps
    log_approx, sign_approx = _signed_lse(log_u[rows] + log_v[:, cols].T, sign_v[:, cols].T, dim=1)
    log_s, sign_s = _signed_lse(torch.stack([log_exact, log_approx], 1),
                                torch.stack([torch.ones_like(sign_approx), -sign_approx], 1), dim=1)
    return LCNFactors(log_u, log_v, sign_v, rows, cols, log_s, sign_s)


def _log_kernel_times(f: LCNFactors, w: torch.Tensor, dim: int, size: int):
    """log (K~ exp(w)) along dim (dim=1: rows, sum over columns; dim=0: columns),
    as (log|sum|, sign)."""
    if dim == 1:
        log_w, sign_w = _signed_lse(f.log_v + w[None, :], f.sign_v, dim=1)        # (L,)
        low_log, low_sign = f.log_u + log_w[None, :], sign_w.expand_as(f.log_u)   # (n, L)
        corr_log, idx = f.log_s + w[f.cols], f.rows
    else:
        log_w = torch.logsumexp(f.log_u + w[:, None], dim=0)                      # (L,)
        low_log, low_sign = (f.log_v + log_w[:, None]).T, f.sign_v.T              # (m, L)
        corr_log, idx = f.log_s + w[f.rows], f.cols
    mx = low_log.amax(dim=1)
    mx = torch.maximum(mx, torch.full_like(mx, -torch.inf).scatter_reduce(0, idx, corr_log, "amax"))
    total = (low_sign * torch.exp(low_log - mx[:, None])).sum(1)
    total = total.index_add(0, idx, f.sign_s * torch.exp(corr_log - mx[idx]))
    return torch.log(total.abs()) + mx, torch.sign(total)


class LCNResult(NamedTuple):
    u: torch.Tensor          # f / eps
    v: torch.Tensor          # g / eps
    iters: int
    converged: bool
    failed: bool             # a row/column sum of the approximate kernel was not positive


def lcn_sinkhorn(f: LCNFactors, a, b, eps: float, max_iter: int, tol: float,
                 check_every: int = 5) -> LCNResult:
    """Alternating log-domain Sinkhorn on the LCN kernel; stops when
    max(|df|, |dg|) < tol between checkpoints, as in the benchmark."""
    log_a, log_b = a.double().log(), b.double().log()
    u = torch.zeros_like(log_a)
    v = torch.zeros_like(log_b)
    pu, pv = u, v
    for it in range(1, max_iter + 1):
        lr, sr = _log_kernel_times(f, v, 1, u.numel())
        u = log_a - lr
        lc, sc = _log_kernel_times(f, u, 0, v.numel())
        v = log_b - lc
        if it % check_every == 0:
            if bool((sr <= 0).any() or (sc <= 0).any() or not torch.isfinite(u).all()
                    or not torch.isfinite(v).all()):
                return LCNResult(u, v, it, False, True)
            change = eps * max(float((u - pu).abs().max()), float((v - pv).abs().max()))
            if change < tol:
                return LCNResult(u, v, it, True, False)
            pu, pv = u, v
    return LCNResult(u, v, max_iter, False, False)


def lcn_plan_metrics(f: LCNFactors, res: LCNResult, x, y, chunk: Optional[int] = None) -> dict:
    """<C, P>, row and column sums of P = diag(e^u) K~ diag(e^v), and the total
    negative mass of P, from the dense plan built in row chunks (float64)."""
    x, y = x.double(), y.double()
    n, m = x.shape[0], y.shape[0]
    chunk = chunk or max(1, int(1e8 // (f.log_u.shape[1] * m)))   # ~0.8 GB of float64 per chunk
    cost, neg = 0.0, 0.0
    r = torch.zeros(n, dtype=torch.float64, device=x.device)
    c = torch.zeros(m, dtype=torch.float64, device=x.device)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        lp, sp = _signed_lse(f.log_u[s:e, :, None] + f.log_v[None] + res.u[s:e, None, None]
                             + res.v[None, None, :], f.sign_v[None].expand(e - s, -1, -1), dim=1)
        p = sp * torch.exp(lp)
        sel = (f.rows >= s) & (f.rows < e)
        p.index_put_((f.rows[sel] - s, f.cols[sel]),
                     f.sign_s[sel] * torch.exp(f.log_s[sel] + res.u[f.rows[sel]] + res.v[f.cols[sel]]),
                     accumulate=True)
        cost += float((_sqdist(x[s:e], y) * p).sum())
        neg += float(p.clamp_max(0).sum())
        r[s:e] = p.sum(1)
        c += p.sum(0)
    return {"plan_cost": cost, "row_sums": r, "col_sums": c, "negative_mass": -neg}
