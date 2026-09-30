"""LCN-Sinkhorn and sparse Sinkhorn through the authors' own package
(github.com/gasteigerjo/lcn, Gasteiger, Lienen, Günnemann, ICML 2021).

The package builds the approximate cost matrices (its get_cost_matrix, with its own
landmark / neighbour selection) and supplies the per-iteration log-sum-exp kernels
(sparse_lse_uv, lcn_lse_uv). What is added here, without editing the package:
  - import shims: the package targets torch==1.4 and needs torch_scatter and munch.
    torch.jit.script is replaced by the identity while importing (its scripted
    functions are documented as broken on torch>=1.8), and pure-torch stand-ins for
    the four torch_scatter functions it calls (and for munch.Munch, if munch is not
    installed) are registered;
  - SqEuclidean, a Distance of the package's interface for the benchmark's cost
    ||x - y||^2 (the package ships p-norms, not squared ones);
  - the Sinkhorn loop: the package's arg_log_*_sinkhorn run a fixed niter with
    uniform marginals from a zero start, so the loop is rewritten around the
    package's own update kernels with u = log a - LSE(...), v = log b - LSE(...)
    and the benchmark's stop rule max(|df|, |dg|) < tol every check_every
    iterations.
A row or column sum that is not positive (LCN: signed Nyström factor; sparse: a
point with no kept pair) makes the log undefined; the solve then stops, failed=True.
"""

import sys
import types
from typing import NamedTuple, Optional

import torch

from sinkslot.bench.lcn_sinkhorn import LCNFactors, LCNResult, lcn_plan_metrics


def _scatter(src, index, dim=-1, out=None, dim_size=None, reduce="sum"):
    dim = dim % src.dim()
    shape = list(src.shape)
    if index.dim() != src.dim():
        view = [1] * src.dim()
        view[dim] = -1
        index = index.view(view)
    index = index.expand_as(src)
    if out is None:
        shape[dim] = int(dim_size if dim_size is not None else int(index.max()) + 1)
        out = torch.zeros(shape, dtype=src.dtype, device=src.device)
        include_self = False
    else:
        include_self = True
    op = {"sum": "sum", "add": "sum", "mean": "mean", "max": "amax", "min": "amin"}[reduce]
    return out.scatter_reduce_(dim, index, src, op, include_self=include_self)


def _scatter_logsumexp(src, index, dim=-1, out=None, dim_size=None):
    mx = _scatter(src, index, dim, None, dim_size if out is None else out.shape[dim], "max")
    dim = dim % src.dim()
    idx = index.view([-1 if i == dim else 1 for i in range(src.dim())]).expand_as(src) \
        if index.dim() != src.dim() else index
    s = _scatter((src - mx.gather(dim, idx)).exp(), index, dim, None, mx.shape[dim], "sum")
    res = s.log() + mx
    if out is not None:
        out.copy_(torch.logaddexp(out, res))
        return out
    return res


def _segment_coo(src, index, out=None, dim_size=None, reduce="sum"):
    return _scatter(src, index, -1, out, dim_size, reduce)


def _segment_csr(src, indptr, out=None, reduce="sum"):
    index = torch.repeat_interleave(torch.arange(indptr.numel() - 1, device=src.device),
                                    indptr[1:] - indptr[:-1])
    return _scatter(src, index, -1, out, indptr.numel() - 1, reduce)


# Settings of the authors' embedding-alignment experiments (configs/embedding_alignment.yaml).
NYSTROM_CONFIG = {"landmark_method": "sampling_kmeanspp", "num_clusters": 20}           # "lcn"
SPARSE_CONFIGS = {
    "kmeans_hier": {"method": "lsh", "neighbor_method": "kmeans_hier", "num_clusters": [10, 100],
                    "num_hash_bands": 1, "num_hashes_per_band": 1},                        # "lcn"
    "angular_lsh": {"method": "lsh", "neighbor_method": "angular_lsh", "num_clusters": 130,
                    "num_hash_bands": 16, "num_hashes_per_band": 2},                      # "sparse"
}


def import_lcn(path: str):
    """Import the lcn package from its source checkout at `path`."""
    if "torch_scatter" not in sys.modules:
        ts = types.ModuleType("torch_scatter")
        ts.scatter, ts.segment_coo, ts.segment_csr = _scatter, _segment_coo, _segment_csr
        ts.composite = types.SimpleNamespace(scatter_logsumexp=_scatter_logsumexp)
        sys.modules["torch_scatter"] = ts
    try:
        import munch  # noqa: F401
    except ImportError:
        m = types.ModuleType("munch")

        class Munch(dict):
            __getattr__ = dict.__getitem__
            __setattr__ = dict.__setitem__

            def copy(self):
                return Munch(self)

        m.Munch = Munch
        sys.modules["munch"] = m
    torch.jit.script = lambda fn=None, *a, **k: fn
    if path not in sys.path:
        sys.path.insert(0, path)
    import lcn.cost.cost_matrix
    import lcn.cost.distances
    import lcn.lcn_sinkhorn
    import lcn.sparse_sinkhorn
    import lcn.utils
    return lcn


def sq_euclidean(lcn):
    class SqEuclidean(lcn.cost.distances.Distance):
        def norm(self, x):
            return x.square().sum(-1)

        def cdist(self, x1, x2):
            return torch.cdist(x1, x2).square()

        def pairwise_distance(self, x1, x2):
            return (x1 - x2).square().sum(-1)
    return SqEuclidean()


def cost_matrix(lcn, x, y, eps: float, nystrom: Optional[dict], sparse: Optional[dict]):
    """The package's approximate cost matrix for one problem (batch of 1), in float32 as
    the package expects. Clustering and landmark selection use its Euclidean PNorm(2)."""
    emb = [x.float()[None], y.float()[None]]
    num_points = torch.tensor([[x.shape[0]], [y.shape[0]]], device=x.device)
    reg = torch.tensor([eps], dtype=torch.float32, device=x.device)
    return lcn.cost.cost_matrix.get_cost_matrix(
        emb, num_points, nystrom=nystrom, sparse=sparse, sinkhorn_reg=reg, sinkhorn_niter=50,
        alpha=None, dist=sq_euclidean(lcn), dist_cluster=lcn.cost.distances.PNorm(p=2),
        bp_cost_matrix=False)


class _Stop(NamedTuple):
    max_iter: int
    tol: float
    check_every: int


def _loop(update_u, update_v, n, m, eps, stop: _Stop, device):
    u = torch.zeros(n, dtype=torch.float64, device=device)
    v = torch.zeros(m, dtype=torch.float64, device=device)
    pu, pv = u, v
    for it in range(1, stop.max_iter + 1):
        u = update_u(v)
        v = update_v(u)
        if it % stop.check_every == 0:
            if not (torch.isfinite(u).all() and torch.isfinite(v).all()):
                return LCNResult(u, v, it, False, True)
            change = eps * max(float((u - pu).abs().max()), float((v - pv).abs().max()))
            if change < stop.tol:
                return LCNResult(u, v, it, True, False)
            pu, pv = u, v
    return LCNResult(u, v, stop.max_iter, False, False)


def sparse_solve(lcn, cm, a, b, eps, max_iter, tol, check_every=5):
    """Sparse Sinkhorn on the package's sparse cost matrix, with its sparse_lse_uv."""
    lse = lcn.sparse_sinkhorn.sparse_lse_uv
    sim = -cm.costs.double() / eps
    log_a, log_b = a.double().log(), b.double().log()
    return _loop(
        lambda v: log_a - lse(sim, cm.cost_idx1, cm.cost_idx2, cm.norms1_batch_idx, v, dim=2),
        lambda u: log_b - lse(sim, cm.cost_idx1, cm.cost_idx2, cm.norms2_batch_idx, u, dim=1),
        a.numel(), b.numel(), eps, _Stop(max_iter, tol, check_every), a.device)


def sparse_metrics(cm, res: LCNResult, eps) -> dict:
    costs = cm.costs.double()
    p = torch.exp(-costs / eps + res.u[cm.cost_idx1] + res.v[cm.cost_idx2])
    n, m = res.u.numel(), res.v.numel()
    return {"plan_cost": float((p * costs).sum()),
            "row_sums": torch.zeros(n, dtype=p.dtype, device=p.device).index_add(0, cm.cost_idx1, p),
            "col_sums": torch.zeros(m, dtype=p.dtype, device=p.device).index_add(0, cm.cost_idx2, p),
            "negative_mass": 0.0, "pairs": int(cm.costs.numel())}


def _lcn_parts(lcn, cm, eps):
    """float64 copies of the package's (float32) LCN factors, and its correction terms."""
    sim_1a = -cm.cost_1a.double() / eps
    sim_exact = -cm.cost_exact.double() / eps
    sim_corr, corr_sign = lcn.utils.logdiffexp(sim_exact, cm.sim_approx_scaled.double(),
                                               cm.sign_approx.double())
    return sim_1a, sim_corr, corr_sign


def lcn_solve(lcn, cm, a, b, eps, max_iter, tol, check_every=5):
    """LCN-Sinkhorn on the package's LCN cost matrix, with its lcn_lse_uv."""
    lse = lcn.lcn_sinkhorn.lcn_lse_uv
    sim_1a, sim_corr, corr_sign = _lcn_parts(lcn, cm, eps)
    args = (sim_1a, cm.sim_a2_scaled.double(), cm.sign_a2.double(), sim_corr, corr_sign, cm.corr_batch_idx,
            cm.corr_idx1, cm.corr_idx2, cm.corr_idx1_nooffset, cm.corr_idx2_nooffset,
            cm.norms1_batch_idx, cm.norms1_idx, cm.norms2_batch_idx, cm.norms2_idx)
    log_a, log_b = a.double().log()[None], b.double().log()[None]
    res = _loop(lambda v: (log_a - lse(*args, v[None], dim=2))[0],
                lambda u: (log_b - lse(*args, u[None], dim=1))[0],
                a.numel(), b.numel(), eps, _Stop(max_iter, tol, check_every), a.device)
    return res


def lcn_metrics(lcn, cm, res: LCNResult, x, y, eps) -> dict:
    """Plan metrics of the LCN plan, with the package's factors in LCNFactors form."""
    sim_1a, sim_corr, corr_sign = _lcn_parts(lcn, cm, eps)
    f = LCNFactors(sim_1a[0], cm.sim_a2_scaled[0].double(), cm.sign_a2[0].double(),
                   cm.corr_idx1_nooffset, cm.corr_idx2_nooffset, sim_corr,
                   corr_sign.to(torch.float64))
    out = lcn_plan_metrics(f, res, x, y)
    out["pairs"] = int(cm.cost_exact.numel())
    return out
