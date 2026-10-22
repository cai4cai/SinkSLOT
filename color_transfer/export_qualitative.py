"""Qualitative color-transfer figure: source, target, and the transferred
image (source recolored via each method's own barycentric projection),
under this PR's potential-change convergence criterion.

    python -m color_transfer.export_qualitative --output_dir DIR

pixels_and_weights() (see main.py) collapses each image to its unique
colors before solving; this module's load_pixels_weights_inverse also
keeps torch.unique's own return_inverse map, so the transferred image is
just the transferred *colors* array indexed back out through it, reshaped
to the original (H, W, 3) layout.

Barycentric projection T(x_i) = (1/a_i) * sum_j P_ij * y_j, computed
directly from each method's own solved representation (FlashSinkhorn's
(f, g); SinkSLOT's (phi, psi) over its sparse support), row-blocked for the
dense case so no (n, m) tensor is ever materialized.
"""

import argparse
import os

import torch
from PIL import Image

import color_transfer.main as ct
from color_transfer.main import DEFAULT_PAINTINGS_DIR, StopCfg, list_images
from sinkslot.bench.devices import default_device, describe, methods_for_device
from sinkslot.bench.reference_solvers import geomloss_multiscale, geomloss_online, kmeans_labels


def load_pixels_weights_inverse(path, device, dtype):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    raw = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).view(-1, 3).to(device)
    total = raw.shape[0]
    uniq, inverse, counts = torch.unique(raw, dim=0, return_inverse=True, return_counts=True)
    pixels = uniq.to(dtype) / 255.0
    weights = counts.to(dtype) / total
    return pixels, weights, inverse.reshape(-1), (h, w)


def _barycentric_dense_chunked(f, g, sc, tc, sw, tw, eps, block_n=2048):
    n = sc.shape[0]
    d = tc.shape[1]
    T = torch.empty((n, d), device=sc.device, dtype=sc.dtype)
    for start in range(0, n, block_n):
        end = min(start + block_n, n)
        C_blk = torch.cdist(sc[start:end][None], tc[None], p=2).pow(2)[0]
        W_blk = tw[None, :] * torch.exp((f[start:end, None] + g[None, :] - C_blk) / eps)
        row_sum = W_blk.sum(1, keepdim=True).clamp_min(torch.finfo(sc.dtype).tiny)
        T[start:end] = (W_blk @ tc) / row_sum
        del C_blk, W_blk
    return T


def _barycentric_kept(f, g, sc, tc, sw, tw, eps, support, block_n=2048):
    """_barycentric_dense_chunked over the entries a truncated GeomLoss multiscale kernel kept
    (support = (label_x, label_y, keep); None: every pair). Dropped pairs are masked before exp."""
    if support is None:
        return _barycentric_dense_chunked(f, g, sc, tc, sw, tw, eps, block_n)
    label_x, label_y, keep = support
    T = torch.empty((sc.shape[0], tc.shape[1]), device=sc.device, dtype=sc.dtype)
    for start in range(0, sc.shape[0], block_n):
        end = min(start + block_n, sc.shape[0])
        C_blk = torch.cdist(sc[start:end][None], tc[None], p=2).pow(2)[0]
        logits = ((f[start:end, None] + g[None, :] - C_blk) / eps).masked_fill(
            ~keep[label_x[start:end]][:, label_y], float("-inf"))
        W_blk = tw[None, :] * logits.exp()
        T[start:end] = (W_blk @ tc) / W_blk.sum(1, keepdim=True).clamp_min(torch.finfo(sc.dtype).tiny)
        del C_blk, W_blk, logits
    return T


def _barycentric_sparse(phi, psi, rows, cols, S, cost, tc, eps, n):
    tiny = torch.finfo(cost.dtype).tiny
    log_S = S.clamp_min(tiny).log()
    vals = (phi[rows] + psi[cols] + log_S - cost / eps).exp()
    d = tc.shape[1]
    row_sum = torch.zeros(n, device=tc.device, dtype=vals.dtype).index_add_(0, rows, vals).clamp_min(tiny)
    weighted = torch.zeros((n, d), device=tc.device, dtype=vals.dtype)
    weighted.index_add_(0, rows, vals[:, None] * tc[cols])
    return weighted / row_sum[:, None]


def solve_and_project(method, sc, tc, sw, tw, eps, max_iter, tol, check_every, sinkslot_L=100,
                      sym_alpha=0.5, geomloss_truncate=5.0, geomloss_points_per_cluster=None):
    n = sc.shape[0]
    if method == "sinkslot":
        from sinkslot.sinkhorn_solvers import sinkslot_alternating_fused
        from sinkslot.solver import (
            ot1d_for, sot_plan_coo, sparse_sqeuclidean_cost, to_csr,
        )
        ot1d = ot1d_for(sc.device)
        rows, cols, S = sot_plan_coo(sc, tc, sw, tw, L=sinkslot_L, seed=0, ot1d=ot1d)
        cost = sparse_sqeuclidean_cost(sc, tc, rows, cols)
        log_S = S.clamp_min(torch.finfo(S.dtype).tiny).log()
        lam = log_S - cost / eps
        r_ptr, r_idx, r_lam, _ = to_csr(rows, cols, lam, sc.shape[0])
        c_ptr, c_idx, c_lam, _ = to_csr(cols, rows, lam, tc.shape[0])
        phi, psi, it, converged, change = sinkslot_alternating_fused(
            r_ptr, r_idx, r_lam, c_ptr, c_idx, c_lam, sw.log(), tw.log(),
            sc.shape[0], tc.shape[0], max_iter,
            stop=StopCfg(mode="potential", max_iter=max_iter, check_every=check_every, tol=tol), eps=eps)
        print(f"    sinkslot: iters={it} converged={converged} change={change:.3e}")
        return _barycentric_sparse(phi, psi, rows, cols, S, cost, tc, eps, n)

    if method == "geomloss_multiscale":
        # As color_transfer/main.py: grid clustering, or K-means with round(n / points_per_cluster) clusters.
        labels = None if geomloss_points_per_cluster is None else tuple(
            kmeans_labels(z, max(1, round(len(z) / geomloss_points_per_cluster))) for z in (sc, tc))
        f, g, it, converged, cost_val, _, info = geomloss_multiscale(
            sc, tc, sw, tw, eps, max_iter, threshold=tol, check_every=check_every, truncate=geomloss_truncate,
            alpha=sym_alpha, labels=labels)
        print(f"    geomloss_multiscale: iters={it} converged={converged} cost={cost_val:.6f} "
              f"kept={info['kept_fraction']:.3f}")
        return _barycentric_kept(f, g, sc, tc, sw, tw, eps, info["support"])

    if method == "geomloss_online":
        # Damped by sym_alpha; stops at sym_alpha * tol, as in color_transfer/main.py.
        f, g, it, converged, cost_val, _ = geomloss_online(
            sc, tc, sw, tw, eps, max_iter, threshold=tol, check_every=check_every, alpha=sym_alpha)
        print(f"    geomloss_online: iters={it} converged={converged} cost={cost_val:.6f}")
        return _barycentric_dense_chunked(f, g, sc, tc, sw, tw, eps)

    symmetric = method == "flashsinkhorn_symmetric"
    if symmetric:
        # Damped by sym_alpha (ct.SYM_ALPHA): the library's solver at 0.5, flash_symmetric_alpha otherwise.
        f, g, it = ct._flash_symmetric(sc, tc, sw, tw, eps=eps, n_iters=max_iter, threshold=sym_alpha * tol,
                                       check_every=check_every, allow_tf32=True)
        it = int(it) - 1  # it counts the initial alpha=1 step
    else:
        from flash_sinkhorn.sinkhorn_solvers import sinkhorn_flashstyle_alternating
        f, g, it = sinkhorn_flashstyle_alternating(
            sc, tc, sw, tw, eps=eps, n_iters=max_iter, threshold=tol,
            check_every=check_every, return_n_iters=True,
        )
    it = int(it)
    converged = it < max_iter
    cost_val = float((sw * f).sum() + (tw * g).sum())
    print(f"    {method}: iters={it} converged={converged} cost={cost_val:.6f}")
    return _barycentric_dense_chunked(f, g, sc, tc, sw, tw, eps)


def transfer_image(source_path, target_path, method, eps, max_iter, tol, check_every,
                    device, dtype, sinkslot_L=100, **solver_kw):
    sc, sw, s_inverse, (h, w) = load_pixels_weights_inverse(source_path, device, dtype)
    tc, tw, _, _ = load_pixels_weights_inverse(target_path, device, dtype)
    T = solve_and_project(method, sc, tc, sw, tw, eps, max_iter, tol, check_every, sinkslot_L=sinkslot_L, **solver_kw)
    T = T.clamp(0.0, 1.0)
    pixels = T[s_inverse]
    img_tensor = (pixels.reshape(h, w, 3) * 255.0).round().to(torch.uint8).cpu()
    return img_tensor


def save_image(img_tensor, path):
    Image.fromarray(img_tensor.numpy(), mode="RGB").save(path)


def load_as_tensor(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    raw = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
    return raw.reshape(h, w, 3)


_METHODS = [
    ("sinkslot", "sinkslot"),
    ("flashsinkhorn", "flashsinkhorn"),
    ("flashsinkhorn_symmetric", "flashsinkhorn_symmetric"),
    ("geomloss_online", "geomloss_online"),
    ("geomloss_multiscale", "geomloss_multiscale"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--paintings_dir", type=str, default=str(DEFAULT_PAINTINGS_DIR))
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--max_iter", type=int, default=8000)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--check_every", type=int, default=5)
    p.add_argument("--sinkslot_L", type=int, default=100)
    p.add_argument("--pair_idx", type=int, nargs="+", default=[2, 9],
                    help="One or more ordered pairs, flattened: 'i1 j1 i2 j2 ...' "
                         "(e.g. --pair_idx 2 9 5 7 for two pairs). Each pair gets its own "
                         "set of output images, named by its own source/target stem.")
    p.add_argument("--sym_alpha", type=float, default=0.5,
                   help="Damping of FlashSinkhorn symmetric and GeomLoss (0.5 the libraries' default, 0.9 the paper's).")
    p.add_argument("--geomloss_truncate", type=float, default=5.0,
                   help="GeomLoss multiscale: kernel truncation margin, in units of eps (GeomLoss's default 5; "
                        "the paper uses 1).")
    p.add_argument("--geomloss_points_per_cluster", type=int, default=None,
                   help="GeomLoss multiscale: K-means with round(n / this) clusters per image (the paper: 250); "
                        "default GeomLoss's grid clustering.")
    p.add_argument("--device", type=str, default=None,
                    help="Default: the best device present (CUDA, then MPS, then CPU). Only "
                         "SinkSLOT runs off CUDA.")
    p.add_argument("--methods", type=str, nargs="+", default=None,
                    choices=[stem for stem, _ in _METHODS],
                    help="Subset of methods to run (default: all).")
    return p.parse_args()


def main():
    args = parse_args()
    if len(args.pair_idx) % 2 != 0:
        raise ValueError(f"--pair_idx must have an even number of ints, got {args.pair_idx}")
    device = torch.device(args.device or default_device())
    stems = methods_for_device(device, args.methods, [s for s, _ in _METHODS])
    print(f"Device: {describe(device)}")
    dtype = torch.float32

    ct.SYM_ALPHA = args.sym_alpha  # read by ct._flash_symmetric
    os.makedirs(args.output_dir, exist_ok=True)
    paths = list_images(args.paintings_dir)
    pairs = list(zip(args.pair_idx[0::2], args.pair_idx[1::2]))

    for i, j in pairs:
        source_path, target_path = paths[i], paths[j]
        src_name, tgt_name = os.path.basename(source_path), os.path.basename(target_path)
        pair_stem = f"{os.path.splitext(src_name)[0]}__{os.path.splitext(tgt_name)[0]}"
        print(f"Pair: {src_name} -> {tgt_name}")

        save_image(load_as_tensor(source_path), os.path.join(args.output_dir, f"{pair_stem}_source.png"))
        save_image(load_as_tensor(target_path), os.path.join(args.output_dir, f"{pair_stem}_target.png"))

        methods = [(s, m) for s, m in _METHODS if s in stems]
        for stem, method in methods:
            print(f"  {stem} ...")
            img_tensor = transfer_image(
                source_path, target_path, method, args.eps, args.max_iter, args.tol,
                args.check_every, device, dtype, sinkslot_L=args.sinkslot_L, sym_alpha=args.sym_alpha,
                geomloss_truncate=args.geomloss_truncate,
                geomloss_points_per_cluster=args.geomloss_points_per_cluster)
            out_path = os.path.join(args.output_dir, f"{pair_stem}_{stem}.png")
            save_image(img_tensor, out_path)
            print(f"    Saved: {out_path}")


if __name__ == "__main__":
    main()
