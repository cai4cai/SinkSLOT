"""Qualitative color-transfer figure: source, target, and the transferred
image (source recolored via each method's own barycentric projection),
for a handful of representative image pairs -- the real replacement for
the placeholder figure borrowed from defense.tex in iclr_v0.tex Section 5.2.

    python -m color_transfer.export_qualitative --output_dir DIR

Reconstructing an image from a solved OT plan: main_pixel.py's
pixels_and_weights() collapses each image down to its UNIQUE colors before
solving (duplicate pixels share one point), which is exactly what makes
this problem tractable at real-image scale -- but it also means the
solved plan only assigns each unique color a new color, not each pixel
directly. This module adds the missing piece: torch.unique's own
return_inverse gives, for every original pixel, which row of the unique
array it came from, so the transferred image is just the transferred
*colors* array indexed back out through that inverse map, reshaped to the
original (H, W, 3) layout. Never solves for a different point cloud than
main_pixel.py's own sweep does, so the images shown here are literally the
same objects those numbers were measured on.

Barycentric projection T(x_i) = (1/a_i) * sum_j P_ij * y_j, the same
"assign each source point the mass-weighted average of the target colors
it transports to" rule used throughout entropic OT (e.g. Peyre & Cuturi's
textbook). Computed directly from each method's own solved representation
(FlashSinkhorn/GeomLoss's (f, g); SinkSLOT's (phi, psi) over its sparse
support), row-blocked for the dense methods so no (n, m) tensor is ever
materialized, mirroring _cost_dense_chunked's own pattern.
"""

import argparse
import os

import torch
from PIL import Image

from color_transfer.main_pixel import (
    DEFAULT_PAINTINGS_DIR,
    StopCfg,
    _GEOMLOSS_BACKENDS,
    list_images,
)


def load_pixels_weights_inverse(path, device, dtype):
    """Like main_pixel.pixels_and_weights, but also returns `inverse`
    (per-original-pixel index into the unique-color array, from
    torch.unique's own return_inverse) and `shape` (H, W) -- everything
    needed to reconstruct a full image from a per-unique-color result."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    raw = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).view(-1, 3).to(device)
    total = raw.shape[0]
    uniq, inverse, counts = torch.unique(raw, dim=0, return_inverse=True, return_counts=True)
    pixels = uniq.to(dtype) / 255.0
    weights = counts.to(dtype) / total
    return pixels, weights, inverse.reshape(-1), (h, w)


def _barycentric_dense_chunked(f, g, sc, tc, sw, tw, eps, block_n=2048):
    """T(x_i) for FlashSinkhorn's/GeomLoss's shared same-as-cost-scale (f, g)
    convention: P_ij = a_i*b_j*exp((f_i+g_j-C_ij)/eps), so
    T_i = sum_j b_j*exp((f_i+g_j-C_ij)/eps)*y_j / sum_j b_j*exp((f_i+g_j-C_ij)/eps)
    (the a_i factor cancels; the denominator re-normalizes by the row's own
    mass instead of assuming it is exactly a_i, so this is well-defined even
    from an unconverged, not-quite-feasible plan). Row-blocked, mirroring
    _cost_dense_chunked -- never materializes a dense (n, m) tensor."""
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


def _barycentric_sparse(phi, psi, rows, cols, S, cost, tc, eps, n):
    """T(x_i) for SinkSLOT's sparse (phi, psi) convention, restricted to its
    own support (rows, cols, S): P_ij = exp(phi_i+psi_j+log(S_ij)-C_ij/eps)
    for (i,j) in the support, 0 elsewhere. Row-normalized by the support's
    own row mass (same feasibility caveat as the dense case above)."""
    tiny = torch.finfo(cost.dtype).tiny
    log_S = S.clamp_min(tiny).log()
    vals = (phi[rows] + psi[cols] + log_S - cost / eps).exp()
    d = tc.shape[1]
    row_sum = torch.zeros(n, device=tc.device, dtype=vals.dtype).index_add_(0, rows, vals).clamp_min(tiny)
    weighted = torch.zeros((n, d), device=tc.device, dtype=vals.dtype)
    weighted.index_add_(0, rows, vals[:, None] * tc[cols])
    return weighted / row_sum[:, None]


def solve_and_project(method, sc, tc, sw, tw, eps, max_iter, tol, check_every,
                       geomloss_backend="online", sinkslot_L=100):
    """Runs `method` to convergence (marginal violation, matching the rest
    of this PR's sweep) and returns T(source_colors) -- the barycentrically
    projected color for every unique source color, in [0, 1]^3."""
    stop = StopCfg(mode="marginal", max_iter=max_iter, check_every=check_every, tol=tol)
    n = sc.shape[0]

    if method == "sinkslot":
        from sinkslot.sinkhorn_solvers import sinkslot_alternating_triton
        from sinkslot.solver import sot_plan_coo, sparse_sqeuclidean_cost, to_csr
        rows, cols, S = sot_plan_coo(sc, tc, sw, tw, L=sinkslot_L, seed=0)
        cost = sparse_sqeuclidean_cost(sc, tc, rows, cols)
        log_S = S.clamp_min(torch.finfo(S.dtype).tiny).log()
        lam = log_S - cost / eps
        r_ptr, r_idx, r_lam, _ = to_csr(rows, cols, lam, sc.shape[0])
        c_ptr, c_idx, c_lam, _ = to_csr(cols, rows, lam, tc.shape[0])
        phi, psi, it, converged, viol = sinkslot_alternating_triton(
            r_ptr, r_idx, r_lam, c_ptr, c_idx, c_lam, sw.log(), tw.log(),
            sc.shape[0], tc.shape[0], max_iter, stop=stop)
        print(f"    sinkslot: iters={it} converged={converged} viol={viol}")
        return _barycentric_sparse(phi, psi, rows, cols, S, cost, tc, eps, n)

    elif method in ("flashsinkhorn", "flashsinkhorn_symmetric"):
        symmetric = method == "flashsinkhorn_symmetric"
        if symmetric:
            from flash_sinkhorn.sinkhorn_solvers import sinkhorn_flashstyle_symmetric
            f, g, it = sinkhorn_flashstyle_symmetric(
                sc, tc, sw, tw, use_epsilon_scaling=False, eps=eps, n_iters=max_iter,
                threshold=tol, check_every=check_every, stop_mode="marginal",
                allow_tf32=False, return_n_iters=True)
        else:
            from flash_sinkhorn.sinkhorn_solvers import sinkhorn_flashstyle_alternating
            f, g, it = sinkhorn_flashstyle_alternating(
                sc, tc, sw, tw, eps=eps, n_iters=max_iter, threshold=tol,
                check_every=check_every, stop_mode="marginal",
                allow_tf32=False, return_n_iters=True)
        print(f"    {method}: iters={it}")
        return _barycentric_dense_chunked(f, g, sc, tc, sw, tw, eps)

    else:  # geomloss
        solve = _GEOMLOSS_BACKENDS[geomloss_backend]
        f, g = solve(sc, tc, sw, tw, eps, max_iter)
        return _barycentric_dense_chunked(f, g, sc, tc, sw, tw, eps)


def transfer_image(source_path, target_path, method, eps, max_iter, tol, check_every,
                    device, dtype, geomloss_backend="online", sinkslot_L=100):
    """Full pipeline: solve OT between source's and target's unique colors,
    barycentrically project, and reconstruct a full-resolution transferred
    image (as a uint8 numpy-free torch tensor, (H, W, 3))."""
    sc, sw, s_inverse, (h, w) = load_pixels_weights_inverse(source_path, device, dtype)
    tc, tw, _, _ = load_pixels_weights_inverse(target_path, device, dtype)

    T = solve_and_project(method, sc, tc, sw, tw, eps, max_iter, tol, check_every,
                           geomloss_backend=geomloss_backend, sinkslot_L=sinkslot_L)
    T = T.clamp(0.0, 1.0)
    pixels = T[s_inverse]  # (H*W, 3), each original pixel's new color
    img_tensor = (pixels.reshape(h, w, 3) * 255.0).round().to(torch.uint8).cpu()
    return img_tensor


def save_image(img_tensor, path):
    Image.fromarray(img_tensor.numpy(), mode="RGB").save(path)


def load_as_tensor(path):
    """Raw (H, W, 3) uint8 tensor of an image, for saving the untouched
    source/target alongside each method's transferred version."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    raw = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
    return raw.reshape(h, w, 3)


_METHODS = [
    ("sinkslot", "sinkslot"),
    ("flashsinkhorn", "flashsinkhorn"),
    ("flashsinkhorn_symmetric", "flashsinkhorn_symmetric"),
    ("geomloss_online", "geomloss"),
    ("geomloss_multiscale", "geomloss"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--paintings_dir", type=str, default=str(DEFAULT_PAINTINGS_DIR))
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--max_iter", type=int, default=5000)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--check_every", type=int, default=500)
    p.add_argument("--sinkslot_L", type=int, default=100)
    p.add_argument("--pairs", type=str, nargs="+", default=[
        "antibes-in-the-morning.jpg:argenteuil-yachts-02.jpg",
        "haystacks-at-giverny.jpg:water-lilies-evening-effect-1899.jpg",
        "the-toques-at-saint-arnoult-1891.jpg:banks-of-rivers-the-thames-hampton-court-first-week-of-october-1874.jpg",
    ], help="source.jpg:target.jpg pairs (filenames within --paintings_dir).")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This script requires a CUDA GPU.")
    device = torch.device(args.device)
    dtype = torch.float32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    os.makedirs(args.output_dir, exist_ok=True)

    for pair_str in args.pairs:
        src_name, tgt_name = pair_str.split(":")
        source_path = os.path.join(args.paintings_dir, src_name)
        target_path = os.path.join(args.paintings_dir, tgt_name)
        pair_stem = f"{os.path.splitext(src_name)[0]}__{os.path.splitext(tgt_name)[0]}"
        print(f"Pair: {src_name} -> {tgt_name}")

        save_image(load_as_tensor(source_path), os.path.join(args.output_dir, f"{pair_stem}_source.png"))
        save_image(load_as_tensor(target_path), os.path.join(args.output_dir, f"{pair_stem}_target.png"))

        for stem, method in _METHODS:
            print(f"  {stem} ...")
            geomloss_backend = "multiscale" if stem == "geomloss_multiscale" else "online"
            img_tensor = transfer_image(
                source_path, target_path, method, args.eps, args.max_iter, args.tol,
                args.check_every, device, dtype, geomloss_backend=geomloss_backend,
                sinkslot_L=args.sinkslot_L)
            out_path = os.path.join(args.output_dir, f"{pair_stem}_{stem}.png")
            save_image(img_tensor, out_path)
            print(f"    Saved: {out_path}")


if __name__ == "__main__":
    main()
