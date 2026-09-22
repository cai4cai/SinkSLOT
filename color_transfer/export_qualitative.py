"""Qualitative color-transfer figure: source, target, and the transferred
image (source recolored via each method's own barycentric projection),
under this PR's potential-change convergence criterion.

    python -m color_transfer.export_qualitative --output_dir DIR

pixels_and_weights() (see trajectory_potential.py) collapses each image to
its unique colors before solving; this module's load_pixels_weights_inverse
also keeps torch.unique's own return_inverse map, so the transferred image
is just the transferred *colors* array indexed back out through it,
reshaped to the original (H, W, 3) layout.

Barycentric projection T(x_i) = (1/a_i) * sum_j P_ij * y_j, computed
directly from each method's own solved representation (FlashSinkhorn's
(f, g); SinkSLOT's (phi, psi) over its sparse support), row-blocked for the
dense case so no (n, m) tensor is ever materialized.
"""

import argparse
import os

import torch
from PIL import Image

from color_transfer.trajectory_potential import DEFAULT_PAINTINGS_DIR, StopCfg, list_images
from sinkslot.bench.reference_solvers import flashsinkhorn_native_run


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


def _barycentric_sparse(phi, psi, rows, cols, S, cost, tc, eps, n):
    tiny = torch.finfo(cost.dtype).tiny
    log_S = S.clamp_min(tiny).log()
    vals = (phi[rows] + psi[cols] + log_S - cost / eps).exp()
    d = tc.shape[1]
    row_sum = torch.zeros(n, device=tc.device, dtype=vals.dtype).index_add_(0, rows, vals).clamp_min(tiny)
    weighted = torch.zeros((n, d), device=tc.device, dtype=vals.dtype)
    weighted.index_add_(0, rows, vals[:, None] * tc[cols])
    return weighted / row_sum[:, None]


def solve_and_project(method, sc, tc, sw, tw, eps, max_iter, tol, check_every, sinkslot_L=100):
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
        phi, psi, it, converged, change = sinkslot_alternating_triton(
            r_ptr, r_idx, r_lam, c_ptr, c_idx, c_lam, sw.log(), tw.log(),
            sc.shape[0], tc.shape[0], max_iter,
            stop=StopCfg(mode="potential", max_iter=max_iter, check_every=check_every, tol=tol), eps=eps)
        print(f"    sinkslot: iters={it} converged={converged} change={change:.3e}")
        return _barycentric_sparse(phi, psi, rows, cols, S, cost, tc, eps, n)

    symmetric = method == "flashsinkhorn_symmetric"
    f, g, it, converged, cost_val = flashsinkhorn_native_run(
        sc, tc, sw, tw, eps, max_iter, threshold=tol, check_every=check_every, symmetric=symmetric)
    print(f"    {method}: iters={it} converged={converged} cost={cost_val:.6f}")
    return _barycentric_dense_chunked(f, g, sc, tc, sw, tw, eps)


def transfer_image(source_path, target_path, method, eps, max_iter, tol, check_every,
                    device, dtype, sinkslot_L=100):
    sc, sw, s_inverse, (h, w) = load_pixels_weights_inverse(source_path, device, dtype)
    tc, tw, _, _ = load_pixels_weights_inverse(target_path, device, dtype)
    T = solve_and_project(method, sc, tc, sw, tw, eps, max_iter, tol, check_every, sinkslot_L=sinkslot_L)
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
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--paintings_dir", type=str, default=str(DEFAULT_PAINTINGS_DIR))
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--max_iter", type=int, default=2000)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--check_every", type=int, default=5)
    p.add_argument("--sinkslot_L", type=int, default=100)
    p.add_argument("--pair_idx", type=int, nargs=2, default=[2, 9])
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This script requires a CUDA GPU.")
    device = torch.device(args.device)
    dtype = torch.float32

    os.makedirs(args.output_dir, exist_ok=True)
    paths = list_images(args.paintings_dir)
    i, j = args.pair_idx
    source_path, target_path = paths[i], paths[j]
    src_name, tgt_name = os.path.basename(source_path), os.path.basename(target_path)
    pair_stem = f"{os.path.splitext(src_name)[0]}__{os.path.splitext(tgt_name)[0]}"
    print(f"Pair: {src_name} -> {tgt_name}")

    save_image(load_as_tensor(source_path), os.path.join(args.output_dir, f"{pair_stem}_source.png"))
    save_image(load_as_tensor(target_path), os.path.join(args.output_dir, f"{pair_stem}_target.png"))

    for stem, method in _METHODS:
        print(f"  {stem} ...")
        img_tensor = transfer_image(
            source_path, target_path, method, args.eps, args.max_iter, args.tol,
            args.check_every, device, dtype, sinkslot_L=args.sinkslot_L)
        out_path = os.path.join(args.output_dir, f"{pair_stem}_{stem}.png")
        save_image(img_tensor, out_path)
        print(f"    Saved: {out_path}")


if __name__ == "__main__":
    main()
