#!/usr/bin/env python3
"""Cross-backend portability check: the SAME @triton.jit source on Metal and CUDA.

Runs three standard Triton kernels (vector-add, fused-softmax, tiled matmul), each
checked against the SAME NumPy reference, and saves the outputs so the two backends can
be diffed against each other element-wise. This is the harness behind PORTABILITY.md.

    # on an NVIDIA box (torch + triton installed):
    python3 benchmarks/cross_backend_verify.py cuda     # -> out_cuda.npz

    # pull out_cuda.npz back, then on the Mac:
    python3 benchmarks/cross_backend_verify.py mps      # -> out_mps.npz + prints Mac<->CUDA deltas

The matmul is run at two precisions: `tf32` (NVIDIA's default for fp32 `tl.dot`) and
`ieee` (true fp32). On Metal there is no tf32 hardware, so triton-msl refuses the tf32
request — which the harness reports rather than hiding.
"""
import os
import sys

import numpy as np
import torch
import triton
import triton.language as tl

DEV = "mps" if "mps" in sys.argv else "cuda"
if DEV == "mps":
    import triton_msl  # noqa: F401  (registers the Metal backend)
SYNC = (lambda: torch.mps.synchronize()) if DEV == "mps" else (lambda: torch.cuda.synchronize())


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    tl.store(out_ptr + offsets,
             tl.load(x_ptr + offsets, mask=mask) + tl.load(y_ptr + offsets, mask=mask),
             mask=mask)


@triton.jit
def softmax_kernel(out_ptr, in_ptr, row_stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(in_ptr + row * row_stride + cols, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    numer = tl.exp(x)
    tl.store(out_ptr + row * row_stride + cols, numer / tl.sum(numer, axis=0), mask=mask)


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, IP: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM); offs_n = pid_n * BN + tl.arange(0, BN); offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_m[:, None] * sam + offs_k[None, :] * sak
    b_ptrs = b_ptr + offs_k[:, None] * sbk + offs_n[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        km = offs_k < K - k * BK
        a = tl.load(a_ptrs, mask=km[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=km[:, None], other=0.0)
        acc += tl.dot(a, b, input_precision=IP)
        a_ptrs += BK * sak; b_ptrs += BK * sbk
    tl.store(c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn, acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def to_dev(a):
    return torch.from_numpy(a).to(DEV)


def main():
    outs, rows = {}, []
    np.random.seed(0)

    n = 98_432
    x = np.random.randn(n).astype(np.float32); y = np.random.randn(n).astype(np.float32)
    o = torch.empty(n, device=DEV)
    add_kernel[(triton.cdiv(n, 1024),)](to_dev(x), to_dev(y), o, n, BLOCK=1024); SYNC()
    outs["add"] = o.cpu().numpy(); rows.append(("vector_add", float(np.abs(outs["add"] - (x + y)).max())))

    R, C = 128, 781
    xm = np.random.randn(R, C).astype(np.float32); xt = to_dev(xm); om = torch.empty_like(xt)
    softmax_kernel[(R,)](om, xt, xt.stride(0), C, BLOCK=triton.next_power_of_2(C)); SYNC()
    outs["softmax"] = om.cpu().numpy()
    ref = np.exp(xm - xm.max(1, keepdims=True)); ref /= ref.sum(1, keepdims=True)
    rows.append(("fused_softmax", float(np.abs(outs["softmax"] - ref).max())))

    M, N, K = 256, 256, 256
    am = np.random.randn(M, K).astype(np.float32); bm = np.random.randn(K, N).astype(np.float32)
    ref_mm = am.astype(np.float64) @ bm.astype(np.float64)
    for ip in ("tf32", "ieee"):
        try:
            at, bt = to_dev(am), to_dev(bm); ct = torch.zeros(M, N, device=DEV)
            grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
            matmul_kernel[grid](at, bt, ct, M, N, K, *at.stride(), *bt.stride(), *ct.stride(),
                                BM=64, BN=64, BK=32, IP=ip)
            SYNC()
            outs[f"matmul_{ip}"] = ct.cpu().numpy()
            rows.append((f"matmul[{ip}]", float(np.abs(outs[f"matmul_{ip}"] - ref_mm).max())))
        except Exception as e:
            rows.append((f"matmul[{ip}]", f"refused/unsupported: {type(e).__name__}"))

    np.savez(f"out_{DEV}.npz", **outs)
    print(f"\n=== {DEV.upper()}  (triton {triton.__version__}, torch {torch.__version__}) ===")
    for name, err in rows:
        print(f"  {name:<16} vs NumPy: {err if isinstance(err, str) else f'{err:.3e}'}")

    other = "out_cuda.npz" if DEV == "mps" else "out_mps.npz"
    if os.path.exists(other):
        here = np.load(f"out_{DEV}.npz"); there = np.load(other)
        print(f"\n=== cross-backend delta ({DEV} <-> {other.split('_')[1].split('.')[0]}) ===")
        for kkey in here.files:
            if kkey in there.files:
                d = float(np.abs(here[kkey].astype(np.float64) - there[kkey].astype(np.float64)).max())
                print(f"  {kkey:<16} max|Δ| = {d:.3e}")


if __name__ == "__main__":
    main()
