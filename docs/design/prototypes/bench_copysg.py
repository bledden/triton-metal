"""Benchmark: baseline matmul vs copy-simdgroup prefetch matmul on Apple Silicon.

Compares two standalone Metal kernels:
  * /tmp/baseline_matmul.metal — 4 simdgroups, all compute, cooperative copy
  * /tmp/copysg_matmul.metal   — sg0 dedicated to prefetch, sg1-3 compute, double-buffered

Problem: 480 x 480 x 512 float32 matmul (M x N x K), dispatched with a 96 x 96
output tile so both kernels use exactly 5 x 5 = 25 threadgroups.

Reports GFLOPS and speedup ratio.
"""

from __future__ import annotations

import ctypes
import statistics
import subprocess
import sys
import time

import numpy as np
import Foundation
import Metal


# -------------------------------------------------------------------------------------
# 1. Compile .metal -> .metallib
# -------------------------------------------------------------------------------------

def _run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def compile_metallibs() -> None:
    _run([
        "xcrun", "-sdk", "macosx", "metal",
        "-O3", "-ffast-math",
        "-c", "/tmp/baseline_matmul.metal",
        "-o", "/tmp/baseline_matmul.air",
    ])
    _run([
        "xcrun", "-sdk", "macosx", "metallib",
        "/tmp/baseline_matmul.air",
        "-o", "/tmp/baseline_matmul.metallib",
    ])
    _run([
        "xcrun", "-sdk", "macosx", "metal",
        "-O3", "-ffast-math",
        "-c", "/tmp/copysg_matmul.metal",
        "-o", "/tmp/copysg_matmul.air",
    ])
    _run([
        "xcrun", "-sdk", "macosx", "metallib",
        "/tmp/copysg_matmul.air",
        "-o", "/tmp/copysg_matmul.metallib",
    ])


# -------------------------------------------------------------------------------------
# 2. Metal helpers
# -------------------------------------------------------------------------------------

def load_pipeline(device, metallib_path: str, func_name: str):
    url = Foundation.NSURL.fileURLWithPath_(metallib_path)
    lib, err = device.newLibraryWithURL_error_(url, None)
    if err is not None:
        raise RuntimeError(f"load {metallib_path}: {err}")
    fn = lib.newFunctionWithName_(func_name)
    if fn is None:
        names = [lib.functionNames().objectAtIndex_(i)
                 for i in range(lib.functionNames().count())]
        raise RuntimeError(f"function {func_name!r} not in {metallib_path}; available: {names}")
    pso, err = device.newComputePipelineStateWithFunction_error_(fn, None)
    if err is not None:
        raise RuntimeError(f"pso {func_name}: {err}")
    return pso


def make_buffer(device, arr: np.ndarray):
    """Create a shared-storage MTLBuffer and copy arr into it."""
    nbytes = arr.nbytes
    buf = device.newBufferWithLength_options_(
        nbytes, Metal.MTLResourceStorageModeShared
    )
    ptr = buf.contents()
    raw = ptr.as_buffer(nbytes)
    ctypes.memmove(
        (ctypes.c_char * nbytes).from_buffer(raw),
        arr.tobytes(),
        nbytes,
    )
    return buf


def make_uint_buffer(device, value: int):
    buf = device.newBufferWithLength_options_(4, Metal.MTLResourceStorageModeShared)
    raw = buf.contents().as_buffer(4)
    ctypes.memmove(
        (ctypes.c_char * 4).from_buffer(raw),
        np.uint32(value).tobytes(),
        4,
    )
    return buf


def read_buffer(buf, shape, dtype) -> np.ndarray:
    n = int(np.prod(shape))
    nbytes = n * np.dtype(dtype).itemsize
    raw = buf.contents().as_buffer(nbytes)
    a = np.frombuffer(bytes(raw), dtype=dtype).reshape(shape).copy()
    return a


# -------------------------------------------------------------------------------------
# 3. Dispatch
# -------------------------------------------------------------------------------------

def dispatch_once(
    device,
    queue,
    pso,
    buffers,
    grid_xyz,
    tg_xyz,
    *,
    sync: bool = True,
):
    cb = queue.commandBuffer()
    enc = cb.computeCommandEncoder()
    enc.setComputePipelineState_(pso)
    for i, buf in enumerate(buffers):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(*grid_xyz),
        Metal.MTLSizeMake(*tg_xyz),
    )
    enc.endEncoding()
    cb.commit()
    if sync:
        cb.waitUntilCompleted()
        if cb.status() == Metal.MTLCommandBufferStatusError:
            raise RuntimeError(f"dispatch failed: {cb.error()}")
    return cb


# -------------------------------------------------------------------------------------
# 4. Benchmark loop
# -------------------------------------------------------------------------------------

def benchmark(
    name: str,
    device,
    queue,
    pso,
    buffers,
    grid_xyz,
    tg_xyz,
    *,
    warmup: int = 5,
    iters: int = 100,
) -> tuple[float, float]:
    # Warmup
    for _ in range(warmup):
        dispatch_once(device, queue, pso, buffers, grid_xyz, tg_xyz, sync=True)

    # Timed: batch all dispatches into a single command buffer to minimize
    # per-launch CPU overhead. Then sync once.
    t0 = time.perf_counter()
    cb = queue.commandBuffer()
    for _ in range(iters):
        enc = cb.computeCommandEncoder()
        enc.setComputePipelineState_(pso)
        for i, buf in enumerate(buffers):
            enc.setBuffer_offset_atIndex_(buf, 0, i)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(*grid_xyz),
            Metal.MTLSizeMake(*tg_xyz),
        )
        enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    t1 = time.perf_counter()
    if cb.status() == Metal.MTLCommandBufferStatusError:
        raise RuntimeError(f"{name} failed: {cb.error()}")

    total = t1 - t0
    per_iter = total / iters
    return total, per_iter


def main() -> None:
    M, N, K = 480, 480, 512
    assert M % 96 == 0 and N % 96 == 0 and K % 32 == 0, \
        "M,N must be divisible by 96 and K by 32 for this prototype"

    np.random.seed(0)
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    C_ref = A @ B  # numpy reference

    compile_metallibs()

    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        raise RuntimeError("No Metal device")
    print(f"device: {device.name()}")
    queue = device.newCommandQueue()

    # Upload inputs once; we'll reuse for both kernels.
    bufA = make_buffer(device, A)
    bufB = make_buffer(device, B)
    bufM = make_uint_buffer(device, M)
    bufN = make_uint_buffer(device, N)
    bufK = make_uint_buffer(device, K)

    # Two output buffers so results don't collide.
    C_base = np.zeros((M, N), dtype=np.float32)
    C_copy = np.zeros((M, N), dtype=np.float32)
    bufC_base = make_buffer(device, C_base)
    bufC_copy = make_buffer(device, C_copy)

    grid = (N // 96, M // 96, 1)
    tg = (128, 1, 1)

    pso_base = load_pipeline(device, "/tmp/baseline_matmul.metallib", "baseline_matmul")
    pso_copy = load_pipeline(device, "/tmp/copysg_matmul.metallib", "copysg_matmul")

    # -- Correctness check --
    dispatch_once(device, queue, pso_base,
                  [bufA, bufB, bufC_base, bufM, bufN, bufK],
                  grid, tg)
    dispatch_once(device, queue, pso_copy,
                  [bufA, bufB, bufC_copy, bufM, bufN, bufK],
                  grid, tg)

    out_base = read_buffer(bufC_base, (M, N), np.float32)
    out_copy = read_buffer(bufC_copy, (M, N), np.float32)

    max_err_base = float(np.max(np.abs(out_base - C_ref)))
    max_err_copy = float(np.max(np.abs(out_copy - C_ref)))
    rel_base = max_err_base / (float(np.max(np.abs(C_ref))) + 1e-12)
    rel_copy = max_err_copy / (float(np.max(np.abs(C_ref))) + 1e-12)
    print(f"baseline max abs err: {max_err_base:.3e}  rel: {rel_base:.3e}")
    print(f"copysg   max abs err: {max_err_copy:.3e}  rel: {rel_copy:.3e}")

    # fp32 matmul with K=512 typically accumulates ~1e-4 relative error.
    tol = 5e-4
    ok_base = rel_base < tol
    ok_copy = rel_copy < tol
    if not ok_base:
        print(f"  WARNING: baseline rel err {rel_base:.3e} exceeds {tol}")
    if not ok_copy:
        print(f"  WARNING: copysg rel err {rel_copy:.3e} exceeds {tol}")

    # -- Benchmark --
    iters = 100
    flops = 2.0 * M * N * K  # per matmul

    # Several runs each to average out noise
    def bench_trials(name, pso, buf_out, n_trials=5):
        per_iters = []
        for _ in range(n_trials):
            total, per_iter = benchmark(
                name, device, queue, pso,
                [bufA, bufB, buf_out, bufM, bufN, bufK],
                grid, tg,
                warmup=3, iters=iters,
            )
            per_iters.append(per_iter)
        best = min(per_iters)
        med = statistics.median(per_iters)
        return best, med, per_iters

    print(f"\nProblem:  {M}x{N}x{K} fp32, tile=96x96x32")
    print(f"Grid:     {grid}  TG:{tg}   iters/trial={iters}")

    best_b, med_b, all_b = bench_trials("baseline", pso_base, bufC_base)
    best_c, med_c, all_c = bench_trials("copysg",   pso_copy, bufC_copy)

    g_b = flops / best_b / 1e9
    g_c = flops / best_c / 1e9
    g_b_med = flops / med_b / 1e9
    g_c_med = flops / med_c / 1e9

    print("\n=== Results (best per-iter across trials) ===")
    print(f"baseline: {best_b*1e6:8.1f} us/iter   {g_b:8.1f} GFLOPS")
    print(f"copysg:   {best_c*1e6:8.1f} us/iter   {g_c:8.1f} GFLOPS")
    print(f"speedup:  {best_b / best_c:.3f}x  (>1 means copysg wins)")

    print("\n=== Results (median per-iter across trials) ===")
    print(f"baseline: {med_b*1e6:8.1f} us/iter   {g_b_med:8.1f} GFLOPS")
    print(f"copysg:   {med_c*1e6:8.1f} us/iter   {g_c_med:8.1f} GFLOPS")
    print(f"speedup:  {med_b / med_c:.3f}x")

    print("\nper-trial (us) baseline:", [f"{p*1e6:.1f}" for p in all_b])
    print("per-trial (us) copysg  :", [f"{p*1e6:.1f}" for p in all_c])

    if not (ok_base and ok_copy):
        print("\nCORRECTNESS FAIL: check kernel output before trusting perf numbers")
        sys.exit(2)


if __name__ == "__main__":
    main()
