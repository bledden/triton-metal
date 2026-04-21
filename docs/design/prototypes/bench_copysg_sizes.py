"""Sweep problem sizes to see if copy-SG pattern ever wins.

Same kernels, vary (M, N, K) to probe:
  * Compute-bound regime (large K)
  * Memory-bound-ish regime (small K with lots of loads)
  * Different tile counts
"""
from __future__ import annotations
import ctypes, subprocess, time, statistics
import numpy as np
import Foundation, Metal

def load_pipeline(device, path, name):
    url = Foundation.NSURL.fileURLWithPath_(path)
    lib, err = device.newLibraryWithURL_error_(url, None)
    if err: raise RuntimeError(err)
    fn = lib.newFunctionWithName_(name)
    pso, err = device.newComputePipelineStateWithFunction_error_(fn, None)
    if err: raise RuntimeError(err)
    return pso

def mkbuf(device, arr):
    buf = device.newBufferWithLength_options_(arr.nbytes, Metal.MTLResourceStorageModeShared)
    raw = buf.contents().as_buffer(arr.nbytes)
    ctypes.memmove((ctypes.c_char * arr.nbytes).from_buffer(raw), arr.tobytes(), arr.nbytes)
    return buf

def mku32(device, v):
    buf = device.newBufferWithLength_options_(4, Metal.MTLResourceStorageModeShared)
    raw = buf.contents().as_buffer(4)
    ctypes.memmove((ctypes.c_char*4).from_buffer(raw), np.uint32(v).tobytes(), 4)
    return buf

def bench(device, queue, pso, bufs, grid, tg, iters=50, warmup=3):
    for _ in range(warmup):
        cb = queue.commandBuffer()
        enc = cb.computeCommandEncoder()
        enc.setComputePipelineState_(pso)
        for i, b in enumerate(bufs): enc.setBuffer_offset_atIndex_(b, 0, i)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSizeMake(*grid), Metal.MTLSizeMake(*tg))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()

    t0 = time.perf_counter()
    cb = queue.commandBuffer()
    for _ in range(iters):
        enc = cb.computeCommandEncoder()
        enc.setComputePipelineState_(pso)
        for i, b in enumerate(bufs): enc.setBuffer_offset_atIndex_(b, 0, i)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSizeMake(*grid), Metal.MTLSizeMake(*tg))
        enc.endEncoding()
    cb.commit(); cb.waitUntilCompleted()
    return (time.perf_counter() - t0) / iters

def run_one(device, queue, pso_b, pso_c, M, N, K):
    assert M % 96 == 0 and N % 96 == 0 and K % 32 == 0
    np.random.seed(0)
    A = np.random.randn(M,K).astype(np.float32)
    B = np.random.randn(K,N).astype(np.float32)
    bufA = mkbuf(device, A); bufB = mkbuf(device, B)
    bufM = mku32(device, M); bufN = mku32(device, N); bufKk = mku32(device, K)
    bufCa = mkbuf(device, np.zeros((M,N), np.float32))
    bufCb = mkbuf(device, np.zeros((M,N), np.float32))

    grid = (N//96, M//96, 1); tg = (128,1,1)
    trials = 3
    bs = [bench(device, queue, pso_b, [bufA,bufB,bufCa,bufM,bufN,bufKk], grid, tg) for _ in range(trials)]
    cs = [bench(device, queue, pso_c, [bufA,bufB,bufCb,bufM,bufN,bufKk], grid, tg) for _ in range(trials)]
    bt, ct = min(bs), min(cs)
    fl = 2.0*M*N*K
    return bt, ct, fl/bt/1e9, fl/ct/1e9

def main():
    device = Metal.MTLCreateSystemDefaultDevice()
    queue = device.newCommandQueue()
    pso_b = load_pipeline(device, '/tmp/baseline_matmul.metallib', 'baseline_matmul')
    pso_c = load_pipeline(device, '/tmp/copysg_matmul.metallib', 'copysg_matmul')

    configs = [
        (480, 480, 512),
        (480, 480, 1024),
        (480, 480, 2048),
        (960, 960, 512),
        (960, 960, 1024),
        (1920, 1920, 512),
        (1920, 1920, 1024),
    ]
    print(f"{'M':>5} {'N':>5} {'K':>5}  {'base us':>10} {'copy us':>10}  {'base GF':>9} {'copy GF':>9}  {'speedup':>8}")
    for M,N,K in configs:
        bt, ct, bg, cg = run_one(device, queue, pso_b, pso_c, M, N, K)
        print(f"{M:>5} {N:>5} {K:>5}  {bt*1e6:>10.1f} {ct*1e6:>10.1f}  {bg:>9.1f} {cg:>9.1f}  {bt/ct:>8.3f}x")

if __name__ == "__main__":
    main()
