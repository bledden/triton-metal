#!/usr/bin/env python3
"""Benchmark C++ metallib vs MSL metallib compilation time.

Measures wall-clock time for full kernel compilation (TTGIR → metallib).
Runs each kernel 3 times with all caches cleared between runs.

Usage:
    .venv/bin/python scripts/bench_compilation.py
"""
import os
import sys
import time
import shutil

# Ensure triton-metal is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import triton
import triton.language as tl


CACHE_DIRS = [
    os.path.expanduser("~/.triton/cache"),
    os.path.expanduser("~/.cache/triton_metal"),
]


def clear_caches():
    for d in CACHE_DIRS:
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs,
             tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask),
             mask=mask)


@triton.jit
def softmax_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=float('-inf'))
    mx = tl.max(x, axis=0)
    e = tl.exp(x - mx)
    tl.store(out_ptr + offs, e / tl.sum(e, axis=0), mask=mask)


@triton.jit
def chain_kernel(a_ptr, b_ptr, out_ptr, n, alpha, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, (a * alpha + b) * (a - b), mask=mask)


def run_kernel(name, kernel, args, grid, constexprs):
    """Run a kernel and return wall-clock compilation+execution time in ms."""
    clear_caches()
    t0 = time.perf_counter()
    kernel[grid](*args, **constexprs)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000


def bench(use_cpp, runs=3):
    """Benchmark all kernels."""
    if use_cpp:
        os.environ["TRITON_METAL_USE_CPP"] = "1"
    else:
        os.environ.pop("TRITON_METAL_USE_CPP", None)

    N = 1024
    results = {}

    kernels = [
        ("vector_add", add_kernel,
         lambda: (torch.randn(N), torch.randn(N), torch.zeros(N), N),
         lambda: (triton.cdiv(N, 256),), {"BLOCK": 256}),
        ("softmax", softmax_kernel,
         lambda: (torch.randn(N), torch.zeros(N), N),
         lambda: (1,), {"BLOCK": 1024}),
        ("chain", chain_kernel,
         lambda: (torch.randn(N), torch.randn(N), torch.zeros(N), N, 2.5),
         lambda: (triton.cdiv(N, 256),), {"BLOCK": 256}),
    ]

    for name, kernel, make_args, make_grid, constexprs in kernels:
        times = []
        for _ in range(runs):
            args = make_args()
            grid = make_grid()
            t = run_kernel(name, kernel, args, grid, constexprs)
            times.append(t)
        results[name] = times

    return results


def main():
    print("C++ vs MSL Metallib Compilation Benchmark")
    print("=" * 50)
    print(f"Each kernel compiled {3} times (caches cleared each run)\n")

    print("MSL Path (default):")
    msl = bench(False)
    for name, times in msl.items():
        avg = sum(times) / len(times)
        print(f"  {name:12s}: {avg:7.1f}ms avg  ({', '.join(f'{t:.0f}' for t in times)})")

    print(f"\nC++ Metallib Path (TRITON_METAL_USE_CPP=1):")
    cpp = bench(True)
    for name, times in cpp.items():
        avg = sum(times) / len(times)
        print(f"  {name:12s}: {avg:7.1f}ms avg  ({', '.join(f'{t:.0f}' for t in times)})")

    print(f"\nSpeedup (MSL avg / C++ avg):")
    for name in msl:
        msl_avg = sum(msl[name]) / len(msl[name])
        cpp_avg = sum(cpp[name]) / len(cpp[name])
        ratio = msl_avg / cpp_avg if cpp_avg > 0 else float('inf')
        faster = "C++" if ratio > 1 else "MSL"
        print(f"  {name:12s}: {ratio:.2f}x ({faster} faster)")


if __name__ == "__main__":
    main()
