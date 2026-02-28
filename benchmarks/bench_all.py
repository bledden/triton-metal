"""Benchmark all kernel types on Metal GPU with GPU-precise timing.

Reports throughput in GB/s and GFLOP/s where applicable.
Uses MTLCommandBuffer.GPUStartTime/GPUEndTime for timing.

Usage: python benchmarks/bench_all.py
"""

import hashlib
import os
import struct
import subprocess
import tempfile

# GPU setup
import Metal

from triton_metal.profiling.metal_bench import (
    MetalBenchmark,
    compute_gflops,
    compute_throughput,
    format_benchmark_result,
)
from triton_metal.codegen.msl_emitter import (
    make_vector_add_kernel,
    make_elementwise_kernel,
    make_silu_kernel,
    make_gelu_kernel,
    make_reduce_kernel,
    make_softmax_kernel,
    make_matmul_kernel,
)


def compile_and_load(device, msl_src, kernel_name):
    """Compile MSL and load pipeline state."""
    import Foundation

    cache_dir = os.path.join(tempfile.gettempdir(), "triton_metal_bench_cache")
    os.makedirs(cache_dir, exist_ok=True)

    src_hash = hashlib.sha256(msl_src.encode()).hexdigest()[:16]
    base = f"{kernel_name}_{src_hash}"
    metal_path = os.path.join(cache_dir, f"{base}.metal")
    air_path = os.path.join(cache_dir, f"{base}.air")
    metallib_path = os.path.join(cache_dir, f"{base}.metallib")

    if not os.path.exists(metallib_path):
        with open(metal_path, "w") as f:
            f.write(msl_src)
        subprocess.check_call(
            ["xcrun", "-sdk", "macosx", "metal", "-c", metal_path,
             "-o", air_path, "-std=metal3.2", "-O2"],
            stderr=subprocess.PIPE,
        )
        subprocess.check_call(
            ["xcrun", "-sdk", "macosx", "metallib", air_path,
             "-o", metallib_path],
            stderr=subprocess.PIPE,
        )

    url = Foundation.NSURL.fileURLWithPath_(metallib_path)
    library, error = device.newLibraryWithURL_error_(url, None)
    assert error is None, f"Load failed: {error}"

    function = library.newFunctionWithName_(kernel_name)
    assert function is not None, f"Kernel '{kernel_name}' not found"

    pipeline, error = device.newComputePipelineStateWithFunction_error_(
        function, None
    )
    assert error is None, f"Pipeline failed: {error}"
    return pipeline


def make_float_buffer(device, n, pattern="ramp"):
    """Create a float buffer with a fill pattern. Much faster than Python lists."""
    buf = device.newBufferWithLength_options_(
        n * 4, Metal.MTLResourceStorageModeShared
    )
    view = buf.contents().as_buffer(n * 4)
    if pattern == "ramp":
        for i in range(n):
            struct.pack_into("f", view, i * 4, float(i % 1000) * 0.01)
    elif pattern == "ones":
        for i in range(n):
            struct.pack_into("f", view, i * 4, 1.0)
    elif pattern == "small":
        for i in range(n):
            struct.pack_into("f", view, i * 4, float(i % 10) * 0.1)
    return buf


def make_empty_buffer(device, n):
    return device.newBufferWithLength_options_(
        n * 4, Metal.MTLResourceStorageModeShared
    )


def make_uint_buffer(device, value):
    buf = device.newBufferWithLength_options_(
        4, Metal.MTLResourceStorageModeShared
    )
    view = buf.contents().as_buffer(4)
    struct.pack_into("I", view, 0, value)
    return buf


def main():
    device = Metal.MTLCreateSystemDefaultDevice()
    print(f"Device: {device.name()}")
    print(f"Max threadgroup memory: {device.maxThreadgroupMemoryLength()} bytes")
    print()

    bench = MetalBenchmark()

    # =========================================================================
    # Elementwise benchmarks
    # =========================================================================
    print("=" * 60)
    print("ELEMENTWISE BENCHMARKS")
    print("=" * 60)

    for n in [1024, 65536, 1_000_000]:
        print(f"\n--- n = {n:,} ---")

        # Vector add: reads 2 inputs + writes 1 output = 3 * n * 4 bytes
        msl = make_vector_add_kernel(block_size=256)
        pipeline = compile_and_load(device, msl, "vector_add")

        a_buf = make_float_buffer(device, n, "ramp")
        b_buf = make_float_buffer(device, n, "ones")
        out_buf = make_empty_buffer(device, n)
        n_buf = make_uint_buffer(device, n)

        result = bench.time_kernel(pipeline, [a_buf, b_buf, out_buf, n_buf], n)
        n_bytes = 3 * n * 4
        print(format_benchmark_result("vector_add", result, n_bytes=n_bytes, n_flops=n))

        # SiLU
        msl = make_silu_kernel(block_size=256)
        pipeline = compile_and_load(device, msl, "silu_kernel")

        in_buf = make_float_buffer(device, n, "ramp")
        out_buf = make_empty_buffer(device, n)
        n_buf = make_uint_buffer(device, n)

        result = bench.time_kernel(pipeline, [in_buf, out_buf, n_buf], n)
        n_bytes = 2 * n * 4  # 1 read + 1 write
        print(format_benchmark_result("silu", result, n_bytes=n_bytes, n_flops=4 * n))

        # GELU
        msl = make_gelu_kernel(block_size=256)
        pipeline = compile_and_load(device, msl, "gelu_kernel")

        result = bench.time_kernel(pipeline, [in_buf, out_buf, n_buf], n)
        print(format_benchmark_result("gelu", result, n_bytes=n_bytes, n_flops=8 * n))

    # =========================================================================
    # Reduction benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("REDUCTION BENCHMARKS")
    print("=" * 60)

    for n in [256, 1024, 65536, 1_000_000]:
        print(f"\n--- n = {n:,} ---")

        msl = make_reduce_kernel("reduce_sum", "sum", block_size=256)
        pipeline = compile_and_load(device, msl, "reduce_sum")

        in_buf = make_float_buffer(device, n, "ramp")
        out_buf = make_empty_buffer(device, 1)
        n_buf = make_uint_buffer(device, n)

        result = bench.time_kernel(pipeline, [in_buf, out_buf, n_buf], n)
        n_bytes = n * 4  # read only
        print(format_benchmark_result("reduce_sum", result, n_bytes=n_bytes, n_flops=n))

    # =========================================================================
    # Softmax benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("SOFTMAX BENCHMARKS")
    print("=" * 60)

    for n_cols in [64, 256, 1024, 4096]:
        n_rows = 128
        print(f"\n--- {n_rows} rows x {n_cols} cols ---")

        msl = make_softmax_kernel(block_size=256)
        pipeline = compile_and_load(device, msl, "softmax_kernel")

        total = n_rows * n_cols
        in_buf = make_float_buffer(device, total, "ramp")
        out_buf = make_empty_buffer(device, total)
        ncols_buf = make_uint_buffer(device, n_cols)

        # Softmax: custom dispatch with n_rows threadgroups
        import Metal as M
        gpu_times_us = []
        for _ in range(10):  # warmup
            cmd = bench.queue.commandBuffer()
            enc = cmd.computeCommandEncoder()
            enc.setComputePipelineState_(pipeline)
            for i, buf in enumerate([in_buf, out_buf, ncols_buf]):
                enc.setBuffer_offset_atIndex_(buf, 0, i)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                M.MTLSizeMake(n_rows, 1, 1),
                M.MTLSizeMake(256, 1, 1),
            )
            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()

        for _ in range(100):
            cmd = bench.queue.commandBuffer()
            enc = cmd.computeCommandEncoder()
            enc.setComputePipelineState_(pipeline)
            for i, buf in enumerate([in_buf, out_buf, ncols_buf]):
                enc.setBuffer_offset_atIndex_(buf, 0, i)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                M.MTLSizeMake(n_rows, 1, 1),
                M.MTLSizeMake(256, 1, 1),
            )
            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
            gpu_times_us.append(
                (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e6
            )

        gpu_times_us.sort()
        nn = len(gpu_times_us)
        result = {
            "median_us": gpu_times_us[nn // 2],
            "min_us": gpu_times_us[0],
            "max_us": gpu_times_us[-1],
            "p10_us": gpu_times_us[int(0.1 * (nn - 1))],
            "p90_us": gpu_times_us[int(0.9 * (nn - 1))],
            "all_us": gpu_times_us,
        }
        # Softmax: 3 reads + 2 writes per element
        n_bytes = total * 4 * 5
        # ~5 flops per element (max, sub, exp, sum, div)
        n_flops = total * 5
        print(format_benchmark_result("softmax", result, n_bytes=n_bytes, n_flops=n_flops))

    # =========================================================================
    # Matmul benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("MATMUL BENCHMARKS")
    print("=" * 60)

    for size in [32, 64, 128, 256]:
        M_dim, N_dim, K_dim = size, size, size
        block_m, block_n, block_k = 32, 32, 32
        print(f"\n--- {M_dim}x{K_dim} @ {K_dim}x{N_dim} ---")

        msl = make_matmul_kernel(block_m=block_m, block_n=block_n, block_k=block_k)
        pipeline = compile_and_load(device, msl, "matmul_kernel")

        A_buf = make_float_buffer(device, M_dim * K_dim, "small")
        B_buf = make_float_buffer(device, K_dim * N_dim, "small")
        C_buf = make_empty_buffer(device, M_dim * N_dim)
        M_buf = make_uint_buffer(device, M_dim)
        N_buf = make_uint_buffer(device, N_dim)
        K_buf = make_uint_buffer(device, K_dim)

        n_tile_cols = (N_dim + block_n - 1) // block_n
        n_tile_rows = (M_dim + block_m - 1) // block_m
        n_groups = n_tile_rows * n_tile_cols
        threads_per_tg = block_m * block_n

        gpu_times_us = []
        for _ in range(10):  # warmup
            cmd = bench.queue.commandBuffer()
            enc = cmd.computeCommandEncoder()
            enc.setComputePipelineState_(pipeline)
            for i, buf in enumerate([A_buf, B_buf, C_buf, M_buf, N_buf, K_buf]):
                enc.setBuffer_offset_atIndex_(buf, 0, i)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                M.MTLSizeMake(n_groups, 1, 1),
                M.MTLSizeMake(threads_per_tg, 1, 1),
            )
            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()

        for _ in range(100):
            cmd = bench.queue.commandBuffer()
            enc = cmd.computeCommandEncoder()
            enc.setComputePipelineState_(pipeline)
            for i, buf in enumerate([A_buf, B_buf, C_buf, M_buf, N_buf, K_buf]):
                enc.setBuffer_offset_atIndex_(buf, 0, i)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                M.MTLSizeMake(n_groups, 1, 1),
                M.MTLSizeMake(threads_per_tg, 1, 1),
            )
            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
            gpu_times_us.append(
                (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e6
            )

        gpu_times_us.sort()
        nn = len(gpu_times_us)
        result = {
            "median_us": gpu_times_us[nn // 2],
            "min_us": gpu_times_us[0],
            "max_us": gpu_times_us[-1],
            "p10_us": gpu_times_us[int(0.1 * (nn - 1))],
            "p90_us": gpu_times_us[int(0.9 * (nn - 1))],
            "all_us": gpu_times_us,
        }
        # Matmul: 2*M*N*K flops, reads A+B + writes C
        n_flops = 2 * M_dim * N_dim * K_dim
        n_bytes = (M_dim * K_dim + K_dim * N_dim + M_dim * N_dim) * 4
        print(format_benchmark_result("matmul", result, n_bytes=n_bytes, n_flops=n_flops))

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("M4 Max theoretical peak:")
    print("  Memory bandwidth: 546 GB/s")
    print("  FP32 compute: ~17.2 TFLOP/s (40 cores x 128 ALUs x 2 ops x 1.65 GHz)")
    print("=" * 60)


if __name__ == "__main__":
    main()
