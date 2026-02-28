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
    make_swiglu_kernel,
    make_reduce_kernel,
    make_softmax_kernel,
    make_matmul_kernel,
    make_simdgroup_matmul_kernel,
    make_rms_norm_kernel,
    make_rope_kernel,
    make_layer_norm_kernel,
    make_cross_entropy_kernel,
    make_flash_attention_kernel,
    make_fused_linear_kernel,
    make_top_k_kernel,
    make_kv_cache_attention_kernel,
    make_residual_add_kernel,
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


def make_float_scalar_buffer(device, value):
    buf = device.newBufferWithLength_options_(
        4, Metal.MTLResourceStorageModeShared
    )
    view = buf.contents().as_buffer(4)
    struct.pack_into("f", view, 0, value)
    return buf


def bench_custom_dispatch(bench, pipeline, buffers, n_groups, block_size,
                          n_warmup=10, n_iters=100):
    """Time a kernel with custom threadgroup dispatch."""
    import Metal as M

    for _ in range(n_warmup):
        cmd = bench.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(pipeline)
        for i, buf in enumerate(buffers):
            enc.setBuffer_offset_atIndex_(buf, 0, i)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(
            M.MTLSizeMake(n_groups, 1, 1),
            M.MTLSizeMake(block_size, 1, 1),
        )
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

    gpu_times_us = []
    for _ in range(n_iters):
        cmd = bench.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(pipeline)
        for i, buf in enumerate(buffers):
            enc.setBuffer_offset_atIndex_(buf, 0, i)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(
            M.MTLSizeMake(n_groups, 1, 1),
            M.MTLSizeMake(block_size, 1, 1),
        )
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        gpu_times_us.append(
            (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e6
        )

    gpu_times_us.sort()
    nn = len(gpu_times_us)
    return {
        "median_us": gpu_times_us[nn // 2],
        "min_us": gpu_times_us[0],
        "max_us": gpu_times_us[-1],
        "p10_us": gpu_times_us[int(0.1 * (nn - 1))],
        "p90_us": gpu_times_us[int(0.9 * (nn - 1))],
        "all_us": gpu_times_us,
    }


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

        result = bench_custom_dispatch(bench, pipeline,
                                        [in_buf, out_buf, ncols_buf],
                                        n_rows, 256)
        n_bytes = total * 4 * 5
        n_flops = total * 5
        print(format_benchmark_result("softmax", result, n_bytes=n_bytes, n_flops=n_flops))

    # =========================================================================
    # Layer Norm benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("LAYER NORM BENCHMARKS")
    print("=" * 60)

    for n_cols in [256, 1024, 4096]:
        n_rows = 128
        print(f"\n--- {n_rows} rows x {n_cols} cols ---")

        msl = make_layer_norm_kernel(block_size=256)
        pipeline = compile_and_load(device, msl, "layer_norm_kernel")

        total = n_rows * n_cols
        in_buf = make_float_buffer(device, total, "ramp")
        gamma_buf = make_float_buffer(device, n_cols, "ones")
        beta_buf = make_float_buffer(device, n_cols, "small")
        out_buf = make_empty_buffer(device, total)
        ncols_buf = make_uint_buffer(device, n_cols)

        result = bench_custom_dispatch(bench, pipeline,
                                        [in_buf, gamma_buf, beta_buf, out_buf, ncols_buf],
                                        n_rows, 256)
        n_bytes = (total * 3 + n_cols * 2) * 4
        n_flops = total * 6
        print(format_benchmark_result("layer_norm", result, n_bytes=n_bytes, n_flops=n_flops))

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

        result = bench_custom_dispatch(bench, pipeline,
                                        [A_buf, B_buf, C_buf, M_buf, N_buf, K_buf],
                                        n_groups, threads_per_tg)
        n_flops = 2 * M_dim * N_dim * K_dim
        n_bytes = (M_dim * K_dim + K_dim * N_dim + M_dim * N_dim) * 4
        print(format_benchmark_result("matmul", result, n_bytes=n_bytes, n_flops=n_flops))

    # =========================================================================
    # simdgroup_matrix matmul benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("SIMDGROUP MATRIX MATMUL BENCHMARKS (hardware 8x8 MMA)")
    print("=" * 60)

    for size in [32, 64, 128, 256, 512]:
        M_dim, N_dim, K_dim = size, size, size
        print(f"\n--- {M_dim}x{K_dim} @ {K_dim}x{N_dim} ---")

        msl = make_simdgroup_matmul_kernel()
        pipeline = compile_and_load(device, msl, "simdgroup_matmul")

        A_buf = make_float_buffer(device, M_dim * K_dim, "small")
        B_buf = make_float_buffer(device, K_dim * N_dim, "small")
        C_buf = make_empty_buffer(device, M_dim * N_dim)
        M_buf = make_uint_buffer(device, M_dim)
        N_buf = make_uint_buffer(device, N_dim)
        K_buf = make_uint_buffer(device, K_dim)

        n_tile_cols = (N_dim + 31) // 32
        n_tile_rows = (M_dim + 31) // 32
        n_groups = n_tile_rows * n_tile_cols

        result = bench_custom_dispatch(bench, pipeline,
                                        [A_buf, B_buf, C_buf, M_buf, N_buf, K_buf],
                                        n_groups, 128)
        n_flops = 2 * M_dim * N_dim * K_dim
        n_bytes = (M_dim * K_dim + K_dim * N_dim + M_dim * N_dim) * 4
        print(format_benchmark_result("simdgroup_matmul", result,
                                       n_bytes=n_bytes, n_flops=n_flops))

    # =========================================================================
    # Fused Linear (matmul + bias) benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("FUSED LINEAR BENCHMARKS (simdgroup_matrix + bias)")
    print("=" * 60)

    for size in [32, 64, 128, 256]:
        M_dim, N_dim, K_dim = size, size, size
        print(f"\n--- {M_dim}x{K_dim} @ {K_dim}x{N_dim} + bias ---")

        msl = make_fused_linear_kernel(has_bias=True)
        pipeline = compile_and_load(device, msl, "fused_linear")

        in_buf = make_float_buffer(device, M_dim * K_dim, "small")
        wt_buf = make_float_buffer(device, N_dim * K_dim, "small")
        C_buf = make_empty_buffer(device, M_dim * N_dim)
        bias_buf = make_float_buffer(device, N_dim, "small")
        M_buf = make_uint_buffer(device, M_dim)
        N_buf = make_uint_buffer(device, N_dim)
        K_buf = make_uint_buffer(device, K_dim)

        n_tile_cols = (N_dim + 31) // 32
        n_tile_rows = (M_dim + 31) // 32
        n_groups = n_tile_rows * n_tile_cols

        result = bench_custom_dispatch(bench, pipeline,
                                        [in_buf, wt_buf, C_buf, bias_buf,
                                         M_buf, N_buf, K_buf],
                                        n_groups, 128)
        n_flops = 2 * M_dim * N_dim * K_dim + M_dim * N_dim
        n_bytes = (M_dim * K_dim + N_dim * K_dim + M_dim * N_dim + N_dim) * 4
        print(format_benchmark_result("fused_linear", result,
                                       n_bytes=n_bytes, n_flops=n_flops))

    # =========================================================================
    # RMS Norm benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("RMS NORM BENCHMARKS")
    print("=" * 60)

    for n_cols in [64, 256, 1024, 4096]:
        n_rows = 128
        print(f"\n--- {n_rows} rows x {n_cols} cols ---")

        msl = make_rms_norm_kernel(block_size=256)
        pipeline = compile_and_load(device, msl, "rms_norm_kernel")

        total = n_rows * n_cols
        in_buf = make_float_buffer(device, total, "ramp")
        wt_buf = make_float_buffer(device, n_cols, "ones")
        out_buf = make_empty_buffer(device, total)
        ncols_buf = make_uint_buffer(device, n_cols)

        result = bench_custom_dispatch(bench, pipeline,
                                        [in_buf, wt_buf, out_buf, ncols_buf],
                                        n_rows, 256)
        n_bytes = total * 4 * 3 + n_cols * 4
        n_flops = total * 4
        print(format_benchmark_result("rms_norm", result,
                                       n_bytes=n_bytes, n_flops=n_flops))

    # =========================================================================
    # RoPE benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("ROPE BENCHMARKS")
    print("=" * 60)

    for dim in [64, 128, 256]:
        seq_len = 512
        print(f"\n--- seq_len={seq_len}, dim={dim} ---")

        msl = make_rope_kernel(block_size=256)
        pipeline = compile_and_load(device, msl, "rope_kernel")

        total = seq_len * dim
        in_buf = make_float_buffer(device, total, "ramp")
        freq_buf = make_float_buffer(device, dim // 2, "small")
        out_buf = make_empty_buffer(device, total)
        dim_buf = make_uint_buffer(device, dim)
        pos_buf = make_uint_buffer(device, 0)

        result = bench_custom_dispatch(bench, pipeline,
                                        [in_buf, freq_buf, out_buf, dim_buf, pos_buf],
                                        seq_len, 256)
        n_bytes = (total + dim // 2 + total) * 4
        n_flops = (dim // 2) * seq_len * 8
        print(format_benchmark_result("rope", result,
                                       n_bytes=n_bytes, n_flops=n_flops))

    # =========================================================================
    # Cross-Entropy benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("CROSS-ENTROPY LOSS BENCHMARKS")
    print("=" * 60)

    for n_classes in [256, 1024, 32768]:
        n_rows = 64
        print(f"\n--- {n_rows} rows x {n_classes} classes ---")

        msl = make_cross_entropy_kernel(block_size=256)
        pipeline = compile_and_load(device, msl, "cross_entropy_kernel")

        total = n_rows * n_classes
        in_buf = make_float_buffer(device, total, "ramp")
        # targets buffer (uint)
        tgt_buf = device.newBufferWithLength_options_(
            n_rows * 4, Metal.MTLResourceStorageModeShared
        )
        tgt_view = tgt_buf.contents().as_buffer(n_rows * 4)
        for i in range(n_rows):
            struct.pack_into("I", tgt_view, i * 4, i % n_classes)
        out_buf = make_empty_buffer(device, n_rows)
        ncls_buf = make_uint_buffer(device, n_classes)

        result = bench_custom_dispatch(bench, pipeline,
                                        [in_buf, tgt_buf, out_buf, ncls_buf],
                                        n_rows, 256)
        n_bytes = (total + n_rows + n_rows) * 4
        n_flops = total * 3
        print(format_benchmark_result("cross_entropy", result,
                                       n_bytes=n_bytes, n_flops=n_flops))

    # =========================================================================
    # SwiGLU benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("SWIGLU ACTIVATION BENCHMARKS")
    print("=" * 60)

    for n in [65536, 1_000_000]:
        print(f"\n--- n = {n:,} ---")

        msl = make_swiglu_kernel(block_size=256)
        pipeline = compile_and_load(device, msl, "swiglu_kernel")

        x_buf = make_float_buffer(device, n, "ramp")
        gate_buf = make_float_buffer(device, n, "small")
        out_buf = make_empty_buffer(device, n)
        n_buf = make_uint_buffer(device, n)

        result = bench.time_kernel(pipeline, [x_buf, gate_buf, out_buf, n_buf], n)
        n_bytes = 3 * n * 4
        print(format_benchmark_result("swiglu", result, n_bytes=n_bytes, n_flops=5 * n))

    # =========================================================================
    # Residual Add benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("RESIDUAL ADD BENCHMARKS")
    print("=" * 60)

    for n in [65536, 1_000_000]:
        print(f"\n--- n = {n:,} ---")

        msl = make_residual_add_kernel(has_bias=False)
        pipeline = compile_and_load(device, msl, "residual_add_kernel")

        x_buf = make_float_buffer(device, n, "ramp")
        r_buf = make_float_buffer(device, n, "small")
        out_buf = make_empty_buffer(device, n)
        n_buf = make_uint_buffer(device, n)

        result = bench.time_kernel(pipeline, [x_buf, r_buf, out_buf, n_buf], n)
        n_bytes = 3 * n * 4
        print(format_benchmark_result("residual_add", result, n_bytes=n_bytes, n_flops=n))

    # =========================================================================
    # KV-Cache Attention benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("KV-CACHE ATTENTION BENCHMARKS (single query)")
    print("=" * 60)

    import math
    for seq_len in [64, 256, 1024]:
        head_dim = 64
        print(f"\n--- seq_len={seq_len}, head_dim={head_dim} ---")

        msl = make_kv_cache_attention_kernel(head_dim=head_dim)
        pipeline = compile_and_load(device, msl, "kv_cache_attention")

        q_buf = make_float_buffer(device, head_dim, "small")
        k_buf = make_float_buffer(device, seq_len * head_dim, "ramp")
        v_buf = make_float_buffer(device, seq_len * head_dim, "ramp")
        out_buf = make_empty_buffer(device, head_dim)
        sl_buf = make_uint_buffer(device, seq_len)
        scale_buf = make_float_scalar_buffer(device, 1.0 / math.sqrt(head_dim))

        result = bench_custom_dispatch(bench, pipeline,
                                        [q_buf, k_buf, v_buf, out_buf,
                                         sl_buf, scale_buf],
                                        1, 256)
        # Q@K^T: seq_len*head_dim*2 flops, P@V: seq_len*head_dim*2 flops
        n_flops = 4 * seq_len * head_dim
        n_bytes = (head_dim + 2 * seq_len * head_dim + head_dim) * 4
        print(format_benchmark_result("kv_cache_attn", result,
                                       n_bytes=n_bytes, n_flops=n_flops))

    # =========================================================================
    # Top-K benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("TOP-K SAMPLING BENCHMARKS")
    print("=" * 60)

    for vocab_size in [1024, 32768, 128000]:
        k = 50
        print(f"\n--- vocab={vocab_size:,}, k={k} ---")

        msl = make_top_k_kernel(k=k, block_size=256)
        pipeline = compile_and_load(device, msl, "top_k")

        logits_buf = make_float_buffer(device, vocab_size, "ramp")
        val_buf = make_empty_buffer(device, k)
        idx_buf = device.newBufferWithLength_options_(
            k * 4, Metal.MTLResourceStorageModeShared
        )
        vsz_buf = make_uint_buffer(device, vocab_size)
        k_buf = make_uint_buffer(device, k)

        result = bench_custom_dispatch(bench, pipeline,
                                        [logits_buf, val_buf, idx_buf,
                                         vsz_buf, k_buf],
                                        1, 256)
        n_bytes = vocab_size * 4 + k * 8
        print(format_benchmark_result("top_k", result, n_bytes=n_bytes))

    # =========================================================================
    # Flash Attention benchmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("FLASH ATTENTION BENCHMARKS")
    print("=" * 60)

    for seq_len in [64, 128, 256]:
        head_dim = 64
        Br, Bc = 16, 16
        print(f"\n--- seq_len={seq_len}, head_dim={head_dim} ---")

        msl = make_flash_attention_kernel(head_dim=head_dim, Br=Br, Bc=Bc)
        pipeline = compile_and_load(device, msl, "flash_attention")

        q_buf = make_float_buffer(device, seq_len * head_dim, "small")
        k_buf = make_float_buffer(device, seq_len * head_dim, "small")
        v_buf = make_float_buffer(device, seq_len * head_dim, "small")
        out_buf = make_empty_buffer(device, seq_len * head_dim)
        sl_buf = make_uint_buffer(device, seq_len)

        n_groups = (seq_len + Br - 1) // Br
        result = bench_custom_dispatch(bench, pipeline,
                                        [q_buf, k_buf, v_buf, out_buf, sl_buf],
                                        n_groups, 256)
        n_flops = 4 * seq_len * seq_len * head_dim
        n_bytes = 4 * seq_len * head_dim * 4
        print(format_benchmark_result("flash_attn", result,
                                       n_bytes=n_bytes, n_flops=n_flops))

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
