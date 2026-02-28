"""Tests for the MSL emitter: generate kernels, compile, run on GPU, verify.

Each test validates:
1. The emitter produces valid MSL
2. The MSL compiles to a metallib
3. The kernel runs on the M4 Max GPU
4. Results match a reference implementation
"""

import math
import platform
import random

import pytest

from tests.conftest import requires_metal

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="Metal backend requires macOS",
)

# runner fixture is provided by conftest.py


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@requires_metal
def test_vector_add(runner):
    """output = a + b"""
    from triton_metal.codegen.msl_emitter import make_vector_add_kernel

    n = 4096
    msl = make_vector_add_kernel(block_size=256)
    path = runner.compile(msl, "vector_add")
    pipeline = runner.load(path, "vector_add")

    a_data = [float(i) for i in range(n)]
    b_data = [float(i) * 0.5 for i in range(n)]

    a_buf = runner.make_float_buffer(a_data)
    b_buf = runner.make_float_buffer(b_data)
    out_buf = runner.make_empty_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [a_buf, b_buf, out_buf, n_buf], n)

    result = runner.read_float_buffer(out_buf, n)
    for i in range(n):
        expected = a_data[i] + b_data[i]
        assert abs(result[i] - expected) < 1e-4, (
            f"[{i}] got {result[i]}, expected {expected}"
        )


@requires_metal
def test_elementwise_sub(runner):
    """output = a - b"""
    from triton_metal.codegen.msl_emitter import make_elementwise_kernel

    n = 2048
    msl = make_elementwise_kernel("sub_kernel", 2, "sub")
    path = runner.compile(msl, "sub_kernel")
    pipeline = runner.load(path, "sub_kernel")

    a_data = [float(i) * 3.0 for i in range(n)]
    b_data = [float(i) for i in range(n)]

    a_buf = runner.make_float_buffer(a_data)
    b_buf = runner.make_float_buffer(b_data)
    out_buf = runner.make_empty_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [a_buf, b_buf, out_buf, n_buf], n)

    result = runner.read_float_buffer(out_buf, n)
    for i in range(n):
        expected = a_data[i] - b_data[i]
        assert abs(result[i] - expected) < 1e-4, (
            f"[{i}] got {result[i]}, expected {expected}"
        )


@requires_metal
def test_elementwise_mul(runner):
    """output = a * b"""
    from triton_metal.codegen.msl_emitter import make_elementwise_kernel

    n = 2048
    msl = make_elementwise_kernel("mul_kernel", 2, "mul")
    path = runner.compile(msl, "mul_kernel")
    pipeline = runner.load(path, "mul_kernel")

    a_data = [float(i) * 0.01 for i in range(n)]
    b_data = [float(i) * 0.02 for i in range(n)]

    a_buf = runner.make_float_buffer(a_data)
    b_buf = runner.make_float_buffer(b_data)
    out_buf = runner.make_empty_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [a_buf, b_buf, out_buf, n_buf], n)

    result = runner.read_float_buffer(out_buf, n)
    for i in range(n):
        expected = a_data[i] * b_data[i]
        assert abs(result[i] - expected) < 1e-2, (
            f"[{i}] got {result[i]}, expected {expected}"
        )


@requires_metal
def test_scalar_mul(runner):
    """output = input * scalar"""
    from triton_metal.codegen.msl_emitter import make_scalar_mul_kernel

    n = 1024
    scalar = 3.14
    msl = make_scalar_mul_kernel()
    path = runner.compile(msl, "scalar_mul")
    pipeline = runner.load(path, "scalar_mul")

    input_data = [float(i) for i in range(n)]
    input_buf = runner.make_float_buffer(input_data)
    out_buf = runner.make_empty_buffer(n)
    scalar_buf = runner.make_float_scalar_buffer(scalar)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [input_buf, out_buf, scalar_buf, n_buf], n)

    result = runner.read_float_buffer(out_buf, n)
    for i in range(n):
        expected = input_data[i] * scalar
        assert abs(result[i] - expected) < 1e-2, (
            f"[{i}] got {result[i]}, expected {expected}"
        )


@requires_metal
def test_silu(runner):
    """output = x * sigmoid(x) = x / (1 + exp(-x))"""
    from triton_metal.codegen.msl_emitter import make_silu_kernel

    n = 1024
    msl = make_silu_kernel()
    path = runner.compile(msl, "silu_kernel")
    pipeline = runner.load(path, "silu_kernel")

    # Test range [-5, 5] to exercise both sides of sigmoid
    input_data = [(i - n // 2) * 0.01 for i in range(n)]
    input_buf = runner.make_float_buffer(input_data)
    out_buf = runner.make_empty_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [input_buf, out_buf, n_buf], n)

    result = runner.read_float_buffer(out_buf, n)
    for i in range(n):
        x = input_data[i]
        expected = x / (1.0 + math.exp(-x))
        assert abs(result[i] - expected) < 1e-4, (
            f"[{i}] x={x}: got {result[i]}, expected {expected}"
        )


@requires_metal
def test_gelu(runner):
    """output = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))"""
    from triton_metal.codegen.msl_emitter import make_gelu_kernel

    n = 1024
    msl = make_gelu_kernel()
    path = runner.compile(msl, "gelu_kernel")
    pipeline = runner.load(path, "gelu_kernel")

    input_data = [(i - n // 2) * 0.01 for i in range(n)]
    input_buf = runner.make_float_buffer(input_data)
    out_buf = runner.make_empty_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [input_buf, out_buf, n_buf], n)

    result = runner.read_float_buffer(out_buf, n)
    for i in range(n):
        x = input_data[i]
        # GELU tanh approximation
        expected = 0.5 * x * (1.0 + math.tanh(
            0.7978845608028654 * (x + 0.044715 * x ** 3)
        ))
        assert abs(result[i] - expected) < 1e-4, (
            f"[{i}] x={x}: got {result[i]}, expected {expected}"
        )


@requires_metal
def test_exp(runner):
    """output = exp(x)"""
    from triton_metal.codegen.msl_emitter import make_elementwise_kernel

    n = 512
    msl = make_elementwise_kernel("exp_kernel", 1, "exp")
    path = runner.compile(msl, "exp_kernel")
    pipeline = runner.load(path, "exp_kernel")

    # Avoid overflow: test range [-10, 10]
    input_data = [(i - n // 2) * 0.04 for i in range(n)]
    input_buf = runner.make_float_buffer(input_data)
    out_buf = runner.make_empty_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [input_buf, out_buf, n_buf], n)

    result = runner.read_float_buffer(out_buf, n)
    for i in range(n):
        x = input_data[i]
        expected = math.exp(x)
        # Relative tolerance for large values
        tol = max(1e-4, abs(expected) * 1e-5)
        assert abs(result[i] - expected) < tol, (
            f"[{i}] x={x}: got {result[i]}, expected {expected}"
        )


@requires_metal
def test_non_power_of_2_size(runner):
    """Test with n not divisible by block_size (tests masking)."""
    from triton_metal.codegen.msl_emitter import make_vector_add_kernel

    n = 1000  # Not divisible by 256
    msl = make_vector_add_kernel(block_size=256)
    path = runner.compile(msl, "vector_add")
    pipeline = runner.load(path, "vector_add")

    a_data = [float(i) for i in range(n)]
    b_data = [1.0] * n

    # Allocate buffers slightly larger to detect out-of-bounds writes
    a_buf = runner.make_float_buffer(a_data)
    b_buf = runner.make_float_buffer(b_data)
    out_buf = runner.make_empty_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [a_buf, b_buf, out_buf, n_buf], n)

    result = runner.read_float_buffer(out_buf, n)
    for i in range(n):
        expected = a_data[i] + b_data[i]
        assert abs(result[i] - expected) < 1e-4, (
            f"[{i}] got {result[i]}, expected {expected}"
        )


@requires_metal
def test_large_buffer(runner):
    """Test with a large buffer (1M elements)."""
    from triton_metal.codegen.msl_emitter import make_vector_add_kernel

    n = 1_000_000
    msl = make_vector_add_kernel(block_size=256)
    path = runner.compile(msl, "vector_add")
    pipeline = runner.load(path, "vector_add")

    # Use simple patterns to avoid slow Python list creation
    a_buf = runner.make_float_buffer([float(i % 1000) for i in range(n)])
    b_buf = runner.make_float_buffer([1.0] * n)
    out_buf = runner.make_empty_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [a_buf, b_buf, out_buf, n_buf], n)

    # Spot-check a few values
    result = runner.read_float_buffer(out_buf, n)
    for i in [0, 1, 999, 1000, 500_000, 999_999]:
        expected = float(i % 1000) + 1.0
        assert abs(result[i] - expected) < 1e-4, (
            f"[{i}] got {result[i]}, expected {expected}"
        )


# ---------------------------------------------------------------------------
# Reduction tests
# ---------------------------------------------------------------------------

@requires_metal
def test_reduce_sum(runner):
    """output[0] = sum(input)"""
    from triton_metal.codegen.msl_emitter import make_reduce_kernel

    n = 256  # One threadgroup
    msl = make_reduce_kernel("reduce_sum", "sum", block_size=256)
    path = runner.compile(msl, "reduce_sum")
    pipeline = runner.load(path, "reduce_sum")

    input_data = [float(i) * 0.01 for i in range(n)]
    input_buf = runner.make_float_buffer(input_data)
    out_buf = runner.make_empty_buffer(1)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [input_buf, out_buf, n_buf], n)

    result = runner.read_float_buffer(out_buf, 1)
    expected = sum(input_data)
    assert abs(result[0] - expected) < 0.5, (
        f"got {result[0]}, expected {expected}"
    )


@requires_metal
def test_reduce_max(runner):
    """output[0] = max(input)"""
    from triton_metal.codegen.msl_emitter import make_reduce_kernel

    n = 256
    msl = make_reduce_kernel("reduce_max", "max", block_size=256)
    path = runner.compile(msl, "reduce_max")
    pipeline = runner.load(path, "reduce_max")

    random.seed(42)
    input_data = [random.uniform(-100.0, 100.0) for _ in range(n)]
    input_buf = runner.make_float_buffer(input_data)
    out_buf = runner.make_empty_buffer(1)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [input_buf, out_buf, n_buf], n)

    result = runner.read_float_buffer(out_buf, 1)
    expected = max(input_data)
    assert abs(result[0] - expected) < 1e-3, (
        f"got {result[0]}, expected {expected}"
    )


@requires_metal
def test_reduce_min(runner):
    """output[0] = min(input)"""
    from triton_metal.codegen.msl_emitter import make_reduce_kernel

    n = 256
    msl = make_reduce_kernel("reduce_min", "min", block_size=256)
    path = runner.compile(msl, "reduce_min")
    pipeline = runner.load(path, "reduce_min")

    random.seed(42)
    input_data = [random.uniform(-100.0, 100.0) for _ in range(n)]
    input_buf = runner.make_float_buffer(input_data)
    out_buf = runner.make_empty_buffer(1)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [input_buf, out_buf, n_buf], n)

    result = runner.read_float_buffer(out_buf, 1)
    expected = min(input_data)
    assert abs(result[0] - expected) < 1e-3, (
        f"got {result[0]}, expected {expected}"
    )


# ---------------------------------------------------------------------------
# Softmax tests
# ---------------------------------------------------------------------------

@requires_metal
def test_softmax(runner):
    """Row-wise softmax: output[i] = exp(x[i] - max) / sum(exp(x - max))"""
    from triton_metal.codegen.msl_emitter import make_softmax_kernel

    n_rows = 4
    n_cols = 64
    msl = make_softmax_kernel(block_size=256)
    path = runner.compile(msl, "softmax_kernel")
    pipeline = runner.load(path, "softmax_kernel")

    # Generate input: multiple rows
    random.seed(123)
    input_data = [random.uniform(-3.0, 3.0) for _ in range(n_rows * n_cols)]
    input_buf = runner.make_float_buffer(input_data)
    out_buf = runner.make_empty_buffer(n_rows * n_cols)
    ncols_buf = runner.make_uint_buffer(n_cols)

    # Launch: one threadgroup per row
    runner.run(pipeline, [input_buf, out_buf, ncols_buf], n_cols,
               block_size=256)

    # Override run to use n_rows threadgroups
    import Metal
    n_groups = n_rows
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([input_buf, out_buf, ncols_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_groups, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_float_buffer(out_buf, n_rows * n_cols)

    # Verify each row
    for row in range(n_rows):
        start = row * n_cols
        row_input = input_data[start:start + n_cols]
        row_result = result[start:start + n_cols]

        # Reference softmax
        mx = max(row_input)
        exps = [math.exp(x - mx) for x in row_input]
        s = sum(exps)
        expected = [e / s for e in exps]

        for j in range(n_cols):
            assert abs(row_result[j] - expected[j]) < 1e-4, (
                f"row={row} col={j}: got {row_result[j]}, expected {expected[j]}"
            )

        # Verify probabilities sum to ~1.0
        row_sum = sum(row_result)
        assert abs(row_sum - 1.0) < 1e-4, (
            f"row={row}: sum={row_sum}, expected 1.0"
        )


@requires_metal
def test_softmax_large_row(runner):
    """Softmax with row larger than block_size (tests strided access)."""
    from triton_metal.codegen.msl_emitter import make_softmax_kernel

    n_cols = 512  # Larger than block_size=256
    msl = make_softmax_kernel(block_size=256)
    path = runner.compile(msl, "softmax_kernel")
    pipeline = runner.load(path, "softmax_kernel")

    random.seed(456)
    input_data = [random.uniform(-5.0, 5.0) for _ in range(n_cols)]
    input_buf = runner.make_float_buffer(input_data)
    out_buf = runner.make_empty_buffer(n_cols)
    ncols_buf = runner.make_uint_buffer(n_cols)

    import Metal
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([input_buf, out_buf, ncols_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_float_buffer(out_buf, n_cols)

    # Reference
    mx = max(input_data)
    exps = [math.exp(x - mx) for x in input_data]
    s = sum(exps)
    expected = [e / s for e in exps]

    for j in range(n_cols):
        assert abs(result[j] - expected[j]) < 1e-4, (
            f"col={j}: got {result[j]}, expected {expected[j]}"
        )

    assert abs(sum(result) - 1.0) < 1e-3


# ---------------------------------------------------------------------------
# Matmul tests
# ---------------------------------------------------------------------------

@requires_metal
def test_matmul_small(runner):
    """C = A @ B for small matrices (fits in one tile)."""
    from triton_metal.codegen.msl_emitter import make_matmul_kernel

    M, N, K = 16, 16, 16
    block_m, block_n, block_k = 32, 32, 32

    msl = make_matmul_kernel(block_m=block_m, block_n=block_n, block_k=block_k)
    path = runner.compile(msl, "matmul_kernel")
    pipeline = runner.load(path, "matmul_kernel")

    # Simple test: identity-ish matrices
    random.seed(789)
    A_data = [random.uniform(-1.0, 1.0) for _ in range(M * K)]
    B_data = [random.uniform(-1.0, 1.0) for _ in range(K * N)]

    A_buf = runner.make_float_buffer(A_data)
    B_buf = runner.make_float_buffer(B_data)
    C_buf = runner.make_empty_buffer(M * N)
    M_buf = runner.make_uint_buffer(M)
    N_buf = runner.make_uint_buffer(N)
    K_buf = runner.make_uint_buffer(K)

    # Grid: one threadgroup (16x16 fits in 32x32 tile)
    n_tile_cols = (N + block_n - 1) // block_n
    n_tile_rows = (M + block_m - 1) // block_m
    n_groups = n_tile_rows * n_tile_cols
    threads_per_tg = block_m * block_n

    import Metal
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([A_buf, B_buf, C_buf, M_buf, N_buf, K_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_groups, 1, 1),
        Metal.MTLSizeMake(threads_per_tg, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4, f"Kernel failed, status={cmd.status()}"

    result = runner.read_float_buffer(C_buf, M * N)

    # Reference matmul
    for i in range(M):
        for j in range(N):
            expected = sum(A_data[i * K + k] * B_data[k * N + j] for k in range(K))
            got = result[i * N + j]
            assert abs(got - expected) < 1e-2, (
                f"C[{i},{j}]: got {got}, expected {expected}"
            )


@requires_metal
def test_matmul_rectangular(runner):
    """C = A @ B for non-square matrices."""
    from triton_metal.codegen.msl_emitter import make_matmul_kernel

    M, N, K = 24, 16, 32
    block_m, block_n, block_k = 32, 32, 32

    msl = make_matmul_kernel(block_m=block_m, block_n=block_n, block_k=block_k)
    path = runner.compile(msl, "matmul_kernel")
    pipeline = runner.load(path, "matmul_kernel")

    random.seed(101)
    A_data = [random.uniform(-1.0, 1.0) for _ in range(M * K)]
    B_data = [random.uniform(-1.0, 1.0) for _ in range(K * N)]

    A_buf = runner.make_float_buffer(A_data)
    B_buf = runner.make_float_buffer(B_data)
    C_buf = runner.make_empty_buffer(M * N)
    M_buf = runner.make_uint_buffer(M)
    N_buf = runner.make_uint_buffer(N)
    K_buf = runner.make_uint_buffer(K)

    n_tile_cols = (N + block_n - 1) // block_n
    n_tile_rows = (M + block_m - 1) // block_m
    n_groups = n_tile_rows * n_tile_cols
    threads_per_tg = block_m * block_n

    import Metal
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([A_buf, B_buf, C_buf, M_buf, N_buf, K_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_groups, 1, 1),
        Metal.MTLSizeMake(threads_per_tg, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4, f"Kernel failed, status={cmd.status()}"

    result = runner.read_float_buffer(C_buf, M * N)

    for i in range(M):
        for j in range(N):
            expected = sum(A_data[i * K + k] * B_data[k * N + j] for k in range(K))
            got = result[i * N + j]
            assert abs(got - expected) < 1e-2, (
                f"C[{i},{j}]: got {got}, expected {expected}"
            )


@requires_metal
def test_matmul_multi_tile(runner):
    """C = A @ B with multiple tiles (matrix larger than tile size)."""
    from triton_metal.codegen.msl_emitter import make_matmul_kernel

    M, N, K = 64, 64, 64
    block_m, block_n, block_k = 32, 32, 32

    msl = make_matmul_kernel(block_m=block_m, block_n=block_n, block_k=block_k)
    path = runner.compile(msl, "matmul_kernel")
    pipeline = runner.load(path, "matmul_kernel")

    random.seed(202)
    A_data = [random.uniform(-0.5, 0.5) for _ in range(M * K)]
    B_data = [random.uniform(-0.5, 0.5) for _ in range(K * N)]

    A_buf = runner.make_float_buffer(A_data)
    B_buf = runner.make_float_buffer(B_data)
    C_buf = runner.make_empty_buffer(M * N)
    M_buf = runner.make_uint_buffer(M)
    N_buf = runner.make_uint_buffer(N)
    K_buf = runner.make_uint_buffer(K)

    n_tile_cols = (N + block_n - 1) // block_n  # 2
    n_tile_rows = (M + block_m - 1) // block_m  # 2
    n_groups = n_tile_rows * n_tile_cols  # 4
    threads_per_tg = block_m * block_n

    import Metal
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([A_buf, B_buf, C_buf, M_buf, N_buf, K_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_groups, 1, 1),
        Metal.MTLSizeMake(threads_per_tg, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4, f"Kernel failed, status={cmd.status()}"

    result = runner.read_float_buffer(C_buf, M * N)

    # Spot-check several elements (full check is slow for 64x64)
    for i in range(0, M, 8):
        for j in range(0, N, 8):
            expected = sum(A_data[i * K + k] * B_data[k * N + j] for k in range(K))
            got = result[i * N + j]
            assert abs(got - expected) < 1e-1, (
                f"C[{i},{j}]: got {got}, expected {expected}"
            )


# ---------------------------------------------------------------------------
# FP16 tests
# ---------------------------------------------------------------------------

@requires_metal
def test_vector_add_fp16(runner):
    """output = a + b in half precision"""
    from triton_metal.codegen.msl_emitter import make_vector_add_kernel

    n = 1024
    msl = make_vector_add_kernel(block_size=256, dtype="fp16")
    path = runner.compile(msl, "vector_add")
    pipeline = runner.load(path, "vector_add")

    a_data = [float(i) * 0.01 for i in range(n)]
    b_data = [float(i) * 0.005 for i in range(n)]

    a_buf = runner.make_half_buffer(a_data)
    b_buf = runner.make_half_buffer(b_data)
    out_buf = runner.make_empty_half_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [a_buf, b_buf, out_buf, n_buf], n)

    result = runner.read_half_buffer(out_buf, n)
    for i in range(n):
        expected = a_data[i] + b_data[i]
        # FP16 has ~3 decimal digits of precision
        tol = max(1e-2, abs(expected) * 1e-2)
        assert abs(result[i] - expected) < tol, (
            f"[{i}] got {result[i]}, expected {expected}"
        )


@requires_metal
def test_silu_fp16(runner):
    """output = x * sigmoid(x) in half precision"""
    from triton_metal.codegen.msl_emitter import make_silu_kernel

    n = 512
    msl = make_silu_kernel(block_size=256, dtype="fp16")
    path = runner.compile(msl, "silu_kernel")
    pipeline = runner.load(path, "silu_kernel")

    input_data = [(i - n // 2) * 0.01 for i in range(n)]
    input_buf = runner.make_half_buffer(input_data)
    out_buf = runner.make_empty_half_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [input_buf, out_buf, n_buf], n)

    result = runner.read_half_buffer(out_buf, n)
    for i in range(n):
        x = input_data[i]
        expected = x / (1.0 + math.exp(-x))
        tol = max(1e-2, abs(expected) * 5e-2)
        assert abs(result[i] - expected) < tol, (
            f"[{i}] x={x}: got {result[i]}, expected {expected}"
        )


@requires_metal
def test_elementwise_mul_fp16(runner):
    """output = a * b in half precision"""
    from triton_metal.codegen.msl_emitter import make_elementwise_kernel

    n = 1024
    msl = make_elementwise_kernel("mul_fp16", 2, "mul", dtype="fp16")
    path = runner.compile(msl, "mul_fp16")
    pipeline = runner.load(path, "mul_fp16")

    a_data = [float(i) * 0.01 for i in range(n)]
    b_data = [float(i) * 0.02 for i in range(n)]

    a_buf = runner.make_half_buffer(a_data)
    b_buf = runner.make_half_buffer(b_data)
    out_buf = runner.make_empty_half_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [a_buf, b_buf, out_buf, n_buf], n)

    result = runner.read_half_buffer(out_buf, n)
    for i in range(n):
        expected = a_data[i] * b_data[i]
        tol = max(1e-1, abs(expected) * 5e-2)
        assert abs(result[i] - expected) < tol, (
            f"[{i}] got {result[i]}, expected {expected}"
        )


# ---------------------------------------------------------------------------
# simdgroup_matrix matmul tests
# ---------------------------------------------------------------------------

def _dispatch_simdgroup_matmul(runner, pipeline, buffers, M, N):
    """Dispatch a simdgroup matmul kernel with correct grid dimensions."""
    import Metal

    n_tile_cols = (N + 31) // 32
    n_tile_rows = (M + 31) // 32
    n_groups = n_tile_rows * n_tile_cols

    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate(buffers):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_groups, 1, 1),
        Metal.MTLSizeMake(128, 1, 1),  # 4 SIMD groups x 32 threads
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4, f"Kernel failed, status={cmd.status()}"


def _ref_matmul(A, B, M, N, K):
    """Reference matmul for testing."""
    C = [0.0] * (M * N)
    for i in range(M):
        for j in range(N):
            s = 0.0
            for k in range(K):
                s += A[i * K + k] * B[k * N + j]
            C[i * N + j] = s
    return C


@requires_metal
def test_simdgroup_matmul_32x32(runner):
    """simdgroup_matrix matmul: 32x32 @ 32x32 (single tile)."""
    from triton_metal.codegen.msl_emitter import make_simdgroup_matmul_kernel

    M, N, K = 32, 32, 32
    msl = make_simdgroup_matmul_kernel()
    path = runner.compile(msl, "simdgroup_matmul")
    pipeline = runner.load(path, "simdgroup_matmul")

    random.seed(303)
    A_data = [random.uniform(-1.0, 1.0) for _ in range(M * K)]
    B_data = [random.uniform(-1.0, 1.0) for _ in range(K * N)]

    A_buf = runner.make_float_buffer(A_data)
    B_buf = runner.make_float_buffer(B_data)
    C_buf = runner.make_empty_buffer(M * N)
    M_buf = runner.make_uint_buffer(M)
    N_buf = runner.make_uint_buffer(N)
    K_buf = runner.make_uint_buffer(K)

    _dispatch_simdgroup_matmul(runner, pipeline,
                               [A_buf, B_buf, C_buf, M_buf, N_buf, K_buf], M, N)

    result = runner.read_float_buffer(C_buf, M * N)
    expected = _ref_matmul(A_data, B_data, M, N, K)

    for i in range(M):
        for j in range(N):
            idx = i * N + j
            assert abs(result[idx] - expected[idx]) < 1e-2, (
                f"C[{i},{j}]: got {result[idx]}, expected {expected[idx]}"
            )


@requires_metal
def test_simdgroup_matmul_64x64(runner):
    """simdgroup_matrix matmul: 64x64 @ 64x64 (2x2 tiles)."""
    from triton_metal.codegen.msl_emitter import make_simdgroup_matmul_kernel

    M, N, K = 64, 64, 64
    msl = make_simdgroup_matmul_kernel()
    path = runner.compile(msl, "simdgroup_matmul")
    pipeline = runner.load(path, "simdgroup_matmul")

    random.seed(404)
    A_data = [random.uniform(-0.5, 0.5) for _ in range(M * K)]
    B_data = [random.uniform(-0.5, 0.5) for _ in range(K * N)]

    A_buf = runner.make_float_buffer(A_data)
    B_buf = runner.make_float_buffer(B_data)
    C_buf = runner.make_empty_buffer(M * N)
    M_buf = runner.make_uint_buffer(M)
    N_buf = runner.make_uint_buffer(N)
    K_buf = runner.make_uint_buffer(K)

    _dispatch_simdgroup_matmul(runner, pipeline,
                               [A_buf, B_buf, C_buf, M_buf, N_buf, K_buf], M, N)

    result = runner.read_float_buffer(C_buf, M * N)
    expected = _ref_matmul(A_data, B_data, M, N, K)

    # Spot-check every 4th element
    for i in range(0, M, 4):
        for j in range(0, N, 4):
            idx = i * N + j
            assert abs(result[idx] - expected[idx]) < 1e-1, (
                f"C[{i},{j}]: got {result[idx]}, expected {expected[idx]}"
            )


@requires_metal
def test_simdgroup_matmul_rectangular(runner):
    """simdgroup_matrix matmul: 64x32 @ 32x64 (rectangular)."""
    from triton_metal.codegen.msl_emitter import make_simdgroup_matmul_kernel

    M, N, K = 64, 64, 32
    msl = make_simdgroup_matmul_kernel()
    path = runner.compile(msl, "simdgroup_matmul")
    pipeline = runner.load(path, "simdgroup_matmul")

    random.seed(505)
    A_data = [random.uniform(-1.0, 1.0) for _ in range(M * K)]
    B_data = [random.uniform(-1.0, 1.0) for _ in range(K * N)]

    A_buf = runner.make_float_buffer(A_data)
    B_buf = runner.make_float_buffer(B_data)
    C_buf = runner.make_empty_buffer(M * N)
    M_buf = runner.make_uint_buffer(M)
    N_buf = runner.make_uint_buffer(N)
    K_buf = runner.make_uint_buffer(K)

    _dispatch_simdgroup_matmul(runner, pipeline,
                               [A_buf, B_buf, C_buf, M_buf, N_buf, K_buf], M, N)

    result = runner.read_float_buffer(C_buf, M * N)
    expected = _ref_matmul(A_data, B_data, M, N, K)

    for i in range(0, M, 4):
        for j in range(0, N, 4):
            idx = i * N + j
            assert abs(result[idx] - expected[idx]) < 1e-1, (
                f"C[{i},{j}]: got {result[idx]}, expected {expected[idx]}"
            )


# ---------------------------------------------------------------------------
# FP16 simdgroup_matrix matmul tests
# ---------------------------------------------------------------------------

@requires_metal
def test_simdgroup_matmul_fp16_compiles(runner):
    """FP16 simdgroup matmul MSL compiles."""
    from triton_metal.codegen.msl_emitter import make_simdgroup_matmul_kernel

    msl = make_simdgroup_matmul_kernel(dtype="fp16")
    runner.compile(msl, "simdgroup_matmul")


@requires_metal
def test_simdgroup_matmul_fp16_32x32(runner):
    """FP16 simdgroup matmul: half inputs, float accumulation, 32x32."""
    from triton_metal.codegen.msl_emitter import make_simdgroup_matmul_kernel

    M, N, K = 32, 32, 32
    msl = make_simdgroup_matmul_kernel(dtype="fp16")
    path = runner.compile(msl, "simdgroup_matmul")
    pipeline = runner.load(path, "simdgroup_matmul")

    random.seed(707)
    A_data = [random.uniform(-1.0, 1.0) for _ in range(M * K)]
    B_data = [random.uniform(-1.0, 1.0) for _ in range(K * N)]

    # Half-precision inputs, float output
    A_buf = runner.make_half_buffer(A_data)
    B_buf = runner.make_half_buffer(B_data)
    C_buf = runner.make_empty_buffer(M * N)  # float output
    M_buf = runner.make_uint_buffer(M)
    N_buf = runner.make_uint_buffer(N)
    K_buf = runner.make_uint_buffer(K)

    _dispatch_simdgroup_matmul(runner, pipeline,
                               [A_buf, B_buf, C_buf, M_buf, N_buf, K_buf], M, N)

    result = runner.read_float_buffer(C_buf, M * N)
    expected = _ref_matmul(A_data, B_data, M, N, K)

    for i in range(M):
        for j in range(N):
            idx = i * N + j
            # FP16 inputs lose precision → wider tolerance
            tol = max(0.1, abs(expected[idx]) * 0.05)
            assert abs(result[idx] - expected[idx]) < tol, (
                f"C[{i},{j}]: got {result[idx]}, expected {expected[idx]}"
            )


@requires_metal
def test_simdgroup_matmul_fp16_64x64(runner):
    """FP16 simdgroup matmul: 64x64 multi-tile."""
    from triton_metal.codegen.msl_emitter import make_simdgroup_matmul_kernel

    M, N, K = 64, 64, 64
    msl = make_simdgroup_matmul_kernel(dtype="fp16")
    path = runner.compile(msl, "simdgroup_matmul")
    pipeline = runner.load(path, "simdgroup_matmul")

    random.seed(808)
    A_data = [random.uniform(-0.5, 0.5) for _ in range(M * K)]
    B_data = [random.uniform(-0.5, 0.5) for _ in range(K * N)]

    A_buf = runner.make_half_buffer(A_data)
    B_buf = runner.make_half_buffer(B_data)
    C_buf = runner.make_empty_buffer(M * N)
    M_buf = runner.make_uint_buffer(M)
    N_buf = runner.make_uint_buffer(N)
    K_buf = runner.make_uint_buffer(K)

    _dispatch_simdgroup_matmul(runner, pipeline,
                               [A_buf, B_buf, C_buf, M_buf, N_buf, K_buf], M, N)

    result = runner.read_float_buffer(C_buf, M * N)
    expected = _ref_matmul(A_data, B_data, M, N, K)

    for i in range(0, M, 4):
        for j in range(0, N, 4):
            idx = i * N + j
            tol = max(0.15, abs(expected[idx]) * 0.1)
            assert abs(result[idx] - expected[idx]) < tol, (
                f"C[{i},{j}]: got {result[idx]}, expected {expected[idx]}"
            )


# ---------------------------------------------------------------------------
# RMS normalization tests
# ---------------------------------------------------------------------------

@requires_metal
def test_rms_norm(runner):
    """RMS norm: output = x * rsqrt(mean(x^2) + eps) * weight"""
    from triton_metal.codegen.msl_emitter import make_rms_norm_kernel

    n_cols = 64
    n_rows = 4
    eps = 1e-6
    msl = make_rms_norm_kernel(block_size=256, eps=eps)
    path = runner.compile(msl, "rms_norm_kernel")
    pipeline = runner.load(path, "rms_norm_kernel")

    random.seed(321)
    input_data = [random.gauss(0, 1) for _ in range(n_rows * n_cols)]
    weight_data = [random.uniform(0.5, 1.5) for _ in range(n_cols)]

    input_buf = runner.make_float_buffer(input_data)
    weight_buf = runner.make_float_buffer(weight_data)
    out_buf = runner.make_empty_buffer(n_rows * n_cols)
    ncols_buf = runner.make_uint_buffer(n_cols)

    import Metal
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([input_buf, weight_buf, out_buf, ncols_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_rows, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_float_buffer(out_buf, n_rows * n_cols)

    # Reference RMS norm
    for row in range(n_rows):
        row_in = input_data[row * n_cols:(row + 1) * n_cols]
        row_out = result[row * n_cols:(row + 1) * n_cols]

        mean_sq = sum(x * x for x in row_in) / n_cols
        rms = 1.0 / math.sqrt(mean_sq + eps)

        for j in range(n_cols):
            expected = row_in[j] * rms * weight_data[j]
            tol = max(1e-4, abs(expected) * 1e-3)
            assert abs(row_out[j] - expected) < tol, (
                f"Row {row}[{j}] got {row_out[j]}, expected {expected}"
            )


@requires_metal
def test_rms_norm_large_row(runner):
    """RMS norm with row larger than block_size (tests strided access)."""
    from triton_metal.codegen.msl_emitter import make_rms_norm_kernel

    n_cols = 512
    eps = 1e-6
    msl = make_rms_norm_kernel(block_size=256, eps=eps)
    path = runner.compile(msl, "rms_norm_kernel")
    pipeline = runner.load(path, "rms_norm_kernel")

    random.seed(654)
    input_data = [random.gauss(0, 2) for _ in range(n_cols)]
    weight_data = [1.0] * n_cols  # unity weight for simplicity

    input_buf = runner.make_float_buffer(input_data)
    weight_buf = runner.make_float_buffer(weight_data)
    out_buf = runner.make_empty_buffer(n_cols)
    ncols_buf = runner.make_uint_buffer(n_cols)

    import Metal
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([input_buf, weight_buf, out_buf, ncols_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_float_buffer(out_buf, n_cols)

    mean_sq = sum(x * x for x in input_data) / n_cols
    rms = 1.0 / math.sqrt(mean_sq + eps)
    for j in range(n_cols):
        expected = input_data[j] * rms
        tol = max(1e-4, abs(expected) * 1e-3)
        assert abs(result[j] - expected) < tol, (
            f"[{j}] got {result[j]}, expected {expected}"
        )


# ---------------------------------------------------------------------------
# RoPE (rotary position embeddings) tests
# ---------------------------------------------------------------------------

@requires_metal
def test_rope(runner):
    """RoPE: apply rotary position embeddings."""
    from triton_metal.codegen.msl_emitter import make_rope_kernel

    dim = 64
    seq_len = 4
    msl = make_rope_kernel(block_size=256)
    path = runner.compile(msl, "rope_kernel")
    pipeline = runner.load(path, "rope_kernel")

    random.seed(987)
    input_data = [random.gauss(0, 1) for _ in range(seq_len * dim)]

    # Pre-compute inverse frequencies: 1 / (10000^(2i/dim))
    freqs = [1.0 / (10000.0 ** (2 * i / dim)) for i in range(dim // 2)]

    input_buf = runner.make_float_buffer(input_data)
    freqs_buf = runner.make_float_buffer(freqs)
    out_buf = runner.make_empty_buffer(seq_len * dim)
    dim_buf = runner.make_uint_buffer(dim)
    pos_buf = runner.make_uint_buffer(0)  # pos_offset = 0

    import Metal
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([input_buf, freqs_buf, out_buf, dim_buf, pos_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(seq_len, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_float_buffer(out_buf, seq_len * dim)

    # Reference RoPE
    for pos in range(seq_len):
        for i in range(dim // 2):
            theta = pos * freqs[i]
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            x0 = input_data[pos * dim + 2 * i]
            x1 = input_data[pos * dim + 2 * i + 1]
            exp0 = x0 * cos_t - x1 * sin_t
            exp1 = x0 * sin_t + x1 * cos_t

            got0 = result[pos * dim + 2 * i]
            got1 = result[pos * dim + 2 * i + 1]
            assert abs(got0 - exp0) < 1e-4, (
                f"pos={pos} pair={i}[0]: got {got0}, expected {exp0}"
            )
            assert abs(got1 - exp1) < 1e-4, (
                f"pos={pos} pair={i}[1]: got {got1}, expected {exp1}"
            )


# ---------------------------------------------------------------------------
# Layer normalization tests
# ---------------------------------------------------------------------------

@requires_metal
def test_layer_norm(runner):
    """Layer norm: output = (x - mean) / sqrt(var + eps) * gamma + beta"""
    from triton_metal.codegen.msl_emitter import make_layer_norm_kernel

    n_cols = 64
    n_rows = 4
    eps = 1e-6
    msl = make_layer_norm_kernel(block_size=256, eps=eps)
    path = runner.compile(msl, "layer_norm_kernel")
    pipeline = runner.load(path, "layer_norm_kernel")

    random.seed(111)
    input_data = [random.gauss(0, 1) for _ in range(n_rows * n_cols)]
    gamma_data = [random.uniform(0.5, 1.5) for _ in range(n_cols)]
    beta_data = [random.uniform(-0.5, 0.5) for _ in range(n_cols)]

    input_buf = runner.make_float_buffer(input_data)
    gamma_buf = runner.make_float_buffer(gamma_data)
    beta_buf = runner.make_float_buffer(beta_data)
    out_buf = runner.make_empty_buffer(n_rows * n_cols)
    ncols_buf = runner.make_uint_buffer(n_cols)

    import Metal
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([input_buf, gamma_buf, beta_buf, out_buf, ncols_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_rows, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_float_buffer(out_buf, n_rows * n_cols)

    for row in range(n_rows):
        row_in = input_data[row * n_cols:(row + 1) * n_cols]
        row_out = result[row * n_cols:(row + 1) * n_cols]

        mean = sum(row_in) / n_cols
        var = sum((x - mean) ** 2 for x in row_in) / n_cols
        inv_std = 1.0 / math.sqrt(var + eps)

        for j in range(n_cols):
            expected = (row_in[j] - mean) * inv_std * gamma_data[j] + beta_data[j]
            tol = max(1e-4, abs(expected) * 1e-3)
            assert abs(row_out[j] - expected) < tol, (
                f"Row {row}[{j}] got {row_out[j]}, expected {expected}"
            )


@requires_metal
def test_layer_norm_large_row(runner):
    """Layer norm with row larger than block_size."""
    from triton_metal.codegen.msl_emitter import make_layer_norm_kernel

    n_cols = 512
    eps = 1e-6
    msl = make_layer_norm_kernel(block_size=256, eps=eps)
    path = runner.compile(msl, "layer_norm_kernel")
    pipeline = runner.load(path, "layer_norm_kernel")

    random.seed(222)
    input_data = [random.gauss(0, 2) for _ in range(n_cols)]
    gamma_data = [1.0] * n_cols
    beta_data = [0.0] * n_cols

    input_buf = runner.make_float_buffer(input_data)
    gamma_buf = runner.make_float_buffer(gamma_data)
    beta_buf = runner.make_float_buffer(beta_data)
    out_buf = runner.make_empty_buffer(n_cols)
    ncols_buf = runner.make_uint_buffer(n_cols)

    import Metal
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([input_buf, gamma_buf, beta_buf, out_buf, ncols_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_float_buffer(out_buf, n_cols)

    mean = sum(input_data) / n_cols
    var = sum((x - mean) ** 2 for x in input_data) / n_cols
    inv_std = 1.0 / math.sqrt(var + eps)

    for j in range(n_cols):
        expected = (input_data[j] - mean) * inv_std
        tol = max(1e-3, abs(expected) * 1e-2)
        assert abs(result[j] - expected) < tol, (
            f"[{j}] got {result[j]}, expected {expected}"
        )

    # Output should have mean ~0 and variance ~1
    out_mean = sum(result) / n_cols
    assert abs(out_mean) < 0.05, f"Output mean = {out_mean}, expected ~0"


# ---------------------------------------------------------------------------
# Cross-entropy loss tests
# ---------------------------------------------------------------------------

@requires_metal
def test_cross_entropy(runner):
    """Cross-entropy loss: loss = log_sum_exp(logits) - logits[target]"""
    from triton_metal.codegen.msl_emitter import make_cross_entropy_kernel

    n_rows = 4
    vocab_size = 32
    msl = make_cross_entropy_kernel(block_size=256)
    path = runner.compile(msl, "cross_entropy_kernel")
    pipeline = runner.load(path, "cross_entropy_kernel")

    random.seed(333)
    logits_data = [random.gauss(0, 2) for _ in range(n_rows * vocab_size)]
    targets_data = [random.randint(0, vocab_size - 1) for _ in range(n_rows)]

    logits_buf = runner.make_float_buffer(logits_data)
    # targets need int32 buffer
    import Metal
    import struct as st
    targets_buf = runner.device.newBufferWithLength_options_(
        n_rows * 4, Metal.MTLResourceStorageModeShared
    )
    view = targets_buf.contents().as_buffer(n_rows * 4)
    for i, t in enumerate(targets_data):
        st.pack_into("i", view, i * 4, t)

    losses_buf = runner.make_empty_buffer(n_rows)
    vocab_buf = runner.make_uint_buffer(vocab_size)

    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([logits_buf, targets_buf, losses_buf, vocab_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_rows, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_float_buffer(losses_buf, n_rows)

    for row in range(n_rows):
        row_logits = logits_data[row * vocab_size:(row + 1) * vocab_size]
        target = targets_data[row]

        # Reference: log_sum_exp - logits[target]
        mx = max(row_logits)
        log_sum_exp = mx + math.log(sum(math.exp(x - mx) for x in row_logits))
        expected = log_sum_exp - row_logits[target]

        assert abs(result[row] - expected) < 1e-3, (
            f"Row {row}: got {result[row]}, expected {expected}"
        )
        # Loss should be non-negative
        assert result[row] >= -1e-6, f"Row {row}: negative loss {result[row]}"


@requires_metal
def test_cross_entropy_large_vocab(runner):
    """Cross-entropy with vocab larger than block_size."""
    from triton_metal.codegen.msl_emitter import make_cross_entropy_kernel

    vocab_size = 1024  # > block_size of 256
    msl = make_cross_entropy_kernel(block_size=256)
    path = runner.compile(msl, "cross_entropy_kernel")
    pipeline = runner.load(path, "cross_entropy_kernel")

    random.seed(444)
    logits_data = [random.gauss(0, 1) for _ in range(vocab_size)]
    target = 500  # middle of vocab

    logits_buf = runner.make_float_buffer(logits_data)
    import Metal
    import struct as st
    targets_buf = runner.device.newBufferWithLength_options_(
        4, Metal.MTLResourceStorageModeShared
    )
    view = targets_buf.contents().as_buffer(4)
    st.pack_into("i", view, 0, target)

    losses_buf = runner.make_empty_buffer(1)
    vocab_buf = runner.make_uint_buffer(vocab_size)

    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([logits_buf, targets_buf, losses_buf, vocab_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_float_buffer(losses_buf, 1)

    mx = max(logits_data)
    log_sum_exp = mx + math.log(sum(math.exp(x - mx) for x in logits_data))
    expected = log_sum_exp - logits_data[target]

    assert abs(result[0] - expected) < 1e-2, (
        f"got {result[0]}, expected {expected}"
    )


# ---------------------------------------------------------------------------
# Flash Attention tests
# ---------------------------------------------------------------------------

@requires_metal
def test_flash_attention_single_head(runner):
    """Flash Attention: single head, short sequence."""
    from triton_metal.codegen.msl_emitter import make_flash_attention_kernel

    seq_len = 16
    head_dim = 64
    n_heads = 1
    scale = 1.0 / math.sqrt(head_dim)

    msl = make_flash_attention_kernel(head_dim=head_dim, Br=16, Bc=16)
    path = runner.compile(msl, "flash_attention")
    pipeline = runner.load(path, "flash_attention")

    random.seed(555)
    Q_data = [random.gauss(0, 0.5) for _ in range(n_heads * seq_len * head_dim)]
    K_data = [random.gauss(0, 0.5) for _ in range(n_heads * seq_len * head_dim)]
    V_data = [random.gauss(0, 0.5) for _ in range(n_heads * seq_len * head_dim)]

    Q_buf = runner.make_float_buffer(Q_data)
    K_buf = runner.make_float_buffer(K_data)
    V_buf = runner.make_float_buffer(V_data)
    O_buf = runner.make_empty_buffer(n_heads * seq_len * head_dim)
    seq_buf = runner.make_uint_buffer(seq_len)
    scale_buf = runner.make_float_scalar_buffer(scale)

    n_q_blocks = (seq_len + 15) // 16
    n_groups = n_heads * n_q_blocks

    import Metal
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([Q_buf, K_buf, V_buf, O_buf, seq_buf, scale_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_groups, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4, f"Kernel failed, status={cmd.status()}"

    result = runner.read_float_buffer(O_buf, n_heads * seq_len * head_dim)

    # Reference attention: softmax(Q @ K^T * scale) @ V
    for h in range(n_heads):
        ho = h * seq_len * head_dim
        # Compute S = Q @ K^T
        S = [[0.0] * seq_len for _ in range(seq_len)]
        for i in range(seq_len):
            for j in range(seq_len):
                dot = 0.0
                for d in range(head_dim):
                    dot += Q_data[ho + i * head_dim + d] * K_data[ho + j * head_dim + d]
                S[i][j] = dot * scale

        # Softmax each row
        P = []
        for i in range(seq_len):
            mx = max(S[i])
            exps = [math.exp(s - mx) for s in S[i]]
            s = sum(exps)
            P.append([e / s for e in exps])

        # O = P @ V
        for i in range(seq_len):
            for d in range(head_dim):
                expected = sum(P[i][j] * V_data[ho + j * head_dim + d] for j in range(seq_len))
                got = result[ho + i * head_dim + d]
                tol = max(1e-3, abs(expected) * 1e-2)
                assert abs(got - expected) < tol, (
                    f"head={h} pos={i} dim={d}: got {got}, expected {expected}"
                )


@requires_metal
def test_flash_attention_multi_block(runner):
    """Flash Attention: sequence longer than one block."""
    from triton_metal.codegen.msl_emitter import make_flash_attention_kernel

    seq_len = 48  # 3 blocks of 16
    head_dim = 64
    n_heads = 1
    scale = 1.0 / math.sqrt(head_dim)

    msl = make_flash_attention_kernel(head_dim=head_dim, Br=16, Bc=16)
    path = runner.compile(msl, "flash_attention")
    pipeline = runner.load(path, "flash_attention")

    random.seed(666)
    Q_data = [random.gauss(0, 0.3) for _ in range(n_heads * seq_len * head_dim)]
    K_data = [random.gauss(0, 0.3) for _ in range(n_heads * seq_len * head_dim)]
    V_data = [random.gauss(0, 0.3) for _ in range(n_heads * seq_len * head_dim)]

    Q_buf = runner.make_float_buffer(Q_data)
    K_buf = runner.make_float_buffer(K_data)
    V_buf = runner.make_float_buffer(V_data)
    O_buf = runner.make_empty_buffer(n_heads * seq_len * head_dim)
    seq_buf = runner.make_uint_buffer(seq_len)
    scale_buf = runner.make_float_scalar_buffer(scale)

    n_q_blocks = (seq_len + 15) // 16
    n_groups = n_heads * n_q_blocks

    import Metal
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([Q_buf, K_buf, V_buf, O_buf, seq_buf, scale_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_groups, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4, f"Kernel failed, status={cmd.status()}"

    result = runner.read_float_buffer(O_buf, n_heads * seq_len * head_dim)

    # Reference: full attention
    S = [[0.0] * seq_len for _ in range(seq_len)]
    for i in range(seq_len):
        for j in range(seq_len):
            dot = sum(Q_data[i * head_dim + d] * K_data[j * head_dim + d] for d in range(head_dim))
            S[i][j] = dot * scale

    P = []
    for i in range(seq_len):
        mx = max(S[i])
        exps = [math.exp(s - mx) for s in S[i]]
        s = sum(exps)
        P.append([e / s for e in exps])

    # Spot-check every 4th position, every 8th dim
    for i in range(0, seq_len, 4):
        for d in range(0, head_dim, 8):
            expected = sum(P[i][j] * V_data[j * head_dim + d] for j in range(seq_len))
            got = result[i * head_dim + d]
            tol = max(1e-2, abs(expected) * 0.05)
            assert abs(got - expected) < tol, (
                f"pos={i} dim={d}: got {got}, expected {expected}"
            )


# ---------------------------------------------------------------------------
# BFloat16 tests
# ---------------------------------------------------------------------------

@requires_metal
def test_vector_add_bf16(runner):
    """output = a + b in bfloat16 precision"""
    from triton_metal.codegen.msl_emitter import make_vector_add_kernel

    n = 1024
    msl = make_vector_add_kernel(block_size=256, dtype="bf16")
    path = runner.compile(msl, "vector_add")
    pipeline = runner.load(path, "vector_add")

    a_data = [float(i) * 0.01 for i in range(n)]
    b_data = [float(i) * 0.005 for i in range(n)]

    a_buf = runner.make_bf16_buffer(a_data)
    b_buf = runner.make_bf16_buffer(b_data)
    out_buf = runner.make_empty_bf16_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [a_buf, b_buf, out_buf, n_buf], n)

    result = runner.read_bf16_buffer(out_buf, n)
    for i in range(0, n, 16):  # spot-check every 16th
        expected = a_data[i] + b_data[i]
        # BF16 has ~2-3 decimal digits of precision
        tol = max(0.1, abs(expected) * 0.02)
        assert abs(result[i] - expected) < tol, (
            f"[{i}] got {result[i]}, expected {expected}"
        )


@requires_metal
def test_silu_bf16(runner):
    """output = x * sigmoid(x) in bfloat16 precision"""
    from triton_metal.codegen.msl_emitter import make_silu_kernel

    n = 512
    msl = make_silu_kernel(block_size=256, dtype="bf16")
    path = runner.compile(msl, "silu_kernel")
    pipeline = runner.load(path, "silu_kernel")

    input_data = [(i - n // 2) * 0.01 for i in range(n)]
    input_buf = runner.make_bf16_buffer(input_data)
    out_buf = runner.make_empty_bf16_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [input_buf, out_buf, n_buf], n)

    result = runner.read_bf16_buffer(out_buf, n)
    for i in range(0, n, 8):
        x = input_data[i]
        expected = x / (1.0 + math.exp(-x))
        tol = max(0.05, abs(expected) * 0.1)
        assert abs(result[i] - expected) < tol, (
            f"[{i}] x={x}: got {result[i]}, expected {expected}"
        )


# ---------------------------------------------------------------------------
# Residual + bias add tests
# ---------------------------------------------------------------------------

@requires_metal
def test_residual_add_with_bias(runner):
    """output = input + residual + bias"""
    from triton_metal.codegen.msl_emitter import make_residual_add_kernel

    n = 1024
    msl = make_residual_add_kernel(block_size=256, has_bias=True)
    path = runner.compile(msl, "residual_add_kernel")
    pipeline = runner.load(path, "residual_add_kernel")

    in_data = [float(i) * 0.1 for i in range(n)]
    res_data = [float(i) * 0.05 for i in range(n)]
    bias_data = [0.5] * n

    in_buf = runner.make_float_buffer(in_data)
    res_buf = runner.make_float_buffer(res_data)
    bias_buf = runner.make_float_buffer(bias_data)
    out_buf = runner.make_empty_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [in_buf, res_buf, bias_buf, out_buf, n_buf], n)

    result = runner.read_float_buffer(out_buf, n)
    for i in range(n):
        expected = in_data[i] + res_data[i] + bias_data[i]
        assert abs(result[i] - expected) < 1e-4, (
            f"[{i}] got {result[i]}, expected {expected}"
        )


@requires_metal
def test_residual_add_no_bias(runner):
    """output = input + residual (no bias)"""
    from triton_metal.codegen.msl_emitter import make_residual_add_kernel

    n = 1024
    msl = make_residual_add_kernel(block_size=256, has_bias=False)
    path = runner.compile(msl, "residual_add_kernel")
    pipeline = runner.load(path, "residual_add_kernel")

    in_data = [float(i) * 0.1 for i in range(n)]
    res_data = [float(i) * 0.05 for i in range(n)]

    in_buf = runner.make_float_buffer(in_data)
    res_buf = runner.make_float_buffer(res_data)
    out_buf = runner.make_empty_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [in_buf, res_buf, out_buf, n_buf], n)

    result = runner.read_float_buffer(out_buf, n)
    for i in range(n):
        expected = in_data[i] + res_data[i]
        assert abs(result[i] - expected) < 1e-4, (
            f"[{i}] got {result[i]}, expected {expected}"
        )


# ---------------------------------------------------------------------------
# KV-cache attention tests
# ---------------------------------------------------------------------------

@requires_metal
def test_kv_cache_attention(runner):
    """KV-cache attention: single query token attending to cached KV."""
    from triton_metal.codegen.msl_emitter import make_kv_cache_attention_kernel

    head_dim = 64
    seq_len = 8
    n_heads = 1
    scale = 1.0 / math.sqrt(head_dim)

    msl = make_kv_cache_attention_kernel(head_dim=head_dim, block_size=256)
    path = runner.compile(msl, "kv_cache_attention")
    pipeline = runner.load(path, "kv_cache_attention")

    random.seed(777)
    Q_data = [random.gauss(0, 0.5) for _ in range(n_heads * head_dim)]
    K_data = [random.gauss(0, 0.5) for _ in range(n_heads * seq_len * head_dim)]
    V_data = [random.gauss(0, 0.5) for _ in range(n_heads * seq_len * head_dim)]

    Q_buf = runner.make_float_buffer(Q_data)
    K_buf = runner.make_float_buffer(K_data)
    V_buf = runner.make_float_buffer(V_data)
    O_buf = runner.make_empty_buffer(n_heads * head_dim)
    seq_buf = runner.make_uint_buffer(seq_len)
    scale_buf = runner.make_float_scalar_buffer(scale)

    import Metal
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([Q_buf, K_buf, V_buf, O_buf, seq_buf, scale_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_heads, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4, f"Kernel failed, status={cmd.status()}"

    result = runner.read_float_buffer(O_buf, n_heads * head_dim)

    # Reference: softmax(Q @ K^T * scale) @ V
    for h in range(n_heads):
        scores = []
        for j in range(seq_len):
            dot = sum(Q_data[h * head_dim + d] * K_data[h * seq_len * head_dim + j * head_dim + d]
                      for d in range(head_dim))
            scores.append(dot * scale)

        mx = max(scores)
        exps = [math.exp(s - mx) for s in scores]
        s = sum(exps)
        attn = [e / s for e in exps]

        for d in range(head_dim):
            expected = sum(attn[j] * V_data[h * seq_len * head_dim + j * head_dim + d]
                          for j in range(seq_len))
            got = result[h * head_dim + d]
            tol = max(1e-3, abs(expected) * 0.01)
            assert abs(got - expected) < tol, (
                f"head={h} dim={d}: got {got}, expected {expected}"
            )


# ---------------------------------------------------------------------------
# SwiGLU tests
# ---------------------------------------------------------------------------

@requires_metal
def test_swiglu(runner):
    """SwiGLU: output = SiLU(gate) * x"""
    from triton_metal.codegen.msl_emitter import make_swiglu_kernel

    n = 1024
    msl = make_swiglu_kernel(block_size=256)
    path = runner.compile(msl, "swiglu_kernel")
    pipeline = runner.load(path, "swiglu_kernel")

    random.seed(888)
    x_data = [random.gauss(0, 1) for _ in range(n)]
    gate_data = [random.gauss(0, 1) for _ in range(n)]

    x_buf = runner.make_float_buffer(x_data)
    gate_buf = runner.make_float_buffer(gate_data)
    out_buf = runner.make_empty_buffer(n)
    n_buf = runner.make_uint_buffer(n)

    runner.run(pipeline, [x_buf, gate_buf, out_buf, n_buf], n)

    result = runner.read_float_buffer(out_buf, n)
    for i in range(n):
        g = gate_data[i]
        silu_g = g / (1.0 + math.exp(-g))
        expected = silu_g * x_data[i]
        tol = max(1e-4, abs(expected) * 1e-3)
        assert abs(result[i] - expected) < tol, (
            f"[{i}] got {result[i]}, expected {expected}"
        )


# ---------------------------------------------------------------------------
# Embedding lookup tests
# ---------------------------------------------------------------------------

@requires_metal
def test_embedding_lookup(runner):
    """Embedding lookup: output[i] = table[indices[i]]"""
    from triton_metal.codegen.msl_emitter import make_embedding_kernel

    vocab_size = 32
    embed_dim = 64
    batch_size = 8

    msl = make_embedding_kernel(block_size=256)
    path = runner.compile(msl, "embedding_kernel")
    pipeline = runner.load(path, "embedding_kernel")

    random.seed(999)
    table_data = [random.gauss(0, 1) for _ in range(vocab_size * embed_dim)]
    indices = [random.randint(0, vocab_size - 1) for _ in range(batch_size)]

    table_buf = runner.make_float_buffer(table_data)

    import Metal
    import struct as st
    indices_buf = runner.device.newBufferWithLength_options_(
        batch_size * 4, Metal.MTLResourceStorageModeShared
    )
    view = indices_buf.contents().as_buffer(batch_size * 4)
    for i, idx in enumerate(indices):
        st.pack_into("i", view, i * 4, idx)

    out_buf = runner.make_empty_buffer(batch_size * embed_dim)
    dim_buf = runner.make_uint_buffer(embed_dim)

    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([table_buf, indices_buf, out_buf, dim_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(batch_size, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_float_buffer(out_buf, batch_size * embed_dim)

    for b in range(batch_size):
        token_idx = indices[b]
        for d in range(embed_dim):
            expected = table_data[token_idx * embed_dim + d]
            got = result[b * embed_dim + d]
            assert abs(got - expected) < 1e-5, (
                f"batch={b} dim={d}: got {got}, expected {expected}"
            )


# ---------------------------------------------------------------------------
# Causal Flash Attention tests
# ---------------------------------------------------------------------------

@requires_metal
def test_flash_attention_causal(runner):
    """Causal Flash Attention: future tokens should be masked."""
    from triton_metal.codegen.msl_emitter import make_flash_attention_kernel

    seq_len = 16
    head_dim = 64
    n_heads = 1
    scale = 1.0 / math.sqrt(head_dim)

    msl = make_flash_attention_kernel(head_dim=head_dim, Br=16, Bc=16, causal=True)
    path = runner.compile(msl, "flash_attention")
    pipeline = runner.load(path, "flash_attention")

    random.seed(1234)
    Q_data = [random.gauss(0, 0.5) for _ in range(n_heads * seq_len * head_dim)]
    K_data = [random.gauss(0, 0.5) for _ in range(n_heads * seq_len * head_dim)]
    V_data = [random.gauss(0, 0.5) for _ in range(n_heads * seq_len * head_dim)]

    Q_buf = runner.make_float_buffer(Q_data)
    K_buf = runner.make_float_buffer(K_data)
    V_buf = runner.make_float_buffer(V_data)
    O_buf = runner.make_empty_buffer(n_heads * seq_len * head_dim)
    seq_buf = runner.make_uint_buffer(seq_len)
    scale_buf = runner.make_float_scalar_buffer(scale)

    n_q_blocks = (seq_len + 15) // 16
    n_groups = n_heads * n_q_blocks

    import Metal
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([Q_buf, K_buf, V_buf, O_buf, seq_buf, scale_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_groups, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_float_buffer(O_buf, n_heads * seq_len * head_dim)

    # Reference: causal attention (lower triangular mask)
    S = [[0.0] * seq_len for _ in range(seq_len)]
    for i in range(seq_len):
        for j in range(seq_len):
            if j <= i:  # causal: attend only to past + self
                dot = sum(Q_data[i * head_dim + d] * K_data[j * head_dim + d]
                          for d in range(head_dim))
                S[i][j] = dot * scale
            else:
                S[i][j] = float('-inf')

    P = []
    for i in range(seq_len):
        mx = max(S[i])
        exps = [math.exp(s - mx) if s > float('-inf') else 0.0 for s in S[i]]
        s = sum(exps)
        P.append([e / s if s > 0 else 0.0 for e in exps])

    for i in range(seq_len):
        for d in range(0, head_dim, 8):  # spot-check every 8th dim
            expected = sum(P[i][j] * V_data[j * head_dim + d] for j in range(seq_len))
            got = result[i * head_dim + d]
            tol = max(1e-3, abs(expected) * 0.02)
            assert abs(got - expected) < tol, (
                f"pos={i} dim={d}: got {got}, expected {expected}"
            )

    # Verify causal property: first token output should only depend on first token
    # (i.e., O[0] = V[0] since attention is all on position 0)
    for d in range(head_dim):
        got = result[d]
        expected = V_data[d]  # position 0 attends only to itself
        tol = max(1e-3, abs(expected) * 0.02)
        assert abs(got - expected) < tol, (
            f"Causal check: O[0][{d}] = {got}, V[0][{d}] = {expected}"
        )


# ---------------------------------------------------------------------------
# Fused linear kernel tests
# ---------------------------------------------------------------------------

def _dispatch_fused_linear(runner, pipeline, buffers, M, N):
    """Dispatch a fused linear kernel with correct grid dimensions."""
    import Metal

    n_tile_cols = (N + 31) // 32
    n_tile_rows = (M + 31) // 32
    n_groups = n_tile_rows * n_tile_cols

    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate(buffers):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_groups, 1, 1),
        Metal.MTLSizeMake(128, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4, f"Kernel failed, status={cmd.status()}"


@requires_metal
def test_fused_linear_no_bias(runner):
    """Fused linear: output = input @ weight^T (no bias)."""
    from triton_metal.codegen.msl_emitter import make_fused_linear_kernel

    M, N, K = 32, 32, 32
    msl = make_fused_linear_kernel(has_bias=False)
    path = runner.compile(msl, "fused_linear")
    pipeline = runner.load(path, "fused_linear")

    random.seed(1111)
    input_data = [random.uniform(-1.0, 1.0) for _ in range(M * K)]
    # weight is [N, K] (row-major)
    weight_data = [random.uniform(-1.0, 1.0) for _ in range(N * K)]

    input_buf = runner.make_float_buffer(input_data)
    weight_buf = runner.make_float_buffer(weight_data)
    C_buf = runner.make_empty_buffer(M * N)
    M_buf = runner.make_uint_buffer(M)
    N_buf = runner.make_uint_buffer(N)
    K_buf = runner.make_uint_buffer(K)

    _dispatch_fused_linear(runner, pipeline,
                           [input_buf, weight_buf, C_buf, M_buf, N_buf, K_buf], M, N)

    result = runner.read_float_buffer(C_buf, M * N)

    # Reference: output = input @ weight^T
    for i in range(M):
        for j in range(N):
            expected = sum(input_data[i * K + k] * weight_data[j * K + k] for k in range(K))
            got = result[i * N + j]
            assert abs(got - expected) < 1e-2, (
                f"C[{i},{j}]: got {got}, expected {expected}"
            )


@requires_metal
def test_fused_linear_with_bias(runner):
    """Fused linear: output = input @ weight^T + bias."""
    from triton_metal.codegen.msl_emitter import make_fused_linear_kernel

    M, N, K = 32, 32, 32
    msl = make_fused_linear_kernel(has_bias=True)
    path = runner.compile(msl, "fused_linear")
    pipeline = runner.load(path, "fused_linear")

    random.seed(2222)
    input_data = [random.uniform(-1.0, 1.0) for _ in range(M * K)]
    weight_data = [random.uniform(-1.0, 1.0) for _ in range(N * K)]
    bias_data = [random.uniform(-0.5, 0.5) for _ in range(N)]

    input_buf = runner.make_float_buffer(input_data)
    weight_buf = runner.make_float_buffer(weight_data)
    C_buf = runner.make_empty_buffer(M * N)
    bias_buf = runner.make_float_buffer(bias_data)
    M_buf = runner.make_uint_buffer(M)
    N_buf = runner.make_uint_buffer(N)
    K_buf = runner.make_uint_buffer(K)

    _dispatch_fused_linear(runner, pipeline,
                           [input_buf, weight_buf, C_buf, bias_buf, M_buf, N_buf, K_buf], M, N)

    result = runner.read_float_buffer(C_buf, M * N)

    for i in range(M):
        for j in range(N):
            matmul = sum(input_data[i * K + k] * weight_data[j * K + k] for k in range(K))
            expected = matmul + bias_data[j]
            got = result[i * N + j]
            assert abs(got - expected) < 1e-1, (
                f"C[{i},{j}]: got {got}, expected {expected}"
            )


# ---------------------------------------------------------------------------
# Grouped Query Attention (GQA) tests
# ---------------------------------------------------------------------------


@requires_metal
def test_gqa_attention(runner):
    """GQA: 4 query heads share 1 KV head, verify attention output."""
    from triton_metal.codegen.msl_emitter import make_gqa_attention_kernel

    n_q_heads = 4
    n_kv_heads = 1
    n_q_per_kv = n_q_heads // n_kv_heads
    seq_len = 16
    head_dim = 64

    msl = make_gqa_attention_kernel(n_q_per_kv=n_q_per_kv)
    path = runner.compile(msl, "gqa_attention")
    pipeline = runner.load(path, "gqa_attention")

    random.seed(7777)
    # Q: [n_q_heads, head_dim] — each query head is different
    q_data = [random.uniform(-0.5, 0.5) for _ in range(n_q_heads * head_dim)]
    # K: [n_kv_heads, seq_len, head_dim]
    k_data = [random.uniform(-0.5, 0.5)
              for _ in range(n_kv_heads * seq_len * head_dim)]
    # V: [n_kv_heads, seq_len, head_dim]
    v_data = [random.uniform(-0.5, 0.5)
              for _ in range(n_kv_heads * seq_len * head_dim)]

    q_buf = runner.make_float_buffer(q_data)
    k_buf = runner.make_float_buffer(k_data)
    v_buf = runner.make_float_buffer(v_data)
    out_buf = runner.make_empty_buffer(n_q_heads * head_dim)
    seq_buf = runner.make_uint_buffer(seq_len)
    scale_val = 1.0 / math.sqrt(head_dim)
    scale_buf = runner.make_float_scalar_buffer(scale_val)

    import Metal
    n_groups = n_q_heads  # one threadgroup per query head
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([q_buf, k_buf, v_buf, out_buf,
                              seq_buf, scale_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_groups, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4, f"Kernel failed, status={cmd.status()}"

    result = runner.read_float_buffer(out_buf, n_q_heads * head_dim)

    # Reference: softmax(Q @ K^T / sqrt(d)) @ V for each query head
    # All query heads share the same KV (since n_kv_heads=1)
    scale = 1.0 / math.sqrt(head_dim)
    for qh in range(n_q_heads):
        kv_h = qh // n_q_per_kv  # which KV head to use

        # Compute attention scores
        scores = []
        for s in range(seq_len):
            dot = sum(q_data[qh * head_dim + d] *
                      k_data[(kv_h * seq_len + s) * head_dim + d]
                      for d in range(head_dim))
            scores.append(dot * scale)

        # Softmax
        max_s = max(scores)
        exp_s = [math.exp(s - max_s) for s in scores]
        sum_exp = sum(exp_s)
        attn = [e / sum_exp for e in exp_s]

        # Weighted sum of V
        for d in range(head_dim):
            expected = sum(attn[s] *
                          v_data[(kv_h * seq_len + s) * head_dim + d]
                          for s in range(seq_len))
            got = result[qh * head_dim + d]
            assert abs(got - expected) < 0.05, (
                f"head {qh}, dim {d}: got {got}, expected {expected}"
            )

    # Verify different query heads produce different outputs
    head0 = result[:head_dim]
    head1 = result[head_dim:2 * head_dim]
    diff = sum(abs(a - b) for a, b in zip(head0, head1))
    assert diff > 0.01, "Different query heads should produce different outputs"


# ---------------------------------------------------------------------------
# INT8 Quantized Matmul tests
# ---------------------------------------------------------------------------


@requires_metal
def test_int8_matmul(runner):
    """INT8 weight-only quantized matmul with per-row scale/zero_point."""
    from triton_metal.codegen.msl_emitter import make_int8_matmul_kernel
    import struct as struct_mod

    M, N, K = 16, 16, 32
    msl = make_int8_matmul_kernel()
    path = runner.compile(msl, "int8_matmul")
    pipeline = runner.load(path, "int8_matmul")

    random.seed(8888)
    # Input: float [M, K]
    input_data = [random.uniform(-1.0, 1.0) for _ in range(M * K)]
    # Quantized weights: int8 [N, K] stored as char
    scale = 0.1
    zero_point = 0
    # Generate int8 weights in range [-10, 10]
    weight_int8 = [random.randint(-10, 10) for _ in range(N * K)]
    # Scale per row: [N]
    scale_data = [scale] * N
    # Zero point per row: [N]
    zp_data = [float(zero_point)] * N

    # Create int8 weight buffer (packed as signed bytes)
    import Metal
    w_buf = runner.device.newBufferWithLength_options_(
        N * K, Metal.MTLResourceStorageModeShared
    )
    w_view = w_buf.contents().as_buffer(N * K)
    for i, val in enumerate(weight_int8):
        struct_mod.pack_into("b", w_view, i, val)

    input_buf = runner.make_float_buffer(input_data)
    out_buf = runner.make_empty_buffer(M * N)
    scale_buf = runner.make_float_buffer(scale_data)
    zp_buf = runner.make_float_buffer(zp_data)
    M_buf = runner.make_uint_buffer(M)
    N_buf = runner.make_uint_buffer(N)
    K_buf = runner.make_uint_buffer(K)

    # Dispatch: 1D grid, one thread per output element
    total_elements = M * N
    block_size = 256
    n_groups = (total_elements + block_size - 1) // block_size
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    # Buffer order matches kernel: input, weight, output, scales, zeros, M, N, K
    for i, buf in enumerate([input_buf, w_buf, out_buf, scale_buf, zp_buf,
                              M_buf, N_buf, K_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_groups, 1, 1),
        Metal.MTLSizeMake(block_size, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4, f"Kernel failed, status={cmd.status()}"

    result = runner.read_float_buffer(out_buf, M * N)

    # Reference: dequantize weights then matmul
    for i in range(M):
        for j in range(N):
            expected = sum(
                input_data[i * K + k] *
                (float(weight_int8[j * K + k]) - zero_point) * scale
                for k in range(K)
            )
            got = result[i * N + j]
            assert abs(got - expected) < 0.5, (
                f"C[{i},{j}]: got {got}, expected {expected}"
            )


# ---------------------------------------------------------------------------
# Concat and Split tests
# ---------------------------------------------------------------------------


@requires_metal
def test_concat_two_tensors(runner):
    """Concatenate two 1D tensors into one."""
    from triton_metal.codegen.msl_emitter import make_concat_kernel

    n0 = 128
    n1 = 256
    total = n0 + n1

    msl = make_concat_kernel(n_inputs=2)
    path = runner.compile(msl, "concat_kernel")
    pipeline = runner.load(path, "concat_kernel")

    a_data = [float(i) for i in range(n0)]
    b_data = [float(i + 1000) for i in range(n1)]

    a_buf = runner.make_float_buffer(a_data)
    b_buf = runner.make_float_buffer(b_data)
    out_buf = runner.make_empty_buffer(total)
    n0_buf = runner.make_uint_buffer(n0)
    n1_buf = runner.make_uint_buffer(n1)

    runner.run(pipeline, [a_buf, b_buf, out_buf, n0_buf, n1_buf],
               total, block_size=256)

    result = runner.read_float_buffer(out_buf, total)

    # First n0 elements should be a_data
    for i in range(n0):
        assert result[i] == pytest.approx(a_data[i], abs=0.01), (
            f"idx {i}: got {result[i]}, expected {a_data[i]}"
        )
    # Next n1 elements should be b_data
    for i in range(n1):
        assert result[n0 + i] == pytest.approx(b_data[i], abs=0.01), (
            f"idx {n0+i}: got {result[n0+i]}, expected {b_data[i]}"
        )


@requires_metal
def test_split_two_chunks(runner):
    """Split a tensor into two equal chunks."""
    from triton_metal.codegen.msl_emitter import make_split_kernel

    chunk_size = 128
    total = chunk_size * 2

    msl = make_split_kernel(n_outputs=2)
    path = runner.compile(msl, "split_kernel")
    pipeline = runner.load(path, "split_kernel")

    data = [float(i) for i in range(total)]
    in_buf = runner.make_float_buffer(data)
    out0_buf = runner.make_empty_buffer(chunk_size)
    out1_buf = runner.make_empty_buffer(chunk_size)
    cs_buf = runner.make_uint_buffer(chunk_size)

    runner.run(pipeline, [in_buf, out0_buf, out1_buf, cs_buf],
               total, block_size=256)

    r0 = runner.read_float_buffer(out0_buf, chunk_size)
    r1 = runner.read_float_buffer(out1_buf, chunk_size)

    for i in range(chunk_size):
        assert r0[i] == pytest.approx(data[i], abs=0.01)
        assert r1[i] == pytest.approx(data[chunk_size + i], abs=0.01)

# ---------------------------------------------------------------------------
# Top-K Sampling tests
# ---------------------------------------------------------------------------


@requires_metal
def test_top_k_basic(runner):
    """Top-k: find top 5 values from 1024-element vocabulary."""
    from triton_metal.codegen.msl_emitter import make_top_k_kernel
    import struct as struct_mod

    vocab_size = 1024
    k = 5

    msl = make_top_k_kernel(k=k, block_size=256)
    path = runner.compile(msl, "top_k")
    pipeline = runner.load(path, "top_k")

    # Create logits with known top-k values
    random.seed(9999)
    logits = [random.uniform(-2.0, 2.0) for _ in range(vocab_size)]
    # Plant known large values at specific indices
    top_indices = [17, 42, 500, 731, 999]
    top_values = [10.0, 9.5, 9.0, 8.5, 8.0]
    for idx, val in zip(top_indices, top_values):
        logits[idx] = val

    logits_buf = runner.make_float_buffer(logits)
    out_val_buf = runner.make_empty_buffer(k)
    # Create uint output buffer for indices
    import Metal
    out_idx_buf = runner.device.newBufferWithLength_options_(
        k * 4, Metal.MTLResourceStorageModeShared
    )
    vocab_buf = runner.make_uint_buffer(vocab_size)
    k_buf = runner.make_uint_buffer(k)

    runner.run(pipeline, [logits_buf, out_val_buf, out_idx_buf, vocab_buf, k_buf],
               256, block_size=256)

    result_vals = runner.read_float_buffer(out_val_buf, k)
    # Read uint indices
    idx_view = out_idx_buf.contents().as_buffer(k * 4)
    result_idxs = [struct_mod.unpack_from("I", idx_view, i * 4)[0] for i in range(k)]

    # Verify the top-k values are returned in descending order
    assert result_vals[0] == pytest.approx(10.0, abs=0.01)
    assert result_vals[1] == pytest.approx(9.5, abs=0.01)
    assert result_vals[2] == pytest.approx(9.0, abs=0.01)
    assert result_vals[3] == pytest.approx(8.5, abs=0.01)
    assert result_vals[4] == pytest.approx(8.0, abs=0.01)

    # Verify indices match
    assert set(result_idxs) == set(top_indices)
    assert result_idxs[0] == 17   # index of highest value
    assert result_idxs[1] == 42


@requires_metal
def test_top_k_large_vocab(runner):
    """Top-k with a larger vocabulary (32K) and k=10."""
    from triton_metal.codegen.msl_emitter import make_top_k_kernel
    import struct as struct_mod

    vocab_size = 32768
    k = 10

    msl = make_top_k_kernel(k=k, block_size=256)
    path = runner.compile(msl, "top_k")
    pipeline = runner.load(path, "top_k")

    random.seed(1234)
    logits = [random.uniform(-5.0, 5.0) for _ in range(vocab_size)]

    # Reference: sort and take top k
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    expected_indices = [idx for idx, _ in indexed[:k]]
    expected_values = [val for _, val in indexed[:k]]

    logits_buf = runner.make_float_buffer(logits)
    out_val_buf = runner.make_empty_buffer(k)
    import Metal
    out_idx_buf = runner.device.newBufferWithLength_options_(
        k * 4, Metal.MTLResourceStorageModeShared
    )
    vocab_buf = runner.make_uint_buffer(vocab_size)
    k_buf = runner.make_uint_buffer(k)

    runner.run(pipeline, [logits_buf, out_val_buf, out_idx_buf, vocab_buf, k_buf],
               256, block_size=256)

    result_vals = runner.read_float_buffer(out_val_buf, k)
    idx_view = out_idx_buf.contents().as_buffer(k * 4)
    result_idxs = [struct_mod.unpack_from("I", idx_view, i * 4)[0] for i in range(k)]

    # All returned values should match the reference top-k
    for i in range(k):
        assert result_vals[i] == pytest.approx(expected_values[i], abs=0.01), (
            f"top-{i}: got {result_vals[i]}, expected {expected_values[i]}"
        )
        assert result_idxs[i] == expected_indices[i], (
            f"top-{i} index: got {result_idxs[i]}, expected {expected_indices[i]}"
        )


# ---------------------------------------------------------------------------
# Top-P (Nucleus) Sampling tests
# ---------------------------------------------------------------------------


@requires_metal
def test_top_p_sampling(runner):
    """Top-p: nucleus sampling with p=0.9 and temperature=1.0."""
    from triton_metal.codegen.msl_emitter import make_top_p_kernel
    import struct as struct_mod

    vocab_size = 512
    max_k = 256
    temperature = 1.0
    p_threshold = 0.9

    msl = make_top_p_kernel(max_k=max_k, block_size=256)
    path = runner.compile(msl, "top_p")
    pipeline = runner.load(path, "top_p")

    # Create logits with a peaked distribution (few tokens dominate)
    random.seed(5555)
    logits = [random.uniform(-10.0, -5.0) for _ in range(vocab_size)]
    # Make a few tokens very likely (high logits vs very low background)
    logits[10] = 10.0
    logits[20] = 9.5
    logits[30] = 9.0
    logits[40] = 8.5

    logits_buf = runner.make_float_buffer(logits)
    out_val_buf = runner.make_empty_buffer(max_k)
    import Metal
    out_idx_buf = runner.device.newBufferWithLength_options_(
        max_k * 4, Metal.MTLResourceStorageModeShared
    )
    out_count_buf = runner.device.newBufferWithLength_options_(
        4, Metal.MTLResourceStorageModeShared
    )
    vocab_buf = runner.make_uint_buffer(vocab_size)
    temp_buf = runner.make_float_scalar_buffer(temperature)
    p_buf = runner.make_float_scalar_buffer(p_threshold)

    runner.run(pipeline,
               [logits_buf, out_val_buf, out_idx_buf, out_count_buf,
                vocab_buf, temp_buf, p_buf],
               256, block_size=256)

    # Read count
    count_view = out_count_buf.contents().as_buffer(4)
    count = struct_mod.unpack_from("I", count_view, 0)[0]

    result_vals = runner.read_float_buffer(out_val_buf, count)
    idx_view = out_idx_buf.contents().as_buffer(count * 4)
    result_idxs = [struct_mod.unpack_from("I", idx_view, i * 4)[0] for i in range(count)]

    # Verify: probabilities should be in descending order
    for i in range(len(result_vals) - 1):
        assert result_vals[i] >= result_vals[i + 1] - 1e-6, (
            f"Not sorted: p[{i}]={result_vals[i]} < p[{i+1}]={result_vals[i+1]}"
        )

    # Verify: cumulative probability should reach p_threshold
    cum = sum(result_vals)
    assert cum >= p_threshold - 0.01, f"Cumulative {cum} < threshold {p_threshold}"

    # Verify: top indices should include our peaked tokens
    assert 10 in result_idxs, "Highest logit token should be in nucleus"
    assert 20 in result_idxs, "Second highest logit token should be in nucleus"

    # The nucleus should be small since the distribution is peaked
    assert count < 50, f"Peaked distribution should have small nucleus, got {count}"


# ---------------------------------------------------------------------------
# Batched KV-Cache Decode tests
# ---------------------------------------------------------------------------


@requires_metal
def test_batched_kv_decode(runner):
    """Batched multi-head KV-cache decode with 2 batch items, 2 heads."""
    from triton_metal.codegen.msl_emitter import make_batched_kv_decode_kernel

    batch_size = 2
    n_heads = 2
    head_dim = 32
    max_seq_len = 16
    # Different sequence lengths per batch item
    actual_seq_lens = [8, 12]

    msl = make_batched_kv_decode_kernel(n_heads=n_heads, head_dim=head_dim,
                                         block_size=256)
    path = runner.compile(msl, "batched_kv_decode")
    pipeline = runner.load(path, "batched_kv_decode")

    random.seed(6666)
    # Q: [batch, n_heads, head_dim]
    q_data = [random.uniform(-0.5, 0.5)
              for _ in range(batch_size * n_heads * head_dim)]
    # K/V: [batch, n_heads, max_seq_len, head_dim]
    kv_size = batch_size * n_heads * max_seq_len * head_dim
    k_data = [random.uniform(-0.5, 0.5) for _ in range(kv_size)]
    v_data = [random.uniform(-0.5, 0.5) for _ in range(kv_size)]

    q_buf = runner.make_float_buffer(q_data)
    k_buf = runner.make_float_buffer(k_data)
    v_buf = runner.make_float_buffer(v_data)
    out_buf = runner.make_empty_buffer(batch_size * n_heads * head_dim)

    # seq_lens buffer: array of uints
    import Metal
    import struct as struct_mod
    sl_buf = runner.device.newBufferWithLength_options_(
        batch_size * 4, Metal.MTLResourceStorageModeShared
    )
    sl_view = sl_buf.contents().as_buffer(batch_size * 4)
    for i, sl in enumerate(actual_seq_lens):
        struct_mod.pack_into("I", sl_view, i * 4, sl)

    max_sl_buf = runner.make_uint_buffer(max_seq_len)
    bs_buf = runner.make_uint_buffer(batch_size)

    # Dispatch: one threadgroup per (batch, head)
    n_groups = batch_size * n_heads
    import Metal
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([q_buf, k_buf, v_buf, out_buf, sl_buf,
                              max_sl_buf, bs_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_groups, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4, f"Kernel failed, status={cmd.status()}"

    result = runner.read_float_buffer(out_buf, batch_size * n_heads * head_dim)

    # Reference implementation
    scale = 1.0 / math.sqrt(head_dim)
    for b in range(batch_size):
        seq_len = actual_seq_lens[b]
        for h in range(n_heads):
            q_off = (b * n_heads + h) * head_dim
            kv_off = (b * n_heads + h) * max_seq_len * head_dim
            o_off = q_off

            # Compute scores
            scores = []
            for j in range(seq_len):
                dot = sum(q_data[q_off + d] * k_data[kv_off + j * head_dim + d]
                          for d in range(head_dim))
                scores.append(dot * scale)

            # Softmax
            max_s = max(scores)
            exp_s = [math.exp(s - max_s) for s in scores]
            sum_exp = sum(exp_s)
            attn = [e / sum_exp for e in exp_s]

            # Weighted V
            for d in range(head_dim):
                expected = sum(attn[j] * v_data[kv_off + j * head_dim + d]
                               for j in range(seq_len))
                got = result[o_off + d]
                assert abs(got - expected) < 0.05, (
                    f"batch {b} head {h} dim {d}: got {got}, expected {expected}"
                )


# ---------------------------------------------------------------------------
# INT4 Quantized Matmul tests
# ---------------------------------------------------------------------------


@requires_metal
def test_int4_matmul(runner):
    """INT4 weight-only quantized matmul with per-group scale/zero_point."""
    from triton_metal.codegen.msl_emitter import make_int4_matmul_kernel
    import struct as struct_mod

    M, N, K = 8, 8, 16  # K must be even for int4 packing
    group_size = 16  # One group covers entire K
    msl = make_int4_matmul_kernel(group_size=group_size)
    path = runner.compile(msl, "int4_matmul")
    pipeline = runner.load(path, "int4_matmul")

    random.seed(7070)
    # Input: float [M, K]
    input_data = [random.uniform(-1.0, 1.0) for _ in range(M * K)]
    # INT4 weights: values 0-15, packed 2 per byte
    # Generate random 4-bit values
    weight_int4 = [random.randint(0, 15) for _ in range(N * K)]
    # Pack into bytes: even=low nibble, odd=high nibble
    packed_bytes = []
    for n in range(N):
        for k_pair in range(K // 2):
            low = weight_int4[n * K + k_pair * 2]
            high = weight_int4[n * K + k_pair * 2 + 1]
            packed_bytes.append((high << 4) | low)

    # Scale and zero per group
    n_groups = (K + group_size - 1) // group_size
    scale = 0.1
    zero_point = 8.0  # center of uint4 range
    scale_data = [scale] * (N * n_groups)
    zp_data = [zero_point] * (N * n_groups)

    # Create packed weight buffer
    import Metal
    w_buf = runner.device.newBufferWithLength_options_(
        len(packed_bytes), Metal.MTLResourceStorageModeShared
    )
    w_view = w_buf.contents().as_buffer(len(packed_bytes))
    for i, val in enumerate(packed_bytes):
        struct_mod.pack_into("B", w_view, i, val)

    input_buf = runner.make_float_buffer(input_data)
    out_buf = runner.make_empty_buffer(M * N)
    scale_buf = runner.make_float_buffer(scale_data)
    zp_buf = runner.make_float_buffer(zp_data)
    M_buf = runner.make_uint_buffer(M)
    N_buf = runner.make_uint_buffer(N)
    K_buf = runner.make_uint_buffer(K)

    total = M * N
    block_size = 256
    n_threadgroups = (total + block_size - 1) // block_size
    runner.run(pipeline,
               [input_buf, w_buf, out_buf, scale_buf, zp_buf,
                M_buf, N_buf, K_buf],
               total, block_size=block_size)

    result = runner.read_float_buffer(out_buf, M * N)

    # Reference: dequantize and matmul
    for i in range(M):
        for j in range(N):
            expected = sum(
                input_data[i * K + k] *
                (float(weight_int4[j * K + k]) - zero_point) * scale
                for k in range(K)
            )
            got = result[i * N + j]
            assert abs(got - expected) < 0.5, (
                f"C[{i},{j}]: got {got}, expected {expected}"
            )


# ---------------------------------------------------------------------------
# Fused RoPE + Attention tests
# ---------------------------------------------------------------------------


@requires_metal
def test_rope_attention(runner):
    """Fused RoPE + single-query attention: applies rotary embeddings to Q and K
    on-the-fly during attention, validating against a Python reference."""
    from triton_metal.codegen.msl_emitter import make_rope_attention_kernel

    head_dim = 64
    seq_len = 16
    q_pos = 10  # Position of the query token

    msl = make_rope_attention_kernel(head_dim=head_dim, block_size=256)
    path = runner.compile(msl, "rope_attention")
    pipeline = runner.load(path, "rope_attention")

    random.seed(8888)
    half_d = head_dim // 2

    # Q: [head_dim] — single query vector
    q_data = [random.uniform(-0.5, 0.5) for _ in range(head_dim)]
    # K_cache: [seq_len, head_dim] — UN-rotated cached keys
    k_data = [random.uniform(-0.5, 0.5) for _ in range(seq_len * head_dim)]
    # V_cache: [seq_len, head_dim]
    v_data = [random.uniform(-0.5, 0.5) for _ in range(seq_len * head_dim)]
    # freqs: [head_dim/2] — RoPE frequencies: 1/10000^(2i/d)
    freqs = [1.0 / (10000.0 ** (2.0 * i / head_dim)) for i in range(half_d)]

    q_buf = runner.make_float_buffer(q_data)
    k_buf = runner.make_float_buffer(k_data)
    v_buf = runner.make_float_buffer(v_data)
    freqs_buf = runner.make_float_buffer(freqs)
    out_buf = runner.make_empty_buffer(head_dim)
    seq_buf = runner.make_uint_buffer(seq_len)
    qpos_buf = runner.make_uint_buffer(q_pos)

    # Custom dispatch: single threadgroup
    import Metal
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([q_buf, k_buf, v_buf, freqs_buf, out_buf,
                              seq_buf, qpos_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4, f"Kernel failed, status={cmd.status()}"

    result = runner.read_float_buffer(out_buf, head_dim)

    # Python reference: apply RoPE to Q at q_pos
    q_rot = [0.0] * head_dim
    for d in range(half_d):
        theta = q_pos * freqs[d]
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        q_r = q_data[2 * d]
        q_i = q_data[2 * d + 1]
        q_rot[2 * d] = q_r * cos_t - q_i * sin_t
        q_rot[2 * d + 1] = q_r * sin_t + q_i * cos_t

    # Compute attention scores with RoPE on each K[j]
    scale = 1.0 / math.sqrt(head_dim)
    scores = []
    for j in range(seq_len):
        dot = 0.0
        for d in range(half_d):
            theta = j * freqs[d]
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            k_r = k_data[j * head_dim + 2 * d]
            k_i = k_data[j * head_dim + 2 * d + 1]
            k_rot_r = k_r * cos_t - k_i * sin_t
            k_rot_i = k_r * sin_t + k_i * cos_t
            dot += q_rot[2 * d] * k_rot_r + q_rot[2 * d + 1] * k_rot_i
        scores.append(dot * scale)

    # Softmax
    max_s = max(scores)
    exp_s = [math.exp(s - max_s) for s in scores]
    sum_exp = sum(exp_s)
    attn = [e / sum_exp for e in exp_s]

    # Weighted V (V is NOT rotated)
    for d in range(head_dim):
        expected = sum(attn[j] * v_data[j * head_dim + d] for j in range(seq_len))
        got = result[d]
        assert abs(got - expected) < 0.05, (
            f"dim {d}: got {got}, expected {expected}"
        )


# ---------------------------------------------------------------------------
# Paged Attention tests
# ---------------------------------------------------------------------------


@requires_metal
def test_paged_attention(runner):
    """Paged attention: KV-cache stored in non-contiguous pages with page table."""
    from triton_metal.codegen.msl_emitter import make_paged_attention_kernel
    import struct as struct_mod

    head_dim = 64
    page_size = 16
    seq_len = 48  # 3 full pages
    n_pages = 3
    # Shuffle physical pages: logical [0,1,2] -> physical [2,0,1]
    page_table = [2, 0, 1]

    msl = make_paged_attention_kernel(head_dim=head_dim, page_size=page_size)
    path = runner.compile(msl, "paged_attention")
    pipeline = runner.load(path, "paged_attention")

    random.seed(7777)

    # Q: [head_dim]
    q_data = [random.uniform(-0.5, 0.5) for _ in range(head_dim)]

    # K/V pages: [n_physical_pages, page_size, head_dim]
    # We'll create 3 physical pages
    n_physical_pages = 3
    total_page_elems = n_physical_pages * page_size * head_dim
    k_page_data = [random.uniform(-0.5, 0.5) for _ in range(total_page_elems)]
    v_page_data = [random.uniform(-0.5, 0.5) for _ in range(total_page_elems)]

    q_buf = runner.make_float_buffer(q_data)
    k_buf = runner.make_float_buffer(k_page_data)
    v_buf = runner.make_float_buffer(v_page_data)

    # Page table buffer (uint array)
    import Metal
    pt_buf = runner.device.newBufferWithLength_options_(
        n_pages * 4, Metal.MTLResourceStorageModeShared
    )
    pt_view = pt_buf.contents().as_buffer(n_pages * 4)
    for i, phys in enumerate(page_table):
        struct_mod.pack_into("I", pt_view, i * 4, phys)

    out_buf = runner.make_empty_buffer(head_dim)
    sl_buf = runner.make_uint_buffer(seq_len)
    np_buf = runner.make_uint_buffer(n_pages)

    # Dispatch: 1 threadgroup
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([q_buf, k_buf, v_buf, pt_buf, out_buf,
                              sl_buf, np_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4, f"Kernel failed, status={cmd.status()}"

    result = runner.read_float_buffer(out_buf, head_dim)

    # Python reference: reconstruct contiguous K,V from pages
    scale = 1.0 / math.sqrt(head_dim)
    scores = []
    for pos in range(seq_len):
        page_idx = pos // page_size
        page_offset = pos % page_size
        phys_page = page_table[page_idx]
        k_base = (phys_page * page_size + page_offset) * head_dim

        dot = sum(q_data[d] * k_page_data[k_base + d] for d in range(head_dim))
        scores.append(dot * scale)

    max_s = max(scores)
    exp_s = [math.exp(s - max_s) for s in scores]
    sum_exp = sum(exp_s)
    attn = [e / sum_exp for e in exp_s]

    for d in range(head_dim):
        expected = 0.0
        for pos in range(seq_len):
            page_idx = pos // page_size
            page_offset = pos % page_size
            phys_page = page_table[page_idx]
            v_base = (phys_page * page_size + page_offset) * head_dim
            expected += attn[pos] * v_page_data[v_base + d]
        got = result[d]
        assert abs(got - expected) < 0.05, (
            f"dim {d}: got {got}, expected {expected}"
        )


@requires_metal
def test_paged_attention_partial_page(runner):
    """Paged attention with a partially filled last page (seq_len not page-aligned)."""
    from triton_metal.codegen.msl_emitter import make_paged_attention_kernel
    import struct as struct_mod

    head_dim = 32
    page_size = 16
    seq_len = 25  # 1 full page + 9 tokens in second page
    n_pages = 2
    page_table = [1, 0]  # Reversed physical order

    msl = make_paged_attention_kernel(head_dim=head_dim, page_size=page_size)
    path = runner.compile(msl, "paged_attention_partial")
    pipeline = runner.load(path, "paged_attention")

    random.seed(5555)
    q_data = [random.uniform(-0.5, 0.5) for _ in range(head_dim)]
    n_physical_pages = 2
    total_elems = n_physical_pages * page_size * head_dim
    k_data = [random.uniform(-0.5, 0.5) for _ in range(total_elems)]
    v_data = [random.uniform(-0.5, 0.5) for _ in range(total_elems)]

    q_buf = runner.make_float_buffer(q_data)
    k_buf = runner.make_float_buffer(k_data)
    v_buf = runner.make_float_buffer(v_data)

    import Metal
    pt_buf = runner.device.newBufferWithLength_options_(
        n_pages * 4, Metal.MTLResourceStorageModeShared
    )
    pt_view = pt_buf.contents().as_buffer(n_pages * 4)
    for i, phys in enumerate(page_table):
        struct_mod.pack_into("I", pt_view, i * 4, phys)

    out_buf = runner.make_empty_buffer(head_dim)
    sl_buf = runner.make_uint_buffer(seq_len)
    np_buf = runner.make_uint_buffer(n_pages)

    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([q_buf, k_buf, v_buf, pt_buf, out_buf,
                              sl_buf, np_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4, f"Kernel failed, status={cmd.status()}"

    result = runner.read_float_buffer(out_buf, head_dim)

    # Reference
    scale = 1.0 / math.sqrt(head_dim)
    scores = []
    for pos in range(seq_len):
        pi = pos // page_size
        po = pos % page_size
        pp = page_table[pi]
        kb = (pp * page_size + po) * head_dim
        dot = sum(q_data[d] * k_data[kb + d] for d in range(head_dim))
        scores.append(dot * scale)

    max_s = max(scores)
    exp_s = [math.exp(s - max_s) for s in scores]
    sum_exp = sum(exp_s)
    attn = [e / sum_exp for e in exp_s]

    for d in range(head_dim):
        expected = 0.0
        for pos in range(seq_len):
            pi = pos // page_size
            po = pos % page_size
            pp = page_table[pi]
            vb = (pp * page_size + po) * head_dim
            expected += attn[pos] * v_data[vb + d]
        got = result[d]
        assert abs(got - expected) < 0.05, (
            f"dim {d}: got {got}, expected {expected}"
        )
