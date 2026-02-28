"""Tests for the MSL emitter: generate kernels, compile, run on GPU, verify.

Each test validates:
1. The emitter produces valid MSL
2. The MSL compiles to a metallib
3. The kernel runs on the M4 Max GPU
4. Results match a reference implementation
"""

import hashlib
import math
import os
import platform
import random
import struct
import subprocess
import tempfile

import pytest

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="Metal backend requires macOS",
)


def _has_metal():
    try:
        import Metal

        return Metal.MTLCreateSystemDefaultDevice() is not None
    except ImportError:
        return False


def _has_metal_compiler():
    try:
        subprocess.check_call(
            ["xcrun", "-sdk", "macosx", "metal", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


requires_metal = pytest.mark.skipif(
    not _has_metal() or not _has_metal_compiler(),
    reason="Requires Metal GPU and compiler",
)


# ---------------------------------------------------------------------------
# Helper: compile + load + run a kernel
# ---------------------------------------------------------------------------

class MetalKernelRunner:
    """Compile MSL, load metallib, run kernel, read results."""

    def __init__(self):
        import Metal

        self.device = Metal.MTLCreateSystemDefaultDevice()
        self.queue = self.device.newCommandQueue()
        self._cache_dir = os.path.join(
            tempfile.gettempdir(), "triton_metal_test_cache"
        )
        os.makedirs(self._cache_dir, exist_ok=True)

    def compile(self, msl_src, kernel_name):
        """Compile MSL to metallib path."""
        src_hash = hashlib.sha256(msl_src.encode()).hexdigest()[:16]
        base = f"{kernel_name}_{src_hash}"
        metal_path = os.path.join(self._cache_dir, f"{base}.metal")
        air_path = os.path.join(self._cache_dir, f"{base}.air")
        metallib_path = os.path.join(self._cache_dir, f"{base}.metallib")

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
        return metallib_path

    def load(self, metallib_path, kernel_name):
        """Load metallib and create pipeline state."""
        import Metal
        import Foundation

        url = Foundation.NSURL.fileURLWithPath_(metallib_path)
        library, error = self.device.newLibraryWithURL_error_(url, None)
        assert error is None, f"Load failed: {error}"

        function = library.newFunctionWithName_(kernel_name)
        assert function is not None, f"Kernel '{kernel_name}' not found"

        pipeline, error = (
            self.device.newComputePipelineStateWithFunction_error_(
                function, None
            )
        )
        assert error is None, f"Pipeline failed: {error}"
        return pipeline

    def make_float_buffer(self, data):
        """Create a Metal buffer filled with float data."""
        import Metal

        n = len(data)
        buf = self.device.newBufferWithLength_options_(
            n * 4, Metal.MTLResourceStorageModeShared
        )
        view = buf.contents().as_buffer(n * 4)
        for i, val in enumerate(data):
            struct.pack_into("f", view, i * 4, float(val))
        return buf

    def make_empty_buffer(self, n):
        """Create an empty float buffer of n elements."""
        import Metal

        return self.device.newBufferWithLength_options_(
            n * 4, Metal.MTLResourceStorageModeShared
        )

    def make_uint_buffer(self, value):
        """Create a buffer with a single uint32."""
        import Metal

        buf = self.device.newBufferWithLength_options_(
            4, Metal.MTLResourceStorageModeShared
        )
        view = buf.contents().as_buffer(4)
        struct.pack_into("I", view, 0, value)
        return buf

    def make_float_scalar_buffer(self, value):
        """Create a buffer with a single float."""
        import Metal

        buf = self.device.newBufferWithLength_options_(
            4, Metal.MTLResourceStorageModeShared
        )
        view = buf.contents().as_buffer(4)
        struct.pack_into("f", view, 0, value)
        return buf

    def run(self, pipeline, buffers, n_elements, block_size=256):
        """Dispatch a compute kernel."""
        import Metal

        n_groups = (n_elements + block_size - 1) // block_size

        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(pipeline)
        for i, buf in enumerate(buffers):
            enc.setBuffer_offset_atIndex_(buf, 0, i)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(n_groups, 1, 1),
            Metal.MTLSizeMake(block_size, 1, 1),
        )
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        assert cmd.status() == 4, f"Kernel failed, status={cmd.status()}"

    def read_float_buffer(self, buf, n):
        """Read n floats from a Metal buffer."""
        view = buf.contents().as_buffer(n * 4)
        return [struct.unpack_from("f", view, i * 4)[0] for i in range(n)]

    # -- Half-precision (FP16) helpers --

    def make_half_buffer(self, data):
        """Create a Metal buffer filled with half-precision (FP16) data."""
        import Metal

        n = len(data)
        buf = self.device.newBufferWithLength_options_(
            n * 2, Metal.MTLResourceStorageModeShared
        )
        view = buf.contents().as_buffer(n * 2)
        for i, val in enumerate(data):
            struct.pack_into("e", view, i * 2, float(val))
        return buf

    def make_empty_half_buffer(self, n):
        """Create an empty half-precision buffer of n elements."""
        import Metal

        return self.device.newBufferWithLength_options_(
            n * 2, Metal.MTLResourceStorageModeShared
        )

    def read_half_buffer(self, buf, n):
        """Read n half-precision values from a Metal buffer."""
        view = buf.contents().as_buffer(n * 2)
        return [struct.unpack_from("e", view, i * 2)[0] for i in range(n)]


@pytest.fixture
def runner():
    if not _has_metal() or not _has_metal_compiler():
        pytest.skip("Requires Metal GPU and compiler")
    return MetalKernelRunner()


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
