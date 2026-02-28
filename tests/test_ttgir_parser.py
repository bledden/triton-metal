"""Tests for the TTGIR MLIR text parser.

Feeds real TTGIR dumps into the parser and verifies:
1. Correct kernel name extraction
2. Correct argument registration (pointers vs scalars)
3. Valid MSL generation
4. MSL compiles with xcrun metal
5. (When possible) GPU execution produces correct results
"""

import math
import platform

import pytest

from tests.conftest import requires_metal

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="Metal backend requires macOS",
)


class FakeOptions:
    """Minimal MetalOptions-like object for testing."""
    def __init__(self, num_warps=4):
        self.num_warps = num_warps


# ---------------------------------------------------------------------------
# TTGIR test inputs
# ---------------------------------------------------------------------------

# Simple vector add: C = A + B
VECADD_TTGIR = """\
module {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) {
    %0 = tt.get_program_id x : i32
    %c256_i32 = arith.constant 256 : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %3 = tt.splat %1 : i32 -> tensor<256xi32>
    %4 = arith.addi %3, %2 : tensor<256xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<256xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<256xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %9 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %10 = tt.addptr %9, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %cst = arith.constant 0.000000e+00 : f32
    %11 = tt.splat %cst : f32 -> tensor<256xf32>
    %12 = tt.load %8, %6, %11 : tensor<256x!tt.ptr<f32>>
    %13 = tt.load %10, %6, %11 : tensor<256x!tt.ptr<f32>>
    %14 = arith.addf %12, %13 : tensor<256xf32>
    %15 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %16 = tt.addptr %15, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %16, %14, %6 : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}
"""

# Simple elementwise multiply: C = A * B
VECMUL_TTGIR = """\
module {
  tt.func public @mul_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) {
    %0 = tt.get_program_id x : i32
    %c256_i32 = arith.constant 256 : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %3 = tt.splat %1 : i32 -> tensor<256xi32>
    %4 = arith.addi %3, %2 : tensor<256xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<256xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<256xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %9 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %10 = tt.addptr %9, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %cst = arith.constant 0.000000e+00 : f32
    %11 = tt.splat %cst : f32 -> tensor<256xf32>
    %12 = tt.load %8, %6, %11 : tensor<256x!tt.ptr<f32>>
    %13 = tt.load %10, %6, %11 : tensor<256x!tt.ptr<f32>>
    %14 = arith.mulf %12, %13 : tensor<256xf32>
    %15 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %16 = tt.addptr %15, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %16, %14, %6 : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}
"""

# Unary exp: B = exp(A)
EXP_TTGIR = """\
module {
  tt.func public @exp_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
    %0 = tt.get_program_id x : i32
    %c256_i32 = arith.constant 256 : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %3 = tt.splat %1 : i32 -> tensor<256xi32>
    %4 = arith.addi %3, %2 : tensor<256xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<256xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<256xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %cst = arith.constant 0.000000e+00 : f32
    %9 = tt.splat %cst : f32 -> tensor<256xf32>
    %10 = tt.load %8, %6, %9 : tensor<256x!tt.ptr<f32>>
    %11 = math.exp %10 : tensor<256xf32>
    %12 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %13 = tt.addptr %12, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %13, %11, %6 : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}
"""


# FP16 vector add: C = A + B with half-precision inputs/outputs
VECADD_FP16_TTGIR = """\
module {
  tt.func public @add_f16_kernel(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: i32) {
    %0 = tt.get_program_id x : i32
    %c256_i32 = arith.constant 256 : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %3 = tt.splat %1 : i32 -> tensor<256xi32>
    %4 = arith.addi %3, %2 : tensor<256xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<256xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<256xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f16>>, tensor<256xi32>
    %9 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>>
    %10 = tt.addptr %9, %4 : tensor<256x!tt.ptr<f16>>, tensor<256xi32>
    %cst = arith.constant 0.000000e+00 : f16
    %11 = tt.splat %cst : f16 -> tensor<256xf16>
    %12 = tt.load %8, %6, %11 : tensor<256x!tt.ptr<f16>>
    %13 = tt.load %10, %6, %11 : tensor<256x!tt.ptr<f16>>
    %14 = arith.extf %12 : tensor<256xf16> to tensor<256xf32>
    %15 = arith.extf %13 : tensor<256xf16> to tensor<256xf32>
    %16 = arith.addf %14, %15 : tensor<256xf32>
    %17 = arith.truncf %16 : tensor<256xf32> to tensor<256xf16>
    %18 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>>
    %19 = tt.addptr %18, %4 : tensor<256x!tt.ptr<f16>>, tensor<256xi32>
    tt.store %19, %17, %6 : tensor<256x!tt.ptr<f16>>
    tt.return
  }
}
"""

# Negate + select: C = A > 0 ? A : -A (abs via select)
SELECT_TTGIR = """\
module {
  tt.func public @abs_select_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
    %0 = tt.get_program_id x : i32
    %c256_i32 = arith.constant 256 : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %3 = tt.splat %1 : i32 -> tensor<256xi32>
    %4 = arith.addi %3, %2 : tensor<256xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<256xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<256xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %cst = arith.constant 0.000000e+00 : f32
    %9 = tt.splat %cst : f32 -> tensor<256xf32>
    %10 = tt.load %8, %6, %9 : tensor<256x!tt.ptr<f32>>
    %11 = arith.negf %10 : tensor<256xf32>
    %12 = arith.addf %10, %11 : tensor<256xf32>
    %13 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %14 = tt.addptr %13, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %14, %12, %6 : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}
"""

# Sum reduction: output = sum(input)
SUM_REDUCE_TTGIR = """\
module {
  tt.func public @sum_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
    %0 = tt.get_program_id x : i32
    %c256_i32 = arith.constant 256 : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %3 = tt.splat %1 : i32 -> tensor<256xi32>
    %4 = arith.addi %3, %2 : tensor<256xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<256xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<256xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %cst = arith.constant 0.000000e+00 : f32
    %9 = tt.splat %cst : f32 -> tensor<256xf32>
    %10 = tt.load %8, %6, %9 : tensor<256x!tt.ptr<f32>>
    %11 = "tt.reduce"(%10) ({
    ^bb0(%arg3: f32, %arg4: f32):
      %13 = arith.addf %arg3, %arg4 : f32
      "tt.reduce.return"(%13) : (f32) -> ()
    }) {axis = 0 : i32} : (tensor<256xf32>) -> f32
    tt.store %arg1, %11 : !tt.ptr<f32>
    tt.return
  }
}
"""

# Max reduction: output = max(input)
MAX_REDUCE_TTGIR = """\
module {
  tt.func public @max_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
    %0 = tt.get_program_id x : i32
    %c256_i32 = arith.constant 256 : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %3 = tt.splat %1 : i32 -> tensor<256xi32>
    %4 = arith.addi %3, %2 : tensor<256xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<256xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<256xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %cst = arith.constant 0.000000e+00 : f32
    %9 = tt.splat %cst : f32 -> tensor<256xf32>
    %10 = tt.load %8, %6, %9 : tensor<256x!tt.ptr<f32>>
    %11 = "tt.reduce"(%10) ({
    ^bb0(%arg3: f32, %arg4: f32):
      %13 = arith.maxf %arg3, %arg4 : f32
      "tt.reduce.return"(%13) : (f32) -> ()
    }) {axis = 0 : i32} : (tensor<256xf32>) -> f32
    tt.store %arg1, %11 : !tt.ptr<f32>
    tt.return
  }
}
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_parse_vecadd_name():
    """Parser extracts kernel name correctly."""
    from triton_metal.codegen.ttgir_parser import parse_ttgir

    kb = parse_ttgir(VECADD_TTGIR, FakeOptions())
    assert kb.name == "add_kernel"


def test_parse_vecadd_args():
    """Parser identifies 3 pointer args and 1 scalar arg."""
    from triton_metal.codegen.ttgir_parser import parse_ttgir

    kb = parse_ttgir(VECADD_TTGIR, FakeOptions())
    ptr_args = [a for a in kb.args if a.is_ptr]
    scalar_args = [a for a in kb.args if not a.is_ptr]
    assert len(ptr_args) == 3
    assert len(scalar_args) == 1
    assert scalar_args[0].dtype == "i32"


def test_parse_vecadd_output_detection():
    """Parser detects which pointer args are outputs (have stores)."""
    from triton_metal.codegen.ttgir_parser import parse_ttgir

    kb = parse_ttgir(VECADD_TTGIR, FakeOptions())
    # arg2 is the output (has tt.store)
    # arg0 and arg1 are inputs (const)
    ptr_args = [a for a in kb.args if a.is_ptr]
    const_count = sum(1 for a in ptr_args if a.const)
    mutable_count = sum(1 for a in ptr_args if not a.const)
    assert const_count == 2, f"Expected 2 const args, got {const_count}"
    assert mutable_count == 1, f"Expected 1 mutable arg, got {mutable_count}"


@requires_metal
def test_parse_vecadd_compiles(runner):
    """MSL generated from parsed vector add TTGIR compiles."""
    from triton_metal.codegen.ttgir_parser import parse_ttgir

    kb = parse_ttgir(VECADD_TTGIR, FakeOptions())
    msl = kb.build()
    runner.compile(msl, "add_kernel")


@requires_metal
def test_parse_vecmul_compiles(runner):
    """MSL generated from parsed multiply TTGIR compiles."""
    from triton_metal.codegen.ttgir_parser import parse_ttgir

    kb = parse_ttgir(VECMUL_TTGIR, FakeOptions())
    msl = kb.build()
    runner.compile(msl, "mul_kernel")


@requires_metal
def test_parse_exp_compiles(runner):
    """MSL generated from parsed exp TTGIR compiles."""
    from triton_metal.codegen.ttgir_parser import parse_ttgir

    kb = parse_ttgir(EXP_TTGIR, FakeOptions())
    msl = kb.build()
    runner.compile(msl, "exp_kernel")


def test_parse_block_size():
    """Parser extracts block size from tt.make_range."""
    from triton_metal.codegen.ttgir_parser import parse_ttgir

    kb = parse_ttgir(VECADD_TTGIR, FakeOptions())
    assert kb.block_size >= 256


def test_parse_vecadd_generates_add_op():
    """Parsed vector add MSL contains an addition operation."""
    from triton_metal.codegen.ttgir_parser import parse_ttgir

    kb = parse_ttgir(VECADD_TTGIR, FakeOptions())
    msl = kb.build()
    assert "+" in msl or "add" in msl.lower(), f"No addition found in MSL:\n{msl}"


def test_parse_vecmul_generates_mul_op():
    """Parsed multiply MSL contains a multiply operation."""
    from triton_metal.codegen.ttgir_parser import parse_ttgir

    kb = parse_ttgir(VECMUL_TTGIR, FakeOptions())
    msl = kb.build()
    assert "*" in msl, f"No multiplication found in MSL:\n{msl}"


def test_parse_exp_generates_exp_op():
    """Parsed exp MSL contains an exp() call."""
    from triton_metal.codegen.ttgir_parser import parse_ttgir

    kb = parse_ttgir(EXP_TTGIR, FakeOptions())
    msl = kb.build()
    assert "exp(" in msl, f"No exp() found in MSL:\n{msl}"


# ---------------------------------------------------------------------------
# GPU execution tests — verify TTGIR-parsed kernels produce correct results
# ---------------------------------------------------------------------------

@requires_metal
def test_ttgir_vecadd_gpu(runner):
    """TTGIR vector add: C = A + B, verified on GPU."""
    from triton_metal.codegen.ttgir_parser import parse_ttgir

    kb = parse_ttgir(VECADD_TTGIR, FakeOptions())
    msl = kb.build()
    n = 512

    metallib = runner.compile(msl, "add_kernel")
    pipeline = runner.load(metallib, "add_kernel")

    a_data = [float(i) for i in range(n)]
    b_data = [float(i) * 0.5 for i in range(n)]
    buf_a = runner.make_float_buffer(a_data)
    buf_b = runner.make_float_buffer(b_data)
    buf_c = runner.make_empty_buffer(n)
    buf_n = runner.make_int_buffer(n)

    runner.run(pipeline, [buf_a, buf_b, buf_c, buf_n], n, block_size=256)

    result = runner.read_float_buffer(buf_c, n)
    for i in range(n):
        expected = a_data[i] + b_data[i]
        assert abs(result[i] - expected) < 1e-5, (
            f"Mismatch at {i}: got {result[i]}, expected {expected}"
        )


@requires_metal
def test_ttgir_vecmul_gpu(runner):
    """TTGIR vector multiply: C = A * B, verified on GPU."""
    from triton_metal.codegen.ttgir_parser import parse_ttgir

    kb = parse_ttgir(VECMUL_TTGIR, FakeOptions())
    msl = kb.build()
    n = 512

    metallib = runner.compile(msl, "mul_kernel")
    pipeline = runner.load(metallib, "mul_kernel")

    a_data = [float(i) * 0.1 for i in range(n)]
    b_data = [float(i) * 0.2 for i in range(n)]
    buf_a = runner.make_float_buffer(a_data)
    buf_b = runner.make_float_buffer(b_data)
    buf_c = runner.make_empty_buffer(n)
    buf_n = runner.make_int_buffer(n)

    runner.run(pipeline, [buf_a, buf_b, buf_c, buf_n], n, block_size=256)

    result = runner.read_float_buffer(buf_c, n)
    for i in range(n):
        expected = a_data[i] * b_data[i]
        tol = max(1e-4, abs(expected) * 1e-6)
        assert abs(result[i] - expected) < tol, (
            f"Mismatch at {i}: got {result[i]}, expected {expected}"
        )


@requires_metal
def test_ttgir_exp_gpu(runner):
    """TTGIR exp: B = exp(A), verified on GPU."""
    from triton_metal.codegen.ttgir_parser import parse_ttgir

    kb = parse_ttgir(EXP_TTGIR, FakeOptions())
    msl = kb.build()
    n = 256

    metallib = runner.compile(msl, "exp_kernel")
    pipeline = runner.load(metallib, "exp_kernel")

    # Use small values to avoid overflow
    a_data = [float(i) * 0.01 for i in range(n)]
    buf_a = runner.make_float_buffer(a_data)
    buf_b = runner.make_empty_buffer(n)
    buf_n = runner.make_int_buffer(n)

    runner.run(pipeline, [buf_a, buf_b, buf_n], n, block_size=256)

    result = runner.read_float_buffer(buf_b, n)
    for i in range(n):
        expected = math.exp(a_data[i])
        assert abs(result[i] - expected) < 1e-4, (
            f"Mismatch at {i}: got {result[i]}, expected {expected}"
        )


@requires_metal
def test_ttgir_vecadd_non_aligned(runner):
    """TTGIR vector add with non-block-aligned size (tests masking)."""
    from triton_metal.codegen.ttgir_parser import parse_ttgir

    kb = parse_ttgir(VECADD_TTGIR, FakeOptions())
    msl = kb.build()
    n = 300  # Not a multiple of 256

    metallib = runner.compile(msl, "add_kernel")
    pipeline = runner.load(metallib, "add_kernel")

    a_data = [float(i) for i in range(n)]
    b_data = [1.0] * n
    # Allocate padded buffers (full block)
    buf_a = runner.make_float_buffer(a_data + [0.0] * (256 - n % 256))
    buf_b = runner.make_float_buffer(b_data + [0.0] * (256 - n % 256))
    buf_c = runner.make_empty_buffer(n + (256 - n % 256))
    buf_n = runner.make_int_buffer(n)

    runner.run(pipeline, [buf_a, buf_b, buf_c, buf_n], n, block_size=256)

    result = runner.read_float_buffer(buf_c, n)
    for i in range(n):
        expected = a_data[i] + b_data[i]
        assert abs(result[i] - expected) < 1e-5, (
            f"Mismatch at {i}: got {result[i]}, expected {expected}"
        )


# ---------------------------------------------------------------------------
# Reduction TTGIR tests
# ---------------------------------------------------------------------------

def test_parse_sum_reduce_detects_reduction():
    """Parser detects tt.reduce with arith.addf as a sum reduction."""
    from triton_metal.codegen.ttgir_parser import parse_ttgir

    kb = parse_ttgir(SUM_REDUCE_TTGIR, FakeOptions())
    assert kb.name == "sum_kernel"
    assert kb._needs_simd_qualifiers


def test_parse_max_reduce_detects_reduction():
    """Parser detects tt.reduce with arith.maxf as a max reduction."""
    from triton_metal.codegen.ttgir_parser import parse_ttgir

    kb = parse_ttgir(MAX_REDUCE_TTGIR, FakeOptions())
    assert kb.name == "max_kernel"
    assert kb._needs_simd_qualifiers


@requires_metal
def test_ttgir_sum_reduce_compiles(runner):
    """TTGIR sum reduction compiles to valid MSL."""
    from triton_metal.codegen.ttgir_parser import parse_ttgir

    kb = parse_ttgir(SUM_REDUCE_TTGIR, FakeOptions())
    msl = kb.build()
    runner.compile(msl, "sum_kernel")


@requires_metal
def test_ttgir_max_reduce_compiles(runner):
    """TTGIR max reduction compiles to valid MSL."""
    from triton_metal.codegen.ttgir_parser import parse_ttgir

    kb = parse_ttgir(MAX_REDUCE_TTGIR, FakeOptions())
    msl = kb.build()
    runner.compile(msl, "max_kernel")


@requires_metal
def test_ttgir_sum_reduce_gpu(runner):
    """TTGIR sum reduction: output = sum(input), verified on GPU."""
    from triton_metal.codegen.ttgir_parser import parse_ttgir

    kb = parse_ttgir(SUM_REDUCE_TTGIR, FakeOptions())
    msl = kb.build()
    n = 256

    metallib = runner.compile(msl, "sum_kernel")
    pipeline = runner.load(metallib, "sum_kernel")

    input_data = [float(i) * 0.01 for i in range(n)]
    buf_in = runner.make_float_buffer(input_data)
    buf_out = runner.make_empty_buffer(1)
    buf_n = runner.make_int_buffer(n)

    runner.run(pipeline, [buf_in, buf_out, buf_n], n, block_size=256)

    result = runner.read_float_buffer(buf_out, 1)
    expected = sum(input_data)
    assert abs(result[0] - expected) < 0.5, (
        f"Sum: got {result[0]}, expected {expected}"
    )


@requires_metal
def test_ttgir_max_reduce_gpu(runner):
    """TTGIR max reduction: output = max(input), verified on GPU."""
    from triton_metal.codegen.ttgir_parser import parse_ttgir
    import random

    kb = parse_ttgir(MAX_REDUCE_TTGIR, FakeOptions())
    msl = kb.build()
    n = 256

    metallib = runner.compile(msl, "max_kernel")
    pipeline = runner.load(metallib, "max_kernel")

    random.seed(606)
    input_data = [random.uniform(-100.0, 100.0) for _ in range(n)]
    buf_in = runner.make_float_buffer(input_data)
    buf_out = runner.make_empty_buffer(1)
    buf_n = runner.make_int_buffer(n)

    runner.run(pipeline, [buf_in, buf_out, buf_n], n, block_size=256)

    result = runner.read_float_buffer(buf_out, 1)
    expected = max(input_data)
    assert abs(result[0] - expected) < 1e-3, (
        f"Max: got {result[0]}, expected {expected}"
    )


# ---------------------------------------------------------------------------
# FP16 and extended ops TTGIR tests
# ---------------------------------------------------------------------------

def test_parse_fp16_args():
    """Parser detects fp16 pointer types correctly."""
    from triton_metal.codegen.ttgir_parser import parse_ttgir

    kb = parse_ttgir(VECADD_FP16_TTGIR, FakeOptions())
    assert kb.name == "add_f16_kernel"
    ptr_args = [a for a in kb.args if a.is_ptr]
    assert all(a.dtype == "fp16" for a in ptr_args)


@requires_metal
def test_ttgir_fp16_vecadd_compiles(runner):
    """TTGIR FP16 vector add compiles to valid MSL."""
    from triton_metal.codegen.ttgir_parser import parse_ttgir

    kb = parse_ttgir(VECADD_FP16_TTGIR, FakeOptions())
    msl = kb.build()
    runner.compile(msl, "add_f16_kernel")


@requires_metal
def test_ttgir_fp16_vecadd_gpu(runner):
    """TTGIR FP16 vector add: C = A + B in half precision, verified on GPU."""
    from triton_metal.codegen.ttgir_parser import parse_ttgir

    kb = parse_ttgir(VECADD_FP16_TTGIR, FakeOptions())
    msl = kb.build()
    n = 512

    metallib = runner.compile(msl, "add_f16_kernel")
    pipeline = runner.load(metallib, "add_f16_kernel")

    a_data = [float(i) * 0.01 for i in range(n)]
    b_data = [float(i) * 0.005 for i in range(n)]
    buf_a = runner.make_half_buffer(a_data)
    buf_b = runner.make_half_buffer(b_data)
    buf_c = runner.make_empty_half_buffer(n)
    buf_n = runner.make_int_buffer(n)

    runner.run(pipeline, [buf_a, buf_b, buf_c, buf_n], n, block_size=256)

    result = runner.read_half_buffer(buf_c, n)
    for i in range(n):
        expected = a_data[i] + b_data[i]
        tol = max(1e-2, abs(expected) * 1e-2)
        assert abs(result[i] - expected) < tol, (
            f"[{i}] got {result[i]}, expected {expected}"
        )


@requires_metal
def test_ttgir_negf_compiles(runner):
    """TTGIR with arith.negf compiles to valid MSL."""
    from triton_metal.codegen.ttgir_parser import parse_ttgir

    kb = parse_ttgir(SELECT_TTGIR, FakeOptions())
    msl = kb.build()
    runner.compile(msl, "abs_select_kernel")


@requires_metal
def test_ttgir_negf_gpu(runner):
    """TTGIR negate + add: output = x + (-x) = 0, verified on GPU."""
    from triton_metal.codegen.ttgir_parser import parse_ttgir

    kb = parse_ttgir(SELECT_TTGIR, FakeOptions())
    msl = kb.build()
    n = 256

    metallib = runner.compile(msl, "abs_select_kernel")
    pipeline = runner.load(metallib, "abs_select_kernel")

    input_data = [float(i) - 128.0 for i in range(n)]
    buf_in = runner.make_float_buffer(input_data)
    buf_out = runner.make_empty_buffer(n)
    buf_n = runner.make_int_buffer(n)

    runner.run(pipeline, [buf_in, buf_out, buf_n], n, block_size=256)

    result = runner.read_float_buffer(buf_out, n)
    for i in range(n):
        # x + (-x) should be 0
        assert abs(result[i]) < 1e-5, (
            f"[{i}] got {result[i]}, expected 0.0"
        )
