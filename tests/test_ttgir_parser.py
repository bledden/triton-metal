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


# Softmax: output = softmax(input) — fused max + exp + sum + divide
SOFTMAX_TTGIR = """\
module {
  tt.func public @softmax_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
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
      %m = arith.maxf %arg3, %arg4 : f32
      "tt.reduce.return"(%m) : (f32) -> ()
    }) {axis = 0 : i32} : (tensor<256xf32>) -> f32
    %12 = tt.splat %11 : f32 -> tensor<256xf32>
    %13 = arith.subf %10, %12 : tensor<256xf32>
    %14 = math.exp %13 : tensor<256xf32>
    %15 = "tt.reduce"(%14) ({
    ^bb0(%arg5: f32, %arg6: f32):
      %s = arith.addf %arg5, %arg6 : f32
      "tt.reduce.return"(%s) : (f32) -> ()
    }) {axis = 0 : i32} : (tensor<256xf32>) -> f32
    %16 = tt.splat %15 : f32 -> tensor<256xf32>
    %17 = arith.divf %14, %16 : tensor<256xf32>
    %18 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %19 = tt.addptr %18, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %19, %17, %6 : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}
"""


# Layer norm: two sum reductions (mean + variance), subtract, rsqrt, multiply
LAYER_NORM_TTGIR = """\
module {
  tt.func public @layer_norm_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: !tt.ptr<f32>, %arg4: i32) {
    %0 = tt.get_program_id x : i32
    %c256_i32 = arith.constant 256 : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %3 = tt.splat %1 : i32 -> tensor<256xi32>
    %4 = arith.addi %3, %2 : tensor<256xi32>
    %5 = tt.splat %arg4 : i32 -> tensor<256xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<256xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %cst = arith.constant 0.000000e+00 : f32
    %9 = tt.splat %cst : f32 -> tensor<256xf32>
    %10 = tt.load %8, %6, %9 : tensor<256x!tt.ptr<f32>>
    %mean_sum = "tt.reduce"(%10) ({
    ^bb0(%arg5: f32, %arg6: f32):
      %s1 = arith.addf %arg5, %arg6 : f32
      "tt.reduce.return"(%s1) : (f32) -> ()
    }) {axis = 0 : i32} : (tensor<256xf32>) -> f32
    %mean_splat = tt.splat %mean_sum : f32 -> tensor<256xf32>
    %centered = arith.subf %10, %mean_splat : tensor<256xf32>
    %sq = arith.mulf %centered, %centered : tensor<256xf32>
    %var_sum = "tt.reduce"(%sq) ({
    ^bb0(%arg7: f32, %arg8: f32):
      %s2 = arith.addf %arg7, %arg8 : f32
      "tt.reduce.return"(%s2) : (f32) -> ()
    }) {axis = 0 : i32} : (tensor<256xf32>) -> f32
    %eps = arith.constant 1.000000e-06 : f32
    %inv_std = math.rsqrt %var_sum : f32
    %11 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %12 = tt.addptr %11, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %gamma_ptr = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %gamma_ptrs = tt.addptr %gamma_ptr, %2 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %gamma = tt.load %gamma_ptrs : tensor<256x!tt.ptr<f32>>
    %normed = arith.mulf %centered, %gamma : tensor<256xf32>
    tt.store %12, %normed, %6 : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}
"""


# scf.for loop: accumulates over chunks with a loop structure
SCF_FOR_TTGIR = """\
module {
  tt.func public @loop_sum_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
    %pid = tt.get_program_id x : i32
    %c0 = arith.constant 0 : i32
    %c256 = arith.constant 256 : i32
    %cst_zero = arith.constant 0.000000e+00 : f32
    %result = scf.for %iv = %c0 to %arg2 step %c256 iter_args(%acc = %cst_zero) -> (f32) {
      %range = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
      %iv_splat = tt.splat %iv : i32 -> tensor<256xi32>
      %offsets = arith.addi %iv_splat, %range : tensor<256xi32>
      %n_splat = tt.splat %arg2 : i32 -> tensor<256xi32>
      %mask = arith.cmpi slt, %offsets, %n_splat : tensor<256xi32>
      %ptr_splat = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
      %ptrs = tt.addptr %ptr_splat, %offsets : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
      %zero_splat = tt.splat %cst_zero : f32 -> tensor<256xf32>
      %loaded = tt.load %ptrs, %mask, %zero_splat : tensor<256x!tt.ptr<f32>>
      %chunk_sum = "tt.reduce"(%loaded) ({
      ^bb0(%a: f32, %b: f32):
        %s = arith.addf %a, %b : f32
        "tt.reduce.return"(%s) : (f32) -> ()
      }) {axis = 0 : i32} : (tensor<256xf32>) -> f32
      %new_acc = arith.addf %acc, %chunk_sum : f32
      scf.yield %new_acc : f32
    }
    tt.store %arg1, %result : !tt.ptr<f32>
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


# ---------------------------------------------------------------------------
# Softmax TTGIR tests (multi-reduce pattern)
# ---------------------------------------------------------------------------

def test_parse_softmax_detects_multi_reduce():
    """Parser detects softmax pattern: max reduce + sum reduce."""
    from triton_metal.codegen.ttgir_parser import TTGIRParser

    parser = TTGIRParser(SOFTMAX_TTGIR, FakeOptions())
    parser._parse_function_signature()
    parser._parse_body()
    assert len(parser.reduce_ops) == 2
    ops = [r[2] for r in parser.reduce_ops]
    assert "max" in ops
    assert "sum" in ops


def test_parse_softmax_routes_to_softmax_kernel():
    """Parser routes multi-reduce (max+sum) to softmax kernel builder."""
    from triton_metal.codegen.ttgir_parser import parse_ttgir

    kb = parse_ttgir(SOFTMAX_TTGIR, FakeOptions())
    assert kb.name == "softmax_kernel"
    # Softmax needs SIMD qualifiers for reductions
    assert kb._needs_simd_qualifiers
    # Should have 2 shared memory arrays (shared_max, shared_sum)
    tg_names = [name for name, _, _ in kb._threadgroup_arrays]
    assert "shared_max" in tg_names
    assert "shared_sum" in tg_names


@requires_metal
def test_ttgir_softmax_compiles(runner):
    """TTGIR softmax compiles to valid MSL."""
    from triton_metal.codegen.ttgir_parser import parse_ttgir

    kb = parse_ttgir(SOFTMAX_TTGIR, FakeOptions())
    msl = kb.build()
    runner.compile(msl, "softmax_kernel")


@requires_metal
def test_ttgir_softmax_gpu(runner):
    """TTGIR softmax: output = softmax(input), verified on GPU."""
    from triton_metal.codegen.ttgir_parser import parse_ttgir

    kb = parse_ttgir(SOFTMAX_TTGIR, FakeOptions())
    msl = kb.build()
    n = 256

    metallib = runner.compile(msl, "softmax_kernel")
    pipeline = runner.load(metallib, "softmax_kernel")

    # Test data: a row of values
    input_data = [float(i) * 0.1 - 12.8 for i in range(n)]
    buf_in = runner.make_float_buffer(input_data)
    buf_out = runner.make_empty_buffer(n)
    buf_n = runner.make_int_buffer(n)

    runner.run(pipeline, [buf_in, buf_out, buf_n], n, block_size=256)

    result = runner.read_float_buffer(buf_out, n)

    # Reference softmax
    max_val = max(input_data)
    exps = [math.exp(x - max_val) for x in input_data]
    sum_exp = sum(exps)
    expected = [e / sum_exp for e in exps]

    for i in range(n):
        tol = max(1e-5, abs(expected[i]) * 1e-4)
        assert abs(result[i] - expected[i]) < tol, (
            f"[{i}] got {result[i]}, expected {expected[i]}"
        )

    # Verify outputs sum to 1.0
    total = sum(result)
    assert abs(total - 1.0) < 1e-4, f"Softmax sum: {total}, expected 1.0"


@requires_metal
def test_ttgir_softmax_multi_row_gpu(runner):
    """TTGIR softmax with multiple rows (one threadgroup per row)."""
    from triton_metal.codegen.ttgir_parser import parse_ttgir
    import random

    kb = parse_ttgir(SOFTMAX_TTGIR, FakeOptions())
    msl = kb.build()
    n_cols = 256
    n_rows = 4

    metallib = runner.compile(msl, "softmax_kernel")
    pipeline = runner.load(metallib, "softmax_kernel")

    random.seed(42)
    input_data = [random.gauss(0, 2) for _ in range(n_rows * n_cols)]
    buf_in = runner.make_float_buffer(input_data)
    buf_out = runner.make_empty_buffer(n_rows * n_cols)
    buf_n = runner.make_int_buffer(n_cols)

    # Dispatch n_rows threadgroups
    n_groups = n_rows
    import Metal
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    enc.setBuffer_offset_atIndex_(buf_in, 0, 0)
    enc.setBuffer_offset_atIndex_(buf_out, 0, 1)
    enc.setBuffer_offset_atIndex_(buf_n, 0, 2)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_groups, 1, 1),
        Metal.MTLSizeMake(256, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4

    result = runner.read_float_buffer(buf_out, n_rows * n_cols)

    # Verify each row sums to 1.0 and matches reference
    for row in range(n_rows):
        row_in = input_data[row * n_cols:(row + 1) * n_cols]
        row_out = result[row * n_cols:(row + 1) * n_cols]

        max_val = max(row_in)
        exps = [math.exp(x - max_val) for x in row_in]
        sum_exp = sum(exps)
        expected = [e / sum_exp for e in exps]

        row_sum = sum(row_out)
        assert abs(row_sum - 1.0) < 1e-3, (
            f"Row {row}: sum={row_sum}, expected 1.0"
        )

        for i in range(n_cols):
            tol = max(1e-5, abs(expected[i]) * 1e-3)
            assert abs(row_out[i] - expected[i]) < tol, (
                f"Row {row}[{i}] got {row_out[i]}, expected {expected[i]}"
            )


# ---------------------------------------------------------------------------
# scf.for loop detection tests
# ---------------------------------------------------------------------------

def test_parse_scf_for_detects_loop():
    """Parser detects scf.for loop structures in TTGIR."""
    from triton_metal.codegen.ttgir_parser import TTGIRParser

    parser = TTGIRParser(SCF_FOR_TTGIR, FakeOptions())
    parser._parse_function_signature()
    parser._scan_scf_for_loops()

    assert len(parser.scf_for_loops) == 1
    loop = parser.scf_for_loops[0]
    assert loop['iv'] == 'iv'
    assert loop['lb'] == '%c0'
    assert loop['ub'] == '%arg2'
    assert loop['step'] == '%c256'
    # Should have iter_args (accumulator)
    assert len(loop['iter_args']) == 1
    assert loop['iter_args'][0][0] == 'acc'  # iter arg name
    assert loop['iter_args'][0][1] == '%cst_zero'  # init value


def test_parse_scf_for_with_reduce():
    """Parser detects both scf.for and tt.reduce in looped reduction."""
    from triton_metal.codegen.ttgir_parser import TTGIRParser

    parser = TTGIRParser(SCF_FOR_TTGIR, FakeOptions())
    parser._parse_function_signature()
    parser._parse_body()

    # Should find the scf.for loop
    assert len(parser.scf_for_loops) == 1
    # Should also find the tt.reduce inside the loop body
    assert len(parser.reduce_ops) == 1
    assert parser.reduce_ops[0][2] == "sum"


def test_parse_scf_for_loop_iv_in_ssa():
    """Loop induction variable is tracked in SSA values."""
    from triton_metal.codegen.ttgir_parser import TTGIRParser

    parser = TTGIRParser(SCF_FOR_TTGIR, FakeOptions())
    parser._parse_function_signature()
    parser._scan_scf_for_loops()

    assert "%iv" in parser.ssa_values
    assert parser.ssa_values["%iv"][0] == "loop_iv"


# ---------------------------------------------------------------------------
# FP64 rejection tests
# ---------------------------------------------------------------------------

# FMA: output = a * b + c (fused multiply-add)
FMA_TTGIR = """\
module {
  tt.func public @fma_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: !tt.ptr<f32>, %arg4: i32) {
    %0 = tt.get_program_id x : i32
    %c256_i32 = arith.constant 256 : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %3 = tt.splat %1 : i32 -> tensor<256xi32>
    %4 = arith.addi %3, %2 : tensor<256xi32>
    %5 = tt.splat %arg4 : i32 -> tensor<256xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<256xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %9 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %10 = tt.addptr %9, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %11 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %12 = tt.addptr %11, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %cst = arith.constant 0.000000e+00 : f32
    %13 = tt.splat %cst : f32 -> tensor<256xf32>
    %14 = tt.load %8, %6, %13 : tensor<256x!tt.ptr<f32>>
    %15 = tt.load %10, %6, %13 : tensor<256x!tt.ptr<f32>>
    %16 = tt.load %12, %6, %13 : tensor<256x!tt.ptr<f32>>
    %17 = math.fma %14, %15, %16 : tensor<256xf32>
    %18 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %19 = tt.addptr %18, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %19, %17, %6 : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}
"""

# RSQRT: output = rsqrt(input) = 1/sqrt(input)
RSQRT_TTGIR = """\
module {
  tt.func public @rsqrt_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
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
    %11 = math.rsqrt %10 : tensor<256xf32>
    %12 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %13 = tt.addptr %12, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %13, %11, %6 : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}
"""

FP64_TTGIR = """\
module {
  tt.func public @fp64_kernel(%arg0: !tt.ptr<f64>, %arg1: !tt.ptr<f64>, %arg2: i32) {
    %0 = tt.get_program_id x : i32
    tt.return
  }
}
"""


def test_fp64_rejected_in_parser():
    """Parser rejects FP64 types with a clear error message."""
    from triton_metal.codegen.ttgir_parser import parse_ttgir

    with pytest.raises(TypeError, match="FP64.*not supported.*Apple Silicon"):
        parse_ttgir(FP64_TTGIR, FakeOptions())


def test_fp64_rejected_in_msl_types():
    """MSL type mapper rejects FP64 types."""
    from triton_metal.codegen.msl_types import triton_type_to_msl

    with pytest.raises(TypeError, match="FP64.*not supported"):
        triton_type_to_msl("fp64")

    with pytest.raises(TypeError, match="FP64.*not supported"):
        triton_type_to_msl("*fp64")


def test_fp64_rejected_in_emitter():
    """MSL emitter rejects FP64 kernel generation."""
    from triton_metal.codegen.msl_emitter import make_elementwise_kernel

    with pytest.raises(TypeError, match="FP64.*not supported"):
        make_elementwise_kernel("bad_kernel", 2, "add", dtype="fp64")


# ---------------------------------------------------------------------------
# math.fma and math.rsqrt tests
# ---------------------------------------------------------------------------

@requires_metal
def test_ttgir_fma_compiles(runner):
    """TTGIR math.fma compiles to valid MSL."""
    from triton_metal.codegen.ttgir_parser import parse_ttgir

    kb = parse_ttgir(FMA_TTGIR, FakeOptions())
    msl = kb.build()
    assert "fma(" in msl
    runner.compile(msl, "fma_kernel")


@requires_metal
def test_ttgir_fma_gpu(runner):
    """TTGIR fma: output = a*b + c, verified on GPU."""
    from triton_metal.codegen.ttgir_parser import parse_ttgir

    kb = parse_ttgir(FMA_TTGIR, FakeOptions())
    msl = kb.build()
    n = 256

    metallib = runner.compile(msl, "fma_kernel")
    pipeline = runner.load(metallib, "fma_kernel")

    a_data = [float(i) * 0.1 for i in range(n)]
    b_data = [float(i) * 0.2 for i in range(n)]
    c_data = [float(i) * 0.05 for i in range(n)]
    buf_a = runner.make_float_buffer(a_data)
    buf_b = runner.make_float_buffer(b_data)
    buf_c = runner.make_float_buffer(c_data)
    buf_out = runner.make_empty_buffer(n)
    buf_n = runner.make_int_buffer(n)

    runner.run(pipeline, [buf_a, buf_b, buf_c, buf_out, buf_n], n, block_size=256)

    result = runner.read_float_buffer(buf_out, n)
    for i in range(n):
        expected = a_data[i] * b_data[i] + c_data[i]
        tol = max(1e-4, abs(expected) * 1e-5)
        assert abs(result[i] - expected) < tol, (
            f"[{i}] got {result[i]}, expected {expected}"
        )


@requires_metal
def test_ttgir_rsqrt_compiles(runner):
    """TTGIR math.rsqrt compiles to valid MSL."""
    from triton_metal.codegen.ttgir_parser import parse_ttgir

    kb = parse_ttgir(RSQRT_TTGIR, FakeOptions())
    msl = kb.build()
    assert "rsqrt(" in msl
    runner.compile(msl, "rsqrt_kernel")


@requires_metal
def test_ttgir_rsqrt_gpu(runner):
    """TTGIR rsqrt: output = 1/sqrt(input), verified on GPU."""
    from triton_metal.codegen.ttgir_parser import parse_ttgir

    kb = parse_ttgir(RSQRT_TTGIR, FakeOptions())
    msl = kb.build()
    n = 256

    metallib = runner.compile(msl, "rsqrt_kernel")
    pipeline = runner.load(metallib, "rsqrt_kernel")

    # Use positive values to avoid NaN
    input_data = [float(i + 1) * 0.1 for i in range(n)]
    buf_in = runner.make_float_buffer(input_data)
    buf_out = runner.make_empty_buffer(n)
    buf_n = runner.make_int_buffer(n)

    runner.run(pipeline, [buf_in, buf_out, buf_n], n, block_size=256)

    result = runner.read_float_buffer(buf_out, n)
    for i in range(n):
        expected = 1.0 / math.sqrt(input_data[i])
        tol = max(1e-4, abs(expected) * 1e-4)
        assert abs(result[i] - expected) < tol, (
            f"[{i}] got {result[i]}, expected {expected}"
        )


# ---------------------------------------------------------------------------
# Layer norm pattern detection tests
# ---------------------------------------------------------------------------

def test_ttgir_layer_norm_detected():
    """Parser detects layer norm pattern (two sum reductions with sub)."""
    from triton_metal.codegen.ttgir_parser import TTGIRParser

    parser = TTGIRParser(LAYER_NORM_TTGIR, FakeOptions())
    parser._parse_function_signature()
    parser._parse_body()

    # Should have 2 reduce ops, both sum
    assert len(parser.reduce_ops) >= 2
    ops = [r[2] for r in parser.reduce_ops]
    assert ops.count("sum") >= 2, f"Expected 2 sum reduces, got {ops}"


def test_ttgir_layer_norm_is_not_softmax():
    """Layer norm should not be detected as softmax."""
    from triton_metal.codegen.ttgir_parser import TTGIRParser

    parser = TTGIRParser(LAYER_NORM_TTGIR, FakeOptions())
    parser._parse_function_signature()
    parser._parse_body()

    assert not parser._is_softmax_pattern(), "Layer norm should not match softmax"
    assert parser._is_layer_norm_pattern(), "Should match layer norm pattern"


@requires_metal
def test_ttgir_layer_norm_compiles(runner):
    """TTGIR layer norm pattern compiles to valid MSL."""
    from triton_metal.codegen.ttgir_parser import parse_ttgir

    kb = parse_ttgir(LAYER_NORM_TTGIR, FakeOptions())
    msl = kb.build()
    assert "mean" in msl.lower() or "inv_std" in msl or "var" in msl.lower()
    runner.compile(msl, "layer_norm_kernel")


# ---------------------------------------------------------------------------
# Matmul (tt.dot) TTGIR
# ---------------------------------------------------------------------------

MATMUL_TTGIR = """
module {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                 %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                 %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                 %arg3: i32, %arg4: i32, %arg5: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked1>
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked2>
    %3 = arith.muli %0, %c32_i32 : i32
    %4 = tt.splat %3 : i32 -> tensor<32xi32, #blocked1>
    %5 = arith.addi %4, %1 : tensor<32xi32, #blocked1>
    %6 = tt.addptr %arg0, %5 : !tt.ptr<f32>, tensor<32xi32, #blocked1>
    %7 = tt.load %6 : !tt.ptr<f32>
    %8 = tt.addptr %arg1, %2 : !tt.ptr<f32>, tensor<32xi32, #blocked2>
    %9 = tt.load %8 : !tt.ptr<f32>
    %10 = "tt.dot"(%7, %9, %cst) {allowTF32 = true} : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %11 = tt.addptr %arg2, %5 : !tt.ptr<f32>, tensor<32xi32, #blocked1>
    tt.store %11, %10 : !tt.ptr<f32>
    tt.return
  }
}
"""


def test_ttgir_matmul_detected():
    """tt.dot operation should be detected as matmul pattern."""
    from triton_metal.codegen.ttgir_parser import TTGIRParser

    parser = TTGIRParser(MATMUL_TTGIR, FakeOptions())
    parser._parse_function_signature()
    parser._parse_body()

    assert len(parser.dot_ops) >= 1, "Should detect tt.dot operation"
    result_ssa, lhs, rhs, acc = parser.dot_ops[0]
    assert result_ssa.startswith("%")


def test_ttgir_matmul_not_reduction():
    """Matmul should not be detected as reduction or softmax."""
    from triton_metal.codegen.ttgir_parser import TTGIRParser

    parser = TTGIRParser(MATMUL_TTGIR, FakeOptions())
    parser._parse_function_signature()
    parser._parse_body()

    assert not parser._is_softmax_pattern(), "Matmul should not match softmax"
    assert len(parser.dot_ops) >= 1, "Should have dot ops"


@requires_metal
def test_ttgir_matmul_compiles(runner):
    """TTGIR matmul pattern compiles to valid MSL (simdgroup_matmul)."""
    from triton_metal.codegen.ttgir_parser import parse_ttgir

    kb = parse_ttgir(MATMUL_TTGIR, FakeOptions())
    msl = kb.build()
    # Should produce simdgroup_matrix-based MSL
    assert "simdgroup" in msl or "matmul" in msl.lower()
    runner.compile(msl, "simdgroup_matmul")


@requires_metal
def test_ttgir_matmul_gpu(runner):
    """TTGIR-parsed matmul produces correct results on GPU."""
    from triton_metal.codegen.ttgir_parser import parse_ttgir
    import random

    kb = parse_ttgir(MATMUL_TTGIR, FakeOptions())
    msl = kb.build()
    path = runner.compile(msl, "simdgroup_matmul")
    pipeline = runner.load(path, "simdgroup_matmul")

    M, N, K = 32, 32, 32
    random.seed(4444)
    a_data = [random.uniform(-1.0, 1.0) for _ in range(M * K)]
    b_data = [random.uniform(-1.0, 1.0) for _ in range(K * N)]

    a_buf = runner.make_float_buffer(a_data)
    b_buf = runner.make_float_buffer(b_data)
    c_buf = runner.make_empty_buffer(M * N)
    m_buf = runner.make_uint_buffer(M)
    n_buf = runner.make_uint_buffer(N)
    k_buf = runner.make_uint_buffer(K)

    import Metal
    cmd = runner.queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(pipeline)
    for i, buf in enumerate([a_buf, b_buf, c_buf, m_buf, n_buf, k_buf]):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1),  # 1 tile for 32x32
        Metal.MTLSizeMake(128, 1, 1),
    )
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == 4, f"Kernel failed, status={cmd.status()}"

    result = runner.read_float_buffer(c_buf, M * N)

    # Reference matmul
    for i in range(M):
        for j in range(N):
            expected = sum(a_data[i * K + k] * b_data[k * N + j] for k in range(K))
            got = result[i * N + j]
            assert abs(got - expected) < 0.1, (
                f"C[{i},{j}]: got {got}, expected {expected}"
            )


# ---------------------------------------------------------------------------
# Flash Attention TTGIR
# ---------------------------------------------------------------------------

FLASH_ATTENTION_TTGIR = """
module {
  tt.func public @fused_attention(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                   %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                   %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                   %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                   %arg4: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x64xf32, #blocked>
    %cst_scale = arith.constant 1.250000e-01 : f32
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #blocked1>
    %2 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #blocked2>

    // Load Q block
    %3 = tt.addptr %arg0, %1 : !tt.ptr<f32>, tensor<16xi32, #blocked1>
    %q = tt.load %3 : !tt.ptr<f32>

    // Load K block
    %4 = tt.addptr %arg1, %2 : !tt.ptr<f32>, tensor<16xi32, #blocked2>
    %k = tt.load %4 : !tt.ptr<f32>

    // Score = Q @ K^T (first tt.dot)
    %scores = "tt.dot"(%q, %k, %cst) {allowTF32 = true} : (tensor<16x64xf32>, tensor<64x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>

    // Max for numerical stability
    %row_max = "tt.reduce"(%scores) ({
    ^bb0(%a: f32, %b: f32):
      %mx = arith.maxf %a, %b : f32
      "tt.reduce.return"(%mx) : (f32) -> ()
    }) {axis = 1 : i32} : (tensor<16x16xf32>) -> tensor<16xf32>

    // Softmax: exp(score - max)
    %max_splat = tt.splat %row_max : tensor<16xf32> -> tensor<16x16xf32, #blocked>
    %shifted = arith.subf %scores, %max_splat : tensor<16x16xf32>
    %p = math.exp %shifted : tensor<16x16xf32>

    // Sum for normalization
    %row_sum = "tt.reduce"(%p) ({
    ^bb0(%c: f32, %d: f32):
      %sm = arith.addf %c, %d : f32
      "tt.reduce.return"(%sm) : (f32) -> ()
    }) {axis = 1 : i32} : (tensor<16x16xf32>) -> tensor<16xf32>

    // Load V block
    %5 = tt.addptr %arg2, %2 : !tt.ptr<f32>, tensor<16xi32, #blocked2>
    %v = tt.load %5 : !tt.ptr<f32>

    // Output = P @ V (second tt.dot)
    %out = "tt.dot"(%p, %v, %cst) {allowTF32 = true} : (tensor<16x16xf32>, tensor<16x64xf32>, tensor<16x64xf32>) -> tensor<16x64xf32>

    // Store output
    %6 = tt.addptr %arg3, %1 : !tt.ptr<f32>, tensor<16xi32, #blocked1>
    tt.store %6, %out : !tt.ptr<f32>
    tt.return
  }
}
"""


def test_ttgir_flash_attention_detected():
    """Flash attention: 2 dot ops + exp + max reduction detected."""
    from triton_metal.codegen.ttgir_parser import TTGIRParser

    parser = TTGIRParser(FLASH_ATTENTION_TTGIR, FakeOptions())
    parser._parse_function_signature()
    parser._parse_body()

    assert len(parser.dot_ops) >= 2, f"Expected 2+ dot ops, got {len(parser.dot_ops)}"
    assert parser._is_flash_attention_pattern(), "Should match flash attention pattern"


def test_ttgir_flash_attention_not_matmul():
    """Flash attention should not be treated as simple matmul."""
    from triton_metal.codegen.ttgir_parser import TTGIRParser

    parser = TTGIRParser(FLASH_ATTENTION_TTGIR, FakeOptions())
    parser._parse_function_signature()
    parser._parse_body()

    # Has 2 dots AND exp+max → flash attention, not simple matmul
    assert parser._is_flash_attention_pattern()
    assert len(parser.dot_ops) >= 2


@requires_metal
def test_ttgir_flash_attention_compiles(runner):
    """Flash attention TTGIR compiles to valid MSL."""
    from triton_metal.codegen.ttgir_parser import parse_ttgir

    kb = parse_ttgir(FLASH_ATTENTION_TTGIR, FakeOptions())
    msl = kb.build()
    assert "flash_attention" in msl or "attention" in msl.lower()
    runner.compile(msl, "flash_attention")
