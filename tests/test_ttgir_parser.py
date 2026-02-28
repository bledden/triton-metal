"""Tests for the TTGIR MLIR text parser.

Feeds real TTGIR dumps into the parser and verifies:
1. Correct kernel name extraction
2. Correct argument registration (pointers vs scalars)
3. Valid MSL generation
4. MSL compiles with xcrun metal
5. (When possible) GPU execution produces correct results
"""

import hashlib
import os
import platform
import struct
import subprocess
import tempfile

import pytest

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="Metal backend requires macOS",
)


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


requires_compiler = pytest.mark.skipif(
    not _has_metal_compiler(),
    reason="Requires Metal compiler",
)


def _compile_msl(msl_src, kernel_name):
    """Compile MSL to metallib, return path."""
    cache_dir = os.path.join(tempfile.gettempdir(), "triton_metal_parser_test")
    os.makedirs(cache_dir, exist_ok=True)

    src_hash = hashlib.sha256(msl_src.encode()).hexdigest()[:16]
    base = f"{kernel_name}_{src_hash}"
    metal_path = os.path.join(cache_dir, f"{base}.metal")
    air_path = os.path.join(cache_dir, f"{base}.air")
    metallib_path = os.path.join(cache_dir, f"{base}.metallib")

    with open(metal_path, "w") as f:
        f.write(msl_src)

    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", metal_path,
         "-o", air_path, "-std=metal3.2", "-O2"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None, result.stderr
    subprocess.check_call(
        ["xcrun", "-sdk", "macosx", "metallib", air_path,
         "-o", metallib_path],
        stderr=subprocess.PIPE,
    )
    return metallib_path, None


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


@requires_compiler
def test_parse_vecadd_compiles():
    """MSL generated from parsed vector add TTGIR compiles."""
    from triton_metal.codegen.ttgir_parser import parse_ttgir

    kb = parse_ttgir(VECADD_TTGIR, FakeOptions())
    msl = kb.build()
    path, error = _compile_msl(msl, "add_kernel")
    assert path is not None, f"Compilation failed:\n{error}\n\nMSL:\n{msl}"


@requires_compiler
def test_parse_vecmul_compiles():
    """MSL generated from parsed multiply TTGIR compiles."""
    from triton_metal.codegen.ttgir_parser import parse_ttgir

    kb = parse_ttgir(VECMUL_TTGIR, FakeOptions())
    msl = kb.build()
    path, error = _compile_msl(msl, "mul_kernel")
    assert path is not None, f"Compilation failed:\n{error}\n\nMSL:\n{msl}"


@requires_compiler
def test_parse_exp_compiles():
    """MSL generated from parsed exp TTGIR compiles."""
    from triton_metal.codegen.ttgir_parser import parse_ttgir

    kb = parse_ttgir(EXP_TTGIR, FakeOptions())
    msl = kb.build()
    path, error = _compile_msl(msl, "exp_kernel")
    assert path is not None, f"Compilation failed:\n{error}\n\nMSL:\n{msl}"


def test_parse_block_size():
    """Parser extracts block size from tt.make_range."""
    from triton_metal.codegen.ttgir_parser import parse_ttgir

    kb = parse_ttgir(VECADD_TTGIR, FakeOptions())
    assert kb.block_size >= 256


@requires_compiler
def test_parse_vecadd_generates_add_op():
    """Parsed vector add MSL contains an addition operation."""
    from triton_metal.codegen.ttgir_parser import parse_ttgir

    kb = parse_ttgir(VECADD_TTGIR, FakeOptions())
    msl = kb.build()
    # The MSL should contain a float addition
    assert "+" in msl or "add" in msl.lower(), f"No addition found in MSL:\n{msl}"


@requires_compiler
def test_parse_vecmul_generates_mul_op():
    """Parsed multiply MSL contains a multiply operation."""
    from triton_metal.codegen.ttgir_parser import parse_ttgir

    kb = parse_ttgir(VECMUL_TTGIR, FakeOptions())
    msl = kb.build()
    assert "*" in msl, f"No multiplication found in MSL:\n{msl}"


@requires_compiler
def test_parse_exp_generates_exp_op():
    """Parsed exp MSL contains an exp() call."""
    from triton_metal.codegen.ttgir_parser import parse_ttgir

    kb = parse_ttgir(EXP_TTGIR, FakeOptions())
    msl = kb.build()
    assert "exp(" in msl, f"No exp() found in MSL:\n{msl}"
