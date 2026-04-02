"""Test the generic op-by-op lowerer against real TTGIR modules.

End-to-end tests: Triton kernel → TTGIR → MLIR walker → generic lowerer → MSL.
Validates that the generated MSL compiles with xcrun metal.
"""

import os
import subprocess
import tempfile
import pytest

try:
    import triton
    import triton.language as tl
    from triton._C.libtriton import ir
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

try:
    import Metal
    HAS_METAL = Metal.MTLCreateSystemDefaultDevice() is not None
except ImportError:
    HAS_METAL = False

requires_triton = pytest.mark.skipif(not HAS_TRITON, reason="Triton not installed")
requires_metal = pytest.mark.skipif(not HAS_METAL, reason="Metal not available")


def _compile_to_ttgir(kernel_fn, sig, constexprs=None):
    """Compile a @triton.jit kernel through TTIR and TTGIR stages."""
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from triton_metal.backend.compiler import MetalBackend

    target = GPUTarget("metal", "apple-m4", 32)
    backend = MetalBackend(target)
    options = backend.parse_options({})

    src = ASTSource(fn=kernel_fn, signature=sig, constexprs=constexprs or {})
    context = ir.context()
    ir.load_dialects(context)
    codegen_fns = backend.get_codegen_implementation(options)
    module_map = backend.get_module_map()
    mod = src.make_ir(target, options, codegen_fns, module_map, context)

    metadata = {}
    mod = backend.make_ttir(mod, metadata, options)
    mod = backend.make_ttgir(mod, metadata, options)

    return mod, metadata, options


def _lower_to_msl(mod, metadata, options):
    """Walk TTGIR module and lower to MSL."""
    from triton_metal.codegen.mlir_walker import walk_ttgir
    from triton_metal.codegen.generic_lowerer import lower_ir_graph

    graph = walk_ttgir(mod, options)
    msl = lower_ir_graph(graph, options)
    return msl, graph


def _validate_msl_compiles(msl_src: str):
    """Verify that MSL source compiles with xcrun metal."""
    with tempfile.NamedTemporaryFile(suffix=".metal", mode="w", delete=False) as f:
        f.write(msl_src)
        metal_path = f.name

    air_path = metal_path.replace(".metal", ".air")
    try:
        result = subprocess.run(
            ["xcrun", "-sdk", "macosx", "metal", "-c", metal_path,
             "-o", air_path, "-std=metal3.2", "-O0"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            print(f"MSL compilation FAILED:\n{result.stderr}")
            print(f"\nGenerated MSL:\n{msl_src}")
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pytest.skip("xcrun metal not available")
    finally:
        for p in (metal_path, air_path):
            if os.path.exists(p):
                os.unlink(p)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@requires_triton
@requires_metal
def test_lower_vector_add():
    """Generic lowerer produces valid MSL for vector_add."""
    @triton.jit
    def vector_add(a_ptr, b_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        a = tl.load(a_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, a + b, mask=mask)

    sig = {"a_ptr": "*fp32", "b_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}
    mod, metadata, options = _compile_to_ttgir(
        vector_add, sig, constexprs={"BLOCK_SIZE": 256}
    )

    msl, graph = _lower_to_msl(mod, metadata, options)

    print(f"\n=== Generated MSL for vector_add ===")
    print(msl)

    # Basic structure checks
    assert "kernel void" in msl
    assert "vector_add" in msl
    assert "a_ptr" in msl
    assert "b_ptr" in msl
    assert "out_ptr" in msl

    # Should have loads, an add, and a store
    assert "+" in msl  # arith.addf

    # Should compile
    assert _validate_msl_compiles(msl), "MSL failed to compile"


@requires_triton
@requires_metal
def test_lower_scalar_mul():
    """Generic lowerer produces valid MSL for scalar multiply."""
    @triton.jit
    def scalar_mul(x_ptr, out_ptr, n, scale, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x * scale, mask=mask)

    sig = {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32", "scale": "fp32"}
    mod, metadata, options = _compile_to_ttgir(
        scalar_mul, sig, constexprs={"BLOCK_SIZE": 256}
    )

    msl, graph = _lower_to_msl(mod, metadata, options)

    print(f"\n=== Generated MSL for scalar_mul ===")
    print(msl)

    assert "kernel void" in msl
    assert "scale" in msl
    assert "*" in msl  # multiplication
    assert _validate_msl_compiles(msl), "MSL failed to compile"


@requires_triton
@requires_metal
def test_lower_fp16_cast():
    """Generic lowerer handles FP16 loads/stores."""
    @triton.jit
    def cast_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
        y = tl.load(y_ptr + offsets, mask=mask).to(tl.float32)
        result = (x * y + x) * 0.5
        tl.store(out_ptr + offsets, result.to(tl.float16), mask=mask)

    sig = {"x_ptr": "*fp16", "y_ptr": "*fp16", "out_ptr": "*fp16", "n": "i32"}
    mod, metadata, options = _compile_to_ttgir(
        cast_kernel, sig, constexprs={"BLOCK_SIZE": 256}
    )

    msl, graph = _lower_to_msl(mod, metadata, options)

    print(f"\n=== Generated MSL for fp16_cast ===")
    print(msl)

    assert "half" in msl  # FP16 storage type
    assert "static_cast" in msl  # Type casts
    assert _validate_msl_compiles(msl), "MSL failed to compile"


@requires_triton
@requires_metal
def test_lower_constant_mul():
    """Generic lowerer handles float constants."""
    @triton.jit
    def const_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask)
        result = x * 0.5
        tl.store(out_ptr + offsets, result, mask=mask)

    sig = {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}
    mod, metadata, options = _compile_to_ttgir(
        const_kernel, sig, constexprs={"BLOCK_SIZE": 256}
    )

    msl, graph = _lower_to_msl(mod, metadata, options)

    print(f"\n=== Generated MSL for constant_mul ===")
    print(msl)

    assert "0.5" in msl  # The constant
    assert _validate_msl_compiles(msl), "MSL failed to compile"


@requires_triton
@requires_metal
def test_lower_sum_reduction():
    """Generic lowerer handles tt.reduce (sum)."""
    @triton.jit
    def sum_kernel(input_ptr, output_ptr, n_elements,
                   BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        result = tl.sum(x, axis=0)
        tl.store(output_ptr + pid, result)

    sig = {"input_ptr": "*fp32", "output_ptr": "*fp32", "n_elements": "i32"}
    mod, metadata, options = _compile_to_ttgir(
        sum_kernel, sig, constexprs={"BLOCK_SIZE": 256}
    )

    msl, graph = _lower_to_msl(mod, metadata, options)

    print(f"\n=== Generated MSL for sum_reduction ===")
    print(msl)

    assert "kernel void" in msl
    assert "simd_sum" in msl or "simd" in msl  # SIMD reduction
    assert "threadgroup" in msl  # Shared memory
    assert _validate_msl_compiles(msl), "MSL failed to compile"


# ---------------------------------------------------------------------------
# Adversarial tests — novel op combinations the pattern matchers can't handle
# ---------------------------------------------------------------------------

@requires_triton
@requires_metal
def test_lower_negation():
    """Generic lowerer handles arith.negf."""
    @triton.jit
    def negate_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, -x, mask=mask)

    sig = {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}
    mod, metadata, options = _compile_to_ttgir(
        negate_kernel, sig, constexprs={"BLOCK_SIZE": 256}
    )

    msl, graph = _lower_to_msl(mod, metadata, options)
    print(f"\n=== Generated MSL for negate ===")
    print(msl)

    assert "kernel void" in msl
    assert _validate_msl_compiles(msl), "MSL failed to compile"


@requires_triton
@requires_metal
def test_lower_math_exp():
    """Generic lowerer handles math.exp (exponential)."""
    @triton.jit
    def exp_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, tl.exp(x), mask=mask)

    sig = {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}
    mod, metadata, options = _compile_to_ttgir(
        exp_kernel, sig, constexprs={"BLOCK_SIZE": 256}
    )

    msl, graph = _lower_to_msl(mod, metadata, options)
    print(f"\n=== Generated MSL for exp ===")
    print(msl)

    assert "exp(" in msl
    assert _validate_msl_compiles(msl), "MSL failed to compile"


@requires_triton
@requires_metal
def test_lower_fused_silu():
    """Adversarial: SiLU (x * sigmoid(x)) — not a named pattern."""
    @triton.jit
    def silu_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask)
        # SiLU = x * sigmoid(x) = x / (1 + exp(-x))
        result = x * tl.sigmoid(x)
        tl.store(out_ptr + offsets, result, mask=mask)

    sig = {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}
    mod, metadata, options = _compile_to_ttgir(
        silu_kernel, sig, constexprs={"BLOCK_SIZE": 256}
    )

    msl, graph = _lower_to_msl(mod, metadata, options)
    print(f"\n=== Generated MSL for fused_silu ===")
    print(msl)

    assert "kernel void" in msl
    assert _validate_msl_compiles(msl), "MSL failed to compile"


@requires_triton
@requires_metal
def test_lower_multi_op_chain():
    """Adversarial: long chain of mixed ops — no single pattern covers this."""
    @triton.jit
    def chain_kernel(a_ptr, b_ptr, out_ptr, n, alpha,
                     BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        a = tl.load(a_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + offsets, mask=mask)
        # Complex expression: (a * alpha + b) * (a - b)
        result = (a * alpha + b) * (a - b)
        tl.store(out_ptr + offsets, result, mask=mask)

    sig = {"a_ptr": "*fp32", "b_ptr": "*fp32", "out_ptr": "*fp32",
           "n": "i32", "alpha": "fp32"}
    mod, metadata, options = _compile_to_ttgir(
        chain_kernel, sig, constexprs={"BLOCK_SIZE": 256}
    )

    msl, graph = _lower_to_msl(mod, metadata, options)
    print(f"\n=== Generated MSL for multi_op_chain ===")
    print(msl)

    assert "kernel void" in msl
    assert "alpha" in msl
    assert _validate_msl_compiles(msl), "MSL failed to compile"


@requires_triton
@requires_metal
def test_lower_reduce_mul_add():
    """Adversarial: reduce(x * y + z) — reduction of a fused expression."""
    @triton.jit
    def reduce_expr_kernel(x_ptr, y_ptr, z_ptr, out_ptr, n,
                           BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        z = tl.load(z_ptr + offsets, mask=mask, other=0.0)
        expr = x * y + z
        result = tl.sum(expr, axis=0)
        tl.store(out_ptr + pid, result)

    sig = {"x_ptr": "*fp32", "y_ptr": "*fp32", "z_ptr": "*fp32",
           "out_ptr": "*fp32", "n": "i32"}
    mod, metadata, options = _compile_to_ttgir(
        reduce_expr_kernel, sig, constexprs={"BLOCK_SIZE": 256}
    )

    msl, graph = _lower_to_msl(mod, metadata, options)
    print(f"\n=== Generated MSL for reduce_mul_add ===")
    print(msl)

    assert "kernel void" in msl
    assert "simd" in msl  # SIMD reduction
    assert _validate_msl_compiles(msl), "MSL failed to compile"


@requires_triton
@requires_metal
def test_lower_max_reduction():
    """Generic lowerer handles tt.reduce with maxf combine."""
    @triton.jit
    def max_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask, other=float('-inf'))
        result = tl.max(x, axis=0)
        tl.store(out_ptr + pid, result)

    sig = {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}
    mod, metadata, options = _compile_to_ttgir(
        max_kernel, sig, constexprs={"BLOCK_SIZE": 256}
    )

    msl, graph = _lower_to_msl(mod, metadata, options)
    print(f"\n=== Generated MSL for max_reduction ===")
    print(msl)

    assert "kernel void" in msl
    assert "simd" in msl
    assert _validate_msl_compiles(msl), "MSL failed to compile"


@requires_triton
@requires_metal
def test_lower_where_clamp():
    """Adversarial: tl.where (arith.select) for clamping."""
    @triton.jit
    def clamp_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask)
        # Clamp to [0, 1] using tl.where
        clamped = tl.where(x > 1.0, 1.0, x)
        clamped = tl.where(clamped < 0.0, 0.0, clamped)
        tl.store(out_ptr + offsets, clamped, mask=mask)

    sig = {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}
    mod, metadata, options = _compile_to_ttgir(
        clamp_kernel, sig, constexprs={"BLOCK_SIZE": 256}
    )

    msl, graph = _lower_to_msl(mod, metadata, options)
    print(f"\n=== Generated MSL for where_clamp ===")
    print(msl)

    assert "kernel void" in msl
    assert "?" in msl  # Ternary from select
    assert _validate_msl_compiles(msl), "MSL failed to compile"


@requires_triton
@requires_metal
def test_lower_int_to_float():
    """Generic lowerer handles sitofp (integer to float conversion)."""
    @triton.jit
    def itof_kernel(idx_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        idx = tl.load(idx_ptr + offsets, mask=mask)
        # Convert int to float and scale
        result = idx.to(tl.float32) * 0.1
        tl.store(out_ptr + offsets, result, mask=mask)

    sig = {"idx_ptr": "*i32", "out_ptr": "*fp32", "n": "i32"}
    mod, metadata, options = _compile_to_ttgir(
        itof_kernel, sig, constexprs={"BLOCK_SIZE": 256}
    )

    msl, graph = _lower_to_msl(mod, metadata, options)
    print(f"\n=== Generated MSL for int_to_float ===")
    print(msl)

    assert "kernel void" in msl
    assert "static_cast" in msl or "float" in msl
    assert _validate_msl_compiles(msl), "MSL failed to compile"


# ---------------------------------------------------------------------------
# Pipeline integration tests — verify emit_msl uses new codegen
# ---------------------------------------------------------------------------

@requires_triton
@requires_metal
def test_emit_msl_uses_new_codegen_for_elementwise():
    """Verify the full emit_msl pipeline uses the new walker+lowerer for elementwise."""
    from triton_metal.codegen.msl_emitter import emit_msl

    @triton.jit
    def add_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        a = tl.load(a_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, a + b, mask=mask)

    sig = {"a_ptr": "*fp32", "b_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}
    mod, metadata, options = _compile_to_ttgir(
        add_kernel, sig, constexprs={"BLOCK_SIZE": 256}
    )

    # Call through the full pipeline — same path as compiler.make_msl
    msl = emit_msl(mod, metadata, options)

    print(f"\n=== emit_msl pipeline output for add_kernel ===")
    print(msl)

    assert "kernel void" in msl
    assert "add_kernel" in msl
    assert metadata.get("name") == "add_kernel"
    assert metadata.get("block_size") == 256
    assert _validate_msl_compiles(msl), "MSL failed to compile"


@requires_triton
@requires_metal
def test_emit_msl_handles_matmul_via_new_pipeline():
    """Verify emit_msl routes matmul (tt.dot) through the new pipeline."""
    from triton_metal.codegen.msl_emitter import emit_msl

    @triton.jit
    def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        c_ptrs = c_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
        tl.store(c_ptrs, acc)

    sig = {
        "a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32",
        "M": "i32", "N": "i32", "K": "i32",
        "stride_am": "i32", "stride_ak": "i32",
        "stride_bk": "i32", "stride_bn": "i32",
        "stride_cm": "i32", "stride_cn": "i32",
    }
    mod, metadata, options = _compile_to_ttgir(
        matmul_kernel, sig,
        constexprs={"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}
    )

    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        msl = emit_msl(mod, metadata, options)
        # Should NOT trigger a legacy fallback deprecation warning
        legacy_warnings = [x for x in w if "legacy" in str(x.message).lower()]
        assert len(legacy_warnings) == 0, \
            f"Matmul should use new pipeline, got legacy warning: {legacy_warnings}"

    print(f"\n=== emit_msl pipeline output for matmul_kernel ===")
    print(f"(first 500 chars): {msl[:500]}")

    assert "kernel void" in msl
    assert "UNSUPPORTED" not in msl
    assert metadata.get("name") is not None
    assert _validate_msl_compiles(msl), "MSL failed to compile"


@requires_triton
@requires_metal
def test_no_legacy_fallback_for_standard_kernels():
    """No standard kernel should fall back to the legacy parser."""
    from triton_metal.codegen.msl_emitter import emit_msl
    import warnings

    kernels = {}

    @triton.jit
    def add_k(a_ptr, b_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        a = tl.load(a_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, a + b, mask=mask)

    kernels["add"] = (
        add_k,
        {"a_ptr": "*fp32", "b_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"},
        {"BLOCK_SIZE": 256},
    )

    @triton.jit
    def relu_k(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask)
        result = tl.where(x > 0, x, 0.0)
        tl.store(out_ptr + offsets, result, mask=mask)

    kernels["relu"] = (
        relu_k,
        {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"},
        {"BLOCK_SIZE": 256},
    )

    @triton.jit
    def sum_k(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        result = tl.sum(x, axis=0)
        tl.store(out_ptr + pid, result)

    kernels["sum"] = (
        sum_k,
        {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"},
        {"BLOCK_SIZE": 256},
    )

    for name, (fn, sig, constexprs) in kernels.items():
        mod, metadata, options = _compile_to_ttgir(fn, sig, constexprs=constexprs)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            msl = emit_msl(mod, metadata, options)
            legacy_warnings = [x for x in w if "legacy" in str(x.message).lower()]
            assert len(legacy_warnings) == 0, \
                f"Kernel '{name}' fell back to legacy parser!"
            assert "UNSUPPORTED" not in msl, \
                f"Kernel '{name}' has unsupported ops in output"
        print(f"  {name}: OK (new pipeline)")


# ---------------------------------------------------------------------------
# Adversarial end-to-end tests — novel combinations through full pipeline
# ---------------------------------------------------------------------------

@requires_triton
@requires_metal
def test_lower_softmax_fused():
    """Adversarial: row-wise softmax — max + sub + exp + sum + div."""
    @triton.jit
    def softmax_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask, other=float('-inf'))
        x_max = tl.max(x, axis=0)
        x_shifted = x - x_max
        numerator = tl.exp(x_shifted)
        denominator = tl.sum(numerator, axis=0)
        result = numerator / denominator
        tl.store(out_ptr + offsets, result, mask=mask)

    sig = {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}
    mod, metadata, options = _compile_to_ttgir(
        softmax_kernel, sig, constexprs={"BLOCK_SIZE": 256}
    )

    msl, graph = _lower_to_msl(mod, metadata, options)
    print(f"\n=== Generated MSL for softmax_fused ===")
    print(msl)

    assert "kernel void" in msl
    assert "exp(" in msl
    assert "simd" in msl  # Reductions
    assert "UNSUPPORTED" not in msl
    assert _validate_msl_compiles(msl), "MSL failed to compile"


@requires_triton
@requires_metal
def test_lower_gelu_sigmoid():
    """Adversarial: GELU sigmoid approximation — x * sigmoid(1.702 * x)."""
    @triton.jit
    def gelu_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask)
        # GELU sigmoid approx: x * sigmoid(1.702 * x)
        result = x * tl.sigmoid(1.702 * x)
        tl.store(out_ptr + offsets, result, mask=mask)

    sig = {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}
    mod, metadata, options = _compile_to_ttgir(
        gelu_kernel, sig, constexprs={"BLOCK_SIZE": 256}
    )

    msl, graph = _lower_to_msl(mod, metadata, options)
    print(f"\n=== Generated MSL for gelu_sigmoid ===")
    print(msl)

    assert "kernel void" in msl
    assert "UNSUPPORTED" not in msl
    assert _validate_msl_compiles(msl), "MSL failed to compile"
