"""Test the C++ MLIR backend infrastructure.

Verifies that the C++ MLIR pass module (triton_metal._triton_metal_cpp) can
be imported and its passes registered alongside Triton's libtriton.so in the
same process. The pybind11 module links against libtriton.so for shared MLIR
symbols, eliminating the previous duplicate-dialect-registration crash.
"""
import pytest

try:
    import triton_metal._triton_metal_cpp as cpp
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False

try:
    import Metal
    _HAS_METAL = Metal.MTLCreateSystemDefaultDevice() is not None
except ImportError:
    _HAS_METAL = False

requires_cpp = pytest.mark.skipif(not _HAS_CPP, reason="C++ backend not built")
requires_metal = pytest.mark.skipif(not _HAS_METAL, reason="Metal not available")


@requires_cpp
def test_cpp_module_importable():
    """C++ MLIR module can be imported."""
    import triton_metal._triton_metal_cpp as mod
    assert hasattr(mod, "register_metal_passes")


@requires_cpp
def test_cpp_passes_register():
    """C++ MLIR passes can be registered without error."""
    cpp.register_metal_passes()


@requires_cpp
def test_cpp_passes_register_idempotent():
    """Calling register_metal_passes() multiple times is safe."""
    cpp.register_metal_passes()
    cpp.register_metal_passes()
    cpp.register_metal_passes()


@requires_cpp
@requires_metal
def test_cpp_pass_runs_on_vector_add():
    """The C++ pass infrastructure works alongside a vector_add compilation.

    Both the C++ module and Triton's compilation pipeline run in the same
    process — no subprocess isolation needed. The pybind11 module links
    against libtriton.so so both use the same MLIR symbols.
    """
    import triton
    import triton.language as tl
    from triton.compiler.compiler import compile as triton_compile, ASTSource
    from triton.backends.compiler import GPUTarget

    # Register our C++ passes in the same process
    cpp.register_metal_passes()

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    target = GPUTarget("metal", "apple-m4", 32)
    sig = {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}
    src = ASTSource(fn=add_kernel, signature=sig, constexprs={"BLOCK": 256})
    compiled = triton_compile(src, target=target)

    assert compiled.asm.get("ttgir") is not None, "TTGIR missing"
    assert len(str(compiled.asm["ttgir"])) > 0, "TTGIR empty"
    assert compiled.asm.get("msl") is not None, "MSL missing"
    assert compiled.asm.get("metallib") is not None, "metallib missing"


@requires_cpp
@requires_metal
def test_vector_add_execution():
    """Vector add produces correct results via the Python compilation path.

    Both the C++ module and Triton kernel execution run in the same process.
    """
    import torch
    import triton
    import triton.language as tl

    # Register our C++ passes in the same process
    cpp.register_metal_passes()

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    n = 1024
    x = torch.randn(n)
    y = torch.randn(n)
    out = torch.zeros(n)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    add_kernel[grid](x, y, out, n, BLOCK=256)

    max_err = (out - (x + y)).abs().max().item()
    assert max_err < 1e-5, f"max error {max_err} exceeds tolerance"


@requires_cpp
@requires_metal
def test_scf_for_accumulation():
    """Accumulation loop compiles and executes through C++ metallib.

    The kernel sums K chunks of an input vector using an explicit loop,
    which Triton lowers to scf.for + scf.yield. The C++ path handles
    this via SCFToControlFlowPass -> cf.br/cf.cond_br -> LLVM branches.
    """
    import os
    import torch
    import triton
    import triton.language as tl

    os.environ["TRITON_METAL_USE_CPP"] = "1"
    try:
        @triton.jit
        def accum_kernel(x_ptr, out_ptr, K: tl.constexpr, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            acc = tl.zeros([BLOCK], dtype=tl.float32)
            for k in range(K):
                val = tl.load(x_ptr + offs * K + k)
                acc += val
            tl.store(out_ptr + offs, acc)

        BLOCK = 256
        K = 4
        n = BLOCK
        x = torch.randn(n * K)
        out = torch.zeros(n)

        accum_kernel[(1,)](x, out, K=K, BLOCK=BLOCK)

        expected = x.view(n, K).sum(dim=1)
        max_err = (out - expected).abs().max().item()
        assert max_err < 1e-4, f"scf.for accumulation: max error {max_err}"
    finally:
        os.environ.pop("TRITON_METAL_USE_CPP", None)


@requires_cpp
@requires_metal
def test_scf_if_conditional():
    """Conditional clamp with float scalar args through C++ metallib.

    Uses tl.where (arith.cmpf + arith.select) and float scalar parameters
    (lo, hi) which are passed as constant buffer pointers in AIR.
    """
    import os
    import torch
    import triton
    import triton.language as tl

    os.environ["TRITON_METAL_USE_CPP"] = "1"
    try:
        @triton.jit
        def clamp_kernel(x_ptr, out_ptr, lo, hi, n, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask)
            x = tl.where(x < lo, lo, x)
            x = tl.where(x > hi, hi, x)
            tl.store(out_ptr + offs, x, mask=mask)

        n = 512
        x = torch.randn(n) * 5
        out = torch.zeros(n)

        clamp_kernel[(triton.cdiv(n, 256),)](x, out, -1.0, 1.0, n, BLOCK=256)

        expected = x.clamp(-1.0, 1.0)
        max_err = (out - expected).abs().max().item()
        assert max_err < 1e-5, f"clamp: max error {max_err}"
    finally:
        os.environ.pop("TRITON_METAL_USE_CPP", None)
