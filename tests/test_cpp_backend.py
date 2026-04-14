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
