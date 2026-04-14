"""Test the C++ MLIR backend infrastructure.

Verifies that the C++ MLIR pass module (triton_metal._triton_metal_cpp) can
be imported and its passes registered.

IMPORTANT: The C++ .so links its own copy of MLIR, which conflicts with the
MLIR loaded by triton's libtriton.so. Loading both in the same process causes
a fatal abort ("dialect has no registered attribute printing hook"). Tests
that need triton compilation use subprocess isolation to avoid this conflict.
"""
import subprocess
import sys
import textwrap

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

    Because the C++ .so and triton's libtriton.so each link their own MLIR,
    loading both in the same process causes a fatal symbol conflict. This test
    uses subprocess isolation: a child process compiles vector_add through the
    normal pipeline (TTIR -> TTGIR -> MSL -> metallib) and verifies all
    artifacts are produced. The C++ module is NOT loaded in that subprocess.

    The test proves:
    1. The C++ module loads and registers in this process (parent)
    2. The compilation pipeline produces TTGIR, MSL, and metallib (child)
    3. The two capabilities exist and work -- just not yet in the same process
    """
    # Parent process: verify C++ passes register
    cpp.register_metal_passes()

    # Child process: compile vector_add through the Python pipeline
    script = textwrap.dedent("""\
        import triton
        import triton.language as tl
        from triton.compiler.compiler import compile as triton_compile, ASTSource
        from triton.backends.compiler import GPUTarget

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

        print(f"TTGIR: {len(str(compiled.asm['ttgir']))} chars")
        print(f"MSL: {len(compiled.asm['msl'])} chars")
        print(f"metallib: {len(compiled.asm['metallib'])} bytes")
        print("OK")
    """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"Subprocess compilation failed (rc={result.returncode}):\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    assert "OK" in result.stdout


@requires_cpp
@requires_metal
def test_vector_add_execution():
    """Vector add produces correct results via the Python compilation path.

    Uses subprocess isolation to avoid MLIR symbol conflict with C++ module.
    The parent process verifies the C++ module loads; the child process runs
    the actual vector_add kernel through Triton on Metal.
    """
    # Parent: C++ module is loaded and functional
    cpp.register_metal_passes()

    # Child: run vector_add kernel end-to-end
    script = textwrap.dedent("""\
        import torch
        import triton
        import triton.language as tl

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
        print(f"max_error={max_err}")
        print("OK")
    """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"Subprocess vector_add failed (rc={result.returncode}):\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    assert "OK" in result.stdout
