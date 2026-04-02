"""Structured error types for triton-metal.

Provides clear, actionable error messages for common failure modes:
- Shader compilation failures
- Codegen lowering failures
- Pre-lowering validation failures
- Hardware-unsupported operations
- Not-yet-implemented operations
- Kernel dispatch/launch failures
"""


class MetalCompilationError(RuntimeError):
    """MSL shader compilation failed (xcrun metal returned an error)."""

    def __init__(self, message, msl_source=None, stderr=None):
        self.msl_source = msl_source
        self.stderr = stderr
        parts = [message]
        if stderr:
            parts.append(f"\nCompiler output:\n{stderr}")
        super().__init__("\n".join(parts))


class MetalCodegenError(RuntimeError):
    """TTGIR → MSL lowering failed for a specific operation."""

    def __init__(self, message, op_name=None, ssa_id=None, type_str=None):
        self.op_name = op_name
        self.ssa_id = ssa_id
        self.type_str = type_str
        parts = [message]
        if op_name:
            parts.append(f"  op: {op_name}")
        if ssa_id:
            parts.append(f"  ssa: {ssa_id}")
        if type_str:
            parts.append(f"  type: {type_str}")
        super().__init__("\n".join(parts))


class MetalUnsupportedError(MetalCodegenError):
    """Operation requires hardware features not available on Apple GPUs.

    Examples: FP64 arithmetic, FP8 types, TF32 tensor cores.
    """

    def __init__(self, message, op_name=None, ssa_id=None, type_str=None):
        full_msg = f"Hardware unsupported: {message}"
        super().__init__(full_msg, op_name=op_name, ssa_id=ssa_id, type_str=type_str)


class MetalValidationError(MetalCodegenError):
    """Pre-lowering validation failed.

    Raised when IR validation detects issues before codegen begins,
    e.g. unsupported tensor ranks, invalid block sizes, or type mismatches
    that can be caught statically.
    """

    def __init__(self, message, op_name=None, ssa_id=None, type_str=None, constraint=None):
        self.constraint = constraint
        full_msg = f"Validation failed: {message}"
        if constraint:
            full_msg += f"\n  constraint: {constraint}"
        super().__init__(full_msg, op_name=op_name, ssa_id=ssa_id, type_str=type_str)


class MetalNotImplementedError(MetalCodegenError):
    """Operation is not yet implemented in triton-metal but could be.

    This indicates a gap in the compiler, not a hardware limitation.
    """

    def __init__(self, message, op_name=None, ssa_id=None, type_str=None):
        full_msg = (
            f"Not yet implemented: {message}\n"
            f"  If you need this, please file an issue at "
            f"https://github.com/bledden/triton-metal/issues"
        )
        super().__init__(full_msg, op_name=op_name, ssa_id=ssa_id, type_str=type_str)


class MetalLaunchError(RuntimeError):
    """Kernel dispatch failed at launch time.

    Raised when a compiled kernel cannot be dispatched to the GPU,
    e.g. buffer allocation failures, invalid grid dimensions, or
    Metal command buffer errors.
    """

    def __init__(self, message, kernel_name=None, grid=None, reason=None):
        self.kernel_name = kernel_name
        self.grid = grid
        self.reason = reason
        parts = [message]
        if kernel_name:
            parts.append(f"  kernel: {kernel_name}")
        if grid:
            parts.append(f"  grid: {grid}")
        if reason:
            parts.append(f"  reason: {reason}")
        super().__init__("\n".join(parts))
