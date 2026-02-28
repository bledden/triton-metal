"""Emit Metal Shading Language (MSL) source from kernel descriptions.

Two-layer architecture:
1. KernelBuilder — captures kernel semantics (args, block size, ops)
2. MSLCodeGen — emits valid MSL compute kernel source from a KernelBuilder

The KernelBuilder can be driven by:
- Direct Python API (standalone, no triton required)
- TTGIR MLIR walking (when triton is available)

Phase 1 targets elementwise kernels: vector add, scalar mul, activation
functions (silu, gelu), and fused elementwise combinations.
"""

from triton_metal.codegen.msl_types import triton_type_to_msl


# ---------------------------------------------------------------------------
# Kernel description API
# ---------------------------------------------------------------------------

class Arg:
    """A kernel argument (buffer pointer or scalar)."""

    def __init__(self, name, dtype, is_ptr=False, const=False):
        self.name = name
        self.dtype = dtype  # Triton type string: "fp32", "i32", etc.
        self.is_ptr = is_ptr
        self.const = const  # Read-only buffer

    def msl_param(self, index):
        """Emit MSL kernel parameter declaration."""
        if self.is_ptr:
            inner = triton_type_to_msl(self.dtype)
            qual = "const " if self.const else ""
            return f"device {qual}{inner}* {self.name} [[buffer({index})]]"
        else:
            msl_ty = triton_type_to_msl(self.dtype)
            return f"constant {msl_ty}& {self.name} [[buffer({index})]]"


class KernelBuilder:
    """Describes a compute kernel's structure for MSL emission."""

    def __init__(self, name, block_size=256):
        self.name = name
        self.block_size = block_size
        self.args = []
        self._body_lines = []
        self._locals = {}
        self._indent = 1

    # -- Argument registration --

    def add_ptr_arg(self, name, dtype="fp32", const=False):
        """Add a device buffer pointer argument."""
        self.args.append(Arg(name, dtype, is_ptr=True, const=const))
        return name

    def add_scalar_arg(self, name, dtype="i32"):
        """Add a scalar argument (passed as constant buffer)."""
        self.args.append(Arg(name, dtype, is_ptr=False))
        return name

    # -- Code generation helpers --

    def _emit(self, line):
        prefix = "    " * self._indent
        self._body_lines.append(f"{prefix}{line}")

    def _var(self, name, expr, ty="auto"):
        """Declare and assign a local variable."""
        self._emit(f"{ty} {name} = {expr};")
        self._locals[name] = ty
        return name

    # -- Triton-like operations --

    def get_program_id(self, var_name="pid"):
        """tt.get_program_id(0) -> threadgroup_position_in_grid."""
        return var_name  # injected as kernel parameter

    def make_block_offsets(self, pid_var="pid", out_var="offsets"):
        """Compute per-thread offsets within a 1D block.

        offsets = pid * BLOCK_SIZE + thread_position_in_threadgroup
        """
        self._var(out_var, f"{pid_var} * {self.block_size} + lid")
        return out_var

    def make_mask(self, offsets_var, n_var, out_var="mask"):
        """Generate a bounds mask: offsets < n_elements."""
        self._var(out_var, f"{offsets_var} < {n_var}", ty="bool")
        return out_var

    def load(self, ptr_var, offsets_var, mask_var=None, out_var=None):
        """Masked load from a buffer pointer + offset."""
        if out_var is None:
            out_var = f"{ptr_var}_val"
        if mask_var:
            self._emit(f"float {out_var} = {mask_var} ? {ptr_var}[{offsets_var}] : 0.0f;")
        else:
            self._emit(f"float {out_var} = {ptr_var}[{offsets_var}];")
        return out_var

    def store(self, ptr_var, offsets_var, val_var, mask_var=None):
        """Masked store to a buffer pointer + offset."""
        if mask_var:
            self._emit(f"if ({mask_var}) {{ {ptr_var}[{offsets_var}] = {val_var}; }}")
        else:
            self._emit(f"{ptr_var}[{offsets_var}] = {val_var};")

    def binary_op(self, op, a_var, b_var, out_var):
        """Emit a binary operation: out = a op b."""
        op_map = {
            "add": "+", "sub": "-", "mul": "*", "div": "/",
            "mod": "%", "and": "&", "or": "|", "xor": "^",
        }
        if op in op_map:
            self._var(out_var, f"{a_var} {op_map[op]} {b_var}", ty="float")
        else:
            raise ValueError(f"Unknown binary op: {op}")
        return out_var

    def unary_op(self, op, x_var, out_var):
        """Emit a unary operation."""
        op_map = {
            "neg": f"-{x_var}",
            "exp": f"exp({x_var})",
            "log": f"log({x_var})",
            "sqrt": f"sqrt({x_var})",
            "rsqrt": f"rsqrt({x_var})",
            "abs": f"abs({x_var})",
            "sigmoid": f"(1.0f / (1.0f + exp(-{x_var})))",
            "tanh": f"tanh({x_var})",
            "sin": f"sin({x_var})",
            "cos": f"cos({x_var})",
        }
        if op in op_map:
            self._var(out_var, op_map[op], ty="float")
        else:
            raise ValueError(f"Unknown unary op: {op}")
        return out_var

    def fused_op(self, op_name, args_vars, out_var):
        """Emit a fused multi-input operation."""
        fused_map = {
            "fma": lambda a: f"fma({a[0]}, {a[1]}, {a[2]})",
            "silu": lambda a: f"({a[0]} / (1.0f + exp(-{a[0]})))",
            # MSL has no erf(). Both gelu variants use the tanh approximation
            # which is standard in ML frameworks.
            "gelu": lambda a: (
                f"({a[0]} * 0.5f * (1.0f + tanh(0.7978845608028654f * "
                f"({a[0]} + 0.044715f * {a[0]} * {a[0]} * {a[0]}))))"
            ),
            "gelu_tanh": lambda a: (
                f"({a[0]} * 0.5f * (1.0f + tanh(0.7978845608028654f * "
                f"({a[0]} + 0.044715f * {a[0]} * {a[0]} * {a[0]}))))"
            ),
        }
        if op_name in fused_map:
            self._var(out_var, fused_map[op_name](args_vars), ty="float")
        else:
            raise ValueError(f"Unknown fused op: {op_name}")
        return out_var

    def raw_line(self, line):
        """Emit a raw MSL line."""
        self._emit(line)

    def comment(self, text):
        """Emit a comment."""
        self._emit(f"// {text}")

    # -- Build the MSL source --

    def build(self):
        """Generate the complete MSL kernel source."""
        gen = MSLCodeGen(self)
        return gen.emit()


class MSLCodeGen:
    """Generates MSL source from a KernelBuilder."""

    def __init__(self, builder):
        self.builder = builder

    def emit(self):
        lines = []
        lines.append("#include <metal_stdlib>")
        lines.append("using namespace metal;")
        lines.append("")

        # Kernel signature
        params = []
        for i, arg in enumerate(self.builder.args):
            params.append(f"    {arg.msl_param(i)}")

        # Thread position qualifiers
        next_idx = len(self.builder.args)  # not used for qualifiers
        params.append("    uint pid [[threadgroup_position_in_grid]]")
        params.append("    uint lid [[thread_position_in_threadgroup]]")
        params.append("    uint tid [[thread_position_in_grid]]")

        lines.append(f"kernel void {self.builder.name}(")
        lines.append(",\n".join(params))
        lines.append(") {")

        # Body
        for line in self.builder._body_lines:
            lines.append(line)

        lines.append("}")
        lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# High-level kernel generators
# ---------------------------------------------------------------------------

def make_elementwise_kernel(name, n_inputs, op, block_size=256, dtype="fp32"):
    """Generate an elementwise kernel: output[i] = op(input_0[i], ..., input_n[i]).

    Args:
        name: Kernel function name.
        n_inputs: Number of input buffers.
        op: Operation name ("add", "mul", "sub", "silu", "gelu", etc.)
        block_size: Elements per threadgroup.
        dtype: Data type.

    Returns:
        MSL source code string.
    """
    kb = KernelBuilder(name, block_size=block_size)

    # Register arguments
    input_names = []
    for i in range(n_inputs):
        input_names.append(kb.add_ptr_arg(f"input{i}", dtype=dtype, const=True))
    out_name = kb.add_ptr_arg("output", dtype=dtype, const=False)
    n_name = kb.add_scalar_arg("n_elements", dtype="u32")

    # Compute offsets and mask
    offsets = kb.make_block_offsets("pid", "offsets")
    mask = kb.make_mask(offsets, n_name, "mask")

    # Load inputs
    val_names = []
    for i, inp in enumerate(input_names):
        val = kb.load(inp, offsets, mask, out_var=f"val{i}")
        val_names.append(val)

    # Apply operation
    if n_inputs == 1:
        # Unary or fused unary
        if op in ("silu", "gelu", "gelu_tanh"):
            result = kb.fused_op(op, val_names, "result")
        else:
            result = kb.unary_op(op, val_names[0], "result")
    elif n_inputs == 2:
        result = kb.binary_op(op, val_names[0], val_names[1], "result")
    elif n_inputs == 3 and op == "fma":
        result = kb.fused_op("fma", val_names, "result")
    else:
        raise ValueError(f"Unsupported: {n_inputs} inputs with op '{op}'")

    # Store result
    kb.store(out_name, offsets, result, mask)

    return kb.build()


def make_vector_add_kernel(block_size=256, dtype="fp32"):
    """Generate a vector add kernel: output = a + b."""
    return make_elementwise_kernel("vector_add", 2, "add", block_size, dtype)


def make_silu_kernel(block_size=256, dtype="fp32"):
    """Generate a SiLU activation kernel: output = x * sigmoid(x)."""
    return make_elementwise_kernel("silu_kernel", 1, "silu", block_size, dtype)


def make_gelu_kernel(block_size=256, dtype="fp32"):
    """Generate a GELU activation kernel."""
    return make_elementwise_kernel("gelu_kernel", 1, "gelu", block_size, dtype)


def make_scalar_mul_kernel(block_size=256, dtype="fp32"):
    """Generate a scalar multiply kernel: output = input * scalar.

    Note: scalar is passed as a separate buffer argument.
    """
    kb = KernelBuilder("scalar_mul", block_size=block_size)

    kb.add_ptr_arg("input", dtype=dtype, const=True)
    kb.add_ptr_arg("output", dtype=dtype, const=False)
    kb.add_scalar_arg("scalar", dtype="fp32")
    kb.add_scalar_arg("n_elements", dtype="u32")

    offsets = kb.make_block_offsets("pid", "offsets")
    mask = kb.make_mask(offsets, "n_elements", "mask")

    val = kb.load("input", offsets, mask, out_var="val")
    result = kb.binary_op("mul", val, "scalar", "result")
    kb.store("output", offsets, result, mask)

    return kb.build()


# ---------------------------------------------------------------------------
# TTGIR integration (requires triton)
# ---------------------------------------------------------------------------

def emit_msl(mod, metadata, options):
    """Convert a TritonGPU IR module to MSL source code.

    This is the entry point called by MetalBackend.make_msl().

    Args:
        mod: The MLIR module after TTGIR passes.
        metadata: Compilation metadata dict.
        options: MetalOptions instance.

    Returns:
        MSL source code as a string.
    """
    ir_text = str(mod)
    kernel_name = _extract_kernel_name(ir_text)
    metadata["name"] = kernel_name

    # TODO: Parse TTGIR and build KernelBuilder from it.
    # For now, generate a passthrough copy kernel.
    num_warps = options.num_warps
    threads_per_tg = num_warps * 32

    kb = KernelBuilder(kernel_name, block_size=threads_per_tg)
    kb.add_ptr_arg("input", dtype="fp32", const=True)
    kb.add_ptr_arg("output", dtype="fp32", const=False)
    kb.add_scalar_arg("n_elements", dtype="u32")

    offsets = kb.make_block_offsets("pid", "offsets")
    mask = kb.make_mask(offsets, "n_elements", "mask")
    val = kb.load("input", offsets, mask)
    kb.store("output", offsets, "input_val", mask)

    return kb.build()


def _extract_kernel_name(ir_text):
    """Extract the kernel function name from MLIR text."""
    import re

    match = re.search(r"tt\.func\s+public\s+@(\w+)\s*\(", ir_text)
    if match:
        return match.group(1)
    match = re.search(r"func\.func\s+@(\w+)\s*\(", ir_text)
    if match:
        return match.group(1)
    return "triton_kernel"
