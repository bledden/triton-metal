"""Emit Metal Shading Language (MSL) source from kernel descriptions.

Two-layer architecture:
1. KernelBuilder — captures kernel semantics (args, block size, ops)
2. MSLCodeGen — emits valid MSL compute kernel source from a KernelBuilder

The KernelBuilder can be driven by:
- Direct Python API (standalone, no triton required)
- TTGIR MLIR walking (when triton is available)

Supports:
- Elementwise ops: vector add, scalar mul, activation functions (silu, gelu)
- Reductions: sum, max, min via SIMD-group intrinsics + threadgroup shared memory
- Softmax: fused row-wise max → subtract → exp → sum → divide
- Matmul: tiled matrix multiplication with threadgroup shared memory
- Layer norm: mean → variance → normalize with gamma/beta
- Cross-entropy: fused log-softmax + target selection loss
- Flash Attention: online softmax with tiled Q@K^T and P@V accumulation
"""

from triton_metal.codegen.msl_types import triton_type_to_msl


# ---------------------------------------------------------------------------
# Type helpers
# ---------------------------------------------------------------------------

def _msl_compute_type(dtype):
    """Get the MSL compute type for a Triton dtype.

    For fp16, computations are done in float and cast back to half on store.
    This matches Triton's behavior and avoids precision issues.
    """
    if dtype in ("fp16", "bf16"):
        return "float"
    return triton_type_to_msl(dtype)


def _msl_zero(dtype):
    """Get the zero literal for a MSL type."""
    if dtype in ("fp16", "bf16", "fp32", "f32"):
        return "0.0f"
    return "0"


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
        self._needs_simd_qualifiers = False
        self._threadgroup_arrays = []  # (name, dtype, size) for static tg memory

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

    def load(self, ptr_var, offsets_var, mask_var=None, out_var=None, dtype="fp32"):
        """Masked load from a buffer pointer + offset.

        For FP16/BF16 buffers, the value is promoted to float for computation.
        """
        if out_var is None:
            out_var = f"{ptr_var}_val"
        compute_ty = _msl_compute_type(dtype)
        zero = _msl_zero(dtype)
        if mask_var:
            self._emit(f"{compute_ty} {out_var} = {mask_var} ? "
                       f"static_cast<{compute_ty}>({ptr_var}[{offsets_var}]) : {zero};")
        else:
            self._emit(f"{compute_ty} {out_var} = "
                       f"static_cast<{compute_ty}>({ptr_var}[{offsets_var}]);")
        return out_var

    def store(self, ptr_var, offsets_var, val_var, mask_var=None, dtype="fp32"):
        """Masked store to a buffer pointer + offset.

        For FP16/BF16 buffers, casts from float compute type back to storage type.
        """
        store_ty = triton_type_to_msl(dtype)
        compute_ty = _msl_compute_type(dtype)
        needs_cast = (store_ty != compute_ty)
        cast_val = f"static_cast<{store_ty}>({val_var})" if needs_cast else val_var
        if mask_var:
            self._emit(f"if ({mask_var}) {{ {ptr_var}[{offsets_var}] = {cast_val}; }}")
        else:
            self._emit(f"{ptr_var}[{offsets_var}] = {cast_val};")

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

    # -- Indentation control --

    def indent(self):
        """Increase indentation level."""
        self._indent += 1

    def dedent(self):
        """Decrease indentation level."""
        self._indent = max(1, self._indent - 1)

    def begin_if(self, condition):
        """Emit an if statement and increase indent."""
        self._emit(f"if ({condition}) {{")
        self._indent += 1

    def end_block(self):
        """Close a block and decrease indent."""
        self._indent -= 1
        self._emit("}")

    # -- Shared memory and barriers --

    def declare_threadgroup_array(self, name, dtype="fp32", size=None):
        """Declare a static threadgroup memory array."""
        if size is None:
            size = (self.block_size + 31) // 32  # one slot per SIMD group
        self._threadgroup_arrays.append((name, dtype, size))
        return name

    def barrier(self, kind="threadgroup"):
        """Emit a memory barrier."""
        from triton_metal.codegen.msl_builtins import BARRIERS
        self._emit(f"{BARRIERS[kind]};")

    # -- Reduction operations --

    def simd_reduce(self, op, val_var, out_var):
        """Emit a SIMD-group reduction: out = simd_op(val).

        Uses hardware SIMD intrinsics (32-wide on Apple Silicon).
        """
        self._needs_simd_qualifiers = True
        from triton_metal.codegen.msl_builtins import SIMD_REDUCTIONS
        intrinsic = SIMD_REDUCTIONS[op]
        self._var(out_var, f"{intrinsic}({val_var})", ty="float")
        return out_var

    def threadgroup_reduce(self, op, val_var, shared_var, out_var):
        """Emit a full threadgroup reduction: SIMD reduce → shared mem → final SIMD reduce.

        Standard two-level pattern:
        1. simd_op within each SIMD group
        2. Lane 0 writes to shared memory
        3. Barrier
        4. SIMD group 0 reads shared and does final reduction

        Variable names are suffixed with out_var to avoid collisions when
        called multiple times in the same kernel.
        """
        self._needs_simd_qualifiers = True
        from triton_metal.codegen.msl_builtins import SIMD_REDUCTIONS

        intrinsic = SIMD_REDUCTIONS[op]
        identity = {
            "sum": "0.0f",
            "max": "-INFINITY",
            "min": "INFINITY",
        }[op]

        # Unique intermediate variable names
        simd_var = f"simd_{out_var}"
        read_var = f"shared_{out_var}"

        # Step 1: SIMD-level reduction
        self._var(simd_var, f"{intrinsic}({val_var})", ty="float")

        # Step 2: Initialize shared memory
        self.begin_if("sgitg == 0")
        self._emit(f"{shared_var}[tiisg] = {identity};")
        self.end_block()
        self.barrier("threadgroup")

        # Step 3: Lane 0 of each SIMD group writes to shared
        self.begin_if("tiisg == 0")
        self._emit(f"{shared_var}[sgitg] = {simd_var};")
        self.end_block()
        self.barrier("threadgroup")

        # Step 4: SIMD group 0 reads back and does final reduction
        self._var(read_var, f"{shared_var}[tiisg]", ty="float")
        self._var(out_var, f"{intrinsic}({read_var})", ty="float")
        return out_var

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
        params.append("    uint pid [[threadgroup_position_in_grid]]")
        params.append("    uint lid [[thread_position_in_threadgroup]]")
        params.append("    uint tid [[thread_position_in_grid]]")

        # SIMD qualifiers (only when reductions are used)
        if self.builder._needs_simd_qualifiers:
            params.append("    uint sgitg [[simdgroup_index_in_threadgroup]]")
            params.append("    uint tiisg [[thread_index_in_simdgroup]]")

        lines.append(f"kernel void {self.builder.name}(")
        lines.append(",\n".join(params))
        lines.append(") {")

        # Static threadgroup memory declarations
        for tg_name, tg_dtype, tg_size in self.builder._threadgroup_arrays:
            msl_ty = triton_type_to_msl(tg_dtype)
            lines.append(f"    threadgroup {msl_ty} {tg_name}[{tg_size}];")

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

    # Load inputs (promoted to float for FP16/BF16)
    val_names = []
    for i, inp in enumerate(input_names):
        val = kb.load(inp, offsets, mask, out_var=f"val{i}", dtype=dtype)
        val_names.append(val)

    # Apply operation (always in float compute precision)
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

    # Store result (cast back to half for FP16/BF16)
    kb.store(out_name, offsets, result, mask, dtype=dtype)

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


def make_swiglu_kernel(block_size=256, dtype="fp32"):
    """Generate a fused SwiGLU activation kernel.

    SwiGLU(x, gate) = SiLU(gate) * x = (gate / (1 + exp(-gate))) * x

    Used in LLaMA, Mistral, and Gemma FFN layers. Fuses the gate
    activation and element-wise multiply into one kernel for memory efficiency.

    Args:
        block_size: Threads per threadgroup.
        dtype: Data type.
    """
    kb = KernelBuilder("swiglu_kernel", block_size=block_size)

    kb.add_ptr_arg("x", dtype=dtype, const=True)
    kb.add_ptr_arg("gate", dtype=dtype, const=True)
    kb.add_ptr_arg("output", dtype=dtype, const=False)
    kb.add_scalar_arg("n_elements", dtype="u32")

    offsets = kb.make_block_offsets("pid", "offsets")
    mask = kb.make_mask(offsets, "n_elements", "mask")

    x_val = kb.load("x", offsets, mask, out_var="x_val", dtype=dtype)
    gate_val = kb.load("gate", offsets, mask, out_var="gate_val", dtype=dtype)

    # SiLU(gate) * x
    silu_gate = kb.fused_op("silu", [gate_val], "silu_gate")
    result = kb.binary_op("mul", silu_gate, x_val, "result")

    kb.store("output", offsets, result, mask, dtype=dtype)

    return kb.build()


def make_embedding_kernel(block_size=256, dtype="fp32"):
    """Generate an embedding lookup kernel.

    output[i, :] = table[indices[i], :]

    Each threadgroup handles one token (one row of the output).
    Threads within the group cooperatively copy the embedding vector.

    Args:
        block_size: Threads per threadgroup.
        dtype: Data type.

    Kernel args:
        table: [vocab_size, embed_dim] embedding table
        indices: [batch_size] int32 token indices
        output: [batch_size, embed_dim] output
        embed_dim: embedding dimension
    """
    kb = KernelBuilder("embedding_kernel", block_size=block_size)

    kb.add_ptr_arg("table", dtype=dtype, const=True)
    kb.add_ptr_arg("indices", dtype="i32", const=True)
    kb.add_ptr_arg("output", dtype=dtype, const=False)
    kb.add_scalar_arg("embed_dim", dtype="u32")

    # pid = token index in the batch
    kb._var("token_idx", "indices[pid]", ty="int")
    kb._var("src_offset", "uint(token_idx) * embed_dim", ty="uint")
    kb._var("dst_offset", "pid * embed_dim", ty="uint")

    # Each thread copies one or more elements of the embedding vector
    kb.raw_line(f"for (uint i = lid; i < embed_dim; i += {block_size}u) {{")
    kb.indent()
    kb.raw_line("output[dst_offset + i] = table[src_offset + i];")
    kb.dedent()
    kb.raw_line("}")

    return kb.build()


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

    val = kb.load("input", offsets, mask, out_var="val", dtype=dtype)
    result = kb.binary_op("mul", val, "scalar", "result")
    kb.store("output", offsets, result, mask, dtype=dtype)

    return kb.build()


# ---------------------------------------------------------------------------
# Reduction kernel generators
# ---------------------------------------------------------------------------

def make_reduce_kernel(name, op, block_size=256, dtype="fp32"):
    """Generate a 1D reduction kernel: output[group] = reduce(input[group*N:...]).

    Each threadgroup reduces block_size elements. For inputs larger than
    block_size, launch multiple threadgroups and reduce the partial results.

    Uses two-level reduction: SIMD intrinsics + threadgroup shared memory.

    Args:
        name: Kernel function name.
        op: Reduction operation ("sum", "max", "min").
        block_size: Elements per threadgroup.
        dtype: Data type.
    """
    n_simd_groups = (block_size + 31) // 32

    kb = KernelBuilder(name, block_size=block_size)
    kb.add_ptr_arg("input", dtype=dtype, const=True)
    kb.add_ptr_arg("output", dtype=dtype, const=False)
    kb.add_scalar_arg("n_elements", dtype="u32")

    # Shared memory for cross-SIMD-group reduction
    kb.declare_threadgroup_array("shared", dtype=dtype, size=n_simd_groups)

    identity = {"sum": "0.0f", "max": "-INFINITY", "min": "INFINITY"}[op]
    combine = {"sum": "+", "max": "max", "min": "min"}[op]

    # Each thread accumulates over strided elements
    kb._var("acc", identity, ty="float")
    kb.raw_line(f"for (uint i = lid; i < n_elements; i += {block_size}) {{")
    kb.indent()
    kb._var("idx", f"pid * n_elements + i", ty="uint")
    if combine in ("+",):
        kb.raw_line("acc += input[idx];")
    else:
        kb.raw_line(f"acc = {combine}(acc, input[idx]);")
    kb.dedent()
    kb.raw_line("}")

    # Two-level threadgroup reduction
    kb.threadgroup_reduce(op, "acc", "shared", "total")

    # Thread 0 writes result
    kb.begin_if("lid == 0")
    kb.raw_line("output[pid] = total;")
    kb.end_block()

    return kb.build()


def make_softmax_kernel(block_size=256, dtype="fp32"):
    """Generate a fused row-wise softmax kernel.

    Each threadgroup processes one row:
    1. Find max(row) — for numerical stability
    2. Compute exp(x - max) for each element
    3. Sum the exponentials
    4. Divide each by the sum

    Args:
        block_size: Threads per threadgroup (should be >= row length or will stride).
        dtype: Data type.
    """
    n_simd_groups = (block_size + 31) // 32

    kb = KernelBuilder("softmax_kernel", block_size=block_size)
    kb.add_ptr_arg("input", dtype=dtype, const=True)
    kb.add_ptr_arg("output", dtype=dtype, const=False)
    kb.add_scalar_arg("n_cols", dtype="u32")

    # Two shared arrays: one for max reduction, one for sum reduction
    kb.declare_threadgroup_array("shared_max", dtype=dtype, size=n_simd_groups)
    kb.declare_threadgroup_array("shared_sum", dtype=dtype, size=n_simd_groups)

    # Row base pointer: each threadgroup handles one row
    kb._var("row_start", "pid * n_cols", ty="uint")

    # Pass 1: Find row max (strided accumulation)
    kb._var("local_max", "-INFINITY", ty="float")
    kb.raw_line(f"for (uint i = lid; i < n_cols; i += {block_size}) {{")
    kb.indent()
    kb.raw_line("local_max = max(local_max, input[row_start + i]);")
    kb.dedent()
    kb.raw_line("}")

    # Reduce max across threadgroup
    kb.threadgroup_reduce("max", "local_max", "shared_max", "row_max")

    # Broadcast row_max to all threads via shared memory
    kb.begin_if("lid == 0")
    kb.raw_line("shared_max[0] = row_max;")
    kb.end_block()
    kb.barrier("threadgroup")
    kb._var("max_val", "shared_max[0]", ty="float")

    # Pass 2: Compute exp(x - max) and accumulate sum
    kb._var("local_sum", "0.0f", ty="float")
    kb.raw_line(f"for (uint i = lid; i < n_cols; i += {block_size}) {{")
    kb.indent()
    kb._var("e", "exp(input[row_start + i] - max_val)", ty="float")
    kb.raw_line("output[row_start + i] = e;")
    kb.raw_line("local_sum += e;")
    kb.dedent()
    kb.raw_line("}")

    # Reduce sum across threadgroup
    kb.threadgroup_reduce("sum", "local_sum", "shared_sum", "row_sum")

    # Broadcast row_sum
    kb.begin_if("lid == 0")
    kb.raw_line("shared_sum[0] = row_sum;")
    kb.end_block()
    kb.barrier("threadgroup")
    kb._var("sum_val", "shared_sum[0]", ty="float")

    # Pass 3: Normalize
    kb.raw_line(f"for (uint i = lid; i < n_cols; i += {block_size}) {{")
    kb.indent()
    kb.raw_line("output[row_start + i] /= sum_val;")
    kb.dedent()
    kb.raw_line("}")

    return kb.build()


def make_matmul_kernel(block_m=32, block_n=32, block_k=32, dtype="fp32"):
    """Generate a tiled matrix multiplication kernel: C = A @ B.

    A is (M, K), B is (K, N), C is (M, N).
    Each threadgroup computes a BLOCK_M x BLOCK_N tile of C.

    Uses threadgroup shared memory for A and B tiles to enable
    coalesced global memory access and data reuse.

    Constrained by Metal's 32KB threadgroup memory limit:
    - 32x32 fp32 tile = 4KB, two tiles = 8KB (well within limit)

    Args:
        block_m: Tile height (rows of A/C per threadgroup).
        block_n: Tile width (cols of B/C per threadgroup).
        block_k: Tile depth (inner dimension chunk).
        dtype: Data type.
    """
    # Threadgroup size: one thread per output element in the tile
    threads_per_tg = block_m * block_n

    kb = KernelBuilder("matmul_kernel", block_size=threads_per_tg)
    kb.add_ptr_arg("A", dtype=dtype, const=True)
    kb.add_ptr_arg("B", dtype=dtype, const=True)
    kb.add_ptr_arg("C", dtype=dtype, const=False)
    kb.add_scalar_arg("M", dtype="u32")
    kb.add_scalar_arg("N", dtype="u32")
    kb.add_scalar_arg("K", dtype="u32")

    # Shared memory tiles
    kb.declare_threadgroup_array("tileA", dtype=dtype, size=block_m * block_k)
    kb.declare_threadgroup_array("tileB", dtype=dtype, size=block_k * block_n)

    # Thread mapping: each thread computes one element of the output tile
    # pid.x = tile column, pid.y = tile row (we use 2D grid)
    # For simplicity, we flatten the 2D threadgroup into 1D and compute
    # the local (row, col) from lid.
    kb._var("local_row", f"lid / {block_n}u", ty="uint")
    kb._var("local_col", f"lid % {block_n}u", ty="uint")

    # Global tile position: pid encodes the 2D tile index.
    # We use a linearized 2D grid: pid = tile_row * n_tile_cols + tile_col
    kb._var("n_tile_cols", f"(N + {block_n}u - 1u) / {block_n}u", ty="uint")
    kb._var("tile_row", "pid / n_tile_cols", ty="uint")
    kb._var("tile_col", "pid % n_tile_cols", ty="uint")

    # Global output row/col for this thread
    kb._var("global_row", f"tile_row * {block_m}u + local_row", ty="uint")
    kb._var("global_col", f"tile_col * {block_n}u + local_col", ty="uint")

    # Accumulator
    kb._var("acc", "0.0f", ty="float")

    # Tile loop over K dimension
    kb._var("n_tiles_k", f"(K + {block_k}u - 1u) / {block_k}u", ty="uint")
    kb.raw_line("for (uint tk = 0; tk < n_tiles_k; tk++) {")
    kb.indent()

    # Load A tile: tileA[local_row][local_col_k] = A[global_row][tk*BLOCK_K + local_col_k]
    # Each thread loads one element. We need block_m * block_k elements loaded
    # by block_m * block_n threads, so some threads load multiple elements.
    kb._var("a_col", f"tk * {block_k}u + local_col", ty="uint")
    kb.raw_line(f"if (global_row < M && a_col < K) {{")
    kb.indent()
    kb.raw_line(f"tileA[local_row * {block_k}u + local_col] = A[global_row * K + a_col];")
    kb.dedent()
    kb.raw_line("} else {")
    kb.indent()
    kb.raw_line(f"tileA[local_row * {block_k}u + local_col] = 0.0f;")
    kb.dedent()
    kb.raw_line("}")

    # Load B tile: tileB[local_row_k][local_col] = B[tk*BLOCK_K + local_row_k][global_col]
    kb._var("b_row", f"tk * {block_k}u + local_row", ty="uint")
    kb.raw_line(f"if (b_row < K && global_col < N) {{")
    kb.indent()
    kb.raw_line(f"tileB[local_row * {block_n}u + local_col] = B[b_row * N + global_col];")
    kb.dedent()
    kb.raw_line("} else {")
    kb.indent()
    kb.raw_line(f"tileB[local_row * {block_n}u + local_col] = 0.0f;")
    kb.dedent()
    kb.raw_line("}")

    kb.barrier("threadgroup")

    # Compute partial dot product for this tile
    kb.raw_line(f"for (uint kk = 0; kk < {block_k}u; kk++) {{")
    kb.indent()
    kb.raw_line(f"acc += tileA[local_row * {block_k}u + kk] * tileB[kk * {block_n}u + local_col];")
    kb.dedent()
    kb.raw_line("}")

    kb.barrier("threadgroup")

    kb.dedent()
    kb.raw_line("}")  # end tile loop

    # Write result
    kb.raw_line("if (global_row < M && global_col < N) {")
    kb.indent()
    kb.raw_line("C[global_row * N + global_col] = acc;")
    kb.dedent()
    kb.raw_line("}")

    return kb.build()


def make_rms_norm_kernel(block_size=256, dtype="fp32", eps=1e-6):
    """Generate a fused RMS normalization kernel.

    Used in LLaMA, Mistral, Gemma, and other modern LLMs.
    For each row: output = x * rsqrt(mean(x^2) + eps) * weight

    Each threadgroup processes one row. Three passes:
    1. Compute sum of squares (strided accumulation + threadgroup reduce)
    2. Compute RMS = rsqrt(mean_sq + eps)
    3. Apply normalization: output[i] = input[i] * rms * weight[i]

    Args:
        block_size: Threads per threadgroup.
        dtype: Data type.
        eps: Epsilon for numerical stability.
    """
    n_simd_groups = (block_size + 31) // 32

    kb = KernelBuilder("rms_norm_kernel", block_size=block_size)
    kb.add_ptr_arg("input", dtype=dtype, const=True)
    kb.add_ptr_arg("weight", dtype=dtype, const=True)
    kb.add_ptr_arg("output", dtype=dtype, const=False)
    kb.add_scalar_arg("n_cols", dtype="u32")

    kb.declare_threadgroup_array("shared_sq", dtype=dtype, size=n_simd_groups)

    # Row base pointer
    kb._var("row_start", "pid * n_cols", ty="uint")

    # Pass 1: Sum of squares (strided)
    kb._var("sq_sum", "0.0f", ty="float")
    kb.raw_line(f"for (uint i = lid; i < n_cols; i += {block_size}u) {{")
    kb.indent()
    kb._var("v", "input[row_start + i]", ty="float")
    kb.raw_line("sq_sum += v * v;")
    kb.dedent()
    kb.raw_line("}")

    # Reduce sum of squares across threadgroup
    kb.threadgroup_reduce("sum", "sq_sum", "shared_sq", "total_sq")

    # Broadcast and compute RMS
    kb.begin_if("lid == 0")
    kb.raw_line("shared_sq[0] = total_sq;")
    kb.end_block()
    kb.barrier("threadgroup")
    kb._var("mean_sq", "shared_sq[0] / float(n_cols)", ty="float")
    kb._var("rms", f"rsqrt(mean_sq + {eps}f)", ty="float")

    # Pass 2: Apply normalization
    kb.raw_line(f"for (uint i = lid; i < n_cols; i += {block_size}u) {{")
    kb.indent()
    kb.raw_line("output[row_start + i] = input[row_start + i] * rms * weight[i];")
    kb.dedent()
    kb.raw_line("}")

    return kb.build()


def make_rope_kernel(block_size=256, dtype="fp32"):
    """Generate a fused RoPE (rotary position embedding) kernel.

    Applies rotary position embeddings to pairs of elements:
    For each pair (x0, x1) at position pos with frequency freq:
        out0 = x0 * cos(theta) - x1 * sin(theta)
        out1 = x0 * sin(theta) + x1 * cos(theta)
    where theta = pos * freq

    Each threadgroup processes elements for one position.
    Frequencies are pre-computed: freq[i] = 1 / (10000^(2i/dim)).

    Args:
        block_size: Threads per threadgroup.
        dtype: Data type.

    Kernel args:
        input: [seq_len, dim] tensor
        freqs: [dim/2] pre-computed inverse frequencies
        output: [seq_len, dim] tensor
        dim: hidden dimension (must be even)
        pos_offset: starting position index
    """
    kb = KernelBuilder("rope_kernel", block_size=block_size)
    kb.add_ptr_arg("input", dtype=dtype, const=True)
    kb.add_ptr_arg("freqs", dtype=dtype, const=True)
    kb.add_ptr_arg("output", dtype=dtype, const=False)
    kb.add_scalar_arg("dim", dtype="u32")
    kb.add_scalar_arg("pos_offset", dtype="u32")

    # Each threadgroup handles one position (pid = position index)
    kb._var("pos", "pid + pos_offset", ty="uint")
    kb._var("row_start", "pid * dim", ty="uint")

    # Each thread handles a pair of elements
    kb.raw_line(f"for (uint i = lid; i < dim / 2u; i += {block_size}u) {{")
    kb.indent()
    kb._var("theta", "float(pos) * freqs[i]", ty="float")
    kb._var("cos_t", "cos(theta)", ty="float")
    kb._var("sin_t", "sin(theta)", ty="float")
    kb._var("x0", "input[row_start + 2u * i]", ty="float")
    kb._var("x1", "input[row_start + 2u * i + 1u]", ty="float")
    kb.raw_line("output[row_start + 2u * i] = x0 * cos_t - x1 * sin_t;")
    kb.raw_line("output[row_start + 2u * i + 1u] = x0 * sin_t + x1 * cos_t;")
    kb.dedent()
    kb.raw_line("}")

    return kb.build()


def make_layer_norm_kernel(block_size=256, dtype="fp32", eps=1e-6):
    """Generate a fused layer normalization kernel.

    Standard layer norm used in transformers:
    output = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta

    Each threadgroup processes one row. Three passes:
    1. Compute mean (strided accumulation + threadgroup reduce)
    2. Compute variance (strided + reduce)
    3. Normalize: (x - mean) * rsqrt(var + eps) * gamma + beta

    Args:
        block_size: Threads per threadgroup.
        dtype: Data type.
        eps: Epsilon for numerical stability.
    """
    n_simd_groups = (block_size + 31) // 32

    kb = KernelBuilder("layer_norm_kernel", block_size=block_size)
    kb.add_ptr_arg("input", dtype=dtype, const=True)
    kb.add_ptr_arg("gamma", dtype=dtype, const=True)
    kb.add_ptr_arg("beta", dtype=dtype, const=True)
    kb.add_ptr_arg("output", dtype=dtype, const=False)
    kb.add_scalar_arg("n_cols", dtype="u32")

    kb.declare_threadgroup_array("shared_mean", dtype=dtype, size=n_simd_groups)
    kb.declare_threadgroup_array("shared_var", dtype=dtype, size=n_simd_groups)

    # Row base pointer
    kb._var("row_start", "pid * n_cols", ty="uint")

    # Pass 1: Compute mean
    kb._var("local_sum", "0.0f", ty="float")
    kb.raw_line(f"for (uint i = lid; i < n_cols; i += {block_size}u) {{")
    kb.indent()
    kb.raw_line("local_sum += input[row_start + i];")
    kb.dedent()
    kb.raw_line("}")

    kb.threadgroup_reduce("sum", "local_sum", "shared_mean", "total_sum")

    kb.begin_if("lid == 0")
    kb.raw_line("shared_mean[0] = total_sum;")
    kb.end_block()
    kb.barrier("threadgroup")
    kb._var("mean_val", "shared_mean[0] / float(n_cols)", ty="float")

    # Pass 2: Compute variance
    kb._var("local_var", "0.0f", ty="float")
    kb.raw_line(f"for (uint i = lid; i < n_cols; i += {block_size}u) {{")
    kb.indent()
    kb._var("diff", "input[row_start + i] - mean_val", ty="float")
    kb.raw_line("local_var += diff * diff;")
    kb.dedent()
    kb.raw_line("}")

    kb.threadgroup_reduce("sum", "local_var", "shared_var", "total_var")

    kb.begin_if("lid == 0")
    kb.raw_line("shared_var[0] = total_var;")
    kb.end_block()
    kb.barrier("threadgroup")
    kb._var("var_val", "shared_var[0] / float(n_cols)", ty="float")
    kb._var("inv_std", f"rsqrt(var_val + {eps}f)", ty="float")

    # Pass 3: Normalize
    kb.raw_line(f"for (uint i = lid; i < n_cols; i += {block_size}u) {{")
    kb.indent()
    kb.raw_line("output[row_start + i] = (input[row_start + i] - mean_val) * inv_std * gamma[i] + beta[i];")
    kb.dedent()
    kb.raw_line("}")

    return kb.build()


def make_cross_entropy_kernel(block_size=256, dtype="fp32"):
    """Generate a fused cross-entropy loss kernel.

    For each sample (row):
    1. Compute max(logits) for numerical stability
    2. Compute log_sum_exp = max + log(sum(exp(logits - max)))
    3. loss = log_sum_exp - logits[target]

    Each threadgroup processes one sample. Outputs per-sample losses.

    Args:
        block_size: Threads per threadgroup.
        dtype: Data type.

    Kernel args:
        logits: [batch, vocab_size] tensor
        targets: [batch] int32 tensor (class indices)
        losses: [batch] float output (per-sample losses)
        vocab_size: number of classes
    """
    n_simd_groups = (block_size + 31) // 32

    kb = KernelBuilder("cross_entropy_kernel", block_size=block_size)
    kb.add_ptr_arg("logits", dtype=dtype, const=True)
    kb.add_ptr_arg("targets", dtype="i32", const=True)
    kb.add_ptr_arg("losses", dtype=dtype, const=False)
    kb.add_scalar_arg("vocab_size", dtype="u32")

    kb.declare_threadgroup_array("shared_max", dtype=dtype, size=n_simd_groups)
    kb.declare_threadgroup_array("shared_sum", dtype=dtype, size=n_simd_groups)

    # Row base pointer
    kb._var("row_start", "pid * vocab_size", ty="uint")

    # Pass 1: Find max logit for numerical stability
    kb._var("local_max", "-INFINITY", ty="float")
    kb.raw_line(f"for (uint i = lid; i < vocab_size; i += {block_size}u) {{")
    kb.indent()
    kb.raw_line("local_max = max(local_max, logits[row_start + i]);")
    kb.dedent()
    kb.raw_line("}")

    kb.threadgroup_reduce("max", "local_max", "shared_max", "row_max")

    kb.begin_if("lid == 0")
    kb.raw_line("shared_max[0] = row_max;")
    kb.end_block()
    kb.barrier("threadgroup")
    kb._var("max_val", "shared_max[0]", ty="float")

    # Pass 2: Compute sum(exp(logits - max))
    kb._var("local_exp_sum", "0.0f", ty="float")
    kb.raw_line(f"for (uint i = lid; i < vocab_size; i += {block_size}u) {{")
    kb.indent()
    kb.raw_line("local_exp_sum += exp(logits[row_start + i] - max_val);")
    kb.dedent()
    kb.raw_line("}")

    kb.threadgroup_reduce("sum", "local_exp_sum", "shared_sum", "total_exp_sum")

    # Thread 0 computes final loss
    kb.begin_if("lid == 0")
    kb._var("target_idx", "targets[pid]", ty="int")
    kb._var("log_sum_exp", "max_val + log(total_exp_sum)", ty="float")
    kb._var("target_logit", "logits[row_start + uint(target_idx)]", ty="float")
    kb.raw_line("losses[pid] = log_sum_exp - target_logit;")
    kb.end_block()

    return kb.build()


def make_flash_attention_kernel(head_dim=64, Br=16, Bc=16, block_size=256, causal=False):
    """Generate a fused Flash Attention kernel for Metal.

    Implements the FlashAttention-2 algorithm with online softmax:
    For each query block:
        O = 0, l = 0, m = -inf
        For each KV block:
            S = Q @ K^T           (Br x Bc scores)
            m_new = max(m, rowmax(S))
            P = exp(S - m_new)    (unnormalized attention)
            l = exp(m - m_new) * l + rowsum(P)
            O = exp(m - m_new) * O + P @ V
            m = m_new
        O = O / l

    When causal=True, attention scores where key position > query position
    are masked to -infinity, implementing autoregressive causal attention.

    Threadgroup memory budget (head_dim=64, Br=Bc=16, fp32):
        Q:  16x64x4 =  4KB
        K:  16x64x4 =  4KB
        V:  16x64x4 =  4KB
        S:  16x16x4 =  1KB
        O:  16x64x4 =  4KB
        l,m: 16x4x2 = 128B
        Total: ~17KB (well within 32KB)

    Args:
        head_dim: Head dimension (d). Must be multiple of 8.
        Br: Query block size (rows of Q per threadgroup).
        Bc: KV block size (rows of K/V loaded per inner loop step).
        block_size: Threads per threadgroup.
        causal: If True, apply causal mask (key pos <= query pos only).

    Kernel args:
        Q: [n_heads * seq_len, head_dim]
        K: [n_heads * seq_len, head_dim]
        V: [n_heads * seq_len, head_dim]
        O: [n_heads * seq_len, head_dim]
        seq_len: sequence length
        scale: 1/sqrt(head_dim)
    """
    # Causal masking: mask out S[i][j] where kv_pos > q_pos
    causal_mask = ""
    if causal:
        causal_mask = """
            uint q_pos = q_start + r;
            uint kv_pos_check = kv_start + c;
            tg_S[i] = (kv_pos < seq_len && kv_pos_check <= q_pos) ? dot * scale : -INFINITY;"""
    else:
        causal_mask = """
            tg_S[i] = (kv_pos < seq_len) ? dot * scale : -INFINITY;"""

    return f"""#include <metal_stdlib>
using namespace metal;

kernel void flash_attention(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* O [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant float& scale [[buffer(5)]],
    uint pid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]]
) {{
    // pid encodes (head_idx * n_q_blocks + q_block_idx)
    // Each threadgroup handles Br={Br} query rows
    const uint BR = {Br}u;
    const uint BC = {Bc}u;
    const uint D = {head_dim}u;

    uint n_q_blocks = (seq_len + BR - 1u) / BR;
    uint head_idx = pid / n_q_blocks;
    uint q_block = pid % n_q_blocks;
    uint q_start = q_block * BR;
    uint head_offset = head_idx * seq_len * D;

    // Threadgroup memory
    threadgroup float tg_Q[{Br} * {head_dim}];    // Br x D
    threadgroup float tg_K[{Bc} * {head_dim}];    // Bc x D
    threadgroup float tg_V[{Bc} * {head_dim}];    // Bc x D
    threadgroup float tg_S[{Br} * {Bc}];          // Br x Bc
    threadgroup float tg_O[{Br} * {head_dim}];    // Br x D
    threadgroup float tg_m[{Br}];                  // row max
    threadgroup float tg_l[{Br}];                  // row sum

    // Load Q block into threadgroup memory
    for (uint i = lid; i < BR * D; i += {block_size}u) {{
        uint r = i / D;
        uint c = i % D;
        uint global_r = q_start + r;
        tg_Q[i] = (global_r < seq_len) ? Q[head_offset + global_r * D + c] : 0.0f;
    }}

    // Initialize O, m, l
    for (uint i = lid; i < BR * D; i += {block_size}u) {{
        tg_O[i] = 0.0f;
    }}
    if (lid < BR) {{
        tg_m[lid] = -INFINITY;
        tg_l[lid] = 0.0f;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Iterate over KV blocks
    uint n_kv_blocks = (seq_len + BC - 1u) / BC;
    for (uint kv_block = 0u; kv_block < n_kv_blocks; kv_block++) {{
        uint kv_start = kv_block * BC;

        // Load K block
        for (uint i = lid; i < BC * D; i += {block_size}u) {{
            uint r = i / D;
            uint c = i % D;
            uint global_r = kv_start + r;
            tg_K[i] = (global_r < seq_len) ? K[head_offset + global_r * D + c] : 0.0f;
        }}

        // Load V block
        for (uint i = lid; i < BC * D; i += {block_size}u) {{
            uint r = i / D;
            uint c = i % D;
            uint global_r = kv_start + r;
            tg_V[i] = (global_r < seq_len) ? V[head_offset + global_r * D + c] : 0.0f;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute S = Q @ K^T (Br x Bc) — each thread computes one element
        for (uint i = lid; i < BR * BC; i += {block_size}u) {{
            uint r = i / BC;
            uint c = i % BC;
            float dot = 0.0f;
            for (uint d = 0u; d < D; d++) {{
                dot += tg_Q[r * D + d] * tg_K[c * D + d];
            }}
            // Mask out-of-bounds KV positions (and causal mask if enabled)
            uint kv_pos = kv_start + c;{causal_mask}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // For each query row: update online softmax and accumulate output
        // Each thread handles one query row
        if (lid < BR) {{
            uint r = lid;
            float m_prev = tg_m[r];
            float l_prev = tg_l[r];

            // Row max of S[r, :]
            float m_new = m_prev;
            for (uint c = 0u; c < BC; c++) {{
                m_new = max(m_new, tg_S[r * BC + c]);
            }}

            // Compute P[r, :] = exp(S[r, :] - m_new) and sum
            float exp_scale = exp(m_prev - m_new);
            float l_new = l_prev * exp_scale;
            for (uint c = 0u; c < BC; c++) {{
                float p = exp(tg_S[r * BC + c] - m_new);
                tg_S[r * BC + c] = p;  // store P in place of S
                l_new += p;
            }}

            // Rescale existing O and accumulate P @ V
            for (uint d = 0u; d < D; d++) {{
                float o_val = tg_O[r * D + d] * exp_scale;
                for (uint c = 0u; c < BC; c++) {{
                    o_val += tg_S[r * BC + c] * tg_V[c * D + d];
                }}
                tg_O[r * D + d] = o_val;
            }}

            tg_m[r] = m_new;
            tg_l[r] = l_new;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // Final normalization: O = O / l
    for (uint i = lid; i < BR * D; i += {block_size}u) {{
        uint r = i / D;
        float l_val = tg_l[r];
        if (l_val > 0.0f) {{
            tg_O[i] /= l_val;
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write output
    for (uint i = lid; i < BR * D; i += {block_size}u) {{
        uint r = i / D;
        uint c = i % D;
        uint global_r = q_start + r;
        if (global_r < seq_len) {{
            O[head_offset + global_r * D + c] = tg_O[i];
        }}
    }}
}}
"""


def make_residual_add_kernel(block_size=256, dtype="fp32", has_bias=True):
    """Generate a fused residual connection kernel.

    output = input + residual + bias (or input + residual if has_bias=False)

    Common in transformer blocks after attention and FFN layers.

    Args:
        block_size: Threads per threadgroup.
        dtype: Data type.
        has_bias: Whether to include a bias term.
    """
    name = "residual_add_kernel"
    kb = KernelBuilder(name, block_size=block_size)

    kb.add_ptr_arg("input", dtype=dtype, const=True)
    kb.add_ptr_arg("residual", dtype=dtype, const=True)
    if has_bias:
        kb.add_ptr_arg("bias", dtype=dtype, const=True)
    kb.add_ptr_arg("output", dtype=dtype, const=False)
    kb.add_scalar_arg("n_elements", dtype="u32")

    offsets = kb.make_block_offsets("pid", "offsets")
    mask = kb.make_mask(offsets, "n_elements", "mask")

    in_val = kb.load("input", offsets, mask, out_var="in_val", dtype=dtype)
    res_val = kb.load("residual", offsets, mask, out_var="res_val", dtype=dtype)

    if has_bias:
        bias_val = kb.load("bias", offsets, mask, out_var="bias_val", dtype=dtype)
        kb._var("sum_val", f"{in_val} + {res_val} + {bias_val}", ty="float")
        kb.store("output", offsets, "sum_val", mask, dtype=dtype)
    else:
        kb._var("sum_val", f"{in_val} + {res_val}", ty="float")
        kb.store("output", offsets, "sum_val", mask, dtype=dtype)

    return kb.build()


def make_kv_cache_attention_kernel(head_dim=64, block_size=256):
    """Generate a KV-cache attention kernel for autoregressive inference.

    Single query token attending to a cached K,V sequence:
    score[j] = Q[0,:] . K[j,:] * scale
    attn = softmax(scores)
    output = sum(attn[j] * V[j,:])

    Each threadgroup handles one attention head.
    Iterates over the KV cache in chunks.

    Args:
        head_dim: Dimension of each attention head.
        block_size: Threads per threadgroup.

    Kernel args:
        Q: [n_heads, head_dim] — single query token per head
        K_cache: [n_heads, max_seq_len, head_dim] — key cache
        V_cache: [n_heads, max_seq_len, head_dim] — value cache
        O: [n_heads, head_dim] — output
        seq_len: current sequence length (how much of cache is valid)
        scale: 1/sqrt(head_dim)
    """
    return f"""#include <metal_stdlib>
using namespace metal;

kernel void kv_cache_attention(
    device const float* Q [[buffer(0)]],
    device const float* K_cache [[buffer(1)]],
    device const float* V_cache [[buffer(2)]],
    device float* O [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant float& scale [[buffer(5)]],
    uint pid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]]
) {{
    // pid = head index
    const uint D = {head_dim}u;
    uint head_offset_q = pid * D;
    uint head_offset_kv = pid * seq_len * D;

    // Shared memory for online softmax
    threadgroup float tg_scores[{block_size}];  // attention scores buffer

    // Phase 1: Compute all attention scores and find max
    // Each thread handles one or more KV positions
    float local_max = -INFINITY;
    for (uint j = lid; j < seq_len; j += {block_size}u) {{
        float dot = 0.0f;
        for (uint d = 0u; d < D; d++) {{
            dot += Q[head_offset_q + d] * K_cache[head_offset_kv + j * D + d];
        }}
        float score = dot * scale;
        tg_scores[j % {block_size}u] = score;  // partial storage
        local_max = max(local_max, score);
    }}

    // Reduce max across threadgroup
    float sg_max = simd_max(local_max);
    threadgroup float shared_max[{(block_size + 31) // 32}];
    if (sgitg == 0u) shared_max[tiisg] = -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0u) shared_max[sgitg] = sg_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rd_max = shared_max[tiisg];
    float global_max = simd_max(rd_max);

    // Phase 2: Compute exp(score - max) and sum
    float local_sum = 0.0f;
    for (uint j = lid; j < seq_len; j += {block_size}u) {{
        float dot = 0.0f;
        for (uint d = 0u; d < D; d++) {{
            dot += Q[head_offset_q + d] * K_cache[head_offset_kv + j * D + d];
        }}
        float score = dot * scale;
        float p = exp(score - global_max);
        local_sum += p;
    }}

    // Reduce sum
    float sg_sum = simd_sum(local_sum);
    threadgroup float shared_sum[{(block_size + 31) // 32}];
    if (sgitg == 0u) shared_sum[tiisg] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0u) shared_sum[sgitg] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rd_sum = shared_sum[tiisg];
    float global_sum = simd_sum(rd_sum);
    float inv_sum = 1.0f / global_sum;

    // Phase 3: Compute weighted V sum
    // Each thread accumulates over its KV positions for all D dimensions
    // To avoid excessive register pressure, process D in chunks
    for (uint d_start = 0u; d_start < D; d_start += {block_size}u) {{
        float o_val = 0.0f;
        uint d = d_start + lid;
        if (d < D) {{
            for (uint j = 0u; j < seq_len; j++) {{
                float dot = 0.0f;
                for (uint dd = 0u; dd < D; dd++) {{
                    dot += Q[head_offset_q + dd] * K_cache[head_offset_kv + j * D + dd];
                }}
                float score = dot * scale;
                float attn_weight = exp(score - global_max) * inv_sum;
                o_val += attn_weight * V_cache[head_offset_kv + j * D + d];
            }}
            O[head_offset_q + d] = o_val;
        }}
    }}
}}
"""


def make_simdgroup_matmul_kernel(dtype="fp32"):
    """Generate a matmul kernel using Apple's simdgroup_matrix hardware.

    C[M,N] = A[M,K] * B[K,N]

    Uses simdgroup_matrix for hardware-accelerated 8x8 matrix
    multiply-accumulate. Each threadgroup (128 threads = 4 SIMD groups)
    computes a 32x32 output tile. Data is staged through threadgroup memory.

    Dispatch: block_size=128, n_groups = ceil(M/32) * ceil(N/32).
    Requires M, N to be multiples of 32 (no boundary masking on simdgroup_store).

    Args:
        dtype: "fp32" or "fp16". FP16 uses half inputs/outputs with float accumulation.
    """
    if dtype in ("fp32", "f32"):
        elem_type = "float"
        tg_type = "float"
        frag_type = "simdgroup_float8x8"
        zero = "0.0f"
        cast_load = "float"
        cast_store = ""
    elif dtype in ("fp16", "f16"):
        elem_type = "half"
        tg_type = "float"  # stage through float for precision
        frag_type = "simdgroup_float8x8"
        zero = "0.0f"
        cast_load = "float"
        cast_store = ""  # accumulator is float, output is float
    else:
        raise ValueError(f"simdgroup_matmul supports fp32 and fp16, got {dtype}")

    # FP16: half inputs, float accumulation, float output
    # (simdgroup_store with float accumulators requires float* destination)
    if dtype in ("fp16", "f16"):
        return f"""#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void simdgroup_matmul(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint pid [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiitg [[thread_index_in_threadgroup]]
) {{
    uint n_tile_cols = (N + 31u) / 32u;
    uint tile_row = pid / n_tile_cols;
    uint tile_col = pid % n_tile_cols;
    uint row_base = tile_row * 32u;
    uint col_base = tile_col * 32u + sgitg * 8u;

    simdgroup_float8x8 acc0(0), acc1(0), acc2(0), acc3(0);
    simdgroup_float8x8 a_frag, b_frag;

    threadgroup float tg_A[32 * 8];
    threadgroup float tg_B[8 * 32];

    for (uint k = 0u; k < K; k += 8u) {{
        for (uint i = tiitg; i < 256u; i += 128u) {{
            uint r = i / 8u, c = i % 8u;
            uint gr = row_base + r, gc = k + c;
            tg_A[i] = (gr < M && gc < K) ? float(A[gr * K + gc]) : 0.0f;
        }}
        uint col_base_tg = tile_col * 32u;
        for (uint i = tiitg; i < 256u; i += 128u) {{
            uint r = i / 32u, c = i % 32u;
            uint gr = k + r, gc = col_base_tg + c;
            tg_B[i] = (gr < K && gc < N) ? float(B[gr * N + gc]) : 0.0f;
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_load(b_frag, tg_B + sgitg * 8u, 32);

        simdgroup_load(a_frag, tg_A, 8);
        simdgroup_multiply_accumulate(acc0, a_frag, b_frag, acc0);

        simdgroup_load(a_frag, tg_A + 64u, 8);
        simdgroup_multiply_accumulate(acc1, a_frag, b_frag, acc1);

        simdgroup_load(a_frag, tg_A + 128u, 8);
        simdgroup_multiply_accumulate(acc2, a_frag, b_frag, acc2);

        simdgroup_load(a_frag, tg_A + 192u, 8);
        simdgroup_multiply_accumulate(acc3, a_frag, b_frag, acc3);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    simdgroup_store(acc0, C + (row_base) * N + col_base, N);
    simdgroup_store(acc1, C + (row_base + 8u) * N + col_base, N);
    simdgroup_store(acc2, C + (row_base + 16u) * N + col_base, N);
    simdgroup_store(acc3, C + (row_base + 24u) * N + col_base, N);
}}
"""

    # FP32 path
    return f"""#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void simdgroup_matmul(
    device const {elem_type}* A [[buffer(0)]],
    device const {elem_type}* B [[buffer(1)]],
    device {elem_type}* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint pid [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiitg [[thread_index_in_threadgroup]]
) {{
    uint n_tile_cols = (N + 31u) / 32u;
    uint tile_row = pid / n_tile_cols;
    uint tile_col = pid % n_tile_cols;
    uint row_base = tile_row * 32u;
    uint col_base = tile_col * 32u + sgitg * 8u;

    {frag_type} acc0(0), acc1(0), acc2(0), acc3(0);
    {frag_type} a_frag, b_frag;

    threadgroup {tg_type} tg_A[32 * 8];
    threadgroup {tg_type} tg_B[8 * 32];

    for (uint k = 0u; k < K; k += 8u) {{
        for (uint i = tiitg; i < 256u; i += 128u) {{
            uint r = i / 8u, c = i % 8u;
            uint gr = row_base + r, gc = k + c;
            tg_A[i] = (gr < M && gc < K) ? {cast_load}(A[gr * K + gc]) : {zero};
        }}
        uint col_base_tg = tile_col * 32u;
        for (uint i = tiitg; i < 256u; i += 128u) {{
            uint r = i / 32u, c = i % 32u;
            uint gr = k + r, gc = col_base_tg + c;
            tg_B[i] = (gr < K && gc < N) ? {cast_load}(B[gr * N + gc]) : {zero};
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_load(b_frag, tg_B + sgitg * 8u, 32);

        simdgroup_load(a_frag, tg_A, 8);
        simdgroup_multiply_accumulate(acc0, a_frag, b_frag, acc0);

        simdgroup_load(a_frag, tg_A + 64u, 8);
        simdgroup_multiply_accumulate(acc1, a_frag, b_frag, acc1);

        simdgroup_load(a_frag, tg_A + 128u, 8);
        simdgroup_multiply_accumulate(acc2, a_frag, b_frag, acc2);

        simdgroup_load(a_frag, tg_A + 192u, 8);
        simdgroup_multiply_accumulate(acc3, a_frag, b_frag, acc3);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    simdgroup_store(acc0, C + (row_base) * N + col_base, N);
    simdgroup_store(acc1, C + (row_base + 8u) * N + col_base, N);
    simdgroup_store(acc2, C + (row_base + 16u) * N + col_base, N);
    simdgroup_store(acc3, C + (row_base + 24u) * N + col_base, N);
}}
"""


# ---------------------------------------------------------------------------
def make_fused_linear_kernel(has_bias=True):
    """Generate a fused linear layer kernel: output = input @ weight^T + bias.

    Uses simdgroup_matrix for hardware-accelerated matmul with optional
    bias addition fused in. Each threadgroup computes a 32x32 output tile.

    This is the most common operation in transformers — used for all
    attention projections (Q, K, V, O) and FFN layers.

    Layout:
        input:  [M, K]
        weight: [N, K] (stored as row-major, transposed during compute)
        bias:   [N] (optional, broadcast across M dimension)
        output: [M, N]

    Dispatch: block_size=128, n_groups = ceil(M/32) * ceil(N/32).

    Args:
        has_bias: Whether to include bias addition.
    """
    bias_buffer = ""
    bias_param = ""
    bias_add = ""

    if has_bias:
        bias_param = "    device const float* bias [[buffer(3)]],\n"
        bias_buffer = ""
        bias_add = """
    // Add bias
    for (uint i = tiitg; i < 32u; i += 128u) {
        uint r = i / 32u;
        uint c = i % 32u;
        uint gc = col_base + c;
        // Bias is broadcast along M dimension — load once per column
        if (gc < N) {
            // For each row in the tile that this thread handles
            for (uint rr = 0u; rr < 32u; rr++) {
                uint gr = row_base + rr;
                if (gr < M) {
                    C[gr * N + gc] += bias[gc];
                }
            }
        }
    }"""
    else:
        bias_param = ""
        bias_add = ""

    # Buffer indices shift based on whether bias is present
    if has_bias:
        m_idx, n_idx, k_idx = 4, 5, 6
    else:
        m_idx, n_idx, k_idx = 3, 4, 5

    return f"""#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void fused_linear(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* C [[buffer(2)]],
{bias_param}    constant uint& M [[buffer({m_idx})]],
    constant uint& N [[buffer({n_idx})]],
    constant uint& K [[buffer({k_idx})]],
    uint pid [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiitg [[thread_index_in_threadgroup]]
) {{
    uint n_tile_cols = (N + 31u) / 32u;
    uint tile_row = pid / n_tile_cols;
    uint tile_col = pid % n_tile_cols;
    uint row_base = tile_row * 32u;
    uint col_base = tile_col * 32u + sgitg * 8u;

    simdgroup_float8x8 acc0(0), acc1(0), acc2(0), acc3(0);
    simdgroup_float8x8 a_frag, b_frag;

    threadgroup float tg_A[32 * 8];
    threadgroup float tg_B[8 * 32];

    for (uint k = 0u; k < K; k += 8u) {{
        // Load input tile: input[row_base:row_base+32, k:k+8]
        for (uint i = tiitg; i < 256u; i += 128u) {{
            uint r = i / 8u, c = i % 8u;
            uint gr = row_base + r, gc = k + c;
            tg_A[i] = (gr < M && gc < K) ? input[gr * K + gc] : 0.0f;
        }}
        // Load weight tile transposed: weight[col_base:col_base+32, k:k+8]
        // weight is [N, K], we want W^T, so we load weight[n, k] as B[k, n]
        uint col_base_tg = tile_col * 32u;
        for (uint i = tiitg; i < 256u; i += 128u) {{
            uint r = i / 32u, c = i % 32u;  // r is the K index, c is the N index
            uint gk = k + r, gn = col_base_tg + c;
            tg_B[i] = (gk < K && gn < N) ? weight[gn * K + gk] : 0.0f;
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_load(b_frag, tg_B + sgitg * 8u, 32);

        simdgroup_load(a_frag, tg_A, 8);
        simdgroup_multiply_accumulate(acc0, a_frag, b_frag, acc0);

        simdgroup_load(a_frag, tg_A + 64u, 8);
        simdgroup_multiply_accumulate(acc1, a_frag, b_frag, acc1);

        simdgroup_load(a_frag, tg_A + 128u, 8);
        simdgroup_multiply_accumulate(acc2, a_frag, b_frag, acc2);

        simdgroup_load(a_frag, tg_A + 192u, 8);
        simdgroup_multiply_accumulate(acc3, a_frag, b_frag, acc3);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    simdgroup_store(acc0, C + (row_base) * N + col_base, N);
    simdgroup_store(acc1, C + (row_base + 8u) * N + col_base, N);
    simdgroup_store(acc2, C + (row_base + 16u) * N + col_base, N);
    simdgroup_store(acc3, C + (row_base + 24u) * N + col_base, N);
{bias_add}
}}
"""


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
    from triton_metal.codegen.ttgir_parser import parse_ttgir

    ir_text = str(mod)
    kernel_name = _extract_kernel_name(ir_text)
    metadata["name"] = kernel_name

    kb = parse_ttgir(ir_text, options)
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
