"""Parse TTGIR (TritonGPU IR) MLIR text into a KernelBuilder.

This module translates Triton's GPU-level MLIR representation into
Metal compute kernel descriptions that can be emitted as MSL.

Supported TTGIR operations:
- tt.get_program_id → threadgroup_position_in_grid
- tt.make_range, tt.splat, arith.addi → block offsets
- arith.cmpi slt → bounds mask
- tt.addptr → pointer arithmetic
- tt.load / tt.store → masked buffer read/write
- arith.addf/subf/mulf/divf → binary float ops
- arith.addi/subi/muli → binary int ops
- math.exp/log/sqrt/abs → unary ops
- tt.reduce → reductions (sum via arith.addf, max via arith.maxf)

The parser is text-based (using str(module)) since Triton's Python
MLIR bindings don't expose a structured walk API.
"""

import re
from collections import OrderedDict

from triton_metal.codegen.msl_emitter import KernelBuilder


# ---------------------------------------------------------------------------
# MLIR type mapping
# ---------------------------------------------------------------------------

def _mlir_type_to_triton_dtype(mlir_type):
    """Convert MLIR type string to Triton dtype string.

    Raises TypeError if the type is FP64 (not supported on Apple Silicon).
    """
    mlir_type = mlir_type.strip()
    if mlir_type == "f64":
        raise TypeError(
            "FP64 (double) is not supported on Apple Silicon GPUs. "
            "Got MLIR type 'f64'. Cast to float32 before running on Metal."
        )
    _map = {
        "f32": "fp32",
        "f16": "fp16",
        "bf16": "bf16",
        "i1": "i1",
        "i8": "i8",
        "i16": "i16",
        "i32": "i32",
        "i64": "i64",
    }
    return _map.get(mlir_type, "fp32")


def _extract_scalar_type(type_str):
    """Extract the scalar element type from a tensor or pointer type.

    Examples:
        'tensor<256xf32>' -> 'f32'
        'tensor<256xf32, #layout>' -> 'f32'
        '!tt.ptr<f32>' -> 'f32'
        'f32' -> 'f32'
        'i32' -> 'i32'
    """
    # Pointer type: !tt.ptr<f32>
    m = re.search(r"!tt\.ptr<(\w+)>", type_str)
    if m:
        return m.group(1)
    # Tensor type: tensor<...xTYPE> or tensor<...xTYPE, #layout>
    m = re.search(r"tensor<[^>]*x(\w+)(?:,\s*#\w+)?>", type_str)
    if m:
        return m.group(1)
    # Scalar type
    m = re.match(r"^(\w+)$", type_str.strip())
    if m:
        return m.group(1)
    return "f32"


def _extract_block_size(ir_text):
    """Extract the block size from tt.make_range op.

    Looks for: tt.make_range {end = N : i32, start = 0 : i32}
    """
    m = re.search(r"tt\.make_range\s*\{end\s*=\s*(\d+)\s*:\s*i32\s*,\s*start\s*=\s*0\s*:\s*i32\}", ir_text)
    if m:
        return int(m.group(1))
    return 256  # default


# ---------------------------------------------------------------------------
# TTGIR Parser
# ---------------------------------------------------------------------------

class TTGIRParser:
    """Parse TTGIR MLIR text and build a KernelBuilder.

    The parser processes the MLIR line by line, tracking SSA values
    and their roles (pointer, offset, mask, data), then emits
    corresponding KernelBuilder operations.
    """

    def __init__(self, ir_text, options):
        self.ir_text = ir_text
        self.options = options
        self.lines = ir_text.strip().split("\n")

        # SSA value tracking
        self.ssa_values = {}      # %name -> role info
        self.ssa_types = {}       # %name -> MLIR type string
        self.ptr_args = OrderedDict()    # arg_name -> (index, dtype, is_output)
        self.scalar_args = OrderedDict() # arg_name -> (index, dtype)
        self.kernel_name = "triton_kernel"
        self.block_size = _extract_block_size(ir_text)

        # Track which SSA values map to which concepts
        self.program_id_var = None
        self.offsets_var = None
        self.mask_var = None
        self.loaded_values = {}  # %ssa -> (ptr_arg_name, msl_var_name)
        self.computed_values = {} # %ssa -> msl_var_name

        # Reduction tracking
        self.reduce_ops = []  # [(result_ssa, input_ssa, op_kind, axis)]

        # Matmul tracking
        self.dot_ops = []  # [(result_ssa, lhs_ssa, rhs_ssa, acc_ssa)]

        # Loop tracking
        self.scf_for_loops = []  # [(lb_ssa, ub_ssa, step_ssa, body_lines)]

        # Operation buffer for the kernel builder
        self.ops = []

    def parse(self):
        """Parse the IR text and return a KernelBuilder."""
        self._parse_function_signature()
        self._parse_body()
        return self._build_kernel()

    def _parse_function_signature(self):
        """Extract kernel name and argument types from the function signature."""
        # Match: tt.func @name(%arg0: TYPE, %arg1: TYPE, ...)
        # or: tt.func public @name(%arg0: TYPE, %arg1: TYPE, ...)
        sig_match = re.search(
            r"tt\.func\s+(?:public\s+)?@(\w+)\s*\(([^)]*)\)",
            self.ir_text, re.DOTALL
        )
        if not sig_match:
            sig_match = re.search(
                r"func\.func\s+@(\w+)\s*\(([^)]*)\)",
                self.ir_text, re.DOTALL
            )
        if not sig_match:
            return

        self.kernel_name = sig_match.group(1)
        args_text = sig_match.group(2)

        # Parse each argument
        # Format: %argN: TYPE {optional attributes}
        arg_pattern = re.compile(
            r"%(\w+)\s*:\s*([^,{}]+(?:\{[^}]*\})?)"
        )
        for i, match in enumerate(arg_pattern.finditer(args_text)):
            arg_name = match.group(1)
            arg_type = match.group(2).strip()

            # Remove attributes like {tt.divisibility = 16 : i32}
            arg_type = re.sub(r"\s*\{[^}]*\}", "", arg_type).strip()

            self.ssa_types[f"%{arg_name}"] = arg_type

            if "!tt.ptr" in arg_type:
                elem_type = _extract_scalar_type(arg_type)
                dtype = _mlir_type_to_triton_dtype(elem_type)
                self.ptr_args[arg_name] = (i, dtype, False)
            else:
                elem_type = arg_type.strip()
                dtype = _mlir_type_to_triton_dtype(elem_type)
                self.scalar_args[arg_name] = (i, dtype)

    def _scan_scf_for_loops(self):
        """Scan for scf.for loop blocks in the IR text.

        scf.for has the form:
            %result = scf.for %iv = %lb to %ub step %step
                      iter_args(%acc = %init) -> (type) {
              ...body...
              scf.yield %new_acc : type
            }
        or without iter_args:
            scf.for %iv = %lb to %ub step %step {
              ...body...
            }
        """
        # Match scf.for with iter_args
        loop_pattern = re.compile(
            r'(?:%(\w+)(?::\d+)?\s*=\s*)?'  # optional result SSA
            r'scf\.for\s+%(\w+)\s*=\s*(%\w+)\s+to\s+(%\w+)\s+step\s+(%\w+)'
            r'(?:\s+iter_args\(([^)]*)\))?'  # optional iter_args
        )
        for m in loop_pattern.finditer(self.ir_text):
            result_ssa = f"%{m.group(1)}" if m.group(1) else None
            iv_name = m.group(2)
            lb_ssa = m.group(3)
            ub_ssa = m.group(4)
            step_ssa = m.group(5)
            iter_args_str = m.group(6)

            # Parse iter_args if present
            iter_args = []
            if iter_args_str:
                for ia_match in re.finditer(r'%(\w+)\s*=\s*(%\w+)', iter_args_str):
                    iter_args.append((ia_match.group(1), ia_match.group(2)))

            self.scf_for_loops.append({
                'result_ssa': result_ssa,
                'iv': iv_name,
                'lb': lb_ssa,
                'ub': ub_ssa,
                'step': step_ssa,
                'iter_args': iter_args,
            })

            # Record the loop variable in SSA tracking
            self.ssa_values[f"%{iv_name}"] = ("loop_iv", lb_ssa, ub_ssa, step_ssa)

    def _scan_reductions(self):
        """Scan for tt.reduce multi-line blocks in the IR text.

        tt.reduce has a nested region:
            %result = "tt.reduce"(%input) ({
            ^bb0(%a: f32, %b: f32):
              %combined = arith.addf %a, %b : f32
              "tt.reduce.return"(%combined) : (f32) -> ()
            }) {axis = 0 : i32} : (tensor<...>) -> f32
        """
        # Match the reduce block: result = "tt.reduce"(input) ({ ... body ... })
        reduce_pattern = re.compile(
            r'%(\w+)\s*=\s*"tt\.reduce"\s*\((%\w+)\)\s*\(\{'
            r'(.*?)'
            r'\}\)\s*\{axis\s*=\s*(\d+)',
            re.DOTALL
        )
        for m in reduce_pattern.finditer(self.ir_text):
            result_ssa = f"%{m.group(1)}"
            input_ssa = m.group(2)
            body = m.group(3)
            axis = int(m.group(4))

            # Detect the combine operation from the body
            if "arith.addf" in body:
                op_kind = "sum"
            elif "arith.maxf" in body or "arith.maxnumf" in body:
                op_kind = "max"
            elif "arith.minf" in body or "arith.minnumf" in body:
                op_kind = "min"
            else:
                op_kind = "sum"  # fallback

            self.reduce_ops.append((result_ssa, input_ssa, op_kind, axis))
            self.ssa_values[result_ssa] = ("reduce", input_ssa, op_kind)

    def _scan_dot_ops(self):
        """Scan for tt.dot operations (matrix multiply-accumulate).

        tt.dot has the form:
            %result = tt.dot %lhs, %rhs, %acc {options} : type * type -> type
        or:
            %result = "tt.dot"(%lhs, %rhs, %acc) {options} : (types) -> type
        """
        # Quoted form
        dot_pattern1 = re.compile(
            r'%(\w+)\s*=\s*"tt\.dot"\s*\((%\w+)\s*,\s*(%\w+)\s*,\s*(%\w+)\)'
        )
        # Unquoted form
        dot_pattern2 = re.compile(
            r'%(\w+)\s*=\s*tt\.dot\s+(%\w+)\s*,\s*(%\w+)\s*,\s*(%\w+)'
        )
        for pattern in [dot_pattern1, dot_pattern2]:
            for m in pattern.finditer(self.ir_text):
                result_ssa = f"%{m.group(1)}"
                lhs_ssa = m.group(2)
                rhs_ssa = m.group(3)
                acc_ssa = m.group(4)
                self.dot_ops.append((result_ssa, lhs_ssa, rhs_ssa, acc_ssa))
                self.ssa_values[result_ssa] = ("dot", lhs_ssa, rhs_ssa, acc_ssa)

    def _parse_body(self):
        """Walk through the body and classify operations."""
        # First scan for multi-line blocks (reduce, scf.for, dot)
        self._scan_scf_for_loops()
        self._scan_reductions()
        self._scan_dot_ops()

        for line in self.lines:
            line = line.strip()
            if not line or line.startswith("//") or line.startswith("module"):
                continue

            # tt.get_program_id
            if "tt.get_program_id" in line:
                m = re.match(r"%(\w+)\s*=\s*tt\.get_program_id\s+(\w+)", line)
                if m:
                    self.program_id_var = f"%{m.group(1)}"
                    self.ssa_values[self.program_id_var] = ("program_id", m.group(2))
                continue

            # tt.make_range
            if "tt.make_range" in line:
                m = re.match(r"%(\w+)\s*=\s*tt\.make_range\s*\{end\s*=\s*(\d+)", line)
                if m:
                    self.ssa_values[f"%{m.group(1)}"] = ("range", int(m.group(2)))
                continue

            # arith.constant
            if "arith.constant" in line:
                m = re.match(r"%(\w+)\s*=\s*arith\.constant\s+(.+)", line)
                if m:
                    val_name = f"%{m.group(1)}"
                    val_str = m.group(2).strip()
                    self.ssa_values[val_name] = ("constant", val_str)
                continue

            # tt.splat (broadcast scalar to tensor)
            if "tt.splat" in line:
                m = re.match(r"%(\w+)\s*=\s*tt\.splat\s+(%\w+)", line)
                if m:
                    result = f"%{m.group(1)}"
                    source = m.group(2)
                    self.ssa_values[result] = ("splat", source)
                continue

            # arith.addi (integer add — used for offset computation)
            if "arith.addi" in line:
                m = re.match(r"%(\w+)\s*=\s*arith\.addi\s+(%\w+)\s*,\s*(%\w+)", line)
                if m:
                    result = f"%{m.group(1)}"
                    lhs, rhs = m.group(2), m.group(3)
                    self.ssa_values[result] = ("addi", lhs, rhs)
                continue

            # arith.muli (integer mul — used for pid * BLOCK_SIZE)
            if "arith.muli" in line:
                m = re.match(r"%(\w+)\s*=\s*arith\.muli\s+(%\w+)\s*,\s*(%\w+)", line)
                if m:
                    result = f"%{m.group(1)}"
                    lhs, rhs = m.group(2), m.group(3)
                    self.ssa_values[result] = ("muli", lhs, rhs)
                continue

            # arith.subi (integer sub)
            if "arith.subi" in line:
                m = re.match(r"%(\w+)\s*=\s*arith\.subi\s+(%\w+)\s*,\s*(%\w+)", line)
                if m:
                    result = f"%{m.group(1)}"
                    self.ssa_values[result] = ("subi", m.group(2), m.group(3))
                continue

            # arith.cmpi (all predicates — used for mask generation and conditionals)
            if "arith.cmpi" in line:
                m = re.match(r"%(\w+)\s*=\s*arith\.cmpi\s+(\w+)\s*,\s*(%\w+)\s*,\s*(%\w+)", line)
                if m:
                    result = f"%{m.group(1)}"
                    pred = m.group(2)  # slt, sle, sgt, sge, eq, ne, ult, ule, ugt, uge
                    self.mask_var = result
                    self.ssa_values[result] = ("mask", m.group(3), m.group(4))
                continue

            # arith.cmpf (float comparison — used for activation conditionals)
            if "arith.cmpf" in line:
                m = re.match(r"%(\w+)\s*=\s*arith\.cmpf\s+(\w+)\s*,\s*(%\w+)\s*,\s*(%\w+)", line)
                if m:
                    result = f"%{m.group(1)}"
                    self.ssa_values[result] = ("mask", m.group(3), m.group(4))
                continue

            # arith.select (ternary: cond ? true_val : false_val)
            if "arith.select" in line:
                m = re.match(r"%(\w+)\s*=\s*arith\.select\s+(%\w+)\s*,\s*(%\w+)\s*,\s*(%\w+)", line)
                if m:
                    result = f"%{m.group(1)}"
                    self.ssa_values[result] = ("select", m.group(2), m.group(3), m.group(4))
                continue

            # arith.maxf / arith.minf (float max/min — not inside tt.reduce)
            if "arith.maxf" in line and "tt.reduce" not in line:
                m = re.match(r"%(\w+)\s*=\s*arith\.maxf\s+(%\w+)\s*,\s*(%\w+)", line)
                if m:
                    result = f"%{m.group(1)}"
                    self.ssa_values[result] = ("fmax", m.group(2), m.group(3))
                continue

            if "arith.minf" in line and "tt.reduce" not in line:
                m = re.match(r"%(\w+)\s*=\s*arith\.minf\s+(%\w+)\s*,\s*(%\w+)", line)
                if m:
                    result = f"%{m.group(1)}"
                    self.ssa_values[result] = ("fmin", m.group(2), m.group(3))
                continue

            # arith.negf (float negate)
            if "arith.negf" in line:
                m = re.match(r"%(\w+)\s*=\s*arith\.negf\s+(%\w+)", line)
                if m:
                    result = f"%{m.group(1)}"
                    self.ssa_values[result] = ("neg", m.group(2))
                continue

            # arith.sitofp / arith.uitofp (int to float conversion)
            if "arith.sitofp" in line or "arith.uitofp" in line:
                m = re.match(r"%(\w+)\s*=\s*arith\.\w+tofp\s+(%\w+)", line)
                if m:
                    result = f"%{m.group(1)}"
                    self.ssa_values[result] = ("passthrough", m.group(2))
                continue

            # arith.fptosi / arith.fptoui (float to int conversion)
            if "arith.fptosi" in line or "arith.fptoui" in line:
                m = re.match(r"%(\w+)\s*=\s*arith\.fpto\w+\s+(%\w+)", line)
                if m:
                    result = f"%{m.group(1)}"
                    self.ssa_values[result] = ("passthrough", m.group(2))
                continue

            # tt.addptr (pointer + offset)
            if "tt.addptr" in line:
                m = re.match(r"%(\w+)\s*=\s*tt\.addptr\s+(%\w+)\s*,\s*(%\w+)", line)
                if m:
                    result = f"%{m.group(1)}"
                    self.ssa_values[result] = ("addptr", m.group(2), m.group(3))
                continue

            # tt.load
            if "tt.load" in line and "=" in line:
                m = re.match(r"%(\w+)\s*=\s*tt\.load\s+(%\w+)", line)
                if m:
                    result = f"%{m.group(1)}"
                    ptr_ssa = m.group(2)
                    # Determine if there's a mask
                    has_mask = "," in line.split("tt.load")[1].split(":")[0]
                    self.ssa_values[result] = ("load", ptr_ssa, has_mask)
                continue

            # tt.store
            if "tt.store" in line:
                m = re.match(r"tt\.store\s+(%\w+)\s*,\s*(%\w+)", line)
                if m:
                    ptr_ssa = m.group(1)
                    val_ssa = m.group(2)
                    has_mask = line.count(",") >= 2
                    self.ops.append(("store", ptr_ssa, val_ssa, has_mask))
                continue

            # Binary float ops: arith.addf, arith.subf, arith.mulf, arith.divf
            for op_name, op_key in [("addf", "add"), ("subf", "sub"),
                                     ("mulf", "mul"), ("divf", "div")]:
                if f"arith.{op_name}" in line:
                    m = re.match(
                        rf"%(\w+)\s*=\s*arith\.{op_name}\s+(%\w+)\s*,\s*(%\w+)",
                        line
                    )
                    if m:
                        result = f"%{m.group(1)}"
                        self.ssa_values[result] = (op_key, m.group(2), m.group(3))
                    break

            # math.fma (fused multiply-add: a*b + c)
            if "math.fma" in line:
                m = re.match(
                    r"%(\w+)\s*=\s*math\.fma\s+(%\w+)\s*,\s*(%\w+)\s*,\s*(%\w+)",
                    line
                )
                if m:
                    result = f"%{m.group(1)}"
                    self.ssa_values[result] = ("fma", m.group(2), m.group(3), m.group(4))
                continue

            # Unary math ops
            for op_name, op_key in [("math.exp", "exp"), ("math.log", "log"),
                                     ("math.sqrt", "sqrt"), ("math.rsqrt", "rsqrt"),
                                     ("math.absf", "abs"), ("math.sin", "sin"),
                                     ("math.cos", "cos"), ("math.tanh", "tanh")]:
                if op_name in line:
                    m = re.match(
                        rf"%(\w+)\s*=\s*{re.escape(op_name)}\s+(%\w+)",
                        line
                    )
                    if m:
                        result = f"%{m.group(1)}"
                        self.ssa_values[result] = (op_key, m.group(2))
                    break

            # arith.extf (fp16 -> fp32 promotion)
            if "arith.extf" in line:
                m = re.match(r"%(\w+)\s*=\s*arith\.extf\s+(%\w+)", line)
                if m:
                    result = f"%{m.group(1)}"
                    self.ssa_values[result] = ("passthrough", m.group(2))
                continue

            # arith.truncf (fp32 -> fp16 demotion)
            if "arith.truncf" in line:
                m = re.match(r"%(\w+)\s*=\s*arith\.truncf\s+(%\w+)", line)
                if m:
                    result = f"%{m.group(1)}"
                    self.ssa_values[result] = ("passthrough", m.group(2))
                continue

            # arith.extsi / arith.trunci (integer extension/truncation)
            if "arith.extsi" in line or "arith.trunci" in line:
                m = re.match(r"%(\w+)\s*=\s*arith\.(?:extsi|trunci)\s+(%\w+)", line)
                if m:
                    result = f"%{m.group(1)}"
                    self.ssa_values[result] = ("int_cast", m.group(2))
                continue

            # tt.reduce (sum/max/min reduction)
            if '"tt.reduce"' in line or "tt.reduce" in line:
                # Reductions are multi-line; we handle them by detecting
                # the pattern in the combined block
                pass

    def _trace_to_ptr_arg(self, ssa):
        """Follow SSA chain to find the original pointer argument name."""
        seen = set()
        current = ssa
        while current and current not in seen:
            seen.add(current)
            val = self.ssa_values.get(current)
            if val is None:
                # Check if it's a function argument
                arg_name = current.lstrip("%")
                if arg_name in self.ptr_args:
                    return arg_name
                return None
            if val[0] == "splat":
                current = val[1]
            elif val[0] == "addptr":
                current = val[1]  # follow the pointer operand
            else:
                return None
        return None

    def _trace_to_scalar_arg(self, ssa):
        """Follow SSA chain to find the original scalar argument."""
        seen = set()
        current = ssa
        while current and current not in seen:
            seen.add(current)
            val = self.ssa_values.get(current)
            if val is None:
                arg_name = current.lstrip("%")
                if arg_name in self.scalar_args:
                    return arg_name
                return None
            if val[0] == "splat":
                current = val[1]
            else:
                return None
        return None

    def _classify_stores(self):
        """Determine which pointer args are outputs (have tt.store)."""
        for op in self.ops:
            if op[0] == "store":
                ptr_ssa = op[1]
                arg_name = self._trace_to_ptr_arg(ptr_ssa)
                if arg_name and arg_name in self.ptr_args:
                    idx, dtype, _ = self.ptr_args[arg_name]
                    self.ptr_args[arg_name] = (idx, dtype, True)

    def _build_kernel(self):
        """Construct a KernelBuilder from parsed IR."""
        self._classify_stores()

        num_warps = self.options.num_warps
        threads_per_tg = num_warps * 32
        block_size = max(self.block_size, threads_per_tg)

        kb = KernelBuilder(self.kernel_name, block_size=block_size)

        # Register pointer arguments in original order
        arg_msl_names = {}
        for arg_name, (idx, dtype, is_output) in self.ptr_args.items():
            msl_name = arg_name
            kb.add_ptr_arg(msl_name, dtype=dtype, const=(not is_output))
            arg_msl_names[arg_name] = msl_name

        # Register scalar arguments
        for arg_name, (idx, dtype) in self.scalar_args.items():
            msl_name = arg_name
            kb.add_scalar_arg(msl_name, dtype=dtype)
            arg_msl_names[arg_name] = msl_name

        # Determine the primary element dtype from the first pointer arg
        primary_dtype = "fp32"
        for arg_name, (idx, dtype, is_output) in self.ptr_args.items():
            if not is_output:
                primary_dtype = dtype
                break

        # Find the n_elements argument (usually the last scalar arg)
        n_arg = None
        for arg_name in self.scalar_args:
            n_arg = arg_name

        # Check if this is flash attention (2 dot ops + reduce/exp)
        if len(self.dot_ops) >= 2 and self._is_flash_attention_pattern():
            return self._build_flash_attention_kernel(kb, primary_dtype)

        # Quantized matmul: dot ops with integer cast (INT8/INT4 weights)
        if self.dot_ops and self._is_quantized_matmul_pattern():
            return self._build_quantized_matmul_kernel(kb, primary_dtype)

        # Check if this is a matmul (tt.dot detected)
        if self.dot_ops:
            return self._build_matmul_kernel(kb, primary_dtype)

        # Check if this is a multi-reduce pattern
        # Cross-entropy: softmax pattern (max+sum) + log + another sum reduce
        if len(self.reduce_ops) >= 3 and self._is_cross_entropy_pattern():
            return self._build_cross_entropy_kernel(kb, n_arg, primary_dtype)
        # Paged attention: exp + mul + reductions with 3+ input ptrs
        if self.reduce_ops and self._is_paged_attention_pattern():
            return self._build_paged_attention_kernel(kb, primary_dtype)
        # Beam search: cmp + add + reductions with 2+ output ptrs
        if self.reduce_ops and self._is_beam_search_pattern():
            return self._build_beam_search_kernel(kb, primary_dtype)
        # Online softmax: reduce in scf.for loop (single-pass streaming)
        if self.reduce_ops and self._is_online_softmax_pattern():
            return self._build_online_softmax_kernel(kb, primary_dtype)
        # Softmax: max + sum reductions
        if len(self.reduce_ops) >= 2 and self._is_softmax_pattern():
            return self._build_softmax_kernel(kb, n_arg, primary_dtype)
        # Fused residual + layer norm: layer norm pattern with arith.addf before first reduce
        if len(self.reduce_ops) >= 2 and self._is_fused_residual_norm_pattern():
            return self._build_fused_residual_norm_kernel(kb, n_arg, primary_dtype)
        # Variance: 2 sum reductions with sub+mul but no rsqrt (no normalization)
        if len(self.reduce_ops) >= 2 and self._is_variance_pattern():
            return self._build_variance_kernel(kb, n_arg, primary_dtype)
        # Standard layer norm
        if len(self.reduce_ops) >= 2 and self._is_layer_norm_pattern():
            return self._build_layer_norm_kernel(kb, n_arg, primary_dtype)
        # RMS norm: single sum reduction + rsqrt, no sub (no mean subtraction)
        if self.reduce_ops and self._is_rms_norm_pattern():
            return self._build_rms_norm_kernel(kb, n_arg, primary_dtype)
        # Group norm: reductions with 3+ input ptrs (input + weight + bias)
        if self.reduce_ops and self._is_group_norm_pattern():
            return self._build_group_norm_kernel(kb, primary_dtype)
        if self.reduce_ops:
            return self._build_reduction_kernel(kb, n_arg, primary_dtype)

        # RoPE: sin + cos + mul pattern (no reductions)
        if self._is_rope_pattern():
            return self._build_rope_kernel(kb, primary_dtype)

        # Speculative decoding: div + cmp with 3+ input ptrs (no reductions)
        if self._is_speculative_decode_pattern():
            return self._build_speculative_decode_kernel(kb, primary_dtype)

        # Top-K sampling: cmp with 2+ output ptrs (no reductions)
        if self._is_top_k_pattern():
            return self._build_top_k_kernel(kb, primary_dtype)

        # Dropout: 2 input ptrs + mask/select + mul, no reductions, 1 output
        if self._is_dropout_pattern():
            return self._build_dropout_kernel(kb, primary_dtype)

        # Batch normalization (eval): 4+ input ptrs, sub+mul, no reductions
        if self._is_batch_norm_pattern():
            return self._build_batch_norm_kernel(kb, primary_dtype)

        # Fused MLP: silu(gate) * up pattern (exp + neg + mul, no reductions)
        if self._is_fused_mlp_pattern():
            return self._build_fused_mlp_kernel(kb, primary_dtype)

        # Gather: 2 input ptrs (data + indices) with int arg, 1 output, no cmp/select
        if self._is_gather_pattern():
            return self._build_gather_kernel(kb, primary_dtype)

        # Activation functions: tanh, sigmoid, elu, leaky_relu, hardswish
        act = self._classify_activation()
        if act:
            return self._build_activation_kernel(kb, act, primary_dtype)

        # Standard elementwise pattern
        offsets = kb.make_block_offsets("pid", "offsets")
        if n_arg:
            mask = kb.make_mask(offsets, n_arg, "mask")
        else:
            mask = None

        # Load from each input pointer
        input_vars = {}
        for arg_name, (idx, dtype, is_output) in self.ptr_args.items():
            if not is_output:
                var_name = f"val_{arg_name}"
                kb.load(arg_name, offsets, mask, out_var=var_name, dtype=dtype)
                input_vars[arg_name] = var_name

        # Analyze the computation between loads and stores
        result_var = self._emit_computation(kb, input_vars, primary_dtype)

        # Store to output
        for arg_name, (idx, dtype, is_output) in self.ptr_args.items():
            if is_output:
                kb.store(arg_name, offsets, result_var, mask, dtype=dtype)

        return kb

    def _build_reduction_kernel(self, kb, n_arg, primary_dtype):
        """Generate a reduction kernel using threadgroup reduce pattern.

        Uses strided accumulation (each thread loops over elements) followed
        by two-level threadgroup reduction (SIMD intrinsics + shared memory).
        """
        reduce_result_ssa, reduce_input_ssa, reduce_op, _ = self.reduce_ops[0]

        # Find the input pointer for the reduction
        input_arg = None
        for arg_name, (idx, dtype, is_output) in self.ptr_args.items():
            if not is_output:
                input_arg = arg_name
                break

        # Find the output pointer
        output_arg = None
        for arg_name, (idx, dtype, is_output) in self.ptr_args.items():
            if is_output:
                output_arg = arg_name
                break

        if not input_arg or not output_arg or not n_arg:
            # Fallback: can't generate reduction, return empty kernel
            return kb

        # Shared memory for cross-SIMD-group reduction
        n_simd_groups = (kb.block_size + 31) // 32
        kb.declare_threadgroup_array("shared", dtype=primary_dtype, size=n_simd_groups)

        # Identity value for the reduction
        identity = {"sum": "0.0f", "max": "-INFINITY", "min": "INFINITY"}[reduce_op]
        combine = {"sum": "+", "max": "max", "min": "min"}[reduce_op]

        # Check if there's a pre-reduce computation (e.g., exp, mul before sum)
        pre_reduce_op = self._find_pre_reduce_op(reduce_input_ssa)

        # Strided accumulation loop
        kb._var("acc", identity, ty="float")
        kb.raw_line(f"for (uint i = lid; i < {n_arg}; i += {kb.block_size}u) {{")
        kb.indent()
        kb._var("idx", f"pid * {n_arg} + i", ty="uint")

        if pre_reduce_op:
            # Apply pre-reduce operation to each element
            op_kind, op_args = pre_reduce_op
            kb._var("loaded", f"{input_arg}[idx]", ty="float")
            if op_kind in ("exp", "log", "sqrt", "abs"):
                kb._var("elem", f"{op_kind}(loaded)", ty="float")
            elif op_kind in ("mul",) and len(op_args) == 2:
                kb._var("elem", f"loaded * loaded", ty="float")  # square
            else:
                kb.raw_line(f"float elem = loaded;")
        else:
            kb._var("elem", f"{input_arg}[idx]", ty="float")

        if combine == "+":
            kb.raw_line("acc += elem;")
        else:
            kb.raw_line(f"acc = {combine}(acc, elem);")
        kb.dedent()
        kb.raw_line("}")

        # Two-level threadgroup reduction
        kb.threadgroup_reduce(reduce_op, "acc", "shared", "total")

        # Thread 0 writes result
        kb.begin_if("lid == 0")
        kb.raw_line(f"{output_arg}[pid] = total;")
        kb.end_block()

        return kb

    def _is_softmax_pattern(self):
        """Check if multi-reduce pattern matches softmax (max + sum)."""
        if len(self.reduce_ops) < 2:
            return False
        ops = [r[2] for r in self.reduce_ops]
        return "max" in ops and "sum" in ops

    def _is_layer_norm_pattern(self):
        """Check if multi-reduce pattern matches layer norm (sum + sum).

        Layer norm has two sum reductions (mean, variance) and operations
        between them (subtract mean, square). Softmax has max + sum.
        """
        if len(self.reduce_ops) < 2:
            return False
        ops = [r[2] for r in self.reduce_ops]
        # Layer norm: two sum reductions (NOT softmax which has max+sum)
        if ops.count("sum") >= 2 and "max" not in ops:
            # Check for subtract or rsqrt between/after reductions
            return ("sub" in [v[0] for v in self.ssa_values.values()]
                    or "rsqrt" in [v[0] for v in self.ssa_values.values()])
        return False

    def _build_softmax_kernel(self, kb, n_arg, primary_dtype):
        """Generate a fused row-wise softmax kernel.

        Detected from 2 tt.reduce ops (max then sum) with exp in between.
        Each threadgroup processes one row:
        1. Find max(row)
        2. Compute exp(x - max)
        3. Sum the exponentials
        4. Divide each by the sum
        """
        n_simd_groups = (kb.block_size + 31) // 32

        # Find input and output pointers
        input_arg = None
        output_arg = None
        for arg_name, (idx, dtype, is_output) in self.ptr_args.items():
            if not is_output and input_arg is None:
                input_arg = arg_name
            if is_output:
                output_arg = arg_name

        if not input_arg or not output_arg or not n_arg:
            return kb

        # Shared memory for reductions
        kb.declare_threadgroup_array("shared_max", dtype=primary_dtype, size=n_simd_groups)
        kb.declare_threadgroup_array("shared_sum", dtype=primary_dtype, size=n_simd_groups)

        # Row base pointer: each threadgroup handles one row
        kb._var("row_start", f"pid * {n_arg}", ty="uint")

        # Pass 1: Find row max (strided accumulation)
        kb._var("local_max", "-INFINITY", ty="float")
        kb.raw_line(f"for (uint i = lid; i < {n_arg}; i += {kb.block_size}u) {{")
        kb.indent()
        kb.raw_line(f"local_max = max(local_max, {input_arg}[row_start + i]);")
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
        kb.raw_line(f"for (uint i = lid; i < {n_arg}; i += {kb.block_size}u) {{")
        kb.indent()
        kb._var("e", f"exp({input_arg}[row_start + i] - max_val)", ty="float")
        kb.raw_line(f"{output_arg}[row_start + i] = e;")
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
        kb.raw_line(f"for (uint i = lid; i < {n_arg}; i += {kb.block_size}u) {{")
        kb.indent()
        kb.raw_line(f"{output_arg}[row_start + i] /= sum_val;")
        kb.dedent()
        kb.raw_line("}")

        return kb

    def _build_layer_norm_kernel(self, kb, n_arg, primary_dtype):
        """Generate a fused layer norm kernel from two sum reductions.

        Detected from 2 tt.reduce sum ops with sub/mul between them.
        Pattern:
        1. sum(input)         → mean = sum / n
        2. sum((x - mean)^2)  → var = sum / n
        3. output = (x - mean) * rsqrt(var + eps) * gamma + beta

        Each threadgroup processes one row. Gamma/beta are detected from
        pointer args that are neither the primary input nor the output.
        """
        n_simd_groups = (kb.block_size + 31) // 32

        # Find input, output, and parameter pointers
        input_arg = None
        output_arg = None
        param_args = []
        for arg_name, (idx, dtype, is_output) in self.ptr_args.items():
            if is_output:
                output_arg = arg_name
            elif input_arg is None:
                input_arg = arg_name
            else:
                param_args.append(arg_name)

        if not input_arg or not output_arg or not n_arg:
            return kb

        # gamma is first param, beta is second (if they exist)
        gamma_arg = param_args[0] if len(param_args) > 0 else None
        beta_arg = param_args[1] if len(param_args) > 1 else None

        # Extract epsilon from constants in the IR
        eps = 1e-6
        for ssa, val in self.ssa_values.items():
            if val[0] == "constant":
                import re as _re
                m = _re.search(r"([\d.e+-]+)\s*:\s*f32", val[1])
                if m:
                    v = float(m.group(1))
                    if 0 < v < 1e-3:  # looks like an epsilon
                        eps = v

        # Shared memory for reductions
        kb.declare_threadgroup_array("shared_mean", dtype=primary_dtype, size=n_simd_groups)
        kb.declare_threadgroup_array("shared_var", dtype=primary_dtype, size=n_simd_groups)

        # Row base pointer
        kb._var("row_start", f"pid * {n_arg}", ty="uint")

        # Pass 1: Compute mean
        kb._var("local_sum", "0.0f", ty="float")
        kb.raw_line(f"for (uint i = lid; i < {n_arg}; i += {kb.block_size}u) {{")
        kb.indent()
        kb.raw_line(f"local_sum += {input_arg}[row_start + i];")
        kb.dedent()
        kb.raw_line("}")

        kb.threadgroup_reduce("sum", "local_sum", "shared_mean", "total_sum")

        kb.begin_if("lid == 0")
        kb.raw_line("shared_mean[0] = total_sum;")
        kb.end_block()
        kb.barrier("threadgroup")
        kb._var("mean_val", f"shared_mean[0] / float({n_arg})", ty="float")

        # Pass 2: Compute variance
        kb._var("local_var", "0.0f", ty="float")
        kb.raw_line(f"for (uint i = lid; i < {n_arg}; i += {kb.block_size}u) {{")
        kb.indent()
        kb._var("diff", f"{input_arg}[row_start + i] - mean_val", ty="float")
        kb.raw_line("local_var += diff * diff;")
        kb.dedent()
        kb.raw_line("}")

        kb.threadgroup_reduce("sum", "local_var", "shared_var", "total_var")

        kb.begin_if("lid == 0")
        kb.raw_line("shared_var[0] = total_var;")
        kb.end_block()
        kb.barrier("threadgroup")
        kb._var("var_val", f"shared_var[0] / float({n_arg})", ty="float")
        kb._var("inv_std", f"rsqrt(var_val + {eps}f)", ty="float")

        # Pass 3: Normalize
        kb.raw_line(f"for (uint i = lid; i < {n_arg}; i += {kb.block_size}u) {{")
        kb.indent()
        if gamma_arg and beta_arg:
            kb.raw_line(f"{output_arg}[row_start + i] = ({input_arg}[row_start + i] - mean_val) * inv_std * {gamma_arg}[i] + {beta_arg}[i];")
        elif gamma_arg:
            kb.raw_line(f"{output_arg}[row_start + i] = ({input_arg}[row_start + i] - mean_val) * inv_std * {gamma_arg}[i];")
        else:
            kb.raw_line(f"{output_arg}[row_start + i] = ({input_arg}[row_start + i] - mean_val) * inv_std;")
        kb.dedent()
        kb.raw_line("}")

        return kb

    def _is_flash_attention_pattern(self):
        """Check if the IR matches a flash attention pattern.

        Flash attention has:
        - 2+ tt.dot ops (Q@K^T and P@V)
        - exp() between them (softmax numerator)
        - A max reduction or explicit max for numerical stability
        """
        if len(self.dot_ops) < 2:
            return False
        # Check for exp in the SSA values (used in softmax between dots)
        has_exp = any(v[0] == "exp" for v in self.ssa_values.values())
        # Check for max reduction or explicit max
        has_max = (any(r[2] == "max" for r in self.reduce_ops) or
                   any(v[0] == "fmax" for v in self.ssa_values.values()))
        return has_exp and has_max

    def _build_flash_attention_kernel(self, kb, primary_dtype):
        """Generate a flash attention kernel from the pattern.

        Detected from 2+ tt.dot ops with exp/max between them.
        Delegates to the pre-built flash attention kernel generator.
        """
        from triton_metal.codegen.msl_emitter import make_flash_attention_kernel

        # Detect head_dim from pointer args
        # In typical flash attention TTGIR, Q/K/V are the first 3 pointer args
        # and the last scalar arg before seq_len determines head_dim
        head_dim = 64  # default

        # Check for causal masking: look for cmpi or "causal" in IR
        causal = "causal" in self.ir_text.lower() or any(
            v[0] == "mask" and "slt" in str(self.ssa_values.get(v[1], ""))
            for v in self.ssa_values.values() if v[0] == "select"
        )

        kb.set_prebuilt_msl(make_flash_attention_kernel(
            head_dim=head_dim, causal=causal
        ))
        return kb

    def _build_matmul_kernel(self, kb, primary_dtype):
        """Generate a tiled matmul kernel from tt.dot pattern.

        Detected from tt.dot operations in the IR. Generates a tiled
        matmul using simdgroup_matrix hardware (8x8 MMA).

        Expected pointer args: A, B, C (+ optional M, N, K scalars).
        Dispatch: one threadgroup per 32x32 output tile, 128 threads each.
        """
        from triton_metal.codegen.msl_emitter import make_simdgroup_matmul_kernel

        # Determine dtype from the dot operation
        dtype_map = {"fp32": "fp32", "fp16": "fp16", "bf16": "bf16"}
        msl_dtype = dtype_map.get(primary_dtype, "fp32")

        # We generate the simdgroup matmul kernel as a standalone MSL
        # and return it as a "prebuilt" kernel via KernelBuilder's raw MSL mode
        kb.set_prebuilt_msl(make_simdgroup_matmul_kernel(dtype=msl_dtype))
        return kb

    def _is_cross_entropy_pattern(self):
        """Check if IR matches cross-entropy loss: max + sum (softmax) + log + sum.

        Cross-entropy has:
        - 3+ reduce ops (max for softmax stability, sum for softmax denominator,
          sum for final loss aggregation)
        - exp between max and first sum (softmax numerator)
        - log after the softmax
        """
        if len(self.reduce_ops) < 3:
            return False
        ops = [r[2] for r in self.reduce_ops]
        if "max" not in ops:
            return False
        if ops.count("sum") < 2:
            return False
        # Check for both exp and log in SSA values
        has_exp = any(v[0] == "exp" for v in self.ssa_values.values())
        has_log = any(v[0] == "log" for v in self.ssa_values.values())
        return has_exp and has_log

    def _build_cross_entropy_kernel(self, kb, n_arg, primary_dtype):
        """Generate a fused cross-entropy loss kernel.

        Detected from 3 reduces (max+sum+sum) with exp and log.
        Delegates to the pre-built cross-entropy kernel.
        """
        from triton_metal.codegen.msl_emitter import make_cross_entropy_kernel
        kb.set_prebuilt_msl(make_cross_entropy_kernel())
        return kb

    def _is_fused_residual_norm_pattern(self):
        """Check if IR matches fused residual add + layer norm.

        This pattern has:
        - 2+ sum reductions (mean + variance, layer norm pattern)
        - An arith.addf before the first reduction (residual connection)
        - rsqrt or sub operations (layer norm normalization)
        - No max reduction (distinguishes from softmax)
        """
        if len(self.reduce_ops) < 2:
            return False
        ops = [r[2] for r in self.reduce_ops]
        if ops.count("sum") < 2 or "max" in ops:
            return False
        # Must have both add and (sub or rsqrt) for residual + norm
        has_add = any(v[0] == "add" for v in self.ssa_values.values())
        has_sub_or_rsqrt = (
            any(v[0] == "sub" for v in self.ssa_values.values()) or
            any(v[0] == "rsqrt" for v in self.ssa_values.values())
        )
        if not (has_add and has_sub_or_rsqrt):
            return False

        # Check the input to the first sum reduction for an add (residual)
        first_sum = None
        for entry in self.reduce_ops:
            result_ssa, input_ssa, op_kind = entry[0], entry[1], entry[2]
            if op_kind == "sum":
                first_sum = (result_ssa, input_ssa)
                break

        if first_sum is None:
            return False

        # Walk back from the first reduce's input to see if there's an add
        # (residual connection: x + residual before normalization)
        def _has_add_in_chain(ssa, depth=5):
            if depth <= 0:
                return False
            val = self.ssa_values.get(ssa)
            if val is None:
                return False
            if val[0] == "add":
                return True
            # Follow operands
            for operand in val[1:]:
                if isinstance(operand, str) and operand.startswith("%"):
                    if _has_add_in_chain(operand, depth - 1):
                        return True
            return False

        return _has_add_in_chain(first_sum[1])

    def _build_fused_residual_norm_kernel(self, kb, n_arg, primary_dtype):
        """Generate a fused residual add + layer norm kernel.

        output = LayerNorm(input + residual, gamma, beta)

        Delegates to the dedicated fused kernel that combines residual
        addition with layer normalization in a single pass.
        """
        from triton_metal.codegen.msl_emitter import make_fused_residual_norm_kernel
        kb.set_prebuilt_msl(make_fused_residual_norm_kernel())
        return kb

    def _is_variance_pattern(self):
        """Check if IR matches a variance computation pattern.

        Variance has:
        - 2 sum reductions (for mean and for sum of squared diffs)
        - subtract (x - mean)
        - multiply (diff * diff, i.e. squaring)
        - NO rsqrt (that would be layer norm)
        - 1 output pointer (variance values)
        """
        if len(self.reduce_ops) < 2:
            return False
        ops_list = [r[2] for r in self.reduce_ops]
        if ops_list.count("sum") < 2 or "max" in ops_list:
            return False
        ssa_ops = {v[0] for v in self.ssa_values.values()}
        has_sub = "sub" in ssa_ops
        has_mul = "mul" in ssa_ops
        has_rsqrt = "rsqrt" in ssa_ops
        # Variance = 2 sums + sub + mul, but NO rsqrt
        return has_sub and has_mul and not has_rsqrt

    def _build_variance_kernel(self, kb, n_arg, primary_dtype):
        """Generate a variance kernel from the pattern."""
        from triton_metal.codegen.msl_emitter import make_variance_kernel
        kb.set_prebuilt_msl(make_variance_kernel())
        return kb

    def _is_rope_pattern(self):
        """Check if IR matches a RoPE (Rotary Position Embedding) pattern.

        RoPE has:
        - sin and cos operations (for rotation matrix)
        - Interleaved multiply and add/sub (x_even*cos - x_odd*sin, x_even*sin + x_odd*cos)
        - No reductions (elementwise operation)
        """
        has_sin = any(v[0] == "sin" for v in self.ssa_values.values())
        has_cos = any(v[0] == "cos" for v in self.ssa_values.values())
        has_mul = any(v[0] == "mul" for v in self.ssa_values.values())
        return has_sin and has_cos and has_mul and not self.reduce_ops

    def _build_rope_kernel(self, kb, primary_dtype):
        """Generate a RoPE kernel from the pattern.

        Delegates to the pre-built RoPE kernel.
        """
        from triton_metal.codegen.msl_emitter import make_rope_kernel
        kb.set_prebuilt_msl(make_rope_kernel())
        return kb

    def _is_quantized_matmul_pattern(self):
        """Check if IR matches a quantized matmul pattern.

        Quantized matmul has:
        - tt.dot ops (matrix multiplication)
        - arith.extsi or int_cast operations (integer weight dequantization)
        - Indicates INT8 or INT4 weights being promoted to float for computation
        """
        if not self.dot_ops:
            return False
        has_int_cast = any(v[0] == "int_cast" for v in self.ssa_values.values())
        return has_int_cast

    def _build_quantized_matmul_kernel(self, kb, primary_dtype):
        """Generate a quantized matmul kernel from the pattern.

        Uses INT8 quantized matmul by default. If the IR has group-size
        related constants, could be INT4, but INT8 is the safer default.
        """
        from triton_metal.codegen.msl_emitter import make_int8_matmul_kernel
        kb.set_prebuilt_msl(make_int8_matmul_kernel())
        return kb

    def _is_fused_mlp_pattern(self):
        """Check if IR matches a fused MLP (SwiGLU) pattern.

        Fused MLP has:
        - exp (for silu/sigmoid: x / (1 + exp(-x)))
        - multiply (gate * up projection fusion)
        - divide or reciprocal (for 1 / (1 + exp(-x)))
        - No reductions (elementwise)
        - No dot ops (not a matmul)
        - 2+ input pointer args (gate and up projections)
        """
        if self.reduce_ops or self.dot_ops:
            return False
        has_exp = any(v[0] == "exp" for v in self.ssa_values.values())
        has_mul = any(v[0] == "mul" for v in self.ssa_values.values())
        has_neg = any(v[0] == "neg" for v in self.ssa_values.values())
        # SiLU requires: exp(-x) → neg + exp, then division
        has_div = any(v[0] == "div" for v in self.ssa_values.values())
        # Need at least 2 input pointers (gate + up)
        n_inputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if not is_out)
        return has_exp and has_mul and n_inputs >= 2 and (has_neg or has_div)

    def _build_fused_mlp_kernel(self, kb, primary_dtype):
        """Generate a fused MLP (SwiGLU) kernel from the pattern.

        output = silu(gate) * up
        """
        from triton_metal.codegen.msl_emitter import make_fused_mlp_kernel
        kb.set_prebuilt_msl(make_fused_mlp_kernel())
        return kb

    def _is_paged_attention_pattern(self):
        """Check if IR matches a paged attention pattern.

        Paged attention has:
        - exp + mul (attention score computation)
        - Reductions (max + sum for softmax)
        - Indirect indexing via page table (load from pointer loaded from pointer)
        - 3+ input pointers (Q, K_cache/V_cache, page_table)
        """
        if not self.reduce_ops:
            return False
        has_exp = any(v[0] == "exp" for v in self.ssa_values.values())
        has_mul = any(v[0] == "mul" for v in self.ssa_values.values())
        n_inputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if not is_out)
        # Page attention needs 3+ input ptrs and exp+mul (attention softmax)
        return has_exp and has_mul and n_inputs >= 3 and not self.dot_ops

    def _build_paged_attention_kernel(self, kb, primary_dtype):
        """Generate a paged attention kernel from the pattern."""
        from triton_metal.codegen.msl_emitter import make_paged_attention_kernel
        kb.set_prebuilt_msl(make_paged_attention_kernel())
        return kb

    def _is_top_k_pattern(self):
        """Check if IR matches a top-k sampling pattern.

        Top-k has:
        - Comparison operations (mask ops from cmpi/cmpf)
        - No dot ops (not a matmul)
        - 1 input pointer (logits), 2 output pointers (values + indices)
        """
        if self.dot_ops:
            return False
        has_cmp = any(v[0] == "mask" for v in self.ssa_values.values())
        n_outputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if is_out)
        return has_cmp and n_outputs >= 2 and not self.reduce_ops

    def _build_top_k_kernel(self, kb, primary_dtype):
        """Generate a top-k sampling kernel from the pattern."""
        from triton_metal.codegen.msl_emitter import make_top_k_kernel
        kb.set_prebuilt_msl(make_top_k_kernel())
        return kb

    def _is_speculative_decode_pattern(self):
        """Check if IR matches a speculative decoding pattern.

        Speculative decoding has:
        - Division (probability ratio: target/draft)
        - Comparison (acceptance test: ratio >= random threshold)
        - 3+ input pointers (draft_probs, target_probs, draft_tokens, random)
        - No dot ops
        """
        if self.dot_ops:
            return False
        has_div = any(v[0] == "div" for v in self.ssa_values.values())
        has_cmp = any(v[0] == "mask" for v in self.ssa_values.values())
        n_inputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if not is_out)
        return has_div and has_cmp and n_inputs >= 3

    def _build_speculative_decode_kernel(self, kb, primary_dtype):
        """Generate a speculative decoding kernel from the pattern."""
        from triton_metal.codegen.msl_emitter import make_speculative_decode_kernel
        kb.set_prebuilt_msl(make_speculative_decode_kernel())
        return kb

    def _is_beam_search_pattern(self):
        """Check if IR matches a beam search pattern.

        Beam search has:
        - Comparison + add (score accumulation and comparison)
        - Reductions (finding top-k beams)
        - 2+ output pointers (scores + indices)
        - No dot ops
        """
        if self.dot_ops:
            return False
        has_cmp = any(v[0] == "mask" for v in self.ssa_values.values())
        has_add = any(v[0] == "add" for v in self.ssa_values.values())
        n_outputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if is_out)
        return has_cmp and has_add and self.reduce_ops and n_outputs >= 2

    def _build_beam_search_kernel(self, kb, primary_dtype):
        """Generate a beam search kernel from the pattern."""
        from triton_metal.codegen.msl_emitter import make_beam_search_kernel
        kb.set_prebuilt_msl(make_beam_search_kernel())
        return kb

    def _classify_activation(self):
        """Classify the activation function from the IR ops.

        Returns the activation name ("tanh", "sigmoid", "elu", "leaky_relu",
        "hardswish") or None if no activation pattern matches.

        Requirements for activation detection:
        - No reductions, no dot ops (elementwise only)
        - 1 input pointer, 1 output pointer
        """
        if self.reduce_ops or self.dot_ops:
            return None
        n_inputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if not is_out)
        n_outputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if is_out)
        if n_inputs != 1 or n_outputs != 1:
            return None

        ssa_ops = {v[0] for v in self.ssa_values.values()}

        # tanh: explicit tanh op
        if "tanh" in ssa_ops:
            return "tanh"
        # sigmoid: exp + div (or neg+exp+add+div), no tanh, single input
        if "exp" in ssa_ops and "div" in ssa_ops and "select" not in ssa_ops:
            return "sigmoid"
        # elu: exp + select (x > 0 ? x : alpha*(exp(x)-1))
        if "exp" in ssa_ops and "select" in ssa_ops:
            return "elu"
        # hardswish: select + add + mul + div, no exp
        if "select" in ssa_ops and "add" in ssa_ops and "div" in ssa_ops and "exp" not in ssa_ops:
            return "hardswish"
        # leaky_relu: select + mul, no exp, no div
        if "select" in ssa_ops and "mul" in ssa_ops and "exp" not in ssa_ops and "div" not in ssa_ops:
            return "leaky_relu"

        return None

    def _build_activation_kernel(self, kb, activation, primary_dtype):
        """Generate an activation kernel from the classified pattern."""
        from triton_metal.codegen.msl_emitter import make_activation_kernel
        kb.set_prebuilt_msl(make_activation_kernel(activation=activation))
        return kb

    def _is_rms_norm_pattern(self):
        """Check if IR matches an RMS norm pattern.

        RMS norm has:
        - Sum reduction (for sum of squares)
        - rsqrt operation
        - mul operation (scale by weight)
        - NO sub (no mean subtraction — distinguishes from layer norm)
        - 2-3 input pointers (input, weight, optionally bias)
        """
        ssa_ops = {v[0] for v in self.ssa_values.values()}
        has_rsqrt = "rsqrt" in ssa_ops
        has_mul = "mul" in ssa_ops
        has_sub = "sub" in ssa_ops
        ops_list = [r[2] for r in self.reduce_ops]
        has_sum = "sum" in ops_list
        # RMS norm: sum + rsqrt + mul but NO sub
        return has_sum and has_rsqrt and has_mul and not has_sub

    def _build_rms_norm_kernel(self, kb, n_arg, primary_dtype):
        """Generate an RMS norm kernel from the pattern."""
        from triton_metal.codegen.msl_emitter import make_rms_norm_kernel
        kb.set_prebuilt_msl(make_rms_norm_kernel())
        return kb

    def _is_group_norm_pattern(self):
        """Check if IR matches a group normalization pattern.

        Group norm has:
        - Sum reductions (for mean and variance within groups)
        - rsqrt or div (normalization step)
        - 3+ input pointers (input, weight, bias)
        - 1 output pointer
        - 2+ scalar args (n_channels, spatial_size)
        """
        n_inputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if not is_out)
        n_outputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if is_out)
        if n_inputs < 3 or n_outputs != 1:
            return False
        ssa_ops = {v[0] for v in self.ssa_values.values()}
        has_rsqrt = "rsqrt" in ssa_ops
        has_div = "div" in ssa_ops
        ops_list = [r[2] for r in self.reduce_ops]
        has_sum = "sum" in ops_list
        return has_sum and (has_rsqrt or has_div) and len(self.scalar_args) >= 2

    def _build_group_norm_kernel(self, kb, primary_dtype):
        """Generate a group norm kernel from the pattern."""
        from triton_metal.codegen.msl_emitter import make_group_norm_kernel
        kb.set_prebuilt_msl(make_group_norm_kernel())
        return kb

    def _is_gather_pattern(self):
        """Check if IR matches a gather (indexed read) pattern.

        Gather has:
        - 2 input pointers (data buffer + index buffer)
        - 1 output pointer
        - No reductions, no dot ops
        - No select/cmp (distinguishes from dropout)
        - Index buffer has integer type
        """
        if self.reduce_ops or self.dot_ops:
            return False
        n_inputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if not is_out)
        n_outputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if is_out)
        if n_inputs != 2 or n_outputs != 1:
            return False
        ssa_ops = {v[0] for v in self.ssa_values.values()}
        # Gather has no comparison/select (distinguishes from dropout)
        has_select = "select" in ssa_ops
        # Check if one input has integer type
        has_int_input = any(
            dtype in ("i32", "i64") for _, (_, dtype, is_out) in self.ptr_args.items()
            if not is_out
        )
        return not has_select and has_int_input

    def _build_gather_kernel(self, kb, primary_dtype):
        """Generate a gather kernel from the pattern."""
        from triton_metal.codegen.msl_emitter import make_gather_kernel
        kb.set_prebuilt_msl(make_gather_kernel())
        return kb

    def _is_dropout_pattern(self):
        """Check if IR matches a dropout pattern.

        Dropout has:
        - 2 input pointers (data, random_mask/threshold)
        - 1 output pointer
        - Comparison (mask) + select + mul operations
        - No reductions, no dot ops
        """
        if self.reduce_ops or self.dot_ops:
            return False
        n_inputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if not is_out)
        n_outputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if is_out)
        if n_inputs != 2 or n_outputs != 1:
            return False
        ssa_ops = {v[0] for v in self.ssa_values.values()}
        has_cmp = "mask" in ssa_ops
        has_select = "select" in ssa_ops
        has_mul = "mul" in ssa_ops
        return has_cmp and has_select and has_mul

    def _build_dropout_kernel(self, kb, primary_dtype):
        """Generate a dropout kernel from the pattern."""
        from triton_metal.codegen.msl_emitter import make_fused_dropout_kernel
        kb.set_prebuilt_msl(make_fused_dropout_kernel())
        return kb

    def _is_batch_norm_pattern(self):
        """Check if IR matches a batch normalization (eval mode) pattern.

        Batch norm eval has:
        - 4+ input pointers (input, running_mean, running_var, weight, optionally bias)
        - 1 output pointer
        - sub + mul operations (normalize and scale)
        - No reductions (uses pre-computed running stats)
        - No dot ops
        """
        if self.reduce_ops or self.dot_ops:
            return False
        n_inputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if not is_out)
        n_outputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if is_out)
        if n_inputs < 4 or n_outputs != 1:
            return False
        ssa_ops = {v[0] for v in self.ssa_values.values()}
        has_sub = "sub" in ssa_ops
        has_mul = "mul" in ssa_ops
        return has_sub and has_mul

    def _build_batch_norm_kernel(self, kb, primary_dtype):
        """Generate a batch norm (eval mode) kernel from the pattern."""
        from triton_metal.codegen.msl_emitter import make_batch_norm_kernel
        kb.set_prebuilt_msl(make_batch_norm_kernel())
        return kb

    def _is_online_softmax_pattern(self):
        """Check if IR matches an online softmax pattern.

        Online softmax has:
        - scf.for loop (streaming single-pass over data)
        - exp operation (for softmax normalization)
        - Reductions (max or sum inside the loop)
        - 1 input pointer, 1 output pointer
        """
        if not self.scf_for_loops:
            return False
        ssa_ops = {v[0] for v in self.ssa_values.values()}
        has_exp = "exp" in ssa_ops
        n_inputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if not is_out)
        n_outputs = sum(1 for _, (_, _, is_out) in self.ptr_args.items() if is_out)
        return has_exp and self.reduce_ops and n_inputs == 1 and n_outputs == 1

    def _build_online_softmax_kernel(self, kb, primary_dtype):
        """Generate an online softmax kernel from the pattern."""
        from triton_metal.codegen.msl_emitter import make_online_softmax_kernel
        kb.set_prebuilt_msl(make_online_softmax_kernel())
        return kb

    def _find_pre_reduce_op(self, reduce_input_ssa):
        """Check if there's a computation between load and reduce.

        Returns (op_kind, op_args) if found, None otherwise.
        """
        val = self.ssa_values.get(reduce_input_ssa)
        if val is None:
            return None

        op = val[0]
        # If the input to reduce is a computation (not a direct load)
        if op in ("exp", "log", "sqrt", "abs"):
            return (op, [val[1]])
        if op in ("add", "sub", "mul", "div"):
            return (op, [val[1], val[2]])

        return None

    def _emit_computation(self, kb, input_vars, dtype):
        """Analyze the computation graph and emit operations.

        Traces from store values back through the SSA graph to find
        the chain of operations between loads and stores.
        """
        if not self.ops:
            # No stores found — just return first loaded value
            if input_vars:
                return list(input_vars.values())[0]
            return "0.0f"

        # Find the value being stored
        store_op = self.ops[0]  # first store
        store_val_ssa = store_op[2]

        # Recursively emit the computation chain
        return self._emit_ssa_value(kb, store_val_ssa, input_vars, dtype, set())

    def _emit_ssa_value(self, kb, ssa, input_vars, dtype, emitted):
        """Recursively emit MSL for an SSA value."""
        if ssa in emitted:
            return self.computed_values.get(ssa, "0.0f")
        emitted.add(ssa)

        # Check if this is a loaded value
        val_info = self.ssa_values.get(ssa)
        if val_info is None:
            # Function argument — check if it's an input pointer we loaded
            arg_name = ssa.lstrip("%")
            if arg_name in input_vars:
                return input_vars[arg_name]
            if arg_name in self.scalar_args:
                return arg_name
            return "0.0f"

        op = val_info[0]

        # Passthrough (extf, truncf)
        if op == "passthrough":
            return self._emit_ssa_value(kb, val_info[1], input_vars, dtype, emitted)

        # Load — map to the loaded variable
        if op == "load":
            ptr_ssa = val_info[1]
            arg_name = self._trace_to_ptr_arg(ptr_ssa)
            if arg_name and arg_name in input_vars:
                self.computed_values[ssa] = input_vars[arg_name]
                return input_vars[arg_name]
            return "0.0f"

        # Binary float ops
        if op in ("add", "sub", "mul", "div"):
            lhs_var = self._emit_ssa_value(kb, val_info[1], input_vars, dtype, emitted)
            rhs_var = self._emit_ssa_value(kb, val_info[2], input_vars, dtype, emitted)
            # Generate a unique variable name
            var_name = f"r_{len(self.computed_values)}"
            kb.binary_op(op, lhs_var, rhs_var, var_name)
            self.computed_values[ssa] = var_name
            return var_name

        # Unary math ops
        if op in ("exp", "log", "sqrt", "rsqrt", "abs", "neg", "sin", "cos"):
            x_var = self._emit_ssa_value(kb, val_info[1], input_vars, dtype, emitted)
            var_name = f"r_{len(self.computed_values)}"
            kb.unary_op(op, x_var, var_name)
            self.computed_values[ssa] = var_name
            return var_name

        # Integer cast (extsi, trunci) — treat as passthrough
        if op == "int_cast":
            return self._emit_ssa_value(kb, val_info[1], input_vars, dtype, emitted)

        # Fused multiply-add: fma(a, b, c) = a*b + c
        if op == "fma":
            a_var = self._emit_ssa_value(kb, val_info[1], input_vars, dtype, emitted)
            b_var = self._emit_ssa_value(kb, val_info[2], input_vars, dtype, emitted)
            c_var = self._emit_ssa_value(kb, val_info[3], input_vars, dtype, emitted)
            var_name = f"r_{len(self.computed_values)}"
            kb.fused_op("fma", [a_var, b_var, c_var], var_name)
            self.computed_values[ssa] = var_name
            return var_name

        # Float max/min
        if op in ("fmax", "fmin"):
            lhs_var = self._emit_ssa_value(kb, val_info[1], input_vars, dtype, emitted)
            rhs_var = self._emit_ssa_value(kb, val_info[2], input_vars, dtype, emitted)
            var_name = f"r_{len(self.computed_values)}"
            msl_op = "max" if op == "fmax" else "min"
            kb._var(var_name, f"{msl_op}({lhs_var}, {rhs_var})", ty="float")
            self.computed_values[ssa] = var_name
            return var_name

        # Select (ternary)
        if op == "select":
            cond_var = self._emit_ssa_value(kb, val_info[1], input_vars, dtype, emitted)
            true_var = self._emit_ssa_value(kb, val_info[2], input_vars, dtype, emitted)
            false_var = self._emit_ssa_value(kb, val_info[3], input_vars, dtype, emitted)
            var_name = f"r_{len(self.computed_values)}"
            kb._var(var_name, f"{cond_var} ? {true_var} : {false_var}", ty="float")
            self.computed_values[ssa] = var_name
            return var_name

        # Constants
        if op == "constant":
            # Try to extract numeric value
            val_str = val_info[1]
            m = re.search(r"([\d.e+-]+)\s*:\s*\w+", val_str)
            if m:
                return f"{float(m.group(1))}f"
            return "0.0f"

        # Splat of a scalar arg
        if op == "splat":
            source = val_info[1]
            return self._emit_ssa_value(kb, source, input_vars, dtype, emitted)

        # Reduce — the result is computed in _build_reduction_kernel
        if op == "reduce":
            return "total"

        # Loop induction variable
        if op == "loop_iv":
            return ssa.lstrip("%")

        return "0.0f"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_ttgir(ir_text, options):
    """Parse TTGIR MLIR text and return a KernelBuilder.

    Args:
        ir_text: MLIR text from str(module).
        options: MetalOptions instance.

    Returns:
        KernelBuilder configured from the parsed IR.
    """
    parser = TTGIRParser(ir_text, options)
    return parser.parse()
