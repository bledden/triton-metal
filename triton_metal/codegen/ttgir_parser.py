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
    """Convert MLIR type string to Triton dtype string."""
    mlir_type = mlir_type.strip()
    _map = {
        "f32": "fp32",
        "f16": "fp16",
        "bf16": "bf16",
        "f64": "fp64",
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

    def _parse_body(self):
        """Walk through the body and classify operations."""
        # First scan for multi-line blocks (reduce, scf.for)
        self._scan_scf_for_loops()
        self._scan_reductions()

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

            # Unary math ops
            for op_name, op_key in [("math.exp", "exp"), ("math.log", "log"),
                                     ("math.sqrt", "sqrt"), ("math.absf", "abs")]:
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

        # Check if this is a multi-reduce pattern (softmax) or single reduction
        if len(self.reduce_ops) >= 2 and self._is_softmax_pattern():
            return self._build_softmax_kernel(kb, n_arg, primary_dtype)
        if self.reduce_ops:
            return self._build_reduction_kernel(kb, n_arg, primary_dtype)

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
        if op in ("exp", "log", "sqrt", "abs", "neg"):
            x_var = self._emit_ssa_value(kb, val_info[1], input_vars, dtype, emitted)
            var_name = f"r_{len(self.computed_values)}"
            kb.unary_op(op, x_var, var_name)
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
