"""Generic op-by-op lowering from IRGraph to MSL via KernelBuilder.

Processes each TTGIR operation independently, mapping it to MSL code.
This replaces the 30+ pattern matchers in ttgir_parser.py with a single
pass that lowers each op based on its type.

Metal-specific considerations:
- No tensor abstraction: each thread processes one element
- tt.splat is a no-op (scalar→per-thread is free in SIMT)
- tt.make_range → thread_position_in_threadgroup (lid)
- tt.reduce → SIMD intrinsics + threadgroup shared memory
- All FP16/BF16 computation done in float, cast at load/store
"""

import re
from typing import Any, Dict, List, Optional

from triton_metal.codegen.mlir_walker import IRGraph, SSAValue, FuncArg, CalledFunc, _extract_shape
from triton_metal.codegen.msl_emitter import KernelBuilder, _msl_compute_type, _sanitize_msl_name
from triton_metal.codegen.msl_types import triton_type_to_msl
from triton_metal.errors import MetalCodegenError, MetalNotImplementedError


# ---------------------------------------------------------------------------
# Comparison predicate maps
# ---------------------------------------------------------------------------

# MLIR integer comparison predicates (arith.cmpi)
CMPI_PREDICATES = {
    0: "==",   # eq
    1: "!=",   # ne
    2: "<",    # slt
    3: "<=",   # sle
    4: ">",    # sgt
    5: ">=",   # sge
    6: "<",    # ult (unsigned, same op in MSL)
    7: "<=",   # ule
    8: ">",    # ugt
    9: ">=",   # uge
}

# Named predicate map
CMPI_NAMED = {
    "eq": "==", "ne": "!=",
    "slt": "<", "sle": "<=", "sgt": ">", "sge": ">=",
    "ult": "<", "ule": "<=", "ugt": ">", "uge": ">=",
}

# MLIR float comparison predicates (arith.cmpf)
CMPF_PREDICATES = {
    0: "false",  # false (always false)
    1: "==",     # oeq
    2: ">",      # ogt
    3: ">=",     # oge
    4: "<",      # olt
    5: "<=",     # ole
    6: "!=",     # one
    # 7-15: unordered variants
}

CMPF_NAMED = {
    "oeq": "==", "ogt": ">", "oge": ">=",
    "olt": "<", "ole": "<=", "one": "!=",
    "ueq": "==", "ugt": ">", "uge": ">=",
    "ult": "<", "ule": "<=", "une": "!=",
}


# ---------------------------------------------------------------------------
# MLIR type → Triton dtype mapping
# ---------------------------------------------------------------------------

def _mlir_to_triton_dtype(mlir_type: str) -> str:
    """Map MLIR element type to Triton dtype string."""
    _map = {
        "f32": "fp32", "f16": "fp16", "bf16": "bf16", "f64": "fp64",
        "i1": "i1", "i8": "i8", "i16": "i16", "i32": "i32", "i64": "i64",
    }
    result = _map.get(mlir_type)
    if result:
        return result
    # Handle multi-dim type strings like "4xi32" → extract base type
    import re
    m = re.search(r"([a-z]\w*)$", mlir_type)
    if m:
        return _map.get(m.group(1), "fp32")
    return "fp32"


# Map MLIR integer elem_type to (MSL type, internal dtype)
_INT_TYPE_MAP = {
    "i1": ("bool", "i1"),
    "i8": ("char", "i8"),
    "i16": ("short", "i16"),
    "i32": ("int", "i32"),
    "i64": ("long", "i64"),
}

# Unsigned variants
_UINT_TYPE_MAP = {
    "i8": ("uchar", "u8"),
    "i16": ("ushort", "u16"),
    "i32": ("uint", "u32"),
    "i64": ("ulong", "u64"),
}


def _msl_int_type(elem_type: str, unsigned: bool = False) -> tuple:
    """Return (msl_type, triton_dtype) for an integer elem_type.

    Args:
        elem_type: MLIR type like "i8", "i16", "i32", "i64"
        unsigned: If True, use unsigned MSL types

    Returns:
        Tuple of (MSL type string, internal dtype string)
    """
    if unsigned and elem_type in _UINT_TYPE_MAP:
        return _UINT_TYPE_MAP[elem_type]
    return _INT_TYPE_MAP.get(elem_type, ("int", "i32"))


def _shape_numel(shape: tuple) -> int:
    """Return the total number of elements in a shape tuple.

    Examples:
        () -> 1 (scalar)
        (32,) -> 32
        (32, 64) -> 2048
    """
    result = 1
    for d in shape:
        result *= d
    return result


# ---------------------------------------------------------------------------
# Generic Lowerer
# ---------------------------------------------------------------------------

class GenericLowerer:
    """Lower an IRGraph to MSL source code via KernelBuilder."""

    def __init__(self, graph: IRGraph, options=None):
        self.graph = graph
        self.options = options
        self.env = {}           # ssa_id -> MSL variable name
        self.env_types = {}     # ssa_id -> triton dtype string
        self.env_is_mask = {}   # ssa_id -> True if this is a bool mask
        self.env_is_ptr = {}    # ssa_id -> (base_ptr_name, offsets_var)
        self.env_shapes = {}    # ssa_id -> shape tuple, e.g., (32, 64)
        self.kb = None
        self._var_counter = 0

        # Track stores for output detection
        self._store_ptr_ids = set()

        # Shared memory counter for reductions
        self._shared_counter = 0

        # Whether kernel uses tt.get_num_programs (needs grid size parameter)
        self._needs_num_programs = False

        # 2D kernel info (populated by _prescan_2d_info)
        self._is_2d = False
        self._effective_2d_shape = None  # e.g., (32, 64)
        self._make_range_dim = {}  # ssa_id -> dimension index (0=row, 1=col)

        # Track which program_id axes are used (for kernel signature)
        self._used_pid_axes = set()  # {0, 1, 2}

        # SSA ids to skip (handled as part of a fused pattern)
        self._skip_ids = set()

    def _next_var(self, prefix="r") -> str:
        name = f"{prefix}_{self._var_counter}"
        self._var_counter += 1
        return name

    # -- Shape tracking helpers --------------------------------------------------

    def _get_shape(self, ssa_id: int) -> tuple:
        """Return the tracked shape for an SSA value.

        Returns the shape tuple from env_shapes if tracked, otherwise
        attempts to infer from the op's type_str via _extract_shape.
        Falls back to () (scalar) if no shape information is available.
        """
        if ssa_id in self.env_shapes:
            return self.env_shapes[ssa_id]
        # Try to infer from the op's type_str
        for op in self.graph.ops:
            if op.id == ssa_id and op.type_str:
                shape = _extract_shape(op.type_str)
                if shape:
                    self.env_shapes[ssa_id] = shape
                    return shape
        return ()

    def _is_scalar(self, ssa_id: int) -> bool:
        """Check if an SSA value has scalar shape (no dimensions).

        A value is scalar if its shape is () — i.e., it has no tensor
        dimensions.  Scalars don't need per-thread indexing; they are
        the same value on every thread.
        """
        return self._get_shape(ssa_id) == ()

    def _propagate_shape_from_type(self, ssa: SSAValue):
        """Set env_shapes[ssa.id] from the op's result type_str.

        Used as a common shape-propagation step after lowering an op.
        If the type_str contains a tensor shape, record it; otherwise
        the value is implicitly scalar (shape = ()).
        """
        if ssa.type_str:
            shape = _extract_shape(ssa.type_str)
            if shape:
                self.env_shapes[ssa.id] = shape
                return
        # No tensor type → scalar
        self.env_shapes[ssa.id] = ()

    def _propagate_shape_elementwise(self, ssa: SSAValue):
        """Propagate shape for element-wise ops (arith, math, select, etc.).

        Element-wise ops inherit the shape of their operands.  When operands
        have different shapes (e.g., scalar + vector due to implicit broadcast),
        we take the "largest" shape — the one with the most elements.

        Falls back to _propagate_shape_from_type if no operand shapes are
        available.
        """
        best_shape = ()
        for op_id in ssa.operand_ids:
            s = self._get_shape(op_id)
            if len(s) > len(best_shape):
                best_shape = s
            elif len(s) == len(best_shape):
                # Same rank — pick the one with more total elements
                if _shape_numel(s) > _shape_numel(best_shape):
                    best_shape = s
        if best_shape != ():
            self.env_shapes[ssa.id] = best_shape
        else:
            self._propagate_shape_from_type(ssa)

    @property
    def _lid_expr(self):
        """Return the per-element index expression.

        When total elements > 1024 and a wrapping loop is active,
        returns '_loop_e' (the loop variable). Otherwise returns 'lid'.
        """
        return "_loop_e" if getattr(self, "_needs_wrapping", False) else "lid"

    # -- Multi-pass reduction helpers ------------------------------------------

    def _split_ops_by_reductions(self):
        """Split ops into phases separated by tt.reduce ops.

        Returns a list of (ops_list, is_reduce) tuples. Reduce ops are
        isolated in their own single-element phases so they can be emitted
        between per-element loops.
        """
        phases = []
        current_phase = []
        for ssa in self.graph.ops:
            if ssa.op == "tt.reduce":
                if current_phase:
                    phases.append((current_phase, False))
                phases.append(([ssa], True))
                current_phase = []
            else:
                current_phase.append(ssa)
        if current_phase:
            phases.append((current_phase, False))
        return phases

    def _get_reduce_combine_info(self, ssa):
        """Extract combine op and identity from a tt.reduce's body region.

        Returns (combine_op, identity_literal) where combine_op is one of
        'sum', 'max', 'min' and identity_literal is the MSL identity value.
        """
        combine_op = "sum"
        if ssa.region_ops:
            has_cmpf_gt = False
            has_cmpf_lt = False
            for body_op in ssa.region_ops:
                op_name = body_op.op
                if "addf" in op_name or "addi" in op_name:
                    combine_op = "sum"
                elif "max" in op_name:
                    combine_op = "max"
                elif "min" in op_name:
                    combine_op = "min"
                elif op_name == "arith.cmpf":
                    pred = body_op.attrs.get("predicate_name", "")
                    if "gt" in pred:
                        has_cmpf_gt = True
                    elif "lt" in pred:
                        has_cmpf_lt = True
            if combine_op == "sum" and has_cmpf_gt:
                combine_op = "max"
            elif combine_op == "sum" and has_cmpf_lt:
                combine_op = "min"

        identities = {"sum": "0.0f", "max": "-INFINITY", "min": "INFINITY"}
        return combine_op, identities.get(combine_op, "0.0f")

    def _collect_tensor_deps(self, target_ops, all_preceding_ops, reduce_result_ids):
        """Find all ops from earlier phases needed to compute target_ops.

        Walks backward through operand_ids from target_ops, collecting any
        ops from all_preceding_ops whose results are tensor-shaped (per-element)
        and therefore must be re-computed inside the current loop.

        Args:
            target_ops: ops in the current phase that need per-element inputs
            all_preceding_ops: all ops from earlier phases (ordered)
            reduce_result_ids: set of SSA IDs that are reduce results (scalars,
                available outside loops)

        Returns:
            List of ops (in original order) that must be re-emitted in the loop.
        """
        # Build lookup: SSA ID → op
        op_by_id = {}
        for ssa in all_preceding_ops:
            op_by_id[ssa.id] = ssa

        # Collect IDs we need by walking dependencies backward
        needed_ids = set()
        worklist = []
        for ssa in target_ops:
            for dep_id in ssa.operand_ids:
                if dep_id in op_by_id and dep_id not in reduce_result_ids:
                    worklist.append(dep_id)

        while worklist:
            dep_id = worklist.pop()
            if dep_id in needed_ids:
                continue
            if dep_id in reduce_result_ids:
                continue
            if dep_id not in op_by_id:
                continue
            needed_ids.add(dep_id)
            dep_op = op_by_id[dep_id]
            for sub_id in dep_op.operand_ids:
                if sub_id not in needed_ids and sub_id in op_by_id:
                    worklist.append(sub_id)

        # Return in original order
        return [op for op in all_preceding_ops if op.id in needed_ids]

    @staticmethod
    def _is_scalar_op(ssa):
        """Return True if an op produces a scalar value (not per-element).

        Scalar ops can be emitted outside loops because they don't depend
        on the per-element index. This includes program_id, scalar
        constants, scalar arithmetic, and passthrough ops like splat on
        a scalar. Tensor ops (loads, stores, tensor arithmetic) must go
        inside per-element loops.
        """
        # Tensor-flagged ops are per-element
        if ssa.is_tensor:
            return False
        # Loads and stores are always per-element
        if ssa.op in ("tt.load", "tt.store", "tt.atomic_rmw", "tt.atomic_cas"):
            return False
        # tt.reduce is handled separately
        if ssa.op == "tt.reduce":
            return False
        return True

    def _lower_multipass_reduction(self, block_size):
        """Emit multi-pass reduction: per-element loops separated by reductions.

        When a kernel has both per-element ops and reductions, we cannot wrap
        everything in a single loop (threadgroup_barrier inside a loop is UB).
        Instead, split into phases:
        - Non-reduce phases: wrap per-element ops in a for-loop with local
          accumulation for the next reduce
        - Reduce phases: emit SIMD + shared memory reduction outside any loop

        Each phase loop re-computes per-element values from scratch (re-loads
        data) because per-element variables from earlier loops are out of scope.
        Scalar ops are hoisted before their phase's loop so that their variables
        remain in scope for later phases.
        """
        total = self._total_elements
        phases = self._split_ops_by_reductions()

        # Collect all reduce result SSA IDs (scalars available across phases)
        reduce_result_ids = set()
        for ops, is_reduce in phases:
            if is_reduce:
                for ssa in ops:
                    reduce_result_ids.add(ssa.id)
                    if ssa.result_ids:
                        for rid in ssa.result_ids:
                            reduce_result_ids.add(rid)

        # Also include function arg IDs as "always available" scalars
        arg_ids = {a.id for a in self.graph.args}

        # Track all preceding non-reduce ops for dependency resolution
        all_preceding_ops = []
        # Track which scalar ops have already been lowered (by SSA id)
        lowered_scalar_ids = set()

        for phase_idx, (phase_ops, is_reduce) in enumerate(phases):
            if is_reduce:
                # Lower the reduce op outside any loop.
                # The reduce's input is already set to the accumulator variable
                # (overridden in self.env by the preceding phase's accumulation).
                for ssa in phase_ops:
                    self._lower_op(ssa)
                continue

            # Determine if the next phase is a reduce (need accumulation)
            next_reduce = None
            if phase_idx + 1 < len(phases) and phases[phase_idx + 1][1]:
                next_reduce = phases[phase_idx + 1][0][0]

            # Separate scalar ops (hoist before loop) from tensor ops (inside loop)
            scalar_ops = [op for op in phase_ops if self._is_scalar_op(op)]
            tensor_ops = [op for op in phase_ops if not self._is_scalar_op(op)]

            # Check if this phase has any tensor ops that need a loop
            has_tensor_ops = len(tensor_ops) > 0

            # Emit scalar ops BEFORE the loop (they stay in function scope)
            for ssa in scalar_ops:
                if ssa.id not in lowered_scalar_ids:
                    self._lower_op(ssa)
                    lowered_scalar_ids.add(ssa.id)

            if not has_tensor_ops and next_reduce is None:
                # Pure scalar phase — no loop needed
                all_preceding_ops.extend(phase_ops)
                continue

            # Determine which earlier ops need to be re-emitted in this loop
            # for their per-element values to be available
            replay_ops = self._collect_tensor_deps(
                tensor_ops, all_preceding_ops, reduce_result_ids | arg_ids | lowered_scalar_ids
            )

            # Also hoist scalar deps from replay_ops before the loop
            replay_scalar = [op for op in replay_ops if self._is_scalar_op(op)]
            replay_tensor = [op for op in replay_ops if not self._is_scalar_op(op)]
            for ssa in replay_scalar:
                if ssa.id not in lowered_scalar_ids:
                    self._lower_op(ssa)
                    lowered_scalar_ids.add(ssa.id)

            # If this phase precedes a reduce, declare the accumulator
            acc_var = None
            if next_reduce:
                combine_op, identity = self._get_reduce_combine_info(next_reduce)
                acc_var = f"_local_acc_{self._shared_counter}"
                # Determine accumulator type from the reduce input
                reduce_input_dtype = self.env_types.get(
                    next_reduce.operand_ids[0], "fp32") if next_reduce.operand_ids else "fp32"
                is_int_reduce = not (
                    reduce_input_dtype.startswith("fp") or reduce_input_dtype.startswith("bf")
                )
                acc_msl_type = "int" if is_int_reduce else "float"
                # Use type-appropriate identity values
                if is_int_reduce:
                    int_identities = {"sum": "0", "max": "INT_MIN", "min": "INT_MAX"}
                    identity = int_identities.get(combine_op, "0")
                self.kb.raw_line(f"    {acc_msl_type} {acc_var} = {identity};")

            # Open the per-element loop
            self._needs_wrapping = True
            self.kb.raw_line(
                f"    for (uint _loop_e = lid; _loop_e < {total}u; "
                f"_loop_e += {block_size}u) {{"
            )

            # Re-emit tensor dependency ops from earlier phases
            for ssa in replay_tensor:
                self._lower_op(ssa)

            # Emit this phase's tensor ops inside the loop
            for ssa in tensor_ops:
                self._lower_op(ssa)

            # Accumulate into the local variable for the next reduce
            if next_reduce and acc_var:
                reduce_input_id = next_reduce.operand_ids[0]
                input_var = self._lookup(reduce_input_id)
                # Cast input to accumulator type to avoid Metal ambiguity
                cast_input = f"({acc_msl_type}){input_var}"
                if combine_op == "sum":
                    self.kb.raw_line(f"        {acc_var} += {cast_input};")
                elif combine_op == "max":
                    self.kb.raw_line(f"        {acc_var} = max({acc_var}, {cast_input});")
                elif combine_op == "min":
                    self.kb.raw_line(f"        {acc_var} = min({acc_var}, {cast_input});")

            # Close the loop
            self.kb.raw_line(f"    }}")
            self._needs_wrapping = False

            # Override the reduce's input to point to the accumulator
            if next_reduce and acc_var:
                reduce_input_id = next_reduce.operand_ids[0]
                self.env[reduce_input_id] = acc_var

            # Add this phase's ops to the preceding ops for future phases
            all_preceding_ops.extend(phase_ops)

    def lower(self) -> str:
        """Lower the IRGraph to MSL source code."""
        # Check for tt.dot — switch to prebuilt matmul template
        if self._requires_matmul_template():
            msl = self._lower_dot_via_prebuilt_template()
            # Matmul template needs block_m * block_n threads (typically 1024)
            self.effective_block_size = self._matmul_block_size
            return msl

        # Check for 3D reduce — switch to prebuilt template
        reduce_3d_info = self._detect_3d_reduce()
        if reduce_3d_info:
            if reduce_3d_info["combine_op"] in ("argmin", "argmax"):
                msl = self._lower_3d_argminmax_template(reduce_3d_info)
            else:
                msl = self._lower_3d_reduce_template(reduce_3d_info)
            self.effective_block_size = reduce_3d_info["block_size"]
            return msl

        # Detect 2D kernel patterns (expand_dims + broadcast)
        self._prescan_2d_info()

        # Use BLOCK_SIZE from the kernel (graph.block_size), not num_warps * 32.
        # For 2D kernels, block_size = product of all dims.
        # For scalar-only kernels (no tt.make_range), use 1 thread.
        if self._is_2d and self._effective_2d_shape:
            block_size = 1
            for d in self._effective_2d_shape:
                block_size *= d
        else:
            block_size = self.graph.block_size
        if not self._has_tensor_ops():
            block_size = 1

        # For kernels with constant tensors (e.g. tl.full) but no make_range,
        # graph.block_size may be too small (defaults to num_warps*32).
        # Scan tensor type_strs to find the actual max tensor size.
        max_tensor_size = block_size
        for ssa in self.graph.ops:
            shape = _extract_shape(ssa.type_str)
            if shape:
                total = 1
                for d in shape:
                    total *= d
                if total > max_tensor_size:
                    max_tensor_size = total
        if max_tensor_size > block_size and max_tensor_size <= 1024:
            block_size = max_tensor_size

        # If total elements exceed the thread count, use a wrapping loop so
        # each thread processes multiple elements.
        self._needs_wrapping = False
        self._total_elements = block_size

        # Determine optimal thread count from TTGIR layout.
        # When sizePerThread > 1, Triton expects fewer threads each handling
        # multiple elements. Use num_warps * warp_size as the thread count
        # and emit a per-thread loop for the extra elements.
        size_per_thread = 1
        if self.graph.size_per_thread:
            for s in self.graph.size_per_thread:
                size_per_thread *= s

        has_reduce_ops = any(
            ssa.op == "tt.reduce" for ssa in self.graph.ops
        )
        has_barrier_ops = any(
            ssa.op in ("tt.reduce", "tt.scan", "tt.debug_barrier", "tt.trans")
            for ssa in self.graph.ops
        )
        # Multi-value reduces (argmin/argmax) need per-element indices which
        # are incompatible with the multi-pass accumulation loop (the loop
        # variable goes out of scope before the reduce handler runs).
        has_multivalue_reduce = any(
            ssa.op == "tt.reduce" and ssa.result_ids and len(ssa.result_ids) >= 2
            for ssa in self.graph.ops
        )
        num_threads = self.graph.num_warps * 32

        # Detect if this 2D kernel has axis-specific reductions that produce
        # per-row/per-column results (not full-array reductions).
        # Multipass is incompatible with these because the per-thread
        # accumulator mixes values from different rows/columns.
        has_2d_axis_reduce = False
        if self._is_2d and self._effective_2d_shape and has_reduce_ops:
            for ssa in self.graph.ops:
                if ssa.op == "tt.reduce" and ssa.operand_ids:
                    reduce_axis = ssa.attrs.get("axis", 0)
                    # Extract shape from the reduce input's type_str in the IR
                    inp_type = self._find_op_type_str(ssa.operand_ids[0])
                    inp_shape = _extract_shape(inp_type) if inp_type else None
                    # A true 2D reduce: multi-row/col input with axis-specific reduction
                    if inp_shape and len(inp_shape) >= 2:
                        if inp_shape[0] > 1 and reduce_axis == 1:
                            has_2d_axis_reduce = True
                        elif inp_shape[1] > 1 and reduce_axis == 0:
                            has_2d_axis_reduce = True

        # Decide wrapping strategy:
        # 1. sizePerThread > 1 with reductions → multi-pass reduction
        # 2. sizePerThread > 1 without barriers → simple wrapping loop
        # 3. block_size > 1024 with reductions → multi-pass reduction
        # 4. block_size > 1024 without reductions → simple wrapping loop (capped at 1024)
        #
        # EXCEPTION: 2D kernels with axis-specific reductions (dim_0 > 1)
        # cannot use multipass because the per-thread accumulator mixes
        # values across rows. For these, keep block_size = total (up to 1024)
        # so each thread handles exactly one element and _lower_reduce_2d
        # can correctly collect all values in shared memory.
        use_multipass = False
        if has_2d_axis_reduce and block_size <= 1024:
            # Skip multipass; use full block_size with one element per thread.
            # _lower_reduce_2d handles the sequential reduction internally.
            pass
        elif has_multivalue_reduce and block_size <= 1024:
            # Multi-value reduces (argmin/argmax) need per-element indices that
            # go out of scope in the multi-pass accumulation loop. Use full
            # block_size with one element per thread.
            pass
        elif size_per_thread > 1 and block_size > num_threads:
            if has_reduce_ops:
                use_multipass = True
                self._total_elements = block_size
                block_size = num_threads
            elif not has_barrier_ops:
                self._needs_wrapping = True
                self._total_elements = block_size
                block_size = num_threads
        elif block_size > 1024:
            if has_reduce_ops:
                use_multipass = True
                self._total_elements = block_size
                block_size = 1024
            else:
                self._needs_wrapping = True
                self._total_elements = block_size
                block_size = 1024  # Cap dispatch to Metal max

        self.effective_block_size = block_size

        self.kb = KernelBuilder(self.graph.func_name, block_size=block_size)

        # Generate device functions for noinline callees (must appear before kernel)
        if self.graph.called_funcs:
            self._lower_called_funcs()

        # Pre-scan stores to identify output pointers
        self._prescan_stores()

        # Register function arguments
        self._register_args()

        if use_multipass:
            # Multi-pass reduction: split kernel into phases separated by
            # reductions, wrap each phase in a per-element loop, emit
            # reductions between loops operating on thread-local accumulators.
            self._lower_multipass_reduction(block_size)
        else:
            # Standard path: single wrapping loop or no loop
            if self._needs_wrapping:
                self.kb.raw_line(f"    for (uint _loop_e = lid; _loop_e < {self._total_elements}u; _loop_e += {block_size}u) {{")

            # Lower each op
            for ssa in self.graph.ops:
                self._lower_op(ssa)

            # Close wrapping loop
            if self._needs_wrapping:
                self.kb.raw_line(f"    }}")

        # Propagate flags to KernelBuilder for MSL emission
        if self._needs_num_programs:
            self.kb._needs_num_programs = True
        if self._used_pid_axes:
            self.kb._used_pid_axes = self._used_pid_axes

        return self.kb.build()

    def get_output_arg_indices(self):
        """Return list of arg positions that are output (stored-to) pointers.

        Must be called after lower(). Returns None if _prescan_stores()
        was not called (e.g., matmul template path), which means the
        driver should conservatively copy back all tensors.
        """
        if not hasattr(self, "_output_arg_ids") or not self._output_arg_ids:
            return None
        indices = []
        for i, arg in enumerate(self.graph.args):
            if arg.id in self._output_arg_ids:
                indices.append(i)
        return indices

    def _requires_matmul_template(self) -> bool:
        """Check if the kernel contains tt.dot (matrix multiply).

        When tt.dot is present, the generic 1D-per-thread lowerer cannot
        produce correct code — tt.dot requires simdgroup_matrix 8×8 MMA
        with threadgroup memory staging and 2D tile accumulation. We
        route these kernels to a prebuilt tiled matmul MSL template.
        """
        for ssa in self.graph.ops:
            if ssa.op == "tt.dot":
                return True
            if ssa.region_ops:
                for body_op in ssa.region_ops:
                    if body_op.op == "tt.dot":
                        return True
        return False

    def _detect_3d_reduce(self):
        """Detect if this kernel is a simple 3D reduce that needs a template.

        Returns dict with shape/axis info if detected, None otherwise.
        Detects both regular reduce (sum/max/min) and argmin/argmax (2 operands).

        Only triggers for simple kernels (load→reduce→store). Complex kernels
        with scf.for loops, multiple reduces, or multi-axis grids must go
        through the generic op-by-op lowerer instead.
        """
        # Reject complex kernels that need op-by-op lowering
        has_scf_for = False
        has_num_programs = False
        reduce_count = 0
        for ssa in self.graph.ops:
            if ssa.op == "scf.for":
                has_scf_for = True
            elif ssa.op == "tt.get_num_programs":
                has_num_programs = True
            elif ssa.op == "tt.reduce":
                reduce_count += 1
        if has_scf_for or has_num_programs or reduce_count > 1:
            return None

        # Look for tt.reduce with a 3D input
        for ssa in self.graph.ops:
            if ssa.op == "tt.reduce" and ssa.operand_ids:
                # Check input shape
                input_type = self._find_op_type_str(ssa.operand_ids[0])
                if input_type:
                    input_shape = _extract_shape(input_type)
                    if input_shape and len(input_shape) == 3:
                        axis = ssa.attrs.get("axis", 0)
                        # Detect argmin/argmax: 2 operands (values, indices)
                        is_argminmax = len(ssa.operand_ids) >= 2
                        # Determine combine op
                        combine_op = "sum"
                        if ssa.region_ops:
                            for body_op in ssa.region_ops:
                                if "max" in body_op.op:
                                    combine_op = "max"
                                elif "min" in body_op.op:
                                    combine_op = "min"
                                elif "addf" in body_op.op or "addi" in body_op.op:
                                    combine_op = "sum"
                                elif body_op.op == "arith.cmpf":
                                    pred = body_op.attrs.get("predicate_name", "")
                                    if "gt" in pred:
                                        combine_op = "max"
                                    elif "lt" in pred:
                                        combine_op = "min"
                        if is_argminmax:
                            # argmin uses cmpf(olt) → detected as "min"
                            # argmax uses cmpf(ogt) → detected as "max"
                            combine_op = "argmin" if combine_op == "min" else "argmax"
                        M, N, K = input_shape
                        total = M * N * K
                        # Use block_size that covers all elements
                        block_size = max(total, self.graph.num_warps * 32)
                        # Cap at 1024 (Metal max threads per threadgroup)
                        block_size = min(block_size, 1024)
                        return {
                            "shape": (M, N, K),
                            "axis": axis,
                            "combine_op": combine_op,
                            "block_size": block_size,
                        }
        return None

    def _lower_3d_reduce_template(self, info) -> str:
        """Generate a complete MSL kernel for 3D axis reduction.

        Bypasses the generic lowerer since 3D→2D dimensionality change
        breaks the per-thread index decomposition.
        """
        M, N, K = info["shape"]
        axis = info["axis"]
        combine_op = info["combine_op"]
        block_size = info["block_size"]
        total = M * N * K

        # Determine result dimensions
        if axis == 0:
            result_dims = (N, K)
            result_total = N * K
            axis_size = M
        elif axis == 1:
            result_dims = (M, K)
            result_total = M * K
            axis_size = N
        else:  # axis == 2
            result_dims = (M, N)
            result_total = M * N
            axis_size = K

        # Determine data type from pointer args
        ptr_args = [a for a in self.graph.args if a.is_ptr]
        msl_type = "float"
        if ptr_args:
            elem = ptr_args[0].elem_type
            if elem in ("i32", "si32"):
                msl_type = "int"

        # Identity and combine expression
        if combine_op == "sum":
            identity = "0.0f" if msl_type == "float" else "0"
            combine_expr = "acc + val"
        elif combine_op == "max":
            identity = "(-INFINITY)" if msl_type == "float" else "INT_MIN"
            combine_expr = "fmax(acc, val)" if msl_type == "float" else "max(acc, val)"
        elif combine_op == "min":
            identity = "INFINITY" if msl_type == "float" else "INT_MAX"
            combine_expr = "fmin(acc, val)" if msl_type == "float" else "min(acc, val)"
        else:
            identity = "0.0f" if msl_type == "float" else "0"
            combine_expr = "acc + val"

        safe_name = _sanitize_msl_name(self.graph.func_name)

        # Build argument list (X and Z pointers)
        arg_decls = []
        for i, arg in enumerate(self.graph.args):
            if arg.is_ptr:
                arg_msl_type = triton_type_to_msl(arg.elem_type)
                arg_decls.append(
                    f"    device {arg_msl_type}* {arg.name} [[buffer({i})]]")
            else:
                arg_decls.append(
                    f"    device int* {arg.name}_buf [[buffer({i})]]")

        x_name = ptr_args[0].name if ptr_args else "X"
        z_name = ptr_args[1].name if len(ptr_args) > 1 else "Z"

        lines = []
        lines.append("#include <metal_stdlib>")
        lines.append("using namespace metal;")
        lines.append("")
        lines.append(f"kernel void {safe_name}(")
        lines.append(",\n".join(arg_decls) + ",")
        lines.append("    uint pid [[threadgroup_position_in_grid]],")
        lines.append("    uint lid [[thread_position_in_threadgroup]],")
        lines.append("    uint tid [[thread_position_in_grid]]")
        lines.append(") {")

        # Shared memory for input staging and result
        lines.append(f"    threadgroup {msl_type} _input[{total}];")
        lines.append(f"    threadgroup {msl_type} _result[{result_total}];")

        # Stage all input values to shared memory
        lines.append(f"    for (uint _e = lid; _e < {total}u; _e += {block_size}u)")
        lines.append(f"        _input[_e] = ({msl_type}){x_name}[_e];")
        lines.append(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Reduce along axis
        lines.append(f"    for (uint _r = lid; _r < {result_total}u; _r += {block_size}u) {{")
        lines.append(f"        {msl_type} acc = {identity};")

        if axis == 0:
            lines.append(f"        uint _j = _r / {K}u;")
            lines.append(f"        uint _k = _r % {K}u;")
            lines.append(f"        for (uint _a = 0; _a < {axis_size}u; _a++) {{")
            lines.append(f"            {msl_type} val = _input[_a * {N * K}u + _j * {K}u + _k];")
        elif axis == 1:
            lines.append(f"        uint _i = _r / {K}u;")
            lines.append(f"        uint _k = _r % {K}u;")
            lines.append(f"        for (uint _a = 0; _a < {axis_size}u; _a++) {{")
            lines.append(f"            {msl_type} val = _input[_i * {N * K}u + _a * {K}u + _k];")
        else:  # axis == 2
            lines.append(f"        uint _i = _r / {N}u;")
            lines.append(f"        uint _j = _r % {N}u;")
            lines.append(f"        for (uint _a = 0; _a < {axis_size}u; _a++) {{")
            lines.append(f"            {msl_type} val = _input[_i * {N * K}u + _j * {K}u + _a];")

        lines.append(f"            acc = {combine_expr};")
        lines.append(f"        }}")
        lines.append(f"        _result[_r] = acc;")
        lines.append(f"    }}")
        lines.append(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Store results to output (row-major 2D)
        R0, R1 = result_dims
        lines.append(f"    for (uint _r = lid; _r < {result_total}u; _r += {block_size}u)")
        lines.append(f"        {z_name}[_r] = _result[_r];")

        lines.append("}")
        lines.append("")

        return "\n".join(lines)

    def _lower_3d_argminmax_template(self, info) -> str:
        """Generate a complete MSL kernel for 3D argmin/argmax.

        Similar to _lower_3d_reduce_template but tracks both value and index,
        storing only the index result.
        """
        M, N, K = info["shape"]
        axis = info["axis"]
        combine_op = info["combine_op"]
        block_size = info["block_size"]
        total = M * N * K
        is_max = (combine_op == "argmax")

        # Determine result dimensions
        if axis == 0:
            result_dims = (N, K)
            result_total = N * K
            axis_size = M
        elif axis == 1:
            result_dims = (M, K)
            result_total = M * K
            axis_size = N
        else:  # axis == 2
            result_dims = (M, N)
            result_total = M * N
            axis_size = K

        # Determine data type from pointer args
        ptr_args = [a for a in self.graph.args if a.is_ptr]
        msl_type = "float"
        if ptr_args:
            elem = ptr_args[0].elem_type
            if elem in ("i32", "si32"):
                msl_type = "int"

        identity = "(-INFINITY)" if is_max and msl_type == "float" else "INFINITY"
        if msl_type == "int":
            identity = "INT_MIN" if is_max else "INT_MAX"
        cmp_op = ">" if is_max else "<"

        safe_name = _sanitize_msl_name(self.graph.func_name)

        # Build argument list
        arg_decls = []
        for i, arg in enumerate(self.graph.args):
            if arg.is_ptr:
                arg_msl_type = triton_type_to_msl(arg.elem_type)
                arg_decls.append(
                    f"    device {arg_msl_type}* {arg.name} [[buffer({i})]]")
            else:
                arg_decls.append(
                    f"    device int* {arg.name}_buf [[buffer({i})]]")

        x_name = ptr_args[0].name if ptr_args else "X"
        z_name = ptr_args[1].name if len(ptr_args) > 1 else "Z"

        lines = []
        lines.append("#include <metal_stdlib>")
        lines.append("using namespace metal;")
        lines.append("")
        lines.append(f"kernel void {safe_name}(")
        lines.append(",\n".join(arg_decls) + ",")
        lines.append("    uint pid [[threadgroup_position_in_grid]],")
        lines.append("    uint lid [[thread_position_in_threadgroup]],")
        lines.append("    uint tid [[thread_position_in_grid]]")
        lines.append(") {")

        # Shared memory for input staging and results
        lines.append(f"    threadgroup {msl_type} _input[{total}];")
        lines.append(f"    threadgroup int _result_idx[{result_total}];")

        # Stage all input values to shared memory
        lines.append(f"    for (uint _e = lid; _e < {total}u; _e += {block_size}u)")
        lines.append(f"        _input[_e] = ({msl_type}){x_name}[_e];")
        lines.append(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Reduce along axis, tracking best value and index
        lines.append(f"    for (uint _r = lid; _r < {result_total}u; _r += {block_size}u) {{")
        lines.append(f"        {msl_type} best_v = {identity};")
        lines.append(f"        int best_i = 0;")

        if axis == 0:
            lines.append(f"        uint _j = _r / {K}u;")
            lines.append(f"        uint _k = _r % {K}u;")
            lines.append(f"        for (uint _a = 0; _a < {axis_size}u; _a++) {{")
            lines.append(f"            {msl_type} val = _input[_a * {N * K}u + _j * {K}u + _k];")
        elif axis == 1:
            lines.append(f"        uint _i = _r / {K}u;")
            lines.append(f"        uint _k = _r % {K}u;")
            lines.append(f"        for (uint _a = 0; _a < {axis_size}u; _a++) {{")
            lines.append(f"            {msl_type} val = _input[_i * {N * K}u + _a * {K}u + _k];")
        else:  # axis == 2
            lines.append(f"        uint _i = _r / {N}u;")
            lines.append(f"        uint _j = _r % {N}u;")
            lines.append(f"        for (uint _a = 0; _a < {axis_size}u; _a++) {{")
            lines.append(f"            {msl_type} val = _input[_i * {N * K}u + _j * {K}u + _a];")

        lines.append(f"            if (val {cmp_op} best_v || (val == best_v && (int)_a < best_i)) {{")
        lines.append(f"                best_v = val; best_i = (int)_a;")
        lines.append(f"            }}")
        lines.append(f"        }}")
        lines.append(f"        _result_idx[_r] = best_i;")
        lines.append(f"    }}")
        lines.append(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Store index results to output
        lines.append(f"    for (uint _r = lid; _r < {result_total}u; _r += {block_size}u)")
        lines.append(f"        {z_name}[_r] = _result_idx[_r];")

        lines.append("}")
        lines.append("")

        return "\n".join(lines)

    def _lower_dot_via_prebuilt_template(self) -> str:
        """Generate strided matmul MSL for kernels containing tt.dot.

        Generates a scalar-loop matmul that handles arbitrary M, N, K
        and strided access patterns. Each thread computes one or more
        output elements via a K-loop, reading directly from global memory.

        For kernels with strides (upstream test_dot), this correctly handles
        row-major, column-major, and transposed layouts.

        For simple 3-pointer kernels (A, B, C with no strides), falls back
        to the optimized simdgroup_matrix template.
        """
        # Extract tile dimensions from tt.make_range ops (for simple template fallback)
        tile_dims = []
        for ssa in self.graph.ops:
            if ssa.op == "tt.make_range":
                end = ssa.attrs.get("end", 32)
                if end not in tile_dims:
                    tile_dims.append(end)

        ptr_args = [a for a in self.graph.args if a.is_ptr]
        scalar_args = [a for a in self.graph.args if not a.is_ptr]

        # Determine dtype from pointer args
        dtype = "fp32"
        if ptr_args:
            dtype = _mlir_to_triton_dtype(ptr_args[0].elem_type)

        # Check if this is a strided kernel (has stride args) vs simple kernel
        has_strides = any("stride" in a.name.lower() for a in scalar_args)
        # If the kernel uses program_id, it's a pid-tiled kernel that
        # needs the simdgroup matmul template with block selection.
        has_pid = any(ssa.op == "tt.get_program_id" for ssa in self.graph.ops)

        # Detect 3D batched dot (e.g. test_dot3d): tt.dot on tensor<BxMxNxT>
        # 3D dot has both strides AND pids — strides for batch/row/col access,
        # pids for spatial tiling over M and N. Use strided template, not simdgroup.
        is_3d_dot = False
        for ssa in self.graph.ops:
            if ssa.op == "tt.dot":
                dot_shape = _extract_shape(ssa.type_str)
                if len(dot_shape) >= 3:
                    is_3d_dot = True
                break

        if not has_strides or (has_pid and not is_3d_dot):
            # Check for constant-input dot (e.g., test_dot_without_load)
            const_info = self._detect_dot_constant_inputs()
            if const_info:
                return self._lower_dot_constant_template(const_info, ptr_args)
            # Fall back to optimized simdgroup template for pid-tiled kernels
            return self._lower_dot_simple_template(tile_dims, ptr_args, dtype)

        # --- Strided dot kernel generation ---
        # Extract M, N, K from tt.dot operand type shapes (reliable).
        # 2D: tensor<MxKxT> * tensor<KxNxT> -> tensor<MxNxT>
        # 3D: tensor<BxMxKxT> * tensor<BxKxNxT> -> tensor<BxMxNxT>
        M, N, K = 32, 32, 32  # fallback
        B_batch = 1
        for ssa in self.graph.ops:
            if ssa.op == "tt.dot":
                dot_shape = _extract_shape(ssa.type_str)
                if len(dot_shape) >= 3:
                    B_batch = dot_shape[0]
                    M, N = dot_shape[1], dot_shape[2]
                elif len(dot_shape) >= 2:
                    M, N = dot_shape[0], dot_shape[1]
                # Get K from first operand shape
                # 2D: [M, K], 3D: [B, M, K]
                if ssa.operand_ids:
                    for ssa2 in self.graph.ops:
                        if ssa2.id == ssa.operand_ids[0]:
                            op_shape = _extract_shape(ssa2.type_str)
                            if len(op_shape) >= 3:
                                K = op_shape[2]
                            elif len(op_shape) >= 2:
                                K = op_shape[1]
                            break
                break

        # Detect accumulator initialization from the IR
        has_accumulator_load = False
        for ssa in self.graph.ops:
            if ssa.op == "arith.addf" and any(
                self.graph.ops[i].op == "tt.dot"
                for i, op in enumerate(self.graph.ops)
                if op.id in (ssa.operand_ids or [])
            ):
                has_accumulator_load = True

        # MSL type mapping
        if dtype in ("fp16", "f16"):
            msl_type = "half"
            compute_type = "float"
        elif dtype in ("bf16",):
            msl_type = "bfloat"
            compute_type = "float"
        else:
            msl_type = "float"
            compute_type = "float"

        # Determine output type from tt.dot result
        out_msl_type = "float"  # tt.dot typically outputs f32
        for ssa in self.graph.ops:
            if ssa.op == "tt.dot":
                dot_out_type = ssa.elem_type
                if dot_out_type in ("f16",):
                    out_msl_type = "half"
                elif dot_out_type in ("bf16",):
                    out_msl_type = "bfloat"
                break

        num_warps = self.graph.num_warps
        block_size = min(num_warps * 32, 1024)
        self._matmul_block_size = block_size

        # For 3D dot, signal that the kernel needs 2D grid dispatch
        if is_3d_dot:
            self._used_pid_axes = {0, 1}

        safe_name = _sanitize_msl_name(self.graph.func_name)

        # Build argument list from IR
        arg_decls = []
        for i, arg in enumerate(self.graph.args):
            if arg.is_ptr:
                arg_msl_type = triton_type_to_msl(arg.elem_type)
                arg_decls.append(
                    f"    volatile device {arg_msl_type}* {arg.name} [[buffer({i})]]")
            else:
                arg_decls.append(
                    f"    volatile device int* {arg.name}_buf [[buffer({i})]]")

        # Generate the kernel
        lines = []
        lines.append("#include <metal_stdlib>")
        lines.append("using namespace metal;")
        lines.append("")
        lines.append(f"kernel void {safe_name}(")
        lines.append(",\n".join(arg_decls) + ",")
        # Use uint3 for grid position when multi-axis dispatch is needed
        used_axes = getattr(self.kb, '_used_pid_axes', {0}) if self.kb else {0}
        if is_3d_dot:
            used_axes = {0, 1}  # 3D dot needs pid3.x and pid3.y for spatial tiling
        if max(used_axes) > 0:
            # Metal requires all thread-index attributes to use the same
            # type (all uint or all uint3). Use uint3 for all when multi-axis.
            lines.append("    uint3 pid3 [[threadgroup_position_in_grid]],")
            lines.append("    uint3 _lid3 [[thread_position_in_threadgroup]],")
            lines.append("    uint3 _tid3 [[thread_position_in_grid]]")
            lines.append(") {")
            lines.append("    uint lid = _lid3.x;")
            lines.append("    uint pid = pid3.x;")
            if 1 in used_axes:
                lines.append("    uint pid_y = pid3.y;")
            if 2 in used_axes:
                lines.append("    uint pid_z = pid3.z;")
        else:
            lines.append("    uint pid [[threadgroup_position_in_grid]],")
            lines.append("    uint lid [[thread_position_in_threadgroup]],")
            lines.append("    uint tid [[thread_position_in_grid]]")
            lines.append(") {")

        # Unpack scalar args from buffers
        for arg in self.graph.args:
            if not arg.is_ptr:
                lines.append(f"    int {arg.name} = {arg.name}_buf[0];")

        # Map scalar stride args to their pointer by name prefix.
        # Triton TTGIR folds stride=1 args to constants, so we may have
        # fewer scalars than expected. Name matching is robust to this.
        #
        # Convention: stride_{ptrbase}{dim} where ptrbase is derived from
        # the pointer name (e.g. "x" from "x_ptr", "x_ptr" → "x").
        # 2D: Each pointer gets [dim0_stride, dim1_stride], default "1".
        # 3D: Each pointer gets [batch_stride, dim0_stride, dim1_stride].
        #
        # IMPORTANT: When one stride is folded (e.g. stride_xm=1 for col_a),
        # the remaining stride must go in the CORRECT slot based on its
        # dimension suffix, not just the first empty slot.
        # For dot A[M,K] @ B[K,N] → C[M,N]:
        #   A: 'k'/'1' suffix → dim1, everything else → dim0
        #   B: 'n'/'1' suffix → dim1, everything else → dim0
        #   C: 'n'/'1' suffix → dim1, everything else → dim0
        #   W: 'l'/'1' suffix → dim1, everything else → dim0
        # For 3D dot A[B,M,K] @ B[B,K,N] → C[B,M,N]:
        #   'b' suffix → batch slot (slot 0), dim suffixes → slots 1,2
        if is_3d_dot:
            stride_map = {p.name: ["1", "1", "1"] for p in ptr_args}
        else:
            stride_map = {p.name: ["1", "1"] for p in ptr_args}

        # Define which suffix characters map to dim1 for each pointer position
        _dim1_suffixes = {
            0: {'k', '1'},    # A (MxK): K-stride is dim1
            1: {'n', '1'},    # B (KxN): N-stride is dim1
        }
        # Last pointer (C/Z) and chain-dot W
        _dim1_last = {'n', '1'}   # C (MxN): N-stride is dim1
        if len(ptr_args) >= 4:
            _dim1_suffixes[2] = {'l', '1'}  # W (NxL): L-stride is dim1

        matched_strides = set()
        for sarg in scalar_args:
            sname = sarg.name.lower()
            if "stride" not in sname:
                continue
            # Match to pointer by name prefix (case-insensitive)
            for pi, p in enumerate(ptr_args):
                base = p.name
                if base.endswith("_ptr"):
                    base = base[:-4]
                base_lower = base.lower()
                prefix = f"stride_{base_lower}"
                alt_prefix = f"s_{base_lower}"
                # Also try reverse pattern: {base}_stride (e.g., "in_stride" for "in1_ptr")
                rev_prefix = f"{base_lower}_stride"
                # Try without trailing digits: "in_stride" matches "in1_ptr"
                base_nodigit = base_lower.rstrip("0123456789")
                rev_prefix_nodigit = f"{base_nodigit}_stride" if base_nodigit != base_lower else ""
                # Try prefix match: "out_stride" matches "output_ptr" (stride base is prefix of ptr base)
                stride_base = sname.replace("_stride", "").replace("stride_", "").replace("s_", "")
                if sname.startswith(prefix):
                    suffix = sname[len(prefix):]
                elif sname.startswith(alt_prefix):
                    suffix = sname[len(alt_prefix):]
                elif sname == rev_prefix or sname.startswith(rev_prefix + "_"):
                    suffix = sname[len(rev_prefix):]
                elif rev_prefix_nodigit and (sname == rev_prefix_nodigit or sname.startswith(rev_prefix_nodigit + "_")):
                    suffix = sname[len(rev_prefix_nodigit):]
                elif stride_base and base_lower.startswith(stride_base) and sname.endswith("_stride"):
                    suffix = ""
                else:
                    continue

                dims = stride_map[p.name]
                # Determine correct dim slot from suffix
                is_last = (pi == len(ptr_args) - 1)
                dim1_chars = _dim1_last if is_last else _dim1_suffixes.get(pi, {'1'})
                if is_3d_dot and suffix and suffix[0] == 'b':
                    # Batch stride → slot 0
                    dims[0] = sarg.name
                elif suffix and suffix[0] in dim1_chars:
                    # Inner dimension (K for A, N for B/C)
                    dims[-1] = sarg.name
                else:
                    # Outer dimension (M for A, K for B, M for C)
                    dims[1 if is_3d_dot else 0] = sarg.name
                matched_strides.add(sarg.name)
                break

        # Positional fallback: if stride args remain unmatched and pointers
        # have default strides, assign remaining strides positionally to dim0
        if has_strides and len(matched_strides) == 0:
            unmatched = [s for s in scalar_args if "stride" in s.name.lower()]
            for si, sarg in enumerate(unmatched):
                if si < len(ptr_args):
                    stride_map[ptr_args[si].name][0] = sarg.name

        if len(ptr_args) < 3:
            return self._lower_dot_simple_template(tile_dims, ptr_args, dtype)

        a_ptr = ptr_args[0]
        b_ptr = ptr_args[1]
        c_ptr = ptr_args[-1]

        if is_3d_dot:
            a_sb, a_s0, a_s1 = stride_map[a_ptr.name]
            b_sb, b_s0, b_s1 = stride_map[b_ptr.name]
            c_sb, c_s0, c_s1 = stride_map[c_ptr.name]
        else:
            a_sb = b_sb = c_sb = "0"
            a_s0, a_s1 = stride_map[a_ptr.name]
            b_s0, b_s1 = stride_map[b_ptr.name]
            c_s0, c_s1 = stride_map[c_ptr.name]

        c_type = triton_type_to_msl(c_ptr.elem_type)

        # Detect epilogue from IR ops after tt.dot
        epilogue = self._detect_dot_epilogue()

        # For chain-dot, we need W pointer and strides
        w_ptr = None
        w_s0 = "1"
        w_s1 = "1"
        if epilogue == "chain-dot" and len(ptr_args) >= 4 and not is_3d_dot:
            w_ptr = ptr_args[2]
            w_s0, w_s1 = stride_map[w_ptr.name]

        # Shared memory for epilogues that need staging
        total = M * N
        if epilogue in ("softmax", "chain-dot"):
            lines.append(f"    threadgroup {compute_type} _shared_c[{total}];")
        elif epilogue == "add-matrix":
            # Stage Z bias into shared memory to avoid aliasing (Z is both source and output)
            lines.append(f"    threadgroup {compute_type} _bias[{total}];")
            lines.append(f"    for (uint _e = lid; _e < {total}u; _e += {block_size}u) {{")
            lines.append(f"        uint _i = _e / {N}u;")
            lines.append(f"        uint _j = _e % {N}u;")
            lines.append(f"        _bias[_e] = ({compute_type}){c_ptr.name}[_i * {c_s0} + _j * {c_s1}];")
            lines.append(f"    }}")
            lines.append(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")
        elif epilogue == "add-rows":
            lines.append(f"    threadgroup {compute_type} _bias[{M}];")
            lines.append(f"    if (lid < {M}u) _bias[lid] = ({compute_type}){c_ptr.name}[lid * {c_s0}];")
            lines.append(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")
        elif epilogue == "add-cols":
            lines.append(f"    threadgroup {compute_type} _bias[{N}];")
            lines.append(f"    if (lid < {N}u) _bias[lid] = ({compute_type}){c_ptr.name}[lid * {c_s1}];")
            lines.append(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Emit the matmul loop
        if is_3d_dot:
            total = B_batch * M * N
            lines.append(f"    // 3D strided dot: [{B_batch}x{M}x{K}] @ [{B_batch}x{K}x{N}] -> [{B_batch}x{M}x{N}]")
            lines.append(f"    uint _pid_m_off = pid3.x * {M}u;")
            lines.append(f"    uint _pid_n_off = pid3.y * {N}u;")
            lines.append(f"    for (uint _e = lid; _e < {total}u; _e += {block_size}u) {{")
            lines.append(f"        uint _b = _e / {M * N}u;")
            lines.append(f"        uint _i = (_e % {M * N}u) / {N}u;")
            lines.append(f"        uint _j = _e % {N}u;")
            lines.append(f"        {compute_type} _sum = 0.0f;")
            lines.append(f"        for (uint _k = 0; _k < {K}u; _k++) {{")
            lines.append(f"            _sum += ({compute_type}){a_ptr.name}[_b * {a_sb} + (_pid_m_off + _i) * {a_s0} + _k * {a_s1}]")
            lines.append(f"                  * ({compute_type}){b_ptr.name}[_b * {b_sb} + _k * {b_s0} + (_pid_n_off + _j) * {b_s1}];")
            lines.append(f"        }}")
            lines.append(f"        {c_ptr.name}[_b * {c_sb} + (_pid_m_off + _i) * {c_s0} + (_pid_n_off + _j) * {c_s1}] = ({c_type})_sum;")
            lines.append(f"    }}")
        else:
            lines.append(f"    // Strided dot: [{M}x{K}] @ [{K}x{N}] -> [{M}x{N}], epilogue={epilogue}")
            lines.append(f"    for (uint _e = lid; _e < {total}u; _e += {block_size}u) {{")
            lines.append(f"        uint _i = _e / {N}u;")
            lines.append(f"        uint _j = _e % {N}u;")

            # Initialize accumulator from staged bias or zero
            if epilogue == "add-matrix":
                lines.append(f"        {compute_type} _sum = _bias[_e];")
            elif epilogue == "add-rows":
                lines.append(f"        {compute_type} _sum = _bias[_i];")
            elif epilogue == "add-cols":
                lines.append(f"        {compute_type} _sum = _bias[_j];")
            else:
                lines.append(f"        {compute_type} _sum = 0.0f;")

            lines.append(f"        for (uint _k = 0; _k < {K}u; _k++) {{")
            lines.append(f"            _sum += ({compute_type}){a_ptr.name}[_i * {a_s0} + _k * {a_s1}]")
            lines.append(f"                  * ({compute_type}){b_ptr.name}[_k * {b_s0} + _j * {b_s1}];")
            lines.append(f"        }}")

        # Epilogue handling (2D only — 3D dot loop already includes store)
        if not is_3d_dot:
            if epilogue == "softmax":
                # Store dot result to shared memory for row-wise softmax
                lines.append(f"        _shared_c[_e] = _sum;")
                lines.append(f"    }}")
                lines.append(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")
                # Row-wise softmax via shared memory
                lines.append(f"    for (uint _e = lid; _e < {total}u; _e += {block_size}u) {{")
                lines.append(f"        uint _i = _e / {N}u;")
                lines.append(f"        uint _j = _e % {N}u;")
                lines.append(f"        // Row-wise max")
                lines.append(f"        {compute_type} _row_max = -INFINITY;")
                lines.append(f"        for (uint _c = 0; _c < {N}u; _c++)")
                lines.append(f"            _row_max = fmax(_row_max, _shared_c[_i * {N}u + _c]);")
                lines.append(f"        {compute_type} _exp_val = exp(_shared_c[_e] - _row_max);")
                lines.append(f"        // Row-wise sum of exp")
                lines.append(f"        {compute_type} _row_sum = 0.0f;")
                lines.append(f"        for (uint _c = 0; _c < {N}u; _c++)")
                lines.append(f"            _row_sum += exp(_shared_c[_i * {N}u + _c] - _row_max);")
                lines.append(f"        {c_ptr.name}[_i * {c_s0} + _j * {c_s1}] = ({c_type})(_exp_val / _row_sum);")
                lines.append(f"    }}")
            elif epilogue == "chain-dot" and w_ptr:
                # Store first dot to shared, then second matmul with W
                lines.append(f"        _shared_c[_e] = _sum;")
                lines.append(f"    }}")  # end first matmul loop
                lines.append(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")
                # Second matmul: shared_c[M,N] @ W[N,N] → result[M,N]
                lines.append(f"    for (uint _e = lid; _e < {total}u; _e += {block_size}u) {{")
                lines.append(f"        uint _i = _e / {N}u;")
                lines.append(f"        uint _j = _e % {N}u;")
                lines.append(f"        {compute_type} _sum2 = 0.0f;")
                lines.append(f"        for (uint _k2 = 0; _k2 < {N}u; _k2++) {{")
                lines.append(f"            _sum2 += _shared_c[_i * {N}u + _k2]")
                lines.append(f"                   * ({compute_type}){w_ptr.name}[_k2 * {w_s0} + _j * {w_s1}];")
                lines.append(f"        }}")
                lines.append(f"        {c_ptr.name}[_i * {c_s0} + _j * {c_s1}] = ({c_type})_sum2;")
                lines.append(f"    }}")
            else:
                # Default store (none, trans, add-*)
                lines.append(f"        {c_ptr.name}[_i * {c_s0} + _j * {c_s1}] = ({c_type})_sum;")
                lines.append(f"    }}")

        lines.append("}")
        lines.append("")

        return "\n".join(lines)

    def _detect_dot_epilogue(self) -> str:
        """Detect epilogue pattern from IR around tt.dot.

        The Triton compiler folds add-matrix/add-rows/add-cols into the
        tt.dot's 3rd operand (accumulator). So these are detected from the
        accumulator source, not from ops after the dot.

        Ops AFTER the dot indicate softmax (tt.reduce) or chain-dot (tt.dot).

        Returns one of: 'none', 'add-matrix', 'add-rows', 'add-cols',
                         'softmax', 'chain-dot'
        """
        dot_op = None
        dot_idx = None
        for i, ssa in enumerate(self.graph.ops):
            if ssa.op == "tt.dot":
                dot_op = ssa
                dot_idx = i
                break
        if dot_op is None:
            return "none"

        # Check ops AFTER the dot
        after_dot = self.graph.ops[dot_idx + 1:]
        n_dot2 = sum(1 for op in after_dot if op.op == "tt.dot")
        n_reduce = sum(1 for op in after_dot if op.op == "tt.reduce")

        if n_dot2 >= 1:
            return "chain-dot"
        if n_reduce >= 1:
            return "softmax"

        # Check accumulator (3rd operand of tt.dot).
        # If it traces back to a tt.load, it's an add epilogue.
        # If it traces to a zero constant or arith.constant, it's 'none'.
        if len(dot_op.operand_ids or []) >= 3:
            acc_id = dot_op.operand_ids[2]
            acc_source = self._trace_dot_accumulator(acc_id)
            if acc_source in ("add-matrix", "add-rows", "add-cols"):
                return acc_source

        return "none"

    def _trace_dot_accumulator(self, acc_id) -> str:
        """Trace the 3rd operand of tt.dot to determine accumulator source.

        Returns: 'zero' (default), 'add-matrix', 'add-rows', 'add-cols'
        """
        # Build a quick lookup
        op_map = {ssa.id: ssa for ssa in self.graph.ops}

        # Follow the chain: convert_layout → load, or convert_layout → broadcast → load
        visited = set()
        current = acc_id
        has_broadcast = False
        expand_dims_shape = None  # Track expand_dims output to distinguish rows vs cols

        while current in op_map and current not in visited:
            visited.add(current)
            op = op_map[current]

            if op.op == "tt.load":
                # Found a load — it's an add epilogue
                if has_broadcast:
                    # Use expand_dims shape to distinguish rows vs cols:
                    # (M, 1) = add-rows ([:, None]), (1, N) = add-cols ([None, :])
                    if expand_dims_shape and len(expand_dims_shape) == 2:
                        if expand_dims_shape[0] == 1:
                            return "add-cols"
                        elif expand_dims_shape[1] == 1:
                            return "add-rows"
                    # Fallback: use load shape vs dot shape
                    load_shape = _extract_shape(op.type_str)
                    dot_shape = _extract_shape(
                        op_map[next(i for i in op_map
                                    if op_map[i].op == "tt.dot")].type_str
                    ) if any(op_map[i].op == "tt.dot" for i in op_map) else []
                    if load_shape and dot_shape and len(dot_shape) >= 2:
                        M_dim, N_dim = dot_shape[0], dot_shape[1]
                        load_size = load_shape[0] if len(load_shape) == 1 else max(load_shape)
                        if M_dim != N_dim:
                            if load_size == M_dim:
                                return "add-rows"
                            elif load_size == N_dim:
                                return "add-cols"
                    return "add-rows"  # default broadcast
                return "add-matrix"

            if op.op == "arith.constant":
                return "zero"

            # Follow through passthrough ops
            if op.op in ("ttg.convert_layout", "tt.broadcast",
                         "tt.expand_dims", "tt.splat",
                         "arith.extf", "arith.truncf",
                         "arith.sitofp", "arith.uitofp"):
                if op.op == "tt.broadcast":
                    has_broadcast = True
                if op.op == "tt.expand_dims":
                    has_broadcast = True
                    expand_dims_shape = _extract_shape(op.type_str)
                if op.operand_ids:
                    current = op.operand_ids[0]
                    continue

            # Unknown op — assume it's derived from a computation (zero)
            break

        return "zero"

    def _detect_dot_constant_inputs(self):
        """Check if tt.dot inputs are compile-time constants (arith.constant).

        Returns (const_a, const_b, M, N, K, dot_elem_type) if both inputs
        are constants, or None otherwise.
        """
        import struct as _struct
        op_by_id = {ssa.id: ssa for ssa in self.graph.ops}

        for ssa in self.graph.ops:
            if ssa.op != "tt.dot":
                continue
            if len(ssa.operand_ids) < 2:
                return None
            a_id, b_id = ssa.operand_ids[0], ssa.operand_ids[1]
            a_op = op_by_id.get(a_id)
            b_op = op_by_id.get(b_id)
            if not (a_op and b_op):
                return None
            if a_op.op != "arith.constant" or b_op.op != "arith.constant":
                return None

            def _get_float_val(op):
                v = op.attrs.get("value")
                if v is None:
                    return 0.0
                if isinstance(v, float):
                    return v
                if isinstance(v, int) and op.elem_type in ("f32", "f16", "bf16"):
                    try:
                        return _struct.unpack('f', _struct.pack('I', v & 0xFFFFFFFF))[0]
                    except _struct.error:
                        return 0.0
                return float(v)

            const_a = _get_float_val(a_op)
            const_b = _get_float_val(b_op)

            dot_shape = _extract_shape(ssa.type_str)
            M = dot_shape[0] if len(dot_shape) >= 1 else 32
            N = dot_shape[1] if len(dot_shape) >= 2 else 32
            a_shape = _extract_shape(a_op.type_str)
            K = a_shape[1] if len(a_shape) >= 2 else 32
            return (const_a, const_b, M, N, K, ssa.elem_type)
        return None

    def _lower_dot_constant_template(self, const_info, ptr_args):
        """Generate MSL for dot product where both inputs are compile-time constants."""
        const_a, const_b, M, N, K, dot_elem_type = const_info
        safe_name = _sanitize_msl_name(self.graph.func_name)
        total = M * N
        num_warps = self.graph.num_warps
        block_size = num_warps * 32
        self._matmul_block_size = total

        c_ptr = ptr_args[-1] if ptr_args else None
        c_msl_type = triton_type_to_msl(c_ptr.elem_type) if c_ptr else "float"

        lines = []
        lines.append("#include <metal_stdlib>")
        lines.append("using namespace metal;")
        lines.append("")
        lines.append(f"kernel void {safe_name}(")
        arg_decls = []
        for i, arg in enumerate(self.graph.args):
            if arg.is_ptr:
                arg_msl_type = triton_type_to_msl(arg.elem_type)
                arg_decls.append(
                    f"    device {arg_msl_type}* {arg.name} [[buffer({i})]]")
        lines.append(",\n".join(arg_decls) + ",")
        lines.append("    uint lid [[thread_position_in_threadgroup]]")
        lines.append(") {")
        lines.append(f"    for (uint _e = lid; _e < {total}u; _e += {block_size}u) {{")
        lines.append(f"        uint _i = _e / {N}u;")
        lines.append(f"        uint _j = _e % {N}u;")
        # Each element = sum over K of const_a * const_b = K * const_a * const_b
        lines.append(f"        float _sum = (float){K}u * {const_a}f * {const_b}f;")
        lines.append(f"        {c_ptr.name}[_i * {N}u + _j] = ({c_msl_type})_sum;")
        lines.append(f"    }}")
        lines.append("}")
        lines.append("")
        return "\n".join(lines)

    def _lower_dot_simple_template(self, tile_dims, ptr_args, dtype) -> str:
        """Fall back to optimized simdgroup matmul template for simple kernels."""
        from triton_metal.codegen.msl_emitter import make_matmul_kernel

        block_m = tile_dims[0] if len(tile_dims) > 0 else 32
        block_n = block_m
        block_k = block_m
        self._matmul_block_size = block_m * block_n

        msl = make_matmul_kernel(
            block_m=block_m, block_n=block_n, block_k=block_k, dtype=dtype,
        )

        safe_name = _sanitize_msl_name(self.graph.func_name)
        msl = msl.replace("matmul_kernel", safe_name, 1)

        if len(ptr_args) >= 3:
            a_name, b_name, c_name = ptr_args[0].name, ptr_args[1].name, ptr_args[2].name
            # Replace parameter declarations -- use regex to match any MSL type (float, half, etc.)
            msl = re.sub(r'(device\s+const\s+\w+\*)\s+A\s', rf'\1 {a_name} ', msl)
            msl = re.sub(r'(device\s+const\s+\w+\*)\s+B\s', rf'\1 {b_name} ', msl)
            msl = re.sub(r'(volatile\s+device\s+\w+\*|device\s+\w+\*)\s+C\s', rf'\1 {c_name} ', msl)
            # Replace body references
            msl = re.sub(r'(?<![a-zA-Z_])A\[', f'{a_name}[', msl)
            msl = re.sub(r'(?<![a-zA-Z_])B\[', f'{b_name}[', msl)
            msl = re.sub(r'(?<![a-zA-Z_])C\[', f'{c_name}[', msl)

        return msl

    def _has_tensor_ops(self) -> bool:
        """Check if the kernel has any tensor-producing ops (tt.make_range, etc.).

        Scalar-only kernels (no tensor operations) should use block_size=1
        to avoid multiple threads racing on the same scalar memory locations.
        """
        def _check_ops(ops):
            for ssa in ops:
                if ssa.op in ("tt.make_range", "tt.splat", "tt.broadcast"):
                    return True
                if ssa.is_tensor:
                    return True
                if ssa.region_ops and _check_ops(ssa.region_ops):
                    return True
                if ssa.else_ops and _check_ops(ssa.else_ops):
                    return True
            return False
        return _check_ops(self.graph.ops)

    def _prescan_stores(self):
        """Scan ops to find which pointer args are stored to (outputs).

        Recursively scans nested regions (scf.for, scf.while, scf.if bodies)
        to find stores inside loops and conditionals.
        """
        self._prescan_stores_recursive(self.graph.ops)

        # Trace through tt.addptr → tt.splat → func_arg (or direct arg)
        # to identify which func_arg pointers are outputs
        self._output_arg_ids = set()
        arg_ids = {a.id for a in self.graph.args if a.is_ptr}

        # Build lookup: ssa_id -> first operand id (for addptr/splat chains)
        first_operand = {}
        self._build_first_operand_map(self.graph.ops, first_operand)

        for store_ptr_id in self._store_ptr_ids:
            # Walk the chain: store_ptr → addptr → splat → arg (or shorter)
            current = store_ptr_id
            for _ in range(5):  # Max chain depth
                if current in arg_ids:
                    self._output_arg_ids.add(current)
                    break
                next_id = first_operand.get(current)
                if next_id is None:
                    break
                current = next_id

    def _prescan_stores_recursive(self, ops):
        """Recursively find all tt.store and tt.atomic_rmw ops including in nested regions."""
        for ssa in ops:
            if ssa.op == "tt.store":
                if ssa.operand_ids:
                    self._store_ptr_ids.add(ssa.operand_ids[0])
            # tt.atomic_rmw modifies memory in-place — treat target as output
            if ssa.op == "tt.atomic_rmw":
                if ssa.operand_ids:
                    self._store_ptr_ids.add(ssa.operand_ids[0])
            # Recurse into nested regions
            if ssa.region_ops:
                self._prescan_stores_recursive(ssa.region_ops)
            if ssa.else_ops:
                self._prescan_stores_recursive(ssa.else_ops)

    def _build_first_operand_map(self, ops, first_operand):
        """Recursively build first-operand lookup for addptr/splat chains."""
        for ssa in ops:
            if ssa.op in ("tt.addptr", "tt.splat") and ssa.operand_ids:
                first_operand[ssa.id] = ssa.operand_ids[0]
            if ssa.region_ops:
                self._build_first_operand_map(ssa.region_ops, first_operand)
            if ssa.else_ops:
                self._build_first_operand_map(ssa.else_ops, first_operand)

    def _prescan_2d_info(self):
        """Detect 2D kernel patterns and compute make_range → dimension mappings.

        Scans the op graph for expand_dims + broadcast chains to determine:
        1. Whether this is a 2D kernel
        2. The effective 2D shape (M, N) from broadcast target types
        3. Which make_range ops correspond to which dimensions

        The pattern is:
            make_range(0, M) → expand_dims(axis=1) → broadcast → tensor<MxNx...>
            make_range(0, N) → expand_dims(axis=0) → broadcast → tensor<MxNx...>

        For a 2D kernel with shape (M, N), thread lid maps to:
            dim 0 (row): lid / N
            dim 1 (col): lid % N
        """
        self._prescan_2d_info_recursive(self.graph.ops)

    def _prescan_2d_info_recursive(self, ops, parent_op_by_id=None):
        """Recursively scan ops for 2D patterns."""
        # Build lookup tables for ops in this scope, including parent scope
        # so tracing can cross scope boundaries (e.g. expand_dims in scf.for
        # body can trace back to make_range in parent scope)
        op_by_id = dict(parent_op_by_id) if parent_op_by_id else {}
        for ssa in ops:
            op_by_id[ssa.id] = ssa

        # Find the max 2D shape from any tensor type in the kernel
        max_2d_shape = None
        for ssa in ops:
            shape = _extract_shape(ssa.type_str)
            if len(shape) >= 2:
                total = 1
                for d in shape:
                    total *= d
                if max_2d_shape is None:
                    max_2d_shape = shape
                else:
                    cur_total = 1
                    for d in max_2d_shape:
                        cur_total *= d
                    if total > cur_total:
                        max_2d_shape = shape
            # Recurse into nested regions
            if ssa.region_ops:
                self._prescan_2d_info_recursive(ssa.region_ops, op_by_id)
            if ssa.else_ops:
                self._prescan_2d_info_recursive(ssa.else_ops, op_by_id)

        if max_2d_shape is None or len(max_2d_shape) < 2:
            return

        self._is_2d = True
        if self._effective_2d_shape is None:
            self._effective_2d_shape = max_2d_shape

        # Find expand_dims ops and trace back to make_range
        for ssa in ops:
            if ssa.op == "tt.expand_dims" and ssa.operand_ids:
                axis = ssa.attrs.get("axis", 0)
                src_id = ssa.operand_ids[0]
                # Trace back through passthroughs to find the make_range
                mr_id = self._trace_to_make_range(src_id, ops, op_by_id)
                if mr_id is not None:
                    # axis tells us which NEW dimension was inserted:
                    # expand_dims(x, axis=1): x becomes row dim (dim 0)
                    # expand_dims(x, axis=0): x becomes col dim (dim 1)
                    dim = 0 if axis == 1 else (len(max_2d_shape) - 1)
                    self._make_range_dim[mr_id] = dim

    def _trace_to_make_range(self, ssa_id, ops, op_by_id):
        """Trace an SSA ID back through passthrough ops to find a make_range.

        Follows: passthroughs (extsi, convert_layout, etc.), tt.load (through
        the pointer operand), tt.addptr (through the offset operand), and
        arithmetic ops (muli, addi — tries both operands).
        This allows tracing from expand_dims through load→addptr→make_range
        chains, which is needed when a 1D load result gets expand_dims'd to 2D.
        """
        visited = set()
        current = ssa_id
        while current not in visited:
            visited.add(current)
            if current in op_by_id:
                op = op_by_id[current]
                if op.op == "tt.make_range":
                    return current
                # Follow through passthroughs (first operand)
                if op.op in ("arith.extsi", "arith.extui", "arith.trunci",
                              "arith.index_cast", "arith.index_castui",
                              "arith.sitofp", "arith.uitofp",
                              "ttg.convert_layout",
                              "tt.load") and op.operand_ids:
                    current = op.operand_ids[0]
                    continue
                # tt.addptr: follow the offset (second operand) to reach make_range
                if op.op == "tt.addptr" and len(op.operand_ids) >= 2:
                    current = op.operand_ids[1]
                    continue
                # Arithmetic ops (muli, addi): the make_range could be either
                # operand (e.g. arange*SIZE or SIZE*arange). Try both.
                if op.op in ("arith.muli", "arith.addi") and op.operand_ids:
                    for oid in op.operand_ids:
                        result = self._trace_to_make_range(oid, ops, op_by_id)
                        if result is not None:
                            return result
                    break
            break
        return None

    def _register_args(self):
        """Register function arguments with KernelBuilder."""
        for arg in self.graph.args:
            triton_dtype = _mlir_to_triton_dtype(arg.elem_type)
            if arg.is_ptr:
                # Never use const for generic-lowered kernels — prescan can miss
                # stores through block args, reductions, and complex chains.
                self.kb.add_ptr_arg(arg.name, dtype=triton_dtype, const=False)
            else:
                self.kb.add_scalar_arg(arg.name, dtype=triton_dtype)
            self.env[arg.id] = arg.name
            self.env_types[arg.id] = triton_dtype
            # Shape: function arguments are always scalar (pointers are
            # base addresses, scalars are single values).  tt.splat lifts
            # them to tensor shapes downstream.
            self.env_shapes[arg.id] = ()

    def _lookup(self, ssa_id: int) -> str:
        """Look up MSL variable name for an SSA value."""
        if ssa_id in self.env:
            return self.env[ssa_id]
        return f"UNKNOWN_{ssa_id}"

    def _lower_op(self, ssa: SSAValue):
        """Lower a single SSA operation to MSL."""
        # Skip ops that were handled as part of a fused pattern
        if ssa.id in self._skip_ids:
            return

        try:
            self._lower_op_dispatch(ssa)
        except (MetalCodegenError, MetalNotImplementedError):
            raise  # Already has context
        except Exception as e:
            raise MetalCodegenError(
                f"Failed to lower operation: {e}",
                op_name=ssa.op,
                ssa_id=ssa.id,
                type_str=ssa.type_str,
            ) from e

    def _lower_op_dispatch(self, ssa: SSAValue):
        """Dispatch a single SSA operation to its lowering handler."""
        op = ssa.op
        ids = ssa.operand_ids

        # Dispatch by op name
        if op == "tt.get_program_id":
            self._lower_get_program_id(ssa)
        elif op == "tt.get_num_programs":
            self._lower_get_num_programs(ssa)
        elif op == "tt.make_range":
            self._lower_make_range(ssa)
        elif op == "tt.splat":
            self._lower_splat(ssa)
        elif op == "tt.expand_dims":
            self._lower_expand_dims(ssa)
        elif op == "tt.broadcast":
            self._lower_broadcast(ssa)
        elif op == "tt.addptr":
            self._lower_addptr(ssa)
        elif op == "tt.load":
            self._lower_load(ssa)
        elif op == "tt.store":
            self._lower_store(ssa)
        elif op == "tt.reduce":
            self._lower_reduce(ssa)
        elif op == "tt.scan":
            self._lower_scan(ssa)
        elif op == "tt.clampf":
            self._lower_clampf(ssa)
        elif op == "tt.dot":
            self._lower_dot(ssa)
        elif op == "arith.constant":
            self._lower_constant(ssa)
        elif op.startswith("arith."):
            self._lower_arith(ssa)
        elif op.startswith("math."):
            self._lower_math(ssa)
        elif op == "scf.for":
            self._lower_scf_for(ssa)
        elif op == "scf.if":
            self._lower_scf_if(ssa)
        elif op == "scf.while":
            self._lower_scf_while(ssa)
        elif op in ("scf.yield", "scf.condition"):
            pass  # Handled by parent op
        elif op == "tt.call":
            self._lower_call(ssa)
        elif op == "tt.return":
            pass  # Kernel return — nothing to emit
        elif op.startswith("ttg."):
            self._lower_ttg(ssa)
        elif op == "tt.reshape":
            self._emit_passthrough(ssa)
            # Propagate output shape from type_str for downstream reduce detection
            if ssa.type_str:
                out_shape = _extract_shape(ssa.type_str)
                if out_shape:
                    self.env_shapes[ssa.id] = out_shape
        elif op == "tt.trans":
            self._lower_tt_trans(ssa)
        elif op == "tt.join":
            self._lower_tt_join(ssa)
        elif op == "tt.cat":
            self._lower_tt_cat(ssa)
        elif op == "tt.split":
            self._lower_tt_split(ssa)
        elif op == "tt.histogram":
            self._lower_tt_histogram(ssa)
        elif op == "tt.gather":
            self._lower_tt_gather(ssa)
        elif op == "tt.unsplat":
            # tt.unsplat: extract scalar from 1-element tensor (inverse of splat)
            # In per-thread model, this is a passthrough.
            self._emit_passthrough(ssa)
        elif op == "tt.map_elementwise":
            self._lower_map_elementwise(ssa)
        elif op == "tt.atomic_rmw":
            self._lower_atomic_rmw(ssa)
        elif op == "tt.atomic_cas":
            self._lower_atomic_cas(ssa)
        elif op == "tt.debug_barrier":
            self.kb.raw_line("    threadgroup_barrier(mem_flags::mem_device);")
        elif op == "tt.mulhiui":
            self._lower_mulhiui(ssa)
        elif op == "tt.bitcast":
            # tt.bitcast can be:
            # 1. Pointer bitcast (e.g. !tt.ptr<i1> -> !tt.ptr<i8>)
            # 2. Value bitcast (e.g. f32 -> i32 for float atomic max)
            # For case 2, we need as_type<T>() in MSL.
            self._lower_tt_bitcast(ssa)
        elif op == "tt.precise_sqrt":
            self._lower_precise_math(ssa, "sqrt")
        elif op == "tt.precise_divf":
            self._lower_precise_math(ssa, "divf")
        elif op == "tt.extern_elementwise":
            self._lower_extern_elementwise(ssa)
        elif op == "tt.assert":
            pass  # Runtime bounds check — skip in MSL
        else:
            # Unknown op — emit comment (don't raise: may be in dead code path)
            self.kb.comment(f"UNSUPPORTED: {op}")

    # -- Program ID and indexing --

    def _lower_get_program_id(self, ssa: SSAValue):
        """tt.get_program_id → pid / pid_y / pid_z.

        Axis 0 (x) → pid, axis 1 (y) → pid_y, axis 2 (z) → pid_z.
        Tracks which axes are used so the kernel signature includes them.
        """
        axis = ssa.attrs.get("axis", 0)
        self._used_pid_axes.add(axis)
        if axis == 0:
            self.env[ssa.id] = "pid"
        elif axis == 1:
            self.env[ssa.id] = "pid_y"
        else:
            self.env[ssa.id] = "pid_z"
        self.env_types[ssa.id] = "i32"
        # Shape: program_id is always scalar
        self.env_shapes[ssa.id] = ()

    def _lower_get_num_programs(self, ssa: SSAValue):
        """tt.get_num_programs → grid dimension (threadgroups_per_grid).

        Uses Metal's [[threadgroups_per_grid]] kernel parameter.
        Since other grid attributes (pid, lid, tid) use scalar uint,
        threadgroups_per_grid must also be scalar uint (Metal requires
        all position attributes to have matching dimensionality).
        For axis 0 this is just 'tpg'. Multi-axis dispatch would need
        uint3 for all grid attributes.
        """
        axis = ssa.attrs.get("axis", 0)
        if axis == 0:
            self.env[ssa.id] = "tpg"
        elif axis == 1:
            self.env[ssa.id] = "tpg_y"
        else:
            self.env[ssa.id] = "tpg_z"
        self.env_types[ssa.id] = "i32"
        # Track which axes need num_programs
        self._used_pid_axes.add(axis)
        # Flag that we need the threadgroups_per_grid parameter
        self._needs_num_programs = True
        # Shape: num_programs is always scalar
        self.env_shapes[ssa.id] = ()

    def _lower_make_range(self, ssa: SSAValue):
        """tt.make_range → lid or 2D index expression.

        In Metal SIMT, tt.make_range {start=0, end=BLOCK_SIZE}
        produces per-thread indices [0, 1, ..., BLOCK_SIZE-1].

        For 1D kernels: maps directly to lid.
        For 2D kernels: maps to lid/N (row dim) or lid%N (col dim),
        determined by the expand_dims + broadcast pre-pass analysis.
        """
        start = ssa.attrs.get("start", 0)
        end = ssa.attrs.get("end", self.graph.block_size)

        # Check if this make_range is part of a 2D pattern
        if self._is_2d and ssa.id in self._make_range_dim:
            dim = self._make_range_dim[ssa.id]
            range_size = end - start
            var_name = self._next_var("idx")
            lid = self._lid_expr
            # Use _total_elements (not capped effective_block_size) for
            # correct index decomposition when wrapping loop is active.
            total = getattr(self, "_total_elements", self.effective_block_size)

            # Compute the inner dimension N from range_size and total.
            # A kernel may have multiple 2D shapes (e.g., transpose: input (2,4)
            # and output (4,2)), so each make_range derives N from its own range.
            # For row (dim 0): range covers rows, N = total / range = inner dim
            # For col (dim 1): range IS the inner dim, N = range_size
            if dim == 0:
                N = total // range_size if range_size > 0 else 1
                expr = f"{lid} / {N}u"
            else:
                expr = f"{lid} % {range_size}u"

            if start != 0:
                expr = f"({expr} + {start}u)"

            self.kb.raw_line(f"    uint {var_name} = {expr};")
            self.env[ssa.id] = var_name
            self.env_types[ssa.id] = "i32"
            self.env_shapes[ssa.id] = (range_size,)
            return

        # 1D make_range in a 2D kernel: this range is used for a 1D operation
        # (like a load) that later gets expand_dims'd to 2D. The range values
        # need to cycle within [start, end) for each thread, so use modular
        # indexing: lid % range_size. This gives each column/row of threads
        # a valid index within the original 1D array.
        range_size = end - start
        total = getattr(self, "_total_elements", self.effective_block_size)
        if self._is_2d and range_size < total:
            lid = self._lid_expr
            var_name = self._next_var("idx")
            if start != 0:
                expr = f"({lid} % {range_size}u + {start}u)"
            else:
                expr = f"{lid} % {range_size}u"
            self.kb.raw_line(f"    uint {var_name} = {expr};")
            self.env[ssa.id] = var_name
            self.env_types[ssa.id] = "i32"
            self.env_shapes[ssa.id] = (range_size,)
            return

        # Pure 1D case (original behavior)
        lid = self._lid_expr
        if start != 0:
            var_name = self._next_var("range")
            self.kb.raw_line(f"    uint {var_name} = {lid} + {start}u;")
            self.env[ssa.id] = var_name
        else:
            self.env[ssa.id] = lid
        self.env_types[ssa.id] = "i32"
        self.env_shapes[ssa.id] = (end - start,)

    def _lower_splat(self, ssa: SSAValue):
        """tt.splat → pass through (broadcast is free in SIMT).

        In Triton IR, tt.splat broadcasts a scalar to a tensor.
        In Metal per-thread execution, every thread already has the scalar,
        so this is a no-op.

        Special case: splatting a pointer arg without addptr (e.g. for
        scalar loads like tl.load(X)) registers the result with offset "0"
        so that tt.load reads from index 0 for all threads.

        Shape tracking: the source is scalar (), the result gets the tensor
        shape from type_str (e.g. tensor<256xf32> → (256,)).  This records
        that the value has been "splatted" but each thread still holds a
        single scalar copy.
        """
        if ssa.operand_ids:
            src_id = ssa.operand_ids[0]
            self.env[ssa.id] = self._lookup(src_id)
            if src_id in self.env_types:
                self.env_types[ssa.id] = self.env_types[src_id]
            if src_id in self.env_is_mask:
                self.env_is_mask[ssa.id] = True
            if src_id in self.env_is_ptr:
                self.env_is_ptr[ssa.id] = self.env_is_ptr[src_id]
            elif "!tt.ptr" in ssa.type_str:
                # Splatting a raw pointer arg (no addptr) — all threads
                # point to the same address, so offset is 0
                self.env_is_ptr[ssa.id] = (self._lookup(src_id), "0")
        # Track splat output shape from result type
        shape = _extract_shape(ssa.type_str)
        if shape:
            self.env_shapes[ssa.id] = shape
        else:
            # Scalar splat (no tensor wrapper) — record as scalar
            self.env_shapes[ssa.id] = ()

    def _lower_expand_dims(self, ssa: SSAValue):
        """tt.expand_dims → passthrough with shape tracking.

        In the 2D model, expand_dims inserts a size-1 dimension.
        The per-thread value doesn't change (index remapping was done
        at make_range level by the 2D pre-pass), so this is a passthrough.

        Shape tracking: records the new shape with the inserted dimension.
        For example, tensor<64xi32> with axis=1 → tensor<64x1xi32>,
        giving shape (64, 1).

        TODO(2D codegen): When the source is a 1D range and expand_dims
        inserts a new axis, this is where the value semantically transitions
        from 1D to 2D.  In full 2D codegen, this is the point where we'd
        need to decide whether this value represents row indices or column
        indices based on the axis parameter.  Currently, the make_range
        pre-pass handles this heuristically, but a proper implementation
        would use the expand_dims axis to assign dimensions explicitly.
        """
        self._emit_passthrough(ssa)
        # Track shape from the result type (overrides passthrough shape)
        shape = _extract_shape(ssa.type_str)
        if shape:
            self.env_shapes[ssa.id] = shape

    def _lower_broadcast(self, ssa: SSAValue):
        """tt.broadcast → passthrough with shape tracking.

        In the 2D model, broadcasting is handled implicitly:
        - make_range already computes the correct 2D index (lid/N or lid%N)
        - Intermediate values (loads, arithmetic) propagate correctly
        - The broadcast just changes the "shape" annotation

        This works because each thread's value is already the correct
        broadcast result based on the 2D index computed at make_range time.

        Shape tracking: records the broadcast target shape.  For example,
        tensor<64x1xi32> broadcast to tensor<64x128xi32> gives shape
        (64, 128).  The source shape (64, 1) → target shape (64, 128)
        tells us dimension 1 was broadcast.

        TODO(2D codegen): In full 2D support, broadcast is where we'd
        need to handle replication.  When a value with shape (M, 1) is
        broadcast to (M, N), each thread in a row should get the same
        value.  Currently the 1D per-thread model makes this implicit
        (all threads independently compute) but a proper 2D model would
        need to ensure the column index is ignored for broadcast dims.
        The shape tracking infrastructure enables detecting these cases
        by comparing source shape to target shape to find broadcast dims.
        """
        self._emit_passthrough(ssa)
        # Track shape from the result type (overrides passthrough shape)
        shape = _extract_shape(ssa.type_str)
        if shape:
            self.env_shapes[ssa.id] = shape

    def _lower_addptr(self, ssa: SSAValue):
        """tt.addptr → pointer + offset indexing.

        tt.addptr(%ptr_tensor, %offset_tensor) computes element addresses.
        In MSL, this becomes array indexing: ptr[offset].
        We track the (base_ptr, offset) pair for use in load/store.
        Chained addptrs accumulate offsets: addptr(addptr(p, a), b) → p[a + b].
        """
        if len(ssa.operand_ids) >= 2:
            ptr_id = ssa.operand_ids[0]
            offset_var = self._lookup(ssa.operand_ids[1])

            # Check if this is a chained addptr (ptr_id is itself an addptr result)
            parent_ptr_info = self.env_is_ptr.get(ptr_id)
            if parent_ptr_info:
                base_ptr, existing_offset = parent_ptr_info
                combined = f"({existing_offset} + {offset_var})"
                self.env_is_ptr[ssa.id] = (base_ptr, combined)
                self.env[ssa.id] = f"{base_ptr}[{combined}]"
            else:
                ptr_var = self._lookup(ptr_id)
                self.env_is_ptr[ssa.id] = (ptr_var, offset_var)
                self.env[ssa.id] = f"{ptr_var}[{offset_var}]"
            # Shape: addptr inherits shape from its operands (typically the
            # offset tensor dictates the shape, or the pointer tensor from splat).
            # TODO(2D codegen): For 2D addptr, the offset computation encodes
            # the memory layout (row * stride + col).  When both operands have
            # shapes, use the larger one.
            self._propagate_shape_elementwise(ssa)

    # -- Load and Store --

    def _lower_load(self, ssa: SSAValue):
        """tt.load → masked buffer read with optional 'other' default value."""
        if not ssa.operand_ids:
            return

        ptr_id = ssa.operand_ids[0]
        ptr_info = self.env_is_ptr.get(ptr_id)

        if ptr_info:
            base_ptr, offsets = ptr_info
        else:
            # Direct pointer (no addptr)
            base_ptr = self._lookup(ptr_id)
            # Scalar load (non-tensor result) → always load from index 0
            # Tensor load without addptr → use lid as offset
            offsets = "0" if not ssa.is_tensor else self._lid_expr

        # Determine dtype from pointer type
        dtype = _mlir_to_triton_dtype(ssa.elem_type)
        compute_type = _msl_compute_type(dtype)
        zero = "0.0f" if dtype in ("fp32", "fp16", "bf16") else "0"

        # Parse operands: tt.load(ptr, mask?, other?)
        # Operands after the pointer: mask (i1 tensor), then other (default value)
        mask_var = None
        other_val = zero

        remaining_ids = ssa.operand_ids[1:]
        for op_id in remaining_ids:
            if op_id in self.env_is_mask or self._is_mask(op_id):
                mask_var = self._lookup(op_id)
            elif mask_var is not None:
                # After mask comes the 'other' value
                other_val = self._lookup(op_id)

        var_name = self._next_var("val")

        if mask_var:
            self.kb.raw_line(
                f"    {compute_type} {var_name} = {mask_var} ? "
                f"static_cast<{compute_type}>({base_ptr}[{offsets}]) : "
                f"static_cast<{compute_type}>({other_val});"
            )
        else:
            self.kb.raw_line(
                f"    {compute_type} {var_name} = "
                f"static_cast<{compute_type}>({base_ptr}[{offsets}]);"
            )

        self.env[ssa.id] = var_name
        self.env_types[ssa.id] = dtype
        # Shape: load inherits shape from pointer operand.
        # TODO(2D codegen): When shape is 2D+, emit proper row/col indexing
        # instead of flat lid-based offset.  For shape (M, N):
        #   row = lid / N, col = lid % N
        #   offset = row_offsets[row] + col_offsets[col]
        ptr_shape = self.env_shapes.get(ptr_id)
        if ptr_shape:
            self.env_shapes[ssa.id] = ptr_shape
        else:
            self._propagate_shape_from_type(ssa)

    def _lower_store(self, ssa: SSAValue):
        """tt.store → masked buffer write.

        TODO(2D codegen): When ptr shape is 2D+, emit proper row/col
        indexing for the store offset instead of flat lid-based offset.
        The shape information from env_shapes can be used to decompose
        the flat thread index into multi-dimensional coordinates.
        """
        if len(ssa.operand_ids) < 2:
            return

        ptr_id = ssa.operand_ids[0]
        val_id = ssa.operand_ids[1]

        # Detect reduce keep_dims pattern: store to (M, 1) or (1, N) shaped pointer
        # where the value comes from a reduce. The generic 2D index decomposition
        # is broken for this case, so we use guarded lid-based indexing instead.
        # Skip when dim_0 == 1 (triton_per_* pattern): the ptr already has
        # the row offset from addptr, and using base_ptr[idx] would lose it.
        ptr_shape = self.env_shapes.get(ptr_id)
        val_shape = self.env_shapes.get(val_id)
        if (self._is_2d and ptr_shape and len(ptr_shape) == 2
                and (ptr_shape[0] == 1 or ptr_shape[1] == 1)
                and ptr_shape[0] != ptr_shape[1]
                and ptr_shape[0] != 1):
            result_size = max(ptr_shape)
            ptr_info = self.env_is_ptr.get(ptr_id)
            if ptr_info:
                base_ptr, offsets = ptr_info
            else:
                base_ptr = self._lookup(ptr_id)
                offsets = self._lid_expr
            val_var = self._lookup(val_id)
            store_dtype = self._trace_ptr_dtype(ptr_id)
            store_type = triton_type_to_msl(store_dtype)
            compute_type = _msl_compute_type(store_dtype)
            needs_cast = (store_type != compute_type)
            cast_val = f"static_cast<{store_type}>({val_var})" if needs_cast else val_var
            idx = self._lid_expr
            # In 2D kernels, lid maps to row index via lid/N where N is the
            # inner dimension.  The guard must cover all threads whose
            # lid/N < result_size, i.e. lid < result_size * N.  Using just
            # result_size (the number of rows) cuts off the upper threads
            # and leaves half the rows unwritten when N > 1.
            guard_size = result_size
            if self._effective_2d_shape and len(self._effective_2d_shape) == 2:
                inner_N = self._effective_2d_shape[1]
                guard_size = result_size * inner_N
            self.kb.raw_line(
                f"    if ({idx} < {guard_size}u) {base_ptr}[{offsets}] = {cast_val};")
            return

        ptr_info = self.env_is_ptr.get(ptr_id)
        if ptr_info:
            base_ptr, offsets = ptr_info
        else:
            base_ptr = self._lookup(ptr_id)
            # Scalar pointer (direct arg, not tensor of pointers) → offset 0
            # Tensor pointer → use lid as offset
            offsets = "0" if self._is_scalar_ptr(ptr_id) else self._lid_expr

        val_var = self._lookup(val_id)

        # Determine storage type
        # Trace back to the function arg to find the pointer dtype
        store_dtype = self._trace_ptr_dtype(ptr_id)
        store_type = triton_type_to_msl(store_dtype)
        compute_type = _msl_compute_type(store_dtype)
        needs_cast = (store_type != compute_type)
        cast_val = f"static_cast<{store_type}>({val_var})" if needs_cast else val_var

        # Get mask if provided
        mask_var = None
        if len(ssa.operand_ids) >= 3:
            mask_id = ssa.operand_ids[2]
            if mask_id in self.env_is_mask or self._is_mask(mask_id):
                mask_var = self._lookup(mask_id)

        # In 2D kernels, 1D store tensors must be guarded to prevent
        # duplicate writes from extra threads (e.g. after 2D→1D reduce).
        store_1d_guard = None
        # Check if the kernel has any ttg.convert_layout that did a real
        # shared memory redistribution. If so, all 1D stores in this kernel
        # should use simple lid < N guards because the convert_layout
        # changed the thread-to-element mapping to simple (thread i = element i).
        val_converted = hasattr(self, '_converted_layout_ids') and bool(getattr(self, '_converted_layout_ids', set()))
        if self._is_2d and not self._is_scalar_ptr(ptr_id):
            store_shape = self.env_shapes.get(ptr_id)
            if not store_shape:
                for op in self.graph.ops:
                    if op.id == ptr_id and op.type_str:
                        store_shape = _extract_shape(op.type_str)
                        break
            if store_shape and len(store_shape) == 1 and store_shape[0] < self.effective_block_size:
                store_1d_guard = store_shape[0]

        if store_1d_guard is not None:
            lid = self._lid_expr
            if val_converted:
                # After convert_layout, thread i has element i. Simple guard.
                guard = f"{lid} < {store_1d_guard}u"
            else:
                # After a 2D reduce (axis=1), the result is per-row and the
                # broadcast uses lid / N (blocked). Fix: use lid / N as the
                # store index and select one thread per row block.
                shape = self._effective_2d_shape
                if (shape and len(shape) >= 2 and store_1d_guard == shape[0]
                        and shape[1] > 0):
                    N = shape[1]
                    offsets = f"({lid} / {N}u)"
                    guard = f"{lid} % {N}u == 0u && {lid} / {N}u < {store_1d_guard}u"
                else:
                    guard = f"{lid} < {store_1d_guard}u"
            if mask_var:
                self.kb.raw_line(f"    if ({guard} && {mask_var}) {{ {base_ptr}[{offsets}] = {cast_val}; }}")
            else:
                self.kb.raw_line(f"    if ({guard}) {{ {base_ptr}[{offsets}] = {cast_val}; }}")
        elif mask_var:
            self.kb.raw_line(f"    if ({mask_var}) {{ {base_ptr}[{offsets}] = {cast_val}; }}")
        else:
            self.kb.raw_line(f"    {base_ptr}[{offsets}] = {cast_val};")

    def _trace_ptr_dtype(self, ptr_id: int) -> str:
        """Trace a pointer SSA value back to its function arg dtype."""
        # ptr_id might be an addptr result
        info = self.env_is_ptr.get(ptr_id)
        if info:
            base_name = info[0]
            # Find the function arg with this name
            for arg in self.graph.args:
                if arg.name == base_name and arg.is_ptr:
                    return _mlir_to_triton_dtype(arg.elem_type)

        # Direct lookup from env
        if ptr_id in self.env_types:
            return self.env_types[ptr_id]

        return "fp32"

    def _is_mask(self, ssa_id: int) -> bool:
        """Check if an SSA value is a boolean mask."""
        if ssa_id in self.env_is_mask:
            return True
        # Check type: i1 or tensor<...xi1> (exact match, not substring)
        for ssa in self.graph.ops:
            if ssa.id == ssa_id:
                return ssa.elem_type == "i1" or ssa.op in ("arith.cmpi", "arith.cmpf")
        return False

    def _is_scalar_ptr(self, ssa_id: int) -> bool:
        """Check if an SSA value is a scalar pointer (not a tensor of pointers).

        A scalar pointer like !tt.ptr<i32> should be indexed at [0],
        while a tensor of pointers like tensor<256x!tt.ptr<i32>> uses [lid].
        """
        # Check function args first
        for arg in self.graph.args:
            if arg.id == ssa_id:
                return arg.is_ptr and "tensor<" not in arg.type_str
        # Check ops
        for ssa in self.graph.ops:
            if ssa.id == ssa_id:
                return "!tt.ptr" in ssa.type_str and "tensor<" not in ssa.type_str
        return False

    # -- Constants --

    def _lower_constant(self, ssa: SSAValue):
        """arith.constant → literal value.

        Handles int, float, bool, and hex-encoded IEEE 754 bit patterns
        (MLIR uses hex integers for special floats like inf/nan).
        """
        import math
        import struct as _struct

        value = ssa.attrs.get("value")
        var_name = self._next_var("c")

        if value is None:
            # Unknown constant — use 0
            self.env[ssa.id] = "0"
            self.env_types[ssa.id] = "i32"
            self.env_shapes[ssa.id] = ()
            return

        # Check if this is a hex integer that should be interpreted as float
        is_float_type = ssa.elem_type in ("f32", "f16", "bf16", "f64")
        if isinstance(value, int) and is_float_type:
            # Hex-encoded IEEE 754 bit pattern
            try:
                float_val = _struct.unpack('f', _struct.pack('I', value & 0xFFFFFFFF))[0]
            except _struct.error:
                float_val = 0.0

            if math.isinf(float_val):
                msl_val = "INFINITY" if float_val > 0 else "(-INFINITY)"
            elif math.isnan(float_val):
                msl_val = "NAN"
            else:
                msl_val = f"{float_val}f"

            if ssa.is_tensor:
                self.env[ssa.id] = msl_val
                self.env_types[ssa.id] = "fp32"
                self._propagate_shape_from_type(ssa)
                return
            self.kb.raw_line(f"    float {var_name} = {msl_val};")
            self.env[ssa.id] = var_name
            self.env_types[ssa.id] = "fp32"
            self.env_shapes[ssa.id] = ()
            return

        # Determine type and format
        if isinstance(value, bool) or (isinstance(value, str) and value in ("true", "false")):
            bool_val = value if isinstance(value, str) else ("true" if value else "false")
            if ssa.is_tensor:
                # Tensor bool: store as int (1/0) for SIMD reduction compatibility
                int_val = "1" if bool_val == "true" else "0"
                self.env[ssa.id] = int_val
                self.env_types[ssa.id] = "i1"
                self._propagate_shape_from_type(ssa)
                return
            self.kb.raw_line(f"    bool {var_name} = {bool_val};")
            self.env[ssa.id] = var_name
            self.env_types[ssa.id] = "i1"
        elif isinstance(value, int):
            is_i64 = ssa.elem_type == "i64" or abs(value) > 0x7FFFFFFF
            int_type = "long" if is_i64 else "int"
            int_dtype = "i64" if is_i64 else "i32"
            if ssa.is_tensor:
                self.env[ssa.id] = str(value)
                self.env_types[ssa.id] = int_dtype
                self._propagate_shape_from_type(ssa)
                return
            self.kb.raw_line(f"    {int_type} {var_name} = {value};")
            self.env[ssa.id] = var_name
            self.env_types[ssa.id] = int_dtype
        elif isinstance(value, float):
            if math.isinf(value):
                msl_val = "INFINITY" if value > 0 else "(-INFINITY)"
            elif math.isnan(value):
                msl_val = "NAN"
            else:
                msl_val = f"{value}f"
            if ssa.is_tensor:
                self.env[ssa.id] = msl_val
                self.env_types[ssa.id] = "fp32"
                self._propagate_shape_from_type(ssa)
                return
            self.kb.raw_line(f"    float {var_name} = {msl_val};")
            self.env[ssa.id] = var_name
            self.env_types[ssa.id] = "fp32"
        else:
            self.env[ssa.id] = str(value)
            self.env_types[ssa.id] = "i32"
        # Shape: constants are scalar unless they have a tensor type
        self._propagate_shape_from_type(ssa)

    # -- Arithmetic ops --

    def _lower_arith(self, ssa: SSAValue):
        """Lower arith.* operations."""
        op = ssa.op
        ids = ssa.operand_ids

        if op in ("arith.addf", "arith.addi"):
            self._emit_binary(ssa, "+")
        elif op in ("arith.subf", "arith.subi"):
            self._emit_binary(ssa, "-")
        elif op in ("arith.mulf", "arith.muli"):
            self._emit_binary(ssa, "*")
        elif op == "arith.divf":
            self._emit_binary(ssa, "/")
        elif op == "arith.divsi":
            self._emit_binary(ssa, "/")
        elif op == "arith.divui":
            self._emit_binary(ssa, "/", force_unsigned=True)
        elif op == "arith.remsi":
            self._emit_binary(ssa, "%")
        elif op == "arith.remui":
            self._emit_binary(ssa, "%", force_unsigned=True)
        elif op == "arith.remf":
            self._emit_builtin_binary(ssa, "fmod")
        elif op == "arith.negf":
            self._emit_unary(ssa, "-")
        elif op in ("arith.maxf", "arith.maxsi"):
            self._emit_builtin_binary(ssa, "max")
        elif op == "arith.maxui":
            self._emit_builtin_binary(ssa, "max", force_unsigned=True)
        elif op in ("arith.minf", "arith.minsi"):
            self._emit_builtin_binary(ssa, "min")
        elif op == "arith.minui":
            self._emit_builtin_binary(ssa, "min", force_unsigned=True)
        # NaN-quiet min/max (IEEE 754 minNum/maxNum): return non-NaN operand
        elif op == "arith.maxnumf":
            self._emit_builtin_binary(ssa, "fmax")
        elif op == "arith.minnumf":
            self._emit_builtin_binary(ssa, "fmin")
        # NaN-propagating min/max: if either operand is NaN, result is NaN
        elif op == "arith.maximumf":
            self._emit_nan_propagating_minmax(ssa, "fmax")
        elif op == "arith.minimumf":
            self._emit_nan_propagating_minmax(ssa, "fmin")
        elif op == "arith.cmpi":
            self._lower_cmpi(ssa)
        elif op == "arith.cmpf":
            self._lower_cmpf(ssa)
        elif op == "arith.select":
            self._lower_select(ssa)
        elif op == "arith.extf":
            self._emit_passthrough(ssa)
            self.env_types[ssa.id] = "fp32"
        elif op == "arith.truncf":
            self._emit_passthrough(ssa)
            self.env_types[ssa.id] = "fp16"
        elif op == "arith.sitofp":
            self._emit_cast(ssa, "float")
            self.env_types[ssa.id] = "fp32"
        elif op == "arith.uitofp":
            self._emit_uitofp(ssa)
            self.env_types[ssa.id] = "fp32"
        elif op in ("arith.fptosi",):
            msl_ty, dtype = _msl_int_type(ssa.elem_type, unsigned=False)
            self._emit_cast(ssa, msl_ty, dtype=dtype)
        elif op == "arith.fptoui":
            msl_ty, dtype = _msl_int_type(ssa.elem_type, unsigned=True)
            self._emit_cast(ssa, msl_ty, dtype=dtype)
        elif op == "arith.extsi":
            self._emit_int_cast(ssa, unsigned=False)
        elif op == "arith.extui":
            self._emit_int_cast(ssa, unsigned=True)
        elif op in ("arith.trunci",):
            self._emit_int_cast(ssa, unsigned=False)
        elif op in ("arith.index_cast", "arith.index_castui"):
            self._emit_cast(ssa, "int")
            self.env_types[ssa.id] = "i32"
        elif op == "arith.bitcast":
            self._lower_arith_bitcast(ssa)
        elif op == "arith.andi":
            self._emit_binary(ssa, "&")
        elif op == "arith.ori":
            self._emit_binary(ssa, "|")
        elif op == "arith.xori":
            self._emit_binary(ssa, "^")
        elif op == "arith.shli":
            self._emit_binary(ssa, "<<")
        elif op == "arith.shrsi":
            self._emit_binary(ssa, ">>")
        elif op == "arith.shrui":
            self._emit_binary(ssa, ">>", force_unsigned=True)
        else:
            self.kb.comment(f"UNSUPPORTED arith: {op}")

    def _emit_binary(self, ssa: SSAValue, op_str: str, force_unsigned=False):
        """Emit a binary operation: result = a op b."""
        if len(ssa.operand_ids) < 2:
            return
        a = self._lookup(ssa.operand_ids[0])
        b = self._lookup(ssa.operand_ids[1])
        var_name = self._next_var("r")
        is_float = self._is_float_op(ssa)
        if is_float:
            ty = "float"
            dtype = "fp32"
        elif force_unsigned:
            # Use the correct unsigned width from elem_type
            ty, dtype = _msl_int_type(ssa.elem_type, unsigned=True)
        else:
            # Use the correct signed width from elem_type
            ty, dtype = _msl_int_type(ssa.elem_type, unsigned=False)
        if force_unsigned and not is_float:
            # Cast operands to the correct unsigned type for unsigned semantics
            unsigned_ty, _ = _msl_int_type(ssa.elem_type, unsigned=True)
            self.kb.raw_line(f"    {ty} {var_name} = ({unsigned_ty}){a} {op_str} ({unsigned_ty}){b};")
        else:
            self.kb.raw_line(f"    {ty} {var_name} = {a} {op_str} {b};")
        self.env[ssa.id] = var_name
        self.env_types[ssa.id] = dtype
        # Shape: element-wise binary inherits shape from operands
        self._propagate_shape_elementwise(ssa)

    def _is_float_op(self, ssa: SSAValue) -> bool:
        """Check if an SSA op produces a float result."""
        # Check element type first (most reliable)
        if ssa.elem_type in ("f32", "f16", "bf16", "f64"):
            return True
        # Check op suffix
        if ssa.op.endswith("f") or ssa.op.endswith("fp"):
            return True
        # Check operand types
        for op_id in ssa.operand_ids[:2]:
            dtype = self.env_types.get(op_id)
            if dtype and dtype.startswith("fp") or dtype in ("bf16",):
                return True
        return False

    def _emit_unary(self, ssa: SSAValue, op_str: str):
        """Emit a unary operation: result = op(a)."""
        if not ssa.operand_ids:
            return
        a = self._lookup(ssa.operand_ids[0])
        var_name = self._next_var("r")
        self.kb.raw_line(f"    float {var_name} = {op_str}{a};")
        self.env[ssa.id] = var_name
        self.env_types[ssa.id] = "fp32"
        # Shape: unary inherits shape from its operand
        self._propagate_shape_elementwise(ssa)

    def _emit_builtin_binary(self, ssa: SSAValue, fn_name: str, force_unsigned=False):
        """Emit a builtin binary function: result = fn(a, b)."""
        if len(ssa.operand_ids) < 2:
            return
        a = self._lookup(ssa.operand_ids[0])
        b = self._lookup(ssa.operand_ids[1])
        var_name = self._next_var("r")
        is_float = self._is_float_op(ssa)
        if is_float:
            ty = "float"
            dtype = "fp32"
        elif force_unsigned:
            ty, dtype = _msl_int_type(ssa.elem_type, unsigned=True)
        else:
            ty, dtype = _msl_int_type(ssa.elem_type, unsigned=False)
        if force_unsigned and not is_float:
            unsigned_ty, _ = _msl_int_type(ssa.elem_type, unsigned=True)
            self.kb.raw_line(f"    {ty} {var_name} = {fn_name}(({unsigned_ty}){a}, ({unsigned_ty}){b});")
        else:
            self.kb.raw_line(f"    {ty} {var_name} = {fn_name}({a}, {b});")
        self.env[ssa.id] = var_name
        self.env_types[ssa.id] = dtype
        # Shape: element-wise builtin binary inherits shape from operands
        self._propagate_shape_elementwise(ssa)

    def _emit_nan_propagating_minmax(self, ssa: SSAValue, fn_name: str):
        """Emit NaN-propagating min/max: if either operand is NaN, result is NaN."""
        if len(ssa.operand_ids) < 2:
            return
        a = self._lookup(ssa.operand_ids[0])
        b = self._lookup(ssa.operand_ids[1])
        var_name = self._next_var("r")
        self.kb.raw_line(
            f"    float {var_name} = (isnan({a}) || isnan({b})) "
            f"? NAN : {fn_name}({a}, {b});"
        )
        self.env[ssa.id] = var_name
        self.env_types[ssa.id] = "fp32"
        # Shape: element-wise binary inherits shape from operands
        self._propagate_shape_elementwise(ssa)

    def _lower_clampf(self, ssa: SSAValue):
        """tt.clampf → clamp(x, min, max) with optional NaN propagation.

        propagateNan = "none": fmin(fmax(x, min), max)  (NaN-quiet)
        propagateNan = "all":  NaN if x is NaN, else fmin(fmax(x, min), max)
        """
        if len(ssa.operand_ids) < 3:
            return
        x = self._lookup(ssa.operand_ids[0])
        lo = self._lookup(ssa.operand_ids[1])
        hi = self._lookup(ssa.operand_ids[2])
        var_name = self._next_var("r")
        propagate = ssa.attrs.get("propagateNan", "none")
        if propagate == "all":
            # NaN-propagating: if x is NaN, result is NaN
            self.kb.raw_line(
                f"    float {var_name} = isnan({x}) "
                f"? NAN : fmin(fmax({x}, {lo}), {hi});"
            )
        else:
            # NaN-quiet: standard clamp
            self.kb.raw_line(
                f"    float {var_name} = fmin(fmax({x}, {lo}), {hi});"
            )
        self.env[ssa.id] = var_name
        self.env_types[ssa.id] = "fp32"
        # Shape: clamp is element-wise
        self._propagate_shape_elementwise(ssa)

    def _lower_tt_bitcast(self, ssa: SSAValue):
        """tt.bitcast → reinterpret bits or change pointer element type.

        Handles both pointer bitcasts (passthrough) and value bitcasts
        (float <-> int requiring MSL as_type<>()).
        """
        if not ssa.operand_ids:
            return
        src_id = ssa.operand_ids[0]

        # Pointer bitcast — preserve ptr tracking
        if src_id in self.env_is_ptr:
            self._emit_passthrough(ssa)
            self.env_is_ptr[ssa.id] = self.env_is_ptr[src_id]
            # Update type to destination
            self.env_types[ssa.id] = _mlir_to_triton_dtype(ssa.elem_type) if ssa.elem_type else "i32"
            return

        # Check if pointer type in type_str (ptr-to-ptr bitcast)
        if "!tt.ptr" in ssa.type_str:
            self._emit_passthrough(ssa)
            return

        # Value bitcast — delegate to arith.bitcast handler
        self._lower_arith_bitcast(ssa)

    def _lower_arith_bitcast(self, ssa: SSAValue):
        """arith.bitcast → reinterpret bits without changing value.

        When source and destination types differ (float <-> int),
        emit as_type<T>() in MSL. When they're the same category
        (e.g., ptr bitcast), pass through.
        """
        if not ssa.operand_ids:
            return
        src_id = ssa.operand_ids[0]
        src_var = self._lookup(src_id)
        src_dtype = self.env_types.get(src_id, "fp32")
        dst_elem = ssa.elem_type or "f32"

        src_is_float = src_dtype.startswith("fp") or src_dtype.startswith("bf")
        dst_is_float = dst_elem in ("f32", "f16", "bf16", "f64")
        dst_is_int = dst_elem.startswith("i")

        if src_is_float and dst_is_int:
            # float -> int bitcast: as_type<int>(val)
            msl_ty, dtype = _msl_int_type(dst_elem, unsigned=False)
            var_name = self._next_var("bc")
            self.kb.raw_line(f"    {msl_ty} {var_name} = as_type<{msl_ty}>({src_var});")
            self.env[ssa.id] = var_name
            self.env_types[ssa.id] = dtype
            self._propagate_shape_elementwise(ssa)
        elif not src_is_float and dst_is_float:
            # int -> float bitcast: as_type<float>(val)
            var_name = self._next_var("bc")
            self.kb.raw_line(f"    float {var_name} = as_type<float>({src_var});")
            self.env[ssa.id] = var_name
            self.env_types[ssa.id] = "fp32"
            self._propagate_shape_elementwise(ssa)
        else:
            # Same category — passthrough (shape propagation handled inside)
            self._emit_passthrough(ssa)
            # Update type to reflect the destination type
            self.env_types[ssa.id] = _mlir_to_triton_dtype(dst_elem)

    def _lower_mulhiui(self, ssa: SSAValue):
        """tt.mulhiui → upper 32 bits of unsigned 32x32→64 multiply."""
        if len(ssa.operand_ids) < 2:
            return
        a = self._lookup(ssa.operand_ids[0])
        b = self._lookup(ssa.operand_ids[1])
        var_name = self._next_var("r")
        self.kb.raw_line(f"    uint {var_name} = mulhi(as_type<uint>({a}), as_type<uint>({b}));")
        self.env[ssa.id] = var_name
        self.env_types[ssa.id] = "i32"
        # Shape: element-wise binary
        self._propagate_shape_elementwise(ssa)

    def _emit_passthrough(self, ssa: SSAValue):
        """Emit a type conversion that's a no-op in MSL (extf, truncf, etc.)."""
        if ssa.operand_ids:
            src_id = ssa.operand_ids[0]
            self.env[ssa.id] = self._lookup(src_id)
            if src_id in self.env_types:
                self.env_types[ssa.id] = self.env_types[src_id]
            if src_id in self.env_is_mask:
                self.env_is_mask[ssa.id] = True
            if src_id in self.env_is_ptr:
                self.env_is_ptr[ssa.id] = self.env_is_ptr[src_id]
            # Propagate shape: passthrough preserves shape from source,
            # unless the result type has a different shape (e.g. reshape).
            if ssa.type_str:
                out_shape = _extract_shape(ssa.type_str)
                if out_shape:
                    self.env_shapes[ssa.id] = out_shape
                elif src_id in self.env_shapes:
                    self.env_shapes[ssa.id] = self.env_shapes[src_id]
            elif src_id in self.env_shapes:
                self.env_shapes[ssa.id] = self.env_shapes[src_id]

    def _emit_cast(self, ssa: SSAValue, target_type: str, dtype: str = None):
        """Emit a type cast."""
        if not ssa.operand_ids:
            return
        a = self._lookup(ssa.operand_ids[0])
        var_name = self._next_var("r")
        self.kb.raw_line(f"    {target_type} {var_name} = static_cast<{target_type}>({a});")
        self.env[ssa.id] = var_name
        if dtype:
            self.env_types[ssa.id] = dtype
        elif target_type == "float":
            self.env_types[ssa.id] = "fp32"
        else:
            # Use the elem_type from MLIR when available
            self.env_types[ssa.id] = _mlir_to_triton_dtype(ssa.elem_type) if ssa.elem_type else "i32"
        # Shape: cast preserves shape from source operand
        self._propagate_shape_elementwise(ssa)

    def _emit_uitofp(self, ssa: SSAValue):
        """Emit unsigned-int-to-float conversion.

        Unlike sitofp, we must first cast the source to its unsigned MSL type
        to prevent sign extension. E.g., for i8 value 241 stored as char(-15),
        static_cast<float>(char(-15)) = -15.0 (wrong), but
        static_cast<float>(static_cast<uchar>(char(-15))) = 241.0 (correct).
        """
        if not ssa.operand_ids:
            return
        src_id = ssa.operand_ids[0]
        a = self._lookup(src_id)
        var_name = self._next_var("r")
        # Determine the source integer type so we can cast to unsigned first
        src_dtype = self.env_types.get(src_id, "i32")
        if src_dtype in _UINT_TYPE_MAP:
            # Source is already tracked as unsigned — direct cast is fine
            # But the MSL variable may still be signed, so always go through unsigned
            src_unsigned_ty, _ = _UINT_TYPE_MAP[src_dtype]
            self.kb.raw_line(
                f"    float {var_name} = static_cast<float>"
                f"(static_cast<{src_unsigned_ty}>({a}));"
            )
        elif src_dtype.startswith("u"):
            # Source is already unsigned (u8, u16, etc.) — direct cast is fine
            self.kb.raw_line(f"    float {var_name} = static_cast<float>({a});")
        else:
            # Source is a signed integer type (i8, i16, etc.) — cast via unsigned
            # to get the correct unsigned interpretation
            src_unsigned_ty, _ = _msl_int_type(src_dtype, unsigned=True)
            self.kb.raw_line(
                f"    float {var_name} = static_cast<float>"
                f"(static_cast<{src_unsigned_ty}>({a}));"
            )
        self.env[ssa.id] = var_name
        self.env_types[ssa.id] = "fp32"
        # Shape: uitofp preserves shape
        self._propagate_shape_elementwise(ssa)

    def _emit_int_cast(self, ssa: SSAValue, unsigned: bool = False):
        """Emit an integer sign-extend, zero-extend, or truncation cast.

        Maps the result type from ssa.elem_type to the correct MSL integer
        type and emits a static_cast. This is needed for arith.extsi,
        arith.extui, and arith.trunci which change integer bitwidths.

        For arith.extui (unsigned=True), we must first cast the source to
        its unsigned equivalent before extending, to prevent sign extension.
        E.g., char(-15) as uchar = 241, then uint(241) = 241 (not 4294967281).
        """
        if not ssa.operand_ids:
            return
        src_id = ssa.operand_ids[0]
        a = self._lookup(src_id)
        # Determine the target type from the MLIR result type
        msl_ty, dtype = _msl_int_type(ssa.elem_type, unsigned=unsigned)
        var_name = self._next_var("r")

        if unsigned and ssa.op == "arith.extui":
            # For unsigned extension, first cast source to unsigned of same
            # width to prevent sign extension, then extend to target width.
            src_dtype = self.env_types.get(src_id, "i32")
            # Get unsigned version of source type
            src_unsigned_ty, _ = _msl_int_type(src_dtype, unsigned=True)
            self.kb.raw_line(
                f"    {msl_ty} {var_name} = static_cast<{msl_ty}>"
                f"(static_cast<{src_unsigned_ty}>({a}));"
            )
        elif ssa.op == "arith.trunci":
            # For integer truncation to narrow types (char, short), the source
            # may actually be a float (e.g. from simd_sum which always returns
            # float). Direct float→char saturates in Metal. Cast through int
            # first: float→int (truncates) → char (wraps modularly).
            if msl_ty in ("char", "short", "uchar", "ushort"):
                self.kb.raw_line(
                    f"    {msl_ty} {var_name} = static_cast<{msl_ty}>"
                    f"(static_cast<int>({a}));")
            else:
                self.kb.raw_line(f"    {msl_ty} {var_name} = static_cast<{msl_ty}>({a});")
        else:
            self.kb.raw_line(f"    {msl_ty} {var_name} = static_cast<{msl_ty}>({a});")

        self.env[ssa.id] = var_name
        self.env_types[ssa.id] = dtype
        # Propagate ptr/mask info
        if src_id in self.env_is_mask:
            self.env_is_mask[ssa.id] = True
        if src_id in self.env_is_ptr:
            self.env_is_ptr[ssa.id] = self.env_is_ptr[src_id]
        # Shape: integer cast preserves shape
        self._propagate_shape_elementwise(ssa)

    def _lower_cmpi(self, ssa: SSAValue):
        """arith.cmpi → comparison with unsigned cast when needed.

        Uses pred_name (from MLIR text, reliable) over pred_int (enum may
        differ between MLIR versions).
        """
        if len(ssa.operand_ids) < 2:
            return
        a = self._lookup(ssa.operand_ids[0])
        b = self._lookup(ssa.operand_ids[1])

        # Get predicate — prefer pred_name (text-parsed, authoritative)
        pred_name = ssa.attrs.get("predicate_name")
        pred_int = ssa.attrs.get("predicate")

        if pred_name and pred_name in CMPI_NAMED:
            op_str = CMPI_NAMED[pred_name]
        elif pred_int is not None and pred_int in CMPI_PREDICATES:
            op_str = CMPI_PREDICATES[pred_int]
        else:
            op_str = "<"

        # Unsigned predicates need (uint) cast for correct semantics.
        # Signed predicates need (int) cast to prevent C++ implicit
        # unsigned promotion when comparing int vs uint (e.g. int32 vs
        # zero-extended uint8 → uint32).
        is_unsigned = pred_name in ("ult", "ule", "ugt", "uge") if pred_name else False
        is_signed = pred_name in ("slt", "sle", "sgt", "sge") if pred_name else False

        var_name = self._next_var("mask")
        if is_unsigned:
            self.kb.raw_line(f"    bool {var_name} = (uint){a} {op_str} (uint){b};")
        elif is_signed:
            self.kb.raw_line(f"    bool {var_name} = (int){a} {op_str} (int){b};")
        else:
            self.kb.raw_line(f"    bool {var_name} = {a} {op_str} {b};")
        self.env[ssa.id] = var_name
        self.env_is_mask[ssa.id] = True
        self.env_types[ssa.id] = "i1"
        # Shape: comparison inherits shape from operands
        self._propagate_shape_elementwise(ssa)

    def _lower_cmpf(self, ssa: SSAValue):
        """arith.cmpf → float comparison with NaN-aware unordered predicates.

        pred_name (from MLIR text parsing) is the primary predicate source.
        pred_int is used as fallback only — its enum values can differ between
        MLIR/Triton versions, so we don't hardcode a mapping.
        """
        if len(ssa.operand_ids) < 2:
            return
        a = self._lookup(ssa.operand_ids[0])
        b = self._lookup(ssa.operand_ids[1])

        pred_name = ssa.attrs.get("predicate_name")
        pred_int = ssa.attrs.get("predicate")

        var_name = self._next_var("mask")

        # Use pred_name as primary source. Fall back to pred_int for op_str only.
        if pred_name == "false":
            self.kb.raw_line(f"    bool {var_name} = false;")
        elif pred_name == "true":
            self.kb.raw_line(f"    bool {var_name} = true;")
        elif pred_name == "uno":
            self.kb.raw_line(f"    bool {var_name} = isnan({a}) || isnan({b});")
        elif pred_name == "ord":
            self.kb.raw_line(f"    bool {var_name} = !isnan({a}) && !isnan({b});")
        elif pred_name == "une":
            # MSL != matches IEEE 754 une semantics (NaN != x is true)
            self.kb.raw_line(f"    bool {var_name} = {a} != {b};")
        elif pred_name and pred_name in CMPF_NAMED:
            op_str = CMPF_NAMED[pred_name]
            if pred_name.startswith("u"):
                self.kb.raw_line(
                    f"    bool {var_name} = isnan({a}) || isnan({b}) || ({a} {op_str} {b});"
                )
            else:
                self.kb.raw_line(f"    bool {var_name} = {a} {op_str} {b};")
        elif pred_int is not None and pred_int in CMPF_PREDICATES:
            # Fallback to pred_int when pred_name unavailable
            op_str = CMPF_PREDICATES[pred_int]
            self.kb.raw_line(f"    bool {var_name} = {a} {op_str} {b};")
        else:
            self.kb.raw_line(f"    bool {var_name} = {a} < {b};")

        self.env[ssa.id] = var_name
        self.env_is_mask[ssa.id] = True
        self.env_types[ssa.id] = "i1"
        # Shape: comparison inherits shape from operands
        self._propagate_shape_elementwise(ssa)

    def _lower_select(self, ssa: SSAValue):
        """arith.select → ternary operator with inferred type."""
        if len(ssa.operand_ids) < 3:
            return
        cond = self._lookup(ssa.operand_ids[0])
        true_val = self._lookup(ssa.operand_ids[1])
        false_val = self._lookup(ssa.operand_ids[2])
        var_name = self._next_var("r")

        # Infer type from the true-value operand
        true_dtype = self.env_types.get(ssa.operand_ids[1], "fp32")
        if true_dtype.startswith("fp") or true_dtype.startswith("bf"):
            ty = "float"
            dtype = "fp32"
        elif true_dtype.startswith("u"):
            ty = "uint"
            dtype = "u32"
        else:
            ty = "int"
            dtype = "i32"

        self.kb.raw_line(f"    {ty} {var_name} = {cond} ? {true_val} : {false_val};")
        self.env[ssa.id] = var_name
        self.env_types[ssa.id] = dtype
        # Shape: select inherits shape from operands (cond, true, false)
        self._propagate_shape_elementwise(ssa)

    # -- Math ops --

    def _lower_math(self, ssa: SSAValue):
        """Lower math.* operations to MSL intrinsics."""
        op = ssa.op
        if not ssa.operand_ids:
            return

        # Map math ops to MSL functions
        unary_map = {
            "math.exp": "exp",
            "math.exp2": "exp2",
            "math.log": "log",
            "math.log2": "log2",
            "math.sqrt": "sqrt",
            "math.rsqrt": "rsqrt",
            "math.abs": "abs",
            "math.absf": "abs",
            "math.sin": "sin",
            "math.cos": "cos",
            "math.tanh": "tanh",
            "math.floor": "floor",
            "math.ceil": "ceil",
            "math.round": "round",
        }

        if op in unary_map:
            a = self._lookup(ssa.operand_ids[0])
            fn = unary_map[op]
            var_name = self._next_var("r")
            self.kb.raw_line(f"    float {var_name} = {fn}({a});")
            self.env[ssa.id] = var_name
            self.env_types[ssa.id] = "fp32"

        elif op == "math.absi":
            # Integer absolute value
            a = self._lookup(ssa.operand_ids[0])
            var_name = self._next_var("r")
            ty, dtype = _msl_int_type(ssa.elem_type, unsigned=False)
            self.kb.raw_line(f"    {ty} {var_name} = abs({a});")
            self.env[ssa.id] = var_name
            self.env_types[ssa.id] = dtype

        elif op == "math.fma":
            if len(ssa.operand_ids) >= 3:
                a = self._lookup(ssa.operand_ids[0])
                b = self._lookup(ssa.operand_ids[1])
                c = self._lookup(ssa.operand_ids[2])
                var_name = self._next_var("r")
                self.kb.raw_line(f"    float {var_name} = fma({a}, {b}, {c});")
                self.env[ssa.id] = var_name

        elif op == "math.erf":
            # MSL has no erf() — Abramowitz & Stegun approximation (max error ~1.5e-7)
            a = self._lookup(ssa.operand_ids[0])
            abs_var = self._next_var("erf_abs")
            t_var = self._next_var("erf_t")
            y_var = self._next_var("erf_y")
            var_name = self._next_var("r")
            self.kb.raw_line(f"    float {abs_var} = abs({a});")
            self.kb.raw_line(f"    float {t_var} = 1.0f / (1.0f + 0.3275911f * {abs_var});")
            self.kb.raw_line(
                f"    float {y_var} = 1.0f - (((((1.061405429f * {t_var} "
                f"- 1.453152027f) * {t_var}) + 1.421413741f) * {t_var} "
                f"- 0.284496736f) * {t_var} + 0.254829592f) * {t_var} "
                f"* exp(-{abs_var} * {abs_var});"
            )
            self.kb.raw_line(f"    float {var_name} = copysign({y_var}, {a});")
            self.env[ssa.id] = var_name
            self.env_types[ssa.id] = "fp32"

        elif op == "math.powf":
            if len(ssa.operand_ids) >= 2:
                a = self._lookup(ssa.operand_ids[0])
                b = self._lookup(ssa.operand_ids[1])
                var_name = self._next_var("r")
                self.kb.raw_line(f"    float {var_name} = pow({a}, {b});")
                self.env[ssa.id] = var_name
                self.env_types[ssa.id] = "fp32"

        elif op == "math.copysign":
            if len(ssa.operand_ids) >= 2:
                a = self._lookup(ssa.operand_ids[0])
                b = self._lookup(ssa.operand_ids[1])
                var_name = self._next_var("r")
                self.kb.raw_line(f"    float {var_name} = copysign({a}, {b});")
                self.env[ssa.id] = var_name
                self.env_types[ssa.id] = "fp32"

        elif op == "math.atan2":
            if len(ssa.operand_ids) >= 2:
                a = self._lookup(ssa.operand_ids[0])
                b = self._lookup(ssa.operand_ids[1])
                var_name = self._next_var("r")
                self.kb.raw_line(f"    float {var_name} = atan2({a}, {b});")
                self.env[ssa.id] = var_name
                self.env_types[ssa.id] = "fp32"

        elif op == "math.roundeven":
            a = self._lookup(ssa.operand_ids[0])
            var_name = self._next_var("r")
            self.kb.raw_line(f"    float {var_name} = rint({a});")
            self.env[ssa.id] = var_name
            self.env_types[ssa.id] = "fp32"

        elif op == "math.trunc":
            a = self._lookup(ssa.operand_ids[0])
            var_name = self._next_var("r")
            self.kb.raw_line(f"    float {var_name} = trunc({a});")
            self.env[ssa.id] = var_name
            self.env_types[ssa.id] = "fp32"

        elif op == "math.log1p":
            # log1p(x) = log(1 + x), more numerically stable near zero
            a = self._lookup(ssa.operand_ids[0])
            var_name = self._next_var("r")
            self.kb.raw_line(f"    float {var_name} = log(1.0f + {a});")
            self.env[ssa.id] = var_name
            self.env_types[ssa.id] = "fp32"

        elif op == "math.expm1":
            # expm1(x) = exp(x) - 1, more numerically stable near zero
            a = self._lookup(ssa.operand_ids[0])
            var_name = self._next_var("r")
            self.kb.raw_line(f"    float {var_name} = (exp({a}) - 1.0f);")
            self.env[ssa.id] = var_name
            self.env_types[ssa.id] = "fp32"

        else:
            self.kb.comment(f"UNSUPPORTED math: {op}")
            return
        # Shape: all math ops are element-wise — inherit from operands
        self._propagate_shape_elementwise(ssa)

    def _lower_precise_math(self, ssa: SSAValue, kind: str):
        """Lower tt.precise_sqrt / tt.precise_divf to MSL."""
        if kind == "sqrt":
            a = self._lookup(ssa.operand_ids[0])
            var_name = self._next_var("r")
            self.kb.raw_line(f"    float {var_name} = precise::sqrt({a});")
            self.env[ssa.id] = var_name
            self.env_types[ssa.id] = "fp32"
        elif kind == "divf":
            a = self._lookup(ssa.operand_ids[0])
            b = self._lookup(ssa.operand_ids[1])
            var_name = self._next_var("r")
            self.kb.raw_line(f"    float {var_name} = precise::divide({a}, {b});")
            self.env[ssa.id] = var_name
            self.env_types[ssa.id] = "fp32"
        # Shape: precise math is element-wise
        self._propagate_shape_elementwise(ssa)

    # -- Extern elementwise --

    def _lower_extern_elementwise(self, ssa: SSAValue):
        """tt.extern_elementwise → direct MSL function call.

        Handles the common case where the extern function maps to a Metal
        standard library function (e.g., sin, cos, exp, etc.).

        The symbol name is extracted from the op's attributes. The TTGIR text
        typically contains: tt.extern_elementwise ... {symbol = "func_name", ...}
        The walker stores raw attributes, so we check for 'symbol', 'libname',
        and 'pure' attributes.
        """
        # Extract function name from attributes
        func_name = ssa.attrs.get("symbol", "")
        if not func_name:
            func_name = ssa.attrs.get("libname", "")
        if not func_name:
            # Fallback: try to extract from the raw_line or op string
            self.kb.comment(f"UNSUPPORTED: tt.extern_elementwise (no symbol)")
            return

        # Sanitize function name for MSL (strip leading underscores from __nv_* etc.)
        # Common pattern: __nv_sinf → sin, __nv_expf → exp
        safe_name = func_name
        if safe_name.startswith("__nv_"):
            # CUDA libdevice function — strip prefix and trailing 'f' if present
            stripped = safe_name[5:]  # remove "__nv_"
            if stripped.endswith("f") and len(stripped) > 1:
                stripped = stripped[:-1]
            safe_name = stripped

        # Build argument list
        args = [self._lookup(oid) for oid in ssa.operand_ids]
        args_str = ", ".join(args)

        # Determine result type
        elem = ssa.elem_type or "f32"
        triton_dtype = _mlir_to_triton_dtype(elem)
        if triton_dtype.startswith("fp") or triton_dtype.startswith("bf"):
            msl_ty = "float"
        elif triton_dtype.startswith("u"):
            msl_ty = "uint"
        elif triton_dtype == "i64":
            msl_ty = "long"
        else:
            msl_ty = triton_type_to_msl(triton_dtype)

        var_name = self._next_var("r")
        self.kb.raw_line(f"    {msl_ty} {var_name} = {safe_name}({args_str});")
        self.env[ssa.id] = var_name
        self.env_types[ssa.id] = triton_dtype

    # -- Transpose --

    def _lower_tt_trans(self, ssa: SSAValue):
        """tt.trans → shared memory transpose.

        In TTGIR, tt.trans {order = array<i32: 1, 0>} swaps dimensions.
        For a 2D tensor of shape (M, N), the transpose requires each
        thread to exchange data with another thread:
        - Thread at (row, col) in source needs value from (col, row) in source
        - Source lid → (lid/N, lid%N), transposed → (lid%N, lid/N)
        - Source lid for transposed value: (lid%N)*N + (lid/N) → wrong!
        - Source lid for transposed value: (lid%M)*M + (lid/M) if target is (N,M)

        Uses threadgroup shared memory for the data exchange.
        """
        if not ssa.operand_ids:
            return
        src_id = ssa.operand_ids[0]
        src_var = self._lookup(src_id)

        # Get source and destination shapes
        src_shape = _extract_shape(
            # Find source op's type_str
            self._find_op_type_str(src_id)
        )
        dst_shape = _extract_shape(ssa.type_str)

        if len(src_shape) < 2 or not self._is_2d:
            # 1D or unknown — passthrough
            self._emit_passthrough(ssa)
            return

        M, N = src_shape[0], src_shape[1]
        total = M * N

        # Determine types
        input_dtype = self.env_types.get(src_id, "fp32")
        is_float = input_dtype.startswith("fp") or input_dtype.startswith("bf")
        msl_type = "float" if is_float else "int"
        shared_dtype = "fp32" if is_float else "i32"

        # Allocate shared memory for transpose
        shared_name = f"trans_shared_{self._shared_counter}"
        self._shared_counter += 1
        self.kb.declare_threadgroup_array(shared_name, dtype=shared_dtype, size=total)

        result_var = self._next_var("trans")

        # Write to shared in row-major order
        self.kb.raw_line(f"    {shared_name}[lid] = {src_var};")
        self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Read from transposed position
        # Source: (row, col) = (lid/N, lid%N) → linear lid
        # Transposed: want value from (col, row) = (lid%N, lid/N)
        # → linear index in source: (lid%N)*N + (lid/N)... wait, that's wrong.
        # Source stored as row-major (M rows, N cols): index = row*N + col
        # We want element at transposed position: (col, row) in source
        # = source[col * N + row]... no, source is M×N row-major.
        # Source element (r, c) is at index r*N + c.
        # After transpose, output position (i, j) = source (j, i).
        # Output is N×M. Thread lid in output maps to (lid/M, lid%M).
        # We want source value at (lid%M, lid/M) = source[(lid%M)*N + (lid/M)].
        self.kb.raw_line(
            f"    {msl_type} {result_var} = {shared_name}["
            f"(lid % {M}u) * {N}u + (lid / {M}u)];"
        )

        self.env[ssa.id] = result_var
        self.env_types[ssa.id] = input_dtype
        if dst_shape:
            self.env_shapes[ssa.id] = dst_shape

    def _find_op_type_str(self, ssa_id: int) -> str:
        """Find the type_str for an SSA value by searching ops."""
        for ssa in self.graph.ops:
            if ssa.id == ssa_id:
                return ssa.type_str
            if ssa.region_ops:
                for inner in ssa.region_ops:
                    if inner.id == ssa_id:
                        return inner.type_str
            if ssa.else_ops:
                for inner in ssa.else_ops:
                    if inner.id == ssa_id:
                        return inner.type_str
        # Check args
        for arg in self.graph.args:
            if arg.id == ssa_id:
                return arg.type_str
        return ""

    # -- Concatenation --

    def _lower_tt_join(self, ssa: SSAValue):
        """tt.join → fused cat (join + trans + reshape = concatenation).

        In Triton, tl.cat(a, b, can_reorder=False) compiles to:
          tt.join(a, b) → tensor<Nx2>
          tt.trans → tensor<2xN>
          tt.reshape → tensor<2*N>

        The result is concatenation: [a[0]..a[N-1], b[0]..b[N-1]].

        Since the kernel runs in 2D mode (due to intermediate 2D shapes),
        make_range(0, N) maps to lid % N, so all 2*N threads have valid
        loaded values. The fused cat is simply:
          result = (lid < N) ? a_val : b_val
        """
        if len(ssa.operand_ids) < 2:
            self.kb.comment("UNSUPPORTED: tt.join with < 2 operands")
            return

        a_id, b_id = ssa.operand_ids[0], ssa.operand_ids[1]
        a_var = self._lookup(a_id)
        b_var = self._lookup(b_id)

        # Get input size N from operand shape
        src_shape = _extract_shape(self._find_op_type_str(a_id))
        N = src_shape[0] if src_shape else self.graph.block_size

        # Detect the join → trans → reshape pattern
        trans_ssa = None
        reshape_ssa = None
        for op in self.graph.ops:
            if op.op == "tt.trans" and ssa.id in op.operand_ids:
                trans_ssa = op
                break
        if trans_ssa:
            for op in self.graph.ops:
                if op.op == "tt.reshape" and trans_ssa.id in op.operand_ids:
                    reshape_ssa = op
                    break

        # Determine type
        input_dtype = self.env_types.get(a_id, "fp32")
        is_float = input_dtype.startswith("fp") or input_dtype.startswith("bf")
        msl_type = "float" if is_float else "int"
        if input_dtype == "fp16":
            msl_type = "half"
        elif input_dtype == "bf16":
            msl_type = "bfloat"

        result_var = self._next_var("cat")

        if trans_ssa and reshape_ssa:
            # Fused cat: in 2D mode, all threads have valid values via wrapping
            self.kb.raw_line(
                f"    {msl_type} {result_var} = (lid < {N}u) ? {a_var} : {b_var};"
            )
            # Register result for all intermediate SSA ids
            self.env[ssa.id] = result_var
            self.env_types[ssa.id] = input_dtype
            self.env[trans_ssa.id] = result_var
            self.env_types[trans_ssa.id] = input_dtype
            self.env[reshape_ssa.id] = result_var
            self.env_types[reshape_ssa.id] = input_dtype
            # Skip the trans and reshape ops
            self._skip_ids.add(trans_ssa.id)
            self._skip_ids.add(reshape_ssa.id)
        else:
            # Standalone join (no trans+reshape) — use shared memory
            shared_a = f"join_shared_a_{self._shared_counter}"
            shared_b = f"join_shared_b_{self._shared_counter}"
            self._shared_counter += 1
            shared_dtype = "fp32" if is_float else "i32"
            self.kb.declare_threadgroup_array(shared_a, dtype=shared_dtype, size=N)
            self.kb.declare_threadgroup_array(shared_b, dtype=shared_dtype, size=N)

            # Stage from threads 0..N-1
            self.kb.raw_line(f"    if (lid < {N}u) {{")
            self.kb.raw_line(f"        {shared_a}[lid] = {a_var};")
            self.kb.raw_line(f"        {shared_b}[lid] = {b_var};")
            self.kb.raw_line(f"    }}")
            self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")
            # Interleave: join[i, 0] = a[i], join[i, 1] = b[i]
            # Linear (row-major): join[lid] = a[lid/2] if lid%2==0, b[lid/2] if lid%2==1
            self.kb.raw_line(
                f"    {msl_type} {result_var} = (lid % 2u == 0u) ? "
                f"{shared_a}[lid / 2u] : {shared_b}[lid / 2u];"
            )
            self.env[ssa.id] = result_var
            self.env_types[ssa.id] = input_dtype
            dst_shape = _extract_shape(ssa.type_str)
            if dst_shape:
                self.env_shapes[ssa.id] = dst_shape

    def _lower_tt_cat(self, ssa: SSAValue):
        """tt.cat → concatenation using shared memory.

        In Triton, tl.cat(a, b, can_reorder=True) may compile directly to
        tt.cat(a, b) → tensor<2*N>. The kernel is 1D with block_size=2*N.

        Since make_range(0, N) maps to lid (1D mode), only threads 0..N-1
        have valid loaded values. Use shared memory to stage and redistribute.
        """
        if len(ssa.operand_ids) < 2:
            self.kb.comment("UNSUPPORTED: tt.cat with < 2 operands")
            return

        a_id, b_id = ssa.operand_ids[0], ssa.operand_ids[1]
        a_var = self._lookup(a_id)
        b_var = self._lookup(b_id)

        # Get input size N
        src_shape = _extract_shape(self._find_op_type_str(a_id))
        N = src_shape[0] if src_shape else self.graph.block_size

        # Determine type
        input_dtype = self.env_types.get(a_id, "fp32")
        is_float = input_dtype.startswith("fp") or input_dtype.startswith("bf")
        msl_type = "float" if is_float else "int"
        shared_dtype = "fp32" if is_float else "i32"
        if input_dtype == "fp16":
            msl_type = "half"
            shared_dtype = "fp16"
        elif input_dtype == "bf16":
            msl_type = "bfloat"
            shared_dtype = "bf16"

        # Allocate shared memory for both halves
        shared_a = f"cat_shared_a_{self._shared_counter}"
        shared_b = f"cat_shared_b_{self._shared_counter}"
        self._shared_counter += 1
        self.kb.declare_threadgroup_array(shared_a, dtype=shared_dtype, size=N)
        self.kb.declare_threadgroup_array(shared_b, dtype=shared_dtype, size=N)

        result_var = self._next_var("cat")

        # Stage: only threads 0..N-1 have valid loaded values
        self.kb.raw_line(f"    if (lid < {N}u) {{")
        self.kb.raw_line(f"        {shared_a}[lid] = {a_var};")
        self.kb.raw_line(f"        {shared_b}[lid] = {b_var};")
        self.kb.raw_line(f"    }}")
        self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Read: cat[lid] = a[lid] for lid < N, b[lid-N] for lid >= N
        self.kb.raw_line(
            f"    {msl_type} {result_var} = (lid < {N}u) ? "
            f"{shared_a}[lid] : {shared_b}[lid - {N}u];"
        )

        self.env[ssa.id] = result_var
        self.env_types[ssa.id] = input_dtype
        dst_shape = _extract_shape(ssa.type_str)
        if dst_shape:
            self.env_shapes[ssa.id] = dst_shape

    def _lower_tt_split(self, ssa: SSAValue):
        """tt.split → de-interleave using shared memory.

        Takes tensor<Nx2> and produces two tensor<N> results.
        split[i, 0] = z1[i], split[i, 1] = z2[i].

        In per-thread model with 2*N threads, each thread has one value
        from the flat input. Stage to shared memory, then read even/odd.
        """
        if not ssa.operand_ids:
            self.kb.comment("UNSUPPORTED: tt.split with no operands")
            return
        if not ssa.result_ids or len(ssa.result_ids) < 2:
            self.kb.comment("UNSUPPORTED: tt.split with < 2 results")
            return

        src_id = ssa.operand_ids[0]
        src_var = self._lookup(src_id)

        # Get input shape (N, 2)
        src_shape = _extract_shape(self._find_op_type_str(src_id))
        if src_shape and len(src_shape) >= 2:
            N = src_shape[0]
        else:
            N = self.effective_block_size // 2

        total = N * 2

        # Determine types
        input_dtype = self.env_types.get(src_id, "i32")
        is_float = input_dtype.startswith("fp") or input_dtype.startswith("bf")
        msl_type = "float" if is_float else "int"
        shared_dtype = "fp32" if is_float else "i32"
        if input_dtype == "fp16":
            msl_type = "half"
            shared_dtype = "fp16"

        # Allocate shared memory
        shared_name = f"split_shared_{self._shared_counter}"
        self._shared_counter += 1
        self.kb.declare_threadgroup_array(shared_name, dtype=shared_dtype, size=total)

        # Stage all values to shared memory
        self.kb.raw_line(f"    {shared_name}[lid] = {src_var};")
        self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Read de-interleaved: z1 = even elements, z2 = odd elements
        z1_var = self._next_var("split")
        z2_var = self._next_var("split")
        self.kb.raw_line(
            f"    {msl_type} {z1_var} = {shared_name}[(lid % {N}u) * 2u];"
        )
        self.kb.raw_line(
            f"    {msl_type} {z2_var} = {shared_name}[(lid % {N}u) * 2u + 1u];"
        )

        # Register both results
        rid1, rid2 = ssa.result_ids[0], ssa.result_ids[1]
        self.env[rid1] = z1_var
        self.env_types[rid1] = input_dtype
        self.env[rid2] = z2_var
        self.env_types[rid2] = input_dtype
        out_shape = (N,)
        self.env_shapes[rid1] = out_shape
        self.env_shapes[rid2] = out_shape

    def _lower_tt_histogram(self, ssa: SSAValue):
        """tt.histogram → threadgroup atomic histogram.

        Input: tensor<Mxi32> of values in [0, N).
        Output: tensor<Nxi32> of bin counts.

        Uses threadgroup atomic_int array for thread-safe counting.
        """
        if not ssa.operand_ids:
            self.kb.comment("UNSUPPORTED: tt.histogram with no operands")
            return

        input_var = self._lookup(ssa.operand_ids[0])

        # Get N (number of bins) from output type
        out_shape = _extract_shape(ssa.type_str)
        if out_shape:
            N = out_shape[0]
        else:
            N = self.effective_block_size

        # Get M (input size) from input type
        in_shape = _extract_shape(self._find_op_type_str(ssa.operand_ids[0]))
        M = in_shape[0] if in_shape else self.effective_block_size

        # Allocate threadgroup atomic histogram
        hist_name = f"hist_{self._shared_counter}"
        self._shared_counter += 1
        # Declare as regular int array — use atomic operations on it
        self.kb.raw_line(f"    threadgroup atomic_int {hist_name}[{N}];")

        # Initialize bins to 0
        self.kb.raw_line(f"    if (lid < {N}u) atomic_store_explicit(&{hist_name}[lid], 0, memory_order_relaxed);")
        self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Each thread with valid input increments its bin (respecting mask if present)
        mask_cond = f"lid < {M}u"
        if len(ssa.operand_ids) >= 2:
            mask_var = self._lookup(ssa.operand_ids[1])
            mask_cond = f"lid < {M}u && {mask_var}"
        self.kb.raw_line(f"    if ({mask_cond}) atomic_fetch_add_explicit(&{hist_name}[(uint){input_var}], 1, memory_order_relaxed);")
        self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Read result: each thread reads its assigned bin
        result_var = self._next_var("hist")
        self.kb.raw_line(f"    int {result_var} = (lid < {N}u) ? atomic_load_explicit(&{hist_name}[lid], memory_order_relaxed) : 0;")

        self.env[ssa.id] = result_var
        self.env_types[ssa.id] = "i32"
        self.env_shapes[ssa.id] = (N,)

    def _lower_tt_gather(self, ssa: SSAValue):
        """tt.gather → shared memory indexed lookup.

        For 1D: src (tensor<Sxf32>), indices (tensor<Ixi32>) → result (tensor<Ixf32>).
        Stage src to shared memory, each thread reads shared[indices[lid]].
        """
        if len(ssa.operand_ids) < 2:
            self.kb.comment("UNSUPPORTED: tt.gather with < 2 operands")
            return

        src_var = self._lookup(ssa.operand_ids[0])
        idx_var = self._lookup(ssa.operand_ids[1])

        # Get source size from type
        src_shape = _extract_shape(self._find_op_type_str(ssa.operand_ids[0]))
        S = src_shape[0] if src_shape else self.effective_block_size

        # Determine types
        src_dtype = self.env_types.get(ssa.operand_ids[0], "fp32")
        is_float = src_dtype.startswith("fp") or src_dtype.startswith("bf")
        msl_type = "float" if is_float else "int"
        shared_dtype = "fp32" if is_float else "i32"

        # Allocate shared memory for source
        shared_name = f"gather_shared_{self._shared_counter}"
        self._shared_counter += 1
        self.kb.declare_threadgroup_array(shared_name, dtype=shared_dtype, size=S)

        # Stage source to shared (only threads with valid src index)
        self.kb.raw_line(f"    if (lid < {S}u) {shared_name}[lid] = {src_var};")
        self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Each thread gathers: result = shared[idx]
        result_var = self._next_var("gathered")
        self.kb.raw_line(f"    {msl_type} {result_var} = {shared_name}[(uint){idx_var}];")

        self.env[ssa.id] = result_var
        self.env_types[ssa.id] = src_dtype
        out_shape = _extract_shape(ssa.type_str)
        if out_shape:
            self.env_shapes[ssa.id] = tuple(out_shape)

    # -- Map elementwise --

    def _lower_map_elementwise(self, ssa: SSAValue):
        """tt.map_elementwise → apply body per element.

        The body region contains basic blocks with cf.cond_br (conditional
        branches) forming a decision tree. We convert this to nested
        ternary expressions in MSL.

        TTGIR pattern:
            %z = "tt.map_elementwise"(%x, %y) <{pack = 1}> ({
            ^bb0(%a: i32, %b: i32):
                %cmp = arith.cmpi slt, %a, %b : i32
                cf.cond_br %cmp, ^bb2(%c-1), ^bb1
            ^bb1:
                %cmp2 = arith.cmpi eq, %a, %b : i32
                cf.cond_br %cmp2, ^bb2(%c0), ^bb2(%c1)
            ^bb2(%result: i32):
                tt.map_elementwise.return %result : i32
            })

        MSL output:
            int v42 = (v10 < v11) ? -1 : ((v10 == v11) ? 0 : 1);
        """
        if not ssa.region_ops:
            self._emit_passthrough(ssa)
            return

        # Get input operands
        input_vars = [self.env.get(oid, f"v{oid}") for oid in ssa.operand_ids]
        msl_type = triton_type_to_msl(ssa.elem_type) if ssa.elem_type else "int"

        # Parse the body to extract the decision tree from the raw TTGIR text.
        # The region_ops contain all ops across basic blocks, but cf.cond_br
        # targets (^bb labels) are only in the raw text. We need to reconstruct
        # the control flow from the ops' _block_id attributes and cf.cond_br args.
        #
        # Strategy: Process ops by basic block. Each cf.cond_br creates a branch.
        # We build a nested ternary expression by resolving the branch targets.

        # Group ops by basic block
        blocks = {}  # block_id → list of ops
        block_order = []
        for op in ssa.region_ops:
            bid = op.attrs.get("_block_id", 0)
            if bid not in blocks:
                blocks[bid] = []
                block_order.append(bid)
            blocks[bid].append(op)

        # Find block args (bb entry parameters)
        block_args = {}  # block_id → list of arg SSA ids
        for op in ssa.region_ops:
            if op.op in ("tt.map_elementwise.return", "tt.reduce.return"):
                continue

        # Simple case: no cf.cond_br/cf.br (direct computation)
        has_cond_br = any(op.op in ("cf.cond_br", "cf.br") for op in ssa.region_ops)

        if not has_cond_br:
            # Direct lowering: process body ops with input bindings
            bb_args = ssa.attrs.get("block_arg_ids", [])
            for i, arg_id in enumerate(bb_args):
                if i < len(input_vars):
                    self.env[arg_id] = input_vars[i]
                    self.env_types[arg_id] = ssa.elem_type or "i32"

            for op in ssa.region_ops:
                if op.op == "tt.map_elementwise.return":
                    if op.operand_ids:
                        result_var = self.env.get(op.operand_ids[0], f"v{op.operand_ids[0]}")
                        var_name = f"v{ssa.id}"
                        self.kb.raw_line(f"    {msl_type} {var_name} = {result_var};")
                        self.env[ssa.id] = var_name
                        self.env_types[ssa.id] = ssa.elem_type or "i32"
                else:
                    self._lower_op_dispatch(op)
            return

        # Complex case: cf.cond_br creates a decision tree
        # Parse from raw TTGIR text to resolve branch targets and block args
        self._lower_map_elementwise_cond_br(ssa, input_vars, msl_type)

    def _lower_map_elementwise_cond_br(self, ssa, input_vars, msl_type):
        """Lower map_elementwise with cf.cond_br decision tree.

        Reconstructs the basic block graph from region_ops and converts
        cf.cond_br branches to nested ternary/if-else expressions.

        cf.cond_br operand_ids are: [condition, true_args..., false_args...].
        The split between true/false args comes from n_true_operands/n_false_operands
        attrs (parsed from TTGIR text by the walker).
        """
        # Group ops by block
        blocks = {}
        block_order = []
        for op in ssa.region_ops:
            bid = op.attrs.get("_block_id", 0)
            if bid not in blocks:
                blocks[bid] = []
                block_order.append(bid)
            blocks[bid].append(op)

        # Bind entry block args to input vars
        bb_arg_ids = ssa.attrs.get("block_arg_ids", [])
        for i, arg_id in enumerate(bb_arg_ids):
            if i < len(input_vars):
                self.env[arg_id] = input_vars[i]
                self.env_types[arg_id] = ssa.elem_type or "i32"

        # Declare result variable
        var_name = f"v{ssa.id}"
        self.kb.raw_line(f"    {msl_type} {var_name};")

        # Process the decision tree using structured if/else
        self._emit_cond_br_block(blocks, block_order, 0, var_name, msl_type)

        self.env[ssa.id] = var_name
        self.env_types[ssa.id] = ssa.elem_type or "i32"

    def _emit_cond_br_block(self, blocks, block_order, block_idx, result_var, msl_type):
        """Recursively emit a basic block as structured if/else."""
        if block_idx >= len(block_order):
            return

        bid = block_order[block_idx]
        ops = blocks[bid]

        for op in ops:
            if op.op == "cf.cond_br":
                cond_var = self.env.get(op.operand_ids[0], f"v{op.operand_ids[0]}") if op.operand_ids else "false"

                # Split operand_ids using walker-parsed arg counts
                n_true = op.attrs.get("n_true_operands", 0)
                n_false = op.attrs.get("n_false_operands", 0)
                true_args = op.operand_ids[1:1 + n_true]
                false_args = op.operand_ids[1 + n_true:1 + n_true + n_false]

                remaining_blocks = block_order[block_idx + 1:]

                if not remaining_blocks:
                    return

                if n_true > 0 and n_false > 0:
                    # Both branches pass values (e.g., both go to return block)
                    true_v = self.env.get(true_args[0], f"v{true_args[0]}")
                    false_v = self.env.get(false_args[0], f"v{false_args[0]}")
                    self.kb.raw_line(f"    {result_var} = {cond_var} ? {true_v} : {false_v};")
                elif n_true > 0 and n_false == 0:
                    # True branch passes value (to return block), false falls through
                    true_v = self.env.get(true_args[0], f"v{true_args[0]}")
                    self.kb.raw_line(f"    if ({cond_var}) {{")
                    self.kb.raw_line(f"        {result_var} = {true_v};")
                    self.kb.raw_line(f"    }} else {{")
                    # Recurse into the next block (false destination)
                    if remaining_blocks:
                        # Find the non-return block to recurse into
                        next_bid = remaining_blocks[0]
                        next_block_idx = block_order.index(next_bid)
                        self._emit_cond_br_block(blocks, block_order, next_block_idx, result_var, msl_type)
                    self.kb.raw_line(f"    }}")
                elif n_true == 0 and n_false > 0:
                    # True branch falls through, false passes value
                    false_v = self.env.get(false_args[0], f"v{false_args[0]}")
                    self.kb.raw_line(f"    if (!{cond_var}) {{")
                    self.kb.raw_line(f"        {result_var} = {false_v};")
                    self.kb.raw_line(f"    }} else {{")
                    if remaining_blocks:
                        next_bid = remaining_blocks[0]
                        next_block_idx = block_order.index(next_bid)
                        self._emit_cond_br_block(blocks, block_order, next_block_idx, result_var, msl_type)
                    self.kb.raw_line(f"    }}")
                return

            elif op.op == "cf.br":
                # Unconditional branch — assign args to result and stop
                if op.operand_ids:
                    val = self.env.get(op.operand_ids[0], f"v{op.operand_ids[0]}")
                    self.kb.raw_line(f"    {result_var} = {val};")
                return

            elif op.op == "tt.map_elementwise.return":
                # Return block — the block arg was set by cf.cond_br assignments
                # Nothing to emit here since result_var was set in the branches
                pass
            else:
                # Process non-terminator ops (e.g., arith.cmpi, arith.subi) in this block
                self._lower_op_dispatch(op)

    # -- Reductions --

    def _lower_reduce(self, ssa: SSAValue):
        """tt.reduce → SIMD + threadgroup shared memory reduction.

        For 1D: standard full reduction using SIMD intrinsics + shared memory.
        For 2D with axis: reduce along one dimension, keeping the other.
            axis=1 on (M, N): reduce N columns per row → (M,) result
            axis=0 on (M, N): reduce M rows per column → (N,) result
        Multi-value reduces (argmax/argmin) are dispatched to a specialized handler.
        """
        if not ssa.operand_ids:
            return

        # Detect multi-value reduce (argmax/argmin): 2+ inputs, 2+ results
        if (len(ssa.operand_ids) >= 2 and ssa.result_ids
                and len(ssa.result_ids) >= 2):
            self._lower_reduce_multi_value(ssa)
            return

        input_var = self._lookup(ssa.operand_ids[0])
        axis = ssa.attrs.get("axis", 0)

        # Determine combine op from body region
        combine_op = "sum"  # default
        if ssa.region_ops:
            has_cmpf_gt = False
            has_cmpf_lt = False
            for body_op in ssa.region_ops:
                op_name = body_op.op
                if "addf" in op_name or "addi" in op_name:
                    combine_op = "sum"
                elif "max" in op_name:
                    combine_op = "max"
                elif "min" in op_name:
                    combine_op = "min"
                elif "xor" in op_name:
                    combine_op = "xor"
                elif op_name == "arith.cmpf":
                    pred = body_op.attrs.get("predicate_name", "")
                    if "gt" in pred:
                        has_cmpf_gt = True
                    elif "lt" in pred:
                        has_cmpf_lt = True
            # cmpf ogt + select = NaN-propagating max (triton_helpers.max2)
            if combine_op == "sum" and has_cmpf_gt:
                combine_op = "max"
            elif combine_op == "sum" and has_cmpf_lt:
                combine_op = "min"

        # Determine type from input operand
        input_dtype = self.env_types.get(ssa.operand_ids[0], "fp32")
        is_int_reduce = not (
            input_dtype.startswith("fp") or input_dtype.startswith("bf")
        )
        shared_dtype = "i32" if is_int_reduce else "fp32"
        msl_type = "int" if is_int_reduce else "float"

        # Check if this is a 2D axis-specific reduction
        input_shape = self.env_shapes.get(ssa.operand_ids[0])
        if not input_shape:
            input_shape = _extract_shape(self._find_op_type_str(ssa.operand_ids[0]))

        if input_shape and len(input_shape) == 3:
            self._lower_reduce_3d(ssa, input_var, axis, combine_op,
                                  msl_type, shared_dtype, input_shape)
            return

        if self._is_2d and input_shape and len(input_shape) >= 2:
            # For triton_per_* kernels with shape (1, N) or (XBLOCK, R_BLOCK)
            # where dim_0 == 1 and axis == 1, this is really a 1D reduction
            # along the reduction dimension.  Use the efficient SIMD path,
            # not the slow sequential shared memory path.
            if input_shape[0] == 1 and axis == 1:
                pass  # Fall through to 1D SIMD reduction below
            else:
                self._lower_reduce_2d(ssa, input_var, axis, combine_op,
                                      msl_type, shared_dtype, input_shape)
                return

        # Cast bool (i1) to int before reduction — MSL SIMD intrinsics reject bool
        if input_dtype == "i1" or (isinstance(input_var, str) and input_var in ("true", "false", "1", "0")):
            cast_var = self._next_var("bool_to_int")
            self.kb.raw_line(f"    int {cast_var} = (int){input_var};")
            input_var = cast_var

        # 1D full reduction (original behavior)
        shared_name = f"shared_{self._shared_counter}"
        self._shared_counter += 1
        n_simd_groups = (self.kb.block_size + 31) // 32
        self.kb.declare_threadgroup_array(shared_name, dtype=shared_dtype, size=n_simd_groups)

        result_var = self._next_var("reduced")
        self.kb.threadgroup_reduce(combine_op, input_var, shared_name, result_var)

        # Narrow-type masking: when reducing in wider type but output is narrow,
        # apply modular arithmetic (i1 sum = XOR, i8 sum = mod 256, etc.)
        out_elem = ssa.elem_type
        if out_elem == "i1":
            masked_var = self._next_var("masked")
            self.kb.raw_line(f"    float {masked_var} = (float)((int){result_var} & 1);")
            result_var = masked_var
        elif out_elem == "i8":
            masked_var = self._next_var("masked")
            self.kb.raw_line(f"    float {masked_var} = (float)((int){result_var} & 0xFF);")
            result_var = masked_var
        elif out_elem == "i16":
            masked_var = self._next_var("masked")
            self.kb.raw_line(f"    float {masked_var} = (float)((int){result_var} & 0xFFFF);")
            result_var = masked_var

        self.env[ssa.id] = result_var
        self.env_types[ssa.id] = shared_dtype

    def _detect_reduce_direction(self, ssa: SSAValue) -> bool:
        """Detect argmax (True) vs argmin (False) from reduce body comparison ops."""
        # Float values: cmpf determines direction unambiguously
        for body_op in (ssa.region_ops or []):
            if body_op.op == "arith.cmpf":
                # Use predicate_name (string) if available, fall back to int code
                pred = body_op.attrs.get("predicate_name", "")
                if not pred:
                    # Integer predicate codes: 1=oeq, 2=ogt, 4=olt
                    code = body_op.attrs.get("predicate", -1)
                    if code == 1:
                        continue  # oeq — tie-break, skip
                    return code == 2  # ogt → max, else min
                if "eq" in pred:
                    continue  # oeq — tie-break, skip
                return "gt" in pred  # ogt → max, olt → min
        # Integer values: sgt/ugt means argmax, absence means argmin
        # (slt is always present for index tie-break, so it's not distinctive)
        for body_op in (ssa.region_ops or []):
            if body_op.op == "arith.cmpi":
                pred = body_op.attrs.get("predicate_name", "")
                if not pred:
                    code = body_op.attrs.get("predicate", -1)
                    if code in (4, 8):  # sgt=4, ugt=8
                        return True
                    continue
                if "sgt" in pred or "ugt" in pred:
                    return True  # argmax
        return False  # default: argmin

    def _lower_reduce_multi_value(self, ssa: SSAValue):
        """Multi-value reduce: argmax/argmin (2-value) or Welford (3-value).

        For 2 inputs: argmax/argmin (value + index) via SIMD shuffle + shared memory.
        For 3 inputs: Welford online variance (mean + m2 + weight) via SIMD shuffle + shared memory.
        """
        # Dispatch Welford (3-value) vs argmax/argmin (2-value)
        if len(ssa.operand_ids) >= 3 and ssa.result_ids and len(ssa.result_ids) >= 3:
            self._lower_reduce_welford(ssa)
            return

        # Check for 2D argmin/argmax
        if len(ssa.operand_ids) >= 2:
            input_shape = self.env_shapes.get(ssa.operand_ids[0])
            if not input_shape:
                input_shape = _extract_shape(
                    self._find_op_type_str(ssa.operand_ids[0]))
            if input_shape and len(input_shape) == 2 and self._is_2d:
                axis = ssa.attrs.get("axis", 0)
                # Skip 2D dispatch when first dim is 1 and axis is 1
                # (really a 1D reduction, same logic as _lower_reduce)
                if not (input_shape[0] == 1 and axis == 1):
                    self._lower_reduce_2d_argminmax(ssa, axis, input_shape)
                    return

        self._lower_reduce_argminmax(ssa)

    def _lower_reduce_welford(self, ssa: SSAValue):
        """Welford online variance reduction: (mean, m2, weight) via SIMD shuffle + shared memory."""
        mean_var = self._lookup(ssa.operand_ids[0])
        m2_var = self._lookup(ssa.operand_ids[1])
        weight_var = self._lookup(ssa.operand_ids[2])

        n_simd_groups = (self.kb.block_size + 31) // 32

        # Shared memory for 3 values
        sh_mean = f"shared_{self._shared_counter}"; self._shared_counter += 1
        sh_m2 = f"shared_{self._shared_counter}"; self._shared_counter += 1
        sh_w = f"shared_{self._shared_counter}"; self._shared_counter += 1
        self.kb.declare_threadgroup_array(sh_mean, dtype="fp32", size=n_simd_groups)
        self.kb.declare_threadgroup_array(sh_m2, dtype="fp32", size=n_simd_groups)
        self.kb.declare_threadgroup_array(sh_w, dtype="fp32", size=n_simd_groups)

        wm = self._next_var("wm")   # working mean
        wv = self._next_var("wv")    # working m2
        ww = self._next_var("ww")    # working weight
        rm = self._next_var("rm")    # result mean
        rv = self._next_var("rv")    # result m2
        rw = self._next_var("rw")    # result weight

        self.kb.raw_line(f"    // Welford reduce")
        self.kb.raw_line(f"    float {wm} = {mean_var};")
        self.kb.raw_line(f"    float {wv} = {m2_var};")
        self.kb.raw_line(f"    float {ww} = {weight_var};")

        # SIMD-level tree reduction
        self.kb.raw_line(f"    for (ushort _d = 16; _d >= 1; _d >>= 1) {{")
        self.kb.raw_line(f"        float _om = simd_shuffle_down({wm}, _d);")
        self.kb.raw_line(f"        float _ov = simd_shuffle_down({wv}, _d);")
        self.kb.raw_line(f"        float _ow = simd_shuffle_down({ww}, _d);")
        self.kb.raw_line(f"        float _delta = _om - {wm};")
        self.kb.raw_line(f"        float _nw = {ww} + _ow;")
        self.kb.raw_line(f"        float _ratio = (_nw == 0.0f) ? 0.0f : _ow / _nw;")
        self.kb.raw_line(f"        {wm} = {wm} + _delta * _ratio;")
        self.kb.raw_line(f"        {wv} = {wv} + _ov + _delta * _delta * {ww} * _ratio;")
        self.kb.raw_line(f"        {ww} = _nw;")
        self.kb.raw_line(f"    }}")

        # Write lane 0 of each SIMD group to shared
        self.kb.raw_line(f"    if (lid % 32 == 0) {{")
        self.kb.raw_line(f"        {sh_mean}[lid / 32] = {wm};")
        self.kb.raw_line(f"        {sh_m2}[lid / 32] = {wv};")
        self.kb.raw_line(f"        {sh_w}[lid / 32] = {ww};")
        self.kb.raw_line(f"    }}")
        self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Cross-SIMD reduction
        if n_simd_groups > 1:
            self.kb.raw_line(f"    if (lid == 0) {{")
            self.kb.raw_line(f"        {wm} = {sh_mean}[0];")
            self.kb.raw_line(f"        {wv} = {sh_m2}[0];")
            self.kb.raw_line(f"        {ww} = {sh_w}[0];")
            self.kb.raw_line(f"        for (uint _s = 1; _s < {n_simd_groups}u; _s++) {{")
            self.kb.raw_line(f"            float _om = {sh_mean}[_s];")
            self.kb.raw_line(f"            float _ov = {sh_m2}[_s];")
            self.kb.raw_line(f"            float _ow = {sh_w}[_s];")
            self.kb.raw_line(f"            float _delta = _om - {wm};")
            self.kb.raw_line(f"            float _nw = {ww} + _ow;")
            self.kb.raw_line(f"            float _ratio = (_nw == 0.0f) ? 0.0f : _ow / _nw;")
            self.kb.raw_line(f"            {wm} = {wm} + _delta * _ratio;")
            self.kb.raw_line(f"            {wv} = {wv} + _ov + _delta * _delta * {ww} * _ratio;")
            self.kb.raw_line(f"            {ww} = _nw;")
            self.kb.raw_line(f"        }}")
            self.kb.raw_line(f"        {sh_mean}[0] = {wm};")
            self.kb.raw_line(f"        {sh_m2}[0] = {wv};")
            self.kb.raw_line(f"        {sh_w}[0] = {ww};")
            self.kb.raw_line(f"    }}")
            self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # All threads read the final result
        self.kb.raw_line(f"    float {rm} = {sh_mean}[0];")
        self.kb.raw_line(f"    float {rv} = {sh_m2}[0];")
        self.kb.raw_line(f"    float {rw} = {sh_w}[0];")

        # Store all 3 results in env
        self.env[ssa.id] = rm
        self.env_types[ssa.id] = "fp32"
        if ssa.result_ids and len(ssa.result_ids) >= 3:
            self.env[ssa.result_ids[0]] = rm
            self.env_types[ssa.result_ids[0]] = "fp32"
            self.env[ssa.result_ids[1]] = rv
            self.env_types[ssa.result_ids[1]] = "fp32"
            self.env[ssa.result_ids[2]] = rw
            self.env_types[ssa.result_ids[2]] = "fp32"

    def _lower_reduce_argminmax(self, ssa: SSAValue):
        """Argmax/argmin: value + index via SIMD shuffle + shared memory."""
        val_var = self._lookup(ssa.operand_ids[0])
        idx_var = self._lookup(ssa.operand_ids[1])

        # Determine value type
        val_dtype = self.env_types.get(ssa.operand_ids[0], "fp32")
        is_int = not (val_dtype.startswith("fp") or val_dtype.startswith("bf"))
        msl_val_type = "int" if is_int else "float"
        val_shared_dtype = "i32" if is_int else "fp32"

        # Detect argmax vs argmin from body ops
        is_max = self._detect_reduce_direction(ssa)
        cmp_op = ">" if is_max else "<"

        # Allocate shared memory for values and indices
        n_simd_groups = (self.kb.block_size + 31) // 32
        shared_val = f"shared_{self._shared_counter}"
        self._shared_counter += 1
        shared_idx = f"shared_{self._shared_counter}"
        self._shared_counter += 1
        self.kb.declare_threadgroup_array(shared_val, dtype=val_shared_dtype, size=n_simd_groups)
        self.kb.declare_threadgroup_array(shared_idx, dtype="i32", size=n_simd_groups)

        # Unique variable names
        mv = self._next_var("mv")
        mi = self._next_var("mi")
        result_val = self._next_var("rval")
        result_idx = self._next_var("ridx")

        tag = "max" if is_max else "min"
        self.kb.raw_line(f"    // Multi-value reduce: arg{tag}")
        self.kb.raw_line(f"    {msl_val_type} {mv} = {val_var};")
        self.kb.raw_line(f"    int {mi} = {idx_var};")

        # SIMD-level tree reduction using simd_shuffle_down
        self.kb.raw_line(f"    for (ushort _d = 16; _d >= 1; _d >>= 1) {{")
        self.kb.raw_line(f"        {msl_val_type} _ov = simd_shuffle_down({mv}, _d);")
        self.kb.raw_line(f"        int _oi = simd_shuffle_down({mi}, _d);")
        self.kb.raw_line(f"        bool _take = (_ov {cmp_op} {mv}) || (_ov == {mv} && _oi < {mi});")
        self.kb.raw_line(f"        {mv} = _take ? _ov : {mv};")
        self.kb.raw_line(f"        {mi} = _take ? _oi : {mi};")
        self.kb.raw_line(f"    }}")

        # Write lane 0 of each SIMD group to shared memory
        self.kb.raw_line(f"    if (lid % 32 == 0) {{")
        self.kb.raw_line(f"        {shared_val}[lid / 32] = {mv};")
        self.kb.raw_line(f"        {shared_idx}[lid / 32] = {mi};")
        self.kb.raw_line(f"    }}")
        self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Cross-SIMD reduction (thread 0 reduces across SIMD groups)
        if n_simd_groups > 1:
            self.kb.raw_line(f"    if (lid == 0) {{")
            self.kb.raw_line(f"        {mv} = {shared_val}[0];")
            self.kb.raw_line(f"        {mi} = {shared_idx}[0];")
            self.kb.raw_line(f"        for (uint _s = 1; _s < {n_simd_groups}u; _s++) {{")
            self.kb.raw_line(f"            {msl_val_type} _ov = {shared_val}[_s];")
            self.kb.raw_line(f"            int _oi = {shared_idx}[_s];")
            self.kb.raw_line(f"            bool _take = (_ov {cmp_op} {mv}) || (_ov == {mv} && _oi < {mi});")
            self.kb.raw_line(f"            {mv} = _take ? _ov : {mv};")
            self.kb.raw_line(f"            {mi} = _take ? _oi : {mi};")
            self.kb.raw_line(f"        }}")
            self.kb.raw_line(f"        {shared_val}[0] = {mv};")
            self.kb.raw_line(f"        {shared_idx}[0] = {mi};")
            self.kb.raw_line(f"    }}")
            self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # All threads read the final result
        self.kb.raw_line(f"    {msl_val_type} {result_val} = {shared_val}[0];")
        self.kb.raw_line(f"    int {result_idx} = {shared_idx}[0];")

        # Store both results in env
        self.env[ssa.id] = result_val
        self.env_types[ssa.id] = val_shared_dtype
        if ssa.result_ids and len(ssa.result_ids) >= 2:
            self.env[ssa.result_ids[0]] = result_val
            self.env_types[ssa.result_ids[0]] = val_shared_dtype
            self.env[ssa.result_ids[1]] = result_idx
            self.env_types[ssa.result_ids[1]] = "i32"

    def _lower_reduce_2d_argminmax(self, ssa, axis, input_shape):
        """Lower 2D argmin/argmax: find min/max value and index along axis.

        For axis=1 on (M, N): each row finds min/max among N values → (M,) values + indices.
        For axis=0 on (M, N): each column finds min/max among M values → (N,) values + indices.
        """
        M, N = input_shape[0], input_shape[1]
        total = M * N

        val_var = self._lookup(ssa.operand_ids[0])
        idx_var = self._lookup(ssa.operand_ids[1])

        val_dtype = self.env_types.get(ssa.operand_ids[0], "fp32")
        is_int = not (val_dtype.startswith("fp") or val_dtype.startswith("bf"))
        msl_val_type = "int" if is_int else "float"
        val_shared_dtype = "i32" if is_int else "fp32"

        is_max = self._detect_reduce_direction(ssa)
        cmp_op = ">" if is_max else "<"
        identity = "(-INFINITY)" if is_max and not is_int else "INFINITY"
        if is_int:
            identity = "INT_MIN" if is_max else "INT_MAX"

        # Shared memory for values and indices
        shared_val = f"shared_{self._shared_counter}"
        self._shared_counter += 1
        shared_idx = f"shared_{self._shared_counter}"
        self._shared_counter += 1
        self.kb.declare_threadgroup_array(shared_val, dtype=val_shared_dtype, size=total)
        self.kb.declare_threadgroup_array(shared_idx, dtype="i32", size=total)

        # Stage values and indices
        self.kb.raw_line(f"    if (lid < {total}u) {{")
        self.kb.raw_line(f"        {shared_val}[lid] = {val_var};")
        self.kb.raw_line(f"        {shared_idx}[lid] = {idx_var};")
        self.kb.raw_line(f"    }}")
        self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Result arrays
        if axis == 1:
            result_size = M
        else:
            result_size = N
        result_val_shared = f"shared_{self._shared_counter}"
        self._shared_counter += 1
        result_idx_shared = f"shared_{self._shared_counter}"
        self._shared_counter += 1
        self.kb.declare_threadgroup_array(result_val_shared, dtype=val_shared_dtype, size=result_size)
        self.kb.declare_threadgroup_array(result_idx_shared, dtype="i32", size=result_size)

        result_val_var = self._next_var("rval")
        result_idx_var = self._next_var("ridx")

        self.kb.raw_line(f"    {msl_val_type} {result_val_var} = {identity};")
        self.kb.raw_line(f"    int {result_idx_var} = 0;")

        if axis == 1:
            # Each row: find argmin/max among N columns
            self.kb.raw_line(f"    if (lid < {M}u) {{")
            self.kb.raw_line(f"        {msl_val_type} best_v = {identity};")
            self.kb.raw_line(f"        int best_i = 0;")
            self.kb.raw_line(f"        for (uint j = 0; j < {N}u; j++) {{")
            self.kb.raw_line(f"            {msl_val_type} v = {shared_val}[lid * {N}u + j];")
            self.kb.raw_line(f"            int idx = {shared_idx}[lid * {N}u + j];")
            self.kb.raw_line(f"            if (v {cmp_op} best_v || (v == best_v && idx < best_i)) {{")
            self.kb.raw_line(f"                best_v = v; best_i = idx;")
            self.kb.raw_line(f"            }}")
            self.kb.raw_line(f"        }}")
            self.kb.raw_line(f"        {result_val_shared}[lid] = best_v;")
            self.kb.raw_line(f"        {result_idx_shared}[lid] = best_i;")
            self.kb.raw_line(f"    }}")
        else:
            # Each column: find argmin/max among M rows
            self.kb.raw_line(f"    if (lid < {N}u) {{")
            self.kb.raw_line(f"        {msl_val_type} best_v = {identity};")
            self.kb.raw_line(f"        int best_i = 0;")
            self.kb.raw_line(f"        for (uint i = 0; i < {M}u; i++) {{")
            self.kb.raw_line(f"            {msl_val_type} v = {shared_val}[i * {N}u + lid];")
            self.kb.raw_line(f"            int idx = {shared_idx}[i * {N}u + lid];")
            self.kb.raw_line(f"            if (v {cmp_op} best_v || (v == best_v && idx < best_i)) {{")
            self.kb.raw_line(f"                best_v = v; best_i = idx;")
            self.kb.raw_line(f"            }}")
            self.kb.raw_line(f"        }}")
            self.kb.raw_line(f"        {result_val_shared}[lid] = best_v;")
            self.kb.raw_line(f"        {result_idx_shared}[lid] = best_i;")
            self.kb.raw_line(f"    }}")

        self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Broadcast result to all threads.
        # Row-major: threads [0..N-1] in row 0, [N..2N-1] in row 1.
        if axis == 1:
            self.kb.raw_line(f"    {result_val_var} = {result_val_shared}[lid / {N}u];")
            self.kb.raw_line(f"    {result_idx_var} = {result_idx_shared}[lid / {N}u];")
        else:
            self.kb.raw_line(f"    {result_val_var} = {result_val_shared}[lid % {N}u];")
            self.kb.raw_line(f"    {result_idx_var} = {result_idx_shared}[lid % {N}u];")

        # Store results
        self.env[ssa.id] = result_val_var
        self.env_types[ssa.id] = val_shared_dtype
        if ssa.result_ids and len(ssa.result_ids) >= 2:
            self.env[ssa.result_ids[0]] = result_val_var
            self.env_types[ssa.result_ids[0]] = val_shared_dtype
            self.env[ssa.result_ids[1]] = result_idx_var
            self.env_types[ssa.result_ids[1]] = "i32"
        # Set output shape
        out_shape = (M,) if axis == 1 else (N,)
        self.env_shapes[ssa.id] = out_shape
        if ssa.result_ids:
            for rid in ssa.result_ids:
                self.env_shapes[rid] = out_shape

    def _lower_reduce_2d(self, ssa, input_var, axis, combine_op,
                         msl_type, shared_dtype, input_shape):
        """Lower a 2D axis-specific reduction.

        For axis=1 on (M, N): each of M rows sums its N values.
        For axis=0 on (M, N): each of N columns sums its M values.

        Uses shared memory to collect all values, then each result-thread
        performs a sequential reduction over its assigned group.
        """
        M, N = input_shape[0], input_shape[1]
        total = M * N

        # Allocate shared memory for the full 2D tensor
        shared_name = f"shared_{self._shared_counter}"
        self._shared_counter += 1
        self.kb.declare_threadgroup_array(shared_name, dtype=shared_dtype, size=total)

        # Identity value for the combine op
        if combine_op == "sum":
            identity = "0.0f" if msl_type == "float" else "0"
        elif combine_op == "max":
            identity = "(-INFINITY)" if msl_type == "float" else "INT_MIN"
        elif combine_op == "min":
            identity = "INFINITY" if msl_type == "float" else "INT_MAX"
        elif combine_op == "xor":
            identity = "0"
        else:
            identity = "0.0f" if msl_type == "float" else "0"

        # Combine expression
        if combine_op == "sum":
            combine_expr = "acc + val"
        elif combine_op == "max":
            combine_expr = "max(acc, val)" if msl_type == "int" else "fmax(acc, val)"
        elif combine_op == "min":
            combine_expr = "min(acc, val)" if msl_type == "int" else "fmin(acc, val)"
        elif combine_op == "xor":
            combine_expr = "acc ^ val"
        else:
            combine_expr = "acc + val"

        result_var = self._next_var("reduced")

        # Store all values to shared memory
        self.kb.raw_line(f"    if (lid < {total}u) {shared_name}[lid] = {input_var};")
        self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Use a second shared array to broadcast results to all threads
        result_shared = f"shared_{self._shared_counter}"
        self._shared_counter += 1

        if axis == 1:
            # Reduce along columns: each row reduces N values → M results
            result_size = M
            self.kb.declare_threadgroup_array(result_shared, dtype=shared_dtype, size=M)
            self.kb.raw_line(f"    {msl_type} {result_var} = {identity};")
            self.kb.raw_line(f"    if (lid < {M}u) {{")
            self.kb.raw_line(f"        {msl_type} acc = {identity};")
            self.kb.raw_line(f"        for (uint j = 0; j < {N}u; j++) {{")
            self.kb.raw_line(f"            {msl_type} val = {shared_name}[lid * {N}u + j];")
            self.kb.raw_line(f"            acc = {combine_expr};")
            self.kb.raw_line(f"        }}")
            self.kb.raw_line(f"        {result_shared}[lid] = acc;")
            self.kb.raw_line(f"    }}")
            self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")
            # All threads read their row's result.
            # Row-major: threads [0..N-1] in row 0, [N..2N-1] in row 1.
            self.kb.raw_line(f"    {result_var} = {result_shared}[lid / {N}u];")
        else:
            # Reduce along rows: each column reduces M values → N results
            result_size = N
            self.kb.declare_threadgroup_array(result_shared, dtype=shared_dtype, size=N)
            self.kb.raw_line(f"    {msl_type} {result_var} = {identity};")
            self.kb.raw_line(f"    if (lid < {N}u) {{")
            self.kb.raw_line(f"        {msl_type} acc = {identity};")
            self.kb.raw_line(f"        for (uint i = 0; i < {M}u; i++) {{")
            self.kb.raw_line(f"            {msl_type} val = {shared_name}[i * {N}u + lid];")
            self.kb.raw_line(f"            acc = {combine_expr};")
            self.kb.raw_line(f"        }}")
            self.kb.raw_line(f"        {result_shared}[lid] = acc;")
            self.kb.raw_line(f"    }}")
            self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")
            # All threads read their column's result
            self.kb.raw_line(f"    {result_var} = {result_shared}[lid % {N}u];")

        self.env[ssa.id] = result_var
        self.env_types[ssa.id] = shared_dtype
        # Result shape is the non-reduced dimension
        if axis == 1:
            self.env_shapes[ssa.id] = (M,)
        else:
            self.env_shapes[ssa.id] = (N,)

    def _lower_reduce_3d(self, ssa, input_var, axis, combine_op,
                         msl_type, shared_dtype, input_shape):
        """Lower a 3D axis-specific reduction.

        For (M, N, K) tensor reducing along axis:
          axis=0: result (N, K), loop over M
          axis=1: result (M, K), loop over N
          axis=2: result (M, N), loop over K

        Uses shared memory staging with loop-based loading for cases where
        total elements > block_size. Reads directly from the source X pointer.
        """
        M, N, K = input_shape[0], input_shape[1], input_shape[2]
        total = M * N * K
        block_size = self.kb.block_size

        # Find the source data pointer (first pointer arg = X)
        x_ptr_name = None
        for arg in self.graph.args:
            if arg.is_ptr:
                x_ptr_name = arg.name
                break
        if x_ptr_name is None:
            # Fallback — shouldn't happen
            return

        # Identity and combine expression
        if combine_op == "sum":
            identity = "0.0f" if msl_type == "float" else "0"
            combine_expr = "acc + val"
        elif combine_op == "max":
            identity = "(-INFINITY)" if msl_type == "float" else "INT_MIN"
            combine_expr = "fmax(acc, val)" if msl_type == "float" else "max(acc, val)"
        elif combine_op == "min":
            identity = "INFINITY" if msl_type == "float" else "INT_MAX"
            combine_expr = "fmin(acc, val)" if msl_type == "float" else "min(acc, val)"
        elif combine_op == "xor":
            identity = "0"
            combine_expr = "acc ^ val"
        else:
            identity = "0.0f" if msl_type == "float" else "0"
            combine_expr = "acc + val"

        # Allocate shared memory for the full 3D tensor
        shared_name = f"shared_{self._shared_counter}"
        self._shared_counter += 1
        self.kb.declare_threadgroup_array(shared_name, dtype=shared_dtype, size=total)

        # Stage values to shared memory from the already-loaded input_var
        # (not the raw pointer, which ignores strided/computed addresses)
        if total <= block_size:
            self.kb.raw_line(f"    if (lid < {total}u) {shared_name}[lid] = {input_var};")
        else:
            # Wrapping loop for large tensors — read from source pointer
            self.kb.raw_line(f"    for (uint _e = lid; _e < {total}u; _e += {block_size}u) {{")
            self.kb.raw_line(f"        {shared_name}[_e] = ({msl_type}){x_ptr_name}[_e];")
            self.kb.raw_line(f"    }}")
        self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Compute result dimensions
        if axis == 0:
            result_dims = (N, K)
            result_total = N * K
            axis_size = M
        elif axis == 1:
            result_dims = (M, K)
            result_total = M * K
            axis_size = N
        else:  # axis == 2
            result_dims = (M, N)
            result_total = M * N
            axis_size = K

        # Allocate result shared memory
        result_shared = f"shared_{self._shared_counter}"
        self._shared_counter += 1
        self.kb.declare_threadgroup_array(result_shared, dtype=shared_dtype, size=result_total)

        # Reduction loop: each result thread reduces along the axis
        self.kb.raw_line(f"    for (uint _r = lid; _r < {result_total}u; _r += {block_size}u) {{")
        self.kb.raw_line(f"        {msl_type} acc = {identity};")

        # Compute result indices and shared memory indexing based on axis
        if axis == 0:
            # result (j, k) at _r: j = _r/K, k = _r%K. Loop over i.
            self.kb.raw_line(f"        uint _j = _r / {K}u;")
            self.kb.raw_line(f"        uint _k = _r % {K}u;")
            self.kb.raw_line(f"        for (uint _a = 0; _a < {axis_size}u; _a++) {{")
            self.kb.raw_line(f"            {msl_type} val = {shared_name}[_a * {N * K}u + _j * {K}u + _k];")
        elif axis == 1:
            # result (i, k) at _r: i = _r/K, k = _r%K. Loop over j.
            self.kb.raw_line(f"        uint _i = _r / {K}u;")
            self.kb.raw_line(f"        uint _k = _r % {K}u;")
            self.kb.raw_line(f"        for (uint _a = 0; _a < {axis_size}u; _a++) {{")
            self.kb.raw_line(f"            {msl_type} val = {shared_name}[_i * {N * K}u + _a * {K}u + _k];")
        else:  # axis == 2
            # result (i, j) at _r: i = _r/N, j = _r%N. Loop over k.
            self.kb.raw_line(f"        uint _i = _r / {N}u;")
            self.kb.raw_line(f"        uint _j = _r % {N}u;")
            self.kb.raw_line(f"        for (uint _a = 0; _a < {axis_size}u; _a++) {{")
            self.kb.raw_line(f"            {msl_type} val = {shared_name}[_i * {N * K}u + _j * {K}u + _a];")

        self.kb.raw_line(f"            acc = {combine_expr};")
        self.kb.raw_line(f"        }}")
        self.kb.raw_line(f"        {result_shared}[_r] = acc;")
        self.kb.raw_line(f"    }}")
        self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # All threads read their result
        result_var = self._next_var("reduced")
        self.kb.raw_line(f"    {msl_type} {result_var} = {result_shared}[lid % {result_total}u];")

        self.env[ssa.id] = result_var
        self.env_types[ssa.id] = shared_dtype
        self.env_shapes[ssa.id] = result_dims

    # -- Prefix scan (tt.scan) --

    def _lower_scan(self, ssa: SSAValue):
        """tt.scan → prefix scan via shared memory.

        For 2D tensors: scan along the specified axis.
            axis=1 on (M, N): each row gets an independent prefix scan along N
            axis=0 on (M, N): each column gets an independent prefix scan along M
        Supports forward and reverse scans, single and multi-value combines.
        """
        axis = ssa.attrs.get("axis", 0)
        reverse = ssa.attrs.get("reverse", False)
        n_values = len(ssa.operand_ids)

        if not ssa.operand_ids:
            return

        # Get input shape from type string
        is_1d = False
        input_shape = _extract_shape(ssa.type_str)
        if not input_shape or len(input_shape) < 2:
            input_shape = _extract_shape(
                self._find_op_type_str(ssa.operand_ids[0]))
        if not input_shape or len(input_shape) < 2:
            # 1D tensor: treat as (1, size) and scan along axis=1
            sz = input_shape[0] if input_shape else self.effective_block_size
            input_shape = (1, sz)
            axis = 1  # 1D scan always scans along the data dimension
            is_1d = True

        M, N = input_shape[0], input_shape[1]
        total = M * N

        # Determine element type and MSL type
        input_dtype = self.env_types.get(ssa.operand_ids[0], "fp32")
        is_int = not (input_dtype.startswith("fp") or input_dtype.startswith("bf"))
        if input_dtype == "bf16":
            msl_type = "float"
            shared_dtype = "fp32"
        elif is_int:
            msl_type = "int"
            shared_dtype = "i32"
        else:
            msl_type = "float"
            shared_dtype = "fp32"

        # Allocate shared memory for each input value
        shared_names = []
        for i in range(n_values):
            shared_name = f"scan_shared_{self._shared_counter}"
            self._shared_counter += 1
            self.kb.declare_threadgroup_array(shared_name, dtype=shared_dtype,
                                              size=total)
            shared_names.append(shared_name)

        # Write input values to shared memory
        for i, operand_id in enumerate(ssa.operand_ids):
            input_var = self._lookup(operand_id)
            cast = f"({msl_type})" if input_dtype == "bf16" else ""
            self.kb.raw_line(
                f"    if (lid < {total}u) {shared_names[i]}[lid] = {cast}{input_var};")
        self.kb.raw_line(
            f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Compute position and base expressions
        if axis == 1:
            scan_size = N
            pos_expr = f"(lid % {N}u)"
            base_expr = f"((lid / {N}u) * {N}u)"
        else:
            scan_size = M
            pos_expr = f"(lid / {N}u)"
            # For axis=0, elements in same column are at stride N

        # Initialize accumulators with first element of scan group
        acc_vars = []
        for i in range(n_values):
            acc_var = self._next_var("scan_acc")
            acc_vars.append(acc_var)
            if axis == 1:
                if not reverse:
                    init_idx = f"{base_expr}"
                else:
                    init_idx = f"({base_expr} + {N - 1}u)"
            else:
                if not reverse:
                    init_idx = f"(lid % {N}u)"
                else:
                    init_idx = f"({(M - 1)}u * {N}u + (lid % {N}u))"
            self.kb.raw_line(
                f"    {msl_type} {acc_var} = ({msl_type}){shared_names[i]}[{init_idx}];")

        # Emit scan loop
        if not reverse:
            self.kb.raw_line(
                f"    for (uint scan_j = 1u; scan_j <= {pos_expr}; scan_j++) {{")
        else:
            self.kb.raw_line(
                f"    for (uint scan_j = 1u; scan_j <= ({scan_size - 1}u - {pos_expr}); scan_j++) {{")

        # Load current elements (rhs) from shared memory
        rhs_vars = []
        for i in range(n_values):
            rhs_var = self._next_var("scan_rhs")
            rhs_vars.append(rhs_var)
            if axis == 1:
                if not reverse:
                    idx_expr = f"{base_expr} + scan_j"
                else:
                    idx_expr = f"{base_expr} + ({N - 1}u - scan_j)"
            else:
                if not reverse:
                    idx_expr = f"scan_j * {N}u + (lid % {N}u)"
                else:
                    idx_expr = f"({M - 1}u - scan_j) * {N}u + (lid % {N}u)"
            self.kb.raw_line(
                f"        {msl_type} {rhs_var} = ({msl_type}){shared_names[i]}[{idx_expr}];")

        # Map block args to accumulator (lhs) and current element (rhs) vars
        block_arg_ids = ssa.attrs.get("block_arg_ids", [])
        if block_arg_ids and len(block_arg_ids) >= 2 * n_values:
            for i in range(n_values):
                self.env[block_arg_ids[i]] = acc_vars[i]
                self.env_types[block_arg_ids[i]] = shared_dtype
                self.env[block_arg_ids[n_values + i]] = rhs_vars[i]
                self.env_types[block_arg_ids[n_values + i]] = shared_dtype

        # Lower body ops (combine function) and find scan.return operands
        scan_return_ids = []
        if ssa.region_ops:
            for body_op in ssa.region_ops:
                if body_op.op == "tt.scan.return":
                    scan_return_ids = body_op.operand_ids
                else:
                    self._lower_op(body_op)

        # Update accumulators from combine results
        for i in range(n_values):
            if i < len(scan_return_ids):
                new_val = self._lookup(scan_return_ids[i])
                self.kb.raw_line(f"        {acc_vars[i]} = {new_val};")

        self.kb.raw_line(f"    }}")

        # For 1D scans, subsequent reshape+broadcast needs all threads to access
        # the scan results. Write back to shared memory and read with modular index.
        if is_1d and total < self.effective_block_size:
            self.kb.raw_line(
                f"    if (lid < {total}u) {shared_names[0]}[lid] = {acc_vars[0]};")
            self.kb.raw_line(
                f"    threadgroup_barrier(mem_flags::mem_threadgroup);")
            result_var = self._next_var("scan_result")
            self.kb.raw_line(
                f"    {msl_type} {result_var} = ({msl_type}){shared_names[0]}[lid % {total}u];")
            for i in range(1, n_values):
                self.kb.raw_line(
                    f"    if (lid < {total}u) {shared_names[i]}[lid] = {acc_vars[i]};")
            if n_values > 1:
                self.kb.raw_line(
                    f"    threadgroup_barrier(mem_flags::mem_threadgroup);")
            acc_vars_out = [result_var]
            for i in range(1, n_values):
                rv = self._next_var("scan_result")
                self.kb.raw_line(
                    f"    {msl_type} {rv} = ({msl_type}){shared_names[i]}[lid % {total}u];")
                acc_vars_out.append(rv)
        else:
            acc_vars_out = acc_vars

        # Map scan results to output variables
        if ssa.result_ids and len(ssa.result_ids) >= n_values:
            for i in range(n_values):
                self.env[ssa.result_ids[i]] = acc_vars_out[i]
                self.env_types[ssa.result_ids[i]] = shared_dtype
                self.env_shapes[ssa.result_ids[i]] = input_shape
        else:
            self.env[ssa.id] = acc_vars_out[0]
            self.env_types[ssa.id] = shared_dtype
            self.env_shapes[ssa.id] = input_shape

    # -- Matrix multiply (tt.dot) --

    def _lower_dot(self, ssa: SSAValue):
        """tt.dot → simdgroup_matrix MMA.

        This should only be reached if _requires_matmul_template() somehow missed it.
        The normal path uses _lower_dot_via_prebuilt_template() instead.
        """
        if len(ssa.operand_ids) >= 3:
            acc = self._lookup(ssa.operand_ids[2])
            self.env[ssa.id] = acc
        else:
            var_name = self._next_var("dot")
            self.kb.raw_line(f"    float {var_name} = 0.0f;")
            self.env[ssa.id] = var_name

    # -- SCF (structured control flow) --

    def _lower_scf_for(self, ssa: SSAValue):
        """scf.for → MSL for loop with iter_args.

        scf.for has operands: [start, end, step, init_0, init_1, ...]
        Results: [result_0, result_1, ...] (same count as iter_args)
        Body block args: [induction_var, iter_arg_0, iter_arg_1, ...]
        """
        if len(ssa.operand_ids) < 3:
            return

        start_var = self._lookup(ssa.operand_ids[0])
        end_var = self._lookup(ssa.operand_ids[1])
        step_var = self._lookup(ssa.operand_ids[2])

        # iter_args initial values: operands[3:]
        init_ids = ssa.operand_ids[3:]
        n_iter_args = len(init_ids)

        # Infer iter_arg types from init values and scf.for result types
        iter_vars = []
        iter_dtypes = []
        # The scf.for result type tells us the true type of iter_args
        result_elem = ssa.elem_type or "f32"  # First result's type
        for i, init_id in enumerate(init_ids):
            var_name = self._next_var("iter")
            init_val = self._lookup(init_id)
            # Prefer result type, fall back to init value type
            init_type = self.env_types.get(init_id, "fp32")
            # Use result type if it's more specific (e.g., i64 vs i32)
            if result_elem in ("i64",) and init_type in ("i32", "fp32"):
                init_type = result_elem
            if init_type.startswith("f") or init_type.startswith("bf"):
                msl_type = "float"
            elif init_type in ("i64",):
                msl_type = "long"
            elif init_type.startswith("u"):
                msl_type = "uint"
            else:
                msl_type = "int"
            self.kb.raw_line(f"    {msl_type} {var_name} = {init_val};")
            iter_vars.append(var_name)
            iter_dtypes.append(init_type)

        # Emit for loop — use long for i64.
        # scf.for semantics: always `iv < ub` (Triton normalizes negative steps).
        start_type = self.env_types.get(ssa.operand_ids[0], "i32")
        is_i64 = start_type == "i64" or "i64" in (ssa.type_str or "")
        loop_type = "long" if is_i64 else "int"
        loop_var = self._next_var("k")

        self.kb.raw_line(
            f"    for ({loop_type} {loop_var} = {start_var}; "
            f"{loop_var} < {end_var}; {loop_var} += {step_var}) {{"
        )

        # Map block args to MSL variables
        block_arg_ids = ssa.attrs.get("block_arg_ids", [])
        if block_arg_ids:
            # First block arg is induction variable
            self.env[block_arg_ids[0]] = loop_var
            self.env_types[block_arg_ids[0]] = start_type
            self.env_shapes[block_arg_ids[0]] = ()  # induction var is scalar
            # Remaining block args are iter_args
            for i, var in enumerate(iter_vars):
                if i + 1 < len(block_arg_ids):
                    ba_id = block_arg_ids[i + 1]
                    self.env[ba_id] = var
                    self.env_types[ba_id] = iter_dtypes[i] if i < len(iter_dtypes) else "fp32"
                    # Propagate shape from init value to block arg
                    init_id = init_ids[i] if i < len(init_ids) else None
                    if init_id is not None and init_id in self.env_shapes:
                        self.env_shapes[ba_id] = self.env_shapes[init_id]

        # Process body ops
        if ssa.region_ops:
            for body_op in ssa.region_ops:
                if body_op.op == "scf.yield":
                    # Update iter_arg variables from yield operands
                    for i, yield_id in enumerate(body_op.operand_ids):
                        if i < len(iter_vars):
                            yield_val = self._lookup(yield_id)
                            self.kb.raw_line(
                                f"        {iter_vars[i]} = {yield_val};"
                            )
                else:
                    self._lower_op(body_op)

        self.kb.raw_line("    }")

        # Map scf.for results to iter_arg variables using proper result IDs
        if ssa.result_ids:
            for i, var in enumerate(iter_vars):
                if i < len(ssa.result_ids):
                    rid = ssa.result_ids[i]
                    self.env[rid] = var
                    self.env_types[rid] = iter_dtypes[i] if i < len(iter_dtypes) else "fp32"
                    # Propagate shape from init value to result
                    if i < len(init_ids) and init_ids[i] in self.env_shapes:
                        self.env_shapes[rid] = self.env_shapes[init_ids[i]]
        elif n_iter_args == 1 and iter_vars:
            self.env[ssa.id] = iter_vars[0]
            self.env_types[ssa.id] = iter_dtypes[0] if iter_dtypes else "fp32"
            if init_ids and init_ids[0] in self.env_shapes:
                self.env_shapes[ssa.id] = self.env_shapes[init_ids[0]]
        elif iter_vars:
            # Fallback: single result maps to first iter_var
            self.env[ssa.id] = iter_vars[0]
            self.env_types[ssa.id] = iter_dtypes[0] if iter_dtypes else "fp32"
            if init_ids and init_ids[0] in self.env_shapes:
                self.env_shapes[ssa.id] = self.env_shapes[init_ids[0]]

    def _lower_scf_if(self, ssa: SSAValue):
        """scf.if → MSL if/else block with optional results."""
        if not ssa.operand_ids:
            return

        cond = self._lookup(ssa.operand_ids[0])
        result_ids = ssa.result_ids or ([ssa.id] if ssa.id is not None else [])

        # Check both then and else for yield with operands
        all_body_ops = list(ssa.region_ops or []) + list(ssa.else_ops or [])
        has_results = any(
            body_op.op == "scf.yield" and body_op.operand_ids
            for body_op in all_body_ops
        )

        # For scf.if with results, declare result variables before the if/else
        result_vars = []
        if has_results:
            # Infer result types from yield operands
            yield_types = []
            for body_op in all_body_ops:
                if body_op.op == "scf.yield" and body_op.operand_ids:
                    for yid in body_op.operand_ids:
                        yt = self.env_types.get(yid, "fp32")
                        yield_types.append(yt)
                    break

            for i, rid in enumerate(result_ids):
                var_name = f"ifr_{abs(rid)}_{i}"
                result_vars.append((rid, var_name))
                yt = yield_types[i] if i < len(yield_types) else "fp32"
                if yt.startswith("fp") or yt.startswith("bf") or yt.startswith("f"):
                    msl_type = "float"
                elif yt.startswith("u"):
                    msl_type = "uint"
                else:
                    msl_type = "int"
                self.kb.raw_line(f"    {msl_type} {var_name};")

        self.kb.raw_line(f"    if ({cond}) {{")

        # Lower "then" body
        if ssa.region_ops:
            for body_op in ssa.region_ops:
                if body_op.op == "scf.yield":
                    for i, yield_id in enumerate(body_op.operand_ids):
                        if i < len(result_vars):
                            yield_val = self._lookup(yield_id)
                            rid, var_name = result_vars[i]
                            self.kb.raw_line(f"        {var_name} = {yield_val};")
                else:
                    self._lower_op(body_op)

        # Lower "else" body
        if ssa.else_ops:
            self.kb.raw_line("    } else {")
            for body_op in ssa.else_ops:
                if body_op.op == "scf.yield":
                    for i, yield_id in enumerate(body_op.operand_ids):
                        if i < len(result_vars):
                            yield_val = self._lookup(yield_id)
                            rid, var_name = result_vars[i]
                            self.kb.raw_line(f"        {var_name} = {yield_val};")
                else:
                    self._lower_op(body_op)

        self.kb.raw_line("    }")

        # Map result variables into env with proper types
        for i, (rid, var_name) in enumerate(result_vars):
            self.env[rid] = var_name
            # Propagate type from yield operands
            yt = yield_types[i] if i < len(yield_types) else "fp32"
            self.env_types[rid] = yt

    def _lower_scf_while(self, ssa: SSAValue):
        """scf.while → MSL while(true) { condition-check; body; } loop.

        scf.while has operands: [init_0, init_1, ...]
        Results: [result_0, result_1, ...] (same count as init values)

        Two regions:
          - "before" (region_ops): evaluates condition, terminates with scf.condition
          - "after" (else_ops): loop body, terminates with scf.yield

        The "before" region's scf.condition carries the loop predicate and
        forwarded values to the "after" region's block arguments.
        """
        init_ids = ssa.operand_ids  # Initial values for iter_args
        n_iter_args = len(init_ids)
        result_ids = ssa.result_ids or ([ssa.id] if ssa.id is not None else [])

        # Declare iter_arg variables from init values
        iter_vars = []
        iter_dtypes = []
        for i, init_id in enumerate(init_ids):
            var_name = self._next_var("wh")
            init_val = self._lookup(init_id)
            init_type = self.env_types.get(init_id, "i32")
            if init_type.startswith("f") or init_type.startswith("bf") or init_type.startswith("fp"):
                msl_type = "float"
            elif init_type in ("i64",):
                msl_type = "long"
            elif init_type.startswith("u"):
                msl_type = "uint"
            else:
                msl_type = "int"
            self.kb.raw_line(f"    {msl_type} {var_name} = {init_val};")
            iter_vars.append(var_name)
            iter_dtypes.append(init_type)

        self.kb.raw_line("    for (;;) {")

        # Map "before" region block args to iter_vars
        before_block_args = ssa.attrs.get("block_arg_ids", [])
        for i, var in enumerate(iter_vars):
            if i < len(before_block_args):
                self.env[before_block_args[i]] = var
                self.env_types[before_block_args[i]] = iter_dtypes[i]

        # Lower "before" region (condition evaluation)
        for body_op in (ssa.region_ops or []):
            if body_op.op == "scf.condition":
                # First operand is the condition
                if body_op.operand_ids:
                    cond_var = self._lookup(body_op.operand_ids[0])
                    self.kb.raw_line(f"        if (!({cond_var})) break;")
                # Remaining operands are forwarded values to "after" block args
                after_block_args = ssa.attrs.get("else_block_arg_ids", [])
                for j, fwd_id in enumerate(body_op.operand_ids[1:]):
                    if j < len(after_block_args):
                        fwd_val = self._lookup(fwd_id)
                        self.env[after_block_args[j]] = fwd_val
                        fwd_type = self.env_types.get(fwd_id, "i32")
                        self.env_types[after_block_args[j]] = fwd_type
            else:
                self._lower_op(body_op)

        # If "after" block args weren't mapped by scf.condition forwarding,
        # map them to iter_vars directly (they share the same values)
        after_block_args = ssa.attrs.get("else_block_arg_ids", [])
        for i, var in enumerate(iter_vars):
            if i < len(after_block_args) and after_block_args[i] not in self.env:
                self.env[after_block_args[i]] = var
                self.env_types[after_block_args[i]] = iter_dtypes[i]

        # Lower "after" region (loop body)
        for body_op in (ssa.else_ops or []):
            if body_op.op == "scf.yield":
                # Update iter_arg variables from yield operands
                for j, yield_id in enumerate(body_op.operand_ids):
                    if j < len(iter_vars):
                        yield_val = self._lookup(yield_id)
                        self.kb.raw_line(f"        {iter_vars[j]} = {yield_val};")
            else:
                self._lower_op(body_op)

        self.kb.raw_line("    }")

        # Map scf.while results to iter_arg variables
        if len(result_ids) > 1:
            for i, var in enumerate(iter_vars):
                if i < len(result_ids):
                    self.env[result_ids[i]] = var
                    self.env_types[result_ids[i]] = iter_dtypes[i] if i < len(iter_dtypes) else "i32"
        elif n_iter_args == 1 and iter_vars:
            self.env[ssa.id] = iter_vars[0]
            self.env_types[ssa.id] = iter_dtypes[0] if iter_dtypes else "i32"
        elif iter_vars:
            self.env[ssa.id] = iter_vars[0]
            self.env_types[ssa.id] = iter_dtypes[0] if iter_dtypes else "i32"

    # -- Atomic ops --

    def _lower_atomic_rmw(self, ssa: SSAValue):
        """tt.atomic_rmw → MSL atomic read-modify-write.

        Operands: [ptr, val, mask] (mask may be absent for scalar atomics)
        Result: the OLD value at the atomic location.

        For integer atomics: cast to device atomic_int*/atomic_uint* and use
        atomic_fetch_add/max/min/and/or/xor/exchange_explicit.

        For float atomic add: cast to device atomic_uint* and use a CAS loop
        (Metal device pointers are declared as float*, can't use atomic_float*).

        Float max/min: Triton decomposes these into bitcast + integer atomic
        in the TTGIR, so we only see integer atomics for those cases.
        """
        if len(ssa.operand_ids) < 2:
            return

        ptr_id = ssa.operand_ids[0]
        val_id = ssa.operand_ids[1]
        mask_id = ssa.operand_ids[2] if len(ssa.operand_ids) >= 3 else None

        rmw_op = ssa.attrs.get("rmw_op", "add")
        val_var = self._lookup(val_id)

        # Resolve pointer info
        ptr_info = self.env_is_ptr.get(ptr_id)
        if ptr_info:
            base_ptr, offsets = ptr_info
        else:
            base_ptr = self._lookup(ptr_id)
            offsets = "0"

        # Determine value type (int vs float)
        val_dtype = self.env_types.get(val_id, "fp32")
        is_float = val_dtype.startswith("fp") or val_dtype.startswith("bf") or val_dtype.startswith("f")

        # Determine the storage element type from the pointer arg
        store_dtype = self._trace_ptr_dtype(ptr_id)
        is_float_ptr = store_dtype.startswith("fp") or store_dtype.startswith("bf")

        # Use float detection: if either the value or the pointer is float
        is_float = is_float or is_float_ptr

        # Check for mask
        mask_var = None
        if mask_id is not None:
            if mask_id in self.env_is_mask or self._is_mask(mask_id):
                mask_var = self._lookup(mask_id)
            else:
                # Could be a splat of true — check if it's a constant true
                lookup_val = self._lookup(mask_id)
                if lookup_val not in ("true", "1"):
                    mask_var = lookup_val

        # Unique variable suffix
        n = self._var_counter
        self._var_counter += 1

        # Determine if this is an unsigned operation
        is_unsigned = rmw_op in ("umax", "umin")

        # MSL atomic function map for integer atomics
        _RMW_TO_MSL = {
            "add": "atomic_fetch_add_explicit",
            "fadd": None,  # handled separately (CAS loop for float)
            "max": "atomic_fetch_max_explicit",
            "umax": "atomic_fetch_max_explicit",
            "min": "atomic_fetch_min_explicit",
            "umin": "atomic_fetch_min_explicit",
            "and": "atomic_fetch_and_explicit",
            "or": "atomic_fetch_or_explicit",
            "xor": "atomic_fetch_xor_explicit",
            "exch": "atomic_exchange_explicit",
        }

        # Determine result type
        result_var = f"old_{n}"
        if is_float and rmw_op in ("fadd", "add", "exch"):
            result_dtype = "fp32"
            result_msl_type = "float"
            result_zero = "0.0f"
        elif is_unsigned:
            result_dtype = "u32"
            result_msl_type = "uint"
            result_zero = "0u"
        else:
            result_dtype = "i32"
            result_msl_type = "int"
            result_zero = "0"

        # Scalar atomics (non-tensor): only thread 0 per threadgroup executes.
        # In Triton, a scalar atomic (ptr is !tt.ptr, not tensor<Nx!tt.ptr>)
        # is per-program, not per-thread. Guard with lid == 0.
        is_scalar = not ssa.is_tensor

        # In 2D kernels, a 1D atomic tensor (e.g. after 2D→1D reduce) must
        # only execute on the first N threads, not all M*N threads.
        atomic_1d_guard = None
        if self._is_2d and ssa.is_tensor:
            atom_shape = _extract_shape(ssa.type_str)
            if len(atom_shape) == 1 and atom_shape[0] < self.effective_block_size:
                atomic_1d_guard = atom_shape[0]

        # Always declare result variable first (needed for mask or not)
        self.kb.raw_line(f"    {result_msl_type} {result_var} = {result_zero};")

        # Build the guard condition
        guard_parts = []
        if is_scalar:
            guard_parts.append("lid == 0")
        elif atomic_1d_guard is not None:
            guard_parts.append(f"lid < {atomic_1d_guard}u")
        if mask_var:
            guard_parts.append(mask_var)

        # Indent prefix — extra indent inside if block
        has_guard = bool(guard_parts)
        indent = "        " if has_guard else "    "

        # Open guard if-block
        if has_guard:
            guard_cond = " && ".join(guard_parts)
            self.kb.raw_line(f"    if ({guard_cond}) {{")

        if is_float and rmw_op in ("fadd", "add"):
            # Float atomic add via CAS loop
            self.kb.raw_line(f"{indent}device atomic_uint* aptr_{n} = (device atomic_uint*)({base_ptr} + {offsets});")
            self.kb.raw_line(f"{indent}uint old_bits_{n} = atomic_load_explicit(aptr_{n}, memory_order_relaxed);")
            self.kb.raw_line(f"{indent}while (true) {{")
            self.kb.raw_line(f"{indent}    float old_val_{n} = as_type<float>(old_bits_{n});")
            self.kb.raw_line(f"{indent}    float new_val_{n} = old_val_{n} + {val_var};")
            self.kb.raw_line(f"{indent}    uint new_bits_{n} = as_type<uint>(new_val_{n});")
            self.kb.raw_line(f"{indent}    if (atomic_compare_exchange_weak_explicit(aptr_{n}, &old_bits_{n}, new_bits_{n},")
            self.kb.raw_line(f"{indent}            memory_order_relaxed, memory_order_relaxed)) break;")
            self.kb.raw_line(f"{indent}}}")
            self.kb.raw_line(f"{indent}{result_var} = as_type<float>(old_bits_{n});")
        elif is_float and rmw_op == "exch":
            # Float atomic exchange via reinterpret as uint
            self.kb.raw_line(f"{indent}device atomic_uint* aptr_{n} = (device atomic_uint*)({base_ptr} + {offsets});")
            self.kb.raw_line(f"{indent}uint exch_bits_{n} = as_type<uint>((float){val_var});")
            self.kb.raw_line(f"{indent}uint old_bits_{n} = atomic_exchange_explicit(aptr_{n}, exch_bits_{n}, memory_order_relaxed);")
            self.kb.raw_line(f"{indent}{result_var} = as_type<float>(old_bits_{n});")
        elif is_unsigned:
            # Unsigned integer atomics
            msl_fn = _RMW_TO_MSL.get(rmw_op, "atomic_fetch_add_explicit")
            self.kb.raw_line(f"{indent}device atomic_uint* aptr_{n} = (device atomic_uint*)({base_ptr} + {offsets});")
            self.kb.raw_line(f"{indent}{result_var} = {msl_fn}(aptr_{n}, (uint){val_var}, memory_order_relaxed);")
        else:
            # Signed integer atomics
            msl_fn = _RMW_TO_MSL.get(rmw_op, "atomic_fetch_add_explicit")
            self.kb.raw_line(f"{indent}device atomic_int* aptr_{n} = (device atomic_int*)({base_ptr} + {offsets});")
            self.kb.raw_line(f"{indent}{result_var} = {msl_fn}(aptr_{n}, (int){val_var}, memory_order_relaxed);")

        # Close guard if-block
        if has_guard:
            self.kb.raw_line(f"    }}")

        self.env[ssa.id] = result_var
        self.env_types[ssa.id] = result_dtype

    def _lower_atomic_cas(self, ssa: SSAValue):
        """tt.atomic_cas → MSL atomic compare-and-swap.

        Operands: [ptr, cmp, val]
        Result: the OLD value at the atomic location.

        CAS semantics: if *ptr == cmp, set *ptr = val. Return old *ptr.
        """
        if len(ssa.operand_ids) < 3:
            return

        ptr_id = ssa.operand_ids[0]
        cmp_id = ssa.operand_ids[1]
        val_id = ssa.operand_ids[2]

        cmp_var = self._lookup(cmp_id)
        val_var = self._lookup(val_id)

        # Resolve pointer info
        ptr_info = self.env_is_ptr.get(ptr_id)
        if ptr_info:
            base_ptr, offsets = ptr_info
        else:
            base_ptr = self._lookup(ptr_id)
            offsets = "0"

        # Determine value type
        val_dtype = self.env_types.get(val_id, "i32")
        is_float = val_dtype.startswith("fp") or val_dtype.startswith("bf") or val_dtype.startswith("f")

        # Also check pointer type
        store_dtype = self._trace_ptr_dtype(ptr_id)
        is_float_ptr = store_dtype.startswith("fp") or store_dtype.startswith("bf")
        is_float = is_float or is_float_ptr

        # Scalar CAS: only thread 0 per threadgroup should execute
        is_scalar = not ssa.is_tensor

        n = self._var_counter
        self._var_counter += 1

        # Determine result type
        if is_float:
            result_msl_type = "float"
            result_zero = "0.0f"
            result_dtype = "fp32"
        else:
            result_msl_type = "int"
            result_zero = "0"
            result_dtype = "i32"

        result_var = f"old_{n}"
        self.kb.raw_line(f"    {result_msl_type} {result_var} = {result_zero};")

        # Scalar guard
        indent = "    "
        if is_scalar:
            self.kb.raw_line(f"    if (lid == 0) {{")
            indent = "        "

        if is_float:
            # Float CAS: use atomic_uint + as_type casts
            self.kb.raw_line(f"{indent}device atomic_uint* aptr_{n} = (device atomic_uint*)({base_ptr} + {offsets});")
            self.kb.raw_line(f"{indent}uint expected_{n} = as_type<uint>((float){cmp_var});")
            self.kb.raw_line(f"{indent}uint desired_{n} = as_type<uint>((float){val_var});")
            self.kb.raw_line(f"{indent}atomic_compare_exchange_weak_explicit(aptr_{n}, &expected_{n}, desired_{n},")
            self.kb.raw_line(f"{indent}    memory_order_relaxed, memory_order_relaxed);")
            self.kb.raw_line(f"{indent}{result_var} = as_type<float>(expected_{n});")
        else:
            # Integer CAS
            self.kb.raw_line(f"{indent}device atomic_int* aptr_{n} = (device atomic_int*)({base_ptr} + {offsets});")
            self.kb.raw_line(f"{indent}int expected_{n} = (int){cmp_var};")
            self.kb.raw_line(f"{indent}atomic_compare_exchange_weak_explicit(aptr_{n}, &expected_{n}, (int){val_var},")
            self.kb.raw_line(f"{indent}    memory_order_relaxed, memory_order_relaxed);")
            self.kb.raw_line(f"{indent}{result_var} = expected_{n};")

        if is_scalar:
            self.kb.raw_line(f"    }}")

        self.env[ssa.id] = result_var
        self.env_types[ssa.id] = result_dtype

    # -- Noinline function calls (tt.call) --

    @staticmethod
    def _sanitize_func_name(name: str) -> str:
        """Sanitize a Triton mangled function name for MSL.

        Replaces dots and other invalid chars with underscores.
        """
        return name.replace(".", "_")

    def _lower_called_funcs(self):
        """Lower all callee (noinline) functions to MSL device functions.

        Each callee function becomes an MSL device function placed before
        the kernel in the output. The lowering reuses the same op-by-op
        approach as the kernel but with a fresh environment.

        Functions are emitted in reverse order so that callees appear
        before their callers (dependency order).
        """
        # Reverse so leaf functions come first
        for cfunc in reversed(self.graph.called_funcs):
            msl = self._lower_one_called_func(cfunc)
            self.kb._device_functions.append(msl)

    def _lower_one_called_func(self, cfunc: CalledFunc) -> str:
        """Lower a single CalledFunc to an MSL device function string.

        Creates a temporary lowering context (env, env_types, etc.) to
        avoid polluting the main kernel's namespace.
        """
        safe_name = self._sanitize_func_name(cfunc.name)

        # Determine return type
        if len(cfunc.return_types) == 0:
            ret_type = "void"
        elif len(cfunc.return_types) == 1:
            ret_type = triton_type_to_msl(
                _mlir_to_triton_dtype(cfunc.return_types[0])
            )
        else:
            # Multi-value return: use a struct
            ret_type = f"_ret_{safe_name}"

        # Build parameter list
        # Pointer params use 'volatile device' to match kernel buffer qualifiers,
        # which prevent the Metal shader compiler from hoisting loads.
        params = []
        for arg in cfunc.args:
            triton_dtype = _mlir_to_triton_dtype(arg.elem_type)
            if arg.is_ptr:
                inner = triton_type_to_msl(triton_dtype)
                params.append(f"volatile device {inner}* {arg.name}")
            else:
                msl_ty = triton_type_to_msl(triton_dtype)
                params.append(f"{msl_ty} {arg.name}")

        params_str = ", ".join(params)

        # Generate the body using a sub-lowerer
        sub = _DeviceFuncLowerer(cfunc, self.options)
        body_lines = sub.lower_body()

        # Assemble the function
        lines = []

        # Multi-return struct definition
        if len(cfunc.return_types) > 1:
            struct_name = ret_type
            lines.append(f"struct {struct_name} {{")
            for i, rt in enumerate(cfunc.return_types):
                msl_ty = triton_type_to_msl(_mlir_to_triton_dtype(rt))
                lines.append(f"    {msl_ty} v{i};")
            lines.append("};")
            lines.append("")

        lines.append(f"{ret_type} {safe_name}({params_str}) {{")
        for line in body_lines:
            lines.append(line)
        lines.append("}")

        return "\n".join(lines)

    def _lower_call(self, ssa: SSAValue):
        """Lower tt.call to an MSL function call.

        Handles:
        - Void calls (no return value)
        - Single return value
        - Multiple return values (via struct)
        """
        callee = ssa.attrs.get("callee", "unknown_fn")
        safe_callee = self._sanitize_func_name(callee)
        args = [self._lookup(oid) for oid in ssa.operand_ids]
        args_str = ", ".join(args)

        # Find the callee function definition to determine return types
        return_types = []
        if self.graph.called_funcs:
            for cfunc in self.graph.called_funcs:
                if cfunc.name == callee:
                    return_types = cfunc.return_types
                    break

        if not return_types:
            # Void call
            self.kb.raw_line(f"    {safe_callee}({args_str});")
        elif len(return_types) == 1:
            # Single return value
            msl_ty = triton_type_to_msl(_mlir_to_triton_dtype(return_types[0]))
            var = self._next_var("r")
            self.kb.raw_line(f"    {msl_ty} {var} = {safe_callee}({args_str});")
            self.env[ssa.id] = var
            self.env_types[ssa.id] = _mlir_to_triton_dtype(return_types[0])
        else:
            # Multiple return values — call returns a struct
            ret_struct = f"_ret_{safe_callee}"
            var = self._next_var("rv")
            self.kb.raw_line(f"    {ret_struct} {var} = {safe_callee}({args_str});")

            # Map each result ID to its struct field
            if ssa.result_ids:
                for i, rid in enumerate(ssa.result_ids):
                    field_var = f"{var}.v{i}"
                    self.env[rid] = field_var
                    if i < len(return_types):
                        self.env_types[rid] = _mlir_to_triton_dtype(return_types[i])
            else:
                # Single result ID (shouldn't happen for multi-return, but be safe)
                self.env[ssa.id] = f"{var}.v0"
                self.env_types[ssa.id] = _mlir_to_triton_dtype(return_types[0])

    # -- TTG ops (TritonGPU dialect) --

    def _lower_ttg(self, ssa: SSAValue):
        """Lower ttg.* ops (TritonGPU dialect).

        Most ttg ops are layout annotations or shared memory management.
        convert_layout requires shared memory redistribution when the
        source and destination layouts map elements to different threads.
        """
        op = ssa.op
        if op == "ttg.convert_layout":
            self._lower_convert_layout(ssa)
        elif op == "ttg.local_alloc":
            # Shared memory allocation — passthrough (data is same)
            self._emit_passthrough(ssa)
        elif op == "ttg.local_load":
            # Load from shared memory — passthrough
            self._emit_passthrough(ssa)
        elif op == "ttg.local_store":
            # Store to shared memory — passthrough
            self._emit_passthrough(ssa)
        else:
            # Other ttg ops: passthrough
            self._emit_passthrough(ssa)

    def _lower_convert_layout(self, ssa: SSAValue):
        """ttg.convert_layout → shared memory redistribution.

        When the source layout's thread-to-element mapping differs from the
        destination layout, elements must be redistributed via shared memory:
          1. Each thread writes its value to shared[source_index]
          2. threadgroup_barrier
          3. Each thread reads shared[dest_index]

        For 1D tensors in our model: after a 2D reduce, the broadcast puts
        results on threads using lid/N or lid%M, but the destination layout
        expects a simple lid-based mapping. The shared memory shuffle fixes
        this mismatch.

        For cases where both layouts use the same mapping (e.g. same blocked
        layout), this is a passthrough.
        """
        if not ssa.operand_ids:
            self._emit_passthrough(ssa)
            return

        src_var = self._lookup(ssa.operand_ids[0])
        src_shape = _extract_shape(ssa.type_str)
        if not src_shape:
            # Can't determine shape — passthrough
            self._emit_passthrough(ssa)
            return

        N = 1
        for d in src_shape:
            N *= d

        # Only do shared memory redistribution when the source layout is
        # a #ttg.slice (from a reduce). For #blocked → #blocked conversions,
        # our 1D model doesn't distinguish between blocked layouts, so
        # passthrough is correct.
        src_type = ""
        for op in self.graph.ops:
            if op.id == ssa.operand_ids[0] and op.type_str:
                src_type = op.type_str
                break
        if not src_type:
            src_type = self._find_op_type_str(ssa.operand_ids[0]) or ""

        # Only redistribute when converting FROM a slice layout TO a
        # non-slice layout (e.g., #ttg.slice → #blocked2). This indicates
        # the reduce result needs remapping to a new thread assignment.
        # When both source and dest are slice layouts, it's a layout
        # variant change that doesn't affect thread mapping in our model.
        dest_type = ssa.type_str or ""
        needs_redistribute = ("ttg.slice" in src_type
                              and "ttg.slice" not in dest_type)

        if not needs_redistribute or not self._is_2d or N <= 1:
            self._emit_passthrough(ssa)
            return

        # Determine source element type
        src_dtype = self.env_types.get(ssa.operand_ids[0], "fp32")
        is_int = not (src_dtype.startswith("fp") or src_dtype.startswith("bf"))
        msl_type = "int" if is_int else "float"
        shared_dtype = "i32" if is_int else "fp32"

        # Allocate shared memory for the redistribution
        shared_name = f"shared_{self._shared_counter}"
        self._shared_counter += 1
        self.kb.declare_threadgroup_array(shared_name, dtype=shared_dtype, size=N)

        # Determine the source write index.
        # After a 2D reduce with broadcast, the thread-to-element mapping
        # uses lid / N_reduced (blocked row indexing), where N_reduced is
        # the inner dim of the reduce input (NOT the global 2D shape).
        # Trace back to find the reduce that produced this value and get
        # its input inner dim.
        N_reduced = None
        reduce_axis = None
        src_id = ssa.operand_ids[0]
        # Trace through passthroughs (addf, etc.) to find the reduce
        visited = set()
        trace_id = src_id
        while trace_id not in visited:
            visited.add(trace_id)
            for op in self.graph.ops:
                if op.id == trace_id:
                    if op.op == "tt.reduce":
                        if op.operand_ids:
                            inp_shape = self.env_shapes.get(op.operand_ids[0])
                            if not inp_shape:
                                inp_type = self._find_op_type_str(op.operand_ids[0])
                                inp_shape = _extract_shape(inp_type) if inp_type else None
                            if inp_shape and len(inp_shape) >= 2:
                                reduce_axis = op.attrs.get("axis", 0)
                                if reduce_axis == 1:
                                    N_reduced = inp_shape[1]
                                elif reduce_axis == 0:
                                    N_reduced = inp_shape[0]
                        break
                    elif op.operand_ids:
                        trace_id = op.operand_ids[0]
                    break

        if N_reduced and N_reduced > 1:
            if reduce_axis == 1:
                # axis=1: broadcast used lid / N_reduced (blocked row)
                src_idx = f"lid / {N_reduced}u"
                self.kb.raw_line(f"    if (lid % {N_reduced}u == 0u && lid / {N_reduced}u < {N}u)")
                self.kb.raw_line(f"        {shared_name}[{src_idx}] = {src_var};")
            else:
                # axis=0: broadcast used lid % N (modular column)
                src_idx = f"lid % {N}u"
                self.kb.raw_line(f"    if (lid < {N}u)")
                self.kb.raw_line(f"        {shared_name}[{src_idx}] = {src_var};")
        else:
            # Standard modular mapping or no reduce found
            self.kb.raw_line(f"    if (lid < {N}u)")
            self.kb.raw_line(f"        {shared_name}[lid % {N}u] = {src_var};")

        self.kb.raw_line(f"    threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Read back in destination layout: thread i gets element i
        result_var = self._next_var("cvt")
        self.kb.raw_line(f"    {msl_type} {result_var} = (lid < {N}u) ? {shared_name}[lid] : ({msl_type})0;")

        self.env[ssa.id] = result_var
        self.env_types[ssa.id] = src_dtype
        if src_shape:
            self.env_shapes[ssa.id] = src_shape
        # Mark this value as having been through convert_layout — the
        # thread-to-element mapping is now simple (thread i = element i).
        # This prevents the store from using 2D-aware guards.
        if not hasattr(self, '_converted_layout_ids'):
            self._converted_layout_ids = set()
        self._converted_layout_ids.add(ssa.id)


# ---------------------------------------------------------------------------
# Device function lowerer (for noinline callees)
# ---------------------------------------------------------------------------

class _DeviceFuncLowerer:
    """Lower the body of a device function (noinline callee) to MSL lines.

    This is a lightweight version of GenericLowerer that operates on
    a CalledFunc instead of an IRGraph. It reuses the same op dispatch
    but produces raw MSL lines instead of using KernelBuilder.
    """

    def __init__(self, cfunc: CalledFunc, options=None):
        self.cfunc = cfunc
        self.options = options
        self.env = {}
        self.env_types = {}
        self.env_is_mask = {}
        self.env_is_ptr = {}
        self._var_counter = 0
        self._lines = []

    def _next_var(self, prefix="r") -> str:
        name = f"{prefix}_{self._var_counter}"
        self._var_counter += 1
        return name

    def _lookup(self, ssa_id: int) -> str:
        if ssa_id in self.env:
            return self.env[ssa_id]
        return f"UNKNOWN_{ssa_id}"

    def _emit(self, line: str):
        self._lines.append(f"    {line}")

    def lower_body(self) -> List[str]:
        """Lower all ops and return lines of MSL body code."""
        # Register function arguments
        for arg in self.cfunc.args:
            self.env[arg.id] = arg.name
            self.env_types[arg.id] = _mlir_to_triton_dtype(arg.elem_type)
            if arg.is_ptr:
                self.env_is_ptr[arg.id] = (arg.name, None)

        # Lower each op
        for ssa in self.cfunc.ops:
            self._lower_op(ssa)

        return self._lines

    def _lower_op(self, ssa: SSAValue):
        """Lower a single op in a device function."""
        op = ssa.op

        if op == "tt.return":
            self._lower_return(ssa)
        elif op == "tt.call":
            self._lower_call(ssa)
        elif op == "tt.load":
            self._lower_load(ssa)
        elif op == "tt.store":
            self._lower_store(ssa)
        elif op == "tt.addptr":
            self._lower_addptr(ssa)
        elif op == "tt.splat":
            self._emit_passthrough(ssa)
        elif op == "tt.broadcast":
            self._emit_passthrough(ssa)
        elif op == "arith.constant":
            self._lower_constant(ssa)
        elif op.startswith("arith."):
            self._lower_arith(ssa)
        elif op.startswith("math."):
            self._lower_math(ssa)
        elif op == "scf.if":
            self._lower_scf_if(ssa)
        elif op in ("scf.yield", "scf.condition"):
            pass
        elif op == "scf.for":
            self._lower_scf_for(ssa)
        elif op == "tt.extern_elementwise":
            self._lower_extern_elementwise(ssa)
        elif op.startswith("ttg.") or op in ("tt.reshape", "tt.expand_dims",
                                               "tt.unsplat", "tt.make_range"):
            self._emit_passthrough(ssa)
        else:
            self._emit(f"// UNSUPPORTED in device func: {op}")

    def _lower_return(self, ssa: SSAValue):
        """tt.return with value(s) in a device function."""
        if not ssa.operand_ids:
            self._emit("return;")
        elif len(ssa.operand_ids) == 1:
            val = self._lookup(ssa.operand_ids[0])
            self._emit(f"return {val};")
        else:
            # Multi-value return: construct struct
            vals = [self._lookup(oid) for oid in ssa.operand_ids]
            ret_struct = f"_ret_{self.cfunc.name.replace('.', '_')}"
            fields = ", ".join(vals)
            self._emit(f"return {ret_struct}{{{fields}}};")

    def _lower_call(self, ssa: SSAValue):
        """tt.call in a device function (nested calls)."""
        callee = ssa.attrs.get("callee", "unknown_fn")
        safe_callee = callee.replace(".", "_")
        args = [self._lookup(oid) for oid in ssa.operand_ids]
        args_str = ", ".join(args)

        n_results = len(ssa.result_ids) if ssa.result_ids else (1 if ssa.type_str else 0)

        if n_results == 0 or not ssa.type_str:
            self._emit(f"{safe_callee}({args_str});")
        elif n_results == 1 or not ssa.result_ids:
            msl_ty = triton_type_to_msl(_mlir_to_triton_dtype(ssa.elem_type or "f32"))
            var = self._next_var("r")
            self._emit(f"{msl_ty} {var} = {safe_callee}({args_str});")
            self.env[ssa.id] = var
            self.env_types[ssa.id] = _mlir_to_triton_dtype(ssa.elem_type or "f32")
        else:
            # Multi-return
            ret_struct = f"_ret_{safe_callee}"
            var = self._next_var("rv")
            self._emit(f"{ret_struct} {var} = {safe_callee}({args_str});")
            for i, rid in enumerate(ssa.result_ids):
                self.env[rid] = f"{var}.v{i}"
                self.env_types[rid] = _mlir_to_triton_dtype(ssa.elem_type or "f32")

    def _lower_load(self, ssa: SSAValue):
        """tt.load — scalar load in device function."""
        if not ssa.operand_ids:
            return
        ptr_id = ssa.operand_ids[0]
        elem = ssa.elem_type or "f32"
        triton_dtype = _mlir_to_triton_dtype(elem)
        msl_ty = _msl_compute_type(triton_dtype)
        var = self._next_var("ld")

        if ptr_id in self.env_is_ptr:
            base, offs = self.env_is_ptr[ptr_id]
            if offs:
                self._emit(f"{msl_ty} {var} = static_cast<{msl_ty}>({base}[{offs}]);")
            else:
                self._emit(f"{msl_ty} {var} = static_cast<{msl_ty}>({base}[0]);")
        else:
            ptr = self._lookup(ptr_id)
            self._emit(f"{msl_ty} {var} = static_cast<{msl_ty}>({ptr}[0]);")

        self.env[ssa.id] = var
        self.env_types[ssa.id] = triton_dtype

    def _lower_store(self, ssa: SSAValue):
        """tt.store — scalar store in device function."""
        if len(ssa.operand_ids) < 2:
            return
        ptr_id = ssa.operand_ids[0]
        val_id = ssa.operand_ids[1]
        val = self._lookup(val_id)

        if ptr_id in self.env_is_ptr:
            base, offs = self.env_is_ptr[ptr_id]
            if offs:
                self._emit(f"{base}[{offs}] = {val};")
            else:
                self._emit(f"{base}[0] = {val};")
        else:
            ptr = self._lookup(ptr_id)
            self._emit(f"{ptr}[0] = {val};")

    def _lower_addptr(self, ssa: SSAValue):
        """tt.addptr — pointer arithmetic."""
        if len(ssa.operand_ids) < 2:
            return
        ptr_id = ssa.operand_ids[0]
        off_id = ssa.operand_ids[1]
        off = self._lookup(off_id)

        if ptr_id in self.env_is_ptr:
            base, existing_off = self.env_is_ptr[ptr_id]
            if existing_off:
                combined = f"({existing_off} + {off})"
            else:
                combined = off
            self.env_is_ptr[ssa.id] = (base, combined)
        else:
            base = self._lookup(ptr_id)
            self.env_is_ptr[ssa.id] = (base, off)

        self.env[ssa.id] = self._lookup(ptr_id)
        if ptr_id in self.env_types:
            self.env_types[ssa.id] = self.env_types[ptr_id]

    def _lower_constant(self, ssa: SSAValue):
        """arith.constant — emit a constant value."""
        val = ssa.attrs.get("value")
        elem = ssa.elem_type or "f32"
        triton_dtype = _mlir_to_triton_dtype(elem)

        if val is None:
            val = 0

        if isinstance(val, bool):
            msl_val = "true" if val else "false"
            msl_ty = "bool"
        elif isinstance(val, float) or (isinstance(val, int) and elem.startswith("f")):
            msl_val = f"{float(val)}f"
            msl_ty = "float"
        else:
            msl_val = str(val)
            msl_ty = triton_type_to_msl(triton_dtype)

        var = self._next_var("c")
        self._emit(f"{msl_ty} {var} = {msl_val};")
        self.env[ssa.id] = var
        self.env_types[ssa.id] = triton_dtype

    def _lower_arith(self, ssa: SSAValue):
        """Lower arith.* ops."""
        op = ssa.op
        ids = ssa.operand_ids

        arith_map = {
            "arith.addf": "+", "arith.subf": "-",
            "arith.mulf": "*", "arith.divf": "/",
            "arith.addi": "+", "arith.subi": "-",
            "arith.muli": "*", "arith.divsi": "/", "arith.divui": "/",
            "arith.remsi": "%", "arith.remui": "%",
            "arith.andi": "&", "arith.ori": "|", "arith.xori": "^",
            "arith.maxnumf": "max", "arith.minnumf": "min",
            "arith.maximumf": "max", "arith.minimumf": "min",
            "arith.maxsi": "max", "arith.minsi": "min",
            "arith.maxui": "max", "arith.minui": "min",
            "arith.shrsi": ">>", "arith.shrui": ">>", "arith.shli": "<<",
        }

        if op in arith_map and len(ids) >= 2:
            a = self._lookup(ids[0])
            b = self._lookup(ids[1])
            symbol = arith_map[op]
            var = self._next_var("r")
            elem = ssa.elem_type or "f32"
            triton_dtype = _mlir_to_triton_dtype(elem)
            msl_ty = triton_type_to_msl(triton_dtype)

            if symbol in ("+", "-", "*", "/", "%", "&", "|", "^", ">>", "<<"):
                self._emit(f"{msl_ty} {var} = {a} {symbol} {b};")
            else:
                self._emit(f"{msl_ty} {var} = {symbol}({a}, {b});")

            self.env[ssa.id] = var
            self.env_types[ssa.id] = triton_dtype
            return

        if op in ("arith.cmpi", "arith.cmpf"):
            pred = ssa.attrs.get("predicate_name", "")
            pred_op = CMPF_NAMED.get(pred, CMPI_NAMED.get(pred, "=="))
            if len(ids) >= 2:
                a = self._lookup(ids[0])
                b = self._lookup(ids[1])
                var = self._next_var("cmp")
                self._emit(f"bool {var} = ({a} {pred_op} {b});")
                self.env[ssa.id] = var
                self.env_types[ssa.id] = "i1"
                self.env_is_mask[ssa.id] = True
            return

        if op == "arith.negf" and len(ids) >= 1:
            a = self._lookup(ids[0])
            var = self._next_var("r")
            self._emit(f"float {var} = -{a};")
            self.env[ssa.id] = var
            self.env_types[ssa.id] = "fp32"
            return

        if op == "arith.select" and len(ids) >= 3:
            cond = self._lookup(ids[0])
            a = self._lookup(ids[1])
            b = self._lookup(ids[2])
            var = self._next_var("sel")
            self._emit(f"float {var} = {cond} ? {a} : {b};")
            self.env[ssa.id] = var
            self.env_types[ssa.id] = "fp32"
            return

        if op in ("arith.extf", "arith.truncf", "arith.sitofp", "arith.fptosi",
                   "arith.extsi", "arith.extui", "arith.trunci", "arith.uitofp",
                   "arith.fptoui", "arith.index_cast", "arith.index_castui",
                   "arith.bitcast"):
            self._emit_passthrough(ssa)
            if ssa.elem_type:
                self.env_types[ssa.id] = _mlir_to_triton_dtype(ssa.elem_type)
            return

        self._emit(f"// UNSUPPORTED arith in device func: {op}")

    def _lower_math(self, ssa: SSAValue):
        """Lower math.* ops."""
        op = ssa.op
        ids = ssa.operand_ids

        math_map = {
            "math.exp": "exp", "math.exp2": "exp2",
            "math.log": "log", "math.log2": "log2",
            "math.sqrt": "sqrt",
            "math.rsqrt": "rsqrt", "math.abs": "abs", "math.absf": "abs",
            "math.ceil": "ceil", "math.floor": "floor",
            "math.sin": "sin", "math.cos": "cos", "math.tanh": "tanh",
            "math.round": "round", "math.roundeven": "rint", "math.trunc": "trunc",
            "math.fma": "fma",
        }

        short = op.split(".")[-1] if "." in op else op
        func = math_map.get(op, short)

        if op == "math.fma" and len(ids) >= 3:
            a, b, c = [self._lookup(i) for i in ids[:3]]
            var = self._next_var("r")
            self._emit(f"float {var} = fma({a}, {b}, {c});")
            self.env[ssa.id] = var
            self.env_types[ssa.id] = "fp32"
        elif op == "math.log1p" and len(ids) >= 1:
            a = self._lookup(ids[0])
            var = self._next_var("r")
            self._emit(f"float {var} = log(1.0f + {a});")
            self.env[ssa.id] = var
            self.env_types[ssa.id] = "fp32"
        elif op == "math.expm1" and len(ids) >= 1:
            a = self._lookup(ids[0])
            var = self._next_var("r")
            self._emit(f"float {var} = (exp({a}) - 1.0f);")
            self.env[ssa.id] = var
            self.env_types[ssa.id] = "fp32"
        elif len(ids) >= 1:
            a = self._lookup(ids[0])
            var = self._next_var("r")
            self._emit(f"float {var} = {func}({a});")
            self.env[ssa.id] = var
            self.env_types[ssa.id] = "fp32"

    def _lower_extern_elementwise(self, ssa: SSAValue):
        """tt.extern_elementwise → direct MSL function call in device func."""
        func_name = ssa.attrs.get("symbol", "")
        if not func_name:
            func_name = ssa.attrs.get("libname", "")
        if not func_name:
            self._emit(f"// UNSUPPORTED: tt.extern_elementwise (no symbol)")
            return

        # Sanitize __nv_* CUDA libdevice names to Metal equivalents
        safe_name = func_name
        if safe_name.startswith("__nv_"):
            stripped = safe_name[5:]
            if stripped.endswith("f") and len(stripped) > 1:
                stripped = stripped[:-1]
            safe_name = stripped

        args = [self._lookup(oid) for oid in ssa.operand_ids]
        args_str = ", ".join(args)

        elem = ssa.elem_type or "f32"
        triton_dtype = _mlir_to_triton_dtype(elem)
        if triton_dtype.startswith("fp") or triton_dtype.startswith("bf"):
            msl_ty = "float"
        elif triton_dtype.startswith("u"):
            msl_ty = "uint"
        elif triton_dtype == "i64":
            msl_ty = "long"
        else:
            msl_ty = triton_type_to_msl(triton_dtype)

        var = self._next_var("r")
        self._emit(f"{msl_ty} {var} = {safe_name}({args_str});")
        self.env[ssa.id] = var
        self.env_types[ssa.id] = triton_dtype

    def _lower_scf_for(self, ssa: SSAValue):
        """scf.for → MSL for loop with iter_args in device function.

        Reuses the same logic as GenericLowerer._lower_scf_for():
        scf.for has operands: [start, end, step, init_0, init_1, ...]
        Results: [result_0, result_1, ...] (same count as iter_args)
        Body block args: [induction_var, iter_arg_0, iter_arg_1, ...]
        """
        if len(ssa.operand_ids) < 3:
            return

        start_var = self._lookup(ssa.operand_ids[0])
        end_var = self._lookup(ssa.operand_ids[1])
        step_var = self._lookup(ssa.operand_ids[2])

        # iter_args initial values: operands[3:]
        init_ids = ssa.operand_ids[3:]
        n_iter_args = len(init_ids)

        # Declare and initialize iter_arg variables
        iter_vars = []
        iter_dtypes = []
        result_elem = ssa.elem_type or "f32"
        for i, init_id in enumerate(init_ids):
            var_name = self._next_var("iter")
            init_val = self._lookup(init_id)
            init_type = self.env_types.get(init_id, "fp32")
            if result_elem in ("i64",) and init_type in ("i32", "fp32"):
                init_type = result_elem
            if init_type.startswith("f") or init_type.startswith("bf"):
                msl_type = "float"
            elif init_type in ("i64",):
                msl_type = "long"
            elif init_type.startswith("u"):
                msl_type = "uint"
            else:
                msl_type = "int"
            self._emit(f"{msl_type} {var_name} = {init_val};")
            iter_vars.append(var_name)
            iter_dtypes.append(init_type)

        # Emit for loop
        start_type = self.env_types.get(ssa.operand_ids[0], "i32")
        is_i64 = start_type == "i64" or "i64" in (ssa.type_str or "")
        loop_type = "long" if is_i64 else "int"
        loop_var = self._next_var("k")

        self._emit(
            f"for ({loop_type} {loop_var} = {start_var}; "
            f"{loop_var} < {end_var}; {loop_var} += {step_var}) {{"
        )

        # Map block args to MSL variables
        block_arg_ids = ssa.attrs.get("block_arg_ids", [])
        if block_arg_ids:
            # First block arg is induction variable
            self.env[block_arg_ids[0]] = loop_var
            self.env_types[block_arg_ids[0]] = start_type
            # Remaining block args are iter_args
            for i, var in enumerate(iter_vars):
                if i + 1 < len(block_arg_ids):
                    self.env[block_arg_ids[i + 1]] = var
                    self.env_types[block_arg_ids[i + 1]] = iter_dtypes[i] if i < len(iter_dtypes) else "fp32"

        # Process body ops
        if ssa.region_ops:
            for body_op in ssa.region_ops:
                if body_op.op == "scf.yield":
                    # Update iter_arg variables from yield operands
                    for i, yield_id in enumerate(body_op.operand_ids):
                        if i < len(iter_vars):
                            yield_val = self._lookup(yield_id)
                            self._emit(f"    {iter_vars[i]} = {yield_val};")
                else:
                    self._lower_op(body_op)

        self._emit("}")

        # Map scf.for results to iter_arg variables
        if ssa.result_ids:
            for i, var in enumerate(iter_vars):
                if i < len(ssa.result_ids):
                    self.env[ssa.result_ids[i]] = var
                    self.env_types[ssa.result_ids[i]] = iter_dtypes[i] if i < len(iter_dtypes) else "fp32"
        elif n_iter_args == 1 and iter_vars:
            self.env[ssa.id] = iter_vars[0]
            self.env_types[ssa.id] = iter_dtypes[0] if iter_dtypes else "fp32"
        elif iter_vars:
            self.env[ssa.id] = iter_vars[0]
            self.env_types[ssa.id] = iter_dtypes[0] if iter_dtypes else "fp32"

    def _lower_scf_if(self, ssa: SSAValue):
        """scf.if in a device function.

        For scf.if with results, we declare result variables before the if
        statement and assign them in each branch via scf.yield. This ensures
        the variables are in scope after the if block.
        """
        if not ssa.operand_ids:
            return
        cond = self._lookup(ssa.operand_ids[0])

        # Pre-declare result variables if the scf.if produces results
        # result_ids may be None for single-result scf.if (walker stores in ssa.id)
        rids = ssa.result_ids if ssa.result_ids else ([ssa.id] if ssa.type_str else [])
        result_vars = []
        if rids:
            for i, rid in enumerate(rids):
                var = self._next_var("if_res")
                # Determine type from elem_type or default to float
                msl_ty = "float"
                if ssa.elem_type and ssa.elem_type.startswith("i"):
                    msl_ty = triton_type_to_msl(_mlir_to_triton_dtype(ssa.elem_type))
                self._emit(f"{msl_ty} {var};")
                result_vars.append(var)
                self.env[rid] = var
                self.env_types[rid] = _mlir_to_triton_dtype(ssa.elem_type or "f32")

        self._emit(f"if ({cond}) {{")

        if ssa.region_ops:
            for sub_op in ssa.region_ops:
                if sub_op.op == "scf.yield":
                    # Assign yielded values to pre-declared result variables
                    if result_vars and sub_op.operand_ids:
                        for var, yid in zip(result_vars, sub_op.operand_ids):
                            val = self._lookup(yid)
                            self._emit(f"    {var} = {val};")
                else:
                    self._lower_op(sub_op)

        if ssa.else_ops:
            self._emit("} else {")
            for sub_op in ssa.else_ops:
                if sub_op.op == "scf.yield":
                    # Assign yielded values to pre-declared result variables
                    if result_vars and sub_op.operand_ids:
                        for var, yid in zip(result_vars, sub_op.operand_ids):
                            val = self._lookup(yid)
                            self._emit(f"    {var} = {val};")
                else:
                    self._lower_op(sub_op)

        self._emit("}")

    def _emit_passthrough(self, ssa: SSAValue):
        """Emit a passthrough (type conversion that's a no-op)."""
        if ssa.operand_ids:
            src_id = ssa.operand_ids[0]
            self.env[ssa.id] = self._lookup(src_id)
            if src_id in self.env_types:
                self.env_types[ssa.id] = self.env_types[src_id]
            if src_id in self.env_is_mask:
                self.env_is_mask[ssa.id] = True
            if src_id in self.env_is_ptr:
                self.env_is_ptr[ssa.id] = self.env_is_ptr[src_id]
            # Propagate shape through passthrough
            if src_id in self.env_shapes:
                self.env_shapes[ssa.id] = self.env_shapes[src_id]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def lower_ir_graph(graph: IRGraph, options=None) -> str:
    """Lower an IRGraph to MSL source code.

    Args:
        graph: The IRGraph from mlir_walker.walk_ttgir().
        options: MetalOptions instance.

    Returns:
        MSL source code string.
    """
    lowerer = GenericLowerer(graph, options)
    return lowerer.lower()
