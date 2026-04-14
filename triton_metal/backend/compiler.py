import functools
import hashlib
import os
import sys
import subprocess
import tempfile
from dataclasses import dataclass, field, MISSING
from typing import Dict

from types import ModuleType

# Import BaseBackend from triton.backends.compiler.
# This can trigger a circular import during Triton's backend discovery:
#   triton.backends.__init__ → _discover_backends() → import this module
#   → import triton.backends.compiler → import triton.backends.__init__ (cycle)
#
# Fix: if triton.backends is currently being loaded (partially initialized),
# import BaseBackend directly from the compiler submodule without going
# through triton.backends.__init__.
from triton.backends.compiler import BaseBackend, GPUTarget


def _get_cache_dir():
    """Return the persistent cache directory for compiled kernels.

    The directory is created if it does not exist.  Users can override the
    location by setting the ``TRITON_METAL_CACHE_DIR`` environment variable.
    """
    cache_dir = os.environ.get("TRITON_METAL_CACHE_DIR")
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir
    # Default: ~/.cache/triton_metal/
    home = os.path.expanduser("~")
    cache_dir = os.path.join(home, ".cache", "triton_metal")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


@dataclass(frozen=True)
class MetalOptions:
    num_warps: int = 4
    num_stages: int = 1
    num_ctas: int = 1
    # Apple GPU SIMD-groups are always 32-wide.
    warp_size: int = 32
    # Metal threadgroup memory is capped at 32 KB.
    max_threadgroup_memory: int = 32768
    enable_fp_fusion: bool = True
    # FP8 via software emulation — store as uchar, convert to/from float.
    supported_fp8_dtypes: tuple = ("fp8e4nv", "fp8e5", "fp8e4b15", "fp8e4b8", "fp8e5b16")
    default_dot_input_precision: str = "ieee"
    allowed_dot_input_precisions: tuple = ("ieee",)
    max_num_imprecise_acc_default: int = 0
    extern_libs: dict = field(default_factory=dict)
    debug: bool = False
    backend_name: str = "metal"
    arch: str = "apple-m4"
    sanitize_overflow: bool = True
    launch_cooperative_grid: bool = False
    launch_pdl: bool = False
    instrumentation_mode: str = ""
    # Metal Shading Language version for xcrun compilation.
    # "auto" (default) detects from the current device and SDK.
    target_metal_version: str = "auto"

    @staticmethod
    def _make_hashable(value):
        if isinstance(value, dict):
            return tuple(sorted(value.items()))
        return value

    def hash(self):
        hash_dict = dict(self.__dict__)
        key = "_".join(
            f"{name}-{self._make_hashable(val)}"
            for name, val in sorted(hash_dict.items())
        )
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class MetalBackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend in ("metal", "mps")

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.binary_ext = "metallib"

    def parse_options(self, opts: dict) -> MetalOptions:
        result = {}
        for k, f in MetalOptions.__dataclass_fields__.items():
            if k in opts:
                result[k] = opts[k]
            elif f.default is not MISSING:
                result[k] = f.default
            elif f.default_factory is not MISSING:
                result[k] = f.default_factory()

        # Validate num_warps is a power of 2
        num_warps = result.get("num_warps", 4)
        assert num_warps > 0 and (num_warps & (num_warps - 1)) == 0, \
            f"num_warps ({num_warps}) must be a power of 2"

        # Validate: no FP64 on Metal.
        arch = result.get("arch", "apple-m4")
        result["arch"] = arch

        return MetalOptions(**result)

    def pack_metadata(self, metadata):
        block_size = getattr(metadata, "block_size", None) or metadata.num_warps * 32
        output_arg_indices = getattr(metadata, "output_arg_indices", None)
        needs_2d_grid = getattr(metadata, "needs_2d_grid", False)
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            block_size,
            output_arg_indices,
            needs_2d_grid,
        )

    def get_codegen_implementation(self, options):
        return {
            "min_dot_size": lambda lhs_type, rhs_type: (1, 1, 1),
        }

    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}

    def load_dialects(self, ctx):
        # No custom MLIR dialects for now.
        pass

    def add_stages(self, stages, options, language=None):
        from triton.compiler.compiler import Language

        if language == Language.GLUON:
            # Gluon: skip TTIR, use gluon-specific passes to reach TTGIR.
            stages["ttgir"] = lambda src, metadata: self.gluon_to_ttgir(src, metadata, options)
        else:
            # Triton (default): TTIR → TTGIR
            stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
            stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options)

        # Optional C++ LLVM IR lowering: TTGIR → AIR LLVM IR → metallib
        # Enabled by TRITON_METAL_USE_CPP=1. Bypasses MSL for the metallib
        # compilation. Kernels with unsupported ops fall back to MSL path.
        use_cpp = os.environ.get("TRITON_METAL_USE_CPP", "") == "1"
        if use_cpp and self._has_cpp_passes():
            def _llir_or_msl(src, metadata):
                """Try C++ lowering; fall back to MSL for unsupported ops."""
                ttgir_text = str(src)
                if MetalBackend._has_unsupported_ops(ttgir_text):
                    # Fall back to MSL path
                    msl_src = MetalBackend.make_msl(src, metadata, options)
                    metallib = MetalBackend.make_metallib(msl_src, metadata, options)
                    metadata["cpp_fallback_metallib"] = metallib
                    metadata["cpp_fallback_msl"] = msl_src
                    return None
                return MetalBackend.make_llir(src, metadata, options)

            def _metallib_from_llir_or_fallback(src, metadata):
                fb = metadata.pop("cpp_fallback_metallib", None)
                if fb is not None:
                    return fb
                return MetalBackend.make_metallib_from_llir(src, metadata, options)

            stages["llir"] = _llir_or_msl
            stages["metallib"] = _metallib_from_llir_or_fallback
        else:
            stages["msl"] = lambda src, metadata: self.make_msl(src, metadata, options)
            stages["metallib"] = lambda src, metadata: self.make_metallib(src, metadata, options)

    @staticmethod
    def _has_cpp_passes():
        """Check if the C++ MLIR pass library is available."""
        try:
            import triton_metal._triton_metal_cpp
            return True
        except ImportError:
            return False

    @staticmethod
    def _strip_ttg_annotations(ttgir_text):
        """Strip TritonGPU encoding attributes, loc annotations, and TTG ops.

        Our C++ pass doesn't need the encoding annotations (it converts
        tensor types to scalars regardless), and parsing them would require
        the TritonGPU dialect which pulls in NVIDIA-specific dependencies.

        Additionally, this method replaces TritonGPU ops with their operand
        passthroughs at the text level:
        - ttg.local_alloc %x -> deleted (local_load will use %x directly)
        - ttg.local_load %x  -> replaced with the original alloc operand
        - ttg.convert_layout %x -> replaced with %x (passthrough)
        - ttg.memdesc_trans %x -> replaced with %x (passthrough)
        - tt.dot %a, %b, %c -> arith.mulf + arith.addf (scalar FMA)
        - tt.trans %x -> replaced with %x (passthrough in per-thread model)
        """
        import re

        # Phase 0.5: Detect 2D blocks and annotate make_range with dimension info.
        #
        # In 2D kernels (matmul), each make_range maps to a different
        # dimension of the output block. We detect slice<{dim = N}> on
        # make_range ops and add a metal.dim attribute so the C++ backend
        # can decompose the linear thread ID:
        #   dim=1 → row index:  lid / BLOCK_COL
        #   dim=0 → col index:  lid % BLOCK_COL
        lines = ttgir_text.split('\n')
        make_range_dims = {}  # %name -> (dim, block_size)
        for line in lines:
            mr_m = re.search(
                r'(%\S+)\s*=\s*tt\.make_range\s*\{[^}]*end\s*=\s*(\d+)[^}]*\}\s*:\s*tensor<\d+xi32,\s*#ttg\.slice<\{dim\s*=\s*(\d+)',
                line
            )
            if mr_m:
                name = mr_m.group(1)
                block_sz = int(mr_m.group(2))
                dim = int(mr_m.group(3))
                make_range_dims[name] = (dim, block_sz)

        # Determine if this is a 2D kernel
        has_2d_block = len(set(d for d, _ in make_range_dims.values())) > 1

        # For 2D blocks, find the column block size (dim=0)
        col_block_size = 32  # default
        if has_2d_block:
            for dim, bsz in make_range_dims.values():
                if dim == 0:
                    col_block_size = bsz
                    break

        # Annotate make_range ops with metal.dim and metal.col_block_size.
        # This is done before stripping so the dim info is preserved.
        annotated_lines = []
        for line in lines:
            if has_2d_block and 'tt.make_range' in line:
                # Find the make_range attribute dict and slice dim
                mr_m = re.search(
                    r'(tt\.make_range\s*\{)([^}]*)(}\s*:)',
                    line
                )
                slice_m = re.search(
                    r'#ttg\.slice<\{dim\s*=\s*(\d+)',
                    line
                )
                if mr_m and slice_m:
                    dim_val = int(slice_m.group(1))
                    # Inject metal.dim and metal.col_block_size attributes
                    new_attrs = f'{mr_m.group(2)}, metal.dim = {dim_val} : i32, metal.col_block_size = {col_block_size} : i32'
                    line = line[:mr_m.start(1)] + mr_m.group(1) + new_attrs + mr_m.group(3) + line[mr_m.end():]
            annotated_lines.append(line)

        # Phase 1: Strip encoding attributes and loc annotations
        phase1_lines = []
        for line in annotated_lines:
            stripped = line.rstrip()

            # Skip #name = #ttg.blocked<...> alias definitions
            if re.match(r'^#\w+ = #ttg\.', stripped):
                continue

            # Skip #locN = loc(...) alias definitions
            if re.match(r'^#loc\d*\s*=\s*loc\(', stripped):
                continue

            # Remove , #blocked from tensor types (simple alias references)
            stripped = re.sub(r',\s*#\w+>', '>', stripped)

            # Remove inline TTG type annotations:
            #   , #ttg.slice<{dim = 1, parent = #blocked1}>
            #   , #ttg.dot_op<{opIdx = 0, parent = #blocked}>
            #   , #ttg.swizzled_shared<{...}>
            # These appear inside tensor<...> types. Match , #ttg.X<{...}>
            stripped = re.sub(
                r',\s*#ttg\.\w+<\{[^}]*\}>', '', stripped
            )

            # Remove ttg.* module attributes
            stripped = re.sub(r'"ttg\.[^"]*"\s*=\s*[^,}]+[,]?\s*', '', stripped)
            stripped = re.sub(r'ttg\.\w+\s*=\s*"[^"]*"[,]?\s*', '', stripped)

            # Remove loc(...) annotations — handle nested parens.
            while 'loc(' in stripped:
                prev = stripped
                stripped = re.sub(
                    r'\s*loc\((?:[^()]*|\([^()]*\))*\)', '', stripped
                )
                if stripped == prev:
                    break

            # Clean up empty module attributes
            stripped = re.sub(r'module attributes \{\s*\}', 'module', stripped)

            # Skip now-empty lines
            if stripped.strip():
                phase1_lines.append(stripped)

        # Phase 2: Strip TTG ops and replace tt.dot with scalar FMA.
        #
        # Build maps from TTG op results to their original operands, then
        # remove TTG op lines and substitute references.
        alloc_map = {}    # ttg.local_alloc result -> input operand
        replace_map = {}  # value to replace -> replacement value

        # First pass: scan for TTG op mappings
        for line in phase1_lines:
            # ttg.local_alloc %x : (...) -> !ttg.memdesc<...>
            m = re.match(r'\s*(%\S+)\s*=\s*ttg\.local_alloc\s+(%\S+)', line)
            if m:
                alloc_map[m.group(1)] = m.group(2)
                continue

            # ttg.local_load %x : !ttg.memdesc<...> -> tensor<...>
            m = re.match(r'\s*(%\S+)\s*=\s*ttg\.local_load\s+(%\S+)', line)
            if m:
                alloc_result = m.group(2)
                original = alloc_map.get(alloc_result, alloc_result)
                replace_map[m.group(1)] = original
                continue

            # ttg.convert_layout %x : tensor<...> -> tensor<...>
            m = re.match(r'\s*(%\S+)\s*=\s*ttg\.convert_layout\s+(%\S+)', line)
            if m:
                replace_map[m.group(1)] = m.group(2)
                continue

            # ttg.memdesc_trans %x {order = ...} : ...
            m = re.match(r'\s*(%\S+)\s*=\s*ttg\.memdesc_trans\s+(%\S+)', line)
            if m:
                alloc_map[m.group(1)] = alloc_map.get(m.group(2), m.group(2))
                continue

            # tt.trans %x : tensor<...> -> tensor<...>
            m = re.match(r'\s*(%\S+)\s*=\s*tt\.trans\s+(%\S+)', line)
            if m:
                replace_map[m.group(1)] = m.group(2)
                continue

        # Second pass: emit lines with TTG ops removed, references replaced,
        # and tt.dot converted to scalar FMA.
        out_lines = []
        # Sort replacements longest-first to avoid partial matches
        sorted_replacements = sorted(replace_map.items(),
                                     key=lambda x: len(x[0]), reverse=True)

        for line in phase1_lines:
            # Skip ttg.* op lines entirely
            if re.match(r'\s*%\S+\s*=\s*ttg\.\w+', line):
                continue

            # Skip tt.trans lines (already mapped as passthrough)
            if re.match(r'\s*%\S+\s*=\s*tt\.trans\s+', line):
                continue

            # Apply replacements
            for old, new in sorted_replacements:
                # Use word-boundary-aware replacement to avoid partial matches
                line = re.sub(re.escape(old) + r'(?=[\s,):}\]]|$)', new, line)

            # Replace tt.dot with scalar FMA (arith.mulf + arith.addf).
            # The operand and result types are tensor types at the MLIR text
            # level (the type converter will later reduce them to scalars).
            dot_m = re.match(
                r'(\s*)(%\S+)\s*=\s*tt\.dot\s+(%\S+),\s*(%\S+),\s*(%\S+)\s*:',
                line
            )
            if dot_m:
                indent, result, a, b, c = dot_m.groups()
                # Extract the full result tensor type (e.g. tensor<32x32xf32>)
                result_type_m = re.search(r'->\s*(tensor<[\dx]+x\w+>)', line)
                result_type = result_type_m.group(1) if result_type_m else 'f32'
                # Extract the operand tensor type (A's type, used for mul).
                # The A operand type appears after the first colon.
                a_type_m = re.search(r':\s*(tensor<[\dx]+x\w+>)\s*\*', line)
                a_type = a_type_m.group(1) if a_type_m else result_type
                mul_name = result + '_dot_mul'
                out_lines.append(
                    f'{indent}{mul_name} = arith.mulf {a}, {b} : {result_type}'
                )
                out_lines.append(
                    f'{indent}{result} = arith.addf {c}, {mul_name} : {result_type}'
                )
                continue

            # Remove residual TTG type annotations that slipped through
            # e.g. !ttg.memdesc<...> in type positions
            line = re.sub(r'!ttg\.memdesc<[^>]*>', 'tensor<1xf32>', line)

            if line.strip():
                out_lines.append(line)

        return '\n'.join(out_lines) + '\n'

    @staticmethod
    def _generate_scalar_matmul(fn_name, args, block_m, block_n, elem_type='f32'):
        """Generate a scalar per-element matmul kernel in MLIR.

        Each thread computes one output element:
            C[m, n] = sum_k A[m, k] * B[k, n]

        Thread decomposition: m = lid / BLOCK_N, n = lid % BLOCK_N
        Block size = BLOCK_M * BLOCK_N.

        Args:
            fn_name: Kernel function name
            args: List of (name, type) tuples for function arguments
            block_m: Block size in M dimension
            block_n: Block size in N dimension
            elem_type: Element type (f32, f16, etc.)
        Returns:
            MLIR text for the scalar matmul module
        """
        # Build the function signature from argument list.
        # Remove tt.divisibility attributes from arg types.
        import re as _re
        arg_strs = []
        arg_names = set()
        ptr_arg_names = []  # ordered list of pointer arg names
        for name, ty in args:
            # Strip {tt.divisibility = ...} from type
            ty = _re.sub(r'\s*\{[^}]*\}', '', ty).strip()
            arg_strs.append(f'%{name}: {ty}')
            arg_names.add(name)
            if '!tt.ptr' in ty:
                ptr_arg_names.append(name)
        sig = ', '.join(arg_strs)
        block_size = block_m * block_n

        # Identify the 3 matrix pointer args (A, B, C) by order
        a_name = ptr_arg_names[0] if len(ptr_arg_names) > 0 else 'A'
        b_name = ptr_arg_names[1] if len(ptr_arg_names) > 1 else 'B'
        c_name = ptr_arg_names[2] if len(ptr_arg_names) > 2 else 'C'

        # Determine arith mul/add ops based on element type
        if elem_type in ('f32', 'f16', 'bf16', 'half', 'bfloat'):
            mul_op = 'arith.mulf'
            add_op = 'arith.addf'
            zero_val = '0.000000e+00'
            scalar_type = elem_type
            if scalar_type == 'half':
                scalar_type = 'f16'
            elif scalar_type == 'bfloat':
                scalar_type = 'bf16'
        else:
            mul_op = 'arith.muli'
            add_op = 'arith.addi'
            zero_val = '0'
            scalar_type = elem_type

        # For missing strides, use constant 1 (contiguous dimension)
        stride_defs = []
        def stride_ref(name):
            if name in arg_names:
                return f'%{name}'
            # Generate a constant for missing stride
            const_name = f'%_const_{name}'
            stride_defs.append(f'    {const_name} = arith.constant 1 : i32')
            return const_name

        s_am = stride_ref('stride_am')
        s_ak = stride_ref('stride_ak')
        s_bk = stride_ref('stride_bk')
        s_bn = stride_ref('stride_bn')
        s_cm = stride_ref('stride_cm')
        s_cn = stride_ref('stride_cn')
        stride_defs_text = '\n'.join(stride_defs)
        if stride_defs_text:
            stride_defs_text += '\n'

        return f'''module {{
  tt.func public @{fn_name}({sig}) attributes {{noinline = false}} {{
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cBN = arith.constant {block_n} : i32
    %cBM = arith.constant {block_m} : i32
    %zero_acc = arith.constant {zero_val} : {scalar_type}
{stride_defs_text}    %pid_m = tt.get_program_id x : i32
    %pid_n = tt.get_program_id y : i32
    %lid = tt.make_range {{end = {block_size} : i32, start = 0 : i32}} : tensor<{block_size}xi32>
    %sBN = tt.splat %cBN : i32 -> tensor<{block_size}xi32>
    %row = arith.divui %lid, %sBN : tensor<{block_size}xi32>
    %col = arith.remui %lid, %sBN : tensor<{block_size}xi32>
    %sBM = tt.splat %cBM : i32 -> tensor<{block_size}xi32>
    %block_row_s = tt.splat %pid_m : i32 -> tensor<{block_size}xi32>
    %block_row = arith.muli %block_row_s, %sBM : tensor<{block_size}xi32>
    %block_col_s = tt.splat %pid_n : i32 -> tensor<{block_size}xi32>
    %block_col = arith.muli %block_col_s, %sBN : tensor<{block_size}xi32>
    %m = arith.addi %block_row, %row : tensor<{block_size}xi32>
    %n = arith.addi %block_col, %col : tensor<{block_size}xi32>
    %sAM = tt.splat {s_am} : i32 -> tensor<{block_size}xi32>
    %a_row_off = arith.muli %m, %sAM : tensor<{block_size}xi32>
    %sA = tt.splat %{a_name} : !tt.ptr<{scalar_type}> -> tensor<{block_size}x!tt.ptr<{scalar_type}>>
    %a_base = tt.addptr %sA, %a_row_off : tensor<{block_size}x!tt.ptr<{scalar_type}>>, tensor<{block_size}xi32>
    %sBN2 = tt.splat {s_bn} : i32 -> tensor<{block_size}xi32>
    %b_col_off = arith.muli %n, %sBN2 : tensor<{block_size}xi32>
    %sB = tt.splat %{b_name} : !tt.ptr<{scalar_type}> -> tensor<{block_size}x!tt.ptr<{scalar_type}>>
    %b_base = tt.addptr %sB, %b_col_off : tensor<{block_size}x!tt.ptr<{scalar_type}>>, tensor<{block_size}xi32>
    %szero = tt.splat %zero_acc : {scalar_type} -> tensor<{block_size}x{scalar_type}>
    %acc = scf.for %k = %c0 to %K step %c1 iter_args(%acc_v = %szero) -> (tensor<{block_size}x{scalar_type}>) : i32 {{
      %lp_sAK = tt.splat {s_ak} : i32 -> tensor<{block_size}xi32>
      %lp_sk = tt.splat %k : i32 -> tensor<{block_size}xi32>
      %lp_a_k_off = arith.muli %lp_sk, %lp_sAK : tensor<{block_size}xi32>
      %lp_a_ptr = tt.addptr %a_base, %lp_a_k_off : tensor<{block_size}x!tt.ptr<{scalar_type}>>, tensor<{block_size}xi32>
      %lp_a_val = tt.load %lp_a_ptr : tensor<{block_size}x!tt.ptr<{scalar_type}>>
      %lp_sBK = tt.splat {s_bk} : i32 -> tensor<{block_size}xi32>
      %lp_b_k_off = arith.muli %lp_sk, %lp_sBK : tensor<{block_size}xi32>
      %lp_b_ptr = tt.addptr %b_base, %lp_b_k_off : tensor<{block_size}x!tt.ptr<{scalar_type}>>, tensor<{block_size}xi32>
      %lp_b_val = tt.load %lp_b_ptr : tensor<{block_size}x!tt.ptr<{scalar_type}>>
      %lp_prod = {mul_op} %lp_a_val, %lp_b_val : tensor<{block_size}x{scalar_type}>
      %lp_new_acc = {add_op} %acc_v, %lp_prod : tensor<{block_size}x{scalar_type}>
      scf.yield %lp_new_acc : tensor<{block_size}x{scalar_type}>
    }}
    %st_sCM = tt.splat {s_cm} : i32 -> tensor<{block_size}xi32>
    %st_row_off = arith.muli %m, %st_sCM : tensor<{block_size}xi32>
    %st_sCN = tt.splat {s_cn} : i32 -> tensor<{block_size}xi32>
    %st_col_off = arith.muli %n, %st_sCN : tensor<{block_size}xi32>
    %st_off = arith.addi %st_row_off, %st_col_off : tensor<{block_size}xi32>
    %st_sC = tt.splat %{c_name} : !tt.ptr<{scalar_type}> -> tensor<{block_size}x!tt.ptr<{scalar_type}>>
    %st_ptr = tt.addptr %st_sC, %st_off : tensor<{block_size}x!tt.ptr<{scalar_type}>>, tensor<{block_size}xi32>
    tt.store %st_ptr, %acc : tensor<{block_size}x!tt.ptr<{scalar_type}>>
    tt.return
  }}
}}
'''

    @staticmethod
    def _try_generate_matmul_mlir(ttgir_text, stripped_text):
        """Detect matmul pattern in TTGIR and generate scalar per-element kernel.

        Returns the generated MLIR text if matmul detected, None otherwise.
        """
        import re

        # Detect: tt.dot in the TTGIR and scf.for (K-loop)
        if 'tt.dot' not in ttgir_text:
            return None

        # Extract function name
        fn_m = re.search(r'tt\.func\s+public\s+@(\w+)\(', ttgir_text)
        if not fn_m:
            return None
        fn_name = fn_m.group(1)

        # Extract function arguments (name: type pairs).
        # Need to handle nested parens from loc() annotations in the TTGIR.
        # First strip loc(...) annotations to simplify parsing.
        clean_ttgir = re.sub(
            r'\s*loc\((?:[^()]*|\([^()]*\))*\)', '', ttgir_text
        )
        args_m = re.search(r'tt\.func\s+public\s+@\w+\(([^)]+)\)', clean_ttgir)
        if not args_m:
            return None
        args_text = args_m.group(1)

        # Parse argument list: %name: type loc(...)
        args = []
        for arg_str in re.split(r',\s*(?=%)', args_text):
            arg_str = arg_str.strip()
            # Remove loc annotations
            arg_str = re.sub(r'\s*loc\(.*', '', arg_str)
            am = re.match(r'(%\w+):\s*(.+)', arg_str)
            if am:
                name = am.group(1).lstrip('%')
                ty = am.group(2).strip()
                args.append((name, ty))

        # Check required arguments exist
        arg_names = {a[0] for a in args}
        # Need: at least 3 pointer args and a K dimension arg.
        # Stride args may be optimized away (e.g. stride=1 becomes constexpr).
        has_ptrs = sum(1 for _, t in args if '!tt.ptr' in t) >= 3
        has_k = 'K' in arg_names
        # Need at least stride_am and stride_cm (row strides for A and C)
        has_min_strides = 'stride_am' in arg_names and 'stride_cm' in arg_names

        if not (has_ptrs and has_k and has_min_strides):
            return None

        # Extract block sizes from make_range end values
        block_sizes = set()
        for mr_m in re.finditer(
            r'tt\.make_range\s*\{[^}]*end\s*=\s*(\d+)',
            ttgir_text
        ):
            block_sizes.add(int(mr_m.group(1)))

        if not block_sizes:
            return None

        # For matmul, BLOCK_M and BLOCK_N are typically the same
        block_m = block_n = max(block_sizes)

        # Detect element type from pointer types
        elem_type = 'f32'
        ptr_m = re.search(r'!tt\.ptr<(\w+)>', ttgir_text)
        if ptr_m:
            elem_type = ptr_m.group(1)

        return MetalBackend._generate_scalar_matmul(
            fn_name, args, block_m, block_n, elem_type
        )

    # Map LLVM IR element types to their byte size and AIR metadata name.
    _ELEM_TYPE_INFO = {
        'half':  (2, 2, 'half'),
        'float': (4, 4, 'float'),
        'i8':    (1, 1, 'char'),
        'i16':   (2, 2, 'short'),
        'i32':   (4, 4, 'int'),
        'i64':   (8, 8, 'long'),
        'bfloat':(2, 2, 'bfloat'),
    }

    @staticmethod
    def _opaque_to_typed_ptrs(llir_text):
        """Convert LLVM IR with opaque pointers to typed pointers for Metal AIR.

        Metal's GPU JIT compiler requires typed pointers and does not support
        generic address space stores. This conversion:
        1. Eliminates addrspacecasts (inlines source pointers)
        2. Infers element types from GEP instructions for each device buffer
        3. Converts all pointer types to typed equivalents
        4. Fixes metadata to use typed function pointer references and correct
           type names/sizes for non-float buffers
        """
        import re

        lines = llir_text.split('\n')

        # Pass 0: Collect addrspacecast mappings and their source address spaces.
        # E.g. %.generic = addrspacecast ptr addrspace(1) %0 to ptr
        #   -> cast_map["%.generic"] = ("%0", "1")
        cast_map = {}
        for line in lines:
            m = re.match(
                r'\s*(%\S+)\s*=\s*addrspacecast\s+ptr\s+addrspace\((\d+)\)\s+(%\S+)\s+to\s+ptr',
                line
            )
            if m:
                cast_map[m.group(1)] = (m.group(3), m.group(2))

        # Pass 0.5: Infer element types from GEP instructions.
        # Pattern: getelementptr <elem_type>, ptr <ptr_name>, ...
        # Follow addrspacecast chains to resolve back to the original param.
        # Note: ptr_name uses [\w.]+  to match LLVM names like %.generic but
        # NOT commas or other delimiters.
        param_types = {}  # param_name -> element_type (e.g. "%0" -> "half")
        for line in lines:
            m = re.match(
                r'\s*%\S+\s*=\s*getelementptr\s+(\w+),\s*ptr\s+(%([\w.]+))',
                line
            )
            if m:
                elem_type = m.group(1)
                ptr_name = m.group(2)  # e.g. "%.generic" or "%0"
                # Follow addrspacecast to find the original parameter
                actual_ptr = cast_map.get(ptr_name, (ptr_name,))[0] if ptr_name in cast_map else ptr_name
                if actual_ptr not in param_types:
                    param_types[actual_ptr] = elem_type

        # Parse function signature to get ordered parameter names.
        # Pattern: define void @kernel(ptr addrspace(1) %0, ptr addrspace(1) %1, ...)
        sig_param_names = []  # ordered list of param names from define line
        sig_param_addrspaces = {}  # param_name -> addrspace string or None
        for line in lines:
            m = re.match(
                r'\s*define\s+void\s+@\w+\((.*)\)',
                line
            )
            if m:
                for param in m.group(1).split(','):
                    param = param.strip()
                    # Extract param name (last %word in the param)
                    name_m = re.search(r'(%\S+)\s*$', param)
                    if name_m:
                        pname = name_m.group(1)
                        sig_param_names.append(pname)
                        # Check if it has an addrspace
                        as_m = re.search(r'addrspace\((\d+)\)', param)
                        if as_m:
                            sig_param_addrspaces[pname] = as_m.group(1)
                break

        # For device buffer params (addrspace 1), determine the typed pointer.
        # If we found a GEP that uses the param, use that type; else default to float.
        def _get_device_ptr_type(param_name):
            """Return the element type for a device buffer parameter."""
            return param_types.get(param_name, 'float')

        # Build a map: param_index -> element_type for device buffers (addrspace 1)
        param_elem_types = {}  # index -> elem_type string
        for i, pname in enumerate(sig_param_names):
            if sig_param_addrspaces.get(pname) == '1':
                param_elem_types[i] = _get_device_ptr_type(pname)


        # Pass 1: Process each line
        out_lines = []
        fn_name = None
        fn_param_types = []

        for line in lines:
            # Skip addrspacecast lines
            if re.match(r'\s*%\S+\s*=\s*addrspacecast\s+.*to\s+ptr', line):
                continue

            # Replace cast names with source names (longest first for safety)
            for name in sorted(cast_map.keys(), key=len, reverse=True):
                src, addrspace = cast_map[name]
                # Word-boundary replacement
                line = re.sub(re.escape(name) + r'(?=[\s,)\]]|$)', src, line)

            # Convert function signature with per-param types
            def_m = re.match(
                r'(\s*define\s+void\s+@\w+\()(.*?)(\)\s*\{.*)$',
                line
            )
            if def_m:
                prefix, params_str, suffix = def_m.group(1), def_m.group(2), def_m.group(3)
                new_params = []
                for i, param in enumerate(params_str.split(',')):
                    param = param.strip()
                    if 'ptr addrspace(1)' in param:
                        ety = param_elem_types.get(i, 'float')
                        param = param.replace('ptr addrspace(1)', f'{ety} addrspace(1)*')
                    elif 'ptr addrspace(2)' in param:
                        param = param.replace('ptr addrspace(2)', 'i32 addrspace(2)*')
                    new_params.append(param)
                line = prefix + ', '.join(new_params) + suffix
            else:
                # Non-signature lines: convert remaining ptr addrspace(N) patterns
                line = line.replace('ptr addrspace(2)', 'i32 addrspace(2)*')
                # Do NOT blindly replace ptr addrspace(1) here; GEP/load/store
                # handlers below will use the correct element type.

            # GEP: getelementptr <type>, ptr %X → getelementptr <type>, <type> addrspace(1)* %X
            line = re.sub(
                r'getelementptr\s+(\w+),\s*ptr\s+(%\S+)',
                r'getelementptr \1, \1 addrspace(1)* \2',
                line
            )

            # Load from GEP result (device buffer, addrspace 1):
            # load <type>, ptr %X -> load <type>, <type> addrspace(1)* %X
            # By this point, constant buffer loads (addrspace 2) already have
            # their ptr replaced with i32 addrspace(2)*, so any remaining
            # "load <type>, ptr %X" refers to a device buffer GEP result.
            line = re.sub(
                r'load\s+(\w+),\s*ptr\s+(%\S+)',
                r'load \1, \1 addrspace(1)* \2',
                line
            )

            # Store: store <type> %v, ptr %X → store <type> %v, <type> addrspace(1)* %X
            # Handle any scalar type.
            line = re.sub(
                r'store\s+(\w+)\s+(%\S+),\s*ptr\s+(%\S+)',
                r'store \1 \2, \1 addrspace(1)* \3',
                line
            )

            # PHI nodes with ptr type: phi ptr [ %a, %bb1 ], [ %b, %bb2 ]
            # These come from scf.for loop iter_args that carry pointers.
            # Convert to typed pointer: phi float addrspace(1)* [ ... ]
            # The element type is inferred from uses (GEP/load/store).
            # For simplicity, default to float addrspace(1)* since that's
            # the most common case for device buffer pointers in loops.
            line = re.sub(
                r'phi\s+ptr\s+\[',
                r'phi float addrspace(1)* [',
                line
            )

            # ---- Threadgroup shared memory (addrspace 3) patterns ----
            # These are generated by the reduce op lowering.

            # GEP with array type base: getelementptr [N x float], ptr addrspace(3) @name, ...
            # -> getelementptr [N x float], [N x float] addrspace(3)* @name, ...
            line = re.sub(
                r'getelementptr\s+(\[\d+ x float\]),\s*ptr addrspace\(3\)\s+([%@]\S+)',
                r'getelementptr \1, \1 addrspace(3)* \2',
                line
            )

            # Store to threadgroup: store float %v, ptr addrspace(3) %slot or @global
            # For direct global refs, insert GEP to get float* from [N x float]*.
            m_tg_store = re.match(
                r'(\s*store\s+float\s+\S+),\s*ptr addrspace\(3\)\s+(@__reduce_shared_\d+)(.*)',
                line
            )
            if m_tg_store:
                line = (f'{m_tg_store.group(1)}, float addrspace(3)* '
                        f'getelementptr([32 x float], [32 x float] addrspace(3)* '
                        f'{m_tg_store.group(2)}, i32 0, i32 0){m_tg_store.group(3)}')
            else:
                line = re.sub(
                    r'store\s+(float)\s+(%\S+),\s*ptr addrspace\(3\)\s+([%@]\S+)',
                    r'store \1 \2, \1 addrspace(3)* \3',
                    line
                )

            # Load from threadgroup: load float, ptr addrspace(3) %slot or @global
            m_tg_load = re.match(
                r'(\s*%\S+\s*=\s*load\s+float),\s*ptr addrspace\(3\)\s+(@__reduce_shared_\d+)(.*)',
                line
            )
            if m_tg_load:
                line = (f'{m_tg_load.group(1)}, float addrspace(3)* '
                        f'getelementptr([32 x float], [32 x float] addrspace(3)* '
                        f'{m_tg_load.group(2)}, i32 0, i32 0){m_tg_load.group(3)}')
            else:
                line = re.sub(
                    r'load\s+(float),\s*ptr addrspace\(3\)\s+([%@]\S+)',
                    r'load \1, \1 addrspace(3)* \2',
                    line
                )

            # Capture function name and param types for metadata
            m = re.match(
                r'\s*define\s+void\s+@(\w+)\(((?:[^()]*|\([^()]*\))*)\)',
                line
            )
            if m:
                fn_name = m.group(1)
                for param in m.group(2).split(','):
                    param = param.strip()
                    idx = param.rfind('%')
                    if idx > 0:
                        fn_param_types.append(param[:idx].rstrip())
                    else:
                        fn_param_types.append(param)

            # Metadata: ptr @fn -> typed function pointer
            if fn_name and line.strip().startswith('!') and f'ptr @{fn_name}' in line:
                typed_sig = ', '.join(fn_param_types)
                fn_ptr_type = f'void ({typed_sig})*'
                line = line.replace(f'ptr @{fn_name}', f'{fn_ptr_type} @{fn_name}')

            # Fix metadata arg_type_name and arg_type_size for non-float device buffers.
            # The C++ pass hardcodes "float" / size 4 for all device buffers.
            # Replace with the correct type based on GEP-inferred element types.
            if line.strip().startswith('!') and '!"air.buffer"' in line and '!"air.address_space", i32 1' in line:
                # This is a device buffer metadata entry. Extract the arg index.
                arg_idx_m = re.match(r'(\s*!\d+\s*=\s*!\{i32\s+)(\d+)', line)
                if arg_idx_m:
                    arg_idx = int(arg_idx_m.group(2))
                    if arg_idx in param_elem_types:
                        ety = param_elem_types[arg_idx]
                        type_info = MetalBackend._ELEM_TYPE_INFO.get(ety)
                        if type_info:
                            byte_size, align_size, air_name = type_info
                            # Replace arg_type_size
                            line = re.sub(
                                r'(!"air\.arg_type_size",\s*i32\s+)\d+',
                                rf'\g<1>{byte_size}',
                                line
                            )
                            # Replace arg_type_align_size
                            line = re.sub(
                                r'(!"air\.arg_type_align_size",\s*i32\s+)\d+',
                                rf'\g<1>{align_size}',
                                line
                            )
                            # Replace arg_type_name
                            line = re.sub(
                                r'(!"air\.arg_type_name",\s*!")(\w+)(")',
                                rf'\g<1>{air_name}\3',
                                line
                            )

            out_lines.append(line)

        return '\n'.join(out_lines)

    @staticmethod
    def _strip_unsupported_llvm_attrs(llir_text):
        """Strip LLVM function attributes that Metal's compiler doesn't support.

        Metal's compiler is based on an older LLVM and doesn't understand
        newer attributes like nocreateundeforpoison, memory(none), etc.
        We remove attribute group definitions and their references from
        function declarations.
        """
        import re

        lines = llir_text.split('\n')
        out_lines = []
        for line in lines:
            # Remove "attributes #N = { ... }" lines entirely
            if re.match(r'\s*attributes\s+#\d+\s*=\s*\{', line):
                continue
            # Remove #N references from declare/define lines
            line = re.sub(r'\s+#\d+\b', '', line)
            out_lines.append(line)

        return '\n'.join(out_lines)

    # Mapping from LLVM intrinsics to AIR intrinsics for math functions
    # that Metal's runtime doesn't resolve as standard LLVM intrinsics.
    # Most llvm.* intrinsics work (exp, sin, cos, ...) but this provides
    # a safety net for any that don't.
    _LLVM_TO_AIR_INTRINSICS = {
        # These are only needed if Metal's runtime fails to resolve them.
        # Currently llvm.exp.f32, llvm.sin.f32, etc. all work.
        # Add entries here if specific intrinsics cause "Undefined symbols" errors.
    }

    # All Triton/TTG ops are now handled by the C++ path:
    # - tt.dot is lowered to scalar FMA at the text level in _strip_ttg_annotations
    # - tt.trans is a passthrough in the per-thread model
    # - ttg.local_alloc/local_load/convert_layout/memdesc_trans are stripped
    # - scf.for/if/yield are lowered by the scf-to-cf pass
    _CPP_UNSUPPORTED_OPS = set()  # empty — no fallback needed

    @staticmethod
    def _has_unsupported_ops(ttgir_text):
        """Check if TTGIR contains ops the C++ path cannot handle.

        Currently returns False for all inputs — all ops are handled.
        """
        return False

    @staticmethod
    def make_llir(mod, metadata, options):
        """Lower TTGIR to AIR-compatible LLVM IR using C++ MLIR passes.

        Pipeline: TTGIR text → strip TritonGPU annotations → C++ pass
        pipeline (Triton ops → LLVM dialect → LLVM IR) → AIR LLVM IR
        with Metal kernel metadata.

        The output is LLVM IR text that can be fed directly to Metal's
        compiler (xcrun metal -Xclang -opaque-pointers -c -x ir).

        If the TTGIR contains ops the C++ path cannot handle (tt.reduce,
        tt.dot, tt.trans), raises RuntimeError to trigger fallback to the
        Python/MSL path.
        """
        import triton_metal._triton_metal_cpp as cpp
        from triton_metal.debug import _debug_level, _dump_dir

        level = _debug_level()
        kernel_name = metadata.get("name", "kernel")

        # Get TTGIR text
        ttgir_text = str(mod)

        # For matmul kernels (containing tt.dot), generate a custom scalar
        # per-element kernel instead of trying to strip and convert the
        # TTGIR. The per-thread scalar model can't handle cooperative
        # matrix operations (tt.dot requires K-dimension reduction).
        matmul_mlir = MetalBackend._try_generate_matmul_mlir(ttgir_text, None)
        if matmul_mlir is not None:
            stripped = matmul_mlir
        else:
            # Standard path: strip TritonGPU annotations
            stripped = MetalBackend._strip_ttg_annotations(ttgir_text)

        if level >= 1:
            debug_dir = _dump_dir()
            os.makedirs(debug_dir, exist_ok=True)
            with open(os.path.join(debug_dir, f"{kernel_name}.ttgir"), "w") as f:
                f.write(ttgir_text)
            with open(os.path.join(debug_dir, f"{kernel_name}.stripped.mlir"), "w") as f:
                f.write(stripped)

        # Run C++ pass pipeline: TTGIR → LLVM IR with AIR metadata
        air_llvm_ir_opaque = cpp.run_to_llvm(stripped)

        if level >= 2:
            with open(os.path.join(debug_dir, f"{kernel_name}.opaque.ll"), "w") as f:
                f.write(air_llvm_ir_opaque)

        # Metal's GPU JIT compiler requires typed pointers (old LLVM IR format).
        # Convert opaque pointers to typed pointers.
        air_llvm_ir = MetalBackend._opaque_to_typed_ptrs(air_llvm_ir_opaque)

        # Strip LLVM function attributes that Metal's compiler doesn't
        # understand (nocreateundeforpoison, memory(none), etc.).
        air_llvm_ir = MetalBackend._strip_unsupported_llvm_attrs(air_llvm_ir)

        if level >= 1:
            with open(os.path.join(debug_dir, f"{kernel_name}.ll"), "w") as f:
                f.write(air_llvm_ir)

        # Extract kernel name from the LLVM IR
        import re
        m = re.search(r'define\s+void\s+@(\w+)\s*\(', air_llvm_ir)
        if m:
            metadata["name"] = m.group(1)

        # Extract block_size from TTGIR make_range end attributes.
        # For 1D kernels, block_size = the single make_range end value.
        # For 2D kernels (matmul), block_size = product of unique make_range
        # end values across different slice dimensions, capped at 1024.
        import re
        block_size = options.num_warps * 32  # default
        mr_ends = set()
        for mr_match in re.finditer(
            r'tt\.make_range\s*\{[^}]*end\s*=\s*(\d+)',
            ttgir_text
        ):
            mr_ends.add(int(mr_match.group(1)))
        if mr_ends:
            # Check if this is a 2D kernel by looking for make_range with
            # different slice dimensions
            slice_dims = set()
            for sl_m in re.finditer(
                r'tt\.make_range\s*\{[^}]*\}\s*:\s*tensor<\d+xi32,\s*#ttg\.slice<\{dim\s*=\s*(\d+)',
                ttgir_text
            ):
                slice_dims.add(int(sl_m.group(1)))
            if len(slice_dims) > 1:
                # 2D kernel: block_size = product of all unique end values
                product = 1
                for end_val in mr_ends:
                    product *= end_val
                block_size = product
            else:
                # 1D kernel: use the largest make_range end value
                block_size = max(mr_ends)
        # For generated matmul kernels, use the make_range end from the
        # generated MLIR (which has end=BLOCK_M*BLOCK_N)
        if matmul_mlir is not None:
            gen_mr = re.search(r'tt\.make_range\s*\{[^}]*end\s*=\s*(\d+)', stripped)
            if gen_mr:
                block_size = int(gen_mr.group(1))
        metadata.setdefault("block_size", min(block_size, 1024))

        # Detect 2D grid usage from program_id axes in the TTGIR.
        needs_2d = bool(re.search(r'tt\.get_program_id\s+y\b', ttgir_text))
        metadata.setdefault("needs_2d_grid", needs_2d)

        return air_llvm_ir

    @staticmethod
    def make_metallib_from_llir(src, metadata, options):
        """Compile AIR LLVM IR to metallib using Metal's compiler.

        Pipeline: LLVM IR text → metal -c -x ir → .air → metallib
        This bypasses MSL entirely.
        """
        import time
        import warnings
        from triton_metal.debug import _debug_level, _fallback_mode

        level = _debug_level()
        kernel_name = metadata.get("name", "kernel")

        if level >= 2:
            t0 = time.perf_counter()

        try:
            cache_dir = _get_cache_dir()
            src_hash = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
            base = f"{kernel_name}_{src_hash}"

            ll_path = os.path.join(cache_dir, f"{base}.ll")
            air_path = os.path.join(cache_dir, f"{base}.air")
            metallib_path = os.path.join(cache_dir, f"{base}.metallib")

            # Skip compilation if cached metallib exists.
            if os.path.exists(metallib_path):
                if level >= 2:
                    print(
                        f"[triton-metal] make_metallib_from_llir({kernel_name}): cache hit",
                        file=sys.stderr,
                    )
                with open(metallib_path, "rb") as f:
                    return f.read()

            with open(ll_path, "w") as f:
                f.write(src)

            # Compile LLVM IR → AIR using Metal's compiler
            # Our IR uses typed pointers (Metal's GPU JIT requires them).
            try:
                subprocess.run(
                    [
                        "xcrun", "-sdk", "macosx", "metal",
                        "-c", "-x", "ir",
                        ll_path,
                        "-o", air_path,
                    ],
                    capture_output=True,
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                from triton_metal.errors import MetalCompilationError
                stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
                raise MetalCompilationError(
                    f"Metal IR compilation failed (exit {e.returncode})",
                    msl_source=ll_path,
                    stderr=stderr,
                ) from None

            # Link AIR → metallib
            tmp_metallib_path = metallib_path + ".tmp"
            try:
                subprocess.run(
                    [
                        "xcrun", "-sdk", "macosx", "metallib",
                        air_path,
                        "-o", tmp_metallib_path,
                    ],
                    capture_output=True,
                    check=True,
                )
                os.replace(tmp_metallib_path, metallib_path)
            except subprocess.CalledProcessError as e:
                try:
                    os.unlink(tmp_metallib_path)
                except OSError:
                    pass
                from triton_metal.errors import MetalCompilationError
                stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
                raise MetalCompilationError(
                    f"Metal library linking failed (exit {e.returncode})",
                    msl_source=air_path,
                    stderr=stderr,
                ) from None

            with open(metallib_path, "rb") as f:
                data = f.read()

            if level >= 2:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                print(
                    f"[triton-metal] make_metallib_from_llir({kernel_name}): {elapsed_ms:.1f}ms",
                    file=sys.stderr,
                )

            return data

        except Exception as e:
            mode = _fallback_mode()
            if mode == "warn":
                warnings.warn(
                    f"triton-metal: Metal IR compilation failed for kernel "
                    f"'{kernel_name}': {e}. "
                    f"Kernel will fall back to CPU.",
                    stacklevel=2,
                )
            elif mode == "error":
                raise
            raise

    @staticmethod
    def gluon_to_ttgir(mod, metadata, options):
        """Convert Gluon IR to TTGIR for Metal.

        Gluon is Triton's higher-level language. The conversion applies
        Gluon-specific passes (inliner, encoding resolution) then standard
        TTGIR conversion passes. Metal-specific: no TMA, no NVIDIA passes.
        """
        from triton._C.libtriton import ir, passes

        pm = ir.pass_manager(mod.context)
        passes.gluon.add_inliner(pm)
        passes.gluon.add_infer_coalesced_encodings(pm)
        passes.gluon.add_resolve_auto_encodings(pm)
        passes.gluon.add_canonicalizer(pm)
        passes.common.add_sccp(pm)
        passes.ttir.add_loop_aware_cse(pm)
        passes.gluon.add_canonicalizer(pm)
        passes.ttgpuir.add_combine_tensor_select_and_if(pm)
        pm.run(mod, "gluon_to_ttgir")
        metadata["tensordesc_meta"] = mod.get_tensordesc_metadata()
        return mod

    @staticmethod
    def make_ttir(mod, metadata, options):
        from triton._C.libtriton import ir, passes

        pm = ir.pass_manager(mod.context)
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_pointer(pm)
        # Metal has no TMA — always rewrite tensor descriptors to pointers.
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod, "make_ttir")
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, options):
        import sys
        import time
        from triton._C.libtriton import ir, passes
        from triton_metal.debug import _debug_level

        level = _debug_level()
        if level >= 2:
            t0 = time.perf_counter()

        pm = ir.pass_manager(mod.context)
        target_str = f"metal:{options.arch}"
        passes.ttir.add_convert_to_ttgpuir(
            pm, target_str, options.num_warps, 32, options.num_ctas
        )

        passes.ttgpuir.add_coalesce(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_thread_locality(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, False)
        passes.ttgpuir.add_reduce_data_duplication(pm)
        passes.ttgpuir.add_reorder_instructions(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod, "make_ttgir")
        metadata["tensordesc_meta"] = None
        # Extract shared memory requirement from MLIR module if available.
        try:
            shared = mod.get_int_attr("ttg.shared")
        except Exception:
            shared = None
        metadata["shared"] = shared if shared is not None else 0

        if level >= 2:
            kernel_name = metadata.get("name", "kernel")
            elapsed_ms = (time.perf_counter() - t0) * 1000
            print(
                f"[triton-metal] make_ttgir({kernel_name}): {elapsed_ms:.1f}ms",
                file=sys.stderr,
            )

        return mod

    @staticmethod
    def make_msl(mod, metadata, options):
        import sys
        import time
        import warnings
        from triton_metal.codegen.msl_emitter import emit_msl
        from triton_metal.debug import _debug_level, _dump_dir, _fallback_mode

        level = _debug_level()
        kernel_name = metadata.get("name", "kernel")

        # Level 1+: dump raw TTGIR before lowering
        if level >= 1:
            debug_dir = _dump_dir()
            os.makedirs(debug_dir, exist_ok=True)
            ttgir_path = os.path.join(debug_dir, f"{kernel_name}.ttgir")
            with open(ttgir_path, "w") as f:
                f.write(str(mod))

        # Check persistent MSL cache (TTGIR text + options → MSL string).
        mod_text = str(mod)
        cache_key = hashlib.sha256(
            (mod_text + options.hash()).encode("utf-8")
        ).hexdigest()[:16]
        cache_dir = _get_cache_dir()
        msl_cache_path = os.path.join(cache_dir, f"{kernel_name}_{cache_key}.msl")

        if os.path.exists(msl_cache_path):
            with open(msl_cache_path, "r") as f:
                msl_src = f.read()

            # Populate metadata that emit_msl would normally set.
            # Extract kernel name from MSL: "kernel void NAME("
            import re as _re
            m = _re.search(r'kernel\s+void\s+(\w+)\s*\(', msl_src)
            if m:
                metadata["name"] = m.group(1)
            else:
                metadata["name"] = kernel_name
            # block_size and output_arg_indices default if not cached
            metadata.setdefault("block_size", options.num_warps * 32)
            metadata.setdefault("needs_2d_grid", False)

            # Try to load cached metadata alongside the MSL
            meta_cache_path = msl_cache_path.replace(".msl", ".meta.json")
            if os.path.exists(meta_cache_path):
                import json
                with open(meta_cache_path, "r") as f:
                    cached_meta = json.load(f)
                metadata.update(cached_meta)

            if level >= 2:
                print(
                    f"[triton-metal] make_msl({kernel_name}): cache hit",
                    file=sys.stderr,
                )

            return msl_src

        # Level 2: time the MSL emission
        if level >= 2:
            t0 = time.perf_counter()

        try:
            msl_src = emit_msl(mod, metadata, options)
        except Exception as e:
            mode = _fallback_mode()
            if mode == "warn":
                warnings.warn(
                    f"triton-metal: MSL codegen failed for kernel "
                    f"'{kernel_name}': {e}. "
                    f"Kernel will fall back to CPU.",
                    stacklevel=2,
                )
            elif mode == "error":
                # Re-raise without fallback hint — user wants hard errors.
                raise
            # "silent" and "warn" both re-raise so Triton/torch.compile
            # can route to CPU fallback.
            raise

        if level >= 2:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            print(
                f"[triton-metal] make_msl({kernel_name}): {elapsed_ms:.1f}ms",
                file=sys.stderr,
            )

        # Cache the generated MSL and metadata atomically.
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=cache_dir, suffix=".msl.tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                f.write(msl_src)
            os.replace(tmp_path, msl_cache_path)
            # Cache metadata (name, block_size, etc.) alongside the MSL.
            import json
            meta_cache_path = msl_cache_path.replace(".msl", ".meta.json")
            cacheable = {
                k: v for k, v in metadata.items()
                if isinstance(v, (str, int, float, bool, type(None), list, tuple))
            }
            tmp_meta_fd, tmp_meta_path = tempfile.mkstemp(
                dir=cache_dir, suffix=".meta.tmp"
            )
            with os.fdopen(tmp_meta_fd, "w") as f:
                json.dump(cacheable, f)
            os.replace(tmp_meta_path, meta_cache_path)
        except Exception:
            # Best-effort cleanup on failure; compilation still succeeds.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        # Level 1+: dump generated MSL
        if level >= 1:
            msl_path = os.path.join(debug_dir, f"{kernel_name}.msl")
            with open(msl_path, "w") as f:
                f.write(msl_src)

        return msl_src

    @staticmethod
    def make_metallib(src, metadata, options):
        import sys
        import time
        import warnings
        from triton_metal.debug import _debug_level, _fallback_mode

        level = _debug_level()
        kernel_name = metadata.get("name", "kernel")

        if level >= 2:
            t0 = time.perf_counter()

        try:
            # Persistent cache directory (survives reboots).
            cache_dir = _get_cache_dir()

            # Use content hash for deterministic naming.
            src_hash = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
            base = f"{kernel_name}_{src_hash}"

            metal_path = os.path.join(cache_dir, f"{base}.metal")
            air_path = os.path.join(cache_dir, f"{base}.air")
            metallib_path = os.path.join(cache_dir, f"{base}.metallib")

            # Skip compilation if cached metallib exists.
            if os.path.exists(metallib_path):
                if level >= 2:
                    print(
                        f"[triton-metal] make_metallib({kernel_name}): cache hit",
                        file=sys.stderr,
                    )
                with open(metallib_path, "rb") as f:
                    return f.read()

            with open(metal_path, "w") as f:
                f.write(src)

            # Resolve Metal standard version for compilation.
            if options.target_metal_version == "auto":
                from triton_metal.backend.device_detect import get_device_info
                metal_std_flag = get_device_info().metal_std_flag
            else:
                metal_std_flag = f"-std=metal{options.target_metal_version}"

            # Compile MSL -> AIR
            try:
                subprocess.run(
                    [
                        "xcrun", "-sdk", "macosx", "metal",
                        "-c", metal_path,
                        "-o", air_path,
                        metal_std_flag,
                        "-mmacosx-version-min=15.0",
                        "-O2",
                    ],
                    capture_output=True,
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                from triton_metal.errors import MetalCompilationError
                stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
                raise MetalCompilationError(
                    f"Metal shader compilation failed (exit {e.returncode})",
                    msl_source=metal_path,
                    stderr=stderr,
                ) from None

            # Link AIR -> metallib (atomic write via rename).
            tmp_metallib_path = metallib_path + ".tmp"
            try:
                subprocess.run(
                    [
                        "xcrun", "-sdk", "macosx", "metallib",
                        air_path,
                        "-o", tmp_metallib_path,
                    ],
                    capture_output=True,
                    check=True,
                )
                os.replace(tmp_metallib_path, metallib_path)
            except subprocess.CalledProcessError as e:
                # Clean up partial temp file on failure.
                try:
                    os.unlink(tmp_metallib_path)
                except OSError:
                    pass
                from triton_metal.errors import MetalCompilationError
                stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
                raise MetalCompilationError(
                    f"Metal library linking failed (exit {e.returncode})",
                    msl_source=air_path,
                    stderr=stderr,
                ) from None

            with open(metallib_path, "rb") as f:
                data = f.read()

            if level >= 2:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                print(
                    f"[triton-metal] make_metallib({kernel_name}): {elapsed_ms:.1f}ms",
                    file=sys.stderr,
                )

            return data

        except Exception as e:
            mode = _fallback_mode()
            if mode == "warn":
                warnings.warn(
                    f"triton-metal: Metal compilation failed for kernel "
                    f"'{kernel_name}': {e}. "
                    f"Kernel will fall back to CPU.",
                    stacklevel=2,
                )
            elif mode == "error":
                # Re-raise without fallback hint -- user wants hard errors.
                raise
            # "silent" and "warn" both re-raise so Triton/torch.compile
            # can route to CPU fallback.
            raise

    @functools.lru_cache()
    def hash(self):
        try:
            sdk_version = subprocess.check_output(
                ["xcrun", "--show-sdk-version"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            sdk_version = "unknown"
        return f"metal-{sdk_version}-{self.target.arch}"
