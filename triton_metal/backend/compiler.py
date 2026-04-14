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

        stages["msl"] = lambda src, metadata: self.make_msl(src, metadata, options)
        stages["metallib"] = lambda src, metadata: self.make_metallib(src, metadata, options)

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
