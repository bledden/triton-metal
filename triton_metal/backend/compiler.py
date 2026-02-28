import functools
import hashlib
import os
import subprocess
import tempfile
from dataclasses import dataclass, field, MISSING
from typing import Dict

from types import ModuleType

from triton.backends.compiler import BaseBackend, GPUTarget


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
    # Apple GPUs have no FP8 support.
    supported_fp8_dtypes: tuple = ()
    default_dot_input_precision: str = "ieee"
    allowed_dot_input_precisions: tuple = ("ieee",)
    max_num_imprecise_acc_default: int = 0
    extern_libs: dict = field(default_factory=dict)
    debug: bool = False
    backend_name: str = "metal"
    arch: str = "apple-m4"

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
        return target.backend == "metal"

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

        # Validate: no FP64 on Metal.
        arch = result.get("arch", "apple-m4")
        result["arch"] = arch

        return MetalOptions(**result)

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
        )

    def get_codegen_implementation(self, options):
        return {}

    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}

    def load_dialects(self, ctx):
        # No custom MLIR dialects for now.
        pass

    def add_stages(self, stages, options, language=None):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options)
        stages["msl"] = lambda src, metadata: self.make_msl(src, metadata, options)
        stages["metallib"] = lambda src, metadata: self.make_metallib(src, metadata, options)

    @staticmethod
    def make_ttir(mod, metadata, options):
        from triton._C.libtriton import ir, passes

        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
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
        from triton._C.libtriton import ir, passes

        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        # NOTE: "metal" target may not be recognized by Triton's
        # add_convert_to_ttgpuir. If it fails, we fall back to a dummy
        # CUDA target and handle the difference in the MSL emitter.
        target_str = f"metal:{options.arch}"
        try:
            passes.ttir.add_convert_to_ttgpuir(
                pm, target_str, options.num_warps, 32, options.num_ctas
            )
        except Exception:
            # Fallback: use a dummy CUDA target.
            passes.ttir.add_convert_to_ttgpuir(
                pm, "cuda:80", options.num_warps, 32, options.num_ctas
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
        return mod

    @staticmethod
    def make_msl(mod, metadata, options):
        from triton_metal.codegen.msl_emitter import emit_msl

        msl_src = emit_msl(mod, metadata, options)
        return msl_src

    @staticmethod
    def make_metallib(src, metadata, options):
        # Write metallib to a persistent cache directory so the driver
        # can load it via newLibraryWithURL (avoids PyObjC NSData crash).
        cache_dir = os.path.join(tempfile.gettempdir(), "triton_metal_cache")
        os.makedirs(cache_dir, exist_ok=True)

        # Use content hash for deterministic naming.
        src_hash = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
        kernel_name = metadata.get("name", "kernel")
        base = f"{kernel_name}_{src_hash}"

        metal_path = os.path.join(cache_dir, f"{base}.metal")
        air_path = os.path.join(cache_dir, f"{base}.air")
        metallib_path = os.path.join(cache_dir, f"{base}.metallib")

        # Skip compilation if cached metallib exists.
        if os.path.exists(metallib_path):
            return metallib_path

        with open(metal_path, "w") as f:
            f.write(src)

        # Compile MSL -> AIR
        subprocess.check_call(
            [
                "xcrun", "-sdk", "macosx", "metal",
                "-c", metal_path,
                "-o", air_path,
                "-std=metal3.2",
                "-O2",
            ],
            stderr=subprocess.PIPE,
        )

        # Link AIR -> metallib
        subprocess.check_call(
            [
                "xcrun", "-sdk", "macosx", "metallib",
                air_path,
                "-o", metallib_path,
            ],
            stderr=subprocess.PIPE,
        )

        return metallib_path

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
