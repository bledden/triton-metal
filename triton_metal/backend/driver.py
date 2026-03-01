import platform
import struct
import tempfile

from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase


def ty_to_cpp(ty):
    """Map Triton type strings to C++ type strings for Metal."""
    if ty[0] == "*":
        # Metal uses raw device pointers.
        return "uint64_t"
    return {
        "i1": "int8_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint8_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
    }[ty]


class MetalUtils:
    """Manages Metal device, command queue, and kernel dispatch."""

    def __init__(self):
        self._device = None
        self._command_queue = None

    @property
    def device(self):
        if self._device is None:
            import Metal

            self._device = Metal.MTLCreateSystemDefaultDevice()
            if self._device is None:
                raise RuntimeError("No Metal GPU device found")
        return self._device

    @property
    def command_queue(self):
        if self._command_queue is None:
            self._command_queue = self.device.newCommandQueue()
        return self._command_queue

    def load_binary(self, name, kernel, shared_mem, device=None):
        """Load a metallib and create a compute pipeline state.

        Uses newLibraryWithURL instead of newLibraryWithData to avoid a
        PyObjC segfault in NSData's interaction with Metal's internal
        SHA256 hashing.

        Args:
            name: kernel function name.
            kernel: metallib bytes (Triton framework) or file path str (legacy).
            shared_mem: bytes of shared memory needed.
            device: ignored (Metal has a single GPU).

        Returns 5-tuple: (library, pipeline_state, n_regs, n_spills, n_max_threads).
        """
        import Foundation

        if isinstance(kernel, (bytes, bytearray)):
            # Triton framework path: write bytes to temp file.
            with tempfile.NamedTemporaryFile(suffix=".metallib", delete=False) as f:
                f.write(kernel)
                tmp_path = f.name
            url = Foundation.NSURL.fileURLWithPath_(tmp_path)
        else:
            # Legacy path: kernel is a file path string.
            url = Foundation.NSURL.fileURLWithPath_(kernel)

        library, error = self.device.newLibraryWithURL_error_(url, None)
        if error is not None:
            raise RuntimeError(f"Failed to load metallib: {error}")

        function = library.newFunctionWithName_(name)
        if function is None:
            available = [
                library.functionNames().objectAtIndex_(i)
                for i in range(library.functionNames().count())
            ]
            raise RuntimeError(
                f"Kernel '{name}' not found in metallib. Available: {available}"
            )

        pipeline_state, error = (
            self.device.newComputePipelineStateWithFunction_error_(function, None)
        )
        if error is not None:
            raise RuntimeError(f"Failed to create pipeline state: {error}")

        n_max_threads = pipeline_state.maxTotalThreadsPerThreadgroup()
        return library, pipeline_state, 0, 0, n_max_threads

    def launch(
        self,
        pipeline_state,
        grid,
        threadgroup_size,
        buffers,
    ):
        """Dispatch a compute kernel.

        Args:
            pipeline_state: MTLComputePipelineState from load_binary.
            grid: (grid_x, grid_y, grid_z) threadgroup counts.
            threadgroup_size: (threads_x, threads_y, threads_z) per threadgroup.
            buffers: list of (MTLBuffer, offset) tuples bound to sequential indices.
        """
        import Metal

        command_buffer = self.command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline_state)

        for i, (buf, offset) in enumerate(buffers):
            encoder.setBuffer_offset_atIndex_(buf, offset, i)

        grid_size = Metal.MTLSizeMake(*grid)
        tg_size = Metal.MTLSizeMake(*threadgroup_size)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(grid_size, tg_size)
        encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

        status = command_buffer.status()
        if status == Metal.MTLCommandBufferStatusError:
            error = command_buffer.error()
            raise RuntimeError(f"Metal kernel execution failed: {error}")

    def make_buffer_from_ptr(self, ptr, nbytes):
        """Create a Metal buffer wrapping an existing pointer (zero-copy UMA).

        Uses a ctypes array (not c_void_p) so PyObjC can validate the
        buffer size for newBufferWithBytesNoCopy.
        """
        import ctypes
        import Metal

        # Wrap pointer as a sized ctypes array so PyObjC accepts it.
        src = (ctypes.c_char * nbytes).from_address(ptr)
        buf = self.device.newBufferWithBytesNoCopy_length_options_deallocator_(
            src,
            nbytes,
            Metal.MTLResourceStorageModeShared,
            None,
        )
        if buf is None:
            raise RuntimeError(
                f"Failed to create Metal buffer from pointer {ptr:#x} ({nbytes} bytes)"
            )
        return buf

    def make_buffer(self, nbytes):
        """Allocate a new Metal buffer."""
        import Metal

        buf = self.device.newBufferWithLength_options_(
            nbytes, Metal.MTLResourceStorageModeShared
        )
        if buf is None:
            raise RuntimeError(f"Failed to allocate Metal buffer ({nbytes} bytes)")
        return buf

    def get_device_properties(self, device=0):
        return {
            "max_shared_mem": 32768,  # 32 KB threadgroup memory
            "max_num_regs": 0,
            "multiprocessor_count": self.device.maxThreadgroupMemoryLength() // 1024,
            "warp_size": 32,
        }

    def unload_module(self, module):
        pass  # Metal libraries are reference-counted by ObjC ARC


_metal_utils = None


def _get_utils():
    """Module-level MetalUtils singleton (Metal device is a system singleton)."""
    global _metal_utils
    if _metal_utils is None:
        _metal_utils = MetalUtils()
    return _metal_utils


class MetalLauncher:
    """Triton kernel launcher for Metal backend.

    Instantiated by the Triton framework as launcher_cls(src, metadata).
    Called as launcher(gridX, gridY, gridZ, stream, function, kernel_metadata,
                       launch_metadata, launch_enter_hook, launch_exit_hook, *args).
    """

    def __init__(self, src, metadata):
        self.constants = src.constants if hasattr(src, "constants") else {}
        self.arg_names = src.fn.arg_names if hasattr(src, "fn") else []
        self.signature = src.signature if hasattr(src, "signature") else {}

    def __call__(
        self,
        gridX,
        gridY,
        gridZ,
        stream,
        function,  # MTLComputePipelineState from load_binary
        kernel_metadata,
        launch_metadata,
        launch_enter_hook,
        launch_exit_hook,
        *args,
    ):
        import ctypes

        if launch_enter_hook:
            launch_enter_hook(kernel_metadata, launch_metadata)

        utils = _get_utils()

        # Unpack kernel metadata: (num_warps, num_ctas, shared, block_size)
        num_warps = kernel_metadata[0] if kernel_metadata else 4
        block_size = kernel_metadata[3] if kernel_metadata and len(kernel_metadata) > 3 else num_warps * 32

        # Pack arguments into Metal buffers.
        # For tensors, we copy data into Metal-allocated buffers to avoid
        # NoCopy's page-alignment requirement (ARM64 pages are 16 KB).
        # After dispatch, we copy output data back.
        buffers = []
        tensor_copies = []  # (metal_buf, tensor, nbytes) for output writeback

        for arg in args:
            if hasattr(arg, "data_ptr"):
                ptr = arg.data_ptr()
                nbytes = arg.nelement() * arg.element_size()
                # Allocate Metal buffer and copy tensor data in.
                buf = utils.make_buffer(nbytes)
                src = (ctypes.c_char * nbytes).from_address(ptr)
                dst = buf.contents().as_buffer(nbytes)
                dst[:] = bytes(src)
                buffers.append((buf, 0))
                # Track all tensor args for output writeback.
                tensor_copies.append((buf, arg, nbytes))
            elif isinstance(arg, bool):
                buf = utils.make_buffer(4)
                view = buf.contents().as_buffer(4)
                struct.pack_into("i", view, 0, int(arg))
                buffers.append((buf, 0))
            elif isinstance(arg, int):
                if arg < -(1 << 31) or arg > 0xFFFFFFFF:
                    buf = utils.make_buffer(8)
                    view = buf.contents().as_buffer(8)
                    struct.pack_into("q", view, 0, arg)  # int64_t
                elif arg < 0:
                    buf = utils.make_buffer(4)
                    view = buf.contents().as_buffer(4)
                    struct.pack_into("i", view, 0, arg)  # int32_t (signed)
                else:
                    buf = utils.make_buffer(4)
                    view = buf.contents().as_buffer(4)
                    struct.pack_into("I", view, 0, arg)  # uint32_t
                buffers.append((buf, 0))
            elif isinstance(arg, float):
                buf = utils.make_buffer(4)
                view = buf.contents().as_buffer(4)
                struct.pack_into("f", view, 0, arg)
                buffers.append((buf, 0))
            else:
                raise TypeError(f"Unsupported argument type: {type(arg)}")

        threads_per_tg = block_size
        # Flatten grid to 1D. Our MSL kernels use scalar uint pid
        # (threadgroup_position_in_grid gives x-component only).
        # Prebuilt kernels decompose the flat index internally.
        grid = (gridX * gridY * gridZ, 1, 1)
        threadgroup_size = (threads_per_tg, 1, 1)

        utils.launch(function, grid, threadgroup_size, buffers)

        # Copy results back from Metal buffers to tensor memory.
        for metal_buf, tensor, nbytes in tensor_copies:
            src_view = metal_buf.contents().as_buffer(nbytes)
            dst_ptr = tensor.data_ptr()
            dst = (ctypes.c_char * nbytes).from_address(dst_ptr)
            dst[:] = bytes(src_view)

        if launch_exit_hook:
            launch_exit_hook(kernel_metadata, launch_metadata)


def _detect_metal_arch():
    """Detect the Apple GPU architecture from the Metal device name."""
    try:
        import Metal

        device = Metal.MTLCreateSystemDefaultDevice()
        if device is None:
            return "apple-unknown"
        name = device.name()
        # e.g. "Apple M4 Max" -> "apple-m4-max"
        return name.lower().replace(" ", "-")
    except (ImportError, Exception):
        return "apple-unknown"


class MetalDriver(DriverBase):

    def __init__(self):
        super().__init__()
        self.utils = MetalUtils()
        self.launcher_cls = MetalLauncher

    @classmethod
    def is_active(cls):
        if platform.system() != "Darwin":
            return False
        try:
            import Metal

            device = Metal.MTLCreateSystemDefaultDevice()
            return device is not None
        except ImportError:
            return False

    def get_current_target(self):
        arch = _detect_metal_arch()
        return GPUTarget("metal", arch, 32)

    def get_current_device(self):
        return 0  # Metal has a single GPU

    def get_current_stream(self, device=0):
        return 0  # Metal has no CUDA-style streams

    def get_active_torch_device(self):
        import torch

        return torch.device("mps")

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty_to_cpp(ty)

    def get_benchmarker(self):
        from triton_metal.profiling.metal_bench import metal_do_bench

        return metal_do_bench
