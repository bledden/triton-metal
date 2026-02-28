import platform

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

    def load_binary(self, name, metallib_path, shared_mem):
        """Load a metallib from a file path and create a compute pipeline state.

        Uses newLibraryWithURL instead of newLibraryWithData to avoid a
        PyObjC segfault in NSData's interaction with Metal's internal
        SHA256 hashing.

        Returns (library, pipeline_state, n_regs, n_spills).
        """
        import Foundation

        url = Foundation.NSURL.fileURLWithPath_(metallib_path)
        library, error = self.device.newLibraryWithURL_error_(url, None)
        if error is not None:
            raise RuntimeError(f"Failed to load metallib from {metallib_path}: {error}")

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

        return library, pipeline_state, 0, 0

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
        """Create a Metal buffer wrapping an existing pointer (zero-copy UMA)."""
        import ctypes
        import Metal

        buf = self.device.newBufferWithBytesNoCopy_length_options_deallocator_(
            ctypes.c_void_p(ptr),
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


class MetalLauncher:
    """Wraps a compiled metallib kernel for repeated dispatch."""

    def __init__(self, utils, name, kernel_bytes, shared_mem, num_warps):
        self.utils = utils
        self.name = name
        self.num_warps = num_warps
        self._library, self._pipeline_state, _, _ = utils.load_binary(
            name, kernel_bytes, shared_mem
        )

    def __call__(self, grid, *args, **kwargs):
        """Launch the kernel with the given grid and arguments.

        Args:
            grid: tuple of (grid_x, grid_y, grid_z) threadgroup counts.
            *args: kernel arguments (torch tensors or scalars).
        """
        import struct

        buffers = []

        for arg in args:
            if hasattr(arg, "data_ptr"):
                # PyTorch tensor — wrap its pointer as a Metal buffer.
                ptr = arg.data_ptr()
                nbytes = arg.nelement() * arg.element_size()
                buf = self.utils.make_buffer_from_ptr(ptr, nbytes)
                buffers.append((buf, 0))
            elif isinstance(arg, int):
                # Pack scalar into a small Metal buffer.
                buf = self.utils.make_buffer(4)
                view = buf.contents().as_buffer(4)
                struct.pack_into("I", view, 0, arg)
                buffers.append((buf, 0))
            elif isinstance(arg, float):
                buf = self.utils.make_buffer(4)
                view = buf.contents().as_buffer(4)
                struct.pack_into("f", view, 0, arg)
                buffers.append((buf, 0))
            else:
                raise TypeError(f"Unsupported argument type: {type(arg)}")

        threads_per_tg = self.num_warps * 32
        threadgroup_size = (threads_per_tg, 1, 1)

        # Normalize grid to 3D.
        if isinstance(grid, int):
            grid = (grid, 1, 1)
        elif len(grid) == 1:
            grid = (grid[0], 1, 1)
        elif len(grid) == 2:
            grid = (grid[0], grid[1], 1)

        self.utils.launch(
            self._pipeline_state,
            grid,
            threadgroup_size,
            buffers,
        )


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

    def get_active_torch_device(self):
        import torch

        return torch.device("mps")

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty_to_cpp(ty)

    def get_benchmarker(self):
        from triton_metal.profiling.metal_bench import metal_do_bench

        return metal_do_bench
