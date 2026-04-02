# triton-metal

Metal (Apple Silicon) backend for [OpenAI Triton](https://github.com/triton-lang/triton). Write `@triton.jit` kernels and run them on your Mac's GPU.

```
@triton.jit → Triton TTIR → TTGIR → MSL → metallib → Apple GPU
```

## Status

**Alpha** — actively developed, not yet production-ready.

- **4,279 / 9,334** upstream Triton tests passing (0 failures)
- **32/32** torch.compile model tests
- **15/15** MLX backend tests
- Triton tutorials 01-03, 05 passing

## Requirements

- Apple Silicon Mac (M1 or later)
- macOS 14 (Sonoma) or later
- Xcode Command Line Tools: `xcode-select --install`
- Python 3.10+

## Install

```bash
pip install triton-metal

# Triton is required but installed separately (macOS wheels may not be available)
pip install triton>=3.6.0

# If no Triton wheel exists for your platform, build from source:
# pip install git+https://github.com/triton-lang/triton.git
```

## Quick Start

### @triton.jit

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

n = 1024
x = torch.randn(n, device="cpu")
y = torch.randn(n, device="cpu")
out = torch.empty(n, device="cpu")
add_kernel[(n + 255) // 256,](x, y, out, n, BLOCK=256)
print(f"Max error: {(out - (x + y)).abs().max():.2e}")
```

### torch.compile

```python
import torch
import triton_metal.inductor  # registers the "metal" backend

model = torch.nn.Sequential(
    torch.nn.Linear(256, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 256),
)

compiled = torch.compile(model, backend="metal")
x = torch.randn(32, 256)
out = compiled(x)
```

### MLX

```python
import mlx.core as mx
import triton
import triton.language as tl
from triton_metal.mlx import triton_call

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

n = 1024
x = mx.random.normal((n,))
y = mx.random.normal((n,))
out = mx.zeros((n,))
results = triton_call(add_kernel, x, y, out, n, grid=(4,), BLOCK=256)
```

## What Works

| Category | Operations |
|----------|-----------|
| **Elementwise** | add, sub, mul, div, exp, log, sqrt, abs, neg, SiLU, GELU, sigmoid, tanh, ReLU, leaky ReLU, clamp, FMA |
| **Reductions** | sum, max, min, argmax, argmin, xor_sum |
| **Dot product** | `tl.dot` with strided matmul template, all epilogues (add, softmax, chain-dot, transpose) |
| **Attention** | Flash attention (causal + non-causal) via Triton |
| **Normalization** | Layer norm, RMS norm, batch norm |
| **Type casts** | FP32, FP16, BF16, INT8, INT16, INT32, bool |
| **Control flow** | `scf.for`, `scf.if`, while loops |
| **Atomics** | atomic_add, atomic_max, atomic_min, atomic_and, atomic_or, atomic_xor, CAS |
| **Tensor ops** | cat, join, split, interleave, reshape, permute, transpose, histogram, gather |
| **torch.compile** | 32 models including MLP, ResBlock, TransformerBlock, SmallGPT, MiniViT, LSTM |
| **MLX** | Zero-copy dispatch via `mx.fast.metal_kernel()` |

## What Doesn't Work

| Feature | Reason |
|---------|--------|
| FP64 | Metal has no FP64 support |
| FP8, TF32 | Not available on Apple GPUs |
| Backward pass / training | Not implemented |
| Multi-GPU | Apple Silicon is single-GPU |
| `tl.dot` with sizePerThread > 1 | Requires 2D cooperative execution model |
| Unstructured control flow (`cf.cond_br`) | Not yet implemented |

## Performance (M4 Max)

Benchmarks from Triton tutorials:

| Kernel | Size | Throughput | vs CPU |
|--------|------|-----------|--------|
| Vector add | 16M elements | 137.5 GB/s | 0.93x |
| Softmax | 8192x1024 | 109.4 GB/s | **1.26x** |
| Matmul | 512x512 | 826 GFLOP/s | 0.32x |
| Layer norm | 4096x1024 | 77.5 GB/s | 0.34x |

**Known bottleneck**: ~0.15ms buffer copy overhead per kernel launch when using MPS tensors (MPS→CPU→Metal→CPU→MPS). Use CPU tensors for best performance, or the MLX backend for zero-copy dispatch.

## Architecture

```
@triton.jit kernel
    → Triton frontend (Python AST → TTIR)
    → Triton optimizer (TTIR → TTGIR)
    → mlir_walker.py: walk TTGIR module → IRGraph
    → generic_lowerer.py: IRGraph → MSL source
    → xcrun metal: MSL → AIR → metallib
    → driver.py: load metallib, dispatch on GPU
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE)
