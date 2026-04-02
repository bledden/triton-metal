# Changelog

## 0.1.0-alpha (2026-03-10)

First public alpha release of triton-metal.

### Milestone 1: First Kernel on Metal
- `@triton.jit` vector add running on Apple GPU via Metal Shading Language

### Milestone 2: Kernel Coverage
- 28 `@triton.jit` tests: sum, max, min, softmax, matmul, SiLU, sigmoid, GELU, SwiGLU, RMS norm, layer norm, fused add+ReLU, leaky ReLU, clamp, FMA, FP16, negation, exp+log

### Milestone 3: Real Compiler
- Replaced pattern-matching parser with proper MLIR walker + op-by-op generic lowerer
- All kernels route through new pipeline (`mlir_walker.py` + `generic_lowerer.py`)
- Legacy parser (`ttgir_parser.py` + `msl_emitter.py`) kept as safety fallback only

### Milestone 4: Upstream Compatibility
- 4,279 / 9,334 upstream `test_core.py` tests passing (0 failures)
- Completed: atomics, while loops, `tt.dot` (strided matmul + all epilogues), 2D/3D reduce, argmax/argmin, `tt.histogram`, `tt.gather`, `tl.cat`, `tl.join`/`tl.split`, reshape, permute, transpose, `scf.for`/`scf.if`, NaN propagation, floor div, shift ops
- Triton tutorials 01 (vector add), 02 (softmax), 03 (matmul), 05 (layer norm) all passing
- `@triton.autotune` working end-to-end

### Milestone 5: torch.compile
- 32/32 torch.compile tests passing
- Models: Identity, ReLU, GELU, SiLU, Sigmoid, Tanh, ELU, LeakyReLU, Dropout, Linear, LayerNorm, BatchNorm2d, GroupNorm, InstanceNorm, Embedding, Conv2d, AvgPool, MaxPool, Softmax, LogSoftmax, MLP, LargeMLP, ResBlock, DepthwiseSeparable, ConvNet, TransformerBlock, MHA, SmallGPT, GPT, MiniViT, LSTM, EmbeddingBag
- `torch.compile(model, backend="metal")` integration via Triton inductor

### Milestone 6: MLX Backend
- 15/15 MLX backend tests passing
- Zero-copy dispatch via `mx.fast.metal_kernel()`
- API: `triton_metal.mlx.triton_call(kernel_fn, *args, grid=(...), **constexpr_kwargs)`

### Performance (M4 Max)
- Vector add (16M): 137.5 GB/s
- Softmax (8192x1024): 109.4 GB/s (1.26x vs CPU)
- Matmul (512x512): 826 GFLOP/s
- Layer norm (4096x1024): 77.5 GB/s
- MLX dispatch: ~0.12ms (zero-copy, comparable to native MLX)

### Known Limitations
- ~0.15ms buffer copy overhead per kernel launch (MPS tensors)
- No FP64, FP8, or TF32 (Metal hardware limitation)
- No backward pass / training support
- 32x32 matmul tile size (larger tiles would improve throughput)
