# Portability — develop on Apple Silicon, run on NVIDIA

triton-msl is a **backend** for [OpenAI Triton](https://github.com/triton-lang/triton),
not a separate language. Your `@triton.jit` source is standard Triton: the frontend
(Python → Triton IR) is shared with the NVIDIA backend, and triton-msl only swaps the
final stage (Triton IR → Metal instead of → PTX). So the kernel you write and debug on a
Mac is the *same* kernel that runs on an NVIDIA GPU — copy it onto a CUDA box, put your
tensors on `cuda` instead of `mps`, and it compiles through Triton's NVIDIA backend
unchanged.

## What's portable, and what isn't

| | Portable? | Why |
|---|---|---|
| **Kernel logic / correctness** | ✅ Yes — verified bit-identical (below) | Same source → same Triton IR; only the backend codegen differs |
| **Performance** | ❌ No — and never will be | Block sizes, occupancy, fast-path routing are hardware-specific (different memory hierarchy + matrix units). Physics, not a gap. |
| **Numerics** | ⚠️ Mostly — one caveat | fp32 is bit-identical with `input_precision="ieee"`; NVIDIA's **tf32** default for `tl.dot` is lower precision (see below) |

The practical consequence: you can develop and correctness-debug a kernel's *logic* on the
Mac you already own — the expensive, iterative part — and trust it transfers. Performance
tuning still needs the target GPU (rent one for minutes; see *Reproduce* below).

## Verified — bit-identical on real NVIDIA silicon

The three example kernels (`examples/local_triton_dev.py`), unmodified, run on both
backends and checked against the **same** NumPy reference:

- **Mac:** Apple M4 Max (Metal) · triton-msl · Triton 3.7.0 · torch 2.12.1
- **NVIDIA:** A40 · Triton 3.0.0 · torch 2.4.1+cu124  *(rented for ~8 min, then terminated)*

| kernel | Mac vs NumPy | NVIDIA vs NumPy | **Mac ↔ NVIDIA Δ** |
|---|---|---|---|
| `vector_add` (n = 98,432) | 0 | 0 | **0 — bit-identical** |
| `fused_softmax` (128 × 781) | 7.45e-9 | 5.59e-9 | 7.45e-9 |
| `matmul` fp32 / `ieee` (256³) | 4.58e-5 | 4.58e-5 | **0 — bit-identical** |
| `matmul` `tf32` (256³) | **refused** *(no tf32 on Metal)* | 6.07e-2 | — |

Two of the three kernels produced *literally the same bits* across Apple Metal and NVIDIA;
softmax matched to fp rounding noise (~1e-9). It also crossed **Triton versions** (3.0.0 on
the pod vs 3.7.0 on the Mac) — the source is stable across both.

> **Scope (honest):** measured on these specific shapes and block sizes. Different tiling
> can introduce last-bit differences from accumulation order. This is a *measured* result,
> not a universal bit-equality guarantee — but the logic is identical, so any divergence is
> fp-rounding-scale, not algorithmic.

## The tf32 caveat (the one real divergence)

NVIDIA's `tl.dot` defaults to **tf32** (~10-bit mantissa) for fp32 inputs — faster, but
lower precision. The *same* matmul kernel:

- on NVIDIA with the default → **6.1e-2** error vs an fp64 reference (the tf32 mantissa loss);
- with `input_precision="ieee"` → **bit-identical** with the Mac.

So your Mac is the *more precise* default: Metal has no tf32 hardware, and triton-msl
**refuses** `input_precision="tf32"` rather than silently approximating it (the same
correct-or-refuse discipline as everywhere else). To match across backends, pass
`input_precision="ieee"` to `tl.dot`, or set the NVIDIA default accordingly.

## Reproduce it yourself

Rent a GPU for a few minutes (any recent CUDA box with `torch` + `triton`; the run above
cost a few cents):

```bash
# on the NVIDIA box:
python3 benchmarks/cross_backend_verify.py cuda     # writes out_cuda.npz + prints vs-NumPy errors
# pull out_cuda.npz back to the Mac, then:
python3 benchmarks/cross_backend_verify.py mps      # writes out_mps.npz
# diff the two .npz element-wise -> the Mac <-> NVIDIA deltas
```

`benchmarks/cross_backend_verify.py` defines the kernels once, runs them on `cuda` or
`mps`, saves the outputs, and (when both files exist) prints the cross-backend deltas.

## The workflow this enables

**Home = the correctness / iteration loop** (free, fast, offline) — get the kernel's logic
right on your Mac. **A rented GPU = the perf + final-validation pass** (minutes, cents). You
don't need to *own* NVIDIA silicon for the inner loop; you need it occasionally for the
perf pass.

The gating factor isn't portability (solved) — it's the supported subset. A kernel
triton-msl **refuses** (see [`docs/SUPPORTED_OPS.md`](docs/SUPPORTED_OPS.md)) can't be
developed locally yet; most of those gaps are lowering paths not yet implemented
(software, in this project's control), a few are genuine Apple-hardware limits (e.g.
head_dim > 128 attention, where the fp32 accumulator alone overflows threadgroup memory).
