# TMA-Inspired "Copy Simdgroup" Pattern for Apple Silicon

**Status:** Design draft (not implemented)
**Date:** 2026-04-19

## Motivation

NVIDIA Hopper's Tensor Memory Accelerator (TMA) decouples address generation
from compute threads — a dedicated copy engine fetches tiles of global memory
into shared memory while compute threads do matmul / reduction. This
asynchrony is a major perf contributor for modern ML kernels.

Apple Silicon pre-M5 has no equivalent hardware primitive. M5's
`simdgroup_async_copy` is the closest analog, but it's hardware-gated.

This doc sketches a software-emulated TMA-inspired pattern that could run on
M1-M4 today: **dedicate one 32-thread simdgroup to cooperative
global→threadgroup memory transfers while the remaining simdgroups do
compute.**

## Core Idea

With `num_warps=4` (128 threads), a threadgroup has 4 simdgroups:

```
Threadgroup (128 threads = 4 simdgroups)
┌──────────┬──────────┬──────────┬──────────┐
│ simdgroup│ simdgroup│ simdgroup│ simdgroup│
│     0    │     1    │     2    │     3    │
│  (copy)  │ (compute)│ (compute)│ (compute)│
└──────────┴──────────┴──────────┴──────────┘
```

While simdgroups 1-3 compute on tile N (matmul, reduction, etc.), simdgroup 0
pre-fetches tile N+1 from global to threadgroup memory. At the end of the
stage, a `wg.barrier` sync then everyone advances.

This trades ~25% of compute bandwidth for overlap of memory latency with
compute. Net win when compute time ≥ copy time (typical for matmul + reduce
patterns).

## Codegen Strategy

### Detection

This pattern only helps when:
- Kernel has `scf.for` with loads inside (the loop is the staging unit)
- Load volume per iteration is large enough that copy time > barrier overhead
- Compute phase is non-trivial (e.g. tt.dot, multi-op reduce chain)

FlashAttention and K-loop matmul are the canonical beneficiaries.

### Lowering (per-iteration)

```
// Iteration k:
if (sgitg == 0) {
    // Copy simdgroup: prefetch tile N+1 into shared_next
    for (i = tiisg; i < TILE_SIZE; i += 32) {
        shared_next[i] = global[base + k*STRIDE + TILE_SIZE + i];
    }
} else {
    // Compute simdgroups: work on tile N (already in shared_current)
    <matmul / reduce using shared_current>
}
wg.barrier();
// Swap: shared_current ↔ shared_next
```

Double-buffered shared memory (2× threadgroup memory cost) enables the
prefetch.

### Integration Point

This is a C++ MLIR pass that runs BEFORE the existing SCF→CF lowering. It
recognizes the pattern (`scf.for` with `tt.load` + `tt.dot`/`tt.reduce`) and
restructures the IR to have simdgroup-conditional branches.

Alternatively: implement as an MSL template in `_simple_dot_inline` /
`_k_loop_dot_inline` that the generic lowerer picks up when a
"pipelined=true" option is set.

## Cost-Benefit Estimate

**Costs:**
- Loss of 1/4 compute simdgroups during the compute phase (not a total 25%
  loss — copy simdgroup is idle during pure compute, could still contribute
  if we schedule it carefully)
- 2× threadgroup memory (double buffer)
- Barrier overhead (but we already barrier once per K iteration)
- Compiler complexity: new detection + emission pass

**Benefits (hypothetical):**
- Hides global memory latency behind compute
- On matmul with K-loop and ≥ 8 iterations: maybe 20-30% speedup
- On FA: potentially more, since the K tile is small and copy per
  iteration is frequent

**Risk:**
- No equivalent in Apple's published performance guidance
- May hit unexpected cache or coherence penalties
- Barrier bloat if stages are too short

## Comparison with M5's `simdgroup_async_copy`

M5's `simdgroup_async_copy` is a true async DMA issued by the simdgroup. It
doesn't consume simdgroup compute resources during the copy. Our emulation
DOES consume a simdgroup, so the gain ceiling is lower. But we get *some*
overlap on M1-M4 today, vs. waiting for M5 hardware.

## Proposed Prototype

1. Implement as a standalone MSL template (not a C++ MLIR pass initially)
2. Target a simple matmul kernel (e.g. 128x128 with BLOCK_K=32)
3. Measure against the existing `_k_loop_dot_inline` template
4. If >15% speedup on a real workload: productize into the generic lowerer
5. If <5% or negative: document findings and shelve

## Open Questions

- Does Apple's GPU actually overlap compute and memory at the
  simdgroup granularity? Or does the hardware serialize at the
  threadgroup scheduler level, making this a nop?
- Would the detection pattern false-positive on kernels where the
  copy-compute ratio is inverted?
- Can we get similar effect by just increasing `num_stages` and trusting
  the compiler's pipeline optimizer?

## Status

**Not implemented.** Tracked as future research direction. Current Apple
matmul path uses MSL's `simdgroup_matrix_multiply_accumulate` template
without explicit copy/compute simdgroup split.
