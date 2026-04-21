# TMA-Inspired "Copy Simdgroup" Pattern for Apple Silicon

**Status:** Prototyped and **shelved** — benchmarks show net slowdown on M4 Max.
**Date:** 2026-04-19 (design), 2026-04-21 (prototype + benchmark)

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

**Prototyped on 2026-04-21 — result: negative, shelved.**

### Prototype

Two standalone MSL kernels were written in `docs/design/prototypes/`:
- `baseline_matmul.metal` — 96x96x16 tile, 4 simdgroups, all doing both
  cooperative global->threadgroup copy AND simdgroup MMA compute.
- `copysg_matmul.metal` — same tile, but sg0 dedicated to prefetching the
  NEXT K-tile into double-buffered threadgroup memory while sg1-sg3 do MMA
  on the CURRENT tile. Uses 2x threadgroup memory and a per-K-iter barrier.

Both produce bit-identical output vs NumPy's reference matmul (fp32). The
driver `bench_copysg.py` / `bench_copysg_sizes.py` compiles each with
`-O3 -ffast-math`, batches 100 dispatches into one command buffer, and
reports best-of-N per-iter time.

### Measurements (Apple M4 Max, 40 GPU cores)

| Problem (MxNxK) | Baseline | Copy-SG  | Speedup |
|-----------------|----------|----------|---------|
| 480x480x512     | 457 GF   | 395 GF   | 0.86x   |
| 480x480x1024    | 440 GF   | 337 GF   | 0.77x   |
| 480x480x2048    | 438 GF   | 330 GF   | 0.75x   |
| 960x960x512     | 1288 GF  | 1114 GF  | 0.86x   |
| 960x960x1024    | 1320 GF  | 1138 GF  | 0.86x   |
| 1920x1920x512   | 1053 GF  | 1032 GF  | 0.98x   |
| 1920x1920x1024  | 1053 GF  | 1014 GF  | 0.96x   |

**Across every tested configuration, copy-SG is slower than baseline.** The
gap narrows at larger problem sizes (where more in-flight threadgroups hide
memory latency at the hardware level), but never closes.

### Why it loses

1.  **The workload is compute-bound.** 96x96x16 fp32 tile has arithmetic
    intensity = 24 FLOPs/byte. On M4 Max (peak ~4.6 TF32, bandwidth
    ~546 GB/s), the memory-bound ceiling is ~13 TFLOPS — far above the
    ~460 GFLOPS we achieve. Hiding memory latency behind compute cannot
    help if compute is already the bottleneck.

2.  **Copy-SG sacrifices 25% of compute throughput.** Baseline uses 4 SGs
    for MMA; copy-SG uses only 3. That 25% loss is not offset by
    prefetch savings because the prefetch wasn't on the critical path.

3.  **Cooperative copy is already ~4x faster than 1-SG copy.** In the
    baseline, 128 threads load 4608 tile elements cooperatively
    (~36 loads/thread). In copy-SG, 32 threads do the full 4608
    (~144 loads/thread). The copy SG likely becomes the barrier
    bottleneck rather than hiding behind compute.

4.  **Apple's GPU scheduler already overlaps memory across threadgroups.**
    With 25 threadgroups dispatched against 40 GPU cores, multiple TGs
    are resident per core, and the hardware naturally interleaves their
    global memory accesses with others' compute. Adding an explicit
    software pipeline on top gains nothing and costs a simdgroup.

5.  **The 32KB threadgroup memory limit hurts.** Double-buffering forced
    BK=16 (vs 32 for a non-pipelined kernel), halving per-K-iter compute
    and increasing per-iter barrier overhead. This is a structural
    disadvantage of the pattern on Apple GPUs.

### Answers to the original open questions

- *Does Apple's GPU overlap compute and memory at simdgroup granularity?*
  Not usefully. The scheduler overlaps at the **threadgroup** level via
  concurrent resident threadgroups. Within a threadgroup, dedicating
  hardware to explicit prefetch is worse than using it for compute.
- *Can we get similar effect by just increasing num_stages?* The
  compiler's pipelining is already implicit via TG-level scheduling.
  Software TMA emulation has no room to add value here.

### Recommendation

**Shelve** this pattern for M1-M4. The theoretical justification (TMA on
Hopper gets big wins) does not translate, because:
- Apple's cache hierarchy and TG-level scheduling already hide the
  memory latency this pattern targeted.
- The 25% simdgroup tax is a real cost, not a theoretical one.
- The 32KB threadgroup memory cap restricts the tile sizes where
  double-buffering is viable.

Future direction: if M5's `simdgroup_async_copy` ships, re-evaluate the
pattern with hardware async copy (which doesn't consume an SG). That is
a fundamentally different cost/benefit equation.

### Artifacts

- Prototypes: `docs/design/prototypes/baseline_matmul.metal`,
  `docs/design/prototypes/copysg_matmul.metal`
- Drivers: `docs/design/prototypes/bench_copysg.py`,
  `docs/design/prototypes/bench_copysg_sizes.py`
- No integration into `generic_lowerer.py` was attempted.
