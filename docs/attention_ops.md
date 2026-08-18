# Frontier attention ops (KDA + FlashAttention backward)

Two attention capabilities the standard `@triton.jit` → simdgroup-FlashAttention path does
not cover, exposed as direct Metal ops. Both are validated against ground-truth references
(`tests/test_kda.py`, `tests/test_fa_backward.py`) and dispatched through `compile_shader`.

## KDA — Kimi Delta Attention / gated DeltaNet (`triton_msl.kda`)

Linear/delta-rule attention: a per-key-dimension forget gate combined with the delta rule's
fast-weight correction. This is *not* softmax attention, and its chunked (parallel-prefill)
form needs a UT-transform triangular solve, so it cannot be expressed as one `@triton.jit`
kernel — it ships as a direct op rather than a detector/route.

```python
from triton_msl.kda import kda_attention, kda_decode_step

# Prefill (chunked, MMA-optimized ~7-12x over a naive kernel):
out = kda_attention(q, k, v, a, beta)          # q,k,v,a: [ZH,T,64]; beta: [ZH,T]; mps

# Decode (one autoregressive step, state updated in place):
S = torch.zeros(ZH, 64, 64, device="mps")
o = kda_decode_step(q_t, k_t, v_t, a_t, beta_t, S)   # q_t..: [ZH,64]; beta_t: [ZH]
```

- `a` is the per-key-dim gate in `(0, 1)`; `beta` is the delta step (typically a sigmoid).
- float32 or float16 I/O (accumulate and state are always fp32); output dtype matches `q`.
- **Constraints (enforced with a loud `ValueError`):** head dim `== 64`, `T % 8 == 0`.
- **Head dim is fixed at 64 by design:** the recurrent state is `d_k × d_v`, so `d = 128`
  would need a 64 KB threadgroup state, over Metal's 32 KB limit. A `d = 128` variant would
  require a device-resident state (memory-bound, much slower) or a low-rank-state redesign;
  `d = 64` is the sweet spot for the on-chip fast path.

## FlashAttention backward — trainable attention (`triton_msl.fa_backward`)

The forward FlashAttention path is inference-only. `flash_attention` wraps a
`torch.autograd.Function` whose backward dispatches tiled, MMA-optimized Metal dQ/dK/dV
kernels (the FA-2 backward), so an attention call inside a training loop gets Metal gradients.

```python
from triton_msl.fa_backward import flash_attention

o = flash_attention(q, k, v, causal=True)   # q,k,v: [..., N, 64] mps; folds leading dims
o.backward(dO)                              # dQ/dK/dV computed on Metal
```

- head dim `== 64`, `N % 16 == 0`, full and causal.
- The forward currently recomputes the logsumexp in torch to hand it to the backward. This
  is correct and fine at realistic sizes (transient `O(N²)`); a fused forward kernel that
  emits the logsumexp directly is an optional perf follow-up, not a correctness gap.
