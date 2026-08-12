# Changelog

## Unreleased

### Performance

- **FlashAttention now beats PyTorch SDPA (was ~0.3× — a dispatch bug, not a slow
  kernel).** The simdgroup FA kernel (head_dim=128, BM=32/BN=64) was always fast, but
  its **2-D threadgroup grid** `(n_q_blocks, Z·H)` disqualified it from the zero-copy
  `compile_shader` fast-path (which hard-required a 1-D grid), so every launch fell to
  the **host-roundtrip metallib path — ~2.5–4.2× slower**. Attribution was decisive: the
  *same* kernel dispatched via `compile_shader` with its **native** 2-D grid vs a
  1-D-linearized grid measured identically (6.32 vs 6.31 TF), so the entire gap is
  dispatch overhead, not the kernel or the grid shape. A new `flash_attention` dispatch
  descriptor routes the FA kernel through `compile_shader` with its native grid
  (**fail-OPEN**: unlike the quantized path, the host path is equally correct, so any
  miss simply falls through — never wrong, never refused). Result, end-to-end through the
  shipping `@triton.jit` path (head_dim=128, cold A/B vs `F.scaled_dot_product_attention`):
  **fp16 full 1.65–1.99×**, fp32 full 1.27–1.53× (causal below, with the skip). A split-KV /
  flash-decoding variant was prototyped and **rejected** — once the dispatch is fixed, the
  online-softmax combine overhead makes it lose to the plain kernel at every size.

- **Causal FlashAttention skips the masked upper triangle.** The causal kernel ran
  *every* KV block and masked per-element, doing ~2× the necessary work (the fully-masked
  upper-triangle blocks). A q-block covering rows `[q_start, q_start+BM)` can stop its KV
  loop once `kv_start > q_start+BM-1` — every later block is entirely masked. This prunes
  only provably-all-masked blocks, so the per-element mask still handles the diagonal block
  and the numerics are **byte-identical**. On the zero-copy path it is a strict
  **1.03–1.69× over the un-skipped causal kernel** (the same loop-clamp was tried and
  reverted earlier, but on the old dispatch-bound path halving compute was invisible).
  Causal vs SDPA now: **fp16 1.42–1.95×**, fp32 1.01–1.29× for N≥1024 (only tiny N=512
  stays ~0.77–0.95×).

- **head_dim=64 FlashAttention now uses the simd MMA kernel too (and fp16 hd64 runs on
  the GPU at all for the first time).** The simd kernel is parameterized on head_dim
  (`TPG = (head_dim//8)//n_groups = head_dim//64`), so it is correct for any multiple of
  64 — but the router only ever sent head_dim=128 to it. head_dim=64 fell through to the
  generic path, where it was slow in fp32 (~0.19–0.33× SDPA) and, in **fp16, silently
  CPU-fell-back**: the generic K^T is a 32×64 `tt.trans` that exceeds the 1024-thread
  cap, so it refused and ran on CPU. A guarded hd64 branch now routes contiguous
  32×32 fp16/fp32 hd64 to the simd template (inheriting the zero-copy dispatch and causal
  skip for free); **any** detection refusal or ineligibility (e.g. non-contiguous)
  falls through to the unchanged generic path, so nothing regresses. hd64 vs SDPA:
  **fp16 full 2.00–2.89×, fp16 causal 1.55–2.31×, fp32 full 1.80–2.27×, fp32 causal
  1.47–1.93×** — and fp16 hd64 attention runs on the device instead of the CPU.

- **Asymmetric-head_dim FlashAttention (the MLA / DeepSeek-style attention core).** The
  2026-frontier models (DeepSeek-V3, Kimi) use Multi-head Latent Attention, whose reference
  form has a *different* head_dim for the QK score (192 = 128 nope + 64 rope) than for V/output
  (128). The simd FA kernel now takes an optional `v_head_dim`: the QK contraction runs over
  `head_dim` while the output/V tiling runs over `v_head_dim`. Because registers scale with
  the *output* width only, MLA's qk=192 / v=128 has the **same register footprint as hd128** —
  the 192-wide QK just adds loop iterations (a *symmetric* head_dim=192 overflows the register
  file and won't even build). `v_head_dim=None` is the default and is byte-identical to before.
  The MLA-core kernel is correct vs SDPA (err ≤2e-3 fp16) and **beats it 1.32–1.67× (full),
  1.26–1.81× (causal)**. fp32 qk=192 exceeds the 32 KB threadgroup budget (192-wide Q
  staging), so this targets fp16/bf16 — the dtype MLA runs in anyway.

- **`@triton.jit` auto-routing for asymmetric (MLA-shaped) attention.** `_detect_flash_attention`
  now reads a *separate* `v_head_dim` from the P@V output (it required `[block_m, head_dim]`
  before, refusing anything asymmetric), and the FA router accepts `_fa_maxdim` up to 256,
  routing an asymmetric shape to the simd template when it's simd-eligible (contiguous, v ∈
  {64,128}, fp16 for qk>128) and falling through to a loud refuse otherwise (the scalar/tiled
  fallback is symmetric-only). Symmetric hd64/hd128 are byte-identical (the FA suite is
  unchanged). A canonical asymmetric `@triton.jit` attention (qk=128/v=64, fp16) now
  auto-routes and matches SDPA. A gen-side budget guard turns an over-budget head_dim (e.g.
  qk=256 fp16 → 33 KB) into a clear error instead of a cryptic pipeline-state failure.
  Note: `tl.arange` needs power-of-2 bounds and qk=256 overflows the budget, so expressing
  real MLA (qk=192) from `@jit` uses the nope/rope split (see below).

- **Real MLA (nope/rope) attention auto-routes from `@triton.jit` — DeepSeek/Kimi-style
  attention now runs on Metal.** A real MLA kernel computes the QK score as two chained
  dots, `q_nope@k_nope^T + q_rope@k_rope^T` (128 + 64), over *separate* tensors — three
  `tt.dot`s total (Triton fuses the sum into the dot accumulator chain). The FA detector
  now recognizes this 3-dot shape (guarded, disjoint from the 2-dot symmetric path so it's
  byte-identical): it identifies the two QK dots (whose A operand is a load) vs the PV dot
  (A is the softmax `exp`, no load), extracts all four QK pointer chains, and sets
  `qk_head_dim = nope + rope = 192`. Routing is a **fail-closed cat-dispatch**: it
  concatenates `[q_nope|q_rope]` / `[k_nope|k_rope]` into contiguous `[.,192]` (a ~0.01 ms
  copy) and runs the validated qk=192 / v=128 kernel; the host path can't run the
  different-ABI kernel, so a miss (non-MPS, compile_shader off) refuses. A canonical
  pre-scaled MLA `@jit` kernel auto-routes, runs on GPU, matches SDPA (err ≤1e-3 fp16,
  full + causal), and **beats it 1.19–1.54× (full), 1.11–1.59× (causal)** end-to-end. The
  post-scaled form (scale on the summed dot result) still refuses cleanly (fused-scale
  gate) — correct-or-refuse preserved. fp16/bf16 only (fp32 qk=192 exceeds the 32 KB budget).

- **Split-K for skinny/deep fp32 matmul.** Small-M/N, deep-K matmuls (e.g.
  M=64,N=64,K=8192) were ~0.1× torch — occupancy-starved: a handful of output tiles
  means a handful of threadgroups, each running the whole K-loop serially. A new
  **deterministic two-pass** split-K path (each of G threadgroups computes 1/G of K
  into a partials buffer, then a reduce kernel sums over G — **no atomics**, so it is
  byte-identical run-to-run) fires for fp32 shapes with `n_tiles < 64` and
  `K >= 2048`: M=64,N=64,K=8192 **0.16 → 0.67 TF** (now > torch), M=128,N=128,K=8192
  **0.58 → 1.43 TF** (1.34× torch). Isolated + additive — any non-fit (moderate tile
  counts, shallow K, fp16/bf16) falls through to the unchanged single-pass fast
  kernel, so no shape regresses (the threshold was tuned by measuring the *shipped*
  path, where the partials-alloc + second-dispatch overhead made n_tiles=128 regress).

### Portability

- **Verified byte-identical on AMD, not just NVIDIA.** The example `@triton.jit` kernels,
  unmodified, ran on a rented AMD Instinct MI300X (ROCm 7.2.4, gfx942, Triton 3.6.0)
  alongside the earlier NVIDIA A40 and the Apple M4 Max. `vector_add` and the `ieee` matmul
  produced the **same SHA-256** across all three vendors (Metal == CUDA == ROCm); softmax
  matched to fp rounding (~7.5e-9); tf32 is refused on Metal and absent on AMD. The central
  claim — develop on the Mac you own, run unchanged on datacenter silicon — now holds across
  three ISAs (Metal / PTX / CDNA3) and three Triton versions (3.0.0 / 3.6.0 / 3.7.0).
  PORTABILITY.md + README updated.

### Platform

- **Validated on macOS 26.6 (Tahoe), build 25G72** — full project suite 1971
  passed / 0 failed on real hardware. The Metal-version probe already runtime-checks
  each `-std=metalX.Y` for actual loadability rather than guessing from SDK version
  numbers, so it was immune to the 15.x→26.x SDK-version jump that broke PyTorch MPS
  version parsing. No code change was needed — the design anticipated it.
- **Metal Toolchain detection (macOS 26 / Xcode 26)** — on Tahoe the Metal shader
  compiler is a separate on-demand component; when absent (fresh setup or an Xcode
  update), `xcrun metal` fails permanently. triton-msl now detects this specific
  failure and raises an actionable error naming the fix
  (`sudo xcodebuild -downloadComponent MetalToolchain`) instead of retrying it as a
  transient flake and reporting a misleading "transient, all 3 attempts failed". The
  Requirements doc calls out the component too.
- **C++ path AIR version now dynamic (macOS 26)** — the opt-in C++/LLVM-IR lowering
  path hardcoded `air.version = {2,7,0}` / `Metal 3.2`, which macOS 26's toolchain
  rejects (it wants AIR 2.8 / Metal 4.0), silently degrading the C++ path to CPU
  execution. device_detect now runtime-probes the toolchain's actual AIR /
  Metal-language version and threads it through `run_to_llvm` into the emitted
  metadata, so the C++ path runs on GPU on macOS 26 without breaking older macOS
  (the probe falls back to 2.7 / 3.2). Requires rebuilding `_triton_msl_cpp`.

### Gluon

- **Basic Gluon kernels now run on Metal.** Implemented `MetalBackend.get_target_name`
  (a hole in Triton's out-of-tree `BaseBackend` contract — the Gluon runtime calls it
  generically, so any backend without it raised `AttributeError`) and populated
  `metadata["shared"]` in the Gluon lowering path (`gluon_to_ttgir`, which the Gluon
  path runs instead of `make_ttgir`). Gluon copy / elementwise kernels with an explicit
  `BlockedLayout` (size_per_thread 1–8 × 4–8 warps) compile and run correctly.
  NVIDIA-specific Gluon (mma, warp specialization, mbarrier, TMA) remains out of scope.
  A companion upstream change — a default `get_target_name` on `BaseBackend` — would
  let any out-of-tree backend get Gluon for free (it is a hole in the base contract).

### Correctness

- **Weight-only int4 (GPTQ/AWQ per-group) decode GEMV now auto-routes** (was refused —
  the generic lowerer can't emit the bitwise nibble unpack). A canonical int4 decode
  GEMV written as a `@triton.jit` — packed weight `[N,K/2]` uchar (2 nibbles/byte),
  per-group fp32 scale/zero, `nib = (extui(w) >> ((k%2)*4)) & 0xF; acc += tl.sum(x *
  (sitofp(nib) - zero[g]) * scale[g])` — is recognized at compile time and dispatched
  to `make_int4_gemv` (~1.47× over int8, half the weight bytes). The recognizer
  (`_maybe_quant_gemv_int4_descriptor`) is **correct-or-refuse**: it verifies the
  nibble-unpack chain and **every packing constant** (byte = k//2, shift = (k%2)*4,
  mask 0xF) against the kernel's assumption, extracts and matches the per-group size,
  and binds each leg to its arg — so a mismatched packing refuses rather than
  mis-computes; int8 GEMVs (no nibble unpack) are not misrouted.
- **Weight-only int8 decode GEMV now RUNS on Metal** (was refused). A canonical
  weight-only int8 decode GEMV written as an in-loop row reduce —
  `acc += tl.sum(x[None,:] * (w_i8.to(f32) - zero[:,None]), axis=1)` over K, then
  `out = acc * scale` — with weight `[N,K]` contiguous (GPTQ/AWQ), per-N float
  scale/zero, fp32 in/out, is recognized at compile time and dispatched to the
  dedicated `make_int8_gemv` kernel (one simdgroup per output column, char4/float4
  coalesced, `simd_sum`) via `compile_shader`. Correct (~1e-5 err) and near the
  memory roofline (~426 GB/s at N=11008, K=4096 — the dominant LLM decode shape). It
  reuses the quantized-matmul dispatch (tagged `"gemv"`) and its fail-closed driver
  path; the recognizer (`_maybe_quant_gemv_descriptor`) is **correct-or-refuse** —
  it verifies the sum-reduce, the exact `mulf(bcast(x), subf(sitofp(w), bcast(zero)))`
  input, and the `acc*scale` epilogue, tracing every leg to its arg. A plain fp32
  GEMV (no dequant) is **not** misrouted — it refuses via the reduce-layout guard.
- **In-loop 2-D axis reduce (GEMV via `tl.sum`) now refuses instead of silently
  collapsing to row 0.** A `tl.sum` (or max/…) over one axis of a 2-D tile produces
  its per-row result in the row-broadcast layout (thread `lid` holds row `lid/N`).
  When that result is accumulated across a K-loop (`acc += tl.sum(x[None,:]*w,
  axis=1)` inside `for k in range(0, K, BK)`), the row-broadcast layout is not
  propagated across the `scf.for` loop-carry, so the loop-carried accumulator + 1-D
  store instead assume one-row-per-thread and every output row silently collapsed to
  the first (a plain or quantized GEMV written this way returned `o[n] = o[0]`). The
  existing 2-D-reduce coverage guard did not catch this (the tile fits the
  threadgroup) — it is a distinct *layout* bug. Detected structurally (a 2-D→1-D
  reduce whose result feeds an `scf.for` carried value) and refused loudly; the
  single-tile form (no K-loop) is correct and unaffected. Full fix (propagating the
  broadcast layout across the loop-carry) is a follow-up. Found via the quantized
  GEMV/decode investigation.
- **Weight-only int8 quantized matmul now RUNS on Metal** (was refused). A canonical
  weight-only int8 matmul — `out = a @ ((w_i8.to(f32) - zero) * scale)` with the
  standard `(input, weight, output, scale, zero, M, N, K, ...strides)` signature,
  weight stored `[K,N]` contiguous, per-N float scale/zero, fp32 in/out — is
  recognized at compile time and dispatched to a dedicated dequant kernel
  (`make_int8_matmul_fast`, layout `kn`) via `compile_shader`: the int8 tile is staged
  to a float threadgroup tile with the zero folded in, a float simdgroup MMA runs on
  it, and the epilogue applies the per-N scale. Near-bit-exact vs the float reference
  (max abs err ~1e-4 at fp32 accumulate). The recognizer
  (`_maybe_quant_matmul_descriptor`) is **correct-or-refuse**: it verifies the exact
  dequant tree, traces `scale`→arg3 / `zero`→arg4, confirms the layout via
  address-traced strides, and anchors the K-loop bound to the K arg — any deviation
  refuses. **Both weight layouts are routed**: `[K,N]` (natural `tl.dot(a, w)`, layout
  `kn`) and `[N,K]` (GPTQ/AWQ prefill, `out = a @ tl.trans(dequant(w))` — the
  transpose lowers to a `ttg.memdesc_trans`, detected and routed to layout `nk`, so
  prefill and decode share the `[N,K]` weight layout). The path is **fail-closed** end
  to end: a launch that can't use the fast kernel (non-MPS, `compile_shader` off, or
  dims not `M%32 / N%16 / K%32`) is refused, never mis-run. Not yet routed (each
  refuses cleanly): symmetric (no-zero) quant, fp16 in/out, and int4.
- **Matmul role-resolution fixed for kernels with extra ptr args.** In a K-loop matmul
  the A/B pointers are loop-carried (advanced each iteration) so they don't trace back
  to a func-arg — only C (the store target) traces reliably. The role-resolution
  fallback then picked the *last* ptr arg as C: correct by luck for a 3-pointer (A,B,C)
  matmul, but wrong for one with **extra ptr args** (a quantized matmul's scale/zero, or
  a bias), which mis-inferred strides and over-refused it as a "batched matmul". Now the
  reliably-traced C is kept and A/B fall back per-leg.
- **MEPT chained-addptr fix** — an array-of-offsets base pointer (`x_ptr + offs*K`)
  advanced by a *scalar* loop variable (`+ k`, inside a runtime loop) fell through to
  the scalar-offset lowering and emitted an invalid double subscript
  (`base[arr[0]][k]`) that failed to compile. This was always a **loud** failure
  (`MetalCompilationError`), never a silent-wrong. The scalar is now folded into every
  register-array slot. Surfaced by the macOS-26 C++-backend suite; guarded by a new
  default-path regression test with a full red-green cycle.

## 0.1.0a2 (2026-07-01)

The project suite grew **877 → 1,968 passed / 0 failed** over this cycle.

### Portability — verified on NVIDIA silicon

- **The central claim is now measured, not argued**: the same unmodified `@triton.jit`
  kernels were run on an Apple M4 Max (Metal, Triton 3.7.0) and a rented NVIDIA A40
  (CUDA, Triton 3.0.0), each checked against the same NumPy reference. vector-add and
  the fp32/`ieee` matmul produced **bit-identical** outputs across vendors (Δ = 0);
  softmax matched to fp rounding (~1e-9). The one divergence is NVIDIA's **tf32**
  default for fp32 `tl.dot` (6.1e-2 vs an fp64 reference; `input_precision="ieee"` →
  bit-identical). Metal has no tf32, and triton-msl refuses it rather than silently
  approximating. See the new [`PORTABILITY.md`](PORTABILITY.md) and the reproduce
  harness `benchmarks/cross_backend_verify.py`.
- **Runnable local-dev example** — `examples/local_triton_dev.py`: the three canonical
  Triton tutorial kernels (vector-add, fused softmax, tiled matmul) on deliberately
  non-multiple shapes, each verified against NumPy on the Metal GPU, with a regression
  test pinning them.

### Correctness — the anti-silent-wrong campaign

Systematic adversarial audits (multiple independent rounds, plus three new fuzzers)
closed **~75 silent-wrong bugs** across the dot/reduce/store surfaces; the dot and
reduce surfaces are now correct-or-refuse by construction:

- **General address-traced matmul stride inference** — transposed / sliced /
  column-major / pre-transposed-staged operands now compute correctly or refuse
  loudly, independent of variable naming. The three legacy stride mechanisms were
  deleted in favor of the one address-traced inference; twin dot-lowering paths
  (simple-dot vs K-loop epilogue, fragment selection, bias-init detection) were
  de-duplicated so a fix can no longer land in one copy and miss the other.
  Chain-dot / 3-D / batched matmuls and ambiguous tile-vs-batch shapes refuse.
- **Reduce-combine classifier rewritten structurally** — reduce lowering now
  exact-matches the combine region (sum/max/min/and/or/xor, argmax/argmin, Welford)
  instead of substring-sniffing, and refuses anything it cannot prove. Closed the
  unsigned (u8/16/32/64) max/min/argminmax-computed-signed class, i64 coverage
  holes, NaN-propagation handling, and shared-memory races (missing barriers) on
  broadcast reads in 2-D/3-D/N-D reduce and fused argminmax.
- **Masked-store correctness** — matmul/FA templates compute the full tile and clip
  only at the tile boundary, which silently dropped tighter user store-masks. Now:
  FlashAttention head_dim ≤ 64 non-tile-boundary output masks **compute + clip**
  (the honoring generic path); head_dim=128 FA and matmul templates **refuse**; a
  constant mask bound on a multi-block (`pid*BLOCK + arange`) index is correctly
  treated as non-trivial (it clips later blocks) instead of being dropped.
- **fp8 round-to-nearest-even** — the fp32→fp8 downcast rounded half-away-from-zero;
  now RTNE, matching hardware casts.
- **Three systemic gates added** — a 97-cell differential routing-boundary sweep
  (invariant: correct OR loud-refuse, never silent/cryptic, with an OOB canary) and
  matmul/reduce/combination fuzzers run in the suite.

### torch.compile

- **Persistent MSL stash** — the zero-copy dispatch path now survives Inductor
  cache restores: warm GPT-2 small went 50.7 ms → **2.1 ms (~24×)**, faster than
  PyTorch's own native MPS Inductor backend (4.6 ms) and eager (7.3 ms) on M4 Max.
- **NaN-propagating max/min** lower correctly (was refused → broke compiled softmax
  and training). CNN/BatchNorm compile via under-filling persistent-reduction
  configs; product reductions and a 2-D reduce fused with a 2-D scan in one kernel
  are supported; genuinely-impossible tiles (>1024 threads) refuse loudly.

### Performance

- **bf16 fast matmul** — bf16 is now a fast-matmul input via the M-series
  `simdgroup_bfloat8x8` matrix unit (bf16 in + float32 accumulate, fp32/bf16 out),
  ~12 TFLOP/s vs the ~2.4 TFLOP/s generic float-compute fallback (~4.9×). bf16 is
  the dominant training dtype. FlashAttention bf16 stays refused (FA kernel is
  fp16/fp32 only); the fused matmul+softmax bf16 path now uses the bfloat MMA unit.
- **Deterministic occupancy-gated matmul tile selection** — extends the fast path
  to unaligned M (`M%32≠0`): ~3.7–4.8× for large unaligned-M matmuls vs the generic
  path, no-op aligned, never-regress small (`TRITON_MSL_MATMUL_AUTOTUNE=0` opts out).
- **N%32 / N%16 / N%8 fast path** — the tuner's strip-width gate was relaxed so
  unaligned-N shapes keep simdgroup speed (N%16 ~11 TF, N%8 ~5.7 TF, byte-exact),
  closing the N-alignment perf cliff symmetrically with the M-side rescue.
- **simdgroup-MMA FlashAttention** at head_dim=128 (fp32 + fp16, causal + non-causal).
- **`num_stages`** is a documented, honest no-op (pipelining measured not to help on
  Apple — no `cp.async`; the fast paths already overlap load/compute).

### Reporting honesty

- **3 silent-wrongs fixed** (2026-06-21 dual-lens audit): the fp16/bf16 simple-dot
  epilogue raced on a shared threadgroup slot; bf16 FlashAttention at head_dim 32/64
  dispatched wrong (now refused via a dtype gate); a 3D reduce with a pre-reduce op
  (`tl.sum(a*s)`) silently dropped the op (now refused, since both reduce paths
  mis-handle it). Each is regression-tested.
- **Reporting-honesty pass** — fixed a skip-count parser undercount (3,634 → true
  3,782), refreshed the stale conformance ratchet baseline (4,280 → 5,560), corrected
  the fp16 matmul headline label (fp16-in/fp32-out), and stale doc counts.

### Tooling / CI

- Hosted CI gates on **lint + format** (`ruff check` + `ruff format --check`; a
  repo-wide `ruff format` pass landed, verified behavior-preserving by the full
  suite). GPU/Metal correctness is validated locally (hosted macOS runners cannot
  build Triton within their time limit) — documented in CONTRIBUTING.
- sdist no longer ships a broken half-copy of the test suite (tests and repo-only
  dirs are pruned; the wheel was always clean).

## 0.1.0a1 — first PyPI release as `triton-msl` (2026-06-19)

First public release on PyPI: `pip install triton-msl`, `import triton_msl`.

### Rebrand → `triton-msl` (2026-06-19)

- Renamed the project/import package `triton_metal` → `triton_msl` and the PyPI
  distribution `triton-metal` → **`triton-msl`**. The obvious name is taken on PyPI by
  an unrelated project and blocked by PEP 541 confusability; `triton-msl` is a distinct
  stem (MSL = Metal Shading Language, which this backend emits). The `metal`
  backend/device id and every Apple-Metal API term (`Metal*` classes, `metal::`,
  `.metal`, `xcrun metal`) are unchanged. Env vars hard-renamed `TRITON_METAL_*` →
  `TRITON_MSL_*`. Verified regression-free: project suite 787/0, `test_core`
  5,560/0/3,782.

### torch.compile + training via the inductor backend (2026-06-18)

- `torch.compile(model, backend="inductor")` (and `backend="metal"`) routes through
  triton-msl on Python 3.10–3.14, for **inference and training** — AOTAutograd's backward
  graph lowers to ordinary Triton kernels — static and `dynamic=True`. 32/32 torch.compile
  model tests plus the training suite pass. Fixed four latent silent-wrong bugs exposed
  once torch.compile actually ran: a native-MPS device-op-override registration clobber,
  Metal fork-unsafe compile subprocesses corrupting the cache, a cross-graph MSL
  cache-key collision (re-keyed by content hash), and a softmax template arg mis-map
  (now refuses to the generic path). The inductor autotuner is disabled on Metal (noisy
  timing selected miscompiled tiles); an under-filling persistent-reduction config that
  produced corrupt gradients is also structurally filtered out.

### 2D `tt.gather` (2026-06-18)

- 2D `tt.gather` lowers correctly — axis=0 (including ragged row counts) and same-shape
  axis=1 — via full-tile shared staging; the upstream `test_gather[[4,4]->[8,4],0]` case
  now passes. A previously *silent-wrong* 2D path was first closed with a loud refusal,
  then implemented. Oversized tiles (>1024 threads), ragged axis=1, and register-array
  operands are refused loudly, never guessed.

### FlashAttention — large head_dim + integrity hardening (2026-06-17)

- **head_dim 128 @ `BLOCK_M=BLOCK_N=32`** now supported (fp32 + fp16, causal + non-causal).
  A real `@triton.jit` FlashAttention-2 kernel is routed to a new head-dim-tiled FA2 MSL
  template (`make_flash_attention_kernel_tiled`) that chunks the head dimension to fit
  Metal's 32 KB threadgroup budget (the un-tiled lowering hit `OutOfResources` at 128).
  Routing is a prescan detector with **refuse-on-any-ambiguity** — an FA-shaped kernel whose
  pointers/strides/scale can't be resolved unambiguously is refused, never guessed; the
  detector is robust to Triton's `equal_to_1` arg specialization. Stays FA2 (FA3/FA4 are
  Hopper/Blackwell async-hardware co-designs with no Apple analog).
- **Closed a small-block silent-wrong hole:** FlashAttention at `BLOCK_M`/`BLOCK_N` < 32
  silently mis-computed (rows past the first → garbage) for *any* head_dim — including the
  otherwise-supported 32/64 — which the previous head_dim>64-only guard missed. The prescan
  now refuses min-dot-tile-dim < 32. head_dim > 128, other block sizes, and bf16 matmul
  inputs are refused loudly.
- **Skip-list reclaim:** +28 upstream `test_core` passes recovered from over-broad skips
  (a Gluon-tool base-name collision wrongly skipping core `test_cat`/`test_split`, plus
  `test_tl_range_num_stages`, a uint16/fp16 modulus, `test_pointer_arguments[cpu_pinned]`) —
  each verified a real pass, not a loose-assertion false-pass.
- **Tooling:** `scripts/run_upstream_tests.py` now loads `-p conftest_metal`, so the cited
  source-of-truth command reproduces the documented conformance number.

### Phase 4 — zero-copy execution + fast matmul + Phase-5 readiness audit (2026-06-16)

- **Zero-copy MPS execution** via `torch.mps.compile_shader`: routes emitted MSL through
  PyTorch's compiler so kernels run against MPS tensors without the per-launch host
  round-trip. ~10× on memory-bound kernels (vector_add 28 → ~347 GB/s ≈ 64% of the M4 Max
  546 GB/s roof). Flag `TRITON_MSL_COMPILE_SHADER` (default-on, `=0` escape hatch).
- **Fast simdgroup matmul** (`make_simdgroup_matmul_kernel_fast`) dispatched zero-copy for
  aligned MPS matmuls. Measured at 2048³: fp32 ~9.6–11.5 TFLOP/s (~55–62% of the 18.4
  fp32 peak — competitive with MLX/MPS GEMM), fp16 ~7.8–12, fp16-output ~12.3 — vs the
  ~2.8 generic fallback. Float accumulation (precision); fp16 output via a cast epilogue.
  Flag `TRITON_MSL_FAST_MATMUL`; correctness-gated (test_core dot/matmul on==off identical).
  This is **not** MLX-parity (fp16 runs at ~fp32 rate to keep float accumulation); the
  earlier "~13.8 TFLOP/s MLX parity" docstring claim was an overstatement and is corrected.
- **MEPT** multi-element-per-thread register-array model is the default lowering path.
- **Test suite (Triton 3.7.0):** upstream `test_core.py` **5,560 passed / 0 failed /
  3,782 feature-gap skips** (each a loud refusal or HW-impossible); the single source of
  truth is `scripts/run_upstream_tests.py` (`--device cpu`, which loads the `conftest_metal`
  skip plugin), not hand-maintained counts.
  Project suite **754 passed / 0 failed**. FlashAttention causal + non-causal at HEAD_DIM
  32 / 64 / 128 via the **Python/MSL** lowering — the C++ MLIR→LLVM path named in the
  2026-05-30 snapshot below was shelved (AGX compiler blocker; Python/MSL is primary).
- **Phase-5 readiness audit** (dual NVIDIA/Triton + MLX/Apple lens) recorded in
  `docs/audits/2026-06-16-phase5-readiness-audit.md`; remaining pre-1.0 items tracked there.

### Integrity prescan (silent-wrong → loud refusal)

Added a structural integrity contract: when the compiler recognizes a kernel
but cannot lower it correctly, it raises `MetalNonRecoverableError` (surfaced
as a clear error to the user) instead of emitting wrong numbers. The catalog
lives in `GenericLowerer._refuse_unsafe_unsupported_ops`; the underlying
audit method was *classifying* the skip-listed feature-gap tests as
loud-failure-safe vs silent-wrong, and migrating the latter to refusals.

Cases now refused (each was a silent-wrong producer before the guard):

- `tt.dot_scaled` — microscaling matmul; no Apple hardware (`test_scaled_dot`).
- pid-tiled matmul with constexpr-baked M/N (`test_dot_mulbroadcasted`).
- rank ≥ 3 `tt.trans` with a non-identity permutation (`test_trans_4d`).
- rank ≥ 2 `tt.cat` / `tt.join` (`test_cat_nd`).
- `tt.dot` inside a noinline device function (`test_noinline[shared]`).
- `tt.join` result feeding `tt.dot` (`test_join_with_mma`).
- unstructured kernel-level control flow / `cf.cond_br` (`test_nested_if_else_return`).

### False-pass exposed by the `cf.cond_br` refusal

`test_constexpr_if_return` shares the void-early-return shape of
`test_nested_if_else_return`; before the guard the legacy parser had been
emitting `Out = pid + 0` for it — dropping the `atomic_add`, dropping the
early return, writing out of bounds — but the test only asserts `out >= 0`,
which `pid + 0` happens to satisfy. The garbage went undetected. Now
correctly refused; skip-listed with that rationale. *A passing test is not
the same as a correct kernel.*

### GPU-hang root cause

`test_dot_max_num_imprecise_acc` was investigated for a per-config hang and
*ruled out* as a per-kernel defect: each config either runs correctly or
raises `OutOfResources` cleanly. The apparent hang was reproduced as **GPU
driver-state accumulation** from many back-to-back fp8 dispatches in a
single process — the same environmental class as concurrent test sweeps.
Documented; not a code fix.

### Test suite (as of 2026-05-30, fresh cache, Python 3.14)

- `test_core.py` (upstream Triton): **4,326 passed / 5,016 skipped / 0 failed**.
- Project suite (codegen / GPU correctness / integration / FlashAttention /
  MLX, excluding torch.compile suites blocked on Py 3.14): **507 passed /
  0 failed**.
- FlashAttention: 11/11 at HEAD_DIM=32 (via the C++ MLIR→LLVM path).
- MLX backend: 15/15.
- `torch.compile` suites (32 + 9 tests) are environment-blocked on Python
  3.14 — PyTorch's own platform guard refuses `torch.compile` on 3.14;
  honest skips with `skipif(py>=3.14)`, will auto-lift when PyTorch ships
  3.14 Dynamo support.

### Hardware profiling harness (WS0/C6)

- Added `benchmarks/hw_harness.py` + `triton_msl/profiling/roofline.py` +
  `triton_msl/profiling/disasm.py`: per-kernel GPU-timestamp timing →
  roofline classification (% of the M4 Max 546 GB/s memory roof / estimated
  compute roof, memory- vs compute-bound), pipeline-reflection occupancy,
  best-effort native-AGX disassembly, and an MLX comparison ratio. Emits
  per-kernel JSON + summary.md + baseline.json. This is the empirical
  backbone for the WS1 perf work ("optimal bounds = saturate the limiting
  counter"). First run surfaced a concrete target: reduce_sum at ~16% of the
  bandwidth roof / 1.3x slower than MLX, vs vector_add (72% of roof, 0.89x
  MLX) and silu (48%, 0.59x MLX).
- Vendored `applegpu` (dougallj — REFERENCES.md [11]) under
  `third_party/applegpu/` for native-AGX disassembly. Honest scope: live GPU
  counters (ALU%/occupancy/registers) are NOT programmatically available on
  Apple Silicon (the device vends only the `timestamp` counter set), and
  applegpu is M1-era so M4/AGX2 disassembly is partial (the harness reports a
  decode-coverage %). `docs/INSTRUMENTS.md` documents the Xcode-capture /
  Instruments path for the counters the programmatic API can't provide. No
  Swift counter-helper was built — it would hit the same Metal API wall.
- Tests: `tests/test_roofline.py` (9), `tests/test_disasm.py` (6, incl.
  fat-header parsing for the 0xCBFEBABE GPU-archive magic + graceful
  degradation).

### Documentation & roadmap

- Added `REFERENCES.md` and `CITING.md` (citations for Triton [1],
  FlashAttention v1/v2 [4,5], online softmax [6], MLX [7], Asahi/`applegpu`
  [10,11], MSL spec [8], PyTorch Inductor [12], M4 Max hardware [13]).
- Added `docs/superpowers/specs/2026-05-30-triton-msl-roadmap.md`
  (umbrella roadmap: WS0 foundation, WS1 the register-array spine,
  WS2 orthogonal-refusal cleanup, WS3 experimental sub-AIR AGX).
- Added `docs/superpowers/specs/2026-05-30-ws0-foundation-design.md`
  (the first workstream's full design — documentation truth, citations,
  test hygiene, C++ build hardening, integrity single source of truth,
  hardware profiling + disassembly harness).
- `docs/ARCHITECTURE.md` reconciled: corrected stale upstream-test numbers,
  added integrity-model section, refusal catalog, MEPT experimental
  charter, scoped FlashAttention claim to HEAD_DIM=32.

## 0.1.0-alpha (2026-03-10)

First public alpha release of triton-msl.

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
- API: `triton_msl.mlx.triton_call(kernel_fn, *args, grid=(...), **constexpr_kwargs)`

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
