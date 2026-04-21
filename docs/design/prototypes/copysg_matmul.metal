// Copy-simdgroup (TMA-inspired) matmul kernel.
//
// Problem: C = A @ B  with A (M x K), B (K x N), C (M x N), all float32 row-major.
// Launch: one threadgroup per 96x96 output tile. 128 threads = 4 simdgroups.
// Tile:  BLOCK_M = 96, BLOCK_N = 96, BLOCK_K = 32.
//
// SG0 is dedicated to global->threadgroup prefetch of the NEXT K-tile.
// SG1-SG3 do compute on the CURRENT K-tile. Threadgroup memory is double-buffered.
//
// Column partition across compute SGs: 96/3 = 32 cols = 4 frag columns each.
// Each compute SG iterates 12 row-frags and does K/8 = 4 inner MMAs per K-step.
// Total MMAs per compute SG per K-iter: 4 * 12 = 48.
//
// Expected dispatch (driver): grid = (ceil(N/96), ceil(M/96), 1), tg = (128,1,1).

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#define BM 96u
#define BN 96u
#define BK 16u               // reduced from 32 to fit double-buffer under 32KB threadgroup limit
#define TG_THREADS 128u
#define NUM_SG 4u
#define COPY_SG 0u
#define COMPUTE_SG_COUNT 3u
#define SG_COLS 32u          // BN / COMPUTE_SG_COUNT = 96 / 3
#define SG_COL_FRAGS 4u      // SG_COLS / 8
#define ROW_FRAGS 12u        // BM / 8
#define K_INNER 2u           // BK / 8
#define COPY_THREADS 32u     // one simdgroup

kernel void copysg_matmul(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device       float* C [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& N      [[buffer(4)]],
    constant uint& K      [[buffer(5)]],
    uint3 pid             [[threadgroup_position_in_grid]],
    uint  sgitg           [[simdgroup_index_in_threadgroup]],
    uint  tiisg           [[thread_index_in_simdgroup]],
    uint  tiitg           [[thread_index_in_threadgroup]])
{
    const uint row_base = pid.y * BM;
    const uint col_base = pid.x * BN;

    // Double-buffered threadgroup storage for A and B
    threadgroup float tg_A0[BM * BK];
    threadgroup float tg_A1[BM * BK];
    threadgroup float tg_B0[BK * BN];
    threadgroup float tg_B1[BK * BN];

    threadgroup float* tg_A_cur = tg_A0;
    threadgroup float* tg_A_nxt = tg_A1;
    threadgroup float* tg_B_cur = tg_B0;
    threadgroup float* tg_B_nxt = tg_B1;

    // Per-compute-SG accumulators
    simdgroup_float8x8 acc[ROW_FRAGS][SG_COL_FRAGS];
    for (uint r = 0; r < ROW_FRAGS; ++r)
        for (uint c = 0; c < SG_COL_FRAGS; ++c)
            acc[r][c] = simdgroup_float8x8(0);

    simdgroup_float8x8 a_frag, b_frag[SG_COL_FRAGS];

    // Compute-SG index 0..2 and its column base
    const bool is_compute_sg = (sgitg != COPY_SG);
    const uint compute_sg_idx = sgitg - 1u;             // valid only if is_compute_sg
    const uint sg_col_base = compute_sg_idx * SG_COLS;  // 0, 32, 64

    // =============================
    // Prologue: copy SG fills the first K-tile into tg_*_cur.
    // All SGs participate in the first fill to amortize the prologue cost
    // (otherwise 32 threads loading 3072 elts = 96 loads/thread before compute).
    // =============================
    {
        uint kk0 = 0;
        for (uint i = tiitg; i < BM * BK; i += TG_THREADS) {
            uint r = i / BK;
            uint c = i % BK;
            uint gr = row_base + r;
            uint gc = kk0 + c;
            tg_A_cur[i] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (uint i = tiitg; i < BK * BN; i += TG_THREADS) {
            uint r = i / BN;
            uint c = i % BN;
            uint gr = kk0 + r;
            uint gc = col_base + c;
            tg_B_cur[i] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // =============================
    // Main K loop with prefetch overlap.
    // Iteration k: compute on tg_*_cur (tile k), copy SG prefetches tile k+1 into tg_*_nxt.
    // =============================
    const uint K_TILES = K / BK;
    for (uint kt = 0; kt < K_TILES; ++kt) {
        const uint kk_next = (kt + 1u) * BK;
        const bool has_next = (kt + 1u) < K_TILES;

        if (sgitg == COPY_SG) {
            // --- Copy simdgroup: prefetch next tile ---
            if (has_next) {
                // 3072 A elts / 32 threads = 96 loads per thread (unrolled by stride)
                for (uint i = tiisg; i < BM * BK; i += COPY_THREADS) {
                    uint r = i / BK;
                    uint c = i % BK;
                    uint gr = row_base + r;
                    uint gc = kk_next + c;
                    tg_A_nxt[i] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
                }
                for (uint i = tiisg; i < BK * BN; i += COPY_THREADS) {
                    uint r = i / BN;
                    uint c = i % BN;
                    uint gr = kk_next + r;
                    uint gc = col_base + c;
                    tg_B_nxt[i] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
                }
            }
        } else {
            // --- Compute simdgroups: work on current tile ---
            for (uint kf = 0; kf < K_INNER; ++kf) {
                for (uint cf = 0; cf < SG_COL_FRAGS; ++cf) {
                    simdgroup_load(b_frag[cf],
                                   tg_B_cur + (kf * 8u) * BN + sg_col_base + cf * 8u,
                                   BN);
                }
                for (uint rf = 0; rf < ROW_FRAGS; ++rf) {
                    simdgroup_load(a_frag,
                                   tg_A_cur + (rf * 8u) * BK + kf * 8u,
                                   BK);
                    for (uint cf = 0; cf < SG_COL_FRAGS; ++cf) {
                        simdgroup_multiply_accumulate(acc[rf][cf], a_frag, b_frag[cf], acc[rf][cf]);
                    }
                }
            }
        }

        // Sync: wait for both compute and (if applicable) copy to finish.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Swap buffers (cheap pointer swap; every thread does it identically).
        if (has_next) {
            threadgroup float* tmpA = tg_A_cur; tg_A_cur = tg_A_nxt; tg_A_nxt = tmpA;
            threadgroup float* tmpB = tg_B_cur; tg_B_cur = tg_B_nxt; tg_B_nxt = tmpB;
        }
    }

    // =============================
    // Store: only compute simdgroups have valid accumulators.
    // =============================
    if (is_compute_sg) {
        for (uint rf = 0; rf < ROW_FRAGS; ++rf) {
            for (uint cf = 0; cf < SG_COL_FRAGS; ++cf) {
                uint gr = row_base + rf * 8u;
                uint gc = col_base + sg_col_base + cf * 8u;
                simdgroup_store(acc[rf][cf], C + gr * N + gc, N);
            }
        }
    }
}
