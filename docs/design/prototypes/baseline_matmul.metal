// Baseline matmul kernel for copy-simdgroup TMA-inspired perf prototype.
//
// Problem: C = A @ B  with A (M x K), B (K x N), C (M x N), all float32 row-major.
// Launch: one threadgroup per 96x96 output tile. 128 threads = 4 simdgroups.
// Tile:  BLOCK_M = 96, BLOCK_N = 96, BLOCK_K = 32.
// All 4 simdgroups participate in both cooperative loading AND compute.
//
// Column partition across SGs: each SG owns 96/4 = 24 cols = 3 frag columns (8 each).
// Each SG iterates all 12 row-frags (96/8) and does K/8 = 4 inner MMAs per K-step.
// Total MMAs per SG per K-iter: 3 * 12 = 36.
//
// Expected dispatch (driver): grid = (ceil(N/96), ceil(M/96), 1), tg = (128,1,1).

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#define BM 96u
#define BN 96u
#define BK 16u       // matches copysg variant for a fair compute-per-K-iter comparison
#define TG_THREADS 128u
#define NUM_SG 4u
#define SG_COLS 24u   // BN / NUM_SG
#define SG_COL_FRAGS 3u  // SG_COLS / 8
#define ROW_FRAGS 12u    // BM / 8
#define K_INNER 2u       // BK / 8

kernel void baseline_matmul(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device       float* C [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& N      [[buffer(4)]],
    constant uint& K      [[buffer(5)]],
    uint3 pid             [[threadgroup_position_in_grid]],
    uint  sgitg           [[simdgroup_index_in_threadgroup]],
    uint  tiitg           [[thread_index_in_threadgroup]])
{
    const uint row_base = pid.y * BM;
    const uint col_base = pid.x * BN;

    // Threadgroup staging for A (BM x BK) and B (BK x BN)
    threadgroup float tg_A[BM * BK];
    threadgroup float tg_B[BK * BN];

    // 12 row-frags * 3 col-frags = 36 accumulators per SG
    simdgroup_float8x8 acc[ROW_FRAGS][SG_COL_FRAGS];
    for (uint r = 0; r < ROW_FRAGS; ++r)
        for (uint c = 0; c < SG_COL_FRAGS; ++c)
            acc[r][c] = simdgroup_float8x8(0);

    simdgroup_float8x8 a_frag, b_frag[SG_COL_FRAGS];

    const uint sg_col_base = sgitg * SG_COLS;  // 0, 24, 48, 72

    for (uint kk = 0; kk < K; kk += BK) {
        // Cooperative load of A[row_base .. row_base+BM, kk .. kk+BK]
        // Total elems = BM*BK = 96*32 = 3072; 128 threads -> 24 elems per thread
        for (uint i = tiitg; i < BM * BK; i += TG_THREADS) {
            uint r = i / BK;
            uint c = i % BK;
            uint gr = row_base + r;
            uint gc = kk + c;
            tg_A[i] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        // Cooperative load of B[kk .. kk+BK, col_base .. col_base+BN]
        // Total elems = BK*BN = 32*96 = 3072; 24 elems per thread
        for (uint i = tiitg; i < BK * BN; i += TG_THREADS) {
            uint r = i / BN;
            uint c = i % BN;
            uint gr = kk + r;
            uint gc = col_base + c;
            tg_B[i] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Inner MMA over BK in steps of 8
        for (uint kf = 0; kf < K_INNER; ++kf) {
            // Load 3 B fragments for this SG's column slice
            for (uint cf = 0; cf < SG_COL_FRAGS; ++cf) {
                // b_frag is 8 rows x 8 cols starting at row (kf*8), col (sg_col_base + cf*8)
                simdgroup_load(b_frag[cf],
                               tg_B + (kf * 8u) * BN + sg_col_base + cf * 8u,
                               BN);
            }
            for (uint rf = 0; rf < ROW_FRAGS; ++rf) {
                // a_frag is 8 rows x 8 cols at row (rf*8), col (kf*8)
                simdgroup_load(a_frag,
                               tg_A + (rf * 8u) * BK + kf * 8u,
                               BK);
                for (uint cf = 0; cf < SG_COL_FRAGS; ++cf) {
                    simdgroup_multiply_accumulate(acc[rf][cf], a_frag, b_frag[cf], acc[rf][cf]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store accumulators to C
    for (uint rf = 0; rf < ROW_FRAGS; ++rf) {
        for (uint cf = 0; cf < SG_COL_FRAGS; ++cf) {
            uint gr = row_base + rf * 8u;
            uint gc = col_base + sg_col_base + cf * 8u;
            // Assume M, N divisible by 8 for this benchmark prototype
            simdgroup_store(acc[rf][cf], C + gr * N + gc, N);
        }
    }
}
