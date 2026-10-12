// QSA projection phases (q tiles, k/v/indexer rows, attn_output) as device phases. dense_tc.cu is included with its
// external symbols renamed and its Fmt renamed (DtcFmt) so it can share a TU with the MoE and IQ kernels.
#pragma once
#include "basalt/kernels/dense_tc.hpp"
#include "basalt/kernels/pdl.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "basalt/ggml/ggml_common.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "qsa_full_rename.h"
#define Fmt DtcFmt
#include "dense_tc.cu"
#undef dense_tc
#undef dense_tc_supported
#undef dense_tc_type_on
#undef dense_tc_pair
#undef dense_tc_pair_supported
#undef dense_tc_pair_mask
#undef dense_tc_l2_prefetch
#undef dense_tc_mask
namespace basalt::kernels {
namespace {

__device__ __forceinline__ void qp_named_bar(int id) { asm volatile("bar.sync %0, 256;" ::"r"(id) : "memory"); }
__device__ __forceinline__ void qp_mb_wait(uint32_t bb, uint32_t par) {
    uint32_t done = 0;
    do {
        asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                     " selp.u32 %0, 1, 0, p;\n}" : "=r"(done) : "r"(bb), "r"(par) : "memory");
    } while (!done);
}
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
template<bool CG>
__device__ __forceinline__ void qp_load_x(XFrag& xf, const uint8_t* __restrict__ x, int c, int hb, int g, int t, int nb32,
                                          int ncols) {
#pragma unroll
    for (int k4 = 0; k4 < 4; ++k4) {
        const int k = 16 * c + 4 * hb + k4;
        xf.b0[k4] = xf.b1[k4] = 0;
        if (g < ncols) {
            const uint8_t* xb = x + ((size_t) g * nb32 + k) * kQ81;
            xf.b0[k4] = CG ? __ldcg(reinterpret_cast<const unsigned int*>(xb + 4) + t) : __ldg(reinterpret_cast<const unsigned int*>(xb + 4) + t);
            xf.b1[k4] = CG ? __ldcg(reinterpret_cast<const unsigned int*>(xb + 20) + t) : __ldg(reinterpret_cast<const unsigned int*>(xb + 20) + t);
        }
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int col = 2 * t + e;
            xf.d8[k4][e] = col < ncols ? __low2float(CG ? __ldcg(reinterpret_cast<const half2*>(
                                             x + ((size_t) col * nb32 + k) * kQ81)) : __ldg(reinterpret_cast<const half2*>(
                                             x + ((size_t) col * nb32 + k) * kQ81)))
                                       : 0.0f;
        }
    }
}

#endif

constexpr int QP_RED = 8 * 16 * kMaxCols * 4;   // one 8-warp group's warp sums
template<int TYPE> constexpr int qp_tile_smem(int n_in) { return 128 + 2 * QP_RED + 2 * 16 * (n_in / 256) * Fmt<TYPE>::SB; }
template<int TYPE> constexpr int qp_one_smem(int n_in) {
    return 128 + std::max(24 * 16 * kMaxCols * 4, 16 * one_row_stride((n_in / 256) * Fmt<TYPE>::SB));
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
// smem: qp_tile_smem bytes.  Tiles G, G + nG, ... of group G = 2 * blockIdx.x + (warp / 8).
template<int TYPE>
__device__ void qp_tile_phase(uint8_t* smem, const uint8_t* __restrict__ w, const uint8_t* __restrict__ x,
                              float* __restrict__ y, int n_in, int n_out, int ncols) {
    constexpr int SB = Fmt<TYPE>::SB, CB = 2 * SB;
    const int tid = threadIdx.x, wpf = tid >> 5, gi = wpf >> 3, wp = wpf & 7, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int ltid = tid & 255;
    const int hb = wp & 3, sbi = hb >> 1, n = hb & 1;
    const int nchunk = n_in / 512, nb32 = n_in / 32, row_bytes = (n_in / 256) * SB, tb = 16 * row_bytes;
    const int ntiles = (n_out + 15) / 16, G = (int) blockIdx.x * 2 + gi, nG = (int) gridDim.x * 2;
    const int mine = G < ntiles ? (ntiles - 1 - G) / nG + 1 : 0;
    uint64_t* bar = reinterpret_cast<uint64_t*>(smem) + gi;
    float* red = reinterpret_cast<float*>(smem + 128 + gi * QP_RED);
    uint8_t* slot = smem + 128 + 2 * QP_RED + (size_t) gi * tb;
    const uint32_t bb = (uint32_t) __cvta_generic_to_shared(bar);
    auto issue = [&](int i) {   // warp 0 of the group, one bulk copy of the 16-row tile
        const int tile = G + i * nG, rows = min(16, n_out - tile * 16);
        const uint32_t bytes = (uint32_t) ((rows * row_bytes + 15) & ~15);
        if (lane == 0) {
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bb), "r"(bytes) : "memory");
            asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                         ::"r"((uint32_t) __cvta_generic_to_shared(slot)), "l"(w + (size_t) tile * tb), "r"(bytes),
                         "r"(bb) : "memory");
        }
    };
    if (wp == 0 && lane == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(bb));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (wp == 0 && mine > 0) issue(0);
    const uint32_t* tbl = reinterpret_cast<const uint32_t*>(dtc_iq4nl);
    uint32_t t0 = 0, t1 = 0, t2 = 0, t3 = 0;
    if constexpr (TYPE == 23) { t0 = tbl[0]; t1 = tbl[1]; t2 = tbl[2]; t3 = tbl[3]; }
    for (int i = 0; i < mine; ++i) {
        const int row0 = (G + i * nG) * 16;
        qp_mb_wait(bb, (uint32_t) (i & 1));
        const uint8_t* rowA = slot + g * row_bytes + sbi * SB;
        const uint8_t* rowB = slot + (g + 8) * row_bytes + sbi * SB;
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int c = wp >> 2; c < nchunk; c += 2) {
            XFrag xf;
            qp_load_x<false>(xf, x, c, hb, g, t, nb32, ncols);
            const uint8_t* const p[2] = {rowA + c * CB, rowB + c * CB};
            half_sb_terms<TYPE>(acc, p, n, t, xf, t0, t1, t2, t3, nullptr);
        }
        float* rw = red + wp * 16 * kMaxCols;
        rw[g * kMaxCols + 2 * t] = acc[0];
        rw[g * kMaxCols + 2 * t + 1] = acc[1];
        rw[(g + 8) * kMaxCols + 2 * t] = acc[2];
        rw[(g + 8) * kMaxCols + 2 * t + 1] = acc[3];
        qp_named_bar(1 + gi);   // the slot is read and the warp sums are in
        if (wp == 0 && i + 1 < mine) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            issue(i + 1);
        }
        if (ltid < 16 * kMaxCols) {
            const int row = ltid / kMaxCols, col = ltid % kMaxCols;
            if (col < ncols && row0 + row < n_out) {
                const float* rr = red + row * kMaxCols + col;
                constexpr int WS = 16 * kMaxCols;
                y[(size_t) col * n_out + row0 + row] = ((rr[0] + rr[4 * WS]) + (rr[WS] + rr[5 * WS])) +
                                                       ((rr[2 * WS] + rr[6 * WS]) + (rr[3 * WS] + rr[7 * WS]));
            }
        }
        qp_named_bar(1 + gi);   // the sums are read
    }
    if (wp == 0 && lane == 0) asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(bb) : "memory");
}

// tile blockIdx.x of 16 rows with dense_tc_one_block's 24 virtual warps; smem: qp_one_smem bytes
template<int TYPE>
__device__ void qp_one_phase(uint8_t* smem, const uint8_t* __restrict__ w, const uint8_t* __restrict__ x,
                             float* __restrict__ y, int n_in, int n_out, int ncols) {
    constexpr int SB = Fmt<TYPE>::SB, CB = 2 * SB, NW = 24, NG = 6;
    const int blk = blockIdx.x;
    if (blk * 16 >= n_out) return;
    uint64_t* bar = reinterpret_cast<uint64_t*>(smem);
    uint8_t* slot = smem + 128;
    float* red = reinterpret_cast<float*>(slot);
    const int tid = threadIdx.x, pw = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int nchunk = n_in / 512, nb32 = n_in / 32;
    const int row_bytes = (n_in / 256) * SB, rs = one_row_stride(row_bytes);
    const int row0 = blk * 16, rows = min(16, n_out - row0);
    const uint32_t bb = (uint32_t) __cvta_generic_to_shared(bar);
    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(bb));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bb), "r"((uint32_t) (rows * row_bytes))
                     : "memory");
    }
    __syncthreads();
    if (tid < rows)
        asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                     ::"r"((uint32_t) __cvta_generic_to_shared(slot + tid * rs)),
                     "l"(w + (size_t) (row0 + tid) * row_bytes), "r"((uint32_t) row_bytes), "r"(bb) : "memory");
    const uint32_t* tbl = reinterpret_cast<const uint32_t*>(dtc_iq4nl);
    uint32_t t0 = 0, t1 = 0, t2 = 0, t3 = 0;
    if constexpr (TYPE == 23) { t0 = tbl[0]; t1 = tbl[1]; t2 = tbl[2]; t3 = tbl[3]; }
    qp_mb_wait(bb, 0u);
    float acc[2][4];
#pragma unroll
    for (int v = 0; v < 2; ++v) {
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[v][q] = 0.0f;
        const int vw = pw + 16 * v;
        if (vw < NW) {
            const int hb = vw & 3, sbi = hb >> 1, n = hb & 1;
            const uint8_t* rowA = slot + g * rs + sbi * SB;
            const uint8_t* rowB = slot + (g + 8) * rs + sbi * SB;
            for (int c = vw >> 2; c < nchunk; c += NG) {
                XFrag xf;
                qp_load_x<true>(xf, x, c, hb, g, t, nb32, ncols);
                const uint8_t* const p[2] = {rowA + c * CB, rowB + c * CB};
                half_sb_terms<TYPE>(acc[v], p, n, t, xf, t0, t1, t2, t3, nullptr);
            }
        }
    }
    __syncthreads();   // the rows are read
    if (tid == 0) asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(bb) : "memory");
#pragma unroll
    for (int v = 0; v < 2; ++v) {
        const int vw = pw + 16 * v;
        if (vw < NW) {
            float* rw = red + vw * 16 * kMaxCols;
            rw[g * kMaxCols + 2 * t] = acc[v][0];
            rw[g * kMaxCols + 2 * t + 1] = acc[v][1];
            rw[(g + 8) * kMaxCols + 2 * t] = acc[v][2];
            rw[(g + 8) * kMaxCols + 2 * t + 1] = acc[v][3];
        }
    }
    __syncthreads();
    if (tid < 16 * kMaxCols) {
        const int row = tid / kMaxCols, col = tid % kMaxCols;
        if (col < ncols && row0 + row < n_out) {
            constexpr int WS = 16 * kMaxCols;
            const float* rr = red + row * kMaxCols + col;
            float h[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                float v = rr[q * WS];
#pragma unroll
                for (int j = 1; j < NG; ++j) v += rr[(4 * j + q) * WS];
                h[q] = v;
            }
            y[(size_t) col * n_out + row0 + row] = (h[0] + h[1]) + (h[2] + h[3]);
        }
    }
}
#endif
}  // namespace
}  // namespace basalt::kernels
#undef Fmt
#include "qsa_full_rows.cuh"
#include "basalt/kernels/bf16_gemv.hpp"
namespace basalt::kernels {
namespace mkb {

constexpr int QR_K = 512, QR_V = 512, QR_IK = 128, QR_IQ = 512, QR_ROWS = QR_K + QR_V + QR_IK + QR_IQ;

struct RowArgs {
    const uint8_t* wk; const uint8_t* wv; const uint16_t* wik; const uint16_t* wiq;
    const block_q8_1* xq;      // [T][80] q8_1 blocks of the window's mixed rows
    const float* mixed;        // [T][2560]
    float* kcur; float* vcur; float* idx_raw; float* qidx;   // [T][512], [T][512], [T][128], [T][512]
    int T;
};

template<int TY, int TM>
__device__ __forceinline__ void proj_row(const uint8_t* w, int r, const block_q8_1* xq, int T, float* o, int os, int lane) {
    if constexpr (TY == 23) sh_gate_row<TM>(w, r, xq, T, o + r, os, lane);
    else shx_row<TY, TM>(w + (size_t) r * (N / 256 * ShxF<TY>::BB), N / 256, xq, XQB, T, o + r, os, lane);
}

// bf16_f32_mmvf_multi_kernel<256>: thread t (warp v = t / 32) takes pairs t + 256 m, m < 5, two ordered fmas per pair;
// each token's warp sums, then the 8 warp sums through one more butterfly (24 zeros)
template<int TM>
__device__ __forceinline__ void idx_row(const uint16_t* wi, const float* x, int ldx, int T, float* o, int os, int lane) {
    const uint32_t* w2 = (const uint32_t*) wi;
    float P[TM];
#pragma unroll
    for (int t = 0; t < TM; ++t) P[t] = 0.0f;
#pragma unroll 1
    for (int v = 0; v < 8; ++v) {
        float acc[TM];
#pragma unroll
        for (int t = 0; t < TM; ++t) acc[t] = 0.0f;
#pragma unroll
        for (int m = 0; m < 5; ++m) {
            const int p = lane + 32 * v + 256 * m;
            const uint32_t ww = __ldg(w2 + p);
#pragma unroll
            for (int t = 0; t < TM; ++t)
                if (t < T) {
                    const float2 in = __ldcg(reinterpret_cast<const float2*>(x + (size_t) t * ldx) + p);
                    acc[t] = fma_ftz(__uint_as_float(ww << 16), in.x, acc[t]);
                    acc[t] = fma_ftz(__uint_as_float(ww & 0xffff0000u), in.y, acc[t]);
                }
        }
#pragma unroll
        for (int t = 0; t < TM; ++t) {
            const float s = wsum_ftz(acc[t]);
            if (lane == v) P[t] = s;
        }
    }
#pragma unroll
    for (int t = 0; t < TM; ++t) {
        const float r = wsum_ftz(P[t]);
        if (t < T && lane == 0) o[t * os] = r;
    }
}

// the phase: warp-rows gw, gw + 2720, ... of the 1664 projection rows (K, V, indexer k, indexer q)
template<int TYK, int TYV, int TM>
__device__ void rows_phase(const RowArgs& a) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int gw = blockIdx.x * 16 + warp, nw = gridDim.x * 16;
    for (int row = gw; row < QR_ROWS; row += nw) {
        if (row < QR_K) proj_row<TYK, TM>(a.wk, row, a.xq, a.T, a.kcur, QR_K, lane);
        else if (row < QR_K + QR_V) proj_row<TYV, TM>(a.wv, row - QR_K, a.xq, a.T, a.vcur, QR_V, lane);
        else if (row < QR_K + QR_V + QR_IK) {
            const int r = row - QR_K - QR_V;
            idx_row<TM>(a.wik + (size_t) r * N, a.mixed, N, a.T, a.idx_raw + r, QR_IK, lane);
        } else {
            const int r = row - QR_K - QR_V - QR_IK;
            idx_row<TM>(a.wiq + (size_t) r * N, a.mixed, N, a.T, a.qidx + r, QR_IQ, lane);
        }
    }
}
}  // namespace mkb
}  // namespace basalt::kernels
