// src/kernels/cuda/dense_tc.cu - see include/strata/kernels/dense_tc.hpp.
//
// A tile is 16 MT rows; its 512-value chunks (two super-blocks of each row) are one ring slot each.  Warp w of a
// tile takes m-tile (w / 4) % MT (16 rows) and half super-block w % 4 of every chunk (4 blocks of 32 values, whose
// weight words it loads once: Q6_K and Q5_K share their high bits, Q4_K / Q5_K pairs share their bytes).  One mma per
// block (two for Q6_K's 16-value scales) gives the exact integer dots of 16 rows x 8 columns, and each lane adds its
// four (row, column) terms in a fixed order: blocks in ascending order within the half super-block, chunks in order.
// The four warps of an m-tile are then added in order.  The result does not depend on MT.  KS = 2: the chunk is 1024
// values and warp group w / (4 MT) takes its half; the groups' sums are added in order after their own.
//
// Persistent: block b takes tiles b, b + G, ...; their chunks stream through one ring of S slots (16-byte cp.async
// per thread, issued S - 1 chunks ahead across tile boundaries, the first ones before the grid dependency wait:
// weights are constant).  The q8_1 columns are read through L1.
#include "strata/kernels/dense_tc.hpp"
#include "strata/kernels/pdl.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int kMaxCols = 8;
constexpr int kQ81 = 36;   // bytes of a q8_1 block: half2 (d, d * sum), 32 codes

template<int TYPE> struct Fmt;
template<> struct Fmt<12> { static constexpr int SB = 144; };
template<> struct Fmt<13> { static constexpr int SB = 176; };
template<> struct Fmt<14> { static constexpr int SB = 210; };
template<> struct Fmt<23> { static constexpr int SB = 136; };
template<> struct Fmt<21> { static constexpr int SB = 110; };

// a staged row of a slot: two super-blocks from the 16-byte unit holding their first byte, padded to an odd number
// of 16-byte units so the eight rows a load touches fall in different banks
__host__ __device__ constexpr int row_stride(int sb) {
    const int r = (2 * sb + 15 + 15) & ~15;
    return ((r / 16) & 1) ? r : r + 16;
}

// a row's bytes rounded up to an odd number of 16-byte units (row_bytes is a multiple of 16)
__host__ __device__ constexpr int one_row_stride(int row_bytes) {
    return ((row_bytes / 16) & 1) ? row_bytes : row_bytes + 16;
}

__device__ __align__(4) int8_t dtc_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                                1,    13,   25,  38,  53,  69,  89,  113};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
__device__ __forceinline__ void mma_k32(int (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                        uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(0));
}
__device__ __forceinline__ void mma_k16(int (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a0), "r"(a1), "r"(b0), "r"(0));
}
__device__ __forceinline__ uint32_t ld32(const uint8_t* p) { return *reinterpret_cast<const uint32_t*>(p); }
__device__ __forceinline__ uint32_t ld32_b2(const uint8_t* p) {   // 2-byte aligned
    const uint16_t* q = reinterpret_cast<const uint16_t*>(p);
    return (uint32_t) q[0] | ((uint32_t) q[1] << 16);
}
__device__ __forceinline__ int byte_of(uint32_t w, int b) { return (int) ((w >> (8 * b)) & 0xffu); }
__device__ __forceinline__ int sbyte_of(uint32_t w, int b) { return (int) (int8_t) (uint8_t) (w >> (8 * b)); }
// eight 4-bit codes to their int8 codebook values: .x the low nibbles, .y the high ones
__device__ __forceinline__ uint2 iq4_lookup(uint32_t q4, uint32_t t0, uint32_t t1, uint32_t t2, uint32_t t3) {
    const uint32_t sel = 0x32103210 | ((q4 & 0x88888888) >> 1);
    const uint32_t r0 = __byte_perm(__byte_perm(t0, t1, q4), __byte_perm(t2, t3, q4), sel);
    const uint32_t r1 = __byte_perm(__byte_perm(t0, t1, q4 >> 16), __byte_perm(t2, t3, q4 >> 16), sel >> 16);
    return make_uint2(__byte_perm(r0, r1, 0x6420), __byte_perm(r0, r1, 0x7531));
}

// IQ3_S: the grid word g with byte j negated where bit j of nib is set
__device__ __forceinline__ uint32_t iq3s_signed(uint32_t g, uint32_t nib) {
    const uint32_t b = (nib * 0x204081u) & 0x01010101u;   // bit j of nib to bit 0 of byte j
    const uint32_t m = (b << 8) - b;                      // 0xff in those bytes
    return (g ^ m) + b;
}

// The column data of a warp's four blocks: column g's codes 4t.. (b0) and 16 + 4t.. (b1), and the q8_1 scales of
// columns 2t and 2t + 1.
struct XFrag {
    uint32_t b0[4], b1[4];
    float d8[4][2];
};

// Half super-block n (blocks 4n..4n + 3) of the super-blocks at p[0] (row g) and p[1] (row g + 8): the four
// (row, column) terms added to acc.
template<int TYPE>
__device__ __forceinline__ void half_sb_terms(float (&acc)[4], const uint8_t* const (&p)[2], int n, int t,
                                              const XFrag& x, uint32_t t0, uint32_t t1, uint32_t t2, uint32_t t3,
                                              const uint32_t* grid = nullptr) {
    if constexpr (TYPE == 12 || TYPE == 13) {
        uint32_t qw[2][4], hw[2][2], sw[2][3];
        float2 dm[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint8_t* qs = p[h] + (TYPE == 12 ? 16 : 48) + 64 * n;
#pragma unroll
            for (int i = 0; i < 4; ++i) qw[h][i] = ld32(qs + 16 * i + 4 * t);   // pairs 2n, 2n + 1: words 4t, 16 + 4t
            if constexpr (TYPE == 13) {
                hw[h][0] = ld32(p[h] + 16 + 4 * t);
                hw[h][1] = ld32(p[h] + 32 + 4 * t);
            }
#pragma unroll
            for (int i = 0; i < 3; ++i) sw[h][i] = ld32(p[h] + 4 + 4 * i);
            dm[h] = __half22float2(*reinterpret_cast<const half2*>(p[h]));
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int j = 4 * n + i, pp = i >> 1, sh = 4 * (i & 1);
            uint32_t lo[2], hi[2];
            float f1[2], f2[2];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                lo[h] = (qw[h][2 * pp] >> sh) & 0x0f0f0f0fu;
                hi[h] = (qw[h][2 * pp + 1] >> sh) & 0x0f0f0f0fu;
                if constexpr (TYPE == 13) {
                    lo[h] |= ((hw[h][0] >> j) & 0x01010101u) << 4;
                    hi[h] |= ((hw[h][1] >> j) & 0x01010101u) << 4;
                }
                int sc, m;   // get_scale_min_k4
                if (n == 0) {
                    sc = byte_of(sw[h][0], i) & 63;
                    m = byte_of(sw[h][1], i) & 63;
                } else {
                    sc = (byte_of(sw[h][2], i) & 0xF) | ((byte_of(sw[h][0], i) >> 6) << 4);
                    m = (byte_of(sw[h][2], i) >> 4) | ((byte_of(sw[h][1], i) >> 6) << 4);
                }
                f1[h] = dm[h].x * (float) sc;
                f2[h] = dm[h].y * (float) m;
            }
            int d[4];
            mma_k32(d, lo[0], lo[1], hi[0], hi[1], x.b0[i], x.b1[i]);
            // the code sums of columns 2t and 2t + 1 (column g's from lanes 4g..4g + 3)
            int is = __dp4a(0x01010101, (int) x.b0[i], __dp4a(0x01010101, (int) x.b1[i], 0));
            is += __shfl_xor_sync(0xffffffffu, is, 1);
            is += __shfl_xor_sync(0xffffffffu, is, 2);
            const int isum[2] = {__shfl_sync(0xffffffffu, is, 8 * t), __shfl_sync(0xffffffffu, is, 8 * t + 4)};
#pragma unroll
            for (int q = 0; q < 4; ++q)
                acc[q] += x.d8[i][q & 1] * (f1[q >> 1] * (float) d[q] - f2[q >> 1] * (float) isum[q & 1]);
        }
    } else if constexpr (TYPE == 14) {
        uint32_t lw[2][4], hw[2][2], sw[2][2];
        float dd[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint8_t* ql = p[h] + 64 * n;
            const uint8_t* qh = p[h] + 128 + 32 * n;
#pragma unroll
            for (int i = 0; i < 4; ++i) lw[h][i] = ld32_b2(ql + 16 * i + 4 * t);
            hw[h][0] = ld32_b2(qh + 4 * t);
            hw[h][1] = ld32_b2(qh + 16 + 4 * t);
            sw[h][0] = ld32_b2(p[h] + 192 + 8 * n);
            sw[h][1] = ld32_b2(p[h] + 196 + 8 * n);
            dd[h] = __half2float(*reinterpret_cast<const __half*>(p[h] + 208));
        }
        float part[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int i = 0; i < 4; ++i) {   // block 4n + i: ql bytes 32 (i & 1) + .., nibble i >> 1, qh bits 2i
            const int sh = 4 * (i >> 1), sq = 2 * i, li = 2 * (i & 1);
            uint32_t lo[2], hi[2];
            int sc0[2], sc1[2];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                lo[h] = __vsub4(((lw[h][li] >> sh) & 0x0f0f0f0fu) | (((hw[h][0] >> sq) & 0x03030303u) << 4),
                                0x20202020u);
                hi[h] = __vsub4(((lw[h][li + 1] >> sh) & 0x0f0f0f0fu) | (((hw[h][1] >> sq) & 0x03030303u) << 4),
                                0x20202020u);
                sc0[h] = sbyte_of(sw[h][i >> 1], 2 * (i & 1));
                sc1[h] = sbyte_of(sw[h][i >> 1], 2 * (i & 1) + 1);
            }
            int e0[4], e1[4];
            mma_k16(e0, lo[0], lo[1], x.b0[i]);
            mma_k16(e1, hi[0], hi[1], x.b1[i]);
#pragma unroll
            for (int q = 0; q < 4; ++q)
                part[q] += x.d8[i][q & 1] * (float) (sc0[q >> 1] * e0[q] + sc1[q >> 1] * e1[q]);
        }
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[q] += dd[q >> 1] * part[q];
    } else if constexpr (TYPE == 21) {   // IQ3_S: d, qs[64], qh[8], signs[32], scales[4]; 2-byte aligned
        float part[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float dw[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) dw[h] = __half2float(*reinterpret_cast<const __half*>(p[h]));
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int ib = 4 * n + i;
            uint32_t lo[2], hi[2];
            int ls[2];
#pragma unroll
            for (int h = 0; h < 2; ++h) {   // values 4t.. and 16 + 4t.. of block ib
                const uint8_t* bp = p[h];
                const uint32_t qh = bp[66 + ib];
                const uint32_t sg = ld32_b2(bp + 74 + 4 * ib);
                lo[h] = iq3s_signed(grid[bp[2 + 8 * ib + t] | (((qh >> t) & 1u) << 8)], (sg >> (4 * t)) & 15u);
                hi[h] = iq3s_signed(grid[bp[6 + 8 * ib + t] | (((qh >> (4 + t)) & 1u) << 8)], (sg >> (16 + 4 * t)) & 15u);
                ls[h] = 1 + 2 * ((bp[106 + ib / 2] >> (4 * (ib & 1))) & 0x0F);
            }
            int d[4];
            mma_k32(d, lo[0], lo[1], hi[0], hi[1], x.b0[i], x.b1[i]);
#pragma unroll
            for (int q = 0; q < 4; ++q) part[q] += x.d8[i][q & 1] * (float) (d[q] * ls[q >> 1]);
        }
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[q] += dw[q >> 1] * part[q];
    } else {   // IQ4_XS
        uint32_t qw[2][4];
        int sl[2], shh[2];
        float dw[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
#pragma unroll
            for (int i = 0; i < 4; ++i) qw[h][i] = ld32(p[h] + 8 + 16 * (4 * n + i) + 4 * t);
            sl[h] = *reinterpret_cast<const uint16_t*>(p[h] + 4 + 2 * n);
            shh[h] = *reinterpret_cast<const uint16_t*>(p[h] + 2);
            dw[h] = __half2float(*reinterpret_cast<const __half*>(p[h]));
        }
        float part[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int j = 4 * n + i;
            uint32_t lo[2], hi[2];
            int ls[2];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const uint2 v = iq4_lookup(qw[h][i], t0, t1, t2, t3);
                lo[h] = v.x;
                hi[h] = v.y;
                ls[h] = (((sl[h] >> (4 * i)) & 0xf) | (((shh[h] >> (2 * j)) & 3) << 4)) - 32;
            }
            int d[4];
            mma_k32(d, lo[0], lo[1], hi[0], hi[1], x.b0[i], x.b1[i]);
#pragma unroll
            for (int q = 0; q < 4; ++q) part[q] += x.d8[i][q & 1] * (float) (d[q] * ls[q >> 1]);
        }
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[q] += dw[q >> 1] * part[q];
    }
}
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
// block blk of nblk; smem: smem_bytes<TYPE, S, MT, KS>() bytes, 128-byte aligned
template<int TYPE, int S, int MT, int KS>
__device__ __forceinline__ void dense_tc_block(int blk, int nblk, uint8_t* smem, const uint8_t* __restrict__ w,
                                               const uint8_t* __restrict__ x, float* __restrict__ y, int n_in,
                                               int n_out, int ncols, int probe) {
    constexpr int SB = Fmt<TYPE>::SB, CB = 2 * KS * SB, RS = row_stride(KS * SB), R = 16 * MT, SLOT = R * RS;
    constexpr int NU = RS / 16, NT = MT * KS * 128;
    uint8_t* ring = smem;
    float* red = reinterpret_cast<float*>(smem + S * SLOT);   // [warp][row][column]
    const int tid = threadIdx.x, wp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int mt = (wp >> 2) % MT, ks = wp / (4 * MT), hb = wp & 3, sbi = 2 * ks + (hb >> 1), n = hb & 1;
    const int nchunk = n_in / (512 * KS), nb32 = n_in / 32;
    const size_t row_bytes = (size_t) (n_in / 256) * SB;
    const int ntiles = (n_out + R - 1) / R;
    const int mine = blk < ntiles ? (ntiles - 1 - blk) / nblk + 1 : 0;
    const int nunits = mine * nchunk;
    auto issue = [&](int u) {   // unit u = (tile, chunk) into slot u % S
        const int tile = blk + (u / nchunk) * nblk, c = u % nchunk;
        uint8_t* dst = ring + (u % S) * SLOT;
        for (int q = tid; q < R * NU; q += NT) {
            const int r = q / NU, k = q - r * NU;
            const int row = min(tile * R + r, n_out - 1);
            const uintptr_t src = (uintptr_t) (w + (size_t) row * row_bytes + (size_t) c * CB);
            const uintptr_t a0 = src & ~(uintptr_t) 15;
            if (k * 16 < (int) (src - a0) + CB)
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                             ::"r"((unsigned) __cvta_generic_to_shared(dst + r * RS + 16 * k)), "l"(a0 + 16 * k)
                             : "memory");
        }
    };
    auto commit = [] { asm volatile("cp.async.commit_group;\n" ::: "memory"); };
#pragma unroll
    for (int q = 0; q < S - 1; ++q) {
        if (q < nunits && !(probe & 2)) issue(q);
        commit();
    }
    pdl_wait();
    const uint32_t* tbl = reinterpret_cast<const uint32_t*>(dtc_iq4nl);
    uint32_t t0 = 0, t1 = 0, t2 = 0, t3 = 0;
    if constexpr (TYPE == 23) { t0 = tbl[0]; t1 = tbl[1]; t2 = tbl[2]; t3 = tbl[3]; }
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int u = 0; u < nunits; ++u) {
        const int tile = blk + (u / nchunk) * nblk, c = u % nchunk;
        XFrag xf;   // read before the wait: it does not depend on the slot
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int k = 16 * (KS * c + ks) + 4 * hb + i;   // the 32-value block of the columns
            xf.b0[i] = xf.b1[i] = 0;
            if (g < ncols) {
                const uint8_t* xb = x + ((size_t) g * nb32 + k) * kQ81;
                xf.b0[i] = __ldg(reinterpret_cast<const unsigned int*>(xb + 4) + t);
                xf.b1[i] = __ldg(reinterpret_cast<const unsigned int*>(xb + 20) + t);
            }
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int col = 2 * t + e;
                xf.d8[i][e] = col < ncols ? __low2float(__ldg(reinterpret_cast<const half2*>(
                                                x + ((size_t) col * nb32 + k) * kQ81)))
                                          : 0.0f;
            }
        }
        asm volatile("cp.async.wait_group %0;\n" ::"n"(S - 2) : "memory");
        __syncthreads();   // unit u is in for every thread; the slot of unit u - 1 is free
        if (u + S - 1 < nunits && !(probe & 2)) issue(u + S - 1);
        commit();
        const int row0 = tile * R;
        const int rA = min(row0 + 16 * mt + g, n_out - 1), rB = min(row0 + 16 * mt + g + 8, n_out - 1);
        const uint8_t* slot = ring + (u % S) * SLOT;
        const uint8_t* const p[2] = {
            slot + (16 * mt + g) * RS + ((uintptr_t) (w + (size_t) rA * row_bytes + (size_t) c * CB) & 15) + sbi * SB,
            slot + (16 * mt + g + 8) * RS + ((uintptr_t) (w + (size_t) rB * row_bytes + (size_t) c * CB) & 15) +
                sbi * SB};
        if (!(probe & 1)) half_sb_terms<TYPE>(acc, p, n, t, xf, t0, t1, t2, t3);
        if (c == nchunk - 1) {   // the tile: each m-tile's warp sums, added in warp order (hb, then ks)
            float* rw = red + wp * 16 * kMaxCols;
            rw[g * kMaxCols + 2 * t] = acc[0];
            rw[g * kMaxCols + 2 * t + 1] = acc[1];
            rw[(g + 8) * kMaxCols + 2 * t] = acc[2];
            rw[(g + 8) * kMaxCols + 2 * t + 1] = acc[3];
            acc[0] = acc[1] = acc[2] = acc[3] = 0.0f;
            if (u == nunits - 1) pdl_launch_dependents();
            __syncthreads();
            const int row = tid / kMaxCols, col = tid % kMaxCols;   // R * kMaxCols threads
            if (tid < R * kMaxCols && col < ncols && row0 + row < n_out) {
                float v = 0.0f;
#pragma unroll
                for (int q = 0; q < KS; ++q) {
                    const float* rr = red + ((4 * (q * MT + row / 16)) * 16 + row % 16) * kMaxCols + col;
                    const float vq = ((rr[0] + rr[16 * kMaxCols]) + rr[32 * kMaxCols]) + rr[48 * kMaxCols];
                    v = q == 0 ? vq : v + vq;
                }
                y[(size_t) col * n_out + row0 + row] = v;
            }
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}
#endif

template<int TYPE, int S, int MT, int KS>
__global__ void __launch_bounds__(MT * KS * 128) dense_tc_kernel(const uint8_t* __restrict__ w,
                                                                 const uint8_t* __restrict__ x, float* __restrict__ y,
                                                                 int n_in, int n_out, int ncols, int probe) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    extern __shared__ __align__(128) uint8_t smem[];
    dense_tc_block<TYPE, S, MT, KS>((int) blockIdx.x, (int) gridDim.x, smem, w, x, y, n_in, n_out, ncols, probe);
#elif defined(__CUDA_ARCH__)
    __trap();
#endif
}

// Whole tiles: a 16-row tile is contiguous in memory, so it lands in one of two shared-memory slots with one bulk
// copy (the first two before the grid dependency wait).  Warp w takes half super-block w % 4 of the chunks
// w / 4, w / 4 + 2, ... in order; a row's eight warp sums are added as
// ((w0 + w4) + (w1 + w5)) + ((w2 + w6) + (w3 + w7)).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
// block blk of nblk (256 threads); smem: 128 + 4096 + 2 tiles of 16 rs bytes, 128-byte aligned.  rs: the staged rows'
// stride - the row's bytes (one bulk copy per tile) or more (one per row, STRATA_DEC_DTC2 & 32: one_row_stride)
template<int TYPE>
__device__ __forceinline__ void dense_tc_tile_block(int blk, int nblk, uint8_t* smem, const uint8_t* __restrict__ w,
                                                    const uint8_t* __restrict__ x, float* __restrict__ y, int n_in,
                                                    int n_out, int ncols, int probe, int rs) {
    constexpr int SB = Fmt<TYPE>::SB, CB = 2 * SB;
    uint64_t* bar = reinterpret_cast<uint64_t*>(smem);
    float* red = reinterpret_cast<float*>(smem + 128);   // [warp][row][column]
    uint8_t* slots = smem + 128 + 8 * 16 * kMaxCols * 4;
    const int tid = threadIdx.x, wp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int hb = wp & 3, sbi = hb >> 1, n = hb & 1;
    const int nchunk = n_in / 512, nb32 = n_in / 32;
    const int row_bytes = (n_in / 256) * SB, tb = 16 * row_bytes, ts = 16 * rs;
    const int ntiles = (n_out + 15) / 16;
    const int mine = blk < ntiles ? (ntiles - 1 - blk) / nblk + 1 : 0;
    // tile i into slot i & 1 by warp 0; the last tile's rows past n_out are not copied (and never stored)
    auto issue = [&](int i) {
        const int tile = blk + i * nblk;
        const int rows = min(16, n_out - tile * 16);
        const uint32_t sb = (uint32_t) __cvta_generic_to_shared(slots + (size_t) (i & 1) * ts);
        const uint32_t bb = (uint32_t) __cvta_generic_to_shared(bar + (i & 1));
        if (rs == row_bytes) {
            const uint32_t bytes = (uint32_t) ((rows * row_bytes + 15) & ~15);
            if (lane == 0) {
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bb), "r"(bytes) : "memory");
                asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                             ::"r"(sb), "l"(w + (size_t) tile * tb), "r"(bytes), "r"(bb) : "memory");
            }
        } else {
            if (lane == 0)
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bb),
                             "r"((uint32_t) (rows * row_bytes)) : "memory");
            __syncwarp();
            if (lane < rows)
                asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                             ::"r"(sb + (uint32_t) (lane * rs)), "l"(w + (size_t) tile * tb + (size_t) lane * row_bytes),
                             "r"((uint32_t) row_bytes), "r"(bb) : "memory");
        }
    };
    if (wp == 0) {
        if (lane == 0) {
            for (int q = 0; q < 2; ++q)
                asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"((uint32_t) __cvta_generic_to_shared(bar + q)));
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
        __syncwarp();
        if (!(probe & 2))
            for (int i = 0; i < 2 && i < mine; ++i) issue(i);
    }
    __syncthreads();
    pdl_wait();
    const uint32_t* tbl = reinterpret_cast<const uint32_t*>(dtc_iq4nl);
    uint32_t t0 = 0, t1 = 0, t2 = 0, t3 = 0;
    if constexpr (TYPE == 23) { t0 = tbl[0]; t1 = tbl[1]; t2 = tbl[2]; t3 = tbl[3]; }
    for (int i = 0; i < mine; ++i) {
        const int row0 = (blk + i * nblk) * 16;
        if (!(probe & 2)) {
            uint32_t done = 0;
            const uint32_t bb = (uint32_t) __cvta_generic_to_shared(bar + (i & 1)), par = (uint32_t) (i >> 1) & 1u;
            do {
                asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                             " selp.u32 %0, 1, 0, p;\n}" : "=r"(done) : "r"(bb), "r"(par) : "memory");
            } while (!done);
        }
        const uint8_t* slot = slots + (size_t) (i & 1) * ts;
        // rows past n_out read the slot's stale bytes: finite or not, they are never stored
        const uint8_t* rowA = slot + g * rs + sbi * SB;
        const uint8_t* rowB = slot + (g + 8) * rs + sbi * SB;
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int c = wp >> 2; c < nchunk; c += 2) {
            XFrag xf;
#pragma unroll
            for (int k4 = 0; k4 < 4; ++k4) {
                const int k = 16 * c + 4 * hb + k4;
                xf.b0[k4] = xf.b1[k4] = 0;
                if (g < ncols) {
                    const uint8_t* xb = x + ((size_t) g * nb32 + k) * kQ81;
                    xf.b0[k4] = __ldg(reinterpret_cast<const unsigned int*>(xb + 4) + t);
                    xf.b1[k4] = __ldg(reinterpret_cast<const unsigned int*>(xb + 20) + t);
                }
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int col = 2 * t + e;
                    xf.d8[k4][e] = col < ncols ? __low2float(__ldg(reinterpret_cast<const half2*>(
                                                     x + ((size_t) col * nb32 + k) * kQ81)))
                                               : 0.0f;
                }
            }
            const uint8_t* const p[2] = {rowA + c * CB, rowB + c * CB};
            if (!(probe & 1)) half_sb_terms<TYPE>(acc, p, n, t, xf, t0, t1, t2, t3);
        }
        float* rw = red + wp * 16 * kMaxCols;
        rw[g * kMaxCols + 2 * t] = acc[0];
        rw[g * kMaxCols + 2 * t + 1] = acc[1];
        rw[(g + 8) * kMaxCols + 2 * t] = acc[2];
        rw[(g + 8) * kMaxCols + 2 * t + 1] = acc[3];
        if (i == mine - 1) pdl_launch_dependents();
        __syncthreads();   // the slot is read and the warp sums are in
        if (wp == 0 && i + 2 < mine && !(probe & 2)) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            issue(i + 2);
        }
        if (tid < 16 * kMaxCols) {
            const int row = tid / kMaxCols, col = tid % kMaxCols;
            if (col < ncols && row0 + row < n_out) {
                const float* rr = red + row * kMaxCols + col;
                constexpr int WS = 16 * kMaxCols;   // one warp's sums
                const float v = ((rr[0] + rr[4 * WS]) + (rr[WS] + rr[5 * WS])) +
                                ((rr[2 * WS] + rr[6 * WS]) + (rr[3 * WS] + rr[7 * WS]));
                y[(size_t) col * n_out + row0 + row] = v;
            }
        }
        __syncthreads();   // the sums are read
    }
}
#endif

template<int TYPE>
__global__ void __launch_bounds__(256) dense_tc_tile_kernel(const uint8_t* __restrict__ w,
                                                           const uint8_t* __restrict__ x, float* __restrict__ y,
                                                           int n_in, int n_out, int ncols, int probe, int rs) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    extern __shared__ __align__(128) uint8_t smem[];
    dense_tc_tile_block<TYPE>((int) blockIdx.x, (int) gridDim.x, smem, w, x, y, n_in, n_out, ncols, probe, rs);
#elif defined(__CUDA_ARCH__)
    __trap();
#endif
}

// One tile per block (STRATA_DEC_DTC2 & 2, where every tile has a block of its own): the 16 rows land by one bulk copy
// each, issued before the grid dependency wait, and NW warps share its chunks: warp w takes half super-block
// w % 4 of the chunks w / 4, w / 4 + NW / 4, ...  A row's sums: per half super-block the warp groups in order, then
// (hb0 + hb1) + (hb2 + hb3).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
// tile blk; smem: one_smem_bytes<TYPE, NW>(n_in) bytes, 128-byte aligned
template<int TYPE, int NW>
__device__ __forceinline__ void dense_tc_one_block(int blk, uint8_t* smem, const uint8_t* __restrict__ w,
                                                   const uint8_t* __restrict__ x, float* __restrict__ y, int n_in,
                                                   int n_out, int ncols, int probe) {
    constexpr int SB = Fmt<TYPE>::SB, CB = 2 * SB, NG = NW / 4;
    uint64_t* bar = reinterpret_cast<uint64_t*>(smem);
    uint8_t* slot = smem + 128;
    float* red = reinterpret_cast<float*>(slot);   // [warp][row][column], over the rows once they are read
    const int tid = threadIdx.x, wp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int hb = wp & 3, sbi = hb >> 1, n = hb & 1;
    const int nchunk = n_in / 512, nb32 = n_in / 32;
    const int row_bytes = (n_in / 256) * SB, rs = one_row_stride(row_bytes);
    const int row0 = blk * 16, rows = min(16, n_out - row0);
    const uint32_t bb = (uint32_t) __cvta_generic_to_shared(bar);
    uint32_t* grid = reinterpret_cast<uint32_t*>(slot + max(NW * 16 * kMaxCols * 4, 16 * rs));   // IQ3_S: the codebook
    if constexpr (TYPE == 21)
        for (int i = tid; i < 512; i += NW * 32) grid[i] = iq3s_grid[i];
    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(bb));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        // rows past n_out are not copied (and never stored)
        if (!(probe & 2))
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bb), "r"((uint32_t) (rows * row_bytes))
                         : "memory");
    }
    __syncthreads();
    if (tid < rows && !(probe & 2))   // one bulk copy per row, at a stride that spreads the rows over the banks
        asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                     ::"r"((uint32_t) __cvta_generic_to_shared(slot + tid * rs)),
                     "l"(w + (size_t) (row0 + tid) * row_bytes), "r"((uint32_t) row_bytes), "r"(bb) : "memory");
    pdl_wait();
    const uint32_t* tbl = reinterpret_cast<const uint32_t*>(dtc_iq4nl);
    uint32_t t0 = 0, t1 = 0, t2 = 0, t3 = 0;
    if constexpr (TYPE == 23) { t0 = tbl[0]; t1 = tbl[1]; t2 = tbl[2]; t3 = tbl[3]; }
    // the first chunk's column data while the tile lands
    auto load_x = [&](XFrag& xf, int c) {
#pragma unroll
        for (int k4 = 0; k4 < 4; ++k4) {
            const int k = 16 * c + 4 * hb + k4;
            xf.b0[k4] = xf.b1[k4] = 0;
            if (g < ncols) {
                const uint8_t* xb = x + ((size_t) g * nb32 + k) * kQ81;
                xf.b0[k4] = __ldg(reinterpret_cast<const unsigned int*>(xb + 4) + t);
                xf.b1[k4] = __ldg(reinterpret_cast<const unsigned int*>(xb + 20) + t);
            }
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int col = 2 * t + e;
                xf.d8[k4][e] = col < ncols ? __low2float(__ldg(reinterpret_cast<const half2*>(
                                                 x + ((size_t) col * nb32 + k) * kQ81)))
                                           : 0.0f;
            }
        }
    };
    XFrag xf;
    int c = wp >> 2;
    if (c < nchunk) load_x(xf, c);
    if (!(probe & 2)) {
        uint32_t done = 0;
        do {
            asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                         " selp.u32 %0, 1, 0, p;\n}" : "=r"(done) : "r"(bb), "r"(0u) : "memory");
        } while (!done);
    }
    pdl_launch_dependents();
    // rows past n_out read the slot's stale bytes: finite or not, they are never stored
    const uint8_t* rowA = slot + g * rs + sbi * SB;
    const uint8_t* rowB = slot + (g + 8) * rs + sbi * SB;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (; c < nchunk; c += NG) {
        XFrag xn;
        if (c + NG < nchunk) load_x(xn, c + NG);
        const uint8_t* const p[2] = {rowA + c * CB, rowB + c * CB};
        if (!(probe & 1)) half_sb_terms<TYPE>(acc, p, n, t, xf, t0, t1, t2, t3, grid);
        xf = xn;
    }
    __syncthreads();   // the rows are read
    float* rw = red + wp * 16 * kMaxCols;
    rw[g * kMaxCols + 2 * t] = acc[0];
    rw[g * kMaxCols + 2 * t + 1] = acc[1];
    rw[(g + 8) * kMaxCols + 2 * t] = acc[2];
    rw[(g + 8) * kMaxCols + 2 * t + 1] = acc[3];
    __syncthreads();
    if (tid < 16 * kMaxCols) {
        const int row = tid / kMaxCols, col = tid % kMaxCols;
        if (col < ncols && row0 + row < n_out) {
            constexpr int WS = 16 * kMaxCols;   // one warp's sums
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

template<int TYPE, int NW>
__global__ void __launch_bounds__(NW * 32) dense_tc_one_kernel(const uint8_t* __restrict__ w,
                                                              const uint8_t* __restrict__ x, float* __restrict__ y,
                                                              int n_in, int n_out, int ncols, int probe) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    extern __shared__ __align__(128) uint8_t smem[];
    dense_tc_one_block<TYPE, NW>((int) blockIdx.x, smem, w, x, y, n_in, n_out, ncols, probe);
#elif defined(__CUDA_ARCH__)
    __trap();
#endif
}

template<int TYPE, int NW>
constexpr int one_smem_bytes(int n_in) {
    return 128 + std::max(NW * 16 * kMaxCols * 4, 16 * one_row_stride((n_in / 256) * Fmt<TYPE>::SB)) +
           (TYPE == 21 ? 2048 : 0);
}

// Two matrices of one input in one launch (dense_tc_pair): whole 16-row tiles as in dense_tc_tile_block, the first
// matrix's tiles then the second's, each tile with its own format; a tile's arithmetic is dense_tc_tile_block's.
struct PairMat {
    const uint8_t* w;
    float* y;
    int type, n_out;
};
__host__ __device__ constexpr int sb_bytes(int type) {
    return type == 12 ? 144 : type == 13 ? 176 : type == 14 ? 210 : type == 21 ? 110 : 136;
}
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
template<int TYPE>
__device__ __forceinline__ void pair_terms(float (&acc)[4], const uint8_t* rowA, const uint8_t* rowB, int wp, int hb,
                                           int n, int g, int t, const uint8_t* __restrict__ x, int nb32, int nchunk,
                                           int ncols, const uint32_t* grid, int probe) {
    constexpr int CB = 2 * Fmt<TYPE>::SB;
    uint32_t t0 = 0, t1 = 0, t2 = 0, t3 = 0;
    if constexpr (TYPE == 23) {
        const uint32_t* tbl = reinterpret_cast<const uint32_t*>(dtc_iq4nl);
        t0 = tbl[0]; t1 = tbl[1]; t2 = tbl[2]; t3 = tbl[3];
    }
    rowA += (hb >> 1) * Fmt<TYPE>::SB;
    rowB += (hb >> 1) * Fmt<TYPE>::SB;
    for (int c = wp >> 2; c < nchunk; c += 2) {
        XFrag xf;
#pragma unroll
        for (int k4 = 0; k4 < 4; ++k4) {
            const int k = 16 * c + 4 * hb + k4;
            xf.b0[k4] = xf.b1[k4] = 0;
            if (g < ncols) {
                const uint8_t* xb = x + ((size_t) g * nb32 + k) * kQ81;
                xf.b0[k4] = __ldg(reinterpret_cast<const unsigned int*>(xb + 4) + t);
                xf.b1[k4] = __ldg(reinterpret_cast<const unsigned int*>(xb + 20) + t);
            }
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int col = 2 * t + e;
                xf.d8[k4][e] = col < ncols ? __low2float(__ldg(reinterpret_cast<const half2*>(
                                                 x + ((size_t) col * nb32 + k) * kQ81)))
                                           : 0.0f;
            }
        }
        const uint8_t* const p[2] = {rowA + c * CB, rowB + c * CB};
        if (!(probe & 1)) half_sb_terms<TYPE>(acc, p, n, t, xf, t0, t1, t2, t3, grid);
    }
}

// block blk of nblk (256 threads); smem: 128 + 4096 + 2048 (IQ3_S grid) + 2 slots of the larger tile
__device__ __forceinline__ void dense_tc_pair_block(int blk, int nblk, uint8_t* smem, PairMat m0, PairMat m1,
                                                    const uint8_t* __restrict__ x, int n_in, int ncols, int probe) {
    uint64_t* bar = reinterpret_cast<uint64_t*>(smem);
    float* red = reinterpret_cast<float*>(smem + 128);   // [warp][row][column]
    uint32_t* grid = reinterpret_cast<uint32_t*>(smem + 128 + 8 * 16 * kMaxCols * 4);
    uint8_t* slots = smem + 128 + 8 * 16 * kMaxCols * 4 + 2048;
    const int tid = threadIdx.x, wp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int hb = wp & 3, n = hb & 1;
    const int nchunk = n_in / 512, nb32 = n_in / 32;
    const int nt0 = (m0.n_out + 15) / 16, ntiles = nt0 + (m1.n_out + 15) / 16;
    const int tb0 = 16 * (n_in / 256) * sb_bytes(m0.type), tb1 = 16 * (n_in / 256) * sb_bytes(m1.type);
    const int slot_bytes = tb0 > tb1 ? tb0 : tb1;
    const int mine = blk < ntiles ? (ntiles - 1 - blk) / nblk + 1 : 0;
    if (m0.type == 21 || m1.type == 21)
        for (int i = tid; i < 512; i += 256) grid[i] = iq3s_grid[i];
    auto issue = [&](int i) {
        const int tile = blk + i * nblk;
        const bool second = tile >= nt0;
        const PairMat& m = second ? m1 : m0;
        const int lt = second ? tile - nt0 : tile;
        const int row_bytes = (n_in / 256) * sb_bytes(m.type);
        const int rows = min(16, m.n_out - lt * 16);
        const uint32_t bytes = (uint32_t) ((rows * row_bytes + 15) & ~15);
        const uint32_t sb = (uint32_t) __cvta_generic_to_shared(slots + (size_t) (i & 1) * slot_bytes);
        const uint32_t bb = (uint32_t) __cvta_generic_to_shared(bar + (i & 1));
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bb), "r"(bytes) : "memory");
        asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                     ::"r"(sb), "l"(m.w + (size_t) lt * 16 * row_bytes), "r"(bytes), "r"(bb) : "memory");
    };
    if (tid == 0) {
        for (int q = 0; q < 2; ++q)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"((uint32_t) __cvta_generic_to_shared(bar + q)));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        if (!(probe & 2))
            for (int i = 0; i < 2 && i < mine; ++i) issue(i);
    }
    __syncthreads();
    pdl_wait();
    for (int i = 0; i < mine; ++i) {
        const int tile = blk + i * nblk;
        const bool second = tile >= nt0;
        const PairMat& m = second ? m1 : m0;
        const int row0 = (second ? tile - nt0 : tile) * 16;
        const int row_bytes = (n_in / 256) * sb_bytes(m.type);
        if (!(probe & 2)) {
            uint32_t done = 0;
            const uint32_t bb = (uint32_t) __cvta_generic_to_shared(bar + (i & 1)), par = (uint32_t) (i >> 1) & 1u;
            do {
                asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                             " selp.u32 %0, 1, 0, p;\n}" : "=r"(done) : "r"(bb), "r"(par) : "memory");
            } while (!done);
        }
        const uint8_t* slot = slots + (size_t) (i & 1) * slot_bytes;
        const uint8_t* rowA = slot + g * row_bytes;
        const uint8_t* rowB = slot + (g + 8) * row_bytes;
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        switch (m.type) {
            case 12: pair_terms<12>(acc, rowA, rowB, wp, hb, n, g, t, x, nb32, nchunk, ncols, grid, probe); break;
            case 13: pair_terms<13>(acc, rowA, rowB, wp, hb, n, g, t, x, nb32, nchunk, ncols, grid, probe); break;
            case 14: pair_terms<14>(acc, rowA, rowB, wp, hb, n, g, t, x, nb32, nchunk, ncols, grid, probe); break;
            case 21: pair_terms<21>(acc, rowA, rowB, wp, hb, n, g, t, x, nb32, nchunk, ncols, grid, probe); break;
            default: pair_terms<23>(acc, rowA, rowB, wp, hb, n, g, t, x, nb32, nchunk, ncols, grid, probe); break;
        }
        float* rw = red + wp * 16 * kMaxCols;
        rw[g * kMaxCols + 2 * t] = acc[0];
        rw[g * kMaxCols + 2 * t + 1] = acc[1];
        rw[(g + 8) * kMaxCols + 2 * t] = acc[2];
        rw[(g + 8) * kMaxCols + 2 * t + 1] = acc[3];
        if (i == mine - 1) pdl_launch_dependents();
        __syncthreads();   // the slot is read and the warp sums are in
        if (tid == 0 && i + 2 < mine && !(probe & 2)) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            issue(i + 2);
        }
        if (tid < 16 * kMaxCols) {
            const int row = tid / kMaxCols, col = tid % kMaxCols;
            if (col < ncols && row0 + row < m.n_out) {
                const float* rr = red + row * kMaxCols + col;
                constexpr int WS = 16 * kMaxCols;   // one warp's sums
                const float v = ((rr[0] + rr[4 * WS]) + (rr[WS] + rr[5 * WS])) +
                                ((rr[2 * WS] + rr[6 * WS]) + (rr[3 * WS] + rr[7 * WS]));
                m.y[(size_t) col * m.n_out + row0 + row] = v;
            }
        }
        __syncthreads();   // the sums are read
    }
}
#endif

__global__ void __launch_bounds__(256) dense_tc_pair_kernel(PairMat m0, PairMat m1, const uint8_t* __restrict__ x,
                                                            int n_in, int ncols, int probe) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    extern __shared__ __align__(128) uint8_t smem[];
    dense_tc_pair_block((int) blockIdx.x, (int) gridDim.x, smem, m0, m1, x, n_in, ncols, probe);
#elif defined(__CUDA_ARCH__)
    __trap();
#endif
}

template<int TYPE, int S, int MT, int KS>
size_t smem_bytes() {
    return (size_t) S * 16 * MT * row_stride(KS * Fmt<TYPE>::SB) + (size_t) MT * KS * 4 * 16 * kMaxCols * 4;
}

int sm_count() {
    static const int n = [] {
        int dev = 0, v = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev);
        return v;
    }();
    return n;
}

int env_int(const char* name, int def) {
    const char* v = std::getenv(name);
    return v ? std::atoi(v) : def;
}
// bench knobs: STRATA_DEC_DTC_BPS blocks per SM (default as many as fit), STRATA_DEC_DTC_SLOTS ring depth (3 or 4;
// default 3), STRATA_DEC_DTC_MT rows per tile / 16 (1 or 2; default 1), STRATA_DEC_DTC_PROBE 1 skip the
// dots, 2 skip the weight copies
const int g_bps = env_int("STRATA_DEC_DTC_BPS", 0);
const int g_slots = env_int("STRATA_DEC_DTC_SLOTS", 3);
const int g_mt = env_int("STRATA_DEC_DTC_MT", 0);
const int g_probe = env_int("STRATA_DEC_DTC_PROBE", 0);
const int g_ks = env_int("STRATA_DEC_DTC_KS", 0);
const int g_tile = env_int("STRATA_DEC_DTC_TILE", 1);   // 1: never split a tile's chunks over two warp groups
const int g_v2 = env_int("STRATA_DEC_DTC2", 0);

template<int TYPE, int S, int MT, int KS>
void launch_s(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    const size_t smem = smem_bytes<TYPE, S, MT, KS>();
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute(dense_tc_kernel<TYPE, S, MT, KS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
        attr = true;
    }
    int occ = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, dense_tc_kernel<TYPE, S, MT, KS>, MT * KS * 128, smem);
    if (g_bps > 0 && g_bps < occ) occ = g_bps;
    if (occ < 1) occ = 1;
    const int ntiles = (n_out + 16 * MT - 1) / (16 * MT);
    const int grid = ntiles < occ * sm_count() ? ntiles : occ * sm_count();
    launch_pdl(dense_tc_kernel<TYPE, S, MT, KS>, dim3((unsigned) grid), dim3(MT * KS * 128), smem, s,
               (const uint8_t*) w, (const uint8_t*) x, y, n_in, n_out, ncols, g_probe);
}

template<int TYPE, int MT, int KS>
void launch_mt(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    switch (g_slots) {
        case 4: launch_s<TYPE, 4, MT, KS>(w, x, y, n_in, n_out, ncols, s); break;
        default: launch_s<TYPE, 3, MT, KS>(w, x, y, n_in, n_out, ncols, s); break;
    }
}

template<int TYPE>
void launch_tile(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    const int row_bytes = (n_in / 256) * Fmt<TYPE>::SB;
    const int rs = (g_v2 & 32) && row_bytes % 16 == 0 ? one_row_stride(row_bytes) : row_bytes;
    const int smem = 128 + 8 * 16 * kMaxCols * 4 + 2 * 16 * rs;
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute(dense_tc_tile_kernel<TYPE>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
        attr = true;
    }
    int occ = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, dense_tc_tile_kernel<TYPE>, 256, smem);
    if (g_bps > 0 && g_bps < occ) occ = g_bps;
    if (occ < 1) occ = 1;
    const int ntiles = (n_out + 15) / 16;
    const int grid = ntiles < occ * sm_count() ? ntiles : occ * sm_count();
    launch_pdl(dense_tc_tile_kernel<TYPE>, dim3((unsigned) grid), dim3(256), (size_t) smem, s, (const uint8_t*) w,
               (const uint8_t*) x, y, n_in, n_out, ncols, g_probe, rs);
}

// the one-tile path takes the shape (STRATA_DEC_DTC2 & 2; IQ3_S only there)
template<int TYPE>
bool one_ok(const void* w, int n_in, int n_out) {
    return ((g_v2 & 2) || TYPE == 21) && ((uintptr_t) w & 15) == 0 && (n_in / 256) * Fmt<TYPE>::SB % 16 == 0 &&
           (n_out + 15) / 16 <= sm_count() && one_smem_bytes<TYPE, 24>(n_in) <= 99 * 1024;
}

template<int TYPE, int NW>
void launch_one(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute(dense_tc_one_kernel<TYPE, NW>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
        attr = true;
    }
    launch_pdl(dense_tc_one_kernel<TYPE, NW>, dim3((unsigned) ((n_out + 15) / 16)), dim3(NW * 32),
               (size_t) one_smem_bytes<TYPE, NW>(n_in), s, (const uint8_t*) w, (const uint8_t*) x, y, n_in, n_out,
               ncols, g_probe);
}

// The layout follows the shape alone (never the column count): whole 16-row tiles where two fit in shared memory and
// the weights are 16-byte aligned (STRATA_DEC_DTC_TILE=0: never), else 16-row tiles in 512-value chunks, and where
// those leave SMs idle, two warp groups per tile taking alternate 512-value halves of 1024 (STRATA_DEC_DTC2 & 1: four
// groups taking the quarters of 2048 where n_in allows; & 2: where every 16-row tile has a block of its own and fits
// in shared memory, one bulk copy per row, issued before the grid dependency wait; & 32: whole tiles staged by one
// bulk copy per row at a bank-spreading stride; & 64: IQ4_XS never in whole tiles).
template<int TYPE>
void launch(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    const bool aligned = ((uintptr_t) w & 15) == 0;
    if (one_ok<TYPE>(w, n_in, n_out)) {
        static const int nw = env_int("STRATA_DEC_DTC2_NW", 24);
        if (nw == 16) launch_one<TYPE, 16>(w, x, y, n_in, n_out, ncols, s);
        else if (nw == 32 && one_smem_bytes<TYPE, 32>(n_in) <= 99 * 1024) launch_one<TYPE, 32>(w, x, y, n_in, n_out, ncols, s);
        else launch_one<TYPE, 24>(w, x, y, n_in, n_out, ncols, s);
        return;
    }
    if constexpr (TYPE == 21) {
        std::fprintf(stderr, "dense_tc: IQ3_S n_in %d n_out %d needs the one-tile path\n", n_in, n_out);
        std::exit(1);
    } else {
    if (g_tile && aligned && 2 * 16 * (n_in / 256) * Fmt<TYPE>::SB <= 90 * 1024 && !(TYPE == 23 && (g_v2 & 64))) {
        launch_tile<TYPE>(w, x, y, n_in, n_out, ncols, s);
        return;
    }
    const int mt = g_mt == 2 ? 2 : 1;
    const bool few = (n_out + 15) / 16 < sm_count();
    const bool ks4 = mt == 1 && n_in % 2048 == 0 && few && (g_ks == 4 || ((g_v2 & 1) && g_ks == 0));
    const bool ks2 = mt == 1 && n_in % 1024 == 0 && few && g_ks != 1;
    if (ks4) launch_mt<TYPE, 1, 4>(w, x, y, n_in, n_out, ncols, s);
    else if (ks2) launch_mt<TYPE, 1, 2>(w, x, y, n_in, n_out, ncols, s);
    else if (mt == 2) launch_mt<TYPE, 2, 1>(w, x, y, n_in, n_out, ncols, s);
    else launch_mt<TYPE, 1, 1>(w, x, y, n_in, n_out, ncols, s);
    }
}

}  // namespace

bool dense_tc_supported(int t, int n_in) noexcept {
    return n_in > 0 && n_in % 512 == 0 && (t == 12 || t == 13 || t == 14 || t == 23);
}

bool dense_tc_supported(int t, int n_in, int n_out, const void* w) noexcept {
    if (t == 21) return n_in > 0 && n_in % 512 == 0 && n_out > 0 && one_ok<21>(w, n_in, n_out);
    return dense_tc_supported(t, n_in);
}

bool dense_tc_type_on(int t) noexcept {
    static const int m = env_int("STRATA_DEC_DTC_TYPES", 15) | ((g_v2 & 4) ? 16 : 0);
    return (m & (t == 12 ? 1 : t == 13 ? 2 : t == 14 ? 4 : t == 23 ? 8 : t == 21 ? 16 : 0)) != 0;
}

void dense_tc(int type, const void* w, const void* x, float* y, int n_in, int n_out, int ncols, void* stream) {
    if (!dense_tc_supported(type, n_in, n_out, w) || ncols < 1 || ncols > kMaxCols || n_out < 1 || ((uintptr_t) w & 3) != 0 ||
        ((uintptr_t) x & 3) != 0) {
        std::fprintf(stderr, "dense_tc: type %d n_in %d n_out %d ncols %d not supported\n", type, n_in, n_out, ncols);
        std::exit(1);
    }
    const auto s = (cudaStream_t) stream;
    switch (type) {
        case 12: launch<12>(w, x, y, n_in, n_out, ncols, s); break;
        case 13: launch<13>(w, x, y, n_in, n_out, ncols, s); break;
        case 14: launch<14>(w, x, y, n_in, n_out, ncols, s); break;
        case 21: launch<21>(w, x, y, n_in, n_out, ncols, s); break;
        default: launch<23>(w, x, y, n_in, n_out, ncols, s); break;
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "dense_tc: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

bool dense_tc_pair_supported(int t0, int t1, int n_in) noexcept {
    auto ok = [&](int t) { return t == 12 || t == 13 || t == 14 || t == 21 || t == 23; };
    if (!ok(t0) || !ok(t1) || n_in <= 0 || n_in % 512 != 0) return false;
    const int tb = 16 * (n_in / 256) * std::max(sb_bytes(t0), sb_bytes(t1));
    return 128 + 8 * 16 * kMaxCols * 4 + 2048 + 2 * tb <= 99 * 1024;
}

void dense_tc_pair(int t0, const void* w0, float* y0, int n_out0, int t1, const void* w1, float* y1, int n_out1,
                   const void* x_q8_1, int n_in, int ncols, void* stream) {
    if (!dense_tc_pair_supported(t0, t1, n_in) || ncols < 1 || ncols > kMaxCols || n_out0 < 1 || n_out1 < 1 ||
        ((uintptr_t) w0 & 15) != 0 || ((uintptr_t) w1 & 15) != 0 || ((uintptr_t) x_q8_1 & 3) != 0) {
        std::fprintf(stderr, "dense_tc_pair: types %d/%d n_in %d not supported\n", t0, t1, n_in);
        std::exit(1);
    }
    const int smem = 128 + 8 * 16 * kMaxCols * 4 + 2048 + 2 * 16 * (n_in / 256) * std::max(sb_bytes(t0), sb_bytes(t1));
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute(dense_tc_pair_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
        attr = true;
    }
    int occ = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, dense_tc_pair_kernel, 256, smem);
    if (g_bps > 0 && g_bps < occ) occ = g_bps;
    if (occ < 1) occ = 1;
    const int ntiles = (n_out0 + 15) / 16 + (n_out1 + 15) / 16;
    const int grid = ntiles < occ * sm_count() ? ntiles : occ * sm_count();
    const PairMat m0{(const uint8_t*) w0, y0, t0, n_out0}, m1{(const uint8_t*) w1, y1, t1, n_out1};
    launch_pdl(dense_tc_pair_kernel, dim3((unsigned) grid), dim3(256), (size_t) smem, (cudaStream_t) stream, m0, m1,
               (const uint8_t*) x_q8_1, n_in, ncols, g_probe);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "dense_tc_pair: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

int dense_tc_mask() {
    static const int m = [] {
        const char* v = std::getenv("STRATA_DEC_DTC");
        return v != nullptr ? (int) std::strtol(v, nullptr, 0) : 0;
    }();
    return m;
}

}  // namespace strata::kernels
