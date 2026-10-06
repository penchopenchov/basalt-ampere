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

// a staged row of a slot: two super-blocks from the 16-byte unit holding their first byte, padded to an odd number
// of 16-byte units so the eight rows a load touches fall in different banks
__host__ __device__ constexpr int row_stride(int sb) {
    const int r = (2 * sb + 15 + 15) & ~15;
    return ((r / 16) & 1) ? r : r + 16;
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
                                              const XFrag& x, uint32_t t0, uint32_t t1, uint32_t t2, uint32_t t3) {
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

template<int TYPE, int S, int MT, int KS>
__global__ void __launch_bounds__(MT * KS * 128) dense_tc_kernel(const uint8_t* __restrict__ w,
                                                                 const uint8_t* __restrict__ x, float* __restrict__ y,
                                                                 int n_in, int n_out, int ncols, int probe) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    constexpr int SB = Fmt<TYPE>::SB, CB = 2 * KS * SB, RS = row_stride(KS * SB), R = 16 * MT, SLOT = R * RS;
    constexpr int NU = RS / 16, NT = MT * KS * 128;
    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t* ring = smem;
    float* red = reinterpret_cast<float*>(smem + S * SLOT);   // [warp][row][column]
    const int tid = threadIdx.x, wp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int mt = (wp >> 2) % MT, ks = wp / (4 * MT), hb = wp & 3, sbi = 2 * ks + (hb >> 1), n = hb & 1;
    const int nchunk = n_in / (512 * KS), nb32 = n_in / 32;
    const size_t row_bytes = (size_t) (n_in / 256) * SB;
    const int ntiles = (n_out + R - 1) / R;
    const int mine = (int) blockIdx.x < ntiles ? (ntiles - 1 - (int) blockIdx.x) / (int) gridDim.x + 1 : 0;
    const int nunits = mine * nchunk;
    auto issue = [&](int u) {   // unit u = (tile, chunk) into slot u % S
        const int tile = (int) blockIdx.x + (u / nchunk) * (int) gridDim.x, c = u % nchunk;
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
        const int tile = (int) blockIdx.x + (u / nchunk) * (int) gridDim.x, c = u % nchunk;
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
const int g_ks = env_int("STRATA_DEC_DTC_KS", 0);   // 1: never split a tile's chunks over two warp groups

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

// The layout follows the shape alone (never the column count): 16-row tiles, and where they leave SMs idle, two warp
// groups per tile taking alternate 512-value halves of 1024.
template<int TYPE>
void launch(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    const int mt = g_mt == 2 ? 2 : 1;
    const bool ks2 = mt == 1 && n_in % 1024 == 0 && (n_out + 15) / 16 < sm_count() && g_ks != 1;
    if (ks2) launch_mt<TYPE, 1, 2>(w, x, y, n_in, n_out, ncols, s);
    else if (mt == 2) launch_mt<TYPE, 2, 1>(w, x, y, n_in, n_out, ncols, s);
    else launch_mt<TYPE, 1, 1>(w, x, y, n_in, n_out, ncols, s);
}

}  // namespace

bool dense_tc_supported(int t, int n_in) noexcept {
    return n_in > 0 && n_in % 512 == 0 && (t == 12 || t == 13 || t == 14 || t == 23);
}

void dense_tc(int type, const void* w, const void* x, float* y, int n_in, int n_out, int ncols, void* stream) {
    if (!dense_tc_supported(type, n_in) || ncols < 1 || ncols > kMaxCols || n_out < 1 || ((uintptr_t) w & 3) != 0 ||
        ((uintptr_t) x & 3) != 0) {
        std::fprintf(stderr, "dense_tc: type %d n_in %d n_out %d ncols %d not supported\n", type, n_in, n_out, ncols);
        std::exit(1);
    }
    const auto s = (cudaStream_t) stream;
    switch (type) {
        case 12: launch<12>(w, x, y, n_in, n_out, ncols, s); break;
        case 13: launch<13>(w, x, y, n_in, n_out, ncols, s); break;
        case 14: launch<14>(w, x, y, n_in, n_out, ncols, s); break;
        default: launch<23>(w, x, y, n_in, n_out, ncols, s); break;
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "dense_tc: %s\n", cudaGetErrorString(e));
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
