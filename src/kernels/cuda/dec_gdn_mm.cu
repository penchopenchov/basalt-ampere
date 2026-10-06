// src/kernels/cuda/dec_gdn_mm.cu - see include/strata/kernels/dec_gdn.hpp: the dense projections.
//
// Built with the flags of native_mmvq.cu (fast math), whose dot products it transcribes: the K-format and IQ4_XS
// traits below are native_mmvq.cu's, the IQ3_S one iq_kernels.cu's (Split<21>), the q8_1 quantizer
// native_quantize_q8_1_kernel's.
//
// Layout: a persistent grid of one block per SM, NW warps each.  Warp gw of NWT owns rows gw, gw + NWT, ... of the
// concatenated matrices; each row is one cp.async.bulk into the warp's ring of shared-memory slots, issued
// NSLOT rows ahead (the first ones before the grid dependency wait: weights are constant).  The warp computes the
// row for every column:
//   * K formats and IQ4_XS (the exact layout of native_mmvq_multi_kernel: four warps, thread tid = 32 w + lane
//     sums blocks kbx = tid / T + i * BPI in ascending order, warp 0 adds warps 1..3 in order, then the butterfly):
//     lane L replays the chains of threads L, L + 32, L + 64, L + 96 and adds them in that order.
//   * IQ3_S (mmvq_multi_kernel: one warp per row, lane-strided calls): the same calls in the same order.
// So every output is bitwise the reference kernel's.
#include "strata/kernels/dec_gdn.hpp"
#include "strata/kernels/dp4a.hpp"
#include "strata/kernels/pdl.hpp"
#include "strata/kernels/q8_1_finite.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int QK = 256;
constexpr int Q8K = 32;
constexpr int QI = 32;
constexpr int VDR = 2;
constexpr int WARPS = 4;
constexpr int WARP = 32;

struct Q5KBlock {
    half2 dm;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};
struct Q81Block {
    half2 ds;
    int8_t qs[32];
};
struct IQ4XSBlock {
    half d;
    uint16_t scales_h;
    uint8_t scales_l[4];
    uint8_t qs[128];
};
struct Q4KBlock {
    half2 dm;
    uint8_t scales[12];
    uint8_t qs[128];
};
struct Q6KBlock {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    half d;
};
static_assert(sizeof(Q5KBlock) == 176 && sizeof(Q81Block) == 36 && sizeof(IQ4XSBlock) == 136 &&
              sizeof(Q4KBlock) == 144 && sizeof(Q6KBlock) == 210 && sizeof(block_iq3_s) == 110, "block sizes");

// ---------------------------------------------------------------- the dot products
// native_mmvq.cu's Q4_K / Q5_K / Q6_K / IQ4_XS traits and iq_kernels.cu's IQ3_S one, with every operation that
// depends only on the weight (code unpacking, scale extraction, sign and grid decoding) moved into `load`, which
// runs once per (row, block, part); `apply`, once per column, keeps the activation loads, the integer dots and the
// float expression of the reference in the same order.  Integer operations are exact, so moving them changes no bit.
__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, WARP);
    }
    return x;
}

__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, WARP));
    }
    return x;
}

__device__ __forceinline__ int load_int_b2(const void* ptr, int i32) {
    const auto* x = static_cast<const uint16_t*>(ptr);
    int value = x[2 * i32] << 0;
    value |= x[2 * i32 + 1] << 16;
    return value;
}

__device__ __align__(4) int8_t iq4nl_values[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

__device__ __forceinline__ int2 iq4_table_lookup(int q4) {
    const uint32_t* table32 = reinterpret_cast<const uint32_t*>(iq4nl_values);
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = 0x32103210 | ((q4 & 0x88888888) >> 1);
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low = __byte_perm(table32[0], table32[1], q4 >> shift);
        const uint32_t high = __byte_perm(table32[2], table32[3], q4 >> shift);
        tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}

__device__ __forceinline__ int get_int_b2(const void* x, const int& i32) {
    const uint16_t* x16 = (const uint16_t*) x;
    int x32 = x16[2 * i32 + 0] << 0;
    x32 |= x16[2 * i32 + 1] << 16;
    return x32;
}

// q8_1 activation columns.  XSm: shared memory, block b's 32 codes at qs + 8 b (16-byte aligned) and its fp16 scale
// as a float at d[b].  XGl: the q8_1 blocks in global memory.
struct XSm {
    const int* qs;
    const float* d;
    const int* s;   // the sum of each block's codes
    __device__ XSm col(int j, int nb32) const { return XSm{qs + 8 * nb32 * j, d + nb32 * j, s + nb32 * j}; }
    __device__ int isum(int b, int4, int4) const { return s[b]; }
    __device__ int u(int b, int k) const { return qs[8 * b + k]; }
    __device__ int4 u4(int b, int h) const { return reinterpret_cast<const int4*>(qs + 8 * b)[h]; }
    __device__ float d8(int b) const { return d[b]; }
};
struct XGl {
    const Q81Block* x;
    __device__ XGl col(int j, int nb32) const { return XGl{x + nb32 * j}; }
    __device__ int isum(int, int4 lo, int4 hi) const {
        int a = STRATA_DP4A(0x01010101, lo.x, STRATA_DP4A(0x01010101, lo.y, 0));
        int b = STRATA_DP4A(0x01010101, lo.z, STRATA_DP4A(0x01010101, lo.w, 0));
        a = STRATA_DP4A(0x01010101, hi.x, STRATA_DP4A(0x01010101, hi.y, a));
        b = STRATA_DP4A(0x01010101, hi.z, STRATA_DP4A(0x01010101, hi.w, b));
        return a + b;
    }
    __device__ int u(int b, int k) const { return reinterpret_cast<const int*>(x[b].qs)[k]; }
    __device__ int4 u4(int b, int h) const {
        const int* p = reinterpret_cast<const int*>(x[b].qs) + 4 * h;
        return make_int4(p[0], p[1], p[2], p[3]);
    }
    __device__ float d8(int b) const { return __low2float(x[b].ds); }
};

// Q4_K: q4_q8_dot / Q4KTraits.  A call covers sub-blocks bq8_offset, bq8_offset + 1, two ints each.
struct Q4KF {
    using Block = Q4KBlock;
    static constexpr int T = QI / VDR, KBY = QK / Q8K, BPI = VDR * WARPS * WARP / QI;
    __device__ static int kqs(int tid) { return VDR * (tid % (QI / VDR)); }
    struct W { int v0[2], v1[2], sc[2], m[2]; float2 dm; int bq8_offset; };
    __device__ static W load(const Block* __restrict__ bq4, int iqs) {
        W r;
        r.bq8_offset = 2 * ((iqs / 2) / 4);
        const int* ql = reinterpret_cast<const int*>(bq4->qs + 16 * r.bq8_offset + 4 * ((iqs / 2) % 4));
        const int v[2] = {ql[0], ql[4]};
        const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq4->scales);
        const int j = r.bq8_offset / 2;
        const int jm = j & 1;
        const uint32_t s0 = scales[jm];
        const uint32_t s2 = scales[jm + 2];
        const uint32_t s4 = scales[jm + 4];
        const uint32_t hi = uint32_t(-int32_t(j >= 2));
        uint16_t aux[2];
        aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
        aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
        const uint8_t* sc = reinterpret_cast<const uint8_t*>(aux);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            r.v0[i] = (v[0] >> (4 * i)) & 0x0f0f0f0f;
            r.v1[i] = (v[1] >> (4 * i)) & 0x0f0f0f0f;
            r.sc[i] = sc[i];
            r.m[i] = sc[2 + i];
        }
        r.dm = __half22float2(bq4->dm);
        return r;
    }
    template<typename X>
    __device__ static float apply(const W& r, const X& x, int kb, int iqs) {
        const int k = (iqs / 2) % 4;
        float sumf_d = 0.0f;
        float sumf_m = 0.0f;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int b = kb + r.bq8_offset + i;
            const int u0 = x.u(b, k), u1 = x.u(b, k + 4);
            const float d8 = x.d8(b);
            const int dot1 = STRATA_DP4A(r.v1[i], u1, STRATA_DP4A(r.v0[i], u0, 0));
            const int dot2 = STRATA_DP4A(0x01010101, u1, STRATA_DP4A(0x01010101, u0, 0));
            sumf_d += d8 * (dot1 * r.sc[i]);
            sumf_m += d8 * (dot2 * r.m[i]);
        }
        return r.dm.x * sumf_d - r.dm.y * sumf_m;
    }
};

// Q5_K: q5_q8_dot / Q5KTraits, the high bits folded into the codes once.
struct Q5KF {
    using Block = Q5KBlock;
    static constexpr int T = QI / VDR, KBY = QK / Q8K, BPI = VDR * WARPS * WARP / QI;
    __device__ static int kqs(int tid) { return VDR * (tid % (QI / VDR)); }
    struct W { int v0[2], v1[2], sc[2], m[2]; float2 dm; int bq8_offset; };
    __device__ static W load(const Block* __restrict__ bq5, int iqs) {
        W r;
        r.bq8_offset = 2 * ((iqs / 2) / 4);
        const int* ql = reinterpret_cast<const int*>(bq5->qs + 16 * r.bq8_offset + 4 * ((iqs / 2) % 4));
        const int* qh = reinterpret_cast<const int*>(bq5->qh + 4 * ((iqs / 2) % 4));
        const int vl[2] = {ql[0], ql[4]};
        const int vh[2] = {qh[0] >> r.bq8_offset, qh[4] >> r.bq8_offset};
        const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq5->scales);
        const int j = r.bq8_offset / 2;
        const int jm = j & 1;
        const uint32_t s0 = scales[jm];
        const uint32_t s2 = scales[jm + 2];
        const uint32_t s4 = scales[jm + 4];
        const uint32_t hi = uint32_t(-int32_t(j >= 2));
        uint16_t aux[2];
        aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
        aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
        const uint8_t* sc = reinterpret_cast<const uint8_t*>(aux);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int vl0i = (vl[0] >> (4 * i)) & 0x0f0f0f0f;
            const int vl1i = (vl[1] >> (4 * i)) & 0x0f0f0f0f;
            const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
            const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
            r.v0[i] = vl0i | vh0i;
            r.v1[i] = vl1i | vh1i;
            r.sc[i] = sc[i];
            r.m[i] = sc[2 + i];
        }
        r.dm = __half22float2(bq5->dm);
        return r;
    }
    template<typename X>
    __device__ static float apply(const W& r, const X& x, int kb, int iqs) {
        const int k = (iqs / 2) % 4;
        float sumf_d = 0.0f;
        float sumf_m = 0.0f;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int b = kb + r.bq8_offset + i;
            const int u0 = x.u(b, k), u1 = x.u(b, k + 4);
            const float d8 = x.d8(b);
            const int dot1 = STRATA_DP4A(r.v0[i], u0, STRATA_DP4A(r.v1[i], u1, 0));
            const int dot2 = STRATA_DP4A(0x01010101, u0, STRATA_DP4A(0x01010101, u1, 0));
            sumf_d += d8 * (dot1 * r.sc[i]);
            sumf_m += d8 * (dot2 * r.m[i]);
        }
        return r.dm.x * sumf_d - r.dm.y * sumf_m;
    }
};

// Q6_K: q6_q8_dot / Q6KTraits, the signed codes and their scales decoded once.
struct Q6KF {
    using Block = Q6KBlock;
    static constexpr int T = 32, KBY = 8, BPI = WARPS * WARP / 32;
    __device__ static int kqs(int tid) { return tid % 32; }
    struct W { int vi[2], sc[2]; float d; int bq8_offset; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
        r.bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
        const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
        const int vh_shift = 2 * ((iqs % 16) / 8);
        const int vl = load_int_b2(w->ql, iqs);
        const int vh = load_int_b2(w->qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            r.sc[i] = w->scales[scale_offset + 4 * i];
            const int vil = (vl >> (4 * i)) & 0x0f0f0f0f;
            const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
            r.vi[i] = __vsubss4(vil | vih, 0x20202020);
        }
        r.d = w->d;
        return r;
    }
    template<typename X>
    __device__ static float apply(const W& r, const X& x, int kb, int iqs) {
        float sumf = 0.0f;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int b = kb + r.bq8_offset + 2 * i;
            const int u = x.u(b, iqs % 8);
            const float d8 = x.d8(b);
            sumf += d8 * (STRATA_DP4A(r.vi[i], u, 0) * r.sc[i]);
        }
        return r.d * sumf;
    }
};

// IQ4_XS: iq4_xs_q8_dot / IQ4XSTraits.
struct IQ4XSF {
    using Block = IQ4XSBlock;
    static constexpr int T = 8, KBY = 8, BPI = 4 * WARPS * WARP / 32;
    __device__ static int kqs(int tid) { return 4 * (tid % 8); }
    struct W { int2 v[4]; int ls; float dw; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
#pragma unroll
        for (int j = 0; j < 4; ++j) r.v[j] = iq4_table_lookup(reinterpret_cast<const int*>(w->qs)[iqs + j]);
        r.ls = (((w->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0f) | (((w->scales_h >> (iqs / 2)) & 0x03) << 4)) - 32;
        r.dw = __half2float(w->d);
        return r;
    }
    template<typename X>
    __device__ static float apply(const W& r, const X& x, int kb, int iqs) {
        const int b = kb + iqs / 4;
        const int4 lo = x.u4(b, 0), hi = x.u4(b, 1);
        int sumi = 0;
        sumi = STRATA_DP4A(r.v[0].x, lo.x, sumi);
        sumi = STRATA_DP4A(r.v[0].y, hi.x, sumi);
        sumi = STRATA_DP4A(r.v[1].x, lo.y, sumi);
        sumi = STRATA_DP4A(r.v[1].y, hi.y, sumi);
        sumi = STRATA_DP4A(r.v[2].x, lo.z, sumi);
        sumi = STRATA_DP4A(r.v[2].y, hi.z, sumi);
        sumi = STRATA_DP4A(r.v[3].x, lo.w, sumi);
        sumi = STRATA_DP4A(r.v[3].y, hi.w, sumi);
        sumi *= r.ls;
        const float d = r.dw * x.d8(b);
        return d * sumi;
    }
};

// IQ3_S: iq_kernels.cu Split<21>.  Call k of a row: block k / 8, part 2 (k % 8), activation q8_1 block k.
struct IQ3SF {
    struct W { int g[8]; int ls; float dw; };
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs) {
        const block_iq3_s* bq3 = (const block_iq3_s*) vbq + kbx;
        const int2 qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const int qh = bq3->qh[iqs / 2];
        const int signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
        const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
        W r;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int2 grid_pos = make_int2(iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
                                            iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
            const int signs0 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
            const int signs1 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
            r.g[l0 + 0] = __vsub4(grid_pos.x ^ signs0, signs0);
            r.g[l0 + 1] = __vsub4(grid_pos.y ^ signs1, signs1);
        }
        r.ls = 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
        r.dw = __half2float(bq3->d);
        return r;
    }
    template<typename X>
    __device__ static float apply(const W& r, const X& x, int b) {
        const int4 lo = x.u4(b, 0), hi = x.u4(b, 1);
        int sumi = 0;
        sumi = STRATA_DP4A(r.g[0], lo.x, sumi);
        sumi = STRATA_DP4A(r.g[1], lo.y, sumi);
        sumi = STRATA_DP4A(r.g[2], lo.z, sumi);
        sumi = STRATA_DP4A(r.g[3], lo.w, sumi);
        sumi = STRATA_DP4A(r.g[4], hi.x, sumi);
        sumi = STRATA_DP4A(r.g[5], hi.y, sumi);
        sumi = STRATA_DP4A(r.g[6], hi.z, sumi);
        sumi = STRATA_DP4A(r.g[7], hi.w, sumi);
        sumi *= r.ls;
        const float d = r.dw * x.d8(b);
        return d * sumi;
    }
};

// ---------------------------------------------------------------- row dots (one warp per row)
// The exact layout's four warps, replayed by one: lane `lane` sums the chains of reference threads lane + 32 rw.
// NB > 0: the row's block count at compile time, so the four chains unroll into independent instruction streams.
template<typename F, int NC, int NB, typename X>
__device__ __forceinline__ void row_dot4(const uint8_t* row, const X& x, int nb32, int nb_rt, int lane,
                                         float (&acc)[NC]) {
    const auto* w = reinterpret_cast<const typename F::Block*>(row);
    const int nb = NB > 0 ? NB : nb_rt;
    constexpr int ITERS = NB > 0 ? (NB + F::BPI - 1) / F::BPI : 1;
    constexpr int RWU = NC >= 6 ? 1 : WARPS;   // wide windows: one chain at a time (registers)
    X xc[NC];
#pragma unroll
    for (int j = 0; j < NC; ++j) xc[j] = x.col(j, nb32);
#pragma unroll RWU
    for (int rw = 0; rw < WARPS; ++rw) {
        const int tid = WARP * rw + lane;
        const int kqs = F::kqs(tid);
        float c[NC];
#pragma unroll
        for (int j = 0; j < NC; ++j) c[j] = 0.0f;
        if constexpr (NB > 0) {
#pragma unroll
            for (int i = 0; i < ITERS; ++i) {
                const int kbx = tid / F::T + i * F::BPI;
                if (kbx < nb) {
                    const typename F::W wv = F::load(w + kbx, kqs);
#pragma unroll
                    for (int j = 0; j < NC; ++j) c[j] += F::apply(wv, xc[j], kbx * F::KBY, kqs);
                }
            }
        } else {
            for (int kbx = tid / F::T; kbx < nb; kbx += F::BPI) {
                const typename F::W wv = F::load(w + kbx, kqs);
#pragma unroll
                for (int j = 0; j < NC; ++j) c[j] += F::apply(wv, xc[j], kbx * F::KBY, kqs);
            }
        }
#pragma unroll
        for (int j = 0; j < NC; ++j) acc[j] = rw == 0 ? c[j] : acc[j] + c[j];
    }
#pragma unroll
    for (int j = 0; j < NC; ++j) acc[j] = warp_sum(acc[j]);
}

// mmvq_multi_kernel's row_dot_multi for IQ3_S: calls k = lane, lane + 32, ... of 8 per 256-value block.
template<int NC, int NB, typename X>
__device__ __forceinline__ void row_dot_iq3s(const uint8_t* row, const X& x, int nb32, int nb_rt, int lane,
                                             float (&acc)[NC]) {
    const int nk = (NB > 0 ? NB : nb_rt) * 8;
    X xc[NC];
#pragma unroll
    for (int j = 0; j < NC; ++j) xc[j] = x.col(j, nb32);
#pragma unroll
    for (int j = 0; j < NC; ++j) acc[j] = 0.0f;
    if constexpr (NB > 0) {
#pragma unroll
        for (int i = 0; i < (NB * 8 + 31) / 32; ++i) {
            const int k = lane + 32 * i;
            if (k < nk) {
                const IQ3SF::W w = IQ3SF::load(row, k / 8, 2 * (k % 8));
#pragma unroll
                for (int j = 0; j < NC; ++j) acc[j] += IQ3SF::apply(w, xc[j], k);
            }
        }
    } else {
        for (int k = lane; k < nk; k += 32) {
            const IQ3SF::W w = IQ3SF::load(row, k / 8, 2 * (k % 8));
#pragma unroll
            for (int j = 0; j < NC; ++j) acc[j] += IQ3SF::apply(w, xc[j], k);
        }
    }
#pragma unroll
    for (int j = 0; j < NC; ++j) acc[j] = warp_sum(acc[j]);
}

template<int NC, int NB, typename X>
__device__ __forceinline__ void row_dot(int type, const uint8_t* row, const X& x, int nb32, int nb, int lane,
                                        float (&acc)[NC]) {
    switch (type) {
        case 12: row_dot4<Q4KF, NC, NB>(row, x, nb32, nb, lane, acc); break;
        case 13: row_dot4<Q5KF, NC, NB>(row, x, nb32, nb, lane, acc); break;
        case 14: row_dot4<Q6KF, NC, NB>(row, x, nb32, nb, lane, acc); break;
        case 23: row_dot4<IQ4XSF, NC, NB>(row, x, nb32, nb, lane, acc); break;
        default: row_dot_iq3s<NC, NB>(row, x, nb32, nb, lane, acc); break;
    }
}

// ---------------------------------------------------------------- bulk copies into shared memory
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t) __cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
    uint32_t done = 0;
    do {
        asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
                     : "=r"(done) : "r"(smem_u32(bar)), "r"(parity) : "memory");
    } while (!done);
}
// `bytes` from global `src` (both 16-byte aligned, bytes a multiple of 16) into shared `dst`, completing on `bar`
__device__ __forceinline__ void bulk_load(void* dst, const void* src, uint32_t bytes, uint64_t* bar, uint64_t policy) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
                 ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)), "l"(policy) : "memory");
}
__device__ __forceinline__ uint64_t evict_last_policy() {
    uint64_t p;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
    return p;
}
__device__ __forceinline__ uint64_t evict_first_policy() {
    uint64_t p;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
    return p;
}
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

// ---------------------------------------------------------------- the dense kernel
// A persistent block per SM of NW warps.  Warp gw of NWT owns rows gw, gw + NWT, ... of the concatenated matrices and
// streams them through its own ring of nslot shared-memory slots: one bulk copy per row, issued nslot rows ahead
// (the first ones before the grid dependency wait: weights are constant).
constexpr int kMaxMats = 3;
constexpr int kSlotsPerWarp = 8;

struct DenseArgs {
    int type[kMaxMats];
    const uint8_t* w[kMaxMats];
    int n_out[kMaxMats];
    int row_bytes[kMaxMats];
    float* y[kMaxMats];
    int n_mats, rows, n_in;
    const float* x;          // fp32 columns to quantize (by quant_x_kernel into xprep), or
    const uint8_t* xprep;    // their quant_x_kernel layout, copied into shared memory, or
    const Q81Block* xq;      // q8_1 columns in global memory
    int nslot, slot_bytes;   // the ring
};

__device__ __forceinline__ void row_of(const DenseArgs& a, int g, int& m, int& r) {
    m = 0;
    while (m + 1 < a.n_mats && g >= a.n_out[m]) g -= a.n_out[m++];
    r = g;
}

__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
}

// The q8_1 of the in-projection's columns, once: codes | fp16 scales as floats | code sums (the layout dense_block
// copies into shared memory).  native_quantize_q8_1_kernel's codes and scale (the warp max as an integer max of the
// magnitudes' bits: the same value).  `n` = columns x n_in, a multiple of 32.
__device__ __forceinline__ void quant_x_block(const float* __restrict__ x, int n, uint8_t* __restrict__ out, int blk,
                                              int nthreads) {
    const int i = blk * nthreads + (int) threadIdx.x;
    if (i >= n) return;   // whole warps
    int8_t* qs = reinterpret_cast<int8_t*>(out);
    float* d_out = reinterpret_cast<float*>(out + n);
    int* s_out = reinterpret_cast<int*>(d_out + n / Q8K);
    const float xi = x[i];
    const float amax = __uint_as_float(__reduce_max_sync(0xffffffffu, __float_as_uint(fabsf(xi))));
    const float d = q8_1_finite(amax / 127.0f);
    const int8_t q = q8_1_quant(xi, d, amax);
    qs[i] = q;
    const int qsum = __reduce_add_sync(0xffffffffu, (int) q);
    if (i % Q8K == 0) {
        d_out[i / Q8K] = __half2float(__float2half(q8_1_finite(d)));
        s_out[i / Q8K] = qsum;
    }
}

__global__ void __launch_bounds__(256) quant_x_kernel(const float* __restrict__ x, int n, uint8_t* __restrict__ out) {
    pdl_trigger();
    pdl_wait();
    quant_x_block(x, n, out, blockIdx.x, blockDim.x);
}

template<int NC, int NB, bool SMX, int NW>
__device__ __forceinline__ void dense_block(const DenseArgs& a, int blk, int nblk, uint8_t* smem) {
    const int warp = threadIdx.x / WARP, lane = threadIdx.x % WARP;
    const int nb32 = a.n_in / Q8K;
    uint64_t* bars = reinterpret_cast<uint64_t*>(smem) + warp * kSlotsPerWarp;
    int8_t* xs_q = reinterpret_cast<int8_t*>(smem + NW * kSlotsPerWarp * 8);   // the columns' codes, scales, sums
    float* xs_d = reinterpret_cast<float*>(xs_q + NC * a.n_in);
    int* xs_s = reinterpret_cast<int*>(xs_d + NC * nb32);
    const int x_bytes = SMX ? NC * nb32 * ((int) sizeof(Q81Block) + 4) : 0;
    uint8_t* slots = smem + NW * kSlotsPerWarp * 8 + x_bytes + (size_t) warp * a.nslot * a.slot_bytes;
    const int gw = blk * NW + warp, nwt = nblk * NW;
    const int nrows = a.rows > gw ? (a.rows - gw + nwt - 1) / nwt : 0;
    const uint64_t policy = evict_first_policy();
    auto issue = [&](int k) {
        int m, r;
        row_of(a, gw + k * nwt, m, r);
        const uintptr_t src = (uintptr_t) (a.w[m] + (size_t) r * a.row_bytes[m]);
        const uintptr_t a0 = src & ~(uintptr_t) 15;
        const uint32_t bytes = (uint32_t) (((src - a0) + a.row_bytes[m] + 15) & ~(uintptr_t) 15);
        const int s = k % a.nslot;
        bulk_load(slots + (size_t) s * a.slot_bytes, (const void*) a0, bytes, bars + s, policy);
    };
    uint64_t* xbar = reinterpret_cast<uint64_t*>(smem) + NW * kSlotsPerWarp - 1;   // the last warp's spare slot barrier
    if (lane == 0) {
        for (int s = 0; s < a.nslot; ++s) mbar_init(bars + s, 1);
        if (SMX && warp == 0) mbar_init(xbar, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        for (int k = 0; k < a.nslot && k < nrows; ++k) issue(k);
    }
    if constexpr (SMX) __syncthreads();   // xbar initialized
    pdl_trigger();
    pdl_wait();
    if constexpr (SMX) {   // the quantized columns, prepared by quant_x_kernel
        if (threadIdx.x == 0) bulk_load(xs_q, a.xprep, (uint32_t) x_bytes, xbar, evict_last_policy());
        mbar_wait(xbar, 0);
    }
    __syncwarp();
    const auto x = [&] {
        if constexpr (SMX) return XSm{reinterpret_cast<const int*>(xs_q), xs_d, xs_s};
        else return XGl{a.xq};
    }();
    for (int k = 0; k < nrows; ++k) {
        const int s = k % a.nslot;
        mbar_wait(bars + s, (uint32_t) (k / a.nslot) & 1u);
        int m, r;
        row_of(a, gw + k * nwt, m, r);
        const uintptr_t src = (uintptr_t) (a.w[m] + (size_t) r * a.row_bytes[m]);
        const uint8_t* row = slots + (size_t) s * a.slot_bytes + (src & 15);
        float acc[NC];
        row_dot<NC, NB>(a.type[m], row, x, nb32, a.n_in / QK, lane, acc);
        __syncwarp();
        if (lane == 0 && k + a.nslot < nrows) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            issue(k + a.nslot);
        }
#pragma unroll
        for (int j = 0; j < NC; ++j)
            if (lane == j) a.y[m][(size_t) j * a.n_out[m] + r] = acc[j];
    }
}

template<int NC, int NB, bool SMX, int NW>
__global__ void __launch_bounds__(NW * WARP, 1) dense_kernel(const DenseArgs a) {
    extern __shared__ __align__(128) uint8_t smem[];
    dense_block<NC, NB, SMX, NW>(a, blockIdx.x, gridDim.x, smem);
}

int row_bytes_of(int type, int n_in) {
    switch (type) {
        case 12: return n_in / 256 * 144;
        case 13: return n_in / 256 * 176;
        case 14: return n_in / 256 * 210;
        case 21: return n_in / 256 * 110;
        case 23: return n_in / 256 * 136;
        default: return -1;
    }
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

int smem_optin() {
    static const int n = [] {
        int dev = 0, v = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        return v;
    }();
    return n;
}

template<int NC, int NB, bool SMX>
void launch_nb(DenseArgs a, cudaStream_t s) {
    constexpr int NW = NC <= 3 ? 16 : 8;
    const int x_bytes = a.x ? NC * (a.n_in / Q8K) * ((int) sizeof(Q81Block) + 4) : 0;
    const int fixed = NW * kSlotsPerWarp * 8 + x_bytes;
    const int budget = smem_optin() - 1024;
    a.nslot = (budget - fixed) / (NW * a.slot_bytes);
    if (a.nslot > kSlotsPerWarp - 1) a.nslot = kSlotsPerWarp - 1;   // the last barrier of the block is the x copy's
    if (a.nslot < 1) { std::fprintf(stderr, "dec_gdn: rows too wide for shared memory\n"); std::exit(1); }
    const int smem = fixed + NW * a.nslot * a.slot_bytes;
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute(dense_kernel<NC, NB, SMX, NW>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_optin());
        attr = true;
    }
    int blocks = sm_count();
    const int need = (a.rows + NW - 1) / NW;
    if (need < blocks) blocks = need;
    launch_pdl(dense_kernel<NC, NB, SMX, NW>, dim3((unsigned) blocks), dim3(NW * WARP), (size_t) smem, s, a);
}

// the block counts of the GDN projections' rows unrolled, any other one looped
template<int NC>
void launch_nc(const DenseArgs& a, cudaStream_t s) {
    const int nb = a.n_in / QK;
    if (a.x) {
        if (nb == 10) launch_nb<NC, 10, true>(a, s);
        else launch_nb<NC, 0, true>(a, s);
    } else {
        if (nb == 24) launch_nb<NC, 24, false>(a, s);
        else launch_nb<NC, 0, false>(a, s);
    }
}

void launch(DenseArgs a, int n_tok, cudaStream_t s) {
    switch (n_tok) {
        case 1: launch_nc<1>(a, s); break;
        case 2: launch_nc<2>(a, s); break;
        case 3: launch_nc<3>(a, s); break;
        case 4: launch_nc<4>(a, s); break;
        case 5: launch_nc<5>(a, s); break;
        case 6: launch_nc<6>(a, s); break;
        case 7: launch_nc<7>(a, s); break;
        case 8: launch_nc<8>(a, s); break;
        default: std::fprintf(stderr, "dec_gdn: n_tok %d out of range\n", n_tok); std::exit(1);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "dec_gdn dense: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

DenseArgs make_args(const DecMat* m, int n_mats, int n_in) {
    if (n_mats < 1 || n_mats > kMaxMats || n_in % 256 != 0) {
        std::fprintf(stderr, "dec_gdn: bad matrix list\n");
        std::exit(1);
    }
    DenseArgs a{};
    a.n_mats = n_mats;
    a.n_in = n_in;
    int widest = 0;
    for (int i = 0; i < n_mats; ++i) {
        if (!dec_dense_supported(m[i].type, n_in) || ((uintptr_t) m[i].w & 3) != 0) {
            std::fprintf(stderr, "dec_gdn: type %d / n_in %d not supported\n", m[i].type, n_in);
            std::exit(1);
        }
        a.type[i] = m[i].type;
        a.w[i] = (const uint8_t*) m[i].w;
        a.n_out[i] = m[i].n_out;
        a.row_bytes[i] = row_bytes_of(m[i].type, n_in);
        a.y[i] = m[i].y;
        a.rows += m[i].n_out;
        if (a.row_bytes[i] > widest) widest = a.row_bytes[i];
    }
    a.slot_bytes = (widest + 15 + 15) & ~15;
    return a;
}

}  // namespace

bool dec_dense_supported(int t, int n_in) noexcept {
    return n_in % 256 == 0 && (t == 12 || t == 13 || t == 14 || t == 21 || t == 23);
}

void dec_gdn_inproj(const float* x, int n_in, int n_tok, const DecMat* m, int n_mats, void* scratch, void* stream) {
    DenseArgs a = make_args(m, n_mats, n_in);
    if (((uintptr_t) scratch & 15) != 0) { std::fprintf(stderr, "dec_gdn_inproj: scratch must be 16-byte aligned\n"); std::exit(1); }
    a.x = x;
    a.xprep = (const uint8_t*) scratch;
    const int n = n_tok * n_in;
    launch_pdl(quant_x_kernel, dim3((unsigned) ((n + 255) / 256)), dim3(256), 0, (cudaStream_t) stream, x, n,
               (uint8_t*) scratch);
    launch(a, n_tok, (cudaStream_t) stream);
}

void dec_dense(const void* xq, int n_in, int n_tok, const DecMat& m, void* stream) {
    DenseArgs a = make_args(&m, 1, n_in);
    a.xq = (const Q81Block*) xq;
    launch(a, n_tok, (cudaStream_t) stream);
}

int dec_gdn_mode() { return 0; }

bool dec_gdn_enabled() { return dec_gdn_mode() != 0; }

}  // namespace strata::kernels
