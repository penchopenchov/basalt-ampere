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

// ---------------------------------------------------------------- native_mmvq.cu, unchanged
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

__device__ __forceinline__ float q5_q8_dot_impl(
    const int* __restrict__ vl, const int* __restrict__ vh, const int* __restrict__ u,
    const uint8_t* __restrict__ sc, const uint8_t* __restrict__ m, const half2& dm5,
    const float* __restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vl0i = (vl[0] >> (4 * i)) & 0x0f0f0f0f;
        const int vl1i = (vl[1] >> (4 * i)) & 0x0f0f0f0f;
        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;
        const int dot1 = STRATA_DP4A(v0i, u[2 * i], STRATA_DP4A(v1i, u[2 * i + 1], 0));
        const int dot2 = STRATA_DP4A(0x01010101, u[2 * i], STRATA_DP4A(0x01010101, u[2 * i + 1], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const float2 dm5f = __half22float2(dm5);
    return dm5f.x * sumf_d - dm5f.y * sumf_m;
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

__device__ __forceinline__ float q4_q8_dot_impl(
    const int* __restrict__ v, const int* __restrict__ u,
    const uint8_t* __restrict__ sc, const uint8_t* __restrict__ m, const half2& dm4,
    const float* __restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v0i = (v[0] >> (4 * i)) & 0x0f0f0f0f;
        const int v1i = (v[1] >> (4 * i)) & 0x0f0f0f0f;
        const int dot1 = STRATA_DP4A(v1i, u[2 * i + 1], STRATA_DP4A(v0i, u[2 * i], 0));
        const int dot2 = STRATA_DP4A(0x01010101, u[2 * i + 1], STRATA_DP4A(0x01010101, u[2 * i], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const float2 dm4f = __half22float2(dm4);
    return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

__device__ __forceinline__ float q6_q8_dot_impl(int vl, int vh, const int* __restrict__ u,
                                              const int8_t* __restrict__ scales,
                                              float d, const float* __restrict__ d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int sc = scales[4 * i];
        const int vil = (vl >> (4 * i)) & 0x0f0f0f0f;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi = __vsubss4(vil | vih, 0x20202020);
        sumf += d8[i] * (STRATA_DP4A(vi, u[i], 0) * sc);
    }
    return d * sumf;
}

struct Q5KTraits {
    using Block = Q5KBlock;
    static constexpr int DIV = QK, T = QI / VDR, KBY = QK / Q8K, BPI = VDR * WARPS * WARP / QI;
    __device__ static int kqs(int tid) { return VDR * (tid % (QI / VDR)); }
    struct W { int vl[2], vh[2]; uint16_t aux[2]; half2 dm; int bq8_offset; };
    __device__ static W load(const Block* __restrict__ bq5, int iqs) {
        W r;
        r.bq8_offset = 2 * ((iqs / 2) / 4);
        const int* ql = reinterpret_cast<const int*>(bq5->qs + 16 * r.bq8_offset + 4 * ((iqs / 2) % 4));
        const int* qh = reinterpret_cast<const int*>(bq5->qh + 4 * ((iqs / 2) % 4));
        r.vl[0] = ql[0];
        r.vl[1] = ql[4];
        r.vh[0] = qh[0] >> r.bq8_offset;
        r.vh[1] = qh[4] >> r.bq8_offset;
        const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq5->scales);
        const int j = r.bq8_offset / 2;
        const int jm = j & 1;
        const uint32_t s0 = scales[jm];
        const uint32_t s2 = scales[jm + 2];
        const uint32_t s4 = scales[jm + 4];
        const uint32_t hi = uint32_t(-int32_t(j >= 2));
        r.aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
        r.aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
        r.dm = bq5->dm;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ bq8, int iqs) {
        int u[4];
        float d8[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const Q81Block* bq8i = bq8 + r.bq8_offset + i;
            d8[i] = __low2float(bq8i->ds);
            const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
            u[2 * i] = q8[0];
            u[2 * i + 1] = q8[4];
        }
        const uint8_t* sc = reinterpret_cast<const uint8_t*>(r.aux);
        return q5_q8_dot_impl(r.vl, r.vh, u, sc, sc + 2, r.dm, d8);
    }
};
struct Q4KTraits {
    using Block = Q4KBlock;
    static constexpr int DIV = QK, T = QI / VDR, KBY = QK / Q8K, BPI = VDR * WARPS * WARP / QI;
    __device__ static int kqs(int tid) { return VDR * (tid % (QI / VDR)); }
    struct W { int v[2]; uint16_t aux[2]; half2 dm; int bq8_offset; };
    __device__ static W load(const Block* __restrict__ bq4, int iqs) {
        W r;
        r.bq8_offset = 2 * ((iqs / 2) / 4);
        const int* ql = reinterpret_cast<const int*>(bq4->qs + 16 * r.bq8_offset + 4 * ((iqs / 2) % 4));
        r.v[0] = ql[0];
        r.v[1] = ql[4];
        const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq4->scales);
        const int j = r.bq8_offset / 2;
        const int jm = j & 1;
        const uint32_t s0 = scales[jm];
        const uint32_t s2 = scales[jm + 2];
        const uint32_t s4 = scales[jm + 4];
        const uint32_t hi = uint32_t(-int32_t(j >= 2));
        r.aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
        r.aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
        r.dm = bq4->dm;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ bq8, int iqs) {
        int u[4];
        float d8[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const Q81Block* bq8i = bq8 + r.bq8_offset + i;
            d8[i] = __low2float(bq8i->ds);
            const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
            u[2 * i] = q8[0];
            u[2 * i + 1] = q8[4];
        }
        const uint8_t* sc = reinterpret_cast<const uint8_t*>(r.aux);
        return q4_q8_dot_impl(r.v, u, sc, sc + 2, r.dm, d8);
    }
};
struct Q6KTraits {
    using Block = Q6KBlock;
    static constexpr int DIV = 256, T = 32, KBY = 8, BPI = WARPS * WARP / 32;
    __device__ static int kqs(int tid) { return tid % 32; }
    struct W { int vl, vh; float d; const int8_t* scales; int bq8_offset; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
        r.bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
        const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
        const int vh_shift = 2 * ((iqs % 16) / 8);
        r.vl = load_int_b2(w->ql, iqs);
        r.vh = load_int_b2(w->qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
        r.scales = w->scales + scale_offset;
        r.d = w->d;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int u[2];
        float d8[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            u[i] = reinterpret_cast<const int*>(x[r.bq8_offset + 2 * i].qs)[iqs % 8];
            d8[i] = __low2float(x[r.bq8_offset + 2 * i].ds);
        }
        return q6_q8_dot_impl(r.vl, r.vh, u, r.scales, r.d, d8);
    }
};
struct IQ4XSTraits {
    using Block = IQ4XSBlock;
    static constexpr int DIV = 256, T = 8, KBY = 8, BPI = 4 * WARPS * WARP / 32;
    __device__ static int kqs(int tid) { return 4 * (tid % 8); }
    struct W { int2 v[4]; int ls; float dw; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
#pragma unroll
        for (int j = 0; j < 4; ++j) r.v[j] = iq4_table_lookup(reinterpret_cast<const int*>(w->qs)[iqs + j]);
        r.ls = ((w->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0f) | (((w->scales_h >> (iqs / 2)) & 0x03) << 4);
        r.dw = __half2float(w->d);
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int u0 = reinterpret_cast<const int*>(x[iqs / 4].qs)[j];
            const int u1 = reinterpret_cast<const int*>(x[iqs / 4].qs)[j + 4];
            sumi = STRATA_DP4A(r.v[j].x, u0, sumi);
            sumi = STRATA_DP4A(r.v[j].y, u1, sumi);
        }
        sumi *= r.ls - 32;
        const float d = r.dw * __low2float(x[iqs / 4].ds);
        return d * sumi;
    }
};

// ---------------------------------------------------------------- iq_kernels.cu Split<21> (IQ3_S), unchanged
__device__ __forceinline__ int get_int_b2(const void* x, const int& i32) {
    const uint16_t* x16 = (const uint16_t*) x;
    int x32 = x16[2 * i32 + 0] << 0;
    x32 |= x16[2 * i32 + 1] << 16;
    return x32;
}
__device__ __forceinline__ int get_int_b4(const void* x, const int& i32) { return ((const int*) x)[i32]; }

struct IQ3STraits {
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
    __device__ static float apply(const W& r, const Q81Block* __restrict__ bq8_1, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) sumi = STRATA_DP4A(r.g[j], get_int_b4(bq8_1[iqs / 2].qs, j), sumi);
        sumi *= r.ls;
        const float d = r.dw * __low2float(bq8_1[iqs / 2].ds);
        return d * sumi;
    }
};

// ---------------------------------------------------------------- row dots (one warp per row)
// The exact layout's four warps, replayed by one: lane `lane` sums the chains of reference threads lane + 32 rw.
template<typename F, int NC>
__device__ __forceinline__ void row_dot4(const uint8_t* row, const Q81Block* x, int xs, int nb, int lane,
                                         float (&acc)[NC]) {
    const auto* w = reinterpret_cast<const typename F::Block*>(row);
#pragma unroll
    for (int rw = 0; rw < WARPS; ++rw) {
        const int tid = WARP * rw + lane;
        const int kqs = F::kqs(tid);
        float c[NC];
#pragma unroll
        for (int j = 0; j < NC; ++j) c[j] = 0.0f;
        for (int kbx = tid / F::T; kbx < nb; kbx += F::BPI) {
            const typename F::W wv = F::load(w + kbx, kqs);
#pragma unroll
            for (int j = 0; j < NC; ++j) c[j] += F::apply(wv, x + j * xs + kbx * F::KBY, kqs);
        }
#pragma unroll
        for (int j = 0; j < NC; ++j) acc[j] = rw == 0 ? c[j] : acc[j] + c[j];
    }
#pragma unroll
    for (int j = 0; j < NC; ++j) acc[j] = warp_sum(acc[j]);
}

// mmvq_multi_kernel's row_dot_multi for IQ3_S: calls k = lane, lane + 32, ... of 8 per 256-value block.
template<int NC>
__device__ __forceinline__ void row_dot_iq3s(const uint8_t* row, const Q81Block* x, int xs, int nb, int lane,
                                             float (&acc)[NC]) {
#pragma unroll
    for (int j = 0; j < NC; ++j) acc[j] = 0.0f;
    for (int k = lane; k < nb * 8; k += 32) {
        const int kbx = k / 8, iqs = 2 * (k % 8);
        const IQ3STraits::W w = IQ3STraits::load(row, kbx, iqs);
#pragma unroll
        for (int j = 0; j < NC; ++j) acc[j] += IQ3STraits::apply(w, x + j * xs + kbx * 8, iqs);
    }
#pragma unroll
    for (int j = 0; j < NC; ++j) acc[j] = warp_sum(acc[j]);
}

template<int NC>
__device__ __forceinline__ void row_dot(int type, const uint8_t* row, const Q81Block* x, int xs, int nb, int lane,
                                        float (&acc)[NC]) {
    switch (type) {
        case 12: row_dot4<Q4KTraits, NC>(row, x, xs, nb, lane, acc); break;
        case 13: row_dot4<Q5KTraits, NC>(row, x, xs, nb, lane, acc); break;
        case 14: row_dot4<Q6KTraits, NC>(row, x, xs, nb, lane, acc); break;
        case 23: row_dot4<IQ4XSTraits, NC>(row, x, xs, nb, lane, acc); break;
        default: row_dot_iq3s<NC>(row, x, xs, nb, lane, acc); break;
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
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
                 ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)), "l"(policy) : "memory");
}
__device__ __forceinline__ uint64_t evict_first_policy() {
    uint64_t p;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
    return p;
}
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

// ---------------------------------------------------------------- the dense kernel
constexpr int kMaxMats = 3;
constexpr int kNW = 16;            // warps per block
constexpr int kMaxSlots = 4;

struct DenseArgs {
    int type[kMaxMats];
    const uint8_t* w[kMaxMats];
    int n_out[kMaxMats];
    int row_bytes[kMaxMats];
    float* y[kMaxMats];
    int n_mats, rows, n_in;
    const float* x;          // fp32 columns to quantize in shared memory, or
    const Q81Block* xq;      // q8_1 columns in global memory
    int nslot, slot_bytes;   // the per-warp ring
};

__device__ __forceinline__ void issue_row(const DenseArgs& a, int g, void* slot, uint64_t* bar, uint64_t policy) {
    int m = 0;
    while (m + 1 < a.n_mats && g >= a.n_out[m]) g -= a.n_out[m++];
    const uintptr_t src = (uintptr_t) (a.w[m] + (size_t) g * a.row_bytes[m]);
    const uintptr_t a0 = src & ~(uintptr_t) 15;
    const uint32_t bytes = (uint32_t) (((src - a0) + a.row_bytes[m] + 15) & ~(uintptr_t) 15);
    bulk_load(slot, (const void*) a0, bytes, bar, policy);
}

// The block's part of the projections: warp `warp` of block `blk` of `nblk`.  `smem` holds the barriers, the
// quantized columns (with a.x) and the slots.
template<int NC>
__device__ __forceinline__ void dense_block(const DenseArgs& a, int blk, int nblk, uint8_t* smem) {
    const int warp = threadIdx.x / WARP, lane = threadIdx.x % WARP;
    const int nb32 = a.n_in / Q8K;
    uint64_t* bars = reinterpret_cast<uint64_t*>(smem) + warp * kMaxSlots;
    Q81Block* xs = reinterpret_cast<Q81Block*>(smem + kNW * kMaxSlots * 8);
    const int x_bytes = a.x ? (NC * nb32 * (int) sizeof(Q81Block) + 15) & ~15 : 0;
    uint8_t* slots = smem + kNW * kMaxSlots * 8 + x_bytes + warp * a.nslot * a.slot_bytes;
    const int gw = blk * kNW + warp, nwt = nblk * kNW;
    const int nrows = a.rows > gw ? (a.rows - gw + nwt - 1) / nwt : 0;
    const uint64_t policy = evict_first_policy();
    if (lane == 0) {
        for (int s = 0; s < a.nslot; ++s) mbar_init(bars + s, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        for (int k = 0; k < a.nslot && k < nrows; ++k) issue_row(a, gw + k * nwt, slots + k * a.slot_bytes, bars + k, policy);
    }
    pdl_trigger();
    pdl_wait();
    const Q81Block* x = a.xq;
    if (a.x) {   // native_quantize_q8_1_kernel, element by element
        for (int i = threadIdx.x; i < NC * a.n_in; i += blockDim.x) {
            const float xi = a.x[i];
            const float amax = warp_max(fabsf(xi));
            const float sum = warp_sum(xi);
            const float d = q8_1_finite(amax / 127.0f);
            const int8_t q = q8_1_quant(xi, d, amax);
            xs[i / Q8K].qs[i % Q8K] = q;
            if (i % Q8K == 0) xs[i / Q8K].ds = q8_1_ds(d, sum);
        }
        __syncthreads();
        x = xs;
    }
    __syncwarp();
    for (int k = 0; k < nrows; ++k) {
        const int s = k % a.nslot;
        mbar_wait(bars + s, (uint32_t) (k / a.nslot) & 1u);
        int g = gw + k * nwt, m = 0;
        while (m + 1 < a.n_mats && g >= a.n_out[m]) g -= a.n_out[m++];
        const uintptr_t src = (uintptr_t) (a.w[m] + (size_t) g * a.row_bytes[m]);
        const uint8_t* row = slots + s * a.slot_bytes + (src & 15);
        float acc[NC];
        row_dot<NC>(a.type[m], row, x, nb32, a.n_in / QK, lane, acc);
#pragma unroll
        for (int j = 0; j < NC; ++j)
            if (lane == j) a.y[m][(size_t) j * a.n_out[m] + g] = acc[j];
        __syncwarp();
        if (lane == 0 && k + a.nslot < nrows) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            issue_row(a, gw + (k + a.nslot) * nwt, slots + s * a.slot_bytes, bars + s, policy);
        }
    }
}

template<int NC>
__global__ void __launch_bounds__(kNW * WARP, 1) dense_kernel(const DenseArgs a) {
    extern __shared__ __align__(128) uint8_t smem[];
    dense_block<NC>(a, blockIdx.x, gridDim.x, smem);
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

template<int NC>
void launch_nc(DenseArgs a, cudaStream_t s) {
    const int x_bytes = a.x ? (NC * (a.n_in / Q8K) * (int) sizeof(Q81Block) + 15) & ~15 : 0;
    const int fixed = kNW * kMaxSlots * 8 + x_bytes;
    const int budget = smem_optin() - 1024;
    a.nslot = (budget - fixed) / (kNW * a.slot_bytes);
    if (a.nslot > kMaxSlots) a.nslot = kMaxSlots;
    if (a.nslot < 1) { std::fprintf(stderr, "dec_gdn: rows too wide for shared memory\n"); std::exit(1); }
    const int smem = fixed + kNW * a.nslot * a.slot_bytes;
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute(dense_kernel<NC>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_optin());
        attr = true;
    }
    int blocks = sm_count();
    const int need = (a.rows + kNW - 1) / kNW;
    if (need < blocks) blocks = need;
    launch_pdl(dense_kernel<NC>, dim3((unsigned) blocks), dim3(kNW * WARP), (size_t) smem, s, a);
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

void dec_gdn_inproj(const float* x, int n_in, int n_tok, const DecMat* m, int n_mats, void* stream) {
    DenseArgs a = make_args(m, n_mats, n_in);
    a.x = x;
    launch(a, n_tok, (cudaStream_t) stream);
}

void dec_dense(const void* xq, int n_in, int n_tok, const DecMat& m, void* stream) {
    DenseArgs a = make_args(&m, 1, n_in);
    a.xq = (const Q81Block*) xq;
    launch(a, n_tok, (cudaStream_t) stream);
}

bool dec_gdn_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_DEC_GDN");
        return v != nullptr && std::atoi(v) != 0;
    }();
    return on;
}

}  // namespace strata::kernels
