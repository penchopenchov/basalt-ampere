#pragma once
// The MoE half of a decode layer (routed experts resident on CUDA0 + shared expert + combine) as one
// persistent kernel, one CTA per SM, 16 warps (512 threads), bitwise equal to today's kernel sequence
// (dec_moe_glue gate/up + SwiGLU + down, shared_expert_multi, dec_moe_wait, dec_moe_combine).
//
// Input release (as the attention half hands over): the window's top-10 ids/weights and q8_1(mixed) are visible.  Every CTA plans the
// layer itself from the ids (dm_glue's grouping, no plan hop), then:
//  - gate/up items (expert group g, ff block j of 32: 32 gate + 32 up rows) round-robin over CTAs, the shared
//    expert's 20 items first; the item owner applies SwiGLU and writes the q8_1 block of h (one per entry), then
//    counts the item in its group's counter.
//  - down by row ownership: CTA b owns output rows [R0, R1) (multiples of 4, sized so every CTA streams the same
//    bytes) for EVERY group and the shared expert, waiting per group on the gate/up counter, so the combine of those
//    rows is local (no combine hop).
//  - every byte this CTA will read (its gate/up items, its down slices) is prefetched into L2 at the release, in
//    schedule order: DRAM streams the whole layer while compute follows.
// Numerics: every row is computed by today's per-row reduction (gu2 rows2, row_dot_40_sub8 / row_dot_20_sub16, the
// mmvq exact layouts for the shared expert, mmvf for its gate), fast-math files emulated with .ftz/.approx PTX.
// Shared expert formats per layer (LayerM sgt/sut/sdt): gate/up Q4_K, Q5_K, Q6_K, IQ4_XS, IQ3_S; down IQ4_NL, Q8_0.
#include "iq_kernels.cu"
#include "basalt/kernels/shared_expert.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include <algorithm>
#include <cstring>
#include <cstdio>


namespace basalt::kernels {
namespace mkb {

constexpr int N = 2560, NFF = 640, KT = 10, TMX = 4, CT = 512, NWARP = 16, TMXH = 8;
constexpr int XQB = N / 32, HB = NFF / 32;
constexpr int C_REL = 0, C_END = 1, C_SH = 2, C_SG = 3, C_G0 = 4, NCTR = C_G0 + kDmCap + 4;
constexpr int SH_GROW = 1360, SH_UROW = 1100, SH_DROW = 360;
constexpr int MAXR = 48, HMAX = 40, NENT = TMX * KT, LW = 15, CWN = 15, MAXIT = 4, STG = 768, NXE = 16;   // rows per CTA, entries per down batch, entries
constexpr int NST = 32;   // [16, 32): Q(n) probe points
enum { S_SEEN, S_PLAN, S_XQ, S_GU, S_PF, S_DN, S_SHD, S_I1, S_END, S_IDS, S_DW, S_DC, S_DK, S_DB, S_RDY };

struct LayerM {
    NativeExpertLayout L;
    const unsigned long long* ptab;   // [512] CUDA0 blob address or 0
    const int32_t* res1;              // [512] peer slot or -1
    const uint8_t* sg; const uint8_t* su; const uint8_t* sd; const uint16_t* sgi;
    int sgt, sut, sdt;                // the shared expert's gate/up/down GGML types
    int glue;                         // today's glue path covers the layer (else route + native_expert_grouped)
};
struct IterM {
    int layer;
    const int32_t* ids; const float* w;
    const block_q8_1* xq; const float* mixed;
    const block_q8_1* xqr;            // the routed experts' q8_1 rows (== xq on glue layers; IEEE q8_1_store on the others)
    float* bo;
};
struct KP {
    const LayerM* lay; const IterM* it; int iters, T; long long delay_ns; int mode; long long ahead; long long dnpf_ns; long long guw;
    const float* peer_rows; const float* cpu_rows;
    const unsigned* peer_done; const unsigned* cpu_done;
    unsigned* cnt;                     // [iters][NCTR]
    uint4* hq; uint4* hqs;             // tagged q8_1 blocks: [kDmCap][HB][3], [TMX][HB][3] (12 payload bytes + epoch)
    uint2* gs;                         // [TMX] (shared gate dot bits, epoch)
    unsigned long long* tx; unsigned long long* tend;
    unsigned long long* st;            // [iters][G][NST] or null
    float* dbg_parts;                  // [kDmCap][N] (mode & 8) or null
};

// ------------------------------------------------------------ small device helpers
__device__ unsigned* g_err;
__device__ __forceinline__ unsigned ldacq(const unsigned* p) { unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0,[%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void strel(unsigned* p, unsigned v) { asm volatile("st.release.gpu.global.u32 [%0],%1;" ::"l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void redrel(unsigned* p) { asm volatile("red.release.gpu.global.add.u32 [%0],1;" ::"l"(p) : "memory"); }
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(t)); return t; }
// g_err (host-mapped): a wait past the bound: [0] = 1, [1] the waiter's line, [2] block, [3] value, [4] target
__device__ __forceinline__ unsigned long long gtime0() { unsigned long long t; asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(t)); return t; }
#define spin_ge(c, target) spin_ge_(c, target, __LINE__)
__device__ __forceinline__ void spin_ge_(const unsigned* c, unsigned target, int line) {
    unsigned v; unsigned n = 0; unsigned long long t0 = 0;
    while ((v = ldacq(c)) < target) {
        if ((++n & 1023u) == 0) {
            const unsigned long long t = gtime0();
            if (t0 == 0) t0 = t;
            else if (t - t0 > 200000000ull) {   // 0.2 s
                if ((*(volatile unsigned*) &g_err[0]) == 0u && ((*(volatile unsigned*) &g_err[0] = 1u), true)) { g_err[1] = line; g_err[2] = blockIdx.x; g_err[3] = v; g_err[4] = target; __threadfence_system(); }
                asm volatile("trap;");
            }
        }
    }
}
__device__ __forceinline__ float add_ftz(float a, float b) { float d; asm("add.rn.ftz.f32 %0,%1,%2;" : "=f"(d) : "f"(a), "f"(b)); return d; }
__device__ __forceinline__ float mul_ftz(float a, float b) { float d; asm("mul.rn.ftz.f32 %0,%1,%2;" : "=f"(d) : "f"(a), "f"(b)); return d; }
__device__ __forceinline__ float fma_ftz(float a, float b, float c) { float d; asm("fma.rn.ftz.f32 %0,%1,%2,%3;" : "=f"(d) : "f"(a), "f"(b), "f"(c)); return d; }
__device__ __forceinline__ float div_apx_ftz(float a, float b) { float d; asm("div.approx.ftz.f32 %0,%1,%2;" : "=f"(d) : "f"(a), "f"(b)); return d; }
__device__ __forceinline__ float ex2_ftz(float a) { float d; asm("ex2.approx.ftz.f32 %0,%1;" : "=f"(d) : "f"(a)); return d; }
__device__ __forceinline__ float wsum_ftz(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = add_ftz(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float wmax_ftz(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t) __cvta_generic_to_shared(p); }
__device__ __forceinline__ void mb_init(uint64_t* b) { asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(sa(b)) : "memory"); }
__device__ __forceinline__ void mb_wait(uint64_t* b, unsigned ph) {
    unsigned ok = 0; unsigned long long t0 = 0;
    while (true) {
        asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }" : "=r"(ok) : "r"(sa(b)), "r"(ph) : "memory");
        if (ok) break;
        unsigned long long t; asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(t));
        if (t0 == 0) t0 = t;
        else if (t - t0 > 200000000ull) { if ((*(volatile unsigned*) &g_err[0]) == 0u && ((*(volatile unsigned*) &g_err[0] = 1u), true)) { g_err[1] = 9999; g_err[2] = blockIdx.x; __threadfence_system(); } asm volatile("trap;"); }
    }
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, unsigned bytes, uint64_t* b) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(sa(b)), "r"(bytes) : "memory");
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 ::"r"(sa(dst)), "l"(src), "r"(bytes), "r"(sa(b)) : "memory");
}
// [p, p + n) into L2, rounded out to 16 bytes, 32 KB a request
__device__ __forceinline__ void pf_l2(const void* p, size_t n) {
    uintptr_t a = (uintptr_t) p & ~(uintptr_t) 15, e = ((uintptr_t) p + n + 15) & ~(uintptr_t) 15;
    for (; a < e; a += 32768) {
        const unsigned sz = (unsigned) (e - a < 32768 ? e - a : 32768);
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(a), "r"(sz) : "memory");
    }
}

// tagged hop data: a q8_1 block (9 words) as 3 x {3 words, epoch}; no release fence (a fence waits for the SM's
// in-flight prefetches), the consumer checks every vector's epoch
__device__ __forceinline__ void st_rlx4(uint4* p, uint4 v) { asm volatile("st.relaxed.gpu.global.v4.u32 [%0],{%1,%2,%3,%4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory"); }
__device__ __forceinline__ uint4 ld_rlx4(const uint4* p) { uint4 v; asm volatile("ld.relaxed.gpu.global.v4.u32 {%0,%1,%2,%3},[%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory"); return v; }
// lanes 0..2 of the warp that wrote blk: its tagged copy at dst
__device__ __forceinline__ void put_tagged(const block_q8_1* blk, uint4* dst, unsigned tag, int lane) {
    if (lane < 3) { const uint32_t* w = (const uint32_t*) blk; st_rlx4(dst + lane, make_uint4(w[3 * lane], w[3 * lane + 1], w[3 * lane + 2], tag)); }
}
// vector q of a tagged block into its 3 words of dst (a block_q8_1 array), spinning on the epoch
__device__ __forceinline__ void get_tagged(const uint4* src, uint32_t* dstw, unsigned tag, int line) {
    uint4 v; unsigned n = 0; unsigned long long t0 = 0;
    while ((v = ld_rlx4(src)).w != tag) {
        if ((++n & 255u) == 0) {
            const unsigned long long t = gtime0();
            if (t0 == 0) t0 = t;
            else if (t - t0 > 200000000ull) { if ((*(volatile unsigned*) &g_err[0]) == 0u && ((*(volatile unsigned*) &g_err[0] = 1u), true)) { g_err[1] = line; g_err[2] = blockIdx.x; g_err[3] = v.w; g_err[4] = tag; __threadfence_system(); } asm volatile("trap;"); }
        }
    }
    dstw[0] = v.x; dstw[1] = v.y; dstw[2] = v.z;
}

// ------------------------------------------------------------ shared memory
struct Sm {
    alignas(16) block_q8_1 xq[TMX * XQB];
    alignas(16) block_q8_1 xq2[TMX * XQB];    // the routed experts' rows on a non-glue layer
    alignas(16) uint32_t grid[2048];          // the layer's gate/up grid (gu2::Fmt2<TG>::kGridWords)
    alignas(16) uint32_t grid3s[512];         // IQ3_S (the shared up)
    alignas(16) float gu[MAXIT * TMX * 64];   // per item: 32 gate + 32 up rows per entry
    unsigned idone[MAXIT];                    // pairs done per item
    unsigned hrdy[kDmCap + 1];                // group g's h in smem (epoch); [kDmCap] the shared h
    unsigned hclaim[kDmCap + 1];              // the warp copying it (epoch)
    unsigned ucnt, dcnt;                      // gate/up and down units claimed
    float gsv[TMX];                           // the shared gate per token (sigmoid applied)
    int32_t xent[NXE], nxe;                   // the entries of the other tiers
    alignas(16) float prow[NXE * MAXR];       // their rows (this CTA's)
    alignas(16) block_q8_1 h[HMAX * HB];      // the h rows of every CUDA0 entry (plan position)
    alignas(16) block_q8_1 hsh[TMX * HB];
    alignas(16) float out[NENT * MAXR];       // down rows per plan position
    alignas(16) block_q8_1 hst[NWARP];        // a warp's q8_1 block before its tagged store
    alignas(16) float sh[TMX * MAXR];
    alignas(16) unsigned long long ptab[512];
    alignas(16) int32_t res1[512];
    alignas(16) int32_t key[kDmCap];
    int32_t ids[kDmCap], kind[kDmCap], first[kDmCap], cnt[kDmCap], gstart[kDmCap], gidx[kDmCap], pos[kDmCap];
    int32_t start[kDmCap + 1], tok[kDmCap], dst[kDmCap];
    unsigned long long gptr[kDmCap], pp[kDmCap];
    float w[kDmCap];
    int32_t idt[TMX];
    int ng, ne, npeer, ncpu, R0, R1, bnb;
    int nrd[kDmCap];
    long long acc[3], wacc[8][3];
    const uint8_t* ck_a[2 * 8 + kDmCap + 2]; int ck_n[2 * 8 + kDmCap + 2]; long long ck_end[2 * 8 + kDmCap + 2]; int nck, nis;
    uint64_t xbar;
    LayerM lay;
};

// ------------------------------------------------------------ gate/up rows (gu2::rows2 with smem activations)
// rows A, B (B row_bytes after A) for entries e .. e + n of the plan: entry e + c's activation is s.xq + tok * 80;
// row A's sum for entry c to oA[c * os], row B's to oA[c * os + 1]
template<int TY, int NC>
__device__ __forceinline__ void rows2s(const uint8_t* rowA, int rb, const int32_t* tok, int e, int n, const block_q8_1* xq,
                                       float* oA, int os, const uint32_t* grid, int lane) {
    using F = gu2::Fmt2<TY>;
    const uint8_t* rowB = rowA + rb;
    const block_q8_1* xc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) xc[c] = xq + (size_t) tok[e + min(c, n - 1)] * XQB;
    float sA[NC], sB[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) sA[c] = sB[c] = 0.0f;
#pragma unroll
    for (int it = 0; it < 2; ++it) {
        const int k = lane + 32 * it;
        const typename F::W wa = F::load(rowA, k, grid);
        const typename F::W wb = F::load(rowB, k, grid);
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            if (c < n) {
                float d, f;
                F::apply(wa, xc[c] + k, d, f);
                sA[c] = __fmaf_rn(d, f, sA[c]);
                F::apply(wb, xc[c] + k, d, f);
                sB[c] = __fmaf_rn(d, f, sB[c]);
            }
        }
    }
    const int k2 = 64 + (lane & 15);
    const typename F::W w2 = F::load(lane < 16 ? rowA : rowB, k2, grid);
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        if (c < n) {
            float d, f;
            F::apply(w2, xc[c] + k2, d, f);
            const float dB = __shfl_down_sync(0xffffffffu, d, 16), fB = __shfl_down_sync(0xffffffffu, f, 16);
            if (lane < 16) {
                sA[c] = __fmaf_rn(d, f, sA[c]);
                sB[c] = __fmaf_rn(dB, fB, sB[c]);
            }
            sA[c] = warp_sum(sA[c]);
            sB[c] = warp_sum(sB[c]);
            if (lane == 2 * c) oA[c * os] = sA[c];
            if (lane == 2 * c + 1) oA[c * os + 1] = sB[c];
        }
    }
}
// gu2::pair's entry batching over [e0, e1)
template<int TY>
__device__ __forceinline__ void pair_s(const uint8_t* rowA, int rb, const int32_t* tok, int e0, int e1, const block_q8_1* xq,
                                       float* o, int os, const uint32_t* grid, int lane) {
    for (int e = e0; e < e1;) {
        const int n = e1 - e;
        float* oe = o + (e - e0) * os;
        if (n == 1) { rows2s<TY, 1>(rowA, rb, tok, e, 1, xq, oe, os, grid, lane); e += 1; }
        else if (n == 2) { rows2s<TY, 2>(rowA, rb, tok, e, 2, xq, oe, os, grid, lane); e += 2; }
        else if (n == 3) { rows2s<TY, 3>(rowA, rb, tok, e, 3, xq, oe, os, grid, lane); e += 3; }
        else { const int m = min(n, 6); rows2s<TY, 6>(rowA, rb, tok, e, m, xq, oe, os, grid, lane); e += m; }
    }
}

// rows A, B (B rb bytes after A) as native_gu_multi_kernel computes a row (row_dot_multi, GRP_NC entries a call)
template<int TY>
__device__ __forceinline__ void pair_rdm(const uint8_t* rowA, int rb, const int32_t* tok, int e0, int e1, const block_q8_1* xq,
                                         float* o, int os, int lane) {
    const int nb = N / Fmt<TY>::qk;
#pragma unroll 1
    for (int h = 0; h < 2; ++h) {
        const uint8_t* wr = rowA + h * rb;
        for (int e = e0; e < e1; e += GRP_NC) {
            const int n = min(GRP_NC, e1 - e);
            float* oe = o + (e - e0) * os + h;
            if (n == 1) {
                const int off1[1] = { tok[e] * XQB };
                float s1[1];
                row_dot_multi<TY, 1, true>(wr, xq, off1, 1, nb, lane, s1);
                if (lane == 0) oe[0] = s1[0];
            } else if (n == 2) {
                const int off2[2] = { tok[e] * XQB, tok[e + 1] * XQB };
                float s2[2];
                row_dot_multi<TY, 2, true>(wr, xq, off2, 2, nb, lane, s2);
                if (lane < 2) oe[lane * os] = s2[lane];
            } else {
                int off[GRP_NC];
#pragma unroll
                for (int c = 0; c < GRP_NC; ++c) off[c] = tok[e + min(c, n - 1)] * XQB;
                float sx[GRP_NC];
                row_dot_multi<TY, GRP_NC, false>(wr, xq, off, n, nb, lane, sx);
#pragma unroll
                for (int c = 0; c < GRP_NC; ++c) if (c < n && lane == c) oe[c * os] = sx[c];
            }
        }
    }
}

// expert item (group g, ff block j): 16 warps x (one gate pair + one up pair) -> s.gu[(e - e0) * 64 + row]
template<int TG>
__device__ void gu_item_ex(Sm& s, int g, int j, int warp, int lane) {
    const NativeExpertLayout& L = s.lay.L;
    const uint8_t* blob = (const uint8_t*) s.gptr[g];
    const int rb = (int) L.gu_row, e0 = s.start[g], e1 = s.start[g + 1];
#pragma unroll 1
    for (int half = 0; half < 2; ++half) {
        const uint8_t* rowA = blob + (half ? L.up_off : 0) + (size_t) (32 * j + 2 * warp) * rb;
        pair_s<TG>(rowA, rb, s.tok, e0, e1, s.xq, s.gu + half * 32 + 2 * warp, 64, s.grid, lane);
    }
}

// shared gate row r (IQ4_XS, 2560 -> one value per token): native_iq4_xs_mmvq / native_mmvq_multi's exact layout
// (4 warps: thread tid takes block tid / 8, sub-block 4 (tid % 8); warp 0 adds warps 1..3's sums in order, then the
// butterfly), the fast-math file's ftz arithmetic
template<int TM = TMX>
__device__ __forceinline__ void sh_gate_row(const uint8_t* w, int r, const block_q8_1* xq, int T, float* o, int os, int lane) {
    const int kq = 4 * (lane & 7);
    float v[3][TM];
#pragma unroll
    for (int q = 0; q < 3; ++q) {
        const int kbx = 4 * q + (lane >> 3);
#pragma unroll
        for (int t = 0; t < TM; ++t) v[q][t] = 0.0f;
        if (kbx < 10) {
            const Split<23>::W ww = Split<23>::load(w + (size_t) r * SH_GROW, kbx, kq);
#pragma unroll
            for (int t = 0; t < TM; ++t) {
                if (t < T) {
                    const block_q8_1* x = xq + (size_t) t * XQB + kbx * 8 + kq / 4;
                    int sumi = 0;
#pragma unroll
                    for (int jj = 0; jj < 4; ++jj) {
                        sumi = ggml_cuda_dp4a(ww.v[jj].x, get_int_b4(x->qs, jj), sumi);
                        sumi = ggml_cuda_dp4a(ww.v[jj].y, get_int_b4(x->qs, jj + 4), sumi);
                    }
                    sumi *= ww.ls - 32;
                    const float d = mul_ftz(ww.dw, __low2float(x->ds));
                    v[q][t] = fma_ftz(d, (float) sumi, 0.0f);
                }
            }
        }
    }
#pragma unroll
    for (int t = 0; t < TM; ++t) {
        if (t < T) {
            float a = add_ftz(add_ftz(add_ftz(v[0][t], v[1][t]), v[2][t]), 0.0f);
            a = wsum_ftz(a);
            if (lane == t) o[t * os] = a;
        }
    }
}

// native_swiglu_quantize_q8_1 (fast-math file) for one 32-value block: lane = value
__device__ __forceinline__ void swiglu_q8_native(float gi, float ui, block_q8_1* y, int lane) {
    const float e = ex2_ftz(mul_ftz(-gi, 1.4426950408889634f));
    const float xi = mul_ftz(div_apx_ftz(gi, add_ftz(1.0f, e)), ui);
    const float amax = wmax_ftz(fabsf(xi));
    const float sum = wsum_ftz(xi);
    const float d = q8_1_finite(div_apx_ftz(amax, 127.0f));
    int8_t q = 0;
    if (amax != 0.0f) {
        const float v = roundf(div_apx_ftz(xi, d));
        q = (int8_t) (v > 127.0f ? 127.0f : v < -127.0f ? -127.0f : v);
    }
    y->qs[lane] = q;
    if (lane == 0) y->ds = q8_1_ds(d, sum);
}

// shared down row r (IQ4_NL, 640 -> 2560): native_small_mmvq's layout (thread tid: block tid / 2, sub 2 (tid % 2);
// 40 threads, warp 1's 8 added to warp 0's lanes 0..7)
template<int TM = TMX>
__device__ __forceinline__ void sh_down_row(const uint8_t* w, int r, const block_q8_1* h, int T, float* o, int os, int lane) {
    float v[2][TM];
#pragma unroll
    for (int q = 0; q < 2; ++q) {
        const int kbx = 16 * q + (lane >> 1), kq = 2 * (lane & 1);
#pragma unroll
        for (int t = 0; t < TM; ++t) v[q][t] = 0.0f;
        if (kbx < 20) {
            const Split<20>::W ww = Split<20>::load(w + (size_t) r * SH_DROW, kbx, kq);
#pragma unroll
            for (int t = 0; t < TM; ++t) {
                if (t < T) {
                    const block_q8_1* x = h + t * HB + kbx;
                    const int* q8 = (const int*) x->qs + kq;
                    int sumi = 0;
#pragma unroll
                    for (int l = 0; l < 2; ++l) {
                        sumi = ggml_cuda_dp4a(ww.v[l].x, q8[l + 0], sumi);
                        sumi = ggml_cuda_dp4a(ww.v[l].y, q8[l + 4], sumi);
                    }
                    const float d = mul_ftz(ww.dw, __low2float(x->ds));
                    v[q][t] = fma_ftz(d, (float) sumi, 0.0f);
                }
            }
        }
    }
#pragma unroll
    for (int t = 0; t < TM; ++t) {
        if (t < T) {
            float a = add_ftz(add_ftz(add_ftz(v[0][t], v[1][t]), 0.0f), 0.0f);
            a = wsum_ftz(a);
            if (lane == 0) o[t * os] = a;
        }
    }
}


// ------------------------------------------------------------ the shared expert's other formats
// native_mmvq's exact layout (native_mmvq_multi_kernel, 4 warps x 1 row; bitwise equal to its ncols = 1 kernels):
// one warp per row, virtual thread 32 q + lane takes blocks kbx = tid / TPB, + BPI, ...; warp q's partial is added to
// warp 0's in order, then the butterfly.  Arithmetic of the fast-math file: ftz, products contracted into the
// accumulating adds as nvcc contracts them.
#ifndef SHX_KV
#define SHX_KV 0   // Q4_K/Q5_K closing expression dm.x * sumf_d - dm.y * sumf_m: 0 fma(x, d, -(y m)), 1 fma(-y, m, x d), 2 no fma
#endif
__device__ __forceinline__ int shx_b2(const uint8_t* p, int i) {
    const uint16_t* q = (const uint16_t*) p + 2 * i;
    return (int) q[0] | ((int) q[1] << 16);
}
__device__ __forceinline__ float shx_h(const uint8_t* p) { return __half2float(*(const __half*) p); }
__device__ __forceinline__ float shx_dmc(float dx, float sd, float dy, float sm) {
#if SHX_KV == 0
    return fma_ftz(dx, sd, -mul_ftz(dy, sm));
#elif SHX_KV == 1
    return fma_ftz(-dy, sm, mul_ftz(dx, sd));
#else
    return add_ftz(mul_ftz(dx, sd), -mul_ftz(dy, sm));
#endif
}
template<int TY> struct ShxF;
template<> struct ShxF<12> { static constexpr int BB = 144, TPB = 16, BPI = 8, KBY = 8; };   // Q4_K
template<> struct ShxF<13> { static constexpr int BB = 176, TPB = 16, BPI = 8, KBY = 8; };   // Q5_K
template<> struct ShxF<14> { static constexpr int BB = 210, TPB = 32, BPI = 4, KBY = 8; };   // Q6_K
template<> struct ShxF<8> { static constexpr int BB = 34, TPB = 4, BPI = 32, KBY = 1; };     // Q8_0

// one weight block (kbx) for virtual thread tid, every token: acc[t] as tmp[t] += F::apply(...)
template<int TY, int TM>
__device__ __forceinline__ void shx_blk(const uint8_t* row, int kbx, int tid, const block_q8_1* xq, int xs, int T, float* acc) {
    const uint8_t* blk = row + (size_t) kbx * ShxF<TY>::BB;
    if constexpr (TY == 12 || TY == 13) {
        const int iqs = 2 * (tid % 16);
        const int bo = 2 * ((iqs / 2) / 4), sub = (iqs / 2) % 4;
        int vl0, vl1, vh0 = 0, vh1 = 0;
        if constexpr (TY == 12) {
            const int* ql = (const int*) (blk + 16 + 16 * bo + 4 * sub);
            vl0 = ql[0]; vl1 = ql[4];
        } else {
            const int* ql = (const int*) (blk + 48 + 16 * bo + 4 * sub);
            const int* qh = (const int*) (blk + 16 + 4 * sub);
            vl0 = ql[0]; vl1 = ql[4];
            vh0 = qh[0] >> bo; vh1 = qh[4] >> bo;
        }
        const uint16_t* scales = (const uint16_t*) (blk + 4);
        const int j = bo / 2, jm = j & 1;
        const uint32_t s0 = scales[jm], s2 = scales[jm + 2], s4 = scales[jm + 4];
        const uint32_t hi = uint32_t(-int32_t(j >= 2));
        uint16_t aux[2];
        aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
        aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
        const uint8_t* sc = (const uint8_t*) aux;
        const float dx = shx_h(blk), dy = shx_h(blk + 2);
#pragma unroll
        for (int t = 0; t < TM; ++t) {
            if (t < T) {
                const block_q8_1* bq8 = xq + (size_t) t * xs + kbx * ShxF<TY>::KBY;
                float sd = 0.0f, sm = 0.0f;
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const block_q8_1* bi = bq8 + bo + i;
                    const float d8 = __low2float(bi->ds);
                    const int* q8 = (const int*) bi->qs + sub;
                    const int u0 = q8[0], u1 = q8[4];
                    int dot1, dot2;
                    if constexpr (TY == 12) {
                        const int v0i = (vl0 >> (4 * i)) & 0x0f0f0f0f, v1i = (vl1 >> (4 * i)) & 0x0f0f0f0f;
                        dot1 = ggml_cuda_dp4a(v1i, u1, ggml_cuda_dp4a(v0i, u0, 0));
                        dot2 = ggml_cuda_dp4a(0x01010101, u1, ggml_cuda_dp4a(0x01010101, u0, 0));
                    } else {
                        const int v0i = ((vl0 >> (4 * i)) & 0x0f0f0f0f) | (((vh0 >> i) << 4) & 0x10101010);
                        const int v1i = ((vl1 >> (4 * i)) & 0x0f0f0f0f) | (((vh1 >> i) << 4) & 0x10101010);
                        dot1 = ggml_cuda_dp4a(v0i, u0, ggml_cuda_dp4a(v1i, u1, 0));
                        dot2 = ggml_cuda_dp4a(0x01010101, u0, ggml_cuda_dp4a(0x01010101, u1, 0));
                    }
                    sd = fma_ftz(d8, (float) (dot1 * sc[i]), sd);
                    sm = fma_ftz(d8, (float) (dot2 * sc[2 + i]), sm);
                }
                acc[t] = add_ftz(acc[t], shx_dmc(dx, sd, dy, sm));
            }
        }
    } else if constexpr (TY == 14) {
        const int iqs = tid % 32;
        const int bo = 4 * (iqs / 16) + (iqs % 16) / 8, so = 8 * (iqs / 16) + (iqs % 16) / 4, vsh = 2 * ((iqs % 16) / 8);
        const int vl = shx_b2(blk, iqs);
        const int vh = shx_b2(blk + 128, 8 * (iqs / 16) + iqs % 8) >> vsh;
        const int8_t* scs = (const int8_t*) (blk + 192) + so;
        const float d = shx_h(blk + 208);
#pragma unroll
        for (int t = 0; t < TM; ++t) {
            if (t < T) {
                const block_q8_1* x = xq + (size_t) t * xs + kbx * 8;
                float sf = 0.0f;
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const int u = ((const int*) x[bo + 2 * i].qs)[iqs % 8];
                    const float d8 = __low2float(x[bo + 2 * i].ds);
                    const int sc = scs[4 * i];
                    const int vil = (vl >> (4 * i)) & 0x0f0f0f0f;
                    const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
                    const int vi = __vsubss4(vil | vih, 0x20202020);
                    sf = fma_ftz(d8, (float) (ggml_cuda_dp4a(vi, u, 0) * sc), sf);
                }
                acc[t] = fma_ftz(d, sf, acc[t]);
            }
        }
    } else {   // Q8_0
        const int kqs = 2 * (tid % 4);
        const int v0 = shx_b2(blk + 2, kqs), v1 = shx_b2(blk + 2, kqs + 1);
        const float d0 = shx_h(blk);
#pragma unroll
        for (int t = 0; t < TM; ++t) {
            if (t < T) {
                const block_q8_1* x = xq + (size_t) t * xs + kbx;
                const int* u = (const int*) x->qs;
                int sumi = ggml_cuda_dp4a(v0, u[kqs], 0);
                sumi = ggml_cuda_dp4a(v1, u[kqs + 1], sumi);
                acc[t] = fma_ftz(mul_ftz(d0, __low2float(x->ds)), (float) sumi, acc[t]);
            }
        }
    }
}

// row (nb blocks) against T activation columns (stride xs blocks) -> o[t * os] (lane 0 writes)
template<int TY, int TM>
__device__ __forceinline__ void shx_row(const uint8_t* row, int nb, const block_q8_1* xq, int xs, int T, float* o, int os, int lane) {
    float v[4][TM];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
#pragma unroll
        for (int t = 0; t < TM; ++t) v[q][t] = 0.0f;
        const int tid = 32 * q + lane;
#pragma unroll 1
        for (int kbx = tid / ShxF<TY>::TPB; kbx < nb; kbx += ShxF<TY>::BPI) shx_blk<TY, TM>(row, kbx, tid, xq, xs, T, v[q]);
    }
#pragma unroll
    for (int t = 0; t < TM; ++t) {
        if (t < T) {
            float a = add_ftz(add_ftz(add_ftz(v[0][t], v[1][t]), v[2][t]), v[3][t]);
            a = wsum_ftz(a);
            if (lane == 0) o[t * os] = a;
        }
    }
}

__host__ __device__ __forceinline__ int sh_rowbytes(int ty, int n_in) {
    switch (ty) {
        case 12: return n_in / 256 * 144;
        case 13: return n_in / 256 * 176;
        case 14: return n_in / 256 * 210;
        case 23: return n_in / 256 * 136;
        case 21: return n_in / 256 * 110;
        case 20: return n_in / 32 * 18;
        case 8: return n_in / 32 * 34;
    }
    return 0;
}

// shared gate or up rows r, r + 1 of format ty (2560 -> 1 per token) -> o[t * os], o[t * os + 1]
template<int TM = TMX>
__device__ __forceinline__ void sh_gu_pair(int ty, const uint8_t* w, int r, const block_q8_1* xq, const int32_t* idt, int T,
                                           float* o, int os, const uint32_t* grid3s, int lane) {
    const int rb = sh_rowbytes(ty, N);
    switch (ty) {
        case 23: sh_gate_row<TM>(w, r, xq, T, o, os, lane); sh_gate_row<TM>(w, r + 1, xq, T, o + 1, os, lane); break;
        case 21: pair_s<21>(w + (size_t) r * rb, rb, idt, 0, T, xq, o, os, grid3s, lane); break;
        case 12: shx_row<12, TM>(w + (size_t) r * rb, N / 256, xq, XQB, T, o, os, lane);
                 shx_row<12, TM>(w + (size_t) (r + 1) * rb, N / 256, xq, XQB, T, o + 1, os, lane); break;
        case 13: shx_row<13, TM>(w + (size_t) r * rb, N / 256, xq, XQB, T, o, os, lane);
                 shx_row<13, TM>(w + (size_t) (r + 1) * rb, N / 256, xq, XQB, T, o + 1, os, lane); break;
        case 14: shx_row<14, TM>(w + (size_t) r * rb, N / 256, xq, XQB, T, o, os, lane);
                 shx_row<14, TM>(w + (size_t) (r + 1) * rb, N / 256, xq, XQB, T, o + 1, os, lane); break;
    }
}
// shared down row r of format ty (640 -> 2560) against h (T columns of HB blocks) -> o[t * os]
template<int TM = TMX>
__device__ __forceinline__ void sh_dn_row(int ty, const uint8_t* w, int r, const block_q8_1* h, int T, float* o, int os, int lane) {
    if (ty == 8) shx_row<8, TM>(w + (size_t) r * sh_rowbytes(8, NFF), NFF / 32, h, HB, T, o, os, lane);
    else sh_down_row<TM>(w, r, h, T, o, os, lane);
}

// shared item j: gate rows 32j.., up rows 32j.. in the layer's formats
__device__ void gu_item_sh(Sm& s, int j, int T, int warp, int lane) {
    const LayerM& M = s.lay;
    sh_gu_pair(M.sgt, M.sg, 32 * j + 2 * warp, s.xq, s.idt, T, s.gu + 2 * warp, 64, s.grid3s, lane);
    sh_gu_pair(M.sut, M.su, 32 * j + 2 * warp, s.xq, s.idt, T, s.gu + 32 + 2 * warp, 64, s.grid3s, lane);
}

// the shared gate's dot (bf16_f32_mmvf_multi_kernel<256>, fast-math file): 256 threads, thread t the pairs t + 256 m in
// order, two fmas each; the butterfly per warp, then warp 0's butterfly over the 8 warp sums (24 zeros)
__device__ __forceinline__ float sh_gate_dot(const uint16_t* wi, const float* x, int lane) {
    const uint32_t* w2 = (const uint32_t*) wi;
    const float2* x2 = (const float2*) x;
    float P = 0.0f;
#pragma unroll 1
    for (int v = 0; v < 8; ++v) {
        float acc = 0.0f;
#pragma unroll
        for (int m = 0; m < 5; ++m) {
            const int p = lane + 32 * v + 256 * m;
            const uint32_t ww = __ldg(w2 + p);
            const float2 in = __ldcg(x2 + p);
            acc = fma_ftz(__uint_as_float(ww << 16), in.x, acc);
            acc = fma_ftz(__uint_as_float(ww & 0xffff0000u), in.y, acc);
        }
        acc = wsum_ftz(acc);
        if (lane == v) P = acc;
    }
    return wsum_ftz(P);
}

// ------------------------------------------------------------ the kernel
}  // namespace mkb
}  // namespace basalt::kernels
