// M3b: the MoE half of a decode layer (routed experts resident on CUDA0 + shared expert + combine) as one
// persistent kernel, 170 CTAs x 16 warps (512 threads), bitwise equal to today's kernel sequence
// (dec_moe_glue gate/up + SwiGLU + down, shared_expert_multi, dec_moe_wait, dec_moe_combine).
//
// Input release (as M3a hands over): the window's top-10 ids/weights and q8_1(mixed) are visible.  Every CTA plans the
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
#include "basalt/kernels/dec_moe.hpp"
#include "iq_kernels.cu"
#include "basalt/kernels/shared_expert.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include <vector>
#include <random>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <string>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

namespace basalt::kernels {
namespace mkb {

constexpr int N = 2560, NFF = 640, KT = 10, TMX = 4, CT = 512, NWARP = 16, TMXH = 8;
constexpr int XQB = N / 32, HB = NFF / 32;
constexpr int C_REL = 0, C_END = 1, C_SH = 2, C_SG = 3, C_G0 = 4, NCTR = C_G0 + kDmCap + 4, C_TIERS = C_G0 + kDmCap + 2;
constexpr int SH_GROW = 1360, SH_UROW = 1100, SH_DROW = 360;
constexpr int MAXR = 48, HMAX = 40, NENT = TMX * KT, LW = 15, CWN = 15, MAXIT = 4, STG = 768, NXE = 16;
constexpr int SHS = 4, NSH = 20 * SHS, SHP = 32 / SHS;   // shared gate/up sub-items per ff block, their count, pairs each
__device__ float g_shp[20 * TMX * 64];   // shared sub-items' gate/up rows [block][t][64] until the block's last sub-item
__device__ unsigned g_shc[20];           // sub-items done per block (the last one resets it)
constexpr int MAXI = 6, GUE = MAXIT * TMX;   // gate/up items per CTA; their entries' slots in s.gu (packed)   // rows per CTA, entries per down batch, entries
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
    const block_q8_1* xqn;            // glue layers: the shared expert's q8_1 rows of mixed, native_quantize_q8_1's arithmetic
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
    const int32_t* ids_cta = nullptr; const float* w_cta = nullptr;   // fused: per-CTA top-10 copies [G][T][KT] (else I.ids)
    const void* pf[4] = {}; unsigned pfn[4] = {};   // fused: the MoE prologue's records (iteration, layer, ptab, res1) into L2 at kernel start
    IterM it1{}; LayerM lay1{};
    // engine tier handshake: the ring header + mapped rows + doorbells are published by CTA ring_cta after the plan; CTA poll_cta
    // polls the tiers (mapped words) and publishes cnt[C_TIERS]; every other CTA waits on that device word. ring_m == null: the harness
    DmRing* ring_m = nullptr; float* x_m = nullptr; void* xq_m = nullptr; uint32_t* host_seq_m = nullptr; uint32_t* peer_seq_m = nullptr;
    unsigned ring_want = 0; int ring_cta = 100, poll_cta = 101, layer_id = 0; volatile unsigned* abort_m = nullptr;
    unsigned tag0 = 1;              // tag of iteration 0 (C4: global layer ordinal + 1, unique per layer in a launch)   // fused: this layer's iteration and layer records in the parameters (no dependent loads)
};

// ------------------------------------------------------------ small device helpers
__device__ unsigned* g_err;
#ifdef BASALT_PROFILE
__device__ unsigned long long g_pq24[8 * 1024];
__device__ unsigned long long* g_rec24[1024];
__device__ unsigned long long g_kb25[192 * 32];
#endif
__device__ __forceinline__ unsigned ldacq(const unsigned* p) { unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0,[%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void strel(unsigned* p, unsigned v) { asm volatile("st.release.gpu.global.u32 [%0],%1;" ::"l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void redrel(unsigned* p) { asm volatile("red.release.gpu.global.add.u32 [%0],1;" ::"l"(p) : "memory"); }
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(t)); return t; }
// g_err (host-mapped): a wait past the bound: [0] = 1, [1] the waiter's line, [2] block, [3] value, [4] target
__device__ __forceinline__ unsigned long long gtime0() { unsigned long long t; asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(t)); return t; }
#define spin_ge(c, target) spin_ge_(c, target, __LINE__)
__device__ __forceinline__ void spin_ge_(const unsigned* c, unsigned target, int line) {
    unsigned v; unsigned n = 0; unsigned long long t0 = 0;
    mkb_prg(line);
    while ((v = ldacq(c)) < target) {
        if ((++n & 1023u) == 0) {
            const unsigned long long t = gtime0();
            if (t0 == 0) t0 = t;
            else if (t - t0 > 200000000ull) {   // 0.2 s
                if ((*(volatile unsigned*) &g_err[0]) == 0u && ((*(volatile unsigned*) &g_err[0] = 1u), true)) { g_err[1] = line; g_err[2] = blockIdx.x; g_err[3] = v; g_err[4] = target; g_err[5] = ::g_lay[blockIdx.x]; __threadfence_system(); }
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
__device__ __forceinline__ void mb_wait(uint64_t* b, unsigned ph, int line = __builtin_LINE()) {
    mkb_prg(line);
    unsigned ok = 0; unsigned long long t0 = 0;
    while (true) {
        asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }" : "=r"(ok) : "r"(sa(b)), "r"(ph) : "memory");
        if (ok) break;
        unsigned long long t; asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(t));
        if (t0 == 0) t0 = t;
        else if (t - t0 > 200000000ull) { if ((*(volatile unsigned*) &g_err[0]) == 0u && ((*(volatile unsigned*) &g_err[0] = 1u), true)) { g_err[1] = line; g_err[2] = blockIdx.x; g_err[3] = ph ^ 1u; g_err[4] = ph; g_err[5] = ::g_lay[blockIdx.x]; __threadfence_system(); } asm volatile("trap;"); }
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
// in-flight prefetches: mb_pf), the consumer checks every vector's epoch
__device__ __forceinline__ void st_rlx4(uint4* p, uint4 v) { asm volatile("st.relaxed.gpu.global.v4.u32 [%0],{%1,%2,%3,%4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory"); }
__device__ __forceinline__ uint4 ld_rlx4(const uint4* p) { uint4 v; asm volatile("ld.relaxed.gpu.global.v4.u32 {%0,%1,%2,%3},[%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory"); return v; }
// lanes 0..2 of the warp that wrote blk: its tagged copy at dst
__device__ __forceinline__ void put_tagged(const block_q8_1* blk, uint4* dst, unsigned tag, int lane) {
    if (lane < 3) { const uint32_t* w = (const uint32_t*) blk; st_rlx4(dst + lane, make_uint4(w[3 * lane], w[3 * lane + 1], w[3 * lane + 2], tag)); }
}
// vector q of a tagged block into its 3 words of dst (a block_q8_1 array), spinning on the epoch
__device__ __forceinline__ void get_tagged(const uint4* src, uint32_t* dstw, unsigned tag, int line) {
    uint4 v; unsigned n = 0; unsigned long long t0 = 0;
    mkb_prg(line);
    while ((v = ld_rlx4(src)).w != tag) {
        if ((++n & 255u) == 0) {
            const unsigned long long t = gtime0();
            if (t0 == 0) t0 = t;
            else if (t - t0 > 200000000ull) { if ((*(volatile unsigned*) &g_err[0]) == 0u && ((*(volatile unsigned*) &g_err[0] = 1u), true)) { g_err[1] = line; g_err[2] = blockIdx.x; g_err[3] = v.w; g_err[4] = tag; g_err[5] = ::g_lay[blockIdx.x]; __threadfence_system(); } asm volatile("trap;"); }
        }
    }
    dstw[0] = v.x; dstw[1] = v.y; dstw[2] = v.z;
}

// ------------------------------------------------------------ shared memory
struct Sm {
    alignas(16) block_q8_1 xq[TMX * XQB];
    alignas(16) block_q8_1 xq2[TMX * XQB];    // the routed experts' rows on a non-glue layer; the shared expert's on a glue layer
    alignas(16) uint32_t grid[2048];          // the layer's gate/up grid (gu2::Fmt2<TG>::kGridWords)
    alignas(16) uint32_t grid3s[512];         // IQ3_S (the shared up)
    alignas(16) float gu[GUE * 64];           // 32 gate + 32 up rows per entry of this CTA's items (packed, goff)
    unsigned idone[MAXI];                     // pairs done per item
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
            if (c < n && lane == 2 * c) oA[c * os] = sA[c];   // only the n live entries (the slots after them belong to the next item)
            if (c < n && lane == 2 * c + 1) oA[c * os + 1] = sB[c];
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
// NB > 0: nb == NB at compile time; the block loops unroll so the row's weight loads all issue up front
template<int TY, int TM, int NB = 0>
__device__ __forceinline__ void shx_row(const uint8_t* row, int nb, const block_q8_1* xq, int xs, int T, float* o, int os, int lane) {
    float v[4][TM];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
#pragma unroll
        for (int t = 0; t < TM; ++t) v[q][t] = 0.0f;
        const int tid = 32 * q + lane;
        if constexpr (NB > 0) {
            constexpr int NIT = (NB + ShxF<TY>::BPI - 1) / ShxF<TY>::BPI;
#pragma unroll
            for (int it = 0; it < NIT; ++it) {
                const int kbx = tid / ShxF<TY>::TPB + it * ShxF<TY>::BPI;
                if (kbx < NB) shx_blk<TY, TM>(row, kbx, tid, xq, xs, T, v[q]);
            }
        } else {
#pragma unroll 1
            for (int kbx = tid / ShxF<TY>::TPB; kbx < nb; kbx += ShxF<TY>::BPI) shx_blk<TY, TM>(row, kbx, tid, xq, xs, T, v[q]);
        }
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
        case 12: shx_row<12, TM, N / 256>(w + (size_t) r * rb, N / 256, xq, XQB, T, o, os, lane);
                 shx_row<12, TM, N / 256>(w + (size_t) (r + 1) * rb, N / 256, xq, XQB, T, o + 1, os, lane); break;
        case 13: shx_row<13, TM, N / 256>(w + (size_t) r * rb, N / 256, xq, XQB, T, o, os, lane);
                 shx_row<13, TM, N / 256>(w + (size_t) (r + 1) * rb, N / 256, xq, XQB, T, o + 1, os, lane); break;
        case 14: shx_row<14, TM, N / 256>(w + (size_t) r * rb, N / 256, xq, XQB, T, o, os, lane);
                 shx_row<14, TM, N / 256>(w + (size_t) (r + 1) * rb, N / 256, xq, XQB, T, o + 1, os, lane); break;
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
    sh_gu_pair(M.sgt, M.sg, 32 * j + 2 * warp, M.glue ? s.xq2 : s.xq, s.idt, T, s.gu + 2 * warp, 64, s.grid3s, lane);
    sh_gu_pair(M.sut, M.su, 32 * j + 2 * warp, M.glue ? s.xq2 : s.xq, s.idt, T, s.gu + 32 + 2 * warp, 64, s.grid3s, lane);
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
// FUSED: rdy = the chain's hc2-up counter of this layer, rdy_n its full count (every hc2 q8_1 / mixed row written)
// vrow (FUSED): this CTA's chain logical id, the top-10 row it wrote itself (0: I.ids)
template <bool FUSED>
__device__ __forceinline__ void moe_body(const KP& p, const unsigned* rdy = nullptr, unsigned rdy_n = 0, int vrow = 0,
                                          [[maybe_unused]] unsigned long long* srec = nullptr) {
    extern __shared__ __align__(128) char sm_dyn[];
    Sm& s = *reinterpret_cast<Sm*>(sm_dyn);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, b = blockIdx.x, G = gridDim.x;
    const int T = p.T, n = T * KT;
    if constexpr (!FUSED) mkb_setlay((unsigned) p.it[0].layer + 1u);
    if (tid == 0) { mb_init(&s.xbar); asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
    for (int i = tid; i < 512; i += CT) s.grid3s[i] = iq3s_grid[i];
    if (tid < TMX) s.idt[tid] = tid;
    for (int k = tid; k <= kDmCap; k += CT) { s.hrdy[k] = 0; s.hclaim[k] = 0; }
    unsigned xph = 0;
    syncp();
    for (int i = 0; i < p.iters; ++i) {
        const IterM& I = FUSED ? p.it1 : p.it[i];
        const LayerM& LM = FUSED ? p.lay1 : p.lay[I.layer];
        unsigned* c = p.cnt + (size_t) i * NCTR;
        unsigned long long* st = p.st != nullptr && tid == 0 ? p.st + ((size_t) i * G + b) * NST : nullptr;
#ifdef BASALT_PROFILE
        if constexpr (FUSED) st = srec ? g_kb25 + 32 * (size_t) b : nullptr;
#endif
#define Q(n) do { if (st) st[16 + (n)] = gtime(); } while (0)
#ifdef BASALT_PROFILE
        unsigned long long* const s_pq = g_pq24 + 8 * b;   // per-CTA stamp scratch in global memory (no static smem: the arena is at the limit)
        unsigned long long*& s_rec = g_rec24[b];
        if (tid == 0) { s_rec = st; for (int k = 0; k < 8; ++k) s_pq[k] = 0ull; }
#define PQX(k) atomicMax(&s_pq[k], gtime())
#else
#define PQX(k) ((void) 0)
#endif
        unsigned long long* stl = p.st != nullptr && tid == LW * 32 ? p.st + ((size_t) i * G + b) * NST : nullptr;   // the loader's lane 0
#ifdef BASALT_PROFILE
        if constexpr (FUSED) stl = srec && tid == LW * 32 ? g_kb25 + 32 * (size_t) b : nullptr;
#endif
        // ---- before the release: the layer's constants (format, tables, grid) into smem
        if (tid == 0) s.lay = LM;
#ifdef BASALT_PROFILE
        if (st) { st[12] = gtime(); st[11] = 0; st[10] = 0; }
#endif
        for (int e = tid; e < 512; e += CT) { s.ptab[e] = __ldg(LM.ptab + e); s.res1[e] = __ldg(LM.res1 + e); }
        {
            const int tg = LM.L.gu_type;
            const int nw = tg == 16 ? 512 : tg == 17 ? 1024 : tg == 18 ? 256 : tg == 21 ? 512 : tg == 22 ? 2048 : 0;
            const uint32_t* src = tg == 16 ? (const uint32_t*) iq2xxs_grid : tg == 17 ? (const uint32_t*) iq2xs_grid :
                                  tg == 18 ? (const uint32_t*) iq3xxs_grid : tg == 21 ? (const uint32_t*) iq3s_grid :
                                  (const uint32_t*) iq2s_grid;
            for (int e = tid; e < nw; e += CT) s.grid[e] = src[e];
        }
        if ((p.mode & 2) && tid == 32) {   // the shared expert's bytes are static: into L2 ahead of the release
            const int r0 = (int) ((long long) b * 640 / G), r1 = (int) ((long long) (b + 1) * 640 / G);
            pf_l2(LM.sg + (size_t) r0 * sh_rowbytes(LM.sgt, N), (size_t) (r1 - r0) * sh_rowbytes(LM.sgt, N));
            pf_l2(LM.su + (size_t) r0 * sh_rowbytes(LM.sut, N), (size_t) (r1 - r0) * sh_rowbytes(LM.sut, N));
            const int d0 = (int) ((long long) b * 2560 / G), d1 = (int) ((long long) (b + 1) * 2560 / G);
            pf_l2(LM.sd + (size_t) d0 * sh_rowbytes(LM.sdt, NFF), (size_t) (d1 - d0) * sh_rowbytes(LM.sdt, NFF));
        }
        // ---- the release (FUSED: acquire the chain's hc2-up count, which covers xq and mixed; CTA 0 stamps it)
        if (tid == 0) {
            if constexpr (FUSED) { spin_ge(rdy, rdy_n); if (b == 0) p.tx[i] = gtime(); } else if (b == 0) {
                if (i > 0) {
                    spin_ge(&p.cnt[(size_t) (i - 1) * NCTR + C_END], (unsigned) G);
                    const unsigned long long t0 = gtime();
                    while (gtime() - t0 < (unsigned long long) p.delay_ns) {}
                }
                p.tx[i] = gtime();
                strel(&c[C_REL], 1);
            } else {
                spin_ge(&c[C_REL], 1);
            }
            if (st) st[S_SEEN] = gtime();
            asm volatile("fence.proxy.async;" ::: "memory");
            {
                const unsigned xb = (unsigned) (T * XQB * sizeof(block_q8_1));
                const bool two = !LM.glue || b < NSH;   // the second copy: the routed rows (non-glue) or the shared expert's native rows (glue)
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(sa(&s.xbar)), "r"(two ? 2 * xb : xb) : "memory");
                asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(sa(s.xq)), "l"(I.xq), "r"(xb), "r"(sa(&s.xbar)) : "memory");
                if (two) asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(sa(s.xq2)), "l"(LM.glue ? I.xqn : I.xqr), "r"(xb), "r"(sa(&s.xbar)) : "memory");
            }
        }
        syncp();
        // ---- the plan (dm_glue_gu_kernel's grouping of the CUDA0 entries, dm_glue_ring's tiers)
        {
            const int rb = FUSED ? vrow : b;   // fused: the row this CTA's own chain wrote (no cross-CTA ids hop)
            const int32_t* ids_in = p.ids_cta && rb != 0 ? p.ids_cta + (size_t) rb * T * KT : I.ids;   // per-CTA top-10 copy (row 0 is I.ids)
            const float* w_in = p.ids_cta && rb != 0 ? p.w_cta + (size_t) rb * T * KT : I.w;
            const int32_t e0 = tid < n ? __ldcg(ids_in + tid) : -1;
            const bool valid = e0 >= 0 && e0 < 512;
            const unsigned long long praw = valid ? s.ptab[e0] : 0ull;
            if (tid < n) { s.ids[tid] = e0; s.w[tid] = __ldcg(w_in + tid); }
            if (tid == 0) { s.npeer = 0; s.ncpu = 0; s.acc[0] = s.acc[1] = s.acc[2] = 0; }
            if (tid < MAXI) s.idone[tid] = 0;
            if (tid == 0) { s.ucnt = 0; s.dcnt = 0; s.nxe = 0; s.R1 = -1; }
            syncp();
            if (st) st[S_IDS] = gtime();
            int32_t key = -1;
            if (tid < n) {
                bool ok = valid;
                for (int jx = tid - tid % KT; ok && jx < tid; ++jx) ok = s.ids[jx] != e0;
                const unsigned long long pr = ok ? praw : 0ull;
                s.pp[tid] = pr;
                const int kd = pr != 0ull ? 0 : (ok && s.res1[e0] >= 0 ? 2 : -1);
                s.kind[tid] = kd;
                if (kd == 2) atomicAdd(&s.npeer, 1);
                if (kd < 0) atomicAdd(&s.ncpu, 1);
                if (kd != 0) { const int x = atomicAdd(&s.nxe, 1); if (x < NXE) s.xent[x] = tid; }
                key = pr != 0ull ? e0 : -1;
            }
            if (tid < kDmCap) s.key[tid] = key;
            syncp();
            int first = 0, rank = 0;
            Q(0);
            dm_group_keys(s.key, n, tid, s.first, s.cnt, s.gstart, s.gidx, s.start, &s.ng, &s.ne, first, rank);
            if (tid < n && s.kind[tid] == 0) {
                const int pos = s.gstart[first] + rank;
                s.pos[tid] = pos;
                s.tok[pos] = tid / KT;
                s.dst[pos] = tid;
                if (s.first[tid]) s.gptr[s.gidx[tid]] = s.pp[tid];
            }
            Q(1);
#ifdef BASALT_PROFILE
            if (st) st[15] = gtime();
#endif
        }
        syncp();
        Q(2);
#ifdef BASALT_PROFILE
        if (st) st[14] = gtime();
#endif
        if (p.ring_m != nullptr && b == p.ring_cta) {   // dm_glue_ring: the ring header, the mapped rows of the tiers that have entries, the doorbells
            if (tid < n) { p.ring_m->ids[tid] = s.ids[tid]; p.ring_m->kind[tid] = s.kind[tid]; }
            if (tid == 0) {
                DmRing* r = p.ring_m;
                r->n_tok = T; r->layer = p.layer_id; r->n_vram = n - s.npeer - s.ncpu; r->n_peer = s.npeer; r->n_cpu = s.ncpu;
            }
            if (s.npeer > 0) { const int n16 = T * XQB * (int) sizeof(block_q8_1) / 16; for (int x = tid; x < n16; x += CT) ((uint4*) p.xq_m)[x] = __ldcg((const uint4*) I.xq + x); }
            if (s.ncpu > 0) { const int n4 = T * N / 4; for (int x = tid; x < n4; x += CT) ((float4*) p.x_m)[x] = __ldcg((const float4*) I.mixed + x); }
            __threadfence_system();
            syncp();
            if (tid == 0) {
                __threadfence_system();
                *(volatile uint32_t*) p.peer_seq_m = p.ring_want | (uint32_t) T << 16 | (s.npeer > 0 ? 1u << 28 : 0u);
                *(volatile uint32_t*) p.host_seq_m = p.ring_want;
                __threadfence_system();
                Q(12);   // the doorbells are published
            }
        }
        const NativeExpertLayout& L = s.lay.L;
        const int ng = s.ng;
        // ---- the schedule: items round-robin (shared first), down rows by bytes (closed form over the <= 4 segments of
        // CTAs with equal gate/up bytes)
        const int NI = NSH + 20 * ng;
        auto sched = [&] {
            const long long Bsh = 32LL * sh_rowbytes(LM.sgt, N) + 32LL * sh_rowbytes(LM.sut, N), Bex = 64LL * (long long) L.gu_row;
            const long long Bq = 4LL * ((long long) ng * (long long) L.d_row + sh_rowbytes(LM.sdt, NFF));
            const long long target = (20 * Bsh * p.guw / 100 + 20LL * ng * Bex * p.guw / 100 + 640 * Bq) / G;
            const int qi = NI / G, ri = NI % G;
            auto val = [&](int x) { const int cn = x < ri ? qi + 1 : qi; const long long gub = (cn * Bex + (x < NSH && cn > 0 ? Bsh / SHS - Bex : 0)) * p.guw / 100; return target > gub ? target - gub : 0LL; };
            // down rows: each CTA's share is proportional to its val, capped at MAXR rows (its smem buffers); the
            // capped CTAs' surplus is spread over the rest (water-filling over the <= 3 equal-val segments), in
            // 16.16 fixed-point quads so a capped CTA gets exactly MAXR rows after the prefix rounding
            int cut[4] = {0, min(NSH, G), min(ri, G), G}; if (cut[1] > cut[2]) { const int t = cut[1]; cut[1] = cut[2]; cut[2] = t; }
            long long v[3], c[3]; int nk[3]; bool capd[3] = {false, false, false};
            for (int k = 0; k < 3; ++k) { nk[k] = max(cut[k + 1] - cut[k], 0); v[k] = nk[k] ? max(val(cut[k]), 1LL) : 0LL; c[k] = 0; }
            const long long CAP = (long long) (MAXR / 4) << 16, TOT = (long long) (N / 4) << 16;
            for (int it = 0; it < 3; ++it) {
                long long rem = TOT, den = 0;
                for (int k = 0; k < 3; ++k) { if (capd[k]) rem -= nk[k] * CAP; else den += nk[k] * v[k]; }
                int worst = -1;
                for (int k = 0; k < 3; ++k) {
                    if (capd[k] || !nk[k]) continue;
                    c[k] = den > 0 && rem > 0 ? (rem * v[k] + den - 1) / den : 0;
                    if (c[k] > CAP && (worst < 0 || v[k] > v[worst])) worst = k;
                }
                if (worst < 0) break;
                capd[worst] = true;
            }
            for (int k = 0; k < 3; ++k) c[k] = capd[k] ? CAP : min(c[k], CAP);
            auto W = [&](int x) {   // sum of c over [0, x)
                long long sum = 0;
                for (int k = 0; k < 3; ++k) { const int lo = cut[k], hi = min(cut[k + 1], x); if (hi > lo) sum += (long long) (hi - lo) * c[k]; }
                return sum;
            };
            const int sr0 = 4 * (int) min((W(b) + 32768) >> 16, (long long) (N / 4)), sr1 = 4 * (int) min((W(b + 1) + 32768) >> 16, (long long) (N / 4));
            *(volatile int*) &s.R0 = sr0;
            if (sr1 - sr0 > MAXR || (b == G - 1 && sr1 != N)) { g_err[0] = 1; g_err[1] = __LINE__; g_err[2] = b; g_err[3] = sr1 - sr0; g_err[5] = ::g_lay[blockIdx.x]; __threadfence_system(); asm volatile("trap;"); }
            const int r1 = sr1 > N ? N : sr1;
            __threadfence_block();
            *(volatile int*) &s.R1 = r1;   // publishes R0 (s.R1 >= 0)
        };
        int R0 = 0, R1 = 0, nr = 0;
        const int nmine = b < NI ? (NI - 1 - b) / G + 1 : 0;
        if (nmine > MAXI && tid == 0) { g_err[0] = 1; g_err[1] = __LINE__; g_err[2] = blockIdx.x; g_err[5] = ::g_lay[blockIdx.x]; __threadfence_system(); asm volatile("trap;"); }
        const unsigned tag = p.tag0 + (unsigned) i;
        if (st) st[S_PLAN] = gtime();
        int nq = 0;                                  // down units: (group, 4 rows); group ng is the shared expert
        if (warp == LW) {
#ifdef BASALT_PROFILE
            if constexpr (FUSED) stl = tid == LW * 32 ? s_rec : nullptr;
#endif
            // ======== loader warp: prefetches, the shared gate scalar, then the h rows of each group into smem as they
            // complete (epoch-tagged), each published to the compute warps by an smem flag
            // byte-paced prefetch: an SM's requests land in order only when few are queued, so the gate/up rows go out
            // as units (u = item * 32 + pair) PF_AHEAD units ahead of the compute warps' claims, PF_CHUNK units at a
            // time; the down slices once ai items' worth of units remain
            constexpr int PF_AHEAD = 12, PF_CHUNK = 2;
            const int ai = p.ahead > 0 ? (int) p.ahead : 64;
            auto pf_units = [&](int u0, int u1) {   // the rows of units [u0, u1)
                while (u0 < u1) {
                    const int k = u0 >> 5, p0 = u0 & 31, it = b + k * G;
                    const int pe = min(u1 - (k << 5), p0 < 16 ? 16 : 32);
                    if (it < NSH) {   // sub-item it % SHS: pairs SHP * (it % SHS) .., the first half of the pairs gate rows, the rest up rows
                        const int pa = min(p0, SHP), pb = min(pe, SHP);
                        if (pb > pa) {
                            const int j = it / SHS, q0 = SHP * (it % SHS) + pa, ty = q0 < 16 ? LM.sgt : LM.sut;
                            pf_l2((q0 < 16 ? LM.sg : LM.su) + (size_t) (32 * j + 2 * (q0 & 15)) * sh_rowbytes(ty, N), (size_t) (2 * (pb - pa)) * sh_rowbytes(ty, N));
                        }
                    } else {
                        const int g = (it - NSH) / 20, jj = (it - NSH) % 20;
                        pf_l2((const uint8_t*) s.gptr[g] + (p0 < 16 ? 0 : L.up_off) + (size_t) (32 * jj + 2 * (p0 & 15)) * L.gu_row, (size_t) (2 * (pe - p0)) * L.gu_row);
                    }
                    u0 = (k << 5) + pe;
                }
            };
            const int NU = nmine * 32;
            int uiss = min(PF_AHEAD, NU);
            if (lane == 0 && !(p.mode & 4)) pf_units(0, uiss);
            if (st) st[S_PF] = gtime();
            if (stl) stl[16 + 10] = gtime();   // Q(10): the loader has issued the first gate/up prefetches
            if (lane == 0) sched();
            __syncwarp();
            R0 = *(volatile int*) &s.R0; R1 = *(volatile int*) &s.R1; nr = R1 - R0;
            if (b >= G - T) {   // the shared gate's dot for token G - 1 - b
                const int t = G - 1 - b;
                const float gd = sh_gate_dot(LM.sgi, I.mixed + (size_t) t * N, lane);
                if (lane == 0) asm volatile("st.relaxed.gpu.global.v2.u32 [%0],{%1,%2};" ::"l"(p.gs + t), "r"(__float_as_uint(gd)), "r"(tag) : "memory");
            }
            if (!(p.mode & 4)) {
                while (uiss < NU) {
                    const int u1 = min(uiss + PF_CHUNK, NU);
                    if (lane == 0) { const unsigned long long tw = gtime0();
                        while ((int) *(volatile unsigned*) &s.ucnt + PF_AHEAD < u1 && !(p.mode & 1))
                            if (gtime0() - tw > 200000000ull) { g_err[0] = 1; g_err[1] = __LINE__; g_err[2] = b; g_err[5] = ::g_lay[blockIdx.x]; __threadfence_system(); asm volatile("trap;"); } }
                    __syncwarp();
                    if (lane == 0) pf_units(uiss, u1);
                    uiss = u1;
                }
                for (int k = max(ai, nmine); k < nmine + ai; ++k) {   // k == nmine + j: the down slices' turn
                    const unsigned need = (unsigned) (k - ai) * 32u;
                    unsigned long long t0 = 0;
                    if (lane == 0)
                    while (*(volatile unsigned*) &s.ucnt < need && !(p.mode & 1)) {
                        if (t0 == 0) t0 = gtime0(); else if (gtime0() - t0 > 200000000ull) { g_err[0] = 1; g_err[1] = __LINE__; g_err[2] = b; g_err[5] = ::g_lay[blockIdx.x]; __threadfence_system(); asm volatile("trap;"); }
                    }
                    __syncwarp();
                    if (nr > 0)   // down slices: per-lane line prefetches (groups in order)
                        for (int g = 0; g <= ng; ++g) {
                            const uint8_t* q = g < ng ? (const uint8_t*) s.gptr[g] + L.down_off + (size_t) R0 * L.d_row : LM.sd + (size_t) R0 * sh_rowbytes(LM.sdt, NFF);
                            const size_t bytes = (size_t) nr * (g < ng ? L.d_row : sh_rowbytes(LM.sdt, NFF));
                            const uintptr_t a0 = (uintptr_t) q & ~(uintptr_t) 127, a1 = ((uintptr_t) q + bytes + 127) & ~(uintptr_t) 127;
                            for (uintptr_t x = a0 + (uintptr_t) lane * 128; x < a1; x += 32 * 128) asm volatile("prefetch.global.L2 [%0];" ::"l"(x));
                        }
                    break;
                }
                if (st) st[S_I1] = gtime();
            }
            xph ^= 1;   // (the loader does not wait on the xq copy; keep its phase in step)
            // the combine's other inputs into smem: the shared gate per token, the other tiers' rows of this CTA's rows
            if (!(p.mode & 1)) {
                if (lane < T) {
                    uint2 gv; unsigned long long tg0 = gtime0(); mkb_prg(__LINE__);
                    do { asm volatile("ld.relaxed.gpu.global.v2.u32 {%0,%1},[%2];" : "=r"(gv.x), "=r"(gv.y) : "l"(p.gs + lane) : "memory");
                         if (gtime0() - tg0 > 200000000ull) { g_err[0] = 1; g_err[1] = __LINE__; g_err[2] = blockIdx.x; g_err[5] = ::g_lay[blockIdx.x]; __threadfence_system(); asm volatile("trap;"); } } while (gv.y != tag);
                    const float g = __uint_as_float(gv.x);
                    s.gsv[lane] = __fdividef(1.0f, 1.0f + __expf(-g));
                    PQX(4);
                }
                if (lane == 0) {
                    if (p.ring_m == nullptr) {
                        if (s.npeer > 0) spin_ge(p.peer_done, (unsigned) (i + 1));
                        if (s.ncpu > 0) spin_ge(p.cpu_done, (unsigned) (i + 1));
                    } else if (s.npeer + s.ncpu > 0) {
                        if (b == p.poll_cta) {   // the one mapped poller; the others wait on the device word it publishes
                            bool pd = s.npeer == 0, cd = s.ncpu == 0;
                            unsigned long long t0 = gtime0();
                            while (!pd || !cd) {
                                if (!pd && *(volatile const unsigned*) p.peer_done >= p.ring_want) pd = true;
                                if (!cd && *(volatile const unsigned*) p.cpu_done >= p.ring_want) cd = true;
                                if (pd && cd) break;
                                __nanosleep(32);
                                if (gtime0() - t0 > 2000000000ull) {
                                    if (p.abort_m != nullptr) *p.abort_m = 1u;
                                    if ((*(volatile unsigned*) &g_err[0]) == 0u) { g_err[0] = 1; g_err[1] = __LINE__; g_err[2] = b; g_err[3] = pd; g_err[4] = cd; g_err[5] = ::g_lay[blockIdx.x]; __threadfence_system(); }
                                    asm volatile("trap;");
                                }
                            }
                            __threadfence_system();
                            strel(&c[C_TIERS], 1);
                        } else {
                            unsigned long long t0 = gtime0();
                            while (ldacq(&c[C_TIERS]) < 1u)
                                if (gtime0() - t0 > 2500000000ull) { if ((*(volatile unsigned*) &g_err[0]) == 0u) { g_err[0] = 1; g_err[1] = __LINE__; g_err[2] = b; g_err[5] = ::g_lay[blockIdx.x]; __threadfence_system(); } asm volatile("trap;"); }
                        }
                    }
                }
                __syncwarp();
                if (stl) stl[16 + 11] = gtime();   // Q(11): the tiers have landed (or none were needed)
                const int nx = min(s.nxe, NXE);
                for (int q = lane; q < nx * nr; q += 32) {
                    const int x = q / nr, r = q % nr, en = s.xent[x];
                    const float* const src = (s.kind[en] == 2 ? p.peer_rows : p.cpu_rows) + (size_t) en * N + R0 + r;
                    s.prow[x * MAXR + r] = __ldcv(src);
                    // the peer's rows go back to "not yet written" for whichever combine reads them next
                }
            }
            if (stl) stl[16 + 9] = gtime();   // Q(9): the loader's work is done
        } else {
            mb_wait(&s.xbar, xph); xph ^= 1;
            if (st) st[S_XQ] = gtime();
            const int cw = warp;   // 0 .. CWN - 1
            // ======== gate/up pairs: unit u = (my item u / 32, pair u % 32: < 16 gate rows, else up rows), strided over
            // the compute warps across item boundaries; the warp that finishes an item's last pair applies SwiGLU and
            // stores its tagged q8_1 blocks
            bool q3d = false;
            if (!(p.mode & 1))
            for (;;) {
                unsigned u = 0;
                if (lane == 0) u = atomicAdd(&s.ucnt, 1u);
                u = __shfl_sync(0xffffffffu, u, 0);
                if ((int) u >= nmine * 32) break;
#ifdef BASALT_PROFILE
                if (st && !q3d) { st[10] = gtime(); st[13] = (unsigned long long) b | (unsigned long long) s.ng << 16 | (unsigned long long) nmine << 24 | (unsigned long long) nr << 32 | (unsigned long long) (s.npeer + s.ncpu) << 40; }
#endif
                const int k = u >> 5, pp = u & 31, it = b + k * G;
                int go = 0;   // item ks first entry slot in s.gu: shared items hold T entries, an expert item its groups count
                for (int k2 = 0; k2 < k; ++k2) { const int i2 = b + k2 * G; go += i2 < NSH ? T : s.start[(i2 - NSH) / 20 + 1] - s.start[(i2 - NSH) / 20]; }
                { const int ne = it < NSH ? T : s.start[(it - NSH) / 20 + 1] - s.start[(it - NSH) / 20];
                  if (go + ne > GUE) { if (lane == 0) { g_err[0] = 1; g_err[1] = __LINE__; g_err[2] = b; g_err[3] = go; g_err[5] = ::g_lay[blockIdx.x]; __threadfence_system(); } asm volatile("trap;"); } }
                float* gk = s.gu + go * 64;
                if (it < NSH) {
                    if (pp >= SHP) continue;   // a sub-item has SHP pairs; its other unit numbers are empty
                    const int j = it / SHS, q = SHP * (it % SHS) + pp;
                    if (q < 16) {
                        sh_gu_pair(LM.sgt, LM.sg, 32 * j + 2 * q, LM.glue ? s.xq2 : s.xq, s.idt, T, gk + 2 * q, 64, s.grid3s, lane);
                    } else {
                        sh_gu_pair(LM.sut, LM.su, 32 * j + 2 * (q - 16), LM.glue ? s.xq2 : s.xq, s.idt, T, gk + 32 + 2 * (q - 16), 64, s.grid3s, lane);
                    }
                } else {
                    const int g = (it - NSH) / 20, jj = (it - NSH) % 20;
                    const uint8_t* rowA = (const uint8_t*) s.gptr[g] + (pp < 16 ? 0 : L.up_off) + (size_t) (32 * jj + 2 * (pp & 15)) * L.gu_row;
                    float* o = gk + (pp < 16 ? 0 : 32) + 2 * (pp & 15);
                    const int e0 = s.start[g], e1 = s.start[g + 1], rb = (int) L.gu_row;
                    const block_q8_1* xr = s.lay.glue ? s.xq : s.xq2;
                    switch (L.gu_type) {
                        case 16: pair_s<16>(rowA, rb, s.tok, e0, e1, xr, o, 64, s.grid, lane); break;
                        case 17: pair_s<17>(rowA, rb, s.tok, e0, e1, xr, o, 64, s.grid, lane); break;
                        case 18: pair_s<18>(rowA, rb, s.tok, e0, e1, xr, o, 64, s.grid, lane); break;
                        case 21: pair_s<21>(rowA, rb, s.tok, e0, e1, xr, o, 64, s.grid, lane); break;
                        case 22: pair_s<22>(rowA, rb, s.tok, e0, e1, xr, o, 64, s.grid, lane); break;
                        case 23: pair_rdm<23>(rowA, rb, s.tok, e0, e1, xr, o, 64, lane); break;
                        default: break;
                    }
                }
                __syncwarp(); __threadfence_block();
                if (st && q3d && st[11] == 0) st[11] = gtime();
                if (!q3d) { if (lane == 0) PQX(6); if (st) Q(3); q3d = true; }   // Q(3): this warp finished its first gate/up pair
                unsigned old = 0;
                if (lane == 0) old = atomicAdd(&s.idone[k], 1u);
                old = __shfl_sync(0xffffffffu, old, 0);
                if (old == (it < NSH ? (unsigned) SHP - 1u : 31u)) {   // the item's pairs are in smem
                    __threadfence_block();
                    if (it < NSH) {
                        // this sub-item's 2 * SHP rows per token to the scratch; the block's last sub-item takes all 64 back
                        const int j = it / SHS, r0 = 2 * SHP * (it % SHS);
                        const int c0 = r0;   // rows r0 .. r0 + 2 SHP - 1 of the block's 64 (gate 0-31, up 32-63)
                        for (int t = 0; t < T; ++t) if (lane < 2 * SHP) __stcg(g_shp + (j * TMX + t) * 64 + c0 + lane, gk[t * 64 + c0 + lane]);
                        __syncwarp(); __threadfence();
                        unsigned last = 0;
                        if (lane == 0) last = atomicAdd(&g_shc[j], 1u) == (unsigned) SHS - 1u;
                        last = __shfl_sync(0xffffffffu, last, 0);
                        if (!last) continue;
                        if (lane == 0) g_shc[j] = 0;   // the next layer's sub-items start only after this layer's combine
                        __threadfence();
                        for (int t = 0; t < T; ++t) { gk[t * 64 + lane] = __ldcg(g_shp + (j * TMX + t) * 64 + lane); gk[t * 64 + 32 + lane] = __ldcg(g_shp + (j * TMX + t) * 64 + 32 + lane); }
                        __syncwarp();
                        for (int t = 0; t < T; ++t) {
                            swiglu_q8_native(gk[t * 64 + lane], gk[t * 64 + 32 + lane], &s.hst[cw], lane);
                            __syncwarp();
                            put_tagged(&s.hst[cw], p.hqs + (size_t) (t * HB + j) * 3, tag, lane);
                            __syncwarp();
                        }
                        if (lane == 0) PQX(3);
                    } else {
                        const int g = (it - NSH) / 20, jj = (it - NSH) % 20, e0 = s.start[g], ne = s.start[g + 1] - e0;
                        for (int e = 0; e < ne; ++e) {
                            const float gv = gk[e * 64 + lane], uv = gk[e * 64 + 32 + lane];
                            const float hv = __fmul_rn(gv / (1.0f + __expf(-gv)), uv);
                            q8_1_store(hv, &s.hst[cw], lane);
                            __syncwarp();
                            put_tagged(&s.hst[cw], p.hq + (size_t) ((e0 + e) * HB + jj) * 3, tag, lane);
                            __syncwarp();
                        }
                    }
                    if (st && k == 0) st[S_I1] = gtime();
                }
            }
            if (st) st[S_GU] = gtime();
            if (lane == 0) PQX(2);
            {   // the loader's schedule (this CTA's down rows)
                if (lane == 0) { const unsigned long long tw = gtime0();
                    while (*(volatile int*) &s.R1 < 0)
                        if (gtime0() - tw > 200000000ull) { g_err[0] = 1; g_err[1] = __LINE__; g_err[2] = b; g_err[5] = ::g_lay[blockIdx.x]; __threadfence_system(); asm volatile("trap;"); } }
                __syncwarp(); __threadfence_block();
                R0 = *(volatile int*) &s.R0; R1 = *(volatile int*) &s.R1; nr = R1 - R0; nq = (nr + 3) / 4;
            }
            // ======== down units (group-major so the early groups go first), each waiting on its group's smem flag
            if (!(p.mode & 1) && nr > 0) {
                const int TD = L.d_type;
                for (bool first = true;; first = false) {
                    unsigned u = 0;
                    if (lane == 0) u = atomicAdd(&s.dcnt, 1u);
                    mkb_prg(__LINE__);
                    u = __shfl_sync(0xffffffffu, u, 0);
                    if ((int) u >= (ng + 1) * nq) break;
                    const int g = u / nq, rq = 4 * (u % nq);
                    {   // the group's h into smem: the first warp to need it copies it (epoch-checked), the others wait
                        const int hv = g < ng ? g : kDmCap;
                        unsigned hr = 0;
                        if (lane == 0) hr = *(volatile unsigned*) &s.hrdy[hv];
                        hr = __shfl_sync(0xffffffffu, hr, 0);
                        if (hr != tag) {
                            unsigned prev = 0;
                            if (lane == 0) prev = atomicExch(&s.hclaim[hv], tag);
                            mkb_prg(__LINE__);
                            prev = __shfl_sync(0xffffffffu, prev, 0);
                            if (prev != tag) {
                                if (lane == 0 && hv == kDmCap) PQX(5);
                                const uint4* src = g < ng ? p.hq + (size_t) s.start[g] * HB * 3 : p.hqs;
                                uint32_t* dst = g < ng ? (uint32_t*) (s.h + (size_t) s.start[g] * HB) : (uint32_t*) s.hsh;
                                const int nvec = (g < ng ? s.start[g + 1] - s.start[g] : T) * HB * 3;
                                for (int q0 = 0; q0 < nvec; q0 += 32 * 8) {
                                    uint4 v[8];
#pragma unroll
                                    for (int k = 0; k < 8; ++k) { const int q = q0 + k * 32 + lane; if (q < nvec) v[k] = ld_rlx4(src + q); }
#pragma unroll
                                    for (int k = 0; k < 8; ++k) {
                                        const int q = q0 + k * 32 + lane;
                                        if (q >= nvec) continue;
                                        if (v[k].w != tag) { uint32_t w3[3]; get_tagged(src + q, w3, tag, __LINE__); v[k].x = w3[0]; v[k].y = w3[1]; v[k].z = w3[2]; }
                                        uint32_t* d = dst + (q / 3) * 9 + (q % 3) * 3;
                                        d[0] = v[k].x; d[1] = v[k].y; d[2] = v[k].z;
                                    }
                                }
                                __syncwarp(); __threadfence_block();
                                if (lane == 0) *(volatile unsigned*) &s.hrdy[hv] = tag;
                                if (lane == 0 && hv == kDmCap) PQX(1);
                            }
                        }
                        const volatile unsigned* f = &s.hrdy[hv];
                        unsigned long long t0 = 0;
                        while (*f != tag) { if (t0 == 0) t0 = gtime0(); else if (gtime0() - t0 > 200000000ull) { if ((*(volatile unsigned*) &g_err[0]) == 0u && ((*(volatile unsigned*) &g_err[0] = 1u), true)) { g_err[1] = __LINE__; g_err[2] = b; } asm volatile("trap;"); } }
                        __syncwarp();
                        __threadfence_block();
                    }
                    if (first) Q(5);
                    if (g == ng) {
                        for (int r = rq; r < min(rq + 4, nr); ++r) sh_dn_row(LM.sdt, LM.sd, R0 + r, s.hsh, T, s.sh + r, MAXR, lane);
                        continue;
                    }
                    const int e0 = s.start[g], e1 = s.start[g + 1];
                    const int tpr = TD == 20 ? 8 : 16, rpp = 32 / tpr;
                    for (int r0 = rq; r0 < min(rq + 4, nr); r0 += rpp) {
                        const int r = r0 + lane / tpr, t = lane % tpr;
                        const bool live = r < min(rq + 4, nr);
                        const uint8_t* wr = (const uint8_t*) s.gptr[g] + L.down_off + (size_t) (R0 + (live ? r : r0)) * L.d_row;
                        for (int e = e0; e < e1; e += GRP_NC) {
                            const int m = min(GRP_NC, e1 - e), bo = e * HB;
                            float* o = s.out + (size_t) e * MAXR + r;
                            if (TD == 20) {
                                if (m == 1) { const int off1[1] = {bo}; float s1[1]; row_dot_40_sub8<20, 1, true>(wr, s.h, off1, 1, t, s1); if (live && t == 0) o[0] = s1[0]; }
                                else if (m == 2) { const int off2[2] = {bo, bo + HB}; float s2[2]; row_dot_40_sub8<20, 2, true>(wr, s.h, off2, 2, t, s2); if (live && t < 2) o[t * MAXR] = s2[t]; }
                                else { int off[GRP_NC]; for (int cc = 0; cc < GRP_NC; ++cc) off[cc] = bo + min(cc, m - 1) * HB; float sx[GRP_NC]; row_dot_40_sub8<20, GRP_NC>(wr, s.h, off, m, t, sx);
#pragma unroll
                                       for (int cc = 0; cc < GRP_NC; ++cc) if (live && cc < m && t == cc) o[cc * MAXR] = sx[cc]; }
                            } else {
                                if (m == 1) { const int off1[1] = {bo}; float s1[1]; row_dot_20_sub16<42, 1, true>(wr, s.h, off1, 1, t, s1); if (live && t == 0) o[0] = s1[0]; }
                                else if (m == 2) { const int off2[2] = {bo, bo + HB}; float s2[2]; row_dot_20_sub16<42, 2, true>(wr, s.h, off2, 2, t, s2); if (live && t < 2) o[t * MAXR] = s2[t]; }
                                else { int off[GRP_NC]; for (int cc = 0; cc < GRP_NC; ++cc) off[cc] = bo + min(cc, m - 1) * HB; float sx[GRP_NC]; row_dot_20_sub16<42, GRP_NC>(wr, s.h, off, m, t, sx);
#pragma unroll
                                       for (int cc = 0; cc < GRP_NC; ++cc) if (live && cc < m && t == cc) o[cc * MAXR] = sx[cc]; }
                            }
                        }
                    }
                }
            }
            if (st) st[S_DN] = gtime();
            if (lane == 0) PQX(0);
        }
        syncp();
#ifdef BASALT_PROFILE
        if (st) { const unsigned long long tb0 = gtime(); const int qm[7] = {4, 6, 7, 8, 0, 1, 2}; for (int k = 0; k < 7; ++k) st[16 + qm[k]] = *(volatile unsigned long long*) &s_pq[k]; st[16 + 13] = tb0; st[16 + 14] = gtime(); }
#endif
        // ---- the shared rows scaled by the gate (sigmoid_scale_rows_kernel)
        if (!(p.mode & 1) && tid < T * nr) {
            const int t = tid / nr, r = tid % nr;
            s.sh[t * MAXR + r] *= s.gsv[t];
        }
        if (st) st[S_SHD] = gtime();
        // ---- combine (dm_combine4's sums, fast-math file) for this CTA's rows
        syncp();
        if (!(p.mode & 1) && tid < T * nr) {
            const int t = tid / nr, r = tid % nr, row = R0 + r;
            float sum = 0.0f;
#pragma unroll
            for (int jx = 0; jx < KT; ++jx) {
                const int en = t * KT + jx, kd = s.kind[en];
                float v;
                if (kd == 0) v = add_ftz(0.0f, s.out[(size_t) s.pos[en] * MAXR + r]);
                else {
                    int x = 0;
                    while (x < min(s.nxe, NXE) && s.xent[x] != en) ++x;
                    if (x < NXE && x < s.nxe) v = s.prow[x * MAXR + r];
                    else {
                        const float* const src = (kd == 2 ? p.peer_rows : p.cpu_rows) + (size_t) en * N + row;
                        v = __ldcv(src);
                    }
                }
                sum = jx == 0 ? mul_ftz(v, s.w[en]) : fma_ftz(v, s.w[en], sum);
            }
            sum = add_ftz(sum, s.sh[t * MAXR + r]);
            I.bo[(size_t) t * N + row] = sum;
        }
        if ((p.mode & 8) && p.dbg_parts != nullptr && tid < s.ne * nr) {
            const int e = tid / nr, r = tid % nr;
            p.dbg_parts[(size_t) s.dst[e] * N + R0 + r] = s.out[(size_t) e * MAXR + r];
        }
        syncp();
        if (st) st[S_END] = gtime();
        if (tid == 0) { atomicMax(p.tend + i, gtime()); redrel(&c[C_END]); }
#ifdef BASALT_PROFILE
        if constexpr (FUSED) if (st) for (int k = 0; k < 32; ++k) srec[::basalt::kernels::stamp::kB + k] = st[k];
#endif
    }
}

// ------------------------------------------------------------ today's sequence (reference)
#ifndef FZ_DEV_ONLY   // the reference sequence is for the standalone harness
__global__ void __launch_bounds__(CT, 1) k_moe(const __grid_constant__ KP p) { moe_body<false>(p); }
__global__ void k_rel(unsigned long long* t0, int i, long long spin_ns) {
    pdl_trigger();
    const unsigned long long a = gtime();
    while (gtime() - a < (unsigned long long) spin_ns) {}
    t0[i] = gtime();
}
__global__ void k_end(unsigned long long* t1, int i) {
    pdl_wait();
    t1[i] = gtime();
}
#endif

}  // namespace mkb
}  // namespace basalt::kernels
using namespace basalt::kernels;
using namespace basalt::kernels::mkb;

// ------------------------------------------------------------ host
static uint16_t sane_half(std::mt19937& r) { const uint32_t v = r(); return (uint16_t) (((v >> 31) << 15) | ((5u + (v >> 10) % 5u) << 10) | (v & 0x3ffu)); }
template <typename V> V* upload(const std::vector<V>& h) { V* d; CK(cudaMalloc(&d, h.size() * sizeof(V) + 256)); CK(cudaMemcpy(d, h.data(), h.size() * sizeof(V), cudaMemcpyHostToDevice)); return d; }
static int blk_bytes(int t) { switch (t) { case 8: return 34; case 12: return 144; case 13: return 176; case 14: return 210; case 16: return 66; case 17: return 74; case 18: return 98; case 20: return 18; case 21: return 110; case 22: return 82; case 23: return 136; case 42: return 18; } return 0; }
// random bytes, each block's leading fp16 scale sane
static void rand_rows(uint8_t* h, size_t bytes, int type, std::mt19937& r) {
    for (size_t k = 0; k < bytes; ++k) h[k] = (uint8_t) (r() & 0xff);
    const int sb = blk_bytes(type);
    const int ho = type == 14 ? 208 : 0;   // the block's fp16 scale(s); Q4_K/Q5_K carry two
    for (size_t k = 0; k + sb <= bytes; k += sb) {
        const uint16_t v = sane_half(r); memcpy(h + k + ho, &v, 2);
        if (type == 12 || type == 13) { const uint16_t m = sane_half(r); memcpy(h + k + 2, &m, 2); }
    }
}
static double med(std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0 : v[v.size() / 2]; }

#ifndef FX_NO_MAIN
// the FUSED instance (k_layer's experts half) with the chain's hc2-up count already full: one record per launch
__global__ void __launch_bounds__(CT, 1) k_moef(const __grid_constant__ KP p, const unsigned* rdy) {
    mkb_setlay(p.tag0);
    moe_body<true>(p, rdy, 1u, 0);
}
// activations on the q8_1 rounding edge: per 32-block one value carries the block's amax, the others sit within a few
// ulp of (k + 0.5) * amax / 127, where IEEE division and div.approx.ftz round x / d to different integers
static void edge_rows(std::vector<float>& m, std::mt19937& r) {
    std::normal_distribution<float> nd(0.f, 1.f);
    for (size_t b0 = 0; b0 < m.size(); b0 += 32) {
        const float amax = 0.25f + 2.0f * std::fabs(nd(r));
        const float d = amax / 127.0f;
        const int big = (int) (r() % 32);
        for (int l = 0; l < 32; ++l) {
            float x;
            if (l == big) x = r() & 1 ? amax : -amax;
            else {
                const int k = (int) (r() % 253) - 126;   // k + 0.5 in [-125.5, 126.5] stays below amax / d
                x = ((float) k + 0.5f) * d;
                const int u = (int) (r() % 7) - 3;
                for (int j = 0; j < u; ++j) x = std::nextafter(x, INFINITY);
                for (int j = 0; j > u; --j) x = std::nextafter(x, -INFINITY);
                if (std::fabs(x) > amax) x = 0.0f;
            }
            m[b0 + l] = x;
        }
    }
}
// bitwise differences of the last run (bo floats); synth_sets: run on synthetic records instead of a trace file
long long mkb4_bad = 0;
int mkb4_moe_main(int argc, char** argv, const int* synth_sets = nullptr, int n_synth = 0) {
    const int T = argc > 1 ? atoi(argv[1]) : 3;
    const int iters = argc > 2 ? atoi(argv[2]) : 96;
    const char* pack = getenv("MB_PACK") ? getenv("MB_PACK") : "IQ3_XXS";
    const long long gap = getenv("MB_GAP") ? atoll(getenv("MB_GAP")) : 10000;
    const int mode = getenv("MB_MODE") ? atoi(getenv("MB_MODE")) : 0;
    const int hot = getenv("MB_HOT") ? atoi(getenv("MB_HOT")) : 0;   // 1: one record repeated (L2-hot)
    const long long ahead = (getenv("MB_AHEAD") ? atoll(getenv("MB_AHEAD")) : 2);
    const long long guw = getenv("MB_GUW") ? atoll(getenv("MB_GUW")) : 100;
    const long long dnpf = getenv("MB_DNPF") ? atoll(getenv("MB_DNPF")) : 0;
    const int reps = getenv("MB_REPS") ? atoi(getenv("MB_REPS")) : 3;
    if (T > TMX) { printf("T %d > TMX %d\n", T, TMX); return 1; }
    int nrec = 0, fmt[48][2] = {};
    std::vector<int8_t> tier(48 * 512, 0);
    struct Rec { int layer; std::vector<int> ids; std::vector<float> w; };
    std::vector<Rec> recs;
    int sh_syn[48][3] = {};
    if (synth_sets != nullptr) {   // synthetic records: n_synth layers {layer, gate/up, down, shared gate, up, down}, round-robin
        std::mt19937 r(11);
        std::uniform_real_distribution<float> uw(0.01f, 1.f);
        for (int k = 0; k < n_synth; ++k) {
            const int* q = synth_sets + 6 * k;
            fmt[q[0]][0] = q[1]; fmt[q[0]][1] = q[2];
            sh_syn[q[0]][0] = q[3]; sh_syn[q[0]][1] = q[4]; sh_syn[q[0]][2] = q[5];
        }
        for (int l = 0; l < 48; ++l) for (int e = 0; e < 512; ++e) tier[l * 512 + e] = e % 9 == 4 ? 2 : e % 13 == 7 ? 1 : 0;   // CUDA0, peer, CPU
        nrec = iters;
        recs.resize(nrec);
        for (int i = 0; i < nrec; ++i) {
            Rec& rc = recs[i];
            rc.layer = synth_sets[6 * (i % n_synth)];
            rc.ids.resize(T * KT); rc.w.resize(T * KT);
            for (int t = 0; t < T; ++t)
                for (int k = 0; k < KT; ++k) {
                    int e;
                    for (;;) { e = (int) (r() % 512); bool dup = false; for (int j = 0; j < k; ++j) dup |= rc.ids[t * KT + j] == e; if (!dup) break; }
                    rc.ids[t * KT + k] = e; rc.w[t * KT + k] = uw(r);
                }
        }
    } else {
    char fn[256]; snprintf(fn, sizeof fn, "/home/jay/poc/mkb4/trace_%s_T%d.bin", pack, T);
    FILE* f = fopen(fn, "rb"); if (!f) { printf("no %s\n", fn); return 1; }
    int tt; fread(&nrec, 4, 1, f); fread(&tt, 4, 1, f);
    fread(fmt, 4, 96, f);
    fread(tier.data(), 1, tier.size(), f);
    recs.resize(nrec);
    for (auto& r : recs) { r.ids.resize(T * KT); r.w.resize(T * KT); fread(&r.layer, 4, 1, f); fread(r.ids.data(), 4, T * KT, f); fread(r.w.data(), 4, T * KT, f); }
    fclose(f);
    }
    // the experts half sets each peer row it reads back to the peer's "not yet written" tag (the next layer's peer
    // window rewrites it); a launch over several records would read those tags, so MB_NOPEER moves the peer tier's
    // entries to the CPU tier (rows read the same way, never tagged)
    if (getenv("MB_NOPEER")) for (auto& t : tier) if (t == 2) t = 1;
    if (hot) for (int i = 1; i < nrec; ++i) recs[i] = recs[0];
    const int NI = std::min(iters, nrec);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    int nsm; CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
    if (nsm * MAXR < N) { fprintf(stderr, "grid %d x MAXR %d cannot hold %d down rows\n", nsm, MAXR, N); return 1; }
    cudaStream_t cs, ss; CK(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking)); CK(cudaStreamCreateWithFlags(&ss, cudaStreamNonBlocking));
    // ---- layers: the CUDA0 experts the records use, the shared expert, the tables
    std::vector<int> lay_of(48, -1); std::vector<int> layers;
    for (int i = 0; i < NI; ++i) if (lay_of[recs[i].layer] < 0) { lay_of[recs[i].layer] = (int) layers.size(); layers.push_back(recs[i].layer); }
    std::vector<LayerM> hl(layers.size());
    size_t exp_bytes = 0;
    for (size_t li = 0; li < layers.size(); ++li) {
        const int l = layers[li];
        LayerM& M = hl[li];
        M.L = native_expert_layout(fmt[l][0], fmt[l][1], N, NFF);
        std::vector<char> used(512, 0);
        for (int i = 0; i < NI; ++i) if (recs[i].layer == l) for (int e : recs[i].ids) if (e >= 0 && e < 512 && tier[l * 512 + e] == 0) used[e] = 1;
        std::vector<unsigned long long> ptab(512, 0); std::vector<int32_t> res1(512, -1);
        int nu = 0; for (int e = 0; e < 512; ++e) nu += used[e];
        const size_t bb = (M.L.bytes + 255) & ~(size_t) 255;
        uint8_t* base; CK(cudaMalloc(&base, bb * std::max(nu, 1)));
        std::vector<uint8_t> hb(M.L.bytes);
        int k = 0;
        for (int e = 0; e < 512; ++e) {
            if (tier[l * 512 + e] == 2) res1[e] = e;
            if (!used[e]) continue;
            rand_rows(hb.data(), M.L.down_off, M.L.gu_type, rng);
            rand_rows(hb.data() + M.L.down_off, M.L.bytes - M.L.down_off, M.L.d_type, rng);
            CK(cudaMemcpy(base + (size_t) k * bb, hb.data(), M.L.bytes, cudaMemcpyHostToDevice));
            ptab[e] = (unsigned long long) (base + (size_t) k * bb); ++k;
        }
        exp_bytes += (size_t) k * M.L.bytes;
        M.ptab = upload(ptab); M.res1 = upload(res1);
        M.sgt = 23; M.sut = 21; M.sdt = 20;
        { char tf[256]; snprintf(tf, sizeof tf, "/home/jay/poc/mkb4/shtypes_%s.txt", pack);
          if (FILE* q = fopen(tf, "r")) { for (int ll = 0; ll <= l; ++ll) if (fscanf(q, "%d %d %d", &M.sgt, &M.sut, &M.sdt) != 3) break; fclose(q); } }
        if (synth_sets != nullptr) { M.sgt = sh_syn[l][0]; M.sut = sh_syn[l][1]; M.sdt = sh_syn[l][2]; }
        std::vector<uint8_t> sg((size_t) NFF * sh_rowbytes(M.sgt, N)), su((size_t) NFF * sh_rowbytes(M.sut, N)), sd((size_t) N * sh_rowbytes(M.sdt, NFF));
        rand_rows(sg.data(), sg.size(), M.sgt, rng); rand_rows(su.data(), su.size(), M.sut, rng); rand_rows(sd.data(), sd.size(), M.sdt, rng);
        M.sg = upload(sg); M.su = upload(su); M.sd = upload(sd);
        std::vector<uint16_t> gi(N); for (auto& x : gi) { const float v = 0.02f * nd(rng); uint32_t u; memcpy(&u, &v, 4); x = (uint16_t) (u >> 16); }
        M.sgi = upload(gi);
        M.glue = dec_moe_glue_ok(M.L) ? 1 : 0;
    }
    LayerM* dlay = upload(hl);
    printf("T %d, %d iterations over %zu layers, %.2f GB of CUDA0 experts, %d SMs, gap %lld ns, mode %d%s\n", T, NI,
           layers.size(), exp_bytes / 1e9, nsm, gap, mode, hot ? " HOT" : "");
    // ---- per-iteration inputs: 8 activation sets; ids/weights from the records
    constexpr int NX = 8;
    std::vector<float*> mixed(NX); std::vector<block_q8_1*> xq(NX), xqr(NX);
    for (int x = 0; x < NX; ++x) {
        std::vector<float> m((size_t) T * N); for (auto& v : m) v = nd(rng);
        if (getenv("MB_EDGE")) edge_rows(m, rng);
        mixed[x] = upload(m);
        CK(cudaMalloc(&xq[x], (size_t) T * XQB * sizeof(block_q8_1)));
        native_quantize_q8_1(mixed[x], xq[x], N, T, cs);
        CK(cudaMalloc(&xqr[x], (size_t) T * XQB * sizeof(block_q8_1)));
        quantize_q8_1_kernel<<<(T * N + 255) / 256, 256, 0, cs>>>(mixed[x], xqr[x], (long long) T * N);
    }
    CK(cudaDeviceSynchronize());
    std::vector<IterM> hi(NI); std::vector<float*> bo(NI), bo_ref(NI);
    for (int i = 0; i < NI; ++i) {
        IterM& I = hi[i];
        I.layer = lay_of[recs[i].layer];
        I.ids = upload(recs[i].ids); I.w = upload(recs[i].w);
        I.xq = xq[i % NX]; I.mixed = mixed[i % NX];
        if (getenv("MB_IEEE") && hl[I.layer].glue) I.xq = xqr[i % NX];   // the engine (glue layers): the routed experts read the hc read's IEEE q8_1
        I.xqn = xq[i % NX];
        I.xqr = hl[I.layer].glue ? I.xq : xqr[i % NX];
        CK(cudaMalloc(&bo[i], (size_t) T * N * 4)); CK(cudaMalloc(&bo_ref[i], (size_t) T * N * 4));
        I.bo = bo[i];
    }
    IterM* dit = upload(hi);
    // peer / CPU rows (landed), the done words
    float* peer_rows = upload([&] { std::vector<float> v((size_t) kDmCap * N); for (auto& x : v) x = 0.1f * nd(rng); return v; }());
    float* cpu_rows = upload([&] { std::vector<float> v((size_t) kDmCap * N); for (auto& x : v) x = 0.1f * nd(rng); return v; }());
    unsigned* words; CK(cudaMalloc(&words, 64)); { std::vector<unsigned> w(16, 0x7ffffff0u); CK(cudaMemcpy(words, w.data(), 64, cudaMemcpyHostToDevice)); }

    // ================= reference: today's sequence in a graph (glue gu+ring / swiglu / down, shared side branch,
    // dm_wait, combine), one k_rel (PDL trigger, spin) before and one k_end (PDL dependent of the combine) after
    cudaEvent_t evf, evj; CK(cudaEventCreateWithFlags(&evf, cudaEventDisableTiming)); CK(cudaEventCreateWithFlags(&evj, cudaEventDisableTiming));
    DmPlan* plan; CK(cudaMalloc(&plan, sizeof(DmPlan))); CK(cudaMemset(plan, 0, sizeof(DmPlan)));
    void* gscr; CK(cudaMalloc(&gscr, native_expert_scratch_bytes(kDmCap, NFF)));
    float* parts; CK(cudaMalloc(&parts, (size_t) kDmCap * N * 4));
    DmRing* ringd; CK(cudaMalloc(&ringd, sizeof(DmRing)));
    unsigned* rctr; CK(cudaMalloc(&rctr, 16)); CK(cudaMemset(rctr, 0, 16));
    float* x_m; CK(cudaMalloc(&x_m, (size_t) TMXH * N * 4));
    void* xq_m; CK(cudaMalloc(&xq_m, (size_t) TMXH * XQB * 36));
    uint32_t* seqw; CK(cudaMalloc(&seqw, 64)); CK(cudaMemset(seqw, 0, 64));
    std::vector<int32_t> res0h(512);
    std::vector<int32_t*> res0(layers.size());
    for (size_t li = 0; li < layers.size(); ++li) {
        for (int e = 0; e < 512; ++e) res0h[e] = tier[layers[li] * 512 + e] == 0 ? e : -1;
        res0[li] = upload(res0h);
    }
    uint8_t* sh_xq; CK(cudaMalloc(&sh_xq, (size_t) TMXH * XQB * 36));
    uint16_t* sh_bf; CK(cudaMalloc(&sh_bf, (size_t) TMXH * N * 2));
    float *sh_gate, *sh_up, *sh_g, *sh_out;
    CK(cudaMalloc(&sh_gate, (size_t) TMXH * NFF * 4)); CK(cudaMalloc(&sh_up, (size_t) TMXH * NFF * 4)); CK(cudaMalloc(&sh_g, 64)); CK(cudaMalloc(&sh_out, (size_t) TMXH * N * 4));
    float* parts_ref_dbg = nullptr;
    const bool dbg = getenv("MB_DBG") != nullptr;
    std::vector<float> ref_parts_h;
    if (dbg) CK(cudaMalloc(&parts_ref_dbg, (size_t) kDmCap * N * 4));
    shared_expert_set_native_bf16(true);
    unsigned long long *t0d, *t1d; CK(cudaMalloc(&t0d, NI * 8)); CK(cudaMalloc(&t1d, NI * 8));
    const long long pre_ns = getenv("MB_PRE") ? atoll(getenv("MB_PRE")) : 3000;
    auto launch_pdl_k = [&](auto kern, cudaStream_t s, auto... args) {
        cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = 1;
        cudaLaunchConfig_t cfg{}; cfg.gridDim = dim3(1); cfg.blockDim = dim3(32); cfg.stream = s; cfg.attrs = at; cfg.numAttrs = 1;
        CK(cudaLaunchKernelEx(&cfg, kern, args...));
    };
    auto record_ref = [&](int i, bool empty) {
        const IterM& I = hi[i];
        const LayerM& M = hl[I.layer];
        k_rel<<<1, 32, 0, cs>>>(t0d, i, pre_ns);
        if (!empty) {
            cudaEventRecord(evf, cs); cudaStreamWaitEvent(ss, evf, 0);
            DmRouteArgs ra;
            ra.ids = I.ids; ra.x = I.mixed; ra.n_tok = T; ra.k = KT; ra.n_embd = N; ra.n_expert = 512; ra.layer = recs[i].layer;
            ra.res0 = res0[I.layer]; ra.res1 = M.res1; ra.plan = plan; ra.xq = (void*) I.xq;
            ra.ring_m = ringd; ra.x_m = x_m; ra.xq_m = xq_m; ra.host_seq_m = seqw; ra.peer_seq_m = seqw + 1;
            ra.ring = (uint32_t) (i + 1); ra.mode = 2; ra.kinds = true;
            if (M.glue) {
                DmGlueArgs ga;
                ga.ids = I.ids; ga.ptab = M.ptab; ga.n_tok = T; ga.k = KT; ga.n_expert = 512; ga.plan = plan; ga.xq = I.xq;
                ga.ring_on = true; ga.ring = ra;
                dec_moe_glue(M.L, ga, gscr, parts, T * KT, cs);
            } else {   // the route kernel (plan, ring, the routed rows' IEEE q8_1) then native_expert_grouped
                DmRouteArgs r0 = ra;
                r0.mode = 0; r0.kinds = false; r0.xq = (void*) I.xqr; r0.cache_base = nullptr; r0.ctr = rctr;
                r0.slot_off = M.ptab; r0.cache_base = (const uint8_t*) 0; r0.blob = 0;   // slot = the expert id, its offset = its address
                dec_moe_route(r0, cs);
                native_expert_grouped(M.L, plan->ptr, plan->start, plan->counts, plan->dst, plan->tok, T * KT, T * KT, I.xqr, gscr, parts, cs);
            }
            NativeSharedWeights nsw;
            nsw.gate_type = M.sgt; nsw.gate_data = M.sg; nsw.up_type = M.sut; nsw.up_data = M.su; nsw.down_type = M.sdt; nsw.down_data = M.sd;
            nsw.q8_1 = sh_xq;
            shared_expert_multi(T, I.mixed, sh_bf, nsw, M.sgi, sh_gate, sh_up, sh_g, sh_out, N, NFF, ss);
            cudaEventRecord(evj, ss); cudaStreamWaitEvent(cs, evj, 0);
            dec_moe_wait(plan, words, words + 1, (uint32_t) (i + 1), cs, nullptr, recs[i].layer, true, false);
            dec_moe_combine(parts, peer_rows, cpu_rows, plan, I.w, sh_out, bo_ref[i], N, KT, T, nullptr, nullptr,
                            (uint32_t) (i + 1), cs, true, nullptr);
        }
        launch_pdl_k(k_end, cs, t1d, i);
    };
    std::vector<std::vector<double>> ref_per;
    auto run_graph = [&](bool empty, int i_lo, int i_hi) {
        cudaGraph_t g; cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(cs, cudaStreamCaptureModeGlobal));
        for (int i = i_lo; i < i_hi; ++i) record_ref(i, empty);
        CK(cudaStreamEndCapture(cs, &g)); CK(cudaGraphInstantiate(&ge, g, 0));
        CK(cudaGraphLaunch(ge, cs)); CK(cudaStreamSynchronize(cs));
        CK(cudaGraphLaunch(ge, cs)); CK(cudaStreamSynchronize(cs));
        std::vector<unsigned long long> a(NI), z(NI);
        CK(cudaMemcpy(a.data(), t0d, NI * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(z.data(), t1d, NI * 8, cudaMemcpyDeviceToHost));
        std::vector<double> d; for (int i = i_lo + 2; i < i_hi; ++i) d.push_back((z[i] - a[i]) / 1e3);
        if (!empty) { ref_per.assign(layers.size(), {}); for (int i = i_lo + 2; i < i_hi; ++i) ref_per[hi[i].layer].push_back((z[i] - a[i]) / 1e3); }
        CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g));
        return med(d);
    };
    if (dbg) {   // the first iteration's parts (CUDA0 rows) for the per-row compare
        record_ref(0, false); CK(cudaStreamSynchronize(cs));
        ref_parts_h.resize((size_t) kDmCap * N); CK(cudaMemcpy(ref_parts_h.data(), parts, ref_parts_h.size() * 4, cudaMemcpyDeviceToHost));
    }
    // ================= the persistent kernel
    const size_t smem = sizeof(Sm);
    CK(cudaFuncSetAttribute(k_moe, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
    CK(cudaFuncSetAttribute(k_moef, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
    const bool fused = getenv("MB_FUSED") != nullptr;
    unsigned* rdy1; CK(cudaMalloc(&rdy1, 4)); { const unsigned one = 1; CK(cudaMemcpy(rdy1, &one, 4, cudaMemcpyHostToDevice)); }
    if (fused) printf("MB_FUSED: moe_body<true> (k_layer's experts half), one launch per record\n");
    int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k_moe, CT, smem));
    if (occ < 1) { printf("k_moe does not fit (smem %zu)\n", smem); return 1; }
    printf("k_moe smem %zu B\n", smem);
    unsigned* cnt; CK(cudaMalloc(&cnt, (size_t) NI * NCTR * 4));
    unsigned* h_err; CK(cudaHostAlloc(&h_err, 64, cudaHostAllocMapped)); memset(h_err, 0, 64);
    { unsigned* d_err; CK(cudaHostGetDevicePointer(&d_err, h_err, 0)); CK(cudaMemcpyToSymbol(g_err, &d_err, sizeof d_err)); }
    uint4 *hq, *hqs; CK(cudaMalloc(&hq, (size_t) kDmCap * HB * 48)); CK(cudaMalloc(&hqs, (size_t) TMXH * HB * 48));
    CK(cudaMemset(hq, 0, (size_t) kDmCap * HB * 48)); CK(cudaMemset(hqs, 0, (size_t) TMXH * HB * 48));
    uint2* gs; CK(cudaMalloc(&gs, (size_t) TMXH * 8)); CK(cudaMemset(gs, 0, TMXH * 8));
    unsigned long long *tx, *tend, *stp; CK(cudaMalloc(&tx, NI * 8)); CK(cudaMalloc(&tend, NI * 8));
    CK(cudaMalloc(&stp, (size_t) NI * nsm * NST * 8)); CK(cudaMemset(stp, 0, (size_t) NI * nsm * NST * 8));
    float* dbgp = nullptr; if (dbg) { CK(cudaMalloc(&dbgp, (size_t) kDmCap * N * 4)); CK(cudaMemset(dbgp, 0, (size_t) kDmCap * N * 4)); }
    std::vector<std::vector<double>> last_per;
    auto run_mk = [&](int md, bool stamps) {
        CK(cudaMemsetAsync(cnt, 0, (size_t) NI * NCTR * 4, cs)); CK(cudaMemsetAsync(tend, 0, NI * 8, cs));
        CK(cudaMemsetAsync(hq, 0, (size_t) kDmCap * HB * 48, cs)); CK(cudaMemsetAsync(hqs, 0, (size_t) TMXH * HB * 48, cs));
        CK(cudaMemsetAsync(gs, 0, TMXH * 8, cs));
        KP p{dlay, dit, dbg ? 1 : NI, T, gap, md, ahead, dnpf, guw, peer_rows, cpu_rows, words, words + 1, cnt, hq, hqs, gs, tx, tend,
             stamps ? stp : nullptr, dbgp};
        if (fused && !dbg) {
            for (int i = 0; i < NI; ++i) {
                KP q = p;
                q.iters = 1; q.cnt = cnt + (size_t) i * NCTR; q.tx = tx + i; q.tend = tend + i;
                q.st = stamps ? stp + (size_t) i * nsm * NST : nullptr;
                q.it1 = hi[i]; q.lay1 = hl[hi[i].layer]; q.tag0 = (unsigned) i + 1u;
                const unsigned* rp = rdy1;
                void* fa[] = {&q, &rp};
                CK(cudaLaunchCooperativeKernel((void*) k_moef, dim3(nsm), dim3(CT), fa, smem, cs));
            }
        } else {
            void* args[] = {&p};
            CK(cudaLaunchCooperativeKernel((void*) k_moe, dim3(nsm), dim3(CT), args, smem, cs));
        }
        const cudaError_t se = cudaStreamSynchronize(cs);
        if (se != cudaSuccess) {
            const volatile unsigned* e = h_err;
            printf("k_moe failed: %s; wait at line %u, block %u, value %u < %u\n", cudaGetErrorString(se), e[1], e[2], e[3], e[4]);
            exit(1);
        }
        std::vector<unsigned long long> a(NI), z(NI);
        CK(cudaMemcpy(a.data(), tx, NI * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(z.data(), tend, NI * 8, cudaMemcpyDeviceToHost));
        std::vector<double> d; for (int i = 2; i < (dbg ? 1 : NI); ++i) d.push_back((z[i] - a[i]) / 1e3);
        last_per.assign(layers.size(), {});
        for (int i = 2; i < (dbg ? 1 : NI); ++i) last_per[hi[i].layer].push_back((z[i] - a[i]) / 1e3);
        return med(d);
    };
    run_mk(mode, false);
    if (dbg) {
        std::vector<float> mp((size_t) kDmCap * N); CK(cudaMemcpy(mp.data(), dbgp, mp.size() * 4, cudaMemcpyDeviceToHost));
        int bad = 0, tot = 0;
        std::vector<int> ids = recs[0].ids;
        for (int e = 0; e < T * KT; ++e) {
            if (tier[recs[0].layer * 512 + ids[e]] != 0) continue;
            int be = 0; for (int r = 0; r < N; ++r) { ++tot; if (memcmp(&mp[(size_t) e * N + r], &ref_parts_h[(size_t) e * N + r], 4)) { ++be; if (bad + be < 4) printf("  entry %d row %d: %g vs %g\n", e, r, mp[(size_t) e * N + r], ref_parts_h[(size_t) e * N + r]); } }
            bad += be;
        }
        printf("debug parts: %d of %d rows differ\n", bad, tot);
    }
    // ---- bitwise: the reference graph over every iteration once, then compare bo
    {
        cudaGraph_t g; cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(cs, cudaStreamCaptureModeGlobal));
        for (int i = 0; i < NI; ++i) record_ref(i, false);
        CK(cudaStreamEndCapture(cs, &g)); CK(cudaGraphInstantiate(&ge, g, 0));
        CK(cudaGraphLaunch(ge, cs)); CK(cudaStreamSynchronize(cs));
        CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g));
        long long bad = 0, tot = 0; int bad_it = 0;
        std::vector<float> a((size_t) T * N), r((size_t) T * N);
        for (int i = 0; i < (dbg ? 1 : NI); ++i) {
            CK(cudaMemcpy(a.data(), bo[i], a.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(r.data(), bo_ref[i], r.size() * 4, cudaMemcpyDeviceToHost));
            long long b0 = 0;
            for (size_t k = 0; k < a.size(); ++k) if (memcmp(&a[k], &r[k], 4)) { if (bad + b0 < 3) printf("  it %d tok %zu row %zu: %.9g vs %.9g\n", i, k / N, k % N, a[k], r[k]); ++b0; }
            bad += b0; tot += (long long) a.size(); bad_it += b0 > 0;
        }
        mkb4_bad = bad;
        printf("BITWISE bo: %lld of %lld floats differ (%d of %d iterations)%s\n", bad, tot, bad_it, dbg ? 1 : NI, (mode & 1) ? " [mode 1: no compute]" : "");
    }
    if (dbg) return 0;
    // ---- timing: interleaved, reps rounds; ref per-iteration release -> end, minus the empty-graph offset
    std::vector<double> rmk, rref, remp;
    for (int r = 0; r < reps; ++r) {
        rmk.push_back(run_mk(mode, false));
        if (getenv("MB_NOREF")) { rref.push_back(0); remp.push_back(0); continue; }
        rref.push_back(run_graph(false, 0, NI));
        remp.push_back(run_graph(true, 0, NI));
    }
    const double off = med(remp);
    printf("today (graph+PDL) release->combine done: %.2f us (raw %.2f, empty rel->end %.2f) | persistent: %.2f us   [rounds:",
           med(rref) - off, med(rref), off, med(rmk));
    for (int r = 0; r < reps; ++r) printf(" %.2f/%.2f", rref[r] - off, rmk[r]);
    printf("]\n");
    printf("per layer (gate/up x down): persistent med us:");
    for (size_t li = 0; li < layers.size(); ++li) printf("  L%d (%d,%d) %.2f%s", layers[li], hl[li].L.gu_type, hl[li].L.d_type, med(last_per[li]),
        ref_per.empty() ? "" : (" [today " + std::to_string(med(ref_per[li]) - off).substr(0, 5) + "]").c_str());
    printf("\n");
    if (getenv("MB_STAMPS")) {
        run_mk(mode, true);
        std::vector<unsigned long long> st((size_t) NI * nsm * NST), a(NI);
        CK(cudaMemcpy(st.data(), stp, st.size() * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(a.data(), tx, NI * 8, cudaMemcpyDeviceToHost));
        const char* nm[] = {"seen", "plan", "xq", "gu", "pf", "down", "shdown", "item1", "end", "ids", "", "", "", "", "ready"};
        for (int s : {0, 9, 1, 4, 2, 7, 3, 14, 5, 6, 8}) {
            std::vector<double> v, mx;
            for (int i = 2; i < NI; ++i) { double m = 0; for (int b = 0; b < nsm; ++b) { const double d = (st[((size_t) i * nsm + b) * NST + s] - a[i]) / 1e3; v.push_back(d); m = std::max(m, d); } mx.push_back(m); }
            std::sort(v.begin(), v.end());
            printf("  %-7s med %6.2f  p10 %6.2f  p90 %6.2f  max-CTA med %6.2f\n", nm[s], v[v.size() / 2], v[v.size() / 10], v[v.size() * 9 / 10], med(mx));
        }
        for (int q = 16; q < 32; ++q) {
            std::vector<double> v;
            for (int i = 2; i < NI; ++i) for (int b = 0; b < nsm; ++b) { const unsigned long long x = st[((size_t) i * nsm + b) * NST + q]; if (x) v.push_back((x - a[i]) / 1e3); }
            if (v.empty()) continue;
            std::sort(v.begin(), v.end());
            printf("  Q(%d)    med %6.2f  p90 %6.2f\n", q - 16, v[v.size() / 2], v[v.size() * 9 / 10]);
        }
        for (int s : {10, 11, 12, 13}) {
            std::vector<double> v;
            for (int i = 2; i < NI; ++i) for (int b = 0; b < nsm; ++b) v.push_back(st[((size_t) i * nsm + b) * NST + s] / (s == 13 ? 1.0 : 1e3));
            std::sort(v.begin(), v.end());
            printf("  down %-9s med %6.2f p90 %6.2f\n", s == 10 ? "wait us" : s == 11 ? "copy us" : s == 12 ? "comp us" : "batches", v[v.size() / 2], v[v.size() * 9 / 10]);
        }
    }
    return 0;
}

#endif  // FX_NO_MAIN
