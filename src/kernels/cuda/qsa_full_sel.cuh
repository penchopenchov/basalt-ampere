// mkQ2 phase 2 device code: indexer scoring phase + top-k phase (see qsel.cu for the harness).
#pragma once
#include "basalt/kernels/qsa_select.hpp"
#include <cuda_runtime.h>
#include <cstdint>
namespace basalt::kernels { namespace q2sel {
constexpr int IDX_DIM = 128, IDX_HEADS = 4, R = 4, TK_T = 512, TK_W = TK_T / 32;
constexpr int TK_HDR_INTS = 16768 / 4, TK_KCAP = 20480;   // header ints; keys cached in smem up to this many blocks (~82K cells)
constexpr size_t TK_SMEM_B = 16768 + (size_t) TK_KCAP * 4;
struct SelArgs { const float *pooled, *dead, *q_idx; const int32_t* steps; int n; int64_t max_blocks; float* scores; int32_t* ids; int64_t cap; };
__device__ __forceinline__ uint32_t order_key(float s) {
    const float v = s + 0.0f;
    if (!(v == v)) return 0u;
    const uint32_t b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
// the four heads' dot products summed across the warp (xor butterfly pairing, transposed), relu-summed in head order
__device__ __forceinline__ float heads_score(float a0, float a1, float a2, float a3, int lane) {
    const bool hi = (lane & 16) != 0, b = (lane & 8) != 0;
    const float r0 = __shfl_xor_sync(0xffffffffu, hi ? a0 : a2, 16);
    const float r1 = __shfl_xor_sync(0xffffffffu, hi ? a1 : a3, 16);
    const float e0 = (hi ? a2 : a0) + r0, e1 = (hi ? a3 : a1) + r1;
    const float r2 = __shfl_xor_sync(0xffffffffu, b ? e0 : e1, 8);
    float f = (b ? e1 : e0) + r2;
    f += __shfl_xor_sync(0xffffffffu, f, 4);
    f += __shfl_xor_sync(0xffffffffu, f, 2);
    f += __shfl_xor_sync(0xffffffffu, f, 1);
    const float r = f > 0.0f ? f : 0.0f;
    const float g1 = __shfl_sync(0xffffffffu, r, 8), g2 = __shfl_sync(0xffffffffu, r, 16), g3 = __shfl_sync(0xffffffffu, r, 24);
    return ((r + g1) + g2) + g3;
}
// Scores phase. Entry: pooled/dead (written by the kv/indexer append phase) and q_idx (q post phase) visible, steps constant.
// Work: global warp gw (16 per CTA) takes blocks gw, gw + 16*grid, ...; per block one key load, scores for all n tokens.
// Exit: scores[t * max_blocks + b] for b <= n_bid(t) (plain stores, one writer each); nothing written when every
// token's selection is the identity (as the engine's skip_identity). No smem, no barriers.
__device__ __forceinline__ void qsa_scores_phase(const SelArgs& A) {
    const int lane = threadIdx.x & 31;
    bool all = true;
    int64_t top = 0;
    int32_t nbid[8], nkv[8];
    for (int q = 0; q < A.n; ++q) {
        all = all && A.steps[q * kStepCount + kStepNKv] <= A.steps[q * kStepCount + kStepWidth];
        nbid[q] = A.steps[q * kStepCount + kStepNBid]; nkv[q] = A.steps[q * kStepCount + kStepNKv];
        top = nbid[q] > top ? nbid[q] : top;
    }
    if (all) return;
    const int64_t W = (int64_t) gridDim.x * 16, lim = top < A.max_blocks - 1 ? top : A.max_blocks - 1;
    const float4 kd = *reinterpret_cast<const float4*>(A.dead + lane * 4);
    for (int64_t b = (int64_t) blockIdx.x * 16 + (threadIdx.x >> 5); b <= lim; b += W) {
        const float4 cur = *reinterpret_cast<const float4*>(A.pooled + b * IDX_DIM + lane * 4);
#pragma unroll 1
        for (int qi = 0; qi < A.n; ++qi) {
            if (b > nbid[qi]) continue;
            const float4 k4 = (b == nbid[qi]) ? kd : cur;
            float dd[IDX_HEADS];
#pragma unroll
            for (int h = 0; h < IDX_HEADS; ++h) {
                const float4 q4 = *reinterpret_cast<const float4*>(A.q_idx + (qi * IDX_HEADS + h) * IDX_DIM + lane * 4);
                dd[h] = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
            }
            float score = heads_score(dd[0], dd[1], dd[2], dd[3], lane);
            if (lane == 0) {
                if (b == nbid[qi] && nkv[qi] % R != 0) score += 1e9f;
                A.scores[qi * A.max_blocks + b] = score;
            }
        }
    }
}
// Top-k phase: CTA t < n selects for token t with all 512 threads (the engine's block_topk_wide_kernel at 512 threads: radix
// threshold on order_key, cells weighted, ties to the lowest cell, ascending output; the ids are a pure function of the
// multiset of (key, weight)). Entry: this token's scores complete (grid hop after the scores phase). Exit: ids[t * cap + j],
// j < min(n_kv, width). smem arena: 16,640 B header (hist 16*256*4 + (2*16 + 3 + 8)*4 B) + up to 20,480 order keys (81,920 B) = 98,560 B; contexts past 20,480 blocks read the scores from global each pass. Barriers: __syncthreads (CTA-wide, 512) + named barrier 1 (256 threads).
__device__ __forceinline__ void qsa_topk_phase(const SelArgs& A, int* smem_i) {
    if ((int) blockIdx.x >= A.n) return;
    int (*hist)[256] = reinterpret_cast<int (*)[256]>(smem_i);
    int* s_gt = smem_i + TK_W * 256; int* s_eq = s_gt + TK_W; int* s_digit = s_eq + TK_W; int* s_above = s_digit + 1; int* s_above2 = s_above + 1; int* s_ws = s_above2 + 1; int* s_ao = s_ws + 16; int* s_dab = s_ao + 32;
    const int64_t qi = blockIdx.x;
    const int32_t* st = A.steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = A.ids + qi * A.cap;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    if (n_kv <= width) {
        for (int64_t j = t; j < n_kv; j += TK_T) out[j] = (int32_t) j;
        return;
    }
    const float* sc = A.scores + qi * A.max_blocks;
    const int64_t nb = n_bid + 1;
    const int64_t per = (nb + TK_T - 1) / TK_T;
    uint32_t* skeys = reinterpret_cast<uint32_t*>(smem_i + TK_HDR_INTS);
    const bool cached = nb <= TK_KCAP;
    auto weight = [&](int64_t b) -> int { return b < n_bid ? R : (int) (n_kv - n_bid * R); };
    int* hist2 = smem_i;   // two 2048-bin histograms, used alternately
    for (int i = t; i < 4096; i += TK_T) hist2[i] = 0;
    // keys (cached in smem), and the AND / OR of all candidate keys: the radix passes start at the highest bit where they differ
    uint32_t kand = 0xffffffffu, kor = 0u;
    for (int64_t b = t; b < nb; b += TK_T) {
        const uint32_t kk = order_key(sc[b]);
        if (cached) skeys[b] = kk;
        if (weight(b) != 0) { kand &= kk; kor |= kk; }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { kand &= __shfl_xor_sync(0xffffffffu, kand, o); kor |= __shfl_xor_sync(0xffffffffu, kor, o); }
    if (lane == 0) { s_ao[warp * 2] = (int) kand; s_ao[warp * 2 + 1] = (int) kor; }
    __syncthreads();
    kand = 0xffffffffu; kor = 0u;
#pragma unroll
    for (int w2 = 0; w2 < TK_W; ++w2) { kand &= (uint32_t) s_ao[w2 * 2]; kor |= (uint32_t) s_ao[w2 * 2 + 1]; }
    auto keyf = [&](int64_t b) -> uint32_t { return cached ? skeys[b] : order_key(sc[b]); };
    // Threshold: the key of the width-th largest cell (cells weighted), found by radix passes of up to 11 bits over the bits
    // below the common prefix. Same threshold as an 8-bit pass from bit 31, so the same selection.
    const uint32_t diff = kand ^ kor;
    int cur = diff ? 31 - __clz(diff) : -1;
    uint32_t prefix = diff ? (kand & ~((2u << cur) - 1u)) : kand;
    int above = 0, buf = 0;
    while (cur >= 0) {
        const int lo = cur >= 10 ? cur - 10 : 0, wb = cur - lo + 1, nbins = 1 << wb, hi_sh = cur + 1;
        int* H = hist2 + buf * 2048;
        const uint32_t hp = (uint32_t) ((uint64_t) prefix >> hi_sh);
        for (int64_t b = t; b < nb; b += TK_T) {
            const int w = weight(b);
            if (w == 0) continue;
            const uint32_t kk = keyf(b);
            if ((uint32_t) ((uint64_t) kk >> hi_sh) == hp) atomicAdd(&H[(kk >> lo) & (nbins - 1)], w);
        }
        if (t == 0) { s_dab[buf] = 0; s_dab[2 + buf] = above; }
        __syncthreads();
        // zero the other histogram for the next pass; scan this one from the top bin down, 4 bins per thread
        for (int i = t; i < 2048; i += TK_T) hist2[(buf ^ 1) * 2048 + i] = 0;
        int c[4], csum = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) { const int r = 4 * t + j; c[j] = r < nbins ? H[nbins - 1 - r] : 0; csum += c[j]; }
        int x = csum;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) { const int y = __shfl_up_sync(0xffffffffu, x, o); if (lane >= o) x += y; }
        if (lane == 31) s_ws[warp] = x;
        __syncthreads();
        int pre = 0;
        for (int w2 = 0; w2 < warp; ++w2) pre += s_ws[w2];
        int acc = above + pre + x - csum;
        if (acc < width && acc + csum >= width) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (acc + c[j] >= width) { s_dab[buf] = nbins - 1 - (4 * t + j); s_dab[2 + buf] = acc; break; }
                acc += c[j];
            }
        }
        __syncthreads();
        prefix |= (uint32_t) s_dab[buf] << lo;
        above = s_dab[2 + buf];
        buf ^= 1;
        cur = lo - 1;
    }
    const uint32_t thr = prefix;
    const int eq_budget = (int) (width - above);
    const int64_t w0 = (int64_t) warp * 32 * per;
    int gt = 0, eq = 0;
    for (int64_t b = w0 + lane; b < nb && b < w0 + 32 * per; b += 32) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t kk = keyf(b);
        if (kk > thr) gt += w;
        else if (kk == thr) eq += w;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        gt += __shfl_xor_sync(0xffffffffu, gt, o);
        eq += __shfl_xor_sync(0xffffffffu, eq, o);
    }
    if (lane == 0) { s_gt[warp] = gt; s_eq[warp] = eq; }
    __syncthreads();
    if (t == 0) {
        int eb = 0, sb = 0;
        for (int w2 = 0; w2 < TK_W; ++w2) {
            const int g2 = s_gt[w2], e2 = s_eq[w2];
            const int take = eq_budget - eb < 0 ? 0 : (eq_budget - eb > e2 ? e2 : eq_budget - eb);
            s_gt[w2] = sb; s_eq[w2] = eb;
            sb += g2 + take; eb += e2;
        }
    }
    __syncthreads();
    int run_sel = s_gt[warp], run_eq = s_eq[warp];
    for (int64_t r0 = w0; r0 < nb && r0 < w0 + 32 * per; r0 += 32) {
        const int64_t b = r0 + lane;
        const int w = b < nb ? weight(b) : 0;
        const uint32_t kk = w ? keyf(b) : 0u;
        const int my_gt = (w && kk > thr) ? w : 0, my_eq = (w && kk == thr) ? w : 0;
        if (__ballot_sync(0xffffffffu, (my_gt | my_eq) != 0) == 0u) continue;
        int pe = my_eq;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) { const int y = __shfl_up_sync(0xffffffffu, pe, o); if (lane >= o) pe += y; }
        const int left = eq_budget - (run_eq + pe - my_eq);
        const int take = left < 0 ? 0 : (left > my_eq ? my_eq : left);
        const int my_sel = my_gt + take;
        int ps = my_sel;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) { const int y = __shfl_up_sync(0xffffffffu, ps, o); if (lane >= o) ps += y; }
        int32_t* dst = out + run_sel + ps - my_sel;
        for (int c = 0; c < my_sel; ++c) dst[c] = (int32_t) (b * R + c);
        run_eq += __shfl_sync(0xffffffffu, pe, 31);
        run_sel += __shfl_sync(0xffffffffu, ps, 31);
    }
}
} }  // namespace
