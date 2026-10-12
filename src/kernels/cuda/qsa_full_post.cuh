// generated from qsa_dec_fused.cu (device code only), patched: half-CTA barriers, smem arena param
#pragma once
// phase 1 device code: PostArgs, qsa_post_phase (generated dev functions + the phase loop)

#include "basalt/kernels/qsa_seg.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/f16_bits.hpp"
#include "basalt/kernels/kv_q8.hpp"
#include "basalt/kernels/mrope.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace basalt::kernels {
namespace q2post {

constexpr int HD = 256, ID = 128, R = 4, ROT = 64, THREADS = 256;

// two 256-thread halves of a 512-thread CTA run independent items; each half syncs on its own named barrier
__device__ __forceinline__ int tid256() { return threadIdx.x & 255; }
__device__ __forceinline__ void hsync() {
    if (threadIdx.x < 256) asm volatile("bar.sync 3, 256;" ::: "memory");
    else asm volatile("bar.sync 4, 256;" ::: "memory");
}


// Phase 1 is the engine's fast-math file (qsa_dec_fused.cu): flush-to-zero arithmetic and the approximate
// division / rsqrt / sin / cos / ex2 / lg2 / rcp. Spelled as PTX so the phase sits in a TU built without fast math;
// every arithmetic op is an explicit .rn so the compiler cannot contract or re-associate it.
__device__ __forceinline__ float fz_mul(float a, float b) { float y; asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b)); return y; }
__device__ __forceinline__ float fz_add(float a, float b) { float y; asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b)); return y; }
__device__ __forceinline__ float fz_sub(float a, float b) { float y; asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b)); return y; }
__device__ __forceinline__ float fz_fma(float a, float b, float c) { float y; asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(y) : "f"(a), "f"(b), "f"(c)); return y; }
__device__ __forceinline__ float fz_div(float a, float b) { float y; asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b)); return y; }
__device__ __forceinline__ float fz_rcp(float a) { float y; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(a)); return y; }
__device__ __forceinline__ float fz_rsqrt(float a) { float y; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(a)); return y; }
__device__ __forceinline__ float fz_lg2(float a) { float y; asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(a)); return y; }
__device__ __forceinline__ float fz_ex2(float a) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(a)); return y; }
__device__ __forceinline__ float fz_sin(float a) { float y; asm("sin.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(a)); return y; }
__device__ __forceinline__ float fz_cos(float a) { float y; asm("cos.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(a)); return y; }
__device__ __forceinline__ float fz_max(float a, float b) { float y; asm("max.ftz.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b)); return y; }
// powf under fast math: ex2(y * lg2(x))
__device__ __forceinline__ float fz_powf(float x, float y) { return fz_ex2(fz_mul(fz_lg2(x), y)); }

// rope_scaling.hpp's rope_scaled_angle with the fast-math operations
__device__ __forceinline__ void fz_rope_angle(float theta_extrap, float freq_scale, float corr_low, float corr_high,
                                              float ext_factor, float mscale_in, int pair, float& cos_out, float& sin_out) {
    float theta = fz_mul(freq_scale, theta_extrap);
    float mscale = mscale_in;
    if (ext_factor != 0.0f) {
        const float y = fz_div(fz_sub(float(pair), corr_low), fz_max(fz_sub(corr_high, corr_low), 0.001f));
        const float clamped = y < 0.0f ? 0.0f : (y > 1.0f ? 1.0f : y);
        const float ramp_mix = fz_mul(ext_factor, fz_sub(1.0f, clamped));
        theta = fz_fma(theta_extrap, ramp_mix, fz_mul(theta, fz_sub(1.0f, ramp_mix)));
        mscale = fz_mul(mscale, fz_fma(fz_mul(fz_lg2(fz_rcp(freq_scale)), 0.693147182464599609375f), 0.100000001490116119384765625f, 1.0f));
    }
    cos_out = fz_mul(mscale, fz_cos(theta));
    sin_out = fz_mul(mscale, fz_sin(theta));
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) value = fz_add(value, __shfl_xor_sync(0xffffffffu, value, offset, 32));
    return value;
}
// IEEE single-precision operations without flush-to-zero, as a file built without fast math emits them
__device__ __forceinline__ float abs_ieee(float x) { float y; asm("abs.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float max_ieee(float a, float b) { float y; asm("max.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b)); return y; }
__device__ __forceinline__ float div_ieee(float a, float b) { float y; asm("div.rn.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b)); return y; }

// native_qsa.cu's norm<256> for one row of n_cols <= 256 (thread d holds column d, x = 0 past n_cols): the scale
__device__ __forceinline__ float norm_scale256(float x, int d, int n_cols, float epsilon, float* sums) {
    float partial = 0.0f;
    if (d < n_cols) partial = fz_fma(x, x, partial);
    partial = warp_sum(partial);
    const int lane = d % 32;
    if (lane == 0) sums[d / 32] = partial;
    hsync();
    partial = lane < THREADS / 32 ? sums[lane] : 0.0f;
    partial = warp_sum(partial);
    const float mean = fz_div(partial, float(n_cols));
    return fz_rsqrt(fz_add(mean, epsilon));
}

struct RopeArgs {
    float theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale;
    const int32_t* mtab;
    RopeTab rt;
};

// native_rope.cu's apply<TAB> on a row held in shared memory (pairs below n_rot/2 rotate in place)
template <bool TAB>
__device__ __forceinline__ void rope_row(float* row, int pos, int d, const RopeArgs& ra) {
    float r0 = 0.0f, r1 = 0.0f;
    const int pair = d;
    if (pair < ROT / 2) {
        float c, s;
        if (!(TAB && rope_tab_cs(ra.rt, mrope_pos(ra.mtab, pos, pair), pair, c, s))) {
            const float theta_extrap = fz_mul(float(mrope_pos(ra.mtab, pos, pair)), fz_powf(ra.theta_scale, float(pair)));
            fz_rope_angle(theta_extrap, ra.freq_scale, ra.corr_low, ra.corr_high, ra.ext_factor, ra.mscale, pair, c, s);
        }
        const float a = row[pair], b = row[pair + ROT / 2];
        r0 = fz_fma(c, a, -fz_mul(s, b));
        r1 = fz_fma(a, s, fz_mul(c, b));
    }
    hsync();
    if (pair < ROT / 2) {
        row[pair] = r0;
        row[pair + ROT / 2] = r1;
    }
    hsync();
}

// kv_q8.cu's kv_append_q8_kernel for one row of one KV head: thread d holds value d; groups of 64 are warp pairs
__device__ __forceinline__ void append_q8_row(float x, int d, long long pos, int h, int n_kv_heads, int page_size,
                                              const int32_t* table, int8_t* codes, uint16_t* scales, int8_t* host_codes,
                                              uint16_t* host_scales, float* wmax) {
    const int groups = HD / KV_Q8_GROUP, g = d / KV_Q8_GROUP, t = d % KV_Q8_GROUP;
    float a = abs_ieee(x);
    for (int o = 16; o > 0; o >>= 1) a = max_ieee(a, __shfl_xor_sync(0xffffffffu, a, o));
    if ((d & 31) == 0) wmax[d >> 5] = a;
    hsync();
    const float amax = max_ieee(wmax[2 * g], wmax[2 * g + 1]);
    const uint16_t sbits = f16_from_f32(div_ieee(amax, 127.0f));
    const float sf = f32_from_f16(sbits);
    int q = 0;
    if (sf > 0.0f) {
        q = __float2int_rn(div_ieee(x, sf));
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
    }
    const long long page = (long long) table[pos / page_size];
    if (page >= 0) {
        const long long row = (page * n_kv_heads + h) * page_size + (pos % page_size);
        codes[row * HD + g * KV_Q8_GROUP + t] = (int8_t) q;
        if (t == 0) scales[row * groups + g] = sbits;
    }
    if (host_codes != nullptr) {
        const long long row = ((pos / page_size) * n_kv_heads + h) * page_size + (pos % page_size);
        host_codes[row * HD + g * KV_Q8_GROUP + t] = (int8_t) q;
        if (t == 0) host_scales[row * groups + g] = sbits;
    }
    hsync();
}

// native_qsa_indexer.cu's append<TAB> for one row (pos_base 0)
template <bool TAB>
__device__ __forceinline__ void indexer_append_row(const float* raw, int pos, const float* gamma, float epsilon,
                                                   float* tail, float* dead, float* pooled, int32_t* block_pos,
                                                   int max_cells, const RopeArgs& ra, float* values, float* partials) {
    const int d = tid256();
    if (pos < 0 || pos >= max_cells) return;
    const int slot = pos % R;
    float incoming = 0.0f;
    if (d < ID) {
        incoming = __half2float(__float2half_rn(raw[d]));
        if (slot < R - 1) tail[slot * ID + d] = incoming;
    }
    if (pos != 0 && slot != R - 1) return;
    float mean = 0.0f;
    if (d < ID) {
        float sum = pos == 0 ? incoming : tail[d];
#pragma unroll
        for (int j = 1; j < R; ++j) sum = fz_add(sum, pos == 0 || j == R - 1 ? incoming : tail[j * ID + d]);
        mean = fz_fma(0.25f, sum, 0.0f);
    }
    float square_sum = 0.0f;
    if (d < ID) square_sum = fz_fma(mean, mean, square_sum);
    square_sum = warp_sum(square_sum);
    const int lane = d % 32;
    if (lane == 0) partials[d / 32] = square_sum;
    hsync();
    square_sum = lane < THREADS / 32 ? partials[lane] : 0.0f;
    square_sum = warp_sum(square_sum);
    const float scale = fz_rsqrt(fz_add(fz_div(square_sum, float(ID)), epsilon));
    if (d < ID) values[d] = fz_mul(fz_mul(mean, scale), gamma[d]);
    hsync();
    if (d < ID) {
        const int b = pos / R;
        const int rope_pos = pos == 0 ? 0 : R * b;
        float y = values[d];
        if (d < ROT) {
            const int pair = d % (ROT / 2);
            float c, s;
            if (!(TAB && rope_tab_cs(ra.rt, pos == 0 ? 0 : mrope_pos(ra.mtab, rope_pos, pair), pair, c, s))) {
                const float theta_extrap = fz_mul(float(pos == 0 ? 0 : mrope_pos(ra.mtab, rope_pos, pair)), fz_powf(ra.theta_scale, float(pair)));
                fz_rope_angle(theta_extrap, ra.freq_scale, ra.corr_low, ra.corr_high, ra.ext_factor, ra.mscale, pair, c, s);
            }
            const float a = values[pair], z = values[pair + ROT / 2];
            y = d < ROT / 2 ? fz_fma(c, a, -fz_mul(s, z)) : fz_fma(a, s, fz_mul(c, z));
        }
        pooled[std::size_t(b) * ID + d] = y;
        if (pos == 0) dead[d] = y;
        else pooled[std::size_t(b + 1) * ID + d] = dead[d];
        if (d == 0 && pos != 0) *block_pos = rope_pos;
    }
    hsync();
}

// blocks [0, n * n_kv): row t = b / n_kv, KV head b % n_kv - K norm + RoPE, K and V appends; block n * n_kv: the
// indexer tail snapshot and the n indexer appends in row order
template <bool TAB>
__device__ __forceinline__ void kv_post_dev(const QsaDecKvPost& a, int n, int n_kv, int page_size, const RopeArgs& ra,
                                            int b, float* sm) {
    float* row = sm + (threadIdx.x >> 8) * (HD + 32);
    float* sums = row + HD;
    const int d = tid256();
    if (b < n * n_kv) {
        const int t = b / n_kv, h = b % n_kv;
        const long long pos = (long long) __ldg(a.steps + (size_t) t * kStepCount + kStepPos);
        float* kp = a.kcur + ((size_t) t * n_kv + h) * HD;
        const float x = kp[d];
        const float scale = norm_scale256(x, d, HD, a.eps, sums);
        row[d] = fz_mul(fz_mul(x, scale), a.k_norm[d]);
        hsync();
        rope_row<TAB>(row, a.pos_k[t * (a.pos_stride > 0 ? a.pos_stride : n_kv) + h], d, ra);
        const float y = row[d];
        kp[d] = y;
        append_q8_row(y, d, pos, h, n_kv, page_size, a.page_table, a.k_q, a.k_scale, a.host.k_q, a.host.k_scale, sums);
        const float v = a.vcur[((size_t) t * n_kv + h) * HD + d];
        append_q8_row(v, d, pos, h, n_kv, page_size, a.page_table, a.v_q, a.v_scale, a.host.v_q, a.host.v_scale, sums);
        return;
    }
    if (a.tail_snap != nullptr)
        for (int i = d; i < (R - 1) * ID; i += THREADS) a.tail_snap[i] = a.tail[i];
    hsync();
    for (int t = 0; t < n; ++t) {
        const int pos = __ldg(a.steps + (size_t) t * kStepCount + kStepPos);
        indexer_append_row<TAB>(a.idx_raw + (size_t) t * ID, pos, a.idx_k_norm, a.eps, a.tail, a.dead, a.pooled,
                                a.block_pos, a.max_cells, ra, row, sums);
        hsync();
    }
}

// blocks [0, n * n_head): the query heads; then n * idx_n_head blocks: the indexer query heads
template <bool TAB>
__device__ __forceinline__ void q_post_dev(const QsaDecQPost& a, int n, int n_head, int idx_heads, const RopeArgs& ra,
                                           int b, float* sm) {
    float* row = sm + (threadIdx.x >> 8) * (HD + 32);
    float* sums = row + HD;
    const int d = tid256();
    if (b < n * n_head) {
        const float x = a.q_full[(size_t) b * 2 * HD + d];
        const float scale = norm_scale256(x, d, HD, a.eps, sums);
        row[d] = fz_mul(fz_mul(x, scale), a.q_norm[d]);
        hsync();
        rope_row<TAB>(row, a.pos_q[b], d, ra);
        a.qcur[(size_t) b * HD + d] = row[d];
        return;
    }
    b -= n * n_head;
    float* qp = a.q_idx + (size_t) b * ID;
    const float x = d < ID ? qp[d] : 0.0f;
    const float scale = norm_scale256(x, d, ID, a.eps, sums);
    if (d < ID) row[d] = fz_mul(fz_mul(x, scale), a.idx_q_norm[d]);
    hsync();
    rope_row<TAB>(row, a.pos_i[b], d, ra);
    if (d < ID) qp[d] = row[d];
}


static RopeArgs rope_args(const RopeScaling& scaling) {
    RopeArgs ra;
    ra.theta_scale = powf((float) scaling.freq_base, -2.0f / ROT);
    const RopeKernelArgs k = scaling.kernel_args(ROT);
    ra.freq_scale = k.freq_scale; ra.corr_low = k.corr_low; ra.corr_high = k.corr_high;
    ra.ext_factor = k.ext_factor; ra.mscale = k.attn_factor;
    ra.mtab = mrope_table();
    ra.rt = rope_table_for(scaling);
    return ra;
}


struct PostArgs { QsaDecKvPost kv; QsaDecQPost q; int n, n_head, idx_heads, n_kv, page_size; RopeArgs ra; };
// one phase invocation: every half-CTA (2 per CTA) takes items hg, hg+2*grid, ...: items [0, n*n_kv) K/V rows,
// item n*n_kv the indexer appends (rows in order), then the n*n_head query heads, then the n*idx_heads indexer heads.
// Entry: kcur/vcur/idx_raw/q_full/q_idx projections visible. Exit: kcur, qcur, q_idx, caches written (plain stores).
// sm: 2 * (HD + 32) floats. Named barriers 3 (half 0) and 4 (half 1), 256 threads each.
__device__ __forceinline__ void qsa_post_phase(const PostArgs& A, float* sm) {
    const int hg = blockIdx.x * 2 + (threadIdx.x >> 8);
    const int nkvb = A.n * A.n_kv, total = nkvb + 1 + A.n * A.n_head + A.n * A.idx_heads;
    for (int i = hg; i < total; i += gridDim.x * 2) {
        if (i <= nkvb) kv_post_dev<false>(A.kv, A.n, A.n_kv, A.page_size, A.ra, i, sm);
        else q_post_dev<false>(A.q, A.n, A.n_head, A.idx_heads, A.ra, i - nkvb - 1, sm);
    }
}

}  // namespace
}  // namespace basalt::kernels
