// generated from qsa_dec_attn.cu (chunk + merge device code), patched: half-CTA barriers 5/6
#pragma once
#include "basalt/kernels/qsa_dec_attn.hpp"
#include "basalt/kernels/qsa_seg.hpp"
#include "basalt/kernels/kv_q8.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/q8_1_finite.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdio>
#include <cstdlib>

namespace basalt::kernels {
namespace q2attn {

constexpr int HD = 256;          // head_dim
constexpr int G = 12;            // query heads per KV head
constexpr int CHUNK = 64;        // cells per block (the reference's chunk: the partial sums depend on it)
constexpr int THREADS = 256;
__device__ __forceinline__ int tid256() { return threadIdx.x & 255; }
__device__ __forceinline__ void hsync() {
    if (threadIdx.x < 256) asm volatile("bar.sync 5, 256;" ::: "memory");
    else asm volatile("bar.sync 6, 256;" ::: "memory");
}
constexpr int WARPS = THREADS / 32;
constexpr int NSC = HD / KV_Q8_GROUP;   // int8 scales per row
// The draft kernel's cells per CTA, from the window's width: a narrow window gets more CTAs of one tile, a wide one fewer
// of up to four (the merge reads the same rule).
constexpr int DT = 64;
__device__ __forceinline__ int draft_tpc(int n_ids) {
    const int tiles = (n_ids + DT - 1) / DT;
    return tiles >= 256 ? 4 : tiles >= 96 ? 2 : 1;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// The 12 heads' lane sums (part[12..15] zero) reduce-scattered with warp_sum's pairing (xor 16, 8, 4, 2, 1): every
// output adds the same two operands at every level, so each sum is warp_sum's bit for bit.  Lane l ends holding head
// head_of_lane(l); lanes l and l^1 agree.
__device__ __forceinline__ float reduce12(const float (&part)[16], int lane) {
    float r8[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const bool hi = (lane & 16) != 0;
        r8[i] = (hi ? part[i + 8] : part[i]) + __shfl_xor_sync(0xffffffffu, hi ? part[i] : part[i + 8], 16);
    }
    float r4[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const bool hi = (lane & 8) != 0;
        r4[i] = (hi ? r8[i + 4] : r8[i]) + __shfl_xor_sync(0xffffffffu, hi ? r8[i] : r8[i + 4], 8);
    }
    float r2[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const bool hi = (lane & 4) != 0;
        r2[i] = (hi ? r4[i + 2] : r4[i]) + __shfl_xor_sync(0xffffffffu, hi ? r4[i] : r4[i + 2], 4);
    }
    const bool hi2 = (lane & 2) != 0;
    const float r1 = (hi2 ? r2[1] : r2[0]) + __shfl_xor_sync(0xffffffffu, hi2 ? r2[0] : r2[1], 2);
    return r1 + __shfl_xor_sync(0xffffffffu, r1, 1);
}
__device__ __forceinline__ int head_of_lane(int lane) {
    return ((lane >> 4) & 1) * 8 + ((lane >> 3) & 1) * 4 + ((lane >> 2) & 1) * 2 + ((lane >> 1) & 1);
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    const unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async8(void* smem, const void* gmem) {
    const unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::); }

// shared memory of one chunk block; KV_MODE 0: fp16 rows, 1: int8 rows + fp16 scales
template <int KV_MODE> struct ChunkSmem {
    static constexpr int ROW = KV_MODE == 0 ? HD * 2 : HD;   // bytes of one cell's K (or V) row
    alignas(16) unsigned char k[CHUNK][ROW];
    alignas(16) unsigned char v[CHUNK][ROW];
    alignas(16) uint16_t ks[CHUNK][NSC];
    alignas(16) uint16_t vs[CHUNK][NSC];
    float sp[G][CHUNK];        // scores
    alignas(16) float spt[CHUNK][G];   // probabilities, cell-major
    long long row[CHUNK];      // pool row of each cell, -1 masked or past the width
};

struct AttnArgs {
    const float* q;
    QsaAttnPools p;
    QsaSegTable tab;           // SEG: each query's pools come from its slot
    const int32_t* ids;
    const int32_t* steps;
    int n_kv_heads, page_size, n_chunks, cap;
    int chunk = CHUNK;         // cells per partial: CHUNK, or 0 for the draft kernel's (draft_tpc of the width)
    long long stride;          // scratch floats per query
    float scale;
    float* scratch;
    float* attn;
    const float* q_full;       // non-null: the gated output too
    float* attn_gated;
    void* xq;                  // non-null (with the gated output): the gated rows' q8_1 blocks
};

// k8 of the row's dimensions [d0, d0+8) as the reference's load8 computes them
template <int KV_MODE>
__device__ __forceinline__ void k8_from(const ChunkSmem<KV_MODE>& sm, int c, int d0, float* out) {
    if constexpr (KV_MODE == 0) {
        const uint4 raw = *reinterpret_cast<const uint4*>(&sm.k[c][d0 * 2]);
        const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = __half22float2(h2[j]);
            out[2 * j] = f.x;
            out[2 * j + 1] = f.y;
        }
    } else {
        const float sc = __half2float(__ushort_as_half(sm.ks[c][d0 / KV_Q8_GROUP]));
        const uint2 raw = *reinterpret_cast<const uint2*>(&sm.k[c][d0]);
        const int8_t* cc = reinterpret_cast<const int8_t*>(&raw);
#pragma unroll
        for (int j = 0; j < 8; ++j) out[j] = (float) cc[j] * sc;
    }
}
template <int KV_MODE>
__device__ __forceinline__ float v1_from(const ChunkSmem<KV_MODE>& sm, int c, int d) {
    if constexpr (KV_MODE == 0) {
        return __half2float(*reinterpret_cast<const __half*>(&sm.v[c][d * 2]));
    } else {
        const float sc = __half2float(__ushort_as_half(sm.vs[c][d / KV_Q8_GROUP]));
        return (float) (int8_t) sm.v[c][d] * sc;
    }
}

// One chunk of one query and KV head: partial acc / max / sum for the 12 query heads, as attn_chunk_kernel.
template <int KV_MODE, bool SEG>
__device__ __forceinline__ void attn_chunk_dev(const AttnArgs& a, int chunk, int kvh, int qi, ChunkSmem<KV_MODE>& sm) {
    const QsaAttnPools pools = SEG ? a.tab.slots[a.tab.rows[qi].slot].pools() : a.p;
    constexpr int ROW = ChunkSmem<KV_MODE>::ROW;
    constexpr int PER = ROW / 16;
    const float* q = a.q + (size_t) qi * (size_t) (a.n_kv_heads * G) * HD;
    const int32_t* ids = a.ids + (size_t) qi * (size_t) a.cap;
    const int32_t* step = a.steps + (size_t) qi * kStepCount;
    float* part_acc = a.scratch + (size_t) qi * (size_t) a.stride;
    float* part_m = part_acc + (size_t) a.n_chunks * a.n_kv_heads * G * HD;
    float* part_l = part_m + (size_t) a.n_chunks * a.n_kv_heads * G;
    const int t = tid256(), lane = t & 31, warp = t >> 5;
    const int c0 = chunk * CHUNK;
    // this block's cell ids are requested before the width is known (ids past the width are read in bounds and
    // unused), then the query heads' registers (8 of the 256 dimensions per lane)
    const int cell = (t < CHUNK && c0 + t < a.cap) ? ids[c0 + t] : 0;
    const int n_ids = __ldg(step + kStepWidth);
    float qr[G][8];
#pragma unroll
    for (int h = 0; h < G; ++h) {
        const float4 qa = *reinterpret_cast<const float4*>(q + (size_t) (kvh * G + h) * HD + lane * 8);
        const float4 qb = *reinterpret_cast<const float4*>(q + (size_t) (kvh * G + h) * HD + lane * 8 + 4);
        qr[h][0] = qa.x; qr[h][1] = qa.y; qr[h][2] = qa.z; qr[h][3] = qa.w;
        qr[h][4] = qb.x; qr[h][5] = qb.y; qr[h][6] = qb.z; qr[h][7] = qb.w;
    }
    const int n_here = min(CHUNK, n_ids - c0);
    if (n_here <= 0) return;   // the merge reads only the chunks the width reaches
    const int slot = kvh * a.n_chunks + chunk;
    if (t < CHUNK) {
        long long r = -1;
        if (t < n_here) {
            const long long page = (long long) pools.page_table[cell / a.page_size];
            if (page >= 0) r = (page * a.n_kv_heads + kvh) * a.page_size + (cell % a.page_size);
        }
        sm.row[t] = r;
    }
    hsync();
    {
        const unsigned char* kb = KV_MODE == 0 ? (const unsigned char*) pools.k_pool : (const unsigned char*) pools.k_q;
        const unsigned char* vb = KV_MODE == 0 ? (const unsigned char*) pools.v_pool : (const unsigned char*) pools.v_q;
        for (int i = t; i < CHUNK * PER; i += THREADS) {
            const int c = i / PER, j = i % PER;
            const long long r = sm.row[c];
            if (r < 0) {   // a masked or missing cell's values are zero, so the value loop needs no branch
                *reinterpret_cast<uint4*>(&sm.v[c][j * 16]) = make_uint4(0, 0, 0, 0);
                continue;
            }
            cp_async16(&sm.k[c][j * 16], kb + r * ROW + j * 16);
            cp_async16(&sm.v[c][j * 16], vb + r * ROW + j * 16);
        }
        if constexpr (KV_MODE == 1) {
            if (t < 2 * CHUNK) {
                const int c = t & (CHUNK - 1);
                const long long r = sm.row[c];
                if (r >= 0) {
                    if (t < CHUNK) cp_async8(&sm.ks[c][0], pools.k_scale + r * NSC);
                    else cp_async8(&sm.vs[c][0], pools.v_scale + r * NSC);
                } else if (t >= CHUNK) {
                    *reinterpret_cast<uint2*>(&sm.vs[c][0]) = make_uint2(0, 0);
                }
            }
        }
    }
    cp_async_wait_all();
    hsync();
    // scores: warp w takes cells w, w+8, ...; two cells per pass
    const int hd = head_of_lane(lane);
    constexpr int NC = 2;
#pragma unroll 1
    for (int i = 0; i < CHUNK / WARPS; i += NC) {
        int cc[NC];
        bool ok[NC];
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            cc[j] = warp + WARPS * (i + j);
            ok[j] = cc[j] < n_here && sm.row[cc[j]] >= 0;
        }
        float kk[NC][8];
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            if (ok[j]) k8_from<KV_MODE>(sm, cc[j], lane * 8, kk[j]);
            else {
#pragma unroll
                for (int e = 0; e < 8; ++e) kk[j][e] = 0.0f;
            }
        }
        float pp[NC][16];
#pragma unroll
        for (int h = 0; h < G; ++h) {
#pragma unroll
            for (int j = 0; j < NC; ++j)
                pp[j][h] = kk[j][0] * qr[h][0] + kk[j][1] * qr[h][1] + kk[j][2] * qr[h][2] + kk[j][3] * qr[h][3] +
                           kk[j][4] * qr[h][4] + kk[j][5] * qr[h][5] + kk[j][6] * qr[h][6] + kk[j][7] * qr[h][7];
        }
#pragma unroll
        for (int j = 0; j < NC; ++j) {
#pragma unroll
            for (int h = G; h < 16; ++h) pp[j][h] = 0.0f;
            const float sj = reduce12(pp[j], lane);
            if (ok[j]) {
                if ((lane & 1) == 0 && hd < G) sm.sp[hd][cc[j]] = sj * a.scale;
            } else if (lane < G) {
                sm.sp[lane][cc[j]] = -FLT_MAX;
            }
        }
    }
    hsync();
    // per-head chunk max and exp-sum: warp w handles heads w and w+8
    {
        const bool ok0 = lane < n_here && sm.row[lane] >= 0, ok1 = lane + 32 < n_here && sm.row[lane + 32] >= 0;
        const int h0 = warp, h1 = warp + WARPS;   // heads w and w+8 (the second exists for w < 4)
        const bool two = h1 < G;
        const float x0 = sm.sp[h0][lane], x1 = sm.sp[h0][lane + 32];
        const float y0 = two ? sm.sp[h1][lane] : 0.0f, y1 = two ? sm.sp[h1][lane + 32] : 0.0f;
        const float m = warp_max(fmaxf(x0, x1));
        const float n = warp_max(fmaxf(y0, y1));
        const float e0 = ok0 ? __expf(x0 - m) : 0.0f, e1 = ok1 ? __expf(x1 - m) : 0.0f;
        const float f0 = ok0 ? __expf(y0 - n) : 0.0f, f1 = ok1 ? __expf(y1 - n) : 0.0f;
        sm.spt[lane][h0] = e0;
        sm.spt[lane + 32][h0] = e1;
        if (two) {
            sm.spt[lane][h1] = f0;
            sm.spt[lane + 32][h1] = f1;
        }
        const float l = warp_sum(e0 + e1);
        const float l2 = warp_sum(f0 + f1);
        if (lane == 0) {
            part_m[slot * G + h0] = m; part_l[slot * G + h0] = l;
            if (two) { part_m[slot * G + h1] = n; part_l[slot * G + h1] = l2; }
        }
    }
    hsync();
    // values: thread t < 128 owns dimensions 2t, 2t+1 of all 12 heads; cells folded in ascending order.  A masked or
    // missing cell has weight 0 and value 0, which leaves every accumulator unchanged (none is ever -0).
    if (t >= HD / 2) return;
    float acc[G][2];
#pragma unroll
    for (int i = 0; i < G; ++i) acc[i][0] = acc[i][1] = 0.0f;
#pragma unroll 8
    for (int c = 0; c < CHUNK; ++c) {
        float v0, v1;
        if constexpr (KV_MODE == 0) {
            const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&sm.v[c][t * 4]));
            v0 = f.x; v1 = f.y;
        } else {
            const float sc = __half2float(__ushort_as_half(sm.vs[c][t / (KV_Q8_GROUP / 2)]));
            const ushort raw = *reinterpret_cast<const ushort*>(&sm.v[c][t * 2]);
            v0 = (float) (int8_t) (raw & 0xff) * sc;
            v1 = (float) (int8_t) (raw >> 8) * sc;
        }
        const float4 w0 = *reinterpret_cast<const float4*>(&sm.spt[c][0]);
        const float4 w1 = *reinterpret_cast<const float4*>(&sm.spt[c][4]);
        const float4 w2 = *reinterpret_cast<const float4*>(&sm.spt[c][8]);
        const float w[G] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w, w2.x, w2.y, w2.z, w2.w};
#pragma unroll
        for (int i = 0; i < G; ++i) {
            acc[i][0] = fmaf(w[i], v0, acc[i][0]);
            acc[i][1] = fmaf(w[i], v1, acc[i][1]);
        }
    }
#pragma unroll
    for (int i = 0; i < G; ++i)
        *reinterpret_cast<float2*>(&part_acc[((size_t) slot * G + i) * HD + 2 * t]) = make_float2(acc[i][0], acc[i][1]);
}

// sigmoid gate with the fast-math sequence native_qsa_gate_apply compiles to
__device__ __forceinline__ float gate_fast(float x, float raw) {
    float y;
    asm("{\n\t.reg .f32 t, e, d, r;\n\t"
        "mul.ftz.f32 t, %2, 0fBFB8AA3B;\n\t"
        "ex2.approx.ftz.f32 e, t;\n\t"
        "add.ftz.f32 d, e, 0f3F800000;\n\t"
        "rcp.approx.ftz.f32 r, d;\n\t"
        "mul.ftz.f32 %0, %1, r;\n\t}"
        : "=f"(y) : "f"(x), "f"(raw));
    return y;
}

// native_quantize_q8_1_kernel's block (fast-math operations spelled out): lane l holds element l of the block
struct AttnQ81 {
    half2 ds;
    int8_t qs[32];
};
__device__ __forceinline__ float fm_max(float a, float b) { float y; asm("max.ftz.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b)); return y; }
__device__ __forceinline__ float fm_abs(float a) { float y; asm("abs.ftz.f32 %0, %1;" : "=f"(y) : "f"(a)); return y; }
__device__ __forceinline__ float fm_add(float a, float b) { float y; asm("add.ftz.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b)); return y; }
__device__ __forceinline__ float fm_div(float a, float b) { float y; asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b)); return y; }
__device__ __forceinline__ void q8_1_store_fm(float xi, AttnQ81* y, size_t ib, int lane) {
    float amax = fm_abs(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fm_max(amax, __shfl_xor_sync(0xffffffffu, amax, o));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum = fm_add(sum, __shfl_xor_sync(0xffffffffu, sum, o));
    float d = fm_div(amax, 127.0f);
    d = q8_1_finite(d);
    int8_t q = 0;
    if (amax != 0.0f) {
        const float r = roundf(fm_div(xi, d));
        q = (int8_t) (r > 127.0f ? 127.0f : r < -127.0f ? -127.0f : r);
    }
    y[ib].qs[lane] = q;
    if (lane == 0) y[ib].ds = q8_1_ds(d, sum);
}

// One query head of one query: merge the chunks the width reaches, as attn_merge_kernel (which skips the rest).
__device__ __forceinline__ void attn_merge_dev(const AttnArgs& a, int h, int qi, float* s_w, float* s_l, float* s_red) {
    const float* part_acc = a.scratch + (size_t) qi * (size_t) a.stride;
    const float* part_m = part_acc + (size_t) a.n_chunks * a.n_kv_heads * G * HD;
    const float* part_l = part_m + (size_t) a.n_chunks * a.n_kv_heads * G;
    const int n_ids = __ldg(a.steps + (size_t) qi * kStepCount + kStepWidth);
    const int chunk_cells = a.chunk > 0 ? a.chunk : DT * draft_tpc(n_ids);
    const int nc = min(a.n_chunks, (n_ids + chunk_cells - 1) / chunk_cells);
    const int kvh = h / G, hl = h % G;
    const int d = tid256();
    float mx = -FLT_MAX;
    for (int c = d; c < nc; c += THREADS) {
        const int slot = kvh * a.n_chunks + c;
        const float m = part_m[slot * G + hl];
        s_w[c] = m;
        s_l[c] = part_l[slot * G + hl];
        mx = fmaxf(mx, m);
    }
    // the chunks' partial sums are requested before the maxima are known
    constexpr int PRE = 32;
    const float* pa = part_acc + ((size_t) (kvh * a.n_chunks) * G + hl) * HD + d;
    const size_t cs = (size_t) G * HD;
    float xp[PRE];
#pragma unroll
    for (int j = 0; j < PRE; ++j) xp[j] = j < nc ? pa[(size_t) j * cs] : 0.0f;
    mx = warp_max(mx);
    if ((d & 31) == 0) s_red[d >> 5] = mx;
    hsync();
    float M = s_red[0];
#pragma unroll
    for (int w = 1; w < WARPS; ++w) M = fmaxf(M, s_red[w]);
    for (int c = d; c < nc; c += THREADS) {
        const float m = s_w[c];
        s_w[c] = m == -FLT_MAX ? -1.0f : __expf(m - M);   // -1: a chunk the reference skips
    }
    hsync();
    float L = 0.0f, acc = 0.0f;
    int c = 0;
#pragma unroll
    for (int j = 0; j < PRE; ++j) {
        if (j >= nc) break;
        const float w = s_w[j];
        if (w < 0.0f) continue;
        L = fmaf(s_l[j], w, L);
        acc = fmaf(xp[j], w, acc);
    }
    c = min(nc, PRE);
    for (; c + 8 <= nc; c += 8) {
        float x[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) x[j] = pa[(size_t) (c + j) * cs];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float w = s_w[c + j];
            if (w < 0.0f) continue;
            L = fmaf(s_l[c + j], w, L);
            acc = fmaf(x[j], w, acc);
        }
    }
    for (; c < nc; ++c) {
        const float w = s_w[c];
        if (w < 0.0f) continue;
        L = fmaf(s_l[c], w, L);
        acc = fmaf(pa[(size_t) c * cs], w, acc);
    }
    const int n_head = a.n_kv_heads * G;
    const float o = L > 0.0f ? acc / L : 0.0f;
    a.attn[((size_t) qi * n_head + h) * HD + d] = o;
    if (a.attn_gated != nullptr) {
        const float raw = a.q_full[((size_t) qi * n_head + h) * 2 * HD + HD + d];
        const float gv = gate_fast(o, raw);
        a.attn_gated[((size_t) qi * n_head + h) * HD + d] = gv;
        if (a.xq != nullptr) q8_1_store_fm(gv, static_cast<AttnQ81*>(a.xq), (((size_t) qi * n_head + h) * HD + d) / 32, d & 31);
    }
}




constexpr size_t HALF_CHUNK_B = (sizeof(ChunkSmem<1>) + 127) & ~(size_t) 127;
constexpr size_t HALF_B = HALF_CHUNK_B + 512;          // + merge's (2 * n_chunks + 8) floats
constexpr size_t ARENA_B = 2 * HALF_B;
// Chunk phase. Entry: qcur (q post phase), ids (top-k phase), pools visible. Work: item i = (chunk, kv head, token) with
// i = hg, hg + 2*grid, ...; hg = 2*cta + half (two independent 256-thread halves per CTA, named barriers 5 and 6).
// Exit: scratch partials (acc / max / sum per chunk, kv head, token), plain stores, each written once.
// smem: 2 * HALF_B (arena), 16-B aligned; the chunk uses [0, sizeof(ChunkSmem<1>)) of its half.
__device__ __forceinline__ void qsa_attn_chunk_phase(const AttnArgs& a, int n, unsigned char* arena) {
    const int half = threadIdx.x >> 8, hg = blockIdx.x * 2 + half, total = n * a.n_kv_heads * a.n_chunks;
    ChunkSmem<1>& sm = *reinterpret_cast<ChunkSmem<1>*>(arena + half * HALF_B);
    for (int i = hg; i < total; i += gridDim.x * 2) {
        const int chunk = i % a.n_chunks, kvh = (i / a.n_chunks) % a.n_kv_heads, qi = i / (a.n_chunks * a.n_kv_heads);
        attn_chunk_dev<1, false>(a, chunk, kvh, qi, sm);
    }
}
// Merge phase (+ sigmoid gate and the gated rows' q8_1 blocks, the buffer attn_output consumes). Entry: all chunk partials of
// (kv head, token) visible (grid hop after the chunk phase; per-(kvh, token) counters would do). Item = (head, token).
// Exit: attn, attn_gated (F32 [n][24][256]) and xq (q8_1, [n][192] blocks of 36 B), one writer per element.
__device__ __forceinline__ void qsa_attn_merge_phase(const AttnArgs& a, int n, unsigned char* arena) {
    const int half = threadIdx.x >> 8, hg = blockIdx.x * 2 + half, total = n * a.n_kv_heads * G;
    float* msm = reinterpret_cast<float*>(arena + half * HALF_B + HALF_CHUNK_B);
    for (int i = hg; i < total; i += gridDim.x * 2) {
        const int h = i % (a.n_kv_heads * G), qi = i / (a.n_kv_heads * G);
        attn_merge_dev(a, h, qi, msm, msm + a.n_chunks, msm + 2 * a.n_chunks);
        hsync();
    }
}

}  // namespace
}  // namespace basalt::kernels
