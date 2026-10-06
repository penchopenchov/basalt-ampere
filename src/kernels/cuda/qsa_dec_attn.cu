// src/kernels/cuda/qsa_dec_attn.cu - see include/basalt/kernels/qsa_dec_attn.hpp.
#include "basalt/kernels/qsa_dec_attn.hpp"
#include "basalt/kernels/kv_q8.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdio>
#include <cstdlib>

namespace basalt::kernels {
namespace {

constexpr int HD = 256;          // head_dim
constexpr int G = 12;            // query heads per KV head
constexpr int CHUNK = 64;        // cells per block (the reference's chunk: the partial sums depend on it)
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int NSC = HD / KV_Q8_GROUP;   // int8 scales per row

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
    float q[G][HD];
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
    const int32_t* ids;
    const int32_t* steps;
    int n_kv_heads, page_size, n_chunks, cap;
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
template <int KV_MODE>
__device__ __forceinline__ void attn_chunk_dev(const AttnArgs& a, int chunk, int kvh, int qi, ChunkSmem<KV_MODE>& sm) {
    constexpr int ROW = ChunkSmem<KV_MODE>::ROW;
    constexpr int PER = ROW / 16;
    const float* q = a.q + (size_t) qi * (size_t) (a.n_kv_heads * G) * HD;
    const int32_t* ids = a.ids + (size_t) qi * (size_t) a.cap;
    const int32_t* step = a.steps + (size_t) qi * kStepCount;
    float* part_acc = a.scratch + (size_t) qi * (size_t) a.stride;
    float* part_m = part_acc + (size_t) a.n_chunks * a.n_kv_heads * G * HD;
    float* part_l = part_m + (size_t) a.n_chunks * a.n_kv_heads * G;
    const int n_ids = __ldg(step + kStepWidth);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int c0 = chunk * CHUNK;
    const int n_here = min(CHUNK, n_ids - c0);
    if (n_here <= 0) return;   // the merge reads only the chunks the width reaches
    const int slot = kvh * a.n_chunks + chunk;
    // the query heads of this KV head (768 float4), then each cell's rows as soon as its pool row is known
    for (int i = t; i < G * HD / 4; i += THREADS)
        cp_async16(&sm.q[0][0] + 4 * i, q + (size_t) (kvh * G) * HD + 4 * i);
    if (t < CHUNK) {
        long long r = -1;
        if (t < n_here) {
            const int cell = ids[c0 + t];
            const long long page = (long long) a.p.page_table[cell / a.page_size];
            if (page >= 0) r = (page * a.n_kv_heads + kvh) * a.page_size + (cell % a.page_size);
        }
        sm.row[t] = r;
    }
    __syncthreads();
    {
        const unsigned char* kb = KV_MODE == 0 ? (const unsigned char*) a.p.k_pool : (const unsigned char*) a.p.k_q;
        const unsigned char* vb = KV_MODE == 0 ? (const unsigned char*) a.p.v_pool : (const unsigned char*) a.p.v_q;
        for (int i = t; i < CHUNK * PER; i += THREADS) {
            const int c = i / PER, j = i % PER;
            const long long r = sm.row[c];
            if (r < 0) continue;
            cp_async16(&sm.k[c][j * 16], kb + r * ROW + j * 16);
            cp_async16(&sm.v[c][j * 16], vb + r * ROW + j * 16);
        }
        if constexpr (KV_MODE == 1) {
            if (t < 2 * CHUNK) {
                const int c = t & (CHUNK - 1);
                const long long r = sm.row[c];
                if (r >= 0) {
                    if (t < CHUNK) cp_async8(&sm.ks[c][0], a.p.k_scale + r * NSC);
                    else cp_async8(&sm.vs[c][0], a.p.v_scale + r * NSC);
                }
            }
        }
    }
    cp_async_wait_all();
    __syncthreads();
    // scores: warp w takes cells w, w+8, ...; two cells per pass; each lane holds 8 of the 256 dimensions of the
    // 12 query heads in registers
    float qr[G][8];
#pragma unroll
    for (int h = 0; h < G; ++h) {
        const float4 qa = *reinterpret_cast<const float4*>(&sm.q[h][lane * 8]);
        const float4 qb = *reinterpret_cast<const float4*>(&sm.q[h][lane * 8 + 4]);
        qr[h][0] = qa.x; qr[h][1] = qa.y; qr[h][2] = qa.z; qr[h][3] = qa.w;
        qr[h][4] = qb.x; qr[h][5] = qb.y; qr[h][6] = qb.z; qr[h][7] = qb.w;
    }
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
    __syncthreads();
    // per-head chunk max and exp-sum: warp w handles heads w and w+8
    for (int h = warp; h < G; h += WARPS) {
        const float x0 = sm.sp[h][lane], x1 = sm.sp[h][lane + 32];
        const float m = warp_max(fmaxf(x0, x1));
        const float e0 = (lane < n_here && sm.row[lane] >= 0) ? __expf(x0 - m) : 0.0f;
        const float e1 = (lane + 32 < n_here && sm.row[lane + 32] >= 0) ? __expf(x1 - m) : 0.0f;
        sm.spt[lane][h] = e0;
        sm.spt[lane + 32][h] = e1;
        const float l = warp_sum(e0 + e1);
        if (lane == 0) { part_m[slot * G + h] = m; part_l[slot * G + h] = l; }
    }
    __syncthreads();
    // values: thread t < 192 owns dimensions 4*(t%64).. +3 of heads 4*(t/64).. +3; cells folded in ascending order.
    // A masked or missing cell folds weight 0 times value 0, which leaves every accumulator unchanged (none is ever
    // -0), so the loop has no branch.
    if (t >= 192) return;
    const int dg = t & 63, hg = t >> 6;
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;
#pragma unroll 8
    for (int c = 0; c < CHUNK; ++c) {
        const bool ok = c < n_here && sm.row[c] >= 0;
        float v[4];
        if constexpr (KV_MODE == 0) {
            const uint2 raw = *reinterpret_cast<const uint2*>(&sm.v[c][dg * 8]);
            const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
            const float2 f0 = __half22float2(h2[0]), f1 = __half22float2(h2[1]);
            v[0] = f0.x; v[1] = f0.y; v[2] = f1.x; v[3] = f1.y;
        } else {
            const float sc = __half2float(__ushort_as_half(sm.vs[c][dg / (KV_Q8_GROUP / 4)]));
            const char4 cv = *reinterpret_cast<const char4*>(&sm.v[c][dg * 4]);
            v[0] = (float) cv.x * sc; v[1] = (float) cv.y * sc; v[2] = (float) cv.z * sc; v[3] = (float) cv.w * sc;
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = ok ? v[j] : 0.0f;
        const float4 w = *reinterpret_cast<const float4*>(&sm.spt[c][hg * 4]);
        const float wv[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(wv[i], v[j], acc[i][j]);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
        *reinterpret_cast<float4*>(&part_acc[((size_t) slot * G + hg * 4 + i) * HD + dg * 4]) =
            make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
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
    d = fabsf(d) > 65504.0f ? copysignf(65504.0f, d) : d;
    int8_t q = 0;
    if (amax != 0.0f) {
        const float r = roundf(fm_div(xi, d));
        q = (int8_t) (r > 127.0f ? 127.0f : r < -127.0f ? -127.0f : r);
    }
    y[ib].qs[lane] = q;
    if (lane == 0) {
        const float sc = fabsf(sum) > 65504.0f ? copysignf(65504.0f, sum) : sum;
        y[ib].ds = make_half2(__float2half(d), __float2half(sc));
    }
}

// One query head of one query: merge the chunks the width reaches, as attn_merge_kernel (which skips the rest).
__device__ __forceinline__ void attn_merge_dev(const AttnArgs& a, int h, int qi, float* s_w, float* s_l, float* s_red) {
    const float* part_acc = a.scratch + (size_t) qi * (size_t) a.stride;
    const float* part_m = part_acc + (size_t) a.n_chunks * a.n_kv_heads * G * HD;
    const float* part_l = part_m + (size_t) a.n_chunks * a.n_kv_heads * G;
    const int n_ids = __ldg(a.steps + (size_t) qi * kStepCount + kStepWidth);
    const int nc = min(a.n_chunks, (n_ids + CHUNK - 1) / CHUNK);
    const int kvh = h / G, hl = h % G;
    const int d = threadIdx.x;
    float mx = -FLT_MAX;
    for (int c = d; c < nc; c += THREADS) {
        const int slot = kvh * a.n_chunks + c;
        const float m = part_m[slot * G + hl];
        s_w[c] = m;
        s_l[c] = part_l[slot * G + hl];
        mx = fmaxf(mx, m);
    }
    mx = warp_max(mx);
    if ((d & 31) == 0) s_red[d >> 5] = mx;
    __syncthreads();
    float M = s_red[0];
#pragma unroll
    for (int w = 1; w < WARPS; ++w) M = fmaxf(M, s_red[w]);
    for (int c = d; c < nc; c += THREADS) {
        const float m = s_w[c];
        s_w[c] = m == -FLT_MAX ? -1.0f : __expf(m - M);   // -1: a chunk the reference skips
    }
    __syncthreads();
    float L = 0.0f, acc = 0.0f;
    const float* pa = part_acc + ((size_t) (kvh * a.n_chunks) * G + hl) * HD + d;
    const size_t cs = (size_t) G * HD;
    int c = 0;
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

template <int KV_MODE>
__global__ void __launch_bounds__(THREADS) attn_chunk_v2(AttnArgs a) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    attn_chunk_dev<KV_MODE>(a, blockIdx.x, blockIdx.y, blockIdx.z, *reinterpret_cast<ChunkSmem<KV_MODE>*>(smem_raw));
}

__global__ void __launch_bounds__(THREADS) attn_merge_v2(AttnArgs a) {
    extern __shared__ __align__(16) float msm[];
    attn_merge_dev(a, blockIdx.x, blockIdx.y, msm, msm + a.n_chunks, msm + 2 * a.n_chunks);
}

bool a16(const void* p) { return ((uintptr_t) p & 15) == 0; }
int kv_mode_of(const QsaAttnPools& p) {
    if (p.k_q4 != nullptr || p.v_q4 != nullptr) return -1;
    if (p.k_q != nullptr)
        return (p.v_q != nullptr && p.k_scale != nullptr && p.v_scale != nullptr && a16(p.k_q) && a16(p.v_q) &&
                ((uintptr_t) p.k_scale & 7) == 0 && ((uintptr_t) p.v_scale & 7) == 0) ? 1 : -1;
    return (p.k_pool != nullptr && p.v_pool != nullptr && a16(p.k_pool) && a16(p.v_pool)) ? 0 : -1;
}
// the dynamic shared memory opt-in, once per device
template <int MODE> void opt_in(void (*kernel)(AttnArgs), size_t bytes) {
    static bool done[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) dev = 0;
    if (!done[dev]) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) bytes);
        done[dev] = true;
    }
}

}  // namespace

bool qsa_dec_attn_supported(const QsaAttnPools& pools, const QsaShapes& s) {
    return s.head_dim == HD && s.n_head == (int64_t) G * s.n_head_kv && kv_mode_of(pools) >= 0 &&
           pools.page_table != nullptr;
}

void qsa_dec_attn(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
                  const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream, const float* q_full,
                  float* attn_gated, void* xq) {
    if (n_q <= 0) return;
    if (!qsa_dec_attn_supported(pools, s) || cap <= 0 || !scratch || !ids || !steps || n_q > 65535) {
        std::fprintf(stderr, "qsa_dec_attn: unsupported geometry, pools or buffers\n");
        std::exit(1);
    }
    const int mode = kv_mode_of(pools);
    AttnArgs a;
    a.q = q; a.p = pools; a.ids = ids; a.steps = steps;
    a.n_kv_heads = (int) s.n_head_kv; a.page_size = (int) s.page_size;
    a.n_chunks = (int) ((cap + CHUNK - 1) / CHUNK); a.cap = (int) cap;
    a.stride = (long long) qsa_decode_attn_scratch_floats(cap, s);
    a.scale = 1.0f / sqrtf((float) HD);
    a.scratch = scratch; a.attn = attn;
    a.q_full = attn_gated != nullptr ? q_full : nullptr;
    a.attn_gated = q_full != nullptr ? attn_gated : nullptr;
    cudaStream_t st = (cudaStream_t) stream;
    a.xq = a.attn_gated != nullptr ? xq : nullptr;
    const dim3 grid((unsigned) a.n_chunks, (unsigned) s.n_head_kv, (unsigned) n_q);
    if (mode == 1) {
        constexpr size_t bytes = sizeof(ChunkSmem<1>);
        opt_in<1>(attn_chunk_v2<1>, bytes);
        attn_chunk_v2<1><<<grid, THREADS, bytes, st>>>(a);
    } else {
        constexpr size_t bytes = sizeof(ChunkSmem<0>);
        opt_in<0>(attn_chunk_v2<0>, bytes);
        attn_chunk_v2<0><<<grid, THREADS, bytes, st>>>(a);
    }
    const size_t mbytes = (size_t) (2 * a.n_chunks + WARPS) * sizeof(float);
    attn_merge_v2<<<dim3((unsigned) s.n_head, (unsigned) n_q), THREADS, mbytes, st>>>(a);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_dec_attn: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace basalt::kernels
