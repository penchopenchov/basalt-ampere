// src/kernels/cuda/hc_fused.cu - see include/strata/kernels/hc_fused.hpp.
//
// Built WITHOUT fast math, like fused_gr.cu, so the hc arithmetic below is fused_gr.cu's expressions compiled the
// same way.  The router part reproduces kernels built WITH fast math (the MMVF GEMV and the native top-10): those
// operations are spelled out as the flush-to-zero PTX instructions that build emits.
#include "strata/kernels/hc_fused.hpp"
#include "strata/kernels/pdl.hpp"

#include <cuda_runtime.h>

#include <cfloat>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int N = 2560;              // n_embd
constexpr int HC = 4;                // streams
constexpr int D = N * HC;            // 10240
constexpr int LR = 320;              // hc_lr
constexpr int NE = 512;              // experts
constexpr int KT = 10;               // routed per token
constexpr int MAXT = 8;
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int ROWS = 4;              // down rows per down CTA
constexpr int ND = LR / ROWS + 1;    // down CTAs: 80 of w_down, one of w_inject
constexpr int QB = 4;                // q-steps (32 chunks of 8) per ring stage
constexpr int NSTAGE = D / (QB * 256);   // 10
constexpr int STAGE_W = ROWS * QB * 32 * 16;   // weight bytes per stage (8 KB)
constexpr int STAGE_X1 = QB * 256 * 4;         // activation bytes per stage and token (4 KB)
constexpr int SMEM = 98 * 1024;      // dynamic shared memory per CTA
constexpr int MIN_NU = 86;           // up CTAs needed for an up slice (30 columns at most) to fit SMEM
static_assert(D % (QB * 256) == 0, "whole stages");

constexpr int NCOPY = 8;             // copies of xn: the down CTAs read different copies, spreading the L2 load
__device__ float g_hc_xn[NCOPY][MAXT][D];

struct HcSync {
    unsigned norm, down, up, router, exit, pad[11];
};
__device__ HcSync g_hc_sync;

struct HcLaunch {
    HcFusedArgs a;
    int nb;          // CTAs (one per SM)
    int up_early;    // w_up bulk copies at launch (1) or after the norm (0)
    unsigned long long* dbg;   // bench only: per-CTA phase times [nb][16]
};

// ---------------------------------------------------------------- primitives
__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned) __cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_u32(dst)), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int K>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(K) : "memory"); }
__device__ __forceinline__ void mbar_init(uint64_t* bar, unsigned count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait0(uint64_t* bar) {
    unsigned done = 0;
    while (!done) {
        asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0; selp.u32 %0, 1, 0, p; }"
                     : "=r"(done) : "r"(smem_u32(bar)) : "memory");
    }
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, unsigned bytes, uint64_t* bar) {
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void prefetch_l2(const void* p, size_t bytes) {   // 16-byte aligned, multiple of 16
    const char* c = (const char*) p;
    for (size_t o = 0; o < bytes; o += 65536) {
        const unsigned sz = (unsigned) (bytes - o < 65536 ? bytes - o : 65536);
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(c + o), "r"(sz) : "memory");
    }
}
__device__ __forceinline__ void stamp(unsigned long long* dbg, int b, int i, int tid) {
    if (dbg != nullptr && tid == 0) {
        unsigned long long t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        dbg[b * 16 + i] = t;
    }
}
__device__ __forceinline__ unsigned ld_acquire(const unsigned* p) {
    unsigned v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
// every thread's writes so far, then one release increment
__device__ __forceinline__ void cta_signal(unsigned* ctr, int tid) {
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        atomicAdd(ctr, 1u);
    }
}
__device__ __forceinline__ void cta_wait(const unsigned* ctr, unsigned target, int tid) {
    if (tid == 0) {
        while (ld_acquire(ctr) < target) {}
        __threadfence();
    }
    __syncthreads();
}

// the flush-to-zero single-precision operations of a fast-math build
__device__ __forceinline__ float fma_ftz(float a, float b, float c) {
    float d;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
    return d;
}
__device__ __forceinline__ float add_ftz(float a, float b) {
    float d;
    asm("add.ftz.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}
__device__ __forceinline__ float sub_ftz(float a, float b) {
    float d;
    asm("sub.ftz.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}
__device__ __forceinline__ float mul_ftz(float a, float b) {
    float d;
    asm("mul.ftz.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}
__device__ __forceinline__ float max_ftz(float a, float b) {
    float d;
    asm("max.ftz.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}
__device__ __forceinline__ float ex2_ftz(float a) {
    float d;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a));
    return d;
}
__device__ __forceinline__ float rcp_ftz(float a) {
    float d;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a));
    return d;
}
__device__ __forceinline__ float expf_fast(float x) { return ex2_ftz(mul_ftz(x, 1.4426950216293335f)); }
__device__ __forceinline__ float warp_sum_ftz(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = add_ftz(v, __shfl_xor_sync(0xffffffffu, v, o, 32));
    return v;
}
__device__ __forceinline__ float warp_max_ftz(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = max_ftz(v, __shfl_xor_sync(0xffffffffu, v, o, 32));
    return v;
}

// fused_gr.cu's operations
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ float dot8(const uint4 w, const float* x) {
    float acc = 0.0f;
    const uint32_t v[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        acc = fmaf(__uint_as_float(v[j] << 16), x[2 * j], acc);
        acc = fmaf(__uint_as_float(v[j] & 0xffff0000u), x[2 * j + 1], acc);
    }
    return acc;
}
__device__ __forceinline__ float dot8v(const uint4 w, const float4 x0, const float4 x1) {
    float acc = 0.0f;
    acc = fmaf(__uint_as_float(w.x << 16), x0.x, acc);
    acc = fmaf(__uint_as_float(w.x & 0xffff0000u), x0.y, acc);
    acc = fmaf(__uint_as_float(w.y << 16), x0.z, acc);
    acc = fmaf(__uint_as_float(w.y & 0xffff0000u), x0.w, acc);
    acc = fmaf(__uint_as_float(w.z << 16), x1.x, acc);
    acc = fmaf(__uint_as_float(w.z & 0xffff0000u), x1.y, acc);
    acc = fmaf(__uint_as_float(w.w << 16), x1.z, acc);
    acc = fmaf(__uint_as_float(w.w & 0xffff0000u), x1.w, acc);
    return acc;
}

// ---------------------------------------------------------------- the stages (block index passed in)

// (token t, stream c): rs and xn = R' * w_norm * rs.  Thread `tid` visits the float4 at tid*4 + 1024 m that lie in
// stream c, m ascending - the elements and order of the plain read's thread (fused_gr.cu gr_norm_split_kernel) - with
// every load issued up front.  hc_norm_pre loads the constant w_norm part before the dependency wait.
struct NormRegs {
    float4 g[3];
};
__device__ __forceinline__ int norm_m0(int c, int tid) {
    const int lo = c * N - tid * 4;
    return lo <= 0 ? 0 : (lo + 1023) / 1024;
}
__device__ __forceinline__ void hc_norm_pre(const HcFusedArgs& a, int c, int tid, NormRegs& nr) {
    const int m0 = norm_m0(c, tid);
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int i = tid * 4 + 1024 * (m0 + j);
        nr.g[j] = i < (c + 1) * N ? *reinterpret_cast<const float4*>(a.w_norm + i) : make_float4(0.f, 0.f, 0.f, 0.f);
    }
}
// Task (t, c, cp) writes copy cp of xn; copy 0 also writes rs.
__device__ void hc_norm_task(const HcFusedArgs& a, int t, int c, int cp, int tid, const NormRegs& nr, float* part,
                             float* s_rs) {
    const int lane = tid & 31, warp = tid >> 5;
    const float* R = a.R + (size_t) t * D;
    const float* bo = a.bo_prev + (size_t) t * N;
    const int m0 = norm_m0(c, tid);
    float4 r[3], bb[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int i = tid * 4 + 1024 * (m0 + j);
        const bool in = i < (c + 1) * N;
        r[j] = in ? *reinterpret_cast<const float4*>(R + i) : make_float4(0.f, 0.f, 0.f, 0.f);
        bb[j] = in && a.apply ? *reinterpret_cast<const float4*>(bo + i - c * N) : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    const float gw = a.apply ? 2.0f * sigmoidf_(a.inj_prev[t * HC + c] / (float) HC) : 0.0f;
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int i = tid * 4 + 1024 * (m0 + j);
        if (i >= (c + 1) * N) continue;
        if (a.apply) {
            r[j].x = fmaf(bb[j].x, gw, r[j].x); r[j].y = fmaf(bb[j].y, gw, r[j].y);
            r[j].z = fmaf(bb[j].z, gw, r[j].z); r[j].w = fmaf(bb[j].w, gw, r[j].w);
        }
        const float sq = r[j].x * r[j].x + r[j].y * r[j].y + r[j].z * r[j].z + r[j].w * r[j].w;
        ss += sq;
    }
    const float v = warp_sum(ss);
    if (lane == 0) part[warp] = v;
    __syncthreads();
    if (tid == 0) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += part[w];
        *s_rs = rsqrtf(s / (float) N + a.eps);
        if (cp == 0) a.rs[t * HC + c] = *s_rs;
    }
    __syncthreads();
    const float rs = *s_rs;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int i = tid * 4 + 1024 * (m0 + j);
        if (i >= (c + 1) * N) continue;
        const float4 g = nr.g[j];
        const float px = r[j].x * g.x, py = r[j].y * g.y, pz = r[j].z * g.z, pw = r[j].w * g.w;
        *reinterpret_cast<float4*>(&g_hc_xn[cp][t][i]) = make_float4(px * rs, py * rs, pz * rs, pw * rs);
    }
}

// The down CTA's ring: stage s holds chunks [s*128, s*128 + 128) of its four weight rows ([4][128] uint4) and of
// every token's xn as two planes ([T][2][128] float4: floats 0-3, then 4-7 of each chunk).
template <int T>
struct Ring {
    static constexpr int BYTES = STAGE_W + T * STAGE_X1;
    static constexpr int S0 = SMEM / BYTES;
    static constexpr int S = S0 > NSTAGE ? NSTAGE : S0;   // stages in flight
};
__device__ __forceinline__ const uint16_t* down_rows(const HcFusedArgs& a, int b) {
    return b < ND - 1 ? a.w_down + (size_t) b * ROWS * D : a.w_inject;
}
__device__ __forceinline__ void down_issue_w(const uint16_t* w, int s, char* slot, int tid) {
    uint4* W = reinterpret_cast<uint4*>(slot);
#pragma unroll
    for (int m = 0; m < 2; ++m) {
        const int i = tid + THREADS * m, r = i >> 7, jj = i & 127;
        cp_async16(W + i, w + (size_t) r * D + (size_t) (s * 128 + jj) * 8);
    }
}
template <int T>
__device__ __forceinline__ void down_issue_x(const float* xn, int s, char* slot, int tid) {
    float4* X = reinterpret_cast<float4*>(slot + STAGE_W);
#pragma unroll
    for (int m = 0; m < T; ++m) {
        const int i = tid + THREADS * m, k = i >> 8, s4 = i & 255;
        cp_async16(X + k * 256 + (s4 & 1) * 128 + (s4 >> 1), xn + (size_t) k * D + (size_t) s * 1024 + (size_t) s4 * 4);
    }
}

// Before the dependency wait: the weights of the first S stages (one cp.async group), and an L2 prefetch of the rest
// of the CTA's rows.
template <int T>
__device__ void hc_down_prologue(const HcFusedArgs& a, int b, int tid, char* ring) {
    constexpr int S = Ring<T>::S;
    const uint16_t* w = down_rows(a, b);
    if (w == nullptr) return;
    if (tid == 0 && S < NSTAGE)
        for (int r = 0; r < ROWS; ++r)
            prefetch_l2(w + (size_t) r * D + (size_t) S * 1024, (size_t) (NSTAGE - S) * 1024 * 2);
#pragma unroll
    for (int s = 0; s < S; ++s) down_issue_w(w, s, ring + s * Ring<T>::BYTES, tid);
    cp_async_commit();
}

// After the norm: lane L of warp (row r = w & 3, token parity p = w >> 2) sums chunks L + 32 q, q ascending, of row r
// against tokens p, p + 2, ... - the plain read's per-lane order.
template <int T>
__device__ void hc_down_main(const HcFusedArgs& a, int b, int ncopy, int tid, char* ring) {
    constexpr int S = Ring<T>::S, BYTES = Ring<T>::BYTES;
    constexpr int TW = (T + 1) / 2;   // tokens of parity 0 (parity 1 has T / 2)
    const uint16_t* w = down_rows(a, b);
    if (w == nullptr) return;
    const int lane = tid & 31, warp = tid >> 5, r = warp & 3, p = warp >> 2;
#pragma unroll
    for (int s = 0; s < S; ++s) {
        down_issue_x<T>(&g_hc_xn[b % ncopy][0][0], s, ring + s * BYTES, tid);
        cp_async_commit();
    }
    const int nt = p == 0 ? TW : T / 2;
    float acc[TW];
#pragma unroll
    for (int i = 0; i < TW; ++i) acc[i] = 0.0f;
#pragma unroll
    for (int s = 0; s < NSTAGE; ++s) {
        cp_async_wait<S - 1>();
        __syncthreads();
        const char* slot = ring + (s % S) * BYTES;
        const uint4* W = reinterpret_cast<const uint4*>(slot) + r * 128;
        const float4* X = reinterpret_cast<const float4*>(slot + STAGE_W) + p * 256;
#pragma unroll
        for (int qq = 0; qq < QB; ++qq) {
            const int j = lane + 32 * qq;
            const uint4 wv = W[j];
#pragma unroll
            for (int i = 0; i < TW; ++i)
                if (i < nt) acc[i] += dot8v(wv, X[i * 512 + j], X[i * 512 + 128 + j]);
        }
        __syncthreads();
        if (s + S < NSTAGE) {
            char* dst = ring + (s % S) * BYTES;
            down_issue_w(w, s + S, dst, tid);
            down_issue_x<T>(&g_hc_xn[b % ncopy][0][0], s + S, dst, tid);
        }
        cp_async_commit();
    }
#pragma unroll
    for (int i = 0; i < TW; ++i) {
        if (i >= nt) break;
        const int k = p + 2 * i;
        const float sum = warp_sum(acc[i]);
        if (lane != 0) continue;
        if (b < ND - 1) {
            const float x = sum / (float) HC;
            a.lo[(size_t) k * LR + b * ROWS + r] = x / (1.0f + __expf(-x));
        } else {
            a.inject_out[k * HC + r] = sum;
        }
    }
}

// The up CTA u of nu: columns [c0, c1) of every stream; its w_up rows [4][nc][40] uint4 in shared memory.
struct UpSlice {
    int c0, nc;
};
__device__ __forceinline__ UpSlice up_slice(int u, int nu) {
    const int c0 = u * N / nu, c1 = (u + 1) * N / nu;
    return {c0, c1 - c0};
}
__device__ void hc_up_issue(const HcFusedArgs& a, UpSlice sl, char* smem, uint64_t* bar) {
    const unsigned row_bytes = LR * 2, bytes = (unsigned) sl.nc * row_bytes;
    mbar_expect_tx(bar, 4 * bytes);
    for (int c = 0; c < HC; ++c)
        bulk_g2s(smem + (size_t) c * bytes, a.w_up + ((size_t) c * N + sl.c0) * LR, bytes, bar);
}

// Shared memory of an up CTA: [w_up rows 4 nc x 640 B][lo T x LR][rp T x 4 x nc: R'][px T x 4 x nc: R' * w_norm]
// [g T x 4 x nc]
struct UpSmem {
    float *lo, *rp, *px, *g;
};
__device__ __forceinline__ UpSmem up_smem(char* smem, int nc, int T) {
    UpSmem u;
    u.lo = reinterpret_cast<float*>(smem + (size_t) 4 * nc * LR * 2);
    u.rp = u.lo + T * LR;
    u.px = u.rp + T * HC * nc;
    u.g = u.px + T * HC * nc;
    return u;
}

// Before the down projection is done: R' (fused_gr.cu gr_up_multi_kernel's fmaf) and R' * w_norm for the slice.
template <int T>
__device__ void hc_up_pre(const HcFusedArgs& a, UpSlice sl, int tid, char* smem) {
    const int nc = sl.nc;
    const UpSmem u = up_smem(smem, nc, T);
    for (int idx = tid; idx < T * HC * nc; idx += THREADS) {
        const int k = idx / (HC * nc), rem = idx - k * HC * nc, c = rem / nc, dd = rem - c * nc;
        const int i = c * N + sl.c0 + dd;
        float rv = a.R[(size_t) k * D + i];
        if (a.apply) rv = fmaf(a.bo_prev[(size_t) k * N + sl.c0 + dd], 2.0f * sigmoidf_(a.inj_prev[k * HC + c] / (float) HC), rv);
        u.rp[idx] = rv;
        u.px[idx] = rv * a.w_norm[i];
    }
}

// The 32-lane xor tree of `warp_sum`, for 16 values at once: four halving steps (offsets 16, 8, 4, 2) each send half
// of the values to the partner lane and add the other half, then one butterfly step (offset 1).  Every value is summed
// over exactly the pairs, in exactly the tree, of warp_sum (and a + b == b + a), so each total has warp_sum's bits;
// lanes 2m and 2m + 1 end with value m.
__device__ __forceinline__ float warp_sum16(float (&v)[16], int lane) {
#pragma unroll
    for (int o = 16, n = 8; o >= 2; o >>= 1, n >>= 1) {
        const bool h = (lane & o) != 0;
#pragma unroll
        for (int j = 0; j < n; ++j) {
            const float keep = h ? v[j + n] : v[j], send = h ? v[j] : v[j + n];
            v[j] = keep + __shfl_xor_sync(0xffffffffu, send, o);
        }
    }
    return v[0] + __shfl_xor_sync(0xffffffffu, v[0], 1);
}

// Tokens [p0, p0 + P) of the up rows: a warp takes G = 16 / P rows at a time, each lane forms the G x P per-lane sums
// of gr_up_multi_kernel (dot8 of its chunk, plus the second chunk on lanes 0-7), warp_sum16 reduces all of them, and
// lane 2m runs the epilogue of value m.
template <int P>
__device__ __forceinline__ void up_pass(const UpSmem& u, const uint4* U, int nc, int p0, const float (&rs_mine)[HC],
                                        int lane, int warp) {
    constexpr int G = 16 / P;
    float4 l0[P][2], l1[P][2];
    const int j1 = lane < LR / 8 - 32 ? 32 + lane : 32;
#pragma unroll
    for (int q = 0; q < P; ++q) {
        const float4* lk = reinterpret_cast<const float4*>(u.lo + (p0 + q) * LR);
        l0[q][0] = lk[lane * 2]; l0[q][1] = lk[lane * 2 + 1];
        l1[q][0] = lk[j1 * 2]; l1[q][1] = lk[j1 * 2 + 1];
    }
    const int rows = HC * nc, groups = (rows + G - 1) / G;
    const int m = lane >> 1, mg = m / P, mk = m - mg * P;
    for (int gi = warp; gi < groups; gi += WARPS) {
        const int r0 = gi * G;
        float v[16];
#pragma unroll
        for (int g = 0; g < G; ++g) {
            const int r = r0 + g < rows ? r0 + g : rows - 1;
            const uint4 wa = U[r * 40 + lane];
            const uint4 wb = lane < LR / 8 - 32 ? U[r * 40 + 32 + lane] : make_uint4(0, 0, 0, 0);
#pragma unroll
            for (int q = 0; q < P; ++q) {
                float acc = dot8v(wa, l0[q][0], l0[q][1]);
                if (lane < LR / 8 - 32) acc += dot8v(wb, l1[q][0], l1[q][1]);
                v[g * P + q] = acc;
            }
        }
#pragma unroll
        for (int i = G * P; i < 16; ++i) v[i] = 0.0f;
        const float sum = warp_sum16(v, lane);
        const int r = r0 + mg;
        if ((lane & 1) == 0 && mg < G && r < rows) {
            const int c = r / nc, dd = r - c * nc, k = p0 + mk;
            const int idx = (k * HC + c) * nc + dd;
            float rsc = rs_mine[0];
#pragma unroll
            for (int cc = 1; cc < HC; ++cc) rsc = c == cc ? rs_mine[cc] : rsc;
            const float x = u.px[idx] * rsc;
            u.g[idx] = x * sigmoidf_(sum);
        }
    }
}

// fused_gr.cu gr_up_multi_kernel's rows, epilogue and mean for this slice.
template <int T>
__device__ void hc_up_main(const HcFusedArgs& a, UpSlice sl, int tid, char* smem, uint64_t* bar) {
    const int lane = tid & 31, warp = tid >> 5, nc = sl.nc, c0 = sl.c0;
    const UpSmem u = up_smem(smem, nc, T);
    for (int i = tid; i < T * LR; i += THREADS) u.lo[i] = __ldcg(a.lo + i);
    if (a.apply)
        for (int idx = tid; idx < T * HC * nc; idx += THREADS) {
            const int k = idx / (HC * nc), rem = idx - k * HC * nc, c = rem / nc, dd = rem - c * nc;
            a.R_out[(size_t) k * D + c * N + c0 + dd] = u.rp[idx];
        }
    // rs of the token whose epilogue this lane runs: pass one holds tokens 0..P1-1, pass two P1..T-1
    constexpr int P1 = T < 4 ? T : 4, P2 = T - P1;
    float rs1[HC], rs2[HC];
    {
        const int m = lane >> 1;
#pragma unroll
        for (int c = 0; c < HC; ++c) {
            rs1[c] = __ldcg(a.rs + (m % P1) * HC + c);
            rs2[c] = P2 > 0 ? __ldcg(a.rs + (P1 + m % (P2 > 0 ? P2 : 1)) * HC + c) : 0.0f;
        }
    }
    __syncthreads();
    mbar_wait0(bar);
    const uint4* U = reinterpret_cast<const uint4*>(smem);
    up_pass<P1>(u, U, nc, 0, rs1, lane, warp);
    if constexpr (P2 > 0) up_pass<(P2 > 0 ? P2 : 1)>(u, U, nc, P1, rs2, lane, warp);
    __syncthreads();
    for (int i = tid; i < T * nc; i += THREADS) {
        const int k = i / nc, col = i - k * nc;
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += u.g[(k * HC + c) * nc + col];
        a.mixed[(size_t) k * N + c0 + col] = s / (float) HC;
    }
}

// Router rows [e0, e1): bf16_f32_mmvf_multi_kernel<256, *>'s per-row order with the whole CTA on each row.
template <int T>
__device__ void hc_router_rows(const HcFusedArgs& a, int e0, int e1, int tid, float* partials /* [4][T][32] */) {
    const int lane = tid & 31, warp = tid >> 5;
    constexpr int PAIRS = N / 2 / THREADS;   // 5
    for (int eb = e0; eb < e1; eb += 4) {
        const int ne = e1 - eb < 4 ? e1 - eb : 4;
        uint32_t wr[4][PAIRS];
#pragma unroll
        for (int e = 0; e < 4; ++e)
#pragma unroll
            for (int j = 0; j < PAIRS; ++j)
                wr[e][j] = e < ne ? reinterpret_cast<const uint32_t*>(a.w_router + (size_t) (eb + e) * N)[tid + THREADS * j] : 0u;
        float2 x[T][PAIRS];
#pragma unroll
        for (int k = 0; k < T; ++k)
#pragma unroll
            for (int j = 0; j < PAIRS; ++j)
                x[k][j] = __ldcg(reinterpret_cast<const float2*>(a.mixed + (size_t) k * N) + tid + THREADS * j);
        if (tid < 32)
            for (int i = 0; i < 4 * T; ++i) partials[i * 32 + tid] = 0.0f;
        float acc[4][T];
#pragma unroll
        for (int e = 0; e < 4; ++e)
#pragma unroll
            for (int k = 0; k < T; ++k) {
                acc[e][k] = 0.0f;
#pragma unroll
                for (int j = 0; j < PAIRS; ++j) {
                    const float w0 = __uint_as_float(wr[e][j] << 16), w1 = __uint_as_float(wr[e][j] & 0xffff0000u);
                    acc[e][k] = fma_ftz(w0, x[k][j].x, acc[e][k]);
                    acc[e][k] = fma_ftz(w1, x[k][j].y, acc[e][k]);
                }
            }
        __syncthreads();
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            if (e >= ne) break;
#pragma unroll
            for (int k = 0; k < T; ++k) {
                const float v = warp_sum_ftz(acc[e][k]);
                if (lane == 0) partials[(e * T + k) * 32 + warp] = v;
            }
        }
        __syncthreads();
        if (tid < 32) {
            for (int e = 0; e < ne; ++e)
#pragma unroll
                for (int k = 0; k < T; ++k) {
                    const float v = warp_sum_ftz(partials[(e * T + k) * 32 + lane]);
                    if (lane == 0) a.logits[(size_t) k * NE + eb + e] = v;
                }
        }
        __syncthreads();
    }
}

// native_router.cu `route` for token k, on one warp.
__device__ void hc_top10(const HcFusedArgs& a, int k, int lane) {
    const float* logits = a.logits + (size_t) k * NE;
    int32_t* ids = a.ids + (size_t) k * KT;
    float* weights = a.weights + (size_t) k * KT;
    float values[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) values[i] = __ldcg(logits + lane + i * 32);
    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i) maximum = max_ftz(maximum, values[i]);
    maximum = warp_max_ftz(maximum);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] = expf_fast(sub_ftz(values[i], maximum));
        sum = add_ftz(sum, values[i]);
    }
    const float reciprocal = rcp_ftz(warp_sum_ftz(sum));
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] = mul_ftz(values[i], reciprocal);
        if (isnan(values[i])) values[i] = -FLT_MAX;
    }
    float selected = 0.0f, selected_sum = 0.0f;
    unsigned taken = 0;   // bit i: values[i] already selected (now -inf)
#pragma unroll 1
    for (int rank = 0; rank < KT; ++rank) {
        float best = (taken & 1u) ? -INFINITY : values[0];
        int expert = lane;
#pragma unroll
        for (int i = 1; i < 16; ++i) {
            const float v = (taken >> i & 1u) ? -INFINITY : values[i];
            if (v > best) { best = v; expert = lane + i * 32; }
        }
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, best, mask, 32);
            const int other_id = __shfl_xor_sync(0xffffffffu, expert, mask, 32);
            if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
        }
        if ((expert & 31) == lane) {
            taken |= 1u << (expert >> 5);
            ids[rank] = expert;
            selected_sum = add_ftz(selected_sum, best);
        }
        if (rank == lane) selected = best;
    }
    selected_sum = max_ftz(warp_sum_ftz(selected_sum), 6.103515625e-5f);
    const float inverse_selected_sum = rcp_ftz(selected_sum);
    if (lane < KT) weights[lane] = mul_ftz(selected, inverse_selected_sum);
}

// ---------------------------------------------------------------- the kernel
template <int T>
__global__ void __launch_bounds__(THREADS, 1) hc_fused_kernel(HcLaunch L) {
    extern __shared__ __align__(128) char smem[];
    __shared__ __align__(8) uint64_t bar;
    __shared__ float part[WARPS];
    __shared__ float s_rs;
    __shared__ int s_last;
    const HcFusedArgs& a = L.a;
    const int b = blockIdx.x, tid = threadIdx.x, nb = L.nb;
    const int nu = nb - ND;
    unsigned long long* dbg = L.dbg;
    HcSync* sy = &g_hc_sync;
    const bool is_down = b < ND;
    const UpSlice sl = is_down ? UpSlice{0, 0} : up_slice(b - ND, nu);
    const int e0 = b * NE / nb, e1 = (b + 1) * NE / nb;
    // the norm tasks (token, stream, copy of xn) on the first CTAs
    const int ncopy = nb / (4 * T) < NCOPY ? nb / (4 * T) : NCOPY, ntask = 4 * T * ncopy;
    const bool norm_cta = b < ntask;
    stamp(dbg, b, 0, tid);

    // constant data first, before the previous kernel is waited on
    NormRegs nr;
    if (norm_cta) hc_norm_pre(a, b & 3, tid, nr);
    if (is_down) {
        hc_down_prologue<T>(a, b, tid, smem);
    } else if (tid == 0) {
        mbar_init(&bar, 1);
        if (L.up_early) {
            hc_up_issue(a, sl, smem, &bar);
            if (a.w_router) prefetch_l2(a.w_router + (size_t) e0 * N, (size_t) (e1 - e0) * N * 2);
        }
    }
    __syncthreads();
    pdl_wait();
    stamp(dbg, b, 1, tid);

    if (norm_cta) {
        hc_norm_task(a, (b >> 2) % T, b & 3, b / (4 * T), tid, nr, part, &s_rs);
        cta_signal(&sy->norm, tid);
    }
    if (is_down) {
        cta_wait(&sy->norm, (unsigned) ntask, tid);
        stamp(dbg, b, 2, tid);
        hc_down_main<T>(a, b, ncopy, tid, smem);
        cta_signal(&sy->down, tid);
        stamp(dbg, b, 3, tid);
        if (a.w_router && tid == 0 && !L.up_early) prefetch_l2(a.w_router + (size_t) e0 * N, (size_t) (e1 - e0) * N * 2);
    } else {
        hc_up_pre<T>(a, sl, tid, smem);
        if (!L.up_early) {
            cta_wait(&sy->norm, (unsigned) ntask, tid);
            if (tid == 0) {
                hc_up_issue(a, sl, smem, &bar);
                if (a.w_router) prefetch_l2(a.w_router + (size_t) e0 * N, (size_t) (e1 - e0) * N * 2);
            }
        }
        cta_wait(&sy->down, (unsigned) ND, tid);
        stamp(dbg, b, 2, tid);
        hc_up_main<T>(a, sl, tid, smem, &bar);
        cta_signal(&sy->up, tid);
        stamp(dbg, b, 3, tid);
    }

    if (a.w_router) {
        cta_wait(&sy->up, (unsigned) nu, tid);
        stamp(dbg, b, 4, tid);
        hc_router_rows<T>(a, e0, e1, tid, reinterpret_cast<float*>(smem));
        stamp(dbg, b, 5, tid);
        __syncthreads();
        if (tid == 0) {
            __threadfence();
            s_last = atomicAdd(&sy->router, 1u) == (unsigned) nb - 1;
            if (s_last) __threadfence();
        }
        __syncthreads();
        if (s_last && (tid >> 5) < T) hc_top10(a, tid >> 5, tid & 31);
        if (s_last) { __syncthreads(); stamp(dbg, b, 6, tid); }
    }

    // the next read's weights into L2, then the counters back to zero once every CTA is through
    if (tid == 0) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            if (a.pf[r] == nullptr || a.pf_bytes[r] < 16) continue;
            const size_t n16 = a.pf_bytes[r] / 16, lo16 = (size_t) b * n16 / nb, hi16 = (size_t) (b + 1) * n16 / nb;
            if (hi16 > lo16) prefetch_l2((const char*) a.pf[r] + lo16 * 16, (hi16 - lo16) * 16);
        }
    }
    __syncthreads();
    stamp(dbg, b, 7, tid);
    if (tid == 0) {
        __threadfence();
        if (atomicAdd(&sy->exit, 1u) == (unsigned) nb - 1) {
            sy->norm = 0; sy->down = 0; sy->up = 0; sy->router = 0; sy->exit = 0;
            __threadfence();
        }
    }
}

__global__ void hc_l2_prefetch_kernel(const void* p0, size_t n0, const void* p1, size_t n1, const void* p2, size_t n2,
                                      const void* p3, size_t n3) {
    if (threadIdx.x != 0) return;
    const void* p[4] = {p0, p1, p2, p3};
    const size_t n[4] = {n0, n1, n2, n3};
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        if (p[r] == nullptr || n[r] < 16) continue;
        const size_t n16 = n[r] / 16, lo16 = (size_t) blockIdx.x * n16 / gridDim.x, hi16 = (size_t) (blockIdx.x + 1) * n16 / gridDim.x;
        if (hi16 > lo16) prefetch_l2((const char*) p[r] + lo16 * 16, (hi16 - lo16) * 16);
    }
}

using KernelFn = void (*)(HcLaunch);
constexpr KernelFn kKernels[MAXT] = {hc_fused_kernel<1>, hc_fused_kernel<2>, hc_fused_kernel<3>, hc_fused_kernel<4>,
                                     hc_fused_kernel<5>, hc_fused_kernel<6>, hc_fused_kernel<7>, hc_fused_kernel<8>};

int g_up_early = -1;
unsigned long long* g_dbg = nullptr;

struct DevInfo {
    bool checked = false, ok = false;
    int sms = 0;
};
DevInfo g_dev[16];

const DevInfo& dev_info() {
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 16) {
        static DevInfo none;
        return none;
    }
    DevInfo& d = g_dev[dev];
    if (!d.checked) {
        int major = 0, sms = 0, optin = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        d.sms = sms;
        d.ok = major == 12 && sms - ND >= MIN_NU && optin >= SMEM + 1024;
        for (int t = 0; t < MAXT && d.ok; ++t)
            d.ok = cudaFuncSetAttribute(kKernels[t], cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM) == cudaSuccess;
        cudaGetLastError();
        d.checked = true;
    }
    return d;
}

}  // namespace

bool hc_fused_supported() { return dev_info().ok; }

void hc_fused_set_knobs(int up_early) {
    if (up_early >= 0) g_up_early = up_early;
}
void hc_fused_set_debug(unsigned long long* dbg) { g_dbg = dbg; }

void hc_fused_read(const HcFusedArgs& a, void* stream) {
    const DevInfo& d = dev_info();
    if (!d.ok || a.T < 1 || a.T > MAXT || !a.R || !a.R_out || !a.w_norm || !a.w_down || !a.w_up || !a.lo || !a.rs ||
        !a.mixed || !a.xn || (a.w_inject && !a.inject_out) || (a.apply && (!a.bo_prev || !a.inj_prev)) ||
        (a.w_router && (!a.logits || !a.ids || !a.weights)) || (a.apply && a.inj_prev == a.inject_out)) {
        std::fprintf(stderr, "hc_fused_read: invalid arguments or unsupported device\n");
        std::exit(1);
    }
    static const int env_up_early = [] { const char* v = std::getenv("STRATA_DEC_HC_UP_EARLY"); return v ? std::atoi(v) : 1; }();
    HcLaunch L;
    L.a = a;
    L.nb = d.sms;
    L.up_early = g_up_early >= 0 ? g_up_early : env_up_early;
    L.dbg = g_dbg;
    launch_pdl(kKernels[a.T - 1], dim3((unsigned) L.nb), dim3(THREADS), (size_t) SMEM, (cudaStream_t) stream, L);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "hc_fused_read: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

void hc_l2_prefetch(const void* const* ptrs, const size_t* bytes, int n, void* stream) {
    const void* p[4] = {nullptr, nullptr, nullptr, nullptr};
    size_t b[4] = {0, 0, 0, 0};
    for (int i = 0; i < n && i < 4; ++i) { p[i] = ptrs[i]; b[i] = bytes[i]; }
    hc_l2_prefetch_kernel<<<8, 32, 0, (cudaStream_t) stream>>>(p[0], b[0], p[1], b[1], p[2], b[2], p[3], b[3]);
}

}  // namespace strata::kernels
