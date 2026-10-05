// src/kernels/cuda/hc_fused.cu - see include/strata/kernels/hc_fused.hpp.
//
// Built WITHOUT fast math, like fused_gr.cu, so the hc arithmetic below is fused_gr.cu's expressions compiled the
// same way.  The router part reproduces kernels built WITH fast math (the MMVF GEMV and the native top-10): those
// operations are spelled out as the flush-to-zero PTX instructions that build emits.
#include "strata/kernels/hc_fused.hpp"
#include "strata/kernels/pdl.hpp"

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>

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
constexpr int RING_CAP = 3;          // down stages in flight: the weight streams stay just deep enough for full bandwidth
constexpr int UP_PIECES = 8;         // w_up slice copies (stream, half), two in flight
constexpr int RT_MAX_ROWS = (NE + ND - 1) / ND;   // router rows per down CTA
constexpr int MIN_NU = 86;           // up CTAs needed for an up slice (30 columns at most) to fit SMEM
static_assert(D % (QB * 256) == 0, "whole stages");

constexpr int NCOPY = 8;             // copies of xn: the down CTAs read different copies, spreading the L2 load

// each counter in its own 256-byte block: the CTAs polling one do not queue behind the atomics on another
struct alignas(256) HcCtr {
    unsigned v;
};
struct HcSync {
    HcCtr q, norm, down, up, router, exit, nload;
};
// consecutive reads take different counter sets: a read's CTAs claim work before the previous kernel is done
constexpr int NSLOT = 4;
__device__ HcSync g_hc_sync[NSLOT];

struct HcLaunch {
    HcFusedArgs a;
    int nb;          // CTAs (one per SM)
    int slot;        // counter set
    int nu;          // split-K kernel: up slices (even)
    int pf;          // 1 = the down CTAs prefetch the router rows (split-K: and w_up) into L2 after their weights
    int ups;         // 1 = the up rows' second chunks spread over the warp
    float* xc;       // the copies of xn, [NCOPY][MAXT][D]
    int hcx;         // 1 = the top-10 as the last item; 2 = also the router logits from the up slices (see Items)
    int upmma;       // 1 = the up rows on tensor cores (hc_up_main_mma)
    int xh;          // 1 = the copies of xn in fp16 (half the bytes the down CTAs read)
    const unsigned long long* t10_dummy;   // [NE] fixed logits for the top-10's dry runs: 32.32 fixed point, then f32
    float* t10_scratch;                    // [nb][2 * KT] their outputs
    unsigned long long* racc;              // hcx: [MAXT][NE] the logits as 32.32 fixed point, summed over the up slices
    unsigned long long* dbg;   // bench only: per-CTA phase times [nb][16]
    int early;       // 1: the next kernel let in at the start; 2: a first down item stages its weights before the wait
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
__device__ __forceinline__ void mbar_wait(uint64_t* bar, unsigned ph) {
    unsigned done = 0;
    while (!done) {
        asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                     : "=r"(done) : "r"(smem_u32(bar)), "r"(ph) : "memory");
    }
}
__device__ __forceinline__ bool mbar_test(uint64_t* bar, unsigned ph) {
    unsigned done;
    asm volatile("{ .reg .pred p; mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                 : "=r"(done) : "r"(smem_u32(bar)), "r"(ph) : "memory");
    return done != 0;
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
__device__ void hc_norm_task(const HcFusedArgs& a, float* xc, int t, int c, int cp, int tid, const NormRegs& nr,
                             float* part, float* s_rs, unsigned* nload, bool xh = false) {
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
    if (tid == 0) atomicAdd(nload, 1u);   // this task's inputs are in: the weight streams may start
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
        if (xh) {   // fp16 copies (hc_down_main<.., true>)
            const __half2 h0 = __floats2half2_rn(px * rs, py * rs), h1 = __floats2half2_rn(pz * rs, pw * rs);
            uint2 v;
            v.x = *reinterpret_cast<const uint32_t*>(&h0);
            v.y = *reinterpret_cast<const uint32_t*>(&h1);
            *reinterpret_cast<uint2*>(reinterpret_cast<__half*>(xc) + ((size_t) cp * MAXT + t) * D + i) = v;
        } else {
            *reinterpret_cast<float4*>(xc + ((size_t) cp * MAXT + t) * D + i) = make_float4(px * rs, py * rs, pz * rs, pw * rs);
        }
    }
}

// The down CTA's ring: stage s holds chunks [s*128, s*128 + 128) of its four weight rows ([4][128] uint4) and of
// every token's xn as two planes ([T][2][128] float4: floats 0-3, then 4-7 of each chunk).
template <int T, int CAP = RING_CAP, bool XH = false>
struct Ring {
    static constexpr int BYTES = STAGE_W + T * STAGE_X1 / (XH ? 2 : 1);
    static constexpr int S0 = SMEM / BYTES;
    static constexpr int S1 = S0 > NSTAGE ? NSTAGE : S0;
    static constexpr int S = S1 > CAP ? CAP : S1;   // stages in flight
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

// fp16 xn: [T][128] uint4 per stage, eight halves per chunk
template <int T>
__device__ __forceinline__ void down_issue_xh(const __half* xn, int s, char* slot, int tid) {
    uint4* X = reinterpret_cast<uint4*>(slot + STAGE_W);
#pragma unroll
    for (int m = 0; m < (T * 128 + THREADS - 1) / THREADS; ++m) {
        const int i = tid + THREADS * m, k = i >> 7, ch = i & 127;
        if (i < T * 128) cp_async16(X + i, xn + (size_t) k * D + (size_t) s * 1024 + (size_t) ch * 8);
    }
}

// The weights of the first S stages (one cp.async group).
template <int T, int CAP, bool XH = false>
__device__ void hc_down_prologue(const HcFusedArgs& a, int b, int tid, char* ring) {
    constexpr int S = Ring<T, CAP, XH>::S;
    const uint16_t* w = down_rows(a, b);
    if (w == nullptr) return;
#pragma unroll
    for (int s = 0; s < S; ++s) down_issue_w(w, s, ring + s * Ring<T, CAP, XH>::BYTES, tid);
    cp_async_commit();
}

// After the norm: lane L of warp (row r = w & 3, token parity p = w >> 2) sums chunks L + 32 q, q ascending, of row r
// against tokens p, p + 2, ... - the plain read's per-lane order.
template <int T, int CAP, bool XH = false>
__device__ void hc_down_main(const HcFusedArgs& a, const float* xn, int b, int tid, char* ring) {
    constexpr int S = Ring<T, CAP, XH>::S, BYTES = Ring<T, CAP, XH>::BYTES;
    const __half* xh = reinterpret_cast<const __half*>(xn);
    constexpr int TW = (T + 1) / 2;   // tokens of parity 0 (parity 1 has T / 2)
    const uint16_t* w = down_rows(a, b);
    if (w == nullptr) return;
    const int lane = tid & 31, warp = tid >> 5, r = warp & 3, p = warp >> 2;
#pragma unroll
    for (int s = 0; s < S; ++s) {
        if constexpr (XH) down_issue_xh<T>(xh, s, ring + s * BYTES, tid);
        else down_issue_x<T>(xn, s, ring + s * BYTES, tid);
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
        const uint4* XH4 = reinterpret_cast<const uint4*>(slot + STAGE_W) + p * 128;
#pragma unroll
        for (int qq = 0; qq < QB; ++qq) {
            const int j = lane + 32 * qq;
            const uint4 wv = W[j];
#pragma unroll
            for (int i = 0; i < TW; ++i) {
                if (i >= nt) continue;
                if constexpr (XH) {
                    const uint4 h = XH4[i * 256 + j];
                    const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&h.x));
                    const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&h.y));
                    const float2 f2 = __half22float2(*reinterpret_cast<const __half2*>(&h.z));
                    const float2 f3 = __half22float2(*reinterpret_cast<const __half2*>(&h.w));
                    acc[i] += dot8v(wv, make_float4(f0.x, f0.y, f1.x, f1.y), make_float4(f2.x, f2.y, f3.x, f3.y));
                } else {
                    acc[i] += dot8v(wv, X[i * 512 + j], X[i * 512 + 128 + j]);
                }
            }
        }
        __syncthreads();
        if (s + S < NSTAGE) {
            char* dst = ring + (s % S) * BYTES;
            down_issue_w(w, s + S, dst, tid);
            if constexpr (XH) down_issue_xh<T>(xh, s + S, dst, tid);
            else down_issue_x<T>(xn, s + S, dst, tid);
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

// ---- split-K down items: a cluster of two CTAs per group of four rows.  CTA h of the pair owns lanes 16h..16h+15 of
// the plain read's warp, i.e. chunks 16h + l + 32 q (256-byte runs of every row): it forms those lanes' sums in their
// order, CTA 1 hands its 16 sums per (row, token) to CTA 0 through distributed shared memory, and CTA 0 runs the xor
// tree from its first step (offset 16: lane l with lane l + 16) on.
constexpr int SK_QB = 8;                          // q-steps per stage
constexpr int SK_NSTAGE = D / (SK_QB * 32 * 8);   // 5
constexpr int SK_CAP = SK_NSTAGE;                 // every stage in flight when it fits
constexpr int SK_XCH = ROWS * MAXT * 16 * 4;      // the handed-over sums, bytes
static_assert(ROWS * SK_QB * 16 * 16 == STAGE_W && SK_QB * 16 * 8 * 4 == STAGE_X1, "stage sizes as the plain ring");
template <int T>
struct SkRing {
    static constexpr int BYTES = STAGE_W + T * STAGE_X1;
    static constexpr int S0 = SMEM / BYTES;
    static constexpr int S1 = S0 > SK_NSTAGE ? SK_NSTAGE : S0;
    static constexpr int S = S1 > SK_CAP ? SK_CAP : S1;
    static constexpr int XCH = S * BYTES;   // offset of the exchange buffer
    static_assert(XCH + SK_XCH <= SMEM, "exchange buffer after the ring");
};
__device__ __forceinline__ int sk_chunk(int s, int jj, int h) {   // jj = q * 16 + l within the stage
    return (s * SK_QB + (jj >> 4)) * 32 + 16 * h + (jj & 15);
}
__device__ __forceinline__ void sk_issue_w(const uint16_t* w, int s, int h, char* slot, int tid) {
    uint4* W = reinterpret_cast<uint4*>(slot);
#pragma unroll
    for (int m = 0; m < 2; ++m) {
        const int i = tid + THREADS * m, r = i >> 7, jj = i & 127;
        cp_async16(W + i, w + (size_t) r * D + (size_t) sk_chunk(s, jj, h) * 8);
    }
}
template <int T>
__device__ __forceinline__ void sk_issue_x(const float* xn, int s, int h, char* slot, int tid) {
    float4* X = reinterpret_cast<float4*>(slot + STAGE_W);
#pragma unroll
    for (int m = 0; m < T; ++m) {
        const int i = tid + THREADS * m, k = i >> 8, s4 = i & 255, jj = s4 >> 1;
        cp_async16(X + k * 256 + (s4 & 1) * 128 + jj, xn + (size_t) k * D + (size_t) sk_chunk(s, jj, h) * 8 + (s4 & 1) * 4);
    }
}
template <int T>
__device__ void sk_down_prologue(const HcFusedArgs& a, int d, int h, int tid, char* ring) {
    const uint16_t* w = down_rows(a, d);
    if (w == nullptr) return;
#pragma unroll
    for (int s = 0; s < SkRing<T>::S; ++s) sk_issue_w(w, s, h, ring + s * SkRing<T>::BYTES, tid);
    cp_async_commit();
}
// Warp (row r = w & 3, token parity p = w >> 2), lane l: own lane 16h + (l & 15), tokens p + 2 (l >> 4) + 4 i.
template <int T>
__device__ void sk_down_main(const HcFusedArgs& a, const float* xn, int d, int h, int tid, char* ring) {
    constexpr int S = SkRing<T>::S, BYTES = SkRing<T>::BYTES;
    constexpr int TW = (T + 3) / 4;
    namespace cg = cooperative_groups;
    const cg::cluster_group cl = cg::this_cluster();
    const uint16_t* w = down_rows(a, d);
    if (w == nullptr) return;   // both CTAs of the pair
    const int lane = tid & 31, warp = tid >> 5, r = warp & 3, p = warp >> 2, l = lane & 15;
    const int k0 = p + 2 * (lane >> 4);
#pragma unroll
    for (int s = 0; s < S; ++s) {
        sk_issue_x<T>(xn, s, h, ring + s * BYTES, tid);
        cp_async_commit();
    }
    float acc[TW];
#pragma unroll
    for (int i = 0; i < TW; ++i) acc[i] = 0.0f;
#pragma unroll
    for (int s = 0; s < SK_NSTAGE; ++s) {
        cp_async_wait<S - 1>();
        __syncthreads();
        const char* slot = ring + (s % S) * BYTES;
        const uint4* W = reinterpret_cast<const uint4*>(slot) + r * 128;
        const float4* X = reinterpret_cast<const float4*>(slot + STAGE_W);
#pragma unroll
        for (int q = 0; q < SK_QB; ++q) {
            const int j = q * 16 + l;
            const uint4 wv = W[j];
#pragma unroll
            for (int i = 0; i < TW; ++i) {
                const int k = k0 + 4 * i;
                if (k < T) acc[i] += dot8v(wv, X[k * 256 + j], X[k * 256 + 128 + j]);
            }
        }
        __syncthreads();
        if (s + S < SK_NSTAGE) {
            char* dst = ring + (s % S) * BYTES;
            sk_issue_w(w, s + S, h, dst, tid);
            sk_issue_x<T>(xn, s + S, h, dst, tid);
        }
        cp_async_commit();
    }
    float* xch = reinterpret_cast<float*>(ring + SkRing<T>::XCH);   // [ROWS][MAXT][16]
    if (h == 1) {
        float* rx = cl.map_shared_rank(xch, 0);
#pragma unroll
        for (int i = 0; i < TW; ++i) {
            const int k = k0 + 4 * i;
            if (k < T) rx[(r * MAXT + k) * 16 + l] = acc[i];
        }
    }
    cl.sync();
    if (h == 1) return;
#pragma unroll
    for (int i = 0; i < TW; ++i) {
        const int k = k0 + 4 * i;
        float v = acc[i] + (k < T ? xch[(r * MAXT + k) * 16 + l] : 0.0f);
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (l != 0 || k >= T) continue;
        if (d < ND - 1) {
            const float x = v / (float) HC;
            a.lo[(size_t) k * LR + d * ROWS + r] = x / (1.0f + __expf(-x));
        } else {
            a.inject_out[k * HC + r] = v;
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
// Piece p of the slice's w_up rows (stream p / 2, half p % 2) into shared memory, completing on bars[p].
__device__ void hc_up_piece(const HcFusedArgs& a, UpSlice sl, char* smem, uint64_t* bars, int p) {
    const unsigned bytes = (unsigned) sl.nc * LR * 2, half = bytes / 2;
    const int c = p >> 1, h = p & 1;
    mbar_expect_tx(&bars[p], half);
    bulk_g2s(smem + (size_t) c * bytes + h * half, reinterpret_cast<const char*>(a.w_up + ((size_t) c * N + sl.c0) * LR) + h * half,
             half, &bars[p]);
}
// The mma up pass's layout: the w_up rows in groups of UPG, each group followed by 16 bytes, so group q starts on bank
// 4 q and eight rows of eight consecutive groups (what an ldmatrix matrix reads) hit distinct banks.  One bulk copy per
// run of rows in one group and one stream.  Piece p holds rows [nc / 2, nc) of stream p / 2 when p is odd, [0, nc / 2)
// otherwise.
constexpr int UPG = 4;
constexpr int UPGB = UPG * LR * 2 + 16;
__device__ __forceinline__ int up_row_off(int r) { return (r / UPG) * UPGB + (r % UPG) * LR * 2; }
__device__ void hc_up_piece_rows(const HcFusedArgs& a, UpSlice sl, char* smem, uint64_t* bars, int p) {
    const int c = p >> 1, d0 = (p & 1) ? sl.nc / 2 : 0, d1 = (p & 1) ? sl.nc : sl.nc / 2;
    mbar_expect_tx(&bars[p], (unsigned) (d1 - d0) * LR * 2);
    for (int dd = d0; dd < d1;) {
        const int r = c * sl.nc + dd, e = min(d1, dd + UPG - r % UPG);
        bulk_g2s(smem + up_row_off(r), a.w_up + ((size_t) c * N + sl.c0 + dd) * LR, (unsigned) (e - dd) * LR * 2, &bars[p]);
        dd = e;
    }
}
// Tile t's m-index i (0..15) is row UPG (8 (t / 2) + i % 8) + 2 (t % 2) + i / 8: tiles 2u and 2u + 1 share groups
// 8u..8u+7.
__device__ __forceinline__ int up_tile_row(int t, int i) { return UPG * (8 * (t >> 1) + (i & 7)) + 2 * (t & 1) + (i >> 3); }
static_assert(UPG == 4, "two m-tiles per eight groups");
static_assert(30 * UPGB + (8 * (LR + 8) + 3 * 8 * HC * 30 + 8 * 30) * 4 <= SMEM, "an mma up slice of 30 columns, T = 8");
// Thread 0 of an up CTA: the w_up pieces, two in flight (once `nload` reaches `n_in`), while watching for the down
// projection to finish; `ph` is the barriers' phase.
__device__ void hc_up_stream(const HcFusedArgs& a, UpSlice sl, char* smem, uint64_t* bars, const unsigned* nload,
                             unsigned n_in, const unsigned* down, unsigned ph, bool rows = false) {
    auto piece = [&](int p) { if (rows) hc_up_piece_rows(a, sl, smem, bars, p); else hc_up_piece(a, sl, smem, bars, p); };
    while (ld_acquire(nload) < n_in) {}
    piece(0);
    piece(1);
    int issued = 2, landed = 0;
    bool down_ok = false;
    while (landed < UP_PIECES || !down_ok) {
        if (landed < issued && mbar_test(&bars[landed], ph)) {
            ++landed;
            if (issued < UP_PIECES) piece(issued++);
        }
        if (!down_ok) down_ok = ld_acquire(down) >= (unsigned) ND;
    }
    __threadfence();
}

// Shared memory of an up CTA: [w_up rows 4 nc x 640 B][lo T x LR][rp T x 4 x nc: R'][px T x 4 x nc: R' * w_norm]
// [g T x 4 x nc]; mma: w_up rows in padded groups (up_row_off) and lo rows of LOS floats
constexpr int LOS = LR + 8;   // the eight tokens' rows of one k-step on distinct banks per half-warp
struct UpSmem {
    float *lo, *rp, *px, *g;
};
__device__ __forceinline__ UpSmem up_smem(char* smem, int nc, int T, bool mma = false) {
    UpSmem u;
    u.lo = reinterpret_cast<float*>(smem + (size_t) (mma ? 4 * nc / UPG * UPGB : 4 * nc * LR * 2));
    u.rp = u.lo + T * (mma ? LOS : LR);
    u.px = u.rp + T * HC * nc;
    u.g = u.px + T * HC * nc;
    return u;
}

// Before the down projection is done: R' (fused_gr.cu gr_up_multi_kernel's fmaf) and R' * w_norm for the slice.
template <int T>
__device__ void hc_up_pre(const HcFusedArgs& a, UpSlice sl, int tid, char* smem, bool mma = false) {
    const int nc = sl.nc;
    const UpSmem u = up_smem(smem, nc, T, mma);
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
template <int P, bool SPREAD = false>
__device__ __forceinline__ void up_pass(const UpSmem& u, const uint4* U, int nc, int p0, const float (&rs_mine)[HC],
                                        int lane, int warp, unsigned long long* dbg = nullptr, int b = 0) {
    constexpr int G = 16 / P;
    float4 l0[P][2], l1[P][2];
    const int j1 = SPREAD ? 32 + (lane & 7) : lane < LR / 8 - 32 ? 32 + lane : 32;
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
        if constexpr (SPREAD) {
            // the second chunks (32 + L, L < 8) of the G rows spread over the warp: lane l forms rows (l >> 3) + 4 i
            // against chunk 32 + (l & 7); lane L < 8 takes row g's from lane L + 8 (g & 3)
            constexpr int NI = (8 * G + 31) / 32;
            float s2[NI][P];
#pragma unroll
            for (int i = 0; i < NI; ++i) {
                const int g = (lane >> 3) + 4 * i, rr = r0 + g < rows && g < G ? r0 + g : rows - 1;
                const uint4 wb = U[rr * 40 + 32 + (lane & 7)];
#pragma unroll
                for (int q = 0; q < P; ++q) s2[i][q] = dot8v(wb, l1[q][0], l1[q][1]);
            }
#pragma unroll
            for (int g = 0; g < G; ++g) {
                const int r = r0 + g < rows ? r0 + g : rows - 1;
                const uint4 wa = U[r * 40 + lane];
#pragma unroll
                for (int q = 0; q < P; ++q) {
                    float acc = dot8v(wa, l0[q][0], l0[q][1]);
                    const float o = __shfl_sync(0xffffffffu, s2[g >> 2][q], (lane & 7) + 8 * (g & 3));
                    if (lane < LR / 8 - 32) acc += o;
                    v[g * P + q] = acc;
                }
            }
        } else {
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
        if (dbg != nullptr) stamp(dbg, b, gi == warp ? 12 : 13, warp * 32 + lane);
    }
}

// fused_gr.cu gr_up_multi_kernel's rows, epilogue and mean for this slice.
template <int T>
__device__ void hc_up_rout(const HcFusedArgs& a, UpSlice sl, int tid, char* smem, bool mma = false) {
    const int nc = sl.nc, c0 = sl.c0;
    const UpSmem u = up_smem(smem, nc, T, mma);
    for (int idx = tid; idx < T * HC * nc; idx += THREADS) {
        const int k = idx / (HC * nc), rem = idx - k * HC * nc, c = rem / nc, dd = rem - c * nc;
        a.R_out[(size_t) k * D + c * N + c0 + dd] = u.rp[idx];
    }
}
template <int T, bool SPREAD = false>
__device__ void hc_up_main(const HcFusedArgs& a, UpSlice sl, int tid, char* smem, bool rout, unsigned long long* dbg, int b) {
    const int lane = tid & 31, warp = tid >> 5, nc = sl.nc, c0 = sl.c0;
    const UpSmem u = up_smem(smem, nc, T);
    for (int i = tid; i < T * LR; i += THREADS) u.lo[i] = __ldcg(a.lo + i);
    if (a.apply && rout) hc_up_rout<T>(a, sl, tid, smem);
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
    stamp(dbg, b, 9, tid);
    const uint4* U = reinterpret_cast<const uint4*>(smem);
    up_pass<P1, SPREAD>(u, U, nc, 0, rs1, lane, warp, dbg, b);
    if constexpr (P2 > 0) up_pass<(P2 > 0 ? P2 : 1), SPREAD>(u, U, nc, P1, rs2, lane, warp);
    __syncthreads();
    stamp(dbg, b, 4, tid);
    for (int i = tid; i < T * nc; i += THREADS) {
        const int k = i / nc, col = i - k * nc;
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += u.g[(k * HC + c) * nc + col];
        a.mixed[(size_t) k * N + c0 + col] = s / (float) HC;
    }
    stamp(dbg, b, 5, tid);
}

// The up rows on tensor cores: warp w takes the rows of m-tile w (up_tile_row; A, bf16 by ldmatrix) against
// the T tokens' lo as the columns of B, each lo split into bf16 hi + lo parts (two mma per k-step, fp32 accumulate),
// then gr_up_multi_kernel's epilogue per (row, token).  A token's column holds only its own lo, so its sums do not
// depend on the other tokens.
__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void split_bf16(float2 x, uint32_t& hi, uint32_t& lo) {
    const __nv_bfloat162 h = __floats2bfloat162_rn(x.x, x.y);
    const float2 hf = __bfloat1622float2(h);
    const __nv_bfloat162 l = __floats2bfloat162_rn(x.x - hf.x, x.y - hf.y);
    hi = *reinterpret_cast<const uint32_t*>(&h);
    lo = *reinterpret_cast<const uint32_t*>(&l);
}
constexpr int UKS = LR / 16;   // k-steps
template <int T>
__device__ void hc_up_main_mma(const HcFusedArgs& a, UpSlice sl, int tid, char* smem, unsigned long long* dbg, int b) {
    static_assert(T <= 8, "one mma column per token");
    const int lane = tid & 31, warp = tid >> 5, nc = sl.nc, c0 = sl.c0, g = lane >> 2, t4 = lane & 3;
    const UpSmem u = up_smem(smem, nc, T, true);
    for (int i = tid; i < T * LR; i += THREADS) {
        const int k = i / LR;
        u.lo[k * LOS + i - k * LR] = __ldcg(a.lo + i);
    }
    if (a.apply) hc_up_rout<T>(a, sl, tid, smem, true);
    // rs once per CTA: every up CTA reads the same few words, one request per warp and word would queue at one L2 slice
    __shared__ float s_rsv[MAXT * HC];
    if (tid < T * HC) s_rsv[tid] = __ldcg(a.rs + tid);
    __syncthreads();
    float rsr[2][HC];
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int c = 0; c < HC; ++c) rsr[j][c] = 2 * t4 + j < T ? s_rsv[(2 * t4 + j) * HC + c] : 0.0f;
    stamp(dbg, b, 9, tid);
    const int rows = HC * nc;
    if (up_tile_row(warp, 0) < rows) {
        const int ar = min(up_tile_row(warp, lane & 15), rows - 1);
        const unsigned addr = smem_u32(smem + up_row_off(ar) + (lane >> 4) * 16);
        const float* lb = u.lo + (g < T ? g : 0) * LOS + 2 * t4;
        float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};   // two chains: hi and lo parts
#pragma unroll
        for (int s = 0; s < UKS; ++s) {
            uint32_t af[4];
            asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                : "=r"(af[0]), "=r"(af[1]), "=r"(af[2]), "=r"(af[3]) : "r"(addr + s * 32));
            float2 x0 = *reinterpret_cast<const float2*>(lb + 16 * s), x1 = *reinterpret_cast<const float2*>(lb + 16 * s + 8);
            if (g >= T) x0 = x1 = make_float2(0.f, 0.f);
            uint32_t h0, l0, h1, l1;
            split_bf16(x0, h0, l0);
            split_bf16(x1, h1, l1);
            mma_bf16(acc, af, h0, h1);
            mma_bf16(acc2, af, l0, l1);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[j] += acc2[j];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int k = 2 * t4 + (j & 1), r = up_tile_row(warp, g + 8 * (j >> 1));
            if (k >= T || r >= rows) continue;
            const int c = r / nc, dd = r - c * nc, idx = (k * HC + c) * nc + dd;
            float rsc = rsr[j & 1][0];
#pragma unroll
            for (int cc = 1; cc < HC; ++cc) rsc = c == cc ? rsr[j & 1][cc] : rsc;
            const float x = u.px[idx] * rsc;
            u.g[idx] = x * sigmoidf_(acc[j]);
        }
    }
    __syncthreads();
    stamp(dbg, b, 4, tid);
    for (int i = tid; i < T * nc; i += THREADS) {
        const int k = i / nc, col = i - k * nc;
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += u.g[(k * HC + c) * nc + col];
        a.mixed[(size_t) k * N + c0 + col] = s / (float) HC;
    }
    stamp(dbg, b, 5, tid);
}

// warp_sum16 with the flush-to-zero adds of a fast-math build (mmvf_warp_sum)
__device__ __forceinline__ float warp_sum16_ftz(float (&v)[16], int lane) {
#pragma unroll
    for (int o = 16, n = 8; o >= 2; o >>= 1, n >>= 1) {
        const bool h = (lane & o) != 0;
#pragma unroll
        for (int j = 0; j < n; ++j) {
            const float keep = h ? v[j + n] : v[j], send = h ? v[j] : v[j + n];
            v[j] = add_ftz(keep, __shfl_xor_sync(0xffffffffu, send, o));
        }
    }
    return add_ftz(v[0], __shfl_xor_sync(0xffffffffu, v[0], 1));
}

// Router rows [e0, e1) (at most RT_MAX_ROWS, their weights at wsm): bf16_f32_mmvf_multi_kernel<256, *>'s per-row
// order - thread t's pairs t + 256 j, two ordered FMAs each, the warp's xor tree, then the tree over the 8 warp sums
// and 24 zeros - for every row and token at once; the xor trees run 16 values at a time (warp_sum16_ftz).
template <int T, int RMAX = RT_MAX_ROWS>
__device__ void hc_router_rows(const HcFusedArgs& a, const uint16_t* wsm, int e0, int e1, int tid,
                               float* partials /* [RMAX * T][WARPS] */) {
    const int lane = tid & 31, warp = tid >> 5, ne = e1 - e0;
    constexpr int PAIRS = N / 2 / THREADS;   // 5
    constexpr int TG = T < 4 ? T : 4;        // tokens per pass
    constexpr int NV = RMAX * TG;            // values per lane per pass (28 at most: two rounds of 16)
    constexpr int ROUNDS = (NV + 15) / 16;
    static_assert(NV <= 32, "two rounds at most");
    for (int k0 = 0; k0 < T; k0 += TG) {
        float2 x[TG][PAIRS];
#pragma unroll
        for (int q = 0; q < TG; ++q)
#pragma unroll
            for (int j = 0; j < PAIRS; ++j)
                x[q][j] = k0 + q < T ? __ldcg(reinterpret_cast<const float2*>(a.mixed + (size_t) (k0 + q) * N) + tid + THREADS * j)
                                     : make_float2(0.f, 0.f);
        float acc[32];
#pragma unroll
        for (int e = 0; e < RMAX; ++e) {
            uint32_t wr[PAIRS];
#pragma unroll
            for (int j = 0; j < PAIRS; ++j)
                wr[j] = e < ne ? reinterpret_cast<const uint32_t*>(wsm + (size_t) e * N)[tid + THREADS * j] : 0u;
#pragma unroll
            for (int q = 0; q < TG; ++q) {
                float s = 0.0f;
#pragma unroll
                for (int j = 0; j < PAIRS; ++j) {
                    const float w0 = __uint_as_float(wr[j] << 16), w1 = __uint_as_float(wr[j] & 0xffff0000u);
                    s = fma_ftz(w0, x[q][j].x, s);
                    s = fma_ftz(w1, x[q][j].y, s);
                }
                acc[e * TG + q] = s;
            }
        }
#pragma unroll
        for (int i = NV; i < 32; ++i) acc[i] = 0.0f;
        // the warp trees: value m of round r lands on lanes 2m, 2m + 1
#pragma unroll
        for (int r = 0; r < ROUNDS; ++r) {
            float v[16];
#pragma unroll
            for (int i = 0; i < 16; ++i) v[i] = acc[r * 16 + i];
            const float sum = warp_sum16_ftz(v, lane);
            const int m = r * 16 + (lane >> 1);
            if ((lane & 1) == 0 && m < NV) partials[m * WARPS + warp] = sum;
        }
        __syncthreads();
        // the tree over the warps: warp 0, lane l < 8 holds warp l's sum, the rest zeros
        if (warp == 0) {
#pragma unroll
            for (int r = 0; r < ROUNDS; ++r) {
                float v[16];
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const int m = r * 16 + i;
                    v[i] = lane < WARPS && m < NV ? partials[m * WARPS + lane] : 0.0f;
                }
                const float sum = warp_sum16_ftz(v, lane);
                const int m = r * 16 + (lane >> 1), e = m / TG, q = m - e * TG;
                if ((lane & 1) == 0 && m < NV && e < ne && k0 + q < T) a.logits[(size_t) (k0 + q) * NE + e0 + e] = sum;
            }
        }
        __syncthreads();
    }
}

// native_router.cu `route` for token k, on one warp, selecting by candidates instead of ten argmax passes.  The ten
// passes pick, in order, the largest p with ties to the lower expert - a total order - so the same ten come out of:
// theta = the 10th largest of the 32 lane maxima (every expert of the top ten is at least theta), then every expert
// at least theta ranked by counting.  `cand` is shared memory for 2 * 512 + 2 * 10 words.
// `dbg` non-null: stamp the phases.  FIX: the logits come as 32.32 fixed point from `fx` and are written to `logits`
// (when non-null).
template <bool FIX = false>
__device__ void hc_top10(const float* logits, int32_t* ids, float* weights, int lane, uint32_t* cand,
                         unsigned long long* dbg, int b, const unsigned long long* fx = nullptr) {
    float values[16];
    if constexpr (FIX) {
        long long q[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) q[i] = (long long) __ldcg(fx + lane + i * 32);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            values[i] = (float) ((double) q[i] * 0x1p-32);
            if (logits != nullptr) const_cast<float*>(logits)[lane + i * 32] = values[i];
        }
    } else {
#pragma unroll
        for (int i = 0; i < 16; ++i) values[i] = __ldcg(logits + lane + i * 32);
    }
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
    stamp(dbg, b, 12, lane);
    // (v, e) before (v', e') when v > v' or (v == v' and e < e')
    auto before = [](float v, int e, float v2, int e2) { return v > v2 || (v == v2 && e < e2); };
    // the lane's maximum
    float bv = values[0];
    int be = lane;
#pragma unroll
    for (int i = 1; i < 16; ++i)
        if (values[i] > bv) { bv = values[i]; be = lane + 32 * i; }
    // theta: the lane maximum with exactly nine lane maxima before it
    int rank = 0;
#pragma unroll
    for (int m = 0; m < 32; ++m) {
        const float ov = __shfl_sync(0xffffffffu, bv, m);
        const int oe = __shfl_sync(0xffffffffu, be, m);
        rank += before(ov, oe, bv, be) ? 1 : 0;
    }
    const unsigned at9 = __ballot_sync(0xffffffffu, rank == KT - 1);
    const int src = __ffs(at9) - 1;
    const float tv = __shfl_sync(0xffffffffu, bv, src);
    const int te = __shfl_sync(0xffffffffu, be, src);
    // the candidates (not before theta is false), compacted into shared memory as (value bits, expert)
    unsigned mine = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i)
        if (!before(tv, te, values[i], lane + 32 * i)) mine |= 1u << i;
    const int cnt = __popc(mine);
    int off = cnt;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, off, o);
        if (lane >= o) off += y;
    }
    const int total = __shfl_sync(0xffffffffu, off, 31);
    off -= cnt;
#pragma unroll
    for (int i = 0; i < 16; ++i)
        if (mine >> i & 1u) {
            cand[2 * off] = __float_as_uint(values[i]);
            cand[2 * off + 1] = (uint32_t) (lane + 32 * i);
            ++off;
        }
    __syncwarp();
    stamp(dbg, b, 13, lane);
    // rank every candidate; ranks 0..9 are the selection in order
    float* top_v = reinterpret_cast<float*>(cand + 2 * NE);
    int* top_e = reinterpret_cast<int*>(cand + 2 * NE + KT);
    for (int c = lane; c < total; c += 32) {
        const float v = __uint_as_float(cand[2 * c]);
        const int e = (int) cand[2 * c + 1];
        int r = 0;
        for (int d = 0; d < total; ++d) r += before(__uint_as_float(cand[2 * d]), (int) cand[2 * d + 1], v, e) ? 1 : 0;
        if (r < KT) { top_v[r] = v; top_e[r] = e; }
    }
    __syncwarp();
    stamp(dbg, b, 14, lane);
    // route's sums: lane L adds, in rank order, the selected values of its experts (expert & 31 == L)
    float selected_sum = 0.0f;
#pragma unroll
    for (int r = 0; r < KT; ++r)
        if ((top_e[r] & 31) == lane) selected_sum = add_ftz(selected_sum, top_v[r]);
    const float selected = lane < KT ? top_v[lane] : 0.0f;
    const int sel_e = lane < KT ? top_e[lane] : 0;
    selected_sum = max_ftz(warp_sum_ftz(selected_sum), 6.103515625e-5f);
    const float inverse_selected_sum = rcp_ftz(selected_sum);
    if (lane < KT) {
        ids[lane] = sel_e;
        weights[lane] = mul_ftz(selected, inverse_selected_sum);
    }
}

// One copy of the top-10's code for the hcx item and its dry runs, so the dry runs load the code the item runs.
__device__ __noinline__ void hc_top10_item(const float* logits, int32_t* ids, float* weights, int lane, uint32_t* cand,
                                           unsigned long long* dbg, int b) {
    hc_top10(logits, ids, weights, lane, cand, dbg, b);
}
__device__ __noinline__ void hc_top10_item_fx(const unsigned long long* fx, float* logits, int32_t* ids, float* weights,
                                              int lane, uint32_t* cand, unsigned long long* dbg, int b) {
    hc_top10<true>(logits, ids, weights, lane, cand, dbg, b, fx);
}

// hcx 2: the up slice's share of the router logits.  Half-warp h (16 lanes) takes experts h + 16 i, lane l column
// pair l of the slice, so each load instruction reads two rows' contiguous runs.  Every lane forms its pair's products
// for all its (expert, token) values, a 16-lane halving tree sums them (the same tree for every value), and the lane
// left with a value adds it as 32.32 fixed point - integer adds, so the total is the same whatever order the slices
// land in.
constexpr int RQ = NE / (THREADS / 16);   // experts per half-warp
__device__ __forceinline__ UpSlice up_slice_even(int u, int nu) {   // column pairs, so the weights load as u32
    const int c0 = 2 * (u * (N / 2) / nu), c1 = 2 * ((u + 1) * (N / 2) / nu);
    return {c0, c1 - c0};
}
__device__ __forceinline__ void hcx_router_load(const HcFusedArgs& a, UpSlice sl, int tid, uint32_t (&rw)[RQ]) {
    const int hw = tid >> 4, l = tid & 15;
    const uint32_t* w = reinterpret_cast<const uint32_t*>(a.w_router) + (sl.c0 >> 1) + l;
    const bool in = 2 * l < sl.nc;
#pragma unroll
    for (int i = 0; i < RQ; ++i) rw[i] = in ? __ldg(w + (size_t) (hw + 16 * i) * (N / 2)) : 0u;
}
template <int T>
__device__ void hcx_router_part(const UpSlice sl, const uint32_t (&rw)[RQ], const float* mix, int tid,
                                unsigned long long* racc) {
    const int hw = tid >> 4, l = tid & 15;
    float2 m[T];
#pragma unroll
    for (int k = 0; k < T; ++k)
        m[k] = 2 * l < sl.nc ? make_float2(mix[k * sl.nc + 2 * l], mix[k * sl.nc + 2 * l + 1]) : make_float2(0.f, 0.f);
#pragma unroll
    for (int bt = 0; bt < RQ * T / 16; ++bt) {
        float v[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int q = bt * 16 + j, i = q / T, k = q % T;
            v[j] = fmaf(__uint_as_float(rw[i] & 0xffff0000u), m[k].y, __uint_as_float(rw[i] << 16) * m[k].x);
        }
#pragma unroll
        for (int o = 8, n = 8; o >= 1; o >>= 1, n >>= 1) {
            const bool h = (l & o) != 0;
#pragma unroll
            for (int j = 0; j < n; ++j) {
                const float keep = h ? v[j + n] : v[j], send = h ? v[j] : v[j + n];
                v[j] = keep + __shfl_xor_sync(0xffffffffu, send, o);
            }
        }
        const int q = bt * 16 + l, i = q / T, k = q % T;
        const long long fx = __float2ll_rn(v[0] * 4294967296.0f);
        asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" ::"l"(racc + (size_t) k * NE + hw + 16 * i),
                     "l"((unsigned long long) fx) : "memory");
    }
}

// ---------------------------------------------------------------- the kernel
// The read is a queue of work items in dependency order: the norm tasks, the down rows, the up slices, the router
// rows.  A CTA claims one item at a time (in claim order) and waits only on items before its own, all of them claimed
// by running CTAs, so the read finishes however few of its CTAs are resident.  With every CTA resident the first
// claims split the CTAs into norm, down and up roles; the norm CTAs then take the remaining up (and down) items, the
// down CTAs the router rows.
// With hcx (and a router) the top-10 is one item after the router rows; hcx 2: the up slices form the router logits
// too (no router items) and the top-10 item follows them.
template <int T>
struct Items {
    int ncopy, ntask, nu, down, up, rt, t10, end;
    __device__ Items(int nb, int nu_, bool router, int hcx) {
        ncopy = nb / (4 * T) < NCOPY ? nb / (4 * T) : NCOPY;
        ntask = 4 * T * ncopy;
        nu = nu_ > 0 ? nu_ : nb - ND;
        down = ntask;
        up = down + ND;
        rt = up + nu;
        t10 = rt + (router && hcx != 2 ? ND : 0);
        end = t10 + (router && hcx ? 1 : 0);
    }
};

constexpr int CAND_WORDS = 2 * NE + 2 * KT;   // hc_top10's shared memory per warp
// Shared memory bytes an up item of nc columns uses (w_up rows, then UpSmem).
template <int T>
__device__ __forceinline__ int up_smem_bytes(int nc, bool mma = false) {
    return mma ? 4 * nc / UPG * UPGB + (T * LOS + 3 * T * HC * nc) * 4 : 4 * nc * LR * 2 + (T * LR + 3 * T * HC * nc) * 4;
}

template <int T>
__global__ void __launch_bounds__(THREADS, 1) hc_fused_kernel(HcLaunch L) {
    extern __shared__ __align__(128) char smem[];
    __shared__ __align__(8) uint64_t bars[UP_PIECES + 1];   // the w_up pieces, the router rows
    __shared__ float part[WARPS];
    __shared__ float s_rs;
    __shared__ int s_item, s_last;
    const HcFusedArgs& a = L.a;
    const int b = blockIdx.x, tid = threadIdx.x, nb = L.nb;
    unsigned long long* dbg = L.dbg;
    HcSync* sy = &g_hc_sync[L.slot];
    const Items<T> it(nb, L.nu, a.w_router != nullptr, L.hcx);
    stamp(dbg, b, 0, tid);
    if (tid == 0) {
        for (int p = 0; p <= UP_PIECES; ++p) mbar_init(&bars[p], 1);
        s_item = (int) atomicAdd(&sy->q.v, 1u);
    }
    __syncthreads();
    int item = s_item;
    // constant data before the previous kernel is waited on
    NormRegs nr;
    if (item < it.ntask) hc_norm_pre(a, item & 3, tid, nr);
    bool staged = false;   // the first down item's prologue ran before the wait
    if (L.early) {
        if ((L.early & 2) && item >= it.down && item < it.up) {
            if (L.xh) hc_down_prologue<T, RING_CAP, true>(a, item - it.down, tid, smem);
            else hc_down_prologue<T, RING_CAP>(a, item - it.down, tid, smem);
            staged = true;
        }
        if (L.early & 1) pdl_launch_dependents();
    }
    pdl_wait();
    stamp(dbg, b, 1, tid);
    unsigned up_ph = 0, rt_ph = 0;
    bool first = true;
    while (item < it.end) {
        if (item < it.ntask) {
            const int j = item;
            if (!first) hc_norm_pre(a, j & 3, tid, nr);
            hc_norm_task(a, L.xc, (j >> 2) % T, j & 3, j / (4 * T), tid, nr, part, &s_rs, &sy->nload.v, L.xh != 0);
            if (L.hcx == 2 && it.end > it.t10 && j < T * 4 && (j & 3) == 0)   // the token's logit sums, zero before any up slice adds
                for (int e = tid; e < NE; e += THREADS) L.racc[(size_t) (j >> 2) * NE + e] = 0ull;
            stamp(dbg, b, 10, tid);
            cta_signal(&sy->norm.v, tid);
            stamp(dbg, b, 11, tid);
        } else if (item < it.up) {
            const int d = item - it.down;
            if (dbg != nullptr && tid == 0) dbg[b * 16 + 15] = 1;
            if (!staged) {
                // weight reads queued ahead of the norm's would delay it
                cta_wait(&sy->nload.v, (unsigned) it.ntask, tid);
                if (L.xh) hc_down_prologue<T, RING_CAP, true>(a, d, tid, smem);
                else hc_down_prologue<T, RING_CAP>(a, d, tid, smem);
            }
            if (L.pf == 1 && tid == 0 && a.w_router != nullptr) {   // the router rows into L2 behind the down weights
                const size_t r16 = (size_t) NE * N * 2 / 16, q0 = (size_t) d * r16 / ND, q1 = (size_t) (d + 1) * r16 / ND;
                prefetch_l2(reinterpret_cast<const char*>(a.w_router) + q0 * 16, (q1 - q0) * 16);
            }
            cta_wait(&sy->norm.v, (unsigned) it.ntask, tid);
            stamp(dbg, b, 2, tid);
            if (L.xh) hc_down_main<T, RING_CAP, true>(a, L.xc + (size_t) (d % it.ncopy) * MAXT * D / 2, d, tid, smem);
            else hc_down_main<T, RING_CAP>(a, L.xc + (size_t) (d % it.ncopy) * MAXT * D, d, tid, smem);
            cp_async_wait<0>();
            cta_signal(&sy->down.v, tid);
            stamp(dbg, b, 3, tid);
        } else if (item < it.rt) {
            const bool rx = L.hcx == 2 && it.end > it.t10;   // this slice's share of the router logits too
            const UpSlice sl = rx ? up_slice_even(item - it.up, it.nu) : up_slice(item - it.up, it.nu);
            if (dbg != nullptr && tid == 0) dbg[b * 16 + 15] = 2;
            const bool mma = L.upmma != 0;
            hc_up_pre<T>(a, sl, tid, smem, mma);
            if (tid == 0) {
                hc_up_stream(a, sl, smem, bars, &sy->nload.v, (unsigned) it.ntask, &sy->down.v, up_ph, mma);
            } else if (it.end > it.t10) {
                // while the down projection runs, warps 1..3 run the top-10 on fixed logits: its code is then in this
                // SM's instruction caches if this CTA claims the top-10 item
                const int warp = tid >> 5, off = (up_smem_bytes<T>(sl.nc, mma) + 15) & ~15;
                int32_t* ids = reinterpret_cast<int32_t*>(L.t10_scratch + (size_t) b * 2 * KT);
                uint32_t* cand = reinterpret_cast<uint32_t*>(smem + off) + (warp - 1) * CAND_WORDS;
                if (warp >= 1 && warp <= 3 && off + warp * CAND_WORDS * 4 <= SMEM) {
                    if (L.hcx == 2)
                        hc_top10_item_fx(L.t10_dummy, nullptr, ids, L.t10_scratch + (size_t) b * 2 * KT + KT, tid & 31, cand,
                                         nullptr, b);
                    else
                        hc_top10_item(reinterpret_cast<const float*>(L.t10_dummy + NE), ids,
                                      L.t10_scratch + (size_t) b * 2 * KT + KT, tid & 31, cand, nullptr, b);
                }
            }
            up_ph ^= 1u;
            __syncthreads();
            stamp(dbg, b, 2, tid);
            uint32_t rw[RQ];   // the router weights stream in while the up pass runs (after the w_up copies)
            if (rx) hcx_router_load(a, sl, tid, rw);
            if (mma) hc_up_main_mma<T>(a, sl, tid, smem, dbg, b);
            else if (L.ups) hc_up_main<T, true>(a, sl, tid, smem, true, dbg, b);
            else hc_up_main<T>(a, sl, tid, smem, true, dbg, b);
            if (rx) {
                // the slice's mixed columns again (hc_up_main's expression) into shared memory, then the logit shares
                const UpSmem u = up_smem(smem, sl.nc, T, mma);
                float* mix = reinterpret_cast<float*>(smem + ((up_smem_bytes<T>(sl.nc, mma) + 15) & ~15));
                for (int i = tid; i < T * sl.nc; i += THREADS) {
                    const int k = i / sl.nc, col = i - k * sl.nc;
                    float s = 0.0f;
#pragma unroll
                    for (int c = 0; c < HC; ++c) s += u.g[(k * HC + c) * sl.nc + col];
                    mix[i] = s / (float) HC;
                }
                __syncthreads();
                hcx_router_part<T>(sl, rw, mix, tid, L.racc);
                stamp(dbg, b, 5, tid);
            }
            cta_signal(&sy->up.v, tid);
            stamp(dbg, b, 3, tid);
        } else if (item >= it.t10) {
            // the top-10 once every router row (hcx 2: up slice) has added its logits: warp k takes token k
            if (dbg != nullptr && tid == 0) dbg[b * 16 + 15] = 3;
            if (L.hcx == 2) cta_wait(&sy->up.v, (unsigned) it.nu, tid);
            else cta_wait(&sy->router.v, (unsigned) ND, tid);
            stamp(dbg, b, 8, tid);
            const int k = tid >> 5;
            uint32_t* cand = reinterpret_cast<uint32_t*>(smem) + k * CAND_WORDS;
            if (k < T && L.hcx == 2)
                hc_top10_item_fx(L.racc + (size_t) k * NE, a.logits + (size_t) k * NE, a.ids + (size_t) k * KT,
                                 a.weights + (size_t) k * KT, tid & 31, cand, tid < 32 ? dbg : nullptr, b);
            else if (k < T)
                hc_top10_item(a.logits + (size_t) k * NE, a.ids + (size_t) k * KT, a.weights + (size_t) k * KT, tid & 31,
                              cand, tid < 32 ? dbg : nullptr, b);
            __syncthreads();
            stamp(dbg, b, 6, tid);
        } else {
            const int r = item - it.rt, e0 = r * NE / ND, e1 = (r + 1) * NE / ND;
            char* const rt_p = smem + RT_MAX_ROWS * N * 2;
            if (tid == 0) {   // the rows into shared memory while the up projection runs
                const unsigned bytes = (unsigned) (e1 - e0) * N * 2;
                mbar_expect_tx(&bars[UP_PIECES], bytes);
                bulk_g2s(smem, a.w_router + (size_t) e0 * N, bytes, &bars[UP_PIECES]);
            }
            cta_wait(&sy->up.v, (unsigned) it.nu, tid);
            mbar_wait(&bars[UP_PIECES], rt_ph);
            rt_ph ^= 1u;
            stamp(dbg, b, 4, tid);
            hc_router_rows<T>(a, reinterpret_cast<const uint16_t*>(smem), e0, e1, tid, reinterpret_cast<float*>(rt_p));
            stamp(dbg, b, 5, tid);
            if (it.end > it.t10) {   // the top-10 item waits on the count
                cta_signal(&sy->router.v, tid);
                if (tid == 0) s_last = 0;
            } else {
                __syncthreads();
                if (tid == 0) {
                    __threadfence();
                    s_last = atomicAdd(&sy->router.v, 1u) == (unsigned) ND - 1;
                    if (s_last) __threadfence();
                }
            }
            __syncthreads();
            if (s_last) {
                stamp(dbg, b, 8, tid);
                if ((tid >> 5) < T)
                    hc_top10(a.logits + (size_t) (tid >> 5) * NE, a.ids + (size_t) (tid >> 5) * KT,
                             a.weights + (size_t) (tid >> 5) * KT, tid & 31,
                             reinterpret_cast<uint32_t*>(rt_p + RT_MAX_ROWS * MAXT * WARPS * 4) + (tid >> 5) * (2 * NE + 2 * KT),
                             tid < 32 ? dbg : nullptr, b);
                __syncthreads();
                stamp(dbg, b, 6, tid);
            }
        }
        first = false;
        staged = false;
        __syncthreads();
        if (tid == 0) s_item = (int) atomicAdd(&sy->q.v, 1u);
        __syncthreads();
        item = s_item;
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
    stamp(dbg, b, 7, tid);
    if (tid == 0) {
        __threadfence();
        if (atomicAdd(&sy->exit.v, 1u) == (unsigned) nb - 1) {
            sy->q.v = 0; sy->norm.v = 0; sy->down.v = 0; sy->up.v = 0; sy->router.v = 0; sy->nload.v = 0; sy->exit.v = 0;
            __threadfence();
        }
    }
}

// The split-K read: clusters of two CTAs claim tickets in order - a pair of norm tasks, one down item (split-K over
// the pair), a pair of up slices, a pair of router items - so the down rows spread over twice the SMs.
constexpr int SK_NR = 160;                        // router items
constexpr int SK_RMAX = (NE + SK_NR - 1) / SK_NR; // router rows per item
template <int T>
struct SkItems {
    int ncopy, ntask, nu, down, up, rt, end;   // ticket boundaries
    __device__ SkItems(int nb, int nu_, bool router) {
        ncopy = nb / (4 * T) < NCOPY ? nb / (4 * T) : NCOPY;
        ntask = 4 * T * ncopy;
        nu = nu_;
        down = ntask / 2;
        up = down + ND;
        rt = up + nu / 2;
        end = rt + (router ? SK_NR / 2 : 0);
    }
};

template <int T>
__global__ void __launch_bounds__(THREADS, 1) hc_fused_sk_kernel(HcLaunch L) {
    namespace cg = cooperative_groups;
    extern __shared__ __align__(128) char smem[];
    __shared__ __align__(8) uint64_t bars[UP_PIECES + 1];
    __shared__ float part[WARPS];
    __shared__ float s_rs;
    __shared__ int s_tk[2], s_last;
    const cg::cluster_group cl = cg::this_cluster();
    const HcFusedArgs& a = L.a;
    const int b = blockIdx.x, tid = threadIdx.x, nb = L.nb, h = (int) cl.block_rank();
    unsigned long long* dbg = L.dbg;
    HcSync* sy = &g_hc_sync[L.slot];
    const SkItems<T> it(nb, L.nu, a.w_router != nullptr);
    int* const tk_src = h == 0 ? s_tk : cl.map_shared_rank(s_tk, 0);
    stamp(dbg, b, 0, tid);
    if (tid == 0) {
        for (int p = 0; p <= UP_PIECES; ++p) mbar_init(&bars[p], 1);
        if (h == 0) s_tk[0] = (int) atomicAdd(&sy->q.v, 1u);
    }
    cl.sync();
    int par = 0;
    int tk = tk_src[0];
    NormRegs nr;
    if (tk < it.down) hc_norm_pre(a, (2 * tk + h) & 3, tid, nr);
    pdl_wait();
    stamp(dbg, b, 1, tid);
    unsigned up_ph = 0, rt_ph = 0;
    bool first = true;
    while (tk < it.end) {
        if (tk < it.down) {
            const int j = 2 * tk + h;
            if (!first) hc_norm_pre(a, j & 3, tid, nr);
            hc_norm_task(a, L.xc, (j >> 2) % T, j & 3, j / (4 * T), tid, nr, part, &s_rs, &sy->nload.v);
            stamp(dbg, b, 10, tid);
            cta_signal(&sy->norm.v, tid);
            stamp(dbg, b, 11, tid);
        } else if (tk < it.up) {
            const int d = tk - it.down;
            if (dbg != nullptr && tid == 0) dbg[b * 16 + 15] = 1;
            cta_wait(&sy->nload.v, (unsigned) it.ntask, tid);
            sk_down_prologue<T>(a, d, h, tid, smem);
            if (L.pf == 1 && tid == 0) {   // the up and router weights into L2 behind this CTA's down weights
                const int sh = 2 * d + h, nsh = 2 * ND;
                const size_t mat = (size_t) LR * D * 2, rtb = a.w_router != nullptr ? (size_t) NE * N * 2 : 0;
                const size_t m16 = mat / 16, r16 = rtb / 16;
                const size_t u0 = (size_t) sh * m16 / nsh, u1 = (size_t) (sh + 1) * m16 / nsh;
                if (u1 > u0) prefetch_l2(reinterpret_cast<const char*>(a.w_up) + u0 * 16, (u1 - u0) * 16);
                const size_t q0 = (size_t) sh * r16 / nsh, q1 = (size_t) (sh + 1) * r16 / nsh;
                if (q1 > q0) prefetch_l2(reinterpret_cast<const char*>(a.w_router) + q0 * 16, (q1 - q0) * 16);
            }
            cta_wait(&sy->norm.v, (unsigned) it.ntask, tid);
            stamp(dbg, b, 2, tid);
            sk_down_main<T>(a, L.xc + (size_t) (d % it.ncopy) * MAXT * D, d, h, tid, smem);
            cp_async_wait<0>();
            if (h == 0) cta_signal(&sy->down.v, tid);
            stamp(dbg, b, 3, tid);
        } else if (tk < it.rt) {
            const UpSlice sl = up_slice(2 * (tk - it.up) + h, it.nu);
            if (dbg != nullptr && tid == 0) dbg[b * 16 + 15] = 2;
            hc_up_pre<T>(a, sl, tid, smem);
            if (tid == 0) hc_up_stream(a, sl, smem, bars, &sy->nload.v, (unsigned) it.ntask, &sy->down.v, up_ph);
            up_ph ^= 1u;
            __syncthreads();
            stamp(dbg, b, 2, tid);
            if (L.ups) hc_up_main<T, true>(a, sl, tid, smem, true, dbg, b);
            else hc_up_main<T>(a, sl, tid, smem, true, dbg, b);
            cta_signal(&sy->up.v, tid);
            stamp(dbg, b, 3, tid);
        } else {
            const int r = 2 * (tk - it.rt) + h, e0 = r * NE / SK_NR, e1 = (r + 1) * NE / SK_NR;
            char* const rt_p = smem + SK_RMAX * N * 2;
            if (tid == 0) {
                const unsigned bytes = (unsigned) (e1 - e0) * N * 2;
                mbar_expect_tx(&bars[UP_PIECES], bytes);
                bulk_g2s(smem, a.w_router + (size_t) e0 * N, bytes, &bars[UP_PIECES]);
            }
            cta_wait(&sy->up.v, (unsigned) it.nu, tid);
            mbar_wait(&bars[UP_PIECES], rt_ph);
            rt_ph ^= 1u;
            stamp(dbg, b, 4, tid);
            hc_router_rows<T, SK_RMAX>(a, reinterpret_cast<const uint16_t*>(smem), e0, e1, tid, reinterpret_cast<float*>(rt_p));
            stamp(dbg, b, 5, tid);
            __syncthreads();
            if (tid == 0) {
                __threadfence();
                s_last = atomicAdd(&sy->router.v, 1u) == (unsigned) SK_NR - 1;
                if (s_last) __threadfence();
            }
            __syncthreads();
            if (s_last) {
                stamp(dbg, b, 8, tid);
                if ((tid >> 5) < T)
                    hc_top10(a.logits + (size_t) (tid >> 5) * NE, a.ids + (size_t) (tid >> 5) * KT,
                             a.weights + (size_t) (tid >> 5) * KT, tid & 31,
                             reinterpret_cast<uint32_t*>(rt_p + SK_RMAX * MAXT * WARPS * 4) + (tid >> 5) * (2 * NE + 2 * KT),
                             tid < 32 ? dbg : nullptr, b);
                __syncthreads();
                stamp(dbg, b, 6, tid);
            }
        }
        first = false;
        __syncthreads();
        par ^= 1;
        if (h == 0 && tid == 0) s_tk[par] = (int) atomicAdd(&sy->q.v, 1u);
        cl.sync();
        tk = tk_src[par];
    }

    if (tid == 0) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            if (a.pf[r] == nullptr || a.pf_bytes[r] < 16) continue;
            const size_t n16 = a.pf_bytes[r] / 16, lo16 = (size_t) b * n16 / nb, hi16 = (size_t) (b + 1) * n16 / nb;
            if (hi16 > lo16) prefetch_l2((const char*) a.pf[r] + lo16 * 16, (hi16 - lo16) * 16);
        }
    }
    stamp(dbg, b, 7, tid);
    if (tid == 0) {
        __threadfence();
        if (atomicAdd(&sy->exit.v, 1u) == (unsigned) nb - 1) {
            sy->q.v = 0; sy->norm.v = 0; sy->down.v = 0; sy->up.v = 0; sy->router.v = 0; sy->nload.v = 0; sy->exit.v = 0;
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
constexpr KernelFn kSkKernels[MAXT] = {hc_fused_sk_kernel<1>, hc_fused_sk_kernel<2>, hc_fused_sk_kernel<3>,
                                        hc_fused_sk_kernel<4>, hc_fused_sk_kernel<5>, hc_fused_sk_kernel<6>,
                                        hc_fused_sk_kernel<7>, hc_fused_sk_kernel<8>};
// STRATA_DEC_HC_UPS=1: each up row's second chunks formed by the lanes that have none (the same products, fewer
// instructions).  STRATA_DEC_HC_SK=1: the split-K read.  STRATA_DEC_HC_RPF=1: the down CTAs prefetch the router rows (and, split-K,
// w_up) into L2 behind their weights.  STRATA_DEC_HC_NU=<n>: the up slices (the down CTAs take the ones past nb - 81).
// STRATA_DEC_HCX=1: the top-10 as one item after the router rows, claimed by an up CTA that ran its code once while
// waiting (bitwise).  =2: the router logits formed by the up slices instead (fixed-point sums, no router items).
// =3: 1 with the up rows on tensor cores; =4: 2 with the up rows on tensor cores (not bitwise, T-independent).
// STRATA_DEC_HC_XH=1: the norm writes xn in fp16 for the down rows (not bitwise, T-independent).
struct SkConf {
    bool on = false;
    int pf = 0, nu = 0, ups = 0, hcx = 0, upmma = 0, xh = 0;
};
const SkConf& sk_conf() {
    static const SkConf c = [] {
        SkConf k;
        const char* e = std::getenv("STRATA_DEC_HC_SK");
        k.on = e && e[0] == '1';
        e = std::getenv("STRATA_DEC_HC_RPF");
        k.pf = e ? std::atoi(e) : 0;
        e = std::getenv("STRATA_DEC_HC_NU");
        k.nu = e ? std::atoi(e) : 0;
        e = std::getenv("STRATA_DEC_HC_UPS");
        k.ups = e && e[0] == '1' ? 1 : 0;
        e = std::getenv("STRATA_DEC_HCX");
        const int x = e && e[0] >= '1' && e[0] <= '4' ? e[0] - '0' : 0;
        k.hcx = x > 2 ? x - 2 : x;
        k.upmma = x > 2 ? 1 : 0;
        e = std::getenv("STRATA_DEC_HC_XH");
        k.xh = e && e[0] == '1' ? 1 : 0;
        return k;
    }();
    return c;
}

unsigned long long* g_dbg = nullptr;
int g_slot = 0;

struct DevInfo {
    bool checked = false, ok = false;
    int sms = 0;
    float* xc = nullptr;
    unsigned long long* t10 = nullptr;    // [NE] fixed logits (fixed point), [NE] the same as f32 (in NE / 2 words), then
                                          // [sms][2 * KT] float scratch
    unsigned long long* racc = nullptr;   // [NSLOT][MAXT][NE]
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
        d.ok = major == 12 && sms - ND >= MIN_NU && optin >= SMEM + 1024 &&
               cudaMalloc((void**) &d.xc, (size_t) NCOPY * MAXT * D * sizeof(float)) == cudaSuccess &&
               cudaMalloc((void**) &d.t10, (size_t) NE * 12 + (size_t) sms * 2 * KT * 4) == cudaSuccess &&
               cudaMalloc((void**) &d.racc, (size_t) NSLOT * MAXT * NE * 8) == cudaSuccess;
        if (d.ok) {   // distinct logits with a few ties, so the dry run takes the usual path
            std::vector<unsigned long long> v(NE);
            std::vector<float> f(NE);
            for (int e = 0; e < NE; ++e) {
                const double x = ((e * 389) % NE) / 64.0 - 4.0 + (e % 97 == 0 ? 0.0 : 0.001);
                v[e] = (unsigned long long) (long long) (x * 4294967296.0);
                f[e] = (float) x;
            }
            d.ok = cudaMemcpy(d.t10, v.data(), NE * 8, cudaMemcpyHostToDevice) == cudaSuccess &&
                   cudaMemcpy(d.t10 + NE, f.data(), NE * 4, cudaMemcpyHostToDevice) == cudaSuccess;
        }
        for (int t = 0; t < MAXT && d.ok; ++t)
            d.ok = cudaFuncSetAttribute(kKernels[t], cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM) == cudaSuccess;
        for (int t = 0; t < MAXT && d.ok && sk_conf().on; ++t)
            d.ok = cudaFuncSetAttribute(kSkKernels[t], cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM) == cudaSuccess;
        cudaGetLastError();
        d.checked = true;
    }
    return d;
}

}  // namespace

bool hc_fused_supported() { return dev_info().ok; }
void dbg_init(int nb);
void hc_fused_init() {
    const DevInfo& d = dev_info();
    if (d.ok) dbg_init(d.sms);
}

void hc_fused_set_debug(unsigned long long* dbg) { g_dbg = dbg; }

// STRATA_DEC_HC_DBG=<file>: every captured read stamps its CTAs' phases into its own slot; a thread writes the
// median over slots (per T and kind) of each phase's min/max over CTAs, relative to the slot's first CTA start.
constexpr int DBG_SLOTS = 1024;
struct DbgState {
    int dev = -1, nb = 0, n = 0;
    unsigned long long* d = nullptr;
    int meta[DBG_SLOTS];   // T * 2 + router
    std::mutex mu;
};
DbgState g_insitu;
void dbg_dump(const char* path) {
    static const char* const ph[16] = {"start", "dep-wait", "input", "role-done", "rtr-in", "rtr-done", "top10", "exit",
                                       "ticket", "up-lo", "norm-wr", "norm-sig", "t10-smax", "t10-cand", "t10-rank", "role"};
    cudaSetDevice(g_insitu.dev);
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    std::vector<unsigned long long> h;
    for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        int n, nb;
        { std::lock_guard<std::mutex> lk(g_insitu.mu); n = g_insitu.n; nb = g_insitu.nb; }
        h.resize((size_t) n * nb * 16);
        cudaMemcpyAsync(h.data(), g_insitu.d, h.size() * 8, cudaMemcpyDeviceToHost, s);
        cudaStreamSynchronize(s);
        FILE* f = std::fopen(path, "w");
        if (!f) continue;
        for (int key = 2; key <= 2 * MAXT + 1; ++key) {
            // per slot: [phase][grp][min,max]
            std::vector<double> v[16][3][2];
            int cnt = 0;
            for (int sl = 0; sl < n; ++sl) {
                if (g_insitu.meta[sl] != key) continue;
                const unsigned long long* x = h.data() + (size_t) sl * nb * 16;
                unsigned long long t0 = ~0ull;
                for (int b = 0; b < nb; ++b) if (x[b * 16] && x[b * 16] < t0) t0 = x[b * 16];
                if (t0 == ~0ull) continue;
                ++cnt;
                for (int i = 0; i < 16; ++i)
                    for (int g = 0; g < 3; ++g) {
                        double mn = 1e30, mx = -1;
                        for (int b = 0; b < nb; ++b) {
                            if ((x[b * 16 + 15] >= 2 ? (int) x[b * 16 + 15] - 1 : 0) != g) continue;
                            if (i == 15 || !x[b * 16 + i] || x[b * 16 + i] < t0) continue;
                            const double t = (double) (x[b * 16 + i] - t0) / 1e3;
                            mn = std::min(mn, t); mx = std::max(mx, t);
                        }
                        if (mx >= 0) { v[i][g][0].push_back(mn); v[i][g][1].push_back(mx); }
                    }
            }
            if (!cnt) continue;
            std::fprintf(f, "T %d %s: %d reads (median over reads of min/max over CTAs, us)\n", key / 2,
                         key & 1 ? "read+router" : "read", cnt);
            for (int g = 0; g < 3; ++g) {
                bool any = false;
                for (int i = 0; i < 16; ++i) any = any || !v[i][g][0].empty();
                if (!any) continue;
                std::fprintf(f, "  %s:", g == 2 ? "t10 " : g ? "up  " : "down");
                for (int i = 0; i < 16; ++i) {
                    if (v[i][g][0].empty()) continue;
                    double m[2];
                    for (int j = 0; j < 2; ++j) {
                        auto& w = v[i][g][j];
                        std::nth_element(w.begin(), w.begin() + w.size() / 2, w.end());
                        m[j] = w[w.size() / 2];
                    }
                    std::fprintf(f, " %s %.1f/%.1f", ph[i], m[0], m[1]);
                }
                std::fprintf(f, "\n");
            }
        }
        std::fclose(f);
    }
}
void dbg_init(int nb) {   // outside any capture
    const char* path = std::getenv("STRATA_DEC_HC_DBG");
    if (path == nullptr || path[0] == 0 || g_insitu.d != nullptr) return;
    cudaGetDevice(&g_insitu.dev);
    g_insitu.nb = nb;
    if (cudaMalloc((void**) &g_insitu.d, (size_t) DBG_SLOTS * nb * 16 * 8) != cudaSuccess) { g_insitu.d = nullptr; return; }
    cudaMemset(g_insitu.d, 0, (size_t) DBG_SLOTS * nb * 16 * 8);
    cudaDeviceSynchronize();
    std::thread(dbg_dump, path).detach();
}
unsigned long long* dbg_slot(int T, bool router) {
    std::lock_guard<std::mutex> lk(g_insitu.mu);
    if (g_insitu.d == nullptr || g_insitu.n >= DBG_SLOTS) return nullptr;
    g_insitu.meta[g_insitu.n] = T * 2 + (router ? 1 : 0);
    return g_insitu.d + (size_t) (g_insitu.n++) * g_insitu.nb * 16;
}

void hc_fused_read(const HcFusedArgs& a, void* stream) {
    const DevInfo& d = dev_info();
    if (!d.ok || a.T < 1 || a.T > MAXT || !a.R || !a.R_out || !a.w_norm || !a.w_down || !a.w_up || !a.lo || !a.rs ||
        !a.mixed || !a.xn || (a.w_inject && !a.inject_out) || (a.apply && (!a.bo_prev || !a.inj_prev)) ||
        (a.w_router && (!a.logits || !a.ids || !a.weights)) || (a.apply && a.inj_prev == a.inject_out)) {
        std::fprintf(stderr, "hc_fused_read: invalid arguments or unsupported device\n");
        std::exit(1);
    }
    HcLaunch L{};
    L.a = a;
    L.nb = d.sms;
    L.slot = g_slot;
    g_slot = (g_slot + 1) % NSLOT;
    L.xc = d.xc;
    L.dbg = g_dbg;
    L.ups = sk_conf().ups;
    L.hcx = sk_conf().hcx;
    L.upmma = sk_conf().upmma;
    L.xh = sk_conf().xh && !sk_conf().on;
    L.t10_dummy = d.t10;
    L.t10_scratch = reinterpret_cast<float*>(d.t10 + NE) + NE;
    L.racc = d.racc + (size_t) L.slot * MAXT * NE;
    L.early = pdl2_mask() & (a.T <= 4 ? 3 : 2);   // from 5 tokens the next kernel's early blocks slow the read
    if (L.dbg == nullptr) L.dbg = dbg_slot(a.T, a.w_router != nullptr);
    const SkConf& sk = sk_conf();
    if (sk.on) {
        L.nb = d.sms & ~1;
        const int nu = std::max(MIN_NU, sk.nu > 0 ? sk.nu : L.nb - ND);
        L.nu = (nu + 1) & ~1;
        L.pf = sk.pf;
        const KernelFn fn = kSkKernels[a.T - 1];
        cudaLaunchAttribute at[2];
        at[0].id = cudaLaunchAttributeClusterDimension;
        at[0].val.clusterDim.x = 2; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
        int na = 1;
        PdlState& ps = pdl_state();
        bool pdl = false;
        if (ps.on) {
            if (ps.plain > 0) --ps.plain;
            else pdl = pdl_kernel_ok((const void*) fn);
        }
        if (pdl) {
            at[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
            at[1].val.programmaticStreamSerializationAllowed = 1;
            na = 2;
        }
        cudaLaunchConfig_t cfg{};
        cfg.gridDim = dim3((unsigned) L.nb);
        cfg.blockDim = dim3(THREADS);
        cfg.dynamicSmemBytes = (size_t) SMEM;
        cfg.stream = (cudaStream_t) stream;
        cfg.attrs = at;
        cfg.numAttrs = na;
        if (cudaLaunchKernelEx(&cfg, fn, L) == cudaSuccess && pdl) ++pdl_count();
    } else {
        L.nu = sk.nu >= MIN_NU && sk.nu <= L.nb ? sk.nu : 0;
        L.pf = sk.pf;
        launch_pdl(kKernels[a.T - 1], dim3((unsigned) L.nb), dim3(THREADS), (size_t) SMEM, (cudaStream_t) stream, L);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "hc_fused_read: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

namespace {
__global__ void hold_sms_kernel(volatile int* go, int* started) {
    extern __shared__ char hold_smem[];
    if (threadIdx.x == 0) {
        hold_smem[0] = 0;
        atomicAdd(started, 1);
        __threadfence_system();
        while (*go == 0) __nanosleep(1000);
    }
}
}  // namespace
void hc_test_hold_sms(int k, int* go, int* started, void* stream) {
    constexpr int bytes = 64 * 1024;
    cudaFuncSetAttribute(hold_sms_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    hold_sms_kernel<<<k, 32, bytes, (cudaStream_t) stream>>>(go, started);
}

void hc_l2_prefetch(const void* const* ptrs, const size_t* bytes, int n, void* stream) {
    const void* p[4] = {nullptr, nullptr, nullptr, nullptr};
    size_t b[4] = {0, 0, 0, 0};
    for (int i = 0; i < n && i < 4; ++i) { p[i] = ptrs[i]; b[i] = bytes[i]; }
    hc_l2_prefetch_kernel<<<8, 32, 0, (cudaStream_t) stream>>>(p[0], b[0], p[1], b[1], p[2], b[2], p[3], b[3]);
}

}  // namespace strata::kernels
