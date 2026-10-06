// src/kernels/cuda/hc_fused.cu - see include/strata/kernels/hc_fused.hpp.
//
// Built WITHOUT fast math, like fused_gr.cu, so the hc arithmetic below is fused_gr.cu's expressions compiled the
// same way.  The router part reproduces kernels built WITH fast math (the MMVF GEMV and the native top-10): those
// operations are spelled out as the flush-to-zero PTX instructions that build emits.
#include "strata/kernels/hc_fused.hpp"
#include "strata/kernels/pdl.hpp"
#include "strata/kernels/q8_1_finite.hpp"

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <unordered_map>
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

// cn (STRATA_DEC_HC2 & 1): down items over half-streams, no norm tasks
constexpr int CN_HS = 2 * HC;                  // half-streams
constexpr int CN_W = N / 2;                    // elements per half-stream
constexpr int CN_RG = 32;                      // w_down rows per item
constexpr int CN_G = LR / CN_RG;               // row groups
constexpr int CN_ND = CN_G * CN_HS + CN_HS;    // items: w_down's, then w_inject's (one per half-stream)
constexpr int CN_ROWS = LR + HC;               // dot products per (token, half-stream): w_down's rows, then w_inject's
constexpr int CN_SB = CN_W / 32 * 2;           // scale bytes per row and half-stream
constexpr int CN_WB = CN_RG * CN_W;            // weight bytes in shared memory
constexpr int CN_SWB = 4 * CN_SB;              // scale bytes per warp's four rows
constexpr int CN_SWT = 384;                    // tensor copies: a warp's scales on their own 128-byte aligned run
constexpr int CN_XOFF = CN_WB + WARPS * CN_SWT; // x after the weights and scales
constexpr int CN_MIN_NU = N / 32;              // up slices of 32 columns at most
constexpr int CN_MAXCOPY = 8;                  // copies of the sums: the up items read different copies (L2 load spread)
constexpr int CN_PART = MAXT * CN_HS * CN_ROWS;   // floats per copy of the dot products
constexpr int CN_SSN = MAXT * CN_HS;              // floats per copy of the sums of squares
static_assert(LR % CN_RG == 0 && CN_W % 256 == 0 && CN_XOFF % 16 == 0 && CN_SWT >= CN_SWB && CN_SWT % 128 == 0 &&
              CN_WB % 128 == 0 && (4 * CN_W) % 128 == 0, "cn geometry");
static_assert(CN_XOFF + MAXT * CN_W * 4 <= SMEM, "cn item fits");
static_assert(32 * (4 * LR + 16) + 128 + 4 * 32 * 16 * 2 + (MAXT * (LR + 8) + 3 * MAXT * 4 * 32) * 4 + 10240 + 3 * (2 * 512 + 20) * 4 <= SMEM,
              "an 8-bit mma up slice of 32 columns with its B fragments and the top-10's dry run");

// One q8_1 block (iq_kernels.cu's block_q8_1) and its q8_1_store: lane i holds value i of block ib.
struct HcQ81 {
    half2 ds;
    int8_t qs[32];
};
__device__ __forceinline__ void hc_q8_1_store(float xi, HcQ81* __restrict__ y, int ib, int lane) {
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    const float d = q8_1_finite(amax / 127.0f);
    y[ib].qs[lane] = q8_1_quant(xi, d, amax);
    if (lane == 0) y[ib].ds = q8_1_ds(d, sum);
}

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
    int pre8;        // 8-bit read: down stages requested before the norm is done
    int cn;          // 1: down items over half-streams without norm tasks (see cn_rows); | 2: their rows by cp.async;
                     // | 4: the up slices by cp.async
    float* cnp;      // cn: [MAXT][CN_HS][CN_ROWS] the down items' dot products
    float* cns;      // cn: [MAXT][CN_HS] the half-streams' sums of squares
    int bf;          // 1: the 8-bit mma up pass's B fragments formed once per item (hc_up_main_mma8's BF); 2: and HI1
    int cn_upw;      // cn: w_down warps whose rows are in before the up items request w_up (of CN_G * CN_HS * WARPS)
    int cn_copies;   // cn: copies of the sums (up item u reads copy u % cn_copies)
    int tma;         // cn: 1 = each warp's rows by two tensor copies (tm[0..3]); 2 = the item's rows by one thread's
                     // tensor copies (tm[4..5], w_inject's by tm[2..3])
    int up_tma;      // 1: the 8-bit mma up slices by two tensor copies (tm[6..7], hc_up_tma); 2: the first one before
                     // the previous kernel is waited on
    int late_w;      // cn, up_tma: the slice's copy waited on after lo is formed
    int lo_stage;    // cn: an up item's copy of the down items' sums by bulk copies into shared memory (cn_lo's lbar)
    int no_dry;      // cn: the up items skip the top-10 dry run (a down item's CTA claims the top-10 and warms itself)
    CUtensorMap tm[8];   // q_down, s_down, q_inject, s_inject by four rows; q_down, s_down by CN_RG rows; q_up, s_up
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
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, unsigned bytes, uint64_t* bar) {
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
// A 2D tensor copy (box of the map) at element coordinates (x, y) into shared memory, completing on bar.
__device__ __forceinline__ void tma_2d(void* dst, const CUtensorMap* map, int x, int y, uint64_t* bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                 ::"r"(smem_u32(dst)), "l"(reinterpret_cast<uint64_t>(map)), "r"(x), "r"(y), "r"(smem_u32(bar))
                 : "memory");
}
__device__ __forceinline__ void tma_3d(void* dst, const CUtensorMap* map, int x, int y, int z, uint64_t* bar) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
                 ::"r"(smem_u32(dst)), "l"(reinterpret_cast<uint64_t>(map)), "r"(x), "r"(y), "r"(z), "r"(smem_u32(bar))
                 : "memory");
}
__device__ __forceinline__ void tma_4d(void* dst, const CUtensorMap* map, int x, int y, int z, int w, uint64_t* bar) {
    asm volatile("cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5}], [%6];"
                 ::"r"(smem_u32(dst)), "l"(reinterpret_cast<uint64_t>(map)), "r"(x), "r"(y), "r"(z), "r"(w), "r"(smem_u32(bar))
                 : "memory");
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

// 8-bit weights: int8 with one f16 scale per G8 consecutive values of a row
constexpr int G8 = 32;
__device__ __forceinline__ float h2f(uint16_t h) { return __half2float(__ushort_as_half(h)); }
// four int8 as floats, exactly: q + 128 as the low byte of 2^23 + (q + 128), less 2^23 + 128
__device__ __forceinline__ float4 i8x4_f(uint32_t q) {
    const uint32_t u = q ^ 0x80808080u;
    return make_float4(__uint_as_float(__byte_perm(u, 0x4B000000u, 0x7650u)) - 8388736.0f,
                       __uint_as_float(__byte_perm(u, 0x4B000000u, 0x7651u)) - 8388736.0f,
                       __uint_as_float(__byte_perm(u, 0x4B000000u, 0x7652u)) - 8388736.0f,
                       __uint_as_float(__byte_perm(u, 0x4B000000u, 0x7653u)) - 8388736.0f);
}
// bytes 0, 1 (HI: 2, 3) of four int8 as an f16 pair, exactly: 1024 + (q + 128), less 1152
template <bool HI>
__device__ __forceinline__ uint32_t i8x2_h2(uint32_t q) {
    const uint32_t v = __byte_perm(q ^ 0x80808080u, 0x64646464u, HI ? 0x7362u : 0x5140u);
    const __half2 h = __hsub2(*reinterpret_cast<const __half2*>(&v), __halves2half2(__ushort_as_half(0x6480), __ushort_as_half(0x6480)));
    return *reinterpret_cast<const uint32_t*>(&h);
}
__device__ __forceinline__ float dot8ff(const float4 w0, const float4 w1, const float4 x0, const float4 x1) {
    float acc = 0.0f;
    acc = fmaf(w0.x, x0.x, acc);
    acc = fmaf(w0.y, x0.y, acc);
    acc = fmaf(w0.z, x0.z, acc);
    acc = fmaf(w0.w, x0.w, acc);
    acc = fmaf(w1.x, x1.x, acc);
    acc = fmaf(w1.y, x1.y, acc);
    acc = fmaf(w1.z, x1.z, acc);
    acc = fmaf(w1.w, x1.w, acc);
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

// ---- 8-bit down items: the CTA's four rows (every stage) and their scales bulk-copied into shared memory, one
// barrier per stage; the activations through a ring of SX stages behind them.  The lane order is hc_down_main's; each
// lane's chunk of eight lies in one scale group, applied to the chunk's dot product.
constexpr int W8_STAGE = ROWS * 1024;                  // int8 bytes per stage
constexpr int S8_BYTES = ROWS * (D / G8) * 2;          // the rows' scales
constexpr int W8_BYTES = NSTAGE * W8_STAGE + S8_BYTES;
template <int T, bool XH>
struct Ring8 {
    static constexpr int X = T * STAGE_X1 / (XH ? 2 : 1);
    static constexpr int S0 = (SMEM - W8_BYTES) / X;
    static constexpr int SX = S0 > RING_CAP ? RING_CAP : S0;
    static_assert(SX >= 1, "one activation stage at least");
};
struct Down8 {
    const int8_t* q;
    const uint16_t* sc;
};
__device__ __forceinline__ Down8 down_rows8(const HcFusedArgs& a, int b) {
    if (b < ND - 1) return {a.q_down + (size_t) b * ROWS * D, a.s_down + (size_t) b * ROWS * (D / G8)};
    return {a.q_inject, a.s_inject};
}
// Stages [s0, s1) of the rows (stage s completes bars[s]) and, with stage 0, the scales (bars[NSTAGE]); warp r copies
// row r, so every barrier takes ROWS arrivals.
__device__ void hc_down8_issue(const HcFusedArgs& a, int b, int tid, char* smem, uint64_t* bars, int s0, int s1) {
    const Down8 w = down_rows8(a, b);
    if (w.q == nullptr || tid >= ROWS * 32) return;
    const int r = tid >> 5, lane = tid & 31;
    if (lane == 0 && s0 == 0 && s1 > 0) {
        mbar_expect_tx(&bars[NSTAGE], (D / G8) * 2);
        bulk_g2s(smem + NSTAGE * W8_STAGE + r * (D / G8) * 2, w.sc + (size_t) r * (D / G8), (D / G8) * 2, &bars[NSTAGE]);
    }
    for (int st = s0 + lane; st < s1; st += 32) {
        mbar_expect_tx(&bars[st], 1024);
        bulk_g2s(smem + st * W8_STAGE + r * 1024, w.q + (size_t) r * D + st * 1024, 1024, &bars[st]);
    }
}
template <int T, bool XH>
__device__ void hc_down8_main(const HcFusedArgs& a, const float* xn, int b, int tid, char* smem, uint64_t* bars,
                              unsigned ph) {
    constexpr int SX = Ring8<T, XH>::SX, XB = Ring8<T, XH>::X;
    constexpr int TW = (T + 1) / 2;
    if (down_rows8(a, b).q == nullptr) return;
    const __half* xh = reinterpret_cast<const __half*>(xn);
    const int lane = tid & 31, warp = tid >> 5, r = warp & 3, p = warp >> 2;
    char* xr = smem + W8_BYTES - STAGE_W;   // the issue helpers write at slot + STAGE_W
#pragma unroll
    for (int st = 0; st < SX; ++st) {
        if constexpr (XH) down_issue_xh<T>(xh, st, xr + st * XB, tid);
        else down_issue_x<T>(xn, st, xr + st * XB, tid);
        cp_async_commit();
    }
    const int nt = p == 0 ? TW : T / 2;
    float acc[TW];
#pragma unroll
    for (int i = 0; i < TW; ++i) acc[i] = 0.0f;
#pragma unroll
    for (int st = 0; st < NSTAGE; ++st) {
        cp_async_wait<SX - 1>();
        if (st == 0) mbar_wait(&bars[NSTAGE], ph);
        mbar_wait(&bars[st], ph);
        __syncthreads();
        const uint2* W = reinterpret_cast<const uint2*>(smem + st * W8_STAGE + r * 1024);
        const uint16_t* S = reinterpret_cast<const uint16_t*>(smem + NSTAGE * W8_STAGE + r * (D / G8) * 2) + st * 32;
        const char* slot = xr + (st % SX) * XB + STAGE_W;
        const float4* X = reinterpret_cast<const float4*>(slot) + p * 256;
        const uint4* XH4 = reinterpret_cast<const uint4*>(slot) + p * 128;
#pragma unroll
        for (int qq = 0; qq < QB; ++qq) {
            const int j = lane + 32 * qq;
            const uint2 wq = W[j];
            const float4 w0 = i8x4_f(wq.x), w1 = i8x4_f(wq.y);
            const float sc = h2f(S[j >> 2]);
#pragma unroll
            for (int i = 0; i < TW; ++i) {
                if (i >= nt) continue;
                float d;
                if constexpr (XH) {
                    const uint4 h = XH4[i * 256 + j];
                    const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&h.x));
                    const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&h.y));
                    const float2 f2 = __half22float2(*reinterpret_cast<const __half2*>(&h.z));
                    const float2 f3 = __half22float2(*reinterpret_cast<const __half2*>(&h.w));
                    d = dot8ff(w0, w1, make_float4(f0.x, f0.y, f1.x, f1.y), make_float4(f2.x, f2.y, f3.x, f3.y));
                } else {
                    d = dot8ff(w0, w1, X[i * 512 + j], X[i * 512 + 128 + j]);
                }
                acc[i] = fmaf(sc, d, acc[i]);
            }
        }
        __syncthreads();
        if (st + SX < NSTAGE) {
            if constexpr (XH) down_issue_xh<T>(xh, st + SX, xr + (st % SX) * XB, tid);
            else down_issue_x<T>(xn, st + SX, xr + (st % SX) * XB, tid);
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

// The 8-bit mma up pass's B operand, formed once per up item (STRATA_DEC_HC2 & 4): for scale group s, half h and lane
// 4 g + t4, the f16 hi and lo parts of lo[g][32 s + 16 h + 4 t4 .. + 3] as split_f16 forms them (uint4 {hi 0 1, hi 2 3,
// lo 0 1, lo 2 3}; zeros for g >= T).  lo_frag_store writes one value's two halves.
constexpr int BF_BYTES = (LR / 32) * 2 * 32 * 16;
__device__ __forceinline__ void lo_frag_store(float v, int k, int row, uint16_t* bf) {
    const int s = row >> 5, h = (row >> 4) & 1, t4 = (row >> 2) & 3, e = row & 3;
    const __half hi = __float2half_rn(v);
    const __half lo = __float2half_rn(v - __half2float(hi));
    uint16_t* f = bf + (size_t) ((s * 2 + h) * 32 + 4 * k + t4) * 8;
    f[e] = __half_as_ushort(hi);
    f[4 + e] = __half_as_ushort(lo);
}
__device__ __forceinline__ void lo_frag_zero(int T, int tid, uint16_t* bf) {
    for (int i = tid; i < (LR / 32) * 2 * (8 - T) * 4; i += THREADS) {
        const int sh = i / ((8 - T) * 4), l = 4 * T + i % ((8 - T) * 4);
        reinterpret_cast<uint4*>(bf)[sh * 32 + l] = make_uint4(0u, 0u, 0u, 0u);
    }
}

// ---- cn down items: item d < CN_G * CN_HS takes rows [CN_RG (d / CN_HS), +CN_RG) of the 8-bit w_down over half-stream
// h = d % CN_HS (elements [CN_W h, +CN_W)), the later items w_inject's rows over half-stream d - CN_G * CN_HS.  The item
// forms R' = R + bo * gate and x = R' * w_norm of its half-stream itself, so it waits on no other item; it writes the
// dot products with x (rs not applied) and, for the first row group, the half-stream's sum of R'^2.  The up items then
// form lo = silu(sum_c rs_c (dot_2c + dot_2c+1) / hc) (cn_lo).
struct CnRows {
    const int8_t* q;
    const uint16_t* s;
    int row0, nrows, hs;
};
__device__ __forceinline__ CnRows cn_rows(const HcFusedArgs& a, int d) {
    if (d < CN_G * CN_HS) {
        const int g = d / CN_HS;
        return {a.q_down + (size_t) g * CN_RG * D, a.s_down + (size_t) g * CN_RG * (D / G8), g * CN_RG, CN_RG, d % CN_HS};
    }
    return {a.q_inject, a.s_inject, LR, HC, d - CN_G * CN_HS};
}
// Warp w's rows [4 w, 4 w + 4) (weights and scales of the half-stream) into shared memory on bars[w]; warps past the
// rows arrive alone.  False: no rows (no w_inject) or `cp`, the barriers untouched.  `cp`: by the warp's own cp.async
// (cn_dot waits on them) instead of bulk copies.  `tm` (tma 1): two tensor copies per warp, its scales at CN_SWT per
// warp; (tma 2): thread 0 copies every row of a w_down item on bars[0] (cn_dot then waits on bars[0] only).
__device__ bool cn_issue(const HcFusedArgs& a, int d, int tid, char* smem, uint64_t* bars, bool cp, int tma = 0,
                         const CUtensorMap* tm = nullptr) {
    const CnRows w = cn_rows(a, d);
    if (w.q == nullptr) return false;
    const int warp = tid >> 5, lane = tid & 31;
    if (tma == 2 && w.row0 < LR) {
        if (tid == 0) {
            mbar_expect_tx(&bars[0], CN_RG * (CN_W + CN_SB));
            tma_2d(smem, &tm[4], w.hs * (CN_W / 8), w.row0, &bars[0]);
            tma_2d(smem + CN_WB, &tm[5], w.hs * (CN_W / G8), w.row0, &bars[0]);
        }
        return true;
    }
    if (tma) {
        if (4 * warp >= w.nrows) {
            if (lane == 0 && tma == 1) mbar_arrive(&bars[warp]);
            return true;
        }
        if (lane == 0) {
            const CUtensorMap* m = w.row0 < LR ? &tm[0] : &tm[2];
            const int y = (w.row0 < LR ? w.row0 : 0) + 4 * warp;
            uint64_t* bar = &bars[tma == 2 ? 0 : warp];
            mbar_expect_tx(bar, 4 * (CN_W + CN_SB));
            tma_2d(smem + 4 * warp * CN_W, m, w.hs * (CN_W / 8), y, bar);
            tma_2d(smem + CN_WB + warp * (tma == 2 ? CN_SWB : CN_SWT), m + 1, w.hs * (CN_W / G8), y, bar);
        }
        return true;
    }
    if (cp) {
        if (4 * warp >= w.nrows) return false;
        constexpr int RW = CN_W / 16, RS = CN_SB / 16;   // 16-byte pieces per row: weights, scales
#pragma unroll
        for (int i = lane; i < 4 * RW; i += 32) {
            const int r = 4 * warp + i / RW, o = (i % RW) * 16;
            cp_async16(smem + r * CN_W + o, w.q + (size_t) r * D + (size_t) w.hs * CN_W + o);
        }
        if (lane < 4 * RS) {
            const int r = 4 * warp + lane / RS, o = (lane % RS) * 16;
            cp_async16(smem + CN_WB + r * CN_SB + o,
                       reinterpret_cast<const char*>(w.s + (size_t) r * (D / G8) + w.hs * (CN_W / G8)) + o);
        }
        cp_async_commit();
        return false;
    }
    if (4 * warp >= w.nrows) {
        if (lane == 0) mbar_arrive(&bars[warp]);
        return true;
    }
    if (lane == 0) mbar_expect_tx(&bars[warp], 4 * (CN_W + CN_SB));
    __syncwarp();
    if (lane < 4) {
        const int r = 4 * warp + lane;
        bulk_g2s(smem + r * CN_W, w.q + (size_t) r * D + (size_t) w.hs * CN_W, CN_W, &bars[warp]);
        bulk_g2s(smem + CN_WB + r * CN_SB, w.s + (size_t) r * (D / G8) + w.hs * (CN_W / G8), CN_SB, &bars[warp]);
    }
    return true;
}
// x of the half-stream for every token into shared memory as two planes per token ([T][2][CN_W / 8] float4: floats 0-3,
// then 4-7 of each chunk of eight); with `ss`, the half-stream's sum of R'^2 per token (thread order, then the warps in
// order) into cns.
template <int T>
__device__ void cn_x(const HcFusedArgs& a, int hs, int tid, char* smem, float* part, float* cns, bool ss, int ncopy) {
    constexpr int NV = CN_W / 4 / THREADS + 1;   // float4 per thread and token: tid, then THREADS + tid when in range
    const int c = hs >> 1, lane = tid & 31, warp = tid >> 5;
    float4 r[T][NV], bb[T][NV], g[NV];
#pragma unroll
    for (int j = 0; j < NV; ++j) {
        const int e = 4 * (tid + THREADS * j);
        const bool in = e < CN_W;
        g[j] = in ? *reinterpret_cast<const float4*>(a.w_norm + hs * CN_W + e) : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int k = 0; k < T; ++k) {
            r[k][j] = in ? *reinterpret_cast<const float4*>(a.R + (size_t) k * D + hs * CN_W + e) : make_float4(0.f, 0.f, 0.f, 0.f);
            bb[k][j] = in && a.apply ? *reinterpret_cast<const float4*>(a.bo_prev + (size_t) k * N + (hs & 1) * CN_W + e)
                                     : make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }
    float4* X = reinterpret_cast<float4*>(smem + CN_XOFF);
#pragma unroll
    for (int k = 0; k < T; ++k) {
        const float gw = a.apply ? 2.0f * sigmoidf_(a.inj_prev[k * HC + c] / (float) HC) : 0.0f;
        float sq = 0.0f;
#pragma unroll
        for (int j = 0; j < NV; ++j) {
            const int e4 = tid + THREADS * j;
            if (4 * e4 >= CN_W) continue;
            float4 v = r[k][j];
            if (a.apply) {
                v.x = fmaf(bb[k][j].x, gw, v.x); v.y = fmaf(bb[k][j].y, gw, v.y);
                v.z = fmaf(bb[k][j].z, gw, v.z); v.w = fmaf(bb[k][j].w, gw, v.w);
            }
            sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            X[k * (CN_W / 4) + (e4 & 1) * (CN_W / 8) + (e4 >> 1)] = make_float4(v.x * g[j].x, v.y * g[j].y, v.z * g[j].z, v.w * g[j].w);
        }
        if (ss) {
            const float s = warp_sum(sq);
            if (lane == 0) part[k * WARPS + warp] = s;
        }
    }
    __syncthreads();
    if (ss && tid < T) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += part[tid * WARPS + w];
        for (int j = 0; j < ncopy; ++j) cns[j * CN_SSN + tid * CN_HS + hs] = s;
    }
}
// Warp w: rows 4 w .. 4 w + 3 against every token, lane L taking chunks L + 32 q, q ascending (one scale per chunk's dot
// product, as hc_down8_main); the sums of all (row, token) pairs reduced 16 at a time.
// `landed`: one count per warp of a w_down item once its rows are in (the up items' w_up copies wait on it).
// `tma`: cn_issue's mode (the scales' layout and barrier).
template <int T>
__device__ void cn_dot(const CnRows& w, int tid, const char* smem, uint64_t* bars, unsigned ph, float* cnp,
                       unsigned* landed, bool cp, int ncopy, int tma = 0) {
    const int warp = tid >> 5, lane = tid & 31;
    if (4 * warp >= w.nrows) return;
    if (cp) {
        cp_async_wait<0>();
        __syncwarp();
    } else {
        mbar_wait(&bars[tma == 2 ? 0 : warp], ph);
    }
    const int soff = CN_WB + warp * (tma == 1 ? CN_SWT : CN_SWB);
    if (lane == 0 && w.row0 < LR) atomicAdd(landed, 1u);
    const float4* X = reinterpret_cast<const float4*>(smem + CN_XOFF);
    float acc[4 * T];
#pragma unroll
    for (int i = 0; i < 4 * T; ++i) acc[i] = 0.0f;
#pragma unroll
    for (int q = 0; q < CN_W / 256; ++q) {
        const int j = lane + 32 * q;
        float4 x0[T], x1[T];
#pragma unroll
        for (int k = 0; k < T; ++k) {
            x0[k] = X[k * (CN_W / 4) + j];
            x1[k] = X[k * (CN_W / 4) + CN_W / 8 + j];
        }
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = 4 * warp + r;
            const uint2 wq = *reinterpret_cast<const uint2*>(smem + row * CN_W + j * 8);
            const float4 w0 = i8x4_f(wq.x), w1 = i8x4_f(wq.y);
            const float sc = h2f(reinterpret_cast<const uint16_t*>(smem + soff + r * CN_SB)[j >> 2]);
#pragma unroll
            for (int k = 0; k < T; ++k) acc[r * T + k] = fmaf(sc, dot8ff(w0, w1, x0[k], x1[k]), acc[r * T + k]);
        }
    }
    constexpr int ROUNDS = (4 * T + 15) / 16;
#pragma unroll
    for (int rd = 0; rd < ROUNDS; ++rd) {
        float v[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) v[i] = rd * 16 + i < 4 * T ? acc[rd * 16 + i] : 0.0f;
        const float s = warp_sum16(v, lane);
        const int m = rd * 16 + (lane >> 1);
        if ((lane & 1) == 0 && m < 4 * T) {
            const int r = m / T, k = m - r * T;
            for (int j = 0; j < ncopy; ++j) cnp[(size_t) j * CN_PART + ((size_t) k * CN_HS + w.hs) * CN_ROWS + w.row0 + 4 * warp + r] = s;
        }
    }
}
// An up item's lo (into lo_s, row stride ls) and rs (into rs_s) from the down items' sums, as hc_down8_main's
// epilogue; `out`: also the read's lo, rs and inject outputs.
// bf non-null: lo as the mma pass's B fragments (lo_frag_store) instead of lo_s.
template <int T>
// lbar non-null: cnp and cns are a copy in shared memory that completes on lbar (phase lph).
__device__ void cn_lo(const HcFusedArgs& a, const float* cnp, const float* cns, int tid, float* lo_s, int ls,
                      float* rs_s, bool out, uint16_t* bf = nullptr, unsigned long long* dbg = nullptr, int b = 0,
                      uint64_t* lbar = nullptr, unsigned lph = 0) {
    constexpr int NI = (T * LR + THREADS - 1) / THREADS;
    const bool sm = lbar != nullptr;
    const auto ld = [sm](const float* q) { return sm ? *q : __ldcg(q); };
    if (sm) mbar_wait(lbar, lph);
    float p[NI][CN_HS];
#pragma unroll
    for (int i = 0; i < NI; ++i) {
        const int idx = tid + THREADS * i, k = idx / LR, row = idx - k * LR;
#pragma unroll
        for (int h = 0; h < CN_HS; ++h)
            p[i][h] = idx < T * LR ? ld(cnp + ((size_t) k * CN_HS + h) * CN_ROWS + row) : 0.0f;
    }
    if (tid < T * HC) {
        const int k = tid / HC, c = tid - k * HC;
        const float s = ld(cns + k * CN_HS + 2 * c) + ld(cns + k * CN_HS + 2 * c + 1);
        rs_s[tid] = rsqrtf(s / (float) N + a.eps);
    }
    __syncthreads();
    stamp(dbg, b, 12, tid);
#pragma unroll
    for (int i = 0; i < NI; ++i) {
        const int idx = tid + THREADS * i, k = idx / LR, row = idx - k * LR;
        if (idx >= T * LR) continue;
        float v = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) v = fmaf(rs_s[k * HC + c], p[i][2 * c] + p[i][2 * c + 1], v);
        const float x = v / (float) HC;
        const float l = x / (1.0f + __expf(-x));
        if (bf != nullptr) lo_frag_store(l, k, row, bf);
        else lo_s[k * ls + row] = l;
        if (out) a.lo[idx] = l;
    }
    stamp(dbg, b, 14, tid);
    if (out && tid < T * HC) {
        a.rs[tid] = rs_s[tid];
        if (a.inject_out != nullptr && a.q_inject != nullptr) {
            const int k = tid / HC, r = tid - k * HC;
            float v = 0.0f;
#pragma unroll
            for (int c = 0; c < HC; ++c)
                v = fmaf(rs_s[k * HC + c], ld(cnp + ((size_t) k * CN_HS + 2 * c) * CN_ROWS + LR + r) +
                                               ld(cnp + ((size_t) k * CN_HS + 2 * c + 1) * CN_ROWS + LR + r), v);
            a.inject_out[tid] = v;
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
// 8-bit (q_up): into the upper half of the bf16 rows' area, for hc_up_widen; the first half of a stream brings its
// rows' scales (SU8 per row) to `sc`.
constexpr int SU8 = 16;   // f16 per up row's scales (LR / G8 padded: a row's scales are a 16-byte aligned run)
__device__ void hc_up_piece(const HcFusedArgs& a, UpSlice sl, char* smem, uint64_t* bars, int p, uint16_t* sc) {
    const bool q8 = a.q_up != nullptr;
    const unsigned bytes = (unsigned) sl.nc * LR * (q8 ? 1 : 2), half = bytes / 2;
    const int c = p >> 1, h = p & 1;
    const char* src = q8 ? reinterpret_cast<const char*>(a.q_up + ((size_t) c * N + sl.c0) * LR)
                         : reinterpret_cast<const char*>(a.w_up + ((size_t) c * N + sl.c0) * LR);
    char* dst = smem + (q8 ? (size_t) 4 * sl.nc * LR : 0) + (size_t) c * bytes + h * half;
    const unsigned sb = q8 && h == 0 ? (unsigned) sl.nc * SU8 * 2 : 0;
    mbar_expect_tx(&bars[p], half + sb);
    bulk_g2s(dst, src + h * half, half, &bars[p]);
    if (sb) bulk_g2s(sc + c * sl.nc * SU8, a.s_up + ((size_t) c * N + sl.c0) * SU8, sb, &bars[p]);
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
// The 8-bit rows in the same grouping: LR bytes per row, UPGB8 per group (group q again on bank 4 q).
constexpr int UPGB8 = UPG * LR + 16;
__device__ __forceinline__ int up_row_off8(int r) { return (r / UPG) * UPGB8 + (r % UPG) * LR; }
__device__ void hc_up_piece_rows8(const HcFusedArgs& a, UpSlice sl, char* smem, uint64_t* bars, int p, uint16_t* sc) {
    const int c = p >> 1, d0 = (p & 1) ? sl.nc / 2 : 0, d1 = (p & 1) ? sl.nc : sl.nc / 2;
    mbar_expect_tx(&bars[p], (unsigned) (d1 - d0) * (LR + SU8 * 2));
    bulk_g2s(sc + (c * sl.nc + d0) * SU8, a.s_up + ((size_t) c * N + sl.c0 + d0) * SU8, (unsigned) (d1 - d0) * SU8 * 2, &bars[p]);
    for (int dd = d0; dd < d1;) {
        const int r = c * sl.nc + dd, e = min(d1, dd + UPG - r % UPG);
        bulk_g2s(smem + up_row_off8(r), a.q_up + ((size_t) c * N + sl.c0 + dd) * LR, (unsigned) (e - dd) * LR, &bars[p]);
        dd = e;
    }
}
// The 8-bit slice in hc_up_piece_rows8's layout by every thread's cp.async (one commit group).
__device__ void hc_up_cp8(const HcFusedArgs& a, UpSlice sl, char* smem, uint16_t* sc, int tid) {
    constexpr int RP = LR / 16, SP = SU8 * 2 / 16;   // 16-byte pieces per row: weights, scales
    const int rows = HC * sl.nc;
    for (int i = tid; i < rows * (RP + SP); i += THREADS) {
        const int r = i / (RP + SP), j = i - r * (RP + SP), c = r / sl.nc, dd = r - c * sl.nc;
        const size_t gr = (size_t) c * N + sl.c0 + dd;
        if (j < RP) cp_async16(smem + up_row_off8(r) + j * 16, a.q_up + gr * LR + j * 16);
        else cp_async16(reinterpret_cast<char*>(sc + r * SU8) + (j - RP) * 16,
                        reinterpret_cast<const char*>(a.s_up + gr * SU8) + (j - RP) * 16);
    }
    cp_async_commit();
}
// Tile t's m-index i (0..15) is row UPG (8 (t / 2) + i % 8) + 2 (t % 2) + i / 8: tiles 2u and 2u + 1 share groups
// 8u..8u+7.
__device__ __forceinline__ int up_tile_row(int t, int i) { return UPG * (8 * (t >> 1) + (i & 7)) + 2 * (t & 1) + (i >> 3); }
static_assert(UPG == 4, "two m-tiles per eight groups");
static_assert(30 * UPGB + (8 * (LR + 8) + 3 * 8 * HC * 30 + 8 * 30) * 4 <= SMEM, "an mma up slice of 30 columns, T = 8");
// Thread 0 of an up CTA: the w_up pieces, two in flight (the 8-bit slice: all at once) once `nload` reaches `n_in`,
// while watching for the down projection to finish (unless !wait_down); `ph` is the barriers' phase.  `issued` > 0:
// that many pieces were requested already (hc_up_issue8).  `sc`: the 8-bit rows' scales in shared memory.
__device__ __forceinline__ void hc_up_piece_any(const HcFusedArgs& a, UpSlice sl, char* smem, uint64_t* bars, int p,
                                                bool rows, uint16_t* sc) {
    if (rows && a.q_up != nullptr) hc_up_piece_rows8(a, sl, smem, bars, p, sc);
    else if (rows) hc_up_piece_rows(a, sl, smem, bars, p);
    else hc_up_piece(a, sl, smem, bars, p, sc);
}
__device__ void hc_up_issue8(const HcFusedArgs& a, UpSlice sl, char* smem, uint64_t* bars, const unsigned* nload,
                             unsigned n_in, bool rows, uint16_t* sc) {
    while (ld_acquire(nload) < n_in) {}
    for (int p = 0; p < UP_PIECES; ++p) hc_up_piece_any(a, sl, smem, bars, p, rows, sc);
}
__device__ void hc_up_stream(const HcFusedArgs& a, UpSlice sl, char* smem, uint64_t* bars, const unsigned* nload,
                             unsigned n_in, const unsigned* down, unsigned ph, bool rows = false, bool wait_down = true,
                             uint16_t* sc = nullptr, int issued = 0, unsigned nd = ND, unsigned long long* dbg = nullptr,
                             int b = 0) {
    auto piece = [&](int p) { hc_up_piece_any(a, sl, smem, bars, p, rows, sc); };
    int landed = 0;
    if (issued < 0) {   // the slice is loaded otherwise: only the down projection is waited on
        issued = landed = UP_PIECES;
    } else if (issued == 0) {
        while (ld_acquire(nload) < n_in) {}
        piece(0);
        piece(1);
        issued = 2;
    }
    bool down_ok = !wait_down;
    while (landed < UP_PIECES || !down_ok) {
        if (landed < issued && mbar_test(&bars[landed], ph)) {
            ++landed;
            if (issued < UP_PIECES) piece(issued++);
            if (landed == UP_PIECES) stamp(dbg, b, 10, 0);
        }
        if (!down_ok) {
            down_ok = ld_acquire(down) >= nd;
            if (down_ok) stamp(dbg, b, 11, 0);
        }
    }
    __threadfence();
}

// Shared memory of an up CTA: [w_up rows 4 nc x 640 B][lo T x LR][rp T x 4 x nc: R'][px T x 4 x nc: R' * w_norm]
// [g T x 4 x nc][8-bit: the rows' scales 4 nc x SU8 f16, 16-byte aligned]; mma: w_up rows in padded groups
// (up_row_off, 8-bit: up_row_off8) and lo rows of LOS floats
constexpr int LOS = LR + 8;   // the eight tokens' rows of one k-step on distinct banks per half-warp
constexpr int NG8 = LR / G8;  // scale groups per up row
struct UpSmem {
    float *lo, *rp, *px, *g;
    uint16_t* sc;
};
// The 8-bit slice as one tensor copy (STRATA_DEC_HC3 & 4, hc_up_tma): 16-byte k-chunk c of row slot r = 32 stream + col
// at c * UPCM_CH + 16 r (eight consecutive slots are one conflict-free ldmatrix matrix); scales at slot * SU8.
constexpr int UPCM_SL = 32;                    // row slots per stream
constexpr int UPCM_CH = HC * UPCM_SL * 16;     // bytes per k-chunk
constexpr int UPCM = LR / 16 * UPCM_CH;        // the slice's bytes
__device__ __forceinline__ int up_w8_bytes(int nc) { return nc * UPGB8 > UPCM ? nc * UPGB8 : UPCM; }
__device__ __forceinline__ UpSmem up_smem(char* smem, int nc, int T, bool mma = false, bool q8 = false) {
    UpSmem u;
    u.lo = reinterpret_cast<float*>(smem + (size_t) (mma ? (q8 ? up_w8_bytes(nc) : nc * UPGB) : 4 * nc * LR * 2));
    u.rp = u.lo + T * (mma ? LOS : LR);
    u.px = u.rp + T * HC * nc;
    u.g = u.px + T * HC * nc;
    u.sc = reinterpret_cast<uint16_t*>((reinterpret_cast<uintptr_t>(u.g + T * HC * nc) + 127) & ~(uintptr_t) 127);
    return u;
}

// Thread 0: the slice's 8-bit rows (32 slots per stream from column c0; slots past the slice hold other columns or
// zeros) and their scales in up_smem's chunk layout, completing on bar.
__device__ void hc_up_tma(UpSlice sl, char* smem, uint16_t* sc, const CUtensorMap* tm, uint64_t* bar) {
    mbar_expect_tx(bar, UPCM + HC * UPCM_SL * SU8 * 2);
    tma_4d(smem, &tm[0], 0, sl.c0, 0, 0, bar);
    tma_3d(sc, &tm[1], 0, sl.c0, 0, bar);
}

// Before the down projection is done: R' (fused_gr.cu gr_up_multi_kernel's fmaf) and R' * w_norm for the slice.
template <int T>
__device__ void hc_up_pre(const HcFusedArgs& a, UpSlice sl, int tid, char* smem, bool mma = false) {
    const int nc = sl.nc;
    const UpSmem u = up_smem(smem, nc, T, mma, a.q_up != nullptr);
    for (int idx = tid; idx < T * HC * nc; idx += THREADS) {
        const int k = idx / (HC * nc), rem = idx - k * HC * nc, c = rem / nc, dd = rem - c * nc;
        const int i = c * N + sl.c0 + dd;
        float rv = a.R[(size_t) k * D + i];
        if (a.apply) rv = fmaf(a.bo_prev[(size_t) k * N + sl.c0 + dd], 2.0f * sigmoidf_(a.inj_prev[k * HC + c] / (float) HC), rv);
        u.rp[idx] = rv;
        u.px[idx] = rv * a.w_norm[i];
    }
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
__device__ void hc_up_rout(const HcFusedArgs& a, UpSlice sl, int tid, char* smem, bool mma = false, bool q8 = false) {
    const int nc = sl.nc, c0 = sl.c0;
    const UpSmem u = up_smem(smem, nc, T, mma, q8);
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

// The 8-bit up rows on tensor cores: one ldmatrix.x4 of int8 rows covers a whole scale group (32 values) of the tile's
// 16 rows, thread (g, t4) holding bytes 4 t4..4 t4 + 3 of each 16-value half.  Those become f16 pairs (exact), the
// mma's k order permuted alike in A and B: the pair (4 t4, 4 t4 + 1) as k (2 t4, 2 t4 + 1), (4 t4 + 2, 4 t4 + 3) as
// (2 t4 + 8, 2 t4 + 9), so a token's lo loads as one float4 per half.  lo is split into f16 hi + lo parts; each
// group's sums are scaled by their row's scale into the fp32 total.
__device__ __forceinline__ void mma_f16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void split_f16(float a, float b, uint32_t& hi, uint32_t& lo) {
    const __half2 h = __floats2half2_rn(a, b);
    const float2 hf = __half22float2(h);
    const __half2 l = __floats2half2_rn(a - hf.x, b - hf.y);
    hi = *reinterpret_cast<const uint32_t*>(&h);
    lo = *reinterpret_cast<const uint32_t*>(&l);
}
// cnp non-null: lo and rs from the cn down items' sums (cn_lo; `out`: this item writes the read's lo, rs and inject).
// BF (STRATA_DEC_HC2 & 4): the B fragments formed once per item (lo_frag_store, at `bf`) instead of by every warp in
// every k-step; the same values, so the same bits.  HI1 (with BF): lo as its f16 hi part only - half the mma
// (fp32-accumulating f16 mma is the pass's limit).
template <int T, bool BF = false, bool HI1 = false>
__device__ void hc_up_main_mma8(const HcFusedArgs& a, UpSlice sl, int tid, char* smem, unsigned long long* dbg, int b,
                                const float* cnp = nullptr, const float* cns = nullptr, bool out = false,
                                uint16_t* bf = nullptr, bool cm = false, uint64_t* lbar = nullptr, unsigned lph = 0,
                                uint64_t* wbar = nullptr, unsigned wph = 0) {
    static_assert(T <= 8, "one mma column per token");
    const int lane = tid & 31, warp = tid >> 5, nc = sl.nc, c0 = sl.c0, g = lane >> 2, t4 = lane & 3;
    const UpSmem u = up_smem(smem, nc, T, true, true);
    __shared__ float s_rsv8[MAXT * HC];
    if constexpr (BF) lo_frag_zero(T, tid, bf);
    if (cnp != nullptr) {
        cn_lo<T>(a, cnp, cns, tid, u.lo, LOS, s_rsv8, out, BF ? bf : nullptr, dbg, b, lbar, lph);
    } else {
        for (int i = tid; i < T * LR; i += THREADS) {
            const int k = i / LR;
            if constexpr (BF) lo_frag_store(__ldcg(a.lo + i), k, i - k * LR, bf);
            else u.lo[k * LOS + i - k * LR] = __ldcg(a.lo + i);
        }
        if (tid < T * HC) s_rsv8[tid] = __ldcg(a.rs + tid);
    }
    if (a.apply) hc_up_rout<T>(a, sl, tid, smem, true, true);
    __syncthreads();
    float rsr[2][HC];
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int c = 0; c < HC; ++c) rsr[j][c] = 2 * t4 + j < T ? s_rsv8[(2 * t4 + j) * HC + c] : 0.0f;
    if (wbar != nullptr) {   // the slice's copy, waited on only now (lo formed meanwhile)
        mbar_wait(wbar, wph);
        stamp(dbg, b, 10, tid);
    }
    stamp(dbg, b, 9, tid);
    const int rows = HC * nc;
    // cm: the slice in hc_up_tma's layout, warp w on slots 16 w .. 16 w + 15
    const int ks = cm ? 2 * UPCM_CH : 32;   // bytes per k-step
    const auto a_addr = [&]() {
        return cm ? smem_u32(smem + (lane >> 4) * UPCM_CH + (16 * warp + (lane & 15)) * 16)
                  : smem_u32(smem + up_row_off8(min(up_tile_row(warp, lane & 15), rows - 1)) + (lane >> 4) * 16);
    };
    const auto sc_row = [&](int i) { return u.sc + (cm ? 16 * warp + i : min(up_tile_row(warp, i), rows - 1)) * SU8; };
    // the output row of m-index i: false when none
    const auto out_row = [&](int i, int& c, int& dd) {
        if (cm) {
            const int sl8 = 16 * warp + i;
            c = sl8 / UPCM_SL;
            dd = sl8 - c * UPCM_SL;
            return dd < nc;
        }
        const int r = up_tile_row(warp, i);
        c = r / nc;
        dd = r - c * nc;
        return r < rows;
    };
    if (BF && (cm || up_tile_row(warp, 0) < rows)) {
        const unsigned addr = a_addr();
        const uint4* B = reinterpret_cast<const uint4*>(bf) + lane;
        const uint16_t* sc0 = sc_row(g);
        const uint16_t* sc1 = sc_row(g + 8);
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int s = 0; s < NG8; ++s) {
            uint32_t q[4];
            asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                : "=r"(q[0]), "=r"(q[1]), "=r"(q[2]), "=r"(q[3]) : "r"(addr + s * ks));
            float gh[4] = {0.f, 0.f, 0.f, 0.f}, gl[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const uint32_t af[4] = {i8x2_h2<false>(q[2 * h]), i8x2_h2<false>(q[2 * h + 1]), i8x2_h2<true>(q[2 * h]),
                                        i8x2_h2<true>(q[2 * h + 1])};
                const uint4 f = B[(s * 2 + h) * 32];
                mma_f16(gh, af, f.x, f.y);
                if (!HI1) mma_f16(gl, af, f.z, f.w);
            }
            const float s0 = h2f(sc0[s]), s1 = h2f(sc1[s]);
            acc[0] = fmaf(s0, gh[0] + gl[0], acc[0]);
            acc[1] = fmaf(s0, gh[1] + gl[1], acc[1]);
            acc[2] = fmaf(s1, gh[2] + gl[2], acc[2]);
            acc[3] = fmaf(s1, gh[3] + gl[3], acc[3]);
        }
        stamp(dbg, b, 13, tid);   // warp 0's k-steps done
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int k = 2 * t4 + (j & 1);
            int c, dd;
            if (!out_row(g + 8 * (j >> 1), c, dd) || k >= T) continue;
            const int idx = (k * HC + c) * nc + dd;
            float rsc = rsr[j & 1][0];
#pragma unroll
            for (int cc = 1; cc < HC; ++cc) rsc = c == cc ? rsr[j & 1][cc] : rsc;
            const float x = u.px[idx] * rsc;
            u.g[idx] = x * sigmoidf_(acc[j]);
        }
    } else if (!BF && (cm || up_tile_row(warp, 0) < rows)) {
        const unsigned addr = a_addr();
        const float* lb = u.lo + (g < T ? g : 0) * LOS + 4 * t4;
        const uint16_t* sc0 = sc_row(g);
        const uint16_t* sc1 = sc_row(g + 8);
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int s = 0; s < NG8; ++s) {
            uint32_t q[4];
            asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                : "=r"(q[0]), "=r"(q[1]), "=r"(q[2]), "=r"(q[3]) : "r"(addr + s * ks));
            float gh[4] = {0.f, 0.f, 0.f, 0.f}, gl[4] = {0.f, 0.f, 0.f, 0.f};   // the group's hi and lo chains
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const uint32_t af[4] = {i8x2_h2<false>(q[2 * h]), i8x2_h2<false>(q[2 * h + 1]), i8x2_h2<true>(q[2 * h]),
                                        i8x2_h2<true>(q[2 * h + 1])};
                float4 x = *reinterpret_cast<const float4*>(lb + 32 * s + 16 * h);
                if (g >= T) x = make_float4(0.f, 0.f, 0.f, 0.f);
                uint32_t h0, l0, h1, l1;
                split_f16(x.x, x.y, h0, l0);
                split_f16(x.z, x.w, h1, l1);
                mma_f16(gh, af, h0, h1);
                mma_f16(gl, af, l0, l1);
            }
            const float s0 = h2f(sc0[s]), s1 = h2f(sc1[s]);
            acc[0] = fmaf(s0, gh[0] + gl[0], acc[0]);
            acc[1] = fmaf(s0, gh[1] + gl[1], acc[1]);
            acc[2] = fmaf(s1, gh[2] + gl[2], acc[2]);
            acc[3] = fmaf(s1, gh[3] + gl[3], acc[3]);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int k = 2 * t4 + (j & 1);
            int c, dd;
            if (!out_row(g + 8 * (j >> 1), c, dd) || k >= T) continue;
            const int idx = (k * HC + c) * nc + dd;
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

// The 8-bit slice (in the upper half of the bf16 rows' area) widened in place to bf16(q * scale) for hc_up_main, in
// ascending steps: a step's writes reach only int8 bytes that earlier steps (or this one, before its barrier) read.
__device__ __forceinline__ uint32_t bf16x2(float lo, float hi) {
    uint32_t d;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
    return d;
}
__device__ void hc_up_widen(UpSlice sl, int tid, char* smem, const uint16_t* sc) {
    const int B = 4 * sl.nc * LR;
    for (int i0 = 0; i0 < B; i0 += THREADS * 16) {
        const int i = i0 + tid * 16;   // 16 values of one scale group
        uint4 v = make_uint4(0, 0, 0, 0);
        float s = 0.0f;
        if (i < B) {
            v = *reinterpret_cast<const uint4*>(smem + B + i);
            s = h2f(sc[i / LR * SU8 + i % LR / G8]);
        }
        __syncthreads();
        if (i < B) {
            const float4 f0 = i8x4_f(v.x), f1 = i8x4_f(v.y), f2 = i8x4_f(v.z), f3 = i8x4_f(v.w);
            uint4* w = reinterpret_cast<uint4*>(smem + 2 * (size_t) i);
            w[0] = make_uint4(bf16x2(f0.x * s, f0.y * s), bf16x2(f0.z * s, f0.w * s), bf16x2(f1.x * s, f1.y * s),
                              bf16x2(f1.z * s, f1.w * s));
            w[1] = make_uint4(bf16x2(f2.x * s, f2.y * s), bf16x2(f2.z * s, f2.w * s), bf16x2(f3.x * s, f3.y * s),
                              bf16x2(f3.z * s, f3.w * s));
        }
        __syncthreads();
    }
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
    int ncopy, ntask, nd, nu, down, up, rt, t10, end;
    __device__ Items(int nb, int nu_, bool router, int hcx, bool cn = false) {
        ncopy = cn ? 0 : nb / (4 * T) < NCOPY ? nb / (4 * T) : NCOPY;
        ntask = 4 * T * ncopy;
        nd = cn ? CN_ND : ND;
        nu = nu_ > 0 ? nu_ : nb - nd;
        down = ntask;
        up = down + nd;
        rt = up + nu;
        t10 = rt + (router && hcx != 2 ? ND : 0);
        end = t10 + (router && hcx ? 1 : 0);
    }
};

constexpr int CAND_WORDS = 2 * NE + 2 * KT;   // hc_top10's shared memory per warp
// Shared memory bytes an up item of nc columns uses (w_up rows, then UpSmem).
template <int T>
__device__ __forceinline__ int up_smem_bytes(int nc, bool mma = false, bool q8 = false) {
    const int sc = q8 ? 128 + 4 * UPCM_SL * SU8 * 2 : 0;
    return sc + (mma ? (q8 ? up_w8_bytes(nc) : nc * UPGB) + (T * LOS + 3 * T * HC * nc) * 4
                     : 4 * nc * LR * 2 + (T * LR + 3 * T * HC * nc) * 4);
}

// Q8: the matrices from the 8-bit copies (HcFusedArgs::q_*).
template <int T, bool Q8>
__global__ void __launch_bounds__(THREADS, 1) hc_fused_kernel(const __grid_constant__ HcLaunch L) {
    extern __shared__ __align__(128) char smem[];
    __shared__ __align__(8) uint64_t bars[UP_PIECES + 1];   // the w_up pieces, the router rows
    __shared__ __align__(8) uint64_t dbars[Q8 ? NSTAGE + 1 : 1];   // 8-bit down stages, their scales
    __shared__ __align__(8) uint64_t lbar;   // cn: an up item's copy of the down items' sums
    __shared__ float part[WARPS];
    __shared__ float s_ssp[Q8 ? MAXT * WARPS : 1];   // cn: the sums of squares per (token, warp)
    __shared__ float s_rs;
    __shared__ int s_item, s_last;
    const HcFusedArgs& a = L.a;
    const int b = blockIdx.x, tid = threadIdx.x, nb = L.nb;
    unsigned long long* dbg = L.dbg;
    HcSync* sy = &g_hc_sync[L.slot];
    const bool cn = Q8 && L.cn != 0;
    const Items<T> it(nb, L.nu, a.w_router != nullptr, L.hcx, cn);
    stamp(dbg, b, 0, tid);
    if (tid == 0) {
        for (int p = 0; p <= UP_PIECES; ++p) mbar_init(&bars[p], 1);
        if constexpr (Q8)
            for (int p = 0; p <= NSTAGE; ++p) mbar_init(&dbars[p], cn ? 1 : ROWS);
        mbar_init(&lbar, 1);
        s_item = (int) atomicAdd(&sy->q.v, 1u);
    }
    __syncthreads();
    int item = s_item;
    // constant data before the previous kernel is waited on
    NormRegs nr;
    if (item < it.ntask) hc_norm_pre(a, item & 3, tid, nr);
    bool staged = false;   // the first down item's prologue ran before the wait
    bool cn_bars = false;  // cn: the staged item uses dbars
    bool warm = false;     // this CTA ran the top-10's code once
    if (cn && item >= it.down && item < it.up) {   // cn: the item's weights do not depend on the previous kernel
        if constexpr (Q8) cn_bars = cn_issue(a, item - it.down, tid, smem, dbars, L.cn & 2, L.tma, L.tm);
        staged = true;
    }
    bool up_staged = false;   // the first up item's slice requested before the wait
    if (Q8 && cn && L.up_tma == 2 && item >= it.up && item < it.rt) {
        const UpSlice sl = up_slice(item - it.up, it.nu);
        if (tid == 0) hc_up_tma(sl, smem, up_smem(smem, sl.nc, T, true, true).sc, &L.tm[6], &bars[0]);
        up_staged = true;
    }
    if (L.early) {
        if (!staged && (L.early & 2) && item >= it.down && item < it.up) {
            if constexpr (Q8) hc_down8_issue(a, item - it.down, tid, smem, dbars, 0, L.pre8);
            else if (L.xh) hc_down_prologue<T, RING_CAP, true>(a, item - it.down, tid, smem);
            else hc_down_prologue<T, RING_CAP>(a, item - it.down, tid, smem);
            staged = true;
        }
        if (L.early & 1) pdl_launch_dependents();
    }
    pdl_wait();
    stamp(dbg, b, 1, tid);
    unsigned up_ph = 0, rt_ph = 0, dn_ph = 0, lo_ph = 0;
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
        } else if (cn && item < it.up) {
            const int d = item - it.down;
            if (dbg != nullptr && tid == 0) dbg[b * 16 + 15] = 1;
            if constexpr (Q8) {
                if (!staged) cn_bars = cn_issue(a, d, tid, smem, dbars, L.cn & 2, L.tma, L.tm);
                const CnRows w = cn_rows(a, d);
                if (w.q != nullptr) {
                    cn_x<T>(a, w.hs, tid, smem, s_ssp, L.cns, d < CN_HS, L.cn_copies);
                    stamp(dbg, b, 2, tid);
                    cn_dot<T>(w, tid, smem, dbars, dn_ph, L.cnp, &sy->nload.v, L.cn & 2, L.cn_copies, L.tma);
                }
                if (cn_bars) dn_ph ^= 1u;
            }
            cta_signal(&sy->down.v, tid);
            stamp(dbg, b, 3, tid);
        } else if (item < it.up) {
            const int d = item - it.down;
            if (dbg != nullptr && tid == 0) dbg[b * 16 + 15] = 1;
            if (!staged) {
                // weight reads queued ahead of the norm's would delay it
                cta_wait(&sy->nload.v, (unsigned) it.ntask, tid);
                if constexpr (Q8) hc_down8_issue(a, d, tid, smem, dbars, 0, L.pre8);
                else if (L.xh) hc_down_prologue<T, RING_CAP, true>(a, d, tid, smem);
                else hc_down_prologue<T, RING_CAP>(a, d, tid, smem);
            }
            if (L.pf == 1 && tid == 0 && a.w_router != nullptr) {   // the router rows into L2 behind the down weights
                const size_t r16 = (size_t) NE * N * 2 / 16, q0 = (size_t) d * r16 / ND, q1 = (size_t) (d + 1) * r16 / ND;
                prefetch_l2(reinterpret_cast<const char*>(a.w_router) + q0 * 16, (q1 - q0) * 16);
            }
            cta_wait(&sy->norm.v, (unsigned) it.ntask, tid);
            stamp(dbg, b, 2, tid);
            if constexpr (Q8) {
                hc_down8_issue(a, d, tid, smem, dbars, L.pre8, NSTAGE);
                if (L.xh) hc_down8_main<T, true>(a, L.xc + (size_t) (d % it.ncopy) * MAXT * D / 2, d, tid, smem, dbars, dn_ph);
                else hc_down8_main<T, false>(a, L.xc + (size_t) (d % it.ncopy) * MAXT * D, d, tid, smem, dbars, dn_ph);
                dn_ph ^= 1u;
            } else if (L.xh) {
                hc_down_main<T, RING_CAP, true>(a, L.xc + (size_t) (d % it.ncopy) * MAXT * D / 2, d, tid, smem);
            } else {
                hc_down_main<T, RING_CAP>(a, L.xc + (size_t) (d % it.ncopy) * MAXT * D, d, tid, smem);
            }
            cp_async_wait<0>();
            cta_signal(&sy->down.v, tid);
            stamp(dbg, b, 3, tid);
        } else if (item < it.rt) {
            const bool rx = L.hcx == 2 && it.end > it.t10;   // this slice's share of the router logits too
            const UpSlice sl = rx ? up_slice_even(item - it.up, it.nu) : up_slice(item - it.up, it.nu);
            if (dbg != nullptr && tid == 0) dbg[b * 16 + 15] = 2;
            const bool mma = L.upmma != 0;
            uint16_t* const sc8 = Q8 ? up_smem(smem, sl.nc, T, mma, true).sc : nullptr;
            const bool cp = Q8 && mma && (L.cn & 4);   // the slice by every thread's cp.async
            const bool ut = Q8 && mma && L.up_tma && !cp;   // the slice by tensor copies (hc_up_tma)
            if (cp) hc_up_cp8(a, sl, smem, sc8, tid);
            else if (ut && !up_staged && tid == 0) hc_up_tma(sl, smem, sc8, &L.tm[6], &bars[0]);
            else if (Q8 && !ut && tid == 0)   // ahead of R'; cn: once enough w_down rows are in
                hc_up_issue8(a, sl, smem, bars, &sy->nload.v, cn ? (unsigned) L.cn_upw : (unsigned) it.ntask, mma, sc8);
            hc_up_pre<T>(a, sl, tid, smem, mma);
            const int bf_off = (up_smem_bytes<T>(sl.nc, mma, Q8) + 15) & ~15;
            const int t10_off = bf_off + (Q8 && mma && L.bf ? BF_BYTES : 0);
            const bool dry = !L.no_dry && it.end > it.t10 && t10_off + 3 * CAND_WORDS * 4 <= SMEM;
            warm = warm || dry;
            const int cpy = cn ? (item - it.up) % L.cn_copies : 0;
            const float *cnp = L.cnp + (size_t) cpy * CN_PART, *cns = L.cns + (size_t) cpy * CN_SSN;
            // cn: the down items' sums by two bulk copies once they are done (after the top-10 area)
            const int lst_off = (t10_off + 127) & ~127;
            const unsigned lst_p = (unsigned) (T * CN_HS * CN_ROWS * 4), lst_s = (unsigned) (T * CN_HS * 4);
            const bool lst = Q8 && mma && cn && L.lo_stage && !dry && lst_off + (int) (lst_p + lst_s) <= SMEM;
            float* const lsm = reinterpret_cast<float*>(smem + lst_off);
            if (tid == 0) {
                hc_up_stream(a, sl, smem, bars, &sy->nload.v, (unsigned) it.ntask, &sy->down.v, up_ph, mma, !(Q8 && !mma),
                             sc8, cp || ut ? -1 : Q8 ? UP_PIECES : 0, (unsigned) it.nd, dbg, b);
                if (lst) {
                    asm volatile("fence.proxy.async;" ::: "memory");
                    mbar_expect_tx(&lbar, lst_p + lst_s);
                    bulk_g2s(lsm, cnp, lst_p, &lbar);
                    bulk_g2s(lsm + T * CN_HS * CN_ROWS, cns, lst_s, &lbar);
                }
            } else if (dry) {
                // while the down projection runs, warps 1..3 run the top-10 on fixed logits: its code is then in this
                // SM's instruction caches if this CTA claims the top-10 item
                const int warp = tid >> 5, off = t10_off;
                int32_t* ids = reinterpret_cast<int32_t*>(L.t10_scratch + (size_t) b * 2 * KT);
                uint32_t* cand = reinterpret_cast<uint32_t*>(smem + off) + (warp - 1) * CAND_WORDS;
                if (warp >= 1 && warp <= 3) {
                    if (L.hcx == 2)
                        hc_top10_item_fx(L.t10_dummy, nullptr, ids, L.t10_scratch + (size_t) b * 2 * KT + KT, tid & 31, cand,
                                         nullptr, b);
                    else
                        hc_top10_item(reinterpret_cast<const float*>(L.t10_dummy + NE), ids,
                                      L.t10_scratch + (size_t) b * 2 * KT + KT, tid & 31, cand, nullptr, b);
                }
            }
            const unsigned wph = up_ph;
            const bool late_w = ut && L.late_w;   // the slice waited on in the up pass, after lo
            if (cp) {
                cp_async_wait<0>();
            } else {
                if (ut && !late_w) {
                    mbar_wait(&bars[0], up_ph);
                    stamp(dbg, b, 10, tid);
                }
                up_ph ^= 1u;
            }
            __syncthreads();
            if (Q8 && !mma) {   // the 8-bit slice widened while the down projection runs
                hc_up_widen(sl, tid, smem, up_smem(smem, sl.nc, T, false, true).sc);
                cta_wait(&sy->down.v, (unsigned) it.nd, tid);
            }
            stamp(dbg, b, 2, tid);
            uint32_t rw[RQ];   // the router weights stream in while the up pass runs (after the w_up copies)
            if (rx) hcx_router_load(a, sl, tid, rw);
            uint16_t* const bfp = reinterpret_cast<uint16_t*>(smem + bf_off);
            uint64_t* const lb = lst ? &lbar : nullptr;
            uint64_t* const wb = late_w ? &bars[0] : nullptr;
            if (lst) {
                cnp = lsm;
                cns = lsm + T * CN_HS * CN_ROWS;
            }
            if (Q8 && mma && cn && L.bf == 2) hc_up_main_mma8<T, true, true>(a, sl, tid, smem, dbg, b, cnp, cns, item == it.up, bfp, ut, lb, lo_ph, wb, wph);
            else if (Q8 && mma && cn && L.bf) hc_up_main_mma8<T, true>(a, sl, tid, smem, dbg, b, cnp, cns, item == it.up, bfp, ut, lb, lo_ph, wb, wph);
            else if (Q8 && mma && cn) hc_up_main_mma8<T>(a, sl, tid, smem, dbg, b, cnp, cns, item == it.up, nullptr, ut, lb, lo_ph, wb, wph);
            else if (Q8 && mma && L.bf) hc_up_main_mma8<T, true>(a, sl, tid, smem, dbg, b, nullptr, nullptr, false, bfp, ut);
            else if (Q8 && mma) hc_up_main_mma8<T>(a, sl, tid, smem, dbg, b, nullptr, nullptr, false, nullptr, ut);
            else if (mma) hc_up_main_mma<T>(a, sl, tid, smem, dbg, b);
            else if (L.ups) hc_up_main<T, true>(a, sl, tid, smem, true, dbg, b);
            else hc_up_main<T>(a, sl, tid, smem, true, dbg, b);
            if (lst) lo_ph ^= 1u;
            if (rx) {
                // the slice's mixed columns again (hc_up_main's expression) into shared memory, then the logit shares
                const UpSmem u = up_smem(smem, sl.nc, T, mma, Q8);
                float* mix = reinterpret_cast<float*>(smem + ((up_smem_bytes<T>(sl.nc, mma, Q8) + 15) & ~15));
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
            if (cn && !warm) {   // a down CTA may claim the item: its code once on fixed logits while the router runs
                const int warp = tid >> 5;
                if (warp >= 1 && warp <= 3)
                    hc_top10_item(reinterpret_cast<const float*>(L.t10_dummy + NE), reinterpret_cast<int32_t*>(L.t10_scratch + (size_t) b * 2 * KT),
                                  L.t10_scratch + (size_t) b * 2 * KT + KT, tid & 31,
                                  reinterpret_cast<uint32_t*>(smem) + (warp - 1) * CAND_WORDS, nullptr, b);
                __syncthreads();
            }
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
                if (a.xq != nullptr) {   // the item's share of mixed's q8_1 blocks, while the top-10 runs
                    const int nq = T * (N / 32), q0 = r * nq / ND, q1 = (r + 1) * nq / ND;
                    for (int q = q0 + (tid >> 5); q < q1; q += WARPS)
                        hc_q8_1_store(__ldcg(a.mixed + (size_t) q * 32 + (tid & 31)), static_cast<HcQ81*>(a.xq), q,
                                      tid & 31);
                }
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
        up_staged = false;
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
constexpr KernelFn kKernels[MAXT] = {hc_fused_kernel<1, false>, hc_fused_kernel<2, false>, hc_fused_kernel<3, false>,
                                      hc_fused_kernel<4, false>, hc_fused_kernel<5, false>, hc_fused_kernel<6, false>,
                                      hc_fused_kernel<7, false>, hc_fused_kernel<8, false>};
constexpr KernelFn kKernels8[MAXT] = {hc_fused_kernel<1, true>, hc_fused_kernel<2, true>, hc_fused_kernel<3, true>,
                                       hc_fused_kernel<4, true>, hc_fused_kernel<5, true>, hc_fused_kernel<6, true>,
                                       hc_fused_kernel<7, true>, hc_fused_kernel<8, true>};
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
// STRATA_DEC_HC2 (bits; the 8-bit read with HCX=3): 1 = down items over half-streams that form R' and x themselves, no
// norm tasks (cn_rows; not bitwise, T-independent); 4 = the 8-bit mma up pass's B fragments formed once per item
// (bitwise); with 1: 8 = the down items' rows, 16 = the up items' slices by cp.async instead of bulk copies; with 1 and
// 4: 32 = lo as one f16 part in the up pass (half the mma; not bitwise, T-independent).
// STRATA_DEC_HC2_UPW=<percent> (with 1): the up items request w_up once that share of the w_down rows is in.
// STRATA_DEC_HC2_COPIES=<n> (with 1, default 1): copies of the down items' sums.
// STRATA_DEC_HC3 (bits, with HC2 & 1; bitwise): 1 = each warp's w_down rows by two 2D tensor copies instead of eight
// bulk copies, or 2 = a w_down item's rows by two tensor copies of thread 0 (not with HC2 & 8); 4 = an up slice by two
// tensor copies in a chunk layout instead of ~45 bulk copies (not with HC2 & 16); with 4: 8 = the first up slice
// requested before the previous kernel is waited on; 16 = the up items skip the top-10 dry run (its global loads queue
// behind the weight copies and hold the up pass's start); with 16: 32 = an up item's copy of the down items' sums by
// two bulk copies into shared memory (where it fits) instead of every thread's loads; with 4: 128 = the slice's copy
// waited on after lo is formed (the lo phase overlaps the weights' landing).
struct SkConf {
    bool on = false;
    int pf = 0, nu = 0, ups = 0, hcx = 0, upmma = 0, xh = 0, hc2 = 0, upw = 0, copies = 1, hc3 = 0;
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
        e = std::getenv("STRATA_DEC_HC2");
        k.hc2 = e ? (int) std::strtol(e, nullptr, 0) : 0;
        e = std::getenv("STRATA_DEC_HC2_UPW");
        k.upw = std::min(std::max(e ? std::atoi(e) : 0, 0), 100);
        e = std::getenv("STRATA_DEC_HC2_COPIES");
        k.copies = std::min(std::max(e ? std::atoi(e) : 1, 1), CN_MAXCOPY);
        e = std::getenv("STRATA_DEC_HC3");
        k.hc3 = e ? (int) std::strtol(e, nullptr, 0) : 0;
        return k;
    }();
    return c;
}

unsigned long long* g_dbg = nullptr;
int g_slot = 0;
}  // namespace

bool hc_fused_q8_rows_ok() { return !sk_conf().on && sk_conf().hcx == 1; }

namespace {

struct DevInfo {
    bool checked = false, ok = false;
    int sms = 0;
    float* xc = nullptr;
    unsigned long long* t10 = nullptr;    // [NE] fixed logits (fixed point), [NE] the same as f32 (in NE / 2 words), then
                                          // [sms][2 * KT] float scratch
    unsigned long long* racc = nullptr;   // [NSLOT][MAXT][NE]
    float* cn = nullptr;                  // [NSLOT][MAXT * CN_HS * CN_ROWS + MAXT * CN_HS]
};
constexpr size_t CN_SLOT = (size_t) CN_MAXCOPY * (CN_PART + CN_SSN);
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
               cudaMalloc((void**) &d.racc, (size_t) NSLOT * MAXT * NE * 8) == cudaSuccess &&
               cudaMalloc((void**) &d.cn, (size_t) NSLOT * CN_SLOT * 4) == cudaSuccess;
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
            d.ok = cudaFuncSetAttribute(kKernels[t], cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM) == cudaSuccess &&
                   cudaFuncSetAttribute(kKernels8[t], cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM) == cudaSuccess;
        for (int t = 0; t < MAXT && d.ok && sk_conf().on; ++t)
            d.ok = cudaFuncSetAttribute(kSkKernels[t], cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM) == cudaSuccess;
        cudaGetLastError();
        d.checked = true;
    }
    return d;
}

// 8-bit copies: q = round(w / s) in [-127, 127], s = f16(absmax / 127) per G8 consecutive values of a row; a row's
// `gpr` scales start every `srow` f16 (a multiple of 8: 16-byte aligned runs).  `err` (optional): the sums of w^2
// and of the error^2.
__global__ void hc8_quant_kernel(const uint16_t* w, int64_t groups, int gpr, int srow, int8_t* q, uint16_t* sc,
                                 double* err) {
    const int64_t gi = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    double e[2] = {0.0, 0.0};
    if (gi < groups) {
        float x[G8], amax = 0.0f;
        for (int i = 0; i < G8; ++i) {
            x[i] = __uint_as_float((uint32_t) w[gi * G8 + i] << 16);
            amax = fmaxf(amax, fabsf(x[i]));
        }
        const __half h = __float2half_rn(amax / 127.0f);
        const float s = __half2float(h);
        sc[gi / gpr * srow + gi % gpr] = __half_as_ushort(h);
        for (int i = 0; i < G8; ++i) {
            const float v = s > 0.0f ? fminf(fmaxf(rintf(x[i] / s), -127.0f), 127.0f) : 0.0f;
            q[gi * G8 + i] = (int8_t) v;
            e[0] += (double) x[i] * x[i];
            e[1] += (double) (v * s - x[i]) * (v * s - x[i]);
        }
    }
    if (err != nullptr)
        for (int k = 0; k < 2; ++k) {
            double v = e[k];
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if ((threadIdx.x & 31) == 0) atomicAdd(&err[k], v);
        }
}
struct Hc8Copy {
    int8_t* q = nullptr;   // then the scales, in the same allocation
    uint16_t* s = nullptr;
    size_t bytes = 0;
    int rows = 0, cols = 0, srow = 0;
    bool owned = false;    // keyed by q itself: no bf16 matrix behind it
};
std::mutex g_hc8_mu;
std::unordered_map<const void*, Hc8Copy> g_hc8;
size_t g_hc8_bytes = 0;

// the copy of the bf16 matrix w (device memory of the current card); false when there is no VRAM for it
bool hc8_make(const uint16_t* w, int rows, int cols, Hc8Copy& c) {
    const int64_t n = (int64_t) rows * cols, groups = n / G8;
    const int gpr = cols / G8, srow = (gpr + 7) & ~7;
    c.bytes = (size_t) n + (size_t) rows * srow * 2;
    c.rows = rows;
    c.cols = cols;
    c.srow = srow;
    if (cudaMalloc((void**) &c.q, c.bytes) != cudaSuccess) {
        cudaGetLastError();
        std::fprintf(stderr, "strata hc8: no VRAM for an 8-bit copy (%zu bytes)\n", c.bytes);
        return false;
    }
    c.s = reinterpret_cast<uint16_t*>(c.q + n);
    cudaMemset(c.s, 0, (size_t) rows * srow * 2);
    static const bool stats = std::getenv("STRATA_DEC_HC8_STATS") != nullptr;
    double* derr = nullptr;
    if (stats && cudaMalloc((void**) &derr, 2 * sizeof(double)) == cudaSuccess) cudaMemset(derr, 0, 2 * sizeof(double));
    hc8_quant_kernel<<<(unsigned) ((groups + 255) / 256), 256>>>(w, groups, gpr, srow, c.q, c.s, derr);
    cudaDeviceSynchronize();
    if (derr != nullptr) {
        double h[2];
        cudaMemcpy(h, derr, sizeof h, cudaMemcpyDeviceToHost);
        cudaFree(derr);
        std::fprintf(stderr, "strata hc8: %d x %d: relative rms error %.3e\n", rows, cols, std::sqrt(h[1] / h[0]));
    }
    return true;
}

__global__ void hc8_dequant_bf16_kernel(const int8_t* q, const uint16_t* s, int cols, int srow, int64_t row0, int64_t n,
                                        uint16_t* out) {
    const int64_t i = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) * 8;   // 8 values, one scale group
    if (i >= n * cols) return;
    const int64_t r = row0 + i / cols;
    const int k = (int) (i % cols);
    const float sc = __half2float(__ushort_as_half(s[r * srow + k / G8]));
    const uint2 w = *reinterpret_cast<const uint2*>(q + r * cols + k);
    uint16_t o[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int v = (int) (int8_t) ((j < 4 ? w.x : w.y) >> (8 * (j & 3)));
        o[j] = __bfloat16_as_ushort(__float2bfloat16_rn((float) v * sc));
    }
    *reinterpret_cast<uint4*>(out + i) = *reinterpret_cast<const uint4*>(o);
}

}  // namespace

bool hc8_enabled() {
    static const bool on = [] { const char* e = std::getenv("STRATA_DEC_HC8"); return e && e[0] == '1'; }();
    return on;
}

bool hc8_only() {
    static const bool on = [] {
        const char* k = std::getenv("STRATA_DEC_HC8_KEEP");
        return hc8_enabled() && !sk_conf().on && !(k && k[0] == '1');
    }();
    return on;
}

void hc8_prepare(const uint16_t* w, int rows, int cols) {
    if (w == nullptr || rows <= 0 || cols <= 0 || cols % G8 != 0) return;
    std::lock_guard<std::mutex> lk(g_hc8_mu);
    if (g_hc8.count(w)) return;
    int dev = 0;
    cudaPointerAttributes pa{};
    cudaGetDevice(&dev);
    if (cudaPointerGetAttributes(&pa, w) != cudaSuccess || pa.type != cudaMemoryTypeDevice || pa.device != dev) {
        cudaGetLastError();
        return;   // another card's matrix: its reads stay bf16
    }
    Hc8Copy c;
    if (!hc8_make(w, rows, cols, c)) return;   // that read stays bf16
    g_hc8[w] = c;
    g_hc8_bytes += c.bytes;
}

const void* hc8_adopt(const uint16_t* w, int rows, int cols) {
    if (w == nullptr || rows <= 0 || cols <= 0 || cols % G8 != 0) return nullptr;
    std::lock_guard<std::mutex> lk(g_hc8_mu);
    Hc8Copy c;
    if (!hc8_make(w, rows, cols, c)) return nullptr;
    c.owned = true;
    g_hc8[c.q] = c;
    g_hc8_bytes += c.bytes;
    return c.q;
}

bool hc8_lookup(const void* w, const int8_t** q, const uint16_t** s, size_t* bytes) {
    std::lock_guard<std::mutex> lk(g_hc8_mu);
    const auto it = g_hc8.find(w);
    if (it == g_hc8.end()) return false;
    *q = it->second.q;
    *s = it->second.s;
    if (bytes != nullptr) *bytes = it->second.bytes;
    return true;
}

bool hc8_view(const void* w, Hc8View& v) {
    std::lock_guard<std::mutex> lk(g_hc8_mu);
    const auto it = g_hc8.find(w);
    if (it == g_hc8.end()) return false;
    const Hc8Copy& c = it->second;
    v.q = c.q; v.s = c.s; v.rows = c.rows; v.cols = c.cols; v.srow = c.srow; v.bytes = c.bytes; v.owned = c.owned;
    return true;
}

void hc8_dequant_bf16(const Hc8View& v, int64_t row0, int64_t n, uint16_t* out, void* stream) {
    if (n <= 0) return;
    const int64_t units = n * v.cols / 8;
    hc8_dequant_bf16_kernel<<<(unsigned) ((units + 255) / 256), 256, 0, (cudaStream_t) stream>>>(v.q, v.s, v.cols, v.srow,
                                                                                              row0, n, out);
}

size_t hc8_bytes() {
    std::lock_guard<std::mutex> lk(g_hc8_mu);
    return g_hc8_bytes;
}

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
                                       "ticket", "up-lo", "norm-wr/w-in", "norm-sig/dn-in", "t10-smax", "t10-cand", "t10-rank", "role"};
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

// cuTensorMapEncodeTiled from the driver; the maps of a read's 8-bit matrices, kept per q_down.
using EncodeFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                              const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                              CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
EncodeFn encode_fn() {
    static const EncodeFn f = [] {
        void* p = nullptr;
        cudaDriverEntryPointQueryResult q{};
        if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q) != cudaSuccess ||
            q != cudaDriverEntryPointSuccess) {
            cudaGetLastError();
            return (EncodeFn) nullptr;
        }
        return reinterpret_cast<EncodeFn>(p);
    }();
    return f;
}
// dims[0] elements of the type, innermost; strides (bytes) of dims 1.. ; box per dim.
bool tmap(CUtensorMap* m, const void* base, CUtensorMapDataType ty, int rank, const cuuint64_t* dims,
          const cuuint64_t* strides, const cuuint32_t* box) {
    const EncodeFn f = encode_fn();
    if (f == nullptr || base == nullptr) return false;
    const cuuint32_t es[5] = {1, 1, 1, 1, 1};
    return f(m, ty, (cuuint32_t) rank, const_cast<void*>(base), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
             CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}
// A row-major [rows][cols] matrix (row stride `stride` bytes), box [box_r][box_c].
bool tmap_2d(CUtensorMap* m, const void* base, CUtensorMapDataType ty, uint64_t cols, uint64_t rows, uint64_t stride,
             uint32_t box_c, uint32_t box_r) {
    const cuuint64_t dim[2] = {cols, rows}, str[1] = {stride};
    const cuuint32_t box[2] = {box_c, box_r};
    return tmap(m, base, ty, 2, dim, str, box);
}
struct CnMaps {
    CUtensorMap tm[8];
    bool ok, up;
};
// False when the down maps failed; `up`: the up maps are in.
bool cn_maps(const HcFusedArgs& h, CUtensorMap* tm, bool& up) {
    static std::mutex mu;
    static std::unordered_map<const void*, CnMaps> cache;
    std::lock_guard<std::mutex> lk(mu);
    auto it = cache.find(h.q_down);
    if (it == cache.end()) {
        CnMaps c{};
        constexpr auto U64 = CU_TENSOR_MAP_DATA_TYPE_UINT64, U16 = CU_TENSOR_MAP_DATA_TYPE_UINT16;
        c.ok = tmap_2d(&c.tm[0], h.q_down, U64, D / 8, LR, D, CN_W / 8, 4) &&
               tmap_2d(&c.tm[1], h.s_down, U16, D / G8, LR, D / G8 * 2, CN_W / G8, 4) &&
               (h.q_inject == nullptr || (tmap_2d(&c.tm[2], h.q_inject, U64, D / 8, HC, D, CN_W / 8, 4) &&
                                          tmap_2d(&c.tm[3], h.s_inject, U16, D / G8, HC, D / G8 * 2, CN_W / G8, 4))) &&
               tmap_2d(&c.tm[4], h.q_down, U64, D / 8, LR, D, CN_W / 8, CN_RG) &&
               tmap_2d(&c.tm[5], h.s_down, U16, D / G8, LR, D / G8 * 2, CN_W / G8, CN_RG);
        // q_up [HC][N][LR] as (16 bytes, column, stream, k-chunk) -> hc_up_tma's chunk layout; s_up [HC][N][SU8]
        const cuuint64_t qd[4] = {2, N, HC, LR / 16}, qs[3] = {LR, (cuuint64_t) N * LR, 16};
        const cuuint32_t qb[4] = {2, UPCM_SL, HC, LR / 16};
        const cuuint64_t sd[3] = {SU8, N, HC}, ss[2] = {SU8 * 2, (cuuint64_t) N * SU8 * 2};
        const cuuint32_t sb[3] = {SU8, UPCM_SL, HC};
        c.up = tmap(&c.tm[6], h.q_up, U64, 4, qd, qs, qb) && tmap(&c.tm[7], h.s_up, U16, 3, sd, ss, sb);
        it = cache.emplace(h.q_down, c).first;
    }
    std::copy(it->second.tm, it->second.tm + 8, tm);
    up = it->second.up;
    return it->second.ok;
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
    // the 8-bit copies when every matrix of the read has one (the split-K read stays bf16); the prefetches then take
    // the copies too
    {
        HcFusedArgs& h = L.a;
        bool q8 = false;
        if (!sk.on && h.q_down == nullptr && hc8_enabled()) {
            q8 = hc8_lookup(a.w_down, &h.q_down, &h.s_down) && hc8_lookup(a.w_up, &h.q_up, &h.s_up) &&
                 (a.w_inject == nullptr || hc8_lookup(a.w_inject, &h.q_inject, &h.s_inject));
            for (int r = 0; q8 && r < 4; ++r) {
                const int8_t* q = nullptr;
                const uint16_t* sc = nullptr;
                size_t bytes = 0;
                if (h.pf[r] != nullptr && hc8_lookup(h.pf[r], &q, &sc, &bytes)) { h.pf[r] = q; h.pf_bytes[r] = bytes; }
            }
        } else if (!sk.on && h.q_down != nullptr) {
            q8 = h.q_up != nullptr && h.s_down != nullptr && h.s_up != nullptr &&
                 (a.w_inject == nullptr || (h.q_inject != nullptr && h.s_inject != nullptr));
        }
        if (!q8) {
            Hc8View v;
            if ((hc8_view(a.w_down, v) && v.owned) || (hc8_view(a.w_up, v) && v.owned)) {
                std::fprintf(stderr, "hc_fused_read: a matrix held only as its 8-bit copy needs the 8-bit read\n");
                std::exit(1);
            }
            h.q_down = nullptr; h.q_up = nullptr; h.q_inject = nullptr;
        }
        if (a.w_inject == nullptr) h.q_inject = nullptr;
    }
    const bool q8 = L.a.q_down != nullptr;
    L.cn = (sk.hc2 & 1) && q8 && !sk.on && L.upmma && L.hcx == 1 && sk.nu == 0 && L.nb - CN_ND >= CN_MIN_NU;
    if (L.cn && (sk.hc2 & 8)) L.cn |= 2;
    if (L.cn && (sk.hc2 & 16)) L.cn |= 4;
    L.bf = (sk.hc2 & 4) && q8 && !sk.on && L.upmma && L.hcx == 1;
    if (L.bf && L.cn && (sk.hc2 & 32)) L.bf = 2;
    L.cn_upw = sk.upw * CN_G * CN_HS * WARPS / 100;
    L.cnp = d.cn + (size_t) L.slot * CN_SLOT;
    L.cns = L.cnp + (size_t) CN_MAXCOPY * CN_PART;
    L.cn_copies = sk.copies;
    L.no_dry = L.cn && (sk.hc3 & 16) ? 1 : 0;
    L.lo_stage = L.no_dry && (sk.hc3 & 32) ? 1 : 0;
    if (L.cn && (sk.hc3 & 7)) {
        bool up = false;
        const bool down = cn_maps(L.a, L.tm, up);
        L.tma = down && (L.cn & 3) == 1 ? sk.hc3 & 3 : 0;
        if (L.tma == 3) L.tma = 0;
        L.up_tma = up && (sk.hc3 & 4) && !(L.cn & 4) ? (sk.hc3 & 8 ? 2 : 1) : 0;
        L.late_w = L.up_tma && (sk.hc3 & 128) ? 1 : 0;
    }
    static const int pre8 = [] { const char* e = std::getenv("STRATA_DEC_HC8_PRE"); return e ? std::atoi(e) : 3; }();
    L.pre8 = std::min(std::max(pre8, 0), NSTAGE);
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
        launch_pdl((q8 ? kKernels8 : kKernels)[a.T - 1], dim3((unsigned) L.nb), dim3(THREADS), (size_t) SMEM,
                   (cudaStream_t) stream, L);
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
