// src/kernels/cuda/dec_gdn_rec.cu - see include/strata/kernels/dec_gdn.hpp: the recurrence stage.
//
// Built with the flags of verify_kernels.cu, whose kernels it fuses: gdn_conv_l2_multi (conv1d + SiLU + q/k L2
// norm), gdn_ab_multi (alpha/beta gates), gdn_step_norm_multi (the recurrence and the gated RMS norm).  The q8_1 of
// the output is native_quantize_q8_1's (a fast-math file), its two divisions written as the instruction it uses.
//
// A cluster of four blocks per value head; block `rank` owns state columns [32 rank, 32 rank + 32), thread
// (warp rg, lane c) the rows [32 rg, 32 rg + 32) of column 32 rank + c, as gdn_step_commit_kernel splits them.
// Each column's sums are the reference's: four 32-row fmaf chains added in row-group order.  The head's RMS norm
// adds the four warps' butterflies in rank order through distributed shared memory; rank 0 computes alpha, rank 1
// beta, and the others read them the same way.  The state and every constant are loaded before the grid
// dependency wait (the window never writes the state).
#include "strata/kernels/dec_gdn.hpp"
#include "strata/kernels/pdl.hpp"
#include "strata/kernels/q8_1_finite.hpp"

#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace cg = cooperative_groups;

namespace strata::kernels {
namespace {

constexpr int S = 128;   // state size
constexpr int RG = 4;    // row groups
constexpr int RPG = S / RG;
constexpr int NJ = 10;   // uint4 of bf16 per lane in a 2560-wide alpha/beta row

struct Q81 {
    half2 ds;
    int8_t qs[32];
};

__device__ __forceinline__ float div_approx_ftz(float a, float b) {
    float r;
    asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

// q8_1_quant with native_mmvq.cu's division
__device__ __forceinline__ int8_t q8_1_quant_fm(const float xi, const float d, const float amax) {
    if (amax == 0.0f) return 0;
    const float q = roundf(div_approx_ftz(xi, d));
    return (int8_t) (q > 127.0f ? 127.0f : q < -127.0f ? -127.0f : q);
}

// conv1d over [hist(3) | x_0 .. x_t] of channel c, then SiLU: gdn_conv_l2_multi_kernel's expression
__device__ __forceinline__ float conv_silu(const float (&hist)[3], const float (&w)[4], const float* qkv, int C, int c,
                                           int t) {
    float win[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int src = t + j;
        win[j] = src < 3 ? hist[src] : qkv[(size_t) (src - 3) * C + c];
    }
    const float v0 = win[0], v1 = win[1], v2 = win[2], x = qkv[(size_t) t * C + c];
    float sum = v0 * w[0] + v1 * w[1] + v2 * w[2] + x * w[3];
    return sum / (1.0f + __expf(-sum));
}

struct RecSmem {
    float qn[kDecMaxT][S], kn[kDecMaxT][S];
    float v[kDecMaxT][32];
    float oc[kDecMaxT][32];
    float red_kv[RG][32], red_o[RG][32];
    float part[2][kDecMaxT][RG];   // the q/k L2 partials
    float g[kDecMaxT], b[kDecMaxT];
    float wsum[kDecMaxT];          // this rank's sum of squares per token
};

// One block of a head's cluster.  `head` = value head, `rank` = its column group.
template<int MAXT>
__device__ __forceinline__ void rec_block(const DecRecArgs& a, int T, int head, int rank, RecSmem& sm,
                                          cg::cluster_group cluster) {
    const int rg = threadIdx.x / 32, lane = threadIdx.x % 32, tid = threadIdx.x;
    const int col = rank * 32 + lane;
    const int qh = head % a.h_k;
    const int qk = S * a.h_k;
    const int value_dim = S * a.h_v;
    const int C = a.C;
    // ---- constants (the state too: the window does not write it)
    float s[RPG];
    const float* base = a.state + ((size_t) (rg * RPG) * a.h_v + head) * S + col;
    const size_t row_stride = (size_t) a.h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
    const float* hist = a.state + (size_t) S * a.h_v * S;
    const int cq = qh * S + tid, ck = qk + qh * S + tid, cv = 2 * qk + head * S + col;
    float wq[4], wk[4], wv[4], hq[3], hk[3], hv[3];
#pragma unroll
    for (int j = 0; j < 4; ++j) { wq[j] = a.conv_w[cq * 4 + j]; wk[j] = a.conv_w[ck * 4 + j]; wv[j] = a.conv_w[cv * 4 + j]; }
#pragma unroll
    for (int j = 0; j < 3; ++j) { hq[j] = hist[cq * 3 + j]; hk[j] = hist[ck * 3 + j]; hv[j] = hist[cv * 3 + j]; }
    const bool ab_warp = rank < 2 && rg == 0;
    uint4 wab[NJ];
    if (ab_warp) {
        const uint4* w4 = reinterpret_cast<const uint4*>((rank == 1 ? a.w_beta : a.w_alpha) + (size_t) head * a.n_embd);
#pragma unroll
        for (int q = 0; q < NJ; ++q) wab[q] = __ldg(w4 + lane + 32 * q);
    }
    const float dt = a.dt[head], sa = a.ssm_a[head], gam = a.gamma[col];
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");   // the out projection may stage its weights
    pdl_wait();

    // ---- alpha (rank 0) / beta (rank 1): gdn_ab_multi_unrolled_kernel's sums
    if (ab_warp) {
        float acc[MAXT];
#pragma unroll
        for (int t = 0; t < MAXT; ++t) acc[t] = 0.0f;
#pragma unroll
        for (int q = 0; q < NJ; ++q) {
            const int j = lane + 32 * q;
#pragma unroll
            for (int t = 0; t < MAXT; ++t) {
                if (t < T) {
                    const float* xt = a.x + (size_t) t * a.n_embd;
                    const float4 xa = *reinterpret_cast<const float4*>(xt + j * 8);
                    const float4 xb = *reinterpret_cast<const float4*>(xt + j * 8 + 4);
                    float v = acc[t];
                    v = fmaf(__uint_as_float(wab[q].x << 16), xa.x, v); v = fmaf(__uint_as_float(wab[q].x & 0xffff0000u), xa.y, v);
                    v = fmaf(__uint_as_float(wab[q].y << 16), xa.z, v); v = fmaf(__uint_as_float(wab[q].y & 0xffff0000u), xa.w, v);
                    v = fmaf(__uint_as_float(wab[q].z << 16), xb.x, v); v = fmaf(__uint_as_float(wab[q].z & 0xffff0000u), xb.y, v);
                    v = fmaf(__uint_as_float(wab[q].w << 16), xb.z, v); v = fmaf(__uint_as_float(wab[q].w & 0xffff0000u), xb.w, v);
                    acc[t] = v;
                }
            }
        }
#pragma unroll
        for (int t = 0; t < MAXT; ++t) {
            if (t >= T) break;
            float v = acc[t];
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane != 0) continue;
            if (rank == 1) {
                const float bv = 1.0f / (1.0f + __expf(-v));
                sm.b[t] = bv;
                a.beta[(size_t) t * a.h_v + head] = bv;
            } else {
                const float u = v + dt;
                const float sp = u > 20.0f ? u : log1pf(__expf(u));
                const float gv = sp * sa;
                sm.g[t] = gv;
                a.gate[(size_t) t * a.h_v + head] = gv;
            }
        }
    }

    // ---- conv + SiLU of the head's q and k channels (L2-normalized) and of the block's v columns
    const bool write_qk = rank == 0 && head < a.h_k;
    for (int t = 0; t < T; ++t) {
        float yq = conv_silu(hq, wq, a.qkv, C, cq, t);
        float yk = conv_silu(hk, wk, a.qkv, C, ck, t);
        float sq = yq * yq, sk = yk * yk;
        for (int o = 16; o > 0; o >>= 1) {
            sq += __shfl_xor_sync(0xffffffffu, sq, o);
            sk += __shfl_xor_sync(0xffffffffu, sk, o);
        }
        if (lane == 0) { sm.part[0][t][rg] = sq; sm.part[1][t][rg] = sk; }
        if (rg == 0) {
            const float yv = conv_silu(hv, wv, a.qkv, C, cv, t);
            sm.v[t][lane] = yv;
            a.h[(size_t) t * C + cv] = yv;
        }
        __syncthreads();
        const float ssq = sm.part[0][t][0] + sm.part[0][t][1] + sm.part[0][t][2] + sm.part[0][t][3];
        const float ssk = sm.part[1][t][0] + sm.part[1][t][1] + sm.part[1][t][2] + sm.part[1][t][3];
        yq *= rsqrtf(ssq + a.eps);
        yk *= rsqrtf(ssk + a.eps);
        sm.qn[t][tid] = yq;
        sm.kn[t][tid] = yk;
        if (write_qk) {
            a.h[(size_t) t * C + cq] = yq;
            a.h[(size_t) t * C + ck] = yk;
        }
    }
    cluster.sync();   // alpha/beta of ranks 0/1 and every rank's qn/kn/v are ready
    if (tid < T && rank != 0) {
        sm.g[tid] = *cluster.map_shared_rank(&sm.g[tid], 0);
    } else if (tid >= 32 && tid - 32 < T && rank != 1) {
        sm.b[tid - 32] = *cluster.map_shared_rank(&sm.b[tid - 32], 1);
    }
    __syncthreads();

    // ---- the recurrence: gdn_step_norm_multi_kernel's sums
    for (int t = 0; t < T; ++t) {
        const float g = __expf(sm.g[t]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sm.kn[t][rg * RPG + r], kv);
        sm.red_kv[rg][lane] = kv;
        __syncthreads();
        const float kv_col = sm.red_kv[0][lane] + sm.red_kv[1][lane] + sm.red_kv[2][lane] + sm.red_kv[3][lane];
        const float delta = (sm.v[t][lane] - g * kv_col) * sm.b[t];
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sm.kn[t][rg * RPG + r] * delta);
            o = fmaf(s[r], sm.qn[t][rg * RPG + r], o);
        }
        sm.red_o[rg][lane] = o;
        __syncthreads();
        if (rg == 0) {
            const float oc = (sm.red_o[0][lane] + sm.red_o[1][lane] + sm.red_o[2][lane] + sm.red_o[3][lane]) *
                             rsqrtf((float) S);
            float sq_part = oc * oc;
            for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part += __shfl_xor_sync(0xffffffffu, sq_part, o2);
            sm.oc[t][lane] = oc;
            if (lane == 0) sm.wsum[t] = sq_part;
        }
    }
    cluster.sync();   // every rank's sums of squares

    // ---- the gated RMS norm and the q8_1 of the output: one warp, one q8_1 block per token
    if (rg == 0) {
        for (int t = 0; t < T; ++t) {
            const float ss = *cluster.map_shared_rank(&sm.wsum[t], 0) + *cluster.map_shared_rank(&sm.wsum[t], 1) +
                             *cluster.map_shared_rank(&sm.wsum[t], 2) + *cluster.map_shared_rank(&sm.wsum[t], 3);
            const float scale = rsqrtf(ss / (float) S + a.eps);
            const float zz = a.z[(size_t) t * value_dim + head * S + col];
            const float yv = sm.oc[t][lane] * scale * gam * (1.0f / (1.0f + __expf(-zz)));
            if (a.y) a.y[(size_t) t * value_dim + head * S + col] = yv;
            float amax = fabsf(yv), sum = yv;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o, 32));
                sum += __shfl_xor_sync(0xffffffffu, sum, o, 32);
            }
            const float d = q8_1_finite(div_approx_ftz(amax, 127.0f));
            Q81* blk = reinterpret_cast<Q81*>(a.yq) + (size_t) t * (value_dim / 32) + head * (S / 32) + rank;
            blk->qs[lane] = q8_1_quant_fm(yv, d, amax);
            if (lane == 0) blk->ds = q8_1_ds(d, sum);
        }
    }
    cluster.sync();   // no block leaves while another reads its shared memory
}

template<int MAXT>
__global__ void __cluster_dims__(4, 1, 1) __launch_bounds__(S) rec_kernel(const DecRecArgs a, int T) {
    __shared__ RecSmem sm;
    rec_block<MAXT>(a, T, blockIdx.x / 4, blockIdx.x % 4, sm, cg::this_cluster());
}

}  // namespace

void dec_gdn_rec(const DecRecArgs& a, int n_tok, void* stream) {
    if (n_tok < 1 || n_tok > kDecMaxT || a.n_embd != 32 * 8 * NJ || a.h_k <= 0 || a.h_v % a.h_k != 0) {
        std::fprintf(stderr, "dec_gdn_rec: invalid arguments\n");
        std::exit(1);
    }
    const dim3 grid((unsigned) (a.h_v * 4)), block(S);
    const cudaStream_t s = (cudaStream_t) stream;
    if (n_tok <= 2) launch_pdl(rec_kernel<2>, grid, block, 0, s, a, n_tok);
    else if (n_tok <= 4) launch_pdl(rec_kernel<4>, grid, block, 0, s, a, n_tok);
    else launch_pdl(rec_kernel<kDecMaxT>, grid, block, 0, s, a, n_tok);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "dec_gdn_rec: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

}  // namespace strata::kernels
