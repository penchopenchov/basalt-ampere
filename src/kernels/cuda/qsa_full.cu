// src/kernels/cuda/qsa_full.cu - see include/basalt/kernels/qsa_full.hpp.
//
// One translation unit with the phase code (dense_tc.cu's tiles, the K/V/indexer row code, the post, scores, top-k and
// attention phases), so their device functions are shared rather than copied.  Built WITHOUT fast math: the attention
// merge is not bitwise under it.
#include "qsa_full_proj.cuh"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/bf16_gemv.hpp"
#include "basalt/kernels/qsa_full.hpp"
#include "basalt/kernels/qsa_select.hpp"
#include "qsa_full_post.cuh"
#include "qsa_full_sel.cuh"
#include "qsa_full_attn.cuh"
#include "basalt/kernels/stamp_ring.hpp"
#ifdef BASALT_PROFILE
#define BASALT_STAMP_PARAM_O , ::basalt::kernels::stamp::Ctx sc, int ord
#define BASALT_STAMP_ARG_O , ::basalt::kernels::stamp::ctx(), a.ord
#else
#define BASALT_STAMP_PARAM_O
#define BASALT_STAMP_ARG_O
#endif

#include <cstdio>
#include <cstdlib>

#ifndef QF_BAR_NS
#define QF_BAR_NS 2000000000ull
#endif

namespace basalt::kernels {
namespace qf {
constexpr int NBAR = 6, CTAS = 170, CTR_STRIDE = 32;
constexpr size_t ARENA = q2sel::TK_SMEM_B > q2attn::ARENA_B ? q2sel::TK_SMEM_B : q2attn::ARENA_B;

__device__ __forceinline__ unsigned long long gtime() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
    return t;
}
// Bounded grid barrier: every CTA adds one (release) and waits until the launch's counter reaches (idx + 1) * gridDim
// (acquire).  A wait past the budget records where in `fail` and traps.
__device__ __forceinline__ void grid_bar(unsigned* ctr, unsigned* fail, int idx) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned want = (unsigned) (idx + 1) * gridDim.x;
        asm volatile("atom.add.release.gpu.global.u32 _, [%0], 1;" ::"l"(ctr) : "memory");
        unsigned seen;
        const unsigned long long t0 = gtime();
        for (;;) {
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(seen) : "l"(ctr) : "memory");
            if ((int) (seen - want) >= 0) break;
            if (gtime() - t0 > (unsigned long long) QF_BAR_NS) {
                fail[0] = 1; fail[1] = blockIdx.x; fail[2] = idx; fail[3] = seen; fail[4] = want;
                __threadfence_system();
                __trap();
            }
            __nanosleep(32);
        }
    }
    __syncthreads();
}

struct ProjArgs {
    int tq, tk, tv, to;
    const uint8_t *wq, *wk, *wv, *wo;
    const uint16_t *wik, *wiq;
    const uint8_t* xmix;
    const float* mixed;
    float *qfull, *kcur, *vcur, *idx_raw, *qidx, *bo;
};
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
template <int TM>
__device__ __forceinline__ void rows_dispatch(const mkb::RowArgs& a, int tk, int tv) {
    if (tk == 14 && tv == 14) mkb::rows_phase<14, 14, TM>(a);
    else if (tk == 23 && tv == 14) mkb::rows_phase<23, 14, TM>(a);
    else if (tk == 14 && tv == 13) mkb::rows_phase<14, 13, TM>(a);
    else __trap();
}
__device__ __forceinline__ void proj_front(const ProjArgs& R, int n, unsigned char* arena) {
    switch (R.tq) {
        case 12: qp_tile_phase<12>(arena, R.wq, R.xmix, R.qfull, 2560, 12288, n); break;
        case 13: qp_tile_phase<13>(arena, R.wq, R.xmix, R.qfull, 2560, 12288, n); break;
        case 14: qp_tile_phase<14>(arena, R.wq, R.xmix, R.qfull, 2560, 12288, n); break;
        case 23: qp_tile_phase<23>(arena, R.wq, R.xmix, R.qfull, 2560, 12288, n); break;
        default: __trap();
    }
    mkb::RowArgs ra{R.wk, R.wv, R.wik, R.wiq, reinterpret_cast<const block_q8_1*>(R.xmix), R.mixed, R.kcur, R.vcur, R.idx_raw, R.qidx, n};
    if (n <= 4) rows_dispatch<4>(ra, R.tk, R.tv);
    else rows_dispatch<8>(ra, R.tk, R.tv);
}
__device__ __forceinline__ void proj_out(const ProjArgs& R, int n, unsigned char* arena, const uint8_t* xg) {
    if (R.to == 12) qp_one_phase<12>(arena, R.wo, xg, R.bo, 6144, 2560, n);
    else if (R.to == 14) qp_one_phase<14>(arena, R.wo, xg, R.bo, 6144, 2560, n);
    else __trap();
}
#endif

__global__ void __launch_bounds__(512, 1) qsa_full_kernel(ProjArgs R, q2post::PostArgs P, q2sel::SelArgs S, q2attn::AttnArgs A, int n,
                                                          unsigned* ctr, unsigned* fail BASALT_STAMP_PARAM_O) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    extern __shared__ __align__(128) unsigned char arena[];
#ifdef BASALT_PROFILE
    // phase ends: the CTA's own (kA + i) and the grid barrier's release (kA + 8 + i); kA + 6 = after the PDL wait
    namespace sp = ::basalt::kernels::stamp;
    unsigned long long* rec = nullptr;
    const unsigned long long t_in = sp::now();
    if (threadIdx.x == 0) rec = sp::open(sc, sp::kQsa, blockIdx.x, gridDim.x, ord, n, t_in, blockIdx.x);
#define QS(i) do { __syncthreads(); if (rec) rec[sp::kA + (i)] = sp::now(); } while (0)
#else
#define QS(i) do { } while (0)
#endif
    pdl_wait();
    QS(6);
    proj_front(R, n, arena);
    QS(0);
    grid_bar(ctr, fail, 0);
    QS(8);
    q2post::qsa_post_phase(P, reinterpret_cast<float*>(arena));
    QS(1);
    grid_bar(ctr, fail, 1);
    QS(9);
    q2sel::qsa_scores_phase(S);
    QS(2);
    grid_bar(ctr, fail, 2);
    QS(10);
    q2sel::qsa_topk_phase(S, reinterpret_cast<int*>(arena));
    QS(3);
    grid_bar(ctr, fail, 3);
    QS(11);
    q2attn::qsa_attn_chunk_phase(A, n, arena);
    QS(4);
    grid_bar(ctr, fail, 4);
    QS(12);
    q2attn::qsa_attn_merge_phase(A, n, arena);
    QS(5);
    grid_bar(ctr, fail, 5);
    QS(13);
    proj_out(R, n, arena, static_cast<const uint8_t*>(A.xq));
#ifdef BASALT_PROFILE
    __syncthreads();
    if (rec) rec[sp::kExit] = sp::now();
#endif
#undef QS
#endif
}

struct Dev {
    bool checked = false, ok = false;
    unsigned* ctr = nullptr;   // [kQsaFullMaxLaunches][CTR_STRIDE]
    unsigned* fail = nullptr;
};
Dev g_dev[16];
Dev& dev_state() {
    int d = 0;
    cudaGetDevice(&d);
    static Dev none;
    return (d < 0 || d >= 16) ? none : g_dev[d];
}
bool qtype_ok(int t) { return t == 12 || t == 13 || t == 14 || t == 23; }
}  // namespace qf

void qsa_full_init() {
    qf::Dev& d = qf::dev_state();
#ifdef BASALT_PROFILE
    stamp::ctx();
#endif
    if (d.checked) return;
    d.checked = true;
    int dev = 0, optin = 0, sms = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    if (sms < qf::CTAS || optin < (int) qf::ARENA) {   // a device the kernel does not cover: the unfused kernels run, said once
        std::fprintf(stderr, "qsa_full: device not covered (%d SMs, %d B optin smem; need %d, %zu): the unfused QSA kernels run\n", sms, optin, qf::CTAS, (size_t) qf::ARENA);
        return;
    }
    bool ok = cudaMalloc((void**) &d.ctr, sizeof(unsigned) * qf::CTR_STRIDE * kQsaFullMaxLaunches) == cudaSuccess &&
              cudaMemset(d.ctr, 0, sizeof(unsigned) * qf::CTR_STRIDE * kQsaFullMaxLaunches) == cudaSuccess &&
              cudaMalloc((void**) &d.fail, 32) == cudaSuccess && cudaMemset(d.fail, 0, 32) == cudaSuccess &&
              cudaFuncSetAttribute((const void*) qf::qsa_full_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   (int) qf::ARENA) == cudaSuccess;
    cudaDeviceSynchronize();
    d.ok = ok && cudaGetLastError() == cudaSuccess;
    if (!d.ok) {   // a covered device that cannot start the kernel is an error, not a silent fallback
        std::fprintf(stderr, "qsa_full: FATAL: kernel initialisation failed on a covered device (%s)\n", cudaGetErrorString(cudaGetLastError()));
        std::exit(1);
    }
}

bool qsa_full_supported(const QsaFullArgs& a) {
    const qf::Dev& d = qf::dev_state();
    const QsaShapes& s = a.shapes;
    return d.ok && a.n >= 1 && a.n <= kQsaFullMaxT && a.ord >= 0 && a.ord < kQsaFullMaxLaunches && qf::qtype_ok(a.tq) &&
           ((a.tk == 14 && a.tv == 14) || (a.tk == 23 && a.tv == 14) || (a.tk == 14 && a.tv == 13)) &&
           (a.to == 12 || a.to == 14) && s.n_head == 24 && s.n_head_kv == 2 && s.head_dim == 256 && s.idx_n_head == 4 &&
           s.idx_dim == 128 && s.idx_block == 4 && s.n_rot == 64 && qsa_dec_attn_supported(a.pools, s) &&
           qsa_dec_fused_supported(s) && a.xmix != nullptr && a.pools.k_q != nullptr && a.pools.v_q != nullptr &&
           ((uintptr_t) a.wq & 15) == 0 && ((uintptr_t) a.wk & 15) == 0 && ((uintptr_t) a.wv & 15) == 0 &&
           ((uintptr_t) a.wo & 15) == 0 && ((uintptr_t) a.xmix & 15) == 0;
}

void qsa_full(const QsaFullArgs& a, void* stream) {
    qf::Dev& d = qf::dev_state();
    if (!qsa_full_supported(a)) {
        std::fprintf(stderr, "qsa_full: unsupported arguments or device\n");
        std::exit(1);
    }
    const QsaShapes& s = a.shapes;
    qf::ProjArgs R{a.tq, a.tk, a.tv, a.to, (const uint8_t*) a.wq, (const uint8_t*) a.wk, (const uint8_t*) a.wv,
                   (const uint8_t*) a.wo, a.wik, a.wiq, (const uint8_t*) a.xmix, a.mixed, a.qfull, a.kcur, a.vcur,
                   a.idx_raw, a.qidx, a.bo};
    q2post::PostArgs P{a.kv, a.qp, a.n, (int) s.n_head, (int) s.idx_n_head, (int) s.n_head_kv, (int) s.page_size,
                       q2post::rope_args(rope_scaling())};
    q2sel::SelArgs S{a.pooled, a.dead, a.qidx, a.steps, a.n, a.max_blocks, a.scores, a.ids, a.cap};
    q2attn::AttnArgs A{};
    A.q = a.qp.qcur; A.p = a.pools; A.ids = a.ids; A.steps = a.steps;
    A.n_kv_heads = (int) s.n_head_kv; A.page_size = (int) s.page_size;
    A.n_chunks = (int) ((a.cap + q2attn::CHUNK - 1) / q2attn::CHUNK); A.cap = (int) a.cap;
    A.stride = (long long) qsa_decode_attn_scratch_floats(a.cap, s);
    A.scale = 1.0f / sqrtf((float) q2attn::HD);
    A.scratch = a.scratch; A.attn = a.attn; A.q_full = a.qp.q_full; A.attn_gated = a.gated; A.xq = const_cast<void*>(a.xmix);
    // each launch of a graph has its own counter; the first zeroes them all
    if (a.ord == 0) cudaMemsetAsync(d.ctr, 0, sizeof(unsigned) * qf::CTR_STRIDE * kQsaFullMaxLaunches, (cudaStream_t) stream);
    launch_pdl(qf::qsa_full_kernel, dim3(qf::CTAS), dim3(512), qf::ARENA, (cudaStream_t) stream, R, P, S, A, a.n,
               d.ctr + (size_t) a.ord * qf::CTR_STRIDE, d.fail BASALT_STAMP_ARG_O);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_full: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace basalt::kernels
