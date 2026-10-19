// src/kernels/cuda/attn_chain.cu - see include/basalt/kernels/attn_chain.hpp.
//
// One translation unit with the three kernel files whose stage code the chain runs (hc_fused.cu, dense_tc.cu,
// dec_gdn_rec.cu), so their device functions are shared rather than copied.  Built WITHOUT fast math, like them.
#include "hc_fused.cu"
#define smem smem_dtc
#include "dense_tc.cu"
#undef smem
#define div_approx_ftz div_approx_ftz_rec
#include "dec_gdn_rec.cu"
#undef div_approx_ftz
#include "basalt/kernels/attn_chain.hpp"

#include <array>
#include <cstring>
#include <type_traits>

using namespace basalt::kernels;
#include "attn_chain_bound.cuh"
#include "basalt/kernels/stamp_ring.hpp"
#include "attn_chain_kernel.cuh"
#include "chain_shared.hpp"

namespace basalt::kernels {
namespace {
using namespace mkc;

struct ChainDev {
    bool checked = false, ok = false;
    unsigned* cnt = nullptr;                // [2][NCTR] hand-off counters, then the set in use
    float *cpr = nullptr, *mxr = nullptr, *lgr = nullptr;
    uint8_t* xqr = nullptr;
    uint8_t* xqn = nullptr;
    int32_t* ids_all = nullptr;
    float* w_all = nullptr;
    float* abv = nullptr;
};
ChainDev g_chain[16];

ChainDev& chain_dev() {
    int dev = 0;
    cudaGetDevice(&dev);
    static ChainDev none;
    if (dev < 0 || dev >= 16) return none;
    return g_chain[dev];
}

constexpr int kChainCtas = R_CTA0 + R_HV;
using KFn = void (*)(KP);
bool type_ok(int t) { return t == 12 || t == 13 || t == 14 || t == 21 || t == 23; }

// The kernels: one per (window rows, projection types) for the layers' common type triples, and the generic one
// (the types picked at run time) for any other.
#define CHAIN_TRIPLES X(14, 14, 14) X(14, 12, 14) X(13, 14, 14) X(13, 23, 14) X(14, 23, 14) X(12, 12, 14) X(23, 14, 14) \
                      X(14, 13, 14) X(14, 12, 12) X(14, 23, 12) X(14, 13, 12) X(14, 23, 13) X(14, 14, 13) X(13, 14, 13) \
                      X(13, 13, 14) X(13, 12, 14) X(13, 13, 13) X(12, 23, 14) X(14, 21, 23) X(23, 21, 12)
template <int T>
KFn chain_pick_t(int q, int z, int o) {
#define X(a, b, c) if (q == a && z == b && o == c) return k_chain<T, a, b, c>;
    CHAIN_TRIPLES
#undef X
    return k_chain<T, 0, 0, 0>;
}
KFn chain_pick(int T, int q, int z, int o) {
    switch (T) {
        case 1: return chain_pick_t<1>(q, z, o);
        case 2: return chain_pick_t<2>(q, z, o);
        case 3: return chain_pick_t<3>(q, z, o);
        default: return chain_pick_t<4>(q, z, o);
    }
}
}  // namespace

ChainShared chain_shared() {
    const ChainDev& d = chain_dev();
    ChainShared s;
    s.ok = dev_info().ok;
    s.sms = dev_info().sms;
    s.nctr = mkc::NCTR;
    s.hc_launch_bytes = sizeof(HcLaunch);
    s.cnt = d.cnt; s.cpr = d.cpr; s.mxr = d.mxr; s.lgr = d.lgr; s.xqr = d.xqr; s.xqn = d.xqn;
    s.ids_all = d.ids_all; s.w_all = d.w_all; s.abv = d.abv;
    return s;
}

bool chain_hc_prepare(const HcFusedArgs& a, void* L) { return hc_prepare(a, *static_cast<HcLaunch*>(L)); }

void attn_chain_init() {
    ChainDev& d = chain_dev();
#ifdef BASALT_PROFILE
    stamp::ctx();
#endif
    if (d.checked) return;
    d.checked = true;
    const DevInfo& h = dev_info();
    int dev = 0, optin = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    if (!h.ok || h.sms < kChainCtas || optin < SMEM_MAX) {
        std::fprintf(stderr, "attn_chain: device not covered (%d SMs, %d B optin smem; need %d, %d): the unfused kernels run\n", h.sms, optin, kChainCtas, SMEM_MAX);
        return;
    }
    bool ok = cudaMalloc((void**) &d.cnt, (2 * NCTR + 1) * 4) == cudaSuccess &&
              cudaMemset(d.cnt, 0, (2 * NCTR + 1) * 4) == cudaSuccess &&
              cudaMalloc((void**) &d.cpr, (size_t) 4 * NREP * CP_FLOATS(XT) * 4) == cudaSuccess &&
              cudaMalloc((void**) &d.mxr, (size_t) 4 * NREP * XT * N * 4) == cudaSuccess &&
              cudaMalloc((void**) &d.lgr, (size_t) 4 * NREP * XT * NE * 4) == cudaSuccess &&
              cudaMalloc((void**) &d.xqr, (size_t) 2 * NREP * XT * (N / 32) * kQ81) == cudaSuccess &&
              cudaMalloc((void**) &d.xqn, (size_t) XT * (N / 32) * kQ81) == cudaSuccess &&
              cudaMalloc((void**) &d.ids_all, (size_t) kChainCtas * XT * KT * 4) == cudaSuccess &&
              cudaMalloc((void**) &d.w_all, (size_t) kChainCtas * XT * KT * 4) == cudaSuccess &&
              cudaMalloc((void**) &d.abv, (size_t) R_HV * 2 * XT * 4) == cudaSuccess;
    // the shared-memory size of every kernel the dispatch can return
    for (int t = 1; ok && t <= kAttnChainMaxT; ++t) {
        ok = cudaFuncSetAttribute((const void*) chain_pick(t, 0, 0, 0), cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX) == cudaSuccess;
#define X(a, b, c) if (ok) ok = cudaFuncSetAttribute((const void*) chain_pick(t, a, b, c), cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX) == cudaSuccess;
        CHAIN_TRIPLES
#undef X
    }
    cudaDeviceSynchronize();
    d.ok = ok && cudaGetLastError() == cudaSuccess;
    if (!d.ok) {
        std::fprintf(stderr, "attn_chain: FATAL: kernel initialisation failed on a covered device (%s)\n", cudaGetErrorString(cudaGetLastError()));
        std::exit(1);
    }
}

bool attn_chain_supported(const AttnChainArgs& a) {
    const ChainDev& d = chain_dev();
    return d.ok && a.T >= 1 && a.T <= kAttnChainMaxT && type_ok(a.type_q) && type_ok(a.type_z) && type_ok(a.type_o) &&
           a.h1.xq != nullptr && a.h2.w_router != nullptr && a.h1.T == a.T && a.h2.T == a.T &&
           ((uintptr_t) a.wq & 15) == 0 && ((uintptr_t) a.wz & 15) == 0 && ((uintptr_t) a.wo & 15) == 0 && ((uintptr_t) a.rec.state & 15) == 0 &&
           ((uintptr_t) a.h1.xq & 3) == 0;
}

void attn_chain(const AttnChainArgs& a, void* stream) {
    ChainDev& d = chain_dev();
    if (!attn_chain_supported(a)) {
        std::fprintf(stderr, "attn_chain: unsupported arguments or device\n");
        std::exit(1);
    }
    KP p{};
    if (!hc_prepare(a.h1, p.Lc.h1) || !hc_prepare(a.h2, p.Lc.h2)) std::exit(1);
    p.Lc.wq = static_cast<const uint8_t*>(a.wq);
    p.Lc.wz = static_cast<const uint8_t*>(a.wz);
    p.Lc.wo = static_cast<const uint8_t*>(a.wo);
    p.Lc.rec = a.rec;
    p.Lc.geo = make_geo(a.type_q, a.type_z, a.type_o);
    p.Lc.dn1 = p.Lc.dn2 = nullptr;
    CUtensorMap stm_map;
    if (!tmap_2d(&stm_map, a.rec.state, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, (uint64_t) R_HV * S, S, (uint64_t) R_HV * S * 4, S, S)) {
        std::fprintf(stderr, "attn_chain: the state's tensor map failed\n");
        std::exit(1);
    }
    std::memcpy(&p.Lc.stm, &stm_map, sizeof stm_map);
    p.NLAY = 1;
    p.iters = 1;
#ifdef BASALT_PROFILE
    p.sc = stamp::ctx(); p.ord = a.ord;
#endif
    // consecutive launches of a graph alternate between two counter sets; the first zeroes both, the others clear the
    // set the next one uses once their predecessor has completed (see the kernel)
    p.par = (unsigned) (a.ord & 1);
    if (a.ord == 0) cudaMemsetAsync(d.cnt, 0, (2 * NCTR + 1) * 4, (cudaStream_t) stream);
    p.T = a.T;
    p.prt = -3;
    p.cnt = d.cnt;
    p.abv = d.abv;
    p.cpr = d.cpr; p.mxr = d.mxr; p.lgr = d.lgr; p.xqr = d.xqr;
    p.qkv = a.rec.qkv == nullptr ? nullptr : const_cast<float*>(a.rec.qkv);
    p.z = const_cast<float*>(a.rec.z);
    p.ids_all = d.ids_all; p.w_all = d.w_all;
    launch_pdl(chain_pick(a.T, a.type_q, a.type_z, a.type_o), dim3(kChainCtas), dim3(CT), (size_t) SMEM_MAX, (cudaStream_t) stream, p);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "attn_chain: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace basalt::kernels

