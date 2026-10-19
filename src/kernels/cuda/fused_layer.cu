// src/kernels/cuda/fused_layer.cu - see include/basalt/kernels/fused_layer.hpp.
//
// Its own translation unit, so the fused kernels rebuild without the unfused chain's.  It compiles the attention half's
// stage code (the device parts of hc_fused.cu, dense_tc.cu, dec_gdn_rec.cu) and the experts half (fused_moe.cuh, which
// includes iq_kernels.cu); the host registries, the launch preparation and the chain's scratch are attn_chain.cu's, reached
// through chain_shared.hpp.  Built WITHOUT fast math, like them.  The two halves' namespaces and globals are renamed so they
// never meet the attn_chain kernel's own.
#define FZ_DEV_ONLY
#include "hc_fused.cu"
#define smem smem_dtc
#include "dense_tc.cu"
#undef smem
#define div_approx_ftz div_approx_ftz_rec
#include "dec_gdn_rec.cu"
#undef div_approx_ftz
#include "chain_shared.hpp"
namespace basalt::kernels {
bool tmap_2d(CUtensorMap* m, const void* base, CUtensorMapDataType ty, uint64_t cols, uint64_t rows, uint64_t stride, uint32_t box_c, uint32_t box_r);
void attn_chain_init();
}  // namespace basalt::kernels
using namespace basalt::kernels;
#define FZ_CAT2(a, b) a##b
#define FZ_CAT(a, b) FZ_CAT2(a, b)
#ifndef FZ_ONLY_T
#define FZ_ONLY_T 1
#define FZ_P fz_
#define FZ_HOST
#define FZ_NSC mkcf
#define FZ_NSB mkbf
#endif
#define mkc FZ_NSC
#define mkb FZ_NSB
#define g_herr FZ_CAT(FZ_P, g_herr)
#define g_prog FZ_CAT(FZ_P, g_prog)
#define g_lay FZ_CAT(FZ_P, g_lay)
#define mkb_hang FZ_CAT(FZ_P, mkb_hang)
#define mkb_mwait FZ_CAT(FZ_P, mkb_mwait)
#define mkb_now FZ_CAT(FZ_P, mkb_now)
#define mkb_prg FZ_CAT(FZ_P, mkb_prg)
#define mkb_setlay FZ_CAT(FZ_P, mkb_setlay)
#define mkb_try FZ_CAT(FZ_P, mkb_try)
#define syncp FZ_CAT(FZ_P, syncp)
// iq_kernels.cu / dec_moe's externals: this TU\'s copies stay private (the engine uses the ones another TU provides)
#define dec_moe_formats_ok FZ_CAT(FZ_P, dec_moe_formats_ok)
#define dec_moe_glue FZ_CAT(FZ_P, dec_moe_glue)
#define dec_moe_glue_ok FZ_CAT(FZ_P, dec_moe_glue_ok)
#define dec_moe_peer_blocks FZ_CAT(FZ_P, dec_moe_peer_blocks)
#define dec_moe_peer_diag FZ_CAT(FZ_P, dec_moe_peer_diag)
#define dec_moe_peer_launch FZ_CAT(FZ_P, dec_moe_peer_launch)
#define dec_moe_peer_math_test FZ_CAT(FZ_P, dec_moe_peer_math_test)
#define dec_moe_peer_scratch_bytes FZ_CAT(FZ_P, dec_moe_peer_scratch_bytes)
#define dec_moe_route FZ_CAT(FZ_P, dec_moe_route)
#define dec_moe_route_blocks FZ_CAT(FZ_P, dec_moe_route_blocks)
#define dec_moe_stats_text FZ_CAT(FZ_P, dec_moe_stats_text)
#define dec_moe_vram FZ_CAT(FZ_P, dec_moe_vram)
#define dec_moe_vram_scratch_bytes FZ_CAT(FZ_P, dec_moe_vram_scratch_bytes)
#define dec_moe_wait FZ_CAT(FZ_P, dec_moe_wait)
#define embed_type_supported FZ_CAT(FZ_P, embed_type_supported)
#define g_exp_phase FZ_CAT(FZ_P, g_exp_phase)
#define iq_dequant_f16 FZ_CAT(FZ_P, iq_dequant_f16)
#define iq_dequant_f32 FZ_CAT(FZ_P, iq_dequant_f32)
#define iq_dequant_gu_f16 FZ_CAT(FZ_P, iq_dequant_gu_f16)
#define iq_embed_rows FZ_CAT(FZ_P, iq_embed_rows)
#define iq_mmvq FZ_CAT(FZ_P, iq_mmvq)
#define iq_mmvq_mma FZ_CAT(FZ_P, iq_mmvq_mma)
#define iq_mmvq_mma_available FZ_CAT(FZ_P, iq_mmvq_mma_available)
#define iq_mmvq_mma_launches FZ_CAT(FZ_P, iq_mmvq_mma_launches)
#define iq_old_kernels FZ_CAT(FZ_P, iq_old_kernels)
#define iq_row_bytes FZ_CAT(FZ_P, iq_row_bytes)
#define iq_set_mmvq_mma FZ_CAT(FZ_P, iq_set_mmvq_mma)
#define iq_set_old_kernels FZ_CAT(FZ_P, iq_set_old_kernels)
#define iq_supported FZ_CAT(FZ_P, iq_supported)
#define native_expert_grouped FZ_CAT(FZ_P, native_expert_grouped)
#define native_expert_layout FZ_CAT(FZ_P, native_expert_layout)
#define native_expert_scratch_bytes FZ_CAT(FZ_P, native_expert_scratch_bytes)
#define native_expert_set_mode FZ_CAT(FZ_P, native_expert_set_mode)
#define native_expert_supported FZ_CAT(FZ_P, native_expert_supported)
#define native_grouped_set_gu2 FZ_CAT(FZ_P, native_grouped_set_gu2)
#define native_grouped_set_mma_hint FZ_CAT(FZ_P, native_grouped_set_mma_hint)
#define native_grouped_set_v1 FZ_CAT(FZ_P, native_grouped_set_v1)
#define pk_moe_experts FZ_CAT(FZ_P, pk_moe_experts)
#define pk_moe_ok FZ_CAT(FZ_P, pk_moe_ok)
#define pk_moe_scratch_bytes FZ_CAT(FZ_P, pk_moe_scratch_bytes)
#define pk_moe_shared_done FZ_CAT(FZ_P, pk_moe_shared_done)
#define quantize_q8_1_rows FZ_CAT(FZ_P, quantize_q8_1_rows)
#include "basalt/kernels/fused_layer.hpp"
#include <atomic>
#include <chrono>
#include <thread>
#include "basalt/kernels/stamp_ring.hpp"
#include "fused_bound.cuh"
#include "fused_chain.cuh"
// the experts half's TU includes iq_kernels.cu, whose anonymous-namespace Fmt / warp_sum collide with dense_tc / hc_fused
#define Fmt Fmt_iq
#define warp_sum warp_sum_iq
#define FX_NO_MAIN
#include "fused_moe.cuh"
#undef Fmt
#undef warp_sum

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace basalt::kernels {
namespace fzl {
namespace MB = basalt::kernels::FZ_NSB;
using namespace FZ_NSC;

template <int T, int FQ, int FZ, int FO>
__global__ void __launch_bounds__(FZ_NSC::CT, 1) k_layer(const __grid_constant__ FZ_NSC::KP pc, const __grid_constant__ MB::KP pm) {
    if (threadIdx.x == 0 && blockIdx.x < 4 && pm.pfn[blockIdx.x] != 0u) {   // the experts prologue's dependent records into L2
        const char* q = (const char*) pm.pf[blockIdx.x];
        for (unsigned o = 0; o < pm.pfn[blockIdx.x]; o += 128) asm volatile("prefetch.global.L2 [%0];" ::"l"(q + o));
    }
    mkb_setlay((unsigned) pm.tag0);
    unsigned long long* srec = nullptr;   // thread 0's stamp record (profile builds)
    const int vb = FZ_NSC::chain_body<T, FQ, FZ, FO>(pc, srec);
    __syncthreads();
    MB::moe_body<true, T>(pm, pc.cnt + (size_t) pc.par * FZ_NSC::NCTR + FZ_NSC::C_H2UP, (unsigned) FZ_NSC::NUP, vb, srec);
#ifdef BASALT_PROFILE
    __syncthreads();
    if (srec) srec[::basalt::kernels::stamp::kExit] = FZ_NSC::gtime();
#endif
}

#ifdef FZ_HOST
struct FusedDev {
    bool checked = false, ok = false;
    unsigned* cnt = nullptr;                // chain: [2][NCTR] hand-off counters + the ticket word
    float *cpr = nullptr, *mxr = nullptr, *lgr = nullptr;
    uint8_t* xqr = nullptr;
    int32_t* ids_all = nullptr;
    float* w_all = nullptr;
    float* abv = nullptr;
    unsigned* mcnt = nullptr;               // experts: [kFusedMaxLaunches][NCTR]
    uint4 *hq = nullptr, *hqs = nullptr;
    uint2* gs = nullptr;
    unsigned long long *tx = nullptr, *tend = nullptr;
    int32_t* res1_none = nullptr;           // [512] of -1: layers without a peer tier
    unsigned *h_err = nullptr, *h_herr = nullptr;   // host-mapped wait reports (experts side, attention side)
};
FusedDev g_fz[16];
FusedDev& fz_dev() {
    int dev = 0;
    cudaGetDevice(&dev);
    static FusedDev none;
    if (dev < 0 || dev >= 16) return none;
    return g_fz[dev];
}
constexpr int kCtas = FZ_NSC::R_CTA0 + FZ_NSC::R_HV;
bool fz_type_ok(int t) { return t == 12 || t == 13 || t == 14 || t == 21 || t == 23; }
// the experts half's kernels: gate/up types its pair switch has, down types its two down paths (the engine's other
// expert formats keep the separate kernels)
bool gu_type_ok(int t) { return t == 16 || t == 17 || t == 18 || t == 21 || t == 22 || t == 23; }
bool d_type_ok(int t) { return t == 20 || t == 42; }
bool sh_type_ok(int t) { return t == 12 || t == 13 || t == 14 || t == 21 || t == 23; }
#endif
using FzFn = void (*)(FZ_NSC::KP, MB::KP);

// the generic kernels (projection types picked at run time); the per-triple specialisations are a build-time cost left out for now
#define FZ_TRIPLES
template <int T>
FzFn pick_t(int q, int z, int o) {
#define X(a, b, c) if (q == a && z == b && o == c) return k_layer<T, a, b, c>;
    FZ_TRIPLES
#undef X
    return k_layer<T, 0, 0, 0>;
}
static size_t smem_bytes() { return std::max((size_t) FZ_NSC::SMEM_MAX, sizeof(MB::Sm)); }
}  // namespace fzl

// One translation unit per window width T: this one holds only k_layer<FZ_ONLY_T, ...>.  The kernel pointer and the
// module's own device globals (the wait-report pointers) are reached through these two.
const void* FZ_CAT(fz_pick_, FZ_ONLY_T)(int q, int z, int o) { return (const void*) fzl::pick_t<FZ_ONLY_T>(q, z, o); }
bool FZ_CAT(fz_tu_init_, FZ_ONLY_T)(unsigned* de, unsigned* dh) {
    using namespace fzl;
    bool ok = cudaMemcpyToSymbol(MB::g_err, &de, sizeof de) == cudaSuccess && cudaMemcpyToSymbol(g_herr, &dh, sizeof dh) == cudaSuccess;
    const int sm = (int) smem_bytes();
    ok = ok && cudaFuncSetAttribute((const void*) pick_t<FZ_ONLY_T>(0, 0, 0), cudaFuncAttributeMaxDynamicSharedMemorySize, sm) == cudaSuccess;
#define X(a, b, c) if (ok) ok = cudaFuncSetAttribute((const void*) pick_t<FZ_ONLY_T>(a, b, c), cudaFuncAttributeMaxDynamicSharedMemorySize, sm) == cudaSuccess;
    FZ_TRIPLES
#undef X
    return ok;
}

#ifdef FZ_HOST
const void* fz_pick_1(int, int, int);
const void* fz_pick_2(int, int, int);
const void* fz_pick_3(int, int, int);
const void* fz_pick_4(int, int, int);
bool fz_tu_init_1(unsigned*, unsigned*);
bool fz_tu_init_2(unsigned*, unsigned*);
bool fz_tu_init_3(unsigned*, unsigned*);
bool fz_tu_init_4(unsigned*, unsigned*);
namespace fzl {
FzFn pick(int T, int q, int z, int o) {
    switch (T) {
        case 1: return (FzFn) fz_pick_1(q, z, o);
        case 2: return (FzFn) fz_pick_2(q, z, o);
        case 3: return (FzFn) fz_pick_3(q, z, o);
        default: return (FzFn) fz_pick_4(q, z, o);
    }
}
}  // namespace fzl
using namespace fzl;

bool fused_layer_init() {
    FusedDev& d = fz_dev();
#ifdef BASALT_PROFILE
    stamp::ctx();
#endif
    if (d.checked) return d.ok;
    d.checked = true;
    attn_chain_init();
    const ChainShared h = chain_shared();
    int dev = 0, optin = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    if (!h.ok || h.sms < kCtas || (size_t) optin < smem_bytes()) {
        std::fprintf(stderr, "fused_layer: device not covered (%d SMs, %d B optin smem; need %d, %zu): the unfused layer kernels run\n", h.sms, optin, kCtas, smem_bytes());
        return false;
    }
    const size_t ncnt = (2 * FZ_NSC::NCTR + 1) * 4;
    bool ok = cudaMalloc((void**) &d.cnt, ncnt) == cudaSuccess && cudaMemset(d.cnt, 0, ncnt) == cudaSuccess &&
              cudaMalloc((void**) &d.cpr, (size_t) 4 * FZ_NSC::NREP * FZ_NSC::CP_FLOATS(FZ_NSC::XT) * 4) == cudaSuccess &&
              cudaMalloc((void**) &d.mxr, (size_t) 4 * FZ_NSC::NREP * FZ_NSC::XT * N * 4) == cudaSuccess &&
              cudaMalloc((void**) &d.lgr, (size_t) 4 * FZ_NSC::NREP * FZ_NSC::XT * NE * 4) == cudaSuccess &&
              cudaMalloc((void**) &d.xqr, (size_t) 2 * FZ_NSC::NREP * FZ_NSC::XT * (N / 32) * kQ81) == cudaSuccess &&
              cudaMalloc((void**) &d.ids_all, (size_t) kCtas * FZ_NSC::XT * KT * 4) == cudaSuccess &&
              cudaMalloc((void**) &d.w_all, (size_t) kCtas * FZ_NSC::XT * KT * 4) == cudaSuccess &&
              cudaMalloc((void**) &d.abv, (size_t) FZ_NSC::R_HV * 2 * FZ_NSC::XT * 4) == cudaSuccess &&
              cudaMalloc((void**) &d.mcnt, (size_t) kFusedMaxLaunches * MB::NCTR * 4) == cudaSuccess &&
              cudaMalloc((void**) &d.hq, (size_t) kDmCap * MB::HB * 48) == cudaSuccess &&
              cudaMalloc((void**) &d.hqs, (size_t) MB::TMXH * MB::HB * 48) == cudaSuccess &&
              cudaMalloc((void**) &d.gs, (size_t) MB::TMXH * 8) == cudaSuccess &&
              cudaMalloc((void**) &d.tx, 64 * 8) == cudaSuccess && cudaMalloc((void**) &d.tend, 64 * 8) == cudaSuccess &&
              cudaMalloc((void**) &d.res1_none, 512 * 4) == cudaSuccess &&
              cudaMemset(d.res1_none, 0xff, 512 * 4) == cudaSuccess &&
              cudaHostAlloc((void**) &d.h_err, 64, cudaHostAllocMapped) == cudaSuccess &&
              cudaHostAlloc((void**) &d.h_herr, 64, cudaHostAllocMapped) == cudaSuccess;
    if (!ok) { std::fprintf(stderr, "fused_layer: FATAL: scratch allocation failed\n"); std::exit(1); }
    std::memset(d.h_err, 0, 64); std::memset(d.h_herr, 0, 64);
    // a bounded-wait trap kills the context before the engine can look; the host-mapped record is read from a watcher
    // thread and at exit, and printed once
    static std::atomic<bool> printed{false};
    static const volatile unsigned *w_e = d.h_err, *w_c = d.h_herr;
    static auto print_once = [] {
        if ((w_e[0] == 0 && w_c[0] == 0) || printed.exchange(true)) return;
        std::fprintf(stderr, "fused_layer: bounded-wait trap: experts wait line %u cta %u value %u wanted %u layer %u; attention wait kind %u line %u cta %u layer %u wanted %u seen %u\n",
                     w_e[1], w_e[2], w_e[3], w_e[4], w_e[5], w_c[1], w_c[2], w_c[3], w_c[4], w_c[5], w_c[6]);
    };
    std::atexit([] { print_once(); });
    std::thread([] { for (;;) { print_once(); std::this_thread::sleep_for(std::chrono::milliseconds(20)); } }).detach();
    unsigned *de = nullptr, *dh = nullptr;
    cudaHostGetDevicePointer(&de, d.h_err, 0);
    cudaHostGetDevicePointer(&dh, d.h_herr, 0);
    ok = fz_tu_init_1(de, dh) && fz_tu_init_2(de, dh) && fz_tu_init_3(de, dh) && fz_tu_init_4(de, dh);
    cudaDeviceSynchronize();
    d.ok = ok && cudaGetLastError() == cudaSuccess;
    if (!d.ok) { std::fprintf(stderr, "fused_layer: FATAL: kernel initialisation failed on a covered device (%s)\n", cudaGetErrorString(cudaGetLastError())); std::exit(1); }
    return d.ok;
}

bool fused_layer_supported(const FusedLayerArgs& a) {
    const FusedDev& d = fz_dev();
    const AttnChainArgs& c = a.chain;
    return d.ok && c.T >= 1 && c.T <= kAttnChainMaxT && fz_type_ok(c.type_q) && fz_type_ok(c.type_z) && fz_type_ok(c.type_o) &&
           c.h1.xq != nullptr && c.h2.w_router != nullptr && c.h2.xq != nullptr && c.h1.T == c.T && c.h2.T == c.T &&
           ((uintptr_t) c.wq & 15) == 0 && ((uintptr_t) c.wz & 15) == 0 && ((uintptr_t) c.wo & 15) == 0 &&
           ((uintptr_t) c.rec.state & 15) == 0 && ((uintptr_t) c.h1.xq & 3) == 0 && a.ptab != nullptr && a.sg && a.su && a.sd &&
           a.sgi && a.bo && a.ring_m && a.peer_rows && a.cpu_rows && a.peer_done && a.cpu_done && a.ring_m && a.x_m && a.xq_m &&
           a.host_seq_m && a.peer_seq_m && sh_type_ok(a.sgt) && sh_type_ok(a.sut) && (a.sdt == 20 || a.sdt == 8) &&
           gu_type_ok(a.gu_type) && d_type_ok(a.d_type) &&
           native_expert_layout(a.gu_type, a.d_type, N, MB::NFF).bytes > 0 && dec_moe_glue_ok(native_expert_layout(a.gu_type, a.d_type, N, MB::NFF)) &&
           a.ord >= 0 && a.ord < kFusedMaxLaunches;
}

// The one place the kernel parameters are packed from the engine's buffers (a by-value k_run / k_pass changes only this and the launch).
static void fused_pack(const FusedLayerArgs& a, FusedDev& d, void* stream, FZ_NSC::KP& pc, MB::KP& pm) {
    const AttnChainArgs& c = a.chain;
    // the chain half shares the unfused chain's counters and scratch: consecutive launches (fused or not) hand the
    // counter sets over to one another
    attn_chain_init();
    const ChainShared cd = chain_shared();
    if (cd.nctr != FZ_NSC::NCTR || cd.hc_launch_bytes != sizeof(pc.Lc.h1)) {
        std::fprintf(stderr, "fused_layer: the fused chain's counter layout or launch record differs from the unfused chain's\n");
        std::exit(1);
    }
    if (!chain_hc_prepare(c.h1, &pc.Lc.h1) || !chain_hc_prepare(c.h2, &pc.Lc.h2)) std::exit(1);
    pc.Lc.wq = static_cast<const uint8_t*>(c.wq);
    pc.Lc.wz = static_cast<const uint8_t*>(c.wz);
    pc.Lc.wo = static_cast<const uint8_t*>(c.wo);
    pc.Lc.rec = c.rec;
    pc.Lc.geo = FZ_NSC::make_geo(c.type_q, c.type_z, c.type_o);
    pc.Lc.dn1 = pc.Lc.dn2 = nullptr;
    CUtensorMap stm_map;
    if (!tmap_2d(&stm_map, c.rec.state, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, (uint64_t) FZ_NSC::R_HV * S, S, (uint64_t) FZ_NSC::R_HV * S * 4, S, S)) {
        std::fprintf(stderr, "fused_layer: the state's tensor map failed\n");
        std::exit(1);
    }
    std::memcpy(&pc.Lc.stm, &stm_map, sizeof stm_map);
    pc.NLAY = 1; pc.iters = 1;
#ifdef BASALT_PROFILE
    pc.sc = stamp::ctx(); pc.ord = a.ord;
#endif
    pc.par = (unsigned) (a.ord & 1);
    const size_t ncnt = (2 * FZ_NSC::NCTR + 1) * 4;
    if (a.ord == 0) {   // the first launch of the graph zeroes the chain's counters and the experts half's counters and tagged hops
        cudaMemsetAsync(cd.cnt, 0, ncnt, (cudaStream_t) stream);
        cudaMemsetAsync(d.mcnt, 0, (size_t) kFusedMaxLaunches * MB::NCTR * 4, (cudaStream_t) stream);
        cudaMemsetAsync(d.hq, 0, (size_t) kDmCap * MB::HB * 48, (cudaStream_t) stream);
        cudaMemsetAsync(d.hqs, 0, (size_t) MB::TMXH * MB::HB * 48, (cudaStream_t) stream);
        cudaMemsetAsync(d.gs, 0, (size_t) MB::TMXH * 8, (cudaStream_t) stream);
    }
    pc.T = c.T; pc.prt = -3;
    pc.cnt = cd.cnt; pc.abv = cd.abv; pc.cpr = cd.cpr; pc.mxr = cd.mxr; pc.lgr = cd.lgr; pc.xqr = cd.xqr; pc.xqn = cd.xqn;
    pc.qkv = c.rec.qkv == nullptr ? nullptr : const_cast<float*>(c.rec.qkv);
    pc.z = const_cast<float*>(c.rec.z);
    pc.ids_all = cd.ids_all; pc.w_all = cd.w_all;

    MB::LayerM& M = pm.lay1;
    M.L = native_expert_layout(a.gu_type, a.d_type, N, MB::NFF);
    M.ptab = a.ptab;
    M.res1 = a.res1 != nullptr ? a.res1 : d.res1_none;
    M.sg = (const uint8_t*) a.sg; M.su = (const uint8_t*) a.su; M.sd = (const uint8_t*) a.sd; M.sgi = a.sgi;
    M.sgt = a.sgt; M.sut = a.sut; M.sdt = a.sdt;
    M.glue = 1;
    MB::IterM& I = pm.it1;
    I.layer = 0; I.ids = c.h2.ids; I.w = c.h2.weights; I.xq = (const block_q8_1*) c.h2.xq; I.mixed = c.h2.mixed; I.xqr = I.xq; I.xqn = (const block_q8_1*) cd.xqn; I.bo = a.bo;
    pm.lay = nullptr; pm.it = nullptr; pm.iters = 1; pm.T = c.T; pm.delay_ns = 0; pm.mode = 2; pm.ahead = 1; pm.dnpf_ns = 0; pm.guw = 100;
    pm.peer_rows = a.peer_rows; pm.cpu_rows = a.cpu_rows; pm.peer_done = a.peer_done; pm.cpu_done = a.cpu_done;
    pm.cnt = d.mcnt + (size_t) a.ord * MB::NCTR;
    pm.hq = d.hq; pm.hqs = d.hqs; pm.gs = d.gs; pm.tx = d.tx; pm.tend = d.tend; pm.st = nullptr; pm.dbg_parts = nullptr;
    pm.ids_cta = cd.ids_all; pm.w_cta = cd.w_all;
    pm.pfn[0] = pm.pfn[1] = pm.pfn[2] = pm.pfn[3] = 0;
    pm.ring_m = a.ring_m; pm.x_m = a.x_m; pm.xq_m = a.xq_m; pm.host_seq_m = a.host_seq_m; pm.peer_seq_m = a.peer_seq_m;
    pm.ring_want = a.ring; pm.layer_id = a.layer; pm.abort_m = a.abort_m;
    pm.tag0 = (unsigned) a.ord + 1u;
}

void fused_layer(const FusedLayerArgs& a, void* stream) {
    FusedDev& d = fz_dev();
    if (!fused_layer_supported(a)) {
        std::fprintf(stderr, "fused_layer: unsupported arguments or device\n");
        std::exit(1);
    }
    const AttnChainArgs& c = a.chain;
    FZ_NSC::KP pc{};
    MB::KP pm{};
    fused_pack(a, d, stream, pc, pm);
    launch_pdl(pick(c.T, c.type_q, c.type_z, c.type_o), dim3(kCtas), dim3(FZ_NSC::CT), smem_bytes(), (cudaStream_t) stream, pc, pm);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "fused_layer: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

std::string fused_layer_report() {
    const FusedDev& d = fz_dev();
    if (d.h_err == nullptr) return {};
    const volatile unsigned* e = d.h_err; const volatile unsigned* c = d.h_herr;
    char b[256];
    if (e[0] == 0 && c[0] == 0) return {};
    std::snprintf(b, sizeof b, "fused_layer: experts wait line %u cta %u value %u wanted %u layer %u; attention wait kind %u line %u cta %u layer %u wanted %u seen %u",
                  e[1], e[2], e[3], e[4], e[5], c[1], c[2], c[3], c[4], c[5], c[6]);
    return b;
}

#endif  // FZ_HOST

}  // namespace basalt::kernels
