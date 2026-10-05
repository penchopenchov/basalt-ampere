// src/kernels/hc_fused_parity.cpp - hc_fused_read against the kernels it replaces (fused_gr_read_multi, then the
// MMVF router GEMV and the native top-10), bit for bit on every output, with real geometry and random inputs; then
// the time of both with the weights hot in L2 and cold (L2 overwritten before every read).
//
//     build/hc_fused_parity [iters] [T min] [T max]
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/hc_fused.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/pdl.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace K = strata::kernels;

static uint16_t bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setenv("STRATA_PDL", "1", 0);
    setenv("STRATA_GR_DEPTH", "3", 0);
    const int iters = argc > 1 ? std::atoi(argv[1]) : 200;
    const int t_lo = argc > 2 ? std::atoi(argv[2]) : 1, t_hi = argc > 3 ? std::atoi(argv[3]) : 8;
    if (!K::hc_fused_supported()) {
        std::printf("hc_fused_parity: the fused read does not run on this device\n");
        return 1;
    }
    K::fused_gr_check();
    const int N = 2560, HC = 4, D = N * HC, LR = 320, NE = 512, KT = 10, TM = 8;
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto fill = [&](std::vector<float>& v, float sc) { for (auto& x : v) x = sc * nd(rng); };
    std::vector<float> R((size_t) TM * D), bo((size_t) TM * N), inj((size_t) TM * HC), wn(D);
    fill(R, 1.0f); fill(bo, 0.5f); fill(inj, 1.0f); fill(wn, 0.3f);
    for (auto& x : wn) x += 1.0f;
    std::vector<uint16_t> wd((size_t) LR * D), wu((size_t) D * LR), wi((size_t) HC * D), wr((size_t) NE * N);
    for (auto& x : wd) x = bf16(0.02f * nd(rng));
    for (auto& x : wu) x = bf16(0.05f * nd(rng));
    for (auto& x : wi) x = bf16(0.02f * nd(rng));
    for (auto& x : wr) x = bf16(0.05f * nd(rng));
    // a few exact ties in the router rows: two experts with the same weights
    std::memcpy(wr.data() + (size_t) 77 * N, wr.data() + (size_t) 300 * N, (size_t) N * 2);

    struct Out {
        float *R, *xn, *lo, *rs, *io, *mix, *logits, *w;
        int32_t* ids;
    };
    auto alloc_out = [&](Out& o) {
        cudaMalloc((void**) &o.R, (size_t) TM * D * 4);
        cudaMalloc((void**) &o.xn, (size_t) TM * D * 4);
        cudaMalloc((void**) &o.lo, (size_t) TM * LR * 4);
        cudaMalloc((void**) &o.rs, (size_t) TM * HC * 4);
        cudaMalloc((void**) &o.io, (size_t) TM * HC * 4);
        cudaMalloc((void**) &o.mix, (size_t) TM * N * 4);
        cudaMalloc((void**) &o.logits, (size_t) TM * NE * 4);
        cudaMalloc((void**) &o.w, (size_t) TM * KT * 4);
        cudaMalloc((void**) &o.ids, (size_t) TM * KT * 4);
    };
    Out o[2];
    alloc_out(o[0]);
    alloc_out(o[1]);
    float *dbo, *dinj, *dwn;
    uint16_t *dwd, *dwu, *dwi, *dwr;
    cudaMalloc((void**) &dbo, bo.size() * 4);
    cudaMalloc((void**) &dinj, inj.size() * 4);
    cudaMalloc((void**) &dwn, wn.size() * 4);
    cudaMalloc((void**) &dwd, wd.size() * 2);
    cudaMalloc((void**) &dwu, wu.size() * 2);
    cudaMalloc((void**) &dwi, wi.size() * 2);
    cudaMalloc((void**) &dwr, wr.size() * 2);
    cudaMemcpy(dbo, bo.data(), bo.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dinj, inj.data(), inj.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dwn, wn.data(), wn.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dwd, wd.data(), wd.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dwu, wu.data(), wu.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dwi, wi.data(), wi.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dwr, wr.data(), wr.size() * 2, cudaMemcpyHostToDevice);
    const size_t evict_bytes = (size_t) 192 << 20;
    void* devict = nullptr;
    if (cudaMalloc(&devict, evict_bytes) != cudaSuccess) {
        std::printf("hc_fused_parity: no room for the L2 eviction buffer\n");
        return 1;
    }
    float* dRbak = nullptr;   // R and bo rewritten after the eviction: in the decode they come from the previous kernel
    cudaMalloc((void**) &dRbak, R.size() * 4);
    cudaMemcpy(dRbak, R.data(), R.size() * 4, cudaMemcpyHostToDevice);
    cudaStream_t s;
    cudaStreamCreate(&s);
    // w_norm too: the previous read prefetches it (HcFusedArgs::pf) in the decode
    float* dwnbak = nullptr;
    cudaMalloc((void**) &dwnbak, wn.size() * 4);
    cudaMemcpy(dwnbak, wn.data(), wn.size() * 4, cudaMemcpyHostToDevice);
    auto evict = [&](int f, int i) {
        cudaMemsetAsync(devict, i & 0xff, evict_bytes, s);
        cudaMemcpyAsync(o[f].R, dRbak, R.size() * 4, cudaMemcpyDeviceToDevice, s);
        cudaMemcpyAsync(dbo, dRbak, bo.size() * 4, cudaMemcpyDeviceToDevice, s);
        cudaMemcpyAsync(dwn, dwnbak, wn.size() * 4, cudaMemcpyDeviceToDevice, s);
    };
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    // f = 0: the reference kernels; f = 1: the fused read
    auto run = [&](int f, int T, bool apply, bool inject, bool router) {
        Out& x = o[f];
        K::PdlScope ps(true);
        if (f == 0) {
            K::FusedGrArgs a[TM];
            for (int t = 0; t < T; ++t) {
                a[t].R = x.R + (size_t) t * D; a[t].R_out = x.R + (size_t) t * D; a[t].apply = apply;
                a[t].bo_prev = dbo + (size_t) t * N; a[t].inj_prev = dinj + (size_t) t * HC;
                a[t].w_norm = dwn; a[t].w_down = dwd; a[t].w_up = dwu; a[t].w_inject = inject ? dwi : nullptr;
                a[t].lo = x.lo + (size_t) t * LR; a[t].rs = x.rs + (size_t) t * HC;
                a[t].inject_out = x.io + (size_t) t * HC; a[t].mixed = x.mix + (size_t) t * N;
            }
            K::fused_gr_read_multi(a, T, x.xn, s);
            if (router) {
                K::bf16_gemv_fp32_mmvf_multi(x.mix, N, dwr, x.logits, NE, N, NE, T, s);
                K::native_router_top10_multi(x.logits, x.ids, x.w, T, s);
            }
        } else {
            K::HcFusedArgs a;
            a.T = T; a.R = x.R; a.R_out = x.R; a.apply = apply; a.bo_prev = dbo; a.inj_prev = dinj;
            a.w_norm = dwn; a.w_down = dwd; a.w_up = dwu; a.w_inject = inject ? dwi : nullptr;
            a.lo = x.lo; a.rs = x.rs; a.inject_out = x.io; a.mixed = x.mix; a.xn = x.xn;
            if (router) { a.w_router = dwr; a.logits = x.logits; a.ids = x.ids; a.weights = x.w; }
            K::hc_fused_read(a, s);
        }
    };
    auto reset = [&](int f) {
        Out& x = o[f];
        cudaMemcpy(x.R, R.data(), R.size() * 4, cudaMemcpyHostToDevice);
        cudaMemset(x.lo, 0xff, (size_t) TM * LR * 4);
        cudaMemset(x.rs, 0xff, (size_t) TM * HC * 4);
        cudaMemset(x.io, 0xff, (size_t) TM * HC * 4);
        cudaMemset(x.mix, 0xff, (size_t) TM * N * 4);
        cudaMemset(x.logits, 0xff, (size_t) TM * NE * 4);
        cudaMemset(x.w, 0xff, (size_t) TM * KT * 4);
        cudaMemset(x.ids, 0xff, (size_t) TM * KT * 4);
    };
    auto fetch = [&](int f, int T) {
        Out& x = o[f];
        std::vector<float> v((size_t) T * (D + LR + HC + HC + N + NE + KT + KT));
        float* p = v.data();
        cudaMemcpy(p, x.R, (size_t) T * D * 4, cudaMemcpyDeviceToHost); p += (size_t) T * D;
        cudaMemcpy(p, x.lo, (size_t) T * LR * 4, cudaMemcpyDeviceToHost); p += (size_t) T * LR;
        cudaMemcpy(p, x.rs, (size_t) T * HC * 4, cudaMemcpyDeviceToHost); p += (size_t) T * HC;
        cudaMemcpy(p, x.io, (size_t) T * HC * 4, cudaMemcpyDeviceToHost); p += (size_t) T * HC;
        cudaMemcpy(p, x.mix, (size_t) T * N * 4, cudaMemcpyDeviceToHost); p += (size_t) T * N;
        cudaMemcpy(p, x.logits, (size_t) T * NE * 4, cudaMemcpyDeviceToHost); p += (size_t) T * NE;
        cudaMemcpy(p, x.w, (size_t) T * KT * 4, cudaMemcpyDeviceToHost); p += (size_t) T * KT;
        cudaMemcpy(p, x.ids, (size_t) T * KT * 4, cudaMemcpyDeviceToHost);
        return v;
    };
    static const char* const part_name[8] = {"R", "lo", "rs", "inject", "mixed", "logits", "weights", "ids"};
    auto part_of = [&](size_t i, int T) {
        const size_t sz[8] = {(size_t) T * D, (size_t) T * LR, (size_t) T * HC, (size_t) T * HC, (size_t) T * N,
                              (size_t) T * NE, (size_t) T * KT, (size_t) T * KT};
        for (int k = 0; k < 8; ++k) {
            if (i < sz[k]) return k;
            i -= sz[k];
        }
        return 7;
    };

    int failures = 0;
    for (int T = t_lo; T <= t_hi; ++T)
        for (int mode = 0; mode < 3; ++mode) {   // 0: attention-side read, 1: FFN read + router, 2: the head's read
            const bool inject = mode != 2, router = mode == 1;
            for (int apply = 0; apply < 2; ++apply) {
                std::vector<float> v[2];
                for (int f = 0; f < 2; ++f) {
                    reset(f);
                    run(f, T, apply, inject, router);
                    if (cudaStreamSynchronize(s) != cudaSuccess) {
                        std::printf("kernel failed: %s\n", cudaGetErrorString(cudaGetLastError()));
                        return 1;
                    }
                    v[f] = fetch(f, T);
                }
                // repeated launches (the counters must come back to zero) give the same bits
                reset(1);
                for (int i = 0; i < 3; ++i) {
                    cudaMemcpyAsync(o[1].R, R.data(), (size_t) T * D * 4, cudaMemcpyHostToDevice, s);
                    run(1, T, apply, inject, router);
                }
                cudaStreamSynchronize(s);
                const std::vector<float> v2 = fetch(1, T);
                size_t ndiff = 0, first = 0;
                for (size_t i = 0; i < v[0].size(); ++i) {
                    if (!inject && part_of(i, T) == 3) continue;
                    if (!router && part_of(i, T) >= 5) continue;
                    uint32_t u0, u1, u2;
                    std::memcpy(&u0, &v[0][i], 4);
                    std::memcpy(&u1, &v[1][i], 4);
                    std::memcpy(&u2, &v2[i], 4);
                    if (u0 != u1 || u1 != u2) {
                        if (ndiff++ == 0) first = i;
                    }
                }
                if (ndiff) {
                    ++failures;
                    std::printf("T %d mode %d apply %d: %zu values differ, first in %s (index %zu): %a vs %a\n", T, mode,
                                apply, ndiff, part_name[part_of(first, T)], first, v[0][first], v[1][first]);
                }
            }
        }
    std::printf("hc_fused_parity: %d differing cases (T %d..%d, 3 modes, with and without the pending write)\n",
                failures, t_lo, t_hi);

    // time: hot (back to back) and cold (192 MB written before each read, then R, bo and w_norm rewritten as the
    // previous kernels leave them; the read alone timed)
    for (int hold = 1; hold >= 0; --hold) {
    K::hc_fused_set_knobs(hold);
    std::printf("weight streams %s:\n", hold ? "after the norm's inputs (hold)" : "at launch");
    for (int T = t_lo; T <= t_hi; ++T)
        for (int router = 0; router < 2; ++router) {
            const double bytes = 2.0 * LR * D * 2 + HC * D * 2 + (router ? (double) NE * N * 2 : 0.0);
            double hot[2], cold[2];
            for (int f = 0; f < 2; ++f) {
                for (int i = 0; i < 10; ++i) run(f, T, true, true, router);
                cudaEventRecord(e0, s);
                for (int i = 0; i < iters; ++i) run(f, T, true, true, router);
                cudaEventRecord(e1, s);
                cudaEventSynchronize(e1);
                float ms = 0;
                cudaEventElapsedTime(&ms, e0, e1);
                hot[f] = 1e3 * ms / iters;
                double tot = 0;
                const int ci = iters / 4 > 10 ? iters / 4 : 10;
                for (int i = 0; i < ci; ++i) {
                    evict(f, i);
                    cudaEventRecord(e0, s);
                    run(f, T, true, true, router);
                    cudaEventRecord(e1, s);
                    cudaEventSynchronize(e1);
                    cudaEventElapsedTime(&ms, e0, e1);
                    tot += ms;
                }
                cold[f] = 1e3 * tot / ci;
            }
            std::printf("T %d %-12s | hot: old %6.1f us  fused %6.1f us (%5.0f GB/s) | cold: old %6.1f us  fused %6.1f us "
                        "(%5.0f GB/s)\n",
                        T, router ? "read+router" : "read", hot[0], hot[1], bytes / hot[1] / 1e3, cold[0], cold[1],
                        bytes / cold[1] / 1e3);
        }
    }
    K::hc_fused_set_knobs(1);

    // per-phase times of single reads (globaltimer, relative to the first CTA's start)
    {
        int sms = 0;
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
        unsigned long long* ddbg = nullptr;
        cudaMalloc((void**) &ddbg, (size_t) sms * 16 * 8);
        std::vector<unsigned long long> h((size_t) sms * 16);
        static const char* const ph[8] = {"start", "dep-wait", "input", "role-done", "rtr-in", "rtr-done", "top10", "exit"};
        for (int T : {1, 2, 3, 8}) {
            if (T < t_lo || T > t_hi) continue;
            for (int cold = 0; cold < 2; ++cold) {
                for (int i = 0; i < 3; ++i) run(1, T, true, true, true);
                if (cold) evict(1, 1);
                cudaMemsetAsync(ddbg, 0, (size_t) sms * 128, s);
                K::hc_fused_set_debug(ddbg);
                run(1, T, true, true, true);
                K::hc_fused_set_debug(nullptr);
                cudaStreamSynchronize(s);
                cudaMemcpy(h.data(), ddbg, h.size() * 8, cudaMemcpyDeviceToHost);
                unsigned long long t0 = ~0ull;
                for (int b = 0; b < sms; ++b) if (h[b * 16] && h[b * 16] < t0) t0 = h[b * 16];
                std::printf("T %d %s phases (us, min/max over CTAs; down CTAs 0..80 | up CTAs 81..):\n", T, cold ? "cold" : "hot");
                for (int grp = 0; grp < 2; ++grp) {
                    std::printf("  %s:", grp ? "up  " : "down");
                    for (int i = 0; i < 8; ++i) {
                        double mn = 1e30, mx = -1;
                        for (int b = grp ? 81 : 0; b < (grp ? sms : 81); ++b) {
                            if (!h[b * 16 + i]) continue;
                            const double v = (double) (h[b * 16 + i] - t0) / 1e3;
                            mn = v < mn ? v : mn; mx = v > mx ? v : mx;
                        }
                        if (mx >= 0) std::printf(" %s %.1f/%.1f", ph[i], mn, mx);
                    }
                    std::printf("\n");
                }
                for (int b = 0; b < sms; ++b)
                    if (h[b * 16 + 6])
                        std::printf("  top-10 CTA %d: ticket %.2f loads %.2f softmax %.2f select %.2f us after its rows\n", b,
                                    (h[b * 16 + 8] - h[b * 16 + 5]) / 1e3, (h[b * 16 + 9] - h[b * 16 + 8]) / 1e3,
                                    (h[b * 16 + 10] - h[b * 16 + 9]) / 1e3, (h[b * 16 + 6] - h[b * 16 + 10]) / 1e3);
            }
        }
    }
    return failures ? 1 : 0;
}
