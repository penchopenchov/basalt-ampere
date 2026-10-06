// src/kernels/ple_win_parity.cpp - native_ple_window against the verify window's per-row chain (native_gr_post,
// ple_block_projected, ple_history_advance, the history copy), bitwise, for windows of 1..7 rows; then both forms
// captured as graphs and timed at T = 1..4.
//   ple_win_parity [iters]
#include "basalt/kernels/native_gr_postops.hpp"
#include "basalt/kernels/native_ple_postops.hpp"
#include "basalt/kernels/ngram.hpp"
#include "basalt/kernels/ple.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = basalt::kernels;

namespace {
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 256), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 256), "memset");
    return p;
}
template <typename T> void up(T* d, const std::vector<T>& h) { ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "up"); }
template <typename T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "down");
    return h;
}
size_t ndiff(const std::vector<float>& a, const std::vector<float>& b) {
    size_t c = 0;
    for (size_t i = 0; i < a.size(); ++i) c += std::memcmp(&a[i], &b[i], 4) != 0;
    return c;
}
}  // namespace

int main(int argc, char** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 500;
    constexpr int N = k::NG_N_EMBD, H = k::NG_HC, D = k::NG_HC_DIM, HS = k::NG_HIST * k::NG_HC_DIM, MT = 7;
    std::mt19937 rng(23);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto randv = [&](size_t n, float sc, float off) { std::vector<float> v(n); for (auto& x : v) x = off + sc * nd(rng); return v; };
    std::vector<uint16_t> conv(4 * (size_t) D);
    for (auto& w : conv) w = __half_as_ushort(__float2half(0.3f * nd(rng)));
    const auto h_R = randv((size_t) MT * D, 1.0f, 0.0f), h_bo = randv((size_t) MT * N, 0.5f, 0.0f),
               h_inj = randv((size_t) MT * H, 2.0f, 0.0f), h_key = randv((size_t) MT * D, 0.2f, 0.0f),
               h_val = randv((size_t) MT * N, 0.5f, 0.0f), h_hist = randv((size_t) HS, 1.0f, 0.0f),
               h_nk = randv(D, 0.1f, 1.0f), h_nq = randv(D, 0.1f, 1.0f), h_nc = randv(D, 0.1f, 1.0f);
    float *R = dalloc<float>((size_t) MT * D), *bo = dalloc<float>((size_t) MT * N), *inj = dalloc<float>((size_t) MT * H),
          *key = dalloc<float>((size_t) MT * D), *val = dalloc<float>((size_t) MT * N), *hist = dalloc<float>(HS),
          *snap = dalloc<float>((size_t) MT * HS), *nk = dalloc<float>(D), *nq = dalloc<float>(D), *nc = dalloc<float>(D),
          *win = dalloc<float>(2 * (size_t) MT * D);
    uint16_t* cw = dalloc<uint16_t>(4 * (size_t) D);
    up(bo, h_bo); up(inj, h_inj); up(key, h_key); up(val, h_val); up(nk, h_nk); up(nq, h_nq); up(nc, h_nc); up(cw, conv);
    void* scratch = nullptr;
    ck(cudaMalloc(&scratch, k::ple_block_scratch_bytes() + (size_t) D * 4 + 256), "scratch");
    float* normalized = (float*) ((uint8_t*) scratch + k::ple_block_scratch_bytes());
    k::PleWeights w;
    w.norm_key = nk; w.norm_query = nq; w.norm_conv = nc; w.conv1d_f16 = cw;
    cudaStream_t st;
    ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");

    auto reference = [&](int T) {   // the verify window's per-row chain
        for (int t = 0; t < T; ++t) {
            float* Rt = R + (size_t) t * D;
            k::native_gr_post(Rt, bo + (size_t) t * N, inj + (size_t) t * H, Rt, N, H, st);
            k::PleOut po;
            po.normalized = normalized;
            po.result = Rt;
            k::ple_block_projected(key + (size_t) t * D, val + (size_t) t * N, Rt, hist, w, po, scratch, st);
            k::ple_history_advance(hist, normalized, st);
            ck(cudaMemcpyAsync(snap + (size_t) t * HS, hist, (size_t) HS * 4, cudaMemcpyDeviceToDevice, st), "snap");
        }
    };
    auto fused = [&](int T) {
        k::PleWindowArgs a;
        a.R = R; a.block_out = bo; a.inject = inj; a.key = key; a.value = val; a.history = hist; a.hist_snap = snap;
        a.scratch = win; a.T = T;
        k::native_ple_window(a, w, st);
    };
    auto reset = [&] {
        ck(cudaDeviceSynchronize(), "sync");
        up(R, h_R); up(hist, h_hist);
        ck(cudaMemsetAsync(snap, 0, (size_t) MT * HS * 4, st), "memset");
        ck(cudaStreamSynchronize(st), "sync");
    };
    int fail = 0;
    for (int T = 1; T <= MT; ++T) {
        std::vector<float> out[2][3];
        for (int v = 0; v < 2; ++v) {
            reset();
            if (v == 0) reference(T); else fused(T);
            ck(cudaStreamSynchronize(st), "sync");
            out[v][0] = down(R, (size_t) T * D);
            out[v][1] = down(hist, HS);
            out[v][2] = down(snap, (size_t) T * HS);
        }
        const size_t d0 = ndiff(out[0][0], out[1][0]), d1 = ndiff(out[0][1], out[1][1]), d2 = ndiff(out[0][2], out[1][2]);
        std::printf("T=%d  result %zu  history %zu  snapshots %zu  differing values%s\n", T, d0, d1, d2,
                    d0 + d1 + d2 ? "  DIFF" : "");
        fail += d0 + d1 + d2 != 0;
        for (int j = 0; j < 3 && T == 2; ++j)
            for (size_t i = 0; i < out[0][j].size(); ++i)
                if (std::memcmp(&out[0][j][i], &out[1][j][i], 4) != 0) {
                    std::printf("   first diff of %d at %zu: %.9g vs %.9g\n", j, i, out[0][j][i], out[1][j][i]);
                    break;
                }
    }
    // microbench: each form captured once per T, replayed
    for (int T = 1; T <= 4; ++T) {
        float ms[2];
        for (int v = 0; v < 2; ++v) {
            reset();
            cudaGraph_t g;
            cudaGraphExec_t ge;
            ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "capture");
            if (v == 0) reference(T); else fused(T);
            ck(cudaStreamEndCapture(st, &g), "end capture");
            ck(cudaGraphInstantiate(&ge, g, 0), "instantiate");
            for (int i = 0; i < 20; ++i) cudaGraphLaunch(ge, st);
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0); cudaEventCreate(&e1);
            cudaEventRecord(e0, st);
            for (int i = 0; i < iters; ++i) cudaGraphLaunch(ge, st);
            cudaEventRecord(e1, st);
            ck(cudaEventSynchronize(e1), "bench");
            cudaEventElapsedTime(&ms[v], e0, e1);
            cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
        }
        std::printf("bench: T=%d  per-row chain %.2f us  window %.2f us\n", T, 1000.0f * ms[0] / iters, 1000.0f * ms[1] / iters);
    }
    std::printf("%s\n", fail ? "PLE WINDOW PARITY FAILED" : "PLE WINDOW PARITY OK (bitwise)");
    return fail ? 1 : 0;
}
