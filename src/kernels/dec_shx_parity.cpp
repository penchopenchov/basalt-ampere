// src/kernels/dec_shx_parity.cpp - STRATA_DEC_SHX's one launch against shared_expert_multi, bitwise.
//
//     build/dec_shx_parity [--bench]
//
// The reference is the verify window's shared expert: native_quantize_q8_1, native_mmvq gate/up (IQ4_XS),
// native_swiglu_quantize_q8_1, native_mmvq down (IQ4_NL), the BF16 scalar gate and its sigmoid scale.  dec_shx's
// output and gate logits must match it bit for bit at the model's geometry for T = 1..8, with the input's q8_1 rows
// given (native_quantize_q8_1's, and quantize_q8_1_rows' when those are the same bytes: on real activations they
// differ in rare blocks, so the verify window quantizes with native_quantize_q8_1).
// Weights are random bytes with normal fp16 scales, activations random normal.
//
// --bench: both as CUDA graphs over enough layers' distinct weights that the L2 holds none of them, in microseconds
// per layer and GB/s over the weight bytes.
#include "strata/kernels/dec_shx.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/shared_expert.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace strata::kernels;

namespace {

constexpr int N = 2560, F = 640;
constexpr size_t GU_BYTES = (size_t) F * (N / 256) * 136, D_BYTES = (size_t) N * (F / 32) * 18;

#define CK(x)                                                                                    \
    do {                                                                                         \
        const cudaError_t e_ = (x);                                                              \
        if (e_ != cudaSuccess) {                                                                 \
            std::printf("CUDA %s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
            std::exit(1);                                                                        \
        }                                                                                        \
    } while (0)

uint16_t half_bits(float v) {
    const __half h = __float2half(v);
    uint16_t b;
    std::memcpy(&b, &h, 2);
    return b;
}

// random blocks of `block` bytes whose first two bytes are a normal fp16 scale
std::vector<uint8_t> rand_blocks(std::mt19937& rng, size_t bytes, int block, float scale) {
    std::vector<uint8_t> v(bytes);
    for (auto& b : v) b = (uint8_t) rng();
    std::normal_distribution<float> nd(0.0f, scale);
    for (size_t o = 0; o + block <= bytes; o += block) {
        const uint16_t d = half_bits(std::fabs(nd(rng)) + scale * 0.1f);
        std::memcpy(&v[o], &d, 2);
    }
    return v;
}

struct Layer {
    void *gate = nullptr, *up = nullptr, *down = nullptr;
    uint16_t* ginp = nullptr;
};

Layer make_layer(std::mt19937& rng) {
    Layer L;
    const auto g = rand_blocks(rng, GU_BYTES, 136, 0.01f), u = rand_blocks(rng, GU_BYTES, 136, 0.01f);
    const auto d = rand_blocks(rng, D_BYTES, 18, 0.02f);
    std::vector<uint16_t> gi(N);
    std::normal_distribution<float> nd(0.0f, 0.02f);
    for (auto& w : gi) {
        const float f = nd(rng);
        uint32_t b;
        std::memcpy(&b, &f, 4);
        w = (uint16_t) (b >> 16);
    }
    CK(cudaMalloc(&L.gate, GU_BYTES)); CK(cudaMalloc(&L.up, GU_BYTES)); CK(cudaMalloc(&L.down, D_BYTES));
    CK(cudaMalloc(&L.ginp, N * 2));
    CK(cudaMemcpy(L.gate, g.data(), GU_BYTES, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(L.up, u.data(), GU_BYTES, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(L.down, d.data(), D_BYTES, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(L.ginp, gi.data(), N * 2, cudaMemcpyHostToDevice));
    return L;
}

struct Bufs {
    float *x = nullptr, *gate = nullptr, *up = nullptr, *g = nullptr, *out = nullptr, *g2 = nullptr, *out2 = nullptr;
    void *xq = nullptr, *xq2 = nullptr, *q8 = nullptr, *hq = nullptr;
    unsigned int* sync = nullptr;
};

Bufs make_bufs() {
    Bufs b;
    CK(cudaMalloc(&b.x, 8 * N * 4)); CK(cudaMalloc(&b.gate, 8 * F * 4)); CK(cudaMalloc(&b.up, 8 * F * 4));
    CK(cudaMalloc(&b.g, 8 * 4)); CK(cudaMalloc(&b.out, 8 * N * 4)); CK(cudaMalloc(&b.g2, 8 * 4));
    CK(cudaMalloc(&b.out2, 8 * N * 4));
    CK(cudaMalloc(&b.xq, 8 * N / 32 * 36)); CK(cudaMalloc(&b.xq2, 8 * N / 32 * 36));
    CK(cudaMalloc(&b.q8, 8 * N / 32 * 36)); CK(cudaMalloc(&b.hq, 8 * F / 32 * 36));
    CK(cudaMalloc(&b.sync, kShxSyncWords * 4)); CK(cudaMemset(b.sync, 0, kShxSyncWords * 4));
    return b;
}

void reference(const Layer& L, Bufs& b, int n, float* gate, float* up, cudaStream_t s) {
    NativeSharedWeights nw;
    nw.gate_type = 23; nw.up_type = 23; nw.down_type = 20;
    nw.gate_data = L.gate; nw.up_data = L.up; nw.down_data = L.down;
    nw.q8_1 = b.q8;
    shared_expert_multi(n, b.x, nullptr, nw, L.ginp, gate, up, b.g, b.out, N, F, s);
}

ShxArgs args(const Layer& L, Bufs& b, int n, const void* xq) {
    ShxArgs a;
    a.gate_w = L.gate; a.up_w = L.up; a.down_w = L.down; a.gate_inp = L.ginp;
    a.x = b.x; a.xq = xq; a.gate = b.gate; a.up = b.up; a.hq = b.hq; a.g = b.g2; a.out = b.out2; a.sync = b.sync;
    a.n_tok = n;
    return a;
}

bool same(const float* d0, const float* d1, size_t n, const char* what, int T, const char* mode) {
    std::vector<float> a(n), c(n);
    CK(cudaMemcpy(a.data(), d0, n * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(c.data(), d1, n * 4, cudaMemcpyDeviceToHost));
    size_t bad = 0;
    double maxe = 0.0;
    for (size_t i = 0; i < n; ++i)
        if (std::memcmp(&a[i], &c[i], 4) != 0) {
            ++bad;
            maxe = std::max(maxe, (double) std::fabs(a[i] - c[i]));
        }
    if (bad) std::printf("  T=%d %s %s: %zu of %zu differ (max abs %.3g)\n", T, mode, what, bad, n, maxe);
    return bad == 0;
}

bool parity(cudaStream_t s) {
    std::mt19937 rng(1234);
    const Layer L = make_layer(rng);
    Bufs b = make_bufs();
    float *rg = nullptr, *ru = nullptr;
    CK(cudaMalloc(&rg, 8 * F * 4)); CK(cudaMalloc(&ru, 8 * F * 4));
    bool ok = true;
    for (int rep = 0; rep < 12; ++rep)
        for (int T = 1; T <= 8; ++T) {
            std::vector<float> x((size_t) T * N);
            std::normal_distribution<float> nd(0.0f, rep == 3 ? 4.0f : 1.0f);
            for (auto& v : x) v = rep >= 4 ? nd(rng) * std::exp(2.0f * nd(rng)) : nd(rng);   // heavy tails
            CK(cudaMemcpy(b.x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaDeviceSynchronize());   // a pageable copy may still be landing when the call returns
            reference(L, b, T, rg, ru, s);
            native_quantize_q8_1(b.x, b.xq, N, T, s);
            quantize_q8_1_rows(b.x, T, N, b.xq2, s);
            CK(cudaStreamSynchronize(s));
            std::vector<uint8_t> q0((size_t) T * N / 32 * 36), q1(q0.size());
            CK(cudaMemcpy(q0.data(), b.xq, q0.size(), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(q1.data(), b.xq2, q1.size(), cudaMemcpyDeviceToHost));
            const bool rows_same = q0 == q1;
            if (!rows_same) {
                size_t nb = 0;
                for (size_t i = 0; i < q0.size(); ++i) nb += q0[i] != q1[i];
                std::printf("  rep %d T=%d: quantize_q8_1_rows differs from native_quantize_q8_1 in %zu bytes\n", rep, T, nb);
            }
            const void* modes[2] = {b.xq, b.xq2};
            const char* names[2] = {"xq", "rows-xq"};
            for (int m = 0; m < 2; ++m) {
                if (m == 1 && !rows_same) continue;
                CK(cudaMemset(b.out2, 0xff, 8 * N * 4));
                dec_shx(args(L, b, T, modes[m]), s);
                CK(cudaStreamSynchronize(s));
                ok &= same(b.out, b.out2, (size_t) T * N, "out", T, names[m]);
                ok &= same(b.g, b.g2, (size_t) T, "g", T, names[m]);
                ok &= same(rg, b.gate, (size_t) T * F, "gate", T, names[m]);
                ok &= same(ru, b.up, (size_t) T * F, "up", T, names[m]);
            }
            std::vector<unsigned int> sync(kShxSyncWords);
            CK(cudaMemcpy(sync.data(), b.sync, kShxSyncWords * 4, cudaMemcpyDeviceToHost));
            for (unsigned int w : sync)
                if (w != 0) { std::printf("  T=%d: sync words not left zeroed\n", T); ok = false; break; }
        }
    std::printf("dec_shx parity (T 1..8): %s\n", ok ? "BITWISE" : "FAILED");
    return ok;
}

void bench(cudaStream_t s) {
    constexpr int NL = 48;
    std::mt19937 rng(99);
    std::vector<Layer> ls;
    for (int l = 0; l < NL; ++l) ls.push_back(make_layer(rng));
    Bufs b = make_bufs();
    std::vector<float> x(8 * N);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : x) v = nd(rng);
    CK(cudaMemcpy(b.x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaDeviceSynchronize());
    const double mb = (2.0 * GU_BYTES + D_BYTES + N * 2) / 1e6;
    for (int T : {1, 2, 3, 5, 7}) {
        float times[2] = {};
        for (int path = 0; path < 2; ++path) {
            cudaGraph_t g;
            cudaGraphExec_t ge;
            CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
            for (int l = 0; l < NL; ++l) {
                if (path == 0) reference(ls[l], b, T, b.gate, b.up, s);
                else dec_shx(args(ls[l], b, T, b.xq), s);
            }
            CK(cudaStreamEndCapture(s, &g));
            CK(cudaGraphInstantiate(&ge, g, 0));
            cudaEvent_t e0, e1;
            CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
            for (int w = 0; w < 3; ++w) CK(cudaGraphLaunch(ge, s));
            constexpr int REPS = 20;
            CK(cudaEventRecord(e0, s));
            for (int r = 0; r < REPS; ++r) CK(cudaGraphLaunch(ge, s));
            CK(cudaEventRecord(e1, s));
            CK(cudaEventSynchronize(e1));
            float ms = 0;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            times[path] = ms * 1000.0f / (REPS * NL);
            CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g));
        }
        std::printf("bench T=%d: reference %.2f us/layer (%.0f GB/s) | dec_shx %.2f us (%.0f GB/s)  [%.2f MB weights/layer]\n",
                    T, times[0], mb * 1e3 / times[0], times[1], mb * 1e3 / times[1], mb);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    shared_expert_set_native_bf16(true);
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    const bool ok = parity(s);
    if (mode == "--bench") bench(s);
    return ok ? 0 : 1;
}
