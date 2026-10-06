// src/kernels/dense_tc_parity.cpp - dense_tc (STRATA_DEC_DTC) against native_mmvq, and its column contract.
//
//     build/dense_tc_parity [--bench]
//
// Random GGUF blocks (fp16 scales rewritten as normal values) and normal activations, one q8_1 quantization shared
// by both paths, at the model's projection shapes.  Checks:
//   - every column of a T-column dense_tc call is bitwise equal to the same column of the 8-column call (T = 1..8);
//   - the error against native_mmvq's single-column outputs (the current numerics): max |diff| / max |ref| and the
//     RMS of the diff over the RMS of ref, per case; it fails above 1e-3 RMS.
// --bench: native_mmvq and dense_tc timed per T as CUDA graphs over enough weight copies that the L2 holds none
// (launched as programmatic dependents under STRATA_PDL=1).
#include "strata/kernels/dense_tc.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/pdl.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

struct Case {
    const char* name;
    int type, n_in, n_out;
    int block_bytes;
    int scale_at[2];
};

const Case CASES[] = {
    {"qkv Q6_K", 14, 2560, 10240, 210, {208, -1}},
    {"qkv Q4_K", 12, 2560, 10240, 144, {0, 2}},
    {"qkv Q5_K", 13, 2560, 10240, 176, {0, 2}},
    {"qkv IQ4_XS", 23, 2560, 10240, 136, {0, -1}},
    {"z Q6_K", 14, 2560, 6144, 210, {208, -1}},
    {"z IQ4_XS", 23, 2560, 6144, 136, {0, -1}},
    {"out Q6_K", 14, 6144, 2560, 210, {208, -1}},
    {"out IQ4_XS", 23, 6144, 2560, 136, {0, -1}},
    {"q Q6_K", 14, 2560, 12288, 210, {208, -1}},
    {"kv Q6_K", 14, 2560, 512, 210, {208, -1}},
    {"tail Q4_K", 12, 1024, 1000, 144, {0, 2}},
    {"tail Q6_K", 14, 1536, 999, 210, {208, -1}},
};

uint16_t sane_half(std::mt19937& rng) {
    const uint32_t r = rng();
    return (uint16_t) (((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
}

bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::printf("CUDA: %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

std::vector<uint8_t> make_weights(const Case& c, std::mt19937& rng) {
    const std::size_t wbytes = strata::kernels::native_mmvq_weight_bytes(c.type, c.n_in, c.n_out);
    std::vector<uint8_t> w(wbytes);
    for (auto& b : w) b = (uint8_t) (rng() & 0xff);
    const std::size_t nb = wbytes / (std::size_t) c.block_bytes;
    for (std::size_t k = 0; k < nb; ++k)
        for (int at : c.scale_at)
            if (at >= 0) {
                const uint16_t h = sane_half(rng);
                std::memcpy(&w[k * (std::size_t) c.block_bytes + (std::size_t) at], &h, 2);
            }
    return w;
}

bool parity(const Case& c, cudaStream_t s) {
    constexpr int TM = 8;
    std::mt19937 rng(777u + (unsigned) c.type * 31u + (unsigned) c.n_out);
    const std::vector<uint8_t> w = make_weights(c, rng);
    std::vector<float> x((std::size_t) TM * c.n_in);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : x) v = nd(rng);
    const std::size_t qcol = strata::kernels::native_q8_1_bytes(c.n_in, 1);
    void *dw = nullptr, *xq = nullptr;
    float *dx = nullptr, *yref = nullptr, *yt = nullptr;
    const std::size_t ybytes = (std::size_t) TM * c.n_out * 4;
    if (!ck(cudaMalloc(&dw, w.size()), "malloc") || !ck(cudaMalloc(&xq, qcol * TM), "malloc") ||
        !ck(cudaMalloc(&dx, x.size() * 4), "malloc") || !ck(cudaMalloc(&yref, ybytes), "malloc") ||
        !ck(cudaMalloc(&yt, ybytes * (TM + 1)), "malloc"))
        return false;
    cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();   // pageable copies may still be in flight for a non-blocking stream
    strata::kernels::native_quantize_q8_1(dx, xq, c.n_in, TM, s);
    for (int j = 0; j < TM; ++j)
        strata::kernels::native_mmvq(c.type, dw, (const uint8_t*) xq + j * qcol, yref + (std::size_t) j * c.n_out,
                                     c.n_in, c.n_out, 1, s);
    cudaMemsetAsync(yt, 0xff, ybytes * (TM + 1), s);
    for (int T = 1; T <= TM; ++T)
        strata::kernels::dense_tc(c.type, dw, xq, yt + (std::size_t) T * TM * c.n_out, c.n_in, c.n_out, T, s);
    if (!ck(cudaStreamSynchronize(s), "run")) return false;
    std::vector<float> ref((std::size_t) TM * c.n_out), got((std::size_t) (TM + 1) * TM * c.n_out);
    cudaMemcpy(ref.data(), yref, ybytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(got.data(), yt, got.size() * 4, cudaMemcpyDeviceToHost);
    long long bits = 0, nonfinite = 0;
    const float* g8 = got.data() + (std::size_t) TM * TM * c.n_out;
    for (int T = 1; T <= TM; ++T)
        for (int j = 0; j < T; ++j)
            for (int r = 0; r < c.n_out; ++r) {
                const float a = got[((std::size_t) T * TM + j) * c.n_out + r], b = g8[(std::size_t) j * c.n_out + r];
                if (std::memcmp(&a, &b, 4) != 0) ++bits;
                if (!std::isfinite(a)) ++nonfinite;
            }
    double maxd = 0, maxr = 0, sd = 0, sr = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double d = std::fabs((double) g8[i] - (double) ref[i]);
        maxd = std::max(maxd, d);
        maxr = std::max(maxr, std::fabs((double) ref[i]));
        sd += d * d;
        sr += (double) ref[i] * ref[i];
    }
    const double rms = std::sqrt(sd / std::max(sr, 1e-300));
    const bool ok = bits == 0 && nonfinite == 0 && rms < 1e-3;
    std::printf("%-12s %5d x %5d  T-columns bitwise: %s (%lld diff, %lld nonfinite)  vs native: max|d|/max|ref| "
                "%.3e  rms %.3e  %s\n",
                c.name, c.n_in, c.n_out, bits == 0 ? "yes" : "NO", bits, nonfinite, maxd / std::max(maxr, 1e-30), rms,
                ok ? "PASS" : "FAIL");
    cudaFree(dw); cudaFree(xq); cudaFree(dx); cudaFree(yref); cudaFree(yt);
    return ok;
}

// µs per call: NL calls on NL distinct weight copies (> L2), captured once as a graph, best of `reps` replays
template<typename F>
double time_graph(F&& call, int NL, cudaStream_t s, int reps = 7) {
    cudaGraph_t g;
    cudaGraphExec_t ge;
    cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
    {
        strata::kernels::PdlScope ps(true);   // with STRATA_PDL=1, as the verify graphs launch them
        for (int l = 0; l < NL; ++l) call(l);
    }
    cudaStreamEndCapture(s, &g);
    cudaGraphInstantiate(&ge, g, 0);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaGraphLaunch(ge, s);
    std::vector<float> ms;
    for (int r = 0; r < reps; ++r) {
        cudaEventRecord(e0, s);
        cudaGraphLaunch(ge, s);
        cudaEventRecord(e1, s);
        cudaEventSynchronize(e1);
        float m = 0;
        cudaEventElapsedTime(&m, e0, e1);
        ms.push_back(m);
    }
    std::sort(ms.begin(), ms.end());
    cudaGraphExecDestroy(ge);
    cudaGraphDestroy(g);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return 1000.0 * ms[ms.size() / 2] / NL;
}

void bench(cudaStream_t s) {
    const Case* sel[] = {&CASES[0], &CASES[1], &CASES[3], &CASES[4], &CASES[6], &CASES[8]};
    std::printf("\nbench: us per call (median of 7 graph replays), GB/s of the weight bytes\n");
    for (const Case* c : sel) {
        const std::size_t wb = strata::kernels::native_mmvq_weight_bytes(c->type, c->n_in, c->n_out);
        const int NL = (int) std::max<std::size_t>(8, (400u << 20) / wb);
        std::mt19937 rng(99u);
        const std::vector<uint8_t> w = make_weights(*c, rng);
        uint8_t* dw = nullptr;
        void* xq = nullptr;
        float *dx = nullptr, *y = nullptr;
        cudaMalloc(&dw, wb * NL);
        for (int l = 0; l < NL; ++l) cudaMemcpy(dw + wb * l, w.data(), wb, cudaMemcpyHostToDevice);
        cudaMalloc(&xq, strata::kernels::native_q8_1_bytes(c->n_in, 8));
        cudaMalloc(&dx, (std::size_t) 8 * c->n_in * 4);
        cudaMalloc(&y, (std::size_t) 8 * c->n_out * 4);
        std::vector<float> x((std::size_t) 8 * c->n_in);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : x) v = nd(rng);
        cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        strata::kernels::native_quantize_q8_1(dx, xq, c->n_in, 8, s);
        std::printf("%-12s %5d x %5d  %.1f MB  floor %.2f us at 1.79 TB/s\n", c->name, c->n_in, c->n_out, wb / 1e6,
                    wb / 1.79e6);
        std::printf("   T   native us   GB/s   dtc us   GB/s   speedup\n");
        for (int T = 1; T <= 8; ++T) {
            const double tn = time_graph([&](int l) {
                strata::kernels::native_mmvq(c->type, dw + wb * l, xq, y, c->n_in, c->n_out, T, s);
            }, NL, s);
            const double td = time_graph([&](int l) {
                strata::kernels::dense_tc(c->type, dw + wb * l, xq, y, c->n_in, c->n_out, T, s);
            }, NL, s);
            std::printf("  %2d  %8.2f %7.0f %8.2f %7.0f   %.2fx\n", T, tn, wb / tn / 1e3, td, wb / td / 1e3, tn / td);
        }
        cudaFree(dw); cudaFree(xq); cudaFree(dx); cudaFree(y);
    }
}

}  // namespace

int main(int argc, char** argv) {
    int dev = 0, major = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    if (major < 9) {
        std::printf("dense_tc needs sm_90+; skipped\n");
        return 77;
    }
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    const bool only_bench = argc > 1 && std::string(argv[1]) == "--bench-only";
    bool ok = true;
    if (!only_bench)
        for (const Case& c : CASES) ok = parity(c, s) && ok;
    if (argc > 1 && (std::string(argv[1]) == "--bench" || only_bench)) bench(s);
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
