// src/kernels/q5k_head_parity.cpp - q5k_head_mmvq against native_mmvq (Q5_K, the exact layout), bitwise, on random
// Q5_K rows and quantized random columns, 1..8 columns; then a microbench of both at the head's shape.
//   q5k_head_parity [rows] [iters]
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/q5k_head.hpp"
#include "basalt/kernels/sampler.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <random>
#include <vector>

namespace k = basalt::kernels;

namespace {
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
}  // namespace

int main(int argc, char** argv) {
    const int rows = argc > 1 ? std::atoi(argv[1]) : 248320;
    const int iters = argc > 2 ? std::atoi(argv[2]) : 50;
    const int n_in = 2560, MT = 8, blk = 176, bpr = n_in / 256;
    const size_t wbytes = (size_t) rows * bpr * blk;
    std::mt19937 rng(5);
    std::vector<uint8_t> hw(wbytes);
    for (size_t i = 0; i < wbytes; i += 4) {
        const uint32_t r = rng();
        std::memcpy(&hw[i], &r, 4);
    }
    std::normal_distribution<float> nd(0.f, 1.f);
    for (size_t b = 0; b < (size_t) rows * bpr; ++b) {   // finite block scales: d and dmin as small halves
        const __half d = __float2half(std::fabs(nd(rng)) * 0.01f), m = __float2half(std::fabs(nd(rng)) * 0.01f);
        std::memcpy(&hw[b * blk], &d, 2);
        std::memcpy(&hw[b * blk + 2], &m, 2);
    }
    void* w = nullptr;
    ck(cudaMalloc(&w, wbytes), "w");
    ck(cudaMemcpy(w, hw.data(), wbytes, cudaMemcpyHostToDevice), "w up");
    std::vector<float> hx((size_t) MT * n_in);
    for (auto& v : hx) v = nd(rng);
    float* x = nullptr;
    void* xq = nullptr;
    float *y0 = nullptr, *y1 = nullptr;
    ck(cudaMalloc(&x, hx.size() * 4), "x");
    ck(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice), "x up");
    ck(cudaMalloc(&xq, k::native_q8_1_bytes(n_in, MT)), "xq");
    ck(cudaMalloc(&y0, (size_t) MT * rows * 4), "y0");
    ck(cudaMalloc(&y1, (size_t) MT * rows * 4), "y1");
    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");
    k::native_quantize_q8_1(x, xq, n_in, MT, s);
    int fail = 0;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (int nc = 1; nc <= MT; ++nc) {
        ck(cudaMemset(y0, 0xff, (size_t) MT * rows * 4), "m0");
        ck(cudaMemset(y1, 0xee, (size_t) MT * rows * 4), "m1");
        k::native_mmvq(13, w, xq, y0, n_in, rows, nc, s);
        k::q5k_head_mmvq(w, xq, y1, n_in, rows, nc, s);
        ck(cudaStreamSynchronize(s), "sync");
        std::vector<float> a((size_t) nc * rows), b((size_t) nc * rows);
        ck(cudaMemcpy(a.data(), y0, a.size() * 4, cudaMemcpyDeviceToHost), "a");
        ck(cudaMemcpy(b.data(), y1, b.size() * 4, cudaMemcpyDeviceToHost), "b");
        size_t bad = 0;
        for (size_t i = 0; i < a.size(); ++i) bad += std::memcmp(&a[i], &b[i], 4) != 0;
        float ms[2] = {0, 0};
        for (int rep = 0; rep < 2; ++rep) {
            cudaEventRecord(e0, s);
            for (int it = 0; it < iters; ++it) k::native_mmvq(13, w, xq, y0, n_in, rows, nc, s);
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            cudaEventElapsedTime(&ms[0], e0, e1);
            cudaEventRecord(e0, s);
            for (int it = 0; it < iters; ++it) k::q5k_head_mmvq(w, xq, y1, n_in, rows, nc, s);
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            cudaEventElapsedTime(&ms[1], e0, e1);
        }
        const double u0 = 1e3 * ms[0] / iters, u1 = 1e3 * ms[1] / iters;
        std::printf("ncols %d: %s (%zu of %zu differ) | native_mmvq %7.1f us %6.0f GB/s | q5k_head %7.1f us %6.0f GB/s | %.2fx\n",
                    nc, bad ? "FAIL" : "bitwise", bad, a.size(), u0, wbytes / u0 / 1e3, u1, wbytes / u1 / 1e3, u0 / u1);
        if (bad) fail = 1;
    }
    // the fused argmax + probability (one column) against the logits' argmax and row_top_prob's probability
    {
        void* scratch = nullptr;
        int* d_id = nullptr;
        float* d_p = nullptr;
        int* r_id = nullptr;
        float* r_p = nullptr;
        ck(cudaMalloc(&scratch, k::q5k_head_top1_scratch_bytes()), "scratch");
        ck(cudaMemset(scratch, 0, k::q5k_head_top1_scratch_bytes()), "scratch0");
        ck(cudaMalloc(&d_id, 64), "id"); ck(cudaMalloc(&d_p, 64), "p"); ck(cudaMalloc(&r_id, 64), "rid"); ck(cudaMalloc(&r_p, 64), "rp");
        double max_rel = 0;
        int id_bad = 0, logit_bad = 0;
        const int n_rows[] = {rows, 106299, 40525, 1000};
        for (int rr : n_rows) {
            if (rr > rows) continue;
            for (int col = 0; col < MT; ++col) {
                const void* xc = (const char*) xq + (size_t) col * k::native_q8_1_bytes(n_in, 1);
                k::native_mmvq(13, w, xc, y0, n_in, rr, 1, s);
                if (!k::sample_greedy_top_prob(y0, 1, rr, nullptr, r_id, r_p, s)) { std::printf("no reference top-prob\n"); break; }
                if (!k::q5k_head_top1(w, xc, y1, n_in, rr, nullptr, d_id, d_p, scratch, s)) { std::printf("top1 refused\n"); fail = 1; break; }
                ck(cudaStreamSynchronize(s), "sync");
                int a_id, b_id;
                float a_p, b_p;
                ck(cudaMemcpy(&a_id, r_id, 4, cudaMemcpyDeviceToHost), "a"); ck(cudaMemcpy(&b_id, d_id, 4, cudaMemcpyDeviceToHost), "b");
                ck(cudaMemcpy(&a_p, r_p, 4, cudaMemcpyDeviceToHost), "c"); ck(cudaMemcpy(&b_p, d_p, 4, cudaMemcpyDeviceToHost), "d");
                std::vector<float> a((size_t) rr), b((size_t) rr);
                ck(cudaMemcpy(a.data(), y0, a.size() * 4, cudaMemcpyDeviceToHost), "la");
                ck(cudaMemcpy(b.data(), y1, b.size() * 4, cudaMemcpyDeviceToHost), "lb");
                for (size_t i = 0; i < a.size(); ++i) logit_bad += std::memcmp(&a[i], &b[i], 4) != 0;
                id_bad += a_id != b_id;
                max_rel = std::max(max_rel, (double) std::fabs(a_p - b_p) / std::max(1e-30, (double) std::fabs(a_p)));
            }
            float ms[2] = {0, 0};
            for (int rep = 0; rep < 2; ++rep) {
                cudaEventRecord(e0, s);
                for (int it = 0; it < iters; ++it) {
                    k::native_mmvq(13, w, xq, y0, n_in, rr, 1, s);
                    k::sample_greedy_top_prob(y0, 1, rr, nullptr, r_id, r_p, s);
                }
                cudaEventRecord(e1, s);
                cudaEventSynchronize(e1);
                cudaEventElapsedTime(&ms[0], e0, e1);
                cudaEventRecord(e0, s);
                for (int it = 0; it < iters; ++it) k::q5k_head_top1(w, xq, y1, n_in, rr, nullptr, d_id, d_p, scratch, s);
                cudaEventRecord(e1, s);
                cudaEventSynchronize(e1);
                cudaEventElapsedTime(&ms[1], e0, e1);
            }
            std::printf("top1 rows %6d: head + top-prob %7.1f us | fused %7.1f us\n", rr, 1e3 * ms[0] / iters, 1e3 * ms[1] / iters);
        }
        std::printf("top1: logits %s (%d differ), argmax %s (%d differ), probability max rel diff %.2e\n",
                    logit_bad ? "FAIL" : "bitwise", logit_bad, id_bad ? "FAIL" : "equal", id_bad, max_rel);
        if (logit_bad || id_bad || max_rel > 1e-5) fail = 1;
    }
    std::printf("%s\n", fail ? "q5k_head_parity: FAIL" : "q5k_head_parity: OK (bitwise)");
    return fail;
}
