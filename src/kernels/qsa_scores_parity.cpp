// src/kernels/qsa_scores_parity.cpp - the decode indexer behind --exp idx50_indexer (bit 1 block scores with the queries in
// registers, bit 2 the top-k as a reduce-scatter on the cluster): scores bitwise against the warp kernel and within
// double-precision error of a CPU reference at the real indexer geometry (tails with and without cells, windows of 1..8
// rows); the top-k selection identical to the cluster kernel's on continuous, tied and equal scores; then a microbench
// line per kernel with the keys warm in L2 and flushed from it.
//   qsa_scores_parity [iters]
#include "basalt/core/diag.hpp"
#include "basalt/kernels/qsa.hpp"
#include "basalt/kernels/qsa_select.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
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
const k::QsaShapes S_ = k::qsa_real_shapes();
const int IQ = (int) S_.idx_n_head, ID = (int) S_.idx_dim, R = (int) S_.idx_block;
int g_fail = 0;
const int NV = 1;
}  // namespace

int main(int argc, char** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 200;
    const int64_t MAXB = 16384;   // 65,536 cells
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> pooled((size_t) MAXB * ID), dead(ID);
    for (auto& x : pooled) x = nd(rng);
    for (auto& x : dead) x = nd(rng);
    float *d_pooled = dalloc<float>(pooled.size()), *d_dead = dalloc<float>(dead.size());
    up(d_pooled, pooled); up(d_dead, dead);
    float* d_q = dalloc<float>((size_t) 8 * IQ * ID);
    int32_t* d_steps = dalloc<int32_t>(8 * k::kStepCount);
    float *d_a = dalloc<float>((size_t) 8 * MAXB), *d_b = dalloc<float>((size_t) 8 * MAXB);
    cudaStream_t st = nullptr;
    ck(cudaStreamCreate(&st), "stream");
    auto& ex = basalt::core::diagnostics().exp;

    struct Case { const char* label; int nq; std::vector<int> nbid; std::vector<int> nkv; };
    const std::vector<Case> cases = {
        {"1 row, 12.5k blocks, tail", 1, {12500}, {50003}},
        {"3 rows, tail and none", 3, {12500, 12501, 12501}, {50003, 50004, 50006}},
        {"8 rows, mixed lengths", 8, {16000, 100, 31, 32, 33, 8000, 12500, 1}, {64001, 401, 126, 129, 134, 32002, 50000, 4}},
        {"2 rows, n_bid 0 and 1", 2, {0, 1}, {1, 5}},
        {"5 rows, 63..65 blocks", 5, {63, 64, 65, 95, 96}, {253, 257, 261, 383, 386}},
    };
    for (const Case& c : cases) {
        std::vector<float> q((size_t) c.nq * IQ * ID);
        for (auto& x : q) x = nd(rng);
        std::vector<int32_t> steps((size_t) c.nq * k::kStepCount, 0);
        for (int i = 0; i < c.nq; ++i) {
            steps[i * k::kStepCount + k::kStepNKv] = c.nkv[i];
            steps[i * k::kStepCount + k::kStepNBid] = c.nbid[i];
            steps[i * k::kStepCount + k::kStepWidth] = 1;   // not the identity: scores are written
        }
        up(d_q, q); up(d_steps, steps);
        std::vector<float> ref_warp, ref_lane;
        bool b1 = true;
        for (int v = 0; v <= NV; ++v) {
            ex["idx50_indexer"] = v;
            ck(cudaMemset(d_a, 0, (size_t) c.nq * MAXB * 4), "z");
            k::qsa_block_scores(d_pooled, d_dead, d_q, d_steps, c.nq, MAXB, S_, d_a, st, -1, true);
            ck(cudaStreamSynchronize(st), "sync");
            auto r = down(d_a, (size_t) c.nq * MAXB);
            if (v == 0) ref_warp = r; else { ref_lane = r; b1 = b1 && std::memcmp(ref_warp.data(), ref_lane.data(), r.size() * 4) == 0; }
        }
        // CPU reference in double over the blocks each query reaches
        double max_abs = 0, max_rel = 0;
        for (int i = 0; i < c.nq; ++i)
            for (int b = 0; b <= c.nbid[i]; ++b) {
                const float* key = b == c.nbid[i] ? dead.data() : pooled.data() + (size_t) b * ID;
                double sc = 0;
                for (int h = 0; h < IQ; ++h) {
                    double d = 0;
                    for (int j = 0; j < ID; ++j) d += (double) key[j] * q[((size_t) i * IQ + h) * ID + j];
                    sc += d > 0 ? d : 0;
                }
                if (b == c.nbid[i] && c.nkv[i] % R != 0) sc += 1e9;
                const double got = ref_warp[(size_t) i * MAXB + b];
                max_abs = std::max(max_abs, std::fabs(got - sc));
                max_rel = std::max(max_rel, std::fabs(got - sc) / std::max(1.0, std::fabs(sc)));
            }
        const bool ok = b1 && max_rel < 1e-5;
        g_fail += !ok;
        std::printf("%-30s new vs warp: %s; vs CPU double: max |d| %.3g, max rel %.3g  %s\n", c.label,
                    b1 ? "bitwise" : "DIFFERS", max_abs, max_rel, ok ? "ok" : "FAIL");
    }
    // microbench: 3 rows at 50k context, keys warm and flushed
    {
        std::vector<float> q((size_t) 3 * IQ * ID);
        for (auto& x : q) x = nd(rng);
        std::vector<int32_t> steps(3 * k::kStepCount, 0);
        for (int i = 0; i < 3; ++i) {
            steps[i * k::kStepCount + k::kStepNKv] = 50003 + i;
            steps[i * k::kStepCount + k::kStepNBid] = 12500 + (i > 0);
            steps[i * k::kStepCount + k::kStepWidth] = 1;
        }
        up(d_q, q); up(d_steps, steps);
        const size_t flush_n = 256u << 20;
        char* flush = dalloc<char>(flush_n);
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        for (int on = 0; on <= NV; ++on)
            for (int cold = 0; cold < 2; ++cold) {
                ex["idx50_indexer"] = on;
                auto call = [&] { k::qsa_block_scores(d_pooled, d_dead, d_q, d_steps, 3, MAXB, S_, d_a, st, -1, true); };
                double tot = 0;
                for (int it = 0; it < iters; ++it) {
                    if (cold) ck(cudaMemsetAsync(flush, it, flush_n, st), "flush");
                    cudaEventRecord(e0, st);
                    call();
                    cudaEventRecord(e1, st);
                    cudaEventSynchronize(e1);
                    float ms = 0;
                    cudaEventElapsedTime(&ms, e0, e1);
                    tot += ms;
                }
                const double us = tot / iters * 1000.0, bytes = 12501.0 * ID * 4;
                std::printf("bench v%d keys %-5s: %7.2f us/call, keys %.1f MB = %.0f GB/s\n", on, cold ? "cold" : "warm", us, bytes / 1e6, bytes / us / 1e3);
            }
    }
    {   // the floor of this timing: a call that returns at once (every selection the identity)
        std::vector<int32_t> steps(3 * k::kStepCount, 0);
        for (int i = 0; i < 3; ++i) { steps[i * k::kStepCount + k::kStepNKv] = 10; steps[i * k::kStepCount + k::kStepWidth] = 10; }
        up(d_steps, steps);
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        double tot = 0;
        for (int it = 0; it < iters; ++it) {
            cudaEventRecord(e0, st);
            k::qsa_block_scores(d_pooled, d_dead, d_q, d_steps, 3, MAXB, S_, d_a, st, -1, true);
            cudaEventRecord(e1, st);
            cudaEventSynchronize(e1);
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            tot += ms;
        }
        std::printf("bench floor (empty launch, event timing): %.2f us\n", tot / iters * 1000.0);
    }
    {   // top-k on real scores: variants identical to the cluster kernel, timed after the scores kernel
        const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, S_);
        std::vector<float> q((size_t) 3 * IQ * ID);
        for (auto& x : q) x = nd(rng);
        const int nqs[3] = {1, 3, 8};
        for (int nq : nqs) {
            std::vector<int32_t> steps((size_t) nq * k::kStepCount, 0);
            for (int i = 0; i < nq; ++i) {
                const int n_kv = 50003 + i;
                steps[i * k::kStepCount + k::kStepNKv] = n_kv;
                steps[i * k::kStepCount + k::kStepNBid] = n_kv / R;
                steps[i * k::kStepCount + k::kStepWidth] = (int) k::qsa_selection_width(n_kv, S_);
            }
            std::vector<float> qq((size_t) nq * IQ * ID);
            for (auto& x : qq) x = nd(rng);
            up(d_q, qq); up(d_steps, steps);
            ex["idx50_indexer"] = 0;
            k::qsa_block_scores(d_pooled, d_dead, d_q, d_steps, nq, MAXB, S_, d_a, st, -1, false);
            int32_t* d_ids = dalloc<int32_t>((size_t) nq * cap);
            std::vector<int32_t> ref;
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0); cudaEventCreate(&e1);
            const int vs[3] = {0, 2, 3};
            for (int v : vs) {
                ex["idx50_indexer"] = v;
                ck(cudaMemset(d_ids, 0xff, (size_t) nq * cap * 4), "m");
                k::qsa_block_topk(d_a, d_steps, nq, MAXB, cap, S_, d_ids, st);
                ck(cudaStreamSynchronize(st), "sync");
                auto h = down(d_ids, (size_t) nq * cap);
                bool same = true;
                if (v == 0) ref = h;
                else
                    for (int i = 0; i < nq; ++i) {
                        const int w = (int) k::qsa_selection_width(50003 + i, S_);
                        same = same && std::equal(h.begin() + (size_t) i * cap, h.begin() + (size_t) i * cap + w, ref.begin() + (size_t) i * cap);
                    }
                double tot = 0;
                for (int it = 0; it < iters; ++it) {
                    cudaEventRecord(e0, st);
                    k::qsa_block_topk(d_a, d_steps, nq, MAXB, cap, S_, d_ids, st);
                    cudaEventRecord(e1, st);
                    cudaEventSynchronize(e1);
                    float ms = 0;
                    cudaEventElapsedTime(&ms, e0, e1);
                    tot += ms;
                }
                g_fail += !same;
                std::printf("topk v%d nq=%d: %7.2f us/call (scores warm)  ids %s\n", v, nq, tot / iters * 1000.0, same ? "identical" : "DIFFER");
            }
        }
    }
    {   // the reduce-scatter top-k against the cluster kernel on continuous, tied and equal scores
        const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, S_);
        int64_t same_n = 0, total = 0;
        float* d_sc = dalloc<float>((size_t) 16 * MAXB);
        int32_t* d_st = dalloc<int32_t>(16 * k::kStepCount);
        for (int64_t ctx : {2100, 2500, 9001, 33000, 50003, 65002})
            for (int nq : {1, 3, 8, 16})
                for (int pat = 0; pat < 3; ++pat) {
                    const int64_t mb = MAXB;
                    std::vector<int32_t> steps((size_t) nq * k::kStepCount, 0);
                    std::vector<float> sc((size_t) nq * mb);
                    for (int i = 0; i < nq; ++i) {
                        const int64_t n_kv = ctx - i * 3;
                        steps[i * k::kStepCount + k::kStepNKv] = (int) n_kv;
                        steps[i * k::kStepCount + k::kStepNBid] = (int) (n_kv / R);
                        steps[i * k::kStepCount + k::kStepWidth] = (int) k::qsa_selection_width(n_kv, S_);
                        for (int64_t b = 0; b < mb; ++b) {
                            float v = std::fabs(nd(rng)) * 40.0f + 3.0f;
                            if (pat == 1) v = std::floor(v * 0.25f) * 4.0f;   // ties
                            if (pat == 2) v = 17.0f;
                            sc[(size_t) i * mb + b] = v;
                        }
                        if (n_kv % R != 0) sc[(size_t) i * mb + n_kv / R] += 1e9f;
                    }
                    up(d_sc, sc);
                    up(d_st, steps);
                    int32_t* d_ids = dalloc<int32_t>((size_t) nq * cap);
                    std::vector<int32_t> ref;
                    for (int v : {0, 2}) {
                        ex["idx50_indexer"] = v;
                        ck(cudaMemset(d_ids, 0xff, (size_t) nq * cap * 4), "m");
                        k::qsa_block_topk(d_sc, d_st, nq, mb, cap, S_, d_ids, st);
                        ck(cudaStreamSynchronize(st), "sync");
                        auto h = down(d_ids, (size_t) nq * cap);
                        if (v == 0) { ref = h; continue; }
                        for (int i = 0; i < nq; ++i) {
                            const int w = steps[i * k::kStepCount + k::kStepWidth];
                            ++total;
                            same_n += std::equal(h.begin() + (size_t) i * cap, h.begin() + (size_t) i * cap + w, ref.begin() + (size_t) i * cap);
                        }
                    }
                }
        ex["idx50_indexer"] = 0;
        g_fail += same_n != total;
        std::printf("topk reduce-scatter vs cluster: identical selections %lld / %lld (6 contexts x 4 widths x continuous, tied, equal)  %s\n", (long long) same_n, (long long) total, same_n == total ? "ok" : "FAIL");
    }
    std::printf("qsa_scores_parity: %s\n", g_fail ? "FAIL" : "OK");
    return g_fail ? 1 : 0;
}
