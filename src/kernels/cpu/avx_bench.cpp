// src/kernels/cpu/avx_bench.cpp - the CPU expert kernels on each instruction set side by side, at the decode geometry.
//
//     build/avx_bench [tokens] [threads] [experts] [reps]
//
// Each line is one kernel on one instruction set: weight bytes read per second with the worker threads each taking
// a slice of the rows of every expert in a working set far larger than the caches (DRAM-bound), and with one
// thread on one cache-resident expert (compute ceiling).  Calls go straight to each set's kernels, so the CPU can
// time the set it does not dispatch to.  Weights are random bytes with a valid scale field.
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/cpu/expert.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/cpu/iq_avx2.hpp"
#include "basalt/kernels/cpu/iq_avx512.hpp"
#include "basalt/kernels/cpu/kq_avx2.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/kernels/cpu/nvfp4_avx2.hpp"
#include "basalt/kernels/cpu/nvfp4_avx512.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <thread>
#include <vector>

namespace cpu = basalt::kernels::cpu;

namespace {

constexpr int NH = 2560, NFF = 640;
int NT = 4;

struct Op {
    const char* name;
    const char* isa;
    size_t bytes;        // weight bytes one call reads over its whole row range
    int rows;            // rows split across the threads
    std::function<void(const uint8_t* blob, int r0, int r1, int tid)> run;
    size_t stride;       // distance between experts in memory
};

// Random weight bytes; the scale field at the start of each block is a small valid half (or NVFP4 exponent).
void fill(std::vector<uint8_t>& v, int type, size_t blk, uint64_t seed) {
    std::mt19937_64 rng(seed);
    for (size_t i = 0; i + 8 <= v.size(); i += 8) { uint64_t r = rng(); std::memcpy(v.data() + i, &r, 8); }
    for (size_t b = 0; b + blk <= v.size(); b += blk) {
        if (type == 40) { for (int k = 0; k < 4; ++k) v[b + k] = 0x30 + (v[b + k] & 7); }
        else { const uint16_t d = 0x2800 + (v[b] & 0xff); std::memcpy(&v[b], &d, 2); if (type == 12) { const uint16_t dm = 0x2400 + (v[b + 3] & 0xff); std::memcpy(&v[b + 2], &dm, 2); } }
    }
}

struct Acts {
    std::vector<std::vector<uint8_t>> q8k, q8k_h, q80, q80_h;
    std::vector<cpu::ActQ> q21, q21_h;
    std::vector<std::vector<float>> x, h;
};

double run_mt(const Op& op, std::vector<uint8_t*>& blobs, int threads, int reps) {
    std::atomic<int> ready{0}, go{0};
    const int per = (op.rows + threads - 1) / threads;
    std::vector<std::thread> th;
    std::vector<double> secs(threads);
    for (int t = 0; t < threads; ++t) th.emplace_back([&, t] {
        const int r0 = t * per, r1 = std::min(op.rows, r0 + per);
        ready.fetch_add(1);
        while (!go.load()) {}
        const auto t0 = std::chrono::steady_clock::now();
        for (int rep = 0; rep < reps; ++rep)
            for (uint8_t* b : blobs) op.run(b, r0, r1, t);
        secs[t] = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    });
    while (ready.load() < threads) {}
    go.store(1);
    for (auto& x : th) x.join();
    double mx = 0;
    for (double s : secs) mx = std::max(mx, s);
    return (double) op.bytes * blobs.size() * reps / mx / 1e9;
}

double run_1t(const Op& op, uint8_t* blob, int reps) {
    for (int i = 0; i < 3; ++i) op.run(blob, 0, op.rows, 0);
    const auto t0 = std::chrono::steady_clock::now();
    for (int rep = 0; rep < reps; ++rep) op.run(blob, 0, op.rows, 0);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return (double) op.bytes * reps / s / 1e9;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc > 1) NT = std::atoi(argv[1]);
    const int threads = argc > 2 ? std::atoi(argv[2]) : 14;
    const int E = argc > 3 ? std::atoi(argv[3]) : 96;
    const int reps = argc > 4 ? std::atoi(argv[4]) : 3;
    std::printf("avx_bench: %s, %d tokens, %d threads, %d experts per pass, %d passes (weight bytes read / s)\n",
                cpu::cpu_name().c_str(), NT, threads, E, reps);

    // activations (shared by the ops)
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.f, 1.f);
    const auto* tq8k = ggml_get_type_traits_cpu_avx2(GGML_TYPE_Q8_K);
    const auto* tq80 = ggml_get_type_traits_cpu_avx2(GGML_TYPE_Q8_0);
    std::vector<std::vector<float>> xs(NT, std::vector<float>(NH)), hs(NT, std::vector<float>(NFF));
    std::vector<std::vector<uint8_t>> a_k(NT), a_0(NT), h_0(NT);
    std::vector<cpu::ActQ> a_1(NT), h_1(NT);
    const void *pk[8], *p0[8], *ph0[8];
    const cpu::ActQ *p1[8], *ph1[8];
    for (int t = 0; t < NT; ++t) {
        for (auto& v : xs[t]) v = nd(rng);
        for (auto& v : hs[t]) v = nd(rng);
        a_k[t].resize(ggml_row_size(GGML_TYPE_Q8_K, NH));
        a_0[t].resize(ggml_row_size(GGML_TYPE_Q8_0, NH));
        h_0[t].resize(ggml_row_size(GGML_TYPE_Q8_0, NFF));
        tq8k->from_float(xs[t].data(), a_k[t].data(), NH);
        tq80->from_float(xs[t].data(), a_0[t].data(), NH);
        tq80->from_float(hs[t].data(), h_0[t].data(), NFF);
        cpu::act_quant_q8_1_avx2(xs[t].data(), NH, a_1[t]);
        cpu::act_quant_q8_1_avx2(hs[t].data(), NFF, h_1[t]);
        pk[t] = a_k[t].data(); p0[t] = a_0[t].data(); ph0[t] = h_0[t].data(); p1[t] = &a_1[t]; ph1[t] = &h_1[t];
    }
    std::vector<std::vector<float>> ffv(NT, std::vector<float>(NFF)), outv(NT, std::vector<float>(NH));
    // per-thread outputs: each thread writes its own rows, so sharing the arrays is safe
    float* ff[8]; float* out[8];
    for (int t = 0; t < NT; ++t) { ff[t] = ffv[t].data(); out[t] = outv[t].data(); }

    std::vector<Op> ops;
    auto gu_row = [](int type) { return ggml_row_size((ggml_type) type, NH); };
    auto d_row = [](int type) { return ggml_row_size((ggml_type) type, NFF); };
    struct Gu { int type; const char* name; };
    // gate/up: i-quants on both sets
    for (Gu g : {Gu{18, "IQ3_XXS gate/up"}, Gu{21, "IQ3_S gate/up"}, Gu{16, "IQ2_XXS gate/up"}, Gu{17, "IQ2_XS gate/up"}, Gu{22, "IQ2_S gate/up"}}) {
        const size_t row = gu_row(g.type), up = row * NFF, bytes = 2 * up;
        const int type = g.type;
        ops.push_back({g.name, "avx512", bytes, NFF, [=](const uint8_t* b, int r0, int r1, int) { cpu::iq512_gu_rows(type, b, row, up, NH, pk, NT, ff, r0, r1); }, bytes});
        ops.push_back({g.name, "avx2", bytes, NFF, [=](const uint8_t* b, int r0, int r1, int) { cpu::iq256_gu_rows(type, b, row, up, NH, pk, NT, ff, r0, r1); }, bytes});
    }
    {   // Q4_K gate/up: AVX-512 CPUs run ggml's dot per token; AVX2 CPUs the multi-token kernel
        const size_t row = gu_row(12), up = row * NFF, bytes = 2 * up;
        auto tr = [&](const ggml_type_traits_cpu* t) {
            return [=](const uint8_t* b, int r0, int r1, int) {
                for (int r = r0; r < r1; ++r) for (int k = 0; k < NT; ++k) { float g, u; t->vec_dot(NH, &g, 0, b + r * row, 0, pk[k], 0, 1); t->vec_dot(NH, &u, 0, b + up + r * row, 0, pk[k], 0, 1); ff[k][r] = g * u; } };
        };
        ops.push_back({"Q4_K gate/up", "avx512", bytes, NFF, tr(ggml_get_type_traits_cpu_avx512(GGML_TYPE_Q4_K)), bytes});
        ops.push_back({"Q4_K gate/up", "avx2", bytes, NFF, [=](const uint8_t* b, int r0, int r1, int) { cpu::kq256_gu_rows(12, b, row, up, NH, pk, NT, ff, r0, r1); }, bytes});
    }
    {   // NVFP4 gate/up and down
        const size_t row = gu_row(40), up = row * NFF, bytes = 2 * up, drow = d_row(40), dbytes = drow * NH;
        auto tr = [&](const ggml_type_traits_cpu* t) {
            return [=](const uint8_t* b, int r0, int r1, int) {
                for (int r = r0; r < r1; ++r) for (int k = 0; k < NT; ++k) { float g, u; t->vec_dot(NH, &g, 0, b + r * row, 0, p0[k], 0, 1); t->vec_dot(NH, &u, 0, b + up + r * row, 0, p0[k], 0, 1); ff[k][r] = g * u; } };
        };
        auto trd = [&](const ggml_type_traits_cpu* t) {
            return [=](const uint8_t* b, int r0, int r1, int) {
                for (int r = r0; r < r1; ++r) for (int k = 0; k < NT; ++k) t->vec_dot(NFF, &out[k][r], 0, b + r * drow, 0, ph0[k], 0, 1); };
        };
        ops.push_back({"NVFP4 gate/up", "avx512", bytes, NFF, [=](const uint8_t* b, int r0, int r1, int) { cpu::nvfp4_512_gu_rows(b, row, up, NH, 1.f, 1.f, p0, NT, ff, r0, r1); }, bytes});
        ops.push_back({"NVFP4 gate/up", "avx2", bytes, NFF, tr(ggml_get_type_traits_cpu_avx2(GGML_TYPE_NVFP4)), bytes});
        ops.push_back({"NVFP4 gate/up", "avx2 mt", bytes, NFF, [=](const uint8_t* b, int r0, int r1, int) { cpu::nvfp4_256_gu_rows(b, row, up, NH, 1.f, 1.f, p0, NT, ff, r0, r1); }, bytes});
        ops.push_back({"NVFP4 down", "avx2 mt", dbytes, NH, [=](const uint8_t* b, int r0, int r1, int) { cpu::nvfp4_256_down_rows(b, drow, NFF, 1.f, ph0, NT, out, r0, r1); }, dbytes});
        ops.push_back({"NVFP4 down", "avx512", dbytes, NH, [=](const uint8_t* b, int r0, int r1, int) { cpu::nvfp4_512_down_rows(b, drow, NFF, 1.f, ph0, NT, out, r0, r1); }, dbytes});
        ops.push_back({"NVFP4 down", "avx2", dbytes, NH, trd(ggml_get_type_traits_cpu_avx2(GGML_TYPE_NVFP4)), dbytes});
    }
    {   // IQ4_NL down: ggml's dot per token on AVX-512 CPUs, the multi-token kernel on AVX2 CPUs
        const size_t drow = d_row(20), dbytes = drow * NH;
        const auto* t = ggml_get_type_traits_cpu_avx512(GGML_TYPE_IQ4_NL);
        ops.push_back({"IQ4_NL down", "avx512", dbytes, NH, [=](const uint8_t* b, int r0, int r1, int) {
            for (int r = r0; r < r1; ++r) for (int k = 0; k < NT; ++k) t->vec_dot(NFF, &out[k][r], 0, b + r * drow, 0, ph0[k], 0, 1); }, dbytes});
        ops.push_back({"IQ4_NL down", "avx2", dbytes, NH, [=](const uint8_t* b, int r0, int r1, int) { cpu::iq4nl256_down_rows(b, drow, NFF, ph0, NT, out, r0, r1); }, dbytes});
    }
    {   // Q2_0 down
        const size_t drow = d_row(42), dbytes = drow * NH;
        ops.push_back({"Q2_0 down", "avx512", dbytes, NH, [=](const uint8_t* b, int r0, int r1, int) { cpu::q2_0_gguf_rows_multi(b, drow, NFF / 64, ph1, NT, out, r0, r1); }, dbytes});
        ops.push_back({"Q2_0 down", "avx2", dbytes, NH, [=](const uint8_t* b, int r0, int r1, int) { cpu::q2_0_gguf_rows_multi_avx2(b, drow, NFF / 64, ph1, NT, out, r0, r1); }, dbytes});
    }

    {   // activation quantizers, one token: us per call
        auto time_us = [&](const std::function<void()>& f) {
            for (int i = 0; i < 2000; ++i) f();
            const auto t0 = std::chrono::steady_clock::now();
            const int R = 50000;
            for (int i = 0; i < R; ++i) f();
            return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / R;
        };
        std::vector<uint8_t> q(ggml_row_size(GGML_TYPE_Q8_K, NH));
        std::printf("q8_K quantize 2560   ggml scalar %.2f us   avx2 %.2f us   avx512 %.2f us\n",
                    time_us([&] { tq8k->from_float(xs[0].data(), q.data(), NH); }),
                    time_us([&] { cpu::quantize_q8_K_avx2(xs[0].data(), q.data(), NH); }),
                    time_us([&] { cpu::quantize_q8_K_avx512(xs[0].data(), q.data(), NH); }));
        std::vector<uint8_t> q0(ggml_row_size(GGML_TYPE_Q8_0, NH));
        std::printf("q8_0 quantize 2560   ggml avx2 %.2f us   ggml avx512 %.2f us\n",
                    time_us([&] { tq80->from_float(xs[0].data(), q0.data(), NH); }),
                    time_us([&] { ggml_get_type_traits_cpu_avx512(GGML_TYPE_Q8_0)->from_float(xs[0].data(), q0.data(), NH); }));
        cpu::ActQ aq;
        std::printf("q8_1 quantize  640   avx2 %.2f us   avx512 %.2f us\n",
                    time_us([&] { cpu::act_quant_q8_1_avx2(hs[0].data(), NFF, aq); }),
                    time_us([&] { cpu::act_quant_q8_1(hs[0].data(), NFF, aq); }));
    }

    std::vector<uint8_t> proto;
    for (const Op& op : ops) {
        int type = 0;
        const std::string n = op.name;
        size_t blk = 0;
        if (n.rfind("IQ3_XXS", 0) == 0) type = 18; else if (n.rfind("IQ3_S", 0) == 0) type = 21; else if (n.rfind("IQ2_XXS", 0) == 0) type = 16;
        else if (n.rfind("IQ2_XS", 0) == 0) type = 17; else if (n.rfind("IQ2_S", 0) == 0) type = 22; else if (n.rfind("Q4_K", 0) == 0) type = 12;
        else if (n.rfind("NVFP4", 0) == 0) type = 40; else if (n.rfind("IQ4_NL", 0) == 0) type = 20; else if (n.rfind("Q2_0", 0) == 0) type = 42;
        blk = ggml_type_size((ggml_type) type);
        const size_t stride = (op.bytes + 63) & ~size_t(63);
        std::vector<uint8_t> mem(stride * E + 64);
        uint8_t* base = (uint8_t*) (((uintptr_t) mem.data() + 63) & ~uintptr_t(63));
        std::vector<uint8_t> one(op.bytes);
        fill(one, type, blk, 1234);
        // rows begin on block boundaries; the fill's block grid matches since row sizes are whole blocks
        std::vector<uint8_t*> blobs;
        for (int e = 0; e < E; ++e) { std::memcpy(base + stride * e, one.data(), op.bytes); blobs.push_back(base + stride * e); }
        double bw = 0;   // best of three: another process's burst only slows a trial
        for (int trial = 0; trial < 3; ++trial) bw = std::max(bw, run_mt(op, blobs, threads, reps));
        double c1 = 0;
        for (int trial = 0; trial < 3; ++trial) c1 = std::max(c1, run_1t(op, blobs[0], 30));
        std::printf("%-18s %-7s %6.1f GB/s on %d threads   %5.1f GB/s on 1 thread in cache\n", op.name, op.isa, bw, threads, c1);
    }
    return 0;
}
