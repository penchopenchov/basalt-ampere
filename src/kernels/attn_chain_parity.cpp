// src/kernels/attn_chain_parity.cpp - attn_chain against the kernels it replaces (hc_fused_read, dense_tc_pair,
// dec_gdn_rec, dense_tc, hc_fused_read with the router), bit for bit on every output, with real geometry, random
// inputs and every projection-type triple of the packs; a token's outputs must not depend on the window width (a
// T-row launch against its T'-row prefix); then the time of both with the weights hot in L2.  The kernels it
// replaces are checked against plain CPU references by hc_fused_parity, dense_tc_parity and dec_gdn_parity.
//
//     build/attn_chain_parity [iters]
#include "basalt/kernels/attn_chain.hpp"
#include "basalt/kernels/dec_gdn.hpp"
#include "basalt/kernels/dense_tc.hpp"
#include "basalt/kernels/hc_fused.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace K = basalt::kernels;

namespace {
constexpr int N = 2560, HC = 4, D = N * HC, LR = 320, NE = 512, KT = 10, S = 128, HV = 48, HK = 16;
constexpr int C = 2 * HK * S + HV * S, ZV = HV * S, NQ = 10240, NZ = 6144, NO_IN = 6144, TM = K::kAttnChainMaxT;

#define CK(x) do { cudaError_t e_ = (x); if (e_) { std::printf("CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__); std::exit(1); } } while (0)

uint16_t bf16(float x) { uint32_t u; std::memcpy(&u, &x, 4); return (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16); }
uint16_t sane_half(std::mt19937& r) { const uint32_t v = r(); return (uint16_t) (((v >> 31) << 15) | ((5u + (v >> 10) % 5u) << 10) | (v & 0x3ffu)); }
template <typename V> V* upload(const std::vector<V>& h) {
    V* d;
    CK(cudaMalloc(&d, h.size() * sizeof(V) + 64));
    CK(cudaMemcpy(d, h.data(), h.size() * sizeof(V), cudaMemcpyHostToDevice));
    return d;
}
std::vector<float> randn(size_t n, float sd, std::mt19937& r) {
    std::normal_distribution<float> nd(0.f, sd);
    std::vector<float> v(n);
    for (auto& x : v) x = nd(r);
    return v;
}
int sb_bytes(int type) { return type == 12 ? 144 : type == 13 ? 176 : type == 14 ? 210 : type == 21 ? 110 : 136; }
// random super-blocks of a type with finite scales (d, and dmin for Q4_K / Q5_K; Q6_K's d is its last 2 bytes)
uint8_t* rand_type(int type, size_t nb, std::mt19937& r) {
    const int sb = sb_bytes(type);
    std::vector<uint8_t> w(nb * sb);
    for (auto& v : w) v = (uint8_t) (r() & 0xff);
    for (size_t k = 0; k < nb; ++k) {
        const uint16_t h = sane_half(r), h2 = sane_half(r);
        if (type == 14) std::memcpy(&w[k * sb + 208], &h, 2);
        else { std::memcpy(&w[k * sb], &h, 2); if (type == 12 || type == 13) std::memcpy(&w[k * sb + 2], &h2, 2); }
    }
    return upload(w);
}

struct Layer {
    uint16_t *wd[2], *wu[2], *wi[2], *wr;
    float* wn[2];
    uint8_t *wq, *wz, *wo;
    K::DecRecArgs rec;
    int tq, tz, to;
};
struct Bufs {
    float *R0, *R, *boM, *injB0, *injA, *injB, *mixed1, *mixed2, *lo, *rs, *xn, *qkv, *z, *hh, *gate, *beta, *yrec, *bo2, *logits, *wts;
    uint8_t *xq1, *xq2, *yq;
    int32_t* ids;
};
}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int iters = argc > 1 ? std::atoi(argv[1]) : 200;
    K::hc_fused_init();
    K::attn_chain_init();
    if (!K::hc_fused_supported()) { std::printf("attn_chain_parity: no fused hc read on this device\n"); return 77; }
    std::mt19937 rng(17);
    std::normal_distribution<float> nd(0.f, 1.f);
    Bufs b{};
    auto zf = [&](size_t n) { return upload(std::vector<float>(n)); };
    b.R0 = upload(randn((size_t) TM * D, 1.f, rng)); b.R = zf((size_t) TM * D);
    b.boM = upload(randn((size_t) TM * N, 0.5f, rng)); b.injB0 = upload(randn((size_t) TM * HC, 1.f, rng));
    b.injA = zf(TM * HC); b.injB = zf(TM * HC);
    b.mixed1 = zf((size_t) TM * N); b.mixed2 = zf((size_t) TM * N); b.lo = zf((size_t) TM * LR); b.rs = zf(TM * HC); b.xn = zf((size_t) TM * D);
    b.qkv = zf((size_t) TM * C); b.z = zf((size_t) TM * ZV); b.hh = zf((size_t) TM * C); b.gate = zf((size_t) TM * HV); b.beta = zf((size_t) TM * HV);
    b.yrec = zf((size_t) TM * ZV); b.bo2 = zf((size_t) TM * N); b.logits = zf((size_t) TM * NE); b.wts = zf((size_t) TM * KT);
    b.xq1 = upload(std::vector<uint8_t>((size_t) TM * (N / 32) * 36)); b.xq2 = upload(std::vector<uint8_t>((size_t) TM * (N / 32) * 36));
    b.yq = upload(std::vector<uint8_t>((size_t) TM * (ZV / 32) * 36));
    CK(cudaMalloc(&b.ids, (size_t) TM * KT * 4));

    // the type triples (qkv, z, out) of the GDN layers of the packs, plus the one the kernel was first built for
    const std::vector<std::array<int, 3>> triples = {{23, 21, 12}, {14, 14, 14}, {14, 12, 14}, {13, 14, 14}, {12, 12, 14}, {14, 21, 23},
                                                     {23, 23, 14}, {13, 13, 13}, {12, 21, 21}, {14, 23, 12}, {23, 14, 13}, {13, 23, 14},
                                                     {13, 12, 14}, {13, 13, 14}, {13, 14, 13}, {14, 12, 12}, {14, 13, 12}, {14, 13, 14}, {14, 14, 13}, {14, 23, 13}, {14, 23, 14}, {23, 14, 14},
                                                     {21, 21, 21}};
    std::vector<Layer> layers(triples.size());
    for (size_t l = 0; l < layers.size(); ++l) {
        Layer& w = layers[l];
        for (int o = 0; o < 2; ++o) {
            std::vector<uint16_t> wd((size_t) LR * D), wu((size_t) D * LR), wi((size_t) HC * D);
            std::vector<float> wn(D);
            for (auto& x : wd) x = bf16(0.02f * nd(rng));
            for (auto& x : wu) x = bf16(0.05f * nd(rng));
            for (auto& x : wi) x = bf16(0.02f * nd(rng));
            for (auto& x : wn) x = 1.0f + 0.3f * nd(rng);
            w.wd[o] = upload(wd); w.wu[o] = upload(wu); w.wi[o] = upload(wi); w.wn[o] = upload(wn);
            K::hc8_prepare(w.wd[o], LR, D); K::hc8_prepare(w.wu[o], D, LR); K::hc8_prepare(w.wi[o], HC, D);
        }
        std::vector<uint16_t> wr((size_t) NE * N);
        for (auto& x : wr) x = bf16(0.05f * nd(rng));
        w.wr = upload(wr);
        w.tq = triples[l][0]; w.tz = triples[l][1]; w.to = triples[l][2];
        w.wq = rand_type(w.tq, (size_t) NQ * 10, rng);
        w.wz = rand_type(w.tz, (size_t) NZ * 10, rng);
        w.wo = rand_type(w.to, (size_t) 2560 * (NO_IN / 256), rng);
        K::DecRecArgs& a = w.rec;
        a.state = upload(randn((size_t) S * HV * S + (size_t) C * 3, 0.1f, rng));
        a.conv_w = upload(randn((size_t) C * 4, 0.5f, rng));
        std::vector<float> dt = randn(HV, 1.f, rng), sa(HV), gm(S);
        for (auto& v : sa) v = -std::exp(randn(1, 1.f, rng)[0]);
        for (auto& v : gm) v = 1.f + randn(1, 0.1f, rng)[0];
        a.dt = upload(dt); a.ssm_a = upload(sa); a.gamma = upload(gm);
        std::vector<uint16_t> wa((size_t) HV * N), wbt((size_t) HV * N);
        std::normal_distribution<float> nd3(0.f, 0.03f);
        for (auto& v : wa) v = bf16(nd3(rng));
        for (auto& v : wbt) v = bf16(nd3(rng));
        a.w_alpha = upload(wa); a.w_beta = upload(wbt);
        a.qkv = b.qkv; a.x = b.mixed1; a.z = b.z; a.C = C; a.n_embd = N; a.h_k = HK; a.h_v = HV; a.eps = 1e-6f;
        a.h = b.hh; a.gate = b.gate; a.beta = b.beta; a.y = b.yrec; a.yq = b.yq;
    }
    auto hcargs = [&](const Layer& w, int ord, int T) {
        K::HcFusedArgs a;
        a.T = T; a.R = b.R; a.R_out = b.R; a.apply = true; a.w_norm = w.wn[ord]; a.w_down = w.wd[ord]; a.w_up = w.wu[ord]; a.w_inject = w.wi[ord];
        a.lo = b.lo; a.rs = b.rs; a.xn = b.xn;
        if (ord == 0) { a.bo_prev = b.boM; a.inj_prev = b.injB; a.inject_out = b.injA; a.mixed = b.mixed1; a.xq = b.xq1; }
        else {
            a.bo_prev = b.bo2; a.inj_prev = b.injA; a.inject_out = b.injB; a.mixed = b.mixed2; a.xq = b.xq2;
            a.w_router = w.wr; a.logits = b.logits; a.ids = b.ids; a.weights = b.wts;
        }
        return a;
    };
    auto chain_args = [&](const Layer& w, int T) {
        K::AttnChainArgs a;
        a.T = T; a.h1 = hcargs(w, 0, T); a.h2 = hcargs(w, 1, T);
        a.wq = w.wq; a.wz = w.wz; a.wo = w.wo; a.type_q = w.tq; a.type_z = w.tz; a.type_o = w.to; a.rec = w.rec;
        return a;
    };
    auto reset = [&] {
        CK(cudaMemcpy(b.R, b.R0, (size_t) TM * D * 4, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(b.injB, b.injB0, (size_t) TM * HC * 4, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(b.bo2, b.boM, (size_t) TM * N * 4, cudaMemcpyDeviceToDevice));
        CK(cudaMemset(b.mixed1, 0xff, (size_t) TM * N * 4));
    };
    auto old_seq = [&](const Layer& w, int T, cudaStream_t s) {
        K::hc_fused_read(hcargs(w, 0, T), s);
        K::dense_tc_pair(w.tq, w.wq, b.qkv, NQ, w.tz, w.wz, b.z, NZ, b.xq1, N, T, s);
        K::dec_gdn_rec(w.rec, T, s);
        K::dense_tc(w.to, w.wo, b.yq, b.bo2, NO_IN, 2560, T, s);
        K::hc_fused_read(hcargs(w, 1, T), s);
    };
    struct Out { const char* name; const void* p; size_t per_row; };   // per token row
    const std::vector<Out> outs = {{"R", b.R, (size_t) D * 4}, {"mixed1", b.mixed1, (size_t) N * 4}, {"xq1", b.xq1, (size_t) N / 32 * 36}, {"qkv", b.qkv, (size_t) C * 4},
                                   {"z", b.z, (size_t) ZV * 4}, {"h", b.hh, (size_t) C * 4}, {"gate", b.gate, (size_t) HV * 4}, {"beta", b.beta, (size_t) HV * 4},
                                   {"y", b.yrec, (size_t) ZV * 4}, {"yq", b.yq, (size_t) ZV / 32 * 36}, {"bo2", b.bo2, (size_t) N * 4}, {"mixed2", b.mixed2, (size_t) N * 4},
                                   {"xq2", b.xq2, (size_t) N / 32 * 36}, {"logits", b.logits, (size_t) NE * 4}, {"ids", b.ids, (size_t) KT * 4}, {"weights", b.wts, (size_t) KT * 4},
                                   {"injA", b.injA, (size_t) HC * 4}, {"injB", b.injB, (size_t) HC * 4}, {"lo2", b.lo, (size_t) LR * 4}};
    auto snap = [&](int T) {
        std::vector<std::vector<uint8_t>> v;
        for (const Out& o : outs) { std::vector<uint8_t> h(o.per_row * T); CK(cudaMemcpy(h.data(), o.p, h.size(), cudaMemcpyDeviceToHost)); v.push_back(h); }
        return v;
    };
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    int bad = 0;
    for (size_t l = 0; l < layers.size(); ++l) {
        const Layer& w = layers[l];
        if (!K::attn_chain_supported(chain_args(w, 1))) { std::printf("types %d/%d/%d: not supported by the chain\n", w.tq, w.tz, w.to); ++bad; continue; }
        std::vector<std::vector<std::vector<uint8_t>>> got(TM + 1);
        size_t tot = 0;
        for (int T = 1; T <= TM; ++T) {
            reset(); old_seq(w, T, s); CK(cudaStreamSynchronize(s));
            const auto ref = snap(T);
            reset(); K::attn_chain(chain_args(w, T), s); CK(cudaStreamSynchronize(s));
            got[T] = snap(T);
            for (size_t k = 0; k < ref.size(); ++k) for (size_t j = 0; j < ref[k].size(); ++j) tot += ref[k][j] != got[T][k][j];
        }
        // a row's outputs do not depend on the window width: the T = 4 launch against each shorter launch's rows
        size_t prefix = 0;
        for (int T = 1; T < TM; ++T)
            for (size_t k = 0; k < outs.size(); ++k) {
                if (!std::strcmp(outs[k].name, "mixed1") || !std::strcmp(outs[k].name, "xq1")) continue;   // overwritten by the second read
                for (size_t j = 0; j < outs[k].per_row * T; ++j) prefix += got[T][k][j] != got[TM][k][j];
            }
        std::printf("types %2d/%2d/%2d: T1-T4 vs today's kernels %zu differing bytes, T-prefix %zu\n", w.tq, w.tz, w.to, tot, prefix);
        bad += (int) (tot + prefix);
    }
    // time, weights hot in L2 (the same layer repeated), T = 3
    {
        const Layer& w = layers[0];
        const int T = 3;
        reset();
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        for (int pass = 0; pass < 2; ++pass) {
            for (int i = 0; i < 10; ++i) { if (pass == 0) old_seq(w, T, s); else K::attn_chain(chain_args(w, T), s); }
            CK(cudaStreamSynchronize(s));
            CK(cudaEventRecord(e0, s));
            for (int i = 0; i < iters; ++i) { if (pass == 0) old_seq(w, T, s); else K::attn_chain(chain_args(w, T), s); }
            CK(cudaEventRecord(e1, s));
            CK(cudaEventSynchronize(e1));
            float ms = 0;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            std::printf("%-26s T=%d hot: %7.2f us per layer\n", pass == 0 ? "today's 5 kernels" : "attn_chain", T, 1e3 * ms / iters);
        }
    }
    std::printf(bad == 0 ? "attn_chain_parity: bitwise\n" : "attn_chain_parity: %d differing\n", bad);
    return bad != 0;
}
