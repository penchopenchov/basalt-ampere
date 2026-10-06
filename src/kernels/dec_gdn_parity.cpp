// src/kernels/dec_gdn_parity.cpp - the verify window's GDN block around dec_gdn_rec against the per-stage verify
// kernels, bitwise.
//
//     build/dec_gdn_parity [--bench | --bench-only]
//
// The reference is the verify window's GDN sequence: native_quantize_q8_1 + native_mmvq (qkv, z),
// gdn_conv_l2_multi, gdn_ab_multi, gdn_step_norm_multi, native_quantize_q8_1 + native_mmvq (out).  The path under
// test is the native in/out projections around dec_gdn_rec (mode 1) or dec_gdn_rec_split + dec_gdn_rec_norm with z on
// a side stream forked after qkv (mode 2) or after the q8_1 (mode 3).  Every output both paths write (qkv, z, h,
// gate, beta, the q8_1 of y, the out projection) must match bit for bit at the model's geometry, for T = 1..8 and
// every projection type the model uses in every position, and each column must equal the same column of an
// eight-token window; the state must be left untouched.  Weights are random bytes with normal fp16 scales
// (mmvq_multi_parity's rule), activations and state random normal.
//
// --bench: the reference and the three modes timed as CUDA graphs over enough layers' distinct weights that the L2
// holds none of them, per stage and whole, in microseconds per layer and GB/s over the bytes each must move.
#include "basalt/kernels/dec_gdn.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/verify_kernels.hpp"

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

using namespace basalt::kernels;

namespace {

constexpr int N = 2560, HK = 16, HV = 48, S = 128, C = 2 * HK * S + HV * S, ZV = HV * S;
constexpr float EPS = 1e-6f;

#define CK(x)                                                                                    \
    do {                                                                                         \
        const cudaError_t e_ = (x);                                                              \
        if (e_ != cudaSuccess) {                                                                 \
            std::printf("CUDA %s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
            std::exit(1);                                                                        \
        }                                                                                        \
    } while (0)

struct Fmt {
    int type;
    const char* name;
    int block_bytes;
    int scale_at[2];
};
const Fmt FMTS[] = {
    {12, "Q4_K", 144, {0, 2}}, {13, "Q5_K", 176, {0, 2}}, {14, "Q6_K", 210, {208, -1}},
    {21, "IQ3_S", 110, {0, -1}}, {23, "IQ4_XS", 136, {0, -1}},
};
const Fmt& fmt(int t) {
    for (const Fmt& f : FMTS)
        if (f.type == t) return f;
    std::printf("no format %d\n", t);
    std::exit(1);
}

uint16_t sane_half(std::mt19937& rng) {
    const uint32_t r = rng();
    return (uint16_t) (((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
}

size_t mat_bytes(int t, int n_in, int n_out) { return (size_t) n_out * (n_in / 256) * fmt(t).block_bytes; }

void* rand_mat(int t, int n_in, int n_out, std::mt19937& rng) {
    const Fmt& f = fmt(t);
    const size_t nb = (size_t) n_out * (n_in / 256), bytes = nb * f.block_bytes;
    std::vector<uint8_t> w(bytes);
    for (auto& b : w) b = (uint8_t) (rng() & 0xff);
    for (size_t k = 0; k < nb; ++k)
        for (int at : f.scale_at)
            if (at >= 0) {
                const uint16_t h = sane_half(rng);
                std::memcpy(&w[k * f.block_bytes + at], &h, 2);
            }
    void* d = nullptr;
    CK(cudaMalloc(&d, bytes + 64));
    CK(cudaMemcpy(d, w.data(), bytes, cudaMemcpyHostToDevice));
    CK(cudaDeviceSynchronize());   // a pageable copy may return before its DMA lands; the test streams do not wait for it
    return d;
}

template<typename V>
V* upload(const std::vector<V>& h) {
    V* d = nullptr;
    CK(cudaMalloc(&d, h.size() * sizeof(V) + 64));
    CK(cudaMemcpy(d, h.data(), h.size() * sizeof(V), cudaMemcpyHostToDevice));
    CK(cudaDeviceSynchronize());
    return d;
}
std::vector<float> randn(size_t n, float sd, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, sd);
    std::vector<float> v(n);
    for (auto& x : v) x = nd(rng);
    return v;
}
uint16_t bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// One GDN layer's weights and constants.
struct Layer {
    int tq, tz, to;
    void *wqkv, *wz, *wo;
    float *conv_w, *dt, *ssm_a, *gamma, *state;
    uint16_t *wa, *wb;
    size_t bytes = 0;   // the bytes a window must read: weights + state
};
Layer make_layer(int tq, int tz, int to, std::mt19937& rng) {
    Layer L{};
    L.tq = tq; L.tz = tz; L.to = to;
    L.wqkv = rand_mat(tq, N, C, rng);
    L.wz = rand_mat(tz, N, ZV, rng);
    L.wo = rand_mat(to, ZV, N, rng);
    L.conv_w = upload(randn((size_t) C * 4, 0.5f, rng));
    std::vector<float> dt = randn(HV, 1.f, rng), sa(HV), gm(S);
    for (auto& v : sa) v = -std::exp(randn(1, 1.f, rng)[0]);
    for (auto& v : gm) v = 1.f + randn(1, 0.1f, rng)[0];
    L.dt = upload(dt); L.ssm_a = upload(sa); L.gamma = upload(gm);
    std::vector<float> st = randn((size_t) S * HV * S + (size_t) C * 3, 0.1f, rng);
    L.state = upload(st);
    std::vector<uint16_t> a((size_t) HV * N), b((size_t) HV * N);
    std::normal_distribution<float> nd(0.f, 0.03f);
    for (auto& v : a) v = bf16(nd(rng));
    for (auto& v : b) v = bf16(nd(rng));
    L.wa = upload(a); L.wb = upload(b);
    L.bytes = mat_bytes(tq, N, C) + mat_bytes(tz, N, ZV) + mat_bytes(to, ZV, N) + 2 * a.size() * 2 + (size_t) C * 16 +
              st.size() * 4;
    return L;
}

// The window's activations / outputs (one set per path).
struct Bufs {
    float *qkv, *z, *h, *gate, *beta, *y, *bo;
    void* xq;
    void* yq;
};
Bufs make_bufs() {
    Bufs b{};
    CK(cudaMalloc(&b.qkv, sizeof(float) * kDecMaxT * C));
    CK(cudaMalloc(&b.z, sizeof(float) * kDecMaxT * ZV));
    CK(cudaMalloc(&b.h, sizeof(float) * kDecMaxT * C));
    CK(cudaMalloc(&b.gate, sizeof(float) * kDecMaxT * HV));
    CK(cudaMalloc(&b.beta, sizeof(float) * kDecMaxT * HV));
    CK(cudaMalloc(&b.y, sizeof(float) * kDecMaxT * ZV));
    CK(cudaMalloc(&b.bo, sizeof(float) * kDecMaxT * N));
    CK(cudaMalloc(&b.xq, native_q8_1_bytes(ZV, kDecMaxT)));
    CK(cudaMalloc(&b.yq, native_q8_1_bytes(ZV, kDecMaxT)));
    return b;
}
void clear(Bufs& b, cudaStream_t s) {
    CK(cudaMemsetAsync(b.qkv, 0xff, sizeof(float) * kDecMaxT * C, s));
    CK(cudaMemsetAsync(b.z, 0xff, sizeof(float) * kDecMaxT * ZV, s));
    CK(cudaMemsetAsync(b.h, 0xff, sizeof(float) * kDecMaxT * C, s));
    CK(cudaMemsetAsync(b.gate, 0xff, sizeof(float) * kDecMaxT * HV, s));
    CK(cudaMemsetAsync(b.beta, 0xff, sizeof(float) * kDecMaxT * HV, s));
    CK(cudaMemsetAsync(b.bo, 0xff, sizeof(float) * kDecMaxT * N, s));
    CK(cudaMemsetAsync(b.yq, 0xff, native_q8_1_bytes(ZV, kDecMaxT), s));
}

// stage masks for the bench: 1 in-projections, 2 conv/ab/recurrence, 4 out projection
void run_ref(const Layer& L, const float* x, Bufs& b, int T, cudaStream_t s, int stages = 7, unsigned long long* = nullptr) {
    const float* conv = L.state + (size_t) S * HV * S;
    if (stages & 1) {
        native_quantize_q8_1(x, b.xq, N, T, s);
        native_mmvq(L.tq, L.wqkv, b.xq, b.qkv, N, C, T, s);
    }
    if (stages & 2) {
        gdn_conv_l2_multi(conv, b.qkv, L.conv_w, b.h, C, 2 * HK, EPS, T, s, 0);
        gdn_ab_multi(x, L.wa, L.wb, L.dt, L.ssm_a, b.gate, b.beta, N, HV, T, s);
    }
    if (stages & 1) native_mmvq(L.tz, L.wz, b.xq, b.z, N, ZV, T, s);
    if (stages & 2) gdn_step_norm_multi(L.state, b.h, C, b.gate, b.beta, b.z, L.gamma, EPS, b.y, HK, HV, T, nullptr, s, 0);
    if (stages & 4) {
        native_quantize_q8_1(b.y, b.yq, ZV, T, s);
        native_mmvq(L.to, L.wo, b.yq, b.bo, ZV, N, T, s);
    }
}

// native in-projections, dec_gdn_rec (1) or dec_gdn_rec_split + dec_gdn_rec_norm with z on a
// side stream forked after qkv (2) or after the q8_1 (3), native out projection on the q8_1 (stage masks as
// run_ref's; a partial mask runs modes 2/3 on one stream)
cudaStream_t g_side = nullptr;
cudaEvent_t g_fork = nullptr, g_join = nullptr;
int g_gdnf = 1;
void run_gdnf(const Layer& L, const float* x, Bufs& b, int T, cudaStream_t s, int stages = 7,
              unsigned long long* = nullptr) {
    const bool fork = g_gdnf >= 2 && stages == 7;
    auto z_side = [&] {
        CK(cudaEventRecord(g_fork, s));
        CK(cudaStreamWaitEvent(g_side, g_fork, 0));
        PdlScope pz(true, true);
        native_mmvq(L.tz, L.wz, b.xq, b.z, N, ZV, T, g_side);
        CK(cudaEventRecord(g_join, g_side));
    };
    DecRecArgs a;
    a.state = L.state; a.qkv = b.qkv; a.conv_w = L.conv_w; a.x = x; a.w_alpha = L.wa; a.w_beta = L.wb;
    a.dt = L.dt; a.ssm_a = L.ssm_a; a.z = b.z; a.gamma = L.gamma; a.h = b.h; a.gate = b.gate; a.beta = b.beta;
    a.y = nullptr; a.yq = b.yq; a.C = C; a.n_embd = N; a.h_k = HK; a.h_v = HV; a.eps = EPS;
    a.early_trigger = true;
    a.o = b.y;
    if (stages & 1) {
        native_quantize_q8_1(x, b.xq, N, T, s);
        if (fork && g_gdnf == 3) z_side();
        native_mmvq(L.tq, L.wqkv, b.xq, b.qkv, N, C, T, s);
        if (!fork) native_mmvq(L.tz, L.wz, b.xq, b.z, N, ZV, T, s);
    }
    if (stages & 2) {
        if (g_gdnf == 1) {
            dec_gdn_rec(a, T, s);
        } else {
            if (fork && g_gdnf == 2) z_side();
            dec_gdn_rec_split(a, T, s);
            if (fork) CK(cudaStreamWaitEvent(s, g_join, 0));
            PdlScope pn(true, fork);
            dec_gdn_rec_norm(a, T, s);
        }
    }
    if (stages & 4) native_mmvq(L.to, L.wo, b.yq, b.bo, ZV, N, T, s);
}

template<typename V>
std::vector<V> down(const V* d, size_t n) {
    std::vector<V> h(n);
    CK(cudaMemcpy(h.data(), d, n * sizeof(V), cudaMemcpyDeviceToHost));
    return h;
}

// bit differences between two float arrays; prints the first
long long cmp_f(const char* what, const float* a, const float* b, size_t n, double& max_abs) {
    std::vector<float> ha = down(a, n), hb = down(b, n);
    long long d = 0;
    for (size_t i = 0; i < n; ++i) {
        if (std::memcmp(&ha[i], &hb[i], 4) != 0) {
            if (d == 0) std::printf("    %s[%zu]: ref %.9g new %.9g\n", what, i, ha[i], hb[i]);
            ++d;
            const double e = std::fabs((double) ha[i] - hb[i]);
            if (e > max_abs) max_abs = e;
        }
        if (!std::isfinite(ha[i]) || !std::isfinite(hb[i])) { std::printf("    %s[%zu] not finite\n", what, i); ++d; }
    }
    return d;
}
long long cmp_b(const char* what, const void* a, const void* b, size_t n) {
    std::vector<uint8_t> ha = down((const uint8_t*) a, n), hb = down((const uint8_t*) b, n);
    long long d = 0;
    for (size_t i = 0; i < n; ++i) d += ha[i] != hb[i];
    if (d) std::printf("    %s: %lld bytes differ\n", what, d);
    return d;
}

bool parity_gdnf(cudaStream_t s) {
  bool ok = true;
  for (g_gdnf = 1; g_gdnf <= 3; ++g_gdnf) {
    std::mt19937 rng(43);
    const int combos[5][3] = {{14, 23, 14}, {23, 21, 13}, {12, 14, 21}, {13, 12, 23}, {14, 13, 12}};
    Bufs r = make_bufs(), n = make_bufs(), w8 = make_bufs();
    for (const auto& cb : combos) {
        Layer L = make_layer(cb[0], cb[1], cb[2], rng);
        const std::vector<float> st0 = down(L.state, (size_t) S * HV * S + (size_t) C * 3);
        float* x = upload(randn((size_t) kDecMaxT * N, 1.f, rng));
        clear(w8, s);
        run_gdnf(L, x, w8, kDecMaxT, s);
        for (int T = 1; T <= kDecMaxT; ++T) {
            clear(r, s); clear(n, s);
            run_ref(L, x, r, T, s);
            run_gdnf(L, x, n, T, s);
            CK(cudaStreamSynchronize(s));
            double ma = 0;
            long long d = 0, inv = 0;
            d += cmp_f("qkv", r.qkv, n.qkv, (size_t) T * C, ma);
            d += cmp_f("z", r.z, n.z, (size_t) T * ZV, ma);
            d += cmp_f("gate", r.gate, n.gate, (size_t) T * HV, ma);
            d += cmp_f("beta", r.beta, n.beta, (size_t) T * HV, ma);
            d += cmp_f("h", r.h, n.h, (size_t) T * C, ma);
            d += cmp_b("yq", r.yq, n.yq, native_q8_1_bytes(ZV, T));
            d += cmp_f("out", r.bo, n.bo, (size_t) T * N, ma);
            inv += cmp_f("h@8", w8.h, n.h, (size_t) T * C, ma);
            inv += cmp_b("yq@8", w8.yq, n.yq, native_q8_1_bytes(ZV, T));
            inv += cmp_f("out@8", w8.bo, n.bo, (size_t) T * N, ma);
            const std::vector<float> st1 = down(L.state, st0.size());
            const bool st_ok = std::memcmp(st0.data(), st1.data(), st0.size() * 4) == 0;
            std::printf("gdnf=%d qkv %-6s z %-6s out %-6s T=%d: vs ref %s (max abs %.3g), columns vs T=8 %s%s\n",
                        g_gdnf, fmt(cb[0]).name, fmt(cb[1]).name, fmt(cb[2]).name, T, d == 0 ? "bitwise" : "DIFF", ma,
                        inv == 0 ? "bitwise" : "DIFF", st_ok ? "" : " STATE WRITTEN");
            ok = ok && d == 0 && inv == 0 && st_ok;
        }
        CK(cudaFree(x));
    }
  }
    return ok;
}

void bench(cudaStream_t s) {
    std::mt19937 rng(7);
    constexpr int NL = 12;   // distinct layers per graph: far more weight than the L2 holds
    const int types[NL][3] = {{23, 21, 14}, {14, 14, 14}, {12, 12, 13}, {14, 23, 14}, {13, 23, 12}, {14, 21, 14},
                              {23, 12, 14}, {14, 13, 21}, {12, 21, 23}, {14, 14, 14}, {23, 23, 13}, {14, 21, 14}};
    std::vector<Layer> Ls;
    size_t tot = 0, totw[3] = {0, 0, 0};
    for (auto& t : types) {
        Ls.push_back(make_layer(t[0], t[1], t[2], rng));
        tot += Ls.back().bytes;
        totw[0] += mat_bytes(t[0], N, C) + mat_bytes(t[1], N, ZV);
        totw[1] += (size_t) S * HV * S * 4 + (size_t) C * 16 + 2 * (size_t) HV * N * 2;
        totw[2] += mat_bytes(t[2], ZV, N);
    }
    std::printf("bench: %d layers, %.1f MB per layer on average (in %.1f, rec %.1f, out %.1f)\n", NL,
                tot / 1e6 / NL, totw[0] / 1e6 / NL, totw[1] / 1e6 / NL, totw[2] / 1e6 / NL);
    Bufs b = make_bufs();
    float* x = upload(randn((size_t) kDecMaxT * N, 1.f, rng));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    const char* names[4] = {"in-proj", "conv+ab+rec", "out-proj", "whole"};
    const int masks[4] = {1, 2, 4, 7};
    for (int T : {1, 2, 3, 4, 6, 8}) {
        for (int k = 0; k < 4; ++k) {
            double us[4];
            for (int path = 0; path < 4; ++path) {
                g_gdnf = path;
                cudaGraph_t g;
                cudaGraphExec_t ge;
                CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
                {
                    PdlScope ps(true, true);
                    for (const Layer& L : Ls) {
                        if (path == 0) run_ref(L, x, b, T, s, masks[k]);
                        else run_gdnf(L, x, b, T, s, masks[k]);
                    }
                }
                CK(cudaStreamEndCapture(s, &g));
                CK(cudaGraphInstantiate(&ge, g, 0));
                for (int w = 0; w < 3; ++w) CK(cudaGraphLaunch(ge, s));
                const int R = 20;
                CK(cudaEventRecord(e0, s));
                for (int i = 0; i < R; ++i) CK(cudaGraphLaunch(ge, s));
                CK(cudaEventRecord(e1, s));
                CK(cudaEventSynchronize(e1));
                float ms = 0;
                CK(cudaEventElapsedTime(&ms, e0, e1));
                us[path] = ms * 1000.0 / R / NL;
                CK(cudaGraphExecDestroy(ge));
                CK(cudaGraphDestroy(g));
            }
            const double bytes = (k == 3 ? (double) tot : (double) totw[k]) / NL;
            std::printf("T=%d %-12s ref %6.2f us (%5.0f GB/s)  gdnf1 %6.2f  gdnf2 %6.2f  gdnf3 %6.2f us (%5.0f GB/s)\n",
                        T, names[k], us[0], bytes / us[0] / 1e3, us[1], us[2], us[3], bytes / us[3] / 1e3);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    int dev = 0;
    cudaDeviceProp p{};
    CK(cudaGetDeviceProperties(&p, dev));
    if (p.major < 9) { std::printf("dec_gdn_parity: needs sm_90+ (have %d.%d)\n", p.major, p.minor); return 77; }
    std::printf("%s: %d SMs, L2 %d MB, smem/block opt-in %zu KB\n", p.name, p.multiProcessorCount, p.l2CacheSize >> 20,
                p.sharedMemPerBlockOptin >> 10);
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    const std::string mode = argc > 1 ? argv[1] : "";
    CK(cudaStreamCreateWithFlags(&g_side, cudaStreamNonBlocking));
    CK(cudaEventCreateWithFlags(&g_fork, cudaEventDisableTiming));
    CK(cudaEventCreateWithFlags(&g_join, cudaEventDisableTiming));
    const bool ok = mode == "--bench-only" || parity_gdnf(s);
    if (mode != "--bench-only") std::printf("dec_gdn_parity: %s\n", ok ? "PASS (bitwise, columns independent of T)" : "FAIL");
    if (ok && (mode == "--bench" || mode == "--bench-only")) bench(s);
    return ok ? 0 : 1;
}
