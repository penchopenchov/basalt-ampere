// src/kernels/dec_commit_parity.cpp - STRATA_DEC_COMMIT's two launches against the commit graph's per-layer calls.
//
//     build/dec_commit_parity [--bench]
//
// GDN: gdn_commit_layers against gdn_conv_commit + gdn_step_norm_multi (state-only commit) per layer; QSA:
// native_qsa_indexer_commit_layers against the tail restore + n_tok native_qsa_indexer_append calls per layer.
// Every state, conv history and indexer buffer must match bit for bit for n_keep = 0..7 at the model's geometry.
// Inputs random normal.  --bench: both as CUDA graphs over a window's layers (36 GDN, 12 QSA), in microseconds.
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/rope_scaling.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace strata::kernels;

namespace {

constexpr int S = 128, HK = 16, HV = 48, C = 2 * HK * S + HV * S, MT = 7, ID = 128, R = 4, CELLS = 4096;
constexpr size_t GDN_FLOATS = (size_t) S * HV * S + (size_t) C * 3, TS = (size_t) (R - 1) * ID;
constexpr float EPS = 1e-6f;

#define CK(x)                                                                                    \
    do {                                                                                         \
        const cudaError_t e_ = (x);                                                              \
        if (e_ != cudaSuccess) {                                                                 \
            std::printf("CUDA %s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
            std::exit(1);                                                                        \
        }                                                                                        \
    } while (0)

template<typename T> T* dev_rand(std::mt19937& rng, size_t n, float sd) {
    std::vector<float> h(n);
    std::normal_distribution<float> nd(0.0f, sd);
    for (auto& v : h) v = nd(rng);
    T* d = nullptr;
    CK(cudaMalloc(&d, n * sizeof(float)));
    CK(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    return d;
}
std::vector<uint8_t> host(const void* d, size_t bytes) {
    std::vector<uint8_t> h(bytes);
    CK(cudaMemcpy(h.data(), d, bytes, cudaMemcpyDeviceToHost));
    return h;
}

struct Gdn {
    float *state0, *state, *h, *qkv, *gate, *beta, *z, *y, *gamma;
    int nl;
};
Gdn make_gdn(std::mt19937& rng, int nl) {
    Gdn g;
    g.nl = nl;
    g.state0 = dev_rand<float>(rng, nl * GDN_FLOATS, 0.1f);
    CK(cudaMalloc(&g.state, nl * GDN_FLOATS * 4));
    g.h = dev_rand<float>(rng, (size_t) nl * MT * C, 0.1f);
    g.qkv = dev_rand<float>(rng, (size_t) nl * MT * C, 1.0f);
    g.gate = dev_rand<float>(rng, (size_t) nl * MT * HV, 0.3f);
    g.beta = dev_rand<float>(rng, (size_t) nl * MT * HV, 0.3f);
    g.z = dev_rand<float>(rng, (size_t) MT * HV * S, 1.0f);
    g.y = dev_rand<float>(rng, (size_t) MT * HV * S, 1.0f);
    g.gamma = dev_rand<float>(rng, S, 1.0f);
    return g;
}
void gdn_ref(const Gdn& g, const int32_t* n_keep, cudaStream_t s) {
    for (int l = 0; l < g.nl; ++l) {
        float* st = g.state + l * GDN_FLOATS;
        gdn_conv_commit(st + (size_t) S * HV * S, g.qkv + (size_t) l * MT * C, C, n_keep, s);
        gdn_step_norm_multi(st, g.h + (size_t) l * MT * C, C, g.gate + (size_t) l * MT * HV,
                            g.beta + (size_t) l * MT * HV, g.z, g.gamma, EPS, g.y, HK, HV, MT, n_keep, s, MT);
    }
}
void gdn_new(const Gdn& g, const int32_t* n_keep, cudaStream_t s) {
    GdnCommitLayers a;
    a.state = g.state; a.h = g.h; a.qkv = g.qkv; a.gate = g.gate; a.beta = g.beta;
    a.state_stride = GDN_FLOATS; a.h_stride = (size_t) MT * C; a.gb_stride = (size_t) MT * HV;
    a.C = C; a.h_k = HK; a.h_v = HV; a.n_layers = g.nl; a.n_keep = n_keep;
    gdn_commit_layers(a, s);
}

struct Qsa {
    QsaCommitLayers a;
    std::vector<float*> bufs;   // per layer: tail, dead, pooled (the restore target and outputs)
    float *pooled0 = nullptr;
    size_t pooled_n = 0;
};
Qsa make_qsa(std::mt19937& rng, int nl, const int32_t* pos) {
    Qsa q;
    q.pooled_n = (size_t) (CELLS / R + 1) * ID;
    q.a.pos = pos; q.a.epsilon = EPS; q.a.n_layers = nl; q.a.n_tok = MT;
    for (int l = 0; l < nl; ++l) {
        QsaCommitLayer& L = q.a.layer[l];
        L.raw = dev_rand<float>(rng, (size_t) MT * ID, 1.0f);
        L.tail_snap = dev_rand<float>(rng, TS, 1.0f);
        L.gamma = dev_rand<float>(rng, ID, 1.0f);
        L.tail = dev_rand<float>(rng, TS, 1.0f);
        L.dead = dev_rand<float>(rng, ID, 1.0f);
        L.pooled = dev_rand<float>(rng, q.pooled_n, 1.0f);
        CK(cudaMalloc(&L.block_pos, 4)); CK(cudaMemset(L.block_pos, 0, 4));
        L.max_cells = CELLS;
    }
    return q;
}
void qsa_ref(const Qsa& q, cudaStream_t s) {
    const QsaShapes sh = qsa_real_shapes();
    for (int l = 0; l < q.a.n_layers; ++l) {
        const QsaCommitLayer& L = q.a.layer[l];
        CK(cudaMemcpyAsync(L.tail, L.tail_snap, TS * 4, cudaMemcpyDeviceToDevice, s));
        const QsaIndexerBuffers b{L.tail, L.dead, L.pooled, L.block_pos};
        for (int t = 0; t < MT; ++t)
            native_qsa_indexer_append(L.raw + (size_t) t * ID, q.a.pos + t, 0, L.gamma, EPS, b, sh, L.max_cells,
                                      RopeScaling{}, s);
    }
}
// per layer: tail | dead | pooled | block_pos bytes
std::vector<uint8_t> qsa_snapshot(const Qsa& q) {
    std::vector<uint8_t> all;
    for (int l = 0; l < q.a.n_layers; ++l) {
        const QsaCommitLayer& L = q.a.layer[l];
        for (const auto& v : {host(L.tail, TS * 4), host(L.dead, ID * 4), host(L.pooled, q.pooled_n * 4),
                              host(L.block_pos, 4)})
            all.insert(all.end(), v.begin(), v.end());
    }
    return all;
}
void qsa_restore(const Qsa& q, const std::vector<uint8_t>& snap) {
    size_t o = 0;
    for (int l = 0; l < q.a.n_layers; ++l) {
        const QsaCommitLayer& L = q.a.layer[l];
        CK(cudaMemcpy(L.tail, snap.data() + o, TS * 4, cudaMemcpyHostToDevice)); o += TS * 4;
        CK(cudaMemcpy(L.dead, snap.data() + o, ID * 4, cudaMemcpyHostToDevice)); o += ID * 4;
        CK(cudaMemcpy(L.pooled, snap.data() + o, q.pooled_n * 4, cudaMemcpyHostToDevice)); o += q.pooled_n * 4;
        CK(cudaMemcpy(L.block_pos, snap.data() + o, 4, cudaMemcpyHostToDevice)); o += 4;
    }
}

// commit_[0] = n_keep, [1] = n_keep - 1, [2 + t] = p0 + t for t < n_keep else -1
void set_commit(int32_t* d, int n_keep, int p0) {
    int32_t h[2 + MT];
    h[0] = n_keep; h[1] = n_keep - 1;
    for (int t = 0; t < MT; ++t) h[2 + t] = t < n_keep ? p0 + t : -1;
    CK(cudaMemcpy(d, h, sizeof(h), cudaMemcpyHostToDevice));
}

bool parity(cudaStream_t s) {
    std::mt19937 rng(7);
    int32_t* commit = nullptr;
    CK(cudaMalloc(&commit, (2 + MT) * 4));
    Gdn g = make_gdn(rng, 3);
    Qsa q = make_qsa(rng, 3, commit + 2);
    const auto q0 = qsa_snapshot(q);
    bool ok = true;
    for (int n_keep = 0; n_keep <= MT; ++n_keep)
        for (int p0 : {0, 1, 5, 6, 1001}) {
            set_commit(commit, n_keep, p0);
            CK(cudaMemcpy(g.state, g.state0, g.nl * GDN_FLOATS * 4, cudaMemcpyDeviceToDevice));
            CK(cudaDeviceSynchronize());
            gdn_ref(g, commit, s);
            CK(cudaStreamSynchronize(s));
            const auto ra = host(g.state, g.nl * GDN_FLOATS * 4);
            CK(cudaMemcpy(g.state, g.state0, g.nl * GDN_FLOATS * 4, cudaMemcpyDeviceToDevice));
            CK(cudaDeviceSynchronize());
            gdn_new(g, commit, s);
            CK(cudaStreamSynchronize(s));
            const auto rb = host(g.state, g.nl * GDN_FLOATS * 4);
            if (ra != rb) { std::printf("  GDN n_keep=%d p0=%d: state differs\n", n_keep, p0); ok = false; }
            qsa_restore(q, q0);
            CK(cudaDeviceSynchronize());
            qsa_ref(q, s);
            CK(cudaStreamSynchronize(s));
            const auto qa = qsa_snapshot(q);
            qsa_restore(q, q0);
            CK(cudaDeviceSynchronize());
            native_qsa_indexer_commit_layers(q.a, qsa_real_shapes(), RopeScaling{}, s);
            CK(cudaStreamSynchronize(s));
            const auto qb = qsa_snapshot(q);
            if (qa != qb) { std::printf("  QSA n_keep=%d p0=%d: indexer buffers differ\n", n_keep, p0); ok = false; }
            if (qa == q0 && n_keep > 0) { std::printf("  QSA n_keep=%d p0=%d: no change (no power)\n", n_keep, p0); ok = false; }
        }
    std::printf("dec_commit parity (n_keep 0..7, GDN state + conv, QSA indexer): %s\n", ok ? "BITWISE" : "FAILED");
    return ok;
}

void bench(cudaStream_t s) {
    std::mt19937 rng(11);
    int32_t* commit = nullptr;
    CK(cudaMalloc(&commit, (2 + MT) * 4));
    Gdn g = make_gdn(rng, 36);
    Qsa q = make_qsa(rng, 12, commit + 2);
    for (int n_keep : {1, 2, 3, 5}) {
        set_commit(commit, n_keep, 100);
        float us[2];
        for (int path = 0; path < 2; ++path) {
            cudaGraph_t gr;
            cudaGraphExec_t ge;
            CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
            if (path == 0) { gdn_ref(g, commit, s); qsa_ref(q, s); }
            else { gdn_new(g, commit, s); native_qsa_indexer_commit_layers(q.a, qsa_real_shapes(), RopeScaling{}, s); }
            CK(cudaStreamEndCapture(s, &gr));
            CK(cudaGraphInstantiate(&ge, gr, 0));
            cudaEvent_t e0, e1;
            CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
            for (int w = 0; w < 3; ++w) CK(cudaGraphLaunch(ge, s));
            constexpr int REPS = 50;
            CK(cudaEventRecord(e0, s));
            for (int r = 0; r < REPS; ++r) CK(cudaGraphLaunch(ge, s));
            CK(cudaEventRecord(e1, s));
            CK(cudaEventSynchronize(e1));
            float ms = 0;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            us[path] = ms * 1000.0f / REPS;
            CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(gr));
        }
        const double gb = 2.0 * 36 * (double) S * HV * S * 4 / 1e9;   // the states read and written
        std::printf("bench n_keep=%d: per-layer calls %.1f us | two launches %.1f us (GDN states %.0f MB: %.0f GB/s)\n",
                    n_keep, us[0], us[1], gb * 1e3, gb / (us[1] * 1e-6));
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    const bool ok = parity(s);
    if (mode == "--bench") bench(s);
    return ok ? 0 : 1;
}
