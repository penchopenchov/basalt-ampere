// src/kernels/qsa_full_parity.cpp - qsa_full against the kernels it replaces (dense_tc, native_mmvq, bf16_gemv, the K/Q
// post, indexer scores, top-k, qsa_dec_attn, dense_tc), bit for bit on every output and on the cache and indexer state,
// with real geometry, random weights of every projection-type combination of the packs and windows of 1..7 rows at
// several contexts, unmapped pages included; then the attention output and its gate against a double-precision CPU
// reference (read back from the cache the kernel wrote, through the cells it selected); then the time of both.
// qsa_dec_attn_parity, qsa_scores_parity, qsa_topk_parity and qsa_dec_fused_parity check the replaced kernels against
// their own CPU references.
//
//     qsa_full_parity [iters]
#include "basalt/kernels/bf16_gemv.hpp"
#include "basalt/kernels/dense_tc.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/qsa.hpp"
#include "basalt/kernels/qsa_dec_attn.hpp"
#include "basalt/kernels/qsa_dec_fused.hpp"
#include "basalt/kernels/qsa_full.hpp"
#include "basalt/kernels/qsa_select.hpp"
#include "basalt/kernels/rope_scaling.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

namespace k = basalt::kernels;

namespace {
constexpr int NKV = 2, NH = 24, IQ = 4, HD = 256, ID = 128, PG = 4, MAXC = 262144 + 64, TM = k::kQsaFullMaxT, N = 2560;
constexpr size_t XQ_ROW = (size_t) NH * HD / 32 * 36;   // one q8_1 row of the attention output

#define CK(x) do { cudaError_t e_ = (x); if (e_) { std::printf("CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__); std::exit(1); } } while (0)
template <class T> T* dalloc(size_t n) { T* p; CK(cudaMalloc(&p, n * sizeof(T) + 64)); return p; }
template <class T> void dput(T* p, const std::vector<T>& v) { CK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); }
template <class T> std::vector<T> dget(const T* p, size_t n) { std::vector<T> v(n); CK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return v; }
size_t ndiff(const void* a, const void* b, size_t bytes) {
    const auto x = dget((const unsigned char*) a, bytes), y = dget((const unsigned char*) b, bytes);
    size_t c = 0;
    for (size_t i = 0; i < bytes; ++i) c += x[i] != y[i];
    return c;
}
double time_us(const std::function<void()>& f, bool cold, void* flush, size_t fb, int iters) {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int i = 0; i < 10; ++i) f();
    CK(cudaDeviceSynchronize());
    std::vector<float> v;
    if (!cold) {
        cudaEventRecord(e0);
        for (int i = 0; i < iters; ++i) f();
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        return 1000.0 * ms / iters;
    }
    for (int i = 0; i < iters; ++i) {   // the weights and cache come from DRAM: a flush first
        cudaMemsetAsync(flush, i, fb); cudaEventRecord(e0); f(); cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1); v.push_back(ms * 1000);
    }
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// one set of QSA state and output buffers (a copy for the kernels of today, a copy for the fused kernel)
struct Set {
    float *kcur, *vcur, *qfull, *qcur, *qidx, *idx_raw, *tail, *dead, *pooled, *tail_snap, *scores, *scratch, *attn, *gated, *bo;
    int32_t *block_pos, *ids;
    int8_t *k_q, *v_q;
    uint16_t *k_s, *v_s;
    unsigned char* xq;
};
const size_t KQ = (size_t) (MAXC / PG + 2) * NKV * PG * HD, KS = (size_t) (MAXC / PG + 2) * NKV * PG * 4, POOL = (size_t) (MAXC / PG + 3) * ID;
int64_t g_mb, g_cap;
size_t g_scr;
Set make_set() {
    Set s{};
    s.kcur = dalloc<float>(TM * NKV * HD); s.vcur = dalloc<float>(TM * NKV * HD); s.qfull = dalloc<float>(TM * NH * 2 * HD);
    s.qcur = dalloc<float>(TM * NH * HD); s.qidx = dalloc<float>(TM * IQ * ID); s.idx_raw = dalloc<float>(TM * ID);
    s.tail = dalloc<float>(3 * ID); s.dead = dalloc<float>(ID); s.pooled = dalloc<float>(POOL); s.tail_snap = dalloc<float>(3 * ID);
    s.scores = dalloc<float>(TM * g_mb); s.scratch = dalloc<float>(g_scr * TM); s.attn = dalloc<float>(TM * NH * HD);
    s.gated = dalloc<float>(TM * NH * HD); s.bo = dalloc<float>(TM * N);
    s.block_pos = dalloc<int32_t>(1); s.ids = dalloc<int32_t>(TM * g_cap);
    s.k_q = dalloc<int8_t>(KQ); s.v_q = dalloc<int8_t>(KQ); s.k_s = dalloc<uint16_t>(KS); s.v_s = dalloc<uint16_t>(KS);
    s.xq = dalloc<unsigned char>(TM * XQ_ROW);
    return s;
}
struct Common { float *knorm, *iknorm, *qnorm, *iqnorm; int32_t *steps, *pos_k, *pos_q, *pos_i, *table; };
struct Wts { int tq, tk, tv, to; void *wq, *wk, *wv, *wo, *wik, *wiq; unsigned char* xmix; float* mixed; };
Wts g_w;
cudaStream_t g_s;
const k::QsaShapes g_sh = k::qsa_real_shapes();
const k::RopeScaling g_rs;

k::QsaFullArgs make_args(const Set& s, const Common& c, int n, int ord) {
    k::QsaFullArgs a{};
    a.n = n; a.tq = g_w.tq; a.tk = g_w.tk; a.tv = g_w.tv; a.to = g_w.to;
    a.wq = g_w.wq; a.wk = g_w.wk; a.wv = g_w.wv; a.wo = g_w.wo; a.wik = (const uint16_t*) g_w.wik; a.wiq = (const uint16_t*) g_w.wiq;
    a.xmix = g_w.xmix; a.mixed = g_w.mixed;
    a.qfull = s.qfull; a.kcur = s.kcur; a.vcur = s.vcur; a.idx_raw = s.idx_raw; a.qidx = s.qidx; a.bo = s.bo;
    k::QsaDecKvPost& kp = a.kv;
    kp.kcur = s.kcur; kp.vcur = s.vcur; kp.pos_k = c.pos_k; kp.k_norm = c.knorm; kp.steps = c.steps;
    kp.k_q = s.k_q; kp.v_q = s.v_q; kp.k_scale = s.k_s; kp.v_scale = s.v_s; kp.page_table = c.table;
    kp.idx_raw = s.idx_raw; kp.idx_k_norm = c.iknorm; kp.tail = s.tail; kp.dead = s.dead; kp.pooled = s.pooled;
    kp.block_pos = s.block_pos; kp.tail_snap = s.tail_snap; kp.max_cells = MAXC; kp.eps = 1e-6f;
    k::QsaDecQPost& qp = a.qp;
    qp.q_full = s.qfull; qp.qcur = s.qcur; qp.pos_q = c.pos_q; qp.q_norm = c.qnorm; qp.q_idx = s.qidx; qp.pos_i = c.pos_i;
    qp.idx_q_norm = c.iqnorm; qp.eps = 1e-6f;
    a.shapes = g_sh;
    a.pooled = s.pooled; a.dead = s.dead; a.steps = c.steps; a.max_blocks = g_mb; a.cap = g_cap; a.scores = s.scores; a.ids = s.ids;
    a.pools.k_q = s.k_q; a.pools.v_q = s.v_q; a.pools.k_scale = s.k_s; a.pools.v_scale = s.v_s; a.pools.page_table = c.table;
    a.scratch = s.scratch; a.attn = s.attn; a.gated = s.gated;
    a.ord = ord;
    return a;
}
void requant_x() { k::native_quantize_q8_1(g_w.mixed, g_w.xmix, N, TM, g_s); CK(cudaStreamSynchronize(g_s)); }
// the sequence a verify window ran before qsa_full
void run_today(const Set& s, const Common& c, int n) {
    const k::QsaFullArgs a = make_args(s, c, n, 0);
    cudaStream_t st = g_s;
    k::dense_tc(g_w.tq, g_w.wq, g_w.xmix, s.qfull, N, 12288, n, st);
    k::native_mmvq(g_w.tk, g_w.wk, g_w.xmix, s.kcur, N, 512, n, st);
    k::native_mmvq(g_w.tv, g_w.wv, g_w.xmix, s.vcur, N, 512, n, st);
    k::bf16_gemv_fp32_mmvf_multi(g_w.mixed, N, (const uint16_t*) g_w.wik, s.idx_raw, 128, N, 128, n, st);
    k::bf16_gemv_fp32_mmvf_multi(g_w.mixed, N, (const uint16_t*) g_w.wiq, s.qidx, 512, N, 512, n, st);
    k::qsa_dec_kv_post(a.kv, n, g_sh, g_rs, st);
    k::qsa_dec_q_post(a.qp, n, g_sh, g_rs, st);
    k::qsa_block_scores(s.pooled, s.dead, s.qidx, c.steps, n, g_mb, g_sh, s.scores, st, -1, true);
    k::qsa_block_topk(s.scores, c.steps, n, g_mb, g_cap, g_sh, s.ids, st);
    k::qsa_dec_attn(s.qcur, a.pools, s.ids, c.steps, g_cap, g_sh, s.scratch, s.attn, n, st, s.qfull, s.gated, s.xq);
    k::dense_tc(g_w.to, g_w.wo, s.xq, s.bo, 6144, N, n, st);
}
int bb_of(int t) { return t == 12 ? 144 : t == 13 ? 176 : t == 14 ? 210 : 136; }
std::vector<uint8_t> make_weights(int type, int n_in, int n_out, std::mt19937& rng) {
    const size_t bb = bb_of(type), nb = (size_t) n_out * (n_in / 256);
    std::vector<uint8_t> w(nb * bb);
    for (auto& b : w) b = (uint8_t) (rng() & 0xff);
    const int at[2] = {type == 14 ? 208 : 0, (type == 12 || type == 13) ? 2 : -1};   // the fp16 scales stay finite
    for (size_t i = 0; i < nb; ++i)
        for (int a : at)
            if (a >= 0) {
                const uint32_t r = rng();
                const uint16_t h = (uint16_t) (((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
                std::memcpy(&w[i * bb + a], &h, 2);
            }
    return w;
}
std::vector<uint16_t> make_bf16(int n, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 0.05f);
    std::vector<uint16_t> v(n);
    for (auto& e : v) { const float f = nd(rng); uint32_t u; std::memcpy(&u, &f, 4); e = (uint16_t) (u >> 16); }
    return v;
}
void free_w() {
    for (void* p : {g_w.wq, g_w.wk, g_w.wv, g_w.wo, g_w.wik, g_w.wiq, (void*) g_w.xmix, (void*) g_w.mixed}) if (p) cudaFree(p);
    g_w = Wts{};
}
// the (q, k, v, o) projection types of the packs' QSA layers
const int kCombos[7][4] = {{23, 14, 14, 12}, {23, 23, 14, 14}, {12, 14, 13, 14}, {23, 14, 14, 14}, {13, 14, 13, 14}, {13, 14, 14, 14}, {14, 14, 14, 14}};
void setup_w(int ci, std::mt19937& rng) {
    free_w();
    g_w.tq = kCombos[ci][0]; g_w.tk = kCombos[ci][1]; g_w.tv = kCombos[ci][2]; g_w.to = kCombos[ci][3];
    auto up = [&](const void* h, size_t b) { void* d; CK(cudaMalloc(&d, b)); CK(cudaMemcpy(d, h, b, cudaMemcpyHostToDevice)); return d; };
    { auto w = make_weights(g_w.tq, N, 12288, rng); g_w.wq = up(w.data(), w.size()); }
    { auto w = make_weights(g_w.tk, N, 512, rng); g_w.wk = up(w.data(), w.size()); }
    { auto w = make_weights(g_w.tv, N, 512, rng); g_w.wv = up(w.data(), w.size()); }
    { auto w = make_weights(g_w.to, 6144, N, rng); g_w.wo = up(w.data(), w.size()); }
    { auto w = make_bf16(128 * N, rng); g_w.wik = up(w.data(), w.size() * 2); }
    { auto w = make_bf16(512 * N, rng); g_w.wiq = up(w.data(), w.size() * 2); }
    std::vector<float> x(TM * N);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : x) v = nd(rng);
    g_w.mixed = (float*) up(x.data(), x.size() * 4);
    CK(cudaMalloc((void**) &g_w.xmix, TM * XQ_ROW + 64));   // the fused kernel writes the attention rows over the mixed rows
    requant_x();
}
double h2f(uint16_t b) { __half h; std::memcpy(&h, &b, 2); return (double) __half2float(h); }
}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int iters = argc > 1 ? std::atoi(argv[1]) : 200;
    CK(cudaStreamCreate(&g_s));
    k::qsa_full_init();
    g_mb = MAXC / 4 + 2; g_cap = k::qsa_selection_width(k::kTopkMaxCells, g_sh); g_scr = k::qsa_decode_attn_scratch_floats(g_cap, g_sh);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto rv = [&](size_t n, float sd, float off = 0) { std::vector<float> v(n); for (auto& x : v) x = off + sd * nd(rng); return v; };
    Common c{};
    c.knorm = dalloc<float>(HD); c.iknorm = dalloc<float>(ID); c.qnorm = dalloc<float>(HD); c.iqnorm = dalloc<float>(ID);
    c.steps = dalloc<int32_t>(TM * k::kStepCount); c.pos_k = dalloc<int32_t>(TM * NKV); c.pos_q = dalloc<int32_t>(TM * NH);
    c.pos_i = dalloc<int32_t>(TM * IQ); c.table = dalloc<int32_t>(MAXC / PG + 2);
    dput(c.knorm, rv(HD, 0.3f, 1)); dput(c.iknorm, rv(ID, 0.3f, 1)); dput(c.qnorm, rv(HD, 0.3f, 1)); dput(c.iqnorm, rv(ID, 0.3f, 1));
    Set A = make_set(), B = make_set(), P0 = make_set();   // today, fused, pristine state
    setup_w(0, rng);
    {
        const k::QsaFullArgs probe = make_args(B, c, 1, 0);
        if (!k::qsa_full_supported(probe)) { std::printf("qsa_full_parity: qsa_full is not supported on this device\n"); return 77; }
    }
    auto f2h = [](float f) { return __half_as_ushort(__float2half_rn(f)); };
    {
        std::vector<int8_t> q(KQ);
        for (auto& x : q) x = (int8_t) (rng() & 0xff);
        dput(P0.k_q, q);
        for (auto& x : q) x = (int8_t) (rng() & 0xff);
        dput(P0.v_q, q);
        std::vector<uint16_t> s(KS);
        for (auto& x : s) x = f2h(0.002f + 0.03f * (rng() & 1023) / 1023.f);
        dput(P0.k_s, s);
        for (auto& x : s) x = f2h(0.002f + 0.03f * (rng() & 1023) / 1023.f);
        dput(P0.v_s, s);
    }
    dput(P0.tail, rv(3 * ID, 1)); dput(P0.dead, rv(ID, 1)); dput(P0.pooled, rv(POOL, 1));
    { int32_t bp = 12345; CK(cudaMemcpy(P0.block_pos, &bp, 4, cudaMemcpyHostToDevice)); }
    auto restore = [&](Set& d) {
        auto cp = [&](void* x, const void* y, size_t b) { CK(cudaMemcpy(x, y, b, cudaMemcpyDeviceToDevice)); };
        cp(d.k_q, P0.k_q, KQ); cp(d.v_q, P0.v_q, KQ); cp(d.k_s, P0.k_s, KS * 2); cp(d.v_s, P0.v_s, KS * 2);
        cp(d.tail, P0.tail, 3 * ID * 4); cp(d.dead, P0.dead, ID * 4); cp(d.pooled, P0.pooled, POOL * 4); cp(d.block_pos, P0.block_pos, 4);
        CK(cudaMemset(d.qcur, 0, TM * NH * HD * 4)); CK(cudaMemset(d.tail_snap, 0, 3 * ID * 4));
        CK(cudaMemset(d.scores, 0xEE, TM * g_mb * 4)); CK(cudaMemset(d.ids, 0xEE, TM * g_cap * 4)); CK(cudaMemset(d.scratch, 0xEE, g_scr * TM * 4));
        CK(cudaMemset(d.attn, 0xEE, TM * NH * HD * 4)); CK(cudaMemset(d.gated, 0xEE, TM * NH * HD * 4)); CK(cudaMemset(d.xq, 0xEE, TM * XQ_ROW));
        CK(cudaMemset(d.qfull, 0xEE, TM * NH * 2 * HD * 4)); CK(cudaMemset(d.kcur, 0xEE, TM * NKV * HD * 4)); CK(cudaMemset(d.vcur, 0xEE, TM * NKV * HD * 4));
        CK(cudaMemset(d.idx_raw, 0xEE, TM * ID * 4)); CK(cudaMemset(d.qidx, 0xEE, TM * IQ * ID * 4)); CK(cudaMemset(d.bo, 0xEE, TM * N * 4));
    };
    auto windows = [&](int n, int L, int p, std::vector<int32_t>& st) {
        const int pos0 = L == 0 ? 0 : L - 8 + p;
        st.assign((size_t) n * k::kStepCount, 0);
        std::vector<int32_t> pk(n * NKV), pq(n * NH), pi(n * IQ);
        for (int t = 0; t < n; ++t) {
            k::qsa_step_fill(&st[(size_t) t * k::kStepCount], pos0 + t, g_sh);
            for (int h = 0; h < NKV; ++h) pk[t * NKV + h] = pos0 + t;
            for (int h = 0; h < NH; ++h) pq[t * NH + h] = pos0 + t;
            for (int h = 0; h < IQ; ++h) pi[t * IQ + h] = pos0 + t;
        }
        std::vector<int32_t> tbl(MAXC / PG + 2);
        for (size_t i = 0; i < tbl.size(); ++i) tbl[i] = (int) ((i * 7919u + 13) % tbl.size());
        if (p & 2) for (auto& x : tbl) if (rng() % 20 == 0) x = -1;   // unmapped pages
        dput(c.table, tbl); dput(c.steps, st); dput(c.pos_k, pk); dput(c.pos_q, pq); dput(c.pos_i, pi);
        return tbl;
    };
    struct Case { int L, T, p; };
    int fails_tot = 0, cpu_cases = 0, cpu_fails = 0;
    double worst_cpu = 0;
    for (int ci = 0; ci < 7; ++ci) {
        setup_w(ci, rng);
        std::vector<Case> cs;
        if (ci == 6) {
            for (int L : {512, 4096, 32768, 262144}) for (int T = 1; T <= 7; ++T) for (int p = 0; p < 4; ++p) cs.push_back({L, T, p});
            for (int T = 1; T <= 7; ++T) cs.push_back({0, T, 0});
        } else {
            for (int L : {4096, 262144}) for (int T = 1; T <= 7; ++T) cs.push_back({L, T, 0});
        }
        int fails = 0;
        for (const Case& cc : cs) {
            const int n = cc.T;
            std::vector<int32_t> st;
            const auto tbl = windows(n, cc.L, cc.p, st);
            restore(A); restore(B);
            // today's kernels read the mixed rows; the fused kernel writes the attention rows over them
            requant_x();
            run_today(A, c, n);
            CK(cudaStreamSynchronize(g_s));
            requant_x();
            k::qsa_full(make_args(B, c, n, 0), g_s);
            CK(cudaStreamSynchronize(g_s));
            CK(cudaGetLastError());
            size_t d = 0, dd[16] = {};
            int i = 0;
            auto cmp = [&](const void* x, const void* y, size_t b) { dd[i] = ndiff(x, y, b); d += dd[i++]; };
            cmp(A.k_q, B.k_q, KQ); cmp(A.v_q, B.v_q, KQ); cmp(A.k_s, B.k_s, KS * 2); cmp(A.v_s, B.v_s, KS * 2);
            cmp(A.kcur, B.kcur, TM * NKV * HD * 4); cmp(A.qcur, B.qcur, TM * NH * HD * 4); cmp(A.qidx, B.qidx, TM * IQ * ID * 4);
            cmp(A.pooled, B.pooled, POOL * 4); cmp(A.tail, B.tail, 3 * ID * 4); cmp(A.dead, B.dead, ID * 4); cmp(A.tail_snap, B.tail_snap, 3 * ID * 4);
            cmp(A.block_pos, B.block_pos, 4); cmp(A.scores, B.scores, TM * g_mb * 4); cmp(A.ids, B.ids, TM * g_cap * 4);
            cmp(A.scratch, B.scratch, g_scr * n * 4); cmp(A.attn, B.attn, TM * NH * HD * 4);
            const size_t d2 = ndiff(A.gated, B.gated, TM * NH * HD * 4) + ndiff(A.xq, g_w.xmix, n * XQ_ROW);
            const size_t d3 = ndiff(A.qfull, B.qfull, TM * NH * 2 * HD * 4) + ndiff(A.vcur, B.vcur, TM * NKV * HD * 4) +
                              ndiff(A.idx_raw, B.idx_raw, TM * ID * 4) + ndiff(A.bo, B.bo, TM * N * 4);
            d += d2 + d3;
            if (d) {
                ++fails;
                std::printf("FAIL L=%d T=%d p=%d: %zu differing bytes [kq %zu vq %zu ks %zu vs %zu kcur %zu qcur %zu qidx %zu pooled %zu tail %zu dead %zu snap %zu bp %zu scores %zu ids %zu scratch %zu attn %zu gated+xq %zu proj %zu]\n",
                            cc.L, cc.T, cc.p, d, dd[0], dd[1], dd[2], dd[3], dd[4], dd[5], dd[6], dd[7], dd[8], dd[9], dd[10], dd[11], dd[12], dd[13], dd[14], dd[15], d2, d3);
            }
            // the CPU reference of the attention and its gate, through the cells the kernel selected and the cache it wrote
            if (ci == 6 && cc.L >= 4096 && cc.L <= 32768 && (cc.T == 1 || cc.T == 4 || cc.T == 7)) {
                const auto hkq = dget(B.k_q, KQ), hvq = dget(B.v_q, KQ);
                const auto hks = dget(B.k_s, KS), hvs = dget(B.v_s, KS);
                const auto hq = dget(B.qcur, (size_t) n * NH * HD), hqf = dget(B.qfull, (size_t) n * NH * 2 * HD);
                const auto hid = dget(B.ids, (size_t) n * g_cap);
                const auto hg = dget(B.gated, (size_t) n * NH * HD);
                double max_ref = 0, err = 0;
                for (int t = 0; t < n; ++t) {
                    const int w = st[(size_t) t * k::kStepCount + k::kStepWidth];
                    for (int h = 0; h < NH; ++h) {
                        const int kvh = h / (NH / NKV);
                        std::vector<double> sc;
                        std::vector<size_t> rows;
                        for (int j = 0; j < w; ++j) {
                            const int cell = hid[(size_t) t * g_cap + j];
                            if (cell < 0) continue;
                            const int page = tbl[cell / PG];
                            if (page < 0) continue;
                            const size_t r = ((size_t) page * NKV + kvh) * PG + cell % PG;
                            double dot = 0;
                            for (int e = 0; e < HD; ++e) dot += (double) hq[((size_t) t * NH + h) * HD + e] * hkq[r * HD + e] * h2f(hks[r * (HD / 64) + e / 64]);
                            sc.push_back(dot / 16.0);
                            rows.push_back(r);
                        }
                        double mx = -1e300, l = 0;
                        for (double x : sc) mx = std::max(mx, x);
                        std::vector<double> acc(HD, 0.0);
                        for (size_t j = 0; j < sc.size(); ++j) {
                            const double pw = std::exp(sc[j] - mx);
                            l += pw;
                            for (int e = 0; e < HD; ++e) acc[e] += pw * hvq[rows[j] * HD + e] * h2f(hvs[rows[j] * (HD / 64) + e / 64]);
                        }
                        for (int e = 0; e < HD; ++e) {
                            const double gate = 1.0 / (1.0 + std::exp(-(double) hqf[((size_t) t * NH + h) * 2 * HD + HD + e]));
                            const double ref = (l > 0 ? acc[e] / l : 0.0) * gate;
                            max_ref = std::max(max_ref, std::fabs(ref));
                            err = std::max(err, std::fabs(ref - hg[((size_t) t * NH + h) * HD + e]));
                        }
                    }
                }
                ++cpu_cases;
                worst_cpu = std::max(worst_cpu, err / std::max(max_ref, 1e-30));
                if (err > 4e-3 * std::max(max_ref, 1e-6)) {
                    ++cpu_fails;
                    std::printf("CPU FAIL L=%d T=%d p=%d: gated attention err %.3g of max|ref| %.3g\n", cc.L, cc.T, cc.p, err, max_ref);
                }
            }
        }
        std::printf("combo %d (q %d k %d v %d o %d): %zu cases (ctx 0/512/4096/32768/262144 x T1..7, unmapped pages), vs today's kernels bitwise: %d fail\n",
                    ci, g_w.tq, g_w.tk, g_w.tv, g_w.to, cs.size(), fails);
        fails_tot += fails;
    }
    std::printf("CPU reference (gated attention over the selected cells, double precision): %d cases, worst error %.2g of max|ref|, %d fail\n",
                cpu_cases, worst_cpu, cpu_fails);

    // the time of both with the weights and cache hot in L2 and from DRAM
    setup_w(6, rng);
    void* flush;
    const size_t fb = 512u << 20;
    CK(cudaMalloc(&flush, fb));
    for (int L : {4096, 32768}) {
        const int n = 3;
        std::vector<int32_t> st;
        windows(n, L, 0, st);
        restore(A); restore(B);
        requant_x();
        auto ft = [&] { k::native_quantize_q8_1(g_w.mixed, g_w.xmix, N, TM, g_s); run_today(A, c, n); };
        auto ff = [&] { k::native_quantize_q8_1(g_w.mixed, g_w.xmix, N, TM, g_s); k::qsa_full(make_args(B, c, n, 0), g_s); };
        const double hot_t = time_us(ft, false, flush, fb, iters), hot_f = time_us(ff, false, flush, fb, iters);
        const double cold_t = time_us(ft, true, flush, fb, 101), cold_f = time_us(ff, true, flush, fb, 101);
        std::printf("bench ctx %5d T%d: hot today %7.2f us fused %7.2f us (%.2fx) | cold today %7.2f us fused %7.2f us (%.2fx)\n", L, n, hot_t, hot_f,
                    hot_t / hot_f, cold_t, cold_f, cold_t / cold_f);
    }
    std::printf("qsa_full_parity: %s\n", fails_tot || cpu_fails ? "FAIL" : "ok");
    return fails_tot || cpu_fails ? 1 : 0;
}
