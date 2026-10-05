// src/kernels/dec_moe_parity.cpp - STRATA_DEC_MOE's kernels on real GGUF experts against the path they replace.
//
//     build/dec_moe_parity <shard1.gguf> [layer,layer,...] [--bench]
//
// Per layer (default: one of each gate/up + down format pair):
//   peer math   the persistent kernel's work items (dec_moe_peer_math_test) vs native_expert_grouped, every card
//   route       dec_moe_route's plan, kinds, q8_1 rows (vs quantize_q8_1_rows) and mapped publications
//   tail        route + native_expert_grouped + dec_moe_combine vs the old chain (grouped into hit rows, the mapped
//               CPU/peer rows copied, hit add, native_moe_combine_multi)
//   window      (two cards) 48 rings through the peer kernel: the peer rows vs native_expert_grouped on the peer
// All bitwise.  --bench adds: the VRAM share's grouped call (us, GB/s of expert bytes), and per ring on CUDA0 the
// route + wait with and without peer entries, alone and behind the VRAM call (what CUDA0 waits for the peer).
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/dec_moe.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace K = strata::kernels;

namespace {

int g_fail = 0;
constexpr int64_t H = 2560, FF = 640, NE = 512, KK = 10, MT = 8;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}
template<typename T> T* dmal(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc((void**) &p, n * sizeof(T) + 256), "cudaMalloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 256), "cudaMemset");
    ck(cudaDeviceSynchronize(), "alloc sync");   // the tests' streams are non-blocking
    return p;
}
template<typename T> T* hmap(size_t n) {
    T* p = nullptr;
    ck(cudaHostAlloc((void**) &p, n * sizeof(T) + 256, cudaHostAllocMapped | cudaHostAllocPortable), "cudaHostAlloc");
    std::memset((void*) p, 0, n * sizeof(T) + 256);
    return p;
}
template<typename T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return h;
}
void report(const char* what, int layer, size_t ndiff, size_t n) {
    std::printf("  %-34s layer %2d: %s (%zu of %zu differ)\n", what, layer, ndiff ? "DIFFERS" : "bitwise equal", ndiff, n);
    if (ndiff) ++g_fail;
}
size_t ndiff_rows(const std::vector<float>& a, const std::vector<float>& b, const std::vector<int>& rows) {
    size_t d = 0;
    for (int r : rows)
        for (int64_t c = 0; c < H; ++c) d += std::memcmp(&a[(size_t) r * H + c], &b[(size_t) r * H + c], 4) != 0;
    return d;
}

struct Groups {   // a grouped call's lists on one device
    std::vector<unsigned long long> ptr;
    std::vector<int32_t> start, dst, tok;
    unsigned long long* d_ptr = nullptr;
    int32_t *d_start = nullptr, *d_n = nullptr, *d_dst = nullptr, *d_tok = nullptr;
    void upload() {
        const int ng = (int) ptr.size();
        d_ptr = dmal<unsigned long long>(K::kDmCap);
        d_start = dmal<int32_t>(K::kDmCap + 1);
        d_n = dmal<int32_t>(4);
        d_dst = dmal<int32_t>(K::kDmCap);
        d_tok = dmal<int32_t>(K::kDmCap);
        ck(cudaMemcpy(d_ptr, ptr.data(), ptr.size() * 8, cudaMemcpyHostToDevice), "up");
        ck(cudaMemcpy(d_start, start.data(), start.size() * 4, cudaMemcpyHostToDevice), "up");
        ck(cudaMemcpy(d_n, &ng, 4, cudaMemcpyHostToDevice), "up");
        ck(cudaMemcpy(d_dst, dst.data(), dst.size() * 4, cudaMemcpyHostToDevice), "up");
        ck(cudaMemcpy(d_tok, tok.data(), tok.size() * 4, cudaMemcpyHostToDevice), "up");
    }
};

double time_us(cudaStream_t s, int iters, const std::function<void()>& f) {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (int i = 0; i < 5; ++i) f();
    double best = 1e30;
    for (int rep = 0; rep < 3; ++rep) {
        cudaEventRecord(e0, s);
        for (int i = 0; i < iters; ++i) f();
        cudaEventRecord(e1, s);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        best = std::min(best, 1e3 * ms / iters);
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 2) {
        std::fprintf(stderr, "usage: dec_moe_parity <shard1.gguf> [layer,layer,...] [--bench]\n");
        return 2;
    }
    bool bench = false;
    std::vector<int> layers;
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bench") == 0) { bench = true; continue; }
        std::stringstream ss(argv[i]);
        std::string t;
        while (std::getline(ss, t, ',')) layers.push_back(std::atoi(t.c_str()));
    }
    strata::GgufFile gguf(argv[1]);
    int ndev = 0;
    ck(cudaGetDeviceCount(&ndev), "device count");
    const bool two = ndev >= 2;
    std::printf("dec_moe_parity: %d device(s)%s\n", ndev, two ? " (window test on device 1)" : "");
    // the layers' formats; by default the first layer of each pair
    std::vector<std::pair<int, int>> fmt(64, {-1, -1});
    std::vector<int> auto_layers;
    for (int l = 0; l < 64; ++l) {
        const auto* g = gguf.find("blk." + std::to_string(l) + ".ffn_gate_exps.weight");
        const auto* d = gguf.find("blk." + std::to_string(l) + ".ffn_down_exps.weight");
        if (!g || !d) continue;
        fmt[l] = {(int) g->type, (int) d->type};
        bool seen = false;
        for (int a : auto_layers) seen |= fmt[a] == fmt[l];
        if (!seen) auto_layers.push_back(l);
    }
    if (layers.empty()) layers = auto_layers;

    cudaStream_t s0, s1 = nullptr;
    ck(cudaSetDevice(0), "dev0");
    ck(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking), "stream");
    if (two) {
        ck(cudaSetDevice(1), "dev1");
        ck(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking), "stream");
        ck(cudaSetDevice(0), "dev0");
    }
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);

    for (int l : layers) {
        if (l < 0 || l >= 64 || fmt[l].first < 0) { std::printf("layer %d: no experts\n", l); ++g_fail; continue; }
        const int gt = fmt[l].first, dt = fmt[l].second;
        const auto L = K::native_expert_layout(gt, dt, H, FF);
        std::printf("layer %d: gate/up type %d, down type %d, %.2f MB per expert, formats %s\n", l, gt, dt,
                    L.bytes / 1e6, K::dec_moe_formats_ok(gt, dt, H, FF) ? "covered" : "NOT COVERED");
        if (!K::dec_moe_formats_ok(gt, dt, H, FF)) { ++g_fail; continue; }
        const auto* tg = gguf.find("blk." + std::to_string(l) + ".ffn_gate_exps.weight");
        const auto* tu = gguf.find("blk." + std::to_string(l) + ".ffn_up_exps.weight");
        const auto* td = gguf.find("blk." + std::to_string(l) + ".ffn_down_exps.weight");
        // NX distinct experts' blobs; expert id e uses blob e % NX
        constexpr int NX = 48;
        std::vector<uint8_t> blobs((size_t) NX * L.bytes);
        for (int x = 0; x < NX; ++x) {
            const size_t E = (size_t) ((x * 37 + 5) % NE);
            uint8_t* b = blobs.data() + (size_t) x * L.bytes;
            std::memcpy(b, gguf.tensor_data(*tg) + E * L.up_off, L.up_off);
            std::memcpy(b + L.up_off, gguf.tensor_data(*tu) + E * L.up_off, L.up_off);
            std::memcpy(b + L.down_off, gguf.tensor_data(*td) + E * (L.bytes - L.down_off), L.bytes - L.down_off);
        }
        std::vector<float> x((size_t) MT * H);
        for (auto& v : x) v = nd(rng);

        struct Dev {
            uint8_t* blob = nullptr;
            float* x = nullptr;
            void* xq = nullptr;
            void* scr = nullptr;
            void* scr2 = nullptr;
            float* out = nullptr;
            float* out2 = nullptr;
        } dv[2];
        for (int d = 0; d < (two ? 2 : 1); ++d) {
            ck(cudaSetDevice(d), "dev");
            dv[d].blob = dmal<uint8_t>(blobs.size());
            ck(cudaMemcpy(dv[d].blob, blobs.data(), blobs.size(), cudaMemcpyHostToDevice), "blobs");
            dv[d].x = dmal<float>(x.size());
            ck(cudaMemcpy(dv[d].x, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
            dv[d].xq = dmal<uint8_t>((size_t) MT * H / 32 * 36);
            K::quantize_q8_1_rows(dv[d].x, MT, H, dv[d].xq, nullptr);
            dv[d].scr = dmal<uint8_t>(K::native_expert_scratch_bytes(K::kDmCap, FF));
            dv[d].scr2 = dmal<uint8_t>(K::dec_moe_peer_scratch_bytes((int) H, (int) FF));
            dv[d].out = dmal<float>((size_t) K::kDmCap * H);
            dv[d].out2 = dmal<float>((size_t) K::kDmCap * H);
        }
        ck(cudaSetDevice(0), "dev0");

        // ---- peer math: groups of 1..8 entries, scattered destinations
        for (int d = 0; d < (two ? 2 : 1); ++d) {
            ck(cudaSetDevice(d), "dev");
            Groups G;
            const int sizes[] = {1, 2, 3, 4, 5, 6, 7, 8, 1, 2};
            int e = 0;
            std::vector<int> perm(K::kDmCap);
            for (int i = 0; i < K::kDmCap; ++i) perm[i] = i;
            std::shuffle(perm.begin(), perm.end(), rng);
            std::vector<int> rows;
            for (int g = 0; g < 10; ++g) {
                G.ptr.push_back((unsigned long long) (dv[d].blob + (size_t) ((g * 5 + 1) % NX) * L.bytes));
                G.start.push_back(e);
                for (int j = 0; j < sizes[g]; ++j, ++e) {
                    G.tok.push_back((g + 3 * j) % MT);
                    G.dst.push_back(perm[e]);
                    rows.push_back(perm[e]);
                }
            }
            G.start.push_back(e);
            G.upload();
            const cudaStream_t s = d ? s1 : s0;
            ck(cudaMemsetAsync(dv[d].out, 0xff, (size_t) K::kDmCap * H * 4, s), "memset");
            ck(cudaMemsetAsync(dv[d].out2, 0xff, (size_t) K::kDmCap * H * 4, s), "memset");
            K::native_expert_grouped(L, G.d_ptr, G.d_start, G.d_n, G.d_dst, G.d_tok, K::kDmCap, K::kDmCap, dv[d].xq,
                                     dv[d].scr, dv[d].out, s);
            K::dec_moe_peer_math_test(L, G.d_ptr, G.d_start, (int) G.ptr.size(), G.d_tok, G.d_dst, dv[d].xq, dv[d].scr2,
                                      dv[d].out2, s);
            ck(cudaStreamSynchronize(s), "peer math");
            const auto a = down(dv[d].out, (size_t) K::kDmCap * H), b = down(dv[d].out2, (size_t) K::kDmCap * H);
            report(d ? "peer math (device 1)" : "peer math (device 0)", l, ndiff_rows(a, b, rows), rows.size() * H);
        }
        ck(cudaSetDevice(0), "dev0");

        // ---- route + tail on CUDA0, T = 1, 3, 8
        int32_t* d_res0 = dmal<int32_t>(NE);
        int32_t* d_res1 = dmal<int32_t>(NE);
        std::vector<int32_t> res0(NE), res1(NE);
        for (int e = 0; e < NE; ++e) {
            const int r = (int) (rng() % 8);
            res0[e] = r < 6 ? e % NX : -1;            // the slot = the blob
            res1[e] = r == 6 ? e % NX : -1;
        }
        ck(cudaMemcpy(d_res0, res0.data(), NE * 4, cudaMemcpyHostToDevice), "res");
        ck(cudaMemcpy(d_res1, res1.data(), NE * 4, cudaMemcpyHostToDevice), "res");
        K::DmPlan* d_plan = dmal<K::DmPlan>(1);
        void* d_xq2 = dmal<uint8_t>((size_t) MT * H / 32 * 36);
        int32_t* d_ids = dmal<int32_t>(K::kDmCap);
        float* d_w = dmal<float>(K::kDmCap);
        float* d_sh = dmal<float>((size_t) MT * H);
        float* d_bo = dmal<float>((size_t) MT * H);
        float* d_bo2 = dmal<float>((size_t) MT * H);
        float* d_parts = dmal<float>((size_t) K::kDmCap * H);
        float* d_hit = dmal<float>((size_t) K::kDmCap * H);
        unsigned int* d_ctr = dmal<unsigned int>(4);
        void* d_vscr = dmal<uint8_t>(K::dec_moe_vram_scratch_bytes((int) FF));
        unsigned int* d_vsync = dmal<unsigned int>(K::kDmCap + 1);
        constexpr int RINGS = 48;
        K::DmRing* h_rings = hmap<K::DmRing>(RINGS);
        float* h_x = hmap<float>((size_t) RINGS * MT * H);
        uint8_t* h_xq = hmap<uint8_t>((size_t) RINGS * MT * H / 32 * 36);
        uint32_t* h_words = hmap<uint32_t>(64);   // [0] host seq [16] peer seq [32] peer done [48] host flag
        float* h_cpu = hmap<float>((size_t) K::kDmCap * H);
        float* h_peer = hmap<float>((size_t) K::kDmCap * H);
        float* h_ymiss = hmap<float>((size_t) K::kDmCap * H);
        const size_t xq_stride = (size_t) MT * H / 32 * 36;
        for (int T : {1, 3, 8}) {
            const int n = T * (int) KK;
            std::vector<int32_t> ids(n);
            for (int t = 0; t < T; ++t) {
                std::vector<int> pick;
                while ((int) pick.size() < KK) {
                    const int e = (int) (rng() % 64);   // a small pool: tokens share experts
                    if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e);
                }
                for (int j = 0; j < KK; ++j) ids[t * KK + j] = pick[j];
            }
            std::vector<float> w(n), sh((size_t) T * H), cpu((size_t) n * H), peer((size_t) n * H);
            for (auto& v : w) v = std::fabs(nd(rng)) * 0.1f;
            for (auto& v : sh) v = nd(rng);
            for (auto& v : cpu) v = nd(rng);
            for (auto& v : peer) v = nd(rng);
            ck(cudaMemcpy(d_ids, ids.data(), n * 4, cudaMemcpyHostToDevice), "ids");
            ck(cudaMemcpy(d_w, w.data(), n * 4, cudaMemcpyHostToDevice), "w");
            ck(cudaMemcpy(d_sh, sh.data(), sh.size() * 4, cudaMemcpyHostToDevice), "sh");
            std::memcpy(h_cpu, cpu.data(), cpu.size() * 4);
            std::memcpy(h_peer, peer.data(), peer.size() * 4);
            std::memset(h_words, 0, 256);
            K::DmRouteArgs ra;
            ra.ids = d_ids; ra.x = dv[0].x; ra.n_tok = T; ra.k = (int) KK; ra.n_embd = (int) H; ra.n_expert = (int) NE;
            ra.layer = l; ra.res0 = d_res0; ra.res1 = d_res1; ra.cache_base = dv[0].blob; ra.slot_off = nullptr;
            ra.blob = (long long) L.bytes; ra.plan = d_plan; ra.xq = d_xq2; ra.ring_m = h_rings + 5;
            ra.x_m = h_x + (size_t) 5 * MT * H; ra.xq_m = h_xq + 5 * xq_stride; ra.host_seq_m = h_words;
            ra.peer_seq_m = h_words + 16; ra.ring = 6; ra.ctr = d_ctr;
            K::dec_moe_route(ra, s0);
            ck(cudaStreamSynchronize(s0), "route");
            // reference: kinds, groups, xq
            std::vector<int32_t> kind(n);
            int nc = 0, np = 0;
            for (int i = 0; i < n; ++i) {
                kind[i] = res0[ids[i]] >= 0 ? 0 : res1[ids[i]] >= 0 ? 2 : -1;
                nc += kind[i] < 0;
                np += kind[i] == 2;
            }
            K::DmPlan P;
            ck(cudaMemcpy(&P, d_plan, sizeof P, cudaMemcpyDeviceToHost), "plan");
            size_t bad = 0;
            std::vector<int> seen(n, 0);
            for (int g = 0; g < P.counts[0]; ++g)
                for (int e = P.start[g]; e < P.start[g + 1]; ++e) {
                    const int i = P.dst[e];
                    bad += i < 0 || i >= n || kind[i] != 0 || P.tok[e] != i / KK ||
                           P.ptr[g] != (unsigned long long) (dv[0].blob + (size_t) res0[ids[i]] * L.bytes);
                    if (i >= 0 && i < n) ++seen[i];
                }
            for (int i = 0; i < n; ++i) bad += (kind[i] == 0) != (seen[i] == 1) || P.kind[i] != kind[i] ||
                                               h_rings[5].kind[i] != kind[i] || h_rings[5].ids[i] != ids[i];
            bad += P.counts[2] != np || P.counts[3] != nc || h_rings[5].n_peer != np || h_rings[5].n_cpu != nc ||
                   h_rings[5].n_tok != T || h_words[0] != 6 || (h_words[16] & 0xffffu) != 6 ||
                   ((h_words[16] >> 16) & 0xffu) != (uint32_t) T || ((h_words[16] >> 28) & 1u) != (np > 0 ? 1u : 0u);
            const auto q1 = down((const uint8_t*) dv[0].xq, (size_t) T * H / 32 * 36);
            const auto q2 = down((const uint8_t*) d_xq2, (size_t) T * H / 32 * 36);
            bad += std::memcmp(q1.data(), q2.data(), q1.size()) != 0;
            if (np > 0) bad += std::memcmp(q1.data(), h_xq + 5 * xq_stride, q1.size()) != 0;
            if (nc > 0) bad += std::memcmp(x.data(), h_x + (size_t) 5 * MT * H, (size_t) T * H * 4) != 0;
            char what[64];
            std::snprintf(what, sizeof what, "route T=%d (%d cpu, %d peer)", T, nc, np);
            report(what, l, bad, (size_t) n);

            // tail: new = grouped into parts by the device plan, then dec_moe_combine reading the tiers' rows
            ck(cudaMemsetAsync(d_parts, 0xff, (size_t) K::kDmCap * H * 4, s0), "memset");
            K::native_expert_grouped(L, d_plan->ptr, d_plan->start, d_plan->counts, d_plan->dst, d_plan->tok, n, n,
                                     d_xq2, dv[0].scr, d_parts, s0);
            h_words[32] = 6;
            h_words[48] = 6;
            K::dec_moe_combine(d_parts, h_peer, h_cpu, d_plan, d_w, d_sh, d_bo, H, KK, T, h_words + 32, h_words + 48, 6, s0);
            // old: the mapped rows (CPU and peer rows, zeros at hits), copied with the hit rows zeroed, the hits
            // added, then the native combine
            for (int i = 0; i < n; ++i)
                std::memcpy(h_ymiss + (size_t) i * H, kind[i] == 0 ? std::vector<float>(H, 0.f).data()
                                                      : (kind[i] == 2 ? peer.data() : cpu.data()) + (size_t) i * H, H * 4);
            float* d_parts2 = d_hit + 0;   // reuse: hit rows go to d_hit, parts2 is a second buffer below
            float* d_p2 = dmal<float>((size_t) K::kDmCap * H);
            ck(cudaMemsetAsync(d_hit, 0, (size_t) K::kDmCap * H * 4, s0), "memset");
            K::native_expert_grouped(L, d_plan->ptr, d_plan->start, d_plan->counts, d_plan->dst, d_plan->tok, n, n,
                                     d_xq2, dv[0].scr, d_parts2, s0);
            K::copy_rows_from_mapped(d_p2, h_ymiss, n, H, d_plan->dst, d_plan->counts + 1, s0);
            K::moe_hit_add(d_p2, d_parts2, d_plan->dst, d_plan->counts + 1, n, H, s0);
            K::native_moe_combine_multi(d_p2, d_w, d_sh, d_bo2, H, KK, T, s0);
            ck(cudaStreamSynchronize(s0), "tail");
            const auto a = down(d_bo, (size_t) T * H), b = down(d_bo2, (size_t) T * H);
            size_t nd2 = 0;
            for (size_t i = 0; i < a.size(); ++i) nd2 += std::memcmp(&a[i], &b[i], 4) != 0;
            std::snprintf(what, sizeof what, "tail T=%d", T);
            report(what, l, nd2, a.size());
            cudaFree(d_p2);
            if (bench) {   // the combine alone, tiers landed 
                const double uc = time_us(s0, 200, [&] {
                    K::dec_moe_combine(d_parts, h_peer, h_cpu, d_plan, d_w, d_sh, d_bo, H, KK, T, h_words + 32,
                                       h_words + 48, 6, s0);
                });
                std::printf("  bench: combine T=%d (%d peer, %d cpu rows of %d): %.2f us per launch\n", T, np, nc, n, uc);
            }

            // the fused VRAM call vs native_expert_grouped (d_parts2 holds the grouped rows)
            {
                float* d_p3 = dmal<float>((size_t) K::kDmCap * H);
                ck(cudaMemsetAsync(d_p3, 0, (size_t) K::kDmCap * H * 4, s0), "memset");
                K::dec_moe_vram(L, d_plan, d_xq2, d_vscr, d_p3, d_vsync, s0);
                ck(cudaStreamSynchronize(s0), "vram");
                const auto a3 = down(d_parts2, (size_t) n * H), b3 = down(d_p3, (size_t) n * H);
                std::vector<int> rows;
                for (int i = 0; i < n; ++i) if (kind[i] == 0) rows.push_back(i);
                std::snprintf(what, sizeof what, "fused VRAM T=%d (%d groups)", T, P.counts[0]);
                report(what, l, ndiff_rows(a3, b3, rows), rows.size() * H);
                ck(cudaMemsetAsync(d_p3, 0, (size_t) K::kDmCap * H * 4, s0), "memset");
                for (int kind : {1, 2, 3}) {
                    ck(cudaMemsetAsync(d_p3, 0, (size_t) K::kDmCap * H * 4, s0), "memset");
                    K::dec_moe_vram(L, d_plan, d_xq2, d_vscr, d_p3, d_vsync, s0, kind, n);
                    ck(cudaStreamSynchronize(s0), "vram kind");
                    const auto t3 = down(d_p3, (size_t) n * H);
                    std::snprintf(what, sizeof what, "VRAM kind %d T=%d (%d groups)", kind, T, P.counts[0]);
                    report(what, l, ndiff_rows(a3, t3, rows), rows.size() * H);
                }
                if (std::getenv("DM_DEBUG")) {
                    const auto sy = down(d_vsync, K::kDmCap + 1);
                    int nz = 0;
                    for (auto v : sy) nz += v != 0;
                    std::printf("    sync words nonzero after: %d\n", nz);
                    {   // gate/up of the fused call vs the grouped call's scratch (gate | up at the same offsets? no: compare per entry)
                        const auto gf = down((const float*) d_vscr, (size_t) K::kDmCap * FF * 2);
                        const size_t f = (size_t) n * FF * 4, fa = (f + 255) & ~(size_t) 255;
                        const auto gg = down((const float*) dv[0].scr, (size_t) n * FF);
                        const auto ug = down((const float*) ((uint8_t*) dv[0].scr + fa), (size_t) n * FF);
                        size_t dg = 0, du = 0;
                        for (size_t i = 0; i < (size_t) n * FF; ++i) {
                            dg += std::memcmp(&gf[i], &gg[i], 4) != 0;
                            du += std::memcmp(&gf[(size_t) K::kDmCap * FF + i], &ug[i], 4) != 0;
                        }
                        std::printf("    gate differs %zu, up differs %zu (of %zu)\n", dg, du, (size_t) n * FF);
                        for (int rep = 0; rep < 4; ++rep) {
                            ck(cudaMemsetAsync(d_p3, 0, (size_t) K::kDmCap * H * 4, s0), "memset");
                            if (rep & 1) ck(cudaMemsetAsync(d_vscr, 0xff, K::dec_moe_vram_scratch_bytes((int) FF), s0), "nan");
                            K::dec_moe_vram(L, d_plan, d_xq2, d_vscr, d_p3, d_vsync, s0);
                            ck(cudaStreamSynchronize(s0), "vram2");
                            const auto c3 = down(d_p3, (size_t) n * H);
                            size_t zeros = 0, nans = 0;
                            for (int r : rows)
                                for (int64_t c = 0; c < H; ++c) {
                                    const float v = c3[(size_t) r * H + c];
                                    zeros += v == 0.0f && a3[(size_t) r * H + c] != 0.0f;
                                    nans += std::isnan(v);
                                }
                            std::printf("    rerun %d (%s scratch) differs from grouped: %zu (zeros %zu nans %zu)\n", rep,
                                        rep & 1 ? "NaN" : "kept", ndiff_rows(a3, c3, rows), zeros, nans);
                        }
                    }
                    for (int g = 0; g < P.counts[0]; ++g)
                        for (int e = P.start[g]; e < P.start[g + 1]; ++e) {
                            const int i = P.dst[e];
                            size_t d = 0, first = H;
                            for (int64_t c = 0; c < H; ++c)
                                if (std::memcmp(&a3[(size_t) i * H + c], &b3[(size_t) i * H + c], 4) != 0) { ++d; if (first == (size_t) H) first = c; }
                            if (d) std::printf("    group %d entry %d (n %d): %zu differ, first col %zu\n", g, e,
                                               P.start[g + 1] - P.start[g], d, first);
                        }
                }
                cudaFree(d_p3);
            }
            if (bench && T == 3) {
                // eight copies of the blobs (past the L2), the route kernel pointing each call at the next one
                uint8_t* big = dmal<uint8_t>(8 * blobs.size());
                for (int c = 0; c < 8; ++c)
                    ck(cudaMemcpy(big + c * blobs.size(), blobs.data(), blobs.size(), cudaMemcpyHostToDevice), "big");
                int rot = 0;
                auto route = [&] {
                    K::DmRouteArgs rb = ra;
                    rb.cache_base = big + (size_t) (rot++ % 8) * blobs.size();
                    K::dec_moe_route(rb, s0);
                };
                const double r0 = time_us(s0, 200, route);
                const double ug = time_us(s0, 200, [&] {
                    route();
                    K::native_expert_grouped(L, d_plan->ptr, d_plan->start, d_plan->counts, d_plan->dst, d_plan->tok,
                                             n, n, d_xq2, dv[0].scr, d_parts, s0);
                }) - r0;
                const double uf = time_us(s0, 200, [&] { route(); K::dec_moe_vram(L, d_plan, d_xq2, d_vscr, d_parts, d_vsync, s0); }) - r0;
                const double ut = time_us(s0, 200, [&] { route(); K::dec_moe_vram(L, d_plan, d_xq2, d_vscr, d_parts, d_vsync, s0, 1); }) - r0;
                const double u2 = time_us(s0, 200, [&] { route(); K::dec_moe_vram(L, d_plan, d_xq2, d_vscr, d_parts, d_vsync, s0, 2, n); }) - r0;
                const double u4 = time_us(s0, 200, [&] { route(); K::dec_moe_vram(L, d_plan, d_xq2, d_vscr, d_parts, d_vsync, s0, 3, n); }) - r0;
                auto phase = [&](int ph) {
                    K::native_expert_set_mode(-1, ph);
                    const double u = time_us(s0, 200, [&] {
                        route();
                        K::native_expert_grouped(L, d_plan->ptr, d_plan->start, d_plan->counts, d_plan->dst, d_plan->tok,
                                                 n, n, d_xq2, dv[0].scr, d_parts, s0);
                    }) - r0;
                    K::native_expert_set_mode(-1, 0);
                    return u;
                };
                const double ugu = phase(1), udn = phase(2);
                std::printf("  bench: grouped phases: gate/up+SwiGLU %.1f us (%.0f GB/s), down %.1f us (%.0f GB/s)\n", ugu,
                            P.counts[0] * (double) (2 * L.up_off) / ugu / 1e3, udn,
                            P.counts[0] * (double) (L.bytes - L.down_off) / udn / 1e3);
                const double gb = P.counts[0] * (double) L.bytes / 1e3;
                std::printf("  bench: VRAM share T=%d, %d groups (%d entries, %.1f MB), past the L2: route %.1f us | "
                            "grouped %.1f us %.0f GB/s | fused %.1f us %.0f GB/s | TMA %.1f us %.0f GB/s | PF1 %.1f us %.0f GB/s | "
                            "PF2 %.1f us %.0f GB/s\n",
                            T, P.counts[0], P.counts[1], gb / 1e3, r0, ug, gb / ug, uf, gb / uf, ut, gb / ut, u2, gb / u2, u4, gb / u4);
                cudaFree(big);
            }
        }

        // ---- the window on two cards
        if (two) {
            ck(cudaSetDevice(1), "dev1");
            std::vector<unsigned long long> ptab((size_t) 64 * NE, 0ull);
            for (int e = 0; e < NE; ++e)
                if (res1[e] >= 0) ptab[(size_t) l * NE + e] = (unsigned long long) (dv[1].blob + (size_t) res1[e] * L.bytes);
            unsigned long long* d_ptab = dmal<unsigned long long>(ptab.size());
            ck(cudaMemcpy(d_ptab, ptab.data(), ptab.size() * 8, cudaMemcpyHostToDevice), "ptab");
            std::vector<K::NativeExpertLayout> lays(64, L);
            K::NativeExpertLayout* d_lay = dmal<K::NativeExpertLayout>(64);
            ck(cudaMemcpy(d_lay, lays.data(), 64 * sizeof(K::NativeExpertLayout), cudaMemcpyHostToDevice), "lay");
            void* pscr = dmal<uint8_t>(K::dec_moe_peer_scratch_bytes((int) H, (int) FF));
            unsigned int* parr = dmal<unsigned int>(4);
            const int pblocks = K::dec_moe_peer_blocks();
            ck(cudaSetDevice(0), "dev0");
            const int T = 3, n = T * (int) KK;
            // one routing for every ring; peer entries guaranteed
            std::vector<int32_t> ids(n);
            int np = 0;
            for (int t = 0; t < T; ++t)
                for (int j = 0; j < KK; ++j) ids[t * KK + j] = (t * 7 + j * 13) % 64;
            for (int i = 0; i < n; ++i) np += res0[ids[i]] < 0 && res1[ids[i]] >= 0;
            ck(cudaMemcpy(d_ids, ids.data(), n * 4, cudaMemcpyHostToDevice), "ids");
            auto run_window = [&](bool peer_on, bool vram, int rings) {
                std::memset(h_words, 0, 256);
                h_words[48] = 0xfffffff0u;   // no CPU wait
                if (peer_on) {
                    ck(cudaSetDevice(1), "dev1");
                    ck(cudaMemsetAsync(parr, 0, 16, s1), "arrive");
                    K::DmPeerArgs pa;
                    pa.rings_m = h_rings; pa.xq_m = h_xq; pa.xq_stride = xq_stride; pa.seq_m = h_words + 16;
                    pa.done_m = h_words + 32; pa.rows_m = h_peer; pa.ptr = d_ptab; pa.lay = d_lay;
                    pa.n_expert = (int) NE; pa.k = (int) KK; pa.n_embd = (int) H; pa.n_ff = (int) FF; pa.steps = rings;
                    pa.scratch = pscr; pa.arrive = parr;
                    if (!K::dec_moe_peer_launch(pa, pblocks, s1)) std::exit(2);
                    ck(cudaSetDevice(0), "dev0");
                }
                for (int r = 0; r < rings; ++r) {
                    K::DmRouteArgs ra;
                    ra.ids = d_ids; ra.x = dv[0].x + (size_t) (r % 2) * 4 * H;   // rings alternate inputs
                    ra.n_tok = T; ra.k = (int) KK; ra.n_embd = (int) H;
                    ra.n_expert = (int) NE; ra.layer = l; ra.res0 = d_res0; ra.res1 = peer_on ? d_res1 : nullptr;
                    ra.cache_base = dv[0].blob; ra.blob = (long long) L.bytes; ra.plan = d_plan; ra.xq = d_xq2;
                    ra.ring_m = h_rings + r; ra.x_m = h_x + (size_t) r * MT * H; ra.xq_m = h_xq + r * xq_stride;
                    ra.host_seq_m = h_words; ra.peer_seq_m = h_words + 16; ra.ring = (uint32_t) r + 1; ra.ctr = d_ctr;
                    K::dec_moe_route(ra, s0);
                    if (vram)
                        K::native_expert_grouped(L, d_plan->ptr, d_plan->start, d_plan->counts, d_plan->dst,
                                                 d_plan->tok, n, n, d_xq2, dv[0].scr, d_parts, s0);
                    K::dec_moe_wait(d_plan, h_words + 32, h_words + 48, (uint32_t) r + 1, s0);
                }
            };
            run_window(true, false, RINGS);
            ck(cudaStreamSynchronize(s0), "window");
            ck(cudaSetDevice(1), "dev1");
            ck(cudaStreamSynchronize(s1), "window peer");
            // reference for the last ring's peer rows: native_expert_grouped on device 1
            {
                Groups G;
                std::vector<int> rows;
                int e = 0;
                for (int i = 0; i < n; ++i) {
                    if (!(res0[ids[i]] < 0 && res1[ids[i]] >= 0)) continue;
                    bool first = true;
                    for (int j = 0; j < i; ++j) first &= ids[j] != ids[i];
                    if (!first) continue;
                    G.ptr.push_back((unsigned long long) (dv[1].blob + (size_t) res1[ids[i]] * L.bytes));
                    G.start.push_back(e);
                    for (int j = i; j < n; ++j)
                        if (ids[j] == ids[i]) { G.dst.push_back(j); G.tok.push_back(j / KK + 4); rows.push_back(j); ++e; }
                }
                G.start.push_back(e);
                G.upload();
                ck(cudaMemsetAsync(dv[1].out, 0, (size_t) K::kDmCap * H * 4, s1), "memset");
                K::native_expert_grouped(L, G.d_ptr, G.d_start, G.d_n, G.d_dst, G.d_tok, n, n, dv[1].xq, dv[1].scr,
                                         dv[1].out, s1);
                ck(cudaStreamSynchronize(s1), "peer ref");
                const auto a = down(dv[1].out, (size_t) K::kDmCap * H);
                std::vector<float> b(h_peer, h_peer + (size_t) K::kDmCap * H);
                char what[64];
                std::snprintf(what, sizeof what, "window peer rows (%d entries)", (int) rows.size());
                report(what, l, ndiff_rows(a, b, rows), rows.size() * H);
                bool done_ok = h_words[32] == (uint32_t) RINGS;
                if (!done_ok) { std::printf("  peer done word %u, want %d\n", h_words[32], RINGS); ++g_fail; }
            }
            ck(cudaSetDevice(0), "dev0");
            if (bench) {
                auto wall = [&](bool peer_on, bool vram) {
                    double best = 1e30;
                    for (int rep = 0; rep < 5; ++rep) {
                        ck(cudaDeviceSynchronize(), "sync");
                        const auto t0 = std::chrono::steady_clock::now();
                        run_window(peer_on, vram, RINGS);
                        ck(cudaStreamSynchronize(s0), "w");
                        const auto t1 = std::chrono::steady_clock::now();
                        best = std::min(best, std::chrono::duration<double, std::micro>(t1 - t0).count() / RINGS);
                        ck(cudaSetDevice(1), "dev1");
                        ck(cudaStreamSynchronize(s1), "w1");
                        ck(cudaSetDevice(0), "dev0");
                    }
                    return best;
                };
                // the peer alone, rung from the host (the rings the window above left in mapped memory)
                {
                    unsigned long long* h_st = hmap<unsigned long long>((size_t) RINGS * 8);
                    std::memset(h_words, 0, 256);
                    ck(cudaSetDevice(1), "dev1");
                    ck(cudaMemsetAsync(parr, 0, 16, s1), "arrive");
                    K::DmPeerArgs pa;
                    pa.rings_m = h_rings; pa.xq_m = h_xq; pa.xq_stride = xq_stride; pa.seq_m = h_words + 16;
                    pa.done_m = h_words + 32; pa.rows_m = h_peer; pa.ptr = d_ptab; pa.lay = d_lay;
                    pa.n_expert = (int) NE; pa.k = (int) KK; pa.n_embd = (int) H; pa.n_ff = (int) FF; pa.steps = RINGS;
                    pa.scratch = pscr; pa.arrive = parr; pa.stamps_m = h_st;
                    if (!K::dec_moe_peer_launch(pa, pblocks, s1)) std::exit(2);
                    ck(cudaSetDevice(0), "dev0");
                    std::vector<double> rt;
                    for (int r = 0; r < RINGS; ++r) {
                        const auto t0 = std::chrono::steady_clock::now();
                        *(volatile uint32_t*) (h_words + 16) = ((uint32_t) r + 1) | (uint32_t) T << 16 | 1u << 28;
                        while (*(volatile uint32_t*) (h_words + 32) < (uint32_t) r + 1) {}
                        rt.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
                    }
                    ck(cudaSetDevice(1), "dev1");
                    ck(cudaStreamSynchronize(s1), "lat");
                    ck(cudaSetDevice(0), "dev0");
                    std::sort(rt.begin() + 1, rt.end());
                    double ph[4] = {};
                    for (int r = 1; r < RINGS; ++r)
                        for (int i = 0; i < 4; ++i) ph[i] += (double) (h_st[r * 8 + i + 1] - h_st[r * 8 + i]) / 1e3 / (RINGS - 1);
                    std::printf("  bench: peer rung by the host: round trip median %.1f us (min %.1f); block 0 phases us: "
                                "fetch+sync %.1f group %.1f experts %.1f done %.1f\n",
                                rt[RINGS / 2], rt[1], ph[0], ph[1], ph[2], ph[3]);
                }
                const double a0 = wall(false, false), a1 = wall(true, false), b0 = wall(false, true), b1 = wall(true, true);
                std::printf("  bench: per ring on CUDA0 (T=%d, %d peer entries, %d peer blocks): route+wait %.1f us, "
                            "+peer %.1f us | route+VRAM+wait %.1f us, +peer %.1f us\n",
                            T, np, pblocks, a0, a1, b0, b1);
            }
        }
        for (int d = 0; d < (two ? 2 : 1); ++d) {
            ck(cudaSetDevice(d), "dev");
            ck(cudaDeviceSynchronize(), "sync");
            cudaFree(dv[d].blob);
        }
        ck(cudaSetDevice(0), "dev0");
    }
    std::printf("dec_moe_parity: %s\n", g_fail ? "FAILED" : "all bitwise equal");
    return g_fail ? 1 : 0;
}
