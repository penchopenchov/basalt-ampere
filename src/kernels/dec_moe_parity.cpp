// src/kernels/dec_moe_parity.cpp - STRATA_DEC_MOE's kernels on real GGUF experts against the path they replace.
//
//     build/dec_moe_parity <shard1.gguf> [layer,layer,...] [--bench | --stress N]
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

// --stress N: N peer windows (48 rings each) of random routings - T 1..8, rings with no, few, many or only peer
// entries, back to back, rung by CUDA0's route kernel or by the host, some abandoned midway - each under a watchdog:
// a window that does not finish within 5 s is a hang (the words and the kernel's abort record are printed).  Every
// finished window's last ring is checked bitwise against native_expert_grouped on the peer.  DM_STRESS_DUP=1 also
// routes tokens with repeated expert ids and ids past n_expert (the route kernel must keep them off the peer).
// DM_STRESS_TORN=1: the host-rung rings carry an expert id with different tiers in different tokens (what the route
// kernel publishes when it reads a residency table while an upload lands): the peer must neither hang nor fault.
void stress(const strata::GgufFile& gguf, int l, int gt, int dt, int windows) {
    const auto L = K::native_expert_layout(gt, dt, H, FF);
    constexpr int NX0 = 32, NP = 48, NX1 = 64, RINGS = 48;
    const bool dup = std::getenv("DM_STRESS_DUP") != nullptr, torn = std::getenv("DM_STRESS_TORN") != nullptr;
    const auto* tg = gguf.find("blk." + std::to_string(l) + ".ffn_gate_exps.weight");
    const auto* tu = gguf.find("blk." + std::to_string(l) + ".ffn_up_exps.weight");
    const auto* td = gguf.find("blk." + std::to_string(l) + ".ffn_down_exps.weight");
    std::vector<uint8_t> blobs((size_t) NX1 * L.bytes);
    for (int x = 0; x < NX1; ++x) {
        const size_t E = (size_t) ((x * 37 + 5) % NE);
        uint8_t* b = blobs.data() + (size_t) x * L.bytes;
        std::memcpy(b, gguf.tensor_data(*tg) + E * L.up_off, L.up_off);
        std::memcpy(b + L.up_off, gguf.tensor_data(*tu) + E * L.up_off, L.up_off);
        std::memcpy(b + L.down_off, gguf.tensor_data(*td) + E * (L.bytes - L.down_off), L.bytes - L.down_off);
    }
    std::mt19937 rng(std::getenv("DM_STRESS_SEED") ? (unsigned) std::atoi(std::getenv("DM_STRESS_SEED")) : 4242u);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> xh((size_t) MT * H);
    for (auto& v : xh) v = nd(rng);
    const size_t xq_stride = (size_t) MT * H / 32 * 36;
    // experts 0..NX0-1 on CUDA0, NX0..NX0+NP-1 on the peer (slot rotating with the layer), the rest on the CPU
    std::vector<int32_t> res0(NE, -1), res1(NE, -1);
    for (int e = 0; e < NX0; ++e) res0[e] = e;
    for (int e = NX0; e < NX0 + NP; ++e) res1[e] = e - NX0;
    auto slot1 = [&](int ly, int e) { return (e - NX0 + (ly % 5) * 7) % NX1; };
    cudaStream_t s0, s1;
    ck(cudaSetDevice(1), "dev1");
    ck(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking), "stream");
    uint8_t* blob1 = dmal<uint8_t>(blobs.size());
    ck(cudaMemcpy(blob1, blobs.data(), blobs.size(), cudaMemcpyHostToDevice), "blob1");
    std::vector<unsigned long long> ptab((size_t) RINGS * NE, 0ull);
    for (int ly = 0; ly < RINGS; ++ly)
        for (int e = NX0; e < NX0 + NP; ++e)
            ptab[(size_t) ly * NE + e] = (unsigned long long) (blob1 + (size_t) slot1(ly, e) * L.bytes);
    unsigned long long* d_ptab = dmal<unsigned long long>(ptab.size());
    ck(cudaMemcpy(d_ptab, ptab.data(), ptab.size() * 8, cudaMemcpyHostToDevice), "ptab");
    std::vector<K::NativeExpertLayout> lays(RINGS, L);
    K::NativeExpertLayout* d_lay = dmal<K::NativeExpertLayout>(RINGS);
    ck(cudaMemcpy(d_lay, lays.data(), RINGS * sizeof(K::NativeExpertLayout), cudaMemcpyHostToDevice), "lay");
    void* pscr = dmal<uint8_t>(K::dec_moe_peer_scratch_bytes((int) H, (int) FF));
    unsigned int* parr = dmal<unsigned int>(16);
    float* x1 = dmal<float>(xh.size());
    ck(cudaMemcpy(x1, xh.data(), xh.size() * 4, cudaMemcpyHostToDevice), "x1");
    void* xq1 = dmal<uint8_t>(xq_stride);
    K::quantize_q8_1_rows(x1, MT, H, xq1, nullptr);
    void* scr1 = dmal<uint8_t>(K::native_expert_scratch_bytes(K::kDmCap, FF));
    float* ref1 = dmal<float>((size_t) K::kDmCap * H);
    Groups G;
    G.upload();
    const int pblocks = K::dec_moe_peer_blocks();
    ck(cudaSetDevice(0), "dev0");
    ck(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking), "stream");
    uint8_t* blob0 = dmal<uint8_t>((size_t) NX0 * L.bytes);
    ck(cudaMemcpy(blob0, blobs.data(), (size_t) NX0 * L.bytes, cudaMemcpyHostToDevice), "blob0");
    int32_t* d_res0 = dmal<int32_t>(NE);
    int32_t* d_res1 = dmal<int32_t>(NE);
    ck(cudaMemcpy(d_res0, res0.data(), NE * 4, cudaMemcpyHostToDevice), "res0");
    ck(cudaMemcpy(d_res1, res1.data(), NE * 4, cudaMemcpyHostToDevice), "res1");
    float* d_x = dmal<float>(xh.size());
    ck(cudaMemcpy(d_x, xh.data(), xh.size() * 4, cudaMemcpyHostToDevice), "x");
    K::DmPlan* d_plan = dmal<K::DmPlan>(1);
    void* d_xq = dmal<uint8_t>(xq_stride);
    int32_t* d_ids = dmal<int32_t>((size_t) RINGS * K::kDmCap);
    float* d_parts = dmal<float>((size_t) K::kDmCap * H);
    void* d_scr = dmal<uint8_t>(K::native_expert_scratch_bytes(K::kDmCap, FF));
    unsigned int* d_ctr = dmal<unsigned int>(16);
    K::DmRing* h_rings = hmap<K::DmRing>(RINGS);
    float* h_x = hmap<float>((size_t) RINGS * MT * H);
    uint8_t* h_xq = hmap<uint8_t>((size_t) RINGS * xq_stride);
    uint32_t* h_words = hmap<uint32_t>(64);   // [0] host seq [16] peer seq [32] peer done [48] host flag
    float* h_peer = hmap<float>((size_t) K::kDmCap * H);
    uint32_t* h_diag = hmap<uint32_t>(K::kDmDiagWords);
    std::printf("stress layer %d (gate/up %d, down %d), %d peer blocks, %d windows%s%s\n", l, gt, dt, pblocks, windows,
                dup ? ", repeated and invalid ids" : "", torn ? ", torn tiers" : "");

    auto peer_launch = [&](int steps) {
        std::memset(h_diag, 0, K::kDmDiagWords * 4);
        ck(cudaSetDevice(1), "dev1");
        ck(cudaMemsetAsync(parr, 0, 64, s1), "arrive");
        K::DmPeerArgs pa;
        pa.rings_m = h_rings; pa.xq_m = h_xq; pa.xq_stride = xq_stride; pa.seq_m = h_words + 16;
        pa.done_m = h_words + 32; pa.rows_m = h_peer; pa.ptr = d_ptab; pa.lay = d_lay;
        pa.n_expert = (int) NE; pa.k = (int) KK; pa.n_embd = (int) H; pa.n_ff = (int) FF; pa.steps = steps;
        pa.scratch = pscr; pa.arrive = parr; pa.diag_m = h_diag;
        if (!K::dec_moe_peer_launch(pa, pblocks, s1)) std::exit(2);
        ck(cudaSetDevice(0), "dev0");
    };
    auto stream_wait = [&](int dev, cudaStream_t s, double sec) {
        ck(cudaSetDevice(dev), "dev");
        const auto t0 = std::chrono::steady_clock::now();
        cudaError_t q;
        while ((q = cudaStreamQuery(s)) == cudaErrorNotReady &&
               std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < sec) {}
        ck(cudaSetDevice(0), "dev0");
        if (q != cudaErrorNotReady && q != cudaSuccess) {
            std::printf("CUDA error on device %d: %s\n", dev, cudaGetErrorString(q));
            std::exit(2);
        }
        return q == cudaSuccess;
    };
    auto diag_line = [&] {
        char b[512];
        std::snprintf(b, sizeof b, "words: host seq %u, peer seq %#x, peer done %u; peer: %s", h_words[0], h_words[16],
                      h_words[32], K::dec_moe_peer_diag(h_diag).c_str());
        return std::string(b);
    };
    auto hang = [&](const char* what, int w) {
        std::printf("HANG in window %d (%s): %s\n", w, what, diag_line().c_str());
        *(volatile uint32_t*) (h_words + 16) = UINT32_MAX;
        *(volatile uint32_t*) (h_words + 32) = UINT32_MAX;
        const bool p = stream_wait(1, s1, 2.0), c = stream_wait(0, s0, 2.0);
        std::printf("  after the release: peer kernel %s, CUDA0 %s; %s\n", p ? "exited" : "STILL RUNNING",
                    c ? "finished" : "STILL RUNNING", diag_line().c_str());
        std::fflush(stdout);
        std::_Exit(3);
    };
    // ring r's ids: T tokens of KK ids each; the share of peer entries per ring varies widely
    std::vector<int32_t> ids((size_t) RINGS * K::kDmCap);
    auto make_window = [&](int T) {
        for (int r = 0; r < RINGS; ++r) {
            const int mode = (int) (rng() % 6);   // 0: no peer, 1: one peer expert, 2-3: mixed, 4: mostly peer, 5: all
            const int pp = mode == 0 ? 0 : mode == 1 ? 8 : mode <= 3 ? 35 : mode == 4 ? 80 : 100;
            const int npool = mode == 1 ? 1 : 2 + (int) (rng() % (NP - 2));   // distinct peer experts available
            const int pbase = NX0 + (int) (rng() % (NP - npool + 1));
            const int rdup = dup ? (int) (rng() % 4) : 0;   // 1: a repeated id, 2: an id past n_expert, 3: both
            for (int t = 0; t < T; ++t) {
                std::vector<int> pick;
                int guard = 0;
                while ((int) pick.size() < KK) {
                    const int u = (int) (rng() % 100);
                    int e;
                    if (u < pp) e = pbase + (int) (rng() % npool);
                    else if (u < pp + (100 - pp) / 2) e = (int) (rng() % NX0);
                    else e = NX0 + NP + (int) (rng() % (NE - NX0 - NP));
                    if (++guard > 1000) e = NX0 + NP + (int) pick.size();
                    if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e);
                }
                if ((rdup & 1) && t % 2 == 0) pick[KK - 1] = pick[(size_t) (rng() % (KK - 1))];
                if ((rdup & 2) && t == T - 1) pick[KK - 2] = (int) NE + 3;
                for (int j = 0; j < KK; ++j) ids[(size_t) r * K::kDmCap + t * KK + j] = pick[j];
            }
        }
        ck(cudaMemcpy(d_ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice), "ids");
    };
    // CUDA0's rings (route, the VRAM share, the wait), the peer launched first or after the first rings
    auto routed = [&](int T, int w, int abandon_at) {
        std::memset(h_words, 0, 256);
        h_words[48] = 0xfffffff0u;   // the CPU tier: never waited for
        const bool late = rng() % 4 == 0;
        if (!late) peer_launch(RINGS);
        for (int r = 0; r < RINGS; ++r) {
            if (r == abandon_at) break;
            K::DmRouteArgs ra;
            ra.ids = d_ids + (size_t) r * K::kDmCap; ra.x = d_x; ra.n_tok = T; ra.k = (int) KK; ra.n_embd = (int) H;
            ra.n_expert = (int) NE; ra.layer = r; ra.res0 = d_res0; ra.res1 = d_res1; ra.cache_base = blob0;
            ra.blob = (long long) L.bytes; ra.plan = d_plan; ra.xq = d_xq; ra.ring_m = h_rings + r;
            ra.x_m = h_x + (size_t) r * MT * H; ra.xq_m = h_xq + r * xq_stride; ra.host_seq_m = h_words;
            ra.peer_seq_m = h_words + 16; ra.ring = (uint32_t) r + 1; ra.ctr = d_ctr;
            K::dec_moe_route(ra, s0);
            if (r % 3 == 0)
                K::native_expert_grouped(L, d_plan->ptr, d_plan->start, d_plan->counts, d_plan->dst, d_plan->tok,
                                         T * (int) KK, T * (int) KK, d_xq, d_scr, d_parts, s0);
            K::dec_moe_wait(d_plan, h_words + 32, h_words + 48, (uint32_t) r + 1, s0);
            if (late && r == 2) peer_launch(RINGS);
        }
        if (late && abandon_at >= 0 && abandon_at <= 2) peer_launch(RINGS);
        if (!stream_wait(0, s0, 5.0)) hang("routed, CUDA0", w);
        if (abandon_at >= 0) *(volatile uint32_t*) (h_words + 16) = UINT32_MAX;   // the peer stops at its next doorbell
        if (!stream_wait(1, s1, 5.0)) hang(abandon_at >= 0 ? "abandoned, peer" : "routed, peer", w);
    };
    // the peer alone, rung by the host over the rings the last routed window left (each doorbell right after the
    // previous ring landed, the first one before the kernel starts)
    // an earlier entry of a peer expert routed again in a later token moves to the CPU or CUDA0 tier
    auto tear = [&](int T) {
        const int n = T * (int) KK;
        for (int r = 0; r < RINGS; ++r) {
            K::DmRing& R = h_rings[r];
            for (int i = 0; i < n; ++i) {
                if (R.kind[i] != 2) continue;
                for (int j = (i / KK + 1) * KK; j < n; ++j)
                    if (R.ids[j] == R.ids[i] && R.kind[j] == 2 && rng() % 2 == 0) {
                        R.kind[i] = rng() % 2 ? -1 : 0;
                        break;
                    }
            }
            int np = 0;
            for (int i = 0; i < n; ++i) np += R.kind[i] == 2;
            R.n_peer = np;
        }
    };
    auto host_rung = [&](int T, int w) {
        std::memset(h_words, 0, 256);
        peer_launch(RINGS);
        for (int r = 0; r < RINGS; ++r) {
            *(volatile uint32_t*) (h_words + 16) =
                ((uint32_t) r + 1) | (uint32_t) T << 16 | (h_rings[r].n_peer > 0 ? 1u << 28 : 0u);
            if (h_rings[r].n_peer == 0) continue;
            const auto t0 = std::chrono::steady_clock::now();
            while (*(volatile uint32_t*) (h_words + 32) < (uint32_t) r + 1)
                if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > 5.0)
                    hang("host-rung", w);
        }
        if (!stream_wait(1, s1, 5.0)) hang("host-rung, peer exit", w);
    };
    // the last ring's peer rows against native_expert_grouped on the peer
    auto check = [&](int T, int w) {
        const int r = RINGS - 1, n = T * (int) KK;
        G.ptr.clear(); G.start.clear(); G.dst.clear(); G.tok.clear();
        std::vector<int> rows;
        const int32_t* id = ids.data() + (size_t) r * K::kDmCap;
        for (int i = 0; i < n; ++i) {
            if (h_rings[r].kind[i] != 2) continue;
            bool first = true;
            for (int j = 0; j < i; ++j) first &= !(h_rings[r].kind[j] == 2 && id[j] == id[i]);
            if (!first) continue;
            G.ptr.push_back((unsigned long long) (blob1 + (size_t) slot1(r, id[i]) * L.bytes));
            G.start.push_back((int) G.dst.size());
            for (int j = i; j < n; ++j)
                if (h_rings[r].kind[j] == 2 && id[j] == id[i]) { G.dst.push_back(j); G.tok.push_back(j / KK); rows.push_back(j); }
        }
        if (rows.empty()) return 0;
        G.start.push_back((int) G.dst.size());
        ck(cudaSetDevice(1), "dev1");
        const int ng = (int) G.ptr.size();
        // on the stream that reads them: a pageable cudaMemcpy may return before its DMA lands
        ck(cudaMemcpyAsync(G.d_ptr, G.ptr.data(), G.ptr.size() * 8, cudaMemcpyHostToDevice, s1), "up");
        ck(cudaMemcpyAsync(G.d_start, G.start.data(), G.start.size() * 4, cudaMemcpyHostToDevice, s1), "up");
        ck(cudaMemcpyAsync(G.d_n, &ng, 4, cudaMemcpyHostToDevice, s1), "up");
        ck(cudaMemcpyAsync(G.d_dst, G.dst.data(), G.dst.size() * 4, cudaMemcpyHostToDevice, s1), "up");
        ck(cudaMemcpyAsync(G.d_tok, G.tok.data(), G.tok.size() * 4, cudaMemcpyHostToDevice, s1), "up");
        std::vector<float> b(h_peer, h_peer + (size_t) K::kDmCap * H);   // before anything else runs
        K::native_expert_grouped(L, G.d_ptr, G.d_start, G.d_n, G.d_dst, G.d_tok, K::kDmCap, K::kDmCap, xq1, scr1, ref1, s1);
        ck(cudaStreamSynchronize(s1), "ref");
        const auto a = down(ref1, (size_t) K::kDmCap * H);
        K::native_expert_grouped(L, G.d_ptr, G.d_start, G.d_n, G.d_dst, G.d_tok, K::kDmCap, K::kDmCap, xq1, scr1, ref1, s1);
        ck(cudaStreamSynchronize(s1), "ref");
        const auto a2 = down(ref1, (size_t) K::kDmCap * H);
        ck(cudaSetDevice(0), "dev0");
        const size_t nd = ndiff_rows(a, b, rows), nr = ndiff_rows(a, a2, rows);
        if (nd || nr) {
            std::vector<int> bad;
            for (int r : rows)
                if (std::memcmp(&a[(size_t) r * H], &b[(size_t) r * H], H * 4) != 0) bad.push_back(r);
            std::string br;
            for (int r : bad) br += " " + std::to_string(r);
            std::printf("  window %d T=%d: last ring's peer rows differ (%zu of %zu; rows%s)%s\n", w, T, nd,
                        rows.size() * H, br.c_str(), nr ? "; the reference differs between two calls" : "");
            ++g_fail;
        }
        return 1;
    };
    int checked = 0, aborted = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int w = 0; w < windows; ++w) {
        const int T = 1 + (int) (rng() % MT);
        make_window(T);
        const int kind = (int) (rng() % 8);   // 0-4 routed, 5 routed then host-rung, 6-7 abandoned midway
        if (kind <= 5) {
            routed(T, w, -1);
            if (!dup) checked += check(T, w);
            if (kind == 5 || torn) {
                if (torn) tear(T);
                host_rung(T, w);
                if (torn && h_diag[0] == 2u) h_diag[0] = 0;   // a refused ring is the right outcome
            }
        } else {
            routed(T, w, (int) (rng() % RINGS));
            if (h_diag[0] == 3u) h_diag[0] = 0;   // released inside a ring: expected when abandoned
        }
        if (h_diag[0] != 0) {
            ++aborted;
            if (aborted <= 5) std::printf("  window %d T=%d: %s\n", w, T, diag_line().c_str());
            ++g_fail;
        }
        if ((w + 1) % 500 == 0)
            std::printf("  %d windows, %.1f s\n", w + 1,
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    std::printf("stress: %d windows without a hang, %d aborted, %d checked bitwise\n", windows, aborted, checked);
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 2) {
        std::fprintf(stderr, "usage: dec_moe_parity <shard1.gguf> [layer,layer,...] [--bench | --stress N]\n");
        return 2;
    }
    bool bench = false;
    int stress_n = 0;
    std::vector<int> layers;
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bench") == 0) { bench = true; continue; }
        if (std::strcmp(argv[i], "--stress") == 0 && i + 1 < argc) { stress_n = std::atoi(argv[++i]); continue; }
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
    if (stress_n > 0) {   // the first listed layer only
        if (!two || fmt[layers[0]].first < 0) { std::fprintf(stderr, "--stress: two cards and a layer with experts\n"); return 2; }
        stress(gguf, layers[0], fmt[layers[0]].first, fmt[layers[0]].second, stress_n);
        std::printf("dec_moe_parity --stress: %s\n", g_fail ? "FAILED" : "no hang, bitwise equal");
        return g_fail ? 1 : 0;
    }

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
            {   // the split route (mode 1 then mode 2) leaves every byte the one launch left
                const std::vector<uint8_t> ring0((const uint8_t*) (h_rings + 5), (const uint8_t*) (h_rings + 6));
                const std::vector<uint8_t> xm0((const uint8_t*) (h_x + (size_t) 5 * MT * H),
                                               (const uint8_t*) (h_x + (size_t) 5 * MT * H + (size_t) T * H));
                const std::vector<uint8_t> xqm0(h_xq + 5 * xq_stride, h_xq + 5 * xq_stride + q1.size());
                const uint32_t w0 = h_words[0], w16 = h_words[16];
                std::memset(h_rings + 5, 0, sizeof(K::DmRing));
                std::memset(h_x + (size_t) 5 * MT * H, 0, (size_t) T * H * 4);
                std::memset(h_xq + 5 * xq_stride, 0, q1.size());
                h_words[0] = h_words[16] = 0;
                ck(cudaMemset(d_plan, 0xee, sizeof(K::DmPlan)), "memset");
                ck(cudaMemset(d_xq2, 0xee, q1.size()), "memset");
                void* d_pub = dmal<uint8_t>(xq_stride);
                K::DmRouteArgs r1 = ra, r2 = ra;
                r1.mode = 1; r2.mode = 2; r2.xq_pub = d_pub;
                K::dec_moe_route(r1, s0);
                K::dec_moe_route(r2, s0);
                ck(cudaStreamSynchronize(s0), "split route");
                K::DmPlan P2;
                ck(cudaMemcpy(&P2, d_plan, sizeof P2, cudaMemcpyDeviceToHost), "plan");
                const auto q3 = down((const uint8_t*) d_xq2, q1.size());
                size_t bad2 = std::memcmp(P.counts, P2.counts, sizeof P.counts) != 0;
                for (int g = 0; g <= P.counts[0]; ++g) bad2 += P.start[g] != P2.start[g] || (g < P.counts[0] && P.ptr[g] != P2.ptr[g]);
                for (int e = 0; e < P.counts[1]; ++e) bad2 += P.dst[e] != P2.dst[e] || P.tok[e] != P2.tok[e];
                for (int i = 0; i < n; ++i) bad2 += P.kind[i] != P2.kind[i];
                bad2 += std::memcmp(q3.data(), q2.data(), q2.size()) != 0;
                bad2 += std::memcmp(ring0.data(), h_rings + 5, ring0.size()) != 0;
                if (nc > 0) bad2 += std::memcmp(xm0.data(), h_x + (size_t) 5 * MT * H, xm0.size()) != 0;
                if (np > 0) bad2 += std::memcmp(xqm0.data(), h_xq + 5 * xq_stride, xqm0.size()) != 0;
                bad2 += h_words[0] != w0 || h_words[16] != w16;
                cudaFree(d_pub);
                std::snprintf(what, sizeof what, "split route T=%d", T);
                report(what, l, bad2, (size_t) 6);
            }

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
            {   // the combine without its own wait (verify's call; STRATA_DEC_GLUE & 4 takes the four-column kernel)
                float* d_bo3 = dmal<float>((size_t) MT * H);
                K::dec_moe_combine(d_parts, h_peer, h_cpu, d_plan, d_w, d_sh, d_bo3, H, KK, T, nullptr, nullptr, 6, s0);
                ck(cudaStreamSynchronize(s0), "combine");
                const auto c3 = down(d_bo3, (size_t) T * H);
                size_t nd3 = 0;
                for (size_t i = 0; i < a.size(); ++i) nd3 += std::memcmp(&a[i], &c3[i], 4) != 0;
                std::snprintf(what, sizeof what, "combine, no wait T=%d", T);
                report(what, l, nd3, a.size());
                if (bench) {
                    const double uc = time_us(s0, 200, [&] {
                        K::dec_moe_combine(d_parts, h_peer, h_cpu, d_plan, d_w, d_sh, d_bo3, H, KK, T, nullptr, nullptr, 6, s0);
                    });
                    std::printf("  bench: combine, no wait T=%d (%d peer, %d cpu rows of %d): %.2f us per launch\n", T, np,
                                nc, n, uc);
                }
                cudaFree(d_bo3);
            }

            // STRATA_DEC_GLUE: the plan built in the gate/up launch (+ the ring launch's kinds) vs the route's plan,
            // its rows vs native_expert_grouped's (d_parts2); needs STRATA_DEC_GU2=2
            if (K::dec_moe_glue_ok(L)) {
                std::vector<unsigned long long> pt(NE);
                for (int e = 0; e < NE; ++e)
                    pt[e] = res0[e] >= 0 ? (unsigned long long) (dv[0].blob + (size_t) res0[e] * L.bytes) : 0ull;
                unsigned long long* d_pt = dmal<unsigned long long>(NE);
                ck(cudaMemcpy(d_pt, pt.data(), NE * 8, cudaMemcpyHostToDevice), "ptab");
                float* d_p4 = dmal<float>((size_t) K::kDmCap * H);
                void* d_pub = dmal<uint8_t>(xq_stride);
                ck(cudaMemset(d_plan, 0xee, sizeof(K::DmPlan)), "memset");
                K::DmRouteArgs r2 = ra;
                r2.mode = 2; r2.xq_pub = d_pub; r2.kinds = true;
                K::dec_moe_route(r2, s0);
                K::DmGlueArgs ga;
                ga.ids = d_ids; ga.ptab = d_pt; ga.n_tok = T; ga.k = (int) KK; ga.n_expert = (int) NE;
                ga.plan = d_plan; ga.xq = d_xq2;
                K::dec_moe_glue(L, ga, dv[0].scr, d_p4, n, s0);
                ck(cudaStreamSynchronize(s0), "glue");
                K::DmPlan P2;
                ck(cudaMemcpy(&P2, d_plan, sizeof P2, cudaMemcpyDeviceToHost), "plan");
                size_t bad3 = std::memcmp(P.counts, P2.counts, sizeof P.counts) != 0;
                for (int g = 0; g <= P.counts[0]; ++g) bad3 += P.start[g] != P2.start[g] || (g < P.counts[0] && P.ptr[g] != P2.ptr[g]);
                for (int e = 0; e < P.counts[1]; ++e) bad3 += P.dst[e] != P2.dst[e] || P.tok[e] != P2.tok[e];
                for (int i = 0; i < n; ++i) bad3 += P.kind[i] != P2.kind[i];
                std::snprintf(what, sizeof what, "glue plan T=%d", T);
                report(what, l, bad3, (size_t) n);
                const auto a4 = down(d_parts2, (size_t) n * H), b4 = down(d_p4, (size_t) n * H);
                std::vector<int> rows;
                for (int i = 0; i < n; ++i) if (kind[i] == 0) rows.push_back(i);
                std::snprintf(what, sizeof what, "glue rows T=%d (%d groups)", T, P.counts[0]);
                report(what, l, ndiff_rows(a4, b4, rows), rows.size() * H);
                {   // the ring's work in the launch: the ring, the mapped rows, the words and the kinds as route's
                    const std::vector<uint8_t> ring0((const uint8_t*) (h_rings + 5), (const uint8_t*) (h_rings + 6));
                    const std::vector<uint8_t> xm0((const uint8_t*) (h_x + (size_t) 5 * MT * H),
                                                   (const uint8_t*) (h_x + (size_t) 5 * MT * H + (size_t) T * H));
                    const size_t nq = (size_t) T * H / 32 * 36;
                    const std::vector<uint8_t> xqm0(h_xq + 5 * xq_stride, h_xq + 5 * xq_stride + nq);
                    std::memset(h_rings + 5, 0, sizeof(K::DmRing));
                    std::memset(h_x + (size_t) 5 * MT * H, 0, (size_t) T * H * 4);
                    std::memset(h_xq + 5 * xq_stride, 0, nq);
                    h_words[0] = h_words[16] = 0;
                    ck(cudaMemset(d_plan, 0xee, sizeof(K::DmPlan)), "memset");
                    ck(cudaMemsetAsync(d_p4, 0, (size_t) K::kDmCap * H * 4, s0), "memset");
                    K::DmGlueArgs gr = ga;
                    gr.ring_on = true; gr.ring = r2;
                    K::dec_moe_glue(L, gr, dv[0].scr, d_p4, n, s0);
                    ck(cudaStreamSynchronize(s0), "glue ring");
                    K::DmPlan P3;
                    ck(cudaMemcpy(&P3, d_plan, sizeof P3, cudaMemcpyDeviceToHost), "plan");
                    size_t bad4 = std::memcmp(P.counts, P3.counts, sizeof P.counts) != 0;
                    for (int g = 0; g <= P.counts[0]; ++g) bad4 += P.start[g] != P3.start[g] || (g < P.counts[0] && P.ptr[g] != P3.ptr[g]);
                    for (int e = 0; e < P.counts[1]; ++e) bad4 += P.dst[e] != P3.dst[e] || P.tok[e] != P3.tok[e];
                    for (int i = 0; i < n; ++i) bad4 += P.kind[i] != P3.kind[i];
                    bad4 += std::memcmp(ring0.data(), h_rings + 5, ring0.size()) != 0;
                    if (nc > 0) bad4 += std::memcmp(xm0.data(), h_x + (size_t) 5 * MT * H, xm0.size()) != 0;
                    if (np > 0) bad4 += std::memcmp(xqm0.data(), h_xq + 5 * xq_stride, nq) != 0;
                    bad4 += h_words[0] != 6 || (h_words[16] & 0xffffu) != 6;
                    std::snprintf(what, sizeof what, "glue ring T=%d", T);
                    report(what, l, bad4, (size_t) n);
                    const auto c4 = down(d_p4, (size_t) n * H);
                    std::snprintf(what, sizeof what, "glue ring rows T=%d", T);
                    report(what, l, ndiff_rows(a4, c4, rows), rows.size() * H);
                }
                if (bench) {
                    const double ug = time_us(s0, 200, [&] { K::dec_moe_glue(L, ga, dv[0].scr, d_p4, n, s0); });
                    const double ur = time_us(s0, 200, [&] {
                        K::DmRouteArgs r1 = ra;
                        r1.mode = 1;
                        K::dec_moe_route(r1, s0);
                        K::native_expert_grouped(L, d_plan->ptr, d_plan->start, d_plan->counts, d_plan->dst, d_plan->tok,
                                                 n, n, d_xq2, dv[0].scr, d_parts2, s0);
                    });
                    std::printf("  bench: T=%d route + grouped %.2f us, glue %.2f us\n", T, ur, ug);
                }
                cudaFree(d_pt); cudaFree(d_p4); cudaFree(d_pub);
            } else {
                std::printf("  glue: skipped (STRATA_DEC_GU2=2 not set or formats)\n");
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
