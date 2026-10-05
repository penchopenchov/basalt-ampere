// src/kernels/dec_moe3_parity.cpp - STRATA_DEC_MOE3 (dec_moe3) on real GGUF experts and shared experts against the
// path it replaces: dec_moe_route + native_expert_grouped + shared_expert_multi + dec_moe_combine.
//
//     build/dec_moe3_parity <shard1.gguf> [layer,layer,...] [--bench]
//
// Per layer (default: one of each routed format pair and each shared format set):
//   error    the layer's output rows vs the old chain, T = 1..7 (max abs error, max error relative to the row's
//            largest value), with entries on all three tiers (the peer's and the CPU's rows mapped, already landed)
//   T-indep  a pool of 8 tokens put in every window T = 1..7 at every column offset: each token's output row
//            bitwise the same in all of them
// --bench: per layer at T = 2 and 3, the old chain (shared expert on a forked stream, as the decode runs it, and
// serial) and dec_moe3, each captured in a CUDA graph; us per layer and GB/s of the expert bytes they read.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/dec_moe.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/shared_expert.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace K = strata::kernels;

namespace {

int g_fail = 0;
constexpr int64_t H = 2560, FF = 640, NE = 512, KK = 10, MT = 8, POOL = 8;

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
template<typename T> void up(T* d, const std::vector<T>& h) {
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
}

double graph_us(cudaStream_t s, int iters, const std::function<void(cudaStream_t, int)>& body) {
    cudaGraph_t g;
    cudaGraphExec_t ge;
    ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "capture");
    for (int i = 0; i < 8; ++i) body(s, i);   // 8 layers a graph, as consecutive layers run
    ck(cudaStreamEndCapture(s, &g), "end capture");
    ck(cudaGraphInstantiate(&ge, g, 0), "instantiate");
    for (int i = 0; i < 5; ++i) ck(cudaGraphLaunch(ge, s), "launch");
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    double best = 1e30;
    for (int rep = 0; rep < 5; ++rep) {
        cudaEventRecord(e0, s);
        for (int i = 0; i < iters; ++i) cudaGraphLaunch(ge, s);
        cudaEventRecord(e1, s);
        ck(cudaEventSynchronize(e1), "bench");
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        best = std::min(best, 1e3 * ms / (iters * 8));
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaGraphExecDestroy(ge);
    cudaGraphDestroy(g);
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 2) {
        std::fprintf(stderr, "usage: dec_moe3_parity <shard1.gguf> [layer,layer,...] [--bench]\n");
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
    auto tensor = [&](int l, const char* n) { return gguf.find("blk." + std::to_string(l) + "." + n); };
    if (layers.empty()) {   // the first layer of each (routed pair, shared formats)
        std::set<std::vector<int>> seen;
        for (int l = 0; l < 64; ++l) {
            const auto *g = tensor(l, "ffn_gate_exps.weight"), *d = tensor(l, "ffn_down_exps.weight"),
                       *sg = tensor(l, "ffn_gate_shexp.weight"), *su = tensor(l, "ffn_up_shexp.weight"),
                       *sd = tensor(l, "ffn_down_shexp.weight");
            if (!g || !d || !sg || !su || !sd) continue;
            const std::vector<int> key{(int) g->type, (int) d->type, (int) sg->type, (int) su->type, (int) sd->type};
            if (seen.insert(key).second) layers.push_back(l);
        }
    }
    K::native_grouped_set_gu2(2);   // the decode's grouped chain
    cudaStream_t s0, s1;
    ck(cudaSetDevice(0), "dev0");
    ck(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking), "stream");
    ck(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking), "stream");
    cudaEvent_t ev_fork, ev_join;
    ck(cudaEventCreateWithFlags(&ev_fork, cudaEventDisableTiming), "event");
    ck(cudaEventCreateWithFlags(&ev_join, cudaEventDisableTiming), "event");
    std::mt19937 rng(4321);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::printf("dec_moe3_parity: %zu layer(s)%s\n", layers.size(), bench ? ", bench" : "");

    // buffers shared by the layers
    K::DmPlan* d_plan = dmal<K::DmPlan>(1);
    float* d_x = dmal<float>((size_t) MT * H);
    uint16_t* d_xb = dmal<uint16_t>((size_t) MT * H);
    void* d_xq = dmal<uint8_t>((size_t) MT * H / 32 * 36);
    void* d_shq = dmal<uint8_t>((size_t) MT * H / 32 * 36);
    int32_t* d_ids = dmal<int32_t>(K::kDmCap);
    float* d_w = dmal<float>(K::kDmCap);
    float *d_sh = dmal<float>((size_t) MT * H), *d_sh2 = dmal<float>((size_t) MT * H);
    float *d_bo = dmal<float>((size_t) MT * H), *d_bo2 = dmal<float>((size_t) MT * H);
    float *d_parts = dmal<float>((size_t) K::kDmCap * H), *d_parts2 = dmal<float>((size_t) K::kDmCap * H);
    float *d_sgate = dmal<float>((size_t) MT * FF), *d_sup = dmal<float>((size_t) MT * FF), *d_sg = dmal<float>(MT);
    float* d_part = dmal<float>((size_t) MT * H / 256);
    unsigned int* d_ctr = dmal<unsigned int>(4);
    unsigned int* d_m3ctr = dmal<unsigned int>(K::kDmM3Ctr);
    void* d_scr = dmal<uint8_t>(K::native_expert_scratch_bytes(K::kDmCap, FF));
    void* d_m3scr = dmal<uint8_t>(K::dec_moe3_scratch_bytes((int) FF));
    int32_t *d_res0 = dmal<int32_t>(NE), *d_res1 = dmal<int32_t>(NE);
    K::DmRing* h_ring = hmap<K::DmRing>(1);
    float* h_x = hmap<float>((size_t) MT * H);
    uint8_t* h_xq = hmap<uint8_t>((size_t) MT * H / 32 * 36);
    uint32_t* h_words = hmap<uint32_t>(64);   // [0] host seq [16] peer seq [32] peer done [48] host flag
    float *h_cpu = hmap<float>((size_t) K::kDmCap * H), *h_peer = hmap<float>((size_t) K::kDmCap * H);
    ck(cudaDeviceSynchronize(), "alloc");

    for (int l : layers) {
        const auto *tg = tensor(l, "ffn_gate_exps.weight"), *tu = tensor(l, "ffn_up_exps.weight"),
                   *td = tensor(l, "ffn_down_exps.weight"), *sg = tensor(l, "ffn_gate_shexp.weight"),
                   *su = tensor(l, "ffn_up_shexp.weight"), *sd = tensor(l, "ffn_down_shexp.weight"),
                   *gi = tensor(l, "ffn_gate_inp_shexp.weight");
        if (!tg || !tu || !td || !sg || !su || !sd || !gi) { std::printf("layer %d: tensors missing\n", l); ++g_fail; continue; }
        const auto L = K::native_expert_layout((int) tg->type, (int) td->type, H, FF);
        const bool ok = K::dec_moe3_ok(L, (int) sg->type, (int) su->type, (int) sd->type);
        std::printf("layer %d: experts %d/%d, shared %d/%d/%d: %s\n", l, (int) tg->type, (int) td->type, (int) sg->type,
                    (int) su->type, (int) sd->type, ok ? "covered" : "NOT COVERED");
        if (!ok) { ++g_fail; continue; }
        const int NX = bench ? (int) NE : 64;   // bench: every expert its own blob (8 layers' worth, beyond L2)
        std::vector<uint8_t> blobs((size_t) NX * L.bytes);
        for (int x = 0; x < NX; ++x) {
            const size_t E = (size_t) ((x * 37 + 5) % NE);
            uint8_t* b = blobs.data() + (size_t) x * L.bytes;
            std::memcpy(b, gguf.tensor_data(*tg) + E * L.up_off, L.up_off);
            std::memcpy(b + L.up_off, gguf.tensor_data(*tu) + E * L.up_off, L.up_off);
            std::memcpy(b + L.down_off, gguf.tensor_data(*td) + E * (L.bytes - L.down_off), L.bytes - L.down_off);
        }
        uint8_t* d_blob = dmal<uint8_t>(blobs.size());
        up(d_blob, blobs);
        const size_t sgb = K::iq_row_bytes((int) sg->type, H) * FF, sub = K::iq_row_bytes((int) su->type, H) * FF,
                     sdb = K::iq_row_bytes((int) sd->type, FF) * H;
        uint8_t *d_sgw = dmal<uint8_t>(sgb), *d_suw = dmal<uint8_t>(sub), *d_sdw = dmal<uint8_t>(sdb);
        uint16_t* d_gi = dmal<uint16_t>(H);
        // bench: the graph's layer i reads copy i of the shared weights and experts 64 i on (cold in L2, as a window)
        int cur = 0;
        int32_t* d_ids_c[8] = {d_ids};
        const uint8_t *sgw_c[8] = {d_sgw}, *suw_c[8] = {d_suw}, *sdw_c[8] = {d_sdw};
        if (bench)
            for (int i = 1; i < 8; ++i) {
                d_ids_c[i] = dmal<int32_t>(K::kDmCap);
                uint8_t *a = dmal<uint8_t>(sgb), *b = dmal<uint8_t>(sub), *c = dmal<uint8_t>(sdb);
                ck(cudaMemcpy(a, d_sgw, sgb, cudaMemcpyDeviceToDevice), "copy");
                ck(cudaMemcpy(b, d_suw, sub, cudaMemcpyDeviceToDevice), "copy");
                ck(cudaMemcpy(c, d_sdw, sdb, cudaMemcpyDeviceToDevice), "copy");
                sgw_c[i] = a; suw_c[i] = b; sdw_c[i] = c;
            }
        ck(cudaMemcpy(d_sgw, gguf.tensor_data(*sg), sgb, cudaMemcpyHostToDevice), "sg");
        ck(cudaMemcpy(d_suw, gguf.tensor_data(*su), sub, cudaMemcpyHostToDevice), "su");
        ck(cudaMemcpy(d_sdw, gguf.tensor_data(*sd), sdb, cudaMemcpyHostToDevice), "sd");
        ck(cudaMemcpy(d_gi, gguf.tensor_data(*gi), H * 2, cudaMemcpyHostToDevice), "gi");
        // residency, in 16ths: DM3_VRAM on CUDA0 (default 12), DM3_PEER on the peer (default 2), the rest on the CPU
        const int nv16 = std::getenv("DM3_VRAM") ? std::atoi(std::getenv("DM3_VRAM")) : 12;
        const int np16 = std::getenv("DM3_PEER") ? std::atoi(std::getenv("DM3_PEER")) : 2;
        std::vector<int32_t> res0(NE), res1(NE);
        for (int e = 0; e < NE; ++e) {
            const int r = (int) (rng() % 16);
            res0[e] = r < nv16 ? e % NX : -1;
            res1[e] = r >= nv16 && r < nv16 + np16 ? e % NX : -1;
        }
        up(d_res0, res0);
        up(d_res1, res1);
        // the token pool: rows, routes (a small expert pool so tokens share experts), weights, peer / CPU rows
        std::vector<float> px((size_t) POOL * H), pw((size_t) POOL * KK), prow((size_t) POOL * KK * H);
        std::vector<int32_t> pid((size_t) POOL * KK);
        for (auto& v : px) v = nd(rng);
        for (int t = 0; t < POOL; ++t) {
            std::vector<int> pick;
            while ((int) pick.size() < KK) {
                const int e = (int) (rng() % 48);
                if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e);
            }
            for (int j = 0; j < KK; ++j) { pid[t * KK + j] = pick[j]; pw[t * KK + j] = std::fabs(nd(rng)) * 0.1f; }
        }
        for (auto& v : prow) v = nd(rng) * 0.05f;

        K::NativeSharedWeights nsw;
        nsw.gate_type = (int) sg->type; nsw.gate_data = d_sgw;
        nsw.up_type = (int) su->type; nsw.up_data = d_suw;
        nsw.down_type = (int) sd->type; nsw.down_data = d_sdw;
        nsw.q8_1 = d_shq;
        uint32_t ring = 0;
        // window T at column offset o: column c is pool token (c + o) % POOL
        auto load = [&](int T, int o) {
            std::vector<float> x((size_t) T * H), w((size_t) T * KK);
            std::vector<uint16_t> xb((size_t) T * H);
            std::vector<int32_t> ids((size_t) T * KK);
            for (int c = 0; c < T; ++c) {
                const int p = (c + o) % POOL;
                std::memcpy(&x[(size_t) c * H], &px[(size_t) p * H], H * 4);
                std::memcpy(&ids[(size_t) c * KK], &pid[(size_t) p * KK], KK * 4);
                std::memcpy(&w[(size_t) c * KK], &pw[(size_t) p * KK], KK * 4);
                std::memcpy(h_cpu + (size_t) c * KK * H, &prow[(size_t) p * KK * H], KK * H * 4);
                for (int64_t i = 0; i < KK * H; ++i) h_peer[(size_t) c * KK * H + i] = -prow[(size_t) p * KK * H + i];
            }
            for (size_t i = 0; i < x.size(); ++i) xb[i] = K::bf16_from_f32(x[i]);
            up(d_x, x);
            up(d_xb, xb);
            up(d_ids, ids);
            up(d_w, w);
            ++ring;
            h_words[32] = ring;   // both tiers' rows have landed
            h_words[48] = ring;
        };
        auto route = [&](int T, bool m3, cudaStream_t s) {
            K::DmRouteArgs ra;
            ra.ids = d_ids_c[cur]; ra.x = d_x; ra.n_tok = T; ra.k = (int) KK; ra.n_embd = (int) H; ra.n_expert = (int) NE;
            ra.layer = l; ra.res0 = d_res0; ra.res1 = d_res1; ra.cache_base = d_blob; ra.blob = (long long) L.bytes;
            ra.plan = d_plan; ra.xq = d_xq; ra.ring_m = h_ring; ra.x_m = h_x; ra.xq_m = h_xq; ra.host_seq_m = h_words;
            ra.peer_seq_m = h_words + 16; ra.ring = ring; ra.ctr = d_ctr;
            if (m3) { ra.sh_gw = d_gi; ra.sh_part = d_part; }
            K::dec_moe_route(ra, s);
        };
        auto old_chain = [&](int T, cudaStream_t s, bool fork) {
            route(T, false, s);
            cudaStream_t ss = s;
            if (fork) { cudaEventRecord(ev_fork, s); cudaStreamWaitEvent(s1, ev_fork, 0); ss = s1; }
            K::NativeSharedWeights w = nsw;
            w.gate_data = sgw_c[cur]; w.up_data = suw_c[cur]; w.down_data = sdw_c[cur];
            K::shared_expert_multi(T, d_x, d_xb, w, d_gi, d_sgate, d_sup, d_sg, d_sh, H, FF, ss);
            if (fork) cudaEventRecord(ev_join, s1);
            const int cap = T * (int) KK;
            K::native_expert_grouped(L, d_plan->ptr, d_plan->start, d_plan->counts, d_plan->dst, d_plan->tok, cap, cap,
                                     d_xq, d_scr, d_parts, s);
            if (fork) cudaStreamWaitEvent(s, ev_join, 0);
            K::dec_moe_wait(d_plan, h_words + 32, h_words + 48, ring, s);   // as the decode waits: one thread
            K::dec_moe_combine(d_parts, h_peer, h_cpu, d_plan, d_w, d_sh, d_bo, H, KK, T, nullptr, nullptr, ring, s);
        };
        auto new_chain = [&](int T, cudaStream_t s) {
            route(T, true, s);
            K::DmM3Args a;
            a.sg = sgw_c[cur]; a.su = suw_c[cur]; a.sd = sdw_c[cur];
            a.sg_type = (int) sg->type; a.su_type = (int) su->type; a.sd_type = (int) sd->type;
            a.sh_part = d_part; a.shared = d_sh2; a.parts = d_parts2; a.peer_rows_m = h_peer; a.cpu_rows_m = h_cpu;
            a.weights = d_w; a.output = d_bo2; a.peer_done_m = h_words + 32; a.host_flag_m = h_words + 48;
            a.ring = ring; a.ctr = d_m3ctr; a.n_tok = T; a.k = (int) KK;
            K::dec_moe3(L, d_plan, d_xq, d_m3scr, a, T * (int) KK, s);
        };

        // ---- error vs the old chain
        double worst_abs = 0, worst_rel = 0;
        for (int T = 1; T <= 7; ++T) {
            load(T, T);
            old_chain(T, s0, false);
            ck(cudaStreamSynchronize(s0), "old");
            const auto ref = down(d_bo, (size_t) T * H);
            const K::DmPlan P = down(d_plan, 1)[0];
            ck(cudaMemsetAsync(d_bo2, 0xff, (size_t) MT * H * 4, s0), "memset");
            new_chain(T, s0);
            ck(cudaStreamSynchronize(s0), "new");
            const auto got = down(d_bo2, (size_t) T * H);
            double ma = 0, mr = 0;
            bool finite = true;
            for (int t = 0; t < T; ++t) {
                double rmax = 0;
                for (int64_t c = 0; c < H; ++c) rmax = std::max(rmax, (double) std::fabs(ref[(size_t) t * H + c]));
                for (int64_t c = 0; c < H; ++c) {
                    const double a = got[(size_t) t * H + c], b = ref[(size_t) t * H + c];
                    finite &= std::isfinite(a);
                    ma = std::max(ma, std::fabs(a - b));
                    mr = std::max(mr, std::fabs(a - b) / std::max(rmax, 1e-20));
                }
            }
            worst_abs = std::max(worst_abs, ma);
            worst_rel = std::max(worst_rel, mr);
            const bool bad = !finite || mr > 2e-2;
            std::printf("  error   T=%d (%2d VRAM groups, %2d peer, %2d CPU): max abs %.3e, max rel %.3e%s\n", T,
                        P.counts[0], P.counts[2], P.counts[3], ma, mr, bad ? "  FAIL" : "");
            if (bad) ++g_fail;
        }
        // ---- T-independence
        std::vector<std::vector<float>> first(POOL);
        size_t ndiff = 0, nrows = 0;
        for (int T = 1; T <= 7; ++T)
            for (int o = 0; o < POOL; ++o) {
                load(T, o);
                new_chain(T, s0);
                ck(cudaStreamSynchronize(s0), "tind");
                const auto got = down(d_bo2, (size_t) T * H);
                for (int c = 0; c < T; ++c) {
                    const int p = (c + o) % POOL;
                    std::vector<float> row(got.begin() + (size_t) c * H, got.begin() + (size_t) (c + 1) * H);
                    if (first[p].empty()) first[p] = row;
                    else ndiff += std::memcmp(first[p].data(), row.data(), H * 4) != 0;
                    ++nrows;
                }
            }
        std::printf("  T-indep T=1..7 x 8 offsets: %s (%zu of %zu rows differ)\n", ndiff ? "DIFFERS" : "bitwise equal",
                    ndiff, nrows);
        if (ndiff) ++g_fail;
        {
            const auto c = down(d_m3ctr, K::kDmM3Ctr);
            size_t nz = 0;
            for (auto v : c) nz += v != 0;
            if (nz) { std::printf("  counters left nonzero: %zu\n", nz); ++g_fail; }
        }

        if (bench) {
            for (int T : {2, 3}) {
                load(T, 0);
                const K::DmPlan P = [&] { route(T, false, s0); ck(cudaStreamSynchronize(s0), "r"); return down(d_plan, 1)[0]; }();
                const double bytes = (double) P.counts[0] * (double) L.bytes + (double) (sgb + sub + sdb);
                {
                    const auto ids = down(d_ids, (size_t) T * KK);
                    for (int i = 1; i < 8; ++i) {
                        std::vector<int32_t> v(ids);
                        for (auto& e : v) e = (e + 64 * i) % (int) NE;
                        up(d_ids_c[i], v);
                    }
                }
                const double t_fork = graph_us(s0, 50, [&](cudaStream_t s, int i) { cur = i; old_chain(T, s, true); });
                const double t_ser = graph_us(s0, 50, [&](cudaStream_t s, int i) { cur = i; old_chain(T, s, false); });
                const double t_new = graph_us(s0, 50, [&](cudaStream_t s, int i) { cur = i; new_chain(T, s); });
                cur = 0;
                std::printf("  bench   T=%d (%2d VRAM groups, %.1f MB): old forked %.1f us (%.0f GB/s), old serial %.1f us, "
                            "moe3 %.1f us (%.0f GB/s)\n", T, P.counts[0], bytes / 1e6, t_fork, bytes / t_fork / 1e3, t_ser,
                            t_new, bytes / t_new / 1e3);
            }
        }
        for (void* p : {(void*) d_blob, (void*) d_sgw, (void*) d_suw, (void*) d_sdw, (void*) d_gi}) cudaFree(p);
        for (int i = 1; i < 8 && bench; ++i)
            for (const void* p : {(const void*) d_ids_c[i], (const void*) sgw_c[i], (const void*) suw_c[i],
                                  (const void*) sdw_c[i]})
                cudaFree((void*) p);
    }
    std::printf("dec_moe3_parity: %s\n", g_fail ? "FAILED" : "ok");
    return g_fail ? 1 : 0;
}
