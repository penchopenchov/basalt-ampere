// src/core/verify.cpp - see include/strata/core/verify.hpp.
#include "strata/core/verify.hpp"
#include "strata/core/remote_expert_opt.hpp"
#if defined(_WIN32)
#include <intrin.h>
#endif

#include "strata/core/native_head.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/peer_experts.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/hc_fused.hpp"
#include "strata/kernels/cvec.hpp"
#include "strata/kernels/dec_gdn.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/pdl.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_dec_attn.hpp"
#include "strata/kernels/qsa_dec_fused.hpp"
#include "strata/kernels/q5k_head.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/core/progress.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/dec_moe.hpp"
#include "strata/kernels/dec_shx.hpp"

#include <algorithm>
#include <atomic>
#include <map>
#include <string>
#include <thread>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <immintrin.h>

namespace strata::core {

// STRATA_DEC_MOE=1 (dec_moe.hpp): the MoE tail planned on CUDA0 from device residency tables, the peer card served
// from mapped rings by one persistent kernel per window, the CPU pool by the host loop (only its own entries).
struct DecMoe {
    int64_t rings = 0;                       ///< rings per window (one per layer)
    int32_t* res0 = nullptr;                 ///< CUDA0: [n_layers * n_expert] CUDA0 slot or -1 (host_res mirrored)
    int32_t* res1 = nullptr;                 ///< CUDA0: the same for the peer's slots
    std::vector<int32_t> shadow0, shadow1;   ///< what res0 / res1 hold
    strata::kernels::DmPlan* plan = nullptr;
    unsigned int* ctr = nullptr;
    unsigned long long* slot_off = nullptr;  ///< CUDA0's per-slot offsets (null: slot * blob)
    int vram = -1;                           ///< STRATA_DEC_MOE_VRAM: the VRAM share's kernels (dec_moe_vram's kind; -1 grouped)
    void* vscr = nullptr;                    ///< its gate/up rows
    void* xq_pub = nullptr;                  ///< STRATA_DEC_PK & 4: the side stream's q8_1 rows for the ring
    unsigned int* vsync = nullptr;           ///< its counters
    // mapped: per ring header, FFN input and q8_1 rows; [0] the peer's ring, [16] its done ring; the peer's rows
    strata::kernels::DmRing* h_ring = nullptr; strata::kernels::DmRing* m_ring = nullptr;
    float* h_x = nullptr; float* m_x = nullptr;
    uint8_t* h_xq = nullptr; uint8_t* m_xq = nullptr;
    uint32_t* h_w = nullptr; uint32_t* m_w = nullptr;
    float* h_rows = nullptr; float* m_rows = nullptr;
    size_t x_stride = 0, xq_stride = 0;      ///< per ring, floats / bytes
    // the peer card
    int pdev = -1;
    cudaStream_t ps = nullptr;
    unsigned long long* ptab = nullptr;      ///< [n_layers * n_expert] blob address or 0
    strata::kernels::NativeExpertLayout* lay = nullptr;
    void* pscr = nullptr;
    unsigned int* parr = nullptr;
    int pblocks = 0;
    bool peer_live = false;                  ///< a peer window was launched and not yet waited for
};

namespace {

constexpr float EPS = 1e-6f;
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
const bool g_dbg = std::getenv("STRATA_VERIFY_DEBUG") != nullptr;
#define VDBG(...) do { if (g_dbg) { std::fprintf(stderr, "verify dbg: " __VA_ARGS__); std::fflush(stderr); } } while (0)

struct Bump {
    uint8_t* base = nullptr;
    uint64_t used = 0;
    template <typename T> T* take(uint64_t n) {
        T* p = base ? (T*) (base + used) : nullptr;
        used += (n * sizeof(T) + 255) & ~255ull;
        return p;
    }
};

// #649 (HIP, opt-in A/Bs for the gfx1030 verify timeouts; CUDA never reads them)
//   STRATA_VERIFY_COHERENT=1  the handshake words and rows in explicitly coherent (fine-grained) host memory
//   STRATA_DOORBELL_STORE=1   the GPU stores each step's ring instead of read-modify-writing it over PCIe
bool env_on(const char* name) {
    const char* e = std::getenv(name);
    return e != nullptr && e[0] != 0 && e[0] != '0';
}
#if defined(STRATA_USE_HIP)
const bool g_coherent = env_on("STRATA_VERIFY_COHERENT");
const bool g_doorbell_store = env_on("STRATA_DOORBELL_STORE");
#endif
const bool g_trace = env_on("STRATA_VERIFY_TRACE");
// STRATA_DEC_PK (bit mask), the same values: 1 = in windows of 2+ tokens, the shared expert's side kernels at the
// greatest stream priority (their blocks are scheduled ahead of the routed experts' queued blocks, so the branch ends
// beside them, not after); 2 = in windows of 1-2 tokens, a GDN layer's z projection forked at that priority before
// the qkv projection (both stream together), joined before the one-launch recurrence with the gated norm
// (STRATA_DEC_GDNF=1's kernel), no separate norm launch.  Wider windows run slower with either.  4 = with 1, the MoE
// route split: CUDA0's plan and q8_1 rows on the main stream (the routed experts follow it at once), the ring for the
// peer card and the CPU pool (mapped rows, system fences, doorbells) on a stream of its own at that priority.
int dec_pk_mask() {
    static const int m = [] { const char* e = std::getenv("STRATA_DEC_PK"); return e ? std::atoi(e) : 0; }();
    return m;
}
bool dec_pk() { return dec_pk_mask() != 0; }
// STRATA_DEC_COMMIT: 1 = the commit graph in two launches; 2 = its GDN part from STRATA_DEC_COMMIT_CTAS persistent
// blocks (default 32) so it trickles beside the draft instead of filling the card
int dec_commit_mode() {
    static const int m = [] { const char* e = std::getenv("STRATA_DEC_COMMIT"); return e ? std::atoi(e) : 0; }();
    return m;
}
int dec_commit_blocks() {
    static const int b = [] {
        const char* e = std::getenv("STRATA_DEC_COMMIT_CTAS");
        return e ? std::max(0, std::atoi(e)) : (dec_commit_mode() == 2 ? 32 : 0);
    }();
    return b;
}

bool mapped(size_t bytes, void** h, void** d) {
#if defined(STRATA_USE_HIP)
    if (g_coherent) {
        if (hipHostMalloc(h, bytes, hipHostMallocMapped | hipHostMallocCoherent |
                                        (peer_portable() ? hipHostMallocPortable : 0)) != hipSuccess)
            return false;
        std::memset(*h, 0, bytes);
        return cudaHostGetDevicePointer(d, *h, 0) == cudaSuccess;
    }
#endif
    if (cudaHostAlloc(h, bytes, cudaHostAllocMapped | (peer_portable() ? cudaHostAllocPortable : 0)) != cudaSuccess) return false;   // multi-GPU: portable only with a peer - the peer card writes its rows into them
    std::memset(*h, 0, bytes);
    return cudaHostGetDevicePointer(d, *h, 0) == cudaSuccess;
}

strata::kernels::QsaShapes shapes_of(const ModelGeometry& g) {
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head;
    s.n_head_kv = g.n_head_kv;
    s.head_dim = g.head_dim;
    s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    return s;
}

const WeightRef* need(const LayerView& v, const char* suffix, std::string& err) {
    const WeightRef* r = v.get(suffix);
    if (r == nullptr && err.empty()) err = v.name(suffix) + " is missing";
    return r;
}

bool native_of(const WeightRef* w, const std::string& name, std::string& err) {
    if (w == nullptr) return false;
    if (w->native_data == nullptr) {
        err = "verify: " + name + " is not served natively (run with --native)";
        return false;
    }
    return true;
}

}  // namespace

namespace {
std::atomic<const Verifier*> g_diag_verifier{nullptr};
void diag_active_verifier(std::FILE* f) {
    if (const Verifier* v = g_diag_verifier.load()) v->diag(f);
}
// #267: every live verifier (a layer split has one per stage), for the release before the engine ends
constexpr int kLiveMax = 16;
std::atomic<Verifier*> g_live[kLiveMax];
void release_live_verifiers(std::FILE* f) {
    for (auto& slot : g_live)
        if (Verifier* v = slot.load()) {
            const Clock::time_point t0 = Clock::now();
            const bool done = v->release_gpu_waits(5000);
            if (f != nullptr)
                std::fprintf(f, "strata: released the verify window's GPU waits (#267): the GPU %s\n",
                             done ? ("finished in " + std::to_string((long long) ms_since(t0)) + " ms").c_str()
                                  : "did not finish within 5 s");
        }
    if (f != nullptr) std::fflush(f);
}
// #649: the host side of STRATA_VERIFY_TRACE - a ring of the last kTraceN handshake events of every verifier
struct TraceEv {
    int64_t t_ns;
    const void* who;
    int64_t window, step, layer, aux;
    uint32_t seq, flag, a, b;
    const char* what;
};
constexpr uint64_t kTraceN = 4096;
TraceEv g_trace_ring[kTraceN];
std::atomic<uint64_t> g_trace_next{0};
int64_t trace_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}
std::string released_note(bool drained) {
    return drained ? "; its GPU waits were released and the GPU finished (#267)"
                   : "; its GPU waits were released but the GPU did not finish within 5 s (#267)";
}
// #267 test hook: STRATA_TEST_VERIFY_STALL=N withholds the last layer's flag in the N-th window (1-based), so the
// GPU spins on a flag nobody raises - the bounded window wait and the release are then what ends it.  Unset: never.
const int64_t g_test_stall = [] {
    const char* e = std::getenv("STRATA_TEST_VERIFY_STALL");
    return e != nullptr ? (int64_t) std::atoll(e) : (int64_t) 0;
}();
}  // namespace

bool Verifier::release_gpu_waits(int timeout_ms) {
    released_.store(true);
    trace_ev("RELEASE", -1, -1, timeout_ms);
    // the words the spin kernels read (wait_flag_ge, wait_flag_ge_or) are mapped host memory, so a store here
    // reaches them with no API call; UINT32_MAX is past every ring.  (E-6's skip words are device memory, but
    // wait_flag_ge_or also returns on its flag.)  A host function raising flag B later only raises.
    for (uint32_t* p : {h_flag_, h_flagA_, h_flagB_})
        if (p != nullptr) *(volatile uint32_t*) p = UINT32_MAX;
    if (dm_ != nullptr) {   // the peer window stops at its next ring; CUDA0's waits on it pass
        *(volatile uint32_t*) (dm_->h_w + 0) = UINT32_MAX;
        *(volatile uint32_t*) (dm_->h_w + 16) = UINT32_MAX;
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    _mm_sfence();
    const OnDevice on_device(device_);
    const Clock::time_point t0 = Clock::now();
    for (cudaStream_t s : {cs_, copy_}) {
        if (s == nullptr) continue;
        while (cudaStreamQuery(s) == cudaErrorNotReady) {
            if (ms_since(t0) > timeout_ms) {
                trace_ev("RELEASE-NOT-DRAINED", -1, -1, (int64_t) ms_since(t0));
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    trace_ev("RELEASE-DRAINED", -1, -1, (int64_t) ms_since(t0));   // aux: ms the GPU took to finish once released
    return true;
}

void Verifier::trace_ev(const char* what, int64_t step, int64_t layer, int64_t aux) const {
    if (!g_trace) return;
    auto rd = [](const uint32_t* p) { return p ? *(const volatile uint32_t*) p : 0u; };
    TraceEv& e = g_trace_ring[g_trace_next.fetch_add(1) % kTraceN];
    e = {trace_now_ns(), this, windows, step, layer, aux, rd(h_seq_), rd(h_flag_), rd(h_flagA_), rd(h_flagB_), what};
}

void Verifier::trace_dump(std::FILE* f) const {
    if (!g_trace || f == nullptr) return;
    static const char* names[kProfPer] = {"pre", "hc-read0", "qkv gemv", "conv", "ab", "z", "rec", "kv-idx",
                                          "k/v rope", "kv append", "q", "scores+topk", "kv-resolve", "attention",
                                          "gate", "", "out-proj", "router+ring", "shared", "waitA", "VRAM hits", "waitB",
                                          "PCIe grp", "waitCPU", "combine", "", "head", "", "", "", "", "", ""};
    const uint64_t end = g_trace_next.load();
    const uint64_t n = end < 160 ? end : 160;
    std::fprintf(f, "strata verify trace (#649): the last %llu handshake events (host clock, us before the last; "
                    "words: seq = GPU rang, flag = CPU served, A = plan, B = copies)\n", (unsigned long long) n);
    const int64_t t_last = n ? g_trace_ring[(end - 1) % kTraceN].t_ns : 0;
    for (uint64_t i = end - n; i < end; ++i) {
        const TraceEv& e = g_trace_ring[i % kTraceN];
        if (e.what == nullptr) continue;
        std::fprintf(f, "  %10.1f %-20s v%p window %lld step %lld layer %lld aux %lld | seq %u flag %u A %u B %u\n",
                     (double) (t_last - e.t_ns) / 1000.0, e.what, e.who, (long long) e.window, (long long) e.step,
                     (long long) e.layer, (long long) e.aux, e.seq, e.flag, e.a, e.b);
    }
    if (trace_h_ == nullptr || g_ == nullptr) return;
    // the breadcrumbs: every stamp point the GPU passed in this window (ns of the GPU clock after the window's first)
    unsigned long long t0 = ~0ull;
    for (size_t i = 0; i < trace_n_; ++i) {
        const unsigned long long t = *(const volatile unsigned long long*) (trace_h_ + i);
        if (t != 0 && t < t0) t0 = t;
    }
    if (t0 == ~0ull) {
        std::fprintf(f, "strata verify trace: no GPU breadcrumb in this window (the GPU never started it)\n");
        return;
    }
    int64_t last_l = -1;
    for (int64_t l = 0; l <= g_->n_layers; ++l)
        for (int i = 0; i < kProfPer; ++i)
            for (int grp = 0; grp < 2; ++grp)
                if (trace_h_[(size_t) ((l * kProfPer + i) * 2 + grp)] != 0) last_l = l;
    std::fprintf(f, "strata verify trace: GPU breadcrumbs (us after the window's first), the GPU got as far as layer "
                    "%lld:\n", (long long) last_l);
    for (int64_t l = std::max<int64_t>(0, last_l - 2); l <= std::min<int64_t>(g_->n_layers, last_l + 1); ++l)
        for (int grp = 0; grp < 2; ++grp) {
            std::string line;
            char b[64];
            for (int i = 0; i < kProfPer; ++i) {
                const unsigned long long t = trace_h_[(size_t) ((l * kProfPer + i) * 2 + grp)];
                if (t == 0) continue;
                std::snprintf(b, sizeof b, " %s(%d)=%.1f", names[i], i, (double) (t - t0) / 1000.0);
                line += b;
            }
            if (!line.empty())
                std::fprintf(f, "  layer %lld group %d:%s\n", (long long) l, grp, line.c_str());
        }
    std::fflush(f);
}

std::string Verifier::dm_note() const {
    const DecMoe* d = dm_;
    if (d == nullptr) return "";
    auto rd = [](const uint32_t* p) { return *(const volatile uint32_t*) p; };
    const uint32_t seq = rd(h_seq_);
    std::string s = "; device plan: ring " + std::to_string(seq) + " published";
    if (seq >= 1 && (int64_t) seq <= d->rings) {
        const strata::kernels::DmRing& r = d->h_ring[seq - 1];
        s += " (entries: VRAM " + std::to_string(r.n_vram) + ", peer " + std::to_string(r.n_peer) + ", CPU " +
             std::to_string(r.n_cpu) + ")";
    }
    s += ", host flag " + std::to_string(rd(h_flag_));
    if (d->pdev >= 0) {
        const uint32_t w = rd(d->h_w + 0), done = rd(d->h_w + 16);
        std::string st = "not launched";
        if (d->peer_live) {
            const OnDevice on(d->pdev);
            const cudaError_t q = cudaStreamQuery(d->ps);
            st = q == cudaErrorNotReady ? "running" : q == cudaSuccess ? "finished" : cudaGetErrorString(q);
        }
        s += "; peer card: rung for ring " + std::to_string(w & 0xffffu) + ((w >> 28) & 1u ? " (with entries)" : "") +
             ", finished ring " + std::to_string(done) + ", its kernel " + st;
    }
    return s;
}

// After release_gpu_waits: whether the peer window ended once its doorbell was raised (it waited for CUDA0) or
// still runs (it is stuck inside a ring).
std::string Verifier::peer_released() {
    DecMoe* d = dm_;
    if (d == nullptr || d->pdev < 0 || !d->peer_live) return "";
    const OnDevice on(d->pdev);
    const Clock::time_point t0 = Clock::now();
    cudaError_t q;
    while ((q = cudaStreamQuery(d->ps)) == cudaErrorNotReady && ms_since(t0) < 2000)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (q == cudaErrorNotReady) return "; the peer kernel still runs after the release";
    d->peer_live = false;
    return q == cudaSuccess ? "; the peer kernel finished once released" : std::string("; the peer kernel: ") + cudaGetErrorString(q);
}

void Verifier::diag(std::FILE* f) const {
    auto rd = [](const uint32_t* p) { return p ? *(const volatile uint32_t*) p : 0u; };
    // #251: outside a verify stage these are the LAST window's numbers (it finished), not the stalled work's
    const char* where = progress().where.load();
    const bool current = where != nullptr && std::strncmp(where, "verify window", 13) == 0;
    std::fprintf(f, "  verify window%s: %d tokens at position %lld, host at layer step %u; the GPU rang %u; flags: "
                    "served %u, plan (A) %u, copies (B) %u\n", current ? "" : " (last window, not the current stage)",
                 last_t_, (long long) last_pos0_, cur_layer_ + 1, rd(h_seq_), rd(h_flag_), rd(h_flagA_), rd(h_flagB_));
    if (dm_ != nullptr) std::fprintf(f, "  %s\n", dm_note().c_str() + 2);
    trace_dump(f);   // #649: STRATA_VERIFY_TRACE=1 only
}

bool Verifier::dm_init(std::string& err) {
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    const auto& lay = strata::kernels::cpu::expert_layout();
    const char* why = nullptr;
    if (!lay.native) why = "the experts are not a native pack";
    else if (lb_ != 0 || le_ != g.n_layers || hand_in_ != nullptr) why = "a layer split";
    else if (remote_opt_ != nullptr) why = "--remote-expert-opt";
    else if (all_resident_) why = "every expert is resident";
    else if (hits_.h_res == nullptr) why = "no host residency table";
    else if ((int64_t) max_t_ * ss_->k > kDmCap) why = "the window is too large";
    else if (!native_moe_combine_enabled()) why = "the native combine is off";
    for (int64_t l = 0; why == nullptr && l < g.n_layers; ++l) {
        const auto& f = lay.fmt[(size_t) l];
        if (!dec_moe_formats_ok(f.gu_type, f.d_type, f.n_embd, f.n_ff)) why = "an expert format the kernels do not cover";
    }
    if (why != nullptr) {
        std::fprintf(stderr, "strata verify: STRATA_DEC_MOE is off: %s\n", why);
        return true;
    }
    DecMoe* d = new DecMoe;
    dm_ = d;
    const int64_t LE = g.n_layers * g.n_expert, N = g.n_embd, T = max_t_, K = ss_->k;
    d->rings = le_ - lb_;
    d->x_stride = (size_t) (T * N);
    d->xq_stride = (size_t) (T * (N / 32) * 36);
    bool ok = cudaMalloc((void**) &d->res0, (size_t) LE * 4) == cudaSuccess &&
              cudaMalloc((void**) &d->res1, (size_t) LE * 4) == cudaSuccess &&
              cudaMalloc((void**) &d->plan, sizeof(DmPlan)) == cudaSuccess &&
              cudaMalloc((void**) &d->ctr, 64) == cudaSuccess && cudaMemset(d->ctr, 0, 64) == cudaSuccess &&
              cudaMemset(d->res1, 0xff, (size_t) LE * 4) == cudaSuccess &&
              mapped((size_t) d->rings * sizeof(DmRing), (void**) &d->h_ring, (void**) &d->m_ring) &&
              mapped((size_t) d->rings * d->x_stride * 4, (void**) &d->h_x, (void**) &d->m_x) &&
              mapped((size_t) d->rings * d->xq_stride, (void**) &d->h_xq, (void**) &d->m_xq) &&
              mapped(256, (void**) &d->h_w, (void**) &d->m_w) &&
              mapped((size_t) (T * K * N) * 4, (void**) &d->h_rows, (void**) &d->m_rows) &&
              cudaMalloc(&d->vscr, dec_moe_vram_scratch_bytes((int) lay.fmt[0].n_ff)) == cudaSuccess &&
              cudaMalloc((void**) &d->vsync, (kDmCap + 1) * 4) == cudaSuccess &&
              cudaMemset(d->vsync, 0, (kDmCap + 1) * 4) == cudaSuccess &&
              cudaMalloc(&d->xq_pub, d->xq_stride) == cudaSuccess;
    {
        const char* v = std::getenv("STRATA_DEC_MOE_VRAM");
        d->vram = v != nullptr ? std::atoi(v) : -1;
    }
    if (ok && hits_.slot_off != nullptr && hits_.n_slots > 0)
        ok = cudaMalloc((void**) &d->slot_off, (size_t) hits_.n_slots * 8) == cudaSuccess &&
             cudaMemcpy(d->slot_off, hits_.slot_off, (size_t) hits_.n_slots * 8, cudaMemcpyHostToDevice) == cudaSuccess;
    if (!ok) { err = "verify: STRATA_DEC_MOE buffers"; return false; }
    d->shadow0.assign((size_t) LE, INT32_MIN);
    d->shadow1.assign((size_t) LE, -1);
    if (peer_ != nullptr && peer_->valid()) {
        const OnDevice on(peer_->device());
        d->pdev = peer_->device();
        std::vector<NativeExpertLayout> ls((size_t) g.n_layers);
        for (int64_t l = 0; l < g.n_layers; ++l) {
            const auto& f = lay.fmt[(size_t) l];
            ls[(size_t) l] = native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
        }
        d->pblocks = dec_moe_peer_blocks();
        ok = d->pblocks > 0 && cudaStreamCreateWithFlags(&d->ps, cudaStreamNonBlocking) == cudaSuccess &&
             cudaMalloc((void**) &d->ptab, (size_t) LE * 8) == cudaSuccess &&
             cudaMemset(d->ptab, 0, (size_t) LE * 8) == cudaSuccess &&
             cudaMalloc((void**) &d->lay, ls.size() * sizeof(NativeExpertLayout)) == cudaSuccess &&
             cudaMemcpy(d->lay, ls.data(), ls.size() * sizeof(NativeExpertLayout), cudaMemcpyHostToDevice) == cudaSuccess &&
             cudaMalloc(&d->pscr, dec_moe_peer_scratch_bytes((int) N, (int) lay.fmt[0].n_ff)) == cudaSuccess &&
             cudaMemset(d->pscr, 0, dec_moe_peer_scratch_bytes((int) N, (int) lay.fmt[0].n_ff)) == cudaSuccess &&
             cudaMalloc((void**) &d->parr, 64) == cudaSuccess;
        if (!ok) { err = "verify: STRATA_DEC_MOE peer buffers"; return false; }
        d->shadow1.assign((size_t) LE, INT32_MIN);
    }
    std::fprintf(stderr, "strata verify: STRATA_DEC_MOE on: the MoE tail planned on the device%s\n",
                 d->pdev >= 0 ? (", the peer card served by a persistent kernel (" + std::to_string(d->pblocks) +
                                 " blocks)").c_str() : "");
    return true;
}

// The residency tables the window classifies with: host_res and the peer's, as they are now (between windows).
bool Verifier::dm_sync(std::string& err) {
    DecMoe* d = dm_;
    const size_t LE = d->shadow0.size();
    auto span = [&](const int32_t* now, std::vector<int32_t>& sh, size_t& a, size_t& b) {
        a = LE; b = 0;
        if (std::memcmp(now, sh.data(), LE * 4) == 0) return false;
        for (size_t i = 0; i < LE; ++i)
            if (now[i] != sh[i]) { if (a == LE) a = i; b = i + 1; }
        if (a < b) std::memcpy(sh.data() + a, now + a, (b - a) * 4);
        return a < b;
    };
    size_t a = 0, b = 0;
    if (span(hits_.h_res, d->shadow0, a, b) &&
        cudaMemcpy(d->res0 + a, d->shadow0.data() + a, (b - a) * 4, cudaMemcpyHostToDevice) != cudaSuccess) {
        err = "verify: STRATA_DEC_MOE residency upload";
        return false;
    }
    if (d->pdev >= 0 && span(peer_->residency().data(), d->shadow1, a, b)) {
        if (cudaMemcpy(d->res1 + a, d->shadow1.data() + a, (b - a) * 4, cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "verify: STRATA_DEC_MOE peer residency upload";
            return false;
        }
        std::vector<unsigned long long> p(b - a);
        for (size_t i = a; i < b; ++i)
            p[i - a] = d->shadow1[i] >= 0 ? (unsigned long long) peer_->slot_address(d->shadow1[i]) : 0ull;
        const OnDevice on(d->pdev);
        if (cudaMemcpy(d->ptab + a, p.data(), p.size() * 8, cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "verify: STRATA_DEC_MOE peer table upload";
            return false;
        }
    }
    return true;
}

void Verifier::dm_free() {
    DecMoe* d = dm_;
    if (d == nullptr) return;
    if (d->pdev >= 0) {
        const OnDevice on(d->pdev);
        if (d->ps) { cudaStreamSynchronize(d->ps); cudaStreamDestroy(d->ps); }
        for (void* p : {(void*) d->ptab, (void*) d->lay, d->pscr, (void*) d->parr})
            if (p) cudaFree(p);
    }
    for (void* p : {(void*) d->res0, (void*) d->res1, (void*) d->plan, (void*) d->ctr, (void*) d->slot_off, d->vscr,
                    (void*) d->vsync, d->xq_pub})
        if (p) cudaFree(p);
    for (void* h : {(void*) d->h_ring, (void*) d->h_x, (void*) d->h_xq, (void*) d->h_w, (void*) d->h_rows})
        if (h) cudaFreeHost(h);
    delete d;
    dm_ = nullptr;
}

Verifier::~Verifier() {
    const Verifier* self = this;
    g_diag_verifier.compare_exchange_strong(self, nullptr);
    for (auto& slot : g_live) {
        Verifier* me = this;
        slot.compare_exchange_strong(me, nullptr);
    }
    if (cs_) cudaStreamSynchronize(cs_);
    if (sh_cs_) cudaStreamSynchronize(sh_cs_);
    if (sh_hi_) cudaStreamSynchronize(sh_hi_);
    if (ring_hi_) cudaStreamSynchronize(ring_hi_);
    dm_free();
    for (auto& e : exec_)
        if (e) cudaGraphExecDestroy(e);
    if (commit_exec_) cudaGraphExecDestroy(commit_exec_);
    for (auto& kv : exec_bm_)
        if (kv.second) cudaGraphExecDestroy(kv.second);
    for (auto& kv : commit_bm_)
        if (kv.second) cudaGraphExecDestroy(kv.second);
    if (arena_b_) cudaFree(arena_b_);
    if (h_commitb_) cudaFreeHost(h_commitb_);
    if (cs_) cudaStreamDestroy(cs_);
    if (sh_cs_) cudaStreamDestroy(sh_cs_);
    if (sh_hi_) cudaStreamDestroy(sh_hi_);
    if (ring_hi_) cudaStreamDestroy(ring_hi_);
    if (ev_ring_) cudaEventDestroy(ev_ring_);
    if (ev_ring_join_) cudaEventDestroy(ev_ring_join_);
    if (shx_sync_) cudaFree(shx_sync_);
    if (copy_) { cudaStreamSynchronize(copy_); cudaStreamDestroy(copy_); }
    if (commit_done_) cudaEventDestroy(commit_done_);
    if (ev_fork_) cudaEventDestroy(ev_fork_);
    if (ev_join_) cudaEventDestroy(ev_join_);
    if (arena_) cudaFree(arena_);
    void* hosts[] = {h_tok_, h_step_, h_pos_, h_commit_, h_ple_, h_out_, h_x_, h_ids_, h_w_, h_seq_, h_flag_, h_ymiss_,
                     h_flagA_, h_plan_, h_flagB_};
    for (void* h : hosts)
        if (h) cudaFreeHost(h);
}

bool Verifier::init(const WeightTable& wt, const ModelGeometry& g, SessionState& ss, const VerifyHits& hits,
                    const NativeHead* head, int max_t, std::string& err) {
    g_diag_verifier.store(this);
    diag_verify_fn().store(&diag_active_verifier);
    for (auto& slot : g_live) {
        Verifier* none = nullptr;
        if (slot.load() == this || slot.compare_exchange_strong(none, this)) break;
    }
    release_gpu_fn().store(&release_live_verifiers);
    cudaGetDevice(&device_);   // a layer split's stage on another GPU: its streams, graphs and buffers live there
    strata::kernels::fused_gr_check();   // once per card: which bitwise-equal hyper-connection read runs there
    if (const char* e = std::getenv("STRATA_DEC_HC"); e && e[0] == '1') strata::kernels::hc_fused_init();
    wt_ = &wt;
    g_ = &g;
    ss_ = &ss;
    hits_ = hits;
    head_ = head;
    max_t_ = max_t;
    sampling_.greedy = true;      // a fresh verifier samples greedily until set_sampling says otherwise
    sampling_.temperature = 0.0f;
    if (max_t < 2 || max_t > strata::kernels::kVerifyMaxT || max_t > strata::kernels::cpu::MAXT) {
        err = "verify: the window must hold 2.." + std::to_string(strata::kernels::kVerifyMaxT) + " tokens";
        return false;
    }
    if (hits.d_res == nullptr || hits.cache_base == nullptr || hits.blob <= 0) {
        err = "verify: needs the profile-filled VRAM expert tier (--expert-profile and --expert-cache); with "
              "--expert-cache auto, no VRAM was left for it - the 'no VRAM is left for the expert cache' line above "
              "says how much is short and what makes room (a smaller --max-context, --kv q4_0, setup --draft-vocab en, "
              "images on the CPU)";
        return false;
    }
    std::string why;
    if (!layer_verify_compatible(why)) {
        err = "verify: " + why + " (the verify window reproduces the default native decode path)";
        return false;
    }
    if (!strata::kernels::fused_gr_supported(g.n_embd, g.hc, g.hc_lr) || ss.k != 10 || g.ssm_state_size != 128 ||
        g.ssm_d_conv != 4) {
        err = "verify: geometry differs from the artifact's";
        return false;
    }
    if (le_ < 0) le_ = g.n_layers;
    if (lb_ < 0 || lb_ >= le_ || le_ > g.n_layers || (lb_ > 0 && hand_in_ == nullptr) ||
        (le_ < g.n_layers && hand_out_ == nullptr)) {
        err = "verify: the stage's layer range or its hand-off buffers are wrong";
        return false;
    }
    const WeightRef* wo = wt.find("output.weight");
    if (wo == nullptr) { err = "verify: output.weight is missing"; return false; }
    n_vocab_ = wo->ne1;

    const strata::kernels::QsaShapes s = shapes_of(g);
    cap_ = strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, s);
    max_blocks_ = ss.qsa_states[ss.qsa_primary()].max_cells / s.idx_block + 2;
    attn_scratch_floats_ = (int64_t) strata::kernels::qsa_decode_attn_scratch_floats(cap_, s);

    const uint64_t T = (uint64_t) max_t, N = (uint64_t) g.n_embd, HC = (uint64_t) g.hc, K = (uint64_t) ss.k;
    const uint64_t C = (uint64_t) g.ssm_conv_channels, ZV = (uint64_t) g.ssm_value_dim, HV = (uint64_t) g.ssm_v_heads;
    const uint64_t NH = (uint64_t) g.n_head, HD = (uint64_t) g.head_dim, NKV = (uint64_t) g.n_head_kv;
    const uint64_t IQ = (uint64_t) g.idx_q_heads, ID = (uint64_t) g.idx_key_dim;
    const uint64_t nG = (uint64_t) g.n_gdn_layers(), nQ = (uint64_t) g.n_qsa_layers();
    const uint64_t HS = (uint64_t) strata::kernels::NG_HIST * strata::kernels::NG_HC_DIM;
    const uint64_t TS = (uint64_t) (s.idx_block - 1) * ID;
    const int max_in = (int) std::max<uint64_t>(std::max<uint64_t>(N, ZV), NH * HD);

    // ---- mapped staging
    bool ok = mapped(T * 4, (void**) &h_tok_, (void**) &m_tok_) &&
              mapped(T * strata::kernels::kStepCount * 4, (void**) &h_step_, (void**) &m_step_) &&
              mapped(T * (NH + NKV + IQ) * 4, (void**) &h_pos_, (void**) &m_pos_) &&
              mapped((2 + T) * 4 + 16, (void**) &h_commit_, (void**) &m_commit_) &&
              mapped(T * N * 4, (void**) &h_ple_, (void**) &m_ple_) &&
              mapped(T * 4 + 16, (void**) &h_out_, (void**) &m_out_) &&
              mapped(T * N * 4, (void**) &h_x_, (void**) &m_x_) &&
              mapped(T * K * 4, (void**) &h_ids_, (void**) &m_ids_) &&
              mapped(T * K * 4, (void**) &h_w_, (void**) &m_w_) &&
              mapped(64, (void**) &h_seq_, (void**) &m_seq_) &&
              mapped(64, (void**) &h_flag_, (void**) &m_flag_) &&
              mapped(64, (void**) &h_flagA_, (void**) &m_flagA_) &&
              mapped(64, (void**) &h_flagB_, (void**) &m_flagB_) &&
              mapped(T * K * N * 4, (void**) &h_ymiss_, (void**) &m_ymiss_);
    if (!ok) { err = "verify: mapped staging allocation failed"; return false; }
    // the GPU plan: counts(4) | start(cap+1) | dst(cap) | tok(cap) | pad | ptr(cap u64) | ptr2(cap u64) | start2(cap+1)
    {
        const int64_t cap = (int64_t) (T * K);
        const int64_t i32 = 4 + (cap + 1) + cap + cap;
        const int64_t ptr_off = (i32 + 1) & ~1ll;
        plan_i32_ = ptr_off + 4 * cap + (cap + 1) + 1;
        if (!mapped((size_t) plan_i32_ * 4 * 2 + 64, (void**) &h_plan_, (void**) &m_plan_)) {
            err = "verify: mapped plan allocation failed";
            return false;
        }
        sink_.counts = h_plan_;
        sink_.start = h_plan_ + 4;
        sink_.dst = sink_.start + cap + 1;
        sink_.tok = sink_.dst + cap;
        sink_.ptr = (unsigned long long*) (h_plan_ + ptr_off);
        sink_.ptr2 = sink_.ptr + cap;
        sink_.start2 = h_plan_ + ptr_off + 4 * cap;
        sink_.cap = cap;
        sink_.publish = &Verifier::publish_plan;
        sink_.fetch = &Verifier::fetch_dma;
        sink_.ctx = this;
    }

    // ---- the device arena: the same sequence counted, then carved
    auto carve = [&](Bump& b) {
        tok_ = b.take<int32_t>(T); step_ = b.take<int32_t>(T * strata::kernels::kStepCount);
        pos_ = b.take<int32_t>(T * (NH + NKV + IQ)); commit_ = b.take<int32_t>(2 + T);
        ple_ = b.take<float>(T * N); emb_ = b.take<float>(T * N); R_ = b.take<float>(T * HC * N);
        mixed_ = b.take<float>(T * N); bo_ = b.take<float>(T * N);
        inj_ = b.take<float>(T * HC); inj2_ = b.take<float>(T * HC);
        lo_ = b.take<float>(T * (uint64_t) g.hc_lr); rs_ = b.take<float>(T * HC); xn_ = b.take<float>(T * HC * N);
        xq_ = b.take<uint8_t>(strata::kernels::native_q8_1_bytes(max_in, (int) T));
        sh_xq_ = b.take<uint8_t>(strata::kernels::native_q8_1_bytes(max_in, (int) T));
        qkv_L_ = b.take<float>(nG * T * C); h_L_ = b.take<float>(nG * T * C);
        gate_L_ = b.take<float>(nG * T * HV); beta_L_ = b.take<float>(nG * T * HV);
        z_ = b.take<float>(T * ZV); y_ = b.take<float>(T * ZV); y_dummy_ = b.take<float>(T * ZV);
        qfull_ = b.take<float>(T * NH * 2 * HD); qcur_ = b.take<float>(T * NH * HD);
        kcur_ = b.take<float>(T * NKV * HD); vcur_ = b.take<float>(T * NKV * HD);
        idx_raw_L_ = b.take<float>(nQ * T * ID); qidx_ = b.take<float>(T * IQ * ID);
        scores_ = b.take<float>(T * (uint64_t) max_blocks_); sel_ = b.take<int32_t>(T * (uint64_t) cap_);
        attn_ = b.take<float>(T * NH * HD); attn32_ = b.take<float>(T * NH * HD);
        attn_scratch_ = b.take<float>(T * (uint64_t) attn_scratch_floats_);
        tail_snap_ = b.take<float>(nQ * TS);
        logits_ = b.take<float>(T * (uint64_t) g.n_expert); w_ = b.take<float>(T * K); ids_ = b.take<int32_t>(T * K);
        shared_ = b.take<float>(T * N); parts_ = b.take<float>(T * K * N); hit_out_ = b.take<float>(T * K * N);
        hit_slot_ = b.take<int32_t>(T * K); hit_dst_ = b.take<int32_t>(T * K); hit_count_ = b.take<int32_t>(4);
        plan_ = b.take<int32_t>(2 * ((uint64_t) plan_i32_ + 16));
        staging_ = b.take<uint8_t>((uint64_t) kStagingBlobs * strata::kernels::cpu::expert_layout().max_blob);
        hit_xq_ = b.take<uint8_t>(T * (N / 32) * 34); hit_xs_ = b.take<float>(T * (N / 32));
        nat_xq_ = b.take<uint8_t>(T * (N / 32) * 36);
        hit_scratch_ = b.take<uint8_t>(std::max<uint64_t>(
            strata::kernels::moe_hit_grouped_scratch_bytes((int64_t) (T * K), g.n_embd, g.n_ff),
            strata::kernels::native_expert_scratch_bytes((int64_t) (T * K), g.n_ff)));
        head_mixed_ = b.take<float>(T * N); head_inj_ = b.take<float>(HC);
        sh_bf16_ = b.take<uint16_t>(T * N); sh_gate_ = b.take<float>(T * (uint64_t) g.n_ff);
        sh_up_ = b.take<float>(T * (uint64_t) g.n_ff); sh_g_ = b.take<float>(T + 4);
        head_logits_ = b.take<float>(T * (uint64_t) n_vocab_);
        hist_snap_ = b.take<float>(T * HS);
    };
    Bump count;
    carve(count);
    if (cudaMalloc(&arena_, count.used) != cudaSuccess) {
        err = "verify: the device arena (" + std::to_string(count.used >> 20) + " MiB) does not fit";
        return false;
    }
    cudaMemset(arena_, 0, count.used);
    if (g_trace && trace_h_ == nullptr) {   // #649: the breadcrumbs, mapped so they read while the GPU hangs
        trace_n_ = (size_t) (g.n_layers + 1) * kProfPer * 2;
        if (!mapped(trace_n_ * 8, (void**) &trace_h_, (void**) &trace_m_)) {
            (void) cudaGetLastError();
            trace_h_ = trace_m_ = nullptr;
            trace_n_ = 0;
        }
#if defined(STRATA_USE_HIP)
        std::fprintf(stderr, "strata verify trace (#649): on; coherent words %s, doorbell %s, HIP_HOST_COHERENT=%s "
                             "HSA_ENABLE_SDMA=%s GPU_MAX_HW_QUEUES=%s\n", g_coherent ? "explicit" : "default",
                     g_doorbell_store ? "stored" : "incremented", std::getenv("HIP_HOST_COHERENT") ? std::getenv("HIP_HOST_COHERENT") : "-",
                     std::getenv("HSA_ENABLE_SDMA") ? std::getenv("HSA_ENABLE_SDMA") : "-",
                     std::getenv("GPU_MAX_HW_QUEUES") ? std::getenv("GPU_MAX_HW_QUEUES") : "-");
#else
        std::fprintf(stderr, "strata verify trace (#649): on\n");
#endif
    }
    prof_on_ = std::getenv("STRATA_VERIFY_PROFILE") != nullptr;
    if (prof_on_) decode_gaps().host_on = true;
    if (prof_on_) {
        const size_t np = (size_t) g.n_layers * kProfPer + 8;
        if (cudaMalloc((void**) &prof_, np * 8) != cudaSuccess) { prof_on_ = false; prof_ = nullptr; cudaGetLastError(); }
        else { cudaMemset(prof_, 0, np * 8); prof_h_.assign(np, 0); }
    }
    Bump real;
    real.base = (uint8_t*) arena_;
    carve(real);
    sink_.staging = (unsigned long long) staging_;
    sink_.staging_cap = kStagingBlobs;
    (void) TS;
    if (cudaStreamCreateWithFlags(&copy_, cudaStreamNonBlocking) != cudaSuccess) {
        err = "verify: copy stream create failed";
        return false;
    }
    if (cudaStreamCreateWithFlags(&cs_, cudaStreamNonBlocking) != cudaSuccess ||
        cudaStreamCreateWithFlags(&sh_cs_, cudaStreamNonBlocking) != cudaSuccess) {
        err = "verify: stream create failed";
        return false;
    }
    if (dec_pk()) {
        int least = 0, greatest = 0;
        if (cudaDeviceGetStreamPriorityRange(&least, &greatest) != cudaSuccess ||
            cudaStreamCreateWithPriority(&sh_hi_, cudaStreamNonBlocking, greatest) != cudaSuccess ||
            ((dec_pk_mask() & 4) &&
             (cudaStreamCreateWithPriority(&ring_hi_, cudaStreamNonBlocking, greatest) != cudaSuccess ||
              cudaEventCreateWithFlags(&ev_ring_, cudaEventDisableTiming) != cudaSuccess ||
              cudaEventCreateWithFlags(&ev_ring_join_, cudaEventDisableTiming) != cudaSuccess))) {
            err = "verify: STRATA_DEC_PK stream create failed";
            return false;
        }
        std::fprintf(stderr, "strata verify: STRATA_DEC_PK=%d: side branches at stream priority %d\n", dec_pk_mask(), greatest);
    }
    if (cudaEventCreateWithFlags(&commit_done_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&ev_fork_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&ev_join_, cudaEventDisableTiming) != cudaSuccess) {
        err = "verify: event create failed";
        return false;
    }
    // Check whether 100% of experts across [lb_, le_) are resident in this stage's VRAM cache.
    // When true, every layer plans on device and writes directly into parts_ without any CPU doorbells,
    // wait_flag_ge spins, PCIe empty launches, or moe_hit_add copies.
    all_resident_ = false;
    if (hits.h_res != nullptr && hits.d_res != nullptr && hits.cache_base != nullptr) {
        const char* v_ar = std::getenv("STRATA_VERIFY_ALL_RESIDENT");
        if (v_ar == nullptr || std::atoi(v_ar) != 0) {
            bool all_ok = true;
            for (int64_t l = lb_; l < le_ && all_ok; ++l) {
                for (int64_t e = 0; e < g.n_expert; ++e) {
                    if (hits.h_res[l * g.n_expert + e] < 0) {
                        all_ok = false;
                        break;
                    }
                }
            }
            all_resident_ = all_ok;
        }
    }
    // E-6: a layer whose routed experts are all resident is planned on the device (STRATA_VERIFY_DEVICE_PLAN=1: on;
    // exact, but neutral on RIBPC 1-2 GPUs: off by default)
    {
        const char* v = std::getenv("STRATA_VERIFY_DEVICE_PLAN");
        device_plan_ = !all_resident_ && (v != nullptr && std::atoi(v) != 0);
    }
    if (all_resident_ || device_plan_) {
        bool ok2 = true;
        if (device_plan_)
            ok2 = cudaMalloc((void**) &skip_, 64) == cudaSuccess && cudaMemset(skip_, 0, 64) == cudaSuccess;
        if (ok2 && hits.slot_off != nullptr && hits.n_slots > 0) {
            ok2 = cudaMalloc((void**) &slot_off_d_, (size_t) hits.n_slots * sizeof(unsigned long long)) == cudaSuccess &&
                  cudaMemcpy(slot_off_d_, hits.slot_off, (size_t) hits.n_slots * sizeof(unsigned long long),
                             cudaMemcpyHostToDevice) == cudaSuccess;
        }
        if (!ok2) { cudaGetLastError(); all_resident_ = false; device_plan_ = false; }
    }
    if (env_on("STRATA_DEC_MOE") && !dm_init(err)) return false;
    if (env_on("STRATA_DEC_SHX") && dm_ != nullptr && shx_sync_ == nullptr) {   // the shared expert in one launch
        if (cudaMalloc(&shx_sync_, strata::kernels::kShxSyncWords * sizeof(unsigned int)) != cudaSuccess ||
            cudaMemset(shx_sync_, 0, strata::kernels::kShxSyncWords * sizeof(unsigned int)) != cudaSuccess) {
            err = "verify: STRATA_DEC_SHX buffers";
            return false;
        }
        std::fprintf(stderr, "strata verify: STRATA_DEC_SHX on: the shared expert in one launch\n");
    }
    std::fprintf(stderr, "strata verify: window up to %d tokens, %.1f MiB of device buffers%s\n", max_t,
                 (double) count.used / 1048576.0,
                 all_resident_ ? " (100% VRAM resident: zero-doorbell graph)" : "");
    return true;
}

const float* Verifier::final_R(int t) const { return R_ + (size_t) t * (size_t) (g_->hc * g_->n_embd); }

// ================================ THE WINDOW, AS CAPTURED ================================
//
// Plan v0.3 P6 (split window): with `groups_ == 2` the window's tokens are cut into two groups A = [0, T/2 up) and
// B = the rest, and the stream is ordered
//
//     pre(0,A) pre(0,B) | post(0,A) pre(1,A) | post(0,B) pre(1,B) | post(1,A) pre(2,A) | ...
//
// so the CPU computes A's experts of layer l while the GPU runs B's mixer and router of layer l, and B's experts
// while the GPU combines A and runs A's layer l+1.  B's mixer only needs A's mixer of the same layer (K/V, GDN
// state), never A's experts, so nothing waits that did not wait before.  Every token's arithmetic is unchanged.
bool Verifier::record_window(int T, cudaStream_t cs, std::string& err) {
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    const WeightTable& wt = *wt_;
    SessionState& ss = *ss_;
    const int64_t N = g.n_embd, HC = g.hc, K = ss.k, C = g.ssm_conv_channels, ZV = g.ssm_value_dim;
    const int64_t HV = g.ssm_v_heads, HK = g.ssm_k_heads, NH = g.n_head, HD = g.head_dim, NKV = g.n_head_kv;
    const int64_t IQ = g.idx_q_heads, ID = g.idx_key_dim, NE = g.n_expert, MT = max_t_;
    const QsaShapes s = shapes_of(g);
    const GrShapes gs{g.n_embd, g.hc, g.hc_lr};
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    const int64_t HS = (int64_t) NG_HIST * NG_HC_DIM;
    const int64_t TS = (s.idx_block - 1) * ID;
    const bool ple_on = ss.ple.ready() && ple_stage();
    auto Rt = [&](int t) { return R_ + (size_t) t * HC * N; };
    const bool dm = dm_ != nullptr && !batch_rec_;   // STRATA_DEC_MOE: the device-planned tail (one group)
    const int G = (split_ && T >= 2 && !batch_rec_ && !dm) ? 2 : 1;   // a batch window is one group
    const bool shx = dm && shx_sync_ != nullptr;   // STRATA_DEC_SHX: the shared expert in one launch, no side stream
    static const bool dec_batch = [] { const char* v = std::getenv("STRATA_DEC_BATCH"); return v == nullptr || std::atoi(v) != 0; }();
    static const bool dec_qsa = [] { const char* v = std::getenv("STRATA_DEC_QSA"); return v != nullptr && std::atoi(v) != 0; }();
    // STRATA_DEC_HC=1: each hyper-connection read (and the FFN read's router) as one kernel, bitwise the kernels it
    // replaces; the next read's w_norm is prefetched into L2 at its end
    static const bool dec_hc_env = [] { const char* e = std::getenv("STRATA_DEC_HC"); return e && e[0] == '1'; }();
    const bool dec_hc = dec_hc_env && hc_fused_supported();
    // STRATA_DEC_HC_PF=1: each read also prefetches the next read's weight matrices into L2 at its end
    static const bool dec_hc_pf = [] { const char* e = std::getenv("STRATA_DEC_HC_PF"); return e && e[0] == '1'; }();
    // STRATA_PDL=1 (pdl.hpp): the dense chain's kernels may start during the previous kernel's tail.  Only where
    // that previous kernel is one of them or another plain kernel; never with the profile's or trace's stamps.
    const bool pdl = pdl_rec_ && !prof_on_ && trace_m_ == nullptr && !batch_rec_;
    // STRATA_DEC_PDL2=1 (pdl.hpp): the chain's kernels let their dependents in early, which stage their weights
    const bool pdl2 = pdl && pdl2_requested();
    auto stamp = [&](int64_t l, int i, int grp) {
        if (prof_on_ && grp == 0) gpu_stamp(prof_, (int) (l * kProfPer + i), cs);
        if (trace_m_ != nullptr) gpu_stamp(trace_m_, (int) ((l * kProfPer + i) * 2 + grp), cs);   // #649
    };
    const int tb_[2] = {0, (T + 1) / 2}, te_[2] = {G == 2 ? (T + 1) / 2 : T, T};
    if (!batch_rec_) groups_[T] = G;
    const int64_t nQall = g.n_qsa_layers();
    // a batch window: row t is slot t, whose state lives in its own session
    auto slot_ss = [&](int t) -> SessionState& { return batch_rec_ ? *slots_[(size_t) brow_[t]] : ss; };
    const int hrow0 = batch_rec_ ? row_base_ : 0;   // a slot group's own hand-off rows

    if (prof_on_ && !batch_rec_) gpu_stamp(prof_, (int) (g.n_layers * kProfPer + 2), cs);
    // ---- the window's inputs, from mapped staging
    copy_i32_from_mapped(tok_, m_tok_, T, cs);
    copy_i32_from_mapped(step_, m_step_, (int64_t) T * kStepCount, cs);
    copy_i32_from_mapped(pos_, m_pos_, (int64_t) MT * (NH + NKV + IQ), cs);
    // per-ROW positions of the K rows [t][NKV] and the indexer query rows [t][IQ] (for batched RoPE)
    const int32_t* pos_k = pos_ + MT * NH;
    const int32_t* pos_i = pos_ + MT * (NH + NKV);

    // ---- the embeddings, broadcast to the hc streams - or, in a later stage of a layer split, the previous stage's
    // residual, pending write and inject (see set_stage)
    const int64_t HB = Verifier::handoff_floats(g);
    if (lb_ > 0) {
        const float* hin = hand_in_ + (size_t) hrow0 * HB;   // the group's rows: [R][bo][inj] contiguous
        copy_from_mapped(R_, hin, (int64_t) T * HC * N, cs);
        copy_from_mapped(bo_, hin + (size_t) T * HC * N, (int64_t) T * N, cs);
        copy_from_mapped(inj2_, hin + (size_t) T * (HC + 1) * N, (int64_t) T * HC, cs);
    } else if (const NativeEmbed* ne = native_embed()) {       // plan v0.3 P6: the GGUF-form table
        ne->gather_dev(tok_, T, emb_, cs);
        broadcast_streams(emb_, R_, N, (int) HC, T, cs);
    } else {
        const WeightRef* w = wt.find("token_embd.weight");
        if (w == nullptr || w->codebook_iq4nl || (w->code_bits != 2 && w->code_bits != 4 && w->code_bits != 8)) {
            err = "verify: token_embd.weight is missing or not an S2/S4/S8 tensor";
            return false;
        }
        const auto* codes = (const uint8_t*) w->data;
        const auto* scales = (const float*) (codes + w->codes_bytes);
        const auto* offsets = w->has_offset ? (const float*) (codes + w->codes_bytes + w->scales_bytes) : nullptr;
        const uint64_t row_codes = (uint64_t) (w->ne0 / (8 / w->code_bits));
        const uint64_t row_groups = (uint64_t) (w->ne0 / w->group_elems);
        embedding_gather_dev(codes, scales, offsets, tok_, T, w->ne0, w->code_bits, w->code_bias, w->group_elems,
                             row_codes, row_groups, emb_, cs);
        broadcast_streams(emb_, R_, N, (int) HC, T, cs);
    }

    // per-layer state indices (GDN and QSA layers are numbered separately)
    std::vector<int64_t> gdn_idx((size_t) g.n_layers, -1), qsa_idx((size_t) g.n_layers, -1);
    {
        int64_t qi = 0, gi = 0;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            if (is_qsa_layer(g, l)) qsa_idx[(size_t) l] = qi++;
            else gdn_idx[(size_t) l] = gi++;
        }
    }

    // ---------------------------------------------------------------- pre(l, group): up to the ring
    bool ring_fork[2] = {false, false};   // STRATA_DEC_PK & 4: the group's ring stream is joined in post
    auto pre = [&](int64_t l, int grp) -> bool {
        const int tb = tb_[grp], te = te_[grp], n = te - tb;
        stamp(l, 0, grp);
        const LayerView v(wt, l);
        const char* pfx[2] = {"hc_attn_", "hc_ffn_"};
        const WeightRef *wn[2], *wd[2], *wu[2], *wi[2];
        for (int h = 0; h < 2; ++h) {
            wn[h] = need(v, (std::string(pfx[h]) + "norm.weight").c_str(), err);
            wd[h] = need(v, (std::string(pfx[h]) + "down.weight").c_str(), err);
            wu[h] = need(v, (std::string(pfx[h]) + "up.weight").c_str(), err);
            wi[h] = need(v, (std::string(pfx[h]) + "inject.weight").c_str(), err);
            if (!wn[h] || !wd[h] || !wu[h] || !wi[h]) return false;
        }
        // the previous layer's FFN write, folded into this layer's first read (a control vector after it has
        // already applied it)
        bool pending = l > 0 && !cvec().covers(l - 1);
        if (l == 1 && ple_on) {
            if (grp == 0) {
                if (all_resident_ || dm) wait_flag_ge(m_flag_, 1, cs);
                copy_from_mapped(ple_, m_ple_, (int64_t) T * N, cs);
            }
            float* normalized = (float*) ((uint8_t*) ss.ple.scratch + ple_block_scratch_bytes());
            static const bool ple_batch_env = [] {
                const char* e = std::getenv("STRATA_PLE_BATCH");
                return !e || e[0] != '0';
            }();
            const bool ple_batch_kv = ple_batch_env && dec_batch && n > 1 && ss.ple.w.key_bf16 != nullptr &&
                                      ss.ple.w.value_bf16 != nullptr && ple_native_bf16_enabled() &&
                                      ple_native_postops_enabled();
            if (ple_batch_kv) {
                try {
                    bf16_gemv_fp32_mmvf_multi(ple_ + (size_t) tb * N, N, ss.ple.w.key_bf16,
                                              xn_ + (size_t) tb * HC * N, HC * N, N, HC * N, n, cs);
                    bf16_gemv_fp32_mmvf_multi(ple_ + (size_t) tb * N, N, ss.ple.w.value_bf16,
                                              z_ + (size_t) tb * N, N, N, N, n, cs);
                } catch (const std::exception& e) {
                    err = std::string("verify PLE multi: ") + e.what();
                    return false;
                }
            }
            for (int t = tb; t < te; ++t) {
                gr_write(Rt(t), bo_ + t * N, inj2_ + t * HC, gs, Rt(t), cs);
                PleOut po;
                po.normalized = normalized;
                po.result = Rt(t);
                float* hist = batch_rec_ ? slot_ss(t).ple_hist : ss.ple.hist;   // the row's own history
                try {
                    if (ple_batch_kv) {
                        ple_block_projected(xn_ + (size_t) t * HC * N, z_ + (size_t) t * N, Rt(t), hist,
                                            ss.ple.w, po, ss.ple.scratch, cs);
                    } else {
                        ple_block(ple_ + t * N, Rt(t), hist, ss.ple.w, po, ss.ple.scratch, cs);
                    }
                    ple_history_advance(hist, normalized, cs);
                } catch (const std::exception& e) {
                    err = std::string("verify PLE: ") + e.what();
                    return false;
                }
                copy_from_mapped(hist_snap_ + (size_t) t * HS, hist, HS, cs);
            }
            pending = false;
        }
        auto gr_read_group = [&](int half, bool apply, float* inj_prev, float* inj_out) {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = tb; t < te; ++t) {
                FusedGrArgs& a = fa[t - tb];
                a.R = Rt(t); a.R_out = Rt(t); a.apply = apply;
                a.bo_prev = bo_ + t * N; a.inj_prev = inj_prev + t * HC;
                a.w_norm = (const float*) wn[half]->data; a.w_down = (const uint16_t*) wd[half]->data;
                a.w_up = (const uint16_t*) wu[half]->data; a.w_inject = (const uint16_t*) wi[half]->data;
                a.eps = EPS; a.lo = lo_ + t * g.hc_lr; a.rs = rs_ + t * HC;
                a.inject_out = inj_out + t * HC; a.mixed = mixed_ + t * N;
            }
            fused_gr_read_multi(fa, n, xn_ + (size_t) tb * HC * N, cs, (prof_on_ && grp == 0) ? prof_ : nullptr,
                                (int) (l * kProfPer + (half == 0 ? 27 : 30)));
        };
        auto hc_args = [&](int half, bool apply, float* inj_prev, float* inj_out) {
            HcFusedArgs h;
            h.T = n; h.R = Rt(tb); h.R_out = Rt(tb); h.apply = apply;
            h.bo_prev = bo_ + tb * N; h.inj_prev = inj_prev + tb * HC;
            h.w_norm = (const float*) wn[half]->data; h.w_down = (const uint16_t*) wd[half]->data;
            h.w_up = (const uint16_t*) wu[half]->data; h.w_inject = (const uint16_t*) wi[half]->data;
            h.eps = EPS; h.lo = lo_ + tb * g.hc_lr; h.rs = rs_ + tb * HC; h.inject_out = inj_out + tb * HC;
            h.mixed = mixed_ + tb * N; h.xn = xn_ + (size_t) tb * HC * N;
            return h;
        };
        {
            // the read's first kernel follows a combine, copy or wait: a plain launch, except after the device-planned
            // tail's combine kernel (PDL2)
            const bool after_kernel = (pdl2_mask() & 32) && pdl && dm && dec_hc && l > lb_ && !(l == 1 && ple_on);
            PdlScope ps(pdl, !after_kernel);
            if (dec_hc) {
                HcFusedArgs h = hc_args(0, pending, inj2_, inj_);
                h.pf[0] = wn[1]->data; h.pf_bytes[0] = (size_t) HC * N * sizeof(float);
                if (dec_hc_pf) {   // the FFN read's weights and router, read after the mixer
                    const size_t mat = (size_t) g.hc_lr * HC * N * 2;
                    h.pf[1] = wd[1]->data; h.pf_bytes[1] = mat;
                    h.pf[2] = wu[1]->data; h.pf_bytes[2] = mat;
                    const WeightRef* wr = v.get("ffn_gate_inp.weight");
                    if (wr != nullptr) { h.pf[3] = wr->data; h.pf_bytes[3] = (size_t) NE * N * 2; }
                }
                hc_fused_read(h, cs);
            } else {
                gr_read_group(0, pending, inj2_, inj_);
            }
        }
        stamp(l, 1, grp);
        float* xm = mixed_ + tb * N;
        try {
            if (!is_qsa_layer(g, l)) {
                // ======================= GDN =======================
                const WeightRef *wqkv = need(v, "attn_qkv.weight", err), *wg = need(v, "attn_gate.weight", err),
                                *wout = need(v, "ssm_out.weight", err), *wa = need(v, "ssm_alpha.weight", err),
                                *wb = need(v, "ssm_beta.weight", err), *wc = need(v, "ssm_conv1d.weight", err),
                                *wnm = need(v, "ssm_norm.weight", err), *wdt = need(v, "ssm_dt.bias", err),
                                *wsa = need(v, "ssm_a", err);
                if (!wqkv || !wg || !wout || !wa || !wb || !wc || !wnm || !wdt || !wsa) return false;
                if (!native_of(wqkv, v.name("attn_qkv.weight"), err) || !native_of(wg, v.name("attn_gate.weight"), err) ||
                    !native_of(wout, v.name("ssm_out.weight"), err))
                    return false;
                const int64_t gi = gdn_idx[(size_t) l];
                float* state = ss.gdn_state + (size_t) (gi - ss.gdn_ord0) * gdn_floats;
                float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                float* qkv = qkv_L_ + (size_t) gi * MT * C;
                float* hb = h_L_ + (size_t) gi * MT * C;
                float* gate = gate_L_ + (size_t) gi * MT * HV;
                float* beta = beta_L_ + (size_t) gi * MT * HV;
                PdlScope ps(pdl);   // the layer's GDN kernels, after the hc read
                // STRATA_DEC_GDN=1: the same values in three launches (dec_gdn.hpp)
                if (dec_gdn_enabled() && !batch_rec_ && tb == 0 && N == 2560 && dec_dense_supported(wqkv->native_type, (int) N) &&
                    dec_dense_supported(wg->native_type, (int) N) && dec_dense_supported(wout->native_type, (int) ZV)) {
                    const DecMat in[2] = {{wqkv->native_type, wqkv->native_data, (int) C, qkv},
                                          {wg->native_type, wg->native_data, (int) ZV, z_}};
                    dec_gdn_inproj(xm, (int) N, n, in, 2, xq_, cs);
                    for (int i = 2; i <= 5; ++i) stamp(l, i, grp);
                    DecRecArgs ra;
                    ra.state = state; ra.qkv = qkv; ra.conv_w = (const float*) wc->data; ra.x = xm;
                    ra.w_alpha = (const uint16_t*) wa->data; ra.w_beta = (const uint16_t*) wb->data;
                    ra.dt = (const float*) wdt->data; ra.ssm_a = (const float*) wsa->data; ra.z = z_;
                    ra.gamma = (const float*) wnm->data; ra.h = hb; ra.gate = gate; ra.beta = beta; ra.yq = xq_;
                    ra.C = (int) C; ra.n_embd = (int) N; ra.h_k = (int) HK; ra.h_v = (int) HV; ra.eps = EPS;
                    dec_gdn_rec(ra, n, cs);
                    stamp(l, 6, grp);
                    dec_dense(xq_, (int) ZV, n, DecMat{wout->native_type, wout->native_data, (int) N, bo_ + tb * N}, cs);
                    goto gdn_done;
                }
                // STRATA_DEC_GDNF: conv, gates, recurrence, gated norm and the output's q8_1 in dec_gdn_rec's one or
                // two launches; at 2/3 the z projection runs on the side stream beside the recurrence (dec_gdn.hpp)
                {
                const int gdnf = (!batch_rec_ && tb == 0 && N == 2560 && n <= kDecMaxT) ? dec_gdnf_mode() : 0;
                const bool gdnf_fork = gdnf >= 2 && sh_cs_ != nullptr && ev_fork_ != nullptr && ev_join_ != nullptr;
                auto gdnf_z_side = [&] {
                    cudaEventRecord(ev_fork_, cs);
                    cudaStreamWaitEvent(sh_cs_, ev_fork_, 0);
                    PdlScope pz(pdl, true);
                    native_mmvq(wg->native_type, wg->native_data, xq_, z_, (int) N, (int) ZV, n, sh_cs_);
                    cudaEventRecord(ev_join_, sh_cs_);
                };
                native_quantize_q8_1(xm, xq_, (int) N, n, cs);
                const bool pk_z = gdnf_fork && sh_hi_ != nullptr && (dec_pk_mask() & 2) && n <= 2;
                if (pk_z) {   // STRATA_DEC_PK & 2: z at the greatest priority beside qkv, then the one-launch recurrence
                    cudaEventRecord(ev_fork_, cs);
                    cudaStreamWaitEvent(sh_hi_, ev_fork_, 0);
                    {
                        PdlScope pz(pdl, true);
                        native_mmvq(wg->native_type, wg->native_data, xq_, z_, (int) N, (int) ZV, n, sh_hi_);
                    }
                    cudaEventRecord(ev_join_, sh_hi_);
                    native_mmvq(wqkv->native_type, wqkv->native_data, xq_, qkv + (size_t) tb * C, (int) N, (int) C, n, cs);
                    stamp(l, 2, grp);
                    DecRecArgs ra;
                    ra.state = state; ra.qkv = qkv; ra.conv_w = (const float*) wc->data; ra.x = xm;
                    ra.w_alpha = (const uint16_t*) wa->data; ra.w_beta = (const uint16_t*) wb->data;
                    ra.dt = (const float*) wdt->data; ra.ssm_a = (const float*) wsa->data; ra.z = z_;
                    ra.gamma = (const float*) wnm->data; ra.h = hb; ra.gate = gate; ra.beta = beta; ra.yq = xq_;
                    ra.C = (int) C; ra.n_embd = (int) N; ra.h_k = (int) HK; ra.h_v = (int) HV; ra.eps = EPS;
                    cudaStreamWaitEvent(cs, ev_join_, 0);
                    for (int i = 3; i <= 5; ++i) stamp(l, i, grp);
                    {
                        PdlScope pr(pdl, true);   // after the join
                        dec_gdn_rec(ra, n, cs);
                    }
                    stamp(l, 6, grp);
                    native_mmvq(wout->native_type, wout->native_data, xq_, bo_ + tb * N, (int) ZV, (int) N, n, cs);
                    goto gdn_done;
                }
                if (gdnf_fork && gdnf == 3) gdnf_z_side();
                native_mmvq(wqkv->native_type, wqkv->native_data, xq_, qkv + (size_t) tb * C, (int) N, (int) C, n, cs);
                stamp(l, 2, grp);
                if (gdnf) {
                    DecRecArgs ra;
                    ra.state = state; ra.qkv = qkv; ra.conv_w = (const float*) wc->data; ra.x = xm;
                    ra.w_alpha = (const uint16_t*) wa->data; ra.w_beta = (const uint16_t*) wb->data;
                    ra.dt = (const float*) wdt->data; ra.ssm_a = (const float*) wsa->data; ra.z = z_;
                    ra.gamma = (const float*) wnm->data; ra.h = hb; ra.gate = gate; ra.beta = beta; ra.yq = xq_;
                    ra.C = (int) C; ra.n_embd = (int) N; ra.h_k = (int) HK; ra.h_v = (int) HV; ra.eps = EPS;
                    ra.early_trigger = pdl && (pdl2_mask() & 64);   // PDL2: the out projection stages its weights
                    if (gdnf_fork) {
                        if (gdnf == 2) gdnf_z_side();
                        ra.o = y_;
                        dec_gdn_rec_split(ra, n, cs);
                        for (int i = 3; i <= 5; ++i) stamp(l, i, grp);
                        cudaStreamWaitEvent(cs, ev_join_, 0);
                        PdlScope pn(pdl, true);   // the norm follows the join
                        dec_gdn_rec_norm(ra, n, cs);
                        stamp(l, 6, grp);
                        native_mmvq(wout->native_type, wout->native_data, xq_, bo_ + tb * N, (int) ZV, (int) N, n, cs);
                    } else {
                        native_mmvq(wg->native_type, wg->native_data, xq_, z_, (int) N, (int) ZV, n, cs);
                        for (int i = 3; i <= 5; ++i) stamp(l, i, grp);
                        dec_gdn_rec(ra, n, cs);
                        stamp(l, 6, grp);
                        native_mmvq(wout->native_type, wout->native_data, xq_, bo_ + tb * N, (int) ZV, (int) N, n, cs);
                    }
                    goto gdn_done;
                }
                }
                if (batch_rec_) {   // each row from its own slot's conv history, one row each
                    for (int t = tb; t < te; ++t) {
                        SessionState& sx = slot_ss(t);
                        const float* cx = sx.gdn_state + (size_t) (gi - sx.gdn_ord0) * gdn_floats +
                                          (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                        gdn_conv_l2_multi(cx, qkv + (size_t) t * C, (const float*) wc->data, hb + (size_t) t * C, (int) C,
                                          (int) (2 * HK), EPS, 1, cs, 0);
                    }
                } else
                gdn_conv_l2_multi(conv, qkv, (const float*) wc->data, hb, (int) C, (int) (2 * HK), EPS, n, cs, tb);
                stamp(l, 3, grp);
                gdn_ab_multi(xm, (const uint16_t*) wa->data, (const uint16_t*) wb->data, (const float*) wdt->data,
                             (const float*) wsa->data, gate + (size_t) tb * HV, beta + (size_t) tb * HV, (int) N, (int) HV,
                             n, cs);
                stamp(l, 4, grp);
                native_mmvq(wg->native_type, wg->native_data, xq_, z_ + (size_t) tb * ZV, (int) N, (int) ZV, n, cs);
                stamp(l, 5, grp);
                // the recurrence from the untouched state over tokens [0, te); outputs only for this group's
                if (batch_rec_) {   // each row's recurrence from its own slot's state, one token
                    for (int t = tb; t < te; ++t) {
                        SessionState& sx = slot_ss(t);
                        float* stx = sx.gdn_state + (size_t) (gi - sx.gdn_ord0) * gdn_floats;
                        gdn_step_norm_multi(stx, hb + (size_t) t * C, (int) C, gate + (size_t) t * HV, beta + (size_t) t * HV,
                                            z_ + (size_t) t * ZV, (const float*) wnm->data, EPS, y_ + (size_t) t * ZV,
                                            (int) HK, (int) HV, 1, nullptr, cs, 0);
                    }
                } else
                gdn_step_norm_multi(state, hb, (int) C, gate, beta, z_, (const float*) wnm->data, EPS, y_, (int) HK,
                                    (int) HV, te, nullptr, cs, tb);
                stamp(l, 6, grp);
                native_quantize_q8_1(y_ + (size_t) tb * ZV, xq_, (int) ZV, n, cs);
                native_mmvq(wout->native_type, wout->native_data, xq_, bo_ + tb * N, (int) ZV, (int) N, n, cs);
            gdn_done:;
            } else {
                // ======================= QSA =======================
                const int64_t qi = qsa_idx[(size_t) l];
                const QsaState& st = ss.qsa_states[qi];
                const WeightRef *wik = need(v, "indexer.k_proj.weight", err), *wq = need(v, "attn_q.weight", err),
                                *wk = need(v, "attn_k.weight", err), *wv = need(v, "attn_v.weight", err),
                                *wo = need(v, "attn_output.weight", err), *wiq = need(v, "indexer.q_proj.weight", err),
                                *wqn = need(v, "attn_q_norm.weight", err), *wkn = need(v, "attn_k_norm.weight", err),
                                *wiqn = need(v, "indexer.q_norm.weight", err), *wikn = need(v, "indexer.k_norm.weight", err);
                if (!wik || !wq || !wk || !wv || !wo || !wiq || !wqn || !wkn || !wiqn || !wikn) return false;
                if (!native_of(wq, v.name("attn_q.weight"), err) || !native_of(wk, v.name("attn_k.weight"), err) ||
                    !native_of(wv, v.name("attn_v.weight"), err) || !native_of(wo, v.name("attn_output.weight"), err))
                    return false;
                auto norm_rope = [&](float* data, const WeightRef* norm, int rows, int cols, const int32_t* pos) {
                    if (native_qsa_enabled()) native_qsa_rms_norm_weighted(data, (const float*) norm->data, data, cols, rows, EPS, cs);
                    else rms_norm_weighted(data, (const float*) norm->data, rows, cols, EPS, cs);
                    if (native_rope_enabled()) native_rope_apply(data, data, rows, cols, (int) s.n_rot, rope_scaling(), pos, cs);
                    else rope_neox_apply(data, data, rows, cols, (int) s.n_rot, st.cos_tab, st.sin_tab, pos, cs);
                };
                float* idx_raw = idx_raw_L_ + (size_t) qi * MT * ID;
                // the per-token GEMVs / norms / RoPEs / copies of this layer as one launch over the
                // window's rows each - row-wise identical arithmetic (STRATA_DEC_BATCH=0: token by token)
                const bool qb = dec_batch && n > 1 && native_qsa_enabled() && native_rope_enabled() && !st.kv_q4;
                {
                    PdlScope ps(pdl);   // after the hc read, up to the K/V norms
                    native_quantize_q8_1(xm, xq_, (int) N, n, cs);
                    if (qb) bf16_gemv_fp32_mmvf_multi(mixed_ + tb * N, N, (const uint16_t*) wik->data, idx_raw + tb * ID, ID, N, ID, n, cs);
                    else for (int t = tb; t < te; ++t)
                        bf16_gemv_fp32_mmvf(mixed_ + t * N, (const uint16_t*) wik->data, idx_raw + t * ID, (int) N, (int) ID, cs);
                    stamp(l, 7, grp);
                    native_mmvq(wk->native_type, wk->native_data, xq_, kcur_ + tb * NKV * HD, (int) N, (int) (NKV * HD), n, cs);
                    native_mmvq(wv->native_type, wv->native_data, xq_, vcur_ + tb * NKV * HD, (int) N, (int) (NKV * HD), n, cs);
                }
                // STRATA_DEC_QSA=1: the K norm + RoPE, the INT8 appends, the tail snapshot and the indexer appends of the
                // group's rows in one launch, the same bits
                const bool fused_pre = dec_qsa && !batch_rec_ && st.kv_int8 && !st.kv_rot && !st.kv_hybrid && !st.kv_q4 &&
                                       native_qsa_enabled() && native_rope_enabled() && qsa_dec_fused_supported(s);
                if (fused_pre) {
                    QsaDecKvPost kp;
                    kp.kcur = kcur_ + tb * NKV * HD; kp.vcur = vcur_ + tb * NKV * HD; kp.pos_k = pos_k + tb * NKV;
                    kp.k_norm = (const float*) wkn->data; kp.steps = step_ + tb * kStepCount;
                    kp.k_q = st.k_q; kp.v_q = st.v_q; kp.k_scale = st.k_scale; kp.v_scale = st.v_scale;
                    kp.page_table = st.page_table; kp.host = st.host;
                    kp.idx_raw = idx_raw + tb * ID; kp.idx_k_norm = (const float*) wikn->data;
                    kp.tail = st.idx_tail; kp.dead = st.idx_dead; kp.pooled = st.idx_pooled; kp.block_pos = st.idx_block_pos;
                    kp.tail_snap = grp == 0 ? tail_snap_ + (size_t) qi * TS : nullptr;
                    kp.max_cells = (int) st.max_cells; kp.eps = EPS;
                    qsa_dec_kv_post(kp, n, s, rope_scaling(), cs);
                    stamp(l, 8, grp);
                } else {
                    if (qb) norm_rope(kcur_ + tb * NKV * HD, wkn, (int) (n * NKV), (int) HD, pos_k + tb * NKV);
                    else for (int t = tb; t < te; ++t) norm_rope(kcur_ + t * NKV * HD, wkn, (int) NKV, (int) HD, pos_ + t * NH);
                    if (st.kv_rot) {   // K and V rotated before they are stored (kv_q4.hpp)
                        fwht256_inplace_cuda(kcur_ + tb * NKV * HD, (int64_t) n * NKV, cs);
                        fwht256_inplace_cuda(vcur_ + tb * NKV * HD, (int64_t) n * NKV, cs);
                    } else if (st.kv_hybrid) {   // K8V4: only V is rotated
                        fwht256_inplace_cuda(vcur_ + tb * NKV * HD, (int64_t) n * NKV, cs);
                    }
                    stamp(l, 8, grp);
                    if (batch_rec_) {   // every slot's indexer tail, restored by its commit
                        for (int t = tb; t < te; ++t)
                            copy_from_mapped(tail_snap_b_ + ((size_t) brow_[t] * nQall + qi) * TS, slot_ss(t).qsa_states[qi].idx_tail,
                                             TS, cs);
                    } else
                    if (grp == 0) copy_from_mapped(tail_snap_ + (size_t) qi * TS, st.idx_tail, TS, cs);
                    for (int t = tb; t < te; ++t) {
                        const QsaState& st = slot_ss(t).qsa_states[qi];   // the row's own K/V (ss's outside a batch)
                        const int32_t* step_t = step_ + t * kStepCount;
                        if (st.kv_hybrid) {   // K8V4: the unused half's lanes folded onto the used pool (layer.cpp)
                            kv_append_q8_step(st.k_q, st.k_q, st.k_scale, st.k_scale, st.page_table, step_t,
                                              kcur_ + t * NKV * HD, kcur_ + t * NKV * HD, s, cs, nullptr);
                            kv_append_q4_step(st.v_q4, st.v_q4, st.page_table, step_t, vcur_ + t * NKV * HD,
                                              vcur_ + t * NKV * HD, s, cs, nullptr);
                        } else if (st.kv_q4)
                            kv_append_q4_step(st.k_q4, st.v_q4, st.page_table, step_t, kcur_ + t * NKV * HD,
                                              vcur_ + t * NKV * HD, s, cs, &st.host);
                        else if (st.kv_int8)
                            kv_append_q8_step(st.k_q, st.v_q, st.k_scale, st.v_scale, st.page_table, step_t,
                                              kcur_ + t * NKV * HD, vcur_ + t * NKV * HD, s, cs, &st.host);
                        else
                            kv_append_step(st.k_pool, st.v_pool, st.page_table, step_t, kcur_ + t * NKV * HD,
                                           vcur_ + t * NKV * HD, s, cs, &st.host);
                    }
                    for (int t = tb; t < te; ++t) {
                        const QsaState& sx = slot_ss(t).qsa_states[qi];
                        const QsaIndexerBuffers ib{sx.idx_tail, sx.idx_dead, sx.idx_pooled, sx.idx_block_pos};
                        native_qsa_indexer_append(idx_raw + t * ID, step_ + t * kStepCount + kStepPos, 0,
                                                  (const float*) wikn->data, EPS, ib, s, sx.max_cells,
                                                  rope_scaling(), cs);
                    }
                }
                stamp(l, 9, grp);
                {
                    PdlScope pq(pdl2 && fused_pre && qb);   // PDL2: after the K/V post kernel
                    native_mmvq(wq->native_type, wq->native_data, xq_, qfull_ + tb * NH * 2 * HD, (int) N,
                                (int) (NH * 2 * HD), n, cs);
                    if (fused_pre && qb)
                        bf16_gemv_fp32_mmvf_multi(mixed_ + tb * N, N, (const uint16_t*) wiq->data, qidx_ + tb * IQ * ID,
                                                  IQ * ID, N, IQ * ID, n, cs);
                }
                if (fused_pre) {   // the query split, norms and RoPEs in one launch
                    if (qb) {}   // the indexer query projection ran above
                    else for (int t = tb; t < te; ++t)
                        bf16_gemv_fp32_mmvf(mixed_ + t * N, (const uint16_t*) wiq->data, qidx_ + t * IQ * ID, (int) N,
                                            (int) (IQ * ID), cs);
                    QsaDecQPost qp;
                    qp.q_full = qfull_ + tb * NH * 2 * HD; qp.qcur = qcur_ + tb * NH * HD; qp.pos_q = pos_ + tb * NH;
                    qp.q_norm = (const float*) wqn->data; qp.q_idx = qidx_ + tb * IQ * ID; qp.pos_i = pos_i + tb * IQ;
                    qp.idx_q_norm = (const float*) wiqn->data; qp.eps = EPS;
                    qsa_dec_q_post(qp, n, s, rope_scaling(), cs);
                } else if (qb) {
                    if (cudaMemcpy2DAsync(qcur_ + tb * NH * HD, (size_t) HD * 4, qfull_ + tb * NH * 2 * HD, (size_t) HD * 2 * 4,
                                          (size_t) HD * 4, (size_t) (n * NH), cudaMemcpyDeviceToDevice, cs) != cudaSuccess) {
                        err = "verify: the q/gate split failed";
                        return false;
                    }
                    norm_rope(qcur_ + tb * NH * HD, wqn, (int) (n * NH), (int) HD, pos_ + tb * NH);
                    if (st.kv_rot) fwht256_inplace_cuda(qcur_ + tb * NH * HD, (int64_t) n * NH, cs);   // <Hq, Hk> = <q, k>
                    bf16_gemv_fp32_mmvf_multi(mixed_ + tb * N, N, (const uint16_t*) wiq->data, qidx_ + tb * IQ * ID, IQ * ID,
                                              N, IQ * ID, n, cs);
                    norm_rope(qidx_ + tb * IQ * ID, wiqn, (int) (n * IQ), (int) ID, pos_i + tb * IQ);
                } else {
                for (int t = tb; t < te; ++t) {
                    float* qc = qcur_ + t * NH * HD;
                    if (cudaMemcpy2DAsync(qc, (size_t) HD * 4, qfull_ + t * NH * 2 * HD, (size_t) HD * 2 * 4,
                                          (size_t) HD * 4, (size_t) NH, cudaMemcpyDeviceToDevice, cs) != cudaSuccess) {
                        err = "verify: the q/gate split failed";
                        return false;
                    }
                    norm_rope(qc, wqn, (int) NH, (int) HD, pos_ + t * NH);
                    if (st.kv_rot) fwht256_inplace_cuda(qc, NH, cs);   // <Hq, Hk> = <q, k>
                }
                for (int t = tb; t < te; ++t) {
                    float* qx = qidx_ + t * IQ * ID;
                    bf16_gemv_fp32_mmvf(mixed_ + t * N, (const uint16_t*) wiq->data, qx, (int) N, (int) (IQ * ID), cs);
                    norm_rope(qx, wiqn, (int) IQ, (int) ID, pos_ + t * NH);
                }
                }
                stamp(l, 10, grp);
                bool fused_gate = false;
                if (batch_rec_) {   // each row selects and attends over its own slot's K/V
                    for (int t = tb; t < te; ++t) {
                        const QsaState& sx = slot_ss(t).qsa_states[qi];
                        qsa_block_scores(sx.idx_pooled, sx.idx_dead, qidx_ + t * IQ * ID, step_ + t * kStepCount, 1,
                                         max_blocks_, s, scores_ + (size_t) t * max_blocks_, cs);
                        qsa_block_topk(scores_ + (size_t) t * max_blocks_, step_ + t * kStepCount, 1, max_blocks_, cap_, s,
                                       sel_ + (size_t) t * cap_, cs);
                        qsa_kv_resolve(sx, *g_, sel_ + (size_t) t * cap_, step_ + t * kStepCount, 1, cap_, cs);
                        const QsaAttnPools px = qsa_attn_pools(sx);
                        qsa_decode_attn_batch(qcur_ + t * NH * HD, px, sel_ + (size_t) t * cap_, step_ + t * kStepCount,
                                              cap_, s, attn_scratch_ + (size_t) t * attn_scratch_floats_,
                                              attn_ + t * NH * HD, 1, cs);
                    }
                } else {
                qsa_block_scores(st.idx_pooled, st.idx_dead, qidx_ + tb * IQ * ID, step_ + tb * kStepCount, n, max_blocks_,
                                 s, scores_ + (size_t) tb * max_blocks_, cs);
                qsa_block_topk(scores_ + (size_t) tb * max_blocks_, step_ + tb * kStepCount, n, max_blocks_, cap_, s,
                               sel_ + (size_t) tb * cap_, cs);
                stamp(l, 11, grp);
                // KV streaming: the n selections' blocks resident (device-side, inside the graph)
                qsa_kv_resolve(st, *g_, sel_ + (size_t) tb * cap_, step_ + tb * kStepCount, n, cap_, cs);
                stamp(l, 12, grp);
                const QsaAttnPools pools = qsa_attn_pools(st);
                // STRATA_DEC_QSA=1: staged chunks and the gate in the merge, the same bits
                fused_gate = dec_qsa && native_qsa_enabled() && !st.kv_rot && !st.kv_hybrid &&
                             qsa_dec_attn_supported(pools, s);
                if (fused_gate)
                    qsa_dec_attn(qcur_ + tb * NH * HD, pools, sel_ + (size_t) tb * cap_, step_ + tb * kStepCount, cap_, s,
                                 attn_scratch_ + (size_t) tb * attn_scratch_floats_, attn_ + tb * NH * HD, n, cs,
                                 qfull_ + tb * NH * 2 * HD, attn32_ + tb * NH * HD);
                else
                qsa_decode_attn_batch(qcur_ + tb * NH * HD, pools, sel_ + (size_t) tb * cap_, step_ + tb * kStepCount, cap_,
                                      s, attn_scratch_ + (size_t) tb * attn_scratch_floats_, attn_ + tb * NH * HD, n, cs);
                }
                stamp(l, 13, grp);
                if (st.kv_rot || st.kv_hybrid) fwht256_inplace_cuda(attn_ + tb * NH * HD, (int64_t) n * NH, cs);   // back: H^-1 = H
                if (fused_gate) {}   // the merge wrote attn32_
                else if (qb) native_qsa_gate_apply(attn_ + tb * NH * HD, qfull_ + tb * NH * 2 * HD, attn32_ + tb * NH * HD,
                                              (int) (n * NH), (int) HD, cs);
                else
                for (int t = tb; t < te; ++t) {
                    if (native_qsa_enabled())
                        native_qsa_gate_apply(attn_ + t * NH * HD, qfull_ + t * NH * 2 * HD, attn32_ + t * NH * HD,
                                              (int) NH, (int) HD, cs);
                    else
                        qsa_gate_apply_f32(attn_ + t * NH * HD, qfull_ + t * NH * 2 * HD, s, attn32_ + t * NH * HD, cs);
                }
                stamp(l, 14, grp);
                PdlScope ps(pdl);   // after the gate kernel
                native_quantize_q8_1(attn32_ + tb * NH * HD, xq_, (int) (NH * HD), n, cs);
                native_mmvq(wo->native_type, wo->native_data, xq_, bo_ + tb * N, (int) (NH * HD), (int) N, n, cs);
            }
        } catch (const std::exception& e) {
            err = "verify layer " + std::to_string(l) + ": " + e.what();
            return false;
        }
        stamp(l, 16, grp);
        const WeightRef* w_router = v.get("ffn_gate_inp.weight");
        const bool router_multi = dec_batch && n > 1 && w_router != nullptr && native_router_enabled() && NE == 512 && K == 10;
        bool routed = false;   // the fused read ran the router too
        {
            PdlScope ps(pdl);   // after the out projection's kernel
            if (dec_hc) {
                HcFusedArgs h = hc_args(1, true, inj_, inj2_);
                if (router_multi) {
                    h.w_router = (const uint16_t*) w_router->data;
                    h.logits = logits_ + tb * NE; h.ids = ids_ + tb * K; h.weights = w_ + tb * K;
                    routed = true;
                }
                const WeightRef *next_norm = nullptr, *next_down = nullptr, *next_up = nullptr;
                if (l + 1 < g.n_layers) {
                    const LayerView vn(wt, l + 1);
                    next_norm = vn.get("hc_attn_norm.weight");
                    next_down = vn.get("hc_attn_down.weight");
                    next_up = vn.get("hc_attn_up.weight");
                } else {
                    next_norm = wt.find("output_hc_norm.weight");
                    next_down = wt.find("output_hc_down.weight");
                    next_up = wt.find("output_hc_up.weight");
                }
                if (next_norm != nullptr) { h.pf[0] = next_norm->data; h.pf_bytes[0] = (size_t) HC * N * sizeof(float); }
                if (dec_hc_pf && next_down != nullptr && next_up != nullptr) {   // the next read's weights, read after the experts
                    const size_t mat = (size_t) g.hc_lr * HC * N * 2;
                    h.pf[1] = next_down->data; h.pf_bytes[1] = mat;
                    h.pf[2] = next_up->data; h.pf_bytes[2] = mat;
                }
                hc_fused_read(h, cs);
            } else {
                gr_read_group(1, true, inj_, inj2_);
            }
        }
        static const bool sh_stream_env = [] {
            const char* e = std::getenv("STRATA_SH_STREAM");
            return !e || e[0] != '0';
        }();
        const bool sh_fork = !shx && sh_stream_env && !prof_on_ && sh_cs_ != nullptr && ev_fork_ != nullptr &&
                             ev_join_ != nullptr;
        cudaStream_t sh_stream = sh_fork ? (sh_hi_ != nullptr && (dec_pk_mask() & 1) && n >= 2 ? sh_hi_ : sh_cs_) : cs;
        // STRATA_DEC_PK & 4: the ring on its own stream, forked after the router, joined where the side branch is
        const bool pk_ring = dm && sh_fork && sh_stream == sh_hi_ && ring_hi_ != nullptr;
        ring_fork[grp] = pk_ring;
        if (sh_fork) {
            cudaEventRecord(ev_fork_, cs);
            cudaStreamWaitEvent(sh_stream, ev_fork_, 0);
        }
        // the window's rows routed in 2 launches (one router GEMV reading the weight once, one
        // top-10) instead of 2 per token; every row's arithmetic is the single-token call's (STRATA_DEC_BATCH=0: old)
        if (routed) {
        } else if (router_multi) {
            try {
                PdlScope ps(pdl);   // after the hc read's last kernel (the fork adds no node on this stream)
                bf16_gemv_fp32_mmvf_multi(mixed_ + tb * N, N, (const uint16_t*) w_router->data, logits_ + tb * NE, NE, N,
                                          NE, n, cs);
                native_router_top10_multi(logits_ + tb * NE, ids_ + tb * K, w_ + tb * K, n, cs);
            } catch (const std::exception& e) { err = "verify router: " + std::string(e.what()); return false; }
#if defined(STRATA_HIP_GFX906)
        } else if (dec_batch) {
            // gfx906: the Coder's 256-expert router (the native router is 512 x 10 only) for the whole window in one
            // multi-column BF16 projection and one top-k, bitwise the per-token calls (route_window_parity); ~3 ms
            // of a ~55 ms verify window on 2x MI50.  STRATA_ROUTE_PER_TOKEN=1: the per-token calls.
            if (!moe_route_window(wt, g, l, K, ss.moe, mixed_ + tb * N, logits_ + tb * NE, ids_ + tb * K, w_ + tb * K, n,
                                  cs, err))
                return false;
#endif
        } else
        for (int t = tb; t < te; ++t) {
            MoEBuffers mb = ss.moe;
            mb.logits = logits_ + t * NE; mb.ids = ids_ + t * K; mb.weights = w_ + t * K;
            if (!moe_route(wt, g, l, K, mb, mixed_ + t * N, cs, err, nullptr)) return false;
        }
        if (dm) {   // the tiers, CUDA0's plan, the q8_1 rows and the rings in one launch
            DmRouteArgs ra;
            ra.ids = ids_ + tb * K; ra.x = xm; ra.n_tok = n; ra.k = (int) K; ra.n_embd = (int) N;
            ra.n_expert = (int) g.n_expert; ra.layer = (int) l;
            ra.res0 = dm_->res0 + l * g.n_expert;
            ra.res1 = dm_->pdev >= 0 ? dm_->res1 + l * g.n_expert : nullptr;
            ra.cache_base = hits_.cache_base; ra.slot_off = dm_->slot_off; ra.blob = (long long) hits_.blob;
            ra.plan = dm_->plan; ra.xq = nat_xq_ + (size_t) tb * (N / 32) * 36;
            const int64_t r = l - lb_;
            ra.ring_m = dm_->m_ring + r; ra.x_m = dm_->m_x + (size_t) r * dm_->x_stride;
            ra.xq_m = dm_->m_xq + (size_t) r * dm_->xq_stride;
            ra.host_seq_m = m_seq_; ra.peer_seq_m = dm_->m_w;
            ra.ring = (uint32_t) (r + 1); ra.ctr = dm_->ctr;
            PdlScope pr(pdl && (pdl2_mask() & 32));   // after the router's kernel
            if (pk_ring) {   // the ring beside, CUDA0's plan and rows here
                cudaEventRecord(ev_ring_, cs);
                cudaStreamWaitEvent(ring_hi_, ev_ring_, 0);
                DmRouteArgs rp = ra;
                rp.mode = 2; rp.xq_pub = dm_->xq_pub;
                {
                    PdlScope pp(false);
                    dec_moe_route(rp, ring_hi_);
                }
                cudaEventRecord(ev_ring_join_, ring_hi_);
                ra.mode = 1;
            }
            dec_moe_route(ra, cs);
        } else if (all_resident_) {
            resident_plan(ids_ + tb * K, n * (int) K, (int) K, hits_.d_res + l * g.n_expert, (int) g.n_expert,
                          hits_.cache_base, slot_off_d_, (long long) hits_.blob,
                          plan_ + (size_t) grp * (size_t) (plan_i32_ + 16), (long long) max_t_ * K, nullptr, 0, cs);
        } else {
            if (device_plan_)   // E-6: every routed expert resident: this group's plan without the host
                resident_plan(ids_ + tb * K, n * (int) K, (int) K, hits_.d_res + l * g.n_expert, (int) g.n_expert,
                              hits_.cache_base, slot_off_d_, (long long) hits_.blob,
                              plan_ + (size_t) grp * (size_t) (plan_i32_ + 16), (long long) max_t_ * K, skip_ + grp,
                              (uint32_t) ((l - lb_) * G + grp + 1), cs);
#if defined(STRATA_USE_HIP)
            if (g_doorbell_store)   // #649 A/B: the step's ring stored, not incremented over PCIe
                doorbell_publish_value(xm, ids_ + tb * K, w_ + tb * K, (int64_t) n * N, (int64_t) n * K, m_x_ + tb * N,
                                       m_ids_ + tb * K, m_w_ + tb * K, m_seq_, (uint32_t) ((l - lb_) * G + grp + 1), cs);
            else
#endif
            if (remote_opt_)   // #578: the helper GPUs reduce with the routing weights - publish them
                doorbell_publish(xm, ids_ + tb * K, w_ + tb * K, (int64_t) n * N, (int64_t) n * K, m_x_ + tb * N,
                                 m_ids_ + tb * K, m_w_ + tb * K, m_seq_, cs);
            else {
                const int32_t* layer_res = hits_.d_res != nullptr ? (hits_.d_res + l * g.n_expert) : nullptr;
                doorbell_publish_res(xm, ids_ + tb * K, layer_res, (int) g.n_expert, (int64_t) n * N, (int64_t) n * K,
                                     m_x_ + tb * N, m_ids_ + tb * K, m_seq_, cs);
            }
        }
        stamp(l, 17, grp);
        {
            const WeightRef *wgi = need(v, "ffn_gate_inp_shexp.weight", err), *wsg = need(v, "ffn_gate_shexp.weight", err),
                            *wsu = need(v, "ffn_up_shexp.weight", err), *wsd = need(v, "ffn_down_shexp.weight", err);
            if (!wgi || !wsg || !wsu || !wsd) return false;
            if (!native_of(wsg, v.name("ffn_gate_shexp.weight"), err) || !native_of(wsu, v.name("ffn_up_shexp.weight"), err) ||
                !native_of(wsd, v.name("ffn_down_shexp.weight"), err))
                return false;
            NativeSharedWeights nsw;
            nsw.gate_type = wsg->native_type; nsw.gate_data = wsg->native_data;
            nsw.up_type = wsu->native_type; nsw.up_data = wsu->native_data;
            nsw.down_type = wsd->native_type; nsw.down_data = wsd->native_data;
            nsw.q8_1 = sh_fork ? sh_xq_ : xq_;
            if (shx && dec_shx_ok(nsw.gate_type, nsw.up_type, nsw.down_type, N, g.n_ff)) {
                ShxArgs sa;   // xm's own q8_1 rows: the route kernel's (q8_1_store) differ from them in rare blocks
                sa.gate_w = nsw.gate_data; sa.up_w = nsw.up_data; sa.down_w = nsw.down_data;
                sa.gate_inp = (const uint16_t*) wgi->data; sa.x = xm; sa.xq = xq_;
                sa.gate = sh_gate_ + (size_t) tb * g.n_ff; sa.up = sh_up_ + (size_t) tb * g.n_ff; sa.hq = sh_xq_;
                sa.g = sh_g_ + tb; sa.out = shared_ + tb * N; sa.sync = shx_sync_; sa.n_tok = n;
                try {
                    PdlScope ps(pdl);   // after the route kernel
                    native_quantize_q8_1(xm, xq_, (int) N, n, cs);
                    dec_shx(sa, cs);
                } catch (const std::exception& e) {
                    err = std::string("verify shared expert: ") + e.what();
                    return false;
                }
            } else {
            if (!shared_expert_native_bf16_enabled()) {
                if (dec_batch) f32_to_bf16_bulk(mixed_ + tb * N, sh_bf16_ + tb * N, (int64_t) n * N, sh_stream);   // contiguous rows
                else for (int t = tb; t < te; ++t) f32_to_bf16_bulk(mixed_ + t * N, sh_bf16_ + t * N, N, sh_stream);
            }
            try {
                shared_expert_multi(n, xm, sh_bf16_ + tb * N, nsw, (const uint16_t*) wgi->data, sh_gate_ + (size_t) tb * g.n_ff,
                                    sh_up_ + (size_t) tb * g.n_ff, sh_g_ + tb, shared_ + tb * N, N, g.n_ff, sh_stream);
            } catch (const std::exception& e) {
                err = std::string("verify shared expert: ") + e.what();
                return false;
            }
            if (sh_fork) cudaEventRecord(ev_join_, sh_stream);
            }
        }
        if (dm) {}   // the route kernel wrote nat_xq_
        else if (strata::kernels::cpu::expert_layout().native)
            quantize_q8_1_rows(xm, n, N, nat_xq_ + (size_t) tb * (N / 32) * 36, cs);
        else
            quantize_q8_0_scaled(xm, hit_xq_ + (size_t) tb * (N / 32) * 34, hit_xs_ + (size_t) tb * (N / 32), (int64_t) n * N, cs);
        stamp(l, 18, grp);
        return true;
    };

    // ---------------------------------------------------------------- post(l, group): experts, combine
    static const bool fuse_head_gr_env = [] {
        const char* e = std::getenv("STRATA_FUSE_HEAD_GR");
        return e && e[0] == '1';
    }();
    const bool fuse_head_gr = fuse_head_gr_env && (le_ == g.n_layers) && (head_ != nullptr && head_->loaded()) &&
                              !cvec().covers(g.n_layers - 1);
    auto post = [&](int64_t l, int grp) -> bool {
        const int tb = tb_[grp], te = te_[grp], n = te - tb;
        const uint32_t ring = (uint32_t) ((l - lb_) * G + grp + 1);
        const int64_t cap = (int64_t) n * K, capx = (int64_t) max_t_ * K;
        int32_t* pl = plan_ + (size_t) grp * (size_t) (plan_i32_ + 16);
        const int32_t* p_counts = pl;
        const int32_t* p_start = pl + 4;
        const int32_t* p_dst = p_start + capx + 1;
        const int32_t* p_tok = p_dst + capx;
        const int64_t ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
        const unsigned long long* p_ptr = (const unsigned long long*) (pl + ptr_off);
        const unsigned long long* p_ptr2 = p_ptr + capx;
        const int32_t* p_start2 = pl + ptr_off + 4 * capx;
        float* hit_out = hit_out_ + (size_t) tb * K * N;
        float* parts_out = parts_ + (size_t) tb * K * N;
        const auto& lay = strata::kernels::cpu::expert_layout();
        // plan v0.3 P6: the VRAM groups now; the PCIe groups once the copy engine has landed them in staging.
        // `gy`: the native launch's groups side by side (0: cap, one block row per possible group).
        auto grouped = [&](const unsigned long long* gp, const int32_t* gs, const int32_t* gn, int64_t gy, float* dst_buf) {
            if (lay.native) {
                // the layer's GGUF formats (i-quant gate/up, Q2_0 / IQ4_NL down)
                const auto& f = lay.fmt[(size_t) l];
                const NativeExpertLayout L = native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
                native_expert_grouped(L, gp, gs, gn, p_dst, p_tok, cap, cap,
                                      nat_xq_ + (size_t) tb * (N / 32) * 36, hit_scratch_, dst_buf, cs, gy);
            } else {
                moe_grouped_s2(gp, gs, gn, p_dst, p_tok, cap, cap, hit_xq_ + (size_t) tb * (N / 32) * 34,
                               hit_xs_ + (size_t) tb * (N / 32), hit_scratch_, dst_buf, cs);
            }
        };
        if (dm) {   // CUDA0's entries straight into their parts rows; the other tiers are read by the combine
            stamp(l, 19, grp);
            const auto& f = lay.fmt[(size_t) l];
            const NativeExpertLayout L = native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
            DmPlan* P = dm_->plan;
            if (dm_->vram >= 0)
                dec_moe_vram(L, P, nat_xq_ + (size_t) tb * (N / 32) * 36, dm_->vscr, parts_out, dm_->vsync, cs, dm_->vram,
                             (int) cap);
            else
                native_expert_grouped(L, P->ptr, P->start, P->counts, P->dst, P->tok, cap, cap,
                                      nat_xq_ + (size_t) tb * (N / 32) * 36, hit_scratch_, parts_out, cs);
            stamp(l, 20, grp);
            stamp(l, 21, grp);
            stamp(l, 22, grp);
        } else if (all_resident_) {
            stamp(l, 19, grp);
            grouped(p_ptr, p_start, p_counts, 0, parts_out);
            stamp(l, 20, grp);
        } else {
            if (device_plan_) {   // E-6: skipped when the device planned this group (all its experts resident)
                wait_flag_ge_or(m_flagA_, ring, skip_ + grp, cs);
                copy_i32_from_mapped_unless(pl, m_plan_ + (size_t) grp * (size_t) plan_i32_, plan_i32_, skip_ + grp, ring, cs);
            } else {
                wait_flag_ge(m_flagA_, ring, cs);                  // the pool published this group's GPU plan
                copy_i32_from_mapped(pl, m_plan_ + (size_t) grp * (size_t) plan_i32_, plan_i32_, cs);
            }
            stamp(l, 19, grp);
            grouped(p_ptr, p_start, p_counts, 0, hit_out);
            stamp(l, 20, grp);
            if (device_plan_) wait_flag_ge_or(m_flagB_, ring, skip_ + grp, cs);
            else wait_flag_ge(m_flagB_, ring, cs);                 // the PCIe share is in staging (DMA) or mapped
            if (sink_.pcie_mode == 2) {                            // stage it with a copy kernel, then point at staging
                const int64_t per = G == 2 ? kStagingBlobs / 2 : kStagingBlobs;
                uint8_t* stage = staging_ + (size_t) (grp * per) * lay.max_blob;
                fetch_blobs(p_ptr2, p_counts + 2, stage, (int64_t) lay.blob_bytes(l), (int) per, cs);
                rebase_ptrs((unsigned long long*) p_ptr2, p_counts + 2, stage, (int64_t) lay.blob_bytes(l), cs);
            }
            stamp(l, 21, grp);
            // the PCIe share is pcie_frac of the misses: a few groups when the cache is cold, usually none (always none at
            // pcie_frac 0), so its launch is kPcieGroupRows block rows striding over the groups, not cap of them
            grouped(p_ptr2, p_start2, p_counts + 2, kPcieGroupRows, hit_out);
            stamp(l, 22, grp);
            if (device_plan_) {   // no CPU share when the device planned the group: its rows are zeros
                wait_flag_ge_or(m_flag_, ring, skip_ + grp, cs);
                copy_or_zero_from_mapped(parts_out, m_ymiss_ + (size_t) tb * K * N, (long long) n * K * N,
                                         skip_ + grp, ring, cs);
            } else {
                wait_flag_ge(m_flag_, ring, cs);               // the CPU's share is in the mapped rows
                stamp(l, 23, grp);
                if (remote_opt_)   // #578: the helper's rows come back reduced; skip them as well
                    remote_opt_->copy_rows(parts_out, m_ymiss_ + (size_t) tb * K * N, tb, n, p_dst, p_counts + 1, cs);
                else if (dec_batch)   // only the CPU rows cross PCIe (p_dst[0, counts[1]) = the GPU's own rows)
                    copy_rows_from_mapped(parts_out, m_ymiss_ + (size_t) tb * K * N, (int64_t) n * K, N,
                                          p_dst, p_counts + 1, cs);
                else
                    copy_from_mapped(parts_out, m_ymiss_ + (size_t) tb * K * N, (int64_t) n * K * N, cs);
            }
            moe_hit_add(parts_out, hit_out, p_dst, p_counts + 1, cap, N, cs);
        }
        if (!shx && !prof_on_ && sh_cs_ != nullptr && ev_fork_ != nullptr && ev_join_ != nullptr) {
            cudaStreamWaitEvent(cs, ev_join_, 0);
            if (ring_fork[grp]) cudaStreamWaitEvent(cs, ev_ring_join_, 0);   // the ring's launch
        }
        if (dm) {
            dec_moe_wait(dm_->plan, dm_->m_w + 16, m_flag_, ring, cs);
            stamp(l, 23, grp);
            try {
                dec_moe_combine(parts_ + (size_t) tb * K * N, dm_->m_rows, m_ymiss_, dm_->plan, w_ + tb * K,
                                shared_ + tb * N, bo_ + tb * N, N, K, n, nullptr, nullptr, ring, cs);
            } catch (const std::exception& e) { err = "verify combine: " + std::string(e.what()); return false; }
        } else if (dec_batch && n > 1 && native_moe_combine_enabled()) {   // one launch for the window's rows
            try {
                native_moe_combine_multi(parts_ + (size_t) tb * K * N, w_ + tb * K, shared_ + tb * N, bo_ + tb * N, N, K, n, cs);
            } catch (const std::exception& e) { err = "verify combine: " + std::string(e.what()); return false; }
        } else
        for (int t = tb; t < te; ++t) {
            MoEBuffers mb = ss.moe;
            mb.weights = w_ + t * K; mb.shared = shared_ + t * N;
            if (!moe_combine_parts(g, l, K, mb, parts_ + (size_t) t * K * N, bo_ + t * N, cs, err)) return false;
        }
        if (remote_opt_) remote_opt_->combine(bo_ + tb * N, y_dummy_ + tb * N, tb, n,
                              device_plan_ ? skip_ + grp : nullptr, ring, cs);
        stamp(l, 24, grp);
        if (l == g.n_layers - 1) {
            if (!fuse_head_gr) {
                for (int t = tb; t < te; ++t) gr_write(Rt(t), bo_ + t * N, inj2_ + t * HC, gs, Rt(t), cs);
                if (cvec().covers(l)) cvec_apply(Rt(tb), l, n, HC * N, nullptr, 0, nullptr, 0, false, cs);
            }
        } else if (cvec().covers(l)) {
            cvec_apply(Rt(tb), l, n, HC * N, bo_ + tb * N, N, inj2_ + tb * HC, HC, true, cs);
        }
        return true;
    };

    for (int grp = 0; grp < G; ++grp)
        if (!pre(lb_, grp)) return false;
    for (int64_t l = lb_; l < le_; ++l)
        for (int grp = 0; grp < G; ++grp) {
            if (!post(l, grp)) return false;
            if (l + 1 < le_ && !pre(l + 1, grp)) return false;
        }
    if (le_ < g.n_layers) {   // a layer split's earlier stage: hand the residual on, no head
        float* hout = hand_out_ + (size_t) hrow0 * HB;
        copy_from_mapped(hout, R_, (int64_t) T * HC * N, cs);
        copy_from_mapped(hout + (size_t) T * HC * N, bo_, (int64_t) T * N, cs);
        copy_from_mapped(hout + (size_t) T * (HC + 1) * N, inj2_, (int64_t) T * HC, cs);
        return true;
    }

    // ---- the head, T columns, and the argmax of each
    stamp(g.n_layers, 0, 0);
    {
        const WeightRef *hn = wt.find("output_hc_norm.weight"), *hd = wt.find("output_hc_down.weight"),
                        *hu = wt.find("output_hc_up.weight");
        if (!hn || !hd || !hu) { err = "verify: an output_hc_* weight is missing"; return false; }
        if (fuse_head_gr) {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                FusedGrArgs& a = fa[t];
                a.R = Rt(t); a.R_out = Rt(t); a.apply = true;
                a.bo_prev = bo_ + t * N; a.inj_prev = inj2_ + t * HC;
                a.w_norm = (const float*) hn->data;
                a.w_down = (const uint16_t*) hd->data;
                a.w_up = (const uint16_t*) hu->data;
                a.w_inject = nullptr;
                a.eps = EPS;
                a.lo = lo_ + t * g.hc_lr;
                a.rs = rs_ + t * HC;
                a.inject_out = head_inj_;
                a.mixed = head_mixed_ + t * N;
            }
            if (dec_hc) {
                HcFusedArgs h;
                h.T = T; h.R = Rt(0); h.R_out = Rt(0); h.apply = true; h.bo_prev = bo_; h.inj_prev = inj2_;
                h.w_norm = (const float*) hn->data; h.w_down = (const uint16_t*) hd->data; h.w_up = (const uint16_t*) hu->data;
                h.eps = EPS; h.lo = lo_; h.rs = rs_; h.mixed = head_mixed_; h.xn = xn_;
                hc_fused_read(h, cs);
            } else {
                fused_gr_read_multi(fa, T, xn_, cs);
            }
        } else {
            for (int t = 0; t < T; ++t) {
                BlockBuffers bb = ss.block;
                bb.R = Rt(t);
                bb.mixed = head_mixed_ + t * N;
                if (head_ != nullptr && head_->loaded()) {
                    if (!lm_head_mix(wt, g, bb, cs, err)) return false;
                } else if (!lm_head(wt, g, bb, head_logits_ + (size_t) t * n_vocab_, cs, err)) {
                    return false;
                }
            }
        }
        if (head_ != nullptr && head_->loaded()) {
            try {
                native_quantize_q8_1(head_mixed_, xq_, (int) N, T, cs);
                if (dec_qsa && head_->type() == 13 && q5k_head_supported((int) N, T))   // the same bits, one wave
                    q5k_head_mmvq(head_->weights(), xq_, head_logits_, (int) N, (int) n_vocab_, T, cs);
                else
                    native_mmvq(head_->type(), head_->weights(), xq_, head_logits_, (int) N, (int) n_vocab_, T, cs);
            } catch (const std::exception& e) {
                err = std::string("verify head: ") + e.what();
                return false;
            }
        }
        // Greedy, the default, is recorded here as before (no extra launch or sync per window). A request that
        // samples or penalizes is sampled again host-side after the replay (run()) with its own parameters and a
        // fresh draw counter: a captured sampler would bake them in and replay the same draws forever.
        SamplerParams sp;
        sp.greedy = true;
        sp.temperature = 0.0f;
        sample_tokens(head_logits_, T, (int) n_vocab_, nullptr, 0, sp, m_out_, cs);
    }
    stamp(g.n_layers, 1, 0);
    return true;
}

std::string Verifier::profile_report() {
    if (!prof_on_ || prof_windows_ == 0) return std::string();
    static const char* names[kProfPer] = {"-", "hc-read0", "q8+qkv/q-idx gemv", "conv", "ab", "z", "rec", "q8+kv-idx",
                                          "k/v+norm-rope", "kv+idx append", "q+q-idx", "scores+topk", "kv-resolve",
                                          "attention", "gate", "", "out-proj", "hc-read1+router", "shared+quant",
                                          "waitA", "VRAM hits", "waitB", "PCIe grp", "waitCPU", "copy+combine",
                                          "(gap)", "head", "  hc0 norm", "  hc0 down", "  hc0 up", "", "", ""};
    std::string out;
    char b[80];
    double total = 0;
    for (int k = 0; k < 2; ++k) {
        out += k == 0 ? " GDN layers:" : " | QSA layers:";
        for (int i = 0; i < kProfPer; ++i) {
            if (prof_sum_[k][i] <= 0) continue;
            total += prof_sum_[k][i];
            std::snprintf(b, sizeof b, " %s %.2f", names[i], prof_sum_[k][i] / 1e6 / (double) prof_windows_);
            out += b;
        }
    }
    std::snprintf(b, sizeof b, " | total %.2f ms/window over %lld windows", total / 1e6 / (double) prof_windows_, (long long) prof_windows_);
    out += b;
    std::snprintf(b, sizeof b, " | graph launch %.3f ms/window", ms_launch / (double) prof_windows_);
    out += b;
    ms_launch = 0;
    DecodeGaps& dg = decode_gaps();
    if (dg.n_v2d > 0 && dg.n_d2v > 0) {
        std::snprintf(b, sizeof b, " | GPU idle verify->draft %.3f, draft GPU %.3f, idle draft->verify %.3f ms",
                      dg.v2d / 1e6 / (double) dg.n_v2d, dg.draft_gpu / 1e6 / (double) dg.n_v2d,
                      dg.d2v / 1e6 / (double) dg.n_d2v);
        out += b;
    }
    if (dg.host_on) {
        static const char* hn[32] = {"peer/copy sync", "profile read", "host sample", "run tail", "commit", "emit",
                                     "draft stage", "draft launch", "draft wait", "draft sync", "draft prof read",
                                     "run entry", "stage+ple", "dm_sync", "peer launch", "graph launch",
                                     "adapt join", "observe", "apply_pending", "window/penalty", "adapt thread", "", "", "", "", "", "", "", "commit GPU",
                                     "commit>draft", "verify>commit", ""};
        out += " | host ms/window:";
        for (int i = 0; i < 31; ++i) {
            if (!hn[i][0]) continue;
            std::snprintf(b, sizeof b, " %s %.3f", hn[i], dg.hms[i] / (double) prof_windows_);
            out += b;
            dg.hms[i] = 0;
        }
        const double gs = prof_graph0_ / 1e6 / (double) prof_windows_;
        std::snprintf(b, sizeof b, " | graph start->layer0 %.3f ms", gs);
        out += b;
        prof_graph0_ = 0;
    }
    dg.v2d = dg.d2v = dg.draft_gpu = 0;
    dg.n_v2d = dg.n_d2v = 0;
    for (auto& r : prof_sum_) for (double& d : r) d = 0;
    prof_windows_ = 0;
    return out;
}

bool Verifier::capture(int T, std::string& err) {
    if (exec_[T] != nullptr) return true;
    const bool pdl_try = strata::kernels::pdl_requested() && !pdl_off_;
    // a window that does not capture or instantiate with programmatic dependent launch is captured again without it
    auto retry_plain = [&](const std::string& why) {
        std::fprintf(stderr, "strata verify: the %d-token window failed with STRATA_PDL=1 (%s): capturing it without\n", T,
                     why.c_str());
        (void) cudaGetLastError();
        pdl_off_ = true;
        exec_[T] = nullptr;
        return capture(T, err);
    };
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        err = "verify: begin capture failed";
        return false;
    }
    std::string rerr;
    pdl_rec_ = pdl_try;
    strata::kernels::pdl_count() = 0;
    const bool ok = record_window(T, cs_, rerr);
    pdl_rec_ = false;
    const long pdl_n = strata::kernels::pdl_count();
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs_, &graph);
    if (!ok) {
        if (graph) cudaGraphDestroy(graph);
        if (pdl_try) return retry_plain(rerr);
        err = rerr;
        return false;
    }
    if (ce != cudaSuccess) {
        if (pdl_try) return retry_plain(cudaGetErrorString(ce));
        err = std::string("verify: end capture: ") + cudaGetErrorString(ce);
        return false;
    }
#if !defined(STRATA_USE_HIP)   // a CUDA debug listing (node types, kernel names)
    if (std::getenv("STRATA_VERIFY_NODES") != nullptr) {   // what the window graph holds
        size_t nn = 0;
        cudaGraphGetNodes(graph, nullptr, &nn);
        std::vector<cudaGraphNode_t> nodes(nn);
        cudaGraphGetNodes(graph, nodes.data(), &nn);
        std::map<std::string, int> kinds;
        for (cudaGraphNode_t nd : nodes) {
            cudaGraphNodeType ty;
            cudaGraphNodeGetType(nd, &ty);
            std::string name = "type" + std::to_string((int) ty);
            if (ty == cudaGraphNodeTypeKernel) {
                cudaKernelNodeParams kp{};
                if (cudaGraphKernelNodeGetParams(nd, &kp) == cudaSuccess) {
#if CUDART_VERSION >= 12030   // cudaFuncGetName arrived in CUDA 12.3
                    const char* fn = nullptr;
                    if (cudaFuncGetName(&fn, kp.func) == cudaSuccess && fn) name = fn;
#endif
                }
            } else if (ty == cudaGraphNodeTypeMemcpy) name = "memcpy";
            else if (ty == cudaGraphNodeTypeMemset) name = "memset";
            ++kinds[name];
        }
        std::vector<std::pair<int, std::string>> v;
        for (auto& [k2, c] : kinds) v.push_back({c, k2});
        std::sort(v.rbegin(), v.rend());
        std::fprintf(stderr, "strata verify: the %d-token window graph has %zu nodes:", T, nn);
        for (size_t i = 0; i < v.size() && i < 40; ++i) std::fprintf(stderr, " %d x %.60s;", v[i].first, v[i].second.c_str());
        std::fprintf(stderr, "\n");
    }
#endif
    const cudaError_t ie = cudaGraphInstantiateWithFlags(&exec_[T], graph, dec_pk() ? cudaGraphInstantiateFlagUseNodePriority : 0);
    cudaGraphDestroy(graph);
    if (ie != cudaSuccess) {
        if (pdl_try) return retry_plain(cudaGetErrorString(ie));
        err = std::string("verify: instantiate: ") + cudaGetErrorString(ie);
        return false;
    }
    const cudaError_t ue = cudaGraphUpload(exec_[T], cs_);
    const cudaError_t us = cudaStreamSynchronize(cs_);
    if (pdl_try)
        std::fprintf(stderr, "strata verify: captured the %d-token window (upload %s, sync %s), %ld programmatic "
                             "dependent launches\n", T, cudaGetErrorString(ue), cudaGetErrorString(us), pdl_n);
    else
        std::fprintf(stderr, "strata verify: captured the %d-token window (upload %s, sync %s)\n", T,
                     cudaGetErrorString(ue), cudaGetErrorString(us));
    return true;
}

bool Verifier::capture_commit(std::string& err) {
    if (commit_exec_ != nullptr) return true;
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    const QsaShapes s = shapes_of(g);
    const int64_t C = g.ssm_conv_channels, HV = g.ssm_v_heads, ID = g.idx_key_dim, MT = max_t_;
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    const int64_t TS = (s.idx_block - 1) * ID;
    const int64_t HS = (int64_t) NG_HIST * NG_HC_DIM;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        err = "verify: begin commit capture failed";
        return false;
    }
    bool ok = true;
    try {
        if (prof_on_) gpu_stamp(prof_, (int) (g.n_layers * kProfPer + 3), cs_);
        copy_i32_from_mapped(commit_, m_commit_, 2 + MT, cs_);
        int64_t qsa_index = 0, gdn_index = 0;
        for (int64_t l = 0; l < lb_; ++l) (is_qsa_layer(g, l) ? qsa_index : gdn_index) += 1;
        static const bool dec_commit = dec_commit_mode() > 0;
        if (dec_commit) {   // every GDN layer's replay in one launch, every QSA layer's indexer appends in another
            GdnCommitLayers ga;
            ga.state = ss.gdn_state + (size_t) (gdn_index - ss.gdn_ord0) * gdn_floats;
            ga.h = h_L_ + (size_t) gdn_index * MT * C; ga.qkv = qkv_L_ + (size_t) gdn_index * MT * C;
            ga.gate = gate_L_ + (size_t) gdn_index * MT * HV; ga.beta = beta_L_ + (size_t) gdn_index * MT * HV;
            ga.state_stride = gdn_floats; ga.h_stride = (size_t) MT * C; ga.gb_stride = (size_t) MT * HV;
            ga.C = (int) C; ga.h_k = (int) g.ssm_k_heads; ga.h_v = (int) HV; ga.n_keep = commit_;
            QsaCommitLayers qa;
            qa.pos = commit_ + 2; qa.epsilon = EPS; qa.n_tok = (int) MT;
            for (int64_t l = lb_; l < le_ && ok; ++l) {
                if (!is_qsa_layer(g, l)) { ++ga.n_layers; continue; }
                const LayerView v(*wt_, l);
                const WeightRef* wikn = need(v, "indexer.k_norm.weight", err);
                if (!wikn || qa.n_layers >= kQsaCommitMaxLayers) { ok = false; break; }
                const QsaState& st = ss.qsa_states[qsa_index + qa.n_layers];
                QsaCommitLayer& q = qa.layer[qa.n_layers++];
                q.raw = idx_raw_L_ + (size_t) ((qsa_index + qa.n_layers - 1) * MT) * ID;
                q.tail_snap = tail_snap_ + (size_t) (qsa_index + qa.n_layers - 1) * TS;
                q.gamma = (const float*) wikn->data;
                q.tail = st.idx_tail; q.dead = st.idx_dead; q.pooled = st.idx_pooled; q.block_pos = st.idx_block_pos;
                q.max_cells = (int) st.max_cells;
            }
            if (ok) {
                gdn_commit_layers(ga, cs_, dec_commit_blocks());
                native_qsa_indexer_commit_layers(qa, s, rope_scaling(), cs_);
            }
        }
        for (int64_t l = lb_; l < le_ && ok && !dec_commit; ++l) {
            const LayerView v(*wt_, l);
            if (!is_qsa_layer(g, l)) {
                const WeightRef* wnm = need(v, "ssm_norm.weight", err);
                if (!wnm) { ok = false; break; }
                float* state = ss.gdn_state + (size_t) (gdn_index - ss.gdn_ord0) * gdn_floats;
                float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                const float* qkv = qkv_L_ + (size_t) gdn_index * MT * C;
                gdn_conv_commit(conv, qkv, (int) C, commit_, cs_);
                gdn_step_norm_multi(state, h_L_ + (size_t) gdn_index * MT * C, (int) C, gate_L_ + (size_t) gdn_index * MT * HV,
                                    beta_L_ + (size_t) gdn_index * MT * HV, z_, (const float*) wnm->data, EPS, y_dummy_,
                                    (int) g.ssm_k_heads, (int) HV, (int) MT, commit_, cs_, (int) MT);
                ++gdn_index;
            } else {
                const QsaState& st = ss.qsa_states[qsa_index];
                const WeightRef* wikn = need(v, "indexer.k_norm.weight", err);
                if (!wikn) { ok = false; break; }
                copy_from_mapped(st.idx_tail, tail_snap_ + (size_t) qsa_index * TS, TS, cs_);
                const QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                for (int64_t t = 0; t < MT; ++t)
                    native_qsa_indexer_append(idx_raw_L_ + (size_t) (qsa_index * MT + t) * ID, commit_ + 2 + t, 0,
                                              (const float*) wikn->data, EPS, ib, s, st.max_cells,
                                              rope_scaling(), cs_);
                ++qsa_index;
            }
        }
        if (ok && ss.ple.ready() && ple_stage()) copy_indexed(ss.ple.hist, hist_snap_, HS, commit_ + 1, HS, cs_);
        if (prof_on_) gpu_stamp(prof_, (int) (g.n_layers * kProfPer + 4), cs_);
    } catch (const std::exception& e) {
        err = std::string("verify commit: ") + e.what();
        ok = false;
    }
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs_, &graph);
    if (!ok) {
        if (graph) cudaGraphDestroy(graph);
        return false;
    }
    if (ce != cudaSuccess || cudaGraphInstantiate(&commit_exec_, graph, 0) != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        err = std::string("verify: commit capture: ") + cudaGetErrorString(ce);
        return false;
    }
    cudaGraphDestroy(graph);
    return true;
}

void Verifier::stage_inputs(int T, const int32_t* tokens, int64_t pos0) {
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    const QsaShapes s = shapes_of(g);
    int32_t* const pk = h_pos_ + (size_t) max_t_ * g.n_head;
    int32_t* const pi = pk + (size_t) max_t_ * g.n_head_kv;
    for (int t = 0; t < T; ++t) {
        h_tok_[t] = tokens[t];
        qsa_step_fill(h_step_ + t * kStepCount, pos0 + t, s);
        const int32_t pos_t = (int32_t) (pos0 + t);
        for (int64_t h = 0; h < g.n_head; ++h) h_pos_[t * g.n_head + h] = pos_t;
        for (int64_t h = 0; h < g.n_head_kv; ++h) pk[t * g.n_head_kv + h] = pos_t;
        for (int64_t h = 0; h < g.idx_q_heads; ++h) pi[t * g.idx_q_heads + h] = pos_t;
    }
    *(volatile uint32_t*) h_seq_ = 0;
    *(volatile uint32_t*) h_flag_ = 0;
    *(volatile uint32_t*) h_flagA_ = 0;
    *(volatile uint32_t*) h_flagB_ = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    last_t_ = T;
    last_pos0_ = pos0;
    for (int t = 0; t < T; ++t) last_tokens_[t] = tokens[t];
    staged_ = true;
}

DecodeGaps& decode_gaps() {
    static DecodeGaps g;
    return g;
}

void Verifier::collect_profile() {
    const ModelGeometry& g = *g_;
    cudaMemcpy(prof_h_.data(), prof_, prof_h_.size() * 8, cudaMemcpyDeviceToHost);
    const int64_t L = g.n_layers;
    auto at = [&](int64_t l, int i) { return prof_h_[(size_t) (l * kProfPer + i)]; };
    // D8: the derived columns below read stamps the hc-read kernels write themselves or the next
    // layer's first.  A slot no kernel stamped is 0 and its unsigned difference wrapped to ~1e19 ns -
    // which is why (gap)/head/hc0 printed ~1e14 ms per window.  A missing or out-of-order stamp now
    // contributes nothing.
    const auto gap = [](unsigned long long to, unsigned long long from) {
        return (from != 0 && to != 0 && to >= from) ? (double) (to - from) : 0.0;
    };
    for (int64_t l = 0; l < L; ++l) {
        const int kind = is_qsa_layer(g, l) ? 1 : 0;
        unsigned long long prev = at(l, 0);
        for (int i = 1; i <= 24; ++i) {
            const unsigned long long x = at(l, i);
            if (x == 0 || x < prev) continue;
            prof_sum_[kind][i] += (double) (x - prev);
            prev = x;
        }
        if (l + 1 < L) prof_sum_[kind][25] += gap(at(l + 1, 0), at(l, 24));
        const double dn = gap(at(l, 27), at(l, 0)), dd = gap(at(l, 28), at(l, 27)), du = gap(at(l, 1), at(l, 28));
        if (dn > 0 && dd > 0 && du > 0) {   // the split exists: show it split, not twice
            prof_sum_[kind][27] += dn;      // hc-read0: norm
            prof_sum_[kind][28] += dd;      //           down
            prof_sum_[kind][29] += du;      //           up (through both halves, as before)
            prof_sum_[kind][1] -= gap(at(l, 1), at(l, 0));   // (hc-read0 shown split)
        }
    }
    prof_sum_[0][26] += gap(at(L, 1), at(L, 0));
    prof_graph0_ += gap(at(0, 0), at(L, 2));
    DecodeGaps& dg = decode_gaps();
    if (dg.draft_end != 0 && at(0, 0) > dg.draft_end) {
        dg.d2v += (double) (at(0, 0) - dg.draft_end);
        ++dg.n_d2v;
        dg.hms[28] += gap(at(L, 4), at(L, 3)) / 1e6;          // the commit's GPU span
        dg.hms[29] += gap(at(L, 4), dg.draft_end) / 1e6;      // the commit still running after the draft
        dg.hms[30] += gap(at(L, 3), dg.verify_prev_end) / 1e6;   // the window's end to the commit's start
    }
    dg.verify_prev_end = at(L, 1);
    dg.draft_end = 0;
    dg.verify_end = at(L, 1);
    ++prof_windows_;
}

bool Verifier::run(int T, const int32_t* tokens, int64_t pos0, PoolMultiFn pool, void* user, int32_t* out,
                   std::string& err) {
    using namespace strata::kernels;
    const OnDevice on_device(device_);
    last_batch_ = false;
    if (T < 1 || T > max_t_) { err = "verify: window size out of range"; return false; }
    if (released_.load()) { err = "verify: an earlier window never finished on the GPU (#267); restart the engine"; return false; }
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    if (pos0 + T > ss.qsa_states[ss.qsa_primary()].max_cells) { err = "verify: the window runs past the context"; return false; }
    if (!capture(T, err) || !capture_commit(err)) return false;
    VDBG("captured; staging\n");
    DecodeGaps& gm = decode_gaps();
    gm.mark(11);
    const Clock::time_point t0 = Clock::now();
    if (!staged_) stage_inputs(T, tokens, pos0);
    staged_ = false;
    const bool do_ple = ss.ple.ready() && ple_stage();
    uint32_t ple_rows[kVerifyMaxT * PLE_N_HEADS];
    if (do_ple) {
        int32_t prev[2] = {ss.ple_prev[0], ss.ple_prev[1]};
        for (int t = 0; t < T; ++t) {
            ngram_rows(&tokens[t], prev, 1, ss.ple.consts, ple_rows + t * PLE_N_HEADS);
            prev[0] = prev[1];
            prev[1] = tokens[t];
            ss.ple.table->prefetch_rows(ple_rows + t * PLE_N_HEADS);
        }
    }
    if (trace_h_ != nullptr) std::memset(trace_h_, 0, trace_n_ * 8);   // #649: this window's breadcrumbs only
    std::atomic_thread_fence(std::memory_order_seq_cst);
    trace_ev("WINDOW", -1, -1, pos0 * 16 + T);
    gm.mark(12);
    if (dm_ != nullptr) {   // STRATA_DEC_MOE: this window's residency, then the peer window ahead of CUDA0's
        if (!dm_sync(err)) return false;
        gm.mark(13);
        *(volatile uint32_t*) (dm_->h_w + 0) = 0;
        *(volatile uint32_t*) (dm_->h_w + 16) = 0;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (dm_->pdev >= 0) {
            const OnDevice on(dm_->pdev);
            DmPeerArgs pa;
            pa.rings_m = dm_->h_ring; pa.xq_m = dm_->h_xq; pa.xq_stride = dm_->xq_stride;
            pa.seq_m = dm_->h_w + 0; pa.done_m = dm_->h_w + 16; pa.rows_m = dm_->h_rows;
            pa.ptr = dm_->ptab; pa.lay = dm_->lay; pa.n_expert = (int) g_->n_expert; pa.k = (int) ss_->k;
            pa.n_embd = (int) g_->n_embd; pa.n_ff = (int) strata::kernels::cpu::expert_layout().fmt[0].n_ff;
            pa.steps = (int) dm_->rings; pa.scratch = dm_->pscr; pa.arrive = dm_->parr;
            if (cudaMemsetAsync(dm_->parr, 0, 64, dm_->ps) != cudaSuccess || !dec_moe_peer_launch(pa, dm_->pblocks, dm_->ps)) {
                err = "verify: the peer window did not launch";
                return false;
            }
            dm_->peer_live = true;
        }
        gm.mark(14);
    }
    ms_host += ms_since(t0);
    VDBG("staged; launching\n");
    if (prof_on_) decode_gaps().verify_end = 0;   // set again only by a collected window
    const Clock::time_point tl = Clock::now();
    const cudaError_t le = cudaGraphLaunch(exec_[T], cs_);
    ms_launch += ms_since(tl);
    trace_ev("LAUNCHED", -1, -1, (int64_t) le);
    if (le != cudaSuccess) { err = std::string("verify: launch: ") + cudaGetErrorString(le); return false; }
    // STRATA_DEC_CHAIN: a sampled or penalized request's head sampling queued behind the window now, not after a sync
    const bool sampled = !sampling_.greedy && sampling_.temperature > 0.0f;
    const bool early_sample = chain_on() && le_ >= g.n_layers && head_sampling_ && (sampled || hist_d_ != nullptr);
    if (early_sample) {
        SamplerParams sp = sampling_;
        sp.counter = (uint64_t) pos0;
        sample_tokens(head_logits_, T, (int) n_vocab_, hist_d_, hist_len_, sp, m_out_, cs_);
    }
    (void) cudaStreamQuery(cs_);
    gm.mark(15);
    VDBG("launched\n");
    volatile uint32_t* const seq = h_seq_;
    volatile uint32_t* const flag = h_flag_;
    const int G = groups_[T] > 0 ? groups_[T] : 1;
    const int gtb[2] = {0, (T + 1) / 2}, gte[2] = {G == 2 ? (T + 1) / 2 : T, T};
    const int64_t steps = (le_ - lb_) * G;
    const bool test_stall = g_test_stall > 0 && windows + 1 == g_test_stall;   // #267 test hook (off: false)
    if (all_resident_ && !test_stall) {
        if (do_ple) {
            const Clock::time_point tp = Clock::now();
            if (!ss.ple.table->gather_batch(ple_rows, (size_t) T, h_ple_, err)) return false;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            _mm_sfence();
            *flag = 1;
            ms_host += ms_since(tp);
        } else {
            *flag = 1;
        }
    } else
    for (int64_t k = 0; k < steps; ++k) {
        const int64_t l = lb_ + k / G;
        const int grp = (int) (k % G);
        const uint32_t want = (uint32_t) (k + 1);
        const Clock::time_point a = Clock::now();
        auto last_flush = a;
        uint32_t spins = 0;
        progress_at("verify window: waiting for the GPU to reach layer", l);
        while (*seq < want) {
            _mm_pause();
            if ((++spins & 1023u) != 0) continue;
            const auto now = Clock::now();
            if (now - last_flush > std::chrono::microseconds(2000)) {
                last_flush = now;
                const cudaError_t q = cudaStreamQuery(cs_);
                if (q != cudaErrorNotReady && *seq < want) {
                    trace_ev("NEVER-RANG", k, l, (int64_t) q);
                    trace_dump(stderr);
                    err = "verify: layer " + std::to_string(l) + " never rang (" +
                          (q == cudaSuccess ? std::string("graph finished") : std::string(cudaGetErrorString(q))) + ")";
                    return false;
                }
            }
            if (now - a > std::chrono::seconds(20)) {
                // #267: the caller ends the engine; no spin kernel may outlive it
                trace_ev("TIMEOUT", k, l, (int64_t) cudaStreamQuery(cs_));
                const std::string words = dm_note();   // read before the release overwrites them
                if (g_trace) diag(stderr);   // the words and the breadcrumbs before the release ...
                const bool drained = release_gpu_waits(5000);
                if (g_trace) trace_dump(stderr);   // ... and after it: did the GPU move once its waits were raised?
                err = "verify: timed out at layer " + std::to_string(l) + words + released_note(drained) + peer_released();
                return false;
            }
        }
        const Clock::time_point b = Clock::now();
        if (g_trace) trace_ev("RANG", k, l, (int64_t) std::chrono::duration_cast<std::chrono::microseconds>(b - a).count());
        VDBG("layer %lld rang\n", (long long) l);
        cur_layer_ = want - 1;
        set_plan_slot(grp);
        const int tb = gtb[grp], n = gte[grp] - gtb[grp];
        progress_at("verify window: the CPU experts of layer", l);
        if (remote_opt_) remote_opt_->begin(h_w_ + (size_t) tb * ss.k, tb, n);
        if (dm_ != nullptr) {   // the ring's own rows and tiers: the pool computes the CPU's entries only
            const strata::kernels::DmRing& R = dm_->h_ring[k];
            sink_.ext_kind = R.kind;
            if (pool != nullptr)
                pool(user, dm_->h_x + (size_t) k * dm_->x_stride, R.ids, n, ss.k, h_ymiss_ + (size_t) tb * ss.k * g.n_embd, l);
            sink_.ext_kind = nullptr;
        } else if (pool != nullptr)
            pool(user, h_x_ + (size_t) tb * g.n_embd, h_ids_ + (size_t) tb * ss.k, n, ss.k,
                 h_ymiss_ + (size_t) tb * ss.k * g.n_embd, l);
        if (remote_opt_) remote_opt_->end();
        VDBG("layer %lld served\n", (long long) l);
        if (g_trace) trace_ev(*(volatile uint32_t*) h_flagA_ == want ? "SERVED" : "SERVED-NO-PLAN-YET", k, l,
                              (int64_t) ms_since(b));   // aux: ms the CPU experts took
        progress_tick();
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
        if (*(volatile uint32_t*) h_flagA_ != want) {        // the pool did not publish a plan: an empty one
            sink_.counts[0] = 0;
            sink_.counts[1] = 0;
            sink_.counts[2] = 0;
            sink_.start[0] = 0;
            sink_.start2[0] = 0;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile uint32_t*) h_flagA_ = want;
            raise_flag(h_flagB_, want);
        }
        // Layer 1's pre(1, 0) copies h_ple_ -> ple_ after Layer 0's wait_flag_ge(m_flag_, 1).
        // By collecting PLE here at k == 0 (after publishing flagA/flagB for Layer 0 so the GPU can run
        // Layer 0's VRAM experts, and before raising *flag = 1), the NVMe PLE reads overlap with both
        // MtpDrafter::draft and Layer 0's attention + router + expert execution!
        if (k == 0 && do_ple) {
            const Clock::time_point tp = Clock::now();
            if (!ss.ple.table->gather_batch(ple_rows, (size_t) T, h_ple_, err)) return false;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            _mm_sfence();
            ms_host += ms_since(tp);
        }
        if (!(test_stall && k + 1 == steps)) *flag = want;
        ms_wait += std::chrono::duration<double, std::milli>(b - a).count();
        ms_pool += ms_since(b);
    }
    // (#646 staged the next stage's inputs here; 0.1.39b keeps the layer split's order: each stage stages its own)
    progress_at("verify window: waiting for the GPU to finish the window (flags A/B/M raised)", (int64_t) T);
    // #267: a window the GPU never finishes (a spin kernel that never sees its flag) holds the host here; the stall
    // watchdog then releases every verifier's GPU waits (release_live_verifiers) before it ends the engine, so no
    // spin kernel outlives the process - the case that left Windows GPUs "lost" until a power cycle.  The wait
    // itself stays a blocking sync: a cudaStreamQuery poll here cost IQ3_S ~3% decode (a core calling the driver
    // beside the expert workers).
    trace_ev("SYNC", -1, -1, 0);
    const cudaError_t se = cudaStreamSynchronize(cs_);
    trace_ev("SYNCED", -1, -1, (int64_t) se);
    gm.mark(-1);
    if (dm_ != nullptr && dm_->peer_live) {   // the peer window ends with CUDA0's last ring
        const OnDevice on(dm_->pdev);
        dm_->peer_live = false;
        const cudaError_t pe = cudaStreamSynchronize(dm_->ps);
        if (pe != cudaSuccess) { err = std::string("verify: the peer window: ") + cudaGetErrorString(pe); return false; }
    }
    if (se != cudaSuccess) { err = std::string("verify: ") + cudaGetErrorString(se); return false; }
    commit_pending_ = false;
    progress_at("verify window: waiting for the expert copies", (int64_t) T);
    if (copy_used_) {
        cudaStreamSynchronize(copy_);   // no host function of this window may raise flag B in the next one
        copy_used_ = false;
    }
    gm.mark(0);
    if (prof_on_ && G == 1) collect_profile();   // the window's GPU stage stamps
    gm.mark(1);
    // ---- a sampled or penalized request: the head's sampling again, host-side so its parameters are this call's
    // own (a captured kernel would replay the same draws forever).  Row t's draw is Philox(seed, pos0 + t): tied to
    // the POSITION it samples, not to how the text was cut into windows, so a seed replays the same text whatever
    // the drafts were. Exact: a rejected row's draw is discarded, and no kept decision depends on a reused draw.
    if (le_ < g.n_layers) {   // a layer split's earlier stage: the hand-off is written (synced above)
        ++windows;
        return next_ == nullptr || next_->run(T, tokens, pos0, pool, next_user_, out, err);
    }
    if (head_sampling_ && (sampled || hist_d_ != nullptr) && !early_sample) {
        SamplerParams sp = sampling_;
        sp.counter = (uint64_t) pos0;
        sample_tokens(head_logits_, T, (int) n_vocab_, hist_d_, hist_len_, sp, m_out_, cs_);
        if (cudaStreamSynchronize(cs_) != cudaSuccess) {   // m_out_ is the mapped h_out_: synced, it is readable
            err = "verify: the head sampling failed";
            return false;
        }
    }
    gm.mark(2);
    for (int t = 0; t < T; ++t) out[t] = ((volatile int32_t*) h_out_)[t];
    if (static const bool dbg = std::getenv("STRATA_DBG_NAN") != nullptr; dbg) {   // debug: the first non-finite head
        static bool reported = false;
        if (!reported) {
            std::vector<float> h((size_t) T * (size_t) n_vocab_);
            cudaMemcpy(h.data(), head_logits_, h.size() * 4, cudaMemcpyDeviceToHost);
            for (int t = 0; t < T && !reported; ++t) {
                int64_t bad = 0;
                for (int64_t v = 0; v < n_vocab_; ++v) bad += !std::isfinite(h[(size_t) t * n_vocab_ + v]);
                if (bad) {
                    reported = true;
                    std::fprintf(stderr, "strata dbg: verify window at position %lld, row %d: %lld of %lld logits non-finite "
                                         "(token out %d)\n", (long long) pos0, t, (long long) bad, (long long) n_vocab_, out[t]);
                }
            }
        }
    }
    VDBG("window done\n");
    gm.mark(3);
    ++windows;
    progress_at("decode");
    progress_beat();
    return true;
}

void Verifier::set_plan_slot(int grp) {
    const int64_t cap = sink_.cap;
    int32_t* base = h_plan_ + (size_t) grp * (size_t) plan_i32_;
    const int64_t i32 = 4 + (cap + 1) + cap + cap;
    const int64_t ptr_off = (i32 + 1) & ~1ll;
    sink_.counts = base;
    sink_.start = base + 4;
    sink_.dst = sink_.start + cap + 1;
    sink_.tok = sink_.dst + cap;
    sink_.ptr = (unsigned long long*) (base + ptr_off);
    sink_.ptr2 = sink_.ptr + cap;
    sink_.start2 = base + ptr_off + 4 * cap;
    const int G = last_batch_ ? 1 : (groups_[last_t_] > 0 ? groups_[last_t_] : 1);
    const int64_t per = G == 2 ? kStagingBlobs / 2 : kStagingBlobs;
    sink_.staging = (unsigned long long) (staging_ + (size_t) (grp * per) * strata::kernels::cpu::expert_layout().max_blob);
    sink_.staging_cap = per;
}

// Flag B only rises: a host function of an earlier layer may run after a later layer already raised it directly.
void Verifier::raise_flag(uint32_t* flag, uint32_t value) {
    volatile long* f = (volatile long*) flag;
#if defined(_WIN32)
    long cur = *f;
    while ((uint32_t) cur < value) {
        const long prev = _InterlockedCompareExchange(f, (long) value, cur);
        if (prev == cur) break;
        cur = prev;
    }
#else
    uint32_t cur = __atomic_load_n((uint32_t*) flag, __ATOMIC_SEQ_CST);
    while (cur < value && !__atomic_compare_exchange_n((uint32_t*) flag, &cur, value, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {}
#endif
}

// Plan v0.3 P6: the PCIe share by DMA.  The copy engine moves the blobs while the CPU computes its own share and the
// GPU its VRAM experts; a host function raises flag B when they have landed (the graph waits for it before the PCIe
// groups).  Staging is split between the two token groups of a split window.
void Verifier::fetch_dma(void* ctx, const uint8_t* const* src, int n, size_t bytes) {
    Verifier* v = (Verifier*) ctx;
    const uint32_t want = v->cur_layer_ + 1;
    if (n <= 0) { raise_flag(v->h_flagB_, want); return; }
    v->copy_used_ = true;
    uint8_t* stage = (uint8_t*) v->sink_.staging;                  // this group's half in a split window
    for (int i = 0; i < n; ++i) cudaMemcpyAsync(stage + (size_t) i * bytes, src[i], bytes, cudaMemcpyHostToDevice, v->copy_);
    FlagSet& fs = v->flag_sets_[v->cur_layer_ % (sizeof v->flag_sets_ / sizeof v->flag_sets_[0])];
    fs.flag = v->h_flagB_;
    fs.value = want;
    cudaLaunchHostFunc(v->copy_, [](void* p) { FlagSet* s = (FlagSet*) p; raise_flag(s->flag, s->value); }, &fs);
}

void Verifier::publish_plan(void* ctx) {
    Verifier* v = (Verifier*) ctx;
    _mm_sfence();
    *(volatile uint32_t*) v->h_flagA_ = v->cur_layer_ + 1;
}

bool Verifier::window_logprobs(const int32_t* targets, int T, int64_t pos0, int32_t extra_id, std::FILE* out,
                               std::string& err) {
    if (next_ != nullptr) return next_->window_logprobs(targets, T, pos0, extra_id, out, err);
    const OnDevice on_device(device_);
    if (head_logits_ == nullptr || n_vocab_ <= 0 || T <= 0 || targets == nullptr || out == nullptr) {
        err = "window_logprobs: no head logits for this window";
        return false;
    }
    // run() synchronized cs_ before returning, so the head of this window is complete
    std::vector<float> h((size_t) T * (size_t) n_vocab_);
    if (cudaMemcpy(h.data(), head_logits_, h.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        err = "window_logprobs: the head logits copy failed";
        return false;
    }
    for (int t = 0; t < T; ++t) {
        const float* row = h.data() + (size_t) t * (size_t) n_vocab_;
        const int32_t tgt = targets[t];
        if (tgt < 0 || (int64_t) tgt >= n_vocab_) continue;
        int64_t top = 0;
        for (int64_t v = 1; v < n_vocab_; ++v)
            if (row[v] > row[top]) top = v;
        const double maxv = row[top];
        const bool has_extra = extra_id >= 0 && (int64_t) extra_id < n_vocab_;
        double sum = 0.0, sum_without = 0.0;   // the second skips extra_id: no cancellation when it holds ~all mass
        for (int64_t v = 0; v < n_vocab_; ++v) {
            const double e = std::exp((double) row[v] - maxv);
            sum += e;
            if (v != (int64_t) extra_id) sum_without += e;
        }
        const double lse = maxv + std::log(sum);
        const double extra = has_extra ? (double) row[extra_id] - lse : NAN;
        const double without = has_extra && tgt != extra_id && sum_without > 0.0
                                   ? (double) row[tgt] - (maxv + std::log(sum_without)) : NAN;
        std::fprintf(out, "%lld\t%d\t%.9f\t%lld\t%.9f\t%d\t%.9f\t%.9f", (long long) (pos0 + t), (int) tgt,
                     (double) row[tgt] - lse, (long long) top, maxv - lse, (int) (top == (int64_t) tgt), extra,
                     without);
        // STRATA_LOGPOS_TOPK=K: the K most likely tokens and their log-probabilities too (`id:logprob`), for a
        // top-k comparison with another engine on the same tokens (docs/UNSLOTH_Q4.md)
        static const int topk = [] {
            const char* v = std::getenv("STRATA_LOGPOS_TOPK");
            return v != nullptr ? std::max(0, std::min(256, std::atoi(v))) : 0;
        }();
        if (topk > 0) {
            std::vector<int32_t> order((size_t) n_vocab_);
            for (int64_t v = 0; v < n_vocab_; ++v) order[(size_t) v] = (int32_t) v;
            std::partial_sort(order.begin(), order.begin() + topk, order.end(),
                              [&](int32_t a, int32_t b) { return row[a] > row[b]; });
            for (int j = 0; j < topk; ++j)
                std::fprintf(out, "\t%d:%.6f", order[(size_t) j], (double) row[order[(size_t) j]] - lse);
        }
        std::fprintf(out, "\n");
    }
    std::fflush(out);
    return true;
}

namespace { bool g_commit_async = false; }
void Verifier::set_commit_async(bool on) { g_commit_async = on && std::getenv("STRATA_COMMIT_SYNC") == nullptr; }

bool Verifier::chain_on() {
    static const bool on = [] { const char* e = std::getenv("STRATA_DEC_CHAIN"); return e && e[0] == '1'; }();
    return on;
}

bool Verifier::commit(int n_keep, std::string& err, int parts) {
    const OnDevice on_device(device_);
    if (n_keep < 1 || n_keep > last_t_) { err = "verify: commit count out of range"; return false; }
    const Clock::time_point t0 = Clock::now();
    if (parts & 1) {
        h_commit_[0] = n_keep;
        h_commit_[1] = n_keep - 1;
        for (int t = 0; t < max_t_; ++t) h_commit_[2 + t] = t < n_keep ? (int32_t) (last_pos0_ + t) : -1;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        const cudaError_t le = cudaGraphLaunch(commit_exec_, cs_);
        if (le != cudaSuccess) { err = std::string("verify: commit launch: ") + cudaGetErrorString(le); return false; }
        // set_commit_async: no wait here - the next window runs on the same stream after it, and the drafter (its own
        // stream) reads only this window's final rows and its own K/V. h_commit_ is next written after the next window's
        // results are read, i.e. after this graph has run.  Everything else waits on commit_done_ (wait_commit).
        if (!g_commit_async || next_ != nullptr) {
            const cudaError_t se = cudaStreamSynchronize(cs_);
            if (se != cudaSuccess) { err = std::string("verify: commit: ") + cudaGetErrorString(se); return false; }
        } else {
            const cudaError_t re = cudaEventRecord(commit_done_, cs_);
            if (re != cudaSuccess) { err = std::string("verify: commit event: ") + cudaGetErrorString(re); return false; }
            (void) cudaStreamQuery(cs_);
            commit_pending_ = true;
        }
        decode_gaps().mark(4);
    }
    if ((parts & 2) && ple_stage())   // stages that share one session must advance it once
        for (int t = 0; t < n_keep; ++t) {
            ss_->ple_prev[0] = ss_->ple_prev[1];
            ss_->ple_prev[1] = last_tokens_[t];
        }
    ms_commit += ms_since(t0);
    return next_ == nullptr || next_->commit(n_keep, err, parts);
}

bool Verifier::wait_commit(std::string& err) {
    if (commit_pending_) {
        const OnDevice on_device(device_);
        commit_pending_ = false;
        const cudaError_t se = cudaEventSynchronize(commit_done_);
        if (se != cudaSuccess) { err = std::string("verify: commit: ") + cudaGetErrorString(se); return false; }
    }
    return next_ == nullptr || next_->wait_commit(err);
}


// ================================ BATCH WINDOWS (see init_slots) ================================

bool Verifier::init_slots(const std::vector<SessionState*>& slots, std::string& err) {
    const OnDevice on_device(device_);
    if (g_ == nullptr || ss_ == nullptr) { err = "verify: init_slots before init"; return false; }
    if (slots.empty() || (int) slots.size() > max_t_) {
        err = "verify: init_slots needs 1.." + std::to_string(max_t_) + " sessions";
        return false;
    }
    for (SessionState* x : slots) {
        if (x == nullptr || x->layer_lo != ss_->layer_lo || x->layer_hi != ss_->layer_hi ||
            x->max_cells != ss_->max_cells || x->gdn_ord0 != ss_->gdn_ord0 || x->qsa_ord0 != ss_->qsa_ord0) {
            err = "verify: a slot's session is not carved like the stage's own (layer range, context)";
            return false;
        }
    }
    const strata::kernels::QsaShapes s = shapes_of(*g_);
    const int64_t S = (int64_t) slots.size(), CB = 2 + max_t_;
    const int64_t TS = (s.idx_block - 1) * g_->idx_key_dim, nQ = g_->n_qsa_layers();
    void* d = nullptr;
    if (!mapped((size_t) (S * CB * 4 + 16), (void**) &h_commitb_, (void**) &m_commitb_)) {
        err = "verify: the batch commit staging failed";
        return false;
    }
    const uint64_t a = ((uint64_t) S * CB * 4 + 255) & ~255ull;
    if (cudaMalloc(&d, a + (uint64_t) S * std::max<int64_t>(nQ, 1) * TS * 4) != cudaSuccess) {
        err = "verify: the batch buffers do not fit";
        return false;
    }
    arena_b_ = d;
    commitb_ = (int32_t*) d;
    tail_snap_b_ = (float*) ((uint8_t*) d + a);
    slots_ = slots;
    slot_sp_.assign(slots.size(), sampling_);   // greedy until set_slot_sampling
    std::fprintf(stderr, "strata verify: batch windows of up to %lld sequences (layers [%lld, %lld))\n", (long long) S,
                 (long long) lb_, (long long) le_);
    return true;
}

bool Verifier::capture_batch(const int* rows, int S, int hbase, std::string& err) {
    cudaGraphExec_t& ex = exec_bm_[batch_key(rows, S, hbase)];
    if (ex != nullptr) return true;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        err = "verify: begin batch capture failed";
        return false;
    }
    batch_rec_ = true;
    row_base_ = hbase;
    for (int t = 0; t < S; ++t) brow_[t] = rows[t];
    std::string rerr;
    const bool ok = record_window(S, cs_, rerr);
    batch_rec_ = false;
    row_base_ = 0;
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs_, &graph);
    if (!ok || ce != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        err = !ok ? rerr : std::string("verify: end batch capture: ") + cudaGetErrorString(ce);
        return false;
    }
    const cudaError_t ie = cudaGraphInstantiate(&ex, graph, 0);
    cudaGraphDestroy(graph);
    if (ie != cudaSuccess) { err = std::string("verify: batch instantiate: ") + cudaGetErrorString(ie); return false; }
    cudaGraphUpload(ex, cs_);
    cudaStreamSynchronize(cs_);
    std::string list;
    for (int t = 0; t < S; ++t) list += (t ? "," : "") + std::to_string(rows[t]);
    std::fprintf(stderr, "strata verify: captured the batch window over slots %s\n", list.c_str());
    return true;
}

bool Verifier::capture_commit_batch(const int* rows, int S, int hbase, std::string& err) {
    cudaGraphExec_t& cex = commit_bm_[batch_key(rows, S, hbase)];
    if (cex != nullptr) return true;
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    const QsaShapes s = shapes_of(g);
    const int64_t C = g.ssm_conv_channels, HV = g.ssm_v_heads, ZV = g.ssm_value_dim, ID = g.idx_key_dim, MT = max_t_;
    const int64_t CB = 2 + MT, nQ = g.n_qsa_layers();
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    const int64_t TS = (s.idx_block - 1) * ID;
    const int64_t HS = (int64_t) NG_HIST * NG_HC_DIM;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        err = "verify: begin batch commit capture failed";
        return false;
    }
    bool ok = true;
    try {
        for (int t = 0; t < S; ++t)
            copy_i32_from_mapped(commitb_ + (size_t) rows[t] * CB, m_commitb_ + (size_t) rows[t] * CB, CB, cs_);
        int64_t qsa_index = 0, gdn_index = 0;
        for (int64_t l = 0; l < lb_; ++l) (is_qsa_layer(g, l) ? qsa_index : gdn_index) += 1;
        for (int64_t l = lb_; l < le_ && ok; ++l) {
            const LayerView v(*wt_, l);
            if (!is_qsa_layer(g, l)) {
                const WeightRef* wnm = need(v, "ssm_norm.weight", err);
                if (!wnm) { ok = false; break; }
                for (int t = 0; t < S; ++t) {
                    SessionState& sx = *slots_[(size_t) rows[t]];
                    const int32_t* keep = commitb_ + (size_t) rows[t] * CB;
                    float* state = sx.gdn_state + (size_t) (gdn_index - sx.gdn_ord0) * gdn_floats;
                    float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                    gdn_conv_commit(conv, qkv_L_ + (size_t) gdn_index * MT * C + (size_t) t * C, (int) C, keep, cs_);
                    gdn_step_norm_multi(state, h_L_ + (size_t) gdn_index * MT * C + (size_t) t * C, (int) C,
                                        gate_L_ + (size_t) gdn_index * MT * HV + (size_t) t * HV,
                                        beta_L_ + (size_t) gdn_index * MT * HV + (size_t) t * HV, z_ + (size_t) t * ZV,
                                        (const float*) wnm->data, EPS, y_dummy_ + (size_t) t * ZV, (int) g.ssm_k_heads,
                                        (int) HV, 1, keep, cs_);
                }
                ++gdn_index;
            } else {
                const WeightRef* wikn = need(v, "indexer.k_norm.weight", err);
                if (!wikn) { ok = false; break; }
                for (int t = 0; t < S; ++t) {
                    const QsaState& st = slots_[(size_t) rows[t]]->qsa_states[qsa_index];
                    copy_from_mapped(st.idx_tail, tail_snap_b_ + ((size_t) rows[t] * nQ + qsa_index) * TS, TS, cs_);
                    const QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                    native_qsa_indexer_append(idx_raw_L_ + (size_t) (qsa_index * MT + t) * ID, commitb_ + (size_t) rows[t] * CB + 2,
                                              0, (const float*) wikn->data, EPS, ib, s, st.max_cells, rope_scaling(), cs_);
                }
                ++qsa_index;
            }
        }
        if (ok && ss_->ple.ready() && ple_stage())
            for (int t = 0; t < S; ++t)
                copy_indexed(slots_[(size_t) rows[t]]->ple_hist, hist_snap_ + (size_t) t * HS, HS,
                             commitb_ + (size_t) rows[t] * CB + 1, HS, cs_);
    } catch (const std::exception& e) {
        err = std::string("verify batch commit: ") + e.what();
        ok = false;
    }
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs_, &graph);
    if (!ok) {
        if (graph) cudaGraphDestroy(graph);
        return false;
    }
    if (ce != cudaSuccess || cudaGraphInstantiate(&cex, graph, 0) != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        err = std::string("verify: batch commit capture: ") + cudaGetErrorString(ce);
        return false;
    }
    cudaGraphDestroy(graph);
    return true;
}

bool Verifier::stage_batch(const int* rows, int S, int hbase, const int32_t* tokens, const int64_t* pos,
                           std::string& err) {
    using namespace strata::kernels;
    if (S < 1 || S > max_t_ || hbase < 0 || hbase + S > (int) slots_.size()) {
        err = "verify: batch rows out of range (init_slots)";
        return false;
    }
    for (int t = 0; t < S; ++t) {
        bool dup = false;
        for (int u = 0; u < t; ++u) dup = dup || rows[u] == rows[t];
        if (rows[t] < 0 || rows[t] >= (int) slots_.size() || dup) { err = "verify: a batch row's slot is out of range or twice"; return false; }
    }
    if (released_.load()) { err = "verify: an earlier window never finished on the GPU (#267); restart the engine"; return false; }
    const ModelGeometry& g = *g_;
    for (int t = 0; t < S; ++t)
        if (pos[t] < 0 || pos[t] + 1 > slots_[(size_t) rows[t]]->max_cells) {
            err = "verify: slot " + std::to_string(rows[t]) + " runs past its context";
            return false;
        }
    if (!capture_batch(rows, S, hbase, err) || !capture_commit_batch(rows, S, hbase, err)) return false;
    const Clock::time_point t0 = Clock::now();
    const QsaShapes s = shapes_of(g);
    for (int t = 0; t < S; ++t) {
        h_tok_[t] = tokens[t];
        qsa_step_fill(h_step_ + t * kStepCount, pos[t], s);
        for (int64_t h = 0; h < g.n_head; ++h) h_pos_[t * g.n_head + h] = (int32_t) pos[t];
        int32_t* pk = h_pos_ + (size_t) max_t_ * g.n_head;
        int32_t* pi = pk + (size_t) max_t_ * g.n_head_kv;
        for (int64_t h = 0; h < g.n_head_kv; ++h) pk[t * g.n_head_kv + h] = (int32_t) pos[t];
        for (int64_t h = 0; h < g.idx_q_heads; ++h) pi[t * g.idx_q_heads + h] = (int32_t) pos[t];
    }
    if (ss_->ple.ready() && ple_stage()) {
        uint32_t ple_rows[kVerifyMaxT * PLE_N_HEADS];   // (not `rows`: that is the slots of the window's rows)
        for (int t = 0; t < S; ++t) {
            const SessionState& sx = *slots_[(size_t) rows[t]];
            int32_t prev[2] = {sx.ple_prev[0], sx.ple_prev[1]};
            ngram_rows(&tokens[t], prev, 1, ss_->ple.consts, ple_rows + t * PLE_N_HEADS);
        }
        if (!ss_->ple.table->gather_batch(ple_rows, (size_t) S, h_ple_, err)) return false;
    }
    // the commit's rows (a batch window keeps every row): [1, 0, position, -1 ..] per slot
    const int64_t CB = 2 + max_t_;
    for (int t = 0; t < S; ++t) {
        int32_t* c = h_commitb_ + (size_t) rows[t] * CB;
        c[0] = 1;
        c[1] = 0;
        c[2] = (int32_t) pos[t];
        for (int64_t j = 3; j < CB; ++j) c[j] = -1;
    }
    *(volatile uint32_t*) h_seq_ = 0;
    *(volatile uint32_t*) h_flag_ = 0;
    *(volatile uint32_t*) h_flagA_ = 0;
    *(volatile uint32_t*) h_flagB_ = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    last_t_ = S;
    for (int t = 0; t < S; ++t) last_rows_[t] = rows[t];
    row_base_ = hbase;   // the graphs' key (and nothing else) reads it until the next stage_batch
    last_batch_ = true;
    for (int t = 0; t < S; ++t) { last_tokens_[t] = tokens[t]; last_pos_b_[t] = pos[t]; }
    ms_host += ms_since(t0);
    return true;
}

bool Verifier::run_slots(int S, const int32_t* tokens, const int64_t* pos, PoolMultiFn pool, void* user, int32_t* out,
                         std::string& err) {
    int rows[8] = {};
    for (int t = 0; t < S && t < 8; ++t) rows[t] = t;
    return run_slot_rows(rows, S, tokens, pos, pool, user, out, err);
}

bool Verifier::run_slot_rows(const int* rows, int S, const int32_t* tokens, const int64_t* pos, PoolMultiFn pool,
                             void* user, int32_t* out, std::string& err) {
    using namespace strata::kernels;
    const OnDevice on_device(device_);
    const ModelGeometry& g = *g_;
    if (!stage_batch(rows, S, 0, tokens, pos, err)) return false;
    const cudaError_t le = cudaGraphLaunch(exec_bm_[batch_key(rows, S, 0)], cs_);
    if (le != cudaSuccess) { err = std::string("verify: batch launch: ") + cudaGetErrorString(le); return false; }
    (void) cudaStreamQuery(cs_);
    volatile uint32_t* const seq = h_seq_;
    volatile uint32_t* const flag = h_flag_;
    const int64_t steps = le_ - lb_;
    for (int64_t k = 0; k < steps; ++k) {
        const int64_t l = lb_ + k;
        const uint32_t want = (uint32_t) (k + 1);
        const Clock::time_point a = Clock::now();
        auto last_flush = a;
        uint32_t spins = 0;
        progress_at("verify batch: waiting for the GPU to reach layer", l);
        while (*seq < want) {
            _mm_pause();
            if ((++spins & 1023u) != 0) continue;
            const auto now = Clock::now();
            if (now - last_flush > std::chrono::microseconds(2000)) {
                last_flush = now;
                const cudaError_t q = cudaStreamQuery(cs_);
                if (q != cudaErrorNotReady && *seq < want) {
                    err = "verify batch: layer " + std::to_string(l) + " never rang (" +
                          (q == cudaSuccess ? std::string("graph finished") : std::string(cudaGetErrorString(q))) + ")";
                    return false;
                }
            }
            if (now - a > std::chrono::seconds(20)) {
                err = "verify batch: timed out at layer " + std::to_string(l) + released_note(release_gpu_waits(5000));
                return false;
            }
        }
        const Clock::time_point b = Clock::now();
        cur_layer_ = want - 1;
        set_plan_slot(0);
        progress_at("verify batch: the CPU experts of layer", l);
        if (pool != nullptr) pool(user, h_x_, h_ids_, S, ss_->k, h_ymiss_, l);
        progress_tick();
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
        if (*(volatile uint32_t*) h_flagA_ != want) {        // the pool did not publish a plan: an empty one
            sink_.counts[0] = 0;
            sink_.counts[1] = 0;
            sink_.counts[2] = 0;
            sink_.start[0] = 0;
            sink_.start2[0] = 0;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile uint32_t*) h_flagA_ = want;
            raise_flag(h_flagB_, want);
        }
        *flag = want;
        ms_wait += std::chrono::duration<double, std::milli>(b - a).count();
        ms_pool += ms_since(b);
    }
    const cudaError_t se = cudaStreamSynchronize(cs_);
    if (se != cudaSuccess) { err = std::string("verify batch: ") + cudaGetErrorString(se); return false; }
    cudaStreamSynchronize(copy_);
    if (prof_on_) collect_profile();
    ++windows;
    if (le_ < g.n_layers) return next_ == nullptr || next_->run_slot_rows(rows, S, tokens, pos, pool, next_user_, out, err);
    if (!sample_rows(S, err)) return false;
    for (int t = 0; t < S; ++t) out[t] = ((volatile int32_t*) h_out_)[t];
    progress_at("decode");
    progress_beat();
    return true;
}

bool Verifier::commit_slots(std::string& err) {
    const OnDevice on_device(device_);
    if (!last_batch_ || last_t_ < 1) { err = "verify: commit_slots without a batch window"; return false; }
    const int S = last_t_;
    const Clock::time_point t0 = Clock::now();
    std::atomic_thread_fence(std::memory_order_seq_cst);
    const cudaError_t le = cudaGraphLaunch(commit_bm_[batch_key(last_rows_, S, row_base_)], cs_);
    if (le != cudaSuccess) { err = std::string("verify: batch commit launch: ") + cudaGetErrorString(le); return false; }
    const cudaError_t se = cudaStreamSynchronize(cs_);
    if (se != cudaSuccess) { err = std::string("verify: batch commit: ") + cudaGetErrorString(se); return false; }
    if (ple_stage())
        for (int t = 0; t < S; ++t) {
            SessionState& sx = *slots_[(size_t) last_rows_[t]];
            sx.ple_prev[0] = sx.ple_prev[1];
            sx.ple_prev[1] = last_tokens_[t];
        }
    ms_commit += ms_since(t0);
    return next_ == nullptr || next_->commit_slots(err);
}

bool Verifier::sample_rows(int S, std::string& err) {
    bool any = false;
    for (int t = 0; t < S; ++t) {
        strata::kernels::SamplerParams sp = slot_sp_[(size_t) last_rows_[t]];
        if (sp.greedy || sp.temperature <= 0.0f) continue;
        sp.counter = (uint64_t) last_pos_b_[t];   // Philox(seed, position): the solo window's draw for this position
        sp.penalty_last_n = 0;
        strata::kernels::sample_tokens(head_logits_ + (size_t) t * (size_t) n_vocab_, 1, (int) n_vocab_, nullptr, 0, sp,
                                       m_out_ + t, cs_);
        any = true;
    }
    if (any && cudaStreamSynchronize(cs_) != cudaSuccess) { err = "verify batch: the row sampling failed"; return false; }
    return true;
}

bool Verifier::batch_launch(int base, int S, const int32_t* tokens, const int64_t* pos, std::string& err) {
    const OnDevice on_device(device_);
    if (b_running_) { err = "verify: batch_launch while this stage is busy"; return false; }
    int rows[8] = {};
    for (int t = 0; t < S && t < 8; ++t) rows[t] = base + t;
    if (!stage_batch(rows, S, base, tokens, pos, err)) return false;
    cudaError_t le = cudaGraphLaunch(exec_bm_[batch_key(rows, S, base)], cs_);
    if (le == cudaSuccess) le = cudaGraphLaunch(commit_bm_[batch_key(rows, S, base)], cs_);   // right behind it: every row is kept
    if (le != cudaSuccess) { err = std::string("verify: batch launch: ") + cudaGetErrorString(le); return false; }
    (void) cudaStreamQuery(cs_);
    if (ple_stage())   // the host's side of the commit (the hash's last two tokens)
        for (int t = 0; t < S; ++t) {
            SessionState& sx = *slots_[(size_t) rows[t]];
            sx.ple_prev[0] = sx.ple_prev[1];
            sx.ple_prev[1] = tokens[t];
        }
    b_running_ = true;
    b_k_ = 0;
    b_steps_ = le_ - lb_;
    b_last_ = Clock::now();
    return true;
}

int Verifier::batch_poll(PoolMultiFn pool, void* user, std::string& err) {
    if (!b_running_) return 1;
    const OnDevice on_device(device_);
    volatile uint32_t* const seq = h_seq_;
    const int S = last_t_;
    while (b_k_ < b_steps_) {
        const uint32_t want = (uint32_t) (b_k_ + 1);
        if (*seq < want) {
            const auto now = Clock::now();
            if (now - b_last_ > std::chrono::milliseconds(2)) {
                const cudaError_t q = cudaStreamQuery(cs_);
                if (q != cudaErrorNotReady && *seq < want) {
                    err = "verify batch: layer " + std::to_string(lb_ + b_k_) + " never rang (" +
                          (q == cudaSuccess ? std::string("graph finished") : std::string(cudaGetErrorString(q))) + ")";
                    b_running_ = false;
                    return -1;
                }
                if (now - b_last_ > std::chrono::seconds(20)) {
                    err = "verify batch: timed out at layer " + std::to_string(lb_ + b_k_) + released_note(release_gpu_waits(5000));
                    b_running_ = false;
                    return -1;
                }
            }
            return 0;
        }
        const Clock::time_point b = Clock::now();
        cur_layer_ = want - 1;
        set_plan_slot(0);
        if (pool != nullptr) pool(user, h_x_, h_ids_, S, ss_->k, h_ymiss_, lb_ + b_k_);
        progress_tick();
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
        if (*(volatile uint32_t*) h_flagA_ != want) {        // the pool did not publish a plan: an empty one
            sink_.counts[0] = 0;
            sink_.counts[1] = 0;
            sink_.counts[2] = 0;
            sink_.start[0] = 0;
            sink_.start2[0] = 0;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile uint32_t*) h_flagA_ = want;
            raise_flag(h_flagB_, want);
        }
        *(volatile uint32_t*) h_flag_ = want;
        ms_pool += ms_since(b);
        b_last_ = Clock::now();
        ++b_k_;
    }
    const cudaError_t q = cudaStreamQuery(cs_);
    if (q == cudaErrorNotReady) return 0;
    if (q != cudaSuccess) { err = std::string("verify batch: ") + cudaGetErrorString(q); b_running_ = false; return -1; }
    const cudaError_t qc = cudaStreamQuery(copy_);   // no host function of this window may raise flag B in the next
    if (qc == cudaErrorNotReady) return 0;
    if (prof_on_) collect_profile();
    if (last_stage()) {
        if (!sample_rows(S, err)) { b_running_ = false; return -1; }
        for (int t = 0; t < S; ++t) b_out_[t] = ((volatile int32_t*) h_out_)[t];
    }
    ++windows;
    b_running_ = false;
    progress_beat();
    return 1;
}

bool Verifier::copy_logits(int t, float* host) const {
    if (next_ != nullptr) return next_->copy_logits(t, host);   // a layer split: the head is on the last stage
    if (head_logits_ == nullptr || host == nullptr || t < 0 || n_vocab_ <= 0) return false;
    return cudaMemcpy(host, head_logits_ + (size_t) t * (size_t) n_vocab_, (size_t) n_vocab_ * sizeof(float),
                      cudaMemcpyDeviceToHost) == cudaSuccess;
}

}  // namespace strata::core
