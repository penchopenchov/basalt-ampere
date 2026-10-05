// include/strata/kernels/dec_moe.hpp - the verify window's MoE tail without host handshakes (STRATA_DEC_MOE=1).
//
// Per layer: one route kernel on CUDA0 sorts the routed entries into its three tiers from device residency tables
// (CUDA0's VRAM tier, the peer card, the CPU pool), builds CUDA0's grouped plan, quantizes the FFN input, publishes
// what the other tiers need to mapped host memory and rings both. The peer card runs one persistent kernel per
// window that polls those rings, computes its entries and writes their rows back to mapped memory; the CPU pool is
// served by the host loop as before. CUDA0 computes its own entries right away and waits on a tier only when the
// layer has entries there; the combine reads every row from where its tier left it.
//
// Every row is computed by the same arithmetic as the grouped kernels it replaces (native_expert_grouped), so a
// tier gives the same bits as before; only the PCIe share is gone (those entries go to the CPU pool).
#pragma once

#include "strata/kernels/iq_kernels.hpp"

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

constexpr int kDmCap = 128;          ///< entries per window: kVerifyMaxT tokens x top-10 fit

/// One ring (layer) as the route kernel publishes it to mapped memory.
struct DmRing {
    int32_t n_tok = 0, layer = 0, n_vram = 0, n_peer = 0, n_cpu = 0, pad[3] = {};
    int32_t ids[kDmCap] = {};
    int32_t kind[kDmCap] = {};       ///< 0: CUDA0's VRAM tier, 2: the peer card, -1: the CPU pool
};

/// CUDA0's grouped plan (native_expert_grouped's lists) and every entry's tier, on the device.
struct DmPlan {
    int32_t counts[4];               ///< [0] VRAM groups, [1] VRAM entries, [2] peer entries, [3] CPU entries
    int32_t start[kDmCap + 1];
    int32_t dst[kDmCap];
    int32_t tok[kDmCap];
    int32_t kind[kDmCap];
    unsigned long long ptr[kDmCap];
};

struct DmRouteArgs {
    const int32_t* ids = nullptr;    ///< device: n_tok * k routed experts
    const float* x = nullptr;        ///< device: n_tok * n_embd FFN input
    int n_tok = 0, k = 0, n_embd = 0, n_expert = 0, layer = 0;
    const int32_t* res0 = nullptr;   ///< this layer's CUDA0 residency row (slot or -1)
    const int32_t* res1 = nullptr;   ///< this layer's peer residency row (slot or -1); null: no peer
    const uint8_t* cache_base = nullptr;
    const unsigned long long* slot_off = nullptr;   ///< per-slot offsets (null: slot * blob)
    long long blob = 0;
    DmPlan* plan = nullptr;          ///< device
    void* xq = nullptr;              ///< device: n_tok q8_1 rows (block_q8_1, n_embd / 32 per row)
    DmRing* ring_m = nullptr;        ///< mapped: this ring's header
    float* x_m = nullptr;            ///< mapped: this ring's FFN input (written when the CPU has entries)
    void* xq_m = nullptr;            ///< mapped: this ring's q8_1 rows (written when the peer has entries)
    uint32_t* host_seq_m = nullptr;  ///< mapped doorbells: the ring is stored once everything above has landed
    uint32_t* peer_seq_m = nullptr;
    uint32_t ring = 0;
    unsigned int* ctr = nullptr;     ///< device: arrival counter, monotonic across launches
    unsigned int ctr_base = 0;       ///< arrivals before this launch (ring - 1) * blocks
};
/// Blocks dec_moe_route launches for `n_tok` rows (the arrival counter advances by this much).
int dec_moe_route_blocks(int n_tok, int n_embd);
void dec_moe_route(const DmRouteArgs& a, void* stream);

/// The peer card's window: one persistent kernel serves `steps` rings.
struct DmPeerArgs {
    const DmRing* rings_m = nullptr;             ///< mapped: ring s at rings_m[s]
    const uint8_t* xq_m = nullptr;               ///< mapped: ring s's q8_1 rows at xq_m + s * xq_stride
    size_t xq_stride = 0;
    const volatile uint32_t* seq_m = nullptr;    ///< mapped: low 16 bits the last ring CUDA0 published, bits 16-23 its
                                                 ///< tokens, bit 28 set when it has peer entries; UINT32_MAX: stop
    uint32_t* done_m = nullptr;                  ///< mapped: the last ring whose rows have landed
    float* rows_m = nullptr;                     ///< mapped: entry i's row at rows_m + i * n_embd
    const unsigned long long* ptr = nullptr;     ///< peer device: [n_layers * n_expert] blob address or 0
    const NativeExpertLayout* lay = nullptr;     ///< peer device: per layer
    int n_expert = 0, k = 0, n_embd = 0, n_ff = 0, steps = 0;
    void* scratch = nullptr;                     ///< peer device: dec_moe_peer_scratch_bytes
    unsigned int* arrive = nullptr;              ///< peer device: monotonic arrival counter
    unsigned long long* stamps_m = nullptr;      ///< tests: block 0's phase times per ring (8 each), or null
    int gu2 = 0;                                 ///< the grid formats' gate/up two rows a warp (STRATA_DEC_GU2 sets it)
};
size_t dec_moe_peer_scratch_bytes(int n_embd, int n_ff);
/// Blocks the peer kernel runs with on the current device (all co-resident); 0 when it cannot run there.
int dec_moe_peer_blocks();
/// Whether the persistent kernels cover this layer's expert formats.
bool dec_moe_formats_ok(int gu_type, int d_type, int64_t n_embd, int64_t n_ff);
/// Launches the peer window (cooperative) on `stream` of the current device; false on a launch error.
bool dec_moe_peer_launch(const DmPeerArgs& a, int blocks, void* stream);

/// CUDA0's spin until the tiers this layer uses have landed (the combine waits by itself; this one is for profiles).
void dec_moe_wait(const DmPlan* plan, const volatile uint32_t* peer_done_m, const volatile uint32_t* host_flag_m,
                  uint32_t ring, void* stream);

/// CUDA0's VRAM share of the plan in one launch (gate/up, SwiGLU, down; native_expert_grouped's rows bit for bit):
/// entry e's row into row plan->dst[e] of `out`.  `sync`: kDmCap + 1 zeroed device words, left zeroed.
size_t dec_moe_vram_scratch_bytes(int n_ff);
/// kind 0: one launch (co-resident blocks, group counters); 1: the same with the weight rows through bulk copies into
/// shared memory; 2 / 3: gate/up with 1 / 2 rows a warp and each lane's weight parts loaded first, then
/// native_expert_grouped's SwiGLU and down launches over `cap` possible groups (the window's entries).
void dec_moe_vram(const NativeExpertLayout& L, const DmPlan* plan, const void* xq, void* scratch, float* out,
                  unsigned int* sync, void* stream, int kind = 0, int cap = kDmCap);

/// The tests: the persistent kernel's per-group math against native_expert_grouped on one device, without rings.
/// Entries [start[g], start[g+1]) of group g read q8_1 row tok[e] and write row dst[e] of `out` (device).
void dec_moe_peer_math_test(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* start,
                            int n_groups, const int32_t* tok, const int32_t* dst, const void* xq, void* scratch,
                            float* out, void* stream);

/// The layer's FFN output: entry i's row from its tier (CUDA0's `parts`, the peer's or the CPU pool's mapped rows,
/// entry i at row i of each), summed with the router weights in native_moe_combine_multi's order, plus `shared`.
/// With peer_done_m, every block first waits for the tiers the plan uses (peer_done_m / host_flag_m >= ring); null:
/// dec_moe_wait ran before it.  Built with native_moe.cu's math.
void dec_moe_combine(const float* parts, const float* peer_rows_m, const float* cpu_rows_m, const DmPlan* plan,
                     const float* weights, const float* shared, float* output, int64_t n_embd, int64_t k, int n_tok,
                     const volatile uint32_t* peer_done_m, const volatile uint32_t* host_flag_m, uint32_t ring,
                     void* stream);

}  // namespace strata::kernels
