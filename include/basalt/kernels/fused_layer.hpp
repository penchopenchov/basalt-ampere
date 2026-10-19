// include/basalt/kernels/fused_layer.hpp - a GDN layer (attention half + routed/shared experts + combine) as ONE persistent kernel.
//
// The attention half is the one in attn_chain.hpp; the experts half (plan from the kernel's own top-10, gate/up + SwiGLU, down,
// shared expert, combine with the peer and CPU tiers' rows) replaces dec_moe_glue, shared_expert_multi, dec_moe_wait and
// dec_moe_combine for the layer.  The kernel also publishes the tier ring (header, mapped rows, doorbells) and waits for the
// tiers itself.
#pragma once

#include <string>

#include "basalt/kernels/attn_chain.hpp"
#include "basalt/kernels/dec_moe.hpp"

namespace basalt::kernels {

constexpr int kFusedMaxLaunches = 48;

struct FusedLayerArgs {
    AttnChainArgs chain;             ///< h2.ids / h2.weights / h2.xq / h2.mixed are the experts half's inputs
    int layer = 0;                   ///< model layer (ring header)
    int gu_type = 0, d_type = 0;     ///< routed experts' gate/up and down types
    const unsigned long long* ptab = nullptr;   ///< [512] CUDA0 blob address or 0
    const int32_t* res1 = nullptr;              ///< [512] peer slot or -1 (null: no peer tier)
    const void *sg = nullptr, *su = nullptr, *sd = nullptr;   ///< shared expert gate / up / down, native blocks
    const uint16_t* sgi = nullptr;              ///< shared expert's BF16 gate row
    int sgt = 0, sut = 0, sdt = 0;
    float* bo = nullptr;                        ///< combine output [T][2560]
    // tiers (mapped memory): rows, done words and the ring publication
    const float* peer_rows = nullptr; const float* cpu_rows = nullptr;
    const unsigned* peer_done = nullptr; const unsigned* cpu_done = nullptr;
    uint32_t ring = 0;
    DmRing* ring_m = nullptr; float* x_m = nullptr; void* xq_m = nullptr;
    uint32_t* host_seq_m = nullptr; uint32_t* peer_seq_m = nullptr; volatile uint32_t* abort_m = nullptr;
    int ord = 0;                     ///< launch ordinal within the graph (0 zeroes the counters)
};

/// Allocate the kernel's scratch (outside a stream capture, once).  False when the device cannot run it; the reason is printed.
bool fused_layer_init();
bool fused_layer_supported(const FusedLayerArgs& a);
// whether the experts half has kernels for a layer's gate/up and down expert formats
bool fused_layer_experts_ok(int gu_type, int d_type);
/// Enqueue the layer.  Exits on invalid arguments.
void fused_layer(const FusedLayerArgs& a, void* stream);
/// The kernel's recorded wait failures, one line (empty when none); for the engine's error path.
std::string fused_layer_report();

}  // namespace basalt::kernels
