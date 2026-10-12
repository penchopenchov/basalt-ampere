// include/basalt/kernels/attn_chain.hpp - a GDN layer's attention half as ONE persistent kernel.
//
// hc read -> qkv + z projections -> recurrence -> output projection -> hc read with the router and top-10, the same
// operations (and bits) as hc_fused_read, dense_tc_pair, dec_gdn_rec, dense_tc and hc_fused_read in sequence.  One
// CTA per SM; the phases hand over through counters in global memory and the next phase's weights are copied into
// shared memory while the previous one finishes.  The kernel needs every CTA resident, so nothing may hold an SM
// while it waits (the verify window launches it with the stream drained).
#pragma once

#include "basalt/kernels/dec_gdn.hpp"
#include "basalt/kernels/hc_fused.hpp"

namespace basalt::kernels {

constexpr int kAttnChainMaxT = 4;

struct AttnChainArgs {
    int T = 0;                       ///< window rows, 1..kAttnChainMaxT
    HcFusedArgs h1;                  ///< the layer's first read; xq = its q8_1 rows (the projections' input)
    HcFusedArgs h2;                  ///< the second read, with the router (w_router, logits, ids, weights)
    const void* wq = nullptr;        ///< qkv projection, native blocks
    const void* wz = nullptr;        ///< z projection
    const void* wo = nullptr;        ///< output projection
    int type_q = 0, type_z = 0, type_o = 0;
    DecRecArgs rec;                  ///< rec.qkv / rec.z are the projections' outputs
    int ord = 0;                     ///< launch ordinal within the graph (0 = the first: it zeroes the hand-off counters first)
};

/// Allocate the kernel's scratch on the current device (outside a stream capture, once).
void attn_chain_init();
/// The current device and these arguments run the kernel (device, row count, projection types, alignment).
bool attn_chain_supported(const AttnChainArgs& a);
/// Enqueue the layer's attention half on `stream`.  Aborts on invalid arguments.
void attn_chain(const AttnChainArgs& a, void* stream);

}  // namespace basalt::kernels
