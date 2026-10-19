// include/basalt/kernels/mid_chain.hpp - a GDN layer's middle as ONE persistent kernel.
//
// qkv + z projections -> recurrence -> output projection, the same operations (and bits) as dense_tc_pair,
// dec_gdn_rec and dense_tc in sequence.  It runs between the layer's two hc reads (which stay hc_fused_read): its input
// is the first read's q8_1 rows and mixed output, its result the output projection's `bo`, which the second read adds.
// One CTA per SM; the phases hand over through counters in global memory.  The kernel needs every CTA resident, so it
// must be launched with the stream drained of kernels that wait on later work (it is a programmatic dependent of the
// first read and the second read is its dependent).
#pragma once

#include "basalt/kernels/dec_gdn.hpp"

namespace basalt::kernels {

constexpr int kMidChainMaxT = 4;

struct MidChainArgs {
    int T = 0;                       ///< window rows, 1..kMidChainMaxT
    const void* xq = nullptr;        ///< the first read's q8_1 rows: row t at xq + t * (n_embd / 32) * 36
    const void* wq = nullptr;        ///< qkv projection, native blocks
    const void* wz = nullptr;        ///< z projection
    const void* wo = nullptr;        ///< output projection
    int type_q = 0, type_z = 0, type_o = 0;
    DecRecArgs rec;                  ///< rec.qkv / rec.z are the projections' outputs; rec.yq may alias xq
    float* bo = nullptr;             ///< the output projection's result, row t at bo + t * 2560
    int ord = 0;                     ///< launch ordinal within the graph (0 = the first: it zeroes the hand-off counters first)
    bool pair_only = false;          ///< bench: stop after the projection pair
    unsigned long long* stamps = nullptr;   ///< bench: [ord][kMidStampCtas][kMidStampPts] phase times (null: no stamping)
};
constexpr int kMidStampCtas = 170, kMidStampPts = 24;

/// Allocate the kernel's scratch on the current device (outside a stream capture, once).
void mid_chain_init();
/// The current device and these arguments run the kernel (device, row count, projection types, alignment).
bool mid_chain_supported(const MidChainArgs& a);
/// Enqueue the layer's middle on `stream`.  Aborts on invalid arguments.
void mid_chain(const MidChainArgs& a, void* stream);

}  // namespace basalt::kernels
