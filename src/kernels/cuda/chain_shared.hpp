// src/kernels/cuda/chain_shared.hpp - what attn_chain.cu hands to fused_layer.cu, which is its own translation unit: the
// unfused chain's per-device scratch (the fused layer launches share its hand-off counters) and its host-side launch
// preparation.  Both translation units compile the same device code, so the launch record has one layout.
#pragma once
#include <cstddef>
#include <cstdint>

#include "basalt/kernels/hc_fused.hpp"

namespace basalt::kernels {

struct ChainShared {
    bool ok = false;
    int sms = 0;                 // SMs of the current device
    int nctr = 0;                // hand-off counters per set
    size_t hc_launch_bytes = 0;  // sizeof the hc read's launch record
    unsigned* cnt = nullptr;
    float *cpr = nullptr, *mxr = nullptr, *lgr = nullptr;
    uint8_t *xqr = nullptr, *xqn = nullptr;
    int32_t* ids_all = nullptr;
    float* w_all = nullptr;
    float* abv = nullptr;
};

// The current device's chain scratch (attn_chain_init first).
ChainShared chain_shared();
// hc_prepare into the launch record `L` (sizeof == ChainShared::hc_launch_bytes).
bool chain_hc_prepare(const HcFusedArgs& a, void* L);

}  // namespace basalt::kernels
