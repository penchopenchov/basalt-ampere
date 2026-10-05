// include/strata/kernels/hc_fused.hpp - the hyper-connection read of a verify window as ONE kernel (sm_120).
//
// The same outputs as fused_gr_read_multi (and, optionally, the router after it), bit for bit:
//
//   R' = R + bo_prev * 2 sigmoid(inj_prev / hc)                  (when apply)
//   rs[c] = rsqrt(mean(R'[c]^2) + eps),  xn = R' * w_norm * rs
//   lo[k] = silu((w_down[k] . xn) / hc),  inject[c] = w_inject[c] . xn
//   mixed[d] = mean_c xn[c,d] * sigmoid(w_up[c*n_embd + d] . lo),  R <- R'
//   logits = w_router . mixed,  (ids, weights) = top-10 of softmax(logits)   (when w_router)
//
// One CTA per SM; the CTAs split into roles and hand data on through counters in global memory instead of kernel
// boundaries: the first 4T CTAs normalize (token, stream) pairs, 81 CTAs run the down projection (four rows each,
// weights and activations streamed through a shared-memory ring), the other CTAs hold their slice of w_up in shared
// memory (bulk copies issued at launch) and run the up projection, then every CTA takes a few router rows and the
// last one to finish runs the top-10.  Every value is computed with the operations, and in the order, of the kernels
// it replaces (fused_gr.cu's read, the BF16 MMVF router GEMV and the native top-10), so the outputs are the same bits.
//
// `pf` / `pf_bytes`: up to four ranges (e.g. the next read's weights) prefetched into L2 once this read is done.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

struct HcFusedArgs {
    int T = 0;                          ///< tokens, 1..8; token t's rows at the fixed strides below
    const float* R = nullptr;           ///< [T][hc * n_embd]
    float* R_out = nullptr;             ///< == R for the in-place update
    bool apply = false;                 ///< fold the previous half's write
    const float* bo_prev = nullptr;     ///< [T][n_embd]
    const float* inj_prev = nullptr;    ///< [T][hc]
    const float* w_norm = nullptr;      ///< [hc * n_embd] f32
    const uint16_t* w_down = nullptr;   ///< bf16 [hc_lr][hc * n_embd]
    const uint16_t* w_up = nullptr;     ///< bf16 [hc * n_embd][hc_lr]
    const uint16_t* w_inject = nullptr; ///< bf16 [hc][hc * n_embd], or null
    float eps = 1e-6f;
    float* lo = nullptr;                ///< [T][hc_lr]
    float* rs = nullptr;                ///< [T][hc]
    float* inject_out = nullptr;        ///< [T][hc] (when w_inject)
    float* mixed = nullptr;             ///< [T][n_embd]
    float* xn = nullptr;                ///< scratch [T][hc * n_embd]
    const uint16_t* w_router = nullptr; ///< bf16 [512][n_embd], or null (no router)
    float* logits = nullptr;            ///< [T][512]
    int32_t* ids = nullptr;             ///< [T][10]
    float* weights = nullptr;           ///< [T][10]
    const void* pf[4] = {nullptr, nullptr, nullptr, nullptr};
    size_t pf_bytes[4] = {0, 0, 0, 0};
};

/// The current device runs the fused read (sm_120 with enough SMs for the role split).
bool hc_fused_supported();
/// Enqueue the read on `stream` (a PdlScope applies).  Aborts on invalid arguments.
void hc_fused_read(const HcFusedArgs& a, void* stream);
/// Prefetch byte ranges into L2 with a one-CTA kernel (bulk prefetches; no data moves to the SMs).
void hc_l2_prefetch(const void* const* ptrs, const size_t* bytes, int n, void* stream);
/// Tuning knobs for the bench (-1 = leave): up_early = issue w_up's bulk copies at launch (1) or after the norm (0).
void hc_fused_set_knobs(int up_early);
/// The bench only: per-CTA phase times (globaltimer ns) into dbg[nb][8]: start, after the dependency wait, role input
/// ready, role done, router input ready, router rows done, top-10 done (last CTA), exit.  nullptr = off.
void hc_fused_set_debug(unsigned long long* dbg);

}  // namespace strata::kernels
