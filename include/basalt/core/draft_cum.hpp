// include/basalt/core/draft_cum.hpp - the cumulative draft-length rule.
//
// With a min_p the MTP chain stops after the first draft whose own probability is below min_p.  This rule instead
// keeps draft j while the estimated chance that drafts 0..j are ALL accepted stays at or above the threshold: the
// product over i <= j of acc(p_i), where acc maps a draft probability to the observed acceptance rate of drafts in
// that probability bin.  A middling first draft is cheap to verify and still often right, so it is proposed; a long
// run of merely high probabilities stops once the joint chance decays.  The host (the window size, the draft loop)
// and the device (the conditional draft graph) evaluate the same products in the same order, so they agree exactly.
// Only what is proposed changes: acceptance and outputs stay exact.
#pragma once

#include "basalt/core/diag.hpp"

#include <cmath>

#if defined(__CUDACC__)
#define DRAFT_CUM_HD __host__ __device__
#else
#define DRAFT_CUM_HD
#endif

namespace basalt::core {

/// The acceptance rate of a sampled draft by its probability bin.
DRAFT_CUM_HD inline float draft_cum_acc(float p) {
    return p < 0.30f ? 0.12f : p < 0.50f ? 0.26f : p < 0.70f ? 0.43f : p < 0.80f ? 0.51f
         : p < 0.90f ? 0.59f : p < 0.95f ? 0.66f : p < 0.99f ? 0.79f : 0.96f;
}

/// Whether draft j stays in the window: with the rule on, prod_{i <= j} acc(probs[i]) >= th; else probs[j] >= min_p.
DRAFT_CUM_HD inline bool draft_keep(const float* probs, int j, float min_p, float th) {
    if (!(th > 0.0f)) return probs[j] >= min_p;
    float c = 1.0f;
    for (int i = 0; i <= j; ++i) c = c * draft_cum_acc(probs[i]);
    return c >= th;
}

/// The rule's threshold (--exp spk_th=V; 0 switches the rule off, the min_p rule), read once.
inline float draft_cum_env() {
    static const float th = (float) exp_flag("spk_th", 0.35);
    return th;
}

/// Per-request threshold (--exp spk2_adapt=1, read once).  The threshold falls from the base toward kLo as the
/// request's drafts are accepted more often than the bin table predicts: when they run well, longer windows pay.  The
/// observed and expected accepted counts decay per window; the ratio sets the threshold along a linear ramp.  The
/// window's value is fixed before its drafts are made and shared by the host rule and the device graph.
struct DraftCumAdapt {
    static bool enabled() {
        static const bool on = exp_flag("spk2_adapt", 0) != 0;
        return on;
    }
    float base = draft_cum_env();   ///< the threshold while nothing is known (and with the gate off)
    double obs = 0.0, exp = 0.0;
    float cur = base;

    float th() const { return cur; }

    /// One verified window of the MTP chain: `n` drafts were offered with probabilities probs[0..n), `acc` accepted.
    void observe(const float* probs, int n, int acc) {
        if (!enabled() || !(base > 0.0f) || n <= 0) return;
        constexpr double kDecay = 0.97, kRlo = 0.9, kRhi = 1.1;
        constexpr float kLo = 0.25f;
        double c = 1.0, e = 0.0;
        for (int j = 0; j < n; ++j) { c *= (double) draft_cum_acc(probs[j]); e += c; }
        obs = obs * kDecay + (double) acc;
        exp = exp * kDecay + e;
        if (exp < 0.5) return;
        const double t = std::fmin(std::fmax((obs / exp - kRlo) / (kRhi - kRlo), 0.0), 1.0);
        cur = base + (float) t * (std::fmin(kLo, base) - base);
    }
};

}  // namespace basalt::core
