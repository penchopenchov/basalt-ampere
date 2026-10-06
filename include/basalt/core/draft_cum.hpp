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

#if defined(__CUDACC__)
#define STRATA_DRAFT_CUM_HD __host__ __device__
#else
#define STRATA_DRAFT_CUM_HD
#endif

namespace strata::core {

/// The acceptance rate of a sampled draft by its probability bin.
STRATA_DRAFT_CUM_HD inline float draft_cum_acc(float p) {
    return p < 0.30f ? 0.12f : p < 0.50f ? 0.26f : p < 0.70f ? 0.43f : p < 0.80f ? 0.51f
         : p < 0.90f ? 0.59f : p < 0.95f ? 0.66f : p < 0.99f ? 0.79f : 0.96f;
}

/// Whether draft j stays in the window: with the rule on, prod_{i <= j} acc(probs[i]) >= th; else probs[j] >= min_p.
STRATA_DRAFT_CUM_HD inline bool draft_keep(const float* probs, int j, float min_p, float th) {
    if (!(th > 0.0f)) return probs[j] >= min_p;
    float c = 1.0f;
    for (int i = 0; i <= j; ++i) c = c * draft_cum_acc(probs[i]);
    return c >= th;
}

/// The rule's threshold (0 would switch it off, the min_p rule).
inline constexpr float draft_cum_env() { return 0.35f; }

}  // namespace strata::core
