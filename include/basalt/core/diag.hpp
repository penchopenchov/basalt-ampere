// include/basalt/core/diag.hpp - the engine's diagnostic switches, set from the command line before any work starts.
#pragma once

#include <string>

namespace strata::core {

struct Diag {
    bool profile = false;      ///< --profile: per-stage GPU stamps of the verify window (turns PDL off; not for A/B)
    bool wait_stats = false;   ///< --wait-stats: the MoE tail's wait counters, one report per 200 windows
    bool decode_timing = false;   ///< --decode-timing: the per-request decode timing line
    std::string draft_log;     ///< --draft-log FILE: one line per verify window (width, accepted, ms)
};

inline Diag& diagnostics() {
    static Diag d;
    return d;
}

}  // namespace strata::core
