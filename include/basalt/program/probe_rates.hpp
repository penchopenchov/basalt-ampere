// include/basalt/program/probe_rates.hpp - the engine's `PROBE rates` serve command: it measures this machine's
// transfer and compute rates inside the loaded engine and prints one result block.
//
// It runs between requests and leaves the engine as it found it: no tier, cache, KV or RNG is touched, the CPU pool's
// active worker count is restored, and every buffer it allocates is freed before it returns.
#pragma once

#include "basalt/core/expert_source.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/kernels/cpu/pool.hpp"

#include <cstdio>
#include <string>

namespace basalt::program {

struct ProbeEnv {
    kernels::cpu::ExpertPool* pool = nullptr;   ///< the engine's CPU pool (idle between requests)
    core::ExpertSource* src = nullptr;          ///< where the RAM-resident expert blobs are read from
    core::PeerExperts* peer = nullptr;          ///< the peer tier, or null
    int64_t n_layers = 0, n_expert = 0;
    int top_k = 10;                             ///< routed experts per token
};

/// Measures hardware facts, each PCIe link both ways, GPU memory bandwidth, host memory bandwidth by CPU worker
/// count, the CPU expert rate, the peer's round trip and per-entry cost, and the copy costs into each tier.  Writes
/// the block to `out` as lines `PROBE <section> key=value ...` from `PROBE begin rates` to `PROBE end ms=<wall>`.
/// False (with `err`) only when nothing could be measured; a step that cannot run says so in the block.
bool probe_rates(const ProbeEnv& env, std::FILE* out, std::string& err);

}  // namespace basalt::program
