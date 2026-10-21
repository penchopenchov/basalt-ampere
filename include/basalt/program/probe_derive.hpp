// include/basalt/program/probe_derive.hpp - values basalt-probe computes from what it measured and suggests when they
// beat the engine's built-in defaults on this machine. Pure functions, unit tested; the probe prints, it applies nothing.
#pragma once

#include "basalt/program/probe_report.hpp"

#include <string>
#include <vector>

namespace basalt::program {

/// The smallest worker count whose CPU expert time (entries per call = `entries`) is within `tolerance` of the best
/// measured count. False when the rates carry no such rows. `us_best` and `us_pick` are the per-entry times.
struct WorkerPick {
    int workers = 0;
    double us_best = 0, us_pick = 0;
};
bool pick_workers(const ProbeRates& rates, int entries, double tolerance, WorkerPick& out);

/// The prefill CPU share that balances the CPU and the GPU tier when both take part of a streamed prompt chunk:
/// `ms_gpu` is the chunk time at share 0, `ms_cpu` at share 1. Rounded to 0.05, within 0.05..0.95.
double balance_share(double ms_gpu, double ms_cpu);

/// One timed prompt chunk.
struct ShareTrial {
    double share = 0;
    double ms = 0;
};

/// Mean chunk time of the trials at `share`; 0 when there are none.
double mean_ms_at(const std::vector<ShareTrial>& trials, double share);

/// Speed gain of `candidate` over `base` (mean chunk times), as a fraction; 0 when either is missing.
double share_gain(const std::vector<ShareTrial>& trials, double base, double candidate);

/// The suggestion lines for the report. `workers` is empty when the pick is the pool's current count; the share one is
/// empty unless the candidate beat the default by more than `min_gain` in the live chunks.
std::string suggest_workers(const ProbeRates& rates, const WorkerPick& pick);
std::string suggest_share(double candidate, double base, double gain);

}  // namespace basalt::program
