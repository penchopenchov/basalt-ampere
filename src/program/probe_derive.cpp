// src/program/probe_derive.cpp - see probe_derive.hpp.
#include "basalt/program/probe_derive.hpp"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>

namespace basalt::program {

namespace {
std::string fmt(const char* f, ...) {
    char b[256];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(b, sizeof b, f, ap);
    va_end(ap);
    return b;
}
}  // namespace

bool pick_workers(const ProbeRates& rates, int entries, double tolerance, WorkerPick& out) {
    out = WorkerPick{};
    double best = 0;
    for (const auto& r : rates.cpu_expert)
        if (r.entries == entries && r.per_entry_us > 0 && (best == 0 || r.per_entry_us < best)) best = r.per_entry_us;
    if (best <= 0) return false;
    for (const auto& r : rates.cpu_expert) {
        if (r.entries != entries || !(r.per_entry_us > 0) || r.per_entry_us > best * (1.0 + tolerance)) continue;
        if (out.workers == 0 || r.workers < out.workers) {
            out.workers = r.workers;
            out.us_pick = r.per_entry_us;
        }
    }
    out.us_best = best;
    return out.workers > 0;
}

double balance_share(double ms_gpu, double ms_cpu) {
    if (!(ms_gpu > 0) || !(ms_cpu > 0)) return 0.0;
    const double s = ms_gpu / (ms_gpu + ms_cpu);
    return std::min(0.95, std::max(0.05, std::round(s * 20.0) / 20.0));
}

double mean_ms_at(const std::vector<ShareTrial>& trials, double share) {
    double sum = 0;
    int n = 0;
    for (const auto& t : trials)
        if (std::fabs(t.share - share) < 1e-9) { sum += t.ms; ++n; }
    return n ? sum / n : 0.0;
}

double share_gain(const std::vector<ShareTrial>& trials, double base, double candidate) {
    const double a = mean_ms_at(trials, base), b = mean_ms_at(trials, candidate);
    return a > 0 && b > 0 ? a / b - 1.0 : 0.0;
}

std::string suggest_workers(const ProbeRates& rates, const WorkerPick& pick) {
    const int cur = rates.active_workers > 0 ? rates.active_workers : rates.pool_workers;
    if (pick.workers <= 0 || pick.workers >= cur) return "";
    return fmt("--pool-workers %d (CPU experts %.0f us each vs %.0f us best, frees %d cores)", pick.workers, pick.us_pick,
               pick.us_best, cur - pick.workers);
}

std::string suggest_share(double candidate, double base, double gain) {
    if (std::fabs(candidate - base) < 1e-9) return "";
    return fmt("--prefill-cpu-share %.2f (prompt chunk %+.0f%% vs %.2f, live 8k chunks)", candidate, gain * 100.0, base);
}

}  // namespace basalt::program
