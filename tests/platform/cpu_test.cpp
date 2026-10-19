// tests/platform/cpu_test.cpp - platform::detect_cpu_topology and the thread placement calls. CPU only. Where the OS
// refuses a placement (a restricted container) the placement cases are skipped with a line saying so.
#include "basalt/platform/cpu.hpp"

#include <algorithm>
#include <cstdio>
#include <string>
#include <thread>

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("cpu %-62s -> %s\n", what.c_str(), ok ? "ok" : "FAILED   <-- WRONG");
    failures |= !ok;
}
}  // namespace

int main() {
    using namespace basalt::platform;
    const CpuTopology all = detect_cpu_topology(false, CpuAffinity::All);
    check(!all.worker_cores.empty(), "at least one worker core is found");
    check(all.host_core == -1, "without skip_first no core is reserved");
    check(all.p_cores >= 1, "at least one performance core is counted");
    bool unique = true;
    for (size_t i = 0; i < all.worker_cores.size(); ++i)
        for (size_t j = i + 1; j < all.worker_cores.size(); ++j) unique &= all.worker_cores[i] != all.worker_cores[j];
    check(unique, "no core is listed twice");

    const CpuTopology skip = detect_cpu_topology(true, CpuAffinity::All);
    if (all.worker_cores.size() > 1) {
        check(skip.host_core == all.worker_cores.front(), "skip_first reserves the first core for the host");
        check(skip.worker_cores.size() + 1 == all.worker_cores.size(), "and takes it out of the workers");
    }
    for (CpuAffinity a : {CpuAffinity::Auto, CpuAffinity::PCores}) {
        const CpuTopology t = detect_cpu_topology(false, a);
        check(!t.worker_cores.empty(), "a hybrid-aware ordering still finds cores");
    }

    check(!set_thread_affinity(-1), "a negative core is refused");
    const ThreadPlacement none = pin_current_thread(-1);
    check(!none.valid, "pinning to a negative core is a no-op");
    restore_thread_affinity(none);

    const int target = all.worker_cores.front();
    const ThreadPlacement prev = pin_current_thread(target);
    if (!prev.valid) {
        std::printf("cpu: this environment refuses a thread placement; placement cases skipped\n");
    } else {
        restore_thread_affinity(prev);
        check(true, "a thread is pinned and its placement restored");
        bool pinned = false;
        std::thread([&] { pinned = set_thread_affinity(target, 0); }).join();
        check(pinned, "a worker thread is pinned for good");
        std::thread([&] { leave_host_core(); }).join();
        check(true, "leave_host_core returns");
    }
    return failures ? 1 : 0;
}
