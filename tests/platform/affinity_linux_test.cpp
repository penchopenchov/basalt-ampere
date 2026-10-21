#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "basalt/kernels/cpu/pool.hpp"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "affinity_linux.hpp"

#include <pthread.h>
#include <sched.h>
#include <unistd.h>

namespace cpu = basalt::kernels::cpu;

class DynamicCpuSet {
public:
    explicit DynamicCpuSet(size_t count)
        : count_(count), bytes_(CPU_ALLOC_SIZE(count)), set_(CPU_ALLOC(count)) {
        if (set_) CPU_ZERO_S(bytes_, set_);
    }

    ~DynamicCpuSet() {
        if (set_) CPU_FREE(set_);
    }

    DynamicCpuSet(const DynamicCpuSet&) = delete;
    DynamicCpuSet& operator=(const DynamicCpuSet&) = delete;

    cpu_set_t* get() { return set_; }
    const cpu_set_t* get() const { return set_; }
    size_t bytes() const { return bytes_; }
    size_t count() const { return count_; }
    explicit operator bool() const { return set_ != nullptr; }

private:
    size_t count_;
    size_t bytes_;
    cpu_set_t* set_;
};

static size_t initial_cpu_count() {
    const long configured = sysconf(_SC_NPROCESSORS_CONF);
    return (std::max)(size_t(128), configured > 0 ? (size_t) configured : size_t(0));
}

// Query with CPU_ALLOC rather than cpu_set_t: this must observe CPUs beyond CPU_SETSIZE too.
static bool get_affinity(std::vector<int>& ids) {
    size_t count = initial_cpu_count();
    for (int attempt = 0; attempt < 8; ++attempt) {
        DynamicCpuSet set(count);
        if (!set) return false;
        const int error = pthread_getaffinity_np(pthread_self(), set.bytes(), set.get());
        if (error == 0) {
            ids.clear();
            for (size_t i = 0; i < set.count(); ++i)
                if (CPU_ISSET_S(i, set.bytes(), set.get())) ids.push_back((int) i);
            return !ids.empty();
        }
        if (error != EINVAL || count > (size_t(1) << 20)) {
            std::fprintf(stderr, "pthread_getaffinity_np failed: %s\n", std::strerror(error));
            return false;
        }
        count *= 2;
    }
    std::fprintf(stderr, "pthread_getaffinity_np still needs a larger CPU set\n");
    return false;
}

static bool set_affinity(const std::vector<int>& ids) {
    if (ids.empty()) return false;
    const int highest = *std::max_element(ids.begin(), ids.end());
    if (highest < 0) return false;
    const size_t count = (std::max)(initial_cpu_count(), (size_t) highest + 1);
    DynamicCpuSet set(count);
    if (!set) return false;
    for (int id : ids) {
        if (id < 0 || (size_t) id >= count) return false;
        CPU_SET_S((size_t) id, set.bytes(), set.get());
    }
    const int error = pthread_setaffinity_np(pthread_self(), set.bytes(), set.get());
    if (error != 0) {
        std::fprintf(stderr, "pthread_setaffinity_np failed: %s\n", std::strerror(error));
        return false;
    }
    return true;
}

class OriginalAffinity {
public:
    bool capture() { return get_affinity(ids_); }
    const std::vector<int>& ids() const { return ids_; }
    bool restore() { return ids_.empty() || set_affinity(ids_); }
    ~OriginalAffinity() { if (!ids_.empty()) set_affinity(ids_); }

private:
    std::vector<int> ids_;
};

static bool expect_affinity(const std::vector<int>& expected, const char* label) {
    std::vector<int> actual;
    if (!get_affinity(actual)) {
        std::fprintf(stderr, "affinity test: %s: cannot query current affinity\n", label);
        return false;
    }
    if (actual != expected) {
        std::fprintf(stderr, "affinity test: %s: expected %zu CPUs, observed %zu\n",
                     label, expected.size(), actual.size());
        return false;
    }
    return true;
}

static bool topology_respects_current_mask() {
    std::vector<int> allowed;
    if (!get_affinity(allowed)) return false;
    const auto topology = cpu::detect_cpu_topology(false);
    if (topology.worker_cores.empty()) return false;
    for (int core : topology.worker_cores) {
        if (!std::binary_search(allowed.begin(), allowed.end(), core)) {
            std::fprintf(stderr, "affinity test: topology selected disallowed CPU %d\n", core);
            return false;
        }
    }
    return true;
}

static bool topology_under_masks(const std::vector<int>& allowed, bool require_high) {
    if (!topology_respects_current_mask()) return false;
    if (allowed.size() >= 2) {
        const std::vector<int> sparse{allowed.front(), allowed.back()};
        if (!set_affinity(sparse) || !expect_affinity(sparse, "establish sparse mask") ||
            !topology_respects_current_mask()) return false;
        if (!set_affinity(allowed) || !expect_affinity(allowed, "restore full mask")) return false;
    } else {
        std::puts("SKIP sparse mask: fewer than two allowed CPUs");
    }
    const auto it = std::find_if(allowed.begin(), allowed.end(), [](int id) { return id >= 64; });
    if (it == allowed.end()) {
        if (require_high) {
            std::fprintf(stderr, "affinity test: --require-high-cpu requested, but no allowed CPU ID is >= 64\n");
            return false;
        }
        std::puts("SKIP high-CPU mask: no allowed CPU ID is >= 64 (use --require-high-cpu to require it)");
        return true;
    }
    const std::vector<int> high_only{*it};
    if (!set_affinity(high_only) || !expect_affinity(high_only, "establish high-only mask") ||
        !topology_respects_current_mask()) return false;
    return set_affinity(allowed) && expect_affinity(allowed, "restore full mask after high-only");
}

int main(int argc, char** argv) {
    bool require_high = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--require-high-cpu") == 0) require_high = true;
        else {
            std::fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            return 2;
        }
    }

    OriginalAffinity original;
    if (!original.capture()) {
        std::fprintf(stderr, "affinity test: cannot capture original thread affinity\n");
        return 1;
    }
    const auto allowed = original.ids();

    // The pool takes its size from the caller as given (the sizing rules live in program/auto_plan).
    {
        cpu::ExpertPool three(3, cpu::PoolAffinity::All);
        cpu::ExpertPool none(0, cpu::PoolAffinity::Auto);
        if (three.workers() != 3 || none.workers() != 1) {
            std::fprintf(stderr, "affinity test: pool sizes %d / %d, expected 3 / 1\n", three.workers(), none.workers());
            return 1;
        }
    }

    const bool ok = topology_under_masks(allowed, require_high);
    const bool restored = original.restore() && expect_affinity(allowed, "final original affinity cleanup");
    if (!ok || !restored) return 1;

    std::printf("affinity test: topology under full, sparse and high-CPU masks, and cleanup checks passed (%zu allowed CPUs)\n",
                allowed.size());
    return 0;
}

