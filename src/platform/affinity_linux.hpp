#pragma once

#include <pthread.h>
#include <sched.h>

#include <cerrno>
#include <climits>
#include <cstring>
#include <vector>

namespace basalt::platform::detail {

// CPU_ALLOC and the sized macros also handle kernels whose CPU mask exceeds CPU_SETSIZE.
// Keep the native mask opaque: copy its bytes for storage and use CPU_*_S to inspect it.
class LinuxCpuSet {
public:
    explicit LinuxCpuSet(int cpus) : bytes(CPU_ALLOC_SIZE(cpus)), set(CPU_ALLOC(cpus)) {
        if (set) CPU_ZERO_S(bytes, set);
    }
    ~LinuxCpuSet() { CPU_FREE(set); }
    LinuxCpuSet(const LinuxCpuSet&) = delete;
    LinuxCpuSet& operator=(const LinuxCpuSet&) = delete;

    const size_t bytes;
    cpu_set_t* const set;
};

// pthread affinity functions return an error number directly; they do not set errno.
// A too-small query buffer returns EINVAL, even when the number of online CPUs is small.
inline int get_thread_affinity(std::vector<unsigned long>& mask, std::vector<int>* cpus = nullptr) {
    for (int count = CPU_SETSIZE;;) {
        LinuxCpuSet current(count);
        if (!current.set) return ENOMEM;
        const int error = pthread_getaffinity_np(pthread_self(), current.bytes, current.set);
        if (error == 0) {
            mask.resize(current.bytes / sizeof(unsigned long));
            std::memcpy(mask.data(), current.set, current.bytes);
            if (cpus) {
                cpus->clear();
                for (int cpu = 0; cpu < count; ++cpu)
                    if (CPU_ISSET_S(cpu, current.bytes, current.set)) cpus->push_back(cpu);
            }
            return 0;
        }
        if (error != EINVAL || count > INT_MAX / 2) return error;
        count *= 2;
    }
}

}  // namespace basalt::platform::detail
