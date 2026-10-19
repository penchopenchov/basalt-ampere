// include/basalt/platform/cpu.hpp - which processors there are, and putting a thread on one.
//
// A processor is named by one int: its OS number on Linux, `group * 64 + index` on Windows, where a machine with more
// than 64 logical processors has several processor groups.
#pragma once

#include <vector>

namespace basalt::platform {

/// How worker threads are allocated across physical and logical processors.
enum class CpuAffinity {
    Auto,      ///< Hybrid: prioritize physical P-cores, then SMT, then E-cores (defaults to P-core count)
    PCores,    ///< Restrict workers strictly to Performance cores and their SMT siblings
    All,       ///< The default: one worker per physical core in the OS's order, without hybrid distinction
};

struct CpuTopology {
    bool is_hybrid = false;
    int p_cores = 0;                ///< Physical performance cores
    int p_threads = 0;              ///< Total logical threads on performance cores
    int e_cores = 0;                ///< Efficient cores
    std::vector<int> worker_cores;  ///< Ordered CPU IDs, one per physical core unless the affinity adds siblings
    int host_core = -1;             ///< Logical core reserved for host thread
};

/// The processors this process may run on, one per physical core (SMT siblings share a core's load/store bandwidth),
/// ordered for `affinity`. `skip_first` takes the first one out of `worker_cores` and names it `host_core`.
CpuTopology detect_cpu_topology(bool skip_first, CpuAffinity affinity = CpuAffinity::All);

/// Pins the calling thread to `core` for good (a pool-owned thread, which ends with the pool). False when the OS
/// refused, after a note on stderr naming `worker` and the error; the previous affinity is kept then. A negative
/// `core` is a no-op that returns false.
bool set_thread_affinity(int core, int worker = -1);

/// A thread's placement before `pin_current_thread` moved it: the CPU mask on Linux, the selected CPU Set IDs on
/// Windows, kept opaque and complete (processors above 63 included). `valid` is false when querying or setting
/// the placement failed, and then there is nothing to restore.
struct ThreadPlacement {
    std::vector<unsigned long> ids;
    bool valid = false;
};

/// Moves the calling thread onto `core` and returns where it was, to give back with `restore_thread_affinity` on the
/// same thread. A caller's implicit all-group placement is kept as it was, not turned into a hard mask.
ThreadPlacement pin_current_thread(int core);
void restore_thread_affinity(const ThreadPlacement& previous);

/// For a thread started by one that is pinned to a single core and spins on it: moves the calling thread to that
/// core's SMT sibling, so it starts at once and the spinner is not preempted. Does nothing when the caller is not
/// confined to one core, when the sibling is unknown, and on Windows, where a host confined by a CPU Set leaves a new
/// thread on the process's own assignment.
void leave_host_core();

}  // namespace basalt::platform
