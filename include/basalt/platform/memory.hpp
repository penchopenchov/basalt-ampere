// include/basalt/platform/memory.hpp - keep a large host region resident.
//
// An unlocked expert arena is trimmed under memory pressure, which makes the CPU pool's speed vary between
// runs. With the n-gram table out of RAM there is headroom to lock it: mlock.
#pragma once

#include <cstdint>
#include <string>

namespace basalt::platform {

struct LockResult {
    bool ok = false;
    uint64_t locked_bytes = 0;   ///< may be less than requested; the rest stays pageable
    std::string note;            ///< what was done or why it failed, for the startup print
};

/// Lock [p, p + bytes) into physical memory. Partial success is reported, not hidden.
LockResult lock_resident(void* p, uint64_t bytes);

/// Undo lock_resident for the same region (best effort).
void unlock_resident(void* p, uint64_t bytes);

/// The machine's physical RAM in bytes (0 when unknown).
uint64_t total_physical_memory();

/// MemAvailable from /proc/meminfo in bytes (0 when unknown).
uint64_t available_memory();

/// The RAM left alone by anything that locks or fills memory: the larger of 2 GiB and 5% of the machine's RAM.
uint64_t ram_margin();

/// Host RAM the engine takes after the resident expert copy is sized and before it serves: the GPU libraries' own
/// host memory, the prompt path's and the verifier's staging.  A resident copy leaves this much beside the margin.
inline constexpr uint64_t kLateHostBytes = 2ull << 30;

/// The RAM a page-locked or fully resident allocation may take now: MemAvailable less ram_margin().  `known` is
/// false when /proc/meminfo cannot be read (nothing is refused then).
uint64_t ram_budget(bool* known = nullptr);

/// Polls MemAvailable until `stop_memory_watchdog`.  Below `floor_bytes` it prints a `basalt error:` line and
/// _exit(1)s, so a load that is exhausting RAM ends before the machine starts to swap.
void start_memory_watchdog(uint64_t floor_bytes);
/// Tells the watchdog the load builds a resident RAM copy of the experts (`resident`) and how big it is so far, so
/// its error names the copy's cap instead of the resident mode.
void watch_resident_ram(bool resident, uint64_t copy_bytes);
void stop_memory_watchdog();

/// Whether the OS file cache could keep the `read_bytes` the expert file is read for, beside
/// `arena_bytes` of RAM held by the engine's own copy of the experts and `margin` for everything else, with `avail`
/// bytes of RAM available.  The file tier passes only the expert bytes it really reads from the file (the experts
/// outside its resident RAM copy) and the RAM that copy really holds.
inline bool file_cache_keeps(uint64_t avail, uint64_t arena_bytes, uint64_t read_bytes,
                             uint64_t margin = 4ull << 30) {
    const uint64_t room = avail > arena_bytes + margin ? avail - arena_bytes - margin : 0;
    return room >= read_bytes;
}

/// Whether `advise_willneed` asks the OS for anything.
bool read_ahead_enabled();
/// Asks the OS to start reading [p, p + bytes) of a file mapping, without waiting.  The kernel reads at most one
/// readahead window per request, so the range is asked for in 128 KiB steps.
void advise_willneed(const void* p, uint64_t bytes);
/// The same for [offset, offset + bytes) of an open file.
void advise_willneed(int fd, uint64_t offset, uint64_t bytes);

}  // namespace basalt::platform
