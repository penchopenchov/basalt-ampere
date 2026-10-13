// src/platform/memory.cpp - see include/basalt/platform/memory.hpp.
#include "basalt/platform/memory.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace basalt::platform {

LockResult lock_resident(void* p, uint64_t bytes) {
    LockResult r;
    if (p == nullptr || bytes == 0) { r.note = "nothing to lock"; return r; }
    if (mlock(p, bytes) != 0) { r.note = "mlock failed (raise ulimit -l)"; return r; }
    r.ok = true;
    r.locked_bytes = bytes;
    r.note = "mlock";
    return r;
}

void unlock_resident(void* p, uint64_t bytes) {
    if (p != nullptr && bytes != 0) munlock(p, bytes);
}

uint64_t total_physical_memory() {
    const long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGE_SIZE);
    return pages > 0 && page > 0 ? (uint64_t) pages * (uint64_t) page : 0;
}

namespace {
// One "<key>: <n> kB" line of /proc/meminfo in bytes (0 when absent).
uint64_t meminfo_bytes(const char* key) {
    std::ifstream in("/proc/meminfo");
    std::string line;
    const std::string k = std::string(key) + ":";
    while (std::getline(in, line))
        if (line.compare(0, k.size(), k) == 0) return std::strtoull(line.c_str() + k.size(), nullptr, 10) * 1024;
    return 0;
}
}  // namespace

uint64_t available_memory() { return meminfo_bytes("MemAvailable"); }

uint64_t ram_margin() {
    return std::max<uint64_t>(2ull << 30, meminfo_bytes("MemTotal") / 20);
}

uint64_t ram_budget(bool* known) {
    const uint64_t avail = available_memory();
    if (known) *known = avail > 0;
    if (avail == 0) return UINT64_MAX;
    const uint64_t margin = ram_margin();
    return avail > margin ? avail - margin : 0;
}

namespace {
std::atomic<bool> g_watch_stop{false};
std::atomic<bool> g_watch_on{false};
std::atomic<bool> g_resident_mode{false};
std::atomic<uint64_t> g_resident_copy{0};
}  // namespace

void watch_resident_ram(bool resident, uint64_t copy_bytes) {
    g_resident_mode = resident;
    g_resident_copy = copy_bytes;
}

void start_memory_watchdog(uint64_t floor_bytes) {
    if (g_watch_on.exchange(true)) return;
    g_watch_stop = false;
    std::thread([floor_bytes] {
        while (!g_watch_stop.load()) {
            const uint64_t avail = available_memory();
            if (avail > 0 && avail < floor_bytes) {
                const unsigned long long mib = (unsigned long long) (avail >> 20);
                if (g_resident_mode.load()) {
                    // the copy shrunk by what the margin is short of leaves the margin free again
                    const uint64_t copy = g_resident_copy.load(), margin = ram_margin();
                    const uint64_t keep = copy - std::min(copy, margin > avail ? margin - avail : 0);
                    std::fprintf(stderr, "basalt error: RAM ran out loading (%llu MiB free); use "
                                         "--resident-budget-gib %llu\n", mib, (unsigned long long) std::max<uint64_t>(1, keep >> 30));
                } else {
                    std::fprintf(stderr, "basalt error: RAM ran out loading (%llu MiB free); use "
                                         "--resident-experts or --mmap-experts\n", mib);
                }
                std::fflush(stderr);
                _exit(1);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }).detach();   // never joined: the process may end while it sleeps
}

void stop_memory_watchdog() {
    g_watch_stop = true;
    g_watch_on = false;
}

namespace {
std::atomic<bool> g_guard_on{false};
std::atomic<bool> g_guard_low{false};
std::atomic<uint64_t> g_guard_avail{0};
std::atomic<uint64_t> g_guard_floor{0};
}  // namespace

void start_ram_guard(uint64_t floor_bytes) {
    g_guard_floor = floor_bytes;
    if (g_guard_on.exchange(true)) return;
    std::thread([] {
        bool was_low = false;
        for (;;) {
            const uint64_t avail = available_memory(), floor = g_guard_floor.load();
            const bool low = ram_below_floor(avail, floor);
            g_guard_avail = avail;
            g_guard_low = low;
            if (low && !was_low) {
                std::fprintf(stderr, "basalt warning: RAM low: %llu MiB free < %llu MiB guard\n",
                             (unsigned long long) (avail >> 20), (unsigned long long) (floor >> 20));
                std::fflush(stderr);
            }
            was_low = low;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }).detach();   // runs for the life of the process
}

bool ram_guard_low(uint64_t* avail, uint64_t* floor) {
    if (avail) *avail = g_guard_avail.load();
    if (floor) *floor = g_guard_floor.load();
    return g_guard_low.load(std::memory_order_relaxed);
}

namespace {
constexpr uint64_t kAdviseStep = 128ull << 10;
}

bool read_ahead_enabled() { return true; }

void advise_willneed(const void* p, uint64_t bytes) {
    if (p == nullptr || bytes == 0 || !read_ahead_enabled()) return;
    const long ps = sysconf(_SC_PAGE_SIZE);
    const uintptr_t pg = ps > 0 ? (uintptr_t) ps : 4096, end = (uintptr_t) p + bytes;
    for (uintptr_t a = (uintptr_t) p & ~(pg - 1); a < end; a += kAdviseStep)
        (void) madvise((void*) a, (size_t) std::min<uintptr_t>(kAdviseStep, end - a), MADV_WILLNEED);
}

void advise_willneed(int fd, uint64_t offset, uint64_t bytes) {
    if (fd < 0 || !read_ahead_enabled()) return;
    for (uint64_t at = 0; at < bytes; at += kAdviseStep)
        (void) posix_fadvise(fd, (off_t) (offset + at), (off_t) std::min(kAdviseStep, bytes - at), POSIX_FADV_WILLNEED);
}

}  // namespace basalt::platform
