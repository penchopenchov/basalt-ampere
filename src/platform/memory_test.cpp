// src/platform/memory_test.cpp - lock_resident on a 256 MiB region, the RAM floor and the file cache decisions (CPU
// only, no GPU, no model).
#include "basalt/platform/memory.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>


namespace {
uint64_t gib(double g) { return (uint64_t) (g * 1073741824.0); }

// The file tier's unbuffered choice (true = the file cache keeps its reads, so they stay buffered)
int file_cache_keeps_cases() {
    using basalt::platform::file_cache_keeps;
    struct Case { const char* what; double avail, arena, read; bool keeps; };
    const double experts = 71.7;   // routed experts of a large model
    const Case cases[] = {
        // passing the budget asked for and every shard's bytes (103.7 GiB) goes unbuffered although the experts
        // outside the copy would fit the cache
        {"96 GB, budget 72, all shards' bytes", 83.0, 72.0, 103.7, false},
        {"96 GB, budget 72, before the copy (arena = all experts)", 83.0, experts, 0.0, true},
        {"96 GB, budget 72, copy built (48.94 resident)", 83.0 - 48.94, 0.0, experts - 48.94, true},
        {"96 GB, budget 40, before the copy", 83.0, 40.0, experts - 40.0, true},
        {"96 GB, budget 40, copy built", 83.0 - 40.0, 0.0, experts - 40.0, true},
        // low-RAM machines stay unbuffered
        {"64 GB, budget 40, before the copy", 55.0, 40.0, experts - 40.0, false},
        {"64 GB, budget 40, copy built", 15.0, 0.0, experts - 40.0, false},
        {"32 GB, budget 16, before the copy", 26.0, 16.0, experts - 16.0, false},
        {"32 GB, budget 16, copy built", 10.0, 0.0, experts - 16.0, false},
        {"32 GB, Q2_0 experts (34 GiB), budget 8", 26.0, 8.0, 34.0 - 8.0, false},
        {"nothing available", 0.0, 0.0, 1.0, false},
        {"arena larger than the RAM", 10.0, 20.0, 0.0, true},
    };
    int fail = 0;
    for (const Case& c : cases) {
        const bool got = file_cache_keeps(gib(c.avail), gib(c.arena), gib(c.read));
        std::printf("file_cache_keeps %-58s -> %s%s\n", c.what, got ? "buffered" : "unbuffered",
                    got == c.keeps ? "" : "   <-- WRONG");
        fail |= got != c.keeps;
    }
    return fail;
}
// The RAM floor, the largest safe copy and the serve guard's decision, on injected MemAvailable readings
int ram_floor_cases() {
    using namespace basalt::platform;
    int fail = 0;
    auto check = [&](const char* what, double got, double want) {
        const bool ok = got > want - 0.01 && got < want + 0.01;
        std::printf("ram floor %-62s %8.2f GiB%s\n", what, got, ok ? "" : "   <-- WRONG");
        fail |= !ok;
    };
    const double G = 1073741824.0;
    const uint64_t margin = gib(2.0);
    const uint64_t layer_q4 = 512ull * 3170304;   // every expert of a layer of a 4-bit pack (1.51 GiB)
    const uint64_t layer_q2 = 512ull * 1382400;   // the same of the 2-bit pack (0.66 GiB)
    check("nothing read from the file: the margin alone", ram_floor(margin, 0, layer_q4) / G, 2.0);
    check("4-bit pack, experts read from the file: margin + 3.5 + 2 layers", ram_floor(margin, gib(15), layer_q4) / G,
          2.0 + 3.5 + 2.0 * (double) layer_q4 / G);
    check("2-bit pack: margin + 3.5 + 2 layers", ram_floor(margin, gib(1), layer_q2) / G,
          2.0 + 3.5 + 2.0 * (double) layer_q2 / G);
    check("small layers: the 4 GiB file cache minimum applies", file_cache_floor(1, gib(0.1)) / G, 4.0);
    check("one byte from the file is enough to need the cache", file_cache_floor(1, layer_q4) / G,
          3.5 + 2.0 * (double) layer_q4 / G);
    // a 30 GiB machine with 27 GiB available and 2 GiB taken by the server later
    const uint64_t fl = ram_floor(margin, gib(15), layer_q4);
    check("largest safe copy, 27 available, 2 late", max_safe_copy(gib(27), gib(2), fl) / G, 27.0 - 2.0 - fl / G);
    check("largest safe copy when the floor exceeds the RAM", max_safe_copy(gib(8), gib(2), fl) / G, 0.0);
    const uint64_t guard = ram_guard_floor(margin, gib(15));
    check("serve guard, experts read from the file: margin + 3.5", guard / G, 5.5);
    check("serve guard, nothing read from the file: the margin", ram_guard_floor(margin, 0) / G, 2.0);
    struct Case { const char* what; double avail; bool low; };
    const Case cases[] = {
        {"well above the floor", 20.0, false},
        {"at the load floor (sized as asked)", fl / G, false},
        {"exactly at the guard line", guard / G, false},
        {"just below the guard line", guard / G - 0.01, true},
        {"the freeze's reading (2.5 GB)", 2.33, true},
        {"unknown (0)", 0.0, false},
    };
    for (const Case& c : cases) {
        const bool got = ram_below_floor(gib(c.avail), guard);
        std::printf("ram guard %-62s -> %s%s\n", c.what, got ? "stop" : "run", got == c.low ? "" : "   <-- WRONG");
        fail |= got != c.low;
    }
    return fail;
}
}  // namespace

int main() {
    int fail_keeps = file_cache_keeps_cases() | ram_floor_cases();
    const uint64_t bytes = 256ull << 20;
    void* p = std::malloc(bytes);
    if (p == nullptr) return 2;
    std::memset(p, 1, bytes);
    const basalt::platform::LockResult r = basalt::platform::lock_resident(p, bytes);
    std::printf("lock_resident: ok=%d locked=%llu MiB (%s)\n", (int) r.ok, (unsigned long long) (r.locked_bytes >> 20),
                r.note.c_str());
    int fail = !(r.ok && r.locked_bytes == bytes) | fail_keeps;
    basalt::platform::unlock_resident(p, bytes);
    std::free(p);
    std::printf("platform_memory_test: %s\n", fail ? "FAILED" : "OK");
    return fail;
}
