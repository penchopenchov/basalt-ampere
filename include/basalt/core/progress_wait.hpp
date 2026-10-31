#pragma once
#include "basalt/core/progress.hpp"
#include <future>

namespace basalt::core {

/// Only foreground completion progress extends the watchdog while a read future is pending.
inline void progress_wait(std::future<bool>& work, const std::atomic<uint64_t>& completed) {
    uint64_t seen = completed.load(std::memory_order_relaxed);
    while (work.wait_for(std::chrono::milliseconds(250)) != std::future_status::ready) {
        const uint64_t current = completed.load(std::memory_order_relaxed);
        if (current != seen) progress_beat();
        seen = current;
    }
    progress_beat();
}

}  // namespace basalt::core
