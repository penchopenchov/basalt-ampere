#pragma once
// Progress lines for the startup loads, one per line on stderr:
//
//   basalt load: <target> <done_bytes> <total_bytes>
//
// `<target>` is `cuda<N>` (bytes written into that device) or `ram` (experts copied or locked into system RAM).
// A load phase makes a LoadProgress with its total (known before copying), calls add() as bytes land and finish()
// at the end.  A line is printed when the phase starts (done 0), whenever done has advanced by 1% of the total or
// 250 ms have passed, and once with done == total.  Safe to call from several threads.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>

namespace basalt::core {

class LoadProgress {
public:
    LoadProgress(std::string target, uint64_t total) : target_(std::move(target)), total_(total) {
        last_ = std::chrono::steady_clock::now();
        emit(0);
    }
    /// The target of a device ordinal: "cuda<N>".
    static std::string cuda(int device) { return "cuda" + std::to_string(device); }

    void add(uint64_t bytes) {
        std::lock_guard<std::mutex> lk(mu_);
        done_ += bytes;
        if (done_ > total_) done_ = total_;
        const auto now = std::chrono::steady_clock::now();
        const bool by_size = total_ > 0 && (done_ - last_done_) * 100 >= total_;
        const bool by_time = std::chrono::duration<double, std::milli>(now - last_).count() >= 250.0;
        if (!finished_ && (by_size || by_time) && done_ != last_done_) emit_locked(now);
    }
    /// The final line (done == total).
    void finish() {
        std::lock_guard<std::mutex> lk(mu_);
        if (finished_) return;
        finished_ = true;
        done_ = total_;
        emit_locked(std::chrono::steady_clock::now());
    }

private:
    void emit(uint64_t done) {
        std::lock_guard<std::mutex> lk(mu_);
        done_ = done;
        emit_locked(std::chrono::steady_clock::now());
    }
    void emit_locked(std::chrono::steady_clock::time_point now) {
        std::fprintf(stderr, "basalt load: %s %llu %llu\n", target_.c_str(), (unsigned long long) done_,
                     (unsigned long long) total_);
        last_done_ = done_;
        last_ = now;
    }

    std::string target_;
    uint64_t total_;
    uint64_t done_ = 0, last_done_ = 0;
    bool finished_ = false;
    std::mutex mu_;
    std::chrono::steady_clock::time_point last_;
};

}  // namespace basalt::core
