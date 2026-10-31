#include "basalt/core/progress_wait.hpp"
#include <chrono>
#include <cstdio>
#include <future>
#include <thread>

int main() {
    using namespace std::chrono_literals;
    std::atomic<uint64_t> completed{0};
    std::promise<bool> promise;
    auto work = promise.get_future();
    auto& beats = basalt::core::progress().beats;
    const auto initial = beats.load();
    auto waiter = std::async(std::launch::async, [&] { basalt::core::progress_wait(work, completed); });
    std::this_thread::sleep_for(350ms);
    bool fail = beats.load() != initial;
    completed.fetch_add(1);
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (beats.load() == initial && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(10ms);
    fail |= beats.load() == initial;
    const auto after_read = beats.load();
    std::this_thread::sleep_for(350ms);
    fail |= beats.load() != after_read;
    promise.set_value(true);
    waiter.get();
    fail |= !work.get() || beats.load() <= after_read;
    std::printf("progress_wait: %s (idle waits do not beat, completed reads and finished work do)\n", fail ? "FAILED" : "OK");
    return fail ? 1 : 0;
}
