// src/kernels/stamp_ring.cpp - host side of the PROFILE stamp ring: allocation and the dumper thread.
#ifdef BASALT_PROFILE
#include "basalt/kernels/stamp_ring.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

namespace basalt::kernels::stamp {
namespace {
// File: repeated { u64 magic, u64 kid | launch << 32, u64 ncta, u64 pad } + ncta * kStampRec u64 (records ordered by role)
constexpr unsigned long long kMagic = 0x5354414d50303031ull;

volatile unsigned long long* g_ring = nullptr;
std::atomic<bool> g_run{false};

unsigned long long hdr_at(int kid, unsigned slot, int role) {
    return g_ring[((size_t) (kid * kStampLaunches + slot) * kStampMaxCta + role) * kStampRec];
}

void dumper(const char* path, unsigned long long max_bytes) {
    FILE* fp = std::fopen(path, "wb");
    if (!fp) { std::fprintf(stderr, "stamp: cannot open %s\n", path); return; }
    unsigned next[kKids] = {};
    unsigned long long written = 0;
    unsigned long long buf[kStampMaxCta * kStampRec];
    unsigned long long tag[4];
    while (g_run.load(std::memory_order_relaxed) && written < max_bytes) {
        bool any = false;
        for (int kid = 1; kid < kKids; ++kid) {
            for (;;) {
                const unsigned slot = next[kid] % kStampLaunches;
                int n = 0, ncta = 0;
                bool newer = false;
                for (int role = 0; role < kStampMaxCta; ++role) {
                    const unsigned long long h = hdr_at(kid, slot, role);
                    if ((int) (h & 0xff) != kid) continue;
                    const unsigned L = (unsigned) (h >> 32);
                    if ((int) (L - next[kid]) > 0) { newer = true; continue; }
                    if (L != next[kid]) continue;
                    ncta = (int) ((h >> 24) & 0xff);
                    ++n;
                }
                if (ncta == 0 || n < ncta) {
                    // an overrun or a launch that never completed: move on once a later launch has finished
                    if (newer) { ++next[kid]; continue; }
                    break;
                }
                std::atomic_thread_fence(std::memory_order_acquire);
                bool done = true;
                int k = 0;
                for (int role = 0; role < kStampMaxCta && k < ncta; ++role) {
                    const volatile unsigned long long* r = g_ring + ((size_t) (kid * kStampLaunches + slot) * kStampMaxCta + role) * kStampRec;
                    const unsigned long long h = r[0];
                    if ((int) (h & 0xff) != kid || (unsigned) (h >> 32) != next[kid]) continue;
                    if (r[kExit] < r[1] || r[kExit] == 0) { done = false; break; }
                    for (int s = 0; s < kStampRec; ++s) buf[(size_t) k * kStampRec + s] = r[s];
                    ++k;
                }
                if (!done) break;
                if (ncta > 1 && kid == kFgdn && (next[kid] & 63u) == 0u) {   // raw per-CTA records (kid + 16)
                    tag[0] = kMagic; tag[1] = (unsigned long long) (kid + 16) | (unsigned long long) next[kid] << 32; tag[2] = (unsigned long long) ncta; tag[3] = 0;
                    std::fwrite(tag, 8, 4, fp); std::fwrite(buf, 8, (size_t) ncta * kStampRec, fp);
                    written += 32 + (unsigned long long) ncta * kStampRec * 8;
                }
                if (ncta > 1) {   // one record per launch: earliest entry, per-slot maximum, latest exit
                    unsigned long long med[kStampRec - kB - 1] = {};   // fused layer: median of each experts-half slot (over the CTAs that wrote it) goes to kB + 32 + s
                    if (kid == kFgdn)
                        for (int sl = 0; sl < 29; ++sl) {
                            std::vector<unsigned long long> v;
                            for (int c = 0; c < ncta; ++c) { const unsigned long long x = buf[(size_t) c * kStampRec + kB + sl]; if (x) v.push_back(x); }
                            if (!v.empty()) { std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end()); med[sl] = v[v.size() / 2]; }
                        }
                    if (kid == kFgdn) {   // the critical CTA (latest S_SHD): its experts-half slots replace the medians, its role in the last
                        int cc = 0;
                        for (int c = 1; c < ncta; ++c) if (buf[(size_t) c * kStampRec + kB + 6] > buf[(size_t) cc * kStampRec + kB + 6]) cc = c;
                        for (int sl = 0; sl < 28; ++sl) med[sl] = buf[(size_t) cc * kStampRec + kB + sl];
                        med[28] = (unsigned long long) cc;
                    }
                    for (int sl = 2; sl < kExit; ++sl) {
                        unsigned long long m = 0;
                        for (int c = 0; c < ncta; ++c) m = std::max(m, buf[(size_t) c * kStampRec + sl]);
                        buf[sl] = m;
                    }
                    unsigned long long t0 = buf[1], t1 = buf[kExit];
                    for (int c = 1; c < ncta; ++c) {
                        t0 = std::min(t0, buf[(size_t) c * kStampRec + 1]);
                        t1 = std::max(t1, buf[(size_t) c * kStampRec + kExit]);
                    }
                    buf[1] = t0; buf[kExit] = t1;
                    if (kid == kFgdn) for (int sl = 0; sl < 29; ++sl) buf[kB + 32 + sl] = med[sl];
                    ncta = 1;
                }
                tag[0] = kMagic; tag[1] = (unsigned long long) kid | (unsigned long long) next[kid] << 32; tag[2] = (unsigned long long) ncta; tag[3] = 0;
                std::fwrite(tag, 8, 4, fp);
                std::fwrite(buf, 8, (size_t) ncta * kStampRec, fp);
                written += 32 + (unsigned long long) ncta * kStampRec * 8;
                ++next[kid];
                any = true;
            }
        }
        if (any) std::fflush(fp);
        else std::this_thread::sleep_for(std::chrono::microseconds(150));
    }
    std::fflush(fp);
    std::fclose(fp);
}

Ctx g_ctx;
bool g_init = false;
void (*g_binders[32])(const Ctx&);
int g_nb = 0;
}  // namespace

void reg_bind(void (*f)(const Ctx&)) { if (g_nb < 32) g_binders[g_nb++] = f; }

Ctx ctx() {
    if (g_init) return g_ctx;
    g_init = true;
    const char* path = std::getenv("BASALT_STAMP_FILE");
    if (!path || !*path) return g_ctx;
    const size_t n = (size_t) kKids * kStampLaunches * kStampMaxCta * kStampRec;
    void* h = nullptr;
    if (cudaHostAlloc(&h, n * 8, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) { std::fprintf(stderr, "stamp: ring alloc failed\n"); return g_ctx; }
    std::memset(h, 0, n * 8);
    void* d = nullptr;
    unsigned* cnt = nullptr;
    if (cudaHostGetDevicePointer(&d, h, 0) != cudaSuccess || cudaMalloc((void**) &cnt, kKids * kStampMaxCta * 4) != cudaSuccess) { std::fprintf(stderr, "stamp: ring map failed\n"); return g_ctx; }
    cudaMemset(cnt, 0, kKids * kStampMaxCta * 4);
    g_ring = (volatile unsigned long long*) h;
    g_ctx.ring = (unsigned long long*) d;
    g_ctx.cnt = cnt;
    for (int i = 0; i < g_nb; ++i) g_binders[i](g_ctx);
    const char* mb = std::getenv("BASALT_STAMP_MAX_MB");
    const unsigned long long cap = (mb ? std::strtoull(mb, nullptr, 10) : 1500ull) << 20;
    g_run = true;
    std::thread(dumper, path, cap).detach();
    return g_ctx;
}
}  // namespace basalt::kernels::stamp
#endif
