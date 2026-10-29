#pragma once

#include <cuda_runtime.h>
#include <atomic>

namespace basalt::kernels {

inline bool ampere_device() {
    static std::atomic<int> known[64];
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return false;
    int state = known[dev].load(std::memory_order_acquire);
    if (state == 0) {
        int major = 0, minor = 0;
        const bool ok = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
                        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess;
        if (!ok) cudaGetLastError();
        state = ok && major == 8 && minor == 6 ? 1 : -1;
        known[dev].store(state, std::memory_order_release);
    }
    return state > 0;
}

}  // namespace basalt::kernels
