// src/prefill/ggml_cuda_host.cu - prompt-speed plan step 2b: the host-side symbols of llama.cpp's ggml-cuda that its MMQ
// and quantize code reference, for the MMQ kernels compiled into strata_mmq without the rest of ggml-cuda.cu.
#include "common.cuh"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

[[noreturn]] void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    std::fprintf(stderr, "ggml-cuda (strata mmq): %s: %s\n  in %s at %s:%d\n", msg, stmt, func, file, line);
    std::abort();
}

int ggml_cuda_get_device() {
    int id = 0;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}


const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = [] {
        ggml_cuda_device_info in = {};
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess) n = 0;
        n = n > GGML_CUDA_MAX_DEVICES ? GGML_CUDA_MAX_DEVICES : n;
        in.device_count = n;
        in.physical_device_count = n;
        for (int id = 0; id < n; ++id) {
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
            auto & d = in.devices[id];
            // #542: the fields MMQ's choices read come from cudaDeviceGetAttribute, which is ABI-stable: a
            // cudaDeviceProp filled by an older libcudart than the headers (a CUDA 13 build that links a CUDA 12
            // libcudart.so) is shifted - sharedMemPerBlockOptin read 1 there, and every i-quant lost MMQ (#420's
            // fits()).  In a matched build both give the same values.
            auto attr = [id](cudaDeviceAttr a) {
                int v = 0;
                CUDA_CHECK(cudaDeviceGetAttribute(&v, a, id));
                return v;
            };
            d.cc = 100 * attr(cudaDevAttrComputeCapabilityMajor) +
                   10 * attr(cudaDevAttrComputeCapabilityMinor);
            d.smpbo = attr(cudaDevAttrMaxSharedMemoryPerBlockOptin);
            d.integrated = attr(cudaDevAttrIntegrated) != 0;
            d.supports_cooperative_launch = attr(cudaDevAttrCooperativeLaunch) != 0;
            d.nsm = attr(cudaDevAttrMultiProcessorCount);
            d.smpb = (size_t) attr(cudaDevAttrMaxSharedMemoryPerBlock);
            d.warp_size = attr(cudaDevAttrWarpSize);
            d.vmm = false;
            d.total_vram = prop.totalGlobalMem;
            d.physical_device = id;
            d.physical_share_count = 1;
            d.virtual_index = 0;
        }
        return in;
    }();
    return info;
}

namespace {
// Buffers are kept and reused: MMQ asks for the same few sizes every launch (its stream-k fixup tiles).
struct CachingPool : ggml_cuda_pool {
    struct Buf { void * p; size_t size; bool used; };
    std::vector<Buf> bufs;
    std::mutex mu;
    void * alloc(size_t size, size_t * actual_size) override {
        std::lock_guard<std::mutex> lk(mu);
        for (auto & b : bufs)
            if (!b.used && b.size >= size) { b.used = true; *actual_size = b.size; return b.p; }
        void * p = nullptr;
        CUDA_CHECK(cudaMalloc(&p, size));
        bufs.push_back({p, size, true});
        *actual_size = size;
        return p;
    }
    void free(void * ptr, size_t) override {
        std::lock_guard<std::mutex> lk(mu);
        for (auto & b : bufs)
            if (b.p == ptr) { b.used = false; return; }
    }
    ~CachingPool() override {
        for (auto & b : bufs) cudaFree(b.p);
    }
};
}  // namespace

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int, int) {
    return std::unique_ptr<ggml_cuda_pool>(new CachingPool());
}

ggml_backend_cuda_context::~ggml_backend_cuda_context() {}
