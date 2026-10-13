// src/core/vmm.cpp - the driver's virtual memory functions through the runtime's entry points (no link against the
// driver library).  Read the header first.
#include "basalt/core/vmm.hpp"

#include <cuda_runtime.h>
#include <cuda.h>

namespace basalt::core::vmm {
namespace {
struct Fns {
    CUresult (CUDAAPI* device_get)(CUdevice*, int) = nullptr;
    CUresult (CUDAAPI* attribute)(int*, CUdevice_attribute, CUdevice) = nullptr;
    CUresult (CUDAAPI* granularity)(size_t*, const CUmemAllocationProp*, CUmemAllocationGranularity_flags) = nullptr;
    CUresult (CUDAAPI* reserve)(CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long) = nullptr;
    CUresult (CUDAAPI* address_free)(CUdeviceptr, size_t) = nullptr;
    CUresult (CUDAAPI* create)(CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*,
                               unsigned long long) = nullptr;
    CUresult (CUDAAPI* release)(CUmemGenericAllocationHandle) = nullptr;
    CUresult (CUDAAPI* map)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long) = nullptr;
    CUresult (CUDAAPI* unmap)(CUdeviceptr, size_t) = nullptr;
    CUresult (CUDAAPI* set_access)(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t) = nullptr;
    bool ok = false;
};

template <class F> bool entry(const char* name, F& f) {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q{};
#if CUDART_VERSION >= 12050
    const cudaError_t e = cudaGetDriverEntryPointByVersion(name, &p, 12000, cudaEnableDefault, &q);
#else
    const cudaError_t e = cudaGetDriverEntryPoint(name, &p, cudaEnableDefault, &q);
#endif
    if (e != cudaSuccess || q != cudaDriverEntryPointSuccess || p == nullptr) {
        (void) cudaGetLastError();
        return false;
    }
    f = reinterpret_cast<F>(p);
    return true;
}

const Fns& fns() {
    static const Fns v = [] {
        Fns x;
        x.ok = entry("cuDeviceGet", x.device_get) && entry("cuDeviceGetAttribute", x.attribute) &&
               entry("cuMemGetAllocationGranularity", x.granularity) && entry("cuMemAddressReserve", x.reserve) &&
               entry("cuMemAddressFree", x.address_free) && entry("cuMemCreate", x.create) &&
               entry("cuMemRelease", x.release) && entry("cuMemMap", x.map) && entry("cuMemUnmap", x.unmap) &&
               entry("cuMemSetAccess", x.set_access);
        return x;
    }();
    return v;
}

int current_device() {
    int dev = 0;
    cudaGetDevice(&dev);
    return dev;
}

CUmemAllocationProp device_prop(int dev) {
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = dev;
    return prop;
}
}  // namespace

bool supported() {
    const Fns& v = fns();
    int supported = 0;
    CUdevice cu = 0;
    return v.ok && v.device_get(&cu, current_device()) == CUDA_SUCCESS &&
           v.attribute(&supported, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, cu) == CUDA_SUCCESS &&
           supported != 0;
}

uint64_t granularity() {
    const Fns& v = fns();
    if (!v.ok) return 0;
    const CUmemAllocationProp prop = device_prop(current_device());
    size_t g = 0;
    if (v.granularity(&g, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) != CUDA_SUCCESS) return 0;
    return (uint64_t) g;
}

uint8_t* reserve(uint64_t bytes) {
    const Fns& v = fns();
    CUdeviceptr va = 0;
    if (!v.ok || v.reserve(&va, (size_t) bytes, 0, 0, 0) != CUDA_SUCCESS) return nullptr;
    return reinterpret_cast<uint8_t*>(va);
}

void release_range(uint8_t* va, uint64_t bytes) {
    const Fns& v = fns();
    if (v.ok && va != nullptr && bytes > 0) v.address_free(reinterpret_cast<CUdeviceptr>(va), (size_t) bytes);
}

unsigned long long map(uint8_t* va, uint64_t bytes) {
    const Fns& v = fns();
    if (!v.ok) return 0;
    const int dev = current_device();
    const CUmemAllocationProp prop = device_prop(dev);
    const CUdeviceptr p = reinterpret_cast<CUdeviceptr>(va);
    CUmemGenericAllocationHandle h = 0;
    if (v.create(&h, (size_t) bytes, &prop, 0) != CUDA_SUCCESS) return 0;
    if (v.map(p, (size_t) bytes, 0, h, 0) != CUDA_SUCCESS) {
        v.release(h);
        return 0;
    }
    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = dev;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (v.set_access(p, (size_t) bytes, &access, 1) != CUDA_SUCCESS) {
        v.unmap(p, (size_t) bytes);
        v.release(h);
        return 0;
    }
    return (unsigned long long) h;
}

bool unmap(uint8_t* va, uint64_t bytes, unsigned long long handle) {
    const Fns& v = fns();
    if (!v.ok || handle == 0) return false;
    const bool a = v.unmap(reinterpret_cast<CUdeviceptr>(va), (size_t) bytes) == CUDA_SUCCESS;
    const bool b = v.release((CUmemGenericAllocationHandle) handle) == CUDA_SUCCESS;
    return a && b;
}

}  // namespace basalt::core::vmm
