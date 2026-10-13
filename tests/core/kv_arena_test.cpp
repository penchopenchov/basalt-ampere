// The elastic KV pools (kv_arena.hpp).  Mapping backs whole pages of every array, zeroed; unmapping gives the VRAM
// back and keeps the cells below the cut exactly as they were; the addresses never move.  Needs a CUDA device with
// virtual memory management (exits 77 without one).
#include "basalt/core/kv_arena.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <string>
#include <vector>

namespace {
int fail(const char* what, const std::string& err = "") {
    std::fprintf(stderr, "FAIL %s %s\n", what, err.c_str());
    return 1;
}

size_t free_vram() {
    size_t f = 0, t = 0;
    cudaMemGetInfo(&f, &t);
    return f;
}

// every byte of the first `bytes` of `p` equals `v`
bool all(const uint8_t* p, uint64_t bytes, uint8_t v) {
    std::vector<uint8_t> h((size_t) bytes);
    if (cudaMemcpy(h.data(), p, (size_t) bytes, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    for (const uint8_t x : h)
        if (x != v) return false;
    return true;
}
}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::puts("no CUDA device: skipped");
        return 77;
    }
    // INT8 geometry of one QSA layer: per page of 64 cells, 4 KV heads x 128 codes (K, V) and 2 scales each (K, V)
    constexpr int64_t kPage = 64, kPages = 4096;
    const std::vector<uint64_t> page_bytes = {kPage * 4 * 128, kPage * 4 * 128, kPage * 4 * 2 * 2, kPage * 4 * 2 * 2};
    basalt::core::KvArena a;
    std::string err;
    if (!a.open(page_bytes, kPages, kPage, err)) {
        if (err.find("virtual memory management") != std::string::npos) {
            std::printf("%s: skipped\n", err.c_str());
            return 77;
        }
        return fail("open", err);
    }
    if (a.cells() != 0 || a.mapped_bytes() != 0 || a.max_cells() != kPages * kPage) return fail("open sizes");
    std::vector<uint8_t*> base;
    for (size_t i = 0; i < page_bytes.size(); ++i) base.push_back(a.array(i));
    // back 10000 cells: whole pages, and whole granules of each array
    if (!a.map(10000, err)) return fail("map", err);
    if (a.cells() < 10000 || a.cells() % kPage != 0) return fail("map cells");
    if (a.grow_bytes(10000) != 0 || a.mapped_bytes() != a.bytes_for(10000)) return fail("map bytes");
    for (size_t i = 0; i < page_bytes.size(); ++i) {
        const uint64_t b = (uint64_t) a.cells() / kPage * page_bytes[i];
        if (!all(a.array(i), b, 0)) return fail("fresh pages are zero");
        if (cudaMemset(a.array(i), (int) (i + 1), (size_t) b) != cudaSuccess) return fail("write");
    }
    // back 200000 cells: the old cells kept, the new ones zero, the same addresses
    const int64_t c0 = a.cells();
    const size_t before = free_vram();
    const uint64_t add = a.grow_bytes(200000);
    if (add == 0 || !a.map(200000, err) || a.cells() < 200000) return fail("map more", err);
    if (before - free_vram() + (4u << 20) < add) return fail("VRAM taken");
    for (size_t i = 0; i < page_bytes.size(); ++i) {
        if (a.array(i) != base[i]) return fail("address moved");
        const uint64_t b0 = (uint64_t) c0 / kPage * page_bytes[i], b1 = (uint64_t) 200000 / kPage * page_bytes[i];
        if (!all(a.array(i), b0, (uint8_t) (i + 1))) return fail("old cells kept");
        if (!all(a.array(i) + b0, b1 - b0 - page_bytes[i], 0)) return fail("new cells zero");
    }
    // give back down to 10000: the VRAM back, the first cells as they were
    const size_t mid = free_vram();
    if (!a.unmap(10000, err) || a.cells() != c0) return fail("unmap", err);
    if (free_vram() - mid + (4u << 20) < add) return fail("VRAM not given back");
    for (size_t i = 0; i < page_bytes.size(); ++i)
        if (!all(a.array(i), (uint64_t) c0 / kPage * page_bytes[i], (uint8_t) (i + 1))) return fail("kept after unmap");
    // to nothing, then the whole context
    if (!a.unmap(0, err) || a.cells() != 0 || a.mapped_bytes() != 0) return fail("unmap all", err);
    if (!a.map(a.max_cells(), err) || a.cells() != a.max_cells()) return fail("map all", err);
    a.close();
    if (a.valid()) return fail("close");
    std::puts("kv arena: whole pages backed and zeroed on demand, given back past the cut, addresses fixed");
    return 0;
}
