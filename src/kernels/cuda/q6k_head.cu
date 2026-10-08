// src/kernels/cuda/q6k_head.cu - see include/basalt/kernels/q6k_head.hpp.
//
// Built with --use_fast_math like native_mmvq.cu, whose Q6_K x q8_1 dot this repeats.
#include "basalt/kernels/q6k_head.hpp"
#include "basalt/kernels/dp4a.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace basalt::kernels {
namespace {

constexpr int QK = 256, Q8K = 32, WARP = 32;
constexpr int VT = 4;      // virtual threads per lane: native_mmvq's four warps of one row
constexpr int WPB = 8;     // warps (rows in flight) per block
constexpr int BPR = 10;    // Q6_K blocks per row (n_in 2560)

struct Q6KBlock {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    half d;
};
struct Q81Block {
    half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(Q6KBlock) == 210 && sizeof(Q81Block) == 36);

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) x += __shfl_xor_sync(0xffffffff, x, offset, WARP);
    return x;
}

// the 16-bit loads of native_mmvq's load_int_b2: blocks are only two-byte aligned
__device__ __forceinline__ int load_b2(const void* ptr, int i32) {
    const uint16_t* x = static_cast<const uint16_t*>(ptr);
    return (int) x[2 * i32] | ((int) x[2 * i32 + 1] << 16);
}

// native_mmvq's Q6KTraits::load for lane `iqs`: the weight half of one virtual thread's dot
struct W6 {
    int vl, vh;
    int sc;      // the two signed scales, bytes 0 and 1
    float d;
};
__device__ __forceinline__ W6 q6_load(const Q6KBlock* __restrict__ b, int iqs) {
    W6 r;
    const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift = 2 * ((iqs % 16) / 8);
    r.vl = load_b2(b->ql, iqs);
    r.vh = load_b2(b->qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
    r.sc = (int) (uint8_t) b->scales[scale_offset] | ((int) (uint8_t) b->scales[scale_offset + 4] << 8);
    r.d = __half2float(b->d);
    return r;
}
// a column in shared memory: per q8_1 block b and position p (0..7), {q[p], the block's scale as float}
__device__ __forceinline__ float q6_apply(const W6& r, const int2* __restrict__ x, int kby, int iqs) {
    const int bo = 4 * (iqs / 16) + (iqs % 16) / 8;
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int2 e = x[(kby + bo + 2 * i) * 8 + iqs % 8];
        const int sc = (int) (int8_t) (r.sc >> (8 * i));
        const int vil = (r.vl >> (4 * i)) & 0x0f0f0f0f;
        const int vih = ((r.vh >> (4 * i)) << 4) & 0x30303030;
        const int vi = __vsubss4(vil | vih, 0x20202020);
        sumf += __int_as_float(e.y) * (DP4A(vi, e.x, 0) * sc);
    }
    return r.d * sumf;
}

// the running (max, its first row, sum of exp(l - max)) of the logits a warp has produced
struct Top {
    float m;
    int idx;   // 0x7fffffff: no rows yet
    float s;
    __device__ __forceinline__ void init() {
        m = __int_as_float(0xff800000);
        idx = 0x7fffffff;
        s = 0.0f;
    }
    __device__ __forceinline__ void take(float l, int row) {
        if (l > m) {
            s = s * __expf(m - l) + 1.0f;
            m = l;
            idx = row;
        } else {
            s += __expf(l - m);
            if (l == m && row < idx) idx = row;
        }
    }
    __device__ __forceinline__ void merge(const Top& o) {
        if (o.idx == 0x7fffffff) return;   // no rows
        if (idx == 0x7fffffff || o.m > m) {
            s = o.s + s * __expf(m - o.m);
            m = o.m;
            idx = o.idx;
        } else {
            s += o.s * __expf(o.m - m);
            if (o.m == m && o.idx < idx) idx = o.idx;
        }
    }
};

// Lane L stands for native_mmvq's threads L, L+32, L+64, L+96, whose blocks are kbx = v, v + 4, v + 8 (v = tid / 32);
// their sums are added in the cross-warp order before the same xor tree.  The next row's weights are loaded while
// this one is computed.
template <int NC, bool TOP = false>
__device__ __forceinline__ void q6k_rows(const Q6KBlock* __restrict__ w, const int2* const* xc, float* __restrict__ y,
                                         int n_out, int row0, int stride, int lane, Top* top = nullptr,
                                         const int32_t* __restrict__ map = nullptr) {
    constexpr int kb[BPR] = {0, 4, 8, 1, 5, 9, 2, 6, 3, 7};
    constexpr int vt[BPR] = {0, 0, 0, 1, 1, 1, 2, 2, 3, 3};
    W6 cur[BPR];
    if (row0 < n_out) {
#pragma unroll
        for (int c = 0; c < BPR; ++c) cur[c] = q6_load(w + (size_t) row0 * BPR + kb[c], lane);
    }
    for (int row = row0; row < n_out; row += stride) {
        const int nrow = row + stride;
        W6 nxt[BPR];
        if (nrow < n_out) {
#pragma unroll
            for (int c = 0; c < BPR; ++c) nxt[c] = q6_load(w + (size_t) nrow * BPR + kb[c], lane);
        }
        float p[VT][NC];
#pragma unroll
        for (int v = 0; v < VT; ++v)
#pragma unroll
            for (int j = 0; j < NC; ++j) p[v][j] = 0.0f;
#pragma unroll
        for (int c = 0; c < BPR; ++c)
#pragma unroll
            for (int j = 0; j < NC; ++j) p[vt[c]][j] += q6_apply(cur[c], xc[j], kb[c] * (QK / Q8K), lane);
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            float t = p[0][j];
#pragma unroll
            for (int v = 1; v < VT; ++v) t += p[v][j];
            t = warp_sum(t);
            if constexpr (TOP) {
                // a row whose id is negative holds no token: it never wins and adds nothing to the sum
                if (map != nullptr && map[row] < 0) t = __int_as_float(0xff800000);
                else top->take(t, row);
            }
            if (lane == 0) y[(size_t) j * n_out + row] = t;
        }
#pragma unroll
        for (int c = 0; c < BPR; ++c) cur[c] = nxt[c];
    }
}

constexpr int XBLK = BPR * QK / Q8K;   // q8_1 blocks per column

template <int NC>
__global__ void __launch_bounds__(WPB * WARP) q6k_head_kernel(const Q6KBlock* __restrict__ w,
                                                              const Q81Block* __restrict__ x, float* __restrict__ y,
                                                              int n_out) {
    extern __shared__ __align__(16) int2 xe[];
    for (int i = threadIdx.x; i < NC * XBLK * 8; i += blockDim.x) {
        const int b = i / 8, pos = i % 8;
        const Q81Block* blk = x + b;
        const unsigned dsb = __ldg(reinterpret_cast<const unsigned*>(&blk->ds));
        xe[i] = make_int2(__ldg(reinterpret_cast<const int*>(blk->qs) + pos),
                          __float_as_int(__low2float(*reinterpret_cast<const half2*>(&dsb))));
    }
    __syncthreads();
    const int2* xc[NC];
#pragma unroll
    for (int j = 0; j < NC; ++j) xc[j] = xe + (size_t) j * XBLK * 8;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    q6k_rows<NC>(w, xc, y, n_out, blockIdx.x * WPB + warp, gridDim.x * WPB, lane);
}

// One column, n_in 2560: the logits, and the argmax (first on ties) with its softmax probability over all rows,
// by block partials and the last block's merge; `map` turns the row into a token id.
__global__ void __launch_bounds__(WPB * WARP) q6k_head_top1_kernel(const Q6KBlock* __restrict__ w,
                                                                   const Q81Block* __restrict__ x,
                                                                   float* __restrict__ y, int n_out,
                                                                   const int32_t* __restrict__ map, int* __restrict__ out_id,
                                                                   float* __restrict__ out_prob, float* __restrict__ part,
                                                                   unsigned* __restrict__ count) {
    __shared__ __align__(16) int2 xe[XBLK * 8];
    __shared__ Top wt[WPB];
    __shared__ bool last;
    for (int i = threadIdx.x; i < XBLK * 8; i += blockDim.x) {
        const int b = i / 8, pos = i % 8;
        const Q81Block* blk = x + b;
        const unsigned dsb = __ldg(reinterpret_cast<const unsigned*>(&blk->ds));
        xe[i] = make_int2(__ldg(reinterpret_cast<const int*>(blk->qs) + pos),
                          __float_as_int(__low2float(*reinterpret_cast<const half2*>(&dsb))));
    }
    __syncthreads();
    const int2* xc[1] = {xe};
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    Top top;
    top.init();
    q6k_rows<1, true>(w, xc, y, n_out, blockIdx.x * WPB + warp, gridDim.x * WPB, lane, &top, map);
    if (lane == 0) wt[warp] = top;
    __syncthreads();
    if (threadIdx.x == 0) {
        Top b = wt[0];
        for (int i = 1; i < WPB; ++i) b.merge(wt[i]);
        part[3 * blockIdx.x] = b.m;
        part[3 * blockIdx.x + 1] = __int_as_float(b.idx);
        part[3 * blockIdx.x + 2] = b.s;
        __threadfence();
        last = atomicAdd(count, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    Top g;   // each thread a strided share of the block partials, then the block
    g.init();
    for (int i = threadIdx.x; i < (int) gridDim.x; i += blockDim.x) {
        Top o;
        o.m = __ldcg(part + 3 * i);
        o.idx = __float_as_int(__ldcg(part + 3 * i + 1));
        o.s = __ldcg(part + 3 * i + 2);
        g.merge(o);
    }
    for (int off = 16; off > 0; off >>= 1) {
        Top o;
        o.m = __shfl_xor_sync(0xffffffffu, g.m, off);
        o.idx = __shfl_xor_sync(0xffffffffu, g.idx, off);
        o.s = __shfl_xor_sync(0xffffffffu, g.s, off);
        g.merge(o);
    }
    if (lane == 0) wt[warp] = g;
    __syncthreads();
    if (threadIdx.x == 0) {
        Top b = wt[0];
        for (int i = 1; i < WPB; ++i) b.merge(wt[i]);
        const int id = b.idx < n_out ? b.idx : 0;
        *out_id = map != nullptr ? map[id] : id;
        *out_prob = 1.0f / b.s;
        *count = 0u;
    }
}

template <int NC>
void launch_nc(const void* w, const void* x, float* y, int n_out, cudaStream_t s) {
    // one resident wave: every warp walks rows with the grid's stride
    static int grid_max[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) dev = 0;
    const size_t smem = (size_t) NC * XBLK * 8 * sizeof(int2);
    static bool set[64] = {};
    if (!set[dev]) {
        cudaFuncSetAttribute(q6k_head_kernel<NC>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
        set[dev] = true;
    }
    if (grid_max[dev] == 0) {
        int sms = 0, per_sm = 0;
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, q6k_head_kernel<NC>, WPB * WARP, smem);
        grid_max[dev] = (sms > 0 ? sms : 1) * (per_sm > 0 ? per_sm : 1);
    }
    const long long want = ((long long) n_out + WPB - 1) / WPB;
    const unsigned grid = (unsigned) (want < grid_max[dev] ? want : grid_max[dev]);
    q6k_head_kernel<NC><<<grid, WPB * WARP, smem, s>>>((const Q6KBlock*) w, (const Q81Block*) x, y, n_out);
}

}  // namespace

bool q6k_head_supported(int n_in, int ncols) { return n_in == BPR * QK && ncols >= 1 && ncols <= 8; }

size_t q6k_head_top1_scratch_bytes() { return 4096 * 3 * sizeof(float) + 256; }

bool q6k_head_top1(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, const int32_t* map,
                   int* out_id, float* out_prob, void* scratch, void* stream) {
    if (n_in != BPR * QK || n_out <= 0 || !weights || !x_q8_1 || !y || !out_id || !out_prob || !scratch) return false;
    static int grid_max[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) dev = 0;
    if (grid_max[dev] == 0) {
        int sms = 0, per_sm = 0;
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, q6k_head_top1_kernel, WPB * WARP, 0);
        grid_max[dev] = (sms > 0 ? sms : 1) * (per_sm > 0 ? per_sm : 1);
        if (grid_max[dev] > 4096) grid_max[dev] = 4096;
    }
    const long long want = ((long long) n_out + WPB - 1) / WPB;
    const unsigned grid = (unsigned) (want < grid_max[dev] ? want : grid_max[dev]);
    float* part = (float*) scratch;
    unsigned* count = (unsigned*) ((char*) scratch + 4096 * 3 * sizeof(float));
    q6k_head_top1_kernel<<<grid, WPB * WARP, 0, (cudaStream_t) stream>>>((const Q6KBlock*) weights,
                                                                         (const Q81Block*) x_q8_1, y, n_out, map, out_id,
                                                                         out_prob, part, count);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "q6k_head_top1: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}

void q6k_head_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    if (!q6k_head_supported(n_in, ncols) || n_out <= 0 || !weights || !x_q8_1 || !y) {
        std::fprintf(stderr, "q6k_head_mmvq: unsupported shape (n_in %d, ncols %d)\n", n_in, ncols);
        std::exit(1);
    }
    const auto s = (cudaStream_t) stream;
    switch (ncols) {
        case 1: launch_nc<1>(weights, x_q8_1, y, n_out, s); break;
        case 2: launch_nc<2>(weights, x_q8_1, y, n_out, s); break;
        case 3: launch_nc<3>(weights, x_q8_1, y, n_out, s); break;
        case 4: launch_nc<4>(weights, x_q8_1, y, n_out, s); break;
        case 5: launch_nc<5>(weights, x_q8_1, y, n_out, s); break;
        case 6: launch_nc<6>(weights, x_q8_1, y, n_out, s); break;
        case 7: launch_nc<7>(weights, x_q8_1, y, n_out, s); break;
        default: launch_nc<8>(weights, x_q8_1, y, n_out, s); break;
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "q6k_head_mmvq: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace basalt::kernels
