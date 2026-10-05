// src/kernels/cuda/q5k_head.cu - see include/strata/kernels/q5k_head.hpp.
//
// Built with --use_fast_math like native_mmvq.cu, whose Q5_K x q8_1 dot this repeats.
#include "strata/kernels/q5k_head.hpp"
#include "strata/kernels/dp4a.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int QK = 256, Q8K = 32, WARP = 32;
constexpr int VT = 4;      // virtual threads per lane: native_mmvq's four warps of one row
constexpr int WPB = 8;     // warps (rows in flight) per block

struct Q5KBlock {
    half2 dm;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};
struct Q81Block {
    half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(Q5KBlock) == 176 && sizeof(Q81Block) == 36);

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) x += __shfl_xor_sync(0xffffffff, x, offset, WARP);
    return x;
}

// native_mmvq.cu's Q5KTraits load and q5_q8_dot_impl: the weight half of one virtual thread's dot, then one column's
struct W5 {
    int v[2][2];        // the 5-bit values of sub-blocks i = 0, 1 (two ints each)
    int sc[2], m[2];    // their 6-bit scales and mins
    float2 dm;
    int bq8_offset;
};
__device__ __forceinline__ W5 q5_load(const Q5KBlock* __restrict__ bq5, int iqs) {
    W5 r;
    r.bq8_offset = 2 * ((iqs / 2) / 4);
    const int* ql = reinterpret_cast<const int*>(bq5->qs + 16 * r.bq8_offset + 4 * ((iqs / 2) % 4));
    const int* qh = reinterpret_cast<const int*>(bq5->qh + 4 * ((iqs / 2) % 4));
    int vl[2], vh[2];
    vl[0] = __ldg(ql);
    vl[1] = __ldg(ql + 4);
    vh[0] = __ldg(qh) >> r.bq8_offset;
    vh[1] = __ldg(qh + 4) >> r.bq8_offset;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vl0i = (vl[0] >> (4 * i)) & 0x0f0f0f0f;
        const int vl1i = (vl[1] >> (4 * i)) & 0x0f0f0f0f;
        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
        r.v[i][0] = vl0i | vh0i;
        r.v[i][1] = vl1i | vh1i;
    }
    const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq5->scales);
    const int j = r.bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = __ldg(scales + jm);
    const uint32_t s2 = __ldg(scales + jm + 2);
    const uint32_t s4 = __ldg(scales + jm + 4);
    const uint32_t hi = uint32_t(-int32_t(j >= 2));
    uint16_t aux[2];
    aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t* sc = reinterpret_cast<const uint8_t*>(aux);
    r.sc[0] = sc[0]; r.sc[1] = sc[1]; r.m[0] = sc[2]; r.m[1] = sc[3];
    const unsigned dmb = __ldg(reinterpret_cast<const unsigned*>(&bq5->dm));
    r.dm = __half22float2(*reinterpret_cast<const half2*>(&dmb));
    return r;
}
// a column in shared memory: per q8_1 block b and position p (0..3), {q[p], q[p + 4], the activation sum
// dp4a(1, q[p], dp4a(1, q[p + 4], 0)) that q5_q8_dot_impl's dot2 computes, the block's scale as float}; blocks XS
// int4 apart, so the blocks b and b + 2 a quarter warp reads sit in disjoint banks
constexpr int XS = 6;
__device__ __forceinline__ int xslot(int b) { return b * XS; }
struct XCol {
    const int4* e;
};
// q5_q8_dot_impl for one column, with the weight-only decode done once per block by q5_load
__device__ __forceinline__ float q5_apply(const W5& r, const XCol& x, int kby, int iqs) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int b = kby + r.bq8_offset + i;
        const int4 e = x.e[xslot(b) + (iqs / 2) % 4];
        const float d8 = __int_as_float(e.w);
        const int dot1 = STRATA_DP4A(r.v[i][0], e.x, STRATA_DP4A(r.v[i][1], e.y, 0));
        const int dot2 = e.z;
        sumf_d += d8 * (dot1 * r.sc[i]);
        sumf_m += d8 * (dot2 * r.m[i]);
    }
    return r.dm.x * sumf_d - r.dm.y * sumf_m;
}

// One row, NC columns, by one warp: lane L stands for native_mmvq's threads L, L+32, L+64, L+96 (kbx = tid / 16,
// += 8), whose sums it adds in the cross-warp order before the same xor tree.  BPR: Q5_K blocks per row (0: runtime).
template <int NC, int BPR>
__device__ __forceinline__ void q5k_row_dev(const Q5KBlock* __restrict__ w, const XCol* xc, float* __restrict__ y,
                                            int n_out, int bpr_rt, int row, int lane) {
    const int bpr = BPR > 0 ? BPR : bpr_rt;
    const int kqs = 2 * (lane % 16);
    float p[VT][NC];
#pragma unroll
    for (int v = 0; v < VT; ++v)
#pragma unroll
        for (int j = 0; j < NC; ++j) p[v][j] = 0.0f;
    const Q5KBlock* wr = w + (size_t) row * bpr;
#pragma unroll
    for (int v = 0; v < VT; ++v) {
        const int tid = lane + WARP * v;
#pragma unroll
        for (int kbx = tid / 16; kbx < (BPR > 0 ? BPR : 64); kbx += 8) {
            if (BPR == 0 && kbx >= bpr) break;
            const W5 wv = q5_load(wr + kbx, kqs);
            const int kby = kbx * (QK / Q8K);
#pragma unroll
            for (int j = 0; j < NC; ++j) p[v][j] += q5_apply(wv, xc[j], kby, kqs);
        }
    }
#pragma unroll
    for (int j = 0; j < NC; ++j) {
        float t = p[0][j];
#pragma unroll
        for (int v = 1; v < VT; ++v) t += p[v][j];
        t = warp_sum(t);
        if (lane == 0) y[(size_t) j * n_out + row] = t;
    }
}

// Rows of 10 blocks (n_in 2560), the next row's weights loaded while this one is computed.  Lane L's blocks:
// thread L: a = L / 16 and a + 8; threads L + 32, L + 64, L + 96: a + 2, a + 4, a + 6.
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

template <int NC, bool TOP = false>
__device__ __forceinline__ void q5k_rows_10(const Q5KBlock* __restrict__ w, const XCol* xc, float* __restrict__ y,
                                            int n_out, int row0, int stride, int lane, Top* top = nullptr) {
    constexpr int NCH = 5;
    const int a = lane / 16, kqs = 2 * (lane % 16);
    const int kb[NCH] = {a, a + 8, a + 2, a + 4, a + 6};
    constexpr int vt[NCH] = {0, 0, 1, 2, 3};
    W5 cur[NCH];
    if (row0 < n_out) {
#pragma unroll
        for (int c = 0; c < NCH; ++c) cur[c] = q5_load(w + (size_t) row0 * 10 + kb[c], kqs);
    }
    for (int row = row0; row < n_out; row += stride) {
        const int nrow = row + stride;
        W5 nxt[NCH];
        if (nrow < n_out) {
#pragma unroll
            for (int c = 0; c < NCH; ++c) nxt[c] = q5_load(w + (size_t) nrow * 10 + kb[c], kqs);
        }
        float p[VT][NC];
#pragma unroll
        for (int v = 0; v < VT; ++v)
#pragma unroll
            for (int j = 0; j < NC; ++j) p[v][j] = 0.0f;
#pragma unroll
        for (int c = 0; c < NCH; ++c)
#pragma unroll
            for (int j = 0; j < NC; ++j) p[vt[c]][j] += q5_apply(cur[c], xc[j], kb[c] * (QK / Q8K), kqs);
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            float t = p[0][j];
#pragma unroll
            for (int v = 1; v < VT; ++v) t += p[v][j];
            t = warp_sum(t);
            if (lane == 0) y[(size_t) j * n_out + row] = t;
            if constexpr (TOP) top->take(t, row);
        }
#pragma unroll
        for (int c = 0; c < NCH; ++c) cur[c] = nxt[c];
    }
}

template <int NC, int BPR>
__global__ void __launch_bounds__(WPB * WARP) q5k_head_kernel(const Q5KBlock* __restrict__ w,
                                                              const Q81Block* __restrict__ x, float* __restrict__ y,
                                                              int n_in, int n_out) {
    extern __shared__ __align__(16) int4 xe[];
    const int x_stride = n_in / Q8K;
    for (int i = threadIdx.x; i < NC * x_stride * 4; i += blockDim.x) {
        const int b = i / 4, pos = i % 4;
        const Q81Block* blk = x + b;
        const int* q8 = reinterpret_cast<const int*>(blk->qs) + pos;
        const int u0 = __ldg(q8), u1 = __ldg(q8 + 4);
        const unsigned dsb = __ldg(reinterpret_cast<const unsigned*>(&blk->ds));
        const float d8 = __low2float(*reinterpret_cast<const half2*>(&dsb));
        xe[xslot(b) + pos] = make_int4(u0, u1, STRATA_DP4A(0x01010101, u0, STRATA_DP4A(0x01010101, u1, 0)),
                                     __float_as_int(d8));
    }
    __syncthreads();
    XCol xc[NC];
#pragma unroll
    for (int j = 0; j < NC; ++j) xc[j] = XCol{xe + (size_t) j * x_stride * XS};
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if constexpr (BPR == 10) {
        q5k_rows_10<NC>(w, xc, y, n_out, blockIdx.x * WPB + warp, gridDim.x * WPB, lane);
    } else {
        for (int row = blockIdx.x * WPB + warp; row < n_out; row += gridDim.x * WPB)
            q5k_row_dev<NC, BPR>(w, xc, y, n_out, n_in / QK, row, lane);
    }
}

// One column, n_in 2560: the logits, and the argmax (first on ties) with its softmax probability over all rows,
// by block partials and the last block's merge; `map` turns the row into a token id.
__global__ void __launch_bounds__(WPB * WARP) q5k_head_top1_kernel(const Q5KBlock* __restrict__ w,
                                                                   const Q81Block* __restrict__ x,
                                                                   float* __restrict__ y, int n_out,
                                                                   const int32_t* __restrict__ map, int* __restrict__ out_id,
                                                                   float* __restrict__ out_prob, float* __restrict__ part,
                                                                   unsigned* __restrict__ count) {
    __shared__ __align__(16) int4 xe[80 * XS];
    __shared__ Top wt[WPB];
    __shared__ bool last;
    for (int i = threadIdx.x; i < 80 * 4; i += blockDim.x) {
        const int b = i / 4, pos = i % 4;
        const Q81Block* blk = x + b;
        const int* q8 = reinterpret_cast<const int*>(blk->qs) + pos;
        const int u0 = __ldg(q8), u1 = __ldg(q8 + 4);
        const unsigned dsb = __ldg(reinterpret_cast<const unsigned*>(&blk->ds));
        const float d8 = __low2float(*reinterpret_cast<const half2*>(&dsb));
        xe[xslot(b) + pos] = make_int4(u0, u1, STRATA_DP4A(0x01010101, u0, STRATA_DP4A(0x01010101, u1, 0)),
                                       __float_as_int(d8));
    }
    __syncthreads();
    const XCol xc[1] = {XCol{xe}};
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    Top top;
    top.init();
    q5k_rows_10<1, true>(w, xc, y, n_out, blockIdx.x * WPB + warp, gridDim.x * WPB, lane, &top);
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

template <int NC, int BPR>
void launch_bpr(const void* w, const void* x, float* y, int n_in, int n_out, cudaStream_t s) {
    // one resident wave: every warp walks rows with the grid's stride
    static int grid_max[64] = {};
    static size_t smem_set[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) dev = 0;
    const size_t smem = (size_t) NC * (n_in / Q8K) * XS * sizeof(int4);
    if (grid_max[dev] == 0 || smem > smem_set[dev]) {
        int sms = 0, per_sm = 0;
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        cudaFuncSetAttribute(q5k_head_kernel<NC, BPR>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
        smem_set[dev] = smem;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, q5k_head_kernel<NC, BPR>, WPB * WARP, smem);
        grid_max[dev] = (sms > 0 ? sms : 1) * (per_sm > 0 ? per_sm : 1);
    }
    const long long want = ((long long) n_out + WPB - 1) / WPB;
    const unsigned grid = (unsigned) (want < grid_max[dev] ? want : grid_max[dev]);
    q5k_head_kernel<NC, BPR><<<grid, WPB * WARP, smem, s>>>((const Q5KBlock*) w, (const Q81Block*) x, y, n_in, n_out);
}
template <int NC>
void launch_nc(const void* w, const void* x, float* y, int n_in, int n_out, cudaStream_t s) {
    if (n_in == 2560) launch_bpr<NC, 10>(w, x, y, n_in, n_out, s);
    else launch_bpr<NC, 0>(w, x, y, n_in, n_out, s);
}

}  // namespace

bool q5k_head_supported(int n_in, int ncols) {
    return n_in > 0 && n_in % QK == 0 && n_in / QK <= 64 && ncols >= 1 && ncols <= 8 &&
           (size_t) ncols * (n_in / Q8K) * XS * sizeof(int4) <= 96 * 1024;
}

size_t q5k_head_top1_scratch_bytes() { return 4096 * 3 * sizeof(float) + 256; }

bool q5k_head_top1(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, const int32_t* map,
                   int* out_id, float* out_prob, void* scratch, void* stream) {
    if (n_in != 2560 || n_out <= 0 || !weights || !x_q8_1 || !y || !out_id || !out_prob || !scratch) return false;
    static int grid_max[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) dev = 0;
    if (grid_max[dev] == 0) {
        int sms = 0, per_sm = 0;
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, q5k_head_top1_kernel, WPB * WARP, 0);
        grid_max[dev] = (sms > 0 ? sms : 1) * (per_sm > 0 ? per_sm : 1);
        if (grid_max[dev] > 4096) grid_max[dev] = 4096;
    }
    const long long want = ((long long) n_out + WPB - 1) / WPB;
    const unsigned grid = (unsigned) (want < grid_max[dev] ? want : grid_max[dev]);
    float* part = (float*) scratch;
    unsigned* count = (unsigned*) ((char*) scratch + 4096 * 3 * sizeof(float));
    q5k_head_top1_kernel<<<grid, WPB * WARP, 0, (cudaStream_t) stream>>>((const Q5KBlock*) weights,
                                                                         (const Q81Block*) x_q8_1, y, n_out, map, out_id,
                                                                         out_prob, part, count);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "q5k_head_top1: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}

void q5k_head_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    if (!q5k_head_supported(n_in, ncols) || n_out <= 0 || !weights || !x_q8_1 || !y) {
        std::fprintf(stderr, "q5k_head_mmvq: unsupported shape (n_in %d, ncols %d)\n", n_in, ncols);
        std::exit(1);
    }
    const auto s = (cudaStream_t) stream;
    switch (ncols) {
        case 1: launch_nc<1>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 2: launch_nc<2>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 3: launch_nc<3>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 4: launch_nc<4>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 5: launch_nc<5>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 6: launch_nc<6>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 7: launch_nc<7>(weights, x_q8_1, y, n_in, n_out, s); break;
        default: launch_nc<8>(weights, x_q8_1, y, n_in, n_out, s); break;
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "q5k_head_mmvq: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace strata::kernels
