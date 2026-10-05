// src/kernels/cuda/dec_shx.cuh - STRATA_DEC_SHX: the shared expert in one launch (dec_shx.hpp), on this file's row
// dots and quantizers (included at the end of native_mmvq.cu, inside namespace strata::kernels).
// needs strata/kernels/bf16_bits.hpp and dec_shx.hpp, included by native_mmvq.cu

namespace {

constexpr int kShxThreads = 256, kShxWarps = kShxThreads / WARP;
constexpr int kShxEmbd = 2560, kShxFf = 640, kShxChunks = kShxFf / Q8K, kShxMaxT = 8;
constexpr int kShxXBlocks = kShxEmbd / Q8K, kShxRowBlocks = kShxEmbd / QK, kShxDownBlocks = kShxFf / Q8K;
constexpr int kShxRD = 2;                                            // down rows a warp
constexpr int kShxItems = 2 * kShxFf + kShxMaxT;                     // a warp each: gate/up rows, then scalar gates
constexpr int kShxBlocks = (kShxItems + kShxWarps - 1) / kShxWarps;
static_assert(kShxBlocks * kShxWarps * kShxRD >= kShxEmbd, "every down row has a warp");
// sync words: [0, kShxChunks) rows done per chunk, then chunks and gates ready, then blocks out
constexpr int kShxReady = kShxChunks, kShxExit = kShxChunks + 1;
// native_mmvq's exact layout: four warps of one row; IQ4_XS virtual threads 0..79 and IQ4_NL 0..39 have a block
constexpr int kShxYA = 3, kShxYD = 2;
static_assert(IQ4XSTraits::BPI >= kShxRowBlocks && (kShxYA * WARP) / IQ4XSTraits::T >= kShxRowBlocks, "one pass");
static_assert(SmallTraits<IQ4NLBlock, 4>::BPI >= kShxDownBlocks && (kShxYD * WARP) / 2 >= kShxDownBlocks, "one pass");

// native_swiglu_quantize_q8_1_kernel's block c of every token (the warp's lane = the element)
template<int NC>
__device__ __forceinline__ void shx_swiglu_chunk(const float* gate, const float* up, Q81Block* hq, int c, int lane) {
    float gv[NC], uv[NC];
#pragma unroll
    for (int t = 0; t < NC; ++t) {
        gv[t] = __ldcg(gate + t * kShxFf + c * Q8K + lane);
        uv[t] = __ldcg(up + t * kShxFf + c * Q8K + lane);
    }
#pragma unroll
    for (int t = 0; t < NC; ++t) {
        const int i = t * kShxFf + c * Q8K + lane;
        const float gi = gv[t];
        const float xi = __fmul_rn(__fdividef(gi, __fadd_rn(1.0f, __expf(-gi))), uv[t]);
        const float amax = warp_max(fabsf(xi));
        const float sum = warp_sum(xi);
        const float d = amax / 127.0f;
        const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
        hq[i / Q8K].qs[lane] = q;
        if (lane == 0) hq[i / Q8K].ds = make_half2(d, sum);
    }
}

// One warp's share: item gw of the gate/up rows (native_mmvq_multi_kernel's exact layout, virtual thread 32 y + lane,
// warp sums added in warp order, then warp_sum) or a scalar gate (bf16_f32_mmvf_kernel<256>'s order), then down rows
// kShxRD gw.. once every SwiGLU chunk and gate is in.  The weights are read before pdl_wait.
struct ShxShared {
    Q81Block hq[kShxMaxT * kShxChunks];
    float gt[kShxMaxT];
};

template<int NC>
__device__ __forceinline__ void shx_warp(const ShxArgs& a, ShxShared& sm, int gw, int lane) {
    const int r_a = gw < kShxFf ? gw : gw - kShxFf;
    const bool row_a = gw < 2 * kShxFf, gate_a = !row_a && gw - 2 * kShxFf < NC;
    const auto* wrow = static_cast<const IQ4XSBlock*>(gw < kShxFf ? a.gate_w : a.up_w) + (size_t) r_a * kShxRowBlocks;
    IQ4XSTraits::W wa[kShxYA];
    if (row_a) {
#pragma unroll
        for (int y = 0; y < kShxYA; ++y) {
            const int tid = WARP * y + lane, kbx = tid / IQ4XSTraits::T;
            if (kbx < kShxRowBlocks) wa[y] = IQ4XSTraits::load(wrow + kbx, IQ4XSTraits::kqs(tid));
        }
    }
    constexpr int GP = kShxEmbd / 2 / 256;   // pairs per virtual thread of the 256-thread gate sum
    uint32_t wg[kShxWarps][GP];
    if (gate_a) {
        const uint32_t* w2 = reinterpret_cast<const uint32_t*>(a.gate_inp);
#pragma unroll
        for (int v = 0; v < kShxWarps; ++v)
#pragma unroll
            for (int k = 0; k < GP; ++k) wg[v][k] = w2[WARP * v + lane + 256 * k];
    }
    const bool has_d = gw * kShxRD < kShxEmbd;
    int wq[kShxRD][kShxYD][2];
    half wd[kShxRD][kShxYD];
    if (has_d) {
        const auto* dw = static_cast<const IQ4NLBlock*>(a.down_w) + (size_t) gw * kShxRD * kShxDownBlocks;
#pragma unroll
        for (int i = 0; i < kShxRD; ++i)
#pragma unroll
            for (int y = 0; y < kShxYD; ++y) {
                const int tid = WARP * y + lane, kbx = tid / 2, kqs = 2 * (tid % 2);
                if (kbx < kShxDownBlocks) {
                    const IQ4NLBlock* b = dw + i * kShxDownBlocks + kbx;
                    wq[i][y][0] = load_int_b2(b->qs, kqs);
                    wq[i][y][1] = load_int_b2(b->qs, kqs + 1);
                    wd[i][y] = b->d;
                }
            }
    }
    pdl_wait();
    if (row_a) {
        const Q81Block* xq = static_cast<const Q81Block*>(a.xq);
        float tmp[WARPS][NC];
#pragma unroll
        for (int y = 0; y < WARPS; ++y)
#pragma unroll
            for (int j = 0; j < NC; ++j) tmp[y][j] = 0.0f;
#pragma unroll
        for (int y = 0; y < kShxYA; ++y) {
            const int tid = WARP * y + lane, kbx = tid / IQ4XSTraits::T;
            if (kbx < kShxRowBlocks)
#pragma unroll
                for (int j = 0; j < NC; ++j)
                    tmp[y][j] += IQ4XSTraits::apply(wa[y], xq + j * kShxXBlocks + kbx * IQ4XSTraits::KBY,
                                                    IQ4XSTraits::kqs(tid));
        }
        float s[NC];
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            s[j] = tmp[0][j];
#pragma unroll
            for (int y = 1; y < WARPS; ++y) s[j] += tmp[y][j];
            s[j] = warp_sum(s[j]);
        }
        float* dst = gw < kShxFf ? a.gate : a.up;
        unsigned int prev = 0;
        if (lane == 0) {
#pragma unroll
            for (int j = 0; j < NC; ++j) dst[j * kShxFf + r_a] = s[j];
            __threadfence();
            prev = atomicAdd(a.sync + r_a / Q8K, 1u);
        }
        prev = __shfl_sync(0xffffffffu, prev, 0);
        if (prev == 2 * Q8K - 1) {   // the chunk's last row: its SwiGLU blocks
            const int c = r_a / Q8K;
            __threadfence();
            shx_swiglu_chunk<NC>(a.gate, a.up, static_cast<Q81Block*>(a.hq), c, lane);
            __threadfence();
            __syncwarp();
            if (lane == 0) {
                a.sync[c] = 0u;
                atomicAdd(a.sync + kShxReady, 1u);
            }
        }
    } else if (gate_a) {
        const int t = gw - 2 * kShxFf;
        const float2* x2 = reinterpret_cast<const float2*>(a.x + (size_t) t * kShxEmbd);
        float part[kShxWarps];
#pragma unroll
        for (int v = 0; v < kShxWarps; ++v) {
            float acc = 0.0f;
#pragma unroll
            for (int k = 0; k < GP; ++k) {
                const float2 in = x2[WARP * v + lane + 256 * k];
                acc = __fmaf_rn(f32_from_bf16((uint16_t) wg[v][k]), in.x, acc);
                acc = __fmaf_rn(f32_from_bf16((uint16_t) (wg[v][k] >> 16)), in.y, acc);
            }
            part[v] = warp_sum(acc);
        }
        float s = 0.0f;
#pragma unroll
        for (int v = 0; v < kShxWarps; ++v)
            if (lane == v) s = part[v];
        s = warp_sum(s);
        if (lane == 0) {
            a.g[t] = s;
            __threadfence();
            atomicAdd(a.sync + kShxReady, 1u);
        }
    }
    __syncthreads();   // one poll a block
    if (threadIdx.x == 0) {
        while (*(volatile unsigned int*) (a.sync + kShxReady) < (unsigned) (kShxChunks + NC)) __nanosleep(32);
        __threadfence();
    }
    __syncthreads();
    for (int i = threadIdx.x; i < NC * kShxChunks * 9; i += kShxThreads)
        reinterpret_cast<int*>(sm.hq)[i] = __ldcg(static_cast<const int*>(a.hq) + i);
    if (threadIdx.x < NC) sm.gt[threadIdx.x] = __fdividef(1.0f, 1.0f + __expf(-__ldcg(a.g + threadIdx.x)));
    __syncthreads();
    if (!has_d) return;
    const Q81Block* hq = sm.hq;
    float tmp[kShxRD][WARPS][NC];
#pragma unroll
    for (int i = 0; i < kShxRD; ++i)
#pragma unroll
        for (int y = 0; y < WARPS; ++y)
#pragma unroll
            for (int j = 0; j < NC; ++j) tmp[i][y][j] = 0.0f;
#pragma unroll
    for (int y = 0; y < kShxYD; ++y) {
        const int tid = WARP * y + lane, kbx = tid / 2, kqs = 2 * (tid % 2);
        if (kbx < kShxDownBlocks) {
#pragma unroll
            for (int j = 0; j < NC; ++j) {
                const Q81Block* xb = hq + j * kShxChunks + kbx;
                const int* q8 = reinterpret_cast<const int*>(xb->qs) + kqs;
                const int u0 = q8[0], u1 = q8[1], u4 = q8[4], u5 = q8[5];
                const float d8 = __low2float(xb->ds);
#pragma unroll
                for (int i = 0; i < kShxRD; ++i) {   // small_q8_dot (IQ4_NL)
                    int sumi = 0;
                    const int2 v0 = iq4_table_lookup(wq[i][y][0]);
                    sumi = STRATA_DP4A(v0.x, u0, sumi);
                    sumi = STRATA_DP4A(v0.y, u4, sumi);
                    const int2 v1 = iq4_table_lookup(wq[i][y][1]);
                    sumi = STRATA_DP4A(v1.x, u1, sumi);
                    sumi = STRATA_DP4A(v1.y, u5, sumi);
                    const float d = __half2float(wd[i][y]) * d8;
                    tmp[i][y][j] += d * sumi;
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < kShxRD; ++i)
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            float s = tmp[i][0][j];
#pragma unroll
            for (int y = 1; y < WARPS; ++y) s += tmp[i][y][j];
            s = warp_sum(s);
            if (lane == i) a.out[(size_t) j * kShxEmbd + gw * kShxRD + i] = s * sm.gt[j];
        }
}

template<int NC>
__global__ void __launch_bounds__(kShxThreads, 1) dec_shx_kernel(ShxArgs a) {
    __shared__ ShxShared sm;
    shx_warp<NC>(a, sm, (int) blockIdx.x * kShxWarps + (int) threadIdx.x / WARP, (int) threadIdx.x % WARP);
    __syncthreads();   // the last block out leaves the words zeroed
    if (threadIdx.x == 0 && atomicAdd(a.sync + kShxExit, 1u) == gridDim.x - 1) {
        a.sync[kShxReady] = 0u;
        a.sync[kShxExit] = 0u;
    }
}

}  // namespace

bool dec_shx_ok(int gate_type, int up_type, int down_type, int64_t n_embd, int64_t n_ff) {
    if (gate_type != 23 || up_type != 23 || down_type != 20 || n_embd != kShxEmbd || n_ff != kShxFf) return false;
    static int known[64] = {};   // per device: every block co-resident (the down rows wait for the others' chunks)
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) {
        (void) cudaGetLastError();
        return false;
    }
    if (known[dev] == 0) {
        int sms = 0, per = 0;
        const bool q = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess &&
                       cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, dec_shx_kernel<kShxMaxT>, kShxThreads, 0) ==
                           cudaSuccess;
        if (!q) (void) cudaGetLastError();
        known[dev] = q && sms * per >= kShxBlocks ? 1 : -1;
    }
    return known[dev] > 0;
}

void dec_shx(const ShxArgs& a, void* stream) {
    if (a.n_tok < 1 || a.n_tok > kShxMaxT || !a.gate_w || !a.up_w || !a.down_w || !a.gate_inp || !a.x || !a.xq || !a.gate ||
        !a.up || !a.hq || !a.g || !a.out || !a.sync || !stream)
        throw std::invalid_argument("dec_shx: 1..8 tokens, weights, scratch, sync words and a stream");
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    const dim3 grid((unsigned) kShxBlocks), block(kShxThreads);
    switch (a.n_tok) {
        case 1: launch_pdl(dec_shx_kernel<1>, grid, block, 0, s, a); break;
        case 2: launch_pdl(dec_shx_kernel<2>, grid, block, 0, s, a); break;
        case 3: launch_pdl(dec_shx_kernel<3>, grid, block, 0, s, a); break;
        case 4: launch_pdl(dec_shx_kernel<4>, grid, block, 0, s, a); break;
        case 5: launch_pdl(dec_shx_kernel<5>, grid, block, 0, s, a); break;
        case 6: launch_pdl(dec_shx_kernel<6>, grid, block, 0, s, a); break;
        case 7: launch_pdl(dec_shx_kernel<7>, grid, block, 0, s, a); break;
        default: launch_pdl(dec_shx_kernel<8>, grid, block, 0, s, a); break;
    }
    launch_check();
}
