// src/kernels/cuda/dec_moe.cuh - see include/strata/kernels/dec_moe.hpp.  Included at the end of iq_kernels.cu:
// the peer's rows use that file's row dots (Split, row_dot_multi, the sub-warp down dots) and q8_1_store.
#include "strata/kernels/dec_moe.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cooperative_groups.h>

namespace strata::kernels {
namespace {

constexpr int kDmThreads = 256;

// ---------------------------------------------------------------- tier plan
// Entry i (tid < n) of kind `want`: its group (distinct ids in routing order) and position.  s_* are block arrays.
// Writes start[0..ng], and per entry pos[i] (its place in the entry list) for entries of kind `want`, -1 else.
__device__ __forceinline__ void dm_group(const int32_t* s_ids, const int32_t* s_kind, int n, int want, int tid,
                                         int32_t* s_first, int32_t* s_cnt, int32_t* s_gstart, int32_t* s_gidx,
                                         int32_t* s_pos, int32_t* start, int32_t* s_ng, int32_t* s_ne) {
    int first = tid, rank = 0, cnt = 0;
    const bool mine = tid < n && s_kind[tid] == want;
    if (mine) {
        const int32_t e = s_ids[tid];
        for (int j = 0; j < n; ++j)
            if (s_ids[j] == e) {
                if (j < first) first = j;
                if (j < tid) ++rank;
                ++cnt;
            }
    }
    if (tid < kDmCap) {
        s_first[tid] = mine && first == tid;
        s_cnt[tid] = mine && first == tid ? cnt : 0;
    }
    __syncthreads();
    if (mine && first == tid) {
        int gi = 0, es = 0;
        for (int j = 0; j < tid; ++j) { gi += s_first[j]; es += s_cnt[j]; }
        s_gidx[tid] = gi;
        s_gstart[tid] = es;
        start[gi] = es;
    }
    __syncthreads();
    if (tid < kDmCap) s_pos[tid] = mine ? s_gstart[first] + rank : -1;
    if (tid == 0) {
        int ng = 0, ne = 0;
        for (int j = 0; j < n; ++j) { ng += s_first[j]; ne += s_cnt[j]; }
        start[ng] = ne;
        *s_ng = ng;
        *s_ne = ne;
    }
    __syncthreads();
}

// Block 0 plans CUDA0's groups and publishes the ring's header; the other blocks quantize the FFN input (256 values
// each) into the device q8_1 rows and, for the tiers that need them, into mapped memory.  The last block to finish
// rings both doorbells.
__global__ void __launch_bounds__(kDmThreads) dm_route_kernel(DmRouteArgs a) {
    __shared__ int32_t s_ids[kDmCap], s_kind[kDmCap], s_first[kDmCap], s_cnt[kDmCap], s_gstart[kDmCap];
    __shared__ int32_t s_gidx[kDmCap], s_pos[kDmCap];
    __shared__ int32_t s_ncpu, s_npeer, s_ng, s_ne;
    const int tid = threadIdx.x, n = a.n_tok * a.k;
    if (tid == 0) { s_ncpu = 0; s_npeer = 0; }
    __syncthreads();
    if (tid < n) {
        const int32_t e = a.ids[tid];
        int kd = -1;
        if (e >= 0 && e < a.n_expert) {
            if (a.res0[e] >= 0) kd = 0;
            else if (a.res1 != nullptr && a.res1[e] >= 0) kd = 2;
        }
        s_ids[tid] = e;
        s_kind[tid] = kd;
        if (kd < 0) atomicAdd(&s_ncpu, 1);
        else if (kd == 2) atomicAdd(&s_npeer, 1);
    }
    __syncthreads();
    if (blockIdx.x == 0) {
        DmPlan* p = a.plan;
        dm_group(s_ids, s_kind, n, 0, tid, s_first, s_cnt, s_gstart, s_gidx, s_pos, p->start, &s_ng, &s_ne);
        if (tid < n) {
            const int32_t kd = s_kind[tid];
            p->kind[tid] = kd;
            if (kd == 0) {
                p->dst[s_pos[tid]] = tid;
                p->tok[s_pos[tid]] = tid / a.k;
                if (s_first[tid]) {
                    const int32_t slot = a.res0[s_ids[tid]];
                    p->ptr[s_gidx[tid]] = (unsigned long long) (a.cache_base + (a.slot_off ? (size_t) a.slot_off[slot]
                                                                                              : (size_t) slot * (size_t) a.blob));
                }
            }
            a.ring_m->ids[tid] = s_ids[tid];
            a.ring_m->kind[tid] = kd;
        }
        if (tid == 0) {
            p->counts[0] = s_ng;
            p->counts[1] = s_ne;
            p->counts[2] = s_npeer;
            p->counts[3] = s_ncpu;
            DmRing* r = a.ring_m;
            r->n_tok = a.n_tok;
            r->layer = a.layer;
            r->n_vram = s_ne;
            r->n_peer = s_npeer;
            r->n_cpu = s_ncpu;
        }
    } else {
        const long long i = (long long) (blockIdx.x - 1) * kDmThreads + tid;
        const long long nv = (long long) a.n_tok * a.n_embd;
        if (i < nv) {   // n_embd is a multiple of 256: whole warps
            const float xi = a.x[i];
            q8_1_store(xi, (block_q8_1*) a.xq, i);
            if (s_ncpu > 0) a.x_m[i] = xi;
        }
        if (s_npeer > 0) {
            __syncthreads();
            // this block's 8 q8_1 blocks (288 bytes) to mapped memory, 16 bytes per thread
            const long long b0 = (long long) (blockIdx.x - 1) * (kDmThreads / 32);
            if (b0 * 32 < nv && tid < 18) {
                const uint4* src = (const uint4*) ((const block_q8_1*) a.xq + b0);
                uint4* dst = (uint4*) ((block_q8_1*) a.xq_m + b0);
                dst[tid] = src[tid];
            }
        }
    }
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        const unsigned int arrived = atomicAdd(a.ctr, 1u) + 1u;
        if (arrived == gridDim.x) {
            *a.ctr = 0u;
            __threadfence_system();
            *(volatile uint32_t*) a.peer_seq_m = a.ring;
            *(volatile uint32_t*) a.host_seq_m = a.ring;
            __threadfence_system();
        }
    }
}

__global__ void dm_wait_kernel(const DmPlan* plan, const volatile uint32_t* peer_done, const volatile uint32_t* host_flag,
                               uint32_t ring) {
    if (plan->counts[2] > 0) while (*peer_done < ring) strata_spin_pause();
    if (plan->counts[3] > 0) while (*host_flag < ring) strata_spin_pause();
    __threadfence_system();
}

// ---------------------------------------------------------------- the grouped rows, one work item at a time
// native_gu_multi_kernel's body for block row `rb` of one group (entries [e0, e1)): the same calls in the same order.
template<int TG, bool STAGE_GRID>
__device__ __forceinline__ void dm_gu_item(const uint8_t* blob, int e0, int e1, const int32_t* __restrict__ ent_tok,
                                           const block_q8_1* __restrict__ xq, const NativeExpertLayout& L, int rb,
                                           float* __restrict__ gate, float* __restrict__ up, const uint32_t* s_grid,
                                           int warp, int lane) {
    const int row = rb * GU_ROWS + warp;
    if (row >= 2 * L.n_ff) return;
    const bool is_up = row >= L.n_ff;
    const int r = is_up ? row - (int) L.n_ff : row;
    const uint8_t* wr = blob + (is_up ? L.up_off : 0) + (size_t) r * L.gu_row;
    const int nb = (int) (L.n_embd / Fmt<TG>::qk), xb = (int) (L.n_embd / 32);
    float* dst = is_up ? up : gate;
    for (int e = e0; e < e1; e += GRP_NC) {
        const int n = min(GRP_NC, e1 - e);
        if (n == 1) {
            const int off1[1] = { ent_tok[e] * xb };
            float s1[1];
            row_dot_multi<TG, 1, true, STAGE_GRID>(wr, xq, off1, 1, nb, lane, s1, s_grid);
            if (lane == 0) dst[(size_t) e * L.n_ff + r] = s1[0];
        } else if (n == 2) {
            const int off2[2] = { ent_tok[e] * xb, ent_tok[e + 1] * xb };
            float s2[2];
            row_dot_multi<TG, 2, true, STAGE_GRID>(wr, xq, off2, 2, nb, lane, s2, s_grid);
            if (lane < 2) dst[(size_t) (e + lane) * L.n_ff + r] = s2[lane];
        } else {
            int off[GRP_NC];
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c) off[c] = ent_tok[e + min(c, n - 1)] * xb;
            float s[GRP_NC];
            row_dot_multi<TG, GRP_NC, false, STAGE_GRID>(wr, xq, off, n, nb, lane, s, s_grid);
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c)
                if (c < n && lane == c) dst[(size_t) (e + c) * L.n_ff + r] = s[c];
        }
    }
}

// rows per down work item: native_down_multi_kernel's block rows at n_ff == 640
template<int TD> __host__ __device__ constexpr int dm_down_rows() { return TD == 20 ? 32 : TD == 42 ? 16 : 8; }

// The SwiGLU product and its q8_1 blocks for entries [e0, e1) into shared memory (swiglu_q8_1_entries_kernel's values).
__device__ __forceinline__ void dm_swiglu_q8(const float* __restrict__ gate, const float* __restrict__ up, int e0,
                                             int e1, int n_ff, block_q8_1* s_hq, int tid) {
    const int nv = (e1 - e0) * n_ff;
    const size_t base = (size_t) e0 * n_ff;
    for (int i = tid; i < nv; i += kDmThreads) {   // n_ff is a multiple of 32: whole warps
        const float g = gate[base + i];
        const float h = __fmul_rn(g / (1.0f + __expf(-g)), up[base + i]);
        q8_1_store(h, s_hq, i);
    }
}

// native_down_multi_kernel's body (hb == 20) for block row `rb` of one group, from the group's q8_1 rows in s_hq
// (entry e at s_hq + (e - e0) * 20); row r of entry e lands in s_out[(e - e0) * ROWS + r - rb * R], R the item's
// rows (dm_down_rows) and ROWS the stride of s_out's entries.
template<int TD>
__device__ __forceinline__ void dm_down_item(const uint8_t* blob, int e0, int e1, const block_q8_1* s_hq,
                                             const NativeExpertLayout& L, int rb, float* s_out, int tid, int ROWS) {
    const int nb = (int) (L.n_ff / Fmt<TD>::qk);
    (void) nb;
    if constexpr (TD == 20) {
        const int rl = tid >> 3, r = rb * 32 + rl, t = tid & 7;
        if (r >= L.n_embd) return;
        const uint8_t* wr = blob + L.down_off + (size_t) r * L.d_row;
        for (int e = e0; e < e1; e += GRP_NC) {
            const int n = min(GRP_NC, e1 - e), b = (e - e0) * 20;
            if (n == 1) {
                const int off1[1] = { b };
                float s1[1];
                row_dot_40_sub8<TD, 1, true>(wr, s_hq, off1, 1, t, s1);
                if (t == 0) s_out[(e - e0) * ROWS + rl] = s1[0];
            } else if (n == 2) {
                const int off2[2] = { b, b + 20 };
                float s2[2];
                row_dot_40_sub8<TD, 2, true>(wr, s_hq, off2, 2, t, s2);
                if (t < 2) s_out[(e - e0 + t) * ROWS + rl] = s2[t];
            } else {
                int off[GRP_NC];
#pragma unroll
                for (int c = 0; c < GRP_NC; ++c) off[c] = b + min(c, n - 1) * 20;
                float s[GRP_NC];
                row_dot_40_sub8<TD, GRP_NC>(wr, s_hq, off, n, t, s);
#pragma unroll
                for (int c = 0; c < GRP_NC; ++c)
                    if (c < n && t == c) s_out[(e - e0 + c) * ROWS + rl] = s[c];
            }
        }
    } else if constexpr (TD == 42) {
        const int rl = tid >> 4, r = rb * 16 + rl, t = tid & 15;
        if (r >= L.n_embd) return;
        const uint8_t* wr = blob + L.down_off + (size_t) r * L.d_row;
        for (int e = e0; e < e1; e += GRP_NC) {
            const int n = min(GRP_NC, e1 - e), b = (e - e0) * 20;
            if (n == 1) {
                const int off1[1] = { b };
                float s1[1];
                row_dot_20_sub16<TD, 1, true>(wr, s_hq, off1, 1, t, s1);
                if (t == 0) s_out[(e - e0) * ROWS + rl] = s1[0];
            } else if (n == 2) {
                const int off2[2] = { b, b + 20 };
                float s2[2];
                row_dot_20_sub16<TD, 2, true>(wr, s_hq, off2, 2, t, s2);
                if (t < 2) s_out[(e - e0 + t) * ROWS + rl] = s2[t];
            } else {
                int off[GRP_NC];
#pragma unroll
                for (int c = 0; c < GRP_NC; ++c) off[c] = b + min(c, n - 1) * 20;
                float s[GRP_NC];
                row_dot_20_sub16<TD, GRP_NC>(wr, s_hq, off, n, t, s);
#pragma unroll
                for (int c = 0; c < GRP_NC; ++c)
                    if (c < n && t == c) s_out[(e - e0 + c) * ROWS + rl] = s[c];
            }
        }
    } else {
        const int warp = tid >> 5, lane = tid & 31, r = rb * 8 + warp;
        if (r >= L.n_embd) return;
        const uint8_t* wr = blob + L.down_off + (size_t) r * L.d_row;
        for (int e = e0; e < e1; e += GRP_NC) {
            const int n = min(GRP_NC, e1 - e), b = (e - e0) * 20;
            if (n == 1) {
                const int off1[1] = { b };
                float s1[1];
                row_dot_multi<TD, 1, true>(wr, s_hq, off1, 1, nb, lane, s1);
                if (lane == 0) s_out[(e - e0) * ROWS + warp] = s1[0];
            } else if (n == 2) {
                const int off2[2] = { b, b + 20 };
                float s2[2];
                row_dot_multi<TD, 2, true>(wr, s_hq, off2, 2, nb, lane, s2);
                if (lane < 2) s_out[(e - e0 + lane) * ROWS + warp] = s2[lane];
            } else {
                int off[GRP_NC];
#pragma unroll
                for (int c = 0; c < GRP_NC; ++c) off[c] = b + min(c, n - 1) * 20;
                float s[GRP_NC];
                row_dot_multi<TD, GRP_NC>(wr, s_hq, off, n, nb, lane, s);
#pragma unroll
                for (int c = 0; c < GRP_NC; ++c)
                    if (c < n && lane == c) s_out[(e - e0 + c) * ROWS + warp] = s[c];
            }
        }
    }
}

constexpr int kDmMaxEnt = kVerifyMaxT;   // entries of one group: at most one per token

struct DmShared {
    alignas(16) uint32_t grid[2048];                       // the format's codebook grid (stage_iq_grid)
    alignas(16) block_q8_1 hq[kDmMaxEnt * 20];             // one group's SwiGLU rows (n_ff == 640)
    alignas(16) float out[kDmMaxEnt * 64];                 // one item's rows per entry
    int32_t ids[kDmCap], kind[kDmCap], first[kDmCap], cnt[kDmCap], gstart[kDmCap], gidx[kDmCap], pos[kDmCap];
    int32_t start[kDmCap + 1], tok[kDmCap], dst[kDmCap];
    unsigned long long gptr[kDmCap];
    int32_t ng, ne;
};

// gate/up for every (group, block row) item, item = blockIdx-strided
template<int TG>
__device__ void dm_gu_all(DmShared& sm, const block_q8_1* xq, const NativeExpertLayout& L, float* gate, float* up,
                          int item0, int stride) {
    constexpr bool SG = kStageIqGrid<TG>;
    const uint32_t* s_grid = stage_iq_grid<TG, SG>(sm.grid, threadIdx.x, kDmThreads);
    const int nrb = (int) ((2 * L.n_ff + GU_ROWS - 1) / GU_ROWS);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int it = item0; it < sm.ng * nrb; it += stride) {
        const int g = it / nrb, rb = it - g * nrb;
        dm_gu_item<TG, SG>((const uint8_t*) sm.gptr[g], sm.start[g], sm.start[g + 1], sm.tok, xq, L, rb, gate, up,
                           s_grid, warp, lane);
    }
}

// down for every (group, span of kDmSpan rows) item; each item's rows go to row dst[e] of `out` (n_embd floats)
constexpr int kDmSpan = 64;
template<int TD>
__device__ void dm_down_all(DmShared& sm, const NativeExpertLayout& L, const float* gate, const float* up, float* out,
                            int item0, int stride) {
    constexpr int R = dm_down_rows<TD>(), SUB = kDmSpan / R;
    const int nsp = (int) ((L.n_embd + kDmSpan - 1) / kDmSpan);
    const int tid = threadIdx.x;
    int g_staged = -1;
    for (int it = item0; it < sm.ng * nsp; it += stride) {
        const int g = it / nsp, sp = it - g * nsp;
        const int e0 = sm.start[g], e1 = sm.start[g + 1];
        __syncthreads();                         // the previous item's reads of hq and out are done
        if (g != g_staged) {
            dm_swiglu_q8(gate, up, e0, e1, (int) L.n_ff, sm.hq, tid);
            g_staged = g;
            __syncthreads();
        }
        for (int sb = 0; sb < SUB; ++sb)
            dm_down_item<TD>((const uint8_t*) sm.gptr[g], e0, e1, sm.hq, L, sp * SUB + sb, sm.out + sb * R, tid, kDmSpan);
        __syncthreads();
        // kDmSpan floats per entry, 16 bytes per thread
        constexpr int q = kDmSpan / 4;
        const int nq = (e1 - e0) * q;
        for (int i = tid; i < nq; i += kDmThreads) {
            const int e = i / q, c = i - e * q;
            const int r0 = sp * kDmSpan + c * 4;
            if (r0 < L.n_embd) {
                const float4 v = *(const float4*) &sm.out[e * kDmSpan + c * 4];
                *(float4*) (out + (size_t) sm.dst[e0 + e] * L.n_embd + r0) = v;
            }
        }
    }
}

#define STRATA_DM_GU_FMTS(X) X(16) X(17) X(18) X(21) X(22) X(23) X(29) X(42)
#define STRATA_DM_D_FMTS(X) X(20) X(42) X(23)

__device__ void dm_gu_dispatch(DmShared& sm, const block_q8_1* xq, const NativeExpertLayout& L, float* gate,
                               float* up, int item0, int stride) {
    switch (L.gu_type) {
#define STRATA_DM_GU(T) case T: dm_gu_all<T>(sm, xq, L, gate, up, item0, stride); break;
        STRATA_DM_GU_FMTS(STRATA_DM_GU)
#undef STRATA_DM_GU
        default: break;
    }
}
__device__ void dm_down_dispatch(DmShared& sm, const NativeExpertLayout& L, const float* gate, const float* up,
                                 float* out, int item0, int stride) {
    switch (L.d_type) {
#define STRATA_DM_D(T) case T: dm_down_all<T>(sm, L, gate, up, out, item0, stride); break;
        STRATA_DM_D_FMTS(STRATA_DM_D)
#undef STRATA_DM_D
        default: break;
    }
}

// The peer window.  Per ring: wait for CUDA0's doorbell, group the peer's entries (every block, the same lists),
// copy the q8_1 rows to the device, gate/up, a grid barrier, then SwiGLU + down with the rows written straight to
// mapped memory; the last block to finish raises `done`.
__global__ void __launch_bounds__(kDmThreads) dm_peer_kernel(DmPeerArgs a) {
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();
    __shared__ DmShared sm;
    const int tid = threadIdx.x;
    const int nb = gridDim.x;
    float* gate = (float*) a.scratch;
    float* up = gate + (size_t) kDmCap * a.n_ff;
    block_q8_1* xq = (block_q8_1*) (up + (size_t) kDmCap * a.n_ff);
    __shared__ uint32_t s_seq;
    unsigned int served = 0;
    auto stamp = [&](int s, int i) {
        if (a.stamps_m != nullptr && blockIdx.x == 0 && threadIdx.x == 0) {
            unsigned long long t;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
            a.stamps_m[s * 8 + i] = t;
        }
    };
    DmRing* hdr = (DmRing*) (xq + (size_t) kVerifyMaxT * (a.n_embd / 32));
    for (int s = 0; s < a.steps; ++s) {
        const uint32_t ring = (uint32_t) s + 1u;
        // block 0 alone reads host memory: it waits for the ring, then copies its header and q8_1 rows to the device
        if (blockIdx.x == 0) {
            if (tid == 0) {
                uint32_t v;
                while ((v = *a.seq_m) < ring) strata_spin_pause();
                __threadfence_system();
                s_seq = v;
            }
            __syncthreads();
            stamp(s, 0);
            if (s_seq != 0xffffffffu) {
                const uint4* src = (const uint4*) (a.rings_m + s);
                for (int i = tid; i < (int) (sizeof(DmRing) / 16); i += kDmThreads) ((uint4*) hdr)[i] = __ldcv(src + i);
                __syncthreads();
                if (hdr->n_peer > 0) {
                    const int n16 = hdr->n_tok * (a.n_embd / 32) * (int) sizeof(block_q8_1) / 16;
                    const uint4* xs = (const uint4*) (a.xq_m + (size_t) s * a.xq_stride);
                    for (int i = tid; i < n16; i += kDmThreads) ((uint4*) xq)[i] = __ldcv(xs + i);
                }
            } else if (tid == 0) {
                hdr->n_tok = -1;
            }
            __threadfence();
        }
        grid.sync();
        stamp(s, 1);
        if (hdr->n_tok < 0) return;              // released: the window is being abandoned
        if (hdr->n_peer == 0) {
            grid.sync();                         // block 0 rewrites the header next ring
            continue;
        }
        const int n_tok = hdr->n_tok, layer = hdr->layer, n = n_tok * a.k;
        if (tid < n) {
            sm.ids[tid] = hdr->ids[tid];
            sm.kind[tid] = hdr->kind[tid];
        }
        __syncthreads();
        dm_group(sm.ids, sm.kind, n, 2, tid, sm.first, sm.cnt, sm.gstart, sm.gidx, sm.pos, sm.start, &sm.ng, &sm.ne);
        if (tid < n && sm.kind[tid] == 2) {
            sm.tok[sm.pos[tid]] = tid / a.k;
            sm.dst[sm.pos[tid]] = tid;
            if (sm.first[tid]) sm.gptr[sm.gidx[tid]] = a.ptr[(size_t) layer * a.n_expert + sm.ids[tid]];
        }
        __syncthreads();
        bool bad = false;
        for (int g = 0; g < sm.ng; ++g) bad |= sm.gptr[g] == 0ull;
        stamp(s, 2);
        const NativeExpertLayout L = a.lay[layer];
        if (!bad) dm_gu_dispatch(sm, xq, L, gate, up, blockIdx.x, nb);
        stamp(s, 3);
        grid.sync();
        stamp(s, 4);
        if (!bad) dm_down_dispatch(sm, L, gate, up, a.rows_m, blockIdx.x, nb);
        stamp(s, 5);
        __threadfence_system();
        __syncthreads();
        stamp(s, 6);
        ++served;
        if (tid == 0) {
            const unsigned int arrived = atomicAdd(a.arrive, 1u) + 1u;
            if (arrived == served * (unsigned int) nb) {
                __threadfence_system();
                *(volatile uint32_t*) a.done_m = ring;
                __threadfence_system();
                if (a.stamps_m != nullptr) {
                    unsigned long long t;
                    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
                    a.stamps_m[s * 8 + 7] = t;
                }
            }
        }
    }
}

// the tests: the same work items over given groups, in two launches
__global__ void __launch_bounds__(kDmThreads) dm_test_gu_kernel(NativeExpertLayout L, const unsigned long long* grp_ptr,
                                                                 const int32_t* start, int n_groups, const int32_t* tok,
                                                                 const block_q8_1* xq, float* gate, float* up) {
    __shared__ DmShared sm;
    if (threadIdx.x == 0) sm.ng = n_groups;
    for (int i = threadIdx.x; i <= n_groups; i += kDmThreads) {
        sm.start[i] = start[i];
        if (i < n_groups) sm.gptr[i] = grp_ptr[i];
    }
    for (int i = threadIdx.x; i < kDmCap; i += kDmThreads) sm.tok[i] = tok[i];
    __syncthreads();
    dm_gu_dispatch(sm, xq, L, gate, up, blockIdx.x, gridDim.x);
}
__global__ void __launch_bounds__(kDmThreads) dm_test_down_kernel(NativeExpertLayout L, const unsigned long long* grp_ptr,
                                                                   const int32_t* start, int n_groups, const int32_t* dst,
                                                                   const float* gate, const float* up, float* out) {
    __shared__ DmShared sm;
    if (threadIdx.x == 0) sm.ng = n_groups;
    for (int i = threadIdx.x; i <= n_groups; i += kDmThreads) {
        sm.start[i] = start[i];
        if (i < n_groups) sm.gptr[i] = grp_ptr[i];
    }
    for (int i = threadIdx.x; i < kDmCap; i += kDmThreads) sm.dst[i] = dst[i];
    __syncthreads();
    dm_down_dispatch(sm, L, gate, up, out, blockIdx.x, gridDim.x);
}

}  // namespace

int dec_moe_route_blocks(int n_tok, int n_embd) { return 1 + (n_tok * n_embd + kDmThreads - 1) / kDmThreads; }

void dec_moe_route(const DmRouteArgs& a, void* stream) {
    if (a.n_tok * a.k > kDmCap || a.n_embd % kDmThreads != 0) {
        std::fprintf(stderr, "dec_moe_route: %d entries / n_embd %d\n", a.n_tok * a.k, a.n_embd);
        std::exit(1);
    }
    dm_route_kernel<<<(unsigned) dec_moe_route_blocks(a.n_tok, a.n_embd), kDmThreads, 0, (cudaStream_t) stream>>>(a);
    check("dec_moe_route");
}

void dec_moe_wait(const DmPlan* plan, const volatile uint32_t* peer_done_m, const volatile uint32_t* host_flag_m,
                  uint32_t ring, void* stream) {
    dm_wait_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(plan, peer_done_m, host_flag_m, ring);
    check("dec_moe_wait");
}

size_t dec_moe_peer_scratch_bytes(int n_embd, int n_ff) {
    return (size_t) kDmCap * n_ff * 2 * sizeof(float) + (size_t) kVerifyMaxT * (n_embd / 32) * sizeof(block_q8_1) +
           sizeof(DmRing) + 256;
}

bool dec_moe_formats_ok(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) {
    bool gu = false, d = false;
    switch (gu_type) {
#define STRATA_DM_OK(T) case T: gu = kSplit<T>; break;
        STRATA_DM_GU_FMTS(STRATA_DM_OK)
#undef STRATA_DM_OK
        default: break;
    }
    switch (d_type) {
#define STRATA_DM_OK(T) case T: d = true; break;
        STRATA_DM_D_FMTS(STRATA_DM_OK)
#undef STRATA_DM_OK
        default: break;
    }
    return gu && d && n_ff == 640 && n_embd % 256 == 0 && !g_old_kernels && !g_grouped_v1;
}

int dec_moe_peer_blocks() {
    int dev = 0, sms = 0, coop = 0, per = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev) != cudaSuccess ||
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, dm_peer_kernel, kDmThreads, 0) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    if (!coop || per <= 0) return 0;
    return sms * per;
}

bool dec_moe_peer_launch(const DmPeerArgs& a, int blocks, void* stream) {
    DmPeerArgs args = a;
    void* params[] = {&args};
    const cudaError_t e = cudaLaunchCooperativeKernel((const void*) dm_peer_kernel, dim3((unsigned) blocks),
                                                      dim3(kDmThreads), params, 0, (cudaStream_t) stream);
    if (e != cudaSuccess) {
        std::fprintf(stderr, "dec_moe_peer_launch: %s\n", cudaGetErrorString(e));
        return false;
    }
    return true;
}

void dec_moe_peer_math_test(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* start,
                            int n_groups, const int32_t* tok, const int32_t* dst, const void* xq, void* scratch,
                            float* out, void* stream) {
    float* gate = (float*) scratch;
    float* up = gate + (size_t) kDmCap * L.n_ff;
    const int blocks = 2 * 36;
    dm_test_gu_kernel<<<blocks, kDmThreads, 0, (cudaStream_t) stream>>>(L, grp_ptr, start, n_groups, tok,
                                                                        (const block_q8_1*) xq, gate, up);
    dm_test_down_kernel<<<blocks, kDmThreads, 0, (cudaStream_t) stream>>>(L, grp_ptr, start, n_groups, dst, gate, up, out);
    check("dec_moe_peer_math_test");
}

}  // namespace strata::kernels
