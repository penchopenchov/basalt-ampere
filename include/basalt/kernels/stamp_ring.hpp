// include/basalt/kernels/stamp_ring.hpp - in-kernel phase stamps for the persistent kernels (PROFILE builds only).
//
// A PROFILE build (-DBASALT_PROFILE=ON) gives every stamped kernel a pinned, host-mapped ring of fixed-size records,
// one record per (kernel id, launch, CTA).  A host thread copies completed launches to the file named by
// BASALT_STAMP_FILE while the engine runs, so nothing depends on a clean exit.  The normal build compiles all of
// it out: no parameters, no code.
//
// Record (kStampRec u64): [0] header (see stamp_hdr), [1] CTA entry time, [kA..] and [kB..] phase times,
// [kExit] CTA exit time.  Times are %globaltimer ns.  A kernel adds its phases by writing rec[2 + id] on thread 0.
#pragma once

#include <cstdint>

namespace basalt::kernels::stamp {

enum Kid : int { kChain = 1, kQsa = 2, kFgdn = 3, kCombine = 4, kDmGu = 5, kDmSwi = 6, kDmDown = 7, kDmWait = 8, kKids = 9 };
constexpr int kStampRec = 96;       ///< u64 per record
constexpr int kExit = kStampRec - 1; ///< exit-time slot
constexpr int kA = 2;               ///< first stamp slot of a kernel's first phase block (32 slots)
constexpr int kB = 34;              ///< first slot of a second block (fused kernel's experts half, 32 slots)
constexpr int kStampMaxCta = 192;   ///< records per launch (record index = the kernel's own role / CTA number)
constexpr int kStampLaunches = 256; ///< launches kept in the ring per kernel id

// Header: kid | ord << 8 | T << 16 | ncta << 24 | launch << 32
struct Ctx {
    unsigned long long* ring = nullptr;   ///< [kKids][kStampLaunches][kStampMaxCta][kStampRec], host-mapped
    unsigned* cnt = nullptr;              ///< [kKids][kStampMaxCta] launches begun per CTA number (device memory)
};

#ifdef BASALT_PROFILE
#define BASALT_STAMP_PARAM , ::basalt::kernels::stamp::Ctx sc
#define BASALT_STAMP_PARAM_D , ::basalt::kernels::stamp::Ctx sc = {}
#define BASALT_STAMP_ARG(x) , x
/// The ring for the current process (allocated on first use, outside a capture); null ring when BASALT_STAMP_FILE is unset.
Ctx ctx();
/// A translation unit with span kernels registers a binder that copies the ring context into its device symbol.
void reg_bind(void (*f)(const Ctx&));
#else
#define BASALT_STAMP_PARAM
#define BASALT_STAMP_PARAM_D
#define BASALT_STAMP_ARG(x)
#endif

#if defined(__CUDACC__) && defined(BASALT_PROFILE)
__device__ __forceinline__ unsigned long long now() {
    unsigned long long t;
    asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(t));
    return t;
}
/// Thread 0 of a CTA: begin this CTA's record of launch number L; returns it (null when the ring is off).
/// `role` indexes the record within the launch, `ncta` is the launch's record count.
__device__ __forceinline__ unsigned long long* open_at(const Ctx& c, int kid, unsigned L, int role, int ncta, int ord, int T,
                                                       unsigned long long t_in) {
    if (c.ring == nullptr) return nullptr;
    unsigned long long* r = c.ring + ((size_t) (kid * kStampLaunches + (L % kStampLaunches)) * kStampMaxCta + role) * kStampRec;
    r[1] = t_in;
    r[kExit] = 0;
    r[0] = (unsigned long long) kid | (unsigned long long) ord << 8 | (unsigned long long) T << 16 | (unsigned long long) ncta << 24 |
           (unsigned long long) L << 32;
    return r;
}
/// Launch number by per-CTA counting: valid when every CTA runs in every launch of the kernel.
__device__ __forceinline__ unsigned long long* open(const Ctx& c, int kid, int role, int ncta, int ord, int T, unsigned long long t_in,
                                                    int cta) {
    if (c.ring == nullptr) return nullptr;
    return open_at(c, kid, atomicAdd(&c.cnt[kid * kStampMaxCta + cta], 1u), role, ncta, ord, T, t_in);
}
/// Launch number by whole-launch counting, for kernels whose grid size varies and whose launches do not overlap:
/// read it at CTA entry, call seq_done after the record is complete (the last CTA of the launch advances it).
__device__ __forceinline__ unsigned seq_launch(const Ctx& c, int kid) {
    return c.ring ? *(volatile unsigned*) &c.cnt[kid * kStampMaxCta + kStampMaxCta - 1] : 0u;
}
__device__ __forceinline__ void seq_done(const Ctx& c, int kid, int ncta) {
    if (c.ring == nullptr) return;
    __threadfence();
    unsigned* arrive = &c.cnt[kid * kStampMaxCta + kStampMaxCta - 2];
    if (atomicAdd(arrive, 1u) + 1u == (unsigned) ncta) {
        *(volatile unsigned*) arrive = 0u;
        atomicAdd(&c.cnt[kid * kStampMaxCta + kStampMaxCta - 1], 1u);
    }
}
/// Whole-kernel span of a many-CTA kernel (one record per launch): entry = CTA 0's entry (or its rearm() time), exit = the
/// last CTA to finish.  Scope guard: thread 0 of every CTA stamps in the destructor, so early returns are covered.
struct Span {
    const Ctx& c; int kid; unsigned long long t_in; unsigned L;
    __device__ __forceinline__ Span(const Ctx& cc, int k) : c(cc), kid(k) { t_in = now(); L = seq_launch(c, k); }
    __device__ __forceinline__ void rearm() { t_in = now(); }
    __device__ __forceinline__ ~Span() {
        if (c.ring == nullptr || (threadIdx.x | threadIdx.y | threadIdx.z) != 0) return;
        const unsigned n = gridDim.x * gridDim.y * gridDim.z;
        if ((blockIdx.x | blockIdx.y | blockIdx.z) == 0) open_at(c, kid, L, 0, 1, 0, 0, t_in);
        __threadfence();
        unsigned* arrive = &c.cnt[kid * kStampMaxCta + kStampMaxCta - 2];
        if (atomicAdd(arrive, 1u) + 1u == n) {
            *(volatile unsigned*) arrive = 0u;
            unsigned long long* r = c.ring + ((size_t) (kid * kStampLaunches + (L % kStampLaunches)) * kStampMaxCta) * kStampRec;
            r[kExit] = now();
            __threadfence();
            atomicAdd(&c.cnt[kid * kStampMaxCta + kStampMaxCta - 1], 1u);
        }
    }
};
#endif

}  // namespace basalt::kernels::stamp
