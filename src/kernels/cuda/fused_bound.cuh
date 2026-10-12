// Device-side wait bound for the persistent attention chain: every wait loop gives up after MKB_NS of globaltimer,
// reports {kind, line, CTA, layer, wanted, seen} to host-mapped memory (g_herr), and traps, so a lost hand-off ends
// the launch with an error instead of hanging the GPU.
// g_prog (host-mapped, MKB_PROG builds): per CTA, the last wait / barrier line each warp entered (a hang dump).
// g_lay: per CTA, the layer ordinal + 1 it is in (written at each body entry), for the reports.
#pragma once
#ifndef MKB_NS
#define MKB_NS 2000000000u
#endif
constexpr int MKB_PW = 33;   // g_prog words per CTA: 16 warps' lines, the layer, 16 warps' active masks at that line
__device__ unsigned* g_herr;  // [0] flag, [1] kind, [2] line, [3] cta, [4] layer, [5] wanted, [6] seen (chain side)
__device__ unsigned* g_prog;
__device__ unsigned g_lay[256];
__device__ __forceinline__ void mkb_prg(int line) {
#ifdef MKB_PROG
    const unsigned am = __activemask();   // the lowest active lane stamps (lane 0 may be elsewhere in a diverged warp)
    if ((int) (threadIdx.x & 31) == __ffs(am) - 1 && g_prog != nullptr) {
        volatile unsigned* g = g_prog + blockIdx.x * MKB_PW;
        g[threadIdx.x >> 5] = (unsigned) line; g[17 + (threadIdx.x >> 5)] = am;
    }
#endif
}
__device__ __forceinline__ void mkb_setlay(unsigned ord1) {
    if (threadIdx.x == 0) {
        g_lay[blockIdx.x] = ord1;
#ifdef MKB_PROG
        if (g_prog != nullptr) ((volatile unsigned*) g_prog)[blockIdx.x * MKB_PW + 16] = ord1;
#endif
    }
}
__device__ __forceinline__ void syncp(int line = __builtin_LINE()) { mkb_prg(line); __syncthreads(); }
__device__ __noinline__ __attribute__((noreturn)) void mkb_hang(int code, int line, unsigned wanted = 0, unsigned seen = 0) {
    volatile unsigned* e = g_herr;
    if (e != nullptr && atomicCAS((unsigned*) e, 0u, 1u) == 0u) {
        e[1] = (unsigned) code; e[2] = (unsigned) line; e[3] = blockIdx.x; e[4] = g_lay[blockIdx.x]; e[5] = wanted; e[6] = seen;
    }
    asm volatile("membar.sys;\ntrap;" ::: "memory");
    while (true) {}
}
__device__ __forceinline__ unsigned mkb_now() { unsigned t; asm volatile("mov.u32 %0,%%globaltimer_lo;" : "=r"(t)); return t; }
__device__ __forceinline__ bool mkb_try(unsigned bar, unsigned ph) {
    unsigned d;
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }" : "=r"(d) : "r"(bar), "r"(ph) : "memory");
    return d != 0;
}
__device__ __forceinline__ void mkb_mwait(unsigned bar, unsigned ph, int line) {
    mkb_prg(line);
    if (mkb_try(bar, ph)) return;
    const unsigned t0 = mkb_now();
    while (!mkb_try(bar, ph)) if (mkb_now() - t0 > MKB_NS) mkb_hang(1, line, ph, ph ^ 1u);
}
