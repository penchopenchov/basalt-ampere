namespace mkc {
constexpr int CT = 512, NREP = 8, XT = 4, SMEM_MAX = 101376;
// ---- hc
constexpr int NDG = 21, NDN = NDG * CN_HS, NUP = N / 32, RT0 = 80, NRTC = 64, RTR = NE / NRTC;
constexpr int SROW = D / G8;
constexpr int OFF_WD = 0, SZ_WD = 16 * CN_W + 16 * 80, OFF_XA = SZ_WD, SZ_XA = XT * CN_W * 4, SZ_A = SZ_WD + SZ_XA;
constexpr int OFF_WU = SZ_A, SZ_WU = UPCM + HC * UPCM_SL * SU8 * 2, OFF_MH = OFF_WU + SZ_WU;
__host__ __device__ constexpr int CP_FLOATS(int T) { return T * CN_HS * CN_ROWS + T * CN_HS; }
// ---- pair
// the projections' types are per layer (LayerC.ty): Q4_K 12, Q5_K 13, Q6_K 14, IQ3_S 21, IQ4_XS 23 (sb_bytes: a
// 256-value super-block's bytes)
constexpr int P_NIN = 2560, P_NQ = 10240, P_NZ = 6144, P_NTQ = P_NQ / 16, P_NT = P_NTQ + P_NZ / 16;
constexpr int P_SLOTS = 4, OFF_PRED = 87040;   // the ring: up to 4 slots of the layer's larger tile below OFF_PRED
// ---- rec
constexpr int R_HV = 48, R_HK = 16, R_C = 2 * R_HK * S + R_HV * S, R_ZV = R_HV * S, R_CTA0 = 122;   // [0, 64 KB): the head's state tile
// ---- out
// out: the tile [0, 16 * row stride), red (4 columns) after it, x staged after red where it fits below OFF_PERM
constexpr int O_NIN = 6144, O_NOUT = 2560, O_NLW = 24, O_RC = 4, O_REDB = O_NLW * 16 * O_RC * 4;
// ---- permanent tail: barriers + the IQ3_S grid
constexpr int OFF_PERM = SMEM_MAX - 4096;
// the per-layer geometry of the typed phases (host-computed, LayerC.geo -> Perm.geo)
struct Geo {
    int tq, tz, to;     // types
    int tbq, tbz;       // pair tile bytes (16 rows)
    int ns, slotb;      // pair ring: slots, slot bytes
    int orb, ors, ored, ox;   // out: row bytes, row stride, red offset, staged x offset
};
__host__ __device__ constexpr Geo make_geo(int tq, int tz, int to) {
    Geo g{};
    g.tq = tq; g.tz = tz; g.to = to;
    g.tbq = 16 * (P_NIN / 256) * sb_bytes(tq); g.tbz = 16 * (P_NIN / 256) * sb_bytes(tz);
    g.slotb = g.tbq > g.tbz ? g.tbq : g.tbz;
    g.ns = OFF_PRED / g.slotb < P_SLOTS ? OFF_PRED / g.slotb : P_SLOTS;
    g.orb = (O_NIN / 256) * sb_bytes(to); g.ors = one_row_stride(g.orb);
    g.ored = (16 * g.ors + 127) & ~127; g.ox = g.ored + O_REDB;
    return g;
}
// f(integral_constant<TYPE>) for a projection type
template <typename F> __device__ __forceinline__ void by_type(int ty, F f) {
    switch (ty) {
        case 12: f(std::integral_constant<int, 12>()); break;
        case 13: f(std::integral_constant<int, 13>()); break;
        case 14: f(std::integral_constant<int, 14>()); break;
        case 21: f(std::integral_constant<int, 21>()); break;
        default: f(std::integral_constant<int, 23>()); break;
    }
}
// the kernel instance's types: FQ/FZ/FO fixed (a specialized instance; the layer's types are those), or 0 (generic:
// the layer's types at run time)
template <int FQ, int FZ, typename F> __device__ __forceinline__ void by_tq_tz(int ty, int tq, F f) {
    if constexpr (FQ == 0) by_type(ty, f);
    else if (FQ == FZ || ty == tq) f(std::integral_constant<int, FQ>()); else f(std::integral_constant<int, FZ>());
}
template <int FO, typename F> __device__ __forceinline__ void by_to(int ty, F f) {
    if constexpr (FO == 0) by_type(ty, f); else f(std::integral_constant<int, FO>());
}

__device__ __forceinline__ unsigned ldacq(const unsigned* p) { unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0,[%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void strel(unsigned* p, unsigned v) { asm volatile("st.release.gpu.global.u32 [%0],%1;" ::"l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void redrel(unsigned* p) { asm volatile("red.release.gpu.global.add.u32 [%0],1;" ::"l"(p) : "memory"); }
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ void cbar() { asm volatile("bar.sync 1, 512;" ::: "memory"); }
__device__ __forceinline__ void gbar(int g) { asm volatile("bar.sync %0, 256;" ::"r"(2 + g) : "memory"); }
__device__ __forceinline__ void barn(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
#define mwait(b, p) mkb_mwait(smem_u32(b), p, __LINE__)
// thread / block index read opaquely: values derived from them are recomputed per phase instead of being hoisted
// out of the layer loop (and spilled: a local-memory reload after an acquire is an L2 round trip)
__device__ __forceinline__ int tidv() { int t; asm volatile("mov.u32 %0, %%tid.x;" : "=r"(t)); return t; }
__device__ __forceinline__ int ctav() { int t; asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(t)); return t; }
__device__ __forceinline__ void pfence() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ unsigned ldrlx(const unsigned* p) { unsigned v; asm volatile("ld.relaxed.gpu.global.u32 %0,[%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void spin_ge(const unsigned* c, unsigned target) {
    if (ldacq(c) < target) { const unsigned t0_ = mkb_now(); while (ldacq(c) < target) if (mkb_now() - t0_ > MKB_NS) mkb_hang(2, __LINE__); }
}
__device__ __forceinline__ void hop_copy(const unsigned* ctr, unsigned target, void* dst, const void* src, unsigned bytes, uint64_t* bar) {
    spin_ge(ctr, target);
    asm volatile("fence.proxy.async.global;" ::: "memory");
    pfence();
    mbar_expect_tx(bar, bytes);
    bulk_g2s(dst, src, bytes, bar);
}
__device__ __forceinline__ void pf_l2(const void* p, size_t n) {   // n multiple of 16
    const char* c = (const char*) p;
#pragma unroll 1
    for (size_t o = 0; o < n; o += 32768) {
        const unsigned sz = (unsigned) (n - o < 32768 ? n - o : 32768);
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(c + o), "r"(sz) : "memory");
    }
}

enum { PB_WD = 0, PB_WU = 1, PB_A = 2, PB_SLOT = 3, PB_REC = 7, PB_OX = 8, PB_OW = 9 };   // bits of the parity register
#define MW(bar, bit) do { mwait(bar, (PH >> (bit)) & 1u); PH ^= 1u << (bit); } while (0)
enum { C_H1DN, C_H1UP, C_PAIR, C_REC, C_OUT, C_H2DN, C_H2UP, C_RT, C_END, C_REL, C_PZ, NCTR = 16 };
enum { S_REL, S_H1DN, S_H1UP, S_PAIR, S_REC, S_OUT, S_H2DN, S_H2UP, S_RT, S_END, S_PHOP, S_PX, S_PT0, S_PT1, S_RPRE, S_RHOP, S_RCONV, S_RCHAIN, S_OX, S_OW, S_H2SEEN, NSTU, NST = 24 };

struct LayerC {
    HcLaunch h1, h2;
    const uint8_t* wq; const uint8_t* wz;
    DecRecArgs rec;
    const uint8_t* wo;
    const uint8_t* dn1; const uint8_t* dn2;   // hc down rows as per-CTA smem images (or null)
    Geo geo;
    TmaDesc stm;                              // rec state [S rows][R_HV * S], box S x S
};
struct KP {
    LayerC Lc; unsigned par; const LayerC* L; int NLAY, iters, T; long long delay_ns; int pf, prt;
    unsigned* cnt;                       // [iters][NCTR]
    unsigned long long* tx;              // [iters] release time of hc1's input
    unsigned long long* st;              // [iters][G][NST] per-CTA phase-release times (or null)
    float* cpr; float* mxr; float* lgr;  // hc replicas: [4][NREP][...]
    uint8_t* xqr;                        // [2][NREP][XT * 80 * 36] hc1's q8_1 replicas for the pair
    float* qkv; float* z;                // pair outputs
    float* abv;                          // prt -3: the rec heads' alpha/beta [R_HV][2 XT] (job order of the rec code)
    int32_t* ids_all; float* w_all;
    long long* det;                      // [iters][G][64] sub-stamps (clock64) or null
#ifdef BASALT_PROFILE
    ::basalt::kernels::stamp::Ctx sc; int ord;   // PROFILE: the stamp ring and the launch ordinal
#endif
};
struct MiscH {
    HcFusedArgs a;
    union { float sqx[XT][64]; struct { float px[XT * HC * 32]; float gg[XT * HC * 32]; } u; } x;
    float part[XT][8];
    float rsv[XT * HC];
    union { uint4 bf[XT * 81]; float rpart[2][16 * 8]; } y;
};
static_assert(OFF_MH + (int) sizeof(MiscH) <= OFF_PERM, "hc misc");
struct Perm {
    uint64_t full_wd, full_wu, cbarA, slot[P_SLOTS], ofull, oxbar, rbar;
    uint32_t grid[512];
    const uint8_t* lwq; const uint8_t* lwz; const uint8_t* lwo; const uint8_t* lyq; float* lbo; const uint8_t* ldn2;
    HcFusedArgs a2;
    DecRecArgs ra;
    Geo geo;
    long long sst[64];
    int vb;   // logical CTA id (arrival ticket)
};
static_assert(sizeof(Perm) <= 4096, "perm");
struct RecS {   // rec_rank's RecSmem for T <= XT, the alpha/beta rows shared by the 4 ranks
    float qn[XT][S], kn[XT][S];
    float v[4][XT][32], oc[4][XT][32];
    float red_kv[4][RG][32], red_o[4][RG][32];
    float part[4][2][XT][RG];
    float g[4][XT], b[4][XT];
    float wsum[XT][4];
    DecRecArgs a;
};
constexpr int OFF_REC = (OFF_PERM - (int) sizeof(RecS)) & ~127;
static_assert(OFF_REC >= 16 * 5040, "rec above the largest out tile (Q6_K)");

extern __shared__ __align__(128) char sm_dyn[];
__device__ __forceinline__ void dn_dbg_stamp() { reinterpret_cast<Perm*>(sm_dyn + OFF_PERM)->sst[28] = clock64(); }
// weight copies: L2 evict-first policy (read once per layer; keeps the kernel's code and the activations in L2)
__device__ __forceinline__ void bulk_w2s(void* dst, const void* src, unsigned bytes, uint64_t* bar) {
#ifdef MK_NOEF
    bulk_g2s(dst, src, bytes, bar);
#else
    uint64_t pol;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
                 ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)), "l"(pol) : "memory");
#endif
}
// ------------------------------------------------------------ hc helpers (hcy)
struct DnRows { const int8_t* q; const uint16_t* s; int nrows, rowbase, srow; };
__device__ __forceinline__ DnRows dn_rows(const HcFusedArgs& a, int b) {
    const int hs = b % CN_HS, grp = b / CN_HS;
    if (grp < NDG - 1) return {a.q_down + (size_t) (16 * grp) * D + hs * CN_W, a.s_down + (size_t) (16 * grp) * SROW + hs * (CN_W / G8), 16, 16 * grp, SROW};
    if (a.q_inject == nullptr) return {nullptr, nullptr, 0, LR, SROW};
    return {a.q_inject + hs * CN_W, a.s_inject + hs * (CN_W / G8), HC, LR, SROW};
}
constexpr int DN_IMG = 16 * CN_W + 16 * 80;   // one down CTA's smem image (rows, then the rows' scales)
__device__ void fill_dn(const HcFusedArgs& a, const uint8_t* img, int b, char* WD, uint64_t* bar, int lane) {
    if (img != nullptr) {   // the CTA's rows packed as its smem image: one copy
        if (lane == 0 && (a.q_inject != nullptr || b < NDN - CN_HS)) { mbar_expect_tx(bar, DN_IMG); bulk_w2s(WD, img + (size_t) b * DN_IMG, DN_IMG, bar); }
        return;
    }
    const DnRows r = dn_rows(a, b);
    if (r.q == nullptr) return;
    if (lane == 0) mbar_expect_tx(bar, (unsigned) r.nrows * (CN_W + 80));
    __syncwarp();
    if (lane == 0 && threadIdx.x == 0) dn_dbg_stamp();
    if (lane < r.nrows) bulk_w2s(WD + lane * CN_W, r.q + (size_t) lane * D, CN_W, bar);
    else if (lane >= 16 && lane - 16 < r.nrows) bulk_w2s(WD + 16 * CN_W + (lane - 16) * 80, r.s + (size_t) (lane - 16) * r.srow, 80, bar);
}
__device__ void fill_wu(const HcLaunch& L, const uint16_t* w_router, int b, bool router, char* WU, uint64_t* bar) {
    if (b < NUP) hc_up_tma(UpSlice{32 * b, 32}, WU, reinterpret_cast<uint16_t*>(WU + UPCM), reinterpret_cast<const CUtensorMap*>(&L.tm[4]), bar);
    else if (router && b >= RT0 && b < RT0 + NRTC) {
        mbar_expect_tx(bar, RTR * N * 2);
        bulk_w2s(WU, w_router + (size_t) (b - RT0) * RTR * N, RTR * N * 2, bar);
    }
}
// the q8_1 block of `mx` (lane i: value i), native or exact rounding, into NREP + 1 places
template <bool NATIVE>
__device__ __forceinline__ void q8_store_reps(float xi, HcQ81* y0, uint8_t* reps, size_t rep_stride, int ib, int lane) {
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    int8_t q;
    float d;
    if constexpr (NATIVE) {
        d = q8_1_finite(div_approx_ftz(amax, 127.0f));
        q = 0;
        if (amax != 0.0f) {
            const float v = roundf(div_approx_ftz(xi, d));
            q = (int8_t) (v > 127.0f ? 127.0f : v < -127.0f ? -127.0f : v);
        }
    } else {
        d = q8_1_finite(amax / 127.0f);
        q = q8_1_quant(xi, d, amax);
    }
    const half2 ds = q8_1_ds(d, sum);
    y0[ib].qs[lane] = q;
    if (lane == 0) y0[ib].ds = ds;
    if (reps != nullptr)
#pragma unroll
        for (int r = 0; r < NREP; ++r) {
            HcQ81* yr = reinterpret_cast<HcQ81*>(reps + (size_t) r * rep_stride);
            yr[ib].qs[lane] = q;
            if (lane == 0) yr[ib].ds = ds;
        }
}
// t10_fast: exact top-10 (the same ids and weights as hc_top10 / top10_smem).  The softmax is theirs; the selection:
// theta = the tenth largest lane maximum by value (ten values at or above it bound the selection), candidates =
// values >= theta, compacted by ballots and ranked by (value desc, expert asc) against the list read back from smem.
// More than 32 candidates (mass ties, NaN logits): theta by (value, expert) as hc_top10, which bounds the count.
// cand: 2 * 512 + 2 * 10 words of smem.
__device__ __forceinline__ void top10_smem(const float* logits, int32_t* ids, float* weights, int lane, uint32_t* cand) {
    float values[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) values[i] = logits[lane + i * 32];
    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i) maximum = max_ftz(maximum, values[i]);
    maximum = warp_max_ftz(maximum);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) { values[i] = expf_fast(sub_ftz(values[i], maximum)); sum = add_ftz(sum, values[i]); }
    const float reciprocal = rcp_ftz(warp_sum_ftz(sum));
#pragma unroll
    for (int i = 0; i < 16; ++i) { values[i] = mul_ftz(values[i], reciprocal); if (isnan(values[i])) values[i] = -FLT_MAX; }
    float bv = values[0];
    int be = lane;
#pragma unroll
    for (int i = 1; i < 16; ++i) if (values[i] > bv) { bv = values[i]; be = lane + 32 * i; }
    // theta = the tenth largest lane maximum counted with multiplicity: distinct maxima in descending order (order-preserving keys)
    const unsigned bu = __float_as_uint(bv);
    const unsigned key = bv == 0.0f ? 0x80000000u : (bu >> 31 ? ~bu : bu | 0x80000000u);
    unsigned tk = 0;
    {
        bool alive = true;
        int seen = 0;
        for (int k = 0; k < KT; ++k) {
            tk = __reduce_max_sync(0xffffffffu, alive ? key : 0u);
            seen += __popc(__ballot_sync(0xffffffffu, alive && key == tk));
            if (seen >= KT) break;
            alive = alive && key != tk;
        }
    }
    const float tv = __shfl_sync(0xffffffffu, bv, __ffs(__ballot_sync(0xffffffffu, key == tk)) - 1);
    unsigned mine = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) mine |= (values[i] >= tv ? 1u : 0u) << i;
    int total = __reduce_add_sync(0xffffffffu, (unsigned) __popc(mine));
    if (total > 32) {   // theta by (value, expert): the lane maximum with exactly nine lane maxima before it
        auto before = [](float v, int e, float v2, int e2) { return v > v2 || (v == v2 && e < e2); };
        int rank = 0;
        for (int m = 0; m < 32; ++m) {
            const float ov = __shfl_sync(0xffffffffu, bv, m);
            const int oe = __shfl_sync(0xffffffffu, be, m);
            rank += before(ov, oe, bv, be) ? 1 : 0;
        }
        const int src = __ffs(__ballot_sync(0xffffffffu, rank == KT - 1)) - 1;
        const float tv2 = __shfl_sync(0xffffffffu, bv, src);
        const int te2 = __shfl_sync(0xffffffffu, be, src);
        mine = 0;
        for (int i = 0; i < 16; ++i) if (!before(tv2, te2, values[i], lane + 32 * i)) mine |= 1u << i;
        total = __reduce_add_sync(0xffffffffu, (unsigned) __popc(mine));
    }
    // compaction, lane-major (the rank below orders by (value, expert), so candidate order is free)
    const int nmine = __popc(mine);
    int o = nmine;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) { const int u = __shfl_up_sync(0xffffffffu, o, d); if (lane >= d) o += u; }
    o -= nmine;
#pragma unroll
    for (int i = 0; i < 16; ++i)
        if (mine >> i & 1u) { cand[2 * o] = __float_as_uint(values[i]); cand[2 * o + 1] = (uint32_t) (lane + 32 * i); ++o; }
    __syncwarp();
    float* top_v = reinterpret_cast<float*>(cand + 2 * NE);
    int* top_e = reinterpret_cast<int*>(cand + 2 * NE + KT);
    const uint2* cp = reinterpret_cast<const uint2*>(cand);
    for (int c = lane; c < total; c += 32) {
        const uint2 me = cp[c];
        const float v = __uint_as_float(me.x);
        const int e = (int) me.y;
        int r = 0;
        for (int d = 0; d < total; ++d) {
            const uint2 o = cp[d];
            const float ov = __uint_as_float(o.x);
            r += (ov > v || (ov == v && (int) o.y < e)) ? 1 : 0;
        }
        if (r < KT) { top_v[r] = v; top_e[r] = e; }
    }
    __syncwarp();
    float selected_sum = 0.0f;
#pragma unroll
    for (int r = 0; r < KT; ++r) if ((top_e[r] & 31) == lane) selected_sum = add_ftz(selected_sum, top_v[r]);
    const float selected = lane < KT ? top_v[lane] : 0.0f;
    const int sel_e = lane < KT ? top_e[lane] : 0;
    selected_sum = max_ftz(warp_sum_ftz(selected_sum), 6.103515625e-5f);
    const float inverse_selected_sum = rcp_ftz(selected_sum);
    if (lane < KT) { ids[lane] = sel_e; weights[lane] = mul_ftz(selected, inverse_selected_sum); }
}

// One hc read (hcy).  ROUTER: + router + top-10 in every CTA.  The input release: wait_in() (thread 0, then cbar).
// Counters: c[0] down, c[1] up, c[2] router.  xqr: hc1's q8_1 replicas (or null).  st: stamps for [dn, up, rt].
// Returns after the CTA's last release of the read; fills of the next user of the smem are the caller's.
struct NoFill { __device__ void operator()() const {} };
template <int T, bool ROUTER, typename WF, typename LF = NoFill, typename DF = NoFill, typename RF = NoFill>
__device__ __forceinline__ void hc_read(const KP& p, const HcFusedArgs* asrc, char* sm, MiscH& M, Perm& P, unsigned& PH,
                                        unsigned* c, int slot, uint8_t* xqr, unsigned long long* st, WF wait_in,
                                        LF late_fill = LF(), DF dn_fill = DF(), RF rel_hook = RF(), bool copy_args = true) {
#define DS(n) do { if (p.det != nullptr && threadIdx.x == 0) P.sst[(ROUTER ? 32 : 0) + (n)] = clock64(); } while (0)
    DS(18);
    char* A = sm;
    char* WD = sm + OFF_WD;
    char* XA = sm + OFF_XA;
    char* WU = sm + OFF_WU;
    const int tid = tidv(), wp = tid >> 5, lane = tid & 31, b = P.vb, G = gridDim.x;
    const bool dn_role = b < NDN, up_role = b < NUP, rt_role = ROUTER && b >= RT0 && b < RT0 + NRTC;
    {
        const int* src = reinterpret_cast<const int*>(asrc);
        int* dst = reinterpret_cast<int*>(&M.a);
        if (copy_args) for (int w = tid; w < (int) (sizeof(HcFusedArgs) / 4); w += CT) dst[w] = src[w];
    }
    cbar();
    DS(19);
    const HcFusedArgs& a = M.a;
    float* cpr = p.cpr + (size_t) slot * NREP * CP_FLOATS(XT);
    float* mxr = p.mxr + (size_t) slot * NREP * XT * N;
    float* lgr = p.lgr + (size_t) slot * NREP * XT * NE;
    const int hs = b % CN_HS, cs = hs >> 1;
    const DnRows rows = dn_role ? dn_rows(a, b) : DnRows{nullptr, nullptr, 0, 0, 0};
    const int e4 = tid;
    const int c0 = 32 * b;
    const bool up_el = up_role && tid < T * HC * 32;
    const int uk = tid / (HC * 32), uc = (tid / 32) % HC, udd = tid % 32;
    DS(20);
    if (tid == 0) wait_in();
    cbar();
    rel_hook();
    DS(0);
    // ================= down
    if (rows.q != nullptr) {
        float4* X = reinterpret_cast<float4*>(XA);
        float sq0[T];
        if (e4 < CN_W / 4) {
            float4 bb[T], rr[T];
            float gw[T];
#pragma unroll
            for (int k = 0; k < T; ++k)
                bb[k] = a.apply ? *reinterpret_cast<const float4*>(a.bo_prev + (size_t) k * N + (hs & 1) * CN_W + 4 * e4) : make_float4(0.f, 0.f, 0.f, 0.f);
            const float4 gg = *reinterpret_cast<const float4*>(a.w_norm + hs * CN_W + 4 * e4);
#pragma unroll
            for (int k = 0; k < T; ++k) {
                rr[k] = *reinterpret_cast<const float4*>(a.R + (size_t) k * D + hs * CN_W + 4 * e4);
                gw[k] = a.apply ? a.inj_prev[k * HC + cs] : 0.0f;
            }
#pragma unroll
            for (int k = 0; k < T; ++k) gw[k] = a.apply ? 2.0f * sigmoidf_(gw[k] / (float) HC) : 0.0f;
#pragma unroll
            for (int k = 0; k < T; ++k) {
                float4 v = rr[k];
                if (a.apply) {
                    v.x = fmaf(bb[k].x, gw[k], v.x); v.y = fmaf(bb[k].y, gw[k], v.y);
                    v.z = fmaf(bb[k].z, gw[k], v.z); v.w = fmaf(bb[k].w, gw[k], v.w);
                }
                // the sum of squares with its contraction spelled out (hc_fused's: fma(x, x, y*y), then + z*z, + w*w)
                const float sq = fmaf(v.w, v.w, fmaf(v.z, v.z, fmaf(v.x, v.x, v.y * v.y)));
                X[k * (CN_W / 4) + (e4 & 1) * (CN_W / 8) + (e4 >> 1)] = make_float4(v.x * gg.x, v.y * gg.y, v.z * gg.z, v.w * gg.w);
                if (e4 >= THREADS) M.x.sqx[k][e4 - THREADS] = sq;
                sq0[k] = sq;
            }
        }
        late_fill();
        cbar();
        DS(1);
        const bool ss = rows.rowbase == 0;
        if (ss && tid < THREADS) {
#pragma unroll
            for (int k = 0; k < T; ++k) {
                float sq = sq0[k];
                if (tid < CN_W / 4 - THREADS) sq += M.x.sqx[k][tid];
                const float s = warp_sum(sq);
                if (lane == 0) M.part[k][wp] = s;
            }
        }
        MW(&P.full_wd, PB_WD);
        DS(2);
        if (2 * wp < rows.nrows) {
            const float4* Xc = reinterpret_cast<const float4*>(XA);
            float acc[2 * T];
#pragma unroll
            for (int m = 0; m < 2 * T; ++m) acc[m] = 0.0f;
#pragma unroll
            for (int q = 0; q < CN_W / 256; ++q) {
                const int j = lane + 32 * q;
                float4 x0[T], x1[T];
#pragma unroll
                for (int k = 0; k < T; ++k) { x0[k] = Xc[k * (CN_W / 4) + j]; x1[k] = Xc[k * (CN_W / 4) + CN_W / 8 + j]; }
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    const int row = 2 * wp + r;
                    const uint2 wq = *reinterpret_cast<const uint2*>(WD + row * CN_W + j * 8);
                    const float4 w0 = i8x4_f(wq.x), w1 = i8x4_f(wq.y);
                    const float sc = h2f(reinterpret_cast<const uint16_t*>(WD + 16 * CN_W + row * 80)[j >> 2]);
#pragma unroll
                    for (int k = 0; k < T; ++k) acc[r * T + k] = fmaf(sc, dot8ff(w0, w1, x0[k], x1[k]), acc[r * T + k]);
                }
            }
            float v[16];
#pragma unroll
            for (int m = 0; m < 16; ++m) v[m] = m < 2 * T ? acc[m] : 0.0f;
            const float s = warp_sum16(v, lane);
            const int m = lane >> 1;
            if ((lane & 1) == 0 && m < 2 * T) {
                const int r = m / T, k = m - r * T;
                const size_t o = ((size_t) k * CN_HS + hs) * CN_ROWS + rows.rowbase + 2 * wp + r;
#pragma unroll
                for (int rp = 0; rp < NREP; ++rp) cpr[(size_t) rp * CP_FLOATS(XT) + o] = s;
            }
        }
        DS(3);
        cbar();
        DS(4);
        if (ss && tid < T) {
            float s = 0.0f;
            for (int w = 0; w < WARPS; ++w) s += M.part[tid][w];
            const size_t o = (size_t) T * CN_HS * CN_ROWS + tid * CN_HS + hs;
#pragma unroll
            for (int r = 0; r < NREP; ++r) cpr[(size_t) r * CP_FLOATS(XT) + o] = s;
        }
        __syncwarp();
        if (tid == 0) { redrel(&c[0]); if (st) st[0] = gtime(); }
        DS(5);
        dn_fill();
    } else if (dn_role) {
        if (tid == 0) { redrel(&c[0]); if (st) st[0] = gtime(); }
        dn_fill();
    } else dn_fill();
    // ================= up
    if (up_role) {
        const float* CP = reinterpret_cast<const float*>(A);
        float urv = 0.f, urp = 0.f, uwn = 0.f, uinj = 0.f, ubo = 0.f;
        if (up_el) {
            urv = a.R[(size_t) uk * D + uc * N + c0 + udd];
            uwn = a.w_norm[uc * N + c0 + udd];
            if (a.apply) { uinj = a.inj_prev[uk * HC + uc]; ubo = a.bo_prev[(size_t) uk * N + c0 + udd]; }
        }
        if (tid == 0) hop_copy(&c[0], NDN, A, cpr + (size_t) (b % NREP) * CP_FLOATS(XT), (unsigned) CP_FLOATS(T) * 4, &P.cbarA);
        if (up_el) {
            urp = urv;
            if (a.apply) urp = fmaf(ubo, 2.0f * sigmoidf_(uinj / (float) HC), urv);
            urv = urp * uwn;
        }
        if (up_el) M.x.u.px[tid] = urv;
        MW(&P.cbarA, PB_A);
        DS(6);
        if (tid < T * HC) {
            const int k = tid / HC, cc = tid - k * HC;
            const float* cs2 = CP + T * CN_HS * CN_ROWS + k * CN_HS;
            M.rsv[tid] = rsqrtf((cs2[2 * cc] + cs2[2 * cc + 1]) / (float) N + a.eps);
        }
        cbar();
        DS(7);
        if (up_el) a.R_out[(size_t) uk * D + uc * N + c0 + udd] = urp;
        if (ROUTER && p.det != nullptr && tid == 0) P.sst[53] = clock64();
        for (int qd = tid; qd < T * (LR / 4); qd += CT) {
            const int k = qd / (LR / 4), r4 = qd - k * (LR / 4);
            // the thread's four rows of each half-stream by one 16-byte load (a row run is 16-byte aligned: CN_ROWS % 4 == 0)
            float4 h0[HC], h1[HC];
#pragma unroll
            for (int cc = 0; cc < HC; ++cc) {
                h0[cc] = *reinterpret_cast<const float4*>(CP + ((size_t) k * CN_HS + 2 * cc) * CN_ROWS + 4 * r4);
                h1[cc] = *reinterpret_cast<const float4*>(CP + ((size_t) k * CN_HS + 2 * cc + 1) * CN_ROWS + 4 * r4);
            }
            float l4[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                float v = 0.0f;
#pragma unroll
                for (int cc = 0; cc < HC; ++cc) {
                    const float a0 = u == 0 ? h0[cc].x : u == 1 ? h0[cc].y : u == 2 ? h0[cc].z : h0[cc].w;
                    const float a1 = u == 0 ? h1[cc].x : u == 1 ? h1[cc].y : u == 2 ? h1[cc].z : h1[cc].w;
                    v = fmaf(M.rsv[k * HC + cc], a0 + a1, v);
                }
                const float x = v / (float) HC;
                l4[u] = x / (1.0f + __expf(-x));
            }
            if (r4 % NUP == b) *reinterpret_cast<float4*>(a.lo + k * LR + 4 * r4) = make_float4(l4[0], l4[1], l4[2], l4[3]);
            uint4 f;
            split_f16(l4[0], l4[1], f.x, f.z);
            split_f16(l4[2], l4[3], f.y, f.w);
            M.y.bf[k * 81 + r4] = f;
        }
        if (ROUTER && p.det != nullptr && tid == 0) P.sst[54] = clock64();
        if (b == 0 && tid < T * HC) {
            a.rs[tid] = M.rsv[tid];
            if (a.inject_out != nullptr && a.q_inject != nullptr) {
                const int k = tid / HC, r = tid - k * HC;
                float v = 0.0f;
#pragma unroll
                for (int cc = 0; cc < HC; ++cc)
                    v = fmaf(M.rsv[k * HC + cc], CP[((size_t) k * CN_HS + 2 * cc) * CN_ROWS + LR + r] + CP[((size_t) k * CN_HS + 2 * cc + 1) * CN_ROWS + LR + r], v);
                a.inject_out[tid] = v;
            }
        }
        cbar();
        DS(8);
        MW(&P.full_wu, PB_WU);
        DS(9);
        if (wp < 8) {
            const int g = lane >> 2, t4 = lane & 3, warp = wp;
            float rsr[2][HC];
#pragma unroll
            for (int j = 0; j < 2; ++j)
#pragma unroll
                for (int cc = 0; cc < HC; ++cc) rsr[j][cc] = 2 * t4 + j < T ? M.rsv[(2 * t4 + j) * HC + cc] : 0.0f;
            constexpr int ks = 2 * UPCM_CH;
            const unsigned addr = smem_u32(WU + (lane >> 4) * UPCM_CH + (16 * warp + (lane & 15)) * 16);
            const uint16_t* SC = reinterpret_cast<const uint16_t*>(WU + UPCM);
            const uint16_t* sc0 = SC + (16 * warp + g) * SU8;
            const uint16_t* sc1 = SC + (16 * warp + g + 8) * SU8;
            float acc[4] = {0.f, 0.f, 0.f, 0.f};
            const int tk = g & 3;
            const bool hi = g < 4;
            const uint4* lbp = M.y.bf + (tk < T ? tk : 0) * 81 + t4;
#pragma unroll
            for (int s = 0; s < NG8; ++s) {
                uint32_t q[4];
                asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                    : "=r"(q[0]), "=r"(q[1]), "=r"(q[2]), "=r"(q[3]) : "r"(addr + s * ks));
                float c4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const uint32_t af[4] = {i8x2_h2<false>(q[2 * h]), i8x2_h2<false>(q[2 * h + 1]), i8x2_h2<true>(q[2 * h]),
                                            i8x2_h2<true>(q[2 * h + 1])};
                    uint4 f = lbp[8 * s + 4 * h];
                    if (tk >= T) f = make_uint4(0u, 0u, 0u, 0u);
                    mma_f16(c4, af, hi ? f.x : f.z, hi ? f.y : f.w);
                }
                float gl4[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) gl4[j] = __shfl_xor_sync(0xffffffffu, c4[j], 2);
                const float s0 = h2f(sc0[s]), s1 = h2f(sc1[s]);
                acc[0] = fmaf(s0, c4[0] + gl4[0], acc[0]);
                acc[1] = fmaf(s0, c4[1] + gl4[1], acc[1]);
                acc[2] = fmaf(s1, c4[2] + gl4[2], acc[2]);
                acc[3] = fmaf(s1, c4[3] + gl4[3], acc[3]);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int k = 2 * t4 + (j & 1);
                const int sl2 = 16 * warp + g + 8 * (j >> 1), cc = sl2 / UPCM_SL, dd = sl2 - cc * UPCM_SL;
                if (k >= T) continue;
                const int idx = (k * HC + cc) * 32 + dd;
                float rsc = rsr[j & 1][0];
#pragma unroll
                for (int c2 = 1; c2 < HC; ++c2) rsc = cc == c2 ? rsr[j & 1][c2] : rsc;
                const float x = M.x.u.px[idx] * rsc;
                M.x.u.gg[idx] = x * sigmoidf_(acc[j]);
            }
        }
        cbar();
        DS(10);
        if (tid < T * 32) {
            const int k = tid >> 5, col = lane;
            float s = 0.0f;
#pragma unroll
            for (int cc = 0; cc < HC; ++cc) s += M.x.u.gg[(k * HC + cc) * 32 + col];
            const float mx = s / (float) HC;
            a.mixed[(size_t) k * N + c0 + col] = mx;
            if (ROUTER)
#pragma unroll
                for (int r = 0; r < NREP; ++r) mxr[((size_t) r * XT + k) * N + c0 + col] = mx;
            if (a.xq != nullptr) {
                if (ROUTER) q8_store_reps<false>(mx, static_cast<HcQ81*>(a.xq), nullptr, 0, k * (N / 32) + c0 / 32, lane);
                else q8_store_reps<true>(mx, static_cast<HcQ81*>(a.xq), xqr, (size_t) XT * (N / 32) * kQ81, k * (N / 32) + c0 / 32, lane);
            }
        }
        cbar();
        DS(11);
        if (tid == 0) { redrel(&c[1]); if (st) st[1] = gtime(); }
    }
    if constexpr (ROUTER) {
        // ================= router
        if (rt_role) {
            const float* XM = reinterpret_cast<const float*>(A);
            if (tid == 0) hop_copy(&c[1], NUP, A, mxr + (size_t) (b % NREP) * XT * N, (unsigned) T * N * 4, &P.cbarA);
            MW(&P.cbarA, PB_A);
            DS(12);
            MW(&P.full_wu, PB_WU);
            DS(13);
            const int g = tid >> 8, gt = tid & 255, gw2 = gt >> 5, e0 = (b - RT0) * RTR + 4 * g;
            constexpr int PAIRS = N / 2 / THREADS, RM = 4, NV = RM * T;
            float2 x[T][PAIRS];
#pragma unroll
            for (int q = 0; q < T; ++q)
#pragma unroll
                for (int j = 0; j < PAIRS; ++j) x[q][j] = reinterpret_cast<const float2*>(XM + (size_t) q * N)[gt + THREADS * j];
            float acc[16];
#pragma unroll
            for (int e = 0; e < RM; ++e) {
                uint32_t wr[PAIRS];
#pragma unroll
                for (int j = 0; j < PAIRS; ++j) wr[j] = reinterpret_cast<const uint32_t*>(WU + (size_t) (4 * g + e) * N * 2)[gt + THREADS * j];
#pragma unroll
                for (int q = 0; q < T; ++q) {
                    float s = 0.0f;
#pragma unroll
                    for (int j = 0; j < PAIRS; ++j) {
                        const float w0 = __uint_as_float(wr[j] << 16), w1 = __uint_as_float(wr[j] & 0xffff0000u);
                        s = fma_ftz(w0, x[q][j].x, s);
                        s = fma_ftz(w1, x[q][j].y, s);
                    }
                    acc[e * T + q] = s;
                }
            }
#pragma unroll
            for (int m = NV; m < 16; ++m) acc[m] = 0.0f;
            {
                float v[16];
#pragma unroll
                for (int m = 0; m < 16; ++m) v[m] = acc[m];
                const float sum = warp_sum16_ftz(v, lane);
                const int m = lane >> 1;
                if ((lane & 1) == 0 && m < NV) M.y.rpart[g][m * WARPS + gw2] = sum;
            }
            gbar(g);
            DS(14);
            if (gw2 == 0) {
                float v[16];
#pragma unroll
                for (int m = 0; m < 16; ++m) v[m] = lane < WARPS && m < NV ? M.y.rpart[g][m * WARPS + lane] : 0.0f;
                const float sum = warp_sum16_ftz(v, lane);
                const int m = lane >> 1, e = m / T, q = m - e * T;
                if ((lane & 1) == 0 && m < NV) {
                    a.logits[(size_t) q * NE + e0 + e] = sum;
#pragma unroll
                    for (int r = 0; r < NREP; ++r) lgr[((size_t) r * XT + q) * NE + e0 + e] = sum;
                }
            }
            cbar();
            DS(15);
            if (tid == 0) { redrel(&c[2]); if (st) st[2] = gtime(); }
        }
        // ================= top-10 (CTA 0: the only copy that is read)
        if (b == 0) {
        const float* LG = reinterpret_cast<const float*>(A);
        if (tid == 0) hop_copy(&c[2], NRTC, A, lgr + (size_t) (b % NREP) * XT * NE, (unsigned) T * NE * 4, &P.cbarA);
        // warm warp: the top-10 call site once on scratch smem while the logits are in flight, so its code is resident
        const bool warm = !rt_role && wp == CT / 32 - 1;
        constexpr int WARM_OFF = XT * NE * 4 + XT * CAND_WORDS * 4;
        static_assert(CT / 32 - 1 >= XT && WARM_OFF + 2 * KT * 4 + NE * 4 + CAND_WORDS * 4 <= SZ_A, "warm scratch");
        if (!warm) MW(&P.cbarA, PB_A);
        DS(16);
        if (wp < T || warm) {
            int32_t* ids = warm ? reinterpret_cast<int32_t*>(A + WARM_OFF) : b == 0 ? a.ids + (size_t) wp * KT : p.ids_all + ((size_t) b * T + wp) * KT;
            float* wts = warm ? reinterpret_cast<float*>(A + WARM_OFF + KT * 4) : b == 0 ? a.weights + (size_t) wp * KT : p.w_all + ((size_t) b * T + wp) * KT;
            top10_smem(warm ? reinterpret_cast<const float*>(A + WARM_OFF + 2 * KT * 4) : LG + (size_t) wp * NE, ids, wts, lane,
                       reinterpret_cast<uint32_t*>(A + (warm ? WARM_OFF + 2 * KT * 4 + NE * 4 : XT * NE * 4 + wp * CAND_WORDS * 4)));
        }
        if (warm) MW(&P.cbarA, PB_A);
        cbar();
        DS(17);
        }
    }
#undef DS
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
__device__ __forceinline__ unsigned ldca32(const void* p) { unsigned v; asm volatile("ld.global.ca.u32 %0,[%1];" : "=r"(v) : "l"(p)); return v; }
__device__ __forceinline__ unsigned short ldca16(const void* p) { unsigned short v; asm volatile("ld.global.ca.u16 %0,[%1];" : "=h"(v) : "l"(p)); return v; }
// pair_xfrag with L1-cached loads (L1 invalidated by the acquire): the replicas are written in this kernel (the nc path is only for kernel-constant data)
__device__ __forceinline__ void pair_xfrag_cg(XFrag& xf, int c, int hb, int g, int t, const uint8_t* x, int nb32, int ncols) {
#pragma unroll
    for (int k4 = 0; k4 < 4; ++k4) {
        const int k = 16 * c + 4 * hb + k4;
        xf.b0[k4] = xf.b1[k4] = 0;
        if (g < ncols) {
            const uint8_t* xb = x + ((size_t) g * nb32 + k) * kQ81;
            xf.b0[k4] = ldca32(reinterpret_cast<const unsigned int*>(xb + 4) + t);
            xf.b1[k4] = ldca32(reinterpret_cast<const unsigned int*>(xb + 20) + t);
        }
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int col = 2 * t + e;
            unsigned short h = 0;
            if (col < ncols) h = ldca16((x + ((size_t) col * nb32 + k) * kQ81));
            xf.d8[k4][e] = col < ncols ? __half2float(__ushort_as_half(h)) : 0.0f;
        }
    }
}
__device__ __forceinline__ float4 ldca128(const void* p) { float4 v; asm volatile("ld.global.ca.v4.f32 {%0,%1,%2,%3},[%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p)); return v; }
// ab_one with the row from global (kernel-constant: nc) and x from L1-cached loads (x is written in this kernel)
__device__ __forceinline__ float ab_one_g(const DecRecArgs& a, int head, bool is_beta, int t, int lane) {
    const float4* xt = reinterpret_cast<const float4*>(a.x + (size_t) t * a.n_embd);
    const uint4* w_g = reinterpret_cast<const uint4*>((is_beta ? a.w_beta : a.w_alpha) + (size_t) head * a.n_embd);
    float4 xa[NJ], xb[NJ];
    uint4 wv[NJ];
#pragma unroll
    for (int q = 0; q < NJ; ++q) {
        const int j = lane + 32 * q;
        xa[q] = ldca128(xt + 2 * j);
        xb[q] = ldca128(xt + 2 * j + 1);
        wv[q] = __ldg(w_g + j);
    }
    float v = 0.0f;
#pragma unroll
    for (int q = 0; q < NJ; ++q) {
        const uint4 w = wv[q];
        v = fmaf(__uint_as_float(w.x << 16), xa[q].x, v); v = fmaf(__uint_as_float(w.x & 0xffff0000u), xa[q].y, v);
        v = fmaf(__uint_as_float(w.y << 16), xa[q].z, v); v = fmaf(__uint_as_float(w.y & 0xffff0000u), xa[q].w, v);
        v = fmaf(__uint_as_float(w.z << 16), xb[q].x, v); v = fmaf(__uint_as_float(w.z & 0xffff0000u), xb[q].y, v);
        v = fmaf(__uint_as_float(w.w << 16), xb[q].z, v); v = fmaf(__uint_as_float(w.w & 0xffff0000u), xb[q].w, v);
    }
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (is_beta) return 1.0f / (1.0f + __expf(-v));
    const float u = v + a.dt[head];
    const float sp = u > 20.0f ? u : log1pf(__expf(u));
    return sp * a.ssm_a[head];
}
#endif
// dec_gdn_rec's conv_silu with its contraction spelled out (fma(v0, w0, v1*w1), then + v2*w2, + x*w3)
__device__ __forceinline__ float conv_silu_x(const float (&hist)[3], const float (&w)[4], const float (&xv)[XT], int t) {
    float win[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) { const int src = t + j; win[j] = src < 3 ? hist[src] : xv[src - 3]; }
    const float sum = fmaf(xv[t], w[3], fmaf(win[2], w[2], fmaf(win[0], w[0], win[1] * w[1])));
    return sum / (1.0f + __expf(-sum));
}
// ------------------------------------------------------------ pair (k_pair<2>: two 8-warp tile groups)
struct PSplit {
    int first, step, n, fz, nq;   // tile k: k < nq (or nq < 0): first + k step, else fz + k - nq
    __device__ __forceinline__ int tile(int k) const { return nq >= 0 && k >= nq ? fz + k - nq : first + k * step; }
};
constexpr int Q3_0 = 80;   // prt -3: CTAs [Q3_0, Q3_0 + 40) take 3 qkv tiles and compute the rec heads' alpha/beta
__device__ __forceinline__ PSplit p_split(int b, int G, int prt) {
    if (prt == -3) {   // qkv on every CTA (CTAs < 40: 3 tiles, the rest 4 = 2 rounds); z on the non-rec CTAs after their qkv
        const bool three = b >= Q3_0 && b < Q3_0 + 40;
        const int nq = three ? 3 : 4, fq = 4 * b - (b < Q3_0 ? 0 : b < Q3_0 + 40 ? b - Q3_0 : 40);
        if (b >= R_CTA0 && b < R_CTA0 + R_HV) return {fq, 1, nq, 0, nq};
        const int nn = R_CTA0, nzt = P_NT - P_NTQ, pz = nzt / nn, ez = nzt % nn;   // CTAs [Q3_0, Q3_0 + ez) take one more z
        const int xb = b < Q3_0 ? 0 : b < Q3_0 + ez ? b - Q3_0 : ez;
        return {fq, 1, nq + pz + (b >= Q3_0 && b < Q3_0 + ez ? 1 : 0), P_NTQ + b * pz + xb, nq};
    }
    if (prt == -2) {
        if (b >= R_CTA0 && b < R_CTA0 + R_HV) return {0, 1, 0, 0, 0};
        const int nn = G - R_HV, pq = P_NTQ / nn, eq = P_NTQ % nn, nzt = P_NT - P_NTQ, pz = nzt / nn, ez = nzt % nn;
        const int nq = pq + (b < eq ? 1 : 0);
        const int zb = (b - eq + nn) % nn;   // the z extras go to the CTAs with fewer qkv tiles first
        const int nz = pz + (zb < ez ? 1 : 0);
        return {b * pq + (b < eq ? b : eq), 1, nq + nz, P_NTQ + zb * pz + (zb < ez ? zb : ez), nq};
    }
    if (prt < 0) return {b, G, (b < P_NT ? (P_NT - 1 - b) / G + 1 : 0), 0, -1};
    if (b >= R_CTA0 && b < R_CTA0 + R_HV) return {P_NT - R_HV * prt + (b - R_CTA0) * prt, 1, prt, 0, -1};
    const int nn = G - R_HV, tiles = P_NT - R_HV * prt, per = tiles / nn, ex = tiles % nn;
    return {b * per + (b < ex ? b : ex), 1, per + (b < ex ? 1 : 0), 0, -1};
}
__device__ __forceinline__ void p_tile(const Perm& L, int tile, const uint8_t*& src, int& bytes, int& type, float*& y, int& n_out, int& row0, const KP& p) {
    if (tile < P_NTQ) { src = L.lwq + (size_t) tile * L.geo.tbq; bytes = L.geo.tbq; type = L.geo.tq; y = p.qkv; n_out = P_NQ; row0 = tile * 16; }
    else { const int t = tile - P_NTQ; src = L.lwz + (size_t) t * L.geo.tbz; bytes = L.geo.tbz; type = L.geo.tz; y = p.z; n_out = P_NZ; row0 = t * 16; }
}
// tile k of this CTA (k < ns: issued at the start) into slot k % ns
__device__ __forceinline__ void p_fill(const LayerC& L, const KP& p, const PSplit& ps, int k, char* sm, Perm& P) {
    const uint8_t* s; int by, ty, no, r0; float* y;
    p_tile(P, ps.tile(k), s, by, ty, y, no, r0, p);
    const int sl = k % P.geo.ns;
    mbar_expect_tx(&P.slot[sl], by);
    bulk_w2s(sm + (size_t) sl * P.geo.slotb, s, by, &P.slot[sl]);
}
__device__ __forceinline__ int p_mine(int b, int G) { return b < P_NT ? (P_NT - 1 - b) / G + 1 : 0; }

// ------------------------------------------------------------ the kernel
template <int T, int FQ, int FZ, int FO>
__global__ void __launch_bounds__(CT, 1) k_chain(const __grid_constant__ KP p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    extern __shared__ __align__(128) char sm_dyn[];
    char* sm = sm_dyn;
    MiscH& MH = *reinterpret_cast<MiscH*>(sm + OFF_MH);
    Perm& P = *reinterpret_cast<Perm*>(sm + OFF_PERM);
    RecS& RS_ = *reinterpret_cast<RecS*>(sm + OFF_REC);
    const int tid = threadIdx.x, wp = tid >> 5, lane = tid & 31, G = gridDim.x;
    unsigned tk = 0;
    [[maybe_unused]] const unsigned long long t_in = gtime();
    if (tid == 0) tk = atomicInc(&p.cnt[2 * NCTR], (unsigned) (G - 1));   // arrival ticket; wraps, never cleared
    if (tid == 0) {
        mbar_init(&P.full_wd, 1); mbar_init(&P.full_wu, 1); mbar_init(&P.cbarA, 1);
        for (int s = 0; s < P_SLOTS; ++s) mbar_init(&P.slot[s], 1);
        mbar_init(&P.ofull, 1); mbar_init(&P.oxbar, 1); mbar_init(&P.rbar, 1);
    }
    {   // the first layer's arguments (hc1, hc2, rec): one batch of independent loads, issued before any bulk copy
        constexpr int NA = (int) (sizeof(HcFusedArgs) / 4), NR = (int) (sizeof(DecRecArgs) / 4);
        static_assert(2 * NA + NR <= CT, "argument batch");
        int v = 0;
        if (tid < NA) v = reinterpret_cast<const int*>(&p.Lc.h1.a)[tid];
        else if (tid < 2 * NA) v = reinterpret_cast<const int*>(&p.Lc.h2.a)[tid - NA];
        else if (tid < 2 * NA + NR) v = reinterpret_cast<const int*>(&p.Lc.rec)[tid - 2 * NA];
        for (int i = tid; i < 512; i += CT) P.grid[i] = iq3s_grid[i];
        if (tid < NA) reinterpret_cast<int*>(&MH.a)[tid] = v;
        else if (tid < 2 * NA) reinterpret_cast<int*>(&P.a2)[tid - NA] = v;
        else if (tid < 2 * NA + NR) reinterpret_cast<int*>(&P.ra)[tid - 2 * NA] = v;
        if (tid == 0) { P.geo = p.Lc.geo; P.lwq = p.Lc.wq; P.lwz = p.Lc.wz; P.lwo = p.Lc.wo; P.lyq = static_cast<const uint8_t*>(p.Lc.rec.yq); P.lbo = const_cast<float*>(p.Lc.h2.a.bo_prev); P.ldn2 = p.Lc.dn2; }
    }
    if (tid < 64) P.sst[tid] = 0;
    if (tid == 0) P.vb = (int) (tk % (unsigned) G);
    __syncthreads();
    const int b = P.vb;
    [[maybe_unused]] unsigned long long* srec = nullptr;
#ifdef BASALT_PROFILE
    if (tid == 0) {
        srec = ::basalt::kernels::stamp::open(p.sc, ::basalt::kernels::stamp::kChain, blockIdx.x, gridDim.x, p.ord, p.T, t_in, blockIdx.x);
        if (srec) srec[::basalt::kernels::stamp::kA + 31] = (unsigned long long) b;
    }
#endif
    unsigned PH = 0;   // mbarrier parities (PB_*)
#define XS(n) do { if (p.det != nullptr && tid == 0) P.sst[n] = clock64(); } while (0)
    const PSplit ps = p_split(b, G, p.prt);
    const int mine = ps.n;
    const bool rec_role = b >= R_CTA0 && b < R_CTA0 + R_HV, out_role = b < O_NOUT / 16;
    // the first layer's hc1 rows
    if (wp == 0 && b < NDN) fill_dn(MH.a, p.Lc.dn1, b, sm + OFF_WD, &P.full_wd, lane);
    for (int i = 0; i < p.iters; ++i) {
        const LayerC& L = p.Lc;
        const LayerC& Ln = p.Lc;
        unsigned* c = p.cnt + (size_t) p.par * NCTR;
        unsigned long long* st = p.st != nullptr && tid == 0 ? p.st + ((size_t) i * G + b) * NST : nullptr;
#ifdef BASALT_PROFILE
        if (srec) st = srec + ::basalt::kernels::stamp::kA;
#endif
        if (i > 0) for (int w = tid; w < (int) (sizeof(HcFusedArgs) / 4); w += CT) reinterpret_cast<int*>(&P.a2)[w] = reinterpret_cast<const int*>(&L.h2.a)[w];
        if (i > 0) for (int w = tid; w < (int) (sizeof(DecRecArgs) / 4); w += CT) reinterpret_cast<int*>(&P.ra)[w] = reinterpret_cast<const int*>(&L.rec)[w];
        if (i > 0 && tid == 0) { P.geo = L.geo; P.lwq = L.wq; P.lwz = L.wz; P.lwo = L.wo; P.lyq = static_cast<const uint8_t*>(L.rec.yq); P.lbo = const_cast<float*>(L.h2.a.bo_prev); P.ldn2 = L.dn2; }
        // ======== hc1 (input: the previous layer's MoE output, released by CTA 0 `delay` after the last layer's end)
        const bool pre_pair = b >= NUP;   // non-up CTAs: their first pair tiles land during hc1's up (smem free after the down)
        auto pfill = [&] { if (pre_pair && tid == 0) { pfence(); for (int k = 0; k < P.geo.ns && k < mine; ++k) p_fill(L, p, ps, k, sm, P); } };
        hc_read<T, false>(p, &L.h1.a, sm, MH, P, PH, c + C_H1DN, (2 * i) & 3, p.xqr + (size_t) (i & 1) * NREP * XT * (N / 32) * kQ81,
                          st ? st + S_H1DN : nullptr, [&] {
            pdl_wait();
            pdl_launch_dependents();
            if (b == 0) for (int k = 0; k < NCTR; ++k) p.cnt[(1 - p.par) * NCTR + k] = 0;   // the set the next launch uses
            if (st) st[S_REL] = gtime();
        }, NoFill(), pfill, [&] {   // the first layer's hc1 up slices, after the input release
            if (i == 0 && tid == 32) fill_wu(p.Lc.h1, nullptr, b, false, sm + OFF_WU, &P.full_wu);
            if (rec_role && tid == 96) {   // the head's state tile and conv history, read after the pair
                const int head = b - R_CTA0, qh = head % R_HK, qk = S * R_HK;
                const float* st0 = L.rec.state;
                const float* hist = st0 + (size_t) S * R_HV * S;
                pf_l2(st0 + (size_t) head * S * S, (size_t) S * S * 4);
                pf_l2(hist + (size_t) qh * S * 3, (size_t) S * 3 * 4);
                pf_l2(hist + (size_t) (qk + qh * S) * 3, (size_t) S * 3 * 4);
                pf_l2(hist + (size_t) (2 * qk + head * S) * 3, (size_t) S * 3 * 4);
            }
        }, i > 0);
        // ======== pair: the first P_SLOTS tiles into the ring (hc1 is done with this CTA's smem)
        if (!pre_pair && tid == 0) { pfence(); for (int k = 0; k < P.geo.ns && k < mine; ++k) p_fill(L, p, ps, k, sm, P); }
        {
            const int gi = wp >> 3, lw = wp & 7, ltid = tid & 255, g = lane >> 2, t = lane & 3;
            const int hb = lw & 3, n = hb & 1, nchunk = P_NIN / 512, nb32 = P_NIN / 32;
            float* gred = reinterpret_cast<float*>(sm + OFF_PRED) + gi * 1024;
            const uint8_t* xr = p.xqr + (size_t) (i & 1) * NREP * XT * (N / 32) * kQ81 + (size_t) (b % NREP) * XT * (N / 32) * kQ81;
            if (tid == 0) { spin_ge(&c[C_H1UP], NUP); if (st) st[S_PHOP] = gtime(); }
            cbar();
            XFrag xs[3];
#pragma unroll
            for (int j = 0; j < 3; ++j)
                if (mine > 0 && (lw >> 2) + 2 * j < nchunk) pair_xfrag_cg(xs[j], (lw >> 2) + 2 * j, hb, g, t, xr, nb32, T);
            if (st) st[S_PX] = gtime();
            if (ltid == 0 && (gi >= mine || ps.tile(gi) >= P_NTQ)) redrel(&c[C_PAIR]);   // no qkv tile in this group
            for (int k = gi; k < mine; k += 2) {
                const int ns = P.geo.ns, slot = k % ns;
                const uint8_t* s; int by, ty, no, row0; float* y;
                p_tile(P, ps.tile(k), s, by, ty, y, no, row0, p);
                mwait(&P.slot[slot], ((PH >> (PB_SLOT + slot)) ^ (unsigned) (k / ns)) & 1u);   // slot uses before tile k: k / ns
                if (st && k == 0) st[S_PT0] = gtime();
                const uint8_t* sl = reinterpret_cast<const uint8_t*>(sm) + (size_t) slot * P.geo.slotb;
                const int row_bytes = (P_NIN / 256) * sb_bytes(ty);
                float acc[4] = {0.f, 0.f, 0.f, 0.f};
                by_tq_tz<FQ, FZ>(ty, P.geo.tq, [&](auto TY) { pair_terms_xr<decltype(TY)::value>(acc, sl + g * row_bytes, sl + (g + 8) * row_bytes, lw, hb, n, t, nchunk, xs, P.grid); });
                float* rw = gred + lw * 16 * kMaxCols;
                rw[g * kMaxCols + 2 * t] = acc[0];
                rw[g * kMaxCols + 2 * t + 1] = acc[1];
                rw[(g + 8) * kMaxCols + 2 * t] = acc[2];
                rw[(g + 8) * kMaxCols + 2 * t + 1] = acc[3];
                barn(2 + gi, 256);
                const bool qrel = ps.tile(k) < P_NTQ && (k + 2 >= mine || ps.tile(k + 2) >= P_NTQ);   // this group's last qkv tile
                if (ltid == 0 && k + ns < mine && !qrel) { pfence(); p_fill(L, p, ps, k + ns, sm, P); }   // the slot's next tile
                if (ltid < 16 * kMaxCols) {
                    const int row = ltid / kMaxCols, col = ltid % kMaxCols;
                    if (col < T) {
                        const float* rr2 = gred + row * kMaxCols + col;
                        constexpr int WS = 16 * kMaxCols;
                        const float v = ((rr2[0] + rr2[4 * WS]) + (rr2[WS] + rr2[5 * WS])) + ((rr2[2 * WS] + rr2[6 * WS]) + (rr2[3 * WS] + rr2[7 * WS]));
                        y[(size_t) col * no + row0 + row] = v;
                    }
                }
                barn(2 + gi, 256);
                if (p.prt == -3 && b >= Q3_0 && b < Q3_0 + 40 && gi == 1 && k == 1) {   // the rec heads' alpha/beta (job = head * 2T + 2t + beta)
                    for (int j = (b - Q3_0) * 8 + lw; j < R_HV * 2 * T; j += 320) {
                        const int head = j / (2 * T), r = j % (2 * T);
                        const float v = ab_one_g(P.ra, head, r & 1, r >> 1, lane);
                        if (lane == 0) p.abv[head * 2 * XT + r] = v;
                    }
                    barn(2 + gi, 256);
                }
                if (ltid == 0 && qrel) {
                    redrel(&c[C_PAIR]);
                    if (p.st != nullptr) p.st[((size_t) i * G + b) * NST + 21 + gi] = gtime();
                }
                if (ltid == 0 && k + ns < mine && qrel) { pfence(); p_fill(L, p, ps, k + ns, sm, P); }   // after the release
                if (st && k == 0) st[S_PT1] = gtime();
            }
            if (ltid == 0) redrel(&c[C_PZ]);
            // every slot's uses by both groups
            for (int k = 0; k < mine; ++k) PH ^= 1u << (PB_SLOT + k % P.geo.ns);
            cbar();
            if (tid == 0 && st) st[S_PAIR] = gtime();
        }
        // ======== out tile into smem (the pair's ring is free; rec lives above it)
        if (out_role && !rec_role && tid == 32) {
            pfence();
            mbar_expect_tx(&P.ofull, 16 * P.geo.orb);
            for (int r = 0; r < 16; ++r) bulk_w2s(sm + r * P.geo.ors, P.lwo + (size_t) (b * 16 + r) * P.geo.orb, P.geo.orb, &P.ofull);
        }
        // ======== rec (48 CTAs, head = b - R_CTA0; rec_rank's code with RecS)
        if (rec_role) {
            const DecRecArgs& a = P.ra;   // copied at the iteration start
            const int head = b - R_CTA0, rank = tid / 128, ltid = tid % 128, rg = ltid / 32, bid = 8 + rank, col = rank * 32 + lane;
            const int qh = head % a.h_k, qk = S * a.h_k, value_dim = S * a.h_v, Cc = a.C;
            const int cq = qh * S + ltid, ck = qk + qh * S + ltid, cv = 2 * qk + head * S + col;
            // the head's alpha/beta (job j = 2t + beta on warp j; mixed1 is visible since the pair's hop), into every rank's copy
            XS(59);

            float wq[4], wk[4], wv[4], hq[3], hk[3], hv[3];
            {
                const float* hist = a.state + (size_t) S * a.h_v * S;
#pragma unroll
                for (int j = 0; j < 4; ++j) { wq[j] = __ldg(a.conv_w + cq * 4 + j); wk[j] = __ldg(a.conv_w + ck * 4 + j); wv[j] = __ldg(a.conv_w + cv * 4 + j); }
#pragma unroll
                for (int j = 0; j < 3; ++j) { hq[j] = hist[cq * 3 + j]; hk[j] = hist[ck * 3 + j]; hv[j] = hist[cv * 3 + j]; }
            }
            const float gam = __ldg(a.gamma + col);
            XS(54);
            if (st) st[S_RPRE] = gtime();
            if (tid == 0) { spin_ge(&c[C_PAIR], (unsigned) (2 * G)); if (st) st[S_RHOP] = gtime(); }
            cbar();
            if (rg == 0 && lane < 2 * T) {   // every rank's copy of the head's alpha/beta (computed in the pair phase)
                const float v = __uint_as_float(ldca32(p.abv + head * 2 * XT + lane));
                ((lane & 1) ? RS_.b[rank] : RS_.g[rank])[lane >> 1] = v;
            }
            XS(55);
            float xq[XT], xk[XT], xv[XT], zz[XT];
#pragma unroll
            for (int t = 0; t < XT; ++t) {
                xq[t] = t < T ? a.qkv[(size_t) t * Cc + cq] : 0.0f;
                xk[t] = t < T ? a.qkv[(size_t) t * Cc + ck] : 0.0f;
                xv[t] = t < T && rg == 1 ? a.qkv[(size_t) t * Cc + cv] : 0.0f;
                zz[t] = 0.0f;
            }
            // the state tile: after the hop and the qkv loads (an in-flight 64 KB copy delays this SM's loads and polls)
            if (tid == 0) { pfence(); mbar_expect_tx(&P.rbar, S * S * 4); tma_2d(sm, reinterpret_cast<const CUtensorMap*>(&L.stm), head * S, 0, &P.rbar); }
            float yq[XT], yk[XT];
#pragma unroll
            for (int t = 0; t < XT; ++t) {
                if (t >= T) break;
                yq[t] = conv_silu_x(hq, wq, xq, t);
                yk[t] = conv_silu_x(hk, wk, xk, t);
                float sq = yq[t] * yq[t], sk = yk[t] * yk[t];
                for (int o = 16; o > 0; o >>= 1) { sq += __shfl_xor_sync(0xffffffffu, sq, o); sk += __shfl_xor_sync(0xffffffffu, sk, o); }
                if (lane == 0) { RS_.part[rank][0][t][rg] = sq; RS_.part[rank][1][t][rg] = sk; }
                if (rg == 1) {
                    const float yv = conv_silu_x(hv, wv, xv, t);
                    RS_.v[rank][t][lane] = yv;
                    a.h[(size_t) t * Cc + cv] = yv;
                }
            }
            barn(bid, 128);
            if (st) st[S_RCONV] = gtime();
            const bool write_qk = rank == 0 && head < a.h_k;
#pragma unroll
            for (int t = 0; t < XT; ++t) {
                if (t >= T) break;
                const float ssq = RS_.part[rank][0][t][0] + RS_.part[rank][0][t][1] + RS_.part[rank][0][t][2] + RS_.part[rank][0][t][3];
                const float ssk = RS_.part[rank][1][t][0] + RS_.part[rank][1][t][1] + RS_.part[rank][1][t][2] + RS_.part[rank][1][t][3];
                yq[t] *= rsqrtf(ssq + a.eps);
                yk[t] *= rsqrtf(ssk + a.eps);
                if (rank == 0) { RS_.qn[t][ltid] = yq[t]; RS_.kn[t][ltid] = yk[t]; }
                if (write_qk) { a.h[(size_t) t * Cc + cq] = yq[t]; a.h[(size_t) t * Cc + ck] = yk[t]; }
            }
            if (ltid < T) {
                if (rank == 0) a.gate[(size_t) ltid * a.h_v + head] = RS_.g[rank][ltid];
                if (rank == 1) a.beta[(size_t) ltid * a.h_v + head] = RS_.b[rank][ltid];
            }
            MW(&P.rbar, PB_REC);   // the state tile (tensor copy issued after the pair)
            XS(56);
            float sreg[RPG];
            {
                const float* stile = reinterpret_cast<const float*>(sm);
#pragma unroll
                for (int r = 0; r < RPG; ++r) sreg[r] = stile[(rg * RPG + r) * S + col];
            }
            cbar();   // the state tile is consumed (the out tile lands there); also the ranks' qn/kn
            XS(57);
            if (out_role && tid == 32) {
                pfence();
                mbar_expect_tx(&P.ofull, 16 * P.geo.orb);
                for (int r = 0; r < 16; ++r) bulk_w2s(sm + r * P.geo.ors, P.lwo + (size_t) (b * 16 + r) * P.geo.orb, P.geo.orb, &P.ofull);
            }
            float wsum_t = 0.0f;
            for (int t = 0; t < T; ++t) {
                const float gg = __expf(RS_.g[rank][t]);
                float kv = 0.0f;
#pragma unroll
                for (int r = 0; r < RPG; ++r) kv = fmaf(sreg[r], RS_.kn[t][rg * RPG + r], kv);
                RS_.red_kv[rank][rg][lane] = kv;
                barn(bid, 128);
                const float kv_col = RS_.red_kv[rank][0][lane] + RS_.red_kv[rank][1][lane] + RS_.red_kv[rank][2][lane] + RS_.red_kv[rank][3][lane];
                const float delta = (RS_.v[rank][t][lane] - gg * kv_col) * RS_.b[rank][t];
                float o = 0.0f;
#pragma unroll
                for (int r = 0; r < RPG; ++r) {
                    sreg[r] = fmaf(gg, sreg[r], RS_.kn[t][rg * RPG + r] * delta);
                    o = fmaf(sreg[r], RS_.qn[t][rg * RPG + r], o);
                }
                RS_.red_o[rank][rg][lane] = o;
                barn(bid, 128);
                if (rg == 0)
                    RS_.oc[rank][t][lane] = (RS_.red_o[rank][0][lane] + RS_.red_o[rank][1][lane] + RS_.red_o[rank][2][lane] + RS_.red_o[rank][3][lane]) * rsqrtf((float) S);
            }
            if (st) st[S_RCHAIN] = gtime();
            if (rg == 0)   // the sums of squares after the chain (the same butterfly per token)
                for (int t = 0; t < T; ++t) {
                    const float sq_part = sq32(RS_.oc[rank][t][lane]);
                    if (lane == t) wsum_t = sq_part;
                }
            if (rg == 0 && lane < T) RS_.wsum[lane][rank] = wsum_t;
            if (tid == 0) spin_ge(&c[C_PZ], (unsigned) (2 * G));
            cbar();
            if (rg == 0)
#pragma unroll
                for (int t = 0; t < XT; ++t) zz[t] = t < T ? a.z[(size_t) t * value_dim + head * S + col] : 0.0f;
            if (rg == 0) {
#pragma unroll
                for (int t = 0; t < XT; ++t) {
                    if (t >= T) break;
                    const float ss = RS_.wsum[t][0] + RS_.wsum[t][1] + RS_.wsum[t][2] + RS_.wsum[t][3];
                    norm_q8(a, t, head, rank, lane, RS_.oc[rank][t][lane], ss, gam, zz[t]);
                }
            }
            cbar();
            if (tid == 0) { redrel(&c[C_REC]); if (st) st[S_REC] = gtime(); }
        }
        // ======== out (dense_tc_one<type, 24 logical warps> on 16 warps)
        if (out_role) {
            const int g = lane >> 2, t = lane & 3, nb32 = O_NIN / 32;
            const int ors = P.geo.ors, osb = sb_bytes(P.geo.to);
            constexpr bool oxs_fixed = FO != 0 && make_geo(23, 23, FO).ox + T * (O_NIN / 32) * kQ81 <= OFF_PERM;
            const bool oxs = FO != 0 ? oxs_fixed : P.geo.ox + T * nb32 * kQ81 <= OFF_PERM;   // x staged in smem (else read from L2 through L1)
            const uint8_t* xsm = oxs ? reinterpret_cast<const uint8_t*>(sm + P.geo.ox) : P.lyq;
            float* red = reinterpret_cast<float*>(sm + P.geo.ored);
            constexpr int RC = O_RC;
            if (oxs) {
                if (tid == 0) hop_copy(&c[C_REC], R_HV, sm + P.geo.ox, P.lyq, (unsigned) (T * nb32 * kQ81), &P.oxbar);
                MW(&P.oxbar, PB_OX);
            } else {
                if (tid == 0) spin_ge(&c[C_REC], R_HV);
                cbar();
            }
            if (st) st[S_OX] = gtime();
            MW(&P.ofull, PB_OW);
            if (st) st[S_OW] = gtime();
            const uint8_t* slot = reinterpret_cast<const uint8_t*>(sm);
            XS(29);
            for (int lw = wp; lw < O_NLW; lw += 16) {
                if (lw == 16) XS(30);
                const int hb = lw & 3, sbi = hb >> 1, n = hb & 1;
                const uint8_t* rowA = slot + g * ors + sbi * osb;
                const uint8_t* rowB = slot + (g + 8) * ors + sbi * osb;
                float acc[4] = {0.f, 0.f, 0.f, 0.f};
                XFrag xf[2];
#pragma unroll
                for (int cc = 0; cc < 2; ++cc) {
                    const int cch = (lw >> 2) + cc * (O_NLW / 4);
#pragma unroll
                    for (int k4 = 0; k4 < 4; ++k4) {
                        const int k = 16 * cch + 4 * hb + k4;
                        xf[cc].b0[k4] = xf[cc].b1[k4] = 0;
                        if (g < T) {
                            const uint8_t* xb = xsm + ((size_t) g * nb32 + k) * kQ81;
                            if (oxs) {
                                xf[cc].b0[k4] = *(reinterpret_cast<const unsigned int*>(xb + 4) + t);
                                xf[cc].b1[k4] = *(reinterpret_cast<const unsigned int*>(xb + 20) + t);
                            } else {
                                xf[cc].b0[k4] = ldca32(xb + 4 + 4 * t);
                                xf[cc].b1[k4] = ldca32(xb + 20 + 4 * t);
                            }
                        }
#pragma unroll
                        for (int e2 = 0; e2 < 2; ++e2) {
                            const int col = 2 * t + e2;
                            const uint8_t* xd = xsm + ((size_t) col * nb32 + k) * kQ81;
                            xf[cc].d8[k4][e2] = col < T ? (oxs ? __low2float(*reinterpret_cast<const half2*>(xd)) : __half2float(__ushort_as_half(ldca16(xd)))) : 0.0f;
                        }
                    }
                }
                by_to<FO>(P.geo.to, [&](auto TY) {
                    constexpr int TT = decltype(TY)::value, SB = Fmt<TT>::SB;
                    uint32_t t0 = 0, t1 = 0, t2 = 0, t3 = 0;
                    if constexpr (TT == 23) {
                        const uint32_t* tbl = reinterpret_cast<const uint32_t*>(dtc_iq4nl);
                        t0 = tbl[0]; t1 = tbl[1]; t2 = tbl[2]; t3 = tbl[3];
                    }
#pragma unroll
                    for (int cc = 0; cc < 2; ++cc) {
                        const int cch = (lw >> 2) + cc * (O_NLW / 4);
                        const uint8_t* const pp[2] = {rowA + cch * 2 * SB, rowB + cch * 2 * SB};
                        half_sb_terms<TT>(acc, pp, n, t, xf[cc], t0, t1, t2, t3, P.grid);
                    }
                });
                float* rw = red + lw * 16 * RC;
                if (2 * t + 1 < RC) {
                    rw[g * RC + 2 * t] = acc[0];
                    rw[g * RC + 2 * t + 1] = acc[1];
                    rw[(g + 8) * RC + 2 * t] = acc[2];
                    rw[(g + 8) * RC + 2 * t + 1] = acc[3];
                }
            }
            XS(21);
            cbar();
            XS(22);
            if (tid < 16 * RC) {
                const int row = tid / RC, col = tid % RC;
                if (col < T) {
                    constexpr int WS = 16 * RC;
                    const float* rr2 = red + row * RC + col;
                    float h[4];
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        float v = rr2[q * WS];
#pragma unroll
                        for (int j = 1; j < O_NLW / 4; ++j) v += rr2[(4 * j + q) * WS];
                        h[q] = v;
                    }
                    const float bov = (h[0] + h[1]) + (h[2] + h[3]);
                    P.lbo[(size_t) col * O_NOUT + b * 16 + row] = bov;
                }
            }
            XS(23);
            cbar();
            XS(24);
            if (tid == 0) { redrel(&c[C_OUT]); if (st) st[S_OUT] = gtime(); }
            XS(25);
        }
        // ======== hc2: its rows into smem (the out tile / rec state are consumed in this CTA)
        if (wp == 0 && b < NDN) { pfence(); XS(27); fill_dn(P.a2, P.ldn2, b, sm + OFF_WD, &P.full_wd, lane); }
        XS(26);
        auto lfill = [&] { if (b < NUP && tid == 32) { pfence(); fill_wu(L.h2, P.a2.w_router, b, true, sm + OFF_WU, &P.full_wu); } };
        auto dfill = [&] { if (b >= NUP && tid == 32) { pfence(); fill_wu(L.h2, P.a2.w_router, b, true, sm + OFF_WU, &P.full_wu); } };
        hc_read<T, true>(p, &P.a2, sm, MH, P, PH, c + C_H2DN, (2 * i + 1) & 3, nullptr, st ? st + S_H2DN : nullptr,
                         [&] { spin_ge(&c[C_OUT], O_NOUT / 16); if (st) st[S_H2SEEN] = gtime(); }, lfill, dfill);
        if (tid == 0) { redrel(&c[C_END]); if (st) st[S_END] = gtime(); if (p.det) { P.sst[62] = clock64();
            for (int n = 0; n < 64; ++n) { p.det[((size_t) i * G + b) * 64 + n] = P.sst[n]; P.sst[n] = 0; } } }
        // the next layer's hc1 rows
        if (i + 1 < p.iters) {
            if (wp == 1 && b < NDN) pfence();
            if (wp == 1 && b < NDN) fill_dn(Ln.h1.a, Ln.dn1, b, sm + OFF_WD, &P.full_wd, lane);
            if (tid == 64) pfence();
            if (tid == 64) fill_wu(Ln.h1, nullptr, b, false, sm + OFF_WU, &P.full_wu);
            if (tid == 96 && b < 10) pf_l2((const char*) Ln.h1.a.w_norm + (size_t) b * (D * 4 / 10), D * 4 / 10);   // the next hc1's norm weights
        }
    }
#ifdef BASALT_PROFILE
    __syncthreads();
    if (srec) srec[::basalt::kernels::stamp::kExit] = gtime();
#endif
#endif
}
}  // namespace mkc
