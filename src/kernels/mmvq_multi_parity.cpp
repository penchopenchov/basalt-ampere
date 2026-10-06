// src/kernels/mmvq_multi_parity.cpp - the multi_exact contract of native_mmvq, tested bitwise.
//
//     build/mmvq_multi_parity
//
// WHY IT EXISTS.  `include/basalt/kernels/native_mmvq.hpp` says of `native_mmvq_set_multi_exact`:
//
//     "true (default): the ncols == 1 layout, every column bitwise equal to a single-column call"
//
// and `include/basalt/core/verify.hpp` counts "multi-column MMVQ in exact mode" among the reasons the verify window
// produces what greedy decode would, bit for bit.  Nothing tested that sentence: `iq_parity` calls `native_mmvq`
// with two columns and compares against a float64 reference at 2e-2, which cannot see a difference between one
// column and T columns.
//
// WHAT IT COMPARES.  The SAME Q8_1 bytes: one T-column call against T single-column calls, each reading column j
// from the same buffer (the header: "its Q8_1 blocks at j * n_in / 32").  The quantization is shared, so the only
// thing under test is how the columns are laid out.
//
// THE DATA.  The weights are random bytes, except each block's fp16 scales (`d`, and `dmin` where the format has
// one), which are rewritten as normal fp16 values in +-[2^-10, 2^-5): random bytes there would put inf and NaN in
// the outputs, and on the GPU every NaN comes out as the same canonical bit pattern, so a NaN output compares equal
// whatever operations produced it and proves nothing.  With finite scales and finite normal activations (the
// precondition the header puts on the input) every output is finite, and a non-finite one fails the test.
//
// THE NEGATIVE CONTROL.  The same comparison with `multi_exact` off must find differences, or the test has no power
// and its pass means nothing.  Where it has power depends on T: in `native_mmvq.cu` the generic layout uses
// `NW = NCOLS <= 4 ? 4 : 2` warps, and with `WARPS = 4` and `BPI = F::BPI * NW / WARPS` it coincides with the
// exact layout at T <= 4.  So the control is ASSERTED on T > 4 for every case, and at T <= 4 its silence is
// reported as that coincidence rather than skipped.  The control needs a row long enough for the two layouts to
// group blocks differently: IQ4_XS takes 16 blocks per iteration with 4 warps and 8 with 2, so at n_in = 2048 (8
// blocks of 256) each thread holds at most one block in both layouts and they coincide; its case uses n_in = 4096.
//
// THE TENSOR-CORE PATH (STRATA_MMVQ_MMA).  Where the device runs it, IQ4_XS calls with T > 1 in exact mode take
// native_iq4_xs_mma_kernel.  The last section turns it on with native_mmvq_set_mma and compares, on the same Q8_1
// bytes, its T-column call against the exact dp4a multi-column kernel (the setter off) and against T single-column
// calls (the ncols == 1 kernel, which the tensor-core path never replaces), bitwise, over row tails, rows of 1, 10,
// 16, 24 and 48 blocks, both block sizes the launcher picks, and weights / activations off their 8- and 16-byte
// alignment; the launch counter must move on every tensor-core call, or the check checked nothing.  With
// STRATA_MMVQ_MMA=1 in the environment the sweeps above take it for their IQ4_XS case as well.
//
//     build/mmvq_multi_parity [--bench]      --bench: the dp4a and tensor-core kernels timed at the model's shapes
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cstring>
#include <random>
#include <vector>

namespace {

struct Case {
    const char* name;
    int type;          // stable GGML type id, from the list in native_mmvq.hpp
    int n_in;
    int n_out;
    int block_elems;   // elements per weight block
    int block_bytes;
    int scale_at[2];   // byte offsets of the block's fp16 scales in native_mmvq.cu's block structs; -1 = none
};

// the types a head or a projection can have in this model
const Case CASES[] = {
    {"Q4_0", 2, 2048, 512, 32, 18, {0, -1}},          // Q40Block: half d
    {"Q8_0", 8, 2048, 512, 32, 34, {0, -1}},          // Q80Block: half d
    {"Q4_K", 12, 2048, 512, 256, 144, {0, 2}},        // Q4KBlock: half2 dm
    {"Q5_K", 13, 2048, 512, 256, 176, {0, 2}},        // Q5KBlock: half2 dm
    {"Q6_K", 14, 2048, 512, 256, 210, {208, -1}},     // Q6KBlock: half d after ql, qh, scales
    {"IQ4_XS", 23, 4096, 512, 256, 136, {0, -1}},     // IQ4XSBlock: half d; n_in 4096, see THE NEGATIVE CONTROL
    {"Q5_K wide", 13, 4096, 2048, 256, 176, {0, 2}},
};

// a normal fp16 in +-[2^-10, 2^-5): exponent field 5..9 (bias 15), any mantissa, either sign
uint16_t sane_half(std::mt19937& rng) {
    const uint32_t r = rng();
    return (uint16_t) (((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
}

// T = 4 is the last width where the two layouts coincide, T = 5 the first where they differ
const int WIDTHS[] = {1, 2, 3, 4, 5, 6, 8};
constexpr int COINCIDE_MAX_T = 4;

bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::printf("CUDA: %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

// Returns the number of outputs that differ bit for bit; counts the non-finite outputs and the differences on
// outputs that are finite in both paths.
long long compare(const Case& c, int T, cudaStream_t s, long long& nonfinite, long long& diff_finite, bool& ran) {
    ran = false;
    if (!basalt::kernels::native_mmvq_supported(c.type)) return -1;

    const std::size_t wbytes = basalt::kernels::native_mmvq_weight_bytes(c.type, c.n_in, c.n_out);
    const std::size_t qcol = basalt::kernels::native_q8_1_bytes(c.n_in, 1);   // Q8_1 bytes of ONE column
    const std::size_t qall = basalt::kernels::native_q8_1_bytes(c.n_in, T);
    if (qall != qcol * (std::size_t) T) {
        std::printf("%-12s T=%d: the Q8_1 bytes of T columns (%zu) are not T times those of one (%zu): the test "
                    "assumes contiguous columns and they are not\n", c.name, T, qall, qcol);
        return -1;
    }

    std::mt19937 rng(1234u + (unsigned) c.type * 97u + (unsigned) T);
    std::vector<uint8_t> w(wbytes);
    for (auto& b : w) b = (uint8_t) (rng() & 0xff);
    const std::size_t n_blocks = (std::size_t) c.n_out * (std::size_t) (c.n_in / c.block_elems);
    if (n_blocks * (std::size_t) c.block_bytes != wbytes) {
        std::printf("%-12s T=%d: %zu blocks of %d bytes are not the %zu weight bytes native_mmvq expects\n", c.name,
                    T, n_blocks, c.block_bytes, wbytes);
        return -1;
    }
    for (std::size_t k = 0; k < n_blocks; ++k)
        for (int at : c.scale_at)
            if (at >= 0) {
                const uint16_t h = sane_half(rng);
                std::memcpy(&w[k * (std::size_t) c.block_bytes + (std::size_t) at], &h, 2);
            }
    std::vector<float> x((std::size_t) T * c.n_in);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : x) v = nd(rng);

    void* dw = nullptr;
    void* xq = nullptr;
    float* dx = nullptr;
    float* dmulti = nullptr;
    float* dsingle = nullptr;
    const std::size_t ybytes = (std::size_t) T * c.n_out * 4;
    if (!ck(cudaMalloc(&dw, wbytes), "malloc w") || !ck(cudaMalloc(&xq, qall), "malloc xq") ||
        !ck(cudaMalloc(&dx, x.size() * 4), "malloc x") || !ck(cudaMalloc(&dmulti, ybytes), "malloc multi") ||
        !ck(cudaMalloc(&dsingle, ybytes), "malloc single"))
        return -1;
    if (!ck(cudaMemcpy(dw, w.data(), wbytes, cudaMemcpyHostToDevice), "copy w") ||
        !ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "copy x"))
        return -1;

    // ONE quantization, shared by both paths: the quantization is not under test
    basalt::kernels::quantize_q8_1_rows(dx, T, c.n_in, xq, s);
    if (!ck(cudaStreamSynchronize(s), "quantize")) return -1;

    try {
        basalt::kernels::native_mmvq(c.type, dw, xq, dmulti, c.n_in, c.n_out, T, s);
        for (int j = 0; j < T; ++j)
            basalt::kernels::native_mmvq(c.type, dw, (const uint8_t*) xq + (std::size_t) j * qcol,
                                         dsingle + (std::size_t) j * c.n_out, c.n_in, c.n_out, 1, s);
    } catch (const std::exception& e) {
        std::printf("%-12s T=%d: %s\n", c.name, T, e.what());
        return -1;
    }
    if (!ck(cudaStreamSynchronize(s), "mmvq")) return -1;

    std::vector<uint32_t> a((std::size_t) T * c.n_out), b(a.size());
    if (!ck(cudaMemcpy(a.data(), dmulti, ybytes, cudaMemcpyDeviceToHost), "read multi") ||
        !ck(cudaMemcpy(b.data(), dsingle, ybytes, cudaMemcpyDeviceToHost), "read single"))
        return -1;

    long long diff = 0;
    nonfinite = 0;
    diff_finite = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        float fa, fb;
        std::memcpy(&fa, &a[i], 4);
        std::memcpy(&fb, &b[i], 4);
        const bool both_finite = std::isfinite(fa) && std::isfinite(fb);
        if (!both_finite) ++nonfinite;
        if (a[i] != b[i]) {
            ++diff;
            if (both_finite) ++diff_finite;
        }
    }
    cudaFree(dw); cudaFree(xq); cudaFree(dx); cudaFree(dmulti); cudaFree(dsingle);
    ran = true;
    return diff;
}

struct Totals {
    long long checked = 0, diff = 0, diff_finite = 0, nonfinite = 0;   // all widths
    long long diff_finite_wide = 0;                                     // T > COINCIDE_MAX_T only
    int powerless_cases = 0;   // cases where the control found no difference at any T > COINCIDE_MAX_T
};

int sweep(bool exact, cudaStream_t s, Totals& t) {
    basalt::kernels::native_mmvq_set_multi_exact(exact);
    if (basalt::kernels::native_mmvq_multi_exact() != exact) {
        std::printf("native_mmvq_multi_exact() does not reflect what was set\n");
        return 1;
    }
    std::printf("\n=== multi_exact %s ===\n", exact ? "ON (the default)" : "off (negative control)");
    std::printf("%-12s %3s %10s %10s %14s %10s\n", "type", "T", "outputs", "bits diff", "finite diff", "nonfinite");
    int bad = 0;
    for (const Case& c : CASES) {
        long long case_wide = 0;
        for (int T : WIDTHS) {
            long long nf = 0, dfin = 0;
            bool ran = false;
            const long long diff = compare(c, T, s, nf, dfin, ran);
            if (!ran) { ++bad; continue; }
            const long long n = (long long) T * c.n_out;
            t.checked += n;
            t.diff += diff;
            t.diff_finite += dfin;
            t.nonfinite += nf;
            if (T > COINCIDE_MAX_T) { t.diff_finite_wide += dfin; case_wide += dfin; }
            const char* verdict;
            if (exact) verdict = diff == 0 ? "ok" : "FAIL";
            else if (dfin > 0) verdict = "control sees it";
            else verdict = T <= COINCIDE_MAX_T ? "layouts coincide here (NW = 4)" : "no power at this T";
            std::printf("%-12s %3d %10lld %10lld %14lld %10lld  %s\n", c.name, T, n, diff, dfin, nf, verdict);
            if (exact && diff != 0) ++bad;
        }
        if (!exact && case_wide == 0) ++t.powerless_cases;
    }
    return bad;
}

// ---- the tensor-core path
struct MmaCase {
    int n_in, n_out;
};
// row tails (n_out % 16, % 32), 1 / 10 / 16 / 24 / 48 blocks per row, and n_out large enough for 32-row blocks
const MmaCase MMA_CASES[] = {
    {256, 40}, {2560, 1000}, {4096, 520}, {6144, 2560}, {12288, 300}, {512, 8200}, {2560, 10240},
};
constexpr int IQ4XS = 23, IQ4XS_BYTES = 136;

struct Buffers {
    uint8_t* w = nullptr;   // the weights at w + w_off
    uint8_t* xq = nullptr;  // the Q8_1 bytes at xq + x_off
    float* x = nullptr;
    float* y_mma = nullptr;
    float* y_old = nullptr;
    float* y_one = nullptr;
    Buffers() = default;
    Buffers(const Buffers&) = delete;
    Buffers& operator=(const Buffers&) = delete;
    ~Buffers() {
        cudaFree(w); cudaFree(xq); cudaFree(x); cudaFree(y_mma); cudaFree(y_old); cudaFree(y_one);
    }
};

// random IQ4_XS weights with sane fp16 scales and normal activations, quantized once; offsets in bytes (multiples of 4)
bool fill(Buffers& b, int n_in, int n_out, int T, int w_off, int x_off, unsigned seed, cudaStream_t s) {
    const std::size_t wbytes = basalt::kernels::native_mmvq_weight_bytes(IQ4XS, n_in, n_out);
    const std::size_t qall = basalt::kernels::native_q8_1_bytes(n_in, T);
    const std::size_t ybytes = (std::size_t) T * n_out * 4;
    std::mt19937 rng(seed);
    std::vector<uint8_t> w(wbytes);
    for (auto& v : w) v = (uint8_t) (rng() & 0xff);
    for (std::size_t k = 0; k < wbytes / IQ4XS_BYTES; ++k) {
        const uint16_t h = sane_half(rng);
        std::memcpy(&w[k * IQ4XS_BYTES], &h, 2);
    }
    std::vector<float> x((std::size_t) T * n_in);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : x) v = nd(rng);
    if (!ck(cudaMalloc(&b.w, wbytes + 16), "malloc w") || !ck(cudaMalloc(&b.xq, qall + 16), "malloc xq") ||
        !ck(cudaMalloc(&b.x, x.size() * 4), "malloc x") || !ck(cudaMalloc(&b.y_mma, ybytes), "malloc y") ||
        !ck(cudaMalloc(&b.y_old, ybytes), "malloc y") || !ck(cudaMalloc(&b.y_one, ybytes), "malloc y"))
        return false;
    if (!ck(cudaMemcpy(b.w + w_off, w.data(), wbytes, cudaMemcpyHostToDevice), "copy w") ||
        !ck(cudaMemcpy(b.x, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "copy x"))
        return false;
    // the outputs start as three different byte patterns: an output a path does not write cannot compare equal
    if (!ck(cudaMemset(b.y_mma, 0xff, ybytes), "memset") || !ck(cudaMemset(b.y_old, 0xfe, ybytes), "memset") ||
        !ck(cudaMemset(b.y_one, 0xfd, ybytes), "memset"))
        return false;
    basalt::kernels::quantize_q8_1_rows(b.x, T, n_in, b.xq + x_off, s);
    return ck(cudaStreamSynchronize(s), "quantize");
}

long long count_diff(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b, long long& nonfinite) {
    long long diff = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        float fa, fb;
        std::memcpy(&fa, &a[i], 4);
        std::memcpy(&fb, &b[i], 4);
        if (!std::isfinite(fa) || !std::isfinite(fb)) ++nonfinite;
        diff += a[i] != b[i];
    }
    return diff;
}

// One case: the tensor-core call against the dp4a multi-column call and T single-column calls.  False on a
// difference, a non-finite output, a call that did not take the tensor-core kernel, or an error.
bool compare_mma(const MmaCase& c, int T, int w_off, int x_off, cudaStream_t s) {
    namespace k = basalt::kernels;
    Buffers b;
    if (!fill(b, c.n_in, c.n_out, T, w_off, x_off, 777u + (unsigned) c.n_in * 3u + (unsigned) c.n_out + (unsigned) T,
              s))
        return false;
    const uint8_t* w = b.w + w_off;
    const uint8_t* xq = b.xq + x_off;
    const std::size_t qcol = k::native_q8_1_bytes(c.n_in, 1);
    long launches = 0;
    try {
        k::native_mmvq_set_mma(true);
        const long before = k::native_mmvq_mma_launches();
        k::native_mmvq(IQ4XS, w, xq, b.y_mma, c.n_in, c.n_out, T, s);
        launches = k::native_mmvq_mma_launches() - before;
        k::native_mmvq_set_mma(false);
        k::native_mmvq(IQ4XS, w, xq, b.y_old, c.n_in, c.n_out, T, s);
        for (int j = 0; j < T; ++j)
            k::native_mmvq(IQ4XS, w, xq + (std::size_t) j * qcol, b.y_one + (std::size_t) j * c.n_out, c.n_in,
                           c.n_out, 1, s);
    } catch (const std::exception& e) {
        std::printf("IQ4_XS %5d x %5d T=%d: %s\n", c.n_in, c.n_out, T, e.what());
        return false;
    }
    if (!ck(cudaStreamSynchronize(s), "mmvq")) return false;
    const std::size_t n = (std::size_t) T * c.n_out;
    std::vector<uint32_t> ym(n), yo(n), y1(n);
    if (!ck(cudaMemcpy(ym.data(), b.y_mma, n * 4, cudaMemcpyDeviceToHost), "read") ||
        !ck(cudaMemcpy(yo.data(), b.y_old, n * 4, cudaMemcpyDeviceToHost), "read") ||
        !ck(cudaMemcpy(y1.data(), b.y_one, n * 4, cudaMemcpyDeviceToHost), "read"))
        return false;
    long long nonfinite = 0;
    const long long d_old = count_diff(ym, yo, nonfinite);
    const long long d_one = count_diff(ym, y1, nonfinite);
    const bool ok = launches == 1 && d_old == 0 && d_one == 0 && nonfinite == 0;
    std::printf("IQ4_XS %5d x %5d T=%d%s: %7zu outputs, %lld differ from the dp4a multi-column call, %lld from the "
                "single-column calls, %lld non-finite%s  %s\n", c.n_in, c.n_out, T,
                w_off || x_off ? " (off-aligned)" : "", n, d_old, d_one, nonfinite,
                launches == 1 ? "" : ", the tensor-core kernel did not run", ok ? "ok" : "FAIL");
    return ok;
}

int mma_section(cudaStream_t s) {
    namespace k = basalt::kernels;
    std::printf("\n=== tensor-core IQ4_XS (STRATA_MMVQ_MMA) ===\n");
    if (!k::native_mmvq_mma_available()) {
        std::printf("not available on this device or build (sm_80+ CUDA): skipped\n");
        return 0;
    }
    const bool env = k::native_mmvq_mma();
    int bad = 0;
    for (const MmaCase& c : MMA_CASES) {
        for (int T = 2; T <= 8; ++T) bad += !compare_mma(c, T, 0, 0, s);
        for (int T : {3, 8}) bad += !compare_mma(c, T, 4, 4, s);   // the 4-byte copies
    }
    k::native_mmvq_set_mma(env);
    return bad;
}

void bench(cudaStream_t s) {
    namespace k = basalt::kernels;
    if (!k::native_mmvq_mma_available()) {
        std::printf("\n--bench: the tensor-core kernel is not available here\n");
        return;
    }
    const bool env = k::native_mmvq_mma();
    const MmaCase shapes[] = {{2560, 10240}, {6144, 2560}, {2560, 12288}};
    std::printf("\n=== --bench: IQ4_XS us per call, dp4a exact / tensor-core ===\n");
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (const MmaCase& c : shapes) {
        std::printf("%5d -> %5d:", c.n_in, c.n_out);
        for (int T = 2; T <= 8; ++T) {
            Buffers b;
            if (!fill(b, c.n_in, c.n_out, T, 0, 0, 99u, s)) break;
            float us[2] = {0, 0};
            for (int mma = 0; mma < 2; ++mma) {
                k::native_mmvq_set_mma(mma == 1);
                constexpr int N = 200;
                for (int r = 0; r < 20; ++r) k::native_mmvq(IQ4XS, b.w, b.xq, b.y_old, c.n_in, c.n_out, T, s);
                cudaEventRecord(e0, s);
                for (int r = 0; r < N; ++r) k::native_mmvq(IQ4XS, b.w, b.xq, b.y_old, c.n_in, c.n_out, T, s);
                cudaEventRecord(e1, s);
                cudaEventSynchronize(e1);
                float ms = 0;
                cudaEventElapsedTime(&ms, e0, e1);
                us[mma] = 1000.0f * ms / N;
            }
            std::printf("  T%d %.1f/%.1f", T, us[0], us[1]);
        }
        std::printf("\n");
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    k::native_mmvq_set_mma(env);
}

}  // namespace

int main(int argc, char** argv) {
    bool do_bench = false;
    for (int i = 1; i < argc; ++i) do_bench |= std::string(argv[i]) == "--bench";
    cudaStream_t s;
    if (!ck(cudaStreamCreate(&s), "stream create")) return 1;

    Totals on, off;
    const int bad = sweep(true, s, on);
    const int bad_off = sweep(false, s, off);
    basalt::kernels::native_mmvq_set_multi_exact(true);   // back to the default
    const int bad_mma = mma_section(s);
    if (do_bench) bench(s);

    std::printf("\nmulti_exact on:   %lld outputs compared, %lld differ bitwise, %lld non-finite\n",
                on.checked, on.diff, on.nonfinite);
    std::printf("multi_exact off:  %lld outputs compared, %lld differ bitwise (%lld on finite outputs, %lld at T > %d)\n",
                off.checked, off.diff, off.diff_finite, off.diff_finite_wide, COINCIDE_MAX_T);

    if (on.nonfinite != 0 || off.nonfinite != 0) {
        std::printf("\nmmvq_multi_parity: FAILED. %lld non-finite outputs from finite scales and activations: the "
                    "bit comparison is only meaningful on finite outputs.\n", on.nonfinite + off.nonfinite);
        return 1;
    }
    if (bad_mma != 0) {
        std::printf("\nmmvq_multi_parity: FAILED. %d tensor-core IQ4_XS case(s) differ from the dp4a kernels, have "
                    "non-finite outputs, did not take the tensor-core kernel or did not run.\n", bad_mma);
        return 1;
    }
    if (bad != 0 || bad_off != 0) {
        std::printf("\nmmvq_multi_parity: FAILED. %d case(s) with multi_exact on differ from single-column calls "
                    "or did not run, %d case(s) did not run with it off. If they differ, the sentence in "
                    "native_mmvq.hpp does not hold, and neither does the bit-exactness verify.hpp claims.\n",
                    bad, bad_off);
        return 1;
    }
    // THE NEGATIVE CONTROL, asserted where it can see something: past T = 4 the generic layout is a different one,
    // so if it still finds no difference the comparison cannot tell the layouts apart and the pass proves nothing.
    if (off.powerless_cases != 0) {
        std::printf("\nmmvq_multi_parity: FAILED (no power). In %d case(s) no output differs with multi_exact off at "
                    "any T > %d, where the generic layout is not the exact one: for those cases this comparison "
                    "cannot tell the two layouts apart, so their pass above proves nothing.\n", off.powerless_cases,
                    COINCIDE_MAX_T);
        return 1;
    }
    std::printf("\nmmvq_multi_parity: ok. With multi_exact on every column is bitwise equal to a single-column "
                "call; with it off the control finds differences at T > %d in every case, so the comparison has "
                "power.\n",
                COINCIDE_MAX_T);
    return 0;
}
