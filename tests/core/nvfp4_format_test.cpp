// NVFP4 block format: the scalar reference decoder (dequant.hpp) and ggml's row dequantizer (ggml_quants.c)
// against an independent decode of the E2M1 and UE4M3 bit fields in double precision.
//
//     nvfp4_format_test [experts.bin]
//
// 1. Every UE4M3 scale byte and every E2M1 code, exhaustively, against the bit-field formulas.
// 2. Random blocks (every byte random, scales included) at the experts' geometry: both decoders bit-equal to each
//    other and to the double-precision spec decode, and a dot product in float against one in double.
// 3. The round-to-nearest test quantizer: decode(quantize(x)) within the format's error on Gaussian rows.
// 4. With an experts.bin (48 layers x 512 experts of [gate | up | down | {s_gate, s_up, s_down, 0}]): a few real
//    experts decode to finite values, no scale byte is NaN, and the tail's scales are finite and positive.
#include "basalt/artifact/dequant.hpp"
#include "basalt/ggml/ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <vector>

namespace {

constexpr int N_EMBD = 2560, N_FF = 640, N_EXPERT = 512, N_LAYER = 48, BLK = 36;
constexpr size_t GU_ROW = (size_t)N_EMBD / 64 * BLK, D_ROW = (size_t)N_FF / 64 * BLK;
constexpr size_t BLOB = 2 * (size_t)N_FF * GU_ROW + (size_t)N_EMBD * D_ROW + 16;

int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

// E4M3 without sign, bias 7: subnormal m/8 * 2^-6, normal (1 + m/8) * 2^(e-7); 0x7F is NaN (decoded as 0, as
// ggml does). Bit 7 is ignored.
double spec_ue4m3(uint8_t x) {
    const int e = (x >> 3) & 0xF, m = x & 7;
    if (x == 0x7F) return 0.0;
    return e == 0 ? m / 8.0 * std::ldexp(1.0, -6) : (1.0 + m / 8.0) * std::ldexp(1.0, e - 7);
}

// E2M1, bias 1: sign bit 3, exponent bits 2-1, mantissa bit 0; subnormal m * 0.5, normal (1 + m/2) * 2^(e-1).
double spec_e2m1(int c) {
    const int s = (c >> 3) & 1, e = (c >> 1) & 3, m = c & 1;
    const double v = e == 0 ? m * 0.5 : (1.0 + m / 2.0) * std::ldexp(1.0, e - 1);
    return s ? -v : v;
}

void spec_block(const uint8_t* b, double* out) {
    for (int i = 0; i < 64; ++i) {
        const int s = i / 16, k = i % 16;
        const uint8_t byte = b[4 + 8 * s + (k & 7)];
        out[i] = spec_e2m1(k < 8 ? byte & 0xF : byte >> 4) * spec_ue4m3(b[s]);
    }
}

void exhaustive() {
    int bad = 0;
    for (int x = 0; x < 256; ++x)
        if ((double)basalt::ue4m3_to_fp32_half((uint8_t)x) * 2.0 != spec_ue4m3((uint8_t)x)) ++bad;
    check(bad == 0, "UE4M3 decode differs from the bit-field formula");
    bad = 0;
    for (int c = 0; c < 16; ++c)
        if (basalt::kvalues_fp4[c] != spec_e2m1(c) * 2.0) ++bad;
    check(bad == 0, "kvalues_fp4 differs from E2M1 doubled");
    std::printf("exhaustive: 256 scale bytes, 16 codes ok\n");
}

void random_blocks(std::mt19937& rng) {
    const ggml_type_traits* t = ggml_get_type_traits(GGML_TYPE_NVFP4);
    check(t != nullptr && t->blck_size == 64 && t->type_size == BLK, "type traits of NVFP4");
    if (t == nullptr) return;
    std::uniform_int_distribution<int> byte(0, 255);
    std::normal_distribution<float> nd(0.f, 1.f);
    const int rows = 64;
    std::vector<uint8_t> w((size_t)rows * GU_ROW);
    for (uint8_t& b : w) b = (uint8_t)byte(rng);
    std::vector<float> a(N_EMBD), ref(N_EMBD), x(N_EMBD);
    std::vector<double> spec(N_EMBD);
    for (float& v : x) v = nd(rng);
    size_t mism_ggml = 0, mism_spec = 0;
    double worst_rel = 0;
    for (int r = 0; r < rows; ++r) {
        const uint8_t* row = w.data() + (size_t)r * GU_ROW;
        t->to_float(row, ref.data(), N_EMBD);
        for (int b = 0; b < N_EMBD / 64; ++b) {
            basalt::dequantize_nvfp4(row + (size_t)b * BLK, a.data() + 64 * b);
            spec_block(row + (size_t)b * BLK, spec.data() + 64 * b);
        }
        double dd = 0, mag = 0;
        float df = 0;
        for (int i = 0; i < N_EMBD; ++i) {
            mism_ggml += a[i] != ref[i];
            mism_spec += (double)a[i] != spec[i];
            dd += spec[i] * (double)x[i];
            mag += std::fabs(spec[i] * (double)x[i]);
            df += a[i] * x[i];
        }
        worst_rel = std::fmax(worst_rel, std::fabs((double)df - dd) / (mag > 0 ? mag : 1));
    }
    check(mism_ggml == 0, "dequant.hpp differs from ggml's dequantize_row_nvfp4");
    check(mism_spec == 0, "dequant.hpp differs from the spec decode");
    check(worst_rel < 1e-5, "float dot drifts from the double dot");
    std::printf("random: %d rows x %d: ggml mismatches %zu, spec mismatches %zu, float dot vs double max %.2e of sum|wx|\n",
                rows, N_EMBD, mism_ggml, mism_spec, worst_rel);
}

void roundtrip(std::mt19937& rng) {
    const ggml_type_traits* t = ggml_get_type_traits(GGML_TYPE_NVFP4);
    if (t == nullptr || t->from_float_ref == nullptr) return;
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(N_EMBD), y(N_EMBD);
    std::vector<uint8_t> q(GU_ROW);
    double err = 0, energy = 0;
    for (int r = 0; r < 64; ++r) {
        for (float& v : x) v = nd(rng) * 0.02f;
        t->from_float_ref(x.data(), q.data(), N_EMBD);
        t->to_float(q.data(), y.data(), N_EMBD);
        for (int i = 0; i < N_EMBD; ++i) {
            err += (double)(y[i] - x[i]) * (y[i] - x[i]);
            energy += (double)x[i] * x[i];
        }
    }
    const double rel = std::sqrt(err / energy);
    check(rel < 0.15, "round-to-nearest NVFP4 error above the format's");
    std::printf("roundtrip: Gaussian rows, relative RMS error %.4f\n", rel);
}

int real_pack(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", path);
        return 1;
    }
    f.seekg(0, std::ios::end);
    const size_t size = (size_t)f.tellg();
    check(size == BLOB * N_EXPERT * N_LAYER, "experts.bin size is 48 x 512 NVFP4 blobs with scale tails");
    const int picks[][2] = {{0, 0}, {0, 511}, {12, 100}, {24, 300}, {47, 5}};
    std::vector<uint8_t> blob(BLOB);
    std::vector<float> v(64);
    for (const auto& p : picks) {
        f.seekg((std::streamoff)(((size_t)p[0] * N_EXPERT + p[1]) * BLOB));
        f.read((char*)blob.data(), (std::streamsize)BLOB);
        check((bool)f, "read a blob");
        size_t nan_scales = 0, nonfinite = 0;
        double amax = 0;
        for (size_t b = 0; b + 16 < BLOB; b += BLK) {
            for (int s = 0; s < 4; ++s) nan_scales += (blob[b + s] & 0x7F) == 0x7F;
            basalt::dequantize_nvfp4(blob.data() + b, v.data());
            for (float e : v) {
                nonfinite += !std::isfinite(e);
                amax = std::fmax(amax, std::fabs(e));
            }
        }
        float tail[4];
        std::memcpy(tail, blob.data() + BLOB - 16, 16);
        bool tail_ok = tail[3] == 0.f;
        for (int i = 0; i < 3; ++i) tail_ok = tail_ok && std::isfinite(tail[i]) && tail[i] > 0.f;
        check(nan_scales == 0 && nonfinite == 0, "real blocks decode finite with no NaN scale");
        check(tail_ok, "real scale tail {s_gate, s_up, s_down, 0}");
        std::printf("layer %2d expert %3d: block max |v| %.1f, s_gate %.3e s_up %.3e s_down %.3e\n", p[0], p[1], amax,
                    tail[0], tail[1], tail[2]);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::mt19937 rng(40);
    exhaustive();
    random_blocks(rng);
    roundtrip(rng);
    if (argc > 1 && real_pack(argv[1]) != 0) return 1;
    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("nvfp4_format_test: ok\n");
    return 0;
}
