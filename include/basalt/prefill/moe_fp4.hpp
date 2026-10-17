// include/basalt/prefill/moe_fp4.hpp - the prompt path's NVFP4 experts on the FP4 block-scaled tensor cores.
// Weights stay in the pack's NVFP4 blocks (read in place from each expert's blob); activations are rounded to FP4
// codes with a UE4M3 scale per 16 values and one FP32 scale per token (W4A4).  Activation planes are laid out per
// 64-token tile of one expert (each expert's rows padded to whole tiles), so a tile's codes and scales are one
// contiguous, bank-swizzled run.
#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt::prefill::fp4 {

constexpr int kTileTokens = 64;
constexpr int kMaxExperts = 16;   // experts per launch

/// Bytes of the activation planes for `rows` rows of `cols` values (multiple of 128) split over up to `experts`
/// segments, each padded to whole tiles.
size_t a4_bytes(int64_t rows, int64_t cols, int experts);
/// Floats of the per-token scale array for the same split.
size_t scale_count(int64_t rows, int experts);

/// tile_off[0..n]: the first tile of each of the n segments of `bounds` (n + 1 row bounds on the device) and the
/// total.  One tiny launch.
void plan(const int32_t* bounds, int n, int32_t* tile_off, void* stream);

/// out[i] = bounds[i] - bounds[0] for i in 0..n: a run of segments' own row bounds.
void rebase(const int32_t* bounds, int n, int32_t* out, void* stream);

/// Rounds rows to FP4: row i of the segments is row ids[i] of x (or i), `ld` floats per row.  `bounds`/`tile_off` as
/// planned; writes the planes `a4` and the token scales `gs`.
void quantize(const float* x, const int32_t* ids, const int32_t* bounds, const int32_t* tile_off, int n, void* a4,
              float* gs, int64_t cols, int64_t ld, int64_t rows, void* stream);

/// n experts' products: expert j's matrix [w_rows, w_cols] of NVFP4 blocks starts at blob[j] + w_off; its rows
/// [bounds[j], bounds[j+1]) of the planes go to dst rows of the same indices (dst + row * ld_dst).  The result is the
/// plain product (no per-expert tail scale).  `max_rows`: the most rows one segment has.
struct Product {
    const uint8_t* blob[kMaxExperts] = {};
    size_t w_off = 0;
    int64_t w_rows = 0, w_cols = 0;
    int n = 0;
    const void* a4 = nullptr;
    const float* gs = nullptr;
    const int32_t* bounds = nullptr;
    const int32_t* tile_off = nullptr;
    float* dst = nullptr;
    int64_t ld_dst = 0, max_rows = 0;
};
void run(const Product& p, void* stream);
/// The 16-byte scale tail at `tail_off` of each of p's n blobs to dst[4 j .. 4 j + 3] (the tail swiglu reads).
void gather_tails(const Product& p, size_t tail_off, float* dst, void* stream);

}  // namespace basalt::prefill::fp4
