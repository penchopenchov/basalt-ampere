// include/basalt/kernels/q5k_head.hpp - the Q5_K x q8_1 GEMV of a wide head (the output projection, the draft head)
// for 1..8 columns, bitwise equal to native_mmvq (type 13, the exact layout): one warp per row holds what four warps
// did, and a block reads the columns' activations once into shared memory for all its rows instead of once per row.
#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt::kernels {

bool q5k_head_supported(int n_in, int ncols);
/// y[j * n_out + r] = row r of `weights` (Q5_K, n_in / 256 blocks per row) . column j of `x_q8_1`
void q5k_head_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream);

/// Rows of a draft head whose `map` entry is negative hold no token: q5k_head_top1 skips them, and this sets their
/// logits (one column, n entries) to -inf after a q5k_head_mmvq.
void q5k_head_mask_unmapped(float* y, const int32_t* map, int n, void* stream);

/// One column (n_in 2560): the logits as above, and in the same launch the first row of the largest logit (through
/// `map` when non-null; rows with a negative id are skipped) and its softmax probability over the n_out rows.  The probability's sum runs in a different
/// order than row_top_prob's (last bits differ).  `scratch`: q5k_head_top1_scratch_bytes(), zeroed once before first
/// use.  False: the shape is not supported.
size_t q5k_head_top1_scratch_bytes();
bool q5k_head_top1(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, const int32_t* map,
                   int* out_id, float* out_prob, void* scratch, void* stream);

}  // namespace basalt::kernels
