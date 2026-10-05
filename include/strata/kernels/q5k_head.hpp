// include/strata/kernels/q5k_head.hpp - the Q5_K x q8_1 GEMV of a wide head (the output projection, the draft head)
// for 1..8 columns, bitwise equal to native_mmvq (type 13, the exact layout): one warp per row holds what four warps
// did, and a block reads the columns' activations once into shared memory for all its rows instead of once per row.
#pragma once

namespace strata::kernels {

bool q5k_head_supported(int n_in, int ncols);
/// y[j * n_out + r] = row r of `weights` (Q5_K, n_in / 256 blocks per row) . column j of `x_q8_1`
void q5k_head_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream);

}  // namespace strata::kernels
