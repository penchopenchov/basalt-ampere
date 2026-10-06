// include/strata/kernels/dense_tc.hpp - dense projections of a verify window on int8 tensor cores (STRATA_DEC_DTC).
//
// y[c][r] = sum over the row's 32-value blocks of the format's float term on the block's exact integer dot with
// column c's q8_1 codes (mma.sync, 16 rows x 8 columns per tile).  Terms per block:
//   Q4_K / Q5_K   d8 * (d * sc * dot - dmin * m * sum)      (sum: the block's q8_1 code sum)
//   Q6_K          d * d8 * (sc0 * dot0 + sc1 * dot1)        (the two 16-value halves)
//   IQ4_XS        d * d8 * (ls - 32) * dot
// The terms are the native kernels' math with fewer roundings, not their bits.  Every column is summed in one order
// fixed by the shape (n_in, n_out) alone, so a column's value does not depend on how many columns ride along or where
// it sits.
//
// Weights are the GGUF row-major blocks; x_q8_1 holds ncols (1..8) columns of n_in / 32 q8_1 blocks.  n_in must be a
// multiple of 512, x_q8_1 16-byte aligned.  sm_90+ CUDA only.
#pragma once

namespace strata::kernels {

bool dense_tc_supported(int ggml_type, int n_in) noexcept;

void dense_tc(int ggml_type, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
              void* stream);

/// STRATA_DEC_DTC: which verify projections take dense_tc for every window width (bits): 1 GDN qkv, 2 GDN z, 4 GDN
/// out, 8 QSA q, 16 QSA out, 32 QSA k/v.  0 (default) none.
int dense_tc_mask();

}  // namespace strata::kernels
