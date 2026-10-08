// src/vision/gemm_cpu.hpp - CPU matrix products for the encoder.
#pragma once

#include <cstdint>

namespace basalt::vision {

/// Y[M][N] = X[M][K] . W[N][K]^T, W bf16 (bits).  `round_x`: X rounded to bf16 first.  fp32 accumulation.
void gemm_bf16w(const float* X, const uint16_t* W, float* Y, int M, int N, int K, int threads, bool round_x);
/// The same with fp32 weights.
void gemm_f32w(const float* X, const float* W, float* Y, int M, int N, int K, int threads);

}  // namespace basalt::vision
