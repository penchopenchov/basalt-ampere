// include/strata/kernels/dec_gdn.hpp - the GDN block of a verify window in three launches (STRATA_DEC_GDN=1).
//
//   dec_gdn_inproj   q8_1 of the block input (per block, in shared memory) + the qkv and z projections
//   dec_gdn_rec      conv1d + SiLU + q/k L2 norm, the alpha/beta gates, the delta-rule recurrence over the
//                    window from the untouched state, the gated RMS norm, and the q8_1 of its output
//   dec_dense        the out projection from those q8_1 columns
//
// Every value is the one the per-stage kernels produce (native_quantize_q8_1 + native_mmvq in exact mode,
// gdn_conv_l2_multi, gdn_ab_multi, gdn_step_norm_multi, native_quantize_q8_1): the same operations on the same
// values in the same order, so a token's outputs do not depend on T or its column.  The dense dots stage each
// weight row in shared memory with one bulk copy and give a row to one warp, whose lanes replay the reference
// layout's per-thread chains and cross-warp sum before the same butterfly.
//
// Supported projection types (GGML ids): Q4_K 12, Q5_K 13, Q6_K 14, IQ3_S 21, IQ4_XS 23.  sm_90+ only.
#pragma once

#include <cstdint>

namespace strata::kernels {

constexpr int kDecMaxT = 8;

struct DecMat {
    int type = -1;
    const void* w = nullptr;
    int n_out = 0;
    float* y = nullptr;   // column t at y + t * n_out
};

/// The dense types dec_gdn_inproj / dec_dense take.
bool dec_dense_supported(int ggml_type, int n_in) noexcept;

/// q8_1 of x (n_tok columns of n_in floats, column t at x + t * n_in) and every matrix of `m` against it.
void dec_gdn_inproj(const float* x, int n_in, int n_tok, const DecMat* m, int n_mats, void* stream);

/// One matrix against n_tok q8_1 columns (column t at block t * n_in / 32 of xq).
void dec_dense(const void* xq, int n_in, int n_tok, const DecMat& m, void* stream);

struct DecRecArgs {
    const float* state = nullptr;    // [S rows][h_v][S cols] fp32, then the conv history [C][3]
    const float* qkv = nullptr;      // raw projections, token t at qkv + t * C
    const float* conv_w = nullptr;   // [C][4]
    const float* x = nullptr;        // the block input (fp32), token t at x + t * n_embd
    const uint16_t* w_alpha = nullptr;   // bf16 [h_v][n_embd]
    const uint16_t* w_beta = nullptr;
    const float* dt = nullptr;
    const float* ssm_a = nullptr;
    const float* z = nullptr;        // token t at z + t * h_v * S
    const float* gamma = nullptr;    // [S]
    float* h = nullptr;              // conv output (q/k normalized), token t at h + t * C
    float* gate = nullptr;           // token t at gate + t * h_v
    float* beta = nullptr;
    float* y = nullptr;              // optional fp32 output, token t at y + t * h_v * S
    void* yq = nullptr;              // q8_1 of the output, token t at block t * h_v * S / 32
    int C = 0, n_embd = 0, h_k = 0, h_v = 0;
    float eps = 0.0f;
    unsigned long long* stamps = nullptr;   // bench: 10 per-phase timestamps per block
};

/// The recurrence of tokens [0, n_tok) from the untouched state; one cluster of four blocks per value head.
void dec_gdn_rec(const DecRecArgs& a, int n_tok, void* stream);

/// Whether STRATA_DEC_GDN=1 asks for the path above.
bool dec_gdn_enabled();

}  // namespace strata::kernels
