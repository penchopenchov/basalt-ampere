// include/strata/kernels/dec_shx.hpp - the verify window's shared expert in one launch (STRATA_DEC_SHX=1).
//
// gate/up (IQ4_XS), SwiGLU + q8_1, down (IQ4_NL), the BF16 scalar gate and its sigmoid scale: shared_expert_multi's
// seven launches as one.  Every row is native_mmvq's exact layout played by one warp, the SwiGLU blocks are
// native_swiglu_quantize_q8_1's, the scalar gate is bf16_gemv_fp32_mmvf's 256-thread sum, so each token's output is
// bitwise shared_expert_multi's at every T.  A warp per gate/up row or scalar gate, then two down rows; the blocks
// are co-resident: a SwiGLU chunk's last row quantizes it, the down rows start once every chunk and gate is in.
#pragma once

#include <cstdint>

namespace strata::kernels {

struct ShxArgs {
    const void* gate_w = nullptr;     ///< IQ4_XS n_ff x n_embd
    const void* up_w = nullptr;       ///< IQ4_XS n_ff x n_embd
    const void* down_w = nullptr;     ///< IQ4_NL n_embd x n_ff
    const uint16_t* gate_inp = nullptr;   ///< BF16 n_embd: the scalar gate
    const float* x = nullptr;         ///< n_tok x n_embd FFN input
    const void* xq = nullptr;         ///< its q8_1 rows: native_quantize_q8_1's bits
    float* gate = nullptr;            ///< scratch n_tok x n_ff
    float* up = nullptr;              ///< scratch n_tok x n_ff
    void* hq = nullptr;               ///< scratch n_tok x n_ff / 32 q8_1 blocks
    float* g = nullptr;               ///< n_tok gate logits
    float* out = nullptr;             ///< n_tok x n_embd: sigmoid(g) * down
    unsigned int* sync = nullptr;     ///< kShxSyncWords zeroed device words, left zeroed
    int n_tok = 0;
};
constexpr int kShxSyncWords = 32;

/// Whether the one-launch path covers these formats and shapes.
bool dec_shx_ok(int gate_type, int up_type, int down_type, int64_t n_embd, int64_t n_ff);
void dec_shx(const ShxArgs& a, void* stream);

}  // namespace strata::kernels
