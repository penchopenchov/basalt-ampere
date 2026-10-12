// include/basalt/kernels/qsa_full.hpp - a QSA layer's attention half as ONE persistent kernel.
//
// K/V/indexer/query projections -> K/Q post (norms, RoPE, cache appends) -> indexer scores -> top-k -> attention chunks
// -> merge + gate + q8_1 -> output projection, the same operations (and bits) as dense_tc, native_mmvq,
// bf16_gemv_fp32_mmvf_multi, qsa_dec_kv_post, qsa_dec_q_post, qsa_block_scores, qsa_block_topk, qsa_dec_attn and
// dense_tc in sequence.  One CTA per SM; phases are joined by bounded grid barriers, so every CTA must be resident.
#pragma once

#include "basalt/kernels/qsa_dec_attn.hpp"
#include "basalt/kernels/qsa_dec_fused.hpp"
#include "basalt/kernels/rope_scaling.hpp"

namespace basalt::kernels {

constexpr int kQsaFullMaxT = 8;
constexpr int kQsaFullMaxLaunches = 16;   ///< ordinals per recorded window

struct QsaFullArgs {
    int n = 0;                                ///< window rows, 1..kQsaFullMaxT
    int tq = 0, tk = 0, tv = 0, to = 0;       ///< native types of attn_q, attn_k, attn_v, attn_output
    const void *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr;
    const uint16_t *wik = nullptr, *wiq = nullptr;   ///< indexer k / q projections (BF16)
    const void* xmix = nullptr;               ///< q8_1 rows of the mixed input; also receives the attention rows
    const float* mixed = nullptr;             ///< [n][2560]
    float *qfull = nullptr, *kcur = nullptr, *vcur = nullptr, *idx_raw = nullptr, *qidx = nullptr, *bo = nullptr;
    QsaDecKvPost kv;                          ///< as for qsa_dec_kv_post (part unset)
    QsaDecQPost qp;                           ///< as for qsa_dec_q_post
    QsaShapes shapes;
    const float *pooled = nullptr, *dead = nullptr;
    const int32_t* steps = nullptr;
    int64_t max_blocks = 0, cap = 0;
    float* scores = nullptr;
    int32_t* ids = nullptr;
    QsaAttnPools pools;
    float *scratch = nullptr, *attn = nullptr, *gated = nullptr;
    int ord = 0;                              ///< launch ordinal within the graph (0 = the first: it zeroes the barrier counters)
};

/// Allocate the barrier counters and set the kernel's shared-memory size on the current device (once, outside a capture).
void qsa_full_init();
/// The current device and these arguments run the kernel (device, rows, projection types, geometry, alignment).
bool qsa_full_supported(const QsaFullArgs& a);
/// Enqueue the layer's attention half on .  Aborts on invalid arguments.
void qsa_full(const QsaFullArgs& a, void* stream);

}  // namespace basalt::kernels
