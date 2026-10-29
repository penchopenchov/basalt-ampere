# Ampere precision

Experimental solo-window staging is off by default. It can copy pinned RAM
experts into GPU staging for solo multi-row windows, including packs with different expert
sizes by layer. It uses the existing staged-copy and grouped-expert kernels;
some work previously computed by the CPU now uses GPU Q8_1 activations per 32
values and GPU reductions. CPU i-quant gate/up uses Q8_K activations per 256.
Selection depends on the window's misses; single-row windows retain the CPU
path. Greedy identity across window widths is unvalidated.
Numerical differences are possible. Q8_0 PPL/KLD is unvalidated for this path;
kernel parity and greedy checks do not establish model quality equivalence.

Ampere selects existing native quantized projection and per-stage GDN recurrence
kernels when dense TMA or cluster specializations are unavailable. These are
alternative existing reduction paths; their output is not assumed bit-identical
to the Blackwell fused path.

Native W4A4 NVFP4 prefill is disabled on Ampere. NVFP4 weights instead use the
existing MMQ implementation with its int8 activation contract. This changes
activation rounding relative to upstream's native FP4 prefill. Both buffer sizing
and dispatch use the same hardware capability decision.

Dependent launches become ordinary ordered stream launches. Prefetch hints use
instructions available to the compiled target; these scheduling changes do not
intentionally change arithmetic. K/IQ weight formats and stored model data are
unchanged.

Measured IQ3_S and IQ4_XS multi-vector shapes select the existing int8 Tensor
Core implementation at larger window sizes. Weight decoding and q8_1 activation
quantization are unchanged; the accumulation and reduction path differs from
the dp4a implementation. The format parity benchmarks exercise both paths;
the dispatch test also checks launch counts and sentinel outputs against an
independent zero-weight reference. This does not establish model-level greedy
identity or quality equivalence.

The fused prefill hyperconnection up/mix keeps its existing arithmetic. On the
RTX 3090 with CUDA 13.3, cuBLAS's two-step reference changes bits when the same
input prefix is multiplied at width 129 rather than 1000. At width 129 the two
paths' normalized maximum errors against CPU double are 1.1e-7 to 2.1e-7 for
BF16 and dequantized 8-bit weights. The test checks all FP32 values for finite
absolute/relative agreement, exact BF16/BF16-low/FP16 encoding of each path's
own FP32 result, and CPU-reference tokens at the tile boundary. Other tested
widths retain bitwise comparison; the fused path stays bitwise independent of
chunk width and position. This is a reference reduction difference, not a new
engine arithmetic implementation.

Validation results belong in [Ampere setup](../ampere.md). Synthetic parity and
window-width checks do not replace a decode-path perplexity/KLD comparison with
a Q8_0 reference. That reference is not included in the requested IQ3_S pack.
