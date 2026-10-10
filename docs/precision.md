# Precision and quality

This page shows what precision Basalt computes in, that speculative decoding (MTP) does not change its output, and how
its quality compares with llama.cpp on the same GGUFs against the same Q8_0 reference.

Test setup: Basalt at commit 610f2cd, one RTX 5090 at a 400 W power limit, an RTX 5060 Ti as the peer expert
tier, 32 GB of system RAM.

## Summary

- **MTP does not change output.** With the expert tiers pinned, 275 of 275 greedy prompts produced identical tokens on
  both packs tested, and every full-vocabulary fp32 logit row (62,271 and 60,878) was bit-identical between `--spec 0`
  and `--spec 7`.
- **KLD matches llama.cpp.** On the same GGUFs against the same base, Basalt and llama.cpp's own `llama-perplexity`
  agree: IQ3_S 0.15264 vs 0.1526, IQ3_XXS 0.2433 vs 0.2428 at 100 chunks.
- **No requantization.** Weights are the GGUF blocks as shipped. The one exception is the hyper-connection matrices,
  which decode reads from an int8 copy with one fp16 scale per 32 inputs.
- **fp32 where it matters.** Activations are fp32 in the residual stream, norms, router, softmax and logits, and int8
  (q8_1) into the quantized matmuls. GDN state is fp32. KV is fp16 by default, with int8, q4_0 and k8v4 options.
- **Quality is scored through decode.** Prefill is a different computation (fp16/BF16 GEMMs); the quality gate scores
  the decode path.
- **The reference has its own noise.** The reference is a llama.cpp Q8_0 base, which carries about 0.01 to 0.02 KLD of
  its own noise ([see below](#why-q8_0-against-q8_0-is-not-zero)).

## Decode path precision

Sources are `file:line` under `src/`, `include/basalt/` and `tools/convert/`. Model geometry: n_embd 2560, 4
hyper-connection streams, 48 layers (36 GDN, 12 QSA), 512 experts, top-10, head_dim 256.

### Weights

| Tensor class | Format | Source |
|---|---|---|
| Routed experts | GGUF blocks copied byte for byte, no requantization. Gate and up share one type per layer; types are mixed across layers (IQ3_XXS pack: gate/up IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S; down Q2_0 on 30 layers, IQ4_NL on 18. IQ3_S pack: gate/up mostly IQ3_XXS, IQ3_S, IQ2_S, one IQ4_XS; down IQ4_NL on 39, Q2_0 on 9) | README.md (The `.basalt` model file); iq_kernels.cu:541-542; tests/data/native_experts/ |
| Shared expert gate/up/down | GGUF blocks, any native_mmvq type | native_mmvq.hpp:55-58 |
| Shared-expert scalar gate | BF16 | model_sections.py:48-51 |
| QSA q/k/v/o, GDN qkv, z (attn_gate), ssm_out | GGUF blocks (dense_tc takes Q4_K, Q5_K, Q6_K, IQ4_XS, IQ3_S, Q3_K, Q2_0; native_mmvq otherwise) | dense_tc.cu:1063-1075; dense_tc.hpp:2-12 |
| GDN ssm_alpha, ssm_beta; QSA indexer q/k projections | BF16 | model_sections.py:52-53 |
| Router | BF16, not quantized | model_sections.py:48 |
| Norm weights, ssm_a, ssm_dt, conv1d, ssm_norm | F32 | model_sections.py:55-60 |
| Hyper-connection (hc) matrices (down, up, inject; per layer and for the output head) | Stored BF16 in the pack; decode reads an int8 copy: int8 codes with one fp16 scale per 32 consecutive inputs, scale = f16(absmax/127), codes clamped to ±127 | hc_fused.cu:343, 1168-1186, 1210-1227; verify.cpp:561-572 |
| Token embedding | GGUF type kept in mapped host memory, one row dequantized to fp32 per token | native_head.hpp:42-45; verify.cpp:948-961 |
| Output head | A native_mmvq type with dedicated Q5_K and Q6_K kernels; the IQ3 packs use a Q5_K head | verify.cpp:1877-1882 |
| MTP drafter dense | Q8_0 for fc_embedding, fc_hidden, q/k/v/o, indexer projection and the shared-expert projections; BF16 for the rest; 1-D tensors F32 | model_sections.py:334-338, 444-470 |
| MTP drafter experts | Q2_0, all in VRAM | model_sections.py:482-487 |
| MTP draft head | Rows gathered from the main head (same type), optionally a vocabulary subset; the leading rows are first scored from a 2-bit copy and the near-best rows rescored exactly, so a draft differs from the exact argmax only in rare cases, and the draft probability is approximate | mtp.cpp, q5k_head.cu |
| MTP hc matrices | The same int8 copies | mtp.cpp:218-263 |

The drafter only proposes tokens; verify decides every token, so the drafter's precision affects speed, not output.

### Activations at each matmul

| Matmul | Activation format | Source |
|---|---|---|
| hc down and inject (hc read) | fp32, no activation quantization; int8 weights converted exactly, fp32 fmaf per group | hc_fused.cu:343-370, 489 |
| hc up | fp32 input split into an fp16 high and low part (about 22-bit mantissa), fp16 tensor-core mma with fp32 accumulate, per-group fp16 scale applied in fp32 | hc_fused.cu:632-695 |
| Router | fp32 times BF16 weight, fp32 FMA | hc_fused.cu:737-780; native_bf16.cu:34-125 |
| GDN qkv, QSA q, GDN ssm_out, QSA o | q8_1: int8 codes with an fp16 scale per 32 values, int8 tensor-core dot (dense_tc) | native_mmvq.cu:147-162; dense_tc.hpp:2-12 |
| GDN z, QSA k/v | z through dense_tc together with qkv when eligible; otherwise native_mmvq (dp4a on q8_1) | dense_tc.cu:1155; verify.cpp:1145-1170 |
| Routed experts gate/up | q8_1 of the fp32 FFN input; dp4a with exact int32 per block | hc_fused.cu:63-77; iq_kernels.cu:202-228 |
| Routed experts down | fp32 SwiGLU, then q8_1 per 32 | iq_kernels.cu:1909-1920 |
| Shared expert | q8_1 into gate/up; fp32 SwiGLU, q8_1 into down; sigmoid gate fp32 | shared_expert.cu:178-215 |
| GDN alpha/beta, indexer projections | fp32 activation times BF16 weight, fp32 FMA | dec_gdn_rec.cu:88-112; verify.cpp:1315-1327 |
| Output head | q8_1 of the fp32 final stream, mmvq or the Q5_K/Q6_K kernels | verify.cpp:1876-1885 |
| CPU expert tier (experts not resident on a GPU) | The fp32 input quantized with ggml's vec_dot_type (Q8_K per 256 for i-quant gate/up; Q8_0 or Q8_K for down), so it rounds differently from the GPU tiers | native_expert.cpp:52-62, 84-139 |

Unlike llama.cpp, Basalt does not quantize the activations of the hc read to Q8_0.

### Accumulation, state, softmax and logits

| Item | Precision | Source |
|---|---|---|
| int8 dot products | Exact int32 per 32-value block, then fp32 scale terms summed in fp32 | dense_tc.cu:60-68, 1007-1010 |
| Residual stream and hc buffers | fp32 | verify.hpp:468-469; hc_fused.hpp:22-38 |
| RMSNorm | fp32, eps 1e-6 | verify.cpp:116 |
| fp16/BF16 accumulators | None; hc up uses fp16 mma inputs with fp32 accumulate | hc_fused.cu:632-695 |
| KV cache (`--kv`) | fp16, int8, q4_0 (alias q4), k8v4 (int8 K, rotated Q4_0 V). Engine default fp16; the sample config in the README uses int8 | generate.cpp:327, 1736-1743; README.md (Settings) |
| int8 KV layout | int8 codes with an fp16 scale per 64 values (4 per 256-wide head, separate for K and V), scale = f16(amax/127) | kv_q8.hpp:3-5, 26-28; kv_q8.cu:46-52 |
| Attention | K/V dequantized to fp32; scores in fp32; online softmax per 64-cell chunk in fp32 (`__expf`), fp32 merge; output gate sigmoid, then q8_1 for the o-projection | qsa_dec_attn.cu:114-138, 255-420 |
| Draft-only attention | fp16 mma kernel; it changes drafts only, never accepted tokens | qsa_dec_attn.cu:424-427 |
| Indexer | Keys fp16-rounded, held fp32; block scores fp32; radix top-k, ties to the lowest cell | native_qsa_indexer.hpp:14-18; qsa_select.hpp:6-9 |
| GDN recurrent state and conv history | fp32 | dec_gdn.hpp:21; verify.cpp:1134, 1158-1160 |
| Router softmax and top-10 | fp32, ties to the lower expert index, weights renormalized over the ten | native_router.cu:39-91 |
| Expert combine | fp32, in router rank order, plus shared expert | native_moe.cu:35-50 |
| Final logits | fp32 `[T][n_vocab]` | verify.hpp:489 |
| Greedy sampling | fp32 argmax, lowest index wins ties | sampler.cu:79-90 |
| KLD scoring (host) | Log-sum-exp and KLD accumulated in double from the fp32 logits | ppl_gate.hpp:125-154 |

### Prefill

Prefill and decode are different computations that read and write the same KV and state formats.

| Item | Prefill | Source |
|---|---|---|
| Dense projections | Weights dequantized to fp16, activations rounded to fp16, cuBLAS with fp32 accumulate; BF16 tensors BF16 x BF16 with fp32 compute; no q8_1 | gemm.cu:125-185; prefill.cpp:1122-1131 |
| hc matrices | Activations rounded to BF16; down/up use the int8 copy dequantized to BF16; inject uses the BF16 tensor | prefill.cpp:1500-1506; gemm.cu:136-159 |
| Router | BF16 activation (decode: fp32) | prefill.cpp:1631 |
| Shared expert | fp16 activation, fp16 SwiGLU | prefill.cpp:1634-1637 |
| Routed experts | Fused int8 mma kernels, llama.cpp MMQ (q8_1) or dequant-to-fp16 cuBLAS | moe_fused_iq.hpp:1-12; moe_mmq.hpp:1-6 |
| Attention | fp16 mma with hi/lo split of q and p, fp32 accumulate | qsa_prompt_attn.hpp:1-22 |
| Indexer scores | 3xTF32 tensor cores | qsa_select.hpp:36-38 |
| GDN | State fp32; out-projection input fp16 | kernels.cu:609-623 |

## Speculative decoding does not change output

Verify guarantees that a token's logits are bit for bit those of plain decode, whatever the window width
(verify.hpp:3-8). This section tests that end to end.

**Method.** 275 prompts: 272 chat-templated prompts with thinking off (152 prose, 60 chat, 60 code) plus the 3
`data/bench` decode prompts (conv, struct, prose). 256 new tokens each, temperature 0, one engine session per
configuration, same prompt order. A logit-hash diagnostic prints an FNV-1a hash of the full-vocabulary fp32 head logits
of every accepted row. Token ids and the hash at each generated position are compared between `--spec 0` and
`--spec 7 --spec-min-p 0.5`.

Expert tiers are pinned (`--adapt-swaps 0 --peer-adapt-swaps 0 --pcie-frac 0`, with `--expert-cache` and `--peer-slots`
fixed to the sizes auto-sizing picks), so the same experts sit on the same device in both runs.

| Pack | Prompts | Identical tokens | Logit rows compared | Bit-identical rows | First divergence | Accepted drafts | Decode tok/s, spec 0 → spec 7 |
|---|---|---|---|---|---|---|---|
| IQ3_XXS | 275 | 275 (100%) | 62,271 | 62,271 | none | 37,892 / 54,206 (69.9%) | 183.9 → 319.8 |
| IQ3_S | 275 | 275 (100%) | 60,878 | 60,878 | none | 36,650 / 52,312 (70.1%) | 145.1 → 239.5 |

Throughput is decode-only over these 275 requests, at 400 W. Verify windows in the spec 7 run have widths 1 to 7, so the
compared rows include logits computed at every window width T, all equal to the T=1 logits of the spec 0 run. On an
earlier build, a repeated IQ3_XXS spec 0 run was bit-identical to the first (62,491 of 62,491 rows), so the setup is
deterministic run to run.

### Forced-window KLD

The quality gate is teacher-forced: the window tokens are the reference text, forced to width `--t`, and the drafter
output is not used (`src/program/generate.cpp`, the `ppl != nullptr` branches). Forcing the window widths the drafter
produces exercises the same verify kernels as real drafting.

IQ3_XXS, 20 chunks of wikitext-2, against the llama.cpp Q8_0 base (base PPL 2.2815), int8 KV, tiers pinned, 400 W:

| Forced window T | PPL | KLD mean | KLD median | KLD p99 | Same top-1 | rows.tsv md5 |
|---|---|---|---|---|---|---|
| 1 | 2.5330 | 0.27827 | 0.02674 | 3.5108 | 86.80% | 179c7f42f1e6d89459bf27f780e99b35 |
| 3 | 2.5330 | 0.27827 | 0.02674 | 3.5108 | 86.80% | 179c7f42f1e6d89459bf27f780e99b35 |
| 7 | 2.5330 | 0.27827 | 0.02674 | 3.5108 | 86.80% | 179c7f42f1e6d89459bf27f780e99b35 |

All 5,100 per-row lines (NLL and KLD) are byte identical across T=1, 3 and 7 (`ppl_gate.py --diff`: max |dKLD|
0.000000). Window width does not change the logits, so KLD with MTP on equals KLD with MTP off; the 0.278 is the quant
against the base.

### Scope

- Identity is bitwise with the expert tiers pinned. With adaptation on, experts move between the CPU pool and the GPU,
  which round differently: a rounding difference, not a quality difference.
- The GPU tiers (VRAM and peer card) share one arithmetic; the CPU tier quantizes with ggml's vec_dot_type (Q8_K per 256
  for i-quant gate/up, against q8_1 per 32 on the GPU; native_expert.cpp:52-62). In the quality gate about 0.8% of
  routed experts were computed on the CPU (IQ3_XXS, default placement). The split is fixed per placement, so it is the
  same for spec 0 and spec 7.
- Prompts are 19 to 395 tokens with 256 new tokens: this tests the decode path, not long context.

## Quality against Q8_0

`basalt-bench --quality` (`tools/ppl_gate.py`) scores wikitext-2 (`wiki.test.raw`) in 512-token chunks. The first half
of each chunk is prefilled and the second half (rows 256 to 510) is scored through the decode path, against a llama.cpp
Q8_0 base file written by `llama-perplexity --kl-divergence-base`, read in llama.cpp's own `.kld` format.

The harness (`include/basalt/program/ppl_gate.hpp:112-139`) computes KL(base ‖ test) over the full vocabulary from the
fp32 log-softmax of the head logits; the base is decoded as scale*u16+min, and terms with base log-prob ≤ -16 are
skipped (a tiny underestimate). The prompt half is checked equal to the base tokens, the window tokens are the base
tokens, and KLD is flat across positions, so the two are aligned.

llama.cpp (commit 436f6f89e, CUDA) scores the same second half of each chunk, with the whole chunk in one batched
forward pass. 100 chunks, base PPL 3.286070, 400 W:

| Pack | Engine | PPL | KLD mean | KLD median | KLD p99 | Same top-1 |
|---|---|---|---|---|---|---|
| GSQ-RCO IQ3_S | llama.cpp | 3.3728 | 0.1526 | 0.0340 | 1.750 | 87.20% |
| GSQ-RCO IQ3_S | Basalt | 3.3723 | 0.15264 | – | – | 87.16% |
| GSQ-RCO IQ3_XXS | llama.cpp | 3.4675 | 0.2428 | 0.0596 | 2.607 | 84.05% |
| GSQ-RCO IQ3_XXS | Basalt | 3.4678 | 0.2433 | – | – | – |

At 20 chunks (Basalt: the README table; llama.cpp: its running value at chunk 20):

| Pack | Basalt KLD / top-1 | llama.cpp KLD / top-1 |
|---|---|---|
| GSQ-RCO IQ3_S | 0.177 / 89.4% | 0.1753 / 89.22% |
| GSQ-RCO IQ3_XXS | 0.277 / 86.8% | 0.2780 / 86.65% |
| UD-Q4_K_XL | 0.068 / 93.0% | 0.0579 / 93.96% |

IQ3_S and IQ3_XXS agree between the two engines to within 0.002 KLD and 0.2 points of top-1 at both chunk counts. The
first 20 chunks read higher than the 100-chunk set (higher KLD and higher top-1 for the IQ3 packs), so the README table,
which uses 20 chunks, reads higher than the 100-chunk numbers; the 100-chunk numbers are the ones to compare. The one gap
is UD-Q4_K_XL at 20 chunks (0.068 vs 0.0579), explained in the next section.

## Why Q8_0 against Q8_0 is not zero

The Basalt Q8_0 pack scores KLD 0.0456 and top-1 95.6% (4 chunks, fp16 KV) against the stock llama.cpp Q8_0 base. This
is mostly the base's own noise. llama.cpp quantizes the activations of its Q8_0 matmuls. In the hyper-connection read
(down projection 10240 → 320, then up) the sum has cancellation, so Q8_0 quantization of the input (0.2 to 0.4% per
element) becomes about 1% in the mixed stream at layer 0. Basalt's hc read uses fp32 activations and matches the exact
math to 1e-5.

At layer 0, computed in numpy from llama.cpp's own hc_norm output and the GGUF Q8_0 hc weights, against an fp64 result
with no activation quantization: Basalt's mixed stream has relative error 0.00000 and llama.cpp's 1.06%. llama.cpp
against an emulation of Q8_0 activation quantization differs by 0.009%.

With the hc read cast to F32 in llama.cpp (a local patch), 4 chunks, 400 W:

| Comparison | KLD mean | Same top-1 |
|---|---|---|
| Basalt Q8_0 vs stock base | 0.0456 | 95.6% |
| Basalt Q8_0 vs F32-hc base | 0.0233 | 97.16% |
| llama.cpp Q8_0 with F32 hc vs stock base | 0.0436 | 96.0% |
| llama.cpp with ssm_alpha/beta activation quantization also removed vs F32-hc base | 0.0241 | 96.86% |

Basalt's 0.0233 is the level of two independent implementations of this model (0.0241 between two llama.cpp variants).
The stock base's hc noise lifts every pack's KLD by about 0.01 to 0.02 (Basalt, 4 chunks):

| Pack | Stock base | F32-hc base |
|---|---|---|
| UD-Q4_K_XL (hc Q8_0) | 0.0825 | 0.0748 |
| GSQ-RCO IQ3_S (hc BF16) | 0.2100 | 0.1950 |

The noise cancels only when the tested llama.cpp pack also has Q8_0 hc and quantizes the same way, which is why
llama.cpp against its own base scores better on UD-Q4_K_XL (0.0579 vs 0.068 at 20 chunks). Reported KLD numbers
therefore include about 0.01 to 0.02 of the reference's own noise. They are kept against the stock base because anyone
can regenerate it.

This is not a CPU-tier effect: the 4-chunk Q8_0 result holds whatever share of experts sits on the CPU pool (0.0456 to
0.0488 at the default placement, 0.0466 with 79% on the CPU). Other causes tested and excluded: KV mode, window width T,
`--gr-fp32-activations`, dense_tc against native_mmvq (bit-identical), and refitting the hc8 scales.

## Reproduce

Build llama.cpp, then the Q8_0 base from the Q8_0 GGUF (100 chunks):

```sh
llama-perplexity -m MODEL-Q8_0-00001-of-0000N.gguf -f wikitext-2-raw/wiki.test.raw \
    --kl-divergence-base flashnext-q8-base.kld -c 512 -b 2048 -ngl 99 --n-cpu-moe 99 -t 16
```

llama.cpp's numbers for a pack:

```sh
llama-perplexity -m PACK.gguf -f wikitext-2-raw/wiki.test.raw --kl-divergence-base flashnext-q8-base.kld \
    --kl-divergence -c 512 -b 2048 -ngl 99 --n-cpu-moe 99 -t 16
```

Basalt's numbers, and the forced-window identity check (run with `--t` 1, 3 and 7, then diff the rows):

```sh
python3 tools/ppl_gate.py --model PACK.basalt --engine-bin bin/basalt --base flashnext-q8-base.kld \
    --chunks 20 --t {1|3|7} --out DIR --expert-cache N --peer-slots M --kv int8 --resident-experts \
    --peer-exclusive --prefill auto
python3 tools/ppl_gate.py --diff DIR1/rows.tsv DIR7/rows.tsv
```

MTP on/off identity: one engine session per configuration, driven over `--serve`:

```sh
bin/basalt --serve --model MODEL.basalt {--spec 0 | --spec 7 --spec-min-p 0.5} \
    --expert-cache N --peer-slots M --prefill auto --kv int8 --vram-reserve-mib 700 --resident-experts \
    --peer-exclusive --peer-device 1 --max-context 8192 --adapt-swaps 0 --peer-adapt-swaps 0 --pcie-frac 0 \
    --exp concur9_logit_hash=1
```

Wait for a `READY` line on stdout, then write one request per line on stdin: `GEN 256 temperature=0 ID,ID,...` with the
prompt token ids. Each generated token comes back on stdout as `T ID`, and `DONE` ends the request; `QUIT` ends the
session. The logit hashes are printed on stderr as `LH` lines, one per accepted row with its position and hash.

Compare the token ids and the per-position hashes between the two sessions. With `--spec 0` there is one row per token;
the `--spec 7` run may print extra rows past the end of a request, so match rows by position. The pinned sizes used here
were N = 11582, M = 8876 for IQ3_XXS and N = 9979, M = 7562 for IQ3_S.
