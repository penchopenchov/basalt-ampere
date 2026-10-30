# Ampere setup

This fork defaults to CUDA compute capability 8.6 for RTX 3090. The upstream
Blackwell build can still be requested with `-DCMAKE_CUDA_ARCHITECTURES=120`.
Ampere uses the existing quantized MMVQ/MMQ and per-stage recurrence paths.
TMA, thread-block cluster and native block-scaled FP4 specializations require
newer hardware and must not be dispatched by the Ampere build.

## Windows build

Use an x64 Visual Studio 2022 Developer command prompt with CUDA 13 and Ninja:

```bat
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements-test.txt
cmake -S . -B build-ampere -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DBUILD_TESTING=ON -DPython3_EXECUTABLE="%CD%/.venv/Scripts/python.exe"
cmake --build build-ampere -j 6
ctest --test-dir build-ampere -L cpu -LE avx512 --output-on-failure
ctest --test-dir build-ampere -L gpu --output-on-failure
```

The Python server uses that local virtual environment:

```bat
.venv\Scripts\python.exe -m serve.server --help
```

## Requested model

The upstream pack is `jesdga/Qwen3.8-Flash-Next-Basalt`, file
`Qwen3.8-FN-GSQ-RCO-IQ3_S.basalt` (86,932,948,345 bytes).
Its published SHA-256 is:

```text
509b30a0dd476441d2ce874be8fbb20b8da4065037f5215ccfde37a379bc160d
```

The complete local download was verified against this hash.

The IQ3_XXS comparison pack, `Qwen3.8-FN-GSQ-RCO-IQ3_XXS.basalt`, is
79,867,643,037 bytes. Its verified published SHA-256 is:

```text
d02bb1b188bbc9ed6bc88ce5e89c7c2d64247290c7bae70fbb9d9d17fe65ed71
```

Its expert section is 40.0 GiB. The smaller footprint leaves more RAM headroom;
the [upstream model card](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt)
reports lower accuracy than IQ3_S. Neither quant is validated here against Q8_0.

The default server selects IQ3_XXS; see [128K default context](#128k-default-context).

```bat
hf download jesdga/Qwen3.8-Flash-Next-Basalt Qwen3.8-FN-GSQ-RCO-IQ3_XXS.basalt --local-dir models
bin\basalt.exe serve --config basalt_config.json
```

The pack contains 46.84 GiB of expert weights, 1.43 GiB of dense weights and
31.03 GiB of embedded GGUF data, plus the drafter and vision sections. Storage
size is not resident memory: the planner places the expert and dense sections
across VRAM and RAM. Actual placement depends on free memory and engine
headroom. Use `basalt-probe` with the actual pack to assess placement and rates.
No overclock, power-limit or registry change is needed.

## Validation

The target GPU's current power limit is 350 W. Development benchmark results
must state that limit and the exact kernel/shape. Synthetic kernel checks alone
do not establish model-level accuracy or throughput. Perplexity/KLD additionally
need a Q8_0 reference; no quality equivalence is assumed from this port.

The Release build was checked with MSVC 19.44 and CUDA 13.3 on Windows:

- 58 CPU tests passed, excluding the separately labelled AVX-512 tests.
- The GPU suite covers 91 tests plus one generated CPU fixture. After focused
  corrections and reruns, 84 GPU tests pass and 7 report capability skips.
  These include cluster/TMA paths, native FP4, and repinning when the driver
  uses a separate mapped device address. Software NVFP4 MMQ and decode pass.
- 624 Python server tests completed with 8 skips.
- The platform-boundary, stderr-logging and whitespace checks pass.
- Quantized A/B benchmarks pass their parity checks; the default dispatch test
  checks CPU-reference outputs, sentinel writes and actual Tensor Core launches.

The disk-reader self-test exceeded a 120-second limit while the full model's
checksum was running, then passed alone in 35.6 seconds. Its I/O workload and
the model load should be timed separately. The existing Windows asynchronous
close/reopen test also passed 20 repetitions after cancelled-I/O accounting was
corrected. Hyperconnection prefill's cuBLAS width-boundary difference is
documented in [Ampere precision](precision/ampere.md).

Development A/Bs on RTX 3090 at 350 W, two runs, microseconds per operation:

| Format and input/output dimensions | Window | Existing dp4a | Tensor Core |
| --- | ---: | ---: | ---: |
| IQ3_S, 2560 / 6144 | 5 | 34.3 / 29.0 | 22.3 / 22.8 |
| IQ3_S, 2560 / 6144 | 6 | 32.9 / 35.8 | 22.5 / 22.3 |
| IQ4_XS, 2560 / 10240 | 7 | 69.6 / 56.8 | 43.7 / 40.1 |
| IQ4_XS, 2560 / 12288 | 7 | 71.1 / 82.4 | 61.1 / 58.1 |

Automatic Tensor Core dispatch is limited to sm_86, IQ3_S 2560 / 6144 with
windows at least 5, and IQ4_XS 2560 / 10240 or 12288 with windows at least 7.
Other shapes keep their previous dispatch. Explicit test setters still exercise
the full implementations. These are development measurements, not full-model
speed claims or community benchmark results.

## Full-model development runs

The verified IQ3_S pack runs on this RTX 3090 at its unchanged 350 W limit.
The local API returned the exact requested phrase and a coherent Fibonacci
function that passed checks for 0, 1, 2, 10, 20 and a negative input. Code and
prose requests generated at 24.7 and 32.8 tokens/s respectively (204 and 256
generated tokens, including reasoning; the prose request reached its cap).
After reloading the delivered 8K setup, exact text and constrained JSON output
were checked again: the arithmetic request returned `{"answer": 323}` for
17 × 19. Start the local server from the repository root with
`bin\basalt.exe serve --config basalt_config.json`.

`basalt-probe` cold decode, 400 tokens per standard fixture, at 350 W:

| Context | Conversation | Structured | Prose |
| --- | ---: | ---: | ---: |
| 16384 | 34.6 tok/s | 28.2 tok/s | 28.2 tok/s |
| 8192 | 35.7 tok/s | 15.8 tok/s | 16.8 tok/s |

These sequential development runs have different cache and memory conditions;
they do not isolate context length as the cause of the difference. These
8K measurements use automatic placement. The first API load put
9043 experts in 17.15 GiB of GPU cache, 23.01 GiB of experts in pinned RAM, and
left about 6.5 GiB of experts on SSD. Free RAM changes the plan and SSD reads can
dominate a request. The probes measured 746–813 GB/s GPU copies, 26.2–26.3 GB/s
host-to-GPU and 30–33 GB/s host reads at 350 W.

The 16K probe's aggressive prefill-share trial hit the RAM safety guard; its
subsequent adaptation figures are excluded. The default prefill share is kept.
The 8K adaptation trials did not beat the repeated default beyond the measured
30.9% noise band, so the original placement and worker defaults are also kept.
No model-level speed comparison with the upstream hardware path or Q8_0
PPL/KLD quality gate has been completed.

## Experimental RAM staging

Solo-window staging requires an explicit diagnostic opt-in and uses the existing copy and
grouped-expert kernels. Mixed layer sizes use a shared stride sized for the
largest expert. Source aliases refresh between windows after placement changes.
The activation and reduction changes are described in
[Ampere precision](precision/ampere.md).

Development A/B on RTX 3090 at 350 W: IQ3_S, 256 greedy tokens per fixture,
12K context, int8 KV, prompt cache disabled, 9021 fixed GPU expert entries,
adaptive swaps disabled, all remaining experts in RAM. The experimental arm
uses CPU/link cost estimates of 20/24 GB/s. Each arm starts a fresh engine and
runs conversation, structured and prose fixtures in that order.

| Fixture | Path | tok/s | ms/window | passes/s |
| --- | --- | ---: | ---: | ---: |
| Conversation | Baseline | 43.9 | 49.42 | 20.23 |
| Conversation | Staging | 69.1 | 35.98 | 27.79 |
| Structured | Baseline | 33.5 | 165.99 | 6.02 |
| Structured | Staging | 51.0 | 116.68 | 8.57 |
| Prose | Baseline | 31.3 | 71.64 | 13.96 |
| Prose | Staging | 45.4 | 49.91 | 20.04 |

The structured prefix matches all 256 token IDs. Conversation first differs at
token 30 and prose at token 5; these are changed-numerics development results,
not quality equivalence. Window width identity and Q8_0 PPL/KLD remain
unvalidated, so this path remains experimental.

At 350 W, the staged-copy microbench moves four synthetic blobs (12.3 MB) in
512.1 us, or 24.0 GB/s, with eight copy blocks. Captured cross-stream replay
matches the CPU routing reference and resident GPU output, including widened
staging stride and relocated host aliases. Both staging format suites pass;
the existing grouped-IQ format sweep also passes.

## IQ3_XXS standard fixtures

Development A/B on RTX 3090 at 350 W, seed 42, fresh engines, 12K context,
int8 KV, prompt cache disabled and adaptive swaps enabled. Both arms hold
10806 GPU expert entries in 17.52 GiB and read no expert blobs from SSD.
The staging arm uses the same 20/24 GB/s cost estimates as above.

| Fixture | Baseline tok/s | Staging tok/s | Baseline ms/window | Staging ms/window |
| --- | ---: | ---: | ---: | ---: |
| Conversation | 78.8 | 85.1 | 36.10 | 32.43 |
| Structured | 163.7 | 192.3 | 47.25 | 40.35 |
| Prose | 51.5 | 92.2 | 42.61 | 24.37 |

Baseline/staging passes per second are 27.70/30.83, 21.16/24.78 and
23.47/41.03 respectively. Both structured runs produce the expected 7230
tokens. Both arms exercise adaptive expert exchanges; the staging arm
completes 28770 exchanges. These are single-pair development results with
changed outputs. N-gram I/O varies substantially: an additional unchanged-engine
run with 15.0 GiB of GPU expert cache reaches 83.5/157.6/69.8 tok/s at 350 W.
The prose difference cannot be attributed entirely to expert staging.

The n-gram table still reads SSD pages when all expert weights fit in RAM.
The decode timing diagnostic includes cumulative row-cache hits and I/O latencies.
A shorter SSD keep-alive trial did not improve throughput and is not retained.
The real-table reader test checks direct/mapped byte identity; this does not
remove storage-read latency from inference.

The verified smaller pack can be served with the baseline engine:

```bat
bin\basalt.exe serve --config basalt_config.iq3xxs.json
```

That configuration uses automatic placement and 8K context. It does not enable
experimental staging or change the model selected by `basalt_config.json`.

With this configuration and the baseline engine, API smoke checks on RTX 3090
at 350 W return the exact requested phrase and constrained JSON `{"answer": 323}`.
A code request generates 198 tokens at 51.4 tokens/s; a prose request reaches
its 256-token limit at 63.4 tokens/s. Thinking is disabled for these requests.
These short requests differ from the native benchmark fixtures and are not
an isolated comparison with IQ3_S or a quality gate.

## 128K default context

The default configuration serves IQ3_XXS at 131072 tokens with experimental
staging enabled, int8 KV and automatic expert placement. The pack's metadata
declares a 262144-token model context; no RoPE extension is requested.
The separate `basalt_config.iq3xxs.json` retains an 8K context.
The elastic KV cache backs about 0.7 GiB initially and can grow to 2.1 GiB,
reclaiming VRAM from expert slots as requests grow. Short requests leave more
space for experts. The context limit includes the prompt and reply.

Baseline API checks with staging off on RTX 3090 at 350 W, thinking disabled:

- A 129959-token prompt returns the exact key placed at its beginning.
  Prompt processing takes 47.0 seconds; the 11-token reply is too short for
  a representative generation rate. A subsequent short request passes.
- A 130011-token prompt generates 512 prose tokens at 56.2 tokens/s after
  45.8 seconds of prompt processing (2837 prompt tokens/s). It reaches the
  requested output cap at 130523 total tokens.
- The API advertises 131072 context tokens. Existing configuration and
  context-overflow tests run 41 tests with one skip and no failures.
- An API request whose prompt plus requested reply exceeds 131072 tokens
  returns HTTP 400 without truncation. A following short request passes.

The long prompts use repetitive synthetic filler. These checks establish
capacity and a simple retrieval result, not broad long-context accuracy or
Q8_0 quality equivalence. Different documents can have different read and
generation rates; storage and expert placement remain relevant.

The local configuration explicitly enables solo staging with the experimental
20/24 GB/s CPU/link cost estimates. These are scheduling estimates; engine
defaults remain opt-in. At full context, both arms have 9656 GPU expert slots,
with the remaining expert weights in RAM.

Development API comparison at 350 W, 130011 prompt tokens and 512 generated
prose tokens, greedy decoding, the same fixture and request order:

| Measurement | Staging off | Staging on |
| --- | ---: | ---: |
| Generation | 56.2 tokens/s | 82.7 tokens/s |
| Generation time | 9.10 s | 6.19 s |
| Prompt processing | 45.83 s | 45.50 s |
| Whole request | 54.99 s | 51.73 s |

This single development pair improves generation by 47.2%; prompt processing
dominates the whole request, which finishes about 5.9% sooner. The prose first
differs after 1266 common characters, so the comparison includes different
continuations and expert routing. It does not isolate a kernel-only gain.
The exact key retrieval still passes. Near 130K context, constrained arithmetic
JSON returns 323 and the generated iterative Fibonacci function passes inputs
0, 1, 2, 10, 20 and a negative-input check. Overflow rejection and short-request
recovery also pass. Full Q8_0 PPL/KLD and window-width identity remain unvalidated.
