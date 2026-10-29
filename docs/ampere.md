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

```bat
hf download jesdga/Qwen3.8-Flash-Next-Basalt Qwen3.8-FN-GSQ-RCO-IQ3_S.basalt --local-dir models
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
17 × 19. The server is available locally at `http://127.0.0.1:8080/`; restart it
from the repository root with `bin\basalt.exe serve --config basalt_config.json`.

`basalt-probe` cold decode, 400 tokens per standard fixture, at 350 W:

| Context | Conversation | Structured | Prose |
| --- | ---: | ---: | ---: |
| 16384 | 34.6 tok/s | 28.2 tok/s | 28.2 tok/s |
| 8192 | 35.7 tok/s | 15.8 tok/s | 16.8 tok/s |

These sequential development runs have different cache and memory conditions;
they do not isolate context length as the cause of the difference. The delivered
configuration uses 8192 tokens and automatic placement. The first API load put
9043 experts in 17.15 GiB of GPU cache, 23.01 GiB of experts in pinned RAM, and
left about 6.5 GiB of experts on SSD. Free RAM changes the plan and SSD reads can
dominate a request. The probes measured 746–813 GB/s GPU copies, 26.2–26.3 GB/s
host-to-GPU and 30–33 GB/s host reads at 350 W.

The 16K probe's aggressive prefill-share trial hit the RAM safety guard; its
subsequent adaptation figures are excluded. The default prefill share is kept.
The 8K adaptation trials did not beat the repeated default beyond the measured
30.9% noise band, so the original placement and worker defaults are also kept.
No model-level speed comparison with the upstream hardware path, greedy-identity
comparison, or Q8_0 PPL/KLD quality gate has been completed.
