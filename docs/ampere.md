# Ampere setup

This fork defaults to CUDA compute capability 8.6 for RTX 3090. The upstream
Blackwell build can still be requested with `-DCMAKE_CUDA_ARCHITECTURES=120`.
Ampere uses the existing quantized MMVQ/MMQ and per-stage recurrence paths.
TMA, thread-block cluster and native block-scaled FP4 specializations require
newer hardware and must not be dispatched by the Ampere build.

## Windows build

Use an x64 Visual Studio 2022 Developer command prompt with CUDA 13 and Ninja:

```bat
cmake -S . -B build-ampere -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DBUILD_TESTING=ON
cmake --build build-ampere -j 6
ctest --test-dir build-ampere -L cpu -LE avx512 --output-on-failure
ctest --test-dir build-ampere -L gpu --output-on-failure
```

The Python server uses a local virtual environment:

```bat
uv venv .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
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
