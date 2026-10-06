<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/basalt-lockup-dark.png">
    <img src="assets/basalt-lockup-light.png" alt="Basalt" width="360">
  </picture>
</p>

Basalt is a local inference engine for the **Qwen3.8-Flash-Next** mixture-of-experts model. It targets one thing:
Linux with an NVIDIA Blackwell GPU (sm_120). The experts that do not fit in VRAM live in system RAM (and optionally on
a second GPU used as a peer expert tier), so a large MoE model runs on a single workstation.

It is a fork of [Strata](https://github.com/Niko1221/Strata) (MIT), stripped down to that one target. See
[THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).

## What is in the repository

- `src/`, `include/`: the C++/CUDA engine (one binary, `basalt`).
- `serve/`: a Python server with an OpenAI- and Anthropic-compatible API, and a stats page.
- `tools/`: the model-file converter (`convert.py`, `model_file.py`), the drafter fetcher (`mtp_fetch.py`) and the
  PPL/KLD quality gate (`ppl_gate.py`).
- `tests/`: parity and unit tests.

## Requirements

- Linux x86-64 with AVX2 (AVX-512 is used when present).
- An NVIDIA Blackwell GPU (compute capability 12.0) and CUDA 13.
- CMake 3.24 or newer and a C++20 compiler.
- Enough system RAM to hold the experts that do not fit in VRAM.
- Python 3.10 or newer for the server: `pip install -r requirements.txt`.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j        # bin/basalt, bin/basalt-bench
ctest --test-dir build        # the parity tests need a GPU
```

## Run

```sh
bin/basalt serve --model /path/to/model.basalt --host 127.0.0.1 --port 8080 [--alias NAME]
```

The server listens on 127.0.0.1 by default; do not expose it beyond that without `--api-key`.

Diagnostics are engine command-line flags (`--profile`, `--wait-stats`, `--decode-timing`, `--draft-log FILE`); no
environment variables control the engine.

## The `.basalt` model file

A model is one file. It holds everything the engine, the server and the vision sidecar load: the weights, the
tokenizer and chat template, the MTP drafter, the expert profile, the 8-bit hyper-connection copies, the vision
projector, and the default engine arguments and sampling. `basalt serve --model FILE` needs nothing else.

- **Layout:** a 64 KiB header (magic `BASALTMF`, version, a table of up to 511 named sections, each with its offset,
  size and SHA-256), then the sections, each aligned to 2 MiB so it can be memory-mapped or read with `O_DIRECT`.
- **Sections:** `meta` (JSON: geometry, the source GGUF metadata, every tensor's type, shape and offset, engine
  arguments, sampling, provenance and license), `tokenizer/*`, `dense`, `experts` and `experts.layout` (the routed
  experts, laid out per layer), `gguf` (every non-expert source tensor), `mtp/*` (the drafter), `profile`, `hc8` and
  `vision/*`. The authoritative description is [include/basalt/core/model_format.hpp](include/basalt/core/model_format.hpp).
- **No requantization:** tensors are copied byte for byte from the source GGUF. The only exception is
  `--compat-bf16`, for GGUFs that store the small projections the engine reads as BF16 in another type; it converts
  them and records that in `meta`.

Build one from the source GGUF shards (give shard 1; the others are found beside it):

```sh
python3 tools/convert.py --gguf MODEL-00001-of-0000N.gguf --mmproj MMPROJ.gguf --out MODEL.basalt \
    [--mtp-source DIR] [--draft-vocab FILE] [--profile FILE] [--hc8 FILE] [--compat-bf16] [--name NAME]
```

`python3 tools/convert.py --help` describes every input.

## Models

Quality measured with `basalt-bench --quality`: 20 chunks of wikitext-2 (512 tokens each), scored through the decode
path against the Q8_0 reference. Lower KLD and higher top-1 agreement are closer to Q8_0.

| model | source | file | experts | PPL | KLD mean | same top-1 |
|---|---|---:|---:|---:|---:|---:|
| GSQ-RCO IQ3_XXS | [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) | 74.4 GiB | 40.0 GiB | 2.535 | 0.277 | 86.8% |
| GSQ-RCO IQ3_S | [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) | 81.0 GiB | 46.8 GiB | 2.387 | 0.177 | 89.4% |
| UD-Q2_K_XL | [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | 76.5 GiB | 42.9 GiB | 2.755 | 0.323 | 84.3% |
| UD-IQ3_XXS | [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | 79.4 GiB | 45.3 GiB | 2.564 | 0.244 | 87.0% |
| UD-Q3_K_XL | [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | 86.8 GiB | 52.0 GiB | 2.457 | 0.160 | 89.3% |
| UD-Q4_K_XL | [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | 106.7 GiB | 71.7 GiB | 2.321 | 0.068 | 93.0% |
| Q8_0 | [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | 178.3 GiB | 119.5 GiB | 2.282 | reference | |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT, see [LICENSE](LICENSE).
