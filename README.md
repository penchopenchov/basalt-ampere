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
- `tools/convert/`: the model-file converter (`convert.py`), its section builders and the drafter fetcher (`mtp_fetch.py`).
- `tools/`: the .basalt reader and writer (`model_file.py`), the PPL/KLD quality gate (`ppl_gate.py`), the greedy-identity
  check and the tokenizer.
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

Image input needs the vision sidecar, `basalt-vision`, a separate build on llama.cpp's mtmd library. The server looks
for it next to `bin/basalt`:

```sh
cmake -S tools/vision -B build-vision -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=third_party/llama.cpp \
      -DVISION_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native     # leave out VISION_CUDA for a CPU-only encoder
cmake --build build-vision -j --target basalt-vision
cp build-vision/bin/basalt-vision bin/
```

Then start the server with `--vision` (encoder on the GPU; with two cards in `--gpu` it uses the second) or
`--vision-cpu` (no VRAM, slower), or `"vision": true` in the config file's `server` section. Vision is off by default.

## Run

```sh
bin/basalt serve --model /path/to/model.basalt --host 127.0.0.1 --port 8080 [--alias NAME]
```

The server listens on 127.0.0.1 by default; do not expose it beyond that without `--api-key`.

### Settings: `--config`

The engine's built-in defaults are untuned (4096 context, fp16 KV, no speculation, no expert cache), and the model
file carries no engine flags. Put every setting in one `basalt_config.json` and pass `--config`:

```json
{
  "server": {
    "model": "model.basalt",
    "alias": "Qwen 3.8 FN",
    "gpu": "0,1",
    "host": "127.0.0.1",
    "port": 8080
  },
  "engine": {
    "expert-cache": "auto",
    "prefill": "auto",
    "spec": 7,
    "spec-min-p": 0.5,
    "max-context": 262144,
    "kv": "int8",
    "resident-experts": true,
    "vram-reserve-mib": 700
  }
}
```

```sh
bin/basalt serve --config basalt_config.json                     # everything from the file
bin/basalt serve --config basalt_config.json --max-context 32768  # the command line wins
```

### Concurrent requests: `--parallel N`

`--parallel N` (1..8, also `"parallel"` in the config's `server` section) serves N requests at once in the
engine's batch slots; the default 1 is the single-stream path. Further requests wait in arrival order (a later one
may go first only to resume a slot that holds its conversation, a few times at most). A slot is freed when its
request finishes, is cancelled or its client leaves. `/metrics` reports `live.slots` (`total`, `active`,
`queue_depth`). Every slot holds a full `--max-context` of KV cache, taken from the expert cache; a per-slot
context is not an engine option yet, so lower `--max-context` when raising N.

- `server` holds the server's options (`python -m serve.server --help`), named as the flags without the dashes:
  strings or numbers, `true`/`false` for a switch (`vision`, `lazy`, ...), a list for the repeatable ones
  (`allowed-host`, `cors-origin`). `model` and `tokenizer` paths are relative to the file.
- `engine` holds engine flags (`bin/basalt --help`), the same way: a string or number, `true` for a switch, `false`
  to leave it off. Any flag the engine accepts works, on the command line as well (`--kv int8`, `--resident-experts`).
  A name the engine does not accept stops the server at startup, as does one in the wrong section.
- Precedence: command line > config file > built-in defaults. `spec`, `spec-min-p`, `mtp-max-t`, `peer-exclusive` and
  `max-context` are engine flags; they are also server flags on the command line.
- The API key is command line only (`--api-key`); a config file that sets it is refused.
- The startup line `[server] engine: ...` lists the settings that came from the file or the command line; without
  `--config` it says the built-in defaults apply.

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
python3 tools/convert/convert.py --gguf MODEL-00001-of-0000N.gguf --mmproj MMPROJ.gguf --out MODEL.basalt \
    [--mtp-source DIR] [--draft-vocab FILE] [--profile FILE] [--hc8 FILE] [--compat-bf16] [--name NAME]
```

`python3 tools/convert/convert.py --help` describes every input.

## Models

Quality measured with `basalt-bench --quality`: 20 chunks of wikitext-2 (512 tokens each), scored through the decode
path against the Q8_0 reference. Lower KLD and higher top-1 agreement are closer to Q8_0.

| model | source | file | experts | PPL | KLD mean | same top-1 |
|---|---|---:|---:|---:|---:|---:|
| GSQ-RCO Q2_0 | [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) | 64.9 GiB | 31.6 GiB | 2.971 | 0.486 | 80.6% |
| GSQ-RCO IQ3_XXS | [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) | 74.4 GiB | 40.0 GiB | 2.535 | 0.277 | 86.8% |
| GSQ-RCO IQ3_S | [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) | 81.0 GiB | 46.8 GiB | 2.387 | 0.177 | 89.4% |
| UD-Q2_K_XL | [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | 76.5 GiB | 42.9 GiB | 2.755 | 0.323 | 84.3% |
| UD-IQ3_XXS | [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | 79.4 GiB | 45.3 GiB | 2.564 | 0.244 | 87.0% |
| UD-Q3_K_XL | [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | 86.8 GiB | 52.0 GiB | 2.457 | 0.160 | 89.3% |
| UD-Q4_K_XL | [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | 106.7 GiB | 71.7 GiB | 2.321 | 0.068 | 93.0% |
| Q8_0 | [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | 178.3 GiB | 119.5 GiB | 2.282 | reference | |

The IQ3_S, IQ3_XXS and Q2_0 packs carry an MTP layer fine-tuned on the model's own output (self-distillation), which
raises draft acceptance. Output is unchanged, because verify decides every token. Q4_K_XL and Q8_0 keep the original
MTP layer.

The model files are derived from Qwen3.8-Flash-Next and are distributed under the
[Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE), not Basalt's MIT
license. The GSQ-RCO quants come from IST Austria's Deep Algorithms and Systems Lab; please cite the
[GSQ](https://arxiv.org/abs/2604.18556) and [RCO](https://arxiv.org/abs/2605.00649) papers if you use them.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT, see [LICENSE](LICENSE).
