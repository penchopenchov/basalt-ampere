# Building and testing

Clone, requirements and the build commands are in the [README](../README.md). This page has the rest.

## What is in the repository

- `src/`, `include/`: the C++/CUDA engine (one binary, `basalt`).
- `serve/`: a Python server with an OpenAI- and Anthropic-compatible API, and a stats page.
- `tools/convert/`: the model-file converter (`convert.py`), its section builders and the drafter fetcher (`mtp_fetch.py`).
- `tools/`: the .basalt reader and writer (`model_file.py`), the PPL/KLD quality gate (`ppl_gate.py`), the greedy-identity
  check and the tokenizer.
- `tests/`: parity and unit tests.

## Tests

The tests are off by default. Configure the README's build with the extra option `-DBUILD_TESTING=ON`, build, then run `ctest --test-dir build`.

The kernel parity tests need the GPU; the file-format, cache and memory unit tests do not. `expert_parity` and `pool_test`
run on a synthetic expert fixture; pass `--file experts.bin` (the pack's experts section, not shipped) to check real blobs.
`platform_memory_test` locks 256 MiB when the locked-memory limit (`ulimit -l`) allows it and a smaller region otherwise,
and says so; it skips the lock case only when the limit is too low to lock anything.

`basalt-bench` reads its pre-tokenized prompts from `data/bench` (token ids for the Qwen3.8-Flash-Next tokenizer: the
three decode prompts and a prefill filler); run it from the repository root or pass `--fixtures DIR`. `basalt-bench
--help` says how the files are made.

The build needs nothing outside this repository and fetches nothing. The parts of ggml and llama.cpp that Basalt uses
(quantization formats, CPU dot products, the prompt path's MMQ kernels) are extracted into `src/` and `include/basalt/ggml/`,
with their MIT licence in `THIRD_PARTY_LICENSES`.
