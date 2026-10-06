<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/basalt-lockup-dark.png">
    <img src="assets/basalt-lockup-light.png" alt="basalt" width="360">
  </picture>
</p>

basalt is a local inference engine for the **Qwen3.8-Flash-Next** mixture-of-experts model. It targets one thing:
Linux with an NVIDIA Blackwell GPU (sm_120). The experts that do not fit in VRAM live in system RAM (and optionally on
a second GPU used as a peer expert tier), so a large MoE model runs on a single workstation.

It is a fork of [Strata](https://github.com/Niko1221/Strata) (MIT), stripped down to that one target. See
[THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).

## What is in the repository

- `src/`, `include/`: the C++/CUDA engine (one binary, `basalt`).
- `serve/`: a Python server with an OpenAI- and Anthropic-compatible API, and its web app.
- `tools/`: pack builders and converters (`iq_pack.py`, `mtp_pack.py`, ...) and the PPL/KLD quality gate (`ppl_gate.py`).
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
cmake --build build -j
ctest --test-dir build        # the parity tests need a GPU
```

## Run

Build a pack from the model with the tools in `tools/`, then start the server:

```sh
python -m serve.server --engine engine --config config.json --port 8095
```

`config.json` names the engine executable and its arguments. The server listens on 127.0.0.1 by default; do not expose
it beyond that without `--api-key`.

Diagnostics are engine command-line flags (`--profile`, `--wait-stats`, `--decode-timing`, `--draft-log FILE`); no
environment variables control the engine.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT, see [LICENSE](LICENSE).
