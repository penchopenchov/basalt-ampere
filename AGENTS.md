# AGENTS.md

basalt is a local inference engine for the Qwen3.8-Flash-Next mixture-of-experts model, built only for Linux and NVIDIA Blackwell (sm_120). It runs on one RTX 5090, with a second card as a peer expert tier and system RAM holding the rest. It is a fork of Strata (MIT): the C++/CUDA engine is in `src/` and `include/`, and the Python server with an OpenAI- and Anthropic-compatible API is in `serve/`.

## Working rules

`CONTRIBUTING.md` sets the project scope (CUDA + Blackwell only) and what every change must include (parity test, profile, quality check). It applies to agents too.

- **Commits:** small and logical, message style `area: what it does`. Never amend and never force-push; always make new commits. End each message with the co-author line your tool's instructions specify.
- **Code comments:** brief and stateless. Say what the code does and why. No hardware names, measured numbers, dates or history in comments.
- **No notes in the repo.** No markdown notes, scratch files or handoffs get committed.
- **Don't push or publish anywhere** (GitHub, HF, any remote) without the maintainer's approval. The `upstream` remote has push disabled on purpose.
- **Never expose a server beyond 127.0.0.1 without `--api-key`.**
- **Don't write tools nobody asked for.** The maintainer has their own bench and harness scripts.
- **Headline benchmark numbers come from the maintainer's own runs,** at 400 W and 600 W. Agent benches are development A/Bs only. Label every number with its GPU power limit; the daily limit is 400 W.
- **Quality gate:** PPL/KLD against the Q8_0 reference, through the decode path (`ppl.sh` on the test machine). Byte identity with old output is not required, but say whenever numerics change. Bitwise-neutral refactors need only the greedy-identity check.
- **Fast iteration:** quick checks, go/no-go. If something isn't working, write down why and move on.
- **No permission prompts while the maintainer is away** (ssh to other hosts, docker, system changes). A prompt blocks the whole session.
- **No drafting/speculation work** in the performance effort; the focus is the per-pass decode cost.
- **basalt's own kernels are the reference.** Parity tests compare a changed kernel against the current one and against a plain CPU reference; Strata's original kernels are not kept as oracles. Every new kernel gets a parity test (random inputs + real geometry) and a microbench line.

## The test machine (everything builds and runs there; the dev laptop has no CUDA)

- Reach it with `ssh user@test-host` (Ubuntu, CUDA at /usr/local/cuda). sudo is allowed only for `nvidia-smi -i 0 -pl 400|600` and drop_caches.
- **The maintainer's own server may be running on :8080** (`~/scripts/serve_test.sh`, holding the GPU and build locks). Never stop or kill it. Ask before taking the machine.
- Every GPU use goes through `~/scripts/gpu.sh`, or the scripts that take `~/scripts/gpu.lock`. Builds go through `~/scripts/bld.sh` (`~/scripts/build.lock`).
- Never kill processes you didn't start, and never kill `e2e.sh` mid-run: its server keeps both locks.
- Don't `pgrep -f "pattern"` inside an ssh/bash -c command; it matches its own command line. Use bracket patterns like `"[e]2e.sh"` or DONE markers.
- Scratch goes under `~/scratch/basalt-*` only. Don't edit `~/worktrees/rel` or the `~/scripts` scripts; copy them if you need variants.
