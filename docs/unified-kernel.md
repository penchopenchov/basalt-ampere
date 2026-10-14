# Unified kernel (parked)

This branch holds the unified kernel: one persistent CUDA kernel per decoder layer in place of the per-step kernel sequence. It is parked, not merged. On the same box and session it decodes slower than main, so main keeps the separate kernels. The design is kept here for a later attempt.

## What it is

Decode runs a window of T tokens per verify pass (T = 1 to 4 are covered). For each layer the engine normally launches a sequence of kernels: hc1, the attention projections, the recurrence, the output projection, hc2 + router + top-10, expert plan and gate/up, SwiGLU, down, the shared expert, the peer and CPU wait, and the combine. The unified kernel runs the same steps inside one launch, with cross-CTA hand-offs by release/acquire counters instead of kernel boundaries.

| Piece | Source | What it replaces |
|---|---|---|
| Attention chain | `attn_chain.cu`, `attn_chain_kernel.cuh`, `attn_chain_bound.cuh` | hc1, qkv+z pair, recurrence, output projection, hc2 + router + top-10 of a GDN layer |
| Experts half | `fused_moe.cuh` | expert plan, gate/up + SwiGLU, down, shared expert, peer and CPU wait, combine; it also publishes the tier ring and waits for the tiers itself |
| Fused GDN layer | `fused_layer.cu`, `fused_chain.cuh`, `fused_bound.cuh` | the whole GDN layer: `k_layer<T>` runs the chain, then the experts half, in one launch |
| QSA attention half | `qsa_full.cu`, `qsa_full_*.cuh` | projections, K/Q post, indexer scores, top-k, attention, gate and output projection of a QSA layer, as one persistent kernel |

Arithmetic and reduction order are the same as the kernels they replace, so outputs are bitwise identical to the separate-kernel path (see `docs/precision.md`). Windows with T >= 5, QSA layers (both halves) and layers whose gate/up format is not covered keep the separate kernels. The QSA attention half (`qsa_full`) is kept in the tree and its parity test still runs, but the engine does not launch it: it was slower than the separate kernels.

Build structure:

- `attn_chain.cu` and `qsa_full.cu` are translation units built without fast math, like the stage code they include.
- `fused_layer_t2.cu` .. `fused_layer_t4.cu` and `fused_layer.cu` are one translation unit per window width, so changing one width rebuilds one unit. The host side stays in `attn_chain.cu` and is reached through `chain_shared.hpp`.
- `BASALT_PROFILE` (CMake option, off by default) compiles in-kernel phase stamps (`stamp_ring.*`, `BASALT_STAMP_FILE` at run time). Without it the stamps compile to nothing.

## Tests

- `attn_chain_parity`: the attention chain against the kernels it replaces, bit for bit, plus window-width independence of a token's output.
- `qsa_full_parity`: the QSA half against the kernel sequence, bitwise, and the attention against a double-precision CPU reference.
- `fused_moe_parity`: the experts half against the launch sequence it replaces, bitwise, T = 1..4, every IQ3_S shared-expert format triple.

Greedy output is identical to main end to end (conversation, structured and prose prompts of 400 tokens, and 16k to 36k token prompts).

## Results

Setup for every number: IQ3_S pack, `--peer-exclusive`, prose fixture, speculation 7, RTX 5090 at 400 W. Numbers are from agent development A/Bs, not headline runs.

Best verified unified build against main, same session, interleaved:

| build | window | tokens/s | passes/s |
|---|---|---|---|
| main | 6.675 ms | 324.6 | 149.8 |
| unified kernel (this branch) | 7.541 ms | 288.2 | 132.6 |

That is +13.0% per window and -11.2% tokens/s against main. An earlier unified build, first measured against main in the same way, was 278 against 319 tokens/s.

Where it wins and loses, per step, at T = 3 (stamp basis, microseconds per layer, scaled to the plain-run basis):

| step | main | unified | difference |
|---|---|---|---|
| GDN recurrence | 8.5 | 2.7 | -5.8 (hidden under the projection pair) |
| GDN output projection | 8.6 | 6.7 | -1.9 |
| GDN SwiGLU, down, tier wait, shared expert, combine | 23.9 | 13.3 | -10.6 |
| GDN hc1 | 10.4 | 12.0 | +1.6 |
| GDN qkv+z pair | 18.6 | 24.0 | +5.4 |
| GDN plan + gate/up (to the end of gate/up) | 20.7 | 42.1 | +21.4 |
| GDN layer total | 103.6 | 114.9 | +11.3 |
| QSA projections | 17.8 | 31.2 | +13.4 |
| QSA output projection | 5.9 | 16.8 | +11.0 |
| QSA hc2 + router + top-10 | 12.6 | 18.5 | +5.9 |
| QSA layer total | 109.1 | 144.3 | +35.2 |

The GDN attention half is at parity with main in the engine. The loss is in the experts half (about 11 us per GDN layer) and the QSA attention half (about 35 us per QSA layer). Per window that is roughly +0.4 ms from the GDN experts half and +0.4 ms from the QSA half. Head, draft, commit and host time are unchanged.

The best build sits at about 54% of the window's byte floor against 61% for main.

## Fix attempts on the top losses

A second pass on the three largest losses against main (expert gate/up, the projections, the output projection, first hc). Each idea was measured with in-kernel stamps in the engine (PROFILE build, conversation fixture, ABAB in one session), then the survivors in a plain-build A/B (4 fresh processes per arm, ABBA, IQ3_S, `--peer-exclusive`, RTX 5090 at 400 W). Development A/Bs, not headline runs.

| idea | engine stamps (us per layer) | plain A/B | kept |
|---|---|---|---|
| all of a CTA's gate/up rows into L2 at the release (no pacing) | gate/up +3, down -1.2 | - | no |
| line prefetches instead of bulk L2 prefetches for gate/up | 0 | - | no |
| no gate/up or down-slice L2 prefetch in the experts half | gate/up -3.2, down +0.8 | with the next row: conv -0.65% per window, verify T2 -0.3%, T3 +0.3% (noise) | no |
| projection pair tiles into L2 at kernel entry (evict-first) | pair -6.2 (to its byte floor), hc1 +3, combine +0.6 | see above | no |
| QSA projection and output weights into L2 at kernel entry | projections +2 to +3.4, output projection -2.5 | - | no |
| QSA layers back on the separate kernels | - | conv -6.0%, prose -5.0%, structured -7.4% per window against the unified build (measured with the two prefetch rows above, which are neutral) | yes |

- Stamps over-predicted again: the prefetch pair summed to about -4.7 us per GDN layer (-1.7% per window); the plain A/B shows no change.
- Two CTAs per SM do not fit: the attention half also uses the whole 99 KB arena, and 512 threads at 128 registers fill the register file.
- Balancing gate/up items across CTAs: the CTAs with one item more finish about 6 us later at T = 3, which bounds the gain at about 1% per window. Not built.
- The fused layer kernel is about 1.5 MB of SASS and the QSA kernel about 0.55 MB. With cold caches, 18% of the experts half's warp stalls are instruction fetch.

With QSA on the separate kernels the branch is still about 6% slower than main per conversation window and 8% per prose window; structured windows (mostly T >= 5) are level. Windows T = 1 to 4, where the fused GDN layer runs, cost 10 to 12% more than main; T = 5 to 7, which run no fused kernel, cost about 1% more (no tag protocol).

## Why it is parked

- The standalone harness predicted gains the engine did not show. Auditing twelve claims, the harness-to-engine ratio was typically 0.1 to 0.3. Causes: the harness starts every CTA with the top-10 experts already known, while in the engine they arrive about 7 us after the attention output; the shared expert finishes last in the engine; window arithmetic that ignored the window-width mix (T = 5 to 7 and T = 1 gain nothing); and measurement errors (wrong lock, builds running, per-launch memsets in a hot loop).
- The experts half is bound by cross-CTA dependency latency (gate/up round 2, then the h rows published, then down), not by DRAM bandwidth. At T = 3 it spends about 57.6 us against a roughly 30 us byte floor.
- The QSA half serializes projections that main overlaps with side kernels, adds grid barriers, and runs the output projection after the last barrier on fewer CTAs than main's overlapped launch.
- Every remaining down-phase lever measured at 1% or less per window. The gap to main (13%) is larger than all of them together. Closing it needs a design decision, not more tuning of this layout.

## Lessons

- Check register spills on the real engine binary (`cuobjdump --dump-resource-usage`), one change at a time. `nvcc --split-compile` on the kernel units made the layer kernel spill about four times more (stack 576 B against 136 B) and cost about 9% tokens/s; the extra spills were first blamed on the arch target and instance count. Never use it on these units. Per-window-width translation units give fast rebuilds with identical SASS.
- A kernel that does not fit its dynamic shared memory silently falls back; confirm the fused path actually runs before timing it.
- Time with the timing lock, with no builds running, and no per-launch memsets in hot loops. Compare stamp numbers only within one session, ABAB. The same binary read 3.6 us per layer apart across two sessions, so single-microsecond effects do not resolve in the engine A/B.
- Convert per-layer gains to window gains with the window-width mix (about 0.73 x 36 per layer-microsecond), not x36.
- A harness must reproduce the engine's start state (late ids, tier wait, shared-expert tail) before its deltas predict anything.
- Holding the down weights in registers before the h wait cost 80 B of stack on a 128-register kernel and lost 1.6%; prefetching them to L1 instead added issue work and also lost.

## Ideas not tried

- Down without the shared-memory h copy: read the late groups' h straight from L2 (tagged layout). Estimate 1 to 2 us per layer on the late groups.
- Give the round-2 gate/up items to CTAs with fewer down rows.
- Per-triple `k_layer` instances (the engine builds the generic one only), and chain-local half-packed fragments to cut spills in the projection pair.
- A layer-boundary kernel running several layers per launch (spills made it slower in the harness).
