# Context and warm starts

The supplied configuration serves IQ3_XXS at **262144 tokens**, including the
prompt and reply. This is the context declared by the model file; no RoPE
extension is requested. The weight format stays IQ3_XXS and KV stays int8.

## Memory settings

The configuration uses:

```json
{
  "max-context": 262144,
  "kv": "int8",
  "resident-budget-gib": 27.5,
  "ram-guard-gib": 1.5,
  "expert-profile-save": "models/profiles/qwen3.8-flash-next-iq3xxs-ampere.profile",
  "expert-profile-save-every": 1
}
```

The elastic KV cache starts around 0.7 GiB and can grow to 4.0 GiB. It reclaims
VRAM from expert slots as the actual request grows. A short request therefore
does not pay the full context's KV allocation.

On a 48 GiB system, automatic RAM top-up can leave too little headroom for a
near-256K prompt. The explicit 27.5 GiB upper bound limits the RAM expert cache;
its allocation can be lower. The tested start holds all primary expert weights
across GPU and RAM, with about 23.4 GiB allocated to the resident RAM arena.
The serve-time RAM guard is 1.5 GiB (1536 MiB). The automatic load-time reserve
and the 1 GiB emergency floor remain in place. Other programs still need their
own memory. A lower guard allows more serving growth before requests stop;
it does not enlarge the expert cache. The development checks below used the
previous automatic guard.

## Expert placement and the speed ramp

The model file includes an expert ranking used at startup. During prompt reads,
the engine counts expert routing and uses those counts to seed decode-time
adaptation. The adaptive tier continues exchanging experts while decoding.

The saved profile carries the learned expert placement and ranking across
launches. It is written between requests at one-minute intervals and on clean
shutdown. A missing profile uses the model's built-in ranking. Keep the file
local: it describes a workload and belongs with that model's runtime data.
A profile learned from one topic can slow a different topic's first request.
Create `models/profiles` before the first launch: `mkdir -p models/profiles`
on Linux, or `New-Item -ItemType Directory -Force models/profiles` in PowerShell.

The SSD-backed per-layer embedding (PLE) row cache also warms during use.
In a development 128K run on RTX 3090 at **350 W**, the first conversation's
expert-cache hit rate was 96.3%, yet PLE reads accounted for 7.9 seconds of
blocked time. Remembering expert placement does not persist the PLE row cache.
Keeping the server running retains both caches; unseen embedding rows can still
require SSD reads. Draft acceptance also affects generation throughput.

## Development checks

RTX 3090 at **350 W**, same model, int8 KV and experimental staging. The 128K
arm uses automatic RAM sizing and the built-in profile; the 256K arm uses the
bounded RAM cache and a learned profile. Each arm runs conversation, structured
output and prose twice in that order. Rates below are total generated tokens
divided by total decode time across the two requests of each type.

| Workload | 128K configuration | 256K configuration |
| --- | ---: | ---: |
| Conversation | 77.4 tok/s | 83.5 tok/s |
| Structured output | 187.4 tok/s | 191.6 tok/s |
| Prose | 79.6 tok/s | 77.5 tok/s |

These are whole-configuration development A/Bs, not headline benchmark results
or isolated speedups from one setting. Adaptive placement changes rounding and
can change sampled continuations; SSD latency and cache warmth vary. Individual
warm requests also vary, so these averages do not guarantee unchanged throughput.

Near the expanded limit, the 256K configuration passes these checks at **350 W**:

- A 259850-token synthetic archive returns four exact labelled keys distributed
  through the prompt. Prompt processing takes 164.8 seconds.
- A 259820-token synthetic prompt generates 512 prose tokens at **49.8 tok/s**,
  after 103.7 seconds of prompt processing. The larger context has a real cost:
  a 129820-token prompt with automatic RAM sizing runs at 81.0 tok/s in a
  separate 256K development trial.
- Overflow is rejected without truncation; a following short request succeeds.
- The deployed HTTP server, with normal prompt checkpoints, also passes a
  259614-token archive with four-key recall and 512 output tokens at 50.8 tok/s
  after 164.6 seconds of prompt processing, followed by a successful short reply.
- A learned expert profile is saved and loaded across engine starts.
- Existing context/config tests pass (41 tests, one skipped), and the CPU
  expert-profile save/load test passes.

The long fixtures repeat synthetic filler, with prompt checkpoint reuse
disabled. They establish capacity and basic retrieval, not general long-context
accuracy or Q8_0 equivalence. The normal server retains its prompt checkpoints.
Full Q8_0 PPL/KLD for experimental staging remains unvalidated.

## Rollback

Restore the backed-up configuration and matching client context limit to return
to 131072 tokens. Without a backup, set `max-context` to 131072 and remove the
explicit resident budget and profile-save settings. Retain int8 KV and the
existing staging settings for the previous serving mode. Client model limits
must match the running server's context.
