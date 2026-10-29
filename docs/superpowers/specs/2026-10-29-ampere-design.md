# RTX 3090 Ampere port

## Intent
Fork Basalt as basalt-ampere, install a native Windows build in this checkout,
and optimize its existing inference paths for a single RTX 3090 (sm_86, 24 GiB).

## Approach
Use a native CUDA port. Keep the server, model format, memory planner, CPU expert
tier and platform boundaries. Compile device code for sm_86, replace unsupported
instructions with Ampere implementations, and retain existing int8/FP16/BF16
Tensor Core kernels where supported. Specializations that require TMA or native
FP4 must be replaced or bypassed by working, tested paths.

A build-flag-only change cannot work because the source emits unsupported PTX.
Replacing Basalt with another inference engine would abandon the requested fork.

## Validation
Build the engine and tests. Run CPU tests and synthetic GPU parity tests on this
RTX 3090. Benchmark relevant kernels at the existing 350 W power limit; do not
change GPU clocks, power limits, Windows registry or driver settings. Prefer
measured improvements. Never describe unmeasured performance as a speedup.

## Constraints and open validation
This computer has approximately 48 GiB RAM. Full-model feasibility depends on
pack size and free RAM. The user approved downloading the upstream IQ3_S pack (86,932,948,345 bytes).
End-to-end text inference is verified with this pack. Greedy identity against
the upstream hardware path, perplexity and KLD remain unverified; the latter
two require a Q8_0 reference.

Native NVFP4 Tensor Core operations are unavailable on this GPU. Quantized K/IQ
packs are the preferred port target; NVFP4 needs a tested software implementation
or an explicit unsupported-format error, never a silent no-op.

## Builder decisions
Use the existing C++20/CUDA implementation and a local Python virtual environment.
Install the missing supported C++ build tools if no existing compiler is found.
Keep upstream attribution and licenses. Changes belong only in the new fork.

## Feasibility review
The initial port must disable dense Tensor Core predicates that currently return
true although their sm_86 kernel bodies are empty. Route these products to the
existing native MMVQ implementation and test that outputs are actually written.
Guard TMA-dependent device code at compile time, retaining the existing unfused
hyper-connection and middle-chain paths. Remove the global, MMQ and runtime
architecture refusals and the unconditional sm_120a FP4 build target.

NVFP4 software decode and MMQ paths already exist. Prefer those over rejecting
the format if their parity tests pass. Gate native FP4 prefill by device
capability and update buffer sizing together with dispatch so larger token
batches cannot select a layout sized for a different implementation. Test both
sides of the prefill token threshold when that format is enabled.

Preserve existing Ampere PDL and top-k fallbacks. Update the fork's scope docs to
state the requested architecture exception. Use sentinel outputs in fallback
dispatch tests to detect accidental no-op kernels. Keep performance claims
limited to the configurations actually measured.
