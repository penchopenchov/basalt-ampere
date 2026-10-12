// src/kernels/fused_moe_parity.cu - the fused layer's experts half (fused_moe.cuh) against today's launch sequence
// (dec_moe_glue / route + native_expert_grouped, shared_expert_multi, dec_moe_wait, dec_moe_combine), bitwise on the
// layer output, T = 1..4. Both instances run: k_moe (moe_body<false>, records in one launch) and moe_body<true>, the
// one k_layer runs after the attention half (one record per launch, the chain's ready count already full).
// Synthetic weights and routing (CUDA0 and CPU tiers; MB_NOPEER), every shared-expert format triple of the IQ3_S pack
// with its layer's expert formats. The activations sit on the q8_1 rounding edge (MB_EDGE), where IEEE division and
// the fast-math native quantize round apart, so a q8_1 row quantized the wrong way changes the output. The routed
// experts read the hc read's IEEE q8_1 rows as in the engine, and, as a second pass, the native q8_1 rows.
#include "fused_bound.cuh"
#include "fused_moe.cuh"

#include <cstdlib>

int main(int argc, char** argv) {
    int dev = 0, cc = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&cc, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess || cc < 12) {
        std::printf("fused_moe_parity: no sm_120 device, skipped\n");
        return 77;
    }
    // {layer, gate/up type, down type, shared gate, shared up, shared down}
    static const int sets[] = {
        0, 18, 20, 12, 23, 20,   1, 22, 42, 23, 12, 20,   2, 22, 20, 12, 14, 20,   4, 22, 20, 14, 13, 20,
        5, 22, 20, 23, 14, 20,   6, 22, 20, 14, 12, 20,   7, 18, 20, 23, 23, 20,  10, 22, 42, 12, 12, 20,
        12, 18, 42, 14, 14, 20,  15, 22, 20, 14, 23, 20,  18, 18, 20, 13, 13, 20,  21, 21, 42, 23, 12, 20,
        24, 22, 20, 23, 13, 20,  29, 21, 20, 13, 23, 20,  34, 22, 20, 12, 13, 20,  47, 23, 20, 12, 13,  8,
    };
    const int n = (int) (sizeof sets / sizeof sets[0] / 6);
    const int iters = argc > 1 ? std::atoi(argv[1]) : 2 * n;
    setenv("MB_REPS", "0", 1);
    setenv("MB_NOPEER", "1", 1);   // see MB_NOPEER in fused_moe.cuh
    setenv("MB_EDGE", "1", 1);
    long long bad = 0;
    for (int fused = 0; fused <= 1; ++fused) {
        if (fused) setenv("MB_FUSED", "1", 1); else unsetenv("MB_FUSED");
        for (int ieee = 1; ieee >= 0; --ieee) {
            if (ieee) setenv("MB_IEEE", "1", 1); else unsetenv("MB_IEEE");
            for (int T = 1; T <= 4; ++T) {
                char t[8], it[16];
                std::snprintf(t, sizeof t, "%d", T); std::snprintf(it, sizeof it, "%d", iters);
                char* av[] = {argv[0], t, it, nullptr};
                if (mkb4_moe_main(3, av, sets, n) != 0) return 1;
                std::printf("fused_moe_parity: %s T%d %s q8_1: %lld floats differ\n", fused ? "moe_body<true>" : "k_moe", T,
                            ieee ? "IEEE" : "native", mkb4_bad);
                bad += mkb4_bad;
            }
        }
    }
    std::printf("fused_moe_parity: %s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
