// Picks the AVX-512 or the AVX2 build of the extracted ggml dot products for this CPU (see ggml_cpu_dots.c).
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/cpu/expert_layout.hpp"

extern "C" {
const ggml_type_traits_cpu* ggml_get_type_traits_cpu_avx2(ggml_type type);
const ggml_type_traits_cpu* ggml_get_type_traits_cpu_avx512(ggml_type type);

const ggml_type_traits_cpu* ggml_get_type_traits_cpu(ggml_type type) {
    namespace cpu = basalt::kernels::cpu;
    static const bool avx512 = cpu::cpu_avx512_ok();
    static const bool avx2 = cpu::cpu_avx2_ok();
    if (avx512) return ggml_get_type_traits_cpu_avx512(type);
    if (avx2) return ggml_get_type_traits_cpu_avx2(type);
    return nullptr;
}
}
