// strata/core/emulate.hpp - compute-capability and shared-memory queries as the device reports them.
#pragma once

namespace strata {

inline constexpr int emulated_cc() { return 0; }
inline constexpr int cc_major_of(int real_major) { return real_major; }
inline constexpr int cc_minor_of(int real_minor) { return real_minor; }
inline constexpr int smem_optin_of(int real) { return real; }

}  // namespace strata
