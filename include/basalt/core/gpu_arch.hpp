#pragma once

namespace basalt::core {

constexpr bool supported_compute_capability(int major, int minor) {
    return (major == 8 && minor == 6) || (major == 12 && minor == 0);
}

}  // namespace basalt::core
