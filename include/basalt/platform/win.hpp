// include/basalt/platform/win.hpp - the one place <windows.h> is included.
//
// Windows only, and for the platform layer's own sources: engine code never includes it. NOMINMAX keeps the min and
// max macros from breaking std::min and std::max, WIN32_LEAN_AND_MEAN leaves out the rarely used API families.
#pragma once

#if !defined(_WIN32)
#error "basalt/platform/win.hpp is Windows-only"
#endif

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
