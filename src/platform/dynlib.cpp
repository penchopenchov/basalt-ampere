// src/platform/dynlib.cpp - see include/basalt/platform/dynlib.hpp.
#include "basalt/platform/dynlib.hpp"

#if defined(_WIN32)
#include "basalt/platform/win.hpp"
#include <string>
#else
#include <dlfcn.h>
#endif

namespace basalt::platform {

#if defined(_WIN32)
namespace {
/// A driver that keeps nvml.dll outside System32: its NVSMI directory under one of the program-file roots. The
/// root comes from the environment as UTF-16, so the path never goes through the ANSI code page. The DLL's own
/// dependencies are looked up beside it.
HMODULE load_nvsmi() {
    static const wchar_t* const roots[] = { L"ProgramFiles", L"ProgramFiles(x86)" };
    wchar_t dir[4096];
    for (const wchar_t* root : roots) {
        const DWORD n = GetEnvironmentVariableW(root, dir, (DWORD) (sizeof dir / sizeof dir[0]));
        if (n == 0 || n >= sizeof dir / sizeof dir[0]) continue;
        const std::wstring path = std::wstring(dir, n) + L"\\NVIDIA Corporation\\NVSMI\\nvml.dll";
        const HMODULE m = LoadLibraryExW(path.c_str(), nullptr,
                                         LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (m != nullptr) return m;
    }
    return nullptr;
}
}  // namespace

DynLib DynLib::open_nvml() {
    // A bare name is looked up in System32 only: the application and current directories are not searched, so
    // an nvml.dll lying next to the executable cannot stand in for the driver's.
    HMODULE m = LoadLibraryExW(L"nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (m == nullptr) m = load_nvsmi();
    DynLib lib;
    lib.h_ = (void*) m;
    return lib;
}

void* DynLib::address(const char* name) const {
    return h_ == nullptr ? nullptr : (void*) GetProcAddress((HMODULE) h_, name);
}

void DynLib::reset() {
    // The library stays loaded for the life of the process: the function pointers handed out stay valid, and the
    // management library keeps threads of its own that must not outlive their code.
    h_ = nullptr;
}
#else
DynLib DynLib::open_nvml() {
    DynLib lib;
    lib.h_ = dlopen("libnvidia-ml.so.1", RTLD_NOW);
    return lib;
}

void* DynLib::address(const char* name) const { return h_ == nullptr ? nullptr : dlsym(h_, name); }

void DynLib::reset() { h_ = nullptr; }   // never dlclosed, as before: the management library stays loaded
#endif

}  // namespace basalt::platform
