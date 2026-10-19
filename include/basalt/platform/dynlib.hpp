// include/basalt/platform/dynlib.hpp - a shared library loaded at run time, for the driver libraries the engine
// reads optionally (the GPU management library): a machine without one still runs, and the caller says so.
#pragma once

namespace basalt::platform {

class DynLib {
public:
    DynLib() = default;
    DynLib(DynLib&& o) noexcept : h_(o.h_) { o.h_ = nullptr; }
    DynLib& operator=(DynLib&& o) noexcept {
        if (this != &o) { reset(); h_ = o.h_; o.h_ = nullptr; }
        return *this;
    }
    DynLib(const DynLib&) = delete;
    DynLib& operator=(const DynLib&) = delete;
    ~DynLib() { reset(); }

    /// The NVIDIA management library (NVML) of the installed driver; empty when there is none. Where the loader
    /// searches is the platform's own, and never the application's directory, so a library beside the executable
    /// cannot stand in for the driver's.
    static DynLib open_nvml();

    explicit operator bool() const { return h_ != nullptr; }

    /// The function `name`, or null when the library has no such symbol.
    template <class F>
    F sym(const char* name) const { return reinterpret_cast<F>(address(name)); }

    void reset();

private:
    void* address(const char* name) const;
    void* h_ = nullptr;
};

}  // namespace basalt::platform
