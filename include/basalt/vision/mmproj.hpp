// basalt/vision/mmproj.hpp - the vision projector file (GGUF, architecture "clip"): hyper-parameters and tensors.
//
// Opened read-only and mapped; tensor data stays in the mapping.  Only the layout the Qwen3-VL style encoder uses
// is checked, and a file that does not match is refused with the reason.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace basalt::vision {

struct TensorRef {
    std::string name;
    uint32_t type = 0;               ///< ggml type id: 0 F32, 1 F16, 30 BF16
    std::vector<int64_t> ne;         ///< ggml order: ne[0] is the contiguous axis
    const void* data = nullptr;
    size_t bytes = 0;
    int64_t elems() const { int64_t n = 1; for (int64_t d : ne) n *= d; return n; }
};

struct Hparams {
    int image_size = 0, patch = 0, embd = 0, ffn = 0, layers = 0, heads = 0, merge = 0, proj = 0;
    float eps = 0.f;
    float mean[3] = {0, 0, 0}, std[3] = {1, 1, 1};
};

class Mmproj {
public:
    Mmproj() = default;
    ~Mmproj();
    Mmproj(const Mmproj&) = delete;
    Mmproj& operator=(const Mmproj&) = delete;

    /// Maps `path`, parses the header and checks it is the expected projector.  False with `err` otherwise.
    bool open(const std::string& path, std::string& err);

    const Hparams& hp() const { return hp_; }
    /// A tensor by name; null when absent.
    const TensorRef* find(const std::string& name) const;
    /// A required tensor of the given type and element count (ne product); null with `err` set otherwise.
    const TensorRef* need(const std::string& name, uint32_t type, int64_t elems, std::string& err) const;

private:
    void* map_ = nullptr;
    size_t size_ = 0;
    Hparams hp_;
    std::map<std::string, TensorRef> tensors_;
};

}  // namespace basalt::vision
