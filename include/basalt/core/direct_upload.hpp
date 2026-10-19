// include/basalt/core/direct_upload.hpp - file ranges to the GPU through unbuffered reads.
//
// A load phase that copies a range of the model file to VRAM reads it with block-aligned O_DIRECT reads into pinned
// staging buffers, so the file cache neither copies the bytes nor evicts anything, and the read of one chunk overlaps
// the copy of the one before.  A file system without unbuffered reads falls back to ordinary reads into the same
// buffers.  One uploader belongs to the device that is current when it is made.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace basalt::core {

class DirectUploader {
public:
    static constexpr size_t kChunk = 16u << 20;   ///< the most `staged` returns at once

    DirectUploader();
    ~DirectUploader();
    DirectUploader(const DirectUploader&) = delete;
    DirectUploader& operator=(const DirectUploader&) = delete;

    /// `bytes` of `path` from `file_off`, copied to device memory `dst`; complete when this returns.
    bool upload(const std::string& path, uint64_t file_off, uint64_t bytes, void* dst, std::string& err);

    /// `bytes` (at most kChunk) of `path` from `file_off` in pinned memory the uploader owns; valid until the next call.
    const uint8_t* staged(const std::string& path, uint64_t file_off, uint64_t bytes, std::string& err);

    /// Whether the last file opened could be read unbuffered.
    bool unbuffered() const;

private:
    struct Impl;
    Impl* impl_;
};

}  // namespace basalt::core
