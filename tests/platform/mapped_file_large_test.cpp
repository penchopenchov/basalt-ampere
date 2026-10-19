// tests/platform/mapped_file_large_test.cpp - a section of a sparse file past 4 GiB lands on the right bytes (a
// 64-bit view offset). CPU only. CMake registers it off Windows only: there the file system writes the hole out as
// zeros. Where a file this size cannot be made the case reports that and passes.
#include "basalt/platform/file.hpp"
#include "basalt/platform/mapped_file.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("mapped_file_large %-48s -> %s\n", what.c_str(), ok ? "ok" : "FAILED   <-- WRONG");
    failures |= !ok;
}
}  // namespace

int main() {
    using basalt::platform::MapOptions;
    using basalt::platform::MappedFile;
    std::string err;
    // A sparse file past 4 GiB: a section at a 64-bit offset must land on the right bytes.
    {
        const std::string big = basalt::platform::temp_dir() + "/basalt_mapped_file_large_test.bin";
        const uint64_t at = (4ull << 30) + 8192;
        std::FILE* f = basalt::platform::open_file_utf8(big, "wb");
        if (f && basalt::platform::fseek64(f, (int64_t) at, SEEK_SET) == 0) {
            const uint8_t mark[4] = {0xde, 0xad, 0xbe, 0xef};
            std::fwrite(mark, 1, 4, f);
            std::fclose(f);
            MapOptions o;
            o.offset = 4ull << 30;
            o.bytes = 8196;
            auto m = MappedFile::open_ro(big, &err, o);
            check(m.has_value() && m->file_size() == at + 4 && std::memcmp(m->data() + 8192, mark, 4) == 0,
                  "section at a 64-bit offset");
        } else {
            if (f) std::fclose(f);
            std::printf("mapped_file_large: no sparse file here, 64-bit offset case skipped\n");
        }
        std::remove(big.c_str());
    }
    return failures ? 1 : 0;
}
