// Selftest for the optional MAP_SHARED backing of PinnedArena.
#include "basalt/core/pinned.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <unistd.h>

int main() {
    char path[] = "/tmp/basalt-shared-arena-XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0) {
        std::perror("mkstemp");
        return 2;
    }
    close(fd);

    int fail = 0;
    const uint64_t bytes = 64u << 10;
    const uint64_t arena_hash = 0x0123456789abcdefull;
    const std::vector<uint64_t> no_bounds;
    {
        basalt::core::PinnedArena first(bytes, no_bounds, 0, path, arena_hash);
        if (!first.valid()) {
            std::fprintf(stderr, "first shared arena failed: %s\n", first.note.c_str());
            fail = 1;
        } else {
            first.data()[123] = 0x5a;

            basalt::core::PinnedArena second(bytes, no_bounds, 0, path, arena_hash);
            if (!second.valid()) {
                std::fprintf(stderr, "second shared arena failed: %s\n", second.note.c_str());
                fail = 1;
            } else if (second.data()[123] != 0x5a) {
                std::fprintf(stderr, "shared mapping did not expose the same bytes\n");
                fail = 1;
            }

            basalt::core::PinnedArena wrong_hash(bytes, no_bounds, 0, path, arena_hash + 1);
            if (wrong_hash.valid() || wrong_hash.note.find("model hash") == std::string::npos) {
                std::fprintf(stderr, "hash mismatch was not refused: %s\n", wrong_hash.note.c_str());
                fail = 1;
            }

            basalt::core::PinnedArena wrong_size(bytes * 2, no_bounds, 0, path, arena_hash);
            if (wrong_size.valid() || wrong_size.note.find("expected") == std::string::npos) {
                std::fprintf(stderr, "size mismatch was not refused: %s\n", wrong_size.note.c_str());
                fail = 1;
            }
        }
    }

    if (unlink(path) != 0) {
        std::perror("unlink");
        fail = 1;
    }
    std::printf("pinned_shared_test: %s\n", fail ? "FAILED" : "OK");
    return fail;
}
