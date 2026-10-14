// include/basalt/core/logit_bias.hpp - a request's token bias, from the engine protocol's key.
#pragma once

#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace basalt::core {

// `id:value,id:value` -> one bias per vocabulary entry, added to the logits before the sampler chain.  Values are
// -100..100; -100 bans the token (-inf), the others are added as given.  False with `err` set (and `out` empty) for
// an id outside the vocabulary, a repeated id, a value out of range or not a number, or a list that bans everything.
inline bool parse_logit_bias(const std::string& text, int vocab, std::vector<float>& out, std::string& err) {
    out.clear();
    if (text.empty() || vocab <= 0) { err = "empty logit_bias or invalid vocabulary"; return false; }
    std::vector<float> bias((size_t) vocab, 0.0f);
    std::vector<bool> seen((size_t) vocab, false);
    const char* p = text.c_str();
    int banned = 0;
    for (;;) {
        if (*p < '0' || *p > '9') { err = "invalid logit_bias token id"; return false; }
        char* end = nullptr;
        const unsigned long id = std::strtoul(p, &end, 10);
        if (*end != ':' || id >= (unsigned long) vocab || seen[id]) {
            err = "logit_bias id outside the vocabulary, repeated, or missing ':'";
            return false;
        }
        p = end + 1;
        const float value = std::strtof(p, &end);
        if (end == p || !std::isfinite(value) || value < -100 || value > 100 || (*end && *end != ',')) {
            err = "invalid logit_bias value (expected -100..100)";
            return false;
        }
        seen[id] = true;
        bias[id] = value == -100 ? -std::numeric_limits<float>::infinity() : value;
        banned += value == -100;
        if (!*end) break;
        p = end + 1;
    }
    if (banned == vocab) { err = "logit_bias bans the entire vocabulary"; return false; }
    out = std::move(bias);
    return true;
}

}  // namespace basalt::core
