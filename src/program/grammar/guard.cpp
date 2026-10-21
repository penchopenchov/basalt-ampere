// src/program/grammar/guard.cpp - the pack's vocabulary and the per-request window check.
#include "basalt/program/grammar/guard.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <unordered_map>

#include "basalt/program/grammar/schema.hpp"

namespace basalt::program::grammar {

namespace {

// the byte-level BPE spelling: a printable byte stands for itself, the others for U+0100 upward in byte order
const std::unordered_map<uint32_t, uint8_t>& char_to_byte() {
    static const std::unordered_map<uint32_t, uint8_t> table = [] {
        std::unordered_map<uint32_t, uint8_t> t;
        uint32_t next = 256;
        for (int b = 0; b < 256; ++b) {
            const bool plain = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
            t[plain ? (uint32_t) b : next++] = (uint8_t) b;
        }
        return t;
    }();
    return table;
}

void put_utf8(std::string& s, uint32_t c) {
    if (c < 0x80) s += (char) c;
    else if (c < 0x800) { s += (char) (0xC0 | (c >> 6)); s += (char) (0x80 | (c & 0x3F)); }
    else if (c < 0x10000) { s += (char) (0xE0 | (c >> 12)); s += (char) (0x80 | ((c >> 6) & 0x3F)); s += (char) (0x80 | (c & 0x3F)); }
    else { s += (char) (0xF0 | (c >> 18)); s += (char) (0x80 | ((c >> 12) & 0x3F)); s += (char) (0x80 | ((c >> 6) & 0x3F)); s += (char) (0x80 | (c & 0x3F)); }
}

// the next code point of valid UTF-8 at s[i] (a stray byte is its own value)
uint32_t next_cp(std::string_view s, size_t& i) {
    const uint8_t c = (uint8_t) s[i++];
    int extra = c >= 0xF0 ? 3 : c >= 0xE0 ? 2 : c >= 0xC0 ? 1 : 0;
    uint32_t v = extra == 3 ? c & 0x07 : extra == 2 ? c & 0x0F : extra == 1 ? c & 0x1F : c;
    while (extra-- > 0 && i < s.size()) v = (v << 6) | ((uint8_t) s[i++] & 0x3F);
    return v;
}

struct Scan {
    std::string_view s;
    size_t i = 0;
    void ws() { while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\t' || s[i] == '\r')) ++i; }
    bool eat(char c) { ws(); if (i < s.size() && s[i] == c) { ++i; return true; } return false; }
    bool str(std::string& out) {
        ws();
        if (i >= s.size() || s[i] != '"') return false;
        ++i;
        out.clear();
        while (i < s.size() && s[i] != '"') {
            if (s[i] != '\\') { out += s[i++]; continue; }
            if (++i >= s.size()) return false;
            const char e = s[i++];
            switch (e) {
                case 'n': out += '\n'; break;
                case 't': out += '\t'; break;
                case 'r': out += '\r'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'u': {
                    auto hex4 = [&](uint32_t& v) {
                        if (i + 4 > s.size()) return false;
                        v = 0;
                        for (int k = 0; k < 4; ++k) {
                            const char h = s[i + (size_t) k];
                            const int d = h >= '0' && h <= '9' ? h - '0' : h >= 'a' && h <= 'f' ? h - 'a' + 10 : h >= 'A' && h <= 'F' ? h - 'A' + 10 : -1;
                            if (d < 0) return false;
                            v = v * 16 + (uint32_t) d;
                        }
                        i += 4;
                        return true;
                    };
                    uint32_t v = 0;
                    if (!hex4(v)) return false;
                    if (v >= 0xD800 && v < 0xDC00 && i + 1 < s.size() && s[i] == '\\' && s[i + 1] == 'u') {
                        i += 2;
                        uint32_t lo = 0;
                        if (!hex4(lo)) return false;
                        v = 0x10000 + ((v - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    put_utf8(out, v);
                    break;
                }
                default: out += e; break;   // " \ /
            }
        }
        if (i >= s.size()) return false;
        ++i;
        return true;
    }
    bool integer(int64_t& v) {
        ws();
        const size_t b = i;
        v = 0;
        while (i < s.size() && s[i] >= '0' && s[i] <= '9') v = v * 10 + (s[i++] - '0');
        return i > b;
    }
};

}  // namespace

bool vocab_from_pack(std::string_view vocab_json, std::string_view types_json, size_t n_vocab,
                     const std::vector<int32_t>& end_ids, PackVocab& out, std::string& err) {
    std::vector<int8_t> types(n_vocab, -1);
    {
        Scan sc{types_json};
        if (!sc.eat('[')) { err = "token_type.json: not an array"; return false; }
        size_t id = 0;
        if (!sc.eat(']')) {
            do {
                int64_t t = 0;
                if (!sc.integer(t)) { err = "token_type.json: bad entry"; return false; }
                if (id < n_vocab) types[id] = (int8_t) t;
                ++id;
            } while (sc.eat(','));
        }
    }
    std::vector<std::string> bytes(n_vocab);
    out.think_open = out.think_close = -1;
    const auto& to_byte = char_to_byte();
    Scan sc{vocab_json};
    if (!sc.eat('{')) { err = "vocab.json: not an object"; return false; }
    std::string key;
    if (!sc.eat('}')) {
        do {
            int64_t id = 0;
            if (!sc.str(key) || !sc.eat(':') || !sc.integer(id)) { err = "vocab.json: bad entry"; return false; }
            if (id < 0 || (size_t) id >= n_vocab) continue;
            const int ty = types[(size_t) id];
            std::string& b = bytes[(size_t) id];
            if (ty == 3 || ty == 5 || ty == 6) continue;   // control, unused and byte entries never appear as text
            if (ty == 4) {
                b = key;   // an added token keeps its literal text
                if (key == "<think>") out.think_open = (int32_t) id;
                else if (key == "</think>") out.think_close = (int32_t) id;
                continue;
            }
            for (size_t i = 0; i < key.size();) {
                const uint32_t cp = next_cp(key, i);
                const auto it = to_byte.find(cp);
                if (it != to_byte.end()) b += (char) it->second;
                else put_utf8(b, cp);
            }
        } while (sc.eat(','));
    }
    if (!sc.eat('}')) { err = "vocab.json: unterminated"; return false; }
    out.vocab = std::make_shared<const Vocab>(std::move(bytes), end_ids);
    return true;
}

bool Guard::begin(const std::string& grammar_json, const int64_t* prompt, size_t n_prompt, std::string& err) {
    on_ = false;
    windows_ = misses_ = ns_ = 0;
    RequestGrammar rg = grammar_from_request(grammar_json, err);
    if (!err.empty()) return false;
    if (!rg.grammar) return true;
    st_ = GrammarState(rg.grammar, rg.triggers);
    // generation starts inside a thinking block when the prompt's last think tag opens one
    thinking_ = false;
    for (size_t i = n_prompt, seen = 0; i-- > 0 && seen < 16; ++seen) {
        const int64_t t = prompt[i];
        if (t == pv_.think_close) break;
        if (t == pv_.think_open) { thinking_ = true; break; }
    }
    on_ = true;
    return true;
}

bool Guard::feed(int32_t token) {
    if (thinking_) {
        if (token == pv_.think_close) thinking_ = false;
        return true;
    }
    const auto& ends = pv_.vocab->end_ids();
    if (std::find(ends.begin(), ends.end(), token) != ends.end()) return !st_.active() || st_.allows_end();
    if (token < 0 || (size_t) token >= pv_.vocab->size()) return !st_.active();
    const std::string& b = pv_.vocab->bytes(token);
    if (b.empty()) return !st_.active();   // a control token is not text
    return st_.accept(b, token);
}

int Guard::scan(const int32_t* out, int rows) {
    const auto t0 = std::chrono::steady_clock::now();
    int miss = -1;
    for (int t = 0; t < rows; ++t)
        if (!feed(out[t])) { miss = t; break; }
    ++windows_;
    if (miss >= 0) ++misses_;
    ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
    return miss;
}

void Guard::mask_bias(const std::vector<float>& user_bias, std::vector<float>& bias) {
    const auto t0 = std::chrono::steady_clock::now();
    st_.fill_mask(*pv_.vocab, mask_);
    const size_t n = mask_.size();
    bias.resize(n);
    const bool u = user_bias.size() == n;
    for (size_t i = 0; i < n; ++i) bias[i] = mask_[i] ? (u ? user_bias[i] : 0.0f) : -INFINITY;
    ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
}

bool Guard::take(int32_t token) {
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = feed(token);
    if (!ok) {
        std::fprintf(stderr, "basalt grammar: a masked draw was refused by the grammar; enforcement is off for this request\n");
        on_ = false;
    }
    ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
    return ok;
}

}  // namespace basalt::program::grammar
