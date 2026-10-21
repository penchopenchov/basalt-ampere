// src/program/grammar/json_value.hpp - a small ordered JSON value for reading schemas (member order is kept; the
// engine's own Json sorts members and keeps only the head of an array).
#pragma once

#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace basalt::program::grammar {

struct JsonValue {
    enum class Type { Null, Bool, Number, String, Array, Object };
    Type type = Type::Null;
    bool b = false;
    std::string s;                                          ///< String text, or the Number's source text
    std::vector<JsonValue> items;                           ///< Array
    std::vector<std::pair<std::string, JsonValue>> members; ///< Object, in source order

    bool is(Type t) const { return type == t; }
    const JsonValue* get(const std::string& key) const {
        for (const auto& m : members) if (m.first == key) return &m.second;
        return nullptr;
    }
    /// Compact JSON text, as a grammar constant for this value.
    std::string dump() const {
        switch (type) {
            case Type::Null: return "null";
            case Type::Bool: return b ? "true" : "false";
            case Type::Number: return s;
            case Type::String: return quote(s);
            case Type::Array: {
                std::string o = "[";
                for (size_t i = 0; i < items.size(); ++i) o += (i ? "," : "") + items[i].dump();
                return o + "]";
            }
            case Type::Object: {
                std::string o = "{";
                for (size_t i = 0; i < members.size(); ++i)
                    o += (i ? "," : "") + quote(members[i].first) + ":" + members[i].second.dump();
                return o + "}";
            }
        }
        return "null";
    }
    static std::string quote(const std::string& in) {
        std::string o = "\"";
        for (unsigned char c : in) {
            switch (c) {
                case '"': o += "\\\""; break;
                case '\\': o += "\\\\"; break;
                case '\n': o += "\\n"; break;
                case '\r': o += "\\r"; break;
                case '\t': o += "\\t"; break;
                case '\b': o += "\\b"; break;
                case '\f': o += "\\f"; break;
                default:
                    if (c < 0x20) {
                        static const char* hex = "0123456789abcdef";
                        o += "\\u00";
                        o += hex[c >> 4];
                        o += hex[c & 15];
                    } else {
                        o += (char) c;
                    }
            }
        }
        return o + "\"";
    }
};

class JsonReader {
public:
    static bool parse(const std::string& text, JsonValue& out, std::string& err) {
        JsonReader r(text.c_str(), text.c_str() + text.size());
        r.ws();
        if (!r.value(out, 0)) { err = "json: " + r.err; return false; }
        r.ws();
        if (r.p != r.end) { err = "json: trailing data"; return false; }
        return true;
    }

private:
    JsonReader(const char* b, const char* e) : p(b), end(e), begin(b) {}
    const char* p;
    const char* end;
    const char* begin;
    std::string err;

    bool fail(const char* m) { err = std::string(m) + " at byte " + std::to_string((size_t) (p - begin)); return false; }
    void ws() { while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p; }
    bool lit(const char* w) {
        size_t n = std::char_traits<char>::length(w);
        if ((size_t) (end - p) >= n && std::string(p, n) == w) { p += n; return true; }
        return false;
    }
    static void utf8(std::string& o, uint32_t c) {
        if (c < 0x80) { o += (char) c; }
        else if (c < 0x800) { o += (char) (0xC0 | (c >> 6)); o += (char) (0x80 | (c & 0x3F)); }
        else if (c < 0x10000) { o += (char) (0xE0 | (c >> 12)); o += (char) (0x80 | ((c >> 6) & 0x3F)); o += (char) (0x80 | (c & 0x3F)); }
        else { o += (char) (0xF0 | (c >> 18)); o += (char) (0x80 | ((c >> 12) & 0x3F)); o += (char) (0x80 | ((c >> 6) & 0x3F)); o += (char) (0x80 | (c & 0x3F)); }
    }
    bool hex4(uint32_t& v) {
        if (end - p < 4) return fail("bad \\u escape");
        v = 0;
        for (int i = 0; i < 4; ++i, ++p) {
            char c = *p;
            v <<= 4;
            if (c >= '0' && c <= '9') v += (uint32_t) (c - '0');
            else if (c >= 'a' && c <= 'f') v += (uint32_t) (c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v += (uint32_t) (c - 'A' + 10);
            else return fail("bad \\u escape");
        }
        return true;
    }
    bool str(std::string& o) {
        ++p;
        while (p < end && *p != '"') {
            if (*p != '\\') { o += *p++; continue; }
            if (++p >= end) break;
            char c = *p++;
            switch (c) {
                case '"': o += '"'; break;
                case '\\': o += '\\'; break;
                case '/': o += '/'; break;
                case 'b': o += '\b'; break;
                case 'f': o += '\f'; break;
                case 'n': o += '\n'; break;
                case 'r': o += '\r'; break;
                case 't': o += '\t'; break;
                case 'u': {
                    uint32_t v;
                    if (!hex4(v)) return false;
                    if (v >= 0xD800 && v < 0xDC00 && end - p >= 6 && p[0] == '\\' && p[1] == 'u') {
                        const char* save = p;
                        p += 2;
                        uint32_t lo;
                        if (!hex4(lo)) return false;
                        if (lo >= 0xDC00 && lo < 0xE000) v = 0x10000 + ((v - 0xD800) << 10) + (lo - 0xDC00);
                        else p = save;
                    }
                    utf8(o, v);
                    break;
                }
                default: return fail("bad escape");
            }
        }
        if (p >= end) return fail("unterminated string");
        ++p;
        return true;
    }
    bool value(JsonValue& v, int depth) {
        if (depth > 200) return fail("nesting too deep");
        if (p >= end) return fail("unexpected end");
        char c = *p;
        if (c == '{') {
            ++p;
            v.type = JsonValue::Type::Object;
            ws();
            if (p < end && *p == '}') { ++p; return true; }
            for (;;) {
                ws();
                if (p >= end || *p != '"') return fail("expected a key");
                std::string key;
                if (!str(key)) return false;
                ws();
                if (p >= end || *p != ':') return fail("expected ':'");
                ++p;
                ws();
                JsonValue child;
                if (!value(child, depth + 1)) return false;
                v.members.emplace_back(std::move(key), std::move(child));
                ws();
                if (p < end && *p == ',') { ++p; continue; }
                if (p < end && *p == '}') { ++p; return true; }
                return fail("expected ',' or '}'");
            }
        }
        if (c == '[') {
            ++p;
            v.type = JsonValue::Type::Array;
            ws();
            if (p < end && *p == ']') { ++p; return true; }
            for (;;) {
                ws();
                JsonValue child;
                if (!value(child, depth + 1)) return false;
                v.items.push_back(std::move(child));
                ws();
                if (p < end && *p == ',') { ++p; continue; }
                if (p < end && *p == ']') { ++p; return true; }
                return fail("expected ',' or ']'");
            }
        }
        if (c == '"') { v.type = JsonValue::Type::String; return str(v.s); }
        if (lit("true")) { v.type = JsonValue::Type::Bool; v.b = true; return true; }
        if (lit("false")) { v.type = JsonValue::Type::Bool; v.b = false; return true; }
        if (lit("null")) { v.type = JsonValue::Type::Null; return true; }
        if (c == '-' || (c >= '0' && c <= '9')) {
            const char* s = p;
            ++p;
            while (p < end && ((*p >= '0' && *p <= '9') || *p == '.' || *p == 'e' || *p == 'E' || *p == '+' || *p == '-')) ++p;
            v.type = JsonValue::Type::Number;
            v.s.assign(s, p);
            return true;
        }
        return fail("unexpected character");
    }
};

}  // namespace basalt::program::grammar
