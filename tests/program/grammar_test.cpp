// tests/program/grammar_test.cpp - GBNF parse, accept/reject, UTF-8 across tokens, clone/rewind, lazy triggers, and
// the vocabulary mask against a token-by-token check. Ends with a timing line for allows() and fill_mask().
#include "basalt/program/grammar/grammar.hpp"
#include "basalt/program/grammar/schema.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace basalt::program::grammar;

static int g_fail = 0;
#define CHECK(c)                                                              \
    do {                                                                      \
        if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } \
    } while (0)

static std::shared_ptr<const Grammar> parse(const std::string& g) {
    std::string err;
    auto p = Grammar::parse(g, err);
    if (!p) std::printf("parse error: %s\n", err.c_str());
    return p;
}

// A complete sentence of the grammar, fed byte by byte.
static bool full(const std::shared_ptr<const Grammar>& g, const std::string& text) {
    GrammarState s(g);
    for (char c : text) if (!s.accept(std::string_view(&c, 1))) return false;
    return s.accepting();
}
// Every byte is accepted (a prefix of some sentence).
static bool prefix(const std::shared_ptr<const Grammar>& g, const std::string& text) {
    GrammarState s(g);
    for (char c : text) if (!s.accept(std::string_view(&c, 1))) return false;
    return true;
}

static const char* kJson = R"(
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws
object ::= "{" ws ( string ":" ws value ("," ws string ":" ws value)* )? "}" ws
array  ::= "[" ws ( value ("," ws value)* )? "]" ws
string ::= "\"" ( [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4}) )* "\"" ws
number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9] [1-9]{0,15})? ws
ws     ::= | " " | "\n" [ \t]{0,20}
)";

static void test_parse() {
    std::string err;
    CHECK(!Grammar::parse("root ::= a", err) && err.find("undefined") != std::string::npos);
    CHECK(!Grammar::parse("x ::= \"a\"", err) && err.find("root") != std::string::npos);
    CHECK(!Grammar::parse("root ::= root \"a\" | \"b\"", err) && err.find("left recursion") != std::string::npos);
    CHECK(!Grammar::parse("root ::= \"a", err));
    CHECK(!Grammar::parse("root ::= [a-", err));
    CHECK(!Grammar::parse("root ::= *", err));
    CHECK(!Grammar::parse("root ::= \"a\"{3", err));
    CHECK(Grammar::parse("# comment\nroot ::= \"a\" # tail\n", err));
}

static void test_basic() {
    auto g = parse("root ::= \"ab\" [0-9]+ (\"x\" | \"yz\")? \"!\"");
    CHECK(g);
    CHECK(full(g, "ab1!"));
    CHECK(full(g, "ab123x!"));
    CHECK(full(g, "ab9yz!"));
    CHECK(!full(g, "ab!"));
    CHECK(!prefix(g, "ac"));
    CHECK(prefix(g, "ab12y") && !full(g, "ab12y"));
    auto r = parse("root ::= \"a\"{2,3} \"b\"{2}");
    CHECK(full(r, "aabb") && full(r, "aaabb") && !full(r, "abb") && !prefix(r, "aaaa") && !full(r, "aab"));
    auto n = parse("root ::= [^a-c]+ .");
    CHECK(full(n, "xyzq") && !prefix(n, "a") && full(n, "dd\xC3\xA9"));
    auto esc = parse("root ::= \"\\x41\\u00e9\\n\"");
    CHECK(full(esc, "A\xC3\xA9\n"));
}

static void test_json() {
    auto g = parse(kJson);
    CHECK(g);
    CHECK(full(g, "{}"));
    CHECK(full(g, "{\"a\": 1, \"b\": [1, 2.5, -3e+2, true, null], \"c\": {\"d\": \"x\"}}"));
    CHECK(full(g, "{\"s\": \"a\\\"b\\n\\u00e9\"}"));
    CHECK(full(g, "{\"u\": \"\xC3\xA9\xE4\xB8\xAD\xF0\x9F\x98\x80\"}"));
    CHECK(full(g, "[[[[]]], [], {\"k\": []}]") == false);   // root is an object
    CHECK(full(g, "{\"k\": [[[[]]], [], {\"k\": []}]}"));
    CHECK(!prefix(g, "{\"a\": 01"));
    CHECK(!prefix(g, "{\"a\": True"));
    CHECK(!prefix(g, "{\"a\": \"x\ny\"}"));
    CHECK(!prefix(g, "{'a': 1}"));
    CHECK(prefix(g, "{\"a\": 1.") && !full(g, "{\"a\": 1."));
    CHECK(!full(g, "{\"a\": 1,}"));
}

static void test_utf8_tokens() {
    auto g = parse(kJson);
    GrammarState s(g);
    CHECK(s.accept("{\"k\": \""));
    // a three-byte character split over three tokens, and one that stops mid-way
    CHECK(s.allows("\xE4"));
    CHECK(s.accept("\xE4"));
    CHECK(s.accept("\xB8"));
    CHECK(!s.accepting());
    CHECK(s.accept("\xAD\"}"));
    CHECK(s.accepting());
    GrammarState t(g);
    CHECK(!t.allows("\xC3"));                 // not inside a string
    CHECK(t.accept("{\"") && !t.allows("\xC0\x80") && !t.allows("\xFF") && !t.allows("\x80"));
    CHECK(t.allows("\xC3\xA9") && t.allows("\xF0\x9F\x98"));
    CHECK(!t.allows("\xED\xA0\x80"));         // a surrogate
    // a class that excludes the range stops a partial early
    auto ascii = parse("root ::= [a-z]+");
    GrammarState a(ascii);
    CHECK(!a.allows("\xC3"));
    auto latin = parse("root ::= [\\u00e0-\\u00ff]+");
    GrammarState l(latin);
    CHECK(l.allows("\xC3") && !l.allows("\xE4"));
}

static void test_clone_rewind() {
    auto g = parse(kJson);
    GrammarState s(g);
    CHECK(s.accept("{\"a\": "));
    GrammarState mark = s.clone();
    CHECK(s.accept("[1, 2"));
    CHECK(!s.accept("}"));                    // a failed accept leaves the state as it was
    CHECK(s.accept("]}"));
    CHECK(s.accepting());
    CHECK(!mark.accepting());
    CHECK(mark.accept("7}"));
    CHECK(mark.accepting());
    s = mark.clone();
    CHECK(s.accepting());
    s.reset();
    CHECK(!s.accepting() && s.accept("{}") && s.accepting());
}

// A small vocabulary with multi-byte, partial and JSON-shaped pieces.
static Vocab test_vocab(int32_t& eos) {
    std::vector<std::string> v;
    for (int i = 0; i < 256; ++i) v.push_back(std::string(1, (char) i));
    for (const char* t : {"{\"", "\":", "\": ", "\",", "\"}", "true", "false", "null", "[]", "{}", ", ", "\\n", "\\u00",
                          "abc", " abc", "123", "-1", "0.5", "e+", "\xC3\xA9", "\xE4\xB8", "\xE4\xB8\xAD", "\xF0\x9F",
                          "\xF0\x9F\x98\x80", "}\n", "\n  ", "  ", "\"\"", "[1", "]]"})
        v.push_back(t);
    eos = (int32_t) v.size();
    v.push_back("");                           // the end token has no text
    v.push_back("");                           // a special token
    return Vocab(v, {eos});
}

static void test_mask() {
    auto g = parse(kJson);
    int32_t eos;
    Vocab voc = test_vocab(eos);
    const char* prefixes[] = {"", "{", "{\"a\"", "{\"a\": ", "{\"a\": 1", "{\"a\": \"x", "{\"a\": \"\xE4", "{\"a\": [1,",
                              "{\"a\": tr", "{\"a\": 1.", "{}", "{\"a\": \"x\\", "{\"a\": \"x\\u0"};
    for (const char* p : prefixes) {
        GrammarState s(g);
        CHECK(s.accept(p));
        std::vector<uint8_t> mask;
        s.fill_mask(voc, mask);
        std::vector<uint8_t> again;
        s.fill_mask(voc, again);                  // the second call comes from the cache
        CHECK(again == mask);
        s.fill_mask(voc, again, false);
        CHECK(again == mask);
        CHECK(mask.size() == voc.size());
        int bad = 0, allowed = 0;
        for (size_t t = 0; t < voc.size(); ++t) {
            bool want = (int32_t) t == eos ? s.accepting() : (!voc.bytes((int32_t) t).empty() && s.allows(voc.bytes((int32_t) t)));
            if ((mask[t] != 0) != want) ++bad;
            allowed += mask[t];
        }
        if (bad) std::printf("mask mismatch after '%s': %d\n", p, bad);
        CHECK(bad == 0);
        CHECK(allowed > 0);
    }
    GrammarState s(g);
    std::vector<uint8_t> mask;
    s.fill_mask(voc, mask);
    CHECK(mask['{'] && !mask['['] && !mask['a'] && !mask[(size_t) eos]);
    CHECK(s.accept("{}"));
    s.fill_mask(voc, mask);
    CHECK(mask[(size_t) eos] && mask[' ']);
}

static void test_lazy() {
    auto g = parse("root ::= \"<c>\" [a-z]+ \"</c>\"");
    Triggers tr;
    tr.words = {"<c>"};
    GrammarState s(g, tr);
    CHECK(!s.active() && s.accepting());
    CHECK(s.accept("Hello, 123 <b"));            // free text, not a trigger
    CHECK(!s.active());
    CHECK(s.allows("anything at all"));
    CHECK(s.accept("><"));                        // "<b><": no trigger yet
    CHECK(!s.active());
    CHECK(s.accept("c"));                         // "<" "c" is still short of "<c>"
    CHECK(!s.active());
    CHECK(s.accept(">"));                         // "<c>" completes across tokens
    CHECK(s.active() && !s.accepting());
    CHECK(!s.allows("1"));
    CHECK(s.accept("abc") && s.accept("</c>"));
    CHECK(!s.active() && s.accepting());          // complete: waiting for the next trigger
    CHECK(s.accept("more text <c>x</c>"));        // a whole call inside one token
    CHECK(!s.active());
    // a trigger followed by a violation in the same token is refused
    CHECK(!s.allows("<c>1"));
    CHECK(s.accept("<c>q") && s.active());
    // a trigger token
    Triggers tt;
    tt.tokens = {42};
    GrammarState k(g, tt);
    CHECK(k.accept("<c>", 7) == true && !k.active());   // not the trigger token: the text is just text
    CHECK(k.accept("<c>", 42) && k.active());
    int32_t eos;
    Vocab voc = test_vocab(eos);
    GrammarState f(g, tr);
    std::vector<uint8_t> mask;
    f.fill_mask(voc, mask);
    size_t sum = 0;
    for (uint8_t m : mask) sum += m;
    CHECK(sum == voc.size());                     // inactive: nothing is masked
}

static double now_us() {
    using namespace std::chrono;
    return (double) duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count() / 1000.0;
}

// A synthetic vocabulary of the real size: byte tokens, words with and without a leading space, digits, punctuation
// runs and multi-byte characters.
static std::vector<std::string> bench_vocab(size_t n) {
    std::vector<std::string> v;
    for (int i = 0; i < 256; ++i) v.push_back(std::string(1, (char) i));
    uint64_t x = 88172645463325252ull;
    auto rnd = [&]() { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; };
    const char* punct[] = {"\"", "\":", "\",", "{\"", "\"}", "},", "],", "\\n", "\\\"", "://", "=>", "();", "\n\n", "  ", "    "};
    while (v.size() < n) {
        std::string t;
        uint64_t k = rnd() % 100;
        if (k < 60) {
            if (rnd() % 2) t += ' ';
            size_t len = 2 + rnd() % 8;
            for (size_t i = 0; i < len; ++i) t += (char) ('a' + rnd() % 26);
        } else if (k < 70) {
            size_t len = 1 + rnd() % 4;
            for (size_t i = 0; i < len; ++i) t += (char) ('0' + rnd() % 10);
        } else if (k < 85) {
            t = punct[rnd() % 15];
        } else {
            size_t len = 1 + rnd() % 3;
            for (size_t i = 0; i < len; ++i) {
                uint32_t cp = 0x4E00 + (uint32_t) (rnd() % 20000);
                t += (char) (0xE0 | (cp >> 12));
                t += (char) (0x80 | ((cp >> 6) & 0x3F));
                t += (char) (0x80 | (cp & 0x3F));
            }
        }
        v.push_back(t);
    }
    return v;
}

// BASALT_GRAMMAR_VOCAB names a file of (u32 length, bytes) records, one per token id, for a run on the real vocabulary;
// BASALT_GRAMMAR_END is the end token's id.
static bool load_vocab(const char* path, std::vector<std::string>& out) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    uint32_t n;
    while (std::fread(&n, sizeof n, 1, f) == 1) {
        std::string t(n, '\0');
        if (n && std::fread(&t[0], 1, n, f) != n) break;
        out.push_back(std::move(t));
    }
    std::fclose(f);
    return !out.empty();
}

static void bench() {
    std::vector<std::string> tokens;
    const char* vpath = std::getenv("BASALT_GRAMMAR_VOCAB");
    const char* eos = std::getenv("BASALT_GRAMMAR_END");
    bool real = vpath && load_vocab(vpath, tokens);
    if (!real) tokens = bench_vocab(248320);
    const size_t N = tokens.size();
    std::printf("bench grammar: %s vocabulary, %zu tokens\n", real ? "real" : "synthetic", N);
    Vocab voc(std::move(tokens), {eos ? std::atoi(eos) : 0});
    auto g = parse(kJson);
    std::string err;
    auto tools = Grammar::parse(tool_call_gbnf(R"([{"name":"f","parameters":{"properties":{"text":{"type":"string"},"n":{"type":"integer"},"obj":{"type":"object","properties":{"k":{"type":"string"}}}},"required":["text"]}}])", {}, err), err);
    struct Case { const char* where; std::shared_ptr<const Grammar> g; const char* text; } cases[] = {
        {"json, start of an object", g, "{"},
        {"json, inside a string value", g, "{\"key\": \"some text"},
        {"json, inside a nested array of objects", g, "{\"a\": [{\"b\": [1, 2, {\"c\": \"x"},
        {"tool call, inside a string parameter", tools, "<tool_call>\n<function=f>\n<parameter=text>\nhello wor"},
        {"tool call, inside an integer parameter", tools, "<tool_call>\n<function=f>\n<parameter=text>\nhi\n</parameter>\n<parameter=n>\n12"},
        {"tool call, between parameters", tools, "<tool_call>\n<function=f>\n<parameter=text>\nhi\n</parameter>\n"},
    };
    for (const Case& c : cases) {
        GrammarState s(c.g);
        s.accept(c.text);
        const int reps = 2000;
        double t0 = now_us();
        int ok = 0;
        for (int i = 0; i < reps; ++i) ok += s.allows(" the") ? 1 : 0;
        double allows_us = (now_us() - t0) / reps;
        std::vector<uint8_t> mask;
        s.fill_mask(voc, mask, false);     // first call builds the memo tables
        double best = 1e18, cached = 1e18;
        size_t allowed = 0;
        for (int i = 0; i < 5; ++i) {
            double a = now_us();
            s.fill_mask(voc, mask, false);
            best = (std::min)(best, now_us() - a);
        }
        s.fill_mask(voc, mask);
        for (int i = 0; i < 5; ++i) {
            double a = now_us();
            s.fill_mask(voc, mask);
            cached = (std::min)(cached, now_us() - a);
        }
        for (uint8_t m : mask) allowed += m;
        std::printf("bench grammar: %-40s allows(1 token) %.2f us (%d/%d)  fill_mask(%zu tokens) %.2f ms, repeated position %.3f ms  allowed %zu\n",
                    c.where, allows_us, ok, reps, N, best / 1000.0, cached / 1000.0, allowed);
    }
}

int main() {
    test_parse();
    test_basic();
    test_json();
    test_utf8_tokens();
    test_clone_rewind();
    test_mask();
    test_lazy();
    if (std::getenv("BASALT_GRAMMAR_BENCH")) bench();
    if (g_fail) { std::printf("grammar_test: %d failures\n", g_fail); return 1; }
    std::printf("grammar_test: ok\n");
    return 0;
}
