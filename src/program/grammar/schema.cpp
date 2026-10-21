// src/program/grammar/schema.cpp - see schema.hpp. The rule shapes follow llama.cpp's json-schema-to-grammar.
#include "basalt/program/grammar/schema.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_map>

#include "json_value.hpp"

namespace basalt::program::grammar {

namespace {

using Type = JsonValue::Type;

const char* const kSpaceRule = R"(| " " | "\n"{1,2} [ \t]{0,20})";

struct Builtin {
    const char* content;
    std::vector<std::string> deps;
};

const std::unordered_map<std::string, Builtin>& builtins() {
    static const std::unordered_map<std::string, Builtin> t = {
        {"boolean", {R"(("true" | "false"))", {}}},
        {"decimal-part", {R"([0-9]{1,16})", {}}},
        {"integral-part", {R"([0] | [1-9] [0-9]{0,15})", {}}},
        {"number", {R"(("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?)", {"integral-part", "decimal-part"}}},
        {"integer", {R"(("-"? integral-part))", {"integral-part"}}},
        {"value", {R"(object | array | string | number | boolean | null)", {"object", "array", "string", "number", "boolean", "null"}}},
        {"object", {R"("{" space ( string ":" space value ("," space string ":" space value)* )? space "}")", {"string", "value"}}},
        {"array", {R"("[" space ( value ("," space value)* )? space "]")", {"value"}}},
        {"uuid", {R"("\"" [0-9a-fA-F]{8} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{12} "\"")", {}}},
        {"char", {R"([^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4}))", {}}},
        {"string", {R"("\"" char* "\"")", {"char"}}},
        {"null", {R"("null")", {}}},
        {"date", {R"([0-9]{4} "-" ( "0" [1-9] | "1" [0-2] ) "-" ( "0" [1-9] | [1-2] [0-9] | "3" [0-1] ))", {}}},
        {"time", {R"(([01] [0-9] | "2" [0-3]) ":" [0-5] [0-9] ":" [0-5] [0-9] ( "." [0-9]{3} )? ( "Z" | ( "+" | "-" ) ( [01] [0-9] | "2" [0-3] ) ":" [0-5] [0-9] ))", {}}},
        {"date-time", {R"(date "T" time)", {"date", "time"}}},
        {"date-string", {R"("\"" date "\"")", {"date"}}},
        {"time-string", {R"("\"" time "\"")", {"time"}}},
        {"date-time-string", {R"("\"" date-time "\"")", {"date-time"}}},
    };
    return t;
}

bool reserved(const std::string& n) { return n == "root" || n == "space" || builtins().count(n) > 0; }

std::string sanitize(const std::string& in) {
    std::string o;
    bool gap = false;
    for (char c : in) {
        bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-';
        if (ok) {
            if (gap) o += '-';
            gap = false;
            o += c;
        } else {
            gap = true;
        }
    }
    if (gap) o += '-';
    return o;
}

// A string as a GBNF literal.
std::string literal(const std::string& s) {
    std::string o = "\"";
    for (char c : s) {
        if (c == '\r') o += "\\r";
        else if (c == '\n') o += "\\n";
        else if (c == '"') o += "\\\"";
        else if (c == '\\') o += "\\\\";
        else o += c;
    }
    return o + "\"";
}

std::string join(const std::vector<std::string>& v, const std::string& sep) {
    std::string o;
    for (size_t i = 0; i < v.size(); ++i) o += (i ? sep : "") + v[i];
    return o;
}

std::string repetition(const std::string& item, int min_items, int max_items, const std::string& sep = "") {
    const int kNoMax = std::numeric_limits<int>::max();
    bool has_max = max_items != kNoMax;
    if (max_items == 0) return "";
    if (min_items == 0 && max_items == 1) return item + "?";
    if (sep.empty()) {
        if (min_items == 1 && !has_max) return item + "+";
        if (min_items == 0 && !has_max) return item + "*";
        return item + "{" + std::to_string(min_items) + "," + (has_max ? std::to_string(max_items) : "") + "}";
    }
    std::string r = item + " " + repetition("(" + sep + " " + item + ")", min_items == 0 ? 0 : min_items - 1,
                                              has_max ? max_items - 1 : max_items);
    if (min_items == 0) r = "(" + r + ")?";
    return r;
}

class Converter {
public:
    explicit Converter(const JsonValue* root) : root_(root) { rules_["space"] = kSpaceRule; }
    std::vector<std::string> errors;
    const JsonValue* root_;

    std::string add_rule(const std::string& name, const std::string& rule) {
        std::string esc = sanitize(name);
        auto it = rules_.find(esc);
        if (it == rules_.end() || it->second == rule) {
            rules_[esc] = rule;
            return esc;
        }
        int i = 0;
        while (rules_.count(esc + std::to_string(i)) && rules_[esc + std::to_string(i)] != rule) ++i;
        std::string key = esc + std::to_string(i);
        rules_[key] = rule;
        return key;
    }

    std::string primitive(const std::string& name) {
        const Builtin& b = builtins().at(name);
        std::string n = add_rule(name, b.content);
        for (const std::string& dep : b.deps) {
            if (!rules_.count(dep)) primitive(dep);
        }
        return n;
    }

    std::string format() const {
        std::string o;
        for (const auto& kv : rules_) o += kv.first + " ::= " + kv.second + "\n";
        return o;
    }

    const JsonValue* resolve(const std::string& ref) {
        if (ref.empty() || ref[0] != '#') {
            errors.push_back("only local $ref is supported: " + ref);
            return nullptr;
        }
        const JsonValue* cur = root_;
        size_t i = 1;
        while (i < ref.size() && cur) {
            if (ref[i] != '/') break;
            size_t j = ref.find('/', i + 1);
            std::string part = ref.substr(i + 1, j == std::string::npos ? std::string::npos : j - i - 1);
            std::string dec;
            for (size_t k = 0; k < part.size(); ++k) {
                if (part[k] == '~' && k + 1 < part.size() && part[k + 1] == '1') { dec += '/'; ++k; }
                else if (part[k] == '~' && k + 1 < part.size() && part[k + 1] == '0') { dec += '~'; ++k; }
                else dec += part[k];
            }
            if (cur->is(Type::Object)) cur = cur->get(dec);
            else if (cur->is(Type::Array) && !dec.empty() && dec.find_first_not_of("0123456789") == std::string::npos &&
                     std::stoul(dec) < cur->items.size()) cur = &cur->items[std::stoul(dec)];
            else cur = nullptr;
            i = j == std::string::npos ? ref.size() : j;
        }
        if (!cur) errors.push_back("unresolved $ref " + ref);
        return cur;
    }

    // Follows $ref links to the schema they name.
    const JsonValue* deref(const JsonValue* s) {
        for (int guard = 0; s && s->is(Type::Object) && guard < 32; ++guard) {
            const JsonValue* r = s->get("$ref");
            if (!r || !r->is(Type::String)) break;
            s = resolve(r->s);
        }
        return s;
    }

    std::string visit(const JsonValue& schema, const std::string& name) {
        std::string rule_name = reserved(name) ? name + "-" : name.empty() ? "root" : name;
        std::string sub = name + (name.empty() ? "" : "-");
        if (schema.is(Type::Bool)) {
            if (!schema.b) errors.push_back("a false schema accepts nothing");
            return add_rule(rule_name, primitive("value"));
        }
        if (!schema.is(Type::Object)) {
            errors.push_back("a schema must be an object");
            return "";
        }
        if (const JsonValue* ref = schema.get("$ref")) {
            if (!ref->is(Type::String)) { errors.push_back("$ref must be a string"); return ""; }
            return add_rule(rule_name, resolve_ref(*ref));
        }
        for (const char* key : {"oneOf", "anyOf"}) {
            const JsonValue* alts = schema.get(key);
            if (alts && alts->is(Type::Array)) return add_rule(rule_name, union_rule(name, alts->items));
        }
        if (const JsonValue* all = schema.get("allOf")) {
            if (all->is(Type::Array)) return all_of(*all, name, rule_name);
        }
        if (const JsonValue* c = schema.get("const")) return add_rule(rule_name, literal(c->dump()));
        if (const JsonValue* e = schema.get("enum")) {
            if (e->is(Type::Array) && !e->items.empty()) {
                std::vector<std::string> vals;
                for (const JsonValue& v : e->items) vals.push_back(literal(v.dump()));
                return add_rule(rule_name, "(" + join(vals, " | ") + ")");
            }
        }
        const JsonValue* type = schema.get("type");
        if (type && type->is(Type::Array)) {
            std::vector<JsonValue> alts;
            for (const JsonValue& t : type->items) {
                JsonValue copy = schema;
                for (auto& m : copy.members) if (m.first == "type") m.second = t;
                alts.push_back(std::move(copy));
            }
            return add_rule(rule_name, union_rule(name, alts));
        }
        std::string t = type && type->is(Type::String) ? type->s : "";
        const JsonValue* props = schema.get("properties");
        if (t.empty()) {
            if (props) t = "object";
            else if (schema.get("items") || schema.get("prefixItems")) t = "array";
        }
        if (t == "object") return object_rule(schema, name, rule_name);
        if (t == "array") {
            if (const JsonValue* pre = schema.get("prefixItems")) {
                if (pre->is(Type::Array)) {
                    std::string rule = "\"[\" space ";
                    for (size_t i = 0; i < pre->items.size(); ++i) {
                        if (i) rule += " \",\" space ";
                        rule += visit(pre->items[i], sub + "tuple-" + std::to_string(i));
                    }
                    return add_rule(rule_name, rule + " space \"]\"");
                }
            }
            const JsonValue* items = schema.get("items");
            int min_items = int_of(schema.get("minItems"), 0);
            int max_items = int_of(schema.get("maxItems"), std::numeric_limits<int>::max());
            if ((!items || items->is(Type::Bool)) && min_items == 0 && max_items == std::numeric_limits<int>::max())
                return add_rule(rule_name, primitive("array"));
            std::string item = items ? visit(*items, sub + "item") : add_rule(sub + "item", primitive("value"));
            return add_rule(rule_name, "\"[\" space " + repetition(item, min_items, max_items, "\",\" space") +
                                           " space \"]\"");
        }
        if (t == "string") {
            const JsonValue* fmt = schema.get("format");
            if (fmt && fmt->is(Type::String)) {
                if (fmt->s == "uuid") return add_rule(rule_name, primitive("uuid"));
                if (fmt->s == "date" || fmt->s == "time" || fmt->s == "date-time")
                    return add_rule(rule_name, primitive(fmt->s + "-string"));
            }
            int min_len = int_of(schema.get("minLength"), 0);
            int max_len = int_of(schema.get("maxLength"), std::numeric_limits<int>::max());
            if (min_len > 0 || max_len != std::numeric_limits<int>::max())
                return add_rule(rule_name, "\"\\\"\" " + repetition(primitive("char"), min_len, max_len) + " \"\\\"\"");
            return add_rule(rule_name, primitive("string"));
        }
        if (t == "integer" || t == "number" || t == "boolean" || t == "null")
            return add_rule(rule_name, primitive(t));
        if (t.empty()) return add_rule(rule_name, primitive("value"));
        errors.push_back("unknown type '" + t + "'");
        return "";
    }

private:
    std::map<std::string, std::string> rules_;
    std::set<std::string> refs_open_;

    static int int_of(const JsonValue* v, int dflt) {
        if (!v || !v->is(Type::Number)) return dflt;
        double d = std::atof(v->s.c_str());
        if (d < 0) return dflt;
        if (d > 100000) return std::numeric_limits<int>::max();
        return (int) d;
    }

    std::string resolve_ref(const JsonValue& ref) {
        size_t hash = ref.s.find('#');
        std::string frag = hash != std::string::npos ? ref.s.substr(hash + 1) : ref.s;
        std::string ref_name = "ref" + sanitize(frag);
        if (!rules_.count(ref_name) && !refs_open_.count(ref.s)) {
            const JsonValue* target = resolve(ref.s);
            if (!target) return "";
            refs_open_.insert(ref.s);
            ref_name = visit(*target, ref_name);
            refs_open_.erase(ref.s);
        }
        return ref_name;
    }

    std::string union_rule(const std::string& name, const std::vector<JsonValue>& alts) {
        std::vector<std::string> rules;
        for (size_t i = 0; i < alts.size(); ++i)
            rules.push_back(visit(alts[i], name + (name.empty() ? "alternative-" : "-") + std::to_string(i)));
        return join(rules, " | ");
    }

    std::string all_of(const JsonValue& all, const std::string& name, const std::string& rule_name) {
        std::vector<std::pair<std::string, const JsonValue*>> props;
        std::set<std::string> required;
        auto add = [&](const JsonValue* comp, bool is_required) {
            comp = deref(comp);
            if (!comp || !comp->is(Type::Object)) return;
            if (const JsonValue* p = comp->get("properties")) {
                for (const auto& m : p->members) {
                    props.emplace_back(m.first, &m.second);
                    if (is_required) required.insert(m.first);
                }
            }
            if (const JsonValue* r = comp->get("required")) {
                for (const JsonValue& n : r->items) if (n.is(Type::String) && is_required) required.insert(n.s);
            }
        };
        for (const JsonValue& child : all.items) {
            const JsonValue* any = child.is(Type::Object) ? child.get("anyOf") : nullptr;
            if (any && any->is(Type::Array)) {
                for (const JsonValue& alt : any->items) add(&alt, false);
            } else {
                add(&child, true);
            }
        }
        return add_rule(rule_name, build_object(props, required, name, nullptr, false));
    }

    std::string object_rule(const JsonValue& schema, const std::string& name, const std::string& rule_name) {
        const JsonValue* props = schema.get("properties");
        const JsonValue* addl = schema.get("additionalProperties");
        bool addl_any = addl && ((addl->is(Type::Bool) && addl->b) || (addl->is(Type::Object) && addl->members.empty()));
        if ((!props || props->members.empty()) && (!addl || addl_any)) return add_rule(rule_name, primitive("object"));
        std::vector<std::pair<std::string, const JsonValue*>> plist;
        if (props) for (const auto& m : props->members) plist.emplace_back(m.first, &m.second);
        std::set<std::string> required;
        if (const JsonValue* r = schema.get("required")) {
            for (const JsonValue& n : r->items) if (n.is(Type::String)) required.insert(n.s);
        }
        const JsonValue* extra = addl && addl->is(Type::Object) && !addl_any ? addl : nullptr;
        return add_rule(rule_name, build_object(plist, required, name, extra, addl_any));
    }

    std::string build_object(const std::vector<std::pair<std::string, const JsonValue*>>& props,
                             const std::set<std::string>& required, const std::string& name,
                             const JsonValue* extra_schema, bool extra_any) {
        std::vector<std::string> req, opt;
        std::map<std::string, std::string> kv;
        std::string pre = name + (name.empty() ? "" : "-");
        for (const auto& p : props) {
            std::string value = visit(*p.second, pre + p.first);
            kv[p.first] = add_rule(pre + p.first + "-kv", literal(JsonValue::quote(p.first)) + " space \":\" space " + value);
            (required.count(p.first) ? req : opt).push_back(p.first);
        }
        if (extra_schema || extra_any) {
            std::string sub = pre + "additional";
            std::string value = extra_schema ? visit(*extra_schema, sub + "-value") : primitive("value");
            kv["*"] = add_rule(sub + "-kv", primitive("string") + " \":\" space " + value);
            opt.push_back("*");
        }
        if (req.empty() && opt.empty()) return "\"{\" space \"}\"";
        std::string rule = "\"{\" space ";
        for (size_t i = 0; i < req.size(); ++i) rule += (i ? " \",\" space " : "") + kv[req[i]];
        if (!opt.empty()) {
            rule += " (";
            if (!req.empty()) rule += " \",\" space ( ";
            for (size_t i = 0; i < opt.size(); ++i) {
                if (i) rule += " | ";
                rule += optional_chain(kv, pre, std::vector<std::string>(opt.begin() + (std::ptrdiff_t) i, opt.end()), false);
            }
            if (!req.empty()) rule += " )";
            rule += " )?";
        }
        return rule + " space \"}\"";
    }

    std::string optional_chain(std::map<std::string, std::string>& kv, const std::string& pre,
                               const std::vector<std::string>& ks, bool first_optional) {
        if (ks.empty()) return "";
        const std::string& k = ks[0];
        std::string comma = "( \",\" space " + kv[k] + " )";
        std::string res = first_optional ? comma + (k == "*" ? "*" : "?") : kv[k] + (k == "*" ? " " + comma + "*" : "");
        if (ks.size() > 1) {
            res += " " + add_rule(pre + k + "-rest",
                                  optional_chain(kv, pre, std::vector<std::string>(ks.begin() + 1, ks.end()), true));
        }
        return res;
    }
};

bool read_json(const std::string& text, JsonValue& v, std::string& err) {
    if (!JsonReader::parse(text, v, err)) return false;
    return true;
}

}  // namespace

std::string json_schema_to_gbnf(const std::string& schema_json, std::string& err) {
    JsonValue schema;
    if (!read_json(schema_json, schema, err)) return "";
    Converter c(&schema);
    c.visit(schema, "");
    if (!c.errors.empty()) {
        err = "schema: " + join(c.errors, "; ");
        return "";
    }
    return c.format();
}

// ------------------------------------------------------------------------------------------ tool calls

namespace {

std::string hex_escape(unsigned char c) {
    static const char* h = "0123456789abcdef";
    return std::string("\\x") + h[c >> 4] + h[c & 15];
}

// Text that never contains the closing tag: a KMP automaton over the tag, one rule per matched length. The tag's last
// character is the one move the automaton refuses.
std::string add_free_text(Converter& c, const std::string& tag) {
    size_t n = tag.size();
    std::set<unsigned char> alpha(tag.begin(), tag.end());
    std::string excluded;
    for (unsigned char a : alpha) excluded += hex_escape(a);
    for (size_t i = 0; i < n; ++i) {
        std::vector<std::string> alts;
        for (unsigned char a : alpha) {
            std::string seen = tag.substr(0, i) + (char) a;
            size_t k = (std::min)(seen.size(), n);
            while (k > 0 && tag.compare(0, k, seen, seen.size() - k, k) != 0) --k;
            if (k == n) continue;
            alts.push_back(literal(std::string(1, (char) a)) + " free-text-" + std::to_string(k));
        }
        alts.push_back("[^" + excluded + "] free-text-0");
        alts.push_back("\"\"");
        c.add_rule("free-text-" + std::to_string(i), join(alts, " | "));
    }
    return "free-text-0";
}

}  // namespace

std::string tool_call_gbnf(const std::string& tools_json, const ToolGrammarOptions& opt, std::string& err) {
    JsonValue tools;
    if (!read_json(tools_json, tools, err)) return "";
    if (!tools.is(Type::Array) || tools.items.empty()) {
        err = "tools: expected a non-empty array";
        return "";
    }
    Converter c(nullptr);
    std::string text_rule;
    std::vector<std::string> fns;
    for (size_t ti = 0; ti < tools.items.size(); ++ti) {
        const JsonValue* tool = &tools.items[ti];
        if (const JsonValue* f = tool->is(Type::Object) ? tool->get("function") : nullptr) tool = f;
        const JsonValue* nm = tool->is(Type::Object) ? tool->get("name") : nullptr;
        if (!nm || !nm->is(Type::String) || nm->s.empty()) {
            err = "tools: every tool needs a name";
            return "";
        }
        const JsonValue* params = tool->get("parameters");
        if (!params) params = tool->get("input_schema");
        c.root_ = params;
        const JsonValue* props = params && params->is(Type::Object) ? params->get("properties") : nullptr;
        std::set<std::string> required;
        if (const JsonValue* r = params && params->is(Type::Object) ? params->get("required") : nullptr) {
            for (const JsonValue& n : r->items) if (n.is(Type::String)) required.insert(n.s);
        }
        std::string fn_id = "fn" + std::to_string(ti);
        std::vector<std::string> items;
        std::vector<bool> is_req;
        if (props && props->is(Type::Object)) {
            for (const auto& m : props->members) {
                const JsonValue* ps = c.deref(&m.second);
                const JsonValue* ty = ps && ps->is(Type::Object) ? ps->get("type") : nullptr;
                const JsonValue* en = ps && ps->is(Type::Object) ? ps->get("enum") : nullptr;
                bool raw = false;
                std::string value;
                bool typed = ps && ps->is(Type::Object) &&
                             (ps->get("anyOf") || ps->get("oneOf") || ps->get("allOf") || ps->get("const") ||
                              ps->get("properties") || ps->get("items"));
                if (en && en->is(Type::Array) && !en->items.empty()) {
                    bool all_str = std::all_of(en->items.begin(), en->items.end(),
                                               [](const JsonValue& v) { return v.is(Type::String); });
                    if (all_str) {
                        std::vector<std::string> vals;
                        for (const JsonValue& v : en->items) vals.push_back(literal(v.s));
                        value = c.add_rule(fn_id + "-" + m.first + "-enum", "(" + join(vals, " | ") + ")");
                        raw = true;
                    }
                } else if (ty && ty->is(Type::String) && ty->s == "string") {
                    raw = true;
                } else if (ty && ty->is(Type::Array) &&
                           std::any_of(ty->items.begin(), ty->items.end(),
                                       [](const JsonValue& v) { return v.is(Type::String) && v.s == "string"; })) {
                    raw = true;
                } else if (!ty && !typed) {
                    raw = true;
                }
                if (raw && value.empty()) {
                    if (text_rule.empty()) text_rule = add_free_text(c, "</parameter>");
                    value = text_rule;
                }
                if (!raw) value = c.visit(*ps, fn_id + "-" + m.first);
                if (value.empty()) {
                    err = "tools: " + nm->s + "." + m.first + ": " + (c.errors.empty() ? "no rule" : c.errors.back());
                    return "";
                }
                items.push_back(literal("<parameter=" + m.first + ">\n") + " " + value + " " +
                                literal("\n</parameter>\n"));
                is_req.push_back(required.count(m.first) > 0);
            }
        }
        std::string body;
        if (opt.params_in_order) {
            for (size_t i = 0; i < items.size(); ++i) body += (is_req[i] ? items[i] : "(" + items[i] + ")?") + " ";
        } else if (!items.empty()) {
            std::string param = c.add_rule(fn_id + "-param", join(items, " | "));
            body = "(" + param + ")* ";
        }
        fns.push_back(c.add_rule(fn_id, literal("<function=" + nm->s + ">\n") + " " + body + literal("</function>")));
    }
    c.add_rule("call", literal("<tool_call>\n") + " (" + join(fns, " | ") + ") " + literal("\n</tool_call>"));
    c.add_rule("root", opt.multiple_calls ? "call ([ \\t\\n]* call)*" : "call");
    return c.format();
}

RequestGrammar grammar_from_request(const std::string& grammar_json, std::string& err) {
    RequestGrammar out;
    JsonValue req;
    if (!read_json(grammar_json, req, err)) return out;
    const JsonValue* kind = req.is(Type::Object) ? req.get("kind") : nullptr;
    if (!kind || !kind->is(Type::String)) {
        err = "grammar: missing kind";
        return out;
    }
    std::string gbnf;
    if (kind->s == "json_object") {
        gbnf = json_schema_to_gbnf(R"({"type":"object"})", err);
    } else if (kind->s == "json_schema") {
        const JsonValue* sch = req.get("schema");
        if (!sch) {
            err = "grammar: json_schema needs a schema";
            return out;
        }
        gbnf = json_schema_to_gbnf(sch->dump(), err);
    } else if (kind->s == "tools") {
        const JsonValue* tools = req.get("tools");
        if (!tools || !tools->is(Type::Array)) {
            err = "grammar: tools needs an array";
            return out;
        }
        std::string only;   // a named choice
        if (const JsonValue* tc = req.get("tool_choice")) {
            if (tc->is(Type::String) && tc->s == "none") return out;
            if (tc->is(Type::Object)) {
                const JsonValue* fn = tc->get("function");
                const JsonValue* nm = fn && fn->is(Type::Object) ? fn->get("name") : tc->get("name");
                if (nm && nm->is(Type::String)) only = nm->s;
            }
        }
        JsonValue list = *tools;
        if (!only.empty()) {
            list.items.clear();
            for (const JsonValue& t : tools->items) {
                const JsonValue* f = t.is(Type::Object) && t.get("function") ? t.get("function") : &t;
                const JsonValue* nm = f->is(Type::Object) ? f->get("name") : nullptr;
                if (nm && nm->is(Type::String) && nm->s == only) list.items.push_back(t);
            }
        }
        gbnf = tool_call_gbnf(list.dump(), ToolGrammarOptions{}, err);
        out.triggers.words = {tool_call_trigger()};
    } else {
        err = "grammar: unknown kind '" + kind->s + "'";
        return out;
    }
    if (gbnf.empty()) return out;
    out.grammar = Grammar::parse(gbnf, err);
    return out;
}

}  // namespace basalt::program::grammar
