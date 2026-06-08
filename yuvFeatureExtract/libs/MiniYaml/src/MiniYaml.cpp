#include "MiniYaml/MiniYaml.h"
#include <fstream>
#include <sstream>
#include <cctype>
#include <algorithm>

namespace myaml {

// --- 小工具 -----------------------------------------------------------------
static std::string stripComment(const std::string& line) {
    // 去掉未被引号包裹的 # 之后内容（本项目值不含 #，简单处理即可）
    bool inSingle = false, inDouble = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (c == '\'' && !inDouble) inSingle = !inSingle;
        else if (c == '"' && !inSingle) inDouble = !inDouble;
        else if (c == '#' && !inSingle && !inDouble) return line.substr(0, i);
    }
    return line;
}

static std::string rtrim(std::string s) {
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t' ||
                          s.back() == '\r' || s.back() == '\n')) s.pop_back();
    return s;
}
static size_t indentOf(const std::string& s) {
    size_t i = 0; while (i < s.size() && s[i] == ' ') ++i; return i;
}
static std::string trim(std::string s) {
    size_t a = 0; while (a < s.size() && std::isspace((unsigned char)s[a])) ++a;
    s = s.substr(a);
    return rtrim(s);
}
static std::string unquote(std::string s) {
    s = trim(s);
    if (s.size() >= 2 && ((s.front() == '"' && s.back() == '"') ||
                          (s.front() == '\'' && s.back() == '\'')))
        s = s.substr(1, s.size() - 2);
    return s;
}
// 把 "key: value" 拆成 key / value（第一个冒号）
static bool splitKV(const std::string& s, std::string& k, std::string& v) {
    size_t p = s.find(':');
    if (p == std::string::npos) return false;
    k = trim(s.substr(0, p));
    v = unquote(s.substr(p + 1));
    return !k.empty();
}

// --- 主解析 -----------------------------------------------------------------
Document parseString(const std::string& text) {
    Document doc;
    std::istringstream in(text);
    std::string raw;

    bool inSeq = false;        // 是否已进入 sequences:
    int  seqItemIndent = -1;   // 列表项键值的缩进基准
    KV*  cur = nullptr;        // 当前 sequence 条目

    int lineNo = 0;
    while (std::getline(in, raw)) {
        ++lineNo;
        std::string line = rtrim(stripComment(raw));
        if (trim(line).empty()) continue;

        size_t ind = indentOf(line);
        std::string content = trim(line);

        // 顶层键（缩进 0，且不是列表项）
        if (ind == 0 && content.rfind("- ", 0) != 0) {
            std::string k, v;
            if (!splitKV(content, k, v)) {
                doc.error = "第 " + std::to_string(lineNo) + " 行无法解析: " + content;
                return doc;
            }
            if (k == "sequences") {
                inSeq = true; cur = nullptr; seqItemIndent = -1;
            } else {
                inSeq = false; cur = nullptr;
                doc.globals[k] = v;
            }
            continue;
        }

        if (!inSeq) {
            doc.error = "第 " + std::to_string(lineNo) + " 行不在 sequences 中却有缩进: " + content;
            return doc;
        }

        // sequences 列表项
        if (content.rfind("- ", 0) == 0 || content == "-") {
            doc.sequences.emplace_back();
            cur = &doc.sequences.back();
            seqItemIndent = -1;
            std::string rest = (content == "-") ? "" : trim(content.substr(2));
            if (!rest.empty()) {
                std::string k, v;
                if (splitKV(rest, k, v)) (*cur)[k] = v;
            }
            continue;
        }

        // 列表项内部的后续键值（缩进大于 '-'）
        if (cur != nullptr) {
            if (seqItemIndent < 0) seqItemIndent = static_cast<int>(ind);
            std::string k, v;
            if (splitKV(content, k, v)) (*cur)[k] = v;
            continue;
        }

        doc.error = "第 " + std::to_string(lineNo) + " 行解析失败: " + content;
        return doc;
    }

    doc.ok = true;
    return doc;
}

Document parseFile(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        Document d; d.ok = false; d.error = "无法打开配置文件: " + path; return d;
    }
    std::stringstream ss; ss << f.rdbuf();
    return parseString(ss.str());
}

// --- 取值辅助 ---------------------------------------------------------------
bool has(const KV& kv, const std::string& key) { return kv.find(key) != kv.end(); }

std::string getStr(const KV& kv, const std::string& key, const std::string& def) {
    auto it = kv.find(key); return it == kv.end() ? def : it->second;
}
int getInt(const KV& kv, const std::string& key, int def) {
    auto it = kv.find(key); if (it == kv.end() || it->second.empty()) return def;
    try { return std::stoi(it->second); } catch (...) { return def; }
}
double getDouble(const KV& kv, const std::string& key, double def) {
    auto it = kv.find(key); if (it == kv.end() || it->second.empty()) return def;
    try { return std::stod(it->second); } catch (...) { return def; }
}
bool getBool(const KV& kv, const std::string& key, bool def) {
    auto it = kv.find(key); if (it == kv.end()) return def;
    std::string s = it->second;
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    if (s == "true" || s == "1" || s == "yes" || s == "on")  return true;
    if (s == "false"|| s == "0" || s == "no"  || s == "off") return false;
    return def;
}

} // namespace myaml
