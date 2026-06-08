// MiniYaml.h —— 极简 YAML 子集解析（无外部依赖）
// 仅支持本项目所需的结构：
//   - 顶层 `key: value` 标量
//   - 一个 `sequences:` 列表，每个条目是若干缩进的 `key: value`，条目以 `- ` 开头
// 不支持嵌套 map、流式 [..]/{..}、多文档、锚点等复杂特性。
// # 之后视为注释。值两端引号会被去掉。
#pragma once
#include <string>
#include <vector>
#include <map>

namespace myaml {

using KV = std::map<std::string, std::string>;

struct Document {
    KV globals;                 // 顶层标量
    std::vector<KV> sequences;  // sequences: 列表，每项一组键值
    bool ok = false;
    std::string error;
};

// 从文件解析
Document parseFile(const std::string& path);
// 从字符串解析
Document parseString(const std::string& text);

// 取值辅助（带默认值）
std::string getStr(const KV& kv, const std::string& key, const std::string& def = "");
int         getInt(const KV& kv, const std::string& key, int def = 0);
double      getDouble(const KV& kv, const std::string& key, double def = 0.0);
bool        getBool(const KV& kv, const std::string& key, bool def = false);
bool        has(const KV& kv, const std::string& key);

} // namespace myaml
