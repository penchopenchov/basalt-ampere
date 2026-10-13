// src/kernels/cpu/cpu_quota.cpp - the CPU quota of the process's cgroup.  See cpu_quota.hpp.
#include "basalt/kernels/cpu/cpu_quota.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace basalt::kernels::cpu {

namespace {

namespace fs = std::filesystem;

// the group's directory under `top`, or `top` itself when the group is not visible there
fs::path visible_group(const fs::path& top, const std::string& rel) {
    const std::string sub = rel.substr(rel.rfind('/', 0) == 0 ? 1 : 0);
    if (sub.empty()) return top;
    auto path = (top / sub).lexically_normal();
    std::error_code ec;
    if (path.string().rfind(top.string(), 0) != 0 || !fs::is_directory(path, ec)) return top;
    return path;
}

// Walks from the group up to `top`, keeping the smallest quota; `read` fills a group's quota in CPUs (0: none).
template <class Read>
double tightest(const fs::path& top, const fs::path& start, Read read) {
    double best = 0;
    for (auto path = start; path.string().rfind(top.string(), 0) == 0; path = path.parent_path()) {
        const double q = read(path);
        if (q > 0 && (best == 0 || q < best)) best = q;
        if (path == top) break;
    }
    return best;
}

}  // namespace

double cgroup_cpu_quota(const std::string& self_cgroup, const std::string& cgroup_root) {
    const fs::path root(cgroup_root);
    std::string v2_path, v1_path;
    bool v2 = false;
    if (std::ifstream groups(self_cgroup); groups) {
        for (std::string line; std::getline(groups, line);) {
            if (line.rfind("0::/", 0) == 0) {
                v2 = true;
                v2_path = line.substr(4);
                continue;
            }
            // v1: "N:controller[,controller]:/path"
            const size_t c1 = line.find(':'), c2 = c1 == std::string::npos ? c1 : line.find(':', c1 + 1);
            if (c2 == std::string::npos) continue;
            std::istringstream ctl(line.substr(c1 + 1, c2 - c1 - 1));
            for (std::string c; std::getline(ctl, c, ',');)
                if (c == "cpu") v1_path = line.substr(c2 + 1);
        }
    }
    std::error_code ec;
    if (!v1_path.empty()) {
        // The cpu controller is mounted as "cpu,cpuacct" with a "cpu" link, or as plain "cpu".
        fs::path top = root / "cpu";
        if (!fs::is_directory(top, ec)) top = root / "cpu,cpuacct";
        return tightest(top, visible_group(top, v1_path), [](const fs::path& p) {
            std::ifstream qf(p / "cpu.cfs_quota_us"), pf(p / "cpu.cfs_period_us");
            long long quota = 0, period = 0;
            // -1: no quota
            if (qf >> quota && pf >> period && quota > 0 && period > 0) return (double) quota / (double) period;
            return 0.0;
        });
    }
    if (v2) {
        return tightest(root, visible_group(root, v2_path), [](const fs::path& p) {
            std::ifstream f(p / "cpu.max");
            std::string quota;
            long long period = 0;
            // "max 100000" is no quota
            if (f >> quota >> period && quota != "max" && period > 0) {
                try {
                    size_t used = 0;
                    const long long q = std::stoll(quota, &used);
                    if (used == quota.size() && q > 0) return (double) q / (double) period;
                } catch (...) {}
            }
            return 0.0;
        });
    }
    return 0;
}

}  // namespace basalt::kernels::cpu
