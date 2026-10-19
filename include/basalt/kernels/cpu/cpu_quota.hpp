#pragma once

#include <string>

namespace basalt::kernels::cpu {

/// The CPU time this process may use under a cgroup quota, in CPUs (quota / period), or 0 when no group limits it.
/// The tightest quota of the process's group and its ancestors counts: v2 `cpu.max`, v1 `cpu.cfs_quota_us` over
/// `cpu.cfs_period_us`.  A group whose path is not visible under `cgroup_root` (a container's own cgroup namespace)
/// is read at the controller's root, as for the memory limit.  The parameters are for tests; an empty one is the system's
/// own (platform::self_cgroup_path, platform::cgroup_root), and a system with none has no quota.
double cgroup_cpu_quota(const std::string& self_cgroup = "", const std::string& cgroup_root = "");

}  // namespace basalt::kernels::cpu
