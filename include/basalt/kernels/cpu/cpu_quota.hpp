#pragma once

#include <string>

namespace basalt::kernels::cpu {

/// The CPU time this process may use under a cgroup quota, in CPUs (quota / period), or 0 when no group limits it.
/// The tightest quota of the process's group and its ancestors counts: v2 `cpu.max`, v1 `cpu.cfs_quota_us` over
/// `cpu.cfs_period_us`.  A group whose path is not visible under `cgroup_root` (a container's own cgroup namespace)
/// is read at the controller's root, as for the memory limit.  The parameters are for tests.
double cgroup_cpu_quota(const std::string& self_cgroup = "/proc/self/cgroup",
                        const std::string& cgroup_root = "/sys/fs/cgroup");

}  // namespace basalt::kernels::cpu
