/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Shared aliases, constants, error helpers and command-line options of the benchmark.
#ifndef BENCH_COMMON_H
#define BENCH_COMMON_H
#include <PxPhysicsAPI.h>
#include <ovspatialquery/ovspatialquery.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace bench
{
using Clock = std::chrono::steady_clock;
using Vec   = std::array<double, 3>;
using Quat  = std::array<float, 4>;
using physx::PxVec3;
constexpr Quat     identity{0, 0, 0, 1};
constexpr Vec      boxHalf{0.5, 0.375, 0.5};
constexpr double   sphereRadius     = 0.5;
constexpr uint64_t hostBudget       = UINT64_C(1) << 30;
constexpr uint64_t deviceByteBudget = UINT64_C(1) << 30; // The warehouse demo's CUDA device budget.

inline double elapsed(Clock::time_point begin)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
}
inline void require(bool ok, std::string_view message)
{
    if (!ok)
        throw std::runtime_error(std::string(message));
}
inline void check(ovspatialquery_result_t result, const char* action)
{
    if (result.status == OVSPATIALQUERY_API_SUCCESS)
        return;
    const auto error = ovspatialquery_get_last_error();
    throw std::runtime_error(std::string(action) + " status=" + std::to_string(result.status) + " " +
                             std::string(error.ptr ? error.ptr : "", static_cast<size_t>(error.length)));
}
inline Quat               rotation(float angle) { return {0, 0, std::sin(angle / 2), std::cos(angle / 2)}; }
inline PxVec3             px(const Vec& v) { return {float(v[0]), float(v[1]), float(v[2])}; }
inline PxVec3             meshVertex(const std::array<float, 3>& v) { return {v[0], v[1], v[2]}; }
inline physx::PxTransform pose(const Vec& p, const Quat& q)
{
    return {px(p), physx::PxQuat(float(q[0]), float(q[1]), float(q[2]), float(q[3]))};
}
struct Options
{
    std::string backend = "parity", workload = "rays", subset = "all";
    std::string warehouseMeshes = "_build/benchmarks/warehouse-racks.bin";
    size_t      meshSide = 32, uniqueMeshes = 0;
    size_t      objects = 100, queries = 64, frames = 10, warmup = 3, updatePercent = 0, iterations = 1;
    size_t      zoneHitCapacity = 0; // Zero retains the worst-case population-sized capacity.
    size_t      radiusGroups = 0, reusePlans = 1, actorHitCapacity = 32;
    // Registers every mesh with source-face columns that repeat the submitted numbering, so
    // hits and parity are unchanged and only the columns' cost differs (ADR-0080).
    size_t meshSourceFaces    = 0;
    size_t pairToleranceMilli = 0,
           pairCapacity       = 0; // Pairs: contact tolerance in mm; zero selects the default capacity.
};
Options options(int argc, char** argv);
} // namespace bench
#endif
