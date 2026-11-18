/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Synthetic instant point cloud: dense angular beams over mixed geometry with float32 compaction.
#ifndef BENCH_WORKLOADS_SENSOR_H
#define BENCH_WORKLOADS_SENSOR_H
#include "../fixture.h"

#include <array>
#include <vector>

namespace bench
{
void    sensorBeams(std::vector<std::array<float, 3>>& local, size_t count);
void    sensorPose(Fixture& f, const std::vector<std::array<float, 3>>& local);
Fixture sensorFixture(const Options& o);
struct PointCloud
{
    std::vector<uint8_t>              mask;
    std::vector<std::array<float, 3>> xyz;
    std::array<size_t, 5>             familyHits{};
    double                            checksum = 0;
    explicit PointCloud(size_t count) : mask(count) { xyz.reserve(count); }
    void qualify(const Fixture& f, const Results& result)
    {
        require(result.size() == f.queries.size(), "point-cloud row count mismatch");
        familyHits.fill(0);
        for (size_t i = 0; i < result.size(); ++i)
        {
            const auto& hits = result[i].hits;
            require(hits.size() <= 1, "point-cloud closest ray produced multiple hits");
            if (hits.empty())
                continue;
            const auto& h = hits[0];
            require(h.id >= 1 && h.id <= f.objects.size(), "point-cloud collider identity out of range");
            require(std::isfinite(h.distance) && h.distance >= f.queries[i].nearDistance &&
                        h.distance <= f.queries[i].farDistance,
                    "point-cloud distance outside submitted range");
            for (double component : h.position)
                require(std::isfinite(float(component)), "point-cloud position is not finite float32");
            ++familyHits[size_t(h.id - 1)];
        }
    }
    // Timed caller work: reusable membership mask, compact float32 XYZ and output consumption.
    // Qualification and family accounting run separately outside steady timing.
    void compact(const Results& result)
    {
        mask.resize(result.size());
        xyz.clear();
        double sum = 0;
        for (size_t i = 0; i < result.size(); ++i)
        {
            const auto& hits = result[i].hits;
            mask[i]          = uint8_t(!hits.empty());
            sum += double(i + 1) * double(mask[i]);
            if (hits.empty())
                continue;
            std::array<float, 3> point{};
            for (size_t axis = 0; axis < 3; ++axis)
            {
                point[axis] = float(hits[0].position[axis]);
                sum += double(point[axis]) * double(axis + 1);
            }
            xyz.push_back(point);
        }
        checksum += sum + double(xyz.size());
    }
};
void compareClouds(const Fixture& f, const Results& a, const Results& b, const PointCloud& ac, const PointCloud& bc);
void sensorParity(const Options& o, Fixture& f);
} // namespace bench
#endif
