/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Sensor beam lattice, fixture, point-cloud compaction and parity.
#include "sensor.h"

#include "../installed_adapter.h"

#include <cmath>
#include <iostream>
#include <memory>
#include <string>

namespace bench
{
// Six coherent angular patches: five interior hit patches and one empty patch.
// This controlled geometric workload is not a physical sensor/FOV replay.
constexpr Vec sensorOrigin{2, -3, 1};
void          sensorBeams(std::vector<std::array<float, 3>>& local, size_t count)
{
    local.resize(count);
    constexpr size_t columns = 128;
    for (size_t i = 0; i < count; ++i)
    {
        const size_t tile  = i * 6 / count;
        const size_t begin = (tile * count + 5) / 6, end = ((tile + 1) * count + 5) / 6;
        const size_t within = i - begin, rows = (end - begin + columns - 1) / columns;
        const double yaw   = -.75 + .3 * double(tile) + .024 * ((double(within % columns) + .5) / columns - .5);
        const double pitch = .024 * ((double(within / columns) + .5) / double(rows) - .5);
        local[i]           = {float(std::sin(yaw) * std::cos(pitch)), float(std::sin(pitch)),
                              float(std::cos(yaw) * std::cos(pitch))};
    }
}
void sensorPose(Fixture& f, const std::vector<std::array<float, 3>>& local)
{
    f.queries.resize(local.size());
    for (size_t i = 0; i < local.size(); ++i)
    {
        auto& q = f.queries[i];
        q.p     = sensorOrigin;
        // Exact 180-degree Z rotation, shared by geometry and beam transform.
        q.direction    = {-local[i][0], -local[i][1], local[i][2]};
        q.nearDistance = .25;
        q.farDistance  = 10;
    }
}
Fixture sensorFixture(const Options& o)
{
    Fixture f;
    f.kind       = QueryKind::rays;
    f.pointCloud = true;
    Mesh               patch;
    constexpr uint32_t side = 32;
    for (uint32_t y = 0; y <= side; ++y)
        for (uint32_t x = 0; x <= side; ++x)
            patch.vertices.push_back({-.75f + 1.5f * float(x) / side, -.75f + 1.5f * float(y) / side, 0});
    for (uint32_t y = 0; y < side; ++y)
        for (uint32_t x = 0; x < side; ++x)
        {
            const uint32_t a = y * (side + 1) + x, b = a + 1, c = a + side + 1, d = c + 1;
            patch.triangles.push_back({a, c, b});
            patch.triangles.push_back({b, c, d}); // Front faces point toward the sensor.
        }
    f.meshes.push_back(std::move(patch));
    f.convexes.emplace_back();
    for (float z : {-1.f, 1.f})
        for (float y : {-1.f, 1.f})
            for (float x : {-1.f, 1.f})
                f.convexes[0].push_back({x, y, z});
    for (size_t i = 0; i < 5; ++i)
    {
        const double angle = -.75 + .3 * double(i);
        // Quantized translations are identical in the float PhysX pose and double API pose.
        const Vec center{double(float(sensorOrigin[0] - 6 * std::sin(angle))), sensorOrigin[1],
                         double(float(sensorOrigin[2] + 6 * std::cos(angle)))};
        Object    object{i + 1, i + 1, i == 2, center, {0, 0, 1, 0}};
        object.mesh    = i == 0 ? 1 : 0;
        object.capsule = i == 3;
        object.convex  = i == 4 ? 1 : 0;
        f.objects.push_back(object);
    }
    std::vector<std::array<float, 3>> local;
    sensorBeams(local, o.queries);
    sensorPose(f, local);
    return f;
}
void compareClouds(const Fixture& f, const Results& a, const Results& b, const PointCloud& ac, const PointCloud& bc)
{
    require(ac.mask.size() == bc.mask.size(), "point-cloud mask size mismatch");
    for (size_t i = 0; i < ac.mask.size(); ++i)
        require(ac.mask[i] == bc.mask[i], "point-cloud membership parity beam " + std::to_string(i) + " expected=" +
                                              std::to_string(ac.mask[i]) + " actual=" + std::to_string(bc.mask[i]));
    require(ac.xyz.size() == bc.xyz.size(), "point-cloud point count parity failed");
    for (size_t i = 0; i < f.queries.size(); ++i)
    {
        if (!ac.mask[i])
            continue;
        const auto& ah = a[i].hits[0];
        const auto& bh = b[i].hits[0];
        require(ah.id == bh.id && near(ah.distance, bh.distance), "point-cloud collider/distance parity failed");
        for (size_t axis = 0; axis < 3; ++axis)
            require(near(ah.position[axis], bh.position[axis]), "point-cloud contact parity failed");
    }
    for (size_t i = 0; i < ac.xyz.size(); ++i)
        for (size_t axis = 0; axis < 3; ++axis)
            require(near(ac.xyz[i][axis], bc.xyz[i][axis]), "point-cloud float32 output parity failed");
}
void sensorParity(const Options& o, Fixture& f)
{
    Direct  direct(f);
    Spatial spatial(f);
    spatial.makePlan(f);
    // The CUDA plan, where one can be created, is held to the CPU plan's cloud in every probe.
    std::string                         cudaParity;
    std::unique_ptr<Spatial::BatchPlan> cudaPlan;
    if (cudaPlansAvailable(cudaParity))
    {
        cudaPlan   = std::make_unique<Spatial::BatchPlan>(spatial, f, f.queries.size(), DLDevice{kDLCUDA, 0});
        cudaParity = "parity";
    }
    else
        cudaParity = "unavailable: " + cudaParity;
    const bool            scalarAnchor = o.queries <= 16384;
    std::array<size_t, 5> familyHits{};
    size_t                pointCount = 0;
    auto                  probe      = [&](const char* label, bool coverage)
    {
        Results expected(f.queries.size()), batch(f.queries.size());
        direct.query(f, expected);
        spatial.defaultPlan->pack(f);
        spatial.batch(f, batch, spatial.ordinal);
        PointCloud a(f.queries.size()), b(f.queries.size());
        a.qualify(f, expected);
        a.compact(expected);
        b.qualify(f, batch);
        b.compact(batch);
        if (scalarAnchor)
        {
            Results scalar(f.queries.size());
            spatial.scalar(f, scalar, spatial.ordinal);
            PointCloud c(f.queries.size());
            c.qualify(f, scalar);
            c.compact(scalar);
            try
            {
                compareClouds(f, expected, scalar, a, c);
                compareClouds(f, scalar, batch, c, b);
            }
            catch (const std::exception& error)
            {
                throw std::runtime_error(std::string(label) + " scalar anchor: " + error.what());
            }
        }
        try
        {
            compareClouds(f, expected, batch, a, b);
        }
        catch (const std::exception& error)
        {
            throw std::runtime_error(std::string(label) + " PhysX/batch: " + error.what());
        }
        if (cudaPlan)
        {
            Results gpu(f.queries.size());
            cudaPlan->pack(f);
            cudaPlan->batch(f, gpu, spatial.ordinal);
            PointCloud g(f.queries.size());
            g.qualify(f, gpu);
            g.compact(gpu);
            try
            {
                compareClouds(f, batch, gpu, b, g);
            }
            catch (const std::exception& error)
            {
                throw std::runtime_error(std::string(label) + " CPU batch/CUDA: " + error.what());
            }
        }
        if (coverage)
        {
            require(a.xyz.size() < f.queries.size(), "sensor fixture has no misses");
            for (size_t hits : a.familyHits)
                require(hits > 0, "sensor fixture failed to exercise a geometry family");
            familyHits = a.familyHits;
            pointCount = a.xyz.size();
        }
        return a.xyz.size();
    };
    probe("nominal", true);
    const auto original = f.queries;
    for (auto& q : f.queries)
        q.bits = 2; // All five objects carry bits=1.
    require(probe("filter-exclusion", false) == 0, "sensor filter-exclusion diagnostic should miss");
    f.queries = original;
    for (auto& q : f.queries)
        q.farDistance = .5;
    require(probe("far-clip", false) == 0, "sensor far-clip diagnostic should miss");
    f.queries = original;
    for (auto& q : f.queries)
        q.nearDistance = 9;
    require(probe("near-clip", false) == 0, "sensor near-clip diagnostic should miss");
    f.queries.resize(1);
    f.queries[0]   = Query{};
    f.queries[0].p = f.objects[0].p;
    f.queries[0].p[2] += 2;
    f.queries[0].direction = {0, 0, -1};
    require(probe("one-sided mesh back-face", false) == 0, "sensor one-sided mesh back-face diagnostic should miss");
    f.queries = original;
    std::cout << "{\"backend\":\"parity\",\"workload\":\"sensor-point-cloud\",\"parity\":true,"
                 "\"scalar_anchor\":"
              << (scalarAnchor ? "true" : "false") << ",\"cuda_parity\":\"" << cudaParity
              << "\",\"queries\":" << o.queries << ",\"point_count\":" << pointCount
              << ",\"miss_count\":" << o.queries - pointCount
              << ",\"family_order\":[\"triangle_mesh\",\"box\",\"sphere\",\"capsule\",\"convex\"],\"family_hits\":[";
    for (size_t i = 0; i < familyHits.size(); ++i)
        std::cout << (i ? "," : "") << familyHits[i];
    std::cout << "],\"diagnostics\":[\"filter_exclusion\",\"far_clip\",\"near_clip\",\"one_sided_mesh_back_face\"],"
                 "\"synthetic_geometric_analogue\":true}\n";
}
} // namespace bench
