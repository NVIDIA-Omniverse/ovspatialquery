/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Loads the warehouse rack mesh export and builds its ray/sweep/overlap query lattices.
#include "warehouse.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

namespace bench
{
namespace
{
template <class T>
T readValue(std::istream& in)
{
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    return value;
}
double frac(double x) { return x - std::floor(x); }
} // namespace

bool warehouseWorkload(const Options& o) { return o.workload.rfind("warehouse-", 0) == 0; }

Fixture warehouseFixture(const Options& o)
{
    const std::string missing = o.warehouseMeshes + "; run benchmarks/warehouse_meshes.py";
    std::ifstream     in(o.warehouseMeshes, std::ios::binary);
    require(bool(in), "warehouse meshes not found at " + missing);
    char magic[8];
    in.read(magic, sizeof(magic));
    require(bool(in) && std::string(magic, sizeof(magic)) == "OVSQWH01",
            "warehouse meshes at " + missing + " (unexpected header)");
    const uint32_t meshCount      = readValue<uint32_t>(in);
    const uint32_t instanceCount  = readValue<uint32_t>(in);
    const uint32_t candidateCount = readValue<uint32_t>(in);

    Fixture f;
    f.meshes.resize(meshCount);
    for (uint32_t m = 0; m < meshCount; ++m)
    {
        auto&          mesh          = f.meshes[m];
        const uint32_t vertexCount   = readValue<uint32_t>(in);
        const uint32_t triangleCount = readValue<uint32_t>(in);
        mesh.doubleSided             = false;
        mesh.vertices.resize(vertexCount);
        in.read(reinterpret_cast<char*>(mesh.vertices.data()), std::streamsize(sizeof(float) * 3 * vertexCount));
        mesh.triangles.resize(triangleCount);
        in.read(reinterpret_cast<char*>(mesh.triangles.data()), std::streamsize(sizeof(uint32_t) * 3 * triangleCount));
    }
    std::vector<uint32_t> instanceMesh(instanceCount);
    std::vector<Vec>      instanceTranslation(instanceCount);
    for (uint32_t i = 0; i < instanceCount; ++i)
    {
        instanceMesh[i] = readValue<uint32_t>(in);
        Vec t{};
        in.read(reinterpret_cast<char*>(t.data()), std::streamsize(sizeof(double) * 3));
        instanceTranslation[i] = t;
        readValue<uint8_t>(in); // Kind (rack frame vs stored cargo) is provenance only; not used here.
    }
    std::vector<Vec> candidates(candidateCount);
    for (uint32_t c = 0; c < candidateCount; ++c)
    {
        Vec center{};
        in.read(reinterpret_cast<char*>(center.data()), std::streamsize(sizeof(double) * 3));
        candidates[c] = center;
    }
    require(bool(in), "warehouse meshes at " + missing + " (truncated file)");

    require(o.objects == instanceCount, "warehouse fixture has " + std::to_string(instanceCount) +
                                            " instances; pass --objects " + std::to_string(instanceCount));
    require(o.updatePercent == 0, "warehouse racks are static");

    // The installed CPU path returns PhysX float mesh contacts verbatim; on these meshes they sit
    // up to 11 mm off the ray (1024-ray lattice, query 357) while direct PhysX scene contacts differ
    // from them by millimetres. The parity JSON reports each side's largest deviation.
    f.approximateBaselineContacts = true;
    f.contactTolerance            = 2e-2;
    f.objects.reserve(instanceCount);
    for (uint32_t i = 0; i < instanceCount; ++i)
        f.objects.push_back({uint64_t(i) + 1, uint64_t(i) + 1, false, instanceTranslation[i], identity, 1,
                             uint64_t(instanceMesh[i]) + 1});

    if (o.workload == "warehouse-rays")
        f.kind = QueryKind::rays;
    else if (o.workload == "warehouse-sweeps" || o.workload == "warehouse-box-sweeps")
    {
        f.kind   = o.workload == "warehouse-box-sweeps" ? QueryKind::boxSweep : QueryKind::sweep;
        f.radius = 0.5;
        // Direct PhysX sphere sweeps report another face of the object up to 1e-3 later
        // (Fixture::sweepRelativeTolerance). PhysX box distances on these racks came up to 1.95e-4
        // early against an independent solve over 10,914 contacts of the 16,384-row lattice
        // (Fixture::boxSweepBaselineEarlyTolerance).
        if (f.kind == QueryKind::boxSweep)
        {
            f.half                           = {0.25, 0.2, 0.2};
            f.boxSweepBaselineEarlyTolerance = 5e-4;
        }
        if (f.kind == QueryKind::sweep)
            f.sweepRelativeTolerance = 1e-3;
    }
    else
    {
        f.kind            = QueryKind::zones;
        f.half            = {0.6, 0.7, 0.5};
        f.zoneHitCapacity = o.zoneHitCapacity;
    }

    const size_t n = o.queries;
    f.queries.resize(n);
    if (f.kind == QueryKind::rays || f.kind == QueryKind::sweep || f.kind == QueryKind::boxSweep)
    {
        const size_t width = std::min(n, n <= 1024 ? size_t(64) : size_t(128));
        require(n % width == 0, "warehouse ray count must be a multiple of its lattice width");
        const size_t rows = n / width;
        const double pi   = std::acos(-1.0);
        for (size_t k = 0; k < n; ++k)
        {
            const size_t i = k / width, j = k % width;
            const double az = -pi + 2 * pi * double(j) / double(width);
            const double el = rows > 1 ? -0.4 + 1.0 * double(i) / double(rows - 1) : -0.4;
            const double dx = std::sin(az), dy = std::tan(el), dz = -std::cos(az);
            const double length = std::sqrt(dx * dx + dy * dy + dz * dz);
            auto&        q      = f.queries[k];
            q.p                 = {0, 1.8, 3.0};
            q.nearDistance      = 0;
            q.farDistance       = 15;
            q.bits              = 0;
            q.direction         = {float(dx / length), float(dy / length), float(dz / length)};
            if (f.kind == QueryKind::boxSweep)
                q.q = k % 2 ? rotation(.25f) : identity;
        }
    }
    else
    {
        // Rack centers, in first-appearance order, drive the fallback scatter once candidates run out.
        std::vector<Vec> rackCenters;
        for (const auto& t : instanceTranslation)
        {
            const bool seen = std::any_of(rackCenters.begin(), rackCenters.end(),
                                          [&](const Vec& c) { return c[0] == t[0] && c[2] == t[2]; });
            if (!seen)
                rackCenters.push_back(t);
        }
        require(rackCenters.size() == 6, "warehouse overlaps expect six distinct rack placements");
        for (size_t k = 0; k < n; ++k)
        {
            auto& q = f.queries[k];
            q.q     = identity;
            q.bits  = 0;
            if (k < candidates.size())
                q.p = candidates[k];
            else
            {
                const size_t r  = k % 6;
                const double cx = rackCenters[r][0], cz = rackCenters[r][2];
                const double t = frac(double(k) * 0.6180339887);
                const double u = frac(double(k) * 0.7548776662);
                const double v = frac(double(k) * 0.5698402910);
                q.p            = {cx - 0.9 + 1.8 * t, 0.6 + 2.2 * u, cz - 2.1 + 4.2 * v};
            }
        }
    }
    return f;
}

size_t warehouseInstancedTriangles(const Fixture& f)
{
    size_t total = 0;
    for (const auto& ob : f.objects)
        if (ob.mesh)
            total += f.meshes[size_t(ob.mesh - 1)].triangles.size();
    return total;
}
} // namespace bench
