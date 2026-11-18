/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Camera-style coherent ray batches: one pinhole image of rays from a single eye.
// `camera-mesh` with zero objects is the empty scene that isolates the fixed per-ray cost;
// `camera-box` reports one box as a bounds candidate, the per-candidate cost without exact
// geometry; `camera-mesh` with one object traces a closed 69,136-triangle blob, about the
// triangle count of a scanned bunny, and with more objects instances it in layers of 16 one
// behind the other, so most rays cross several objects' bounds.
#include "camera.h"

#include <cmath>

namespace bench
{
namespace
{
constexpr double kPi = 3.14159265358979323846;

// A closed latitude/longitude blob with outward winding, so a single-sided mesh shows its
// front faces to the camera. Vertices are quantized once to float for both SDKs.
Mesh blob()
{
    constexpr uint32_t rings = 150, segments = 232;
    Mesh               mesh;
    const auto         point = [](double theta, double phi)
    {
        const double r = 1 + .12 * std::sin(5 * theta) * std::cos(4 * phi) + .04 * std::sin(11 * phi + 3 * theta);
        return std::array<float, 3>{float(r * std::sin(theta) * std::cos(phi)),
                                    float(r * std::sin(theta) * std::sin(phi)), float(r * std::cos(theta))};
    };
    mesh.vertices.push_back(point(0, 0));
    for (uint32_t i = 1; i < rings; ++i)
        for (uint32_t j = 0; j < segments; ++j)
            mesh.vertices.push_back(point(kPi * i / rings, 2 * kPi * j / segments));
    mesh.vertices.push_back(point(kPi, 0));
    const uint32_t south = uint32_t(mesh.vertices.size() - 1);
    const auto     ring  = [](uint32_t i, uint32_t j) { return 1 + (i - 1) * segments + j % segments; };
    for (uint32_t j = 0; j < segments; ++j)
    {
        mesh.triangles.push_back({0, ring(1, j), ring(1, j + 1)});
        for (uint32_t i = 1; i + 1 < rings; ++i)
        {
            const uint32_t a = ring(i, j), b = ring(i, j + 1), c = ring(i + 1, j), d = ring(i + 1, j + 1);
            mesh.triangles.push_back({a, c, b});
            mesh.triangles.push_back({b, c, d});
        }
        mesh.triangles.push_back({south, ring(rings - 1, j + 1), ring(rings - 1, j)});
    }
    return mesh;
}
} // namespace

bool cameraWorkload(const Options& o) { return o.workload == "camera-mesh" || o.workload == "camera-box"; }

Fixture cameraFixture(const Options& o)
{
    Fixture    f;
    const bool box     = o.workload == "camera-box";
    f.kind             = QueryKind::rays;
    f.boundsCandidates = box;
    if (!box)
        f.meshes.push_back(blob());
    // Named limitation (Fixture::approximateBaselineContacts): PhysX's float instance poses put
    // its contacts up to 1.5e-4 m off the analytic face distance 12 m away on the rotated
    // instances, so the layered scene compares the baseline on identity and distance within
    // this tolerance; the parity record reports the installed paths' own analytic error.
    if (o.objects > 1)
    {
        f.approximateBaselineContacts = true;
        f.contactTolerance            = 5e-4;
    }
    // Layers of 4 x 4 instances 2.6 m apart, each layer 3 m behind the previous one.
    for (size_t i = 0; i < o.objects; ++i)
    {
        const size_t cell = i % 16;
        const Vec    position =
            o.objects == 1 ? Vec{0, 0, 0}
                              : Vec{(double(cell % 4) - 1.5) * 2.6, double(i / 16) * 3, (double(cell / 4) - 1.5) * 2.6};
        // Boxes stay axis-aligned: a bounds candidate is compared with PhysX's box itself.
        f.objects.push_back(
            {i + 1, i + 1, false, position, i % 2 && !box ? rotation(.25f) : identity, 1, box ? 0u : 1u});
    }
    // A square image; the box sits nearer so it covers about as much of it as the blob, and
    // the camera steps back to see a whole layer.
    const size_t side = size_t(std::sqrt(double(o.queries)));
    require(side * side == o.queries, "camera queries must be a perfect square pixel count");
    const double distance = box ? 1.4 : o.objects > 1 ? 12 : 3.2, halfField = .45;
    for (size_t i = 0; i < o.queries; ++i)
    {
        const double u = (2 * (double(i % side) + .5) / double(side) - 1) * halfField;
        const double v = (2 * (double(i / side % side) + .5) / double(side) - 1) * halfField;
        const double n = std::sqrt(u * u + 1 + v * v);
        Query        q;
        q.p            = {0, -distance, .05};
        q.direction    = {float(u / n), float(1 / n), float(v / n)};
        q.nearDistance = 0;
        q.farDistance  = distance + 20;
        f.queries.push_back(q);
    }
    return f;
}
} // namespace bench
