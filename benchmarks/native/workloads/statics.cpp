/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Static scenes modelled on PhysX scene-query stress tests, where the query's reach is far larger
// than the distance to its first contact:
//   sea-box-sweeps-<length>, sea-sweeps-<length>: a flat grid of boxes; each query pair is one
//     diagonal sweep through the sea (first contact at about 12.2 m) and the same sweep above it
//     (no contact), both of the given length.
//   dense-mesh-box-sweeps, dense-mesh-sweeps: one closed, lumpy mesh of 130,560 triangles; sweeps
//     from a surrounding shell aimed through it, ten times its radius long.
//   dense-mesh-sphere-any, dense-mesh-sphere-overlaps: one closed, lumpy mesh of 70,312
//     triangles under spheres of radius 0.15 centred between 0.7 and 1.3 of its radius, so some
//     lie inside or outside it untouched; ANY, or every object touched.
//   level-box-sweeps, level-sweeps, level-box-overlaps, level-sphere-overlaps: contiguous terrain
//     tiles of four times --mesh-side cells (32,640 triangles at the default 32; one shared
//     definition). The sweeps graze the level: from 0.8 m up they descend 8 cm per metre along a
//     40 m path, so their path runs close to the terraced surface;
//     the overlaps are a box or sphere spanning most of the level, reporting every object it
//     overlaps; a tile wholly inside the shape touches it without a triangle test.
#include "statics.h"

#include "mesh.h"

#include <cmath>
#include <cstdlib>
#include <string>

namespace bench
{
namespace
{
double frac(double x) { return x - std::floor(x); }

bool startsWith(const std::string& text, const char* prefix) { return text.rfind(prefix, 0) == 0; }

std::array<float, 3> unitDirection(double x, double y, double z)
{
    const double length = std::sqrt(x * x + y * y + z * z);
    return {float(x / length), float(y / length), float(z / length)};
}

// A closed latitude-longitude sphere of radius about one with low-frequency lumps; front faces
// point outward. 256 rings by 256 segments give 130,560 triangles.
Mesh lumpySphere(uint32_t rings, uint32_t segments)
{
    const double pi = std::acos(-1.0);
    Mesh         mesh;
    const auto   vertex = [&](double theta, double phi)
    {
        const double r =
            1 + .12 * std::sin(3 * theta) * std::cos(4 * phi) + .04 * std::sin(11 * theta) * std::sin(7 * phi);
        mesh.vertices.push_back({float(r * std::sin(theta) * std::cos(phi)), float(r * std::sin(theta) * std::sin(phi)),
                                 float(r * std::cos(theta))});
    };
    vertex(0, 0);
    for (uint32_t ring = 1; ring < rings; ++ring)
        for (uint32_t segment = 0; segment < segments; ++segment)
            vertex(pi * ring / rings, 2 * pi * segment / segments);
    vertex(pi, 0);
    const uint32_t south = uint32_t(mesh.vertices.size() - 1);
    const auto     at = [&](uint32_t ring, uint32_t segment) { return 1 + (ring - 1) * segments + segment % segments; };
    for (uint32_t segment = 0; segment < segments; ++segment)
    {
        mesh.triangles.push_back({0, at(1, segment), at(1, segment + 1)});
        for (uint32_t ring = 1; ring + 1 < rings; ++ring)
        {
            const uint32_t a = at(ring, segment), b = at(ring, segment + 1), c = at(ring + 1, segment),
                           d = at(ring + 1, segment + 1);
            mesh.triangles.push_back({a, c, d});
            mesh.triangles.push_back({a, d, b});
        }
        mesh.triangles.push_back({at(rings - 1, segment), south, at(rings - 1, segment + 1)});
    }
    return mesh;
}

void seaFixture(const Options& o, Fixture& f, const std::string& prefix)
{
    const std::string suffix = o.workload.substr(prefix.size());
    char*             end    = nullptr;
    const double      length = std::strtod(suffix.c_str(), &end);
    require(!suffix.empty() && *end == '\0' && std::isfinite(length) && length > 0,
            "a sea sweep needs a positive length suffix, such as sea-box-sweeps-2000");
    require(o.queries % 2 == 0, "sea sweeps come in pairs: one through the sea, one above it");
    f.radius = .5;
    f.half   = {.5, .5, .5};
    // Direct PhysX box-box sweeps come 2.2e-4 of the distance early here (12.1949 against the
    // independent solve's 12.1976 m); the installed contact meets the solve.
    f.boxSweepBaselineEarlyTolerance = 5e-4;
    // The general fixture's box layout, flat: boxes of boxHalf on a two-metre grid at z = 0.
    const size_t width = size_t(std::ceil(std::sqrt(double(o.objects))));
    for (size_t i = 0; i < o.objects; ++i)
        f.objects.push_back({i + 1, i + 1, false, {double(i % width) * 2, double(i / width) * 2, 0}, identity});
    // Pair p starts outside the sea's corner edge and runs along a diagonal of box centres.
    for (size_t i = 0; i < o.queries; ++i)
    {
        Query q;
        q.p            = {-9.5 + 2 * double((i / 2) % width), -9.5, i % 2 ? 3.0 : 0.0};
        q.direction    = unitDirection(1, 1, 0);
        q.nearDistance = 0;
        q.farDistance  = length;
        f.queries.push_back(q);
    }
}

void denseMeshFixture(const Options& o, Fixture& f)
{
    require(o.objects == 1, "the dense mesh fixture holds one mesh; pass --objects 1");
    f.radius = .08;
    f.half   = {.08, .06, .1};
    f.meshes.push_back(lumpySphere(256, 256));
    f.objects.push_back({1, 1, false, {.3, -.2, .1}, rotation(.25f), 1, 1});
    // Starts on a Fibonacci shell of radius 4, aimed at jittered points within half a metre of the
    // centre: every sweep passes through the mesh and well beyond it.
    for (size_t i = 0; i < o.queries; ++i)
    {
        const double z = 1 - 2 * (double(i) + .5) / double(o.queries), ring = std::sqrt(1 - z * z);
        const double phi = 2.399963229728653 * double(i);
        const Vec    start{.3 + 4 * ring * std::cos(phi), -.2 + 4 * ring * std::sin(phi), .1 + 4 * z};
        const Vec    target{.3 + frac(double(i) * .7548776662) - .5, -.2 + frac(double(i) * .5698402910) - .5,
                         .1 + frac(double(i) * .6180339887) - .5};
        Query        q;
        q.p            = start;
        q.q            = i % 2 ? rotation(.25f) : identity;
        q.direction    = unitDirection(target[0] - start[0], target[1] - start[1], target[2] - start[2]);
        q.nearDistance = 0;
        q.farDistance  = 10;
        f.queries.push_back(q);
    }
}

// Centres on a Fibonacci shell whose radius varies with a golden-ratio sequence.
void denseMeshOverlapFixture(const Options& o, Fixture& f)
{
    require(o.objects == 1, "the dense mesh fixture holds one mesh; pass --objects 1");
    f.radius = .15;
    f.meshes.push_back(lumpySphere(188, 188));
    f.objects.push_back({1, 1, false, {.3, -.2, .1}, rotation(.25f), 1, 1});
    for (size_t i = 0; i < o.queries; ++i)
    {
        const double z = 1 - 2 * (double(i) + .5) / double(o.queries), ring = std::sqrt(1 - z * z);
        const double phi = 2.399963229728653 * double(i), shell = .7 + .6 * frac(double(i) * .6180339887);
        Query        q;
        q.p = {.3 + shell * ring * std::cos(phi), -.2 + shell * ring * std::sin(phi), .1 + shell * z};
        f.queries.push_back(q);
    }
}

void levelFixture(const Options& o, Fixture& f)
{
    f.meshes.push_back(terrain(4 * o.meshSide));
    // Tiles span eight metres: an eight-metre grid makes one contiguous level.
    const size_t width = size_t(std::ceil(std::sqrt(double(o.objects))));
    for (size_t i = 0; i < o.objects; ++i)
        f.objects.push_back({i + 1, i + 1, false, {double(i % width) * 8, double(i / width) * 8, 0}, identity, 1, 1});
    const double centre = double(width) * 4 - 4, reach = double(width) * 3.2;
    if (f.kind == QueryKind::boxSweep || f.kind == QueryKind::sweep)
    {
        f.radius = .3;
        f.half   = {.3, .25, .2};
        // Direct PhysX sphere sweeps on these tiles report float normals about 3e-5 off the
        // installed contact's and distances up to 4.4e-6 of the distance away (Fixture's named
        // limitations, as on the warehouse racks); installed contacts meet the analytic face.
        f.approximateBaselineContacts = f.kind == QueryKind::sweep;
        f.sweepRelativeTolerance      = f.kind == QueryKind::sweep ? 1e-3 : 0;
        for (size_t i = 0; i < o.queries; ++i)
        {
            Query        q;
            const double heading = 6.283185307179586 * frac(double(i) * .6180339887);
            q.p                  = {centre + 2 * reach * (frac(double(i) * .7548776662) - .5),
                                    centre + 2 * reach * (frac(double(i) * .5698402910) - .5), .8};
            q.q                  = i % 2 ? rotation(.25f) : identity;
            q.direction          = unitDirection(std::cos(heading), std::sin(heading), -.08);
            q.nearDistance       = 0;
            q.farDistance        = 40;
            f.queries.push_back(q);
        }
        return;
    }
    f.radius = reach;
    f.half   = {reach, reach, 1};
    for (size_t i = 0; i < o.queries; ++i)
    {
        Query q;
        q.p = {centre + frac(double(i) * .7548776662) - .5, centre + frac(double(i) * .5698402910) - .5, 0};
        q.q = i % 2 ? rotation(.25f) : identity;
        f.queries.push_back(q);
    }
}
} // namespace

bool staticsWorkload(const Options& o)
{
    return startsWith(o.workload, "sea-box-sweeps-") || startsWith(o.workload, "sea-sweeps-") ||
           o.workload == "dense-mesh-box-sweeps" || o.workload == "dense-mesh-sweeps" ||
           o.workload == "dense-mesh-sphere-any" || o.workload == "dense-mesh-sphere-overlaps" ||
           o.workload == "level-box-sweeps" || o.workload == "level-sweeps" || o.workload == "level-box-overlaps" ||
           o.workload == "level-sphere-overlaps";
}

Fixture staticsFixture(const Options& o)
{
    Fixture f;
    f.zoneHitCapacity = o.zoneHitCapacity;
    if (startsWith(o.workload, "sea-box-sweeps-"))
    {
        f.kind = QueryKind::boxSweep;
        seaFixture(o, f, "sea-box-sweeps-");
    }
    else if (startsWith(o.workload, "sea-sweeps-"))
    {
        f.kind = QueryKind::sweep;
        seaFixture(o, f, "sea-sweeps-");
    }
    else if (o.workload == "dense-mesh-box-sweeps" || o.workload == "dense-mesh-sweeps")
    {
        f.kind = o.workload == "dense-mesh-box-sweeps" ? QueryKind::boxSweep : QueryKind::sweep;
        denseMeshFixture(o, f);
    }
    else if (o.workload == "level-box-sweeps" || o.workload == "level-sweeps")
    {
        f.kind = o.workload == "level-box-sweeps" ? QueryKind::boxSweep : QueryKind::sweep;
        levelFixture(o, f);
    }
    else if (o.workload == "dense-mesh-sphere-overlaps" || o.workload == "dense-mesh-sphere-any")
    {
        f.kind = o.workload == "dense-mesh-sphere-any" ? QueryKind::sphereAny : QueryKind::sphereOverlap;
        denseMeshOverlapFixture(o, f);
    }
    else if (o.workload == "level-box-overlaps" || o.workload == "level-sphere-overlaps")
    {
        f.kind = o.workload == "level-box-overlaps" ? QueryKind::zones : QueryKind::sphereOverlap;
        levelFixture(o, f);
    }
    else
        require(false, "unknown statics workload " + o.workload);
    return f;
}
} // namespace bench
