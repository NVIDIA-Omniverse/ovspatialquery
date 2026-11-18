/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Terrain generation, mesh fixtures and the mesh semantic smoke.
#include "mesh.h"

#include "../installed_adapter.h"

#include <cmath>
#include <iostream>
#include <string>

namespace bench
{
// One heightfield tile combines a ramp, terraces and an omitted patch. Vertices
// are quantized once to float so both SDKs receive identical geometric points.
double terrainHeight(double x) { return x < 0 ? .125 * x : .125 * std::floor(x); }
Mesh   terrain(size_t side)
{
    Mesh mesh;
    for (size_t y = 0; y <= side; ++y)
        for (size_t x = 0; x <= side; ++x)
        {
            const float px = float(-4 + 8 * double(x) / double(side)), py = float(-4 + 8 * double(y) / double(side));
            mesh.vertices.push_back({px, py, float(terrainHeight(px))});
        }
    for (size_t y = 0; y < side; ++y)
        for (size_t x = 0; x < side; ++x)
        {
            const double cx = -4 + 8 * (double(x) + .5) / double(side), cy = -4 + 8 * (double(y) + .5) / double(side);
            if (cx > -.5 && cx < .5 && cy > 1 && cy < 2)
                continue;
            const uint32_t a = uint32_t(y * (side + 1) + x), b = a + 1, c = a + uint32_t(side + 1), d = c + 1;
            mesh.triangles.push_back({a, b, d});
            mesh.triangles.push_back({a, d, c});
        }
    return mesh;
}
Fixture meshFixture(const Options& o)
{
    Fixture f;
    f.kind   = o.workload == "mesh-sweeps"       ? QueryKind::sweep
               : o.workload == "mesh-box-sweeps" ? QueryKind::boxSweep
                                                 : QueryKind::rays;
    f.even   = o.subset == "even";
    f.radius = .125;
    if (f.kind == QueryKind::boxSweep)
        f.half = {.12, .12, .125};
    f.meshes.assign(o.uniqueMeshes ? o.objects : 1, terrain(o.meshSide));
    const size_t width = size_t(std::ceil(std::sqrt(double(o.objects))));
    for (size_t i = 0; i < o.objects; ++i)
        f.objects.push_back({i + 1,
                             i + 1,
                             false,
                             {double(i % width) * 10, double(i / width) * 10, 0},
                             i % 2 ? rotation(.25f) : identity,
                             1,
                             o.uniqueMeshes ? i + 1 : 1});
    f.queries.resize(o.queries);
    return f;
}
void meshQueries(Fixture& f, size_t frame)
{
    for (size_t i = 0; i < f.queries.size(); ++i)
    {
        auto&        q      = f.queries[i];
        const auto&  object = f.objects[((i / 4) * 37) % f.objects.size()];
        const double x =
            -3.483 + double((i / 4 * 17) % 61) / 10 + (i % 2 ? .12 : -.12) + .01 * std::sin(double(frame) * .3);
        const double y = -3.487 + double((i / 4 * 29) % 59) / 10 + (i % 4 < 2 ? -.1 : .1);
        const Vec    local{x, y, terrainHeight(x) + .6};
        const auto   world = pose(object.p, object.q).transform(px(local));
        q.p                = {world.x, world.y, world.z};
        if (f.kind == QueryKind::boxSweep)
            q.q = i % 2 ? rotation(.25f) : identity;
        q.nearDistance = 0;
        q.farDistance  = .8;
    }
}

void meshSemanticSmoke()
{
    size_t                   checks = 0;
    std::vector<std::string> diagnostics;
    // Every edge probe below also runs on the CUDA plan when one can be created, so the
    // sidedness, clipping, edge, overlap and update cases gate CPU/CUDA exact-mesh parity.
    std::string cudaParity;
    const bool  cudaAvailable = cudaPlansAvailable(cudaParity);
    cudaParity                = cudaAvailable ? "parity" : "unavailable: " + cudaParity;
    auto probe = [&](const std::string& name, Fixture f, bool expected, bool contact = true, bool commonParity = true,
                     uint64_t expectedFace = UINT64_MAX, double planeDistance = -1)
    {
        Direct  direct(f);
        Spatial spatial(f);
        spatial.makePlan(f);
        Results d(f.queries.size()), s(d.size()), b(d.size());
        direct.query(f, d);
        spatial.scalar(f, s);
        spatial.batch(f, b);
        if (cudaAvailable)
        {
            Results c(d.size());
            spatial.makePlan(f, true);
            spatial.batch(f, c);
            compare(f, b, c, name + " CPU batch/CUDA batch");
            if (planeDistance >= 0)
                require(c[0].hits.size() == 1 && near(c[0].hits[0].distance, planeDistance) &&
                            near(c[0].hits[0].position[2], 0),
                        name + " CUDA missed analytic plane contact");
            spatial.makePlan(f);
        }
        if (commonParity && contact)
        {
            compare(f, d, s, name, false, !f.approximateBaselineContacts, nullptr, true);
        }
        else if (commonParity)
        {
            require(identities(d[0]) == identities(s[0]), name + " initial identities");
            require(!d[0].hits.empty() && d[0].hits[0].distance == 0 && s[0].hits[0].distance == 0,
                    name + " initial distance");
        }
        compare(f, s, b, name + " scalar/batch");
        require(!s[0].hits.empty() == expected, name + " unexpected spatial hit mask");
        if (planeDistance >= 0)
            for (const auto* result : {&d, &s, &b})
                require((*result)[0].hits.size() == 1 && near((*result)[0].hits[0].distance, planeDistance) &&
                            near((*result)[0].hits[0].position[2], 0),
                        name + " missed analytic plane contact");
        if (expectedFace != UINT64_MAX)
            for (const auto* result : {&d, &s, &b})
                require((*result)[0].hits.size() == 1 && (*result)[0].hits[0].sourceFace == expectedFace,
                        name + " original triangle index");
        if (commonParity)
            ++checks;
        else
            diagnostics.push_back("{\"case\":\"" + name + "\",\"physx_hit\":" + (d[0].hits.empty() ? "false" : "true") +
                                  ",\"spatial_hit\":" + (s[0].hits.empty() ? "false" : "true") + "}");
    };
    Fixture plane;
    plane.kind = QueryKind::rays;
    plane.meshes.push_back({{{-2, -2, 0}, {2, -2, 0}, {2, 2, 0}, {-2, 2, 0}}, {{0, 1, 2}, {0, 2, 3}}, false});
    plane.objects.push_back({1, 1, false, {0, 0, 0}, identity, 1, 1});
    Query q;
    q.p            = {-.25, .5, 2};
    q.nearDistance = 0;
    q.farDistance  = 4;
    plane.queries  = {q};
    for (bool sided : {false, true})
    {
        plane.meshes[0].doubleSided = sided;
        const std::string tag       = sided ? "double-sided " : "front-only ";
        plane.queries[0]            = q;
        probe(tag + "front ray", plane, true);
        plane.queries[0].p[2]      = -2;
        plane.queries[0].direction = {0, 0, 1};
        probe(tag + "back ray", plane, sided);
        plane.queries[0]   = q;
        plane.queries[0].p = {0, 0, 2};
        probe(tag + "shared edge ray", plane, true);
        plane.queries[0].p = {2, 2, 2};
        probe(tag + "vertex ray", plane, true);
        plane.queries[0]             = q;
        plane.queries[0].farDistance = 1.99;
        probe(tag + "far clipped ray", plane, false);
        plane.queries[0].farDistance = 2;
        probe(tag + "far endpoint ray", plane, true, true, false);
        plane.queries[0]              = q;
        plane.queries[0].nearDistance = 2;
        probe(tag + "near endpoint ray", plane, true);
        plane.queries[0].nearDistance = 2.01;
        probe(tag + "near clipped ray", plane, false);
        plane.queries[0]      = q;
        plane.queries[0].p[2] = 0;
        probe(tag + "surface origin ray", plane, true);
        plane.kind       = QueryKind::sweep;
        plane.radius     = .25;
        plane.queries[0] = q;
        probe(tag + "front sweep", plane, true);
        plane.queries[0].p[2]      = -2;
        plane.queries[0].direction = {0, 0, 1};
        probe(tag + "back sweep", plane, sided);
        plane.queries[0]      = q;
        plane.queries[0].p[2] = .125;
        probe(tag + "initial sweep overlap", plane, true, false);
        plane.queries[0]   = q;
        plane.queries[0].p = {2.125, 2, 2};
        probe(tag + "edge sweep", plane, true);
        plane.kind                        = QueryKind::boxSweep;
        plane.approximateBaselineContacts = true;
        plane.half                        = {.125, .125, .25};
        plane.queries[0]                  = q;
        plane.queries[0].q                = rotation(.25f);
        plane.queries[0].nearDistance     = .1;
        probe(tag + "front box sweep", plane, true, true, true, UINT64_MAX, 1.75);
        plane.queries[0].farDistance = 1.74;
        probe(tag + "clipped box sweep", plane, false);
        plane.queries[0]           = q;
        plane.queries[0].p[2]      = -2;
        plane.queries[0].q         = rotation(.25f);
        plane.queries[0].direction = {0, 0, 1};
        probe(tag + "back box sweep", plane, sided, true, true, UINT64_MAX, sided ? 1.75 : -1);
        plane.kind            = QueryKind::sphereOverlap;
        plane.queries[0]      = q;
        plane.queries[0].p[2] = .125;
        probe(tag + "sphere overlap front", plane, true);
        plane.queries[0].p[2] = -.125;
        probe(tag + "sphere overlap back", plane, true);
        plane.queries[0].p[2] = .3;
        probe(tag + "sphere overlap separated", plane, false);
        plane.kind            = QueryKind::zones;
        plane.half            = {.1, .2, .125};
        plane.queries[0].p[2] = .1;
        plane.queries[0].q    = rotation(.3f);
        probe(tag + "rotated box overlap", plane, true);
        plane.kind = QueryKind::occupancy;
        probe(tag + "box ANY", plane, true);
        plane.kind = QueryKind::rays;
    }
    // Cooking may reorder triangles; contacts must retain their original source indices.
    Fixture disjoint    = plane;
    disjoint.queries[0] = q;
    disjoint.meshes[0].vertices.clear();
    disjoint.meshes[0].triangles.clear();
    for (float x : {10.f, -10.f, 0.f})
    {
        const auto index = uint32_t(disjoint.meshes[0].vertices.size());
        disjoint.meshes[0].vertices.insert(disjoint.meshes[0].vertices.end(),
                                           {{x - 1, -1, 0}, {x + 1, -1, 0}, {x, 1, 0}});
        disjoint.meshes[0].triangles.push_back({index, index + 1, index + 2});
    }
    uint64_t sourceFace = 0;
    for (float x : {10.f, -10.f, 0.f})
    {
        disjoint.queries[0].p = {x, 0, 2};
        probe("disjoint cooked source remap", disjoint, true, true, true, sourceFace++);
    }
    // A sloped triangle's AABB contains points that are not surface overlaps.
    Fixture slope      = plane;
    slope.meshes[0]    = {{{-2, -2, -.25}, {2, -2, .25}, {-2, 2, -.25}}, {{0, 1, 2}}, false};
    slope.kind         = QueryKind::sphereOverlap;
    slope.radius       = .05;
    slope.queries[0].p = {-1, 0, .2};
    probe("mesh bounds false positive", slope, false);
    slope.kind            = QueryKind::rays;
    slope.queries[0]      = q;
    slope.queries[0].p[0] = -1;
    probe("slope normal", slope, true);
    Fixture tilted      = plane;
    tilted.queries[0]   = q;
    tilted.objects[0].q = {std::sin(.2f), 0, 0, std::cos(.2f)};
    probe("tilted mesh ray", tilted, true);
    tilted.kind   = QueryKind::sweep;
    tilted.radius = .25;
    probe("tilted mesh sweep", tilted, true);
    // Closed meshes retain surface-query semantics; interior overlap is not containment.
    Fixture closed   = plane;
    closed.meshes[0] = {
        {{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1}, {-1, -1, 1}, {1, -1, 1}, {1, 1, 1}, {-1, 1, 1}},
        {{0, 2, 1},
         {0, 3, 2},
         {4, 5, 6},
         {4, 6, 7},
         {0, 4, 7},
         {0, 7, 3},
         {1, 2, 6},
         {1, 6, 5},
         {0, 1, 5},
         {0, 5, 4},
         {3, 7, 6},
         {3, 6, 2}},
        false};
    closed.queries[0]           = q;
    closed.queries[0].p         = {0, 0, 0};
    closed.queries[0].direction = {0, 0, 1};
    probe("closed mesh front-only interior ray", closed, false);
    closed.meshes[0].doubleSided = true;
    probe("closed mesh double-sided interior exit", closed, true);
    closed.kind   = QueryKind::sphereOverlap;
    closed.radius = .125;
    probe("closed mesh interior is not containment", closed, false);
    // Complete collider sets, subset selection, updated visibility and retained retry.
    {
        Fixture layers = plane;
        layers.kind    = QueryKind::sphereOverlap;
        layers.radius  = .25;
        layers.objects.push_back({2, 1, false, {0, 0, .25}, rotation(.25f), 1, 1});
        layers.queries[0]      = q;
        layers.queries[0].p[2] = .125;
        Direct  direct(layers);
        Spatial spatial(layers);
        compareAdapters(layers, direct, spatial);
        Results before(1), after(1), retained(1);
        direct.query(layers, before);
        require(identities(before[0]) == std::vector<uint64_t>({1, 2}), "layered complete set");
        const auto                          old = spatial.ordinal;
        ovspatialquery_overlap_hit_t        hit{};
        ovspatialquery_overlap_hit_buffer_t buffer{&hit, 1};
        ovspatialquery_multiple_result_t    shortResult{};
        auto                                desc = spatial.overlap(layers, layers.queries[0]);
        check(ovspatialquery_overlap_multiple_sync(spatial.instance, &desc, &buffer, &shortResult),
              "short mesh overlap");
        require(shortResult.truncated && shortResult.source_ordinal == old, "mesh truncation ordinal");
        layers.objects[0].p[2] = 3;
        direct.update(layers, 1);
        spatial.update(layers, 1);
        compareAdapters(layers, direct, spatial);
        direct.query(layers, after);
        require(identities(after[0]) == std::vector<uint64_t>({2}), "mesh update visibility");
        spatial.scalar(layers, retained, old);
        compare(layers, before, retained, "mesh retained scalar retry");
        spatial.batch(layers, retained, old);
        compare(layers, before, retained, "mesh retained batch retry");
        layers.even = true;
        compareAdapters(layers, direct, spatial);
        ++checks;
    }
    {
        Options largeOptions;
        largeOptions.objects  = 1;
        largeOptions.queries  = 1;
        largeOptions.meshSide = 224;
        largeOptions.workload = "mesh-sweeps";
        auto large            = meshFixture(largeOptions);
        meshQueries(large, 0);
        Direct  d(large);
        Spatial s(large);
        s.makePlan(large);
        Results a(1), b(1), c(1);
        d.query(large, a);
        s.scalar(large, b);
        s.batch(large, c);
        compare(large, b, c, "dense sweep scalar/batch");
        require(identities(a[0]) == identities(b[0]) && !a[0].hits.empty(), "dense sweep identities");
        require(near(a[0].hits[0].distance, b[0].hits[0].distance),
                "dense sweep distance: PhysX=" + std::to_string(a[0].hits[0].distance) +
                    " spatial=" + std::to_string(b[0].hits[0].distance));
        validateMeshFace(large, a[0].hits[0]);
        diagnostics.push_back(
            "{\"case\":\"dense mesh sweep normal\",\"physx_normal_x\":" + std::to_string(a[0].hits[0].normal[0]) +
            ",\"spatial_normal_x\":" + std::to_string(b[0].hits[0].normal[0]) + "}");
    }
    // Preserve a measured float-transform seam discrepancy as a diagnostic, not a parity claim.
    Options edgeOptions;
    edgeOptions.objects = 1;
    edgeOptions.queries = 1;
    auto edge           = meshFixture(edgeOptions);
    edge.objects[0].p   = {30, 10, 0};
    edge.objects[0].q   = rotation(.25f);
    const auto world    = pose(edge.objects[0].p, edge.objects[0].q).transform(px({-.12, 1, terrainHeight(-.12) + .6}));
    edge.queries[0].p   = {world.x, world.y, world.z};
    edge.queries[0].nearDistance = 0;
    edge.queries[0].farDistance  = .8;
    Direct  edgeDirect(edge);
    Spatial edgeSpatial(edge);
    edgeSpatial.makePlan(edge);
    Results ed(1), es(1), eb(1);
    edgeDirect.query(edge, ed);
    edgeSpatial.scalar(edge, es);
    edgeSpatial.batch(edge, eb);
    compare(edge, es, eb, "float seam scalar/batch");
    std::cout << "{\"backend\":\"parity\",\"mesh_checks\":" << checks << ",\"cuda_parity\":\"" << cudaParity
              << "\",\"qualified_common_parity\":true,\"strict_cross_backend_parity\":false,\"initial_sweep_contact_"
                 "fields_compared\":false,\"float_seam_probe\":{\"physx_hit\":"
              << (!ed[0].hits.empty() ? "true" : "false")
              << ",\"spatial_hit\":" << (!es[0].hits.empty() ? "true" : "false") << "},\"diagnostics\":[";
    for (size_t i = 0; i < diagnostics.size(); ++i)
    {
        if (i)
            std::cout << ',';
        std::cout << diagnostics[i];
    }
    std::cout << "]}\n";
}
} // namespace bench
