/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Identical authored meshes acquired from an Ovstage stage share one preparation and one
// cooked backend mesh, across instances and across a transform-only republication, while a
// changed mesh is prepared afresh. The evidence is the instance's host accounting: cooking a
// mesh of this size charges about a megabyte, sharing charges nothing. Geometry references
// prove identity: unchanged content keeps its reference across republications, including
// points rewritten with the same values at a later ordinal, a mesh entering the population
// takes the reference of the identical content it shares, and changed points or a rewritten
// face triangulation take a new one.
#include <ovspatialquery/ovspatialquery_ovstage.h>

#include <ovstage/ovstage.h>
#include <ovx/path_dictionary/path_dictionary.h>

#include "../../support/native/ovstage_companion_fixtures.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace
{
// A 64 x 64 quad grid in the plane x = 0: 4,225 points and 8,192 triangles. The publication
// that adds its mesh reserves the query structure, 4096 + 16 * points + 32 * triangles bytes
// (about 0.33 MB), and claims the build scratch, 65536 + 16 * points + 192 * triangles bytes
// (about 1.7 MB), which the first exact query releases once it has built the structure
// (ADR-0075). A generation that shares the preparation still charges its prepared arrays
// (about 0.3 MB here) as retained input, so sharing is bounded at 0.5 MiB.
constexpr int      kGridSide       = 64;
constexpr uint64_t kStructureBytes = 4096u + 4225u * 16u + 8192u * 32u;
constexpr uint64_t kScratchBytes   = 65536u + 4225u * 16u + 8192u * 192u;
constexpr uint64_t kSharingMargin  = 512u * 1024u;

std::string gridMesh(const char* name, double translateX)
{
    std::string usda = "    def Mesh \"";
    usda += name;
    usda += "\"\n    {\n        uniform bool doubleSided = 0\n        uniform token orientation = \"rightHanded\"\n"
            "        uniform token subdivisionScheme = \"none\"\n        int[] faceVertexCounts = [";
    for (int face = 0; face < kGridSide * kGridSide; ++face)
    {
        usda += face ? ", 4" : "4";
    }
    usda += "]\n        int[] faceVertexIndices = [";
    for (int row = 0; row < kGridSide; ++row)
    {
        for (int column = 0; column < kGridSide; ++column)
        {
            const int corner = row * (kGridSide + 1) + column;
            char      text[96];
            std::snprintf(text, sizeof text, "%s%d, %d, %d, %d", (row || column) ? ", " : "", corner, corner + 1,
                          corner + kGridSide + 2, corner + kGridSide + 1);
            usda += text;
        }
    }
    usda += "]\n        point3f[] points = [";
    for (int row = 0; row <= kGridSide; ++row)
    {
        for (int column = 0; column <= kGridSide; ++column)
        {
            char text[64];
            std::snprintf(text, sizeof text, "%s(0, %g, %g)", (row || column) ? ", " : "", 0.1 * row, 0.1 * column);
            usda += text;
        }
    }
    char extent[512];
    std::snprintf(extent, sizeof extent,
                  "]\n        float3[] extent = [(0, 0, 0), (0, %g, %g)]\n"
                  "        matrix4d xformOp:transform = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (%g, 0, 0, 1))\n"
                  "        uniform token[] xformOpOrder = [\"xformOp:transform\"]\n    }\n",
                  0.1 * kGridSide, 0.1 * kGridSide, translateX);
    usda += extent;
    return usda;
}

bool populateSharedMeshFixture(ovstage_instance_t* stage, path_dictionary_instance_t* dictionary)
{
    std::string usda = "#usda 1.0\n(\n    defaultPrim = \"World\"\n    upAxis = \"Y\"\n    metersPerUnit = 1\n)\n"
                       "def Xform \"World\"\n{\n";
    usda += gridMesh("First", 0.0);
    usda += gridMesh("Second", 10.0);
    usda += gridMesh("Third", 30.0);
    usda += "}\n";
    const ovstage_population_enqueue_result_t population =
        openUsdWithPhysics(stage, ovx_string_t{usda.data(), usda.size()}, 1u, 0.0);
    ovstage_population_op_wait_result_t populationWait{};
    if (population.status != OVSTAGE_OK ||
        ovstage_population_wait_op(stage, population.op_index, OVSTAGE_TIMEOUT_INFINITE, &populationWait) != OVSTAGE_OK)
    {
        return false;
    }
    const ovstage_write_floor_desc_t floorOne{1u, OVSTAGE_SCOPE_ALL, nullptr, 0u};
    if (!waitStageOperation(stage, ovstage_advance_write_floor(stage, &floorOne)) ||
        !waitStageOperation(stage,
                            ovstage_compute_hierarchy(stage, OVSTAGE_HIERARCHY_COMPUTATION_MODEL_DEFAULT_CPU, 1u, 2u)))
    {
        return false;
    }
    // The third mesh starts outside the population: present, but without bounds.
    double absentBounds[6]{
        absentBoundsValue(true),  absentBoundsValue(true),  absentBoundsValue(true),
        absentBoundsValue(false), absentBoundsValue(false), absentBoundsValue(false),
    };
    if (!writeBounds(stage, dictionary, "/World/Third", sizeof("/World/Third") - 1u, 2u, absentBounds))
    {
        return false;
    }
    const ovstage_write_floor_desc_t floorTwo{2u, OVSTAGE_SCOPE_ALL, nullptr, 0u};
    return waitStageOperation(stage, ovstage_advance_write_floor(stage, &floorTwo));
}

bool sealFloor(ovstage_instance_t* stage, ovstage_ordinal_t ordinal)
{
    const ovstage_write_floor_desc_t floor{ordinal, OVSTAGE_SCOPE_ALL, nullptr, 0u};
    return waitStageOperation(stage, ovstage_advance_write_floor(stage, &floor));
}

bool sameReference(const ovspatialquery_geometry_ref_t& first, const ovspatialquery_geometry_ref_t& second)
{
    return first.type == second.type && first.value == second.value && first.value != 0u;
}

// Casts along +x from `originX` through the middle of a grid cell (not a vertex, where a float
// contact test is not watertight); returns the reported distance or a negative value, and the
// hit's geometry reference when asked.
double castExact(ovspatialquery_instance_handle_t instance, ovspatialquery_population_handle_t population,
                 double originX, ovspatialquery_ordinal_t expectedOrdinal, ovx_primpath_t expectedId,
                 ovspatialquery_geometry_ref_t* outGeometry = nullptr)
{
    ovspatialquery_raycast_desc_t ray{};
    ray.common.target         = populationTarget(population);
    ray.common.precision      = OVSPATIALQUERY_QUERY_EXACT;
    ray.ray.origin[0]         = originX;
    ray.ray.origin[1]         = 0.05 * kGridSide + 0.05;
    ray.ray.origin[2]         = 0.05 * kGridSide + 0.05;
    ray.ray.unit_direction[0] = 1.0f;
    ray.ray.max_distance      = 20.0;
    ovspatialquery_raycast_single_result_t closest{};
    if (!succeeded(ovspatialquery_raycast_closest_sync(instance, &ray, &closest)) ||
        closest.source_ordinal != expectedOrdinal || closest.has_hit != 1u ||
        closest.hit.truth != OVSPATIALQUERY_HIT_EXACT_SURFACE ||
        closest.hit.data.exact.identity.object_id.value != handleValue(expectedId) ||
        closest.hit.data.exact.feature.kind != OVSPATIALQUERY_GEOMETRY_FEATURE_TRIANGLE)
    {
        return -1.0;
    }
    if (outGeometry != nullptr)
    {
        *outGeometry = closest.hit.data.exact.identity.geometry;
    }
    return closest.hit.data.exact.distance;
}

// The same cast must find nothing within 20 m.
bool castMisses(ovspatialquery_instance_handle_t instance, ovspatialquery_population_handle_t population,
                double originX, ovspatialquery_ordinal_t expectedOrdinal)
{
    ovspatialquery_raycast_desc_t ray{};
    ray.common.target         = populationTarget(population);
    ray.common.precision      = OVSPATIALQUERY_QUERY_EXACT;
    ray.ray.origin[0]         = originX;
    ray.ray.origin[1]         = 0.05 * kGridSide + 0.05;
    ray.ray.origin[2]         = 0.05 * kGridSide + 0.05;
    ray.ray.unit_direction[0] = 1.0f;
    ray.ray.max_distance      = 20.0;
    ovspatialquery_raycast_single_result_t closest{};
    return succeeded(ovspatialquery_raycast_closest_sync(instance, &ray, &closest)) &&
           closest.source_ordinal == expectedOrdinal && closest.has_hit == 0u;
}

int testSharedAuthoredMeshes()
{
    ovstage_instance_t*           stage = nullptr;
    const ovstage_instance_desc_t stageDesc{"ovspatialquery-shared-mesh-test"};
    CHECK(ovstage_create_instance(&stageDesc, &stage) == OVSTAGE_OK);
    path_dictionary_instance_t* dictionary = ovstage_get_path_dictionary(stage);
    CHECK(dictionary != nullptr);
    CHECK(populateSharedMeshFixture(stage, dictionary));

    static const char  firstPath[]  = "/World/First";
    static const char  secondPath[] = "/World/Second";
    static const char  thirdPath[]  = "/World/Third";
    const ovx_string_t paths[]      = {{firstPath, sizeof(firstPath) - 1u},
                                       {secondPath, sizeof(secondPath) - 1u},
                                       {thirdPath, sizeof(thirdPath) - 1u}};
    ovx_primpath_t     ids[3]       = {OVX_INVALID_PRIMPATH, OVX_INVALID_PRIMPATH, OVX_INVALID_PRIMPATH};
    CHECK(dictionaryCallSucceeded(dictionary,
                                  dictionary->vtable->create_paths_from_strings(dictionary->context, paths, 3u, ids)));
    ovstage_predicate_t pathPredicate{};
    pathPredicate.attribute.string = ovx_string_t{"usd-path", sizeof("usd-path") - 1u};
    pathPredicate.op               = OVSTAGE_FILTER_OP_IN;
    pathPredicate.values           = paths;
    pathPredicate.value_count      = 3u;

    const ovspatialquery_instance_desc_t instanceDesc = commonInstanceDesc();
    ovspatialquery_instance_handle_t     instance     = OVSPATIALQUERY_INVALID_INSTANCE;
    CHECK(succeeded(ovspatialquery_create_instance(&instanceDesc, &instance)));
    ovspatialquery_ovstage_population_desc_t sourceDesc{};
    sourceDesc.stage                              = stage;
    sourceDesc.filter                             = ovstage_filter_t{&pathPredicate, 1u};
    sourceDesc.common                             = aabbPopulationDesc();
    ovspatialquery_population_handle_t population = OVSPATIALQUERY_INVALID_POPULATION;
    CHECK(succeeded(ovspatialquery_create_ovstage_population(instance, &sourceDesc, &population)));
    uint64_t unpublished = 0u;
    CHECK(readHostBytes(instance, unpublished));
    CHECK(finishPopulationUpdate(instance, ovspatialquery_enqueue_ovstage_population_set(instance, population, 2), 2,
                                 2u, 1u));

    // The publication reserves the shared mesh's structure and its build scratch; the first
    // exact query builds it within them and releases the scratch, the second instance reuses
    // it, and identical content carries one geometry reference.
    uint64_t                      published = 0u, firstCooked = 0u, secondQueried = 0u, moved = 0u, changed = 0u;
    ovspatialquery_geometry_ref_t firstGeometry{}, secondGeometry{}, geometry{};
    CHECK(readHostBytes(instance, published));
    CHECK(published >= unpublished + kStructureBytes + kScratchBytes);
    CHECK(std::abs(castExact(instance, population, -5.0, 2, ids[0], &firstGeometry) - 5.0) < 1.0e-4);
    CHECK(readHostBytes(instance, firstCooked));
    CHECK(firstCooked + kScratchBytes <= published);
    CHECK(std::abs(castExact(instance, population, 5.0, 2, ids[1], &secondGeometry) - 5.0) < 1.0e-4);
    CHECK(readHostBytes(instance, secondQueried));
    CHECK(secondQueried < firstCooked + kSharingMargin);
    CHECK(sameReference(firstGeometry, secondGeometry));

    // A transform-only republication (world matrix and extent rewritten, as the stage's
    // hierarchy pass would) keeps the preparation of the unchanged mesh.
    double worldFromLocal[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 20, 0, 0, 1};
    double movedBounds[6]     = {19.99, -0.01, -0.01, 20.01, 6.41, 6.41};
    CHECK(writeAttribute(stage, dictionary, secondPath, sizeof(secondPath) - 1u, "omni:fabric:worldMatrix", 3u,
                         worldFromLocal, DLDataType{kDLFloat, 64u, 16u}, 1u, OVSTAGE_SEMANTIC_MATRIX));
    CHECK(writeBounds(stage, dictionary, secondPath, sizeof(secondPath) - 1u, 3u, movedBounds));
    CHECK(sealFloor(stage, 3u));
    CHECK(finishPopulationUpdate(instance, ovspatialquery_enqueue_ovstage_population_update(instance, population, 2, 3),
                                 3, 2u, 1u));
    CHECK(std::abs(castExact(instance, population, 15.0, 3, ids[1], &geometry) - 5.0) < 1.0e-4);
    CHECK(sameReference(geometry, secondGeometry));
    CHECK(std::abs(castExact(instance, population, -5.0, 3, ids[0], &geometry) - 5.0) < 1.0e-4);
    CHECK(sameReference(geometry, firstGeometry));
    CHECK(readHostBytes(instance, moved));
    std::fprintf(stderr, "host bytes after move: %llu (second queried %llu)\n", static_cast<unsigned long long>(moved),
                 static_cast<unsigned long long>(secondQueried));
    CHECK(moved < secondQueried + kSharingMargin);

    // Changed points are new content: prepared and built afresh, and queried as authored.
    std::vector<float> points;
    points.reserve(static_cast<size_t>((kGridSide + 1) * (kGridSide + 1) * 3));
    for (int row = 0; row <= kGridSide; ++row)
    {
        for (int column = 0; column <= kGridSide; ++column)
        {
            points.push_back(1.0f);
            points.push_back(0.1f * static_cast<float>(row));
            points.push_back(0.1f * static_cast<float>(column));
        }
    }
    double changedBounds[6] = {0.99, -0.01, -0.01, 1.01, 6.41, 6.41};
    CHECK(writeFloatPointArray(stage, dictionary, firstPath, sizeof(firstPath) - 1u, 4u, points.data(),
                               points.size() / 3u));
    CHECK(writeBounds(stage, dictionary, firstPath, sizeof(firstPath) - 1u, 4u, changedBounds));
    CHECK(sealFloor(stage, 4u));
    CHECK(finishPopulationUpdate(instance, ovspatialquery_enqueue_ovstage_population_update(instance, population, 3, 4),
                                 4, 2u, 1u));
    ovspatialquery_geometry_ref_t changedGeometry{};
    CHECK(std::abs(castExact(instance, population, -5.0, 4, ids[0], &changedGeometry) - 6.0) < 1.0e-4);
    CHECK(changedGeometry.value != 0u && !sameReference(changedGeometry, firstGeometry));
    CHECK(std::abs(castExact(instance, population, 15.0, 4, ids[1], &geometry) - 5.0) < 1.0e-4);
    CHECK(sameReference(geometry, secondGeometry));
    CHECK(readHostBytes(instance, changed));
    std::fprintf(stderr,
                 "host bytes: published %llu, first cooked %llu, second queried %llu, moved %llu, changed %llu\n",
                 static_cast<unsigned long long>(published), static_cast<unsigned long long>(firstCooked),
                 static_cast<unsigned long long>(secondQueried), static_cast<unsigned long long>(moved),
                 static_cast<unsigned long long>(changed));
    CHECK(changed >= moved + kStructureBytes);
    CHECK(std::abs(castExact(instance, population, 15.0, 4, ids[1]) - 5.0) < 1.0e-4);

    // The third mesh enters the population when it gains bounds. It is new to the
    // population, so it is read from the stage; its content is the second mesh's, so
    // it shares that preparation and reference, and its publication reserves no structure
    // or scratch. (The generation itself charges its retained arrays, below that scratch.)
    double thirdBounds[6] = {29.99, -0.01, -0.01, 30.01, 6.41, 6.41};
    CHECK(writeBounds(stage, dictionary, thirdPath, sizeof(thirdPath) - 1u, 5u, thirdBounds));
    CHECK(sealFloor(stage, 5u));
    CHECK(finishPopulationUpdate(instance, ovspatialquery_enqueue_ovstage_population_update(instance, population, 4, 5),
                                 5, 3u, 0u));
    uint64_t entered = 0u, thirdQueried = 0u;
    CHECK(readHostBytes(instance, entered));
    std::fprintf(stderr, "host bytes after the third mesh entered: %llu\n", static_cast<unsigned long long>(entered));
    CHECK(entered < changed + kScratchBytes);
    CHECK(std::abs(castExact(instance, population, 25.0, 5, ids[2], &geometry) - 5.0) < 1.0e-4);
    CHECK(sameReference(geometry, secondGeometry));
    CHECK(readHostBytes(instance, thirdQueried));
    std::fprintf(stderr, "host bytes after the third mesh's first query: %llu\n",
                 static_cast<unsigned long long>(thirdQueried));
    CHECK(thirdQueried < entered + kSharingMargin);
    CHECK(std::abs(castExact(instance, population, -5.0, 5, ids[0], &geometry) - 6.0) < 1.0e-4);
    CHECK(sameReference(geometry, changedGeometry));

    // It leaves again when its bounds are withdrawn; the others keep their geometry.
    double absentBounds[6]{
        absentBoundsValue(true),  absentBoundsValue(true),  absentBoundsValue(true),
        absentBoundsValue(false), absentBoundsValue(false), absentBoundsValue(false),
    };
    CHECK(writeBounds(stage, dictionary, thirdPath, sizeof(thirdPath) - 1u, 6u, absentBounds));
    CHECK(sealFloor(stage, 6u));
    CHECK(finishPopulationUpdate(instance, ovspatialquery_enqueue_ovstage_population_update(instance, population, 5, 6),
                                 6, 2u, 1u));
    CHECK(castMisses(instance, population, 25.0, 6));
    CHECK(std::abs(castExact(instance, population, 15.0, 6, ids[1], &geometry) - 5.0) < 1.0e-4);
    CHECK(sameReference(geometry, secondGeometry));

    // Points rewritten with the values they already have are a change record for the
    // stage but not new content: the adapter compares them with the retained mesh and
    // keeps its preparation and reference.
    CHECK(writeFloatPointArray(stage, dictionary, firstPath, sizeof(firstPath) - 1u, 7u, points.data(),
                               points.size() / 3u));
    CHECK(sealFloor(stage, 7u));
    CHECK(finishPopulationUpdate(instance, ovspatialquery_enqueue_ovstage_population_update(instance, population, 6, 7),
                                 7, 2u, 1u));
    uint64_t rewritten = 0u, rewrittenQueried = 0u;
    CHECK(readHostBytes(instance, rewritten));
    CHECK(rewritten < thirdQueried + kScratchBytes);
    CHECK(std::abs(castExact(instance, population, -5.0, 7, ids[0], &geometry) - 6.0) < 1.0e-4);
    CHECK(sameReference(geometry, changedGeometry));
    CHECK(std::abs(castExact(instance, population, 15.0, 7, ids[1], &geometry) - 5.0) < 1.0e-4);
    CHECK(sameReference(geometry, secondGeometry));
    CHECK(readHostBytes(instance, rewrittenQueried));
    CHECK(rewrittenQueried < rewritten + kSharingMargin);

    // Points that move under unchanged faces keep the retained mesh's topology: the new
    // mesh derives from it, and its first exact query loads a refit of the topology's
    // cooked template instead of cooking a tree. Identical content still shares one
    // preparation and one reference, whether the second prim is read in full (the third
    // mesh returns with the moved points) or derives like the first.
    const auto movedGrid = [](double phase)
    {
        std::vector<float> moved;
        moved.reserve(static_cast<size_t>((kGridSide + 1) * (kGridSide + 1) * 3));
        for (int row = 0; row <= kGridSide; ++row)
        {
            for (int column = 0; column <= kGridSide; ++column)
            {
                moved.push_back(0.05f * static_cast<float>(std::sin(0.5 * row + phase)));
                moved.push_back(0.1f * static_cast<float>(row));
                moved.push_back(0.1f * static_cast<float>(column));
            }
        }
        return moved;
    };
    std::vector<float> movedOnce      = movedGrid(0.0);
    double             secondMoved[6] = {19.94, -0.01, -0.01, 20.06, 6.41, 6.41};
    double             thirdMoved[6]  = {29.94, -0.01, -0.01, 30.06, 6.41, 6.41};
    CHECK(writeFloatPointArray(stage, dictionary, secondPath, sizeof(secondPath) - 1u, 8u, movedOnce.data(),
                               movedOnce.size() / 3u));
    CHECK(writeBounds(stage, dictionary, secondPath, sizeof(secondPath) - 1u, 8u, secondMoved));
    CHECK(writeFloatPointArray(stage, dictionary, thirdPath, sizeof(thirdPath) - 1u, 8u, movedOnce.data(),
                               movedOnce.size() / 3u));
    CHECK(writeBounds(stage, dictionary, thirdPath, sizeof(thirdPath) - 1u, 8u, thirdMoved));
    CHECK(sealFloor(stage, 8u));
    CHECK(finishPopulationUpdate(instance, ovspatialquery_enqueue_ovstage_population_update(instance, population, 7, 8),
                                 8, 3u, 0u));
    uint64_t derivedPublished = 0u, derivedQueried = 0u, derivedShared = 0u;
    CHECK(readHostBytes(instance, derivedPublished));
    ovspatialquery_geometry_ref_t movedGeometry{};
    const double                  secondDistance = castExact(instance, population, 15.0, 8, ids[1], &movedGeometry);
    CHECK(secondDistance > 0.0 && std::abs(secondDistance - 5.0) < 0.06 && std::abs(secondDistance - 5.0) > 1.0e-3);
    CHECK(movedGeometry.value != 0u && !sameReference(movedGeometry, secondGeometry));
    CHECK(readHostBytes(instance, derivedQueried));
    std::fprintf(stderr, "host bytes: derived published %llu, first derived query (template cooked) %llu\n",
                 static_cast<unsigned long long>(derivedPublished), static_cast<unsigned long long>(derivedQueried));
    CHECK(derivedQueried + kScratchBytes <= derivedPublished);
    const double thirdDistance = castExact(instance, population, 25.0, 8, ids[2], &geometry);
    CHECK(std::abs(thirdDistance - secondDistance) < 1.0e-6);
    CHECK(sameReference(geometry, movedGeometry));
    CHECK(readHostBytes(instance, derivedShared));
    CHECK(derivedShared < derivedQueried + kSharingMargin);

    // Both move again, identically: two derived meshes, one preparation, one new
    // reference, and no cook at the first query because the template already exists.
    std::vector<float> movedTwice = movedGrid(1.0);
    CHECK(writeFloatPointArray(stage, dictionary, secondPath, sizeof(secondPath) - 1u, 9u, movedTwice.data(),
                               movedTwice.size() / 3u));
    CHECK(writeFloatPointArray(stage, dictionary, thirdPath, sizeof(thirdPath) - 1u, 9u, movedTwice.data(),
                               movedTwice.size() / 3u));
    CHECK(sealFloor(stage, 9u));
    CHECK(finishPopulationUpdate(instance, ovspatialquery_enqueue_ovstage_population_update(instance, population, 8, 9),
                                 9, 3u, 0u));
    uint64_t twicePublished = 0u, twiceQueried = 0u;
    CHECK(readHostBytes(instance, twicePublished));
    ovspatialquery_geometry_ref_t twiceGeometry{};
    const double                  secondTwice = castExact(instance, population, 15.0, 9, ids[1], &twiceGeometry);
    CHECK(secondTwice > 0.0 && std::abs(secondTwice - 5.0) < 0.06 && std::abs(secondTwice - secondDistance) > 1.0e-3);
    CHECK(twiceGeometry.value != 0u && !sameReference(twiceGeometry, movedGeometry));
    CHECK(std::abs(castExact(instance, population, 25.0, 9, ids[2], &geometry) - secondTwice) < 1.0e-6);
    CHECK(sameReference(geometry, twiceGeometry));
    CHECK(readHostBytes(instance, twiceQueried));
    std::fprintf(stderr, "host bytes: second move published %llu, both queried %llu\n",
                 static_cast<unsigned long long>(twicePublished), static_cast<unsigned long long>(twiceQueried));
    CHECK(twiceQueried + kScratchBytes <= twicePublished);
    // The retained head still answers with the geometry it was published with.
    CHECK(std::abs(castExact(instance, population, -5.0, 9, ids[0], &geometry) - 6.0) < 1.0e-4);
    CHECK(sameReference(geometry, changedGeometry));

    // The first mesh's faces rewritten from their next corner: the same quads, the same
    // points, a different fan triangulation. Changed topology is new content, read in full,
    // prepared and cooked afresh, with a new reference and the same surface; the shared
    // meshes keep theirs. A change record that leaves the content as it is then keeps the
    // new preparation.
    std::vector<int32_t> rotated;
    rotated.reserve(static_cast<size_t>(kGridSide * kGridSide * 4));
    for (int row = 0; row < kGridSide; ++row)
    {
        for (int column = 0; column < kGridSide; ++column)
        {
            const int corner = row * (kGridSide + 1) + column;
            rotated.push_back(corner + 1);
            rotated.push_back(corner + kGridSide + 2);
            rotated.push_back(corner + kGridSide + 1);
            rotated.push_back(corner);
        }
    }
    CHECK(writeAttribute(stage, dictionary, firstPath, sizeof(firstPath) - 1u, "faceVertexIndices", 10u, rotated.data(),
                         DLDataType{kDLInt, 32u, 1u}, rotated.size(), OVSTAGE_SEMANTIC_NONE, true));
    CHECK(sealFloor(stage, 10u));
    CHECK(finishPopulationUpdate(
        instance, ovspatialquery_enqueue_ovstage_population_update(instance, population, 9, 10), 10, 3u, 0u));
    uint64_t rotatedPublished = 0u, rotatedQueried = 0u;
    CHECK(readHostBytes(instance, rotatedPublished));
    ovspatialquery_geometry_ref_t rotatedGeometry{};
    CHECK(std::abs(castExact(instance, population, -5.0, 10, ids[0], &rotatedGeometry) - 6.0) < 1.0e-4);
    CHECK(rotatedGeometry.value != 0u && !sameReference(rotatedGeometry, changedGeometry));
    CHECK(std::abs(castExact(instance, population, 15.0, 10, ids[1], &geometry) - secondTwice) < 1.0e-6);
    CHECK(sameReference(geometry, twiceGeometry));
    CHECK(readHostBytes(instance, rotatedQueried));
    std::fprintf(stderr, "host bytes: rotated faces published %llu, queried %llu\n",
                 static_cast<unsigned long long>(rotatedPublished), static_cast<unsigned long long>(rotatedQueried));
    CHECK(rotatedQueried + kScratchBytes <= rotatedPublished);
    CHECK(writeBounds(stage, dictionary, firstPath, sizeof(firstPath) - 1u, 11u, changedBounds));
    CHECK(sealFloor(stage, 11u));
    CHECK(finishPopulationUpdate(
        instance, ovspatialquery_enqueue_ovstage_population_update(instance, population, 10, 11), 11, 3u, 0u));
    uint64_t keptPublished = 0u, keptQueried = 0u;
    CHECK(readHostBytes(instance, keptPublished));
    CHECK(std::abs(castExact(instance, population, -5.0, 11, ids[0], &geometry) - 6.0) < 1.0e-4);
    CHECK(sameReference(geometry, rotatedGeometry));
    CHECK(readHostBytes(instance, keptQueried));
    CHECK(keptQueried < keptPublished + kSharingMargin);

    // A nonuniform world transform is baked into the mesh for both a retained mesh
    // and a points edit; restoring a rigid transform keeps the mesh queryable.
    double stretched[16] = {2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 20, 0, 0, 1};
    CHECK(writeAttribute(stage, dictionary, secondPath, sizeof(secondPath) - 1u, "omni:fabric:worldMatrix", 12u,
                         stretched, DLDataType{kDLFloat, 64u, 16u}, 1u, OVSTAGE_SEMANTIC_MATRIX));
    double stretchedBounds[6] = {19.89, -0.01, -0.01, 20.11, 6.41, 6.41};
    CHECK(writeBounds(stage, dictionary, secondPath, sizeof(secondPath) - 1u, 12u, stretchedBounds));
    CHECK(sealFloor(stage, 12u));
    CHECK(finishPopulationUpdate(
        instance, ovspatialquery_enqueue_ovstage_population_update(instance, population, 11, 12), 12, 3u, 0u));
    CHECK(std::abs(castExact(instance, population, 15.0, 12, ids[1]) - (5.0 + 2.0 * (secondTwice - 5.0))) < 1.0e-6);
    std::vector<float> movedThrice = movedGrid(2.0);
    CHECK(writeFloatPointArray(stage, dictionary, secondPath, sizeof(secondPath) - 1u, 13u, movedThrice.data(),
                               movedThrice.size() / 3u));
    CHECK(sealFloor(stage, 13u));
    CHECK(finishPopulationUpdate(
        instance, ovspatialquery_enqueue_ovstage_population_update(instance, population, 12, 13), 13, 3u, 0u));
    CHECK(writeAttribute(stage, dictionary, secondPath, sizeof(secondPath) - 1u, "omni:fabric:worldMatrix", 14u,
                         worldFromLocal, DLDataType{kDLFloat, 64u, 16u}, 1u, OVSTAGE_SEMANTIC_MATRIX));
    CHECK(writeBounds(stage, dictionary, secondPath, sizeof(secondPath) - 1u, 14u, secondMoved));
    CHECK(sealFloor(stage, 14u));
    CHECK(finishPopulationUpdate(
        instance, ovspatialquery_enqueue_ovstage_population_update(instance, population, 13, 14), 14, 3u, 0u));
    CHECK(castExact(instance, population, 15.0, 14, ids[1]) > 0.0);

    CHECK(succeeded(ovspatialquery_destroy_population(instance, population)));
    CHECK(succeeded(ovspatialquery_destroy_instance(instance)));
    CHECK(ovstage_destroy_instance(stage) == OVSTAGE_OK);
    return 0;
}
} // namespace

int main()
{
    CHECK(testSharedAuthoredMeshes() == 0);
    return 0;
}
