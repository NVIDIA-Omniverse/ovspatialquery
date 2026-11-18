/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../support/native/hull_box_gap_scene.h"
#include "../../support/native/public_api_inputs.h"
#include "../../support/native/test_failure.h"
#include <ovspatialquery/ovspatialquery_explicit.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>

// ovspatialquery_overlap_any_sync and ovspatialquery_overlap_multiple_sync
// through the real PhysX backend, for all three overlap shape kinds (sphere,
// AABB, oriented box): any-hit selection, the stored-count/truncated
// contract, degenerate-shape rejection with output preservation, and misses.
// Objects are boxes with identity rotation, whose published world AABB
// reproduces the tested bounds exactly under OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE.

namespace
{

constexpr uint64_t kBoxDefinition = 1u;
constexpr double   kHalfExtent    = 0.5;

ovspatialquery_population_handle_t buildPopulation(ovspatialquery_instance_handle_t instance,
                                                   ovspatialquery_ordinal_t         ordinal)
{
    double                                      halfExtents[3] = {kHalfExtent, kHalfExtent, kHalfExtent};
    uint64_t                                    boxValue       = kBoxDefinition;
    InputArray                                  boxValues(&boxValue, 1, kDLUInt);
    InputArray                                  boxExtents(halfExtents, 1, 3, kDLFloat);
    const ovspatialquery_box_definition_table_t boxes{&boxValues.value, &boxExtents.value};
    waitAndDestroy(instance, ovspatialquery_enqueue_box_registration(instance, &boxes));

    const ovspatialquery_population_handle_t population = createExplicitPopulation(instance, 0u);
    // Objects 30 and 10 share bounds [0,1]^3; object 20 sits apart at
    // [4,5]x[0,1]x[0,1]. Box centers are the AABB midpoints.
    InstanceRows rows;
    rows.add(30u, OVSPATIALQUERY_GEOMETRY_BOX, kBoxDefinition, 0.5, 0.5, 0.5);
    rows.add(10u, OVSPATIALQUERY_GEOMETRY_BOX, kBoxDefinition, 0.5, 0.5, 0.5);
    rows.add(20u, OVSPATIALQUERY_GEOMETRY_BOX, kBoxDefinition, 4.5, 0.5, 0.5);
    const ovspatialquery_explicit_population_set_t set{ordinal, rows.table(), nullptr, 0u};
    waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set));
    return population;
}

ovspatialquery_query_common_t candidateCommon(ovspatialquery_population_handle_t population)
{
    ovspatialquery_query_common_t common{};
    common.target    = populationTarget(population);
    common.precision = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    return common;
}

ovspatialquery_overlap_desc_t makeSphereQuery(ovspatialquery_population_handle_t population, double centerX,
                                              double radius)
{
    ovspatialquery_overlap_desc_t query{};
    query.common                      = candidateCommon(population);
    query.shape.kind                  = OVSPATIALQUERY_OVERLAP_SPHERE;
    query.shape.data.sphere.center[0] = centerX;
    query.shape.data.sphere.center[1] = 0.5;
    query.shape.data.sphere.center[2] = 0.5;
    query.shape.data.sphere.radius    = radius;
    return query;
}

ovspatialquery_overlap_desc_t makeAabbQuery(ovspatialquery_population_handle_t population, double minimumX,
                                            double maximumX)
{
    ovspatialquery_overlap_desc_t query{};
    query.common                        = candidateCommon(population);
    query.shape.kind                    = OVSPATIALQUERY_OVERLAP_AABB;
    query.shape.data.aabb.min_corner[0] = minimumX;
    query.shape.data.aabb.min_corner[1] = 0.25;
    query.shape.data.aabb.min_corner[2] = 0.25;
    query.shape.data.aabb.max_corner[0] = maximumX;
    query.shape.data.aabb.max_corner[1] = 0.75;
    query.shape.data.aabb.max_corner[2] = 0.75;
    return query;
}

ovspatialquery_overlap_desc_t makeBoxQuery(ovspatialquery_population_handle_t population, double centerX,
                                           double halfExtentX, float rotationZ, float rotationW)
{
    ovspatialquery_overlap_desc_t query{};
    query.common                          = candidateCommon(population);
    query.shape.kind                      = OVSPATIALQUERY_OVERLAP_BOX;
    query.shape.data.box.center[0]        = centerX;
    query.shape.data.box.center[1]        = 0.5;
    query.shape.data.box.center[2]        = 0.5;
    query.shape.data.box.rotation_xyzw[2] = rotationZ;
    query.shape.data.box.rotation_xyzw[3] = rotationW;
    query.shape.data.box.half_extents[0]  = halfExtentX;
    query.shape.data.box.half_extents[1]  = 0.25;
    query.shape.data.box.half_extents[2]  = 0.25;
    return query;
}

std::set<uint64_t> hitIdSet(const ovspatialquery_overlap_hit_t* hits, uint64_t count)
{
    std::set<uint64_t> ids;
    for (uint64_t index = 0u; index < count; ++index)
    {
        REQUIRE(hits[index].truth == OVSPATIALQUERY_HIT_BOUNDS_CANDIDATE);
        ids.insert(hits[index].identity.object_id.value);
    }
    return ids;
}

// Exercises one shape kind: any-hit selection, the stored/truncated buffer
// contract at zero, partial, and full capacity, a clean miss, and a
// degenerate shape rejected with the caller's output untouched.
void runOverlapModeCase(ovspatialquery_instance_handle_t instance, ovspatialquery_ordinal_t ordinal,
                        const ovspatialquery_overlap_desc_t& hitQuery, const ovspatialquery_overlap_desc_t& missQuery,
                        const ovspatialquery_overlap_desc_t& degenerateQuery)
{
    ovspatialquery_overlap_any_result_t any{};
    REQUIRE(ovspatialquery_overlap_any_sync(instance, &hitQuery, &any).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(any.source_ordinal == ordinal && any.has_hit == 1u);
    REQUIRE(any.hit.truth == OVSPATIALQUERY_HIT_BOUNDS_CANDIDATE);
    REQUIRE(any.hit.identity.object_id.value == 10u || any.hit.identity.object_id.value == 30u);

    ovspatialquery_overlap_hit_buffer_t emptyBuffer{nullptr, 0u};
    ovspatialquery_multiple_result_t    empty{};
    REQUIRE(ovspatialquery_overlap_multiple_sync(instance, &hitQuery, &emptyBuffer, &empty).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(empty.source_ordinal == ordinal && empty.stored_hit_count == 0u && empty.truncated == 1u);

    ovspatialquery_overlap_hit_t        partialHits[1]{};
    ovspatialquery_overlap_hit_buffer_t partialBuffer{partialHits, 1u};
    ovspatialquery_multiple_result_t    partial{};
    REQUIRE(ovspatialquery_overlap_multiple_sync(instance, &hitQuery, &partialBuffer, &partial).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(partial.source_ordinal == ordinal && partial.stored_hit_count == 1u && partial.truncated == 1u);
    REQUIRE(partialHits[0].truth == OVSPATIALQUERY_HIT_BOUNDS_CANDIDATE);
    REQUIRE(partialHits[0].identity.object_id.value == 10u || partialHits[0].identity.object_id.value == 30u);

    ovspatialquery_overlap_hit_t        hits[2]{};
    ovspatialquery_overlap_hit_buffer_t buffer{hits, 2u};
    ovspatialquery_multiple_result_t    multiple{};
    REQUIRE(ovspatialquery_overlap_multiple_sync(instance, &hitQuery, &buffer, &multiple).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(multiple.source_ordinal == ordinal && multiple.stored_hit_count == 2u && multiple.truncated == 0u);
    // Row order is not promised (same contract as raycast_multiple_sync);
    // only the complete identity set is.
    REQUIRE(hitIdSet(hits, 2u) == (std::set<uint64_t>{10u, 30u}));

    ovspatialquery_overlap_any_result_t miss{};
    REQUIRE(ovspatialquery_overlap_any_sync(instance, &missQuery, &miss).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(miss.source_ordinal == ordinal && miss.has_hit == 0u);

    // Rejected calls preserve the caller's output struct byte for byte.
    ovspatialquery_overlap_any_result_t sentinel;
    std::memset(&sentinel, 0xa5, sizeof(sentinel));
    ovspatialquery_overlap_any_result_t invalid = sentinel;
    const ovspatialquery_result_t       status  = ovspatialquery_overlap_any_sync(instance, &degenerateQuery, &invalid);
    REQUIRE(status.status == OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(std::memcmp(&invalid, &sentinel, sizeof(invalid)) == 0);
}

// A box overlap of a convex hull is decided by the exact separation, not by
// GJK's tolerance: on hull_box_gap_scene.h's rows, boxes 1e-3 and 1e-4 from the
// hull overlap nothing and boxes 1e-4 deep overlap it, in a CPU plan with
// float32 and float64 rotation columns and in the synchronous query.
void testHullBoxOverlapsResolveSmallGaps(ovspatialquery_instance_handle_t instance)
{
    HullBoxGapScene                    scene(instance);
    constexpr uint32_t                 kRows = HullBoxGapScene::kRows;
    ovspatialquery_overlap_plan_desc_t desc{};
    desc.target     = populationRef(scene.population);
    desc.hit_mode   = OVSPATIALQUERY_HIT_MODE_ANY;
    desc.precision  = OVSPATIALQUERY_QUERY_EXACT;
    desc.shape.kind = OVSPATIALQUERY_OVERLAP_BOX;
    for (double& extent : desc.shape.data.box.half_extents)
    {
        extent = HullBoxGapScene::kHalfExtent;
    }
    desc.device                             = DLDevice{kDLCPU, 0};
    desc.maximum_query_count                = kRows;
    desc.maximum_hits_per_query             = 1u;
    desc.hit_capacity_per_slot              = kRows;
    desc.result_slot_count                  = 1u;
    ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_overlap_plan(instance, &desc, &plan).status == OVSPATIALQUERY_API_SUCCESS);
    InputArray              translations(scene.translations, kRows, 3, kDLFloat);
    InputArray              rotations64(scene.rotations64, kRows, 4, kDLFloat);
    InputArray              rotations32(scene.rotations32, kRows, 4, kDLFloat);
    const InputArray* const rotationColumns[2]{&rotations32, &rotations64};
    for (const InputArray* rotations : rotationColumns)
    {
        ovspatialquery_overlap_batch_t batch{};
        batch.ordinal.kind   = OVSPATIALQUERY_ORDINAL_HEAD;
        batch.query_count    = kRows;
        batch.translations   = &translations.value;
        batch.rotations_xyzw = &rotations->value;
        ovspatialquery_overlap_batch_reservation_t reservation{};
        ovspatialquery_enqueue_result_t            operation =
            ovspatialquery_enqueue_overlap_plan(instance, plan, &batch, &reservation);
        REQUIRE(operation.status == OVSPATIALQUERY_API_SUCCESS);
        ovspatialquery_op_wait_result_t waited{};
        REQUIRE(ovspatialquery_wait_op(instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status ==
                    OVSPATIALQUERY_API_SUCCESS &&
                waited.state == OVSPATIALQUERY_OP_COMPLETED);
        ovspatialquery_overlap_batch_result_t result{};
        REQUIRE(ovspatialquery_fetch_overlap_batch(instance, operation.op_id, &result).status ==
                OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(std::memcmp(result.columns.hit_ranges.hit_counts.data, scene.overlaps, sizeof(scene.overlaps)) == 0);
        REQUIRE(ovspatialquery_destroy_op(instance, &operation.op_id).status == OVSPATIALQUERY_API_SUCCESS);
    }
    REQUIRE(ovspatialquery_destroy_batch_plan(instance, plan).status == OVSPATIALQUERY_API_SUCCESS);
    for (uint32_t row = 0u; row < kRows; ++row)
    {
        ovspatialquery_overlap_desc_t query{};
        query.common.target    = populationTarget(scene.population);
        query.common.precision = OVSPATIALQUERY_QUERY_EXACT;
        query.shape.kind       = OVSPATIALQUERY_OVERLAP_BOX;
        for (uint32_t axis = 0u; axis < 3u; ++axis)
        {
            query.shape.data.box.center[axis]       = scene.translations[row * 3u + axis];
            query.shape.data.box.half_extents[axis] = HullBoxGapScene::kHalfExtent;
        }
        std::memcpy(query.shape.data.box.rotation_xyzw, scene.rotations32 + row * 4u, sizeof(float) * 4u);
        ovspatialquery_overlap_any_result_t any{};
        REQUIRE(ovspatialquery_overlap_any_sync(instance, &query, &any).status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(any.has_hit == scene.overlaps[row]);
    }
    scene.destroy();
}

// An overlap shortcut must not take a rounding for contact. The prism with base
// (1, -2), (-2, 1), (-5, -2), (-2, -5) and z in [-1, 1] satisfies x + y <= -1;
// a box of half extents (2^54, 2^54, 2) centred at (2^54, 2^54, 0) has x, y >= 0,
// so the two are apart along (1, 1, 0) though their bounds meet, and
// 1 - 2^54 rounds to -2^54, onto the box's face. A float64 rotation row, which
// takes the double-precision test, reports no overlap; the same box moved to
// (2^54 - 2, 2^54 - 2, 0) reaches the prism. A float32 row takes PhysX's float32
// test, which does not resolve a box this large.
void testHugeBoxBesideASmallHullIsApart(ovspatialquery_instance_handle_t instance)
{
    uint64_t    hullValue = 41u, offset = 0u, vertexCount = 8u;
    const float base[4][2]{{1.0f, -2.0f}, {-2.0f, 1.0f}, {-5.0f, -2.0f}, {-2.0f, -5.0f}};
    float       vertices[24]{};
    for (uint32_t corner = 0u; corner < 8u; ++corner)
    {
        vertices[corner * 3u]      = base[corner / 2u][0];
        vertices[corner * 3u + 1u] = base[corner / 2u][1];
        vertices[corner * 3u + 2u] = corner % 2u != 0u ? 1.0f : -1.0f;
    }
    InputArray values(&hullValue, 1, kDLUInt), offsets(&offset, 1, kDLUInt), counts(&vertexCount, 1, kDLUInt),
        points(vertices, 8, 3, kDLFloat);
    const ovspatialquery_convex_definition_table_t hulls{
        &values.value, &offsets.value, &counts.value, nullptr, nullptr, &points.value,
        nullptr,       nullptr,        nullptr,       0.0,     0u,
    };
    waitAndDestroy(instance, ovspatialquery_enqueue_convex_registration(instance, &hulls));
    const ovspatialquery_population_handle_t population = createExplicitPopulation(instance, 1u);
    InstanceRows                             rows;
    rows.add(hullValue, OVSPATIALQUERY_GEOMETRY_CONVEX, hullValue, 0.0, 0.0, 0.0);
    const ovspatialquery_explicit_population_set_t set{1, rows.table(), nullptr, 0u};
    waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set));

    const double                       huge = std::ldexp(1.0, 54);
    double                             translations[6]{huge, huge, 0.0, huge - 2.0, huge - 2.0, 0.0};
    double                             halfExtents[6]{huge, huge, 2.0, huge, huge, 2.0};
    double                             rotations[8]{0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    const uint64_t                     expected[2]{0u, 1u};
    ovspatialquery_overlap_plan_desc_t desc{};
    desc.target     = populationRef(population);
    desc.hit_mode   = OVSPATIALQUERY_HIT_MODE_ANY;
    desc.precision  = OVSPATIALQUERY_QUERY_EXACT;
    desc.shape.kind = OVSPATIALQUERY_OVERLAP_BOX;
    for (double& extent : desc.shape.data.box.half_extents)
    {
        extent = 1.0;
    }
    desc.device                             = DLDevice{kDLCPU, 0};
    desc.maximum_query_count                = 2u;
    desc.maximum_hits_per_query             = 1u;
    desc.hit_capacity_per_slot              = 2u;
    desc.result_slot_count                  = 1u;
    ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_overlap_plan(instance, &desc, &plan).status == OVSPATIALQUERY_API_SUCCESS);
    InputArray translationColumn(translations, 2, 3, kDLFloat), extentColumn(halfExtents, 2, 3, kDLFloat),
        rotationColumn(rotations, 2, 4, kDLFloat);
    ovspatialquery_overlap_batch_t batch{};
    batch.ordinal.kind   = OVSPATIALQUERY_ORDINAL_HEAD;
    batch.query_count    = 2u;
    batch.translations   = &translationColumn.value;
    batch.rotations_xyzw = &rotationColumn.value;
    batch.half_extents   = &extentColumn.value;
    ovspatialquery_overlap_batch_reservation_t reservation{};
    ovspatialquery_enqueue_result_t            operation =
        ovspatialquery_enqueue_overlap_plan(instance, plan, &batch, &reservation);
    REQUIRE(operation.status == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_op_wait_result_t waited{};
    REQUIRE(ovspatialquery_wait_op(instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status ==
                OVSPATIALQUERY_API_SUCCESS &&
            waited.state == OVSPATIALQUERY_OP_COMPLETED);
    ovspatialquery_overlap_batch_result_t result{};
    REQUIRE(ovspatialquery_fetch_overlap_batch(instance, operation.op_id, &result).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(std::memcmp(result.columns.hit_ranges.hit_counts.data, expected, sizeof(expected)) == 0);
    REQUIRE(ovspatialquery_destroy_op(instance, &operation.op_id).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_batch_plan(instance, plan).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_CONVEX, hullValue}).status ==
            OVSPATIALQUERY_API_SUCCESS);
}

// A mesh authored 1e8 from its placement: one triangle with corners (1e8, 1e8, 0),
// (1e8 + 1024, 1e8, 0) and (1e8, 1e8 + 1024, 0), placed at (-1e8, -1e8, 0). A unit sphere
// at (-0.9, -0.9, 0) lies 0.27 from the placed corner, but PhysX's float32 test works in
// the mesh's local coordinates, where the sphere's centre rounds onto the corner, and
// reports a touch. The sphere bounds gate, which drops meshes the sphere misses before
// that test, must leave the decision to it: its margin counts the local coordinates, not
// only the world bounds. A sphere 5,000 away is rejected either way.
void testFarAuthoredMeshKeepsItsExactDecision(ovspatialquery_instance_handle_t instance)
{
    uint64_t   value = 51u, zero = 0u, vertexCount = 3u, triangleCount = 1u;
    float      vertices[9]{1e8f, 1e8f, 0.0f, 1e8f + 1024.0f, 1e8f, 0.0f, 1e8f, 1e8f + 1024.0f, 0.0f};
    uint32_t   triangle[3]{0u, 1u, 2u};
    InputArray values(&value, 1, kDLUInt), vertexOffsets(&zero, 1, kDLUInt), vertexCounts(&vertexCount, 1, kDLUInt),
        triangleOffsets(&zero, 1, kDLUInt), triangleCounts(&triangleCount, 1, kDLUInt),
        vertexData(vertices, 3, 3, kDLFloat), triangleData(triangle, 1, 3, kDLUInt);
    const ovspatialquery_triangle_mesh_definition_table_t meshes{
        &values.value,
        &vertexOffsets.value,
        &vertexCounts.value,
        &triangleOffsets.value,
        &triangleCounts.value,
        &vertexData.value,
        &triangleData.value,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
    };
    waitAndDestroy(instance, ovspatialquery_enqueue_triangle_mesh_registration(instance, &meshes));
    const ovspatialquery_population_handle_t population = createExplicitPopulation(instance, 1u);
    InstanceRows                             rows;
    rows.add(value, OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, value, -1e8, -1e8, 0.0);
    const ovspatialquery_explicit_population_set_t set{1, rows.table(), nullptr, 0u};
    waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set));

    const double   centers[2][3]{{-0.9, -0.9, 0.0}, {-5000.0, -5000.0, 0.0}};
    const uint64_t expected[2]{1u, 0u};
    for (uint32_t row = 0u; row < 2u; ++row)
    {
        ovspatialquery_overlap_desc_t desc{};
        desc.common.target    = populationTarget(population);
        desc.common.precision = OVSPATIALQUERY_QUERY_EXACT;
        desc.shape.kind       = OVSPATIALQUERY_OVERLAP_SPHERE;
        std::memcpy(desc.shape.data.sphere.center, centers[row], sizeof(centers[row]));
        desc.shape.data.sphere.radius = 1.0;
        ovspatialquery_overlap_hit_t        hits[2]{};
        ovspatialquery_overlap_hit_buffer_t buffer{hits, 2u};
        ovspatialquery_multiple_result_t    result{};
        REQUIRE(ovspatialquery_overlap_multiple_sync(instance, &desc, &buffer, &result).status ==
                OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(result.stored_hit_count == expected[row] && result.truncated == 0u);
    }
    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(
        ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, value}).status ==
        OVSPATIALQUERY_API_SUCCESS);
}

} // namespace

int main()
{
    const ovspatialquery_instance_desc_t instanceDesc{4u, 1u, UINT64_C(16) * 1024u * 1024u, nullptr, 0u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&instanceDesc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    const ovspatialquery_population_handle_t population = buildPopulation(instance, 3);

    {
        const ovspatialquery_overlap_desc_t hitQuery        = makeSphereQuery(population, 0.5, 0.75);
        const ovspatialquery_overlap_desc_t missQuery       = makeSphereQuery(population, 2.5, 0.25);
        ovspatialquery_overlap_desc_t       degenerateQuery = missQuery;
        degenerateQuery.shape.data.sphere.radius            = 0.0;
        runOverlapModeCase(instance, 3, hitQuery, missQuery, degenerateQuery);
    }

    {
        const ovspatialquery_overlap_desc_t hitQuery        = makeAabbQuery(population, -0.25, 0.75);
        const ovspatialquery_overlap_desc_t missQuery       = makeAabbQuery(population, 2.0, 3.0);
        ovspatialquery_overlap_desc_t       degenerateQuery = missQuery;
        degenerateQuery.shape.data.aabb.min_corner[0]       = 4.0;
        degenerateQuery.shape.data.aabb.max_corner[0]       = 3.0;
        runOverlapModeCase(instance, 3, hitQuery, missQuery, degenerateQuery);
    }

    {
        const float                         halfSqrt        = static_cast<float>(std::sqrt(0.5));
        const ovspatialquery_overlap_desc_t hitQuery        = makeBoxQuery(population, 0.5, 0.75, halfSqrt, halfSqrt);
        const ovspatialquery_overlap_desc_t missQuery       = makeBoxQuery(population, 2.5, 0.25, 0.0f, 1.0f);
        ovspatialquery_overlap_desc_t       degenerateQuery = missQuery;
        degenerateQuery.shape.data.box.rotation_xyzw[3]     = 0.5f;
        runOverlapModeCase(instance, 3, hitQuery, missQuery, degenerateQuery);
    }

    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(
        ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_BOX, kBoxDefinition}).status ==
        OVSPATIALQUERY_API_SUCCESS);
    testHullBoxOverlapsResolveSmallGaps(instance);
    testHugeBoxBesideASmallHullIsApart(instance);
    testFarAuthoredMeshKeepsItsExactDecision(instance);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
    std::puts("overlap modes test passed");
    return 0;
}
