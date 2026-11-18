/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../support/native/public_api_inputs.h"
#include "../../support/native/test_failure.h"
#include <ovspatialquery/ovspatialquery_explicit.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <utility>
#include <vector>

// Precise triangle-mesh, sphere-sweep, and overlap geometry proven through the
// public synchronous queries (raycast_closest_sync, raycast_multiple_sync,
// sweep_sphere_closest_sync, overlap_any_sync, overlap_multiple_sync) over
// explicit populations: exact hit position, normal, distance, triangle face
// and barycentric weights, front/back sidedness carried by the mesh
// definition, instance pose (rotation and uniform scale), the multiple-hit
// and overlap-multiple truncation contracts, view membership, exact-vs-bounds
// precision, and broad-phase-vs-exact overlap rejection. Reusable batch/plan
// coverage lives in other CTest targets, not here, except the per-launch
// `query_flags`, which the sidedness test checks on CPU plans beside the
// synchronous flag.

namespace
{

void requireNear(double actual, double expected, double tolerance, int line, const char* label)
{
    if (!(std::abs(actual - expected) <= tolerance))
    {
        std::fprintf(stderr,
                     "precise geometry requirement failed at line %d: %s expected "
                     "%.17g, observed %.17g (tolerance %.3g)\n",
                     line, label, expected, actual, tolerance);
        OVSPATIALQUERY_TEST_EXIT_FAILURE();
    }
}

#define REQUIRE_NEAR(actual, expected, tolerance) requireNear((actual), (expected), (tolerance), __LINE__, #actual)

ovspatialquery_instance_handle_t createInstance()
{
    const ovspatialquery_instance_desc_t desc{4u, 2u, UINT64_C(32) * 1024u * 1024u, nullptr, 0u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&desc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    return instance;
}

// A single front-facing triangle in the local X=1 plane. Its outward normal
// is -X (winding 0,1,2 as seen from -X). Used wherever a test needs exactly
// one triangle with predictable barycentric weights for an interior hit.
std::vector<float> singleTriangleVertices() { return {1.0, -1.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, -1.0}; }

std::vector<uint32_t> singleTriangleIndices() { return {0u, 1u, 2u}; }

// Two parallel front-facing triangles (outward normal -X on both) at local
// X=1 and X=2, forming a thin wedge from X=1 to X=2 with the same (Y,Z)
// silhouette on each plane. Used for sidedness, multiple-hit, and
// broad-phase-vs-exact overlap tests that need a real (non-degenerate) X
// extent on the object's world AABB.
std::vector<float> wedgeVertices()
{
    return {
        1.0, -1.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, -1.0, 2.0, -1.0, -1.0, 2.0, 0.0, 1.0, 2.0, 1.0, -1.0,
    };
}

std::vector<uint32_t> wedgeIndices() { return {0u, 1u, 2u, 3u, 4u, 5u}; }

// Enqueues the registration of one immutable triangle-mesh definition; its
// columns are copied on submission.
ovspatialquery_enqueue_result_t enqueueTriangleMesh(ovspatialquery_instance_handle_t instance, uint64_t definitionValue,
                                                    std::vector<float> vertices, std::vector<uint32_t> triangles,
                                                    bool doubleSided)
{
    const int64_t vertexCount        = static_cast<int64_t>(vertices.size() / 3u);
    const int64_t triangleCount      = static_cast<int64_t>(triangles.size() / 3u);
    uint64_t      value              = definitionValue;
    uint64_t      vertexOffset       = 0u;
    uint64_t      vertexCountValue   = static_cast<uint64_t>(vertexCount);
    uint64_t      triangleOffset     = 0u;
    uint64_t      triangleCountValue = static_cast<uint64_t>(triangleCount);
    uint8_t       doubleSidedValue   = doubleSided ? 1u : 0u;

    InputArray                          valueArray(&value, 1, kDLUInt);
    InputArray                          vertexOffsetArray(&vertexOffset, 1, kDLUInt);
    InputArray                          vertexCountArray(&vertexCountValue, 1, kDLUInt);
    InputArray                          triangleOffsetArray(&triangleOffset, 1, kDLUInt);
    InputArray                          triangleCountArray(&triangleCountValue, 1, kDLUInt);
    InputArray                          vertexArray(vertices.data(), vertexCount, 3, kDLFloat);
    InputArray                          triangleArray(triangles.data(), triangleCount, 3, kDLUInt);
    InputArray                          doubleSidedArray(&doubleSidedValue, 1, kDLUInt);
    const ovspatialquery_input_array_t* doubleSidedColumn = doubleSided ? &doubleSidedArray.value : nullptr;

    const ovspatialquery_triangle_mesh_definition_table_t table{&valueArray.value,
                                                                &vertexOffsetArray.value,
                                                                &vertexCountArray.value,
                                                                &triangleOffsetArray.value,
                                                                &triangleCountArray.value,
                                                                &vertexArray.value,
                                                                &triangleArray.value,
                                                                doubleSidedColumn,
                                                                nullptr,
                                                                nullptr,
                                                                nullptr};
    return ovspatialquery_enqueue_triangle_mesh_registration(instance, &table);
}

// Registers one immutable triangle-mesh definition and returns its value.
uint64_t registerTriangleMesh(ovspatialquery_instance_handle_t instance, uint64_t definitionValue,
                              std::vector<float> vertices, std::vector<uint32_t> triangles, bool doubleSided)
{
    waitAndDestroy(instance, enqueueTriangleMesh(instance, definitionValue, std::move(vertices), std::move(triangles),
                                                 doubleSided));
    return definitionValue;
}

struct TriangleInstance
{
    uint64_t objectId;
    uint64_t geometryValue;
    double   translation[3]  = {0.0, 0.0, 0.0};
    float    rotationXyzw[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    double   uniformScale    = 1.0;
};

// Publishes the first (and only) complete set for a fresh explicit
// population whose instances all reference triangle-mesh definitions.
ovspatialquery_population_handle_t publishTrianglePopulation(ovspatialquery_instance_handle_t     instance,
                                                             ovspatialquery_ordinal_t             ordinal,
                                                             const std::vector<TriangleInstance>& instances)
{
    const ovspatialquery_population_desc_t populationDesc{1u, 1u};
    ovspatialquery_population_handle_t     population = OVSPATIALQUERY_INVALID_POPULATION;
    REQUIRE(ovspatialquery_create_explicit_population(instance, &populationDesc, &population).status ==
            OVSPATIALQUERY_API_SUCCESS);

    const int64_t         count = static_cast<int64_t>(instances.size());
    std::vector<uint64_t> objectValues(static_cast<size_t>(count));
    std::vector<int32_t>  geometryTypes(static_cast<size_t>(count), OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH);
    std::vector<uint64_t> geometryValues(static_cast<size_t>(count));
    std::vector<double>   translations(static_cast<size_t>(count) * 3u);
    std::vector<float>    rotations(static_cast<size_t>(count) * 4u);
    std::vector<double>   scales(static_cast<size_t>(count));
    for (int64_t index = 0; index < count; ++index)
    {
        const TriangleInstance& placement          = instances[static_cast<size_t>(index)];
        objectValues[static_cast<size_t>(index)]   = placement.objectId;
        geometryValues[static_cast<size_t>(index)] = placement.geometryValue;
        for (uint32_t axis = 0u; axis < 3u; ++axis)
        {
            translations[static_cast<size_t>(index) * 3u + axis] = placement.translation[axis];
        }
        for (uint32_t component = 0u; component < 4u; ++component)
        {
            rotations[static_cast<size_t>(index) * 4u + component] = placement.rotationXyzw[component];
        }
        scales[static_cast<size_t>(index)] = placement.uniformScale;
    }

    InputArray objectArray(objectValues.data(), count, kDLUInt);
    InputArray typeArray(geometryTypes.data(), count, kDLInt);
    InputArray geometryArray(geometryValues.data(), count, kDLUInt);
    InputArray translationArray(translations.data(), count, 3, kDLFloat);
    InputArray rotationArray(rotations.data(), count, 4, kDLFloat);
    InputArray scaleArray(scales.data(), count, kDLFloat);

    const ovspatialquery_explicit_population_set_t set{ordinal,
                                                       {&objectArray.value, &typeArray.value, &geometryArray.value,
                                                        &translationArray.value, &rotationArray.value,
                                                        &scaleArray.value, nullptr, nullptr},
                                                       nullptr,
                                                       0u};
    waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set));
    return population;
}

ovspatialquery_raycast_single_result_t raycastClosest(
    ovspatialquery_instance_handle_t instance, ovspatialquery_query_target_t target, const double origin[3],
    const float direction[3], double maxDistance,
    ovspatialquery_query_precision_t precision = OVSPATIALQUERY_QUERY_EXACT)
{
    ovspatialquery_raycast_desc_t desc{};
    desc.common.target    = target;
    desc.common.precision = precision;
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        desc.ray.origin[axis]         = origin[axis];
        desc.ray.unit_direction[axis] = direction[axis];
    }
    desc.ray.max_distance = maxDistance;
    ovspatialquery_raycast_single_result_t result{};
    REQUIRE(ovspatialquery_raycast_closest_sync(instance, &desc, &result).status == OVSPATIALQUERY_API_SUCCESS);
    return result;
}

ovspatialquery_sphere_sweep_closest_result_t sweepClosest(ovspatialquery_instance_handle_t instance,
                                                          ovspatialquery_query_target_t target, const double center[3],
                                                          double radius, const float direction[3], double maxDistance)
{
    ovspatialquery_sphere_sweep_desc_t desc{};
    desc.common.target    = target;
    desc.common.precision = OVSPATIALQUERY_QUERY_EXACT;
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        desc.sweep.center[axis]         = center[axis];
        desc.sweep.unit_direction[axis] = direction[axis];
    }
    desc.sweep.radius       = radius;
    desc.sweep.max_distance = maxDistance;
    ovspatialquery_sphere_sweep_closest_result_t result{};
    REQUIRE(ovspatialquery_sweep_sphere_closest_sync(instance, &desc, &result).status == OVSPATIALQUERY_API_SUCCESS);
    return result;
}

ovspatialquery_overlap_any_result_t overlapSphereAny(ovspatialquery_instance_handle_t instance,
                                                     ovspatialquery_query_target_t target, const double center[3],
                                                     double radius)
{
    ovspatialquery_overlap_desc_t desc{};
    desc.common.target    = target;
    desc.common.precision = OVSPATIALQUERY_QUERY_EXACT;
    desc.shape.kind       = OVSPATIALQUERY_OVERLAP_SPHERE;
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        desc.shape.data.sphere.center[axis] = center[axis];
    }
    desc.shape.data.sphere.radius = radius;
    ovspatialquery_overlap_any_result_t result{};
    REQUIRE(ovspatialquery_overlap_any_sync(instance, &desc, &result).status == OVSPATIALQUERY_API_SUCCESS);
    return result;
}

// Destroys `populations`, then the triangle-mesh definitions `meshValues`, then the instance.
void destroyMeshScene(ovspatialquery_instance_handle_t                instance,
                      std::vector<ovspatialquery_population_handle_t> populations, std::vector<uint64_t> meshValues)
{
    for (const ovspatialquery_population_handle_t population : populations)
        REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    for (const uint64_t value : meshValues)
        REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, value})
                    .status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

// One triangle-mesh scene: an instance, one population and one mesh definition.
struct SingleTriangleFixture
{
    ovspatialquery_instance_handle_t   instance      = OVSPATIALQUERY_INVALID_INSTANCE;
    ovspatialquery_population_handle_t population    = OVSPATIALQUERY_INVALID_POPULATION;
    uint64_t                           geometryValue = 0u;

    void destroy() { destroyMeshScene(instance, {population}, {geometryValue}); }
};

// The pose test's ray meets the single triangle at local (Y,Z)=(0,0), whose
// barycentric weights against corners A(-1,-1), B(0,1), C(1,-1) are
// (0.25, 0.5, 0.25). A rigid pose (rotation plus positive uniform scale)
// preserves these ratios because it is an affine map.
constexpr float kInteriorBarycentricA = 0.25f;
constexpr float kInteriorBarycentricB = 0.5f;
constexpr float kInteriorBarycentricC = 0.25f;

void requireInteriorTriangleFeature(const ovspatialquery_geometry_feature_t& feature, int line)
{
    REQUIRE(feature.kind == OVSPATIALQUERY_GEOMETRY_FEATURE_TRIANGLE);
    REQUIRE(feature.data.triangle.source_face_index == 0u);
    REQUIRE(feature.data.triangle.source_face_corner_indices[0] == 0u);
    REQUIRE(feature.data.triangle.source_face_corner_indices[1] == 1u);
    REQUIRE(feature.data.triangle.source_face_corner_indices[2] == 2u);
    requireNear(feature.data.triangle.barycentrics[0], kInteriorBarycentricA, 1.0e-6, line, "barycentrics[0]");
    requireNear(feature.data.triangle.barycentrics[1], kInteriorBarycentricB, 1.0e-6, line, "barycentrics[1]");
    requireNear(feature.data.triangle.barycentrics[2], kInteriorBarycentricC, 1.0e-6, line, "barycentrics[2]");
}

// ADR-0065: OVSPATIALQUERY_QUERY_MESH_BOTH_SIDES lets the single-sided wedge count
// from both sides for one query or one batch launch. Every probe starts at X=3 and
// moves -X, so it meets the back sides of the faces at X=2 and X=1 (outward normal
// -X). A ray reports them with the front normal -X, as PhysX's eMESH_BOTH_SIDES does,
// while the double-sided wedge still reports +X, facing the ray. A sphere or box
// sweep, which ADR-0051 lets ignore a single-sided triangle whose front normal
// points along the motion, stops at X=2 instead. Without the flag every probe
// misses. Overlaps accept the flag; an unknown bit is NOT_SUPPORTED everywhere.
void requireMeshBothSidesCountsTheSingleSidedWedgeFromBehind(ovspatialquery_instance_handle_t instance,
                                                             ovspatialquery_query_target_t    target,
                                                             ovspatialquery_query_target_t    doubleSidedTarget)
{
    constexpr uint32_t kBothSides = OVSPATIALQUERY_QUERY_MESH_BOTH_SIDES;
    constexpr uint32_t kUnknown   = 2u;
    for (const uint32_t flags : {0u, kBothSides, kBothSides | kUnknown})
    {
        const bool                        hits = flags == kBothSides;
        const ovspatialquery_api_status_t expect =
            flags == (kBothSides | kUnknown) ? OVSPATIALQUERY_API_NOT_SUPPORTED : OVSPATIALQUERY_API_SUCCESS;
        ovspatialquery_raycast_desc_t ray{};
        ray.common.target         = target;
        ray.common.precision      = OVSPATIALQUERY_QUERY_EXACT;
        ray.common.query_flags    = flags;
        ray.ray.origin[0]         = 3.0;
        ray.ray.unit_direction[0] = -1.0f;
        ray.ray.max_distance      = 10.0;
        ovspatialquery_raycast_single_result_t single{};
        REQUIRE(ovspatialquery_raycast_closest_sync(instance, &ray, &single).status == expect);
        if (expect != OVSPATIALQUERY_API_SUCCESS)
        {
            ovspatialquery_sphere_sweep_desc_t sphere{};
            sphere.common                  = ray.common;
            sphere.sweep.center[0]         = 3.0;
            sphere.sweep.radius            = 0.25;
            sphere.sweep.unit_direction[0] = -1.0f;
            sphere.sweep.max_distance      = 10.0;
            ovspatialquery_sphere_sweep_closest_result_t swept{};
            REQUIRE(ovspatialquery_sweep_sphere_closest_sync(instance, &sphere, &swept).status == expect);
            ovspatialquery_overlap_desc_t overlap{};
            overlap.common                   = ray.common;
            overlap.shape.kind               = OVSPATIALQUERY_OVERLAP_SPHERE;
            overlap.shape.data.sphere.radius = 0.25;
            ovspatialquery_overlap_any_result_t overlapped{};
            REQUIRE(ovspatialquery_overlap_any_sync(instance, &overlap, &overlapped).status == expect);
            continue;
        }
        REQUIRE(single.has_hit == (hits ? 1u : 0u));
        if (hits)
        {
            REQUIRE_NEAR(single.hit.data.exact.distance, 1.0, 1.0e-12);
            REQUIRE(single.hit.data.exact.feature.data.triangle.source_face_index == 1u);
            REQUIRE_NEAR(single.hit.data.exact.normal[0], -1.0, 1.0e-6);
            // The double-sided wedge reports the normal facing the ray with the flag too.
            ovspatialquery_raycast_desc_t doubleSided = ray;
            doubleSided.common.target                 = doubleSidedTarget;
            ovspatialquery_raycast_single_result_t facing{};
            REQUIRE(ovspatialquery_raycast_closest_sync(instance, &doubleSided, &facing).status ==
                    OVSPATIALQUERY_API_SUCCESS);
            REQUIRE(facing.has_hit == 1u);
            REQUIRE_NEAR(facing.hit.data.exact.normal[0], 1.0, 1.0e-6);
            // A ray with a contact tolerance is evaluated as a thin sweep; its back-face normal
            // follows the same rule, front normal on the single-sided wedge, facing the ray on
            // the double-sided one, touching the face at X=2 one tolerance early.
            const std::pair<ovspatialquery_query_target_t, double> tubes[] = {{target, -1.0}, {doubleSidedTarget, 1.0}};
            for (const auto& [tube, normalX] : tubes)
            {
                ovspatialquery_raycast_desc_t tolerant = ray;
                tolerant.common.target                 = tube;
                tolerant.common.contact_tolerance      = 0.1;
                ovspatialquery_raycast_single_result_t widened{};
                REQUIRE(ovspatialquery_raycast_closest_sync(instance, &tolerant, &widened).status ==
                        OVSPATIALQUERY_API_SUCCESS);
                REQUIRE(widened.has_hit == 1u);
                REQUIRE_NEAR(widened.hit.data.exact.distance, 0.9, 1.0e-9);
                REQUIRE_NEAR(widened.hit.data.exact.normal[0], normalX, 1.0e-6);
            }
        }
        REQUIRE(ovspatialquery_raycast_any_sync(instance, &ray, &single).status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(single.has_hit == (hits ? 1u : 0u));
        std::array<ovspatialquery_raycast_hit_t, 4u> rows{};
        ovspatialquery_raycast_hit_buffer_t          buffer{rows.data(), rows.size()};
        ovspatialquery_multiple_result_t             multiple{};
        REQUIRE(ovspatialquery_raycast_multiple_sync(instance, &ray, &buffer, &multiple).status ==
                OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(multiple.stored_hit_count == (hits ? 2u : 0u) && multiple.truncated == 0u);
        for (uint64_t index = 0u; index < multiple.stored_hit_count; ++index)
        {
            REQUIRE_NEAR(rows[index].data.exact.normal[0], -1.0, 1.0e-6);
        }

        ovspatialquery_sphere_sweep_desc_t sphere{};
        sphere.common                  = ray.common;
        sphere.sweep.center[0]         = 3.0;
        sphere.sweep.radius            = 0.25;
        sphere.sweep.unit_direction[0] = -1.0f;
        sphere.sweep.max_distance      = 10.0;
        ovspatialquery_sphere_sweep_closest_result_t swept{};
        REQUIRE(ovspatialquery_sweep_sphere_closest_sync(instance, &sphere, &swept).status ==
                OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(swept.has_hit == (hits ? 1u : 0u));
        if (hits)
        {
            // A sweep keeps its contact normal, from the contact toward the shape's center (+X),
            // which here is the triangle's back side.
            REQUIRE_NEAR(swept.hit.data.exact.distance, 0.75, 1.0e-9);
            REQUIRE_NEAR(swept.hit.data.exact.normal[0], 1.0, 1.0e-6);
        }
        ovspatialquery_box_sweep_desc_t box{};
        box.common                = ray.common;
        box.sweep.center[0]       = 3.0;
        box.sweep.half_extents[0] = box.sweep.half_extents[1] = box.sweep.half_extents[2] = 0.25;
        box.sweep.rotation_xyzw[3]                                                        = 1.0f;
        box.sweep.unit_direction[0]                                                       = -1.0f;
        box.sweep.max_distance                                                            = 10.0;
        REQUIRE(ovspatialquery_sweep_box_closest_sync(instance, &box, &swept).status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(swept.has_hit == (hits ? 1u : 0u));
        if (hits)
        {
            // A sweep keeps its contact normal, from the contact toward the shape's center (+X),
            // which here is the triangle's back side.
            REQUIRE_NEAR(swept.hit.data.exact.distance, 0.75, 1.0e-9);
            REQUIRE_NEAR(swept.hit.data.exact.normal[0], 1.0, 1.0e-6);
        }

        ovspatialquery_overlap_desc_t overlap{};
        overlap.common                      = ray.common;
        overlap.shape.kind                  = OVSPATIALQUERY_OVERLAP_SPHERE;
        overlap.shape.data.sphere.center[0] = 2.0;
        overlap.shape.data.sphere.radius    = 0.25;
        ovspatialquery_overlap_any_result_t overlapped{};
        REQUIRE(ovspatialquery_overlap_any_sync(instance, &overlap, &overlapped).status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(overlapped.has_hit == 1u);
    }

    // CPU plans take the flags per launch: a launch without them misses on every row, one
    // with MESH_BOTH_SIDES hits on every row, and an unknown bit is refused at enqueue.
    constexpr uint64_t kRows = 2u;
    double             starts[3 * kRows]{3, 0, 0, 3, 0.25, -0.25};
    float              directions[3 * kRows]{-1, 0, 0, -1, 0, 0};
    float              rotations[4 * kRows]{0, 0, 0, 1, 0, 0, 0, 1};
    double             maximums[kRows]{10, 10};
    InputArray         startColumn(starts, kRows, 3, kDLFloat), directionColumn(directions, kRows, 3, kDLFloat),
        rotationColumn(rotations, kRows, 4, kDLFloat), maximumColumn(maximums, kRows, kDLFloat);
    // Runs one launch through `enqueue` and checks its rows: every row hits at `distance`
    // when `hits`, else none does, with normal x `normalX`: a raycast's is the front normal -X,
    // a sweep's points from the contact toward the shape's center, +X.
    const auto requireLaunch =
        [&](auto enqueue, auto fetch, auto columnsOf, bool hits, uint64_t perRow, double distance, double normalX)
    {
        ovspatialquery_enqueue_result_t launch = enqueue();
        REQUIRE(launch.status == OVSPATIALQUERY_API_SUCCESS);
        ovspatialquery_op_wait_result_t waited{};
        REQUIRE(ovspatialquery_wait_op(instance, launch.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status ==
                OVSPATIALQUERY_API_SUCCESS);
        const auto  result   = fetch(launch.op_id);
        const auto& columns  = columnsOf(result);
        const auto* statuses = static_cast<const uint32_t*>(columns.hit_ranges.query_status.data);
        const auto* offsets  = static_cast<const uint64_t*>(columns.hit_ranges.hit_offsets.data);
        const auto* counts   = static_cast<const uint64_t*>(columns.hit_ranges.hit_counts.data);
        for (uint64_t row = 0u; row < kRows; ++row)
        {
            REQUIRE(statuses[row] == OVSPATIALQUERY_BATCH_ROW_SUCCESS);
            REQUIRE(counts[row] == (hits ? perRow : 0u));
            if (hits)
            {
                const uint64_t first = offsets[row];
                REQUIRE_NEAR(static_cast<const double*>(columns.contact.exact.distances.data)[first], distance, 1.0e-9);
                REQUIRE_NEAR(static_cast<const float*>(columns.contact.exact.normals.data)[3 * first], normalX, 1.0e-6);
            }
        }
        REQUIRE(ovspatialquery_destroy_op(instance, &launch.op_id).status == OVSPATIALQUERY_API_SUCCESS);
    };
    const DLDevice cpu{kDLCPU, 0};
    const auto     fields =
        static_cast<uint32_t>(OVSPATIALQUERY_BATCH_RESULT_DISTANCE | OVSPATIALQUERY_BATCH_RESULT_NORMAL);
    for (const ovspatialquery_hit_mode_t mode : {OVSPATIALQUERY_HIT_MODE_CLOSEST, OVSPATIALQUERY_HIT_MODE_MULTIPLE})
    {
        const bool                         multiple = mode == OVSPATIALQUERY_HIT_MODE_MULTIPLE;
        ovspatialquery_raycast_plan_desc_t planDesc{};
        planDesc.target                         = target.target;
        planDesc.hit_mode                       = mode;
        planDesc.precision                      = OVSPATIALQUERY_QUERY_EXACT;
        planDesc.device                         = cpu;
        planDesc.maximum_query_count            = kRows;
        planDesc.maximum_hits_per_query         = multiple ? 2u : 1u;
        planDesc.hit_capacity_per_slot          = multiple ? 2u * kRows : kRows;
        planDesc.result_slot_count              = 1u;
        planDesc.result_fields                  = fields;
        ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
        REQUIRE(ovspatialquery_create_raycast_plan(instance, &planDesc, &plan).status == OVSPATIALQUERY_API_SUCCESS);
        ovspatialquery_raycast_batch_t batch{};
        batch.ordinal.kind    = OVSPATIALQUERY_ORDINAL_HEAD;
        batch.query_count     = kRows;
        batch.origins         = &startColumn.value;
        batch.unit_directions = &directionColumn.value;
        batch.max_distances   = &maximumColumn.value;
        ovspatialquery_raycast_batch_reservation_t reservation{};
        batch.query_flags = kBothSides | kUnknown;
        REQUIRE(ovspatialquery_enqueue_raycast_plan(instance, plan, &batch, &reservation).status ==
                OVSPATIALQUERY_API_NOT_SUPPORTED);
        for (const uint32_t flags : {0u, kBothSides})
        {
            batch.query_flags = flags;
            requireLaunch(
                [&] { return ovspatialquery_enqueue_raycast_plan(instance, plan, &batch, &reservation); },
                [&](ovspatialquery_op_id_t operation)
                {
                    ovspatialquery_raycast_batch_result_t result{};
                    REQUIRE(ovspatialquery_fetch_raycast_batch(instance, operation, &result).status ==
                            OVSPATIALQUERY_API_SUCCESS);
                    return result;
                },
                [](const ovspatialquery_raycast_batch_result_t& result) -> const auto& { return result.columns; },
                flags != 0u, multiple ? 2u : 1u, 1.0, -1.0);
        }
        REQUIRE(ovspatialquery_destroy_batch_plan(instance, plan).status == OVSPATIALQUERY_API_SUCCESS);
    }
    for (const bool boxSweep : {false, true})
    {
        ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
        if (boxSweep)
        {
            ovspatialquery_box_sweep_plan_desc_t planDesc{};
            planDesc.target          = target.target;
            planDesc.precision       = OVSPATIALQUERY_QUERY_EXACT;
            planDesc.half_extents[0] = planDesc.half_extents[1] = planDesc.half_extents[2] = 0.25;
            planDesc.device                                                                = cpu;
            planDesc.maximum_query_count                                                   = kRows;
            planDesc.result_slot_count                                                     = 1u;
            planDesc.result_fields                                                         = fields;
            planDesc.hit_capacity_per_slot                                                 = kRows;
            REQUIRE(ovspatialquery_create_box_sweep_plan(instance, &planDesc, &plan).status ==
                    OVSPATIALQUERY_API_SUCCESS);
        }
        else
        {
            ovspatialquery_sphere_sweep_plan_desc_t planDesc{};
            planDesc.target                = target.target;
            planDesc.precision             = OVSPATIALQUERY_QUERY_EXACT;
            planDesc.radius                = 0.25;
            planDesc.device                = cpu;
            planDesc.maximum_query_count   = kRows;
            planDesc.result_slot_count     = 1u;
            planDesc.result_fields         = fields;
            planDesc.hit_capacity_per_slot = kRows;
            REQUIRE(ovspatialquery_create_sphere_sweep_plan(instance, &planDesc, &plan).status ==
                    OVSPATIALQUERY_API_SUCCESS);
        }
        ovspatialquery_box_sweep_batch_t box{};
        box.ordinal.kind    = OVSPATIALQUERY_ORDINAL_HEAD;
        box.query_count     = kRows;
        box.centers         = &startColumn.value;
        box.rotations_xyzw  = &rotationColumn.value;
        box.unit_directions = &directionColumn.value;
        box.max_distances   = &maximumColumn.value;
        ovspatialquery_sphere_sweep_batch_t sphere{};
        sphere.ordinal.kind    = OVSPATIALQUERY_ORDINAL_HEAD;
        sphere.query_count     = kRows;
        sphere.centers         = &startColumn.value;
        sphere.unit_directions = &directionColumn.value;
        sphere.max_distances   = &maximumColumn.value;
        ovspatialquery_sphere_sweep_batch_reservation_t reservation{};
        const auto                                      enqueue = [&](uint32_t flags)
        {
            box.query_flags = sphere.query_flags = flags;
            return boxSweep ? ovspatialquery_enqueue_box_sweep_plan(instance, plan, &box, &reservation)
                            : ovspatialquery_enqueue_sphere_sweep_plan(instance, plan, &sphere, &reservation);
        };
        REQUIRE(enqueue(kBothSides | kUnknown).status == OVSPATIALQUERY_API_NOT_SUPPORTED);
        for (const uint32_t flags : {0u, kBothSides})
        {
            requireLaunch(
                [&] { return enqueue(flags); },
                [&](ovspatialquery_op_id_t operation)
                {
                    ovspatialquery_sphere_sweep_batch_result_t result{};
                    REQUIRE((boxSweep ? ovspatialquery_fetch_box_sweep_batch(instance, operation, &result)
                                      : ovspatialquery_fetch_sphere_sweep_batch(instance, operation, &result))
                                .status == OVSPATIALQUERY_API_SUCCESS);
                    return result;
                },
                [](const ovspatialquery_sphere_sweep_batch_result_t& result) -> const auto& { return result.columns; },
                flags != 0u, 1u, 0.75, 1.0);
        }
        REQUIRE(ovspatialquery_destroy_batch_plan(instance, plan).status == OVSPATIALQUERY_API_SUCCESS);
    }
}

// `double_sided` belongs to the immutable mesh definition, so the two variants
// are two definitions: a ray that reaches only the triangles' back sides misses
// the single-sided one and hits the double-sided one, its normal facing the ray.
void testTriangleMeshSidednessRejectsBackFaceUnlessDefinitionDoubleSided()
{
    ovspatialquery_instance_handle_t instance = createInstance();
    const uint64_t singleSidedValue = registerTriangleMesh(instance, 5101u, wedgeVertices(), wedgeIndices(), false);
    const uint64_t doubleSidedValue = registerTriangleMesh(instance, 5102u, wedgeVertices(), wedgeIndices(), true);

    TriangleInstance singleInstance;
    singleInstance.objectId      = 1u;
    singleInstance.geometryValue = singleSidedValue;
    const ovspatialquery_population_handle_t singleSidedPopulation =
        publishTrianglePopulation(instance, 1, {singleInstance});

    TriangleInstance doubleInstance;
    doubleInstance.objectId      = 2u;
    doubleInstance.geometryValue = doubleSidedValue;
    const ovspatialquery_population_handle_t doubleSidedPopulation =
        publishTrianglePopulation(instance, 1, {doubleInstance});

    // Approaching from +X toward -X meets both wedge triangles' back side:
    // their outward normal is -X (see wedgeVertices), so unit_direction (-1)
    // has a positive dot product with that normal.
    const double origin[3]    = {3.0, 0.0, 0.0};
    const float  direction[3] = {-1.0f, 0.0f, 0.0f};

    const ovspatialquery_raycast_single_result_t singleSidedResult =
        raycastClosest(instance, populationTarget(singleSidedPopulation), origin, direction, 10.0);
    REQUIRE(singleSidedResult.has_hit == 0u);

    const ovspatialquery_raycast_single_result_t doubleSidedResult =
        raycastClosest(instance, populationTarget(doubleSidedPopulation), origin, direction, 10.0);
    REQUIRE(doubleSidedResult.has_hit == 1u);
    const ovspatialquery_raycast_exact_hit_t& hit = doubleSidedResult.hit.data.exact;
    REQUIRE(hit.identity.object_id.value == 2u);
    REQUIRE_NEAR(hit.distance, 1.0, 1.0e-12);
    REQUIRE_NEAR(hit.position[0], 2.0, 1.0e-12);
    REQUIRE(hit.feature.kind == OVSPATIALQUERY_GEOMETRY_FEATURE_TRIANGLE);
    REQUIRE(hit.feature.data.triangle.source_face_index == 1u);
    REQUIRE_NEAR(hit.normal[0], 1.0, 1.0e-6);

    // Rounding the near-plane origin to float must not revive the excluded
    // face at X=1 or hide the valid second face at X=2.
    ovspatialquery_raycast_desc_t clipped{};
    clipped.common.target         = populationTarget(doubleSidedPopulation);
    clipped.common.precision      = OVSPATIALQUERY_QUERY_EXACT;
    clipped.ray.unit_direction[0] = 1.0f;
    clipped.ray.min_distance      = 1.0 + 1.0e-8;
    clipped.ray.max_distance      = 3.0;
    ovspatialquery_raycast_single_result_t clippedResult{};
    REQUIRE(ovspatialquery_raycast_closest_sync(instance, &clipped, &clippedResult).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(clippedResult.has_hit == 1u);
    REQUIRE_NEAR(clippedResult.hit.data.exact.distance, 2.0, 1.0e-12);
    clipped.ray.max_distance = 2.0 - 1.0e-8;
    REQUIRE(ovspatialquery_raycast_closest_sync(instance, &clipped, &clippedResult).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(clippedResult.has_hit == 0u);

    requireMeshBothSidesCountsTheSingleSidedWedgeFromBehind(instance, populationTarget(singleSidedPopulation),
                                                            populationTarget(doubleSidedPopulation));
    destroyMeshScene(instance, {singleSidedPopulation, doubleSidedPopulation}, {singleSidedValue, doubleSidedValue});
}

// A rotated, uniformly scaled instance carries position, distance, normal and
// barycentrics through its pose. Local vertex v maps to R*(scale*v) + t, with R a
// 180-degree rotation about Z (quaternion (0,0,1,0), matrix diag(-1,-1,1)): the
// local plane X=1 with outward normal -X lands on world X=3 facing +X, so a ray
// travelling -X hits the single-sided mesh.
void testTrianglePoseRotationAndUniformScaleTransformExactHit()
{
    ovspatialquery_instance_handle_t instance = createInstance();
    const uint64_t                   geometryValue =
        registerTriangleMesh(instance, 5201u, singleTriangleVertices(), singleTriangleIndices(), false);
    TriangleInstance placement;
    placement.objectId                                  = 1u;
    placement.geometryValue                             = geometryValue;
    placement.translation[0]                            = 5.0;
    placement.rotationXyzw[2]                           = 1.0f; // 180 degrees about Z: (0,0,1,0)
    placement.rotationXyzw[3]                           = 0.0f;
    placement.uniformScale                              = 2.0;
    const ovspatialquery_population_handle_t population = publishTrianglePopulation(instance, 1, {placement});

    const double                                 origin[3]    = {10.0, 0.0, 0.0};
    const float                                  direction[3] = {-1.0f, 0.0f, 0.0f};
    const ovspatialquery_raycast_single_result_t result =
        raycastClosest(instance, populationTarget(population), origin, direction, 10.0);

    REQUIRE(result.has_hit == 1u);
    const ovspatialquery_raycast_exact_hit_t& hit = result.hit.data.exact;
    REQUIRE_NEAR(hit.distance, 7.0, 1.0e-9);
    REQUIRE_NEAR(hit.position[0], 3.0, 1.0e-9);
    REQUIRE_NEAR(hit.position[1], 0.0, 1.0e-9);
    REQUIRE_NEAR(hit.position[2], 0.0, 1.0e-9);
    // The local normal (-1,0,0) rotates the same way as any local vector:
    // diag(-1,-1,1) * (-1,0,0) = (1,0,0). Uniform scale does not change
    // normal direction.
    REQUIRE_NEAR(hit.normal[0], 1.0, 1.0e-6);
    REQUIRE_NEAR(hit.normal[1], 0.0, 1.0e-6);
    REQUIRE_NEAR(hit.normal[2], 0.0, 1.0e-6);
    requireInteriorTriangleFeature(hit.feature, __LINE__);

    destroyMeshScene(instance, {population}, {geometryValue});
}

// A ray inside the wedge's world AABB (Y,Z in [-1,1]) that misses both triangles
// reports no hit under EXACT precision and the AABB contact under
// BOUNDS_CANDIDATE. At Y=Z=0.9 it passes outside both triangles, which reach
// only Z<=-0.8 once |Y|>0.8.
void testTriangleRaycastExactMissDoesNotFallBackToBoundsCandidate()
{
    ovspatialquery_instance_handle_t instance = createInstance();
    const uint64_t   geometryValue = registerTriangleMesh(instance, 5301u, wedgeVertices(), wedgeIndices(), false);
    TriangleInstance placement;
    placement.objectId                                  = 1u;
    placement.geometryValue                             = geometryValue;
    const ovspatialquery_population_handle_t population = publishTrianglePopulation(instance, 1, {placement});

    const double origin[3]    = {0.0, 0.9, 0.9};
    const float  direction[3] = {1.0f, 0.0f, 0.0f};

    const ovspatialquery_raycast_single_result_t exactResult =
        raycastClosest(instance, populationTarget(population), origin, direction, 10.0, OVSPATIALQUERY_QUERY_EXACT);
    REQUIRE(exactResult.has_hit == 0u);

    const ovspatialquery_raycast_single_result_t candidateResult = raycastClosest(
        instance, populationTarget(population), origin, direction, 10.0, OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE);
    REQUIRE(candidateResult.has_hit == 1u);
    REQUIRE(candidateResult.hit.truth == OVSPATIALQUERY_HIT_BOUNDS_CANDIDATE);
    REQUIRE(candidateResult.hit.data.bounds_candidate.identity.object_id.value == 1u);

    destroyMeshScene(instance, {population}, {geometryValue});
}

// A flat triangle's world AABB has zero thickness in X. BOUNDS_CANDIDATE
// reports the face the ray enters through: +X from the far side, -X from
// the near side, although both faces sit at the same X. The synchronous query
// and a CPU raycast plan reach the slab test through different callers.
void testBoundsCandidateNormalOfFlatAabbFacesTheRay()
{
    ovspatialquery_instance_handle_t instance = createInstance();
    const uint64_t                   geometryValue =
        registerTriangleMesh(instance, 5302u, singleTriangleVertices(), singleTriangleIndices(), true);
    TriangleInstance placement;
    placement.objectId                                  = 1u;
    placement.geometryValue                             = geometryValue;
    const ovspatialquery_population_handle_t population = publishTrianglePopulation(instance, 1, {placement});

    for (const float side : {1.0f, -1.0f})
    {
        const double                                 origin[3]    = {1.0 + 2.0 * side, 0.0, -0.5};
        const float                                  direction[3] = {-side, 0.0f, 0.0f};
        const ovspatialquery_raycast_single_result_t result       = raycastClosest(
            instance, populationTarget(population), origin, direction, 10.0, OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE);
        REQUIRE(result.has_hit == 1u);
        const ovspatialquery_raycast_bounds_candidate_t& hit = result.hit.data.bounds_candidate;
        REQUIRE(hit.distance == 2.0 && hit.bounds_position[0] == 1.0);
        REQUIRE(hit.bounds_normal[0] == side && hit.bounds_normal[1] == 0.0f && hit.bounds_normal[2] == 0.0f);
    }

    ovspatialquery_raycast_plan_desc_t planDesc{};
    planDesc.target                         = populationRef(population);
    planDesc.hit_mode                       = OVSPATIALQUERY_HIT_MODE_CLOSEST;
    planDesc.precision                      = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    planDesc.device                         = DLDevice{kDLCPU, 0};
    planDesc.maximum_query_count            = 2u;
    planDesc.maximum_hits_per_query         = 1u;
    planDesc.hit_capacity_per_slot          = 2u;
    planDesc.result_slot_count              = 1u;
    planDesc.result_fields                  = OVSPATIALQUERY_BATCH_RESULT_NORMAL;
    ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_raycast_plan(instance, &planDesc, &plan).status == OVSPATIALQUERY_API_SUCCESS);
    double                         origins[2][3]    = {{3.0, 0.0, -0.5}, {-1.0, 0.0, -0.5}};
    float                          directions[2][3] = {{-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}};
    double                         maxDistances[2]  = {10.0, 10.0};
    InputArray                     originArray(origins[0], 2, 3, kDLFloat);
    InputArray                     directionArray(directions[0], 2, 3, kDLFloat);
    InputArray                     maxArray(maxDistances, 2, kDLFloat);
    ovspatialquery_raycast_batch_t batch{};
    batch.ordinal.kind    = OVSPATIALQUERY_ORDINAL_HEAD;
    batch.query_count     = 2u;
    batch.origins         = &originArray.value;
    batch.unit_directions = &directionArray.value;
    batch.max_distances   = &maxArray.value;
    ovspatialquery_raycast_batch_reservation_t reservation{};
    const ovspatialquery_enqueue_result_t      enqueued =
        ovspatialquery_enqueue_raycast_plan(instance, plan, &batch, &reservation);
    REQUIRE(enqueued.status == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_op_wait_result_t waited{};
    REQUIRE(ovspatialquery_wait_op(instance, enqueued.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status ==
            OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_raycast_batch_result_t planResult{};
    REQUIRE(ovspatialquery_fetch_raycast_batch(instance, enqueued.op_id, &planResult).status ==
            OVSPATIALQUERY_API_SUCCESS);
    const uint64_t* offsets = static_cast<const uint64_t*>(planResult.columns.hit_ranges.hit_offsets.data);
    const uint64_t* counts  = static_cast<const uint64_t*>(planResult.columns.hit_ranges.hit_counts.data);
    const float*    normals = static_cast<const float*>(planResult.columns.contact.candidate.bounds_normals.data);
    for (uint32_t row = 0u; row < 2u; ++row)
    {
        REQUIRE(counts[row] == 1u && offsets[row] < 2u);
        const float* normal = normals + 3u * offsets[row];
        REQUIRE(normal[0] == -directions[row][0] && normal[1] == 0.0f && normal[2] == 0.0f);
    }
    ovspatialquery_op_id_t operation = enqueued.op_id;
    REQUIRE(ovspatialquery_destroy_op(instance, &operation).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_batch_plan(instance, plan).status == OVSPATIALQUERY_API_SUCCESS);

    destroyMeshScene(instance, {population}, {geometryValue});
}

// ovspatialquery_raycast_multiple_sync reports one hit per triangle crossing by
// default and one per object with MESH_HITS_NEAREST, in unspecified row order,
// so the hits are checked as a set; a short buffer keeps the nearest and sets
// truncated.
void testTriangleRaycastMultipleReportsAllHitsWithTruncationContract()
{
    ovspatialquery_instance_handle_t instance = createInstance();
    const uint64_t geometryValue = registerTriangleMesh(instance, 5401u, wedgeVertices(), wedgeIndices(), false);
    // Each wedge has a front-facing triangle at local X=1 and X=2. With
    // NEAREST, two instances along the ray give two hits: the near wedge's
    // first face at X=1 and the far wedge's at X=6.
    TriangleInstance nearInstance;
    nearInstance.objectId      = 1u;
    nearInstance.geometryValue = geometryValue;
    TriangleInstance farInstance;
    farInstance.objectId       = 2u;
    farInstance.geometryValue  = geometryValue;
    farInstance.translation[0] = 5.0;
    const ovspatialquery_population_handle_t population =
        publishTrianglePopulation(instance, 1, {nearInstance, farInstance});

    ovspatialquery_raycast_desc_t desc{};
    desc.common.target         = populationTarget(population);
    desc.common.precision      = OVSPATIALQUERY_QUERY_EXACT;
    desc.ray.unit_direction[0] = 1.0f;
    desc.ray.max_distance      = 10.0;
    desc.mesh_hits             = OVSPATIALQUERY_MESH_HITS_NEAREST;

    ovspatialquery_raycast_hit_t        shortHits[1]{};
    ovspatialquery_raycast_hit_buffer_t shortBuffer{shortHits, 1u};
    ovspatialquery_multiple_result_t    shortResult{};
    REQUIRE(ovspatialquery_raycast_multiple_sync(instance, &desc, &shortBuffer, &shortResult).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(shortResult.source_ordinal == 1);
    REQUIRE(shortResult.stored_hit_count == 1u);
    REQUIRE(shortResult.truncated == 1u);
    const double shortDistance = shortHits[0].data.exact.distance;
    REQUIRE(std::abs(shortDistance - 1.0) <= 1.0e-9 || std::abs(shortDistance - 6.0) <= 1.0e-9);

    ovspatialquery_raycast_hit_t        fullHits[2]{};
    ovspatialquery_raycast_hit_buffer_t fullBuffer{fullHits, 2u};
    ovspatialquery_multiple_result_t    fullResult{};
    REQUIRE(ovspatialquery_raycast_multiple_sync(instance, &desc, &fullBuffer, &fullResult).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(fullResult.stored_hit_count == 2u);
    REQUIRE(fullResult.truncated == 0u);
    const auto [nearDistance, farDistance] =
        std::minmax(fullHits[0].data.exact.distance, fullHits[1].data.exact.distance);
    REQUIRE_NEAR(nearDistance, 1.0, 1.0e-9);
    REQUIRE_NEAR(farDistance, 6.0, 1.0e-9);

    // The default reports all four crossings, each with its own source face;
    // a buffer of three keeps the three nearest.
    desc.mesh_hits = OVSPATIALQUERY_MESH_HITS_ALL;
    ovspatialquery_raycast_hit_t        crossings[5]{};
    ovspatialquery_raycast_hit_buffer_t crossingBuffer{crossings, 5u};
    ovspatialquery_multiple_result_t    crossingResult{};
    REQUIRE(ovspatialquery_raycast_multiple_sync(instance, &desc, &crossingBuffer, &crossingResult).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(crossingResult.stored_hit_count == 4u && crossingResult.truncated == 0u);
    std::vector<std::pair<double, uint64_t>> found;
    for (uint64_t row = 0u; row < 4u; ++row)
    {
        const ovspatialquery_raycast_exact_hit_t& hit = crossings[row].data.exact;
        REQUIRE(hit.feature.kind == OVSPATIALQUERY_GEOMETRY_FEATURE_TRIANGLE);
        found.emplace_back(hit.distance, hit.feature.data.triangle.source_face_index);
    }
    std::sort(found.begin(), found.end());
    const std::pair<double, uint64_t> expected[] = {{1.0, 0u}, {2.0, 1u}, {6.0, 0u}, {7.0, 1u}};
    for (size_t index = 0u; index < 4u; ++index)
    {
        REQUIRE_NEAR(found[index].first, expected[index].first, 1.0e-6);
        REQUIRE(found[index].second == expected[index].second);
    }
    crossingBuffer.capacity = 3u;
    REQUIRE(ovspatialquery_raycast_multiple_sync(instance, &desc, &crossingBuffer, &crossingResult).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(crossingResult.stored_hit_count == 3u && crossingResult.truncated == 1u);
    double farthest = 0.0;
    for (uint64_t row = 0u; row < 3u; ++row)
    {
        farthest = std::max(farthest, crossings[row].data.exact.distance);
    }
    REQUIRE_NEAR(farthest, 6.0, 1.0e-6);

    // A tolerance tube has no crossing count; NEAREST still serves it.
    desc.common.contact_tolerance = 0.01;
    REQUIRE(ovspatialquery_raycast_multiple_sync(instance, &desc, &crossingBuffer, &crossingResult).status ==
            OVSPATIALQUERY_API_NOT_SUPPORTED);
    desc.mesh_hits = OVSPATIALQUERY_MESH_HITS_NEAREST;
    REQUIRE(ovspatialquery_raycast_multiple_sync(instance, &desc, &crossingBuffer, &crossingResult).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(crossingResult.stored_hit_count == 2u);
    desc.mesh_hits = static_cast<ovspatialquery_mesh_hits_t>(2);
    REQUIRE(ovspatialquery_raycast_multiple_sync(instance, &desc, &crossingBuffer, &crossingResult).status ==
            OVSPATIALQUERY_API_INVALID_ARGUMENT);

    destroyMeshScene(instance, {population}, {geometryValue});
}

// A crossing exactly at max_distance whose PhysX contact position rounds just
// beyond it: every-crossing mode decides that triangle exactly, as the nearest
// mode's fallback does, instead of dropping it.
void testTriangleRaycastMultipleKeepsRoundedEndpointCrossing()
{
    ovspatialquery_instance_handle_t instance      = createInstance();
    const uint64_t                   geometryValue = registerTriangleMesh(
        instance, 5601u, {2.0f, -1.0f, -1.0f, 2.0f, 1.0f, -1.0f, 2.0f, 0.0f, 1.0f}, {0u, 1u, 2u}, true);
    TriangleInstance placement;
    placement.objectId                                  = 1u;
    placement.geometryValue                             = geometryValue;
    const ovspatialquery_population_handle_t population = publishTrianglePopulation(instance, 1, {placement});
    ovspatialquery_raycast_desc_t            desc{};
    desc.common.target         = populationTarget(population);
    desc.ray.unit_direction[0] = 0.9805806875228882f;
    desc.ray.unit_direction[1] = 0.1961161345243454f;
    desc.ray.max_distance      = 2.0 / double(desc.ray.unit_direction[0]);
    static constexpr std::array kPrecisions{OVSPATIALQUERY_QUERY_EXACT, OVSPATIALQUERY_QUERY_EXACT_SINGLE};
    static constexpr std::array kMeshHits{OVSPATIALQUERY_MESH_HITS_ALL, OVSPATIALQUERY_MESH_HITS_NEAREST};
    for (const auto precision : kPrecisions)
    {
        for (const auto meshHits : kMeshHits)
        {
            desc.common.precision = precision;
            desc.mesh_hits        = meshHits;
            ovspatialquery_raycast_hit_t        rows[2]{};
            ovspatialquery_raycast_hit_buffer_t buffer{rows, 2u};
            ovspatialquery_multiple_result_t    result{};
            REQUIRE(ovspatialquery_raycast_multiple_sync(instance, &desc, &buffer, &result).status ==
                    OVSPATIALQUERY_API_SUCCESS);
            REQUIRE(result.stored_hit_count == 1u && result.truncated == 0u);
            REQUIRE_NEAR(rows[0].data.exact.distance, desc.ray.max_distance, 1.0e-6);
        }
    }
    destroyMeshScene(instance, {population}, {geometryValue});
}

// One ray crossing 70 triangles of one mesh: more than PhysX's 64-hit inline
// buffer, so the CPU retries with a larger charged one, and with a short row
// limit the nearest-crossings store compacts many times. Both must keep the
// nearest crossings: every plate with room for them, the first five without.
void testTriangleRaycastMultipleKeepsNearestOfManyCrossings()
{
    ovspatialquery_instance_handle_t instance = createInstance();
    constexpr uint32_t               kPlates  = 70u;
    std::vector<float>               vertices;
    std::vector<uint32_t>            indices;
    for (uint32_t plate = 0u; plate < kPlates; ++plate)
    {
        const float    x     = float(plate + 1u);
        const uint32_t first = plate * 4u;
        vertices.insert(vertices.end(), {x, -1.0f, -1.0f, x, 1.0f, -1.0f, x, 1.0f, 1.0f, x, -1.0f, 1.0f});
        indices.insert(indices.end(), {first, first + 1u, first + 2u, first, first + 2u, first + 3u});
    }
    const uint64_t   geometryValue = registerTriangleMesh(instance, 5501u, vertices, indices, true);
    TriangleInstance placement;
    placement.objectId                                  = 1u;
    placement.geometryValue                             = geometryValue;
    const ovspatialquery_population_handle_t population = publishTrianglePopulation(instance, 1, {placement});

    // y 0.25, z -0.5 stays off each plate's diagonal (y = z).
    ovspatialquery_raycast_desc_t desc{};
    desc.common.target         = populationTarget(population);
    desc.common.precision      = OVSPATIALQUERY_QUERY_EXACT;
    desc.ray.origin[1]         = 0.25;
    desc.ray.origin[2]         = -0.5;
    desc.ray.unit_direction[0] = 1.0f;
    desc.ray.max_distance      = 100.0;
    std::vector<ovspatialquery_raycast_hit_t> rows(80u);
    static constexpr std::array               kCapacities{uint64_t{80u}, uint64_t{5u}};
    for (const uint64_t capacity : kCapacities)
    {
        ovspatialquery_raycast_hit_buffer_t buffer{rows.data(), capacity};
        ovspatialquery_multiple_result_t    result{};
        REQUIRE(ovspatialquery_raycast_multiple_sync(instance, &desc, &buffer, &result).status ==
                OVSPATIALQUERY_API_SUCCESS);
        const uint64_t kept = std::min<uint64_t>(capacity, kPlates);
        REQUIRE(result.stored_hit_count == kept && result.truncated == (capacity < kPlates ? 1u : 0u));
        std::vector<double> distances;
        for (uint64_t row = 0u; row < kept; ++row)
        {
            distances.push_back(rows[row].data.exact.distance);
        }
        std::sort(distances.begin(), distances.end());
        for (uint64_t row = 0u; row < kept; ++row)
        {
            REQUIRE_NEAR(distances[row], double(row + 1u), 1.0e-5);
        }
    }

    destroyMeshScene(instance, {population}, {geometryValue});
}

// Controlled closest-sweep fixture for the CPU sweep clipping round: several
// instances of one front-facing triangle (local X=1 plane) placed so that the
// sweep meets them at equal or ordered distances. Exercises an exact double
// tie at a non-binary distance, the inclusive far endpoint, an earlier-entered
// instance whose contact is later, and initial overlaps that tie at distance
// zero. CLOSEST makes no tied-hit identity promise, so a tie may report either
// tied instance, but never a farther one. The expected values are the analytic
// contacts; the unclipped path must hit them too.
void testSphereSweepClosestTiesEndpointsAndInitialOverlapsAcrossInstances()
{
    ovspatialquery_instance_handle_t instance = createInstance();
    const uint64_t                   triangleValue =
        registerTriangleMesh(instance, 5901u, singleTriangleVertices(), singleTriangleIndices(), false);
    // A second definition whose bounds begin well before its only face on the
    // sweep line: one triangle off to the side at local X=0.6 and one at X=3.
    const uint64_t sidedValue   = registerTriangleMesh(instance, 5902u,
                                                       {0.6f, 3.0f, -1.0f, 0.6f, 3.5f, 1.0f, 0.6f, 4.0f, -1.0f, 3.0f,
                                                        -1.0f, -1.0f, 3.0f, 0.0f, 1.0f, 3.0f, 1.0f, -1.0f},
                                                       {0u, 1u, 2u, 3u, 4u, 5u}, false);
    const float    direction[3] = {1.0f, 0.0f, 0.0f};

    // Two planes at world X=1.3 (translation 0.3 is not binary) whose faces
    // both contain the sweep line at Y=0: an exact double tie at 0.95.
    {
        TriangleInstance high;
        high.objectId       = 7u;
        high.geometryValue  = triangleValue;
        high.translation[0] = 0.3;
        high.translation[1] = 0.3;
        TriangleInstance low;
        low.objectId       = 3u;
        low.geometryValue  = triangleValue;
        low.translation[0] = 0.3;
        low.translation[1] = -0.3;
        // A farther instance whose bounds are entered first cannot displace
        // the tie; it is the nearest candidate the broadphase reports.
        TriangleInstance far;
        far.objectId                                        = 1u;
        far.geometryValue                                   = sidedValue;
        far.translation[0]                                  = -0.3;
        const ovspatialquery_population_handle_t population = publishTrianglePopulation(instance, 1, {high, far, low});
        const double                             center[3]  = {0.1, 0.0, 0.0};
        const ovspatialquery_sphere_sweep_closest_result_t tie =
            sweepClosest(instance, populationTarget(population), center, 0.25, direction, 10.0);
        REQUIRE(tie.has_hit == 1u);
        REQUIRE(tie.hit.truth == OVSPATIALQUERY_HIT_EXACT_SURFACE);
        const uint64_t tied = tie.hit.data.exact.identity.object_id.value;
        REQUIRE(tied == 3u || tied == 7u);
        REQUIRE_NEAR(tie.hit.data.exact.distance, 0.95, 1.0e-12);
        REQUIRE_NEAR(tie.hit.data.exact.object_position[0], 1.3, 1.0e-12);
        REQUIRE_NEAR(tie.hit.data.exact.object_position[1], 0.0, 1.0e-12);
        REQUIRE_NEAR(tie.hit.data.exact.normal[0], -1.0, 1.0e-6);

        // The far endpoint is inclusive: a range ending exactly at the tie
        // distance still reports it, one double step shorter reports nothing.
        const double                                       reported = tie.hit.data.exact.distance;
        const ovspatialquery_sphere_sweep_closest_result_t inclusive =
            sweepClosest(instance, populationTarget(population), center, 0.25, direction, reported);
        REQUIRE(inclusive.has_hit == 1u);
        REQUIRE(inclusive.hit.data.exact.identity.object_id.value == 3u ||
                inclusive.hit.data.exact.identity.object_id.value == 7u);
        REQUIRE_NEAR(inclusive.hit.data.exact.distance, reported, 0.0);
        const ovspatialquery_sphere_sweep_closest_result_t shorter = sweepClosest(
            instance, populationTarget(population), center, 0.25, direction, std::nextafter(reported, 0.0));
        REQUIRE(shorter.has_hit == 0u);

        // Off the tie line only the higher instance's face contains the sweep
        // line (local Y=0.3); the lower instance's edge stays out of reach.
        const double                                       highCenter[3] = {0.1, 0.6, 0.0};
        const ovspatialquery_sphere_sweep_closest_result_t single =
            sweepClosest(instance, populationTarget(population), highCenter, 0.25, direction, 10.0);
        REQUIRE(single.has_hit == 1u);
        REQUIRE(single.hit.data.exact.identity.object_id.value == 7u);
        REQUIRE_NEAR(single.hit.data.exact.distance, 0.95, 1.0e-12);

        // Past both planes the side mesh's second face is the only contact,
        // at X=2.7 minus the radius from X=1.6: 0.85, object 1.
        const double                                       behind[3] = {1.6, 0.0, 0.0};
        const ovspatialquery_sphere_sweep_closest_result_t sided =
            sweepClosest(instance, populationTarget(population), behind, 0.25, direction, 10.0);
        REQUIRE(sided.has_hit == 1u);
        REQUIRE(sided.hit.data.exact.identity.object_id.value == 1u);
        REQUIRE_NEAR(sided.hit.data.exact.distance, 0.85, 1.0e-12);
        REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    }

    // Initial overlaps: two instances already touching the start sphere tie
    // at distance zero, and either beats a farther third instance.
    {
        TriangleInstance nearer;
        nearer.objectId       = 9u;
        nearer.geometryValue  = triangleValue;
        nearer.translation[0] = 0.2;
        TriangleInstance lowest;
        lowest.objectId       = 6u;
        lowest.geometryValue  = triangleValue;
        lowest.translation[0] = 0.15;
        lowest.translation[1] = -0.3;
        TriangleInstance ahead;
        ahead.objectId       = 4u;
        ahead.geometryValue  = triangleValue;
        ahead.translation[0] = 2.0;
        ahead.translation[1] = 0.3;
        const ovspatialquery_population_handle_t population =
            publishTrianglePopulation(instance, 1, {ahead, nearer, lowest});
        const double                                       center[3] = {1.0, 0.0, 0.0};
        const ovspatialquery_sphere_sweep_closest_result_t initial =
            sweepClosest(instance, populationTarget(population), center, 0.25, direction, 10.0);
        REQUIRE(initial.has_hit == 1u);
        const uint64_t touched = initial.hit.data.exact.identity.object_id.value;
        REQUIRE(touched == 6u || touched == 9u);
        REQUIRE_NEAR(initial.hit.data.exact.distance, 0.0, 1.0e-12);
        REQUIRE_NEAR(initial.hit.data.exact.object_position[0], touched == 6u ? 1.15 : 1.2, 1.0e-12);
        REQUIRE_NEAR(initial.hit.data.exact.object_position[1], 0.0, 1.0e-12);
        REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    }

    destroyMeshScene(instance, {}, {triangleValue, sidedValue});
}

// Distributed faces exercise mesh-tree traversal; analytical contacts also
// protect float32 local vertices, double-valued rigid poses, scale, edge and
// initial contacts.
void testMeshSweepsAcrossDistributedFaces()
{
    for (uint32_t pose = 0u; pose < 3u; ++pose)
    {
        SingleTriangleFixture fixture;
        fixture.instance            = createInstance();
        constexpr float       plane = 1.0f;
        std::vector<float>    vertices;
        std::vector<uint32_t> triangles;
        for (uint32_t face = 0u; face < 64u; ++face)
        {
            const float y = static_cast<float>(face) * 4.0f;
            vertices.insert(vertices.end(), {plane, y - 1.0f, -1.0f, plane, y, 1.0f, plane, y + 1.0f, -1.0f});
            triangles.insert(triangles.end(), {face * 3u, face * 3u + 1u, face * 3u + 2u});
        }
        // A coincident face retains the lower source-face tie choice even
        // though tree traversal need not visit faces in their input order.
        triangles.insert(triangles.end(), {0u, 1u, 2u});
        fixture.geometryValue = registerTriangleMesh(fixture.instance, 5801u, vertices, triangles, false);
        TriangleInstance placement{};
        placement.objectId      = 1u;
        placement.geometryValue = fixture.geometryValue;
        const double sign       = pose == 2u ? -1.0 : 1.0;
        const double scale      = pose == 2u ? 2.0 : 1.0;
        if (pose == 1u)
        {
            // Local float32 coordinates remain well-conditioned; the double
            // instance pose carries the large world-space placement.
            placement.translation[0] = 16777216.25;
        }
        if (pose == 2u)
        {
            placement.rotationXyzw[2] = 1.0f;
            placement.rotationXyzw[3] = 0.0f;
            placement.uniformScale    = scale;
            placement.translation[0]  = 5.0;
        }
        fixture.population       = publishTrianglePopulation(fixture.instance, 1u, {placement});
        const float direction[3] = {static_cast<float>(sign), 0.0f, 0.0f};
        for (uint32_t face = 0u; face < 64u; ++face)
        {
            for (uint32_t contact = 0u; contact < 4u; ++contact)
            {
                const double y         = static_cast<double>(face) * 4.0;
                const double localY    = y + (contact == 1u ? 1.2 : (contact == 3u ? 2.4 : 0.0));
                const double localZ    = contact == 1u ? -1.0 : 0.0;
                const double center[3] = {placement.translation[0] +
                                              sign * scale * (plane - (contact == 2u ? 0.125 : 1.0)),
                                          sign * scale * localY, scale * localZ};
                const auto   result    = sweepClosest(fixture.instance, populationTarget(fixture.population), center,
                                                      scale * 0.25, direction, scale * 2.0);
                REQUIRE(result.has_hit == (contact == 3u ? 0u : 1u));
                if (!result.has_hit)
                    continue;
                const auto& hit = result.hit.data.exact;
                REQUIRE_NEAR(hit.distance, scale * (contact == 2u ? 0.0 : (contact == 1u ? 0.85 : 0.75)), 1.0e-8);
                REQUIRE_NEAR(hit.object_position[0], placement.translation[0] + sign * scale * plane, 1.0e-8);
                REQUIRE_NEAR(hit.object_position[1], sign * scale * (y + (contact == 1u ? 1.0 : 0.0)), 1.0e-8);
                REQUIRE_NEAR(hit.object_position[2], scale * localZ, 1.0e-8);
                REQUIRE_NEAR(hit.normal[0], sign * (contact == 1u ? -0.6 : -1.0), 1.0e-6);
                REQUIRE_NEAR(hit.normal[1], contact == 1u ? sign * 0.8 : 0.0, 1.0e-6);
                REQUIRE(hit.feature.data.triangle.source_face_index == face);
            }
        }
        fixture.destroy();
    }
}

// Every triangle the exact sphere predicate accepts must reach it, whatever a
// backend's own sweep reports. The first mesh is one millimetre triangle of the
// warehouse rack export (sweep 840 of the 1,024-row lattice), rotated so the
// sweep runs along +X and wound to face it, since a single-sided triangle the
// sphere moves along is ignored: the sphere grazes its corner C 0.499 from the
// axis, and the packaged PhysX sphere sweep misses it from either side. In the second a
// one-sided wall seen from behind lies on the axis before the only contact, and
// filler triangles off the path make the mesh large enough to be gathered in
// parts, so the gather must continue past the wall. In the third a 1e-7 sphere
// passes 2e-7 from the corner of a micrometre triangle, a contact the predicate
// accepts only through its tolerance.
void testSphereSweepCandidatesCoverEveryAcceptedContact()
{
    ovspatialquery_instance_handle_t instance     = createInstance();
    const float                      direction[3] = {1.0f, 0.0f, 0.0f};
    const double                     origin[3]    = {0.0, 0.0, 0.0};
    const std::vector<float>         grazing      = {3.42352772f, 0.398550808f, 0.305834889f, 3.42879057f, 0.392723948f,
                                                     0.30853501f, 3.42813015f,  0.392111272f, 0.30865261f};
    std::vector<float>               blocked      = {2.0f, -3.0f, -3.0f, 2.0f, 3.0f, -3.0f, 2.0f, 0.0f, 3.0f,
                                                     4.0f, -1.0f, -1.0f, 4.0f, 0.0f, 1.0f,  4.0f, 1.0f, -1.0f};
    std::vector<uint32_t>            blockedTriangles = {0u, 1u, 2u, 3u, 4u, 5u};
    for (uint32_t filler = 0u; filler < 8192u; ++filler)
    {
        const float x = 0.01f * static_cast<float>(filler);
        blocked.insert(blocked.end(), {x, 50.0f, 0.0f, x, 50.0f, 0.01f, x, 50.01f, 0.0f});
        blockedTriangles.insert(blockedTriangles.end(), {6u + 3u * filler, 7u + 3u * filler, 8u + 3u * filler});
    }
    const std::vector<float> corner    = {1.0e-6f, 0.0f, 0.0f, 1.0e-6f, 0.0f, 1.0e-6f, 1.0e-6f, 1.0e-6f, 0.0f};
    const uint64_t           values[3] = {
        registerTriangleMesh(instance, 5701u, grazing, {1u, 0u, 2u}, false),
        registerTriangleMesh(instance, 5702u, blocked, blockedTriangles, false),
        registerTriangleMesh(instance, 5703u, corner, {0u, 1u, 2u}, false),
    };
    for (uint32_t mesh = 0u; mesh < 3u; ++mesh)
    {
        TriangleInstance placement;
        placement.objectId                                  = 1u;
        placement.geometryValue                             = values[mesh];
        const ovspatialquery_population_handle_t population = publishTrianglePopulation(instance, 1, {placement});
        double                                   center[3]  = {origin[0], origin[1], origin[2]};
        double                                   radius     = 0.5;
        double                                   range      = 15.0;
        if (mesh == 2u)
        {
            // A short sweep near the origin, where float rounding does not widen the gather.
            center[1] = -2.0e-7;
            radius    = 1.0e-7;
            range     = 2.0e-6;
        }
        const auto result = sweepClosest(instance, populationTarget(population), center, radius, direction, range);
        REQUIRE(result.has_hit == 1u);
        const ovspatialquery_sphere_sweep_exact_hit_t& hit = result.hit.data.exact;
        if (mesh == 0u)
        {
            const double lateral = double(grazing[7]) * grazing[7] + double(grazing[8]) * grazing[8];
            REQUIRE_NEAR(hit.distance, double(grazing[6]) - std::sqrt(0.25 - lateral), 1.0e-9);
            REQUIRE_NEAR(hit.object_position[0], grazing[6], 1.0e-9);
            REQUIRE_NEAR(hit.object_position[1], grazing[7], 1.0e-9);
            REQUIRE_NEAR(hit.object_position[2], grazing[8], 1.0e-9);
            REQUIRE(hit.feature.data.triangle.source_face_index == 0u);
        }
        else if (mesh == 1u)
        {
            REQUIRE_NEAR(hit.distance, 3.5, 1.0e-12);
            REQUIRE_NEAR(hit.normal[0], -1.0, 1.0e-6);
            REQUIRE(hit.feature.data.triangle.source_face_index == 1u);
        }
        else
        {
            // The face's plane contact, its foot 2e-7 beside the corner.
            REQUIRE_NEAR(hit.distance, double(corner[0]) - 1.0e-7, 1.0e-18);
            REQUIRE_NEAR(hit.object_position[1], 0.0, 1.0e-18);
            REQUIRE(hit.feature.data.triangle.source_face_index == 0u);
        }
        REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    }
    destroyMeshScene(instance, {}, {values[0], values[1], values[2]});
}

// A view filters raycasts, sweeps and overlaps. Object 1 sits at the origin and
// object 2 twenty units down +X; the view selects only object 2. Each query runs
// against the population, proving the excluded object is reachable, and against
// the view, proving it is excluded even where it would be the nearest or only
// candidate.
void testRaycastSweepAndOverlapRespectViewMembership()
{
    ovspatialquery_instance_handle_t instance = createInstance();
    const uint64_t                   geometryValue =
        registerTriangleMesh(instance, 5501u, singleTriangleVertices(), singleTriangleIndices(), false);
    TriangleInstance nearInstance;
    nearInstance.objectId      = 1u;
    nearInstance.geometryValue = geometryValue;
    TriangleInstance farInstance;
    farInstance.objectId       = 2u;
    farInstance.geometryValue  = geometryValue;
    farInstance.translation[0] = 20.0;
    const ovspatialquery_population_handle_t population =
        publishTrianglePopulation(instance, 1, {nearInstance, farInstance});

    uint64_t                                  memberId = 2u;
    InputArray                                memberArray(&memberId, 1, kDLUInt);
    const ovspatialquery_explicit_view_desc_t viewDesc{&memberArray.value};
    ovspatialquery_view_handle_t              view = OVSPATIALQUERY_INVALID_VIEW;
    REQUIRE(ovspatialquery_create_explicit_view(instance, population, &viewDesc, &view).status ==
            OVSPATIALQUERY_API_SUCCESS);

    // Raycast: a ray along the shared (Y,Z)=(0,0) line reaches object 1's
    // plane (X=1) at distance 1 and object 2's plane (X=21) at distance 21.
    const double                                 rayOrigin[3]    = {0.0, 0.0, 0.0};
    const float                                  rayDirection[3] = {1.0f, 0.0f, 0.0f};
    const ovspatialquery_raycast_single_result_t populationRay =
        raycastClosest(instance, populationTarget(population), rayOrigin, rayDirection, 25.0);
    REQUIRE(populationRay.has_hit == 1u);
    REQUIRE(populationRay.hit.data.exact.identity.object_id.value == 1u);
    const ovspatialquery_raycast_single_result_t viewRay =
        raycastClosest(instance, viewTarget(view), rayOrigin, rayDirection, 25.0);
    REQUIRE(viewRay.has_hit == 1u);
    REQUIRE(viewRay.hit.data.exact.identity.object_id.value == 2u);
    REQUIRE_NEAR(viewRay.hit.data.exact.distance, 21.0, 1.0e-9);

    // Sphere sweep: center (0.9,0,0) already overlaps object 1's surface,
    // so it is the sharpest case: the view must reject it even though it is
    // an immediate, distance-0 contact.
    const double                                       sweepCenter[3] = {0.9, 0.0, 0.0};
    const ovspatialquery_sphere_sweep_closest_result_t populationSweep =
        sweepClosest(instance, populationTarget(population), sweepCenter, 0.25, rayDirection, 30.0);
    REQUIRE(populationSweep.has_hit == 1u);
    REQUIRE(populationSweep.hit.data.exact.identity.object_id.value == 1u);
    const ovspatialquery_sphere_sweep_closest_result_t viewSweep =
        sweepClosest(instance, viewTarget(view), sweepCenter, 0.25, rayDirection, 30.0);
    REQUIRE(viewSweep.has_hit == 1u);
    REQUIRE(viewSweep.hit.data.exact.identity.object_id.value == 2u);

    // Overlap: a small sphere at (1,0,0) touches only object 1's surface.
    const double                              overlapAtNear[3] = {1.0, 0.0, 0.0};
    const ovspatialquery_overlap_any_result_t populationOverlap =
        overlapSphereAny(instance, populationTarget(population), overlapAtNear, 0.1);
    REQUIRE(populationOverlap.has_hit == 1u);
    REQUIRE(populationOverlap.hit.identity.object_id.value == 1u);
    const ovspatialquery_overlap_any_result_t viewOverlapAtNear =
        overlapSphereAny(instance, viewTarget(view), overlapAtNear, 0.1);
    REQUIRE(viewOverlapAtNear.has_hit == 0u);
    const double                              overlapAtFar[3] = {21.0, 0.0, 0.0};
    const ovspatialquery_overlap_any_result_t viewOverlapAtFar =
        overlapSphereAny(instance, viewTarget(view), overlapAtFar, 0.1);
    REQUIRE(viewOverlapAtFar.has_hit == 1u);
    REQUIRE(viewOverlapAtFar.hit.identity.object_id.value == 2u);

    REQUIRE(ovspatialquery_destroy_view(instance, view).status == OVSPATIALQUERY_API_SUCCESS);
    destroyMeshScene(instance, {population}, {geometryValue});
}

// Under EXACT precision an AABB or oriented-box shape that touches the wedge's
// face reports a hit, and one that meets the wedge's world AABB only near the
// (0.9, 0.9) corner, outside both triangles, reports none.
void testOverlapAabbAndBoxRejectBroadPhaseFalsePositivesAgainstTriangleMesh()
{
    ovspatialquery_instance_handle_t instance = createInstance();
    const uint64_t   geometryValue = registerTriangleMesh(instance, 5601u, wedgeVertices(), wedgeIndices(), false);
    TriangleInstance placement;
    placement.objectId                                  = 1u;
    placement.geometryValue                             = geometryValue;
    const ovspatialquery_population_handle_t population = publishTrianglePopulation(instance, 1, {placement});

    ovspatialquery_overlap_desc_t aabbNear{};
    aabbNear.common.target                 = populationTarget(population);
    aabbNear.common.precision              = OVSPATIALQUERY_QUERY_EXACT;
    aabbNear.shape.kind                    = OVSPATIALQUERY_OVERLAP_AABB;
    aabbNear.shape.data.aabb.min_corner[0] = 0.9;
    aabbNear.shape.data.aabb.min_corner[1] = -0.1;
    aabbNear.shape.data.aabb.min_corner[2] = -0.1;
    aabbNear.shape.data.aabb.max_corner[0] = 1.1;
    aabbNear.shape.data.aabb.max_corner[1] = 0.1;
    aabbNear.shape.data.aabb.max_corner[2] = 0.1;
    ovspatialquery_overlap_any_result_t aabbNearResult{};
    REQUIRE(ovspatialquery_overlap_any_sync(instance, &aabbNear, &aabbNearResult).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(aabbNearResult.has_hit == 1u);

    ovspatialquery_overlap_desc_t aabbCorner = aabbNear;
    aabbCorner.shape.data.aabb.min_corner[1] = 0.85;
    aabbCorner.shape.data.aabb.min_corner[2] = 0.85;
    aabbCorner.shape.data.aabb.max_corner[1] = 0.95;
    aabbCorner.shape.data.aabb.max_corner[2] = 0.95;
    ovspatialquery_overlap_any_result_t aabbCornerResult{};
    REQUIRE(ovspatialquery_overlap_any_sync(instance, &aabbCorner, &aabbCornerResult).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(aabbCornerResult.has_hit == 0u);

    ovspatialquery_overlap_desc_t boxNear{};
    boxNear.common.target                  = populationTarget(population);
    boxNear.common.precision               = OVSPATIALQUERY_QUERY_EXACT;
    boxNear.shape.kind                     = OVSPATIALQUERY_OVERLAP_BOX;
    boxNear.shape.data.box.center[0]       = 1.0;
    boxNear.shape.data.box.half_extents[0] = 0.1;
    boxNear.shape.data.box.half_extents[1] = 0.2;
    boxNear.shape.data.box.half_extents[2] = 0.2;
    // A 22.5-degree rotation about Z (half-angle form), confirming the
    // exact box overlap honors rotation and not just axis-aligned extent.
    boxNear.shape.data.box.rotation_xyzw[2] = std::sqrt(0.5f - 0.5f * std::sqrt(0.5f));
    boxNear.shape.data.box.rotation_xyzw[3] = std::sqrt(0.5f + 0.5f * std::sqrt(0.5f));
    ovspatialquery_overlap_any_result_t boxNearResult{};
    REQUIRE(ovspatialquery_overlap_any_sync(instance, &boxNear, &boxNearResult).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(boxNearResult.has_hit == 1u);

    ovspatialquery_overlap_desc_t boxCorner  = boxNear;
    boxCorner.shape.data.box.center[1]       = 0.9;
    boxCorner.shape.data.box.center[2]       = 0.9;
    boxCorner.shape.data.box.half_extents[1] = 0.05;
    boxCorner.shape.data.box.half_extents[2] = 0.05;
    ovspatialquery_overlap_any_result_t boxCornerResult{};
    REQUIRE(ovspatialquery_overlap_any_sync(instance, &boxCorner, &boxCornerResult).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(boxCornerResult.has_hit == 0u);

    destroyMeshScene(instance, {population}, {geometryValue});
}

// A mesh wholly inside an overlap shape is reported without a triangle test; one whose bounds
// sit beside the shape, or only inside its world AABB, is not. The wedge's world bounds are
// [2.5,3.5]x[-4,-2]x[-1,1], centred on the line y = -x, and its nearest point to the origin is
// (2.5,-2,-1), 3.35 away. Object 2, one triangle (10,11,10), (11,10,10), (11,11,11), has bounds
// [10,11]^3 and every point of it has x + y >= 21.
void testOverlapReportsMeshWhollyInsideShapeAndNotOneBesideIt()
{
    ovspatialquery_instance_handle_t instance = createInstance();
    const uint64_t geometryValue = registerTriangleMesh(instance, 5602u, wedgeVertices(), wedgeIndices(), false);
    const uint64_t triangleValue = registerTriangleMesh(instance, 5603u, {0, 1, 0, 1, 0, 0, 1, 1, 1}, {0, 1, 2}, false);
    TriangleInstance placement;
    placement.objectId       = 1u;
    placement.geometryValue  = geometryValue;
    placement.translation[0] = 1.5;
    placement.translation[1] = -3.0;
    TriangleInstance triangle;
    triangle.objectId       = 2u;
    triangle.geometryValue  = triangleValue;
    triangle.translation[0] = triangle.translation[1] = triangle.translation[2] = 10.0;
    const ovspatialquery_population_handle_t population = publishTrianglePopulation(instance, 1, {placement, triangle});
    const auto                               overlaps   = [&](const ovspatialquery_overlap_shape_t& shape)
    {
        ovspatialquery_overlap_desc_t desc{};
        desc.common.target    = populationTarget(population);
        desc.common.precision = OVSPATIALQUERY_QUERY_EXACT;
        desc.shape            = shape;
        ovspatialquery_overlap_any_result_t result{};
        REQUIRE(ovspatialquery_overlap_any_sync(instance, &desc, &result).status == OVSPATIALQUERY_API_SUCCESS);
        return result.has_hit == 1u;
    };

    // A long box about the origin turned 45 degrees about Z lies along y = x and misses the
    // wedge; turned -45 degrees it lies along y = -x and holds it.
    ovspatialquery_overlap_shape_t box{};
    box.kind                      = OVSPATIALQUERY_OVERLAP_BOX;
    box.data.box.half_extents[0]  = 6.0;
    box.data.box.half_extents[1]  = 1.5;
    box.data.box.half_extents[2]  = 1.5;
    box.data.box.rotation_xyzw[2] = std::sin(0.125f * 3.14159265f);
    box.data.box.rotation_xyzw[3] = std::cos(0.125f * 3.14159265f);
    REQUIRE(!overlaps(box));
    box.data.box.rotation_xyzw[2] = -box.data.box.rotation_xyzw[2];
    REQUIRE(overlaps(box));

    // A sphere about the origin: radius 2.8 stops short of the wedge though its farthest
    // bounds corner (5.41 away) lies within twice that; radius 6 holds it.
    ovspatialquery_overlap_shape_t sphere{};
    sphere.kind               = OVSPATIALQUERY_OVERLAP_SPHERE;
    sphere.data.sphere.radius = 2.8;
    REQUIRE(!overlaps(sphere));
    sphere.data.sphere.radius = 6.0;
    REQUIRE(overlaps(sphere));

    // AABBs that cut into a mesh's bounds holding only one of its corners touch no triangle:
    // [2.4,3.6]x[-2.2,0]x[0.6,2] holds the wedge's upper corner, where both its triangles lie
    // below z = -0.6 at y = -2.2, and [9,10.1]^3 holds the triangle's lower corner. Grown to
    // [-1,4]x[-5,0]x[-2,2] the first holds the whole wedge.
    const auto aabbOf = [](double x0, double y0, double z0, double x1, double y1, double z1)
    {
        ovspatialquery_overlap_shape_t aabb{};
        aabb.kind           = OVSPATIALQUERY_OVERLAP_AABB;
        const double low[]  = {x0, y0, z0};
        const double high[] = {x1, y1, z1};
        std::copy_n(low, 3, aabb.data.aabb.min_corner);
        std::copy_n(high, 3, aabb.data.aabb.max_corner);
        return aabb;
    };
    REQUIRE(!overlaps(aabbOf(2.4, -2.2, 0.6, 3.6, 0.0, 2.0)));
    REQUIRE(!overlaps(aabbOf(9.0, 9.0, 9.0, 10.1, 10.1, 10.1)));
    REQUIRE(overlaps(aabbOf(-1.0, -5.0, -2.0, 4.0, 0.0, 2.0)));

    destroyMeshScene(instance, {population}, {geometryValue, triangleValue});
}

// overlap_multiple_sync over two triangle instances inside one sphere: a short
// buffer truncates, the full one holds both objects in either order.
void testOverlapMultipleReportsHitsWithTruncationContract()
{
    ovspatialquery_instance_handle_t instance = createInstance();
    const uint64_t                   geometryValue =
        registerTriangleMesh(instance, 5701u, singleTriangleVertices(), singleTriangleIndices(), false);
    TriangleInstance first;
    first.objectId      = 1u;
    first.geometryValue = geometryValue;
    TriangleInstance second;
    second.objectId                                     = 2u;
    second.geometryValue                                = geometryValue;
    second.translation[0]                               = 0.5;
    const ovspatialquery_population_handle_t population = publishTrianglePopulation(instance, 1, {first, second});

    ovspatialquery_overlap_desc_t desc{};
    desc.common.target               = populationTarget(population);
    desc.common.precision            = OVSPATIALQUERY_QUERY_EXACT;
    desc.shape.kind                  = OVSPATIALQUERY_OVERLAP_SPHERE;
    desc.shape.data.sphere.center[0] = 1.25;
    desc.shape.data.sphere.radius    = 0.3;

    ovspatialquery_overlap_hit_t        shortHits[1]{};
    ovspatialquery_overlap_hit_buffer_t shortBuffer{shortHits, 1u};
    ovspatialquery_multiple_result_t    shortResult{};
    REQUIRE(ovspatialquery_overlap_multiple_sync(instance, &desc, &shortBuffer, &shortResult).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(shortResult.stored_hit_count == 1u);
    REQUIRE(shortResult.truncated == 1u);

    ovspatialquery_overlap_hit_t        fullHits[2]{};
    ovspatialquery_overlap_hit_buffer_t fullBuffer{fullHits, 2u};
    ovspatialquery_multiple_result_t    fullResult{};
    REQUIRE(ovspatialquery_overlap_multiple_sync(instance, &desc, &fullBuffer, &fullResult).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(fullResult.stored_hit_count == 2u);
    REQUIRE(fullResult.truncated == 0u);
    const uint64_t firstId = fullHits[0].identity.object_id.value, secondId = fullHits[1].identity.object_id.value;
    REQUIRE(std::min(firstId, secondId) == 1u && std::max(firstId, secondId) == 2u);

    destroyMeshScene(instance, {population}, {geometryValue});
}

// A non-finite vertex is rejected atomically: host usage is unchanged and the
// definition value never becomes live.
void testTriangleMeshRegistrationRejectsNonfiniteVertexAtomically()
{
    ovspatialquery_instance_handle_t instance             = createInstance();
    const uint64_t                   usageBeforeRejection = hostUsage(instance);

    const uint64_t     definitionValue = 5801u;
    std::vector<float> vertices        = singleTriangleVertices();
    vertices[0]                        = std::numeric_limits<float>::infinity();
    requireRejected(enqueueTriangleMesh(instance, definitionValue, vertices, singleTriangleIndices(), false),
                    OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(hostUsage(instance) == usageBeforeRejection);
    REQUIRE(
        ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, definitionValue})
            .status == OVSPATIALQUERY_API_INVALID_HANDLE);

    // The same definition value registers successfully once the vertex is
    // finite, confirming the earlier rejection left no partial state behind.
    const uint64_t registeredValue =
        registerTriangleMesh(instance, definitionValue, singleTriangleVertices(), singleTriangleIndices(), false);
    REQUIRE(hostUsage(instance) > usageBeforeRejection);
    destroyMeshScene(instance, {}, {registeredValue});
}

} // namespace

int main()
{
    testTriangleMeshSidednessRejectsBackFaceUnlessDefinitionDoubleSided();
    testTrianglePoseRotationAndUniformScaleTransformExactHit();
    testTriangleRaycastExactMissDoesNotFallBackToBoundsCandidate();
    testBoundsCandidateNormalOfFlatAabbFacesTheRay();
    testTriangleRaycastMultipleReportsAllHitsWithTruncationContract();
    testTriangleRaycastMultipleKeepsNearestOfManyCrossings();
    testTriangleRaycastMultipleKeepsRoundedEndpointCrossing();
    testSphereSweepClosestTiesEndpointsAndInitialOverlapsAcrossInstances();
    testMeshSweepsAcrossDistributedFaces();
    testSphereSweepCandidatesCoverEveryAcceptedContact();
    testRaycastSweepAndOverlapRespectViewMembership();
    testOverlapAabbAndBoxRejectBroadPhaseFalsePositivesAgainstTriangleMesh();
    testOverlapReportsMeshWhollyInsideShapeAndNotOneBesideIt();
    testOverlapMultipleReportsHitsWithTruncationContract();
    testTriangleMeshRegistrationRejectsNonfiniteVertexAtomically();
    std::puts("precise geometry test passed");
    return 0;
}
