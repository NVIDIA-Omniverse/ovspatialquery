/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../support/native/far_sphere_sweep_scene.h"
#include "../../support/native/public_api_inputs.h"
#include "../../support/native/test_failure.h"
#include <ovspatialquery/ovspatialquery_explicit.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <utility>
#include <vector>

// Differential check of ovspatialquery_raycast_closest_sync and
// ovspatialquery_sweep_sphere_closest_sync against independent double-precision
// oracles computed in this file, exercised through the real PhysX backend.
// Raycast uses OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE against axis-aligned boxes,
// whose published world AABB is exactly the box itself, so a slab ray/AABB
// test is the correct oracle. Sweep uses OVSPATIALQUERY_QUERY_EXACT against
// stored spheres, so a ray/enlarged-sphere test (Minkowski sum with the moving
// sphere radius) is the correct oracle.

namespace
{

struct Instance
{
    uint64_t id;
    double   center[3];
};

constexpr uint64_t kBoxDefinition    = 1u;
constexpr uint64_t kSphereDefinition = 2u;
constexpr double   kBoxHalfExtent    = 1.0;
constexpr double   kSphereRadius     = 1.0;

void registerBoxShape(ovspatialquery_instance_handle_t instance)
{
    uint64_t                                    value          = kBoxDefinition;
    double                                      halfExtents[3] = {kBoxHalfExtent, kBoxHalfExtent, kBoxHalfExtent};
    InputArray                                  values(&value, 1, kDLUInt);
    InputArray                                  extents(halfExtents, 1, 3, kDLFloat);
    const ovspatialquery_box_definition_table_t boxes{&values.value, &extents.value};
    waitAndDestroy(instance, ovspatialquery_enqueue_box_registration(instance, &boxes));
}

void registerSphereShape(ovspatialquery_instance_handle_t instance)
{
    uint64_t                                       value  = kSphereDefinition;
    double                                         radius = kSphereRadius;
    InputArray                                     values(&value, 1, kDLUInt);
    InputArray                                     radii(&radius, 1, kDLFloat);
    const ovspatialquery_sphere_definition_table_t spheres{&values.value, &radii.value};
    waitAndDestroy(instance, ovspatialquery_enqueue_sphere_registration(instance, &spheres));
}

void registerTriangleMeshShape(ovspatialquery_instance_handle_t instance, uint64_t definitionValue,
                               std::vector<float> vertices, std::vector<uint32_t> triangles)
{
    uint64_t   vertexOffset   = 0u;
    uint64_t   vertexCount    = vertices.size() / 3u;
    uint64_t   triangleOffset = 0u;
    uint64_t   triangleCount  = triangles.size() / 3u;
    InputArray values(&definitionValue, 1, kDLUInt);
    InputArray vertexOffsets(&vertexOffset, 1, kDLUInt);
    InputArray vertexCounts(&vertexCount, 1, kDLUInt);
    InputArray triangleOffsets(&triangleOffset, 1, kDLUInt);
    InputArray triangleCounts(&triangleCount, 1, kDLUInt);
    InputArray vertexRows(vertices.data(), static_cast<int64_t>(vertexCount), 3, kDLFloat);
    InputArray triangleRows(triangles.data(), static_cast<int64_t>(triangleCount), 3, kDLUInt);
    const ovspatialquery_triangle_mesh_definition_table_t meshes{&values.value,
                                                                 &vertexOffsets.value,
                                                                 &vertexCounts.value,
                                                                 &triangleOffsets.value,
                                                                 &triangleCounts.value,
                                                                 &vertexRows.value,
                                                                 &triangleRows.value,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr};
    waitAndDestroy(instance, ovspatialquery_enqueue_triangle_mesh_registration(instance, &meshes));
}

ovspatialquery_population_handle_t buildTrianglePopulation(ovspatialquery_instance_handle_t instance,
                                                           ovspatialquery_ordinal_t ordinal, uint64_t definitionValue,
                                                           uint64_t objectId, const double translation[3],
                                                           const double rotationXyzw[4], double uniformScale)
{
    const ovspatialquery_population_handle_t       population   = createExplicitPopulation(instance, 0u);
    int32_t                                        geometryType = OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH;
    InputArray                                     objectValues(&objectId, 1, kDLUInt);
    InputArray                                     geometryTypes(&geometryType, 1, kDLInt);
    InputArray                                     geometryValues(&definitionValue, 1, kDLUInt);
    InputArray                                     translations(translation, 1, 3, kDLFloat);
    InputArray                                     rotations(rotationXyzw, 1, 4, kDLFloat);
    InputArray                                     scales(&uniformScale, 1, kDLFloat);
    const ovspatialquery_explicit_population_set_t set{ordinal,
                                                       {&objectValues.value, &geometryTypes.value,
                                                        &geometryValues.value, &translations.value, &rotations.value,
                                                        &scales.value, nullptr, nullptr},
                                                       nullptr,
                                                       0u};
    waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set));
    return population;
}

// Publishes `instances` of one stored definition, each with an identity rotation.
ovspatialquery_population_handle_t buildPopulation(ovspatialquery_instance_handle_t instance,
                                                   ovspatialquery_ordinal_t ordinal, int32_t type, uint64_t definition,
                                                   const std::vector<Instance>& instances)
{
    InstanceRows rows;
    for (const Instance& placement : instances)
    {
        rows.add(placement.id, type, definition, placement.center[0], placement.center[1], placement.center[2]);
    }
    const ovspatialquery_population_handle_t       population = createExplicitPopulation(instance, 0u);
    const ovspatialquery_explicit_population_set_t set{ordinal, rows.table(), nullptr, 0u};
    waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set));
    return population;
}

ovspatialquery_raycast_desc_t makeRay(ovspatialquery_population_handle_t population, const double origin[3],
                                      const float direction[3], double minDistance, double maxDistance)
{
    ovspatialquery_raycast_desc_t query{};
    query.common.target    = populationTarget(population);
    query.common.precision = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    for (int axis = 0; axis < 3; ++axis)
    {
        query.ray.origin[axis] = origin[axis];
    }
    query.ray.unit_direction[0] = direction[0];
    query.ray.unit_direction[1] = direction[1];
    query.ray.unit_direction[2] = direction[2];
    query.ray.min_distance      = minDistance;
    query.ray.max_distance      = maxDistance;
    return query;
}

ovspatialquery_sphere_sweep_desc_t makeSweep(ovspatialquery_population_handle_t population, const double center[3],
                                             double radius, const float direction[3], double minDistance,
                                             double maxDistance)
{
    ovspatialquery_sphere_sweep_desc_t query{};
    query.common.target    = populationTarget(population);
    query.common.precision = OVSPATIALQUERY_QUERY_EXACT;
    for (int axis = 0; axis < 3; ++axis)
    {
        query.sweep.center[axis] = center[axis];
    }
    query.sweep.radius            = radius;
    query.sweep.unit_direction[0] = direction[0];
    query.sweep.unit_direction[1] = direction[1];
    query.sweep.unit_direction[2] = direction[2];
    query.sweep.min_distance      = minDistance;
    query.sweep.max_distance      = maxDistance;
    return query;
}

// Double-precision oracle: closest entry of a ray into an axis-aligned box,
// clamped to [tMin, tMax]. Matches the BOUNDS_CANDIDATE contract, since a
// tolerance-zero ray against an identity-rotation box's own extents is
// exactly its published world AABB.
bool oracleRayBox(const double origin[3], const float direction[3], double tMin, double tMax, const double center[3],
                  double halfExtent, double& outDistance)
{
    double lo = tMin;
    double hi = tMax;
    for (int axis = 0; axis < 3; ++axis)
    {
        const double minB = center[axis] - halfExtent;
        const double maxB = center[axis] + halfExtent;
        const double dir  = static_cast<double>(direction[axis]);
        if (dir != 0.0)
        {
            double tNear = (minB - origin[axis]) / dir;
            double tFar  = (maxB - origin[axis]) / dir;
            if (tNear > tFar)
            {
                std::swap(tNear, tFar);
            }
            lo = std::max(lo, tNear);
            hi = std::min(hi, tFar);
            if (lo > hi)
            {
                return false;
            }
        }
        else if (origin[axis] < minB || origin[axis] > maxB)
        {
            return false;
        }
    }
    outDistance = lo;
    return true;
}

// Double-precision oracle: closest entry of a ray against a sphere enlarged
// by the moving sphere's radius (Minkowski sum), clamped to [tMin, tMax].
// Matches EXACT sphere-vs-sphere sweep contact with zero contact tolerance.
bool oracleRaySphere(const double origin[3], const float direction[3], double tMin, double tMax, const double center[3],
                     double radius, double& outDistance)
{
    double relative[3];
    double directionSquared = 0.0;
    double dotProduct       = 0.0;
    double distanceSquared  = 0.0;
    for (int axis = 0; axis < 3; ++axis)
    {
        relative[axis]   = origin[axis] - center[axis];
        const double dir = static_cast<double>(direction[axis]);
        directionSquared += dir * dir;
        dotProduct += dir * relative[axis];
        distanceSquared += relative[axis] * relative[axis];
    }
    const double c            = distanceSquared - radius * radius;
    const double b            = 2.0 * dotProduct;
    const double discriminant = b * b - 4.0 * directionSquared * c;
    if (discriminant < 0.0)
    {
        return false;
    }
    const double squareRoot = std::sqrt(discriminant);
    double       t0         = (-b - squareRoot) / (2.0 * directionSquared);
    double       t1         = (-b + squareRoot) / (2.0 * directionSquared);
    if (t0 > t1)
    {
        std::swap(t0, t1);
    }
    const double lo = std::max(t0, tMin);
    const double hi = std::min(t1, tMax);
    if (lo > hi)
    {
        return false;
    }
    outDistance = lo;
    return true;
}

// A query box as the library reads it: half extents and the world directions
// of its float quaternion's local axes, normalised in double, with the 15
// candidate separating axes of it and an axis-aligned box.
struct OrientedBox
{
    double                             half[3];
    double                             axis[3][3];
    std::vector<std::array<double, 3>> separating;
};

OrientedBox orientedBox(const float rotationXyzw[4], const double half[3])
{
    double q[4];
    double length = 0.0;
    for (int component = 0; component < 4; ++component)
    {
        q[component] = double(rotationXyzw[component]);
        length += q[component] * q[component];
    }
    length = std::sqrt(length);
    for (double& component : q)
    {
        component /= length;
    }
    const double x = q[0], y = q[1], z = q[2], w = q[3];
    OrientedBox  box{{half[0], half[1], half[2]},
                     {{1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w), 2.0 * (x * z - y * w)},
                      {2.0 * (x * y - z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + x * w)},
                      {2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y)}},
                     {}};
    for (int world = 0; world < 3; ++world)
    {
        std::array<double, 3> unit{};
        unit[size_t(world)] = 1.0;
        box.separating.push_back(unit);
        box.separating.push_back({box.axis[world][0], box.axis[world][1], box.axis[world][2]});
        for (const double* e : box.axis)
        {
            const std::array<double, 3> cross{unit[1] * e[2] - unit[2] * e[1], unit[2] * e[0] - unit[0] * e[2],
                                              unit[0] * e[1] - unit[1] * e[0]};
            const double norm = std::sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);
            if (norm > 1.0e-9)
            {
                box.separating.push_back({cross[0] / norm, cross[1] / norm, cross[2] / norm});
            }
        }
    }
    return box;
}

// Double-precision oracle: the first distance in [tMin, tMax] at which `box`,
// moving from `start` along `direction`, touches the axis-aligned cube at
// `center`. Two boxes overlap exactly when none of their separating axes parts
// them, and along each axis the moving projections overlap over one interval
// of distances, so the contact is the start of those intervals' intersection.
bool oracleBoxSweepCube(const double start[3], const float direction[3], double tMin, double tMax,
                        const OrientedBox& box, const double center[3], double halfExtent, double& outDistance)
{
    double lo = tMin;
    double hi = tMax;
    for (const std::array<double, 3>& l : box.separating)
    {
        double radius = halfExtent * (std::abs(l[0]) + std::abs(l[1]) + std::abs(l[2]));
        double offset = 0.0;
        double speed  = 0.0;
        for (int axis = 0; axis < 3; ++axis)
        {
            const double* e = box.axis[axis];
            radius += box.half[axis] * std::abs(e[0] * l[0] + e[1] * l[1] + e[2] * l[2]);
            offset += (start[axis] - center[axis]) * l[size_t(axis)];
            speed += double(direction[axis]) * l[size_t(axis)];
        }
        if (speed == 0.0)
        {
            if (std::abs(offset) > radius)
            {
                return false;
            }
            continue;
        }
        double t0 = (-radius - offset) / speed;
        double t1 = (radius - offset) / speed;
        if (t0 > t1)
        {
            std::swap(t0, t1);
        }
        lo = std::max(lo, t0);
        hi = std::min(hi, t1);
        if (lo > hi)
        {
            return false;
        }
    }
    outDistance = lo;
    return true;
}

bool nearlyEqual(double left, double right) { return std::abs(left - right) < 1.0e-6; }

// The oracle's nearest valid entry over every instance and the objects tied with it,
// since the header makes no closest-hit tie-break promise: only that a tied candidate
// with the tied distance is returned.
struct OracleExpectation
{
    bool                  hasHit   = false;
    double                distance = 0.0;
    std::vector<uint64_t> tiedIds;

    bool ties(uint64_t id) const { return std::find(tiedIds.begin(), tiedIds.end(), id) != tiedIds.end(); }
};

template <typename Entry>
OracleExpectation expectedHit(const std::vector<Instance>& instances, Entry entry)
{
    OracleExpectation expectation;
    for (const Instance& instance : instances)
    {
        double distance = 0.0;
        if (!entry(instance.center, distance))
        {
            continue;
        }
        if (!expectation.hasHit || distance < expectation.distance - 1.0e-9)
        {
            expectation = {true, distance, {instance.id}};
        }
        else if (nearlyEqual(distance, expectation.distance))
        {
            expectation.tiedIds.push_back(instance.id);
        }
    }
    return expectation;
}

OracleExpectation expectedRayHit(const double origin[3], const float direction[3], double tMin, double tMax,
                                 const std::vector<Instance>& boxes)
{
    return expectedHit(boxes, [&](const double center[3], double& distance)
                       { return oracleRayBox(origin, direction, tMin, tMax, center, kBoxHalfExtent, distance); });
}

OracleExpectation expectedSweepHit(const double start[3], double sweepRadius, const float direction[3], double tMin,
                                   double tMax, const std::vector<Instance>& spheres)
{
    return expectedHit(
        spheres, [&](const double center[3], double& distance)
        { return oracleRaySphere(start, direction, tMin, tMax, center, kSphereRadius + sweepRadius, distance); });
}

void expectRay(ovspatialquery_instance_handle_t instance, ovspatialquery_ordinal_t expectedOrdinal,
               const ovspatialquery_raycast_desc_t& query, const OracleExpectation& expectation)
{
    ovspatialquery_raycast_single_result_t result{};
    const ovspatialquery_result_t          status = ovspatialquery_raycast_closest_sync(instance, &query, &result);
    REQUIRE(status.status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(result.source_ordinal == expectedOrdinal);
    REQUIRE(result.has_hit == (expectation.hasHit ? 1u : 0u));
    if (expectation.hasHit)
    {
        REQUIRE(result.hit.truth == OVSPATIALQUERY_HIT_BOUNDS_CANDIDATE);
        REQUIRE(expectation.ties(result.hit.data.bounds_candidate.identity.object_id.value));
        REQUIRE(nearlyEqual(result.hit.data.bounds_candidate.distance, expectation.distance));
    }
}

void expectSweep(ovspatialquery_instance_handle_t instance, ovspatialquery_ordinal_t expectedOrdinal,
                 const ovspatialquery_sphere_sweep_desc_t& query, const OracleExpectation& expectation)
{
    ovspatialquery_sphere_sweep_closest_result_t result{};
    const ovspatialquery_result_t status = ovspatialquery_sweep_sphere_closest_sync(instance, &query, &result);
    REQUIRE(status.status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(result.source_ordinal == expectedOrdinal);
    REQUIRE(result.has_hit == (expectation.hasHit ? 1u : 0u));
    if (expectation.hasHit)
    {
        REQUIRE(result.hit.truth == OVSPATIALQUERY_HIT_EXACT_SURFACE);
        REQUIRE(expectation.ties(result.hit.data.exact.identity.object_id.value));
        REQUIRE(nearlyEqual(result.hit.data.exact.distance, expectation.distance));
    }
}

// One rotated box sweep through the cubes of `placements` in CLOSEST, ANY and
// MULTIPLE against the separating-axis oracle. Returns whether CLOSEST hit
// beyond `minimum`.
bool expectRotatedBoxSweep(ovspatialquery_instance_handle_t instance, ovspatialquery_population_handle_t boxes,
                           const std::vector<Instance>& placements, const double start[3], const float direction[3],
                           double minimum, double maximum, const float rotationXyzw[4], const double half[3])
{
    const OrientedBox box   = orientedBox(rotationXyzw, half);
    const auto        reach = [&](const double center[3], double& distance)
    { return oracleBoxSweepCube(start, direction, minimum, maximum, box, center, kBoxHalfExtent, distance); };
    ovspatialquery_box_sweep_desc_t query{};
    query.common.target    = populationTarget(boxes);
    query.common.precision = OVSPATIALQUERY_QUERY_EXACT;
    for (int axis = 0; axis < 3; ++axis)
    {
        query.sweep.center[axis]         = start[axis];
        query.sweep.half_extents[axis]   = half[axis];
        query.sweep.unit_direction[axis] = direction[axis];
    }
    std::copy(rotationXyzw, rotationXyzw + 4, query.sweep.rotation_xyzw);
    query.sweep.min_distance = minimum;
    query.sweep.max_distance = maximum;

    const OracleExpectation                   expected = expectedHit(placements, reach);
    ovspatialquery_box_sweep_closest_result_t closest{};
    REQUIRE(ovspatialquery_sweep_box_closest_sync(instance, &query, &closest).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(closest.has_hit == (expected.hasHit ? 1u : 0u));
    if (expected.hasHit)
    {
        const auto& hit = closest.hit.data.exact;
        REQUIRE(expected.ties(hit.identity.object_id.value) && nearlyEqual(hit.distance, expected.distance));
        // The point lies on the cube and, in the box's own frame, in the moved box.
        const double* center = placements[hit.identity.object_id.value - 1u].center;
        double        onCube = 0.0;
        bool          inBox  = true;
        for (int axis = 0; axis < 3; ++axis)
        {
            onCube       = std::max(onCube, std::abs(hit.object_position[axis] - center[axis]));
            double local = 0.0;
            for (int world = 0; world < 3; ++world)
            {
                local += box.axis[axis][world] *
                         (hit.object_position[world] - start[world] - double(direction[world]) * hit.distance);
            }
            inBox = inBox && std::abs(local) <= half[axis] + 1.0e-6;
        }
        // An initial overlap's point is defined differently.
        if (hit.distance > minimum)
        {
            REQUIRE(nearlyEqual(onCube, kBoxHalfExtent) && inBox);
        }
    }

    // ANY reports one object the box reaches, at that object's first contact.
    ovspatialquery_box_sweep_closest_result_t any{};
    REQUIRE(ovspatialquery_sweep_box_any_sync(instance, &query, &any).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(any.has_hit == (expected.hasHit ? 1u : 0u));
    if (expected.hasHit)
    {
        double distance = 0.0;
        REQUIRE(reach(placements[any.hit.data.exact.identity.object_id.value - 1u].center, distance));
        REQUIRE(nearlyEqual(any.hit.data.exact.distance, distance));
    }

    // MULTIPLE reports exactly the objects the oracle reaches, each at its first contact.
    std::vector<ovspatialquery_box_sweep_hit_t> rows(placements.size());
    ovspatialquery_box_sweep_hit_buffer_t       buffer{rows.data(), rows.size()};
    ovspatialquery_multiple_result_t            multiple{};
    REQUIRE(ovspatialquery_sweep_box_multiple_sync(instance, &query, &buffer, &multiple).status ==
            OVSPATIALQUERY_API_SUCCESS);
    size_t reached = 0u;
    for (const Instance& placement : placements)
    {
        double distance = 0.0;
        if (!reach(placement.center, distance))
        {
            continue;
        }
        ++reached;
        bool found = false;
        for (uint64_t row = 0u; row < multiple.stored_hit_count; ++row)
        {
            const auto& hit = rows[row].data.exact;
            found = found || (hit.identity.object_id.value == placement.id && nearlyEqual(hit.distance, distance));
        }
        REQUIRE(found);
    }
    REQUIRE(!multiple.truncated && multiple.stored_hit_count == reached);
    return expected.hasHit && expected.distance > minimum;
}

// Closest sphere and box sweeps through populations deep enough for a tree of
// several levels, along paths far longer than the distance to their first
// contact: the traversal stops at the running best, and every object it skips
// must lie beyond it. Box sweeps with an identity rotation against axis-aligned
// boxes first meet a box where the ray enters it grown by the query's half
// extents, and the reported point must lie on the stored box and in the moved
// query box. The same paths repeat with a box turned 45 degrees about each axis
// and a long thin box at a random rotation, whose traversal must cover the
// turned box's bounds rather than its local extents.
void testLongSweepsThroughDensePopulations(ovspatialquery_instance_handle_t instance)
{
    uint64_t state = 0x9e3779b97f4a7c15ull;
    auto     next  = [&](double low, double high)
    {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return low + (high - low) * double(state >> 11) * 0x1.0p-53;
    };
    std::vector<Instance> placements;
    for (uint64_t index = 0u; index < 1200u; ++index)
    {
        placements.push_back(
            {index + 1u,
             {4.0 * double(index % 20u) + next(-0.8, 0.8), 4.0 * double(index / 20u % 20u) + next(-0.8, 0.8),
              4.0 * double(index / 400u) + next(-0.8, 0.8)}});
    }
    const ovspatialquery_population_handle_t spheres =
        buildPopulation(instance, 3, OVSPATIALQUERY_GEOMETRY_SPHERE, kSphereDefinition, placements);
    const ovspatialquery_population_handle_t boxes =
        buildPopulation(instance, 4, OVSPATIALQUERY_GEOMETRY_BOX, kBoxDefinition, placements);
    constexpr double kQueryHalf = 0.4;
    size_t           hits       = 0u;
    constexpr double kTwoPi     = 6.283185307179586;
    // 45 degrees about x, then y, then z.
    const float  tilted[4] = {0.19134172f, 0.46193977f, 0.19134172f, 0.84462320f};
    const double flat[3]   = {0.6, 0.3, 0.15};
    const double thin[3]   = {2.5, 0.12, 0.12};
    // Rotations draw from their own sequence, so the paths stay those above.
    uint64_t rotationState = 0x2545f4914f6cdd1dull;
    auto     unit          = [&]
    {
        rotationState ^= rotationState << 13;
        rotationState ^= rotationState >> 7;
        rotationState ^= rotationState << 17;
        return double(rotationState >> 11) * 0x1.0p-53;
    };
    size_t rotatedHits = 0u;
    for (int i = 0; i < 400; ++i)
    {
        const double start[3] = {next(-6.0, 82.0), next(-6.0, 82.0), next(-3.0, 11.0)};
        double       d[3]     = {next(-1.0, 1.0), next(-1.0, 1.0), next(-0.3, 0.3)};
        if (i % 8 == 0)
        {
            // Along an axis, where the box's swept volume fills its path's AABB.
            d[0] = i % 16 ? 0.0 : (i % 32 ? 1.0 : -1.0);
            d[1] = i % 16 ? (i % 32 > 16 ? 1.0 : -1.0) : 0.0;
            d[2] = 0.0;
        }
        const double length       = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        const float  direction[3] = {float(d[0] / length), float(d[1] / length), float(d[2] / length)};
        const double minimum      = i % 4 == 0 ? next(0.0, 3.0) : 0.0;
        const double maximum      = minimum + 300.0;

        const auto sweepHit = expectedSweepHit(start, 0.5, direction, minimum, maximum, placements);
        expectSweep(instance, 3, makeSweep(spheres, start, 0.5, direction, minimum, maximum), sweepHit);

        ovspatialquery_box_sweep_desc_t box{};
        box.common.target    = populationTarget(boxes);
        box.common.precision = OVSPATIALQUERY_QUERY_EXACT;
        for (int axis = 0; axis < 3; ++axis)
        {
            box.sweep.center[axis]         = start[axis];
            box.sweep.half_extents[axis]   = kQueryHalf;
            box.sweep.unit_direction[axis] = direction[axis];
        }
        box.sweep.rotation_xyzw[3] = 1.0f;
        box.sweep.min_distance     = minimum;
        box.sweep.max_distance     = maximum;
        // The boxes are their own bounds, so bounds precision meets them where
        // exact precision does; a contact tolerance grows the query box.
        const auto expectedBox = [&](double grow)
        {
            return expectedHit(placements,
                               [&](const double center[3], double& distance) {
                                   return oracleRayBox(start, direction, minimum, maximum, center,
                                                       kBoxHalfExtent + kQueryHalf + grow, distance);
                               });
        };
        const OracleExpectation expected = expectedBox(0.0);
        for (const auto precision : {OVSPATIALQUERY_QUERY_EXACT, OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE})
        {
            for (const double tolerance : {0.0, 0.05})
            {
                ovspatialquery_box_sweep_desc_t query          = box;
                query.common.precision                         = precision;
                query.common.contact_tolerance                 = tolerance;
                const OracleExpectation                   want = tolerance == 0.0 ? expected : expectedBox(tolerance);
                ovspatialquery_box_sweep_closest_result_t result{};
                REQUIRE(ovspatialquery_sweep_box_closest_sync(instance, &query, &result).status ==
                        OVSPATIALQUERY_API_SUCCESS);
                REQUIRE(result.source_ordinal == 4);
                REQUIRE(result.has_hit == (want.hasHit ? 1u : 0u));
                if (!want.hasHit)
                {
                    continue;
                }
                const bool   exact = precision == OVSPATIALQUERY_QUERY_EXACT;
                const auto   id    = exact ? result.hit.data.exact.identity.object_id.value
                                           : result.hit.data.bounds_candidate.identity.object_id.value;
                const double distance =
                    exact ? result.hit.data.exact.distance : result.hit.data.bounds_candidate.distance;
                REQUIRE(want.ties(id) && nearlyEqual(distance, want.distance));
                if (!exact || tolerance != 0.0)
                {
                    continue;
                }
                ++hits;
                const auto&   hit    = result.hit.data.exact;
                const double* center = placements[hit.identity.object_id.value - 1u].center;
                double        onBox = 0.0, inQuery = 0.0;
                for (int axis = 0; axis < 3; ++axis)
                {
                    onBox   = std::max(onBox, std::abs(hit.object_position[axis] - center[axis]));
                    inQuery = std::max(inQuery, std::abs(hit.object_position[axis] - start[axis] -
                                                         double(direction[axis]) * hit.distance));
                }
                // An initial overlap's point is defined differently.
                if (hit.distance > minimum)
                {
                    REQUIRE(nearlyEqual(onBox, kBoxHalfExtent) && inQuery <= kQueryHalf + 1.0e-6);
                }
            }
        }
        const bool tiltedHit =
            expectRotatedBoxSweep(instance, boxes, placements, start, direction, minimum, maximum, tilted, flat);
        // A uniformly random unit quaternion.
        const double u0 = unit(), u1 = kTwoPi * unit(), u2 = kTwoPi * unit();
        const float  random[4] = {float(std::sqrt(1.0 - u0) * std::sin(u1)), float(std::sqrt(1.0 - u0) * std::cos(u1)),
                                  float(std::sqrt(u0) * std::sin(u2)), float(std::sqrt(u0) * std::cos(u2))};
        const bool   thinHit =
            expectRotatedBoxSweep(instance, boxes, placements, start, direction, minimum, maximum, random, thin);
        rotatedHits += size_t(tiltedHit) + size_t(thinHit);

        // MULTIPLE follows the same traversal without stopping: one row per box
        // the oracle reaches, at its first contact.
        std::vector<ovspatialquery_box_sweep_hit_t> rows(placements.size());
        ovspatialquery_box_sweep_hit_buffer_t       buffer{rows.data(), rows.size()};
        ovspatialquery_multiple_result_t            multiple{};
        REQUIRE(ovspatialquery_sweep_box_multiple_sync(instance, &box, &buffer, &multiple).status ==
                OVSPATIALQUERY_API_SUCCESS);
        size_t reached = 0u;
        for (const Instance& placement : placements)
        {
            double distance = 0.0;
            if (!oracleRayBox(start, direction, minimum, maximum, placement.center, kBoxHalfExtent + kQueryHalf,
                              distance))
            {
                continue;
            }
            ++reached;
            bool found = false;
            for (uint64_t row = 0u; row < multiple.stored_hit_count; ++row)
            {
                const auto& hit = rows[row].data.exact;
                found = found || (hit.identity.object_id.value == placement.id && nearlyEqual(hit.distance, distance));
            }
            REQUIRE(found);
        }
        REQUIRE(!multiple.truncated && multiple.stored_hit_count == reached);
    }
    // Most paths reach an object, so the comparison covers the stopped traversal.
    REQUIRE(hits > 250u && rotatedHits > 250u);
    REQUIRE(ovspatialquery_destroy_population(instance, spheres).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_population(instance, boxes).status == OVSPATIALQUERY_API_SUCCESS);
}

// A rejected call returns `expectedStatus` and preserves the caller's output byte for byte.
template <typename Result, typename Query, typename Call>
void expectRejected(ovspatialquery_instance_handle_t instance, const Query& query, Call call,
                    ovspatialquery_api_status_t expectedStatus)
{
    Result before;
    std::memset(&before, 0xa5, sizeof(before));
    Result after = before;
    REQUIRE(call(instance, &query, &after).status == expectedStatus);
    REQUIRE(std::memcmp(&before, &after, sizeof(before)) == 0);
}

// Parallel staging must not turn an affordable lazy PhysX mesh query into HOST_BUDGET.
// Leave measured free bytes with a ballast plan, and compare a worker-only serial
// launch with the same rows on two workers. The last row first touches the mesh;
// earlier rows fill staged hits with box contacts before that lazy allocation.
bool coldMeshBatchAtBudget(uint32_t workers, uint64_t freeBytes)
{
    constexpr uint64_t                   budget = UINT64_C(16) << 20, count = 128u;
    const ovspatialquery_instance_desc_t desc{4u, workers, budget, nullptr, 0u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&desc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    registerBoxShape(instance);
    registerTriangleMeshShape(instance, 3u, {1.f, 9.f, -1.f, 1.f, 10.f, 1.f, 1.f, 11.f, -1.f}, {0u, 1u, 2u});
    InstanceRows rows;
    rows.add(10u, OVSPATIALQUERY_GEOMETRY_BOX, kBoxDefinition, 0., 0., 0.);
    rows.add(20u, OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, 3u, 0., 0., 0.);
    const auto                                     population = createExplicitPopulation(instance, 0u);
    const ovspatialquery_explicit_population_set_t set{1, rows.table(), nullptr, 0u};
    waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set));
    ovspatialquery_raycast_plan_desc_t planDesc{};
    planDesc.target                 = populationRef(population);
    planDesc.hit_mode               = OVSPATIALQUERY_HIT_MODE_CLOSEST;
    planDesc.precision              = OVSPATIALQUERY_QUERY_EXACT_SINGLE;
    planDesc.device                 = {kDLCPU, 0};
    planDesc.maximum_query_count    = count;
    planDesc.maximum_hits_per_query = 1u;
    planDesc.hit_capacity_per_slot  = count;
    planDesc.result_slot_count      = 1u;
    planDesc.result_fields          = OVSPATIALQUERY_BATCH_RESULT_DISTANCE;
    ovspatialquery_batch_plan_handle_t plan{};
    REQUIRE(ovspatialquery_create_raycast_plan(instance, &planDesc, &plan).status == OVSPATIALQUERY_API_SUCCESS);
    const auto usage = [instance]()
    {
        uint64_t bytes = 0u;
        REQUIRE(ovspatialquery_get_instance_memory_usage(instance, DLDevice{kDLCPU, 0}, &bytes).status ==
                OVSPATIALQUERY_API_SUCCESS);
        return bytes;
    };
    const auto ballast = [&](uint64_t capacity)
    {
        auto d                   = planDesc;
        d.hit_mode               = OVSPATIALQUERY_HIT_MODE_MULTIPLE;
        d.maximum_query_count    = 1u;
        d.maximum_hits_per_query = d.hit_capacity_per_slot = capacity;
        ovspatialquery_batch_plan_handle_t result{};
        REQUIRE(ovspatialquery_create_raycast_plan(instance, &d, &result).status == OVSPATIALQUERY_API_SUCCESS);
        return result;
    };
    const uint64_t base   = usage();
    auto           filler = ballast(1u);
    const uint64_t fixed  = usage() - base;
    REQUIRE(ovspatialquery_destroy_batch_plan(instance, filler).status == OVSPATIALQUERY_API_SUCCESS);
    filler                = ballast(1025u);
    const uint64_t perHit = (usage() - base - fixed) / 1024u;
    REQUIRE(ovspatialquery_destroy_batch_plan(instance, filler).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(perHit != 0u && base + fixed + freeBytes < budget);
    filler = ballast(1u + (budget - base - fixed - freeBytes) / perHit);
    std::vector<double> origins(count * 3u, 0.), distances(count, 10.);
    std::vector<float>  directions(count * 3u, 0.f);
    for (size_t row = 0u; row < count; ++row)
    {
        origins[row * 3u]    = -2.;
        directions[row * 3u] = 1.f;
    }
    origins[(count - 1u) * 3u + 1u] = 10.;
    InputArray originArray(origins.data(), count, 3, kDLFloat), directionArray(directions.data(), count, 3, kDLFloat);
    InputArray distanceArray(distances.data(), count, kDLFloat);
    ovspatialquery_raycast_batch_t batch{};
    batch.ordinal.kind    = OVSPATIALQUERY_ORDINAL_HEAD;
    batch.query_count     = count;
    batch.origins         = &originArray.value;
    batch.unit_directions = &directionArray.value;
    batch.max_distances   = &distanceArray.value;
    ovspatialquery_raycast_batch_reservation_t reservation{};
    auto admission = ovspatialquery_enqueue_raycast_plan(instance, plan, &batch, &reservation);
    bool success   = false;
    REQUIRE(admission.status == OVSPATIALQUERY_API_SUCCESS || admission.status == OVSPATIALQUERY_API_HOST_BUDGET);
    if (admission.status == OVSPATIALQUERY_API_SUCCESS)
    {
        ovspatialquery_op_status_t status{};
        const auto                 deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        do
        {
            REQUIRE(ovspatialquery_query_op_status(instance, admission.op_id, &status).status ==
                    OVSPATIALQUERY_API_SUCCESS);
            REQUIRE(std::chrono::steady_clock::now() < deadline);
            std::this_thread::yield();
        } while (status.state == OVSPATIALQUERY_OP_PENDING);
        success = status.state == OVSPATIALQUERY_OP_COMPLETED;
        if (success)
        {
            ovspatialquery_raycast_batch_result_t result{};
            REQUIRE(ovspatialquery_fetch_raycast_batch(instance, admission.op_id, &result).status ==
                    OVSPATIALQUERY_API_SUCCESS);
            REQUIRE(*static_cast<const uint64_t*>(result.columns.hit_ranges.stored_hit_count.data) == count);
            for (size_t row = 0u; row < count; ++row)
            {
                REQUIRE(static_cast<const uint64_t*>(result.columns.hit_ranges.hit_counts.data)[row] == 1u);
                REQUIRE(static_cast<const uint8_t*>(result.columns.hit_ranges.truncated.data)[row] == 0u);
                REQUIRE(static_cast<const uint64_t*>(result.columns.object_ids.data)[row] ==
                        (row + 1u == count ? 20u : 10u));
                REQUIRE(nearlyEqual(static_cast<const double*>(result.columns.contact.exact.distances.data)[row],
                                    row + 1u == count ? 3. : 1.));
            }
        }
        else
            REQUIRE(status.terminal_status == OVSPATIALQUERY_API_OPERATION_FAILED);
        REQUIRE(ovspatialquery_destroy_op(instance, &admission.op_id).status == OVSPATIALQUERY_API_SUCCESS);
    }
    REQUIRE(ovspatialquery_destroy_batch_plan(instance, filler).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_batch_plan(instance, plan).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(
        ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_BOX, kBoxDefinition}).status ==
        OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, 3u}).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
    return success;
}

void testColdMeshBatchUnderTightBudgets()
{
    uint64_t accepted = 0u;
    for (uint64_t freeBytes = 8u << 10; freeBytes <= (96u << 10); freeBytes += 512u)
    {
        if (coldMeshBatchAtBudget(1u, freeBytes))
        {
            ++accepted;
            REQUIRE(coldMeshBatchAtBudget(2u, freeBytes));
        }
    }
    REQUIRE(accepted != 0u);
}

void testTriangleRaycastPreservesDoublePoseAndSourceFeature()
{
    constexpr uint64_t       definitionValue = 3u;
    const std::vector<float> vertices{
        1.0, 9.0, -1.0, 1.0, 10.0, 1.0, 1.0, 11.0, -1.0, 1.0, -1.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, -1.0,
    };
    const std::vector<uint32_t>          triangles{0u, 1u, 2u, 3u, 4u, 5u};
    const ovspatialquery_instance_desc_t instanceDesc{4u, 1u, UINT64_C(16) * 1024u * 1024u, nullptr, 0u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&instanceDesc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    registerTriangleMeshShape(instance, definitionValue, vertices, triangles);

    const double                             translation[3] = {1000000000.125, -999999999.75, 7.0};
    const double                             rotation[4]    = {0.0, 0.0, 1.0, 0.0};
    const ovspatialquery_population_handle_t posed =
        buildTrianglePopulation(instance, 3, definitionValue, 31u, translation, rotation, 2.0);
    const double                  origin[3]   = {translation[0] + 10.0, translation[1], translation[2]};
    const float                   backward[3] = {-1.0f, 0.0f, 0.0f};
    ovspatialquery_raycast_desc_t ray         = makeRay(posed, origin, backward, 0.0, 12.0);
    ray.common.precision                      = OVSPATIALQUERY_QUERY_EXACT;
    ovspatialquery_raycast_single_result_t result{};
    REQUIRE(ovspatialquery_raycast_closest_sync(instance, &ray, &result).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(result.source_ordinal == 3);
    REQUIRE(result.has_hit == 1u);
    REQUIRE(result.hit.truth == OVSPATIALQUERY_HIT_EXACT_SURFACE);
    const ovspatialquery_raycast_exact_hit_t& hit = result.hit.data.exact;
    REQUIRE(hit.identity.object_id.value == 31u);
    REQUIRE(hit.identity.geometry.type == OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH);
    REQUIRE(hit.identity.geometry.value == definitionValue);
    REQUIRE(nearlyEqual(hit.distance, 12.0));
    REQUIRE(nearlyEqual(hit.position[0], translation[0] - 2.0));
    REQUIRE(nearlyEqual(hit.position[1], translation[1]));
    REQUIRE(nearlyEqual(hit.position[2], translation[2]));
    REQUIRE(nearlyEqual(hit.normal[0], 1.0));
    REQUIRE(nearlyEqual(hit.normal[1], 0.0));
    REQUIRE(nearlyEqual(hit.normal[2], 0.0));
    REQUIRE(hit.feature.kind == OVSPATIALQUERY_GEOMETRY_FEATURE_TRIANGLE);
    REQUIRE(hit.feature.data.triangle.source_face_index == 1u);
    REQUIRE(hit.feature.data.triangle.source_face_corner_indices[0] == 0u);
    REQUIRE(hit.feature.data.triangle.source_face_corner_indices[1] == 1u);
    REQUIRE(hit.feature.data.triangle.source_face_corner_indices[2] == 2u);
    REQUIRE(nearlyEqual(hit.feature.data.triangle.barycentrics[0], 0.25));
    REQUIRE(nearlyEqual(hit.feature.data.triangle.barycentrics[1], 0.5));
    REQUIRE(nearlyEqual(hit.feature.data.triangle.barycentrics[2], 0.25));

    ray.ray.max_distance = 12.0 - 1.0e-8;
    result               = {};
    REQUIRE(ovspatialquery_raycast_closest_sync(instance, &ray, &result).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(result.has_hit == 0u);

    const double                             identityTranslation[3] = {0.0, 0.0, 0.0};
    const double                             identityRotation[4]    = {0.0, 0.0, 0.0, 1.0};
    const ovspatialquery_population_handle_t boundary =
        buildTrianglePopulation(instance, 4, definitionValue, 32u, identityTranslation, identityRotation, 1.0);
    const double boundaryOrigin[3] = {-1048575.0, 0.0, 0.0};
    const float  forward[3]        = {1.0f, 0.0f, 0.0f};
    ray                            = makeRay(boundary, boundaryOrigin, forward, 0.0, 1048576.0);
    ray.common.precision           = OVSPATIALQUERY_QUERY_EXACT;
    result                         = {};
    REQUIRE(ovspatialquery_raycast_closest_sync(instance, &ray, &result).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(result.has_hit == 1u);
    REQUIRE(nearlyEqual(result.hit.data.exact.distance, 1048576.0));

    const double nearOrigin[3] = {0.0, 0.0, 0.0};
    ray                        = makeRay(boundary, nearOrigin, forward, 1.0, 1.0);
    ray.common.precision       = OVSPATIALQUERY_QUERY_EXACT;
    result                     = {};
    REQUIRE(ovspatialquery_raycast_closest_sync(instance, &ray, &result).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(result.has_hit == 1u);
    REQUIRE(nearlyEqual(result.hit.data.exact.distance, 1.0));

    ray                  = makeRay(boundary, nearOrigin, forward, 0.0, 1048577.0);
    ray.common.precision = OVSPATIALQUERY_QUERY_EXACT;
    result               = {};
    REQUIRE(ovspatialquery_raycast_closest_sync(instance, &ray, &result).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(result.has_hit == 1u);
    REQUIRE(nearlyEqual(result.hit.data.exact.distance, 1.0));

    REQUIRE(ovspatialquery_destroy_population(instance, posed).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_population(instance, boundary).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(
        ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, definitionValue})
            .status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

// A face just behind a ray's origin is clipped however far the ray reaches and whatever the
// mesh's size. Front faces at x = s and x = 2s are cast at along +x from 1e-8 s past the
// first, where the origin's float rounding lands on the first face: the hit is the second
// face, (1 - 1e-8) s away, for s = 1 and 1e-12 and ranges of 10 s and 1e8 s. An allowance
// for negative distances that grew with the range, or a fixed one at small sizes, reported
// the first face at 0.
void testTriangleRaycastClipsFaceBehindOrigin()
{
    const ovspatialquery_instance_desc_t instanceDesc{4u, 1u, UINT64_C(16) * 1024u * 1024u, nullptr, 0u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&instanceDesc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    const double translation[3]  = {0.0, 0.0, 0.0};
    const double rotation[4]     = {0.0, 0.0, 0.0, 1.0};
    const float  forward[3]      = {1.0f, 0.0f, 0.0f};
    uint64_t     definitionValue = 6u;
    for (const float size : {1.0f, 1.0e-12f})
    {
        std::vector<float> vertices{
            1.0, -1.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, -1.0, 2.0, -1.0, -1.0, 2.0, 0.0, 1.0, 2.0, 1.0, -1.0,
        };
        for (float& coordinate : vertices)
            coordinate *= size;
        registerTriangleMeshShape(instance, definitionValue, vertices, {0u, 1u, 2u, 3u, 4u, 5u});
        const ovspatialquery_population_handle_t population =
            buildTrianglePopulation(instance, 9, definitionValue, 61u, translation, rotation, 1.0);
        const double origin[3] = {(1.0 + 1.0e-8) * double(vertices[0]), 0.0, 0.0};
        const double expected  = 2.0 * double(size) - origin[0];
        for (const double reach : {10.0, 1.0e8})
        {
            ovspatialquery_raycast_desc_t ray = makeRay(population, origin, forward, 0.0, reach * double(size));
            ray.common.precision              = OVSPATIALQUERY_QUERY_EXACT;
            ovspatialquery_raycast_single_result_t closest{}, any{};
            REQUIRE(ovspatialquery_raycast_closest_sync(instance, &ray, &closest).status == OVSPATIALQUERY_API_SUCCESS);
            REQUIRE(closest.has_hit == 1u &&
                    std::fabs(closest.hit.data.exact.distance - expected) < 1.0e-6 * double(size));
            REQUIRE(closest.hit.data.exact.feature.data.triangle.source_face_index == 1u);
            REQUIRE(ovspatialquery_raycast_any_sync(instance, &ray, &any).status == OVSPATIALQUERY_API_SUCCESS);
            REQUIRE(any.has_hit == 1u && any.hit.data.exact.feature.data.triangle.source_face_index == 1u);
        }
        REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(ovspatialquery_destroy_geometry_definition(instance,
                                                           {OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, definitionValue})
                    .status == OVSPATIALQUERY_API_SUCCESS);
        ++definitionValue;
    }
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

// A face just beyond a short ray's range is clipped. A triangle of size s = 2^-40 crosses a
// ray along +x 2^-60 from its origin: a range of 2^-59 reports it there and one of 2^-61
// reports nothing (an end-of-range allowance of at least 16 double epsilons admitted it).
void testTriangleRaycastClipsFaceBeyondShortRange()
{
    constexpr uint64_t                   definitionValue = 8u;
    const float                          s               = 0x1p-40f;
    const ovspatialquery_instance_desc_t instanceDesc{4u, 1u, UINT64_C(16) * 1024u * 1024u, nullptr, 0u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&instanceDesc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    registerTriangleMeshShape(instance, definitionValue, {0.0f, -s, -s, s, 0.0f, s, 2.0f * s, s, -s}, {0u, 1u, 2u});
    const double                             translation[3] = {0.0, 0.0, 0.0};
    const double                             rotation[4]    = {0.0, 0.0, 0.0, 1.0};
    const ovspatialquery_population_handle_t population =
        buildTrianglePopulation(instance, 10, definitionValue, 81u, translation, rotation, 1.0);
    const double origin[3]  = {double(s) - 0x1p-60, 0.0, 0.0};
    const float  forward[3] = {1.0f, 0.0f, 0.0f};
    for (const double reach : {0x1p-59, 0x1p-61})
    {
        ovspatialquery_raycast_desc_t ray = makeRay(population, origin, forward, 0.0, reach);
        ray.common.precision              = OVSPATIALQUERY_QUERY_EXACT;
        ovspatialquery_raycast_single_result_t closest{}, any{};
        REQUIRE(ovspatialquery_raycast_closest_sync(instance, &ray, &closest).status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(ovspatialquery_raycast_any_sync(instance, &ray, &any).status == OVSPATIALQUERY_API_SUCCESS);
        if (reach > 0x1p-60)
        {
            REQUIRE(closest.has_hit == 1u && std::fabs(closest.hit.data.exact.distance - 0x1p-60) < 0x1p-70);
            REQUIRE(any.has_hit == 1u);
        }
        else
            REQUIRE(closest.has_hit == 0u && any.has_hit == 0u);
    }
    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(
        ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, definitionValue})
            .status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

// A ray from very far away still meets the mesh. A unit triangle in z = 1 is cast at along -z
// from 2^56: the ray starts near the mesh in double, and at that distance rounding can carry
// the start past the mesh, which then reported nothing.
void testTriangleRaycastFromFarAway()
{
    constexpr uint64_t                   definitionValue = 9u;
    const ovspatialquery_instance_desc_t instanceDesc{4u, 1u, UINT64_C(16) * 1024u * 1024u, nullptr, 0u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&instanceDesc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    registerTriangleMeshShape(instance, definitionValue, {-1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 0.0f, 1.0f, 1.0f},
                              {0u, 1u, 2u});
    const double                             translation[3] = {0.0, 0.0, 0.0};
    const double                             rotation[4]    = {0.0, 0.0, 0.0, 1.0};
    const ovspatialquery_population_handle_t population =
        buildTrianglePopulation(instance, 11, definitionValue, 91u, translation, rotation, 1.0);
    const double                  origin[3] = {0.0, 0.0, 0x1p56};
    const float                   down[3]   = {0.0f, 0.0f, -1.0f};
    ovspatialquery_raycast_desc_t ray       = makeRay(population, origin, down, 0.0, 0x1p57);
    ray.common.precision                    = OVSPATIALQUERY_QUERY_EXACT;
    ovspatialquery_raycast_single_result_t closest{}, any{};
    REQUIRE(ovspatialquery_raycast_closest_sync(instance, &ray, &closest).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(closest.has_hit == 1u && std::fabs(closest.hit.data.exact.distance - (0x1p56 - 1.0)) <= 64.0);
    REQUIRE(ovspatialquery_raycast_any_sync(instance, &ray, &any).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(any.has_hit == 1u);
    // MULTIPLE decides each crossing exactly; its contact is still the point on the triangle,
    // which the point at that distance along the ray, 2^56 long, would round off.
    ovspatialquery_raycast_hit_t        hits[1]{};
    ovspatialquery_raycast_hit_buffer_t buffer{hits, 1u};
    ovspatialquery_multiple_result_t    multiple{};
    REQUIRE(ovspatialquery_raycast_multiple_sync(instance, &ray, &buffer, &multiple).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(multiple.stored_hit_count == 1u);
    REQUIRE(std::fabs(hits[0].data.exact.position[0]) <= 1.0e-6 &&
            std::fabs(hits[0].data.exact.position[1]) <= 1.0e-6 &&
            std::fabs(hits[0].data.exact.position[2] - 1.0) <= 1.0e-6);
    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(
        ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, definitionValue})
            .status == OVSPATIALQUERY_API_SUCCESS);

    // Obliquely from as far: a triangle in z = 4.25 cast at along x = z from (2^56, 0, 2^56)
    // crosses it at (4.25, 0, 4.25), inside it. Decided from the far origin, the 2^56 offsets
    // swamp the triangle's coordinates and put the crossing outside it.
    constexpr uint64_t obliqueValue = 15u;
    registerTriangleMeshShape(instance, obliqueValue, {1.5f, -1.0f, 4.25f, 5.5f, -1.0f, 4.25f, 3.5f, 1.0f, 4.25f},
                              {0u, 1u, 2u});
    const ovspatialquery_population_handle_t oblique =
        buildTrianglePopulation(instance, 12, obliqueValue, 92u, translation, rotation, 1.0);
    const double                  far[3]   = {0x1p56, 0.0, 0x1p56};
    const float                   slant[3] = {-0.70710677f, 0.0f, -0.70710677f};
    const double                  reach    = (0x1p56 - 4.25) / 0.70710677f;
    ovspatialquery_raycast_desc_t slanted  = makeRay(oblique, far, slant, 0.0, 0x1p57);
    slanted.common.precision               = OVSPATIALQUERY_QUERY_EXACT;
    multiple                               = {};
    REQUIRE(ovspatialquery_raycast_multiple_sync(instance, &slanted, &buffer, &multiple).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(multiple.stored_hit_count == 1u);
    REQUIRE(std::fabs(hits[0].data.exact.distance - reach) <= 64.0);
    REQUIRE(std::fabs(hits[0].data.exact.position[0] - 4.25) <= 1.0e-6 &&
            std::fabs(hits[0].data.exact.position[1]) <= 1.0e-6 &&
            std::fabs(hits[0].data.exact.position[2] - 4.25) <= 1.0e-6);
    REQUIRE(ovspatialquery_destroy_population(instance, oblique).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, obliqueValue})
                .status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

// A long ray still meets a tiny mesh. A triangle of size s = 2^-69 is cast at along -z from
// s over [0, 2^60 - 2^40]: in the mesh's cooked frame that range nears the largest float, and
// the float ray's traversal, which reaches slightly past it, overflowed and found nothing.
void testTriangleRaycastLongRayAtTinyMesh()
{
    constexpr uint64_t                   definitionValue = 10u;
    const float                          s               = 0x1p-69f;
    const ovspatialquery_instance_desc_t instanceDesc{4u, 1u, UINT64_C(16) * 1024u * 1024u, nullptr, 0u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&instanceDesc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    registerTriangleMeshShape(instance, definitionValue, {-s, -s, 0.0f, s, -s, 0.0f, 0.0f, s, 0.0f}, {0u, 1u, 2u});
    const double                             translation[3] = {0.0, 0.0, 0.0};
    const double                             rotation[4]    = {0.0, 0.0, 0.0, 1.0};
    const ovspatialquery_population_handle_t population =
        buildTrianglePopulation(instance, 12, definitionValue, 101u, translation, rotation, 1.0);
    const double                  origin[3] = {0.0, 0.0, double(s)};
    const float                   down[3]   = {0.0f, 0.0f, -1.0f};
    ovspatialquery_raycast_desc_t ray       = makeRay(population, origin, down, 0.0, 0x1p60 - 0x1p40);
    ray.common.precision                    = OVSPATIALQUERY_QUERY_EXACT;
    ovspatialquery_raycast_single_result_t closest{}, any{};
    REQUIRE(ovspatialquery_raycast_closest_sync(instance, &ray, &closest).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(closest.has_hit == 1u && std::fabs(closest.hit.data.exact.distance - double(s)) <= 1.0e-6 * double(s));
    REQUIRE(ovspatialquery_raycast_any_sync(instance, &ray, &any).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(any.has_hit == 1u);
    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(
        ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, definitionValue})
            .status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

// The nearest crossing wins when the exact decision of a near-edge candidate moves it past
// another face. Along +x, face A in x = 1 is crossed at 1; face B, grazed 3e-5 outside an
// edge, has a float crossing at 0.99997, before A, and an exact one at 1.000015, after it.
// The closest hit is A, with a range just past A's crossing and with a long one.
void testTriangleRaycastRefinedCandidateKeepsOrder()
{
    constexpr uint64_t                   definitionValue = 11u;
    const float                          e = 0x1p-23f, m = 0x1p-10f;
    const ovspatialquery_instance_desc_t instanceDesc{4u, 1u, UINT64_C(16) * 1024u * 1024u, nullptr, 0u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&instanceDesc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    registerTriangleMeshShape(instance, definitionValue,
                              {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 2.0f, 1.0f, 2.0f, 0.0f, 1.0f, 1.0f, 1.0f, 2.0f, 1.0f + m,
                               0.0f, 1.0f, 1.0f + e, -1.0f},
                              {0u, 1u, 2u, 3u, 4u, 5u});
    const double                             translation[3] = {0.0, 0.0, 0.0};
    const double                             rotation[4]    = {0.0, 0.0, 0.0, 1.0};
    const ovspatialquery_population_handle_t population =
        buildTrianglePopulation(instance, 13, definitionValue, 111u, translation, rotation, 1.0);
    const double origin[3]  = {0.0, 1.0 + 3.0 * 0x1p-26, 0.5};
    const float  forward[3] = {1.0f, 0.0f, 0.0f};
    for (const double reach : {1.0 + 0x1p-17, 10.0})
    {
        ovspatialquery_raycast_desc_t ray = makeRay(population, origin, forward, 0.0, reach);
        ray.common.precision              = OVSPATIALQUERY_QUERY_EXACT;
        ovspatialquery_raycast_single_result_t closest{};
        REQUIRE(ovspatialquery_raycast_closest_sync(instance, &ray, &closest).status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(closest.has_hit == 1u && closest.hit.data.exact.feature.data.triangle.source_face_index == 0u);
        REQUIRE(std::fabs(closest.hit.data.exact.distance - 1.0) < 1.0e-6);
    }
    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(
        ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, definitionValue})
            .status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

// Faces a ray crosses, in order, and the rays: each row's range reaches `crossings[row]`, the
// exact distances along it to faces 0, 1, ... in turn, and no other face; a reported distance
// lies within `tolerances[row]` of its face's.
struct GrazingScene
{
    std::vector<float>               vertices;
    std::vector<uint32_t>            triangles;
    std::vector<double>              origins, maximums, tolerances;
    std::vector<float>               directions;
    std::vector<std::vector<double>> crossings;

    void add(const double origin[3], const float direction[3], double maximum, std::vector<double> crossed,
             double tolerance)
    {
        origins.insert(origins.end(), origin, origin + 3);
        directions.insert(directions.end(), direction, direction + 3);
        maximums.push_back(maximum);
        crossings.push_back(std::move(crossed));
        tolerances.push_back(tolerance);
    }
};

// The double distance along `direction` from `origin` to the plane of the scene's face `face`.
double planeCrossing(const GrazingScene& scene, uint32_t face, const double origin[3], const float direction[3])
{
    double corner[3][3]{};
    for (uint32_t k = 0; k < 3; ++k)
        for (uint32_t axis = 0; axis < 3; ++axis)
            corner[k][axis] = scene.vertices[3u * scene.triangles[3u * face + k] + axis];
    double e1[3]{}, e2[3]{};
    for (uint32_t axis = 0; axis < 3; ++axis)
    {
        e1[axis] = corner[1][axis] - corner[0][axis];
        e2[axis] = corner[2][axis] - corner[0][axis];
    }
    const double normal[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2],
                              e1[0] * e2[1] - e1[1] * e2[0]};
    double       height = 0, approach = 0;
    for (uint32_t axis = 0; axis < 3; ++axis)
    {
        height += normal[axis] * (corner[0][axis] - origin[axis]);
        approach += normal[axis] * double(direction[axis]);
    }
    return height / approach;
}

// Every row of `scene`, synchronously and in CPU plans. CLOSEST reports face 0, ANY a face the
// range reaches, and MULTIPLE every face it reaches, each distance at its own face's crossing;
// MULTIPLE rows are matched by face, since their order is unspecified. With room for one hit,
// a MULTIPLE row keeps the nearest, face 0, and is truncated when the range reaches both.
void expectExactOrder(const GrazingScene& scene, uint64_t definitionValue)
{
    const uint64_t                       rows = scene.maximums.size();
    const ovspatialquery_instance_desc_t instanceDesc{4u, 1u, UINT64_C(16) * 1024u * 1024u, nullptr, 0u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&instanceDesc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    registerTriangleMeshShape(instance, definitionValue, scene.vertices, scene.triangles);
    const double                             translation[3] = {0.0, 0.0, 0.0};
    const double                             rotation[4]    = {0.0, 0.0, 0.0, 1.0};
    const ovspatialquery_population_handle_t population =
        buildTrianglePopulation(instance, 14, definitionValue, 131u, translation, rotation, 1.0);
    // MULTIPLE decides every crossing exactly; CLOSEST and ANY keep PhysX's float distance for
    // a crossing it decides away from an edge and the range's end, so theirs is checked to the
    // relative 1e-5 the other mesh ray tests use.
    const auto onFace = [&](uint64_t row, uint64_t face, double distance, bool exact)
    {
        if (face >= scene.crossings[row].size())
            return false;
        const double crossing = scene.crossings[row][face];
        return std::fabs(distance - crossing) <= (exact ? scene.tolerances[row] : 1.0e-5 * crossing);
    };
    // Each face the row reaches, once, at its own crossing.
    const auto everyFace = [&](uint64_t row, const uint64_t* faces, const double* distances, uint64_t count)
    {
        bool seen[2]{};
        for (uint64_t hit = 0u; hit < count; ++hit)
        {
            if (!onFace(row, faces[hit], distances[hit], true) || seen[faces[hit]])
                return false;
            seen[faces[hit]] = true;
        }
        return count == scene.crossings[row].size();
    };

    for (uint64_t row = 0u; row < rows; ++row)
    {
        const uint64_t                reached = scene.crossings[row].size();
        ovspatialquery_raycast_desc_t ray =
            makeRay(population, &scene.origins[row * 3u], &scene.directions[row * 3u], 0.0, scene.maximums[row]);
        ray.common.precision = OVSPATIALQUERY_QUERY_EXACT;
        ovspatialquery_raycast_single_result_t closest{}, any{};
        REQUIRE(ovspatialquery_raycast_closest_sync(instance, &ray, &closest).status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(closest.has_hit == (reached > 0u ? 1u : 0u));
        REQUIRE(!closest.has_hit || (closest.hit.data.exact.feature.data.triangle.source_face_index == 0u &&
                                     onFace(row, 0u, closest.hit.data.exact.distance, false)));
        REQUIRE(ovspatialquery_raycast_any_sync(instance, &ray, &any).status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(any.has_hit == (reached > 0u ? 1u : 0u));
        REQUIRE(!any.has_hit || onFace(row, any.hit.data.exact.feature.data.triangle.source_face_index,
                                       any.hit.data.exact.distance, false));
        for (const uint32_t capacity : {2u, 1u})
        {
            ovspatialquery_raycast_hit_t        hits[2]{};
            ovspatialquery_raycast_hit_buffer_t buffer{hits, capacity};
            ovspatialquery_multiple_result_t    multiple{};
            REQUIRE(ovspatialquery_raycast_multiple_sync(instance, &ray, &buffer, &multiple).status ==
                    OVSPATIALQUERY_API_SUCCESS);
            uint64_t faces[2]{};
            double   distances[2]{};
            for (uint64_t hit = 0u; hit < multiple.stored_hit_count && hit < 2u; ++hit)
            {
                faces[hit]     = hits[hit].data.exact.feature.data.triangle.source_face_index;
                distances[hit] = hits[hit].data.exact.distance;
            }
            if (capacity == 2u)
                REQUIRE(everyFace(row, faces, distances, multiple.stored_hit_count) && multiple.truncated == 0u);
            else
                REQUIRE(multiple.stored_hit_count == std::min<uint64_t>(1u, reached) &&
                        multiple.truncated == (reached > 1u ? 1u : 0u) &&
                        (reached == 0u || (faces[0] == 0u && onFace(row, 0u, distances[0], true))));
        }
    }

    InputArray originArray(scene.origins.data(), rows, 3, kDLFloat);
    InputArray directionArray(scene.directions.data(), rows, 3, kDLFloat);
    InputArray maximumArray(scene.maximums.data(), rows, kDLFloat);
    const struct
    {
        ovspatialquery_hit_mode_t mode;
        uint64_t                  perQuery;
    } plans[] = {{OVSPATIALQUERY_HIT_MODE_CLOSEST, 1u},
                 {OVSPATIALQUERY_HIT_MODE_ANY, 1u},
                 {OVSPATIALQUERY_HIT_MODE_MULTIPLE, 2u},
                 {OVSPATIALQUERY_HIT_MODE_MULTIPLE, 1u}};
    for (const auto& planned : plans)
    {
        ovspatialquery_raycast_plan_desc_t desc{};
        desc.target                 = populationRef(population);
        desc.hit_mode               = planned.mode;
        desc.precision              = OVSPATIALQUERY_QUERY_EXACT;
        desc.device                 = {kDLCPU, 0};
        desc.maximum_query_count    = rows;
        desc.maximum_hits_per_query = planned.perQuery;
        desc.hit_capacity_per_slot  = rows * planned.perQuery;
        desc.result_slot_count      = 1u;
        desc.result_fields          = OVSPATIALQUERY_BATCH_RESULT_DISTANCE | OVSPATIALQUERY_BATCH_RESULT_SOURCE_FEATURE;
        ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
        REQUIRE(ovspatialquery_create_raycast_plan(instance, &desc, &plan).status == OVSPATIALQUERY_API_SUCCESS);
        ovspatialquery_raycast_batch_t batch{};
        batch.ordinal.kind    = OVSPATIALQUERY_ORDINAL_HEAD;
        batch.query_count     = rows;
        batch.origins         = &originArray.value;
        batch.unit_directions = &directionArray.value;
        batch.max_distances   = &maximumArray.value;
        ovspatialquery_raycast_batch_reservation_t reservation{};
        auto admission = ovspatialquery_enqueue_raycast_plan(instance, plan, &batch, &reservation);
        REQUIRE(admission.status == OVSPATIALQUERY_API_SUCCESS);
        ovspatialquery_op_wait_result_t waited{};
        REQUIRE(ovspatialquery_wait_op(instance, admission.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status ==
                OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(waited.state == OVSPATIALQUERY_OP_COMPLETED);
        ovspatialquery_raycast_batch_result_t result{};
        REQUIRE(ovspatialquery_fetch_raycast_batch(instance, admission.op_id, &result).status ==
                OVSPATIALQUERY_API_SUCCESS);
        const auto* offsets   = static_cast<const uint64_t*>(result.columns.hit_ranges.hit_offsets.data);
        const auto* counts    = static_cast<const uint64_t*>(result.columns.hit_ranges.hit_counts.data);
        const auto* truncated = static_cast<const uint8_t*>(result.columns.hit_ranges.truncated.data);
        const auto* distances = static_cast<const double*>(result.columns.contact.exact.distances.data);
        const auto* faces =
            static_cast<const uint64_t*>(result.columns.contact.exact.features.source_face_indices.data);
        for (uint64_t row = 0u; row < rows; ++row)
        {
            const uint64_t reached = scene.crossings[row].size();
            const uint64_t at      = offsets[row];
            if (planned.mode == OVSPATIALQUERY_HIT_MODE_MULTIPLE && planned.perQuery == 2u)
                REQUIRE(everyFace(row, faces + at, distances + at, counts[row]) && truncated[row] == 0u);
            else
            {
                REQUIRE(counts[row] == std::min<uint64_t>(1u, reached));
                const bool multiple = planned.mode == OVSPATIALQUERY_HIT_MODE_MULTIPLE;
                REQUIRE(!multiple || truncated[row] == (reached > 1u ? 1u : 0u));
                // CLOSEST and a one-hit MULTIPLE keep the nearest; ANY any face the range reaches.
                REQUIRE(counts[row] == 0u || ((planned.mode == OVSPATIALQUERY_HIT_MODE_ANY || faces[at] == 0u) &&
                                              onFace(row, faces[at], distances[at], multiple)));
            }
        }
        REQUIRE(ovspatialquery_destroy_op(instance, &admission.op_id).status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(ovspatialquery_destroy_batch_plan(instance, plan).status == OVSPATIALQUERY_API_SUCCESS);
    }

    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(
        ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, definitionValue})
            .status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

// Hits follow the exact distance where PhysX's float distance cannot tell two crossings apart
// or place one on the right side of the range's end.
//
// Stacked faces: face A lies in z = 0 and face B 2^-30 below it, with an edge in y = 0. Rays
// fall at 1 in 10 from a height of 10 onto points within 1e-4 of that edge, so each crosses A
// at 100 and then B 9.3e-9 later, both within PhysX's float rounding; B's crossing near its
// edge is decided exactly. A range beyond B reaches both: CLOSEST reports A, MULTIPLE both,
// each at its own crossing, and a one-hit MULTIPLE keeps A. One ending between them reaches A
// alone, and one ending as far before A reaches neither.
//
// A tilted face: rays fall at 0.3 % incidence from 0.3 above it, where PhysX's float distance
// errs by more than the ulps of the coordinates, with ranges ending 2^-30 of the distance past
// the crossing (a hit) and before it (none).
void testTriangleRaycastGrazingFacesKeepExactOrder()
{
    uint64_t   state   = UINT64_C(0x9e3779b97f4a7c15);
    const auto uniform = [&state](double low, double high)
    {
        state ^= state << 13u;
        state ^= state >> 7u;
        state ^= state << 17u;
        return low + (high - low) * double(state >> 11u) * 0x1.0p-53;
    };
    constexpr uint32_t count = 64u;

    GrazingScene stacked;
    const float  below = -0x1p-30f;
    stacked.vertices   = {-8.f, -8.f, 0.f,   8.f, -8.f, 0.f,   0.f, 8.f, 0.f,
                          -4.f, 0.f,  below, 4.f, 0.f,  below, 0.f, 8.f, below};
    stacked.triangles  = {0u, 1u, 2u, 3u, 4u, 5u};
    for (uint32_t row = 0u; row < count; ++row)
    {
        const double azimuth = uniform(0.0, 6.283185307179586);
        const float  direction[3]{float(0.99498743710662 * std::cos(azimuth)),
                                  float(0.99498743710662 * std::sin(azimuth)), -0.1f};
        const double origin[3]{uniform(-0.5, 0.5) - double(direction[0]) * 100.0,
                               uniform(0.0, 1.0e-4) - double(direction[1]) * 100.0, 10.0};
        const double a    = planeCrossing(stacked, 0u, origin, direction),
                     b    = planeCrossing(stacked, 1u, origin, direction);
        const double half = (b - a) * 0.5;
        stacked.add(origin, direction, 1000.0, {a, b}, half);
        stacked.add(origin, direction, a + half, {a}, half);
        stacked.add(origin, direction, a - half, {}, half);
    }
    expectExactOrder(stacked, 13u);

    GrazingScene tilted;
    const double normal[3]{0.3, 0.5, 0.81};
    const double norm = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
    const double n[3]{normal[0] / norm, normal[1] / norm, normal[2] / norm};
    const double sideLength = std::sqrt(n[0] * n[0] + n[1] * n[1]);
    const double u[3]{n[1] / sideLength, -n[0] / sideLength, 0.0};
    const double w[3]{n[1] * u[2] - n[2] * u[1], n[2] * u[0] - n[0] * u[2], n[0] * u[1] - n[1] * u[0]};
    const double center[3]{1.0 / 3.0, 1.0 / 5.0, 1.0 / 7.0};
    for (const auto& corner : {std::array<double, 2>{-8.0, -8.0}, {8.0, -8.0}, {0.0, 8.0}})
        for (uint32_t axis = 0; axis < 3; ++axis)
            tilted.vertices.push_back(float(center[axis] + corner[0] * u[axis] + corner[1] * w[axis]));
    tilted.triangles  = {0u, 1u, 2u};
    const double sine = 0.003, cosine = std::sqrt(1.0 - sine * sine);
    for (uint32_t row = 0u; row < count; ++row)
    {
        const double azimuth = uniform(0.0, 6.283185307179586), p = uniform(-1.0, 1.0), q = uniform(-1.0, 1.0);
        float        direction[3]{};
        double       origin[3]{};
        for (uint32_t axis = 0; axis < 3; ++axis)
            direction[axis] =
                float(cosine * (std::cos(azimuth) * u[axis] + std::sin(azimuth) * w[axis]) - sine * n[axis]);
        for (uint32_t axis = 0; axis < 3; ++axis)
            origin[axis] = center[axis] + p * u[axis] + q * w[axis] - double(direction[axis]) * 0.3 / sine;
        const double crossing = planeCrossing(tilted, 0u, origin, direction);
        tilted.add(origin, direction, crossing * (1.0 + 0x1p-30), {crossing}, crossing * 0x1p-31);
        tilted.add(origin, direction, crossing * (1.0 - 0x1p-30), {}, 0.0);
    }
    expectExactOrder(tilted, 14u);
}

void testTriangleAnyIgnoresRoundedRangeOnlyHit()
{
    constexpr uint64_t       definitionValue = 5u;
    const float              farDistance     = std::nextafter(1.0f, 2.0f);
    const std::vector<float> vertices{
        farDistance, -1.0, -1.0, farDistance, 0.0, 1.0, farDistance, 1.0, -1.0,
        0.5,         -1.0, -1.0, 0.5,         0.0, 1.0, 0.5,         1.0, -1.0,
    };
    const std::vector<uint32_t>          triangles{0u, 1u, 2u, 3u, 4u, 5u};
    const ovspatialquery_instance_desc_t instanceDesc{4u, 1u, UINT64_C(16) * 1024u * 1024u, nullptr, 0u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&instanceDesc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    registerTriangleMeshShape(instance, definitionValue, vertices, triangles);
    const double                             translation[3] = {0.0, 0.0, 0.0};
    const double                             rotation[4]    = {0.0, 0.0, 0.0, 1.0};
    const ovspatialquery_population_handle_t population =
        buildTrianglePopulation(instance, 6, definitionValue, 51u, translation, rotation, 1.0);

    const double                  origin[3]  = {0.0, 0.0, 0.0};
    const float                   forward[3] = {1.0f, 0.0f, 0.0f};
    ovspatialquery_raycast_desc_t ray        = makeRay(population, origin, forward, 0.0, 1.00000006);
    ray.common.precision                     = OVSPATIALQUERY_QUERY_EXACT;
    ovspatialquery_raycast_single_result_t result{};
    REQUIRE(ovspatialquery_raycast_any_sync(instance, &ray, &result).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(result.has_hit == 1u);
    REQUIRE(result.hit.data.exact.identity.object_id.value == 51u);
    REQUIRE(nearlyEqual(result.hit.data.exact.distance, 0.5));
    REQUIRE(result.hit.data.exact.feature.kind == OVSPATIALQUERY_GEOMETRY_FEATURE_TRIANGLE);
    REQUIRE(result.hit.data.exact.feature.data.triangle.source_face_index == 1u);

    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(
        ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, definitionValue})
            .status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

// far_sphere_sweep_scene.h's rows on CPU: spheres meeting a triangle's face or
// a large cube's near face more than a thousand units from their start must
// report that contact, not miss it or a later one, and spheres grazing the
// triangle from just beyond their radius must miss it.
void testSphereSweepsMeetFarFaces()
{
    const ovspatialquery_instance_desc_t desc{4u, 1u, UINT64_C(16) * 1024u * 1024u, nullptr, 0u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&desc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    FarSphereSweepScene scene(instance);
    const std::array    cases{std::make_pair(scene.meshPopulation, &scene.meshRows),
                           std::make_pair(scene.cubePopulation, &scene.cubeRows)};
    for (const auto& [population, rows] : cases)
    {
        for (size_t row = 0u; row < rows->expected.size(); ++row)
        {
            const ovspatialquery_sphere_sweep_desc_t query =
                makeSweep(population, &rows->origins[row * 3u], FarSphereSweepScene::kRadius,
                          &rows->directions[row * 3u], 0.0, rows->maximumDistances[row]);
            ovspatialquery_sphere_sweep_closest_result_t result{};
            REQUIRE(ovspatialquery_sweep_sphere_closest_sync(instance, &query, &result).status ==
                    OVSPATIALQUERY_API_SUCCESS);
            REQUIRE(result.has_hit == 1u &&
                    std::abs(result.hit.data.exact.distance - rows->expected[row]) <= 1.0e-9 * rows->expected[row]);
        }
    }
    const FarSphereSweepScene::Rows& grazing = scene.grazingRows;
    for (size_t row = 0u; row < grazing.maximumDistances.size(); ++row)
    {
        const ovspatialquery_sphere_sweep_desc_t query =
            makeSweep(scene.meshPopulation, &grazing.origins[row * 3u], FarSphereSweepScene::kRadius,
                      &grazing.directions[row * 3u], 0.0, grazing.maximumDistances[row]);
        ovspatialquery_sphere_sweep_closest_result_t result{};
        REQUIRE(ovspatialquery_sweep_sphere_closest_sync(instance, &query, &result).status ==
                OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(result.has_hit == 0u);
    }
    scene.destroy();
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

// A ray aimed at the edge two mesh triangles share crosses one of them. A unit square of two
// single-sided triangles in z = 0, split on its diagonal, is seen from 10 and from 100,000
// units by rays aimed at points of the diagonal, whose float origins and directions put each
// crossing within rounding of the edge. Every query form must report the crossing where the
// ray reaches z = 0. PhysX's float test with a margin of 2^-22 let about 3 % pass between the
// triangles and report nothing (BH-075); from 100,000 units its rounding exceeds any margin
// unless the float ray starts near the mesh.
void testTriangleRaycastHitsSharedEdges()
{
    constexpr uint64_t                   definitionValue = 7u;
    constexpr uint64_t                   count           = 2000u;
    const ovspatialquery_instance_desc_t instanceDesc{4u, 1u, UINT64_C(64) * 1024u * 1024u, nullptr, 0u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&instanceDesc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    registerTriangleMeshShape(instance, definitionValue, {0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 1.f, 0.f},
                              {0u, 1u, 2u, 0u, 2u, 3u});
    const double                             translation[3] = {0.0, 0.0, 0.0};
    const double                             rotation[4]    = {0.0, 0.0, 0.0, 1.0};
    const ovspatialquery_population_handle_t population =
        buildTrianglePopulation(instance, 8, definitionValue, 71u, translation, rotation, 1.0);

    uint64_t   state   = UINT64_C(0x9e3779b97f4a7c15);
    const auto uniform = [&state](double low, double high)
    {
        state ^= state << 13u;
        state ^= state >> 7u;
        state ^= state << 17u;
        return low + (high - low) * double(state >> 11u) * 0x1.0p-53;
    };
    std::vector<double> origins, maximums(count, 2.0e5), expected;
    std::vector<float>  directions;
    for (uint64_t row = 0u; row < count; ++row)
    {
        const double back  = row % 2u ? 1.0e5 : 10.0;
        const double along = uniform(0.05, 0.95);
        double       away[3]{uniform(-1.0, 1.0), uniform(-1.0, 1.0), uniform(0.2, 1.0)};
        const double norm = std::sqrt(away[0] * away[0] + away[1] * away[1] + away[2] * away[2]);
        double       origin[3]{}, toward[3]{};
        for (int axis = 0; axis < 3; ++axis)
            origin[axis] = double(float((axis < 2 ? along : 0.0) + back * away[axis] / norm));
        for (int axis = 0; axis < 3; ++axis)
            toward[axis] = (axis < 2 ? along : 0.0) - origin[axis];
        const double length = std::sqrt(toward[0] * toward[0] + toward[1] * toward[1] + toward[2] * toward[2]);
        float        direction[3]{};
        for (int axis = 0; axis < 3; ++axis)
            direction[axis] = float(toward[axis] / length);
        origins.insert(origins.end(), origin, origin + 3);
        directions.insert(directions.end(), direction, direction + 3);
        expected.push_back(origin[2] / -double(direction[2]));
    }
    const auto onSquare = [&](uint64_t row, double distance)
    { return std::fabs(distance - expected[row]) <= 1.0e-5 * expected[row]; };

    for (uint64_t row = 0u; row < count; ++row)
    {
        ovspatialquery_raycast_desc_t ray =
            makeRay(population, &origins[row * 3u], &directions[row * 3u], 0.0, maximums[row]);
        ray.common.precision = OVSPATIALQUERY_QUERY_EXACT;
        ovspatialquery_raycast_single_result_t closest{}, any{};
        REQUIRE(ovspatialquery_raycast_closest_sync(instance, &ray, &closest).status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(closest.has_hit == 1u && onSquare(row, closest.hit.data.exact.distance));
        REQUIRE(ovspatialquery_raycast_any_sync(instance, &ray, &any).status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(any.has_hit == 1u && onSquare(row, any.hit.data.exact.distance));
        ovspatialquery_raycast_hit_t        hits[2]{};
        ovspatialquery_raycast_hit_buffer_t buffer{hits, 2u};
        ovspatialquery_multiple_result_t    multiple{};
        REQUIRE(ovspatialquery_raycast_multiple_sync(instance, &ray, &buffer, &multiple).status ==
                OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(multiple.stored_hit_count >= 1u && multiple.truncated == 0u);
        for (uint64_t hit = 0u; hit < multiple.stored_hit_count; ++hit)
            REQUIRE(onSquare(row, hits[hit].data.exact.distance));
    }

    InputArray originArray(origins.data(), count, 3, kDLFloat), directionArray(directions.data(), count, 3, kDLFloat);
    InputArray maximumArray(maximums.data(), count, kDLFloat);
    for (ovspatialquery_hit_mode_t mode :
         {OVSPATIALQUERY_HIT_MODE_CLOSEST, OVSPATIALQUERY_HIT_MODE_ANY, OVSPATIALQUERY_HIT_MODE_MULTIPLE})
    {
        const uint64_t                     perQuery = mode == OVSPATIALQUERY_HIT_MODE_MULTIPLE ? 2u : 1u;
        ovspatialquery_raycast_plan_desc_t desc{};
        desc.target                             = populationRef(population);
        desc.hit_mode                           = mode;
        desc.precision                          = OVSPATIALQUERY_QUERY_EXACT;
        desc.device                             = {kDLCPU, 0};
        desc.maximum_query_count                = count;
        desc.maximum_hits_per_query             = perQuery;
        desc.hit_capacity_per_slot              = count * perQuery;
        desc.result_slot_count                  = 1u;
        desc.result_fields                      = OVSPATIALQUERY_BATCH_RESULT_DISTANCE;
        ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
        REQUIRE(ovspatialquery_create_raycast_plan(instance, &desc, &plan).status == OVSPATIALQUERY_API_SUCCESS);
        ovspatialquery_raycast_batch_t batch{};
        batch.ordinal.kind    = OVSPATIALQUERY_ORDINAL_HEAD;
        batch.query_count     = count;
        batch.origins         = &originArray.value;
        batch.unit_directions = &directionArray.value;
        batch.max_distances   = &maximumArray.value;
        ovspatialquery_raycast_batch_reservation_t reservation{};
        auto admission = ovspatialquery_enqueue_raycast_plan(instance, plan, &batch, &reservation);
        REQUIRE(admission.status == OVSPATIALQUERY_API_SUCCESS);
        ovspatialquery_op_wait_result_t waited{};
        REQUIRE(ovspatialquery_wait_op(instance, admission.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status ==
                OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(waited.state == OVSPATIALQUERY_OP_COMPLETED);
        ovspatialquery_raycast_batch_result_t result{};
        REQUIRE(ovspatialquery_fetch_raycast_batch(instance, admission.op_id, &result).status ==
                OVSPATIALQUERY_API_SUCCESS);
        const auto* offsets   = static_cast<const uint64_t*>(result.columns.hit_ranges.hit_offsets.data);
        const auto* counts    = static_cast<const uint64_t*>(result.columns.hit_ranges.hit_counts.data);
        const auto* distances = static_cast<const double*>(result.columns.contact.exact.distances.data);
        for (uint64_t row = 0u; row < count; ++row)
        {
            REQUIRE(counts[row] >= 1u && counts[row] <= perQuery);
            for (uint64_t hit = 0u; hit < counts[row]; ++hit)
                REQUIRE(onSquare(row, distances[offsets[row] + hit]));
        }
        REQUIRE(ovspatialquery_destroy_op(instance, &admission.op_id).status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(ovspatialquery_destroy_batch_plan(instance, plan).status == OVSPATIALQUERY_API_SUCCESS);
    }

    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(
        ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, definitionValue})
            .status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

} // namespace

int main()
{
    testColdMeshBatchUnderTightBudgets();
    testTriangleRaycastPreservesDoublePoseAndSourceFeature();
    testTriangleAnyIgnoresRoundedRangeOnlyHit();
    testTriangleRaycastClipsFaceBehindOrigin();
    testTriangleRaycastClipsFaceBeyondShortRange();
    testTriangleRaycastFromFarAway();
    testTriangleRaycastLongRayAtTinyMesh();
    testTriangleRaycastRefinedCandidateKeepsOrder();
    testTriangleRaycastGrazingFacesKeepExactOrder();
    testSphereSweepsMeetFarFaces();
    testTriangleRaycastHitsSharedEdges();

    const ovspatialquery_instance_desc_t instanceDesc{4u, 1u, UINT64_C(16) * 1024u * 1024u, nullptr, 0u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&instanceDesc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    registerBoxShape(instance);
    registerSphereShape(instance);

    // Population A holds two coincident boxes (an exact tie); population B's far
    // box proves target selection reaches the right population.
    const std::vector<Instance>              boxesA{{50u, {6.0, 0.0, 0.0}}, {100u, {6.0, 0.0, 0.0}}};
    const ovspatialquery_population_handle_t populationA =
        buildPopulation(instance, 1, OVSPATIALQUERY_GEOMETRY_BOX, kBoxDefinition, boxesA);
    const std::vector<Instance>              boxesB{{200u, {12.0, 0.0, 0.0}}};
    const ovspatialquery_population_handle_t populationB =
        buildPopulation(instance, 7, OVSPATIALQUERY_GEOMETRY_BOX, kBoxDefinition, boxesB);

    const double origin[3]   = {0.0, 0.0, 0.0};
    const float  forward[3]  = {1.0f, 0.0f, 0.0f};
    const float  backward[3] = {-1.0f, 0.0f, 0.0f};

    // Baseline hit: the tied pair's nearest surface.
    ovspatialquery_raycast_desc_t rayTie = makeRay(populationA, origin, forward, 0.0, 20.0);
    expectRay(instance, 1, rayTie, expectedRayHit(origin, forward, 0.0, 20.0, boxesA));

    // A generous min_distance below the entry point does not change the hit.
    ovspatialquery_raycast_desc_t ray = rayTie;
    ray.ray.min_distance              = 2.0;
    expectRay(instance, 1, ray, expectedRayHit(origin, forward, 2.0, 20.0, boxesA));

    // Origin already inside the box: the oracle clamps entry to min_distance.
    ray                          = rayTie;
    ray.ray.origin[0]            = 5.25;
    const double insideOrigin[3] = {5.25, 0.0, 0.0};
    expectRay(instance, 1, ray, expectedRayHit(insideOrigin, forward, 0.0, 20.0, boxesA));

    // Perpendicular offset misses entirely.
    ray                          = rayTie;
    ray.ray.origin[1]            = 3.0;
    const double offsetOrigin[3] = {0.0, 3.0, 0.0};
    expectRay(instance, 1, ray, expectedRayHit(offsetOrigin, forward, 0.0, 20.0, boxesA));

    // A max_distance short of the entry point misses.
    ray                  = rayTie;
    ray.ray.max_distance = 3.0;
    expectRay(instance, 1, ray, expectedRayHit(origin, forward, 0.0, 3.0, boxesA));

    // min_distance == max_distance exactly at the entry point still hits.
    ray                  = rayTie;
    ray.ray.min_distance = 5.0;
    ray.ray.max_distance = 5.0;
    expectRay(instance, 1, ray, expectedRayHit(origin, forward, 5.0, 5.0, boxesA));

    // Selecting population B reaches its own object at its own ordinal.
    ray = makeRay(populationB, origin, forward, 0.0, 20.0);
    expectRay(instance, 7, ray, expectedRayHit(origin, forward, 0.0, 20.0, boxesB));

    // Reversed direction from beyond the box finds the far-side entry.
    const double farOrigin[3] = {20.0, 0.0, 0.0};
    ray                       = makeRay(populationA, farOrigin, backward, 0.0, 20.0);
    expectRay(instance, 1, ray, expectedRayHit(farOrigin, backward, 0.0, 20.0, boxesA));

    // Non-unit direction and an inverted distance range are both rejected
    // with the caller's output untouched.
    ovspatialquery_raycast_desc_t invalidDirection = rayTie;
    invalidDirection.ray.unit_direction[0]         = 2.0f;
    expectRejected<ovspatialquery_raycast_single_result_t>(
        instance, invalidDirection, ovspatialquery_raycast_closest_sync, OVSPATIALQUERY_API_INVALID_ARGUMENT);
    ovspatialquery_raycast_desc_t invalidRange = rayTie;
    invalidRange.ray.min_distance              = 3.0;
    invalidRange.ray.max_distance              = 2.0;
    expectRejected<ovspatialquery_raycast_single_result_t>(instance, invalidRange, ovspatialquery_raycast_closest_sync,
                                                           OVSPATIALQUERY_API_INVALID_ARGUMENT);

    // Sweep coverage mirrors the raycast coverage using stored spheres.
    const std::vector<Instance>              spheresA{{50u, {6.0, 0.0, 0.0}}, {100u, {6.0, 0.0, 0.0}}};
    const ovspatialquery_population_handle_t sweepPopulationA =
        buildPopulation(instance, 1, OVSPATIALQUERY_GEOMETRY_SPHERE, kSphereDefinition, spheresA);
    const std::vector<Instance>              spheresB{{200u, {12.0, 0.0, 0.0}}};
    const ovspatialquery_population_handle_t sweepPopulationB =
        buildPopulation(instance, 7, OVSPATIALQUERY_GEOMETRY_SPHERE, kSphereDefinition, spheresB);

    ovspatialquery_sphere_sweep_desc_t sweepTie = makeSweep(sweepPopulationA, origin, 1.0, forward, 0.0, 20.0);
    expectSweep(instance, 1, sweepTie, expectedSweepHit(origin, 1.0, forward, 0.0, 20.0, spheresA));

    ovspatialquery_sphere_sweep_desc_t sweep = sweepTie;
    sweep.sweep.min_distance                 = 2.0;
    expectSweep(instance, 1, sweep, expectedSweepHit(origin, 1.0, forward, 2.0, 20.0, spheresA));

    sweep                             = sweepTie;
    sweep.sweep.center[1]             = 3.0;
    const double sweepOffsetOrigin[3] = {0.0, 3.0, 0.0};
    expectSweep(instance, 1, sweep, expectedSweepHit(sweepOffsetOrigin, 1.0, forward, 0.0, 20.0, spheresA));

    sweep                    = sweepTie;
    sweep.sweep.max_distance = 2.0;
    expectSweep(instance, 1, sweep, expectedSweepHit(origin, 1.0, forward, 0.0, 2.0, spheresA));

    sweep = makeSweep(sweepPopulationB, origin, 1.0, forward, 0.0, 20.0);
    expectSweep(instance, 7, sweep, expectedSweepHit(origin, 1.0, forward, 0.0, 20.0, spheresB));

    sweep = makeSweep(sweepPopulationA, farOrigin, 1.0, backward, 0.0, 20.0);
    expectSweep(instance, 1, sweep, expectedSweepHit(farOrigin, 1.0, backward, 0.0, 20.0, spheresA));

    ovspatialquery_sphere_sweep_desc_t degenerate = sweepTie;
    degenerate.sweep.radius                       = 0.0;
    expectRejected<ovspatialquery_sphere_sweep_closest_result_t>(
        instance, degenerate, ovspatialquery_sweep_sphere_closest_sync, OVSPATIALQUERY_API_INVALID_ARGUMENT);

    testLongSweepsThroughDensePopulations(instance);

    // A sphere whose grown bounds the sweep enters first (at 2.5) is met at
    // about 3.62; one entered later (at 3.1) is met at 3.1 and must still be
    // evaluated up to the first one's contact.
    const std::vector<Instance>              staggered{{1u, {4.0, 1.45, 0.0}}, {2u, {4.6, 0.0, 0.0}}};
    const ovspatialquery_population_handle_t staggeredPopulation =
        buildPopulation(instance, 6, OVSPATIALQUERY_GEOMETRY_SPHERE, kSphereDefinition, staggered);
    expectSweep(instance, 6, makeSweep(staggeredPopulation, origin, 0.5, forward, 0.0, 20.0),
                expectedSweepHit(origin, 0.5, forward, 0.0, 20.0, staggered));
    REQUIRE(ovspatialquery_destroy_population(instance, staggeredPopulation).status == OVSPATIALQUERY_API_SUCCESS);

    REQUIRE(ovspatialquery_destroy_population(instance, populationA).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_population(instance, populationB).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_population(instance, sweepPopulationA).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_population(instance, sweepPopulationB).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(
        ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_BOX, kBoxDefinition}).status ==
        OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_SPHERE, kSphereDefinition})
                .status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
    std::puts("closest query oracle differential test passed");
    return 0;
}
