/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../../source/ovspatialquery_backend.h"
#include "../../../source/ovspatialquery_instance_host_ledger.h"
#include "../../../source/ovspatialquery_population_table.h"
#include "../../../source/ovspatialquery_typed_geometry.h"

#include "../../support/native/physx_convex_oracle.h"
#include "../../support/native/public_api_inputs.h"
#include "../../support/native/test_failure.h"
#include <ovspatialquery/ovspatialquery_explicit.h>
#include <ovspatialquery/ovspatialquery_pairs.h>

#include <PxPhysicsAPI.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// Low-level PhysX and typed-geometry checks, then population and pair-snapshot
// checks through the public API, with registered box definitions placed by
// identity-rotation instance poses.

namespace
{

void testPackagedPhysXConvexPreparationLimitsAndAtomicity()
{
    using namespace ovspatialquery::detail;
    ovspatialquery_backend_runtime_t* runtime = nullptr;
    REQUIRE(acquireDefaultBackendRuntime(&runtime) == OVSPATIALQUERY_API_SUCCESS);
    const std::vector<float> validPoints = {
        -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0,
        -1.0, -1.0, 1.0,  1.0, -1.0, 1.0,  1.0, 1.0, 1.0,  -1.0, 1.0, 1.0,
    };
    std::shared_ptr<const PreparedConvexHull> preserved;
    REQUIRE(prepareConvexHull(runtime, validPoints, UINT64_MAX, preserved) == ConvexHullPreparationOutcome::success);
    REQUIRE(preserved != nullptr && preserved->residentByteCount() > 0u);
    std::shared_ptr<const PreparedConvexHull> output = preserved;

    std::vector<float> malformed = validPoints;
    malformed[0]                 = std::numeric_limits<float>::quiet_NaN();
    REQUIRE(prepareConvexHull(runtime, malformed, UINT64_MAX, output) == ConvexHullPreparationOutcome::invalidArgument);
    REQUIRE(output == preserved);
    malformed = {
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0,
    };
    REQUIRE(prepareConvexHull(runtime, malformed, UINT64_MAX, output) == ConvexHullPreparationOutcome::invalidArgument);
    REQUIRE(output == preserved);
    REQUIRE(prepareConvexHull(runtime, validPoints, preserved->residentByteCount() - 1u, output) ==
            ConvexHullPreparationOutcome::resourceExhausted);
    REQUIRE(output == preserved);
    std::shared_ptr<const PreparedConvexHull> exactBudgetHull;
    REQUIRE(prepareConvexHull(runtime, validPoints, preserved->residentByteCount(), exactBudgetHull) ==
            ConvexHullPreparationOutcome::success);
    REQUIRE(exactBudgetHull != nullptr && exactBudgetHull->residentByteCount() <= preserved->residentByteCount());

    // 129 points on a sphere, the most whose hull always fits 255 faces (at most 2V - 4).
    constexpr uint32_t kDensePointCount = 129u;
    constexpr double   kGoldenAngle     = 2.3999632297286533222;
    std::vector<float> densePoints;
    densePoints.reserve(kDensePointCount * 3u);
    for (uint32_t index = 0u; index < kDensePointCount; ++index)
    {
        const double z      = 1.0 - 2.0 * (static_cast<double>(index) + 0.5) / static_cast<double>(kDensePointCount);
        const double radius = std::sqrt(std::max(0.0, 1.0 - z * z));
        const double angle  = kGoldenAngle * static_cast<double>(index);
        densePoints.push_back(static_cast<float>(radius * std::cos(angle)));
        densePoints.push_back(static_cast<float>(radius * std::sin(angle)));
        densePoints.push_back(static_cast<float>(z));
    }
    std::shared_ptr<const PreparedConvexHull> denseHull;
    REQUIRE(prepareConvexHull(runtime, densePoints, UINT64_MAX, denseHull) == ConvexHullPreparationOutcome::success);
    REQUIRE(denseHull != nullptr && denseHull->vertexCount() == kDensePointCount);
    std::shared_ptr<const PreparedConvexHull> exactDenseBudgetHull;
    REQUIRE(prepareConvexHull(runtime, densePoints, denseHull->residentByteCount(), exactDenseBudgetHull) ==
            ConvexHullPreparationOutcome::success);
    REQUIRE(exactDenseBudgetHull != nullptr &&
            exactDenseBudgetHull->residentByteCount() == denseHull->residentByteCount() &&
            exactDenseBudgetHull->vertices() == denseHull->vertices() &&
            exactDenseBudgetHull->faceVertexCounts() == denseHull->faceVertexCounts() &&
            exactDenseBudgetHull->faceVertexIndices() == denseHull->faceVertexIndices());
    const std::array<double, 16u> identity = {
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    };
    std::shared_ptr<const ExactGeometry> geometry;
    REQUIRE(createConvexExactGeometry(denseHull, identity, geometry) == ConvexConstructionOutcome::success);
    for (uint32_t index = 0u; index < kDensePointCount; ++index)
    {
        ovspatialquery_overlap_shape_t overlap{};
        overlap.kind                  = OVSPATIALQUERY_OVERLAP_SPHERE;
        overlap.data.sphere.center[0] = densePoints[index * 3u];
        overlap.data.sphere.center[1] = densePoints[index * 3u + 1u];
        overlap.data.sphere.center[2] = densePoints[index * 3u + 2u];
        overlap.data.sphere.radius    = 1.0e-9;
        REQUIRE(geometry->overlaps(overlap));
    }

    constexpr size_t                                       kConcurrentCount = 8u;
    std::vector<std::shared_ptr<const PreparedConvexHull>> concurrent(kConcurrentCount);
    std::vector<ConvexHullPreparationOutcome> outcomes(kConcurrentCount, ConvexHullPreparationOutcome::internalFailure);
    std::vector<std::thread>                  workers;
    workers.reserve(kConcurrentCount);
    for (size_t index = 0u; index < kConcurrentCount; ++index)
    {
        workers.emplace_back(
            [&, index]() { outcomes[index] = prepareConvexHull(runtime, validPoints, UINT64_MAX, concurrent[index]); });
    }
    for (std::thread& worker : workers)
    {
        worker.join();
    }
    for (size_t index = 0u; index < kConcurrentCount; ++index)
    {
        REQUIRE(outcomes[index] == ConvexHullPreparationOutcome::success && concurrent[index] != nullptr &&
                concurrent[index]->vertices() == concurrent[0]->vertices() &&
                concurrent[index]->faceVertexCounts() == concurrent[0]->faceVertexCounts() &&
                concurrent[index]->faceVertexIndices() == concurrent[0]->faceVertexIndices());
    }
    releaseDefaultBackendRuntime(runtime);
}

// Points on an ellipsoid and tapered prisms make PhysX merge nearly coplanar hull
// triangles into polygons whose corners are coplanar only to float precision. Every set
// must keep its cooked hull (ADR-0030), with each authored point within the accepted
// polygon tolerance, 2^-20 of the hull's half extent, of that hull.
void testPackagedPhysXConvexRegistrationAcceptsMergedCookerPolygons()
{
    std::vector<std::vector<float>> pointSets;
    uint64_t                        state   = UINT64_C(0x9E3779B97F4A7C15);
    auto                            uniform = [&state]()
    {
        state = state * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
        return static_cast<double>(state >> 11u) * (1.0 / 9007199254740992.0);
    };
    for (int cloud = 0; cloud < 8; ++cloud)
    {
        for (int count : {12, 16, 24})
        {
            std::vector<float> points;
            for (int point = 0; point < count; ++point)
            {
                const double z     = 2.0 * uniform() - 1.0;
                const double angle = 6.283185307179586 * uniform();
                const double ring  = std::sqrt(1.0 - z * z);
                points.insert(points.end(),
                              {static_cast<float>(0.45 * ring * std::cos(angle)),
                               static_cast<float>(0.35 * ring * std::sin(angle)), static_cast<float>(0.25 * z)});
            }
            pointSets.push_back(std::move(points));
        }
    }
    for (int sides : {4, 6, 8, 16})
    {
        for (double topRadius : {0.45, 0.3})
        {
            std::vector<float> prism;
            for (double z : {-0.3, 0.3})
            {
                const double radius = z < 0.0 ? 0.45 : topRadius;
                for (int side = 0; side < sides; ++side)
                {
                    const double angle = side * 6.283185307179586 / sides + 0.1;
                    prism.insert(prism.end(), {static_cast<float>(radius * std::cos(angle)),
                                               static_cast<float>(radius * std::sin(angle)), static_cast<float>(z)});
                }
            }
            pointSets.push_back(std::move(prism));
        }
    }

    ovspatialquery_backend_runtime_t* runtime = nullptr;
    REQUIRE(ovspatialquery::detail::acquireDefaultBackendRuntime(&runtime) == OVSPATIALQUERY_API_SUCCESS);
    const std::array<double, 16u> identity = {
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    };
    for (const std::vector<float>& points : pointSets)
    {
        std::shared_ptr<const ovspatialquery::detail::PreparedConvexHull> hull;
        REQUIRE(
            ovspatialquery::detail::prepareConvexHull(runtime, points, std::numeric_limits<uint64_t>::max(), hull) ==
            ovspatialquery::detail::ConvexHullPreparationOutcome::success);
        REQUIRE(hull != nullptr && hull->cookedMesh() != nullptr);
        std::shared_ptr<const ovspatialquery::detail::ExactGeometry> geometry;
        REQUIRE(ovspatialquery::detail::createConvexExactGeometry(hull, identity, geometry) ==
                ovspatialquery::detail::ConvexConstructionOutcome::success);
        ovspatialquery::detail::ExactGeometryDescription description;
        REQUIRE(geometry->exportDescription(description));
        const auto& convex = std::get<ovspatialquery::detail::ConvexExactGeometryDescription>(description);
        REQUIRE(convex.planeCount >= 4u);
        double halfExtent = 0.0;
        for (uint32_t axis = 0u; axis < 3u; ++axis)
        {
            float minimum = points[axis], maximum = points[axis];
            for (size_t index = axis; index < points.size(); index += 3u)
            {
                minimum = std::min(minimum, points[index]);
                maximum = std::max(maximum, points[index]);
            }
            halfExtent = std::max(halfExtent, (double(maximum) - double(minimum)) * 0.5);
        }
        for (size_t index = 0u; index < points.size(); index += 3u)
            for (uint64_t plane = 0u; plane < convex.planeCount; ++plane)
            {
                const auto& face   = convex.planes[plane];
                double      height = -face.offset;
                for (uint32_t axis = 0u; axis < 3u; ++axis)
                    height += face.normal[axis] * double(points[index + axis]);
                REQUIRE(height <= std::ldexp(halfExtent, -20));
            }
    }
    ovspatialquery::detail::releaseDefaultBackendRuntime(runtime);

    const ovspatialquery_instance_desc_t desc{2u, 1u, UINT64_C(64) * 1024u * 1024u, nullptr, 0u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&desc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    uint64_t definitionValue = 1u;
    for (std::vector<float>& points : pointSets)
    {
        uint64_t   vertexOffset = 0u;
        uint64_t   vertexCount  = points.size() / 3u;
        InputArray values(&definitionValue, 1, kDLUInt);
        InputArray offsets(&vertexOffset, 1, kDLUInt);
        InputArray counts(&vertexCount, 1, kDLUInt);
        InputArray vertices(points.data(), static_cast<int64_t>(vertexCount), 3, kDLFloat);
        const ovspatialquery_convex_definition_table_t definitions{
            &values.value, &offsets.value, &counts.value, nullptr, nullptr, &vertices.value,
            nullptr,       nullptr,        nullptr,       0.0,     0u};
        waitAndDestroy(instance, ovspatialquery_enqueue_convex_registration(instance, &definitions));
        REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_CONVEX, definitionValue})
                    .status == OVSPATIALQUERY_API_SUCCESS);
        ++definitionValue;
    }
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

void testPackagedPhysXConvexOverlapParityAndLifetime()
{
    using namespace ovspatialquery::detail;

    ovspatialquery_backend_runtime_t* runtime = nullptr;
    REQUIRE(ovspatialquery::detail::acquireDefaultBackendRuntime(&runtime) == OVSPATIALQUERY_API_SUCCESS);
    // An interior point and a repeated corner: preparation accepts both and keeps the eight corners.
    const std::vector<float> cubePoints{
        -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f,  1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f,
        1.0f,  -1.0f, 1.0f,  1.0f, 1.0f,  1.0f,  -1.0f, 1.0f, 1.0f,  0.0f,  0.0f, 0.0f,  -1.0f, -1.0f, -1.0f,
    };
    InstanceHostLedger                        sharedSurfaceLedger(UINT64_MAX);
    std::shared_ptr<const PreparedConvexHull> hull;
    REQUIRE(prepareConvexHull(runtime, cubePoints, std::numeric_limits<uint64_t>::max(), hull) ==
            ConvexHullPreparationOutcome::success);
    REQUIRE(hull != nullptr);
    REQUIRE(hull->vertexCount() == 8u);
    REQUIRE(hull->cookedMesh() != nullptr);
    REQUIRE(hull->preparedSurface() != nullptr);
    REQUIRE(hull->preparedSurface()->mesh != nullptr);
    // The same cube away from the origin cooks about its centre (ADR-0076), so its overlaps
    // below check that the PhysX test places the query in the cooked mesh's frame.
    std::vector<float> offsetCubePoints = cubePoints;
    for (size_t index = 0u; index < offsetCubePoints.size(); ++index)
        offsetCubePoints[index] += index % 3u == 1u ? -100.0f : 100.0f;
    std::shared_ptr<const PreparedConvexHull> offsetHull;
    REQUIRE(prepareConvexHull(runtime, offsetCubePoints, std::numeric_limits<uint64_t>::max(), offsetHull) ==
            ConvexHullPreparationOutcome::success);
    REQUIRE(offsetHull->cookedMesh() != nullptr);

    // The retained cooked mesh owns the PhysX infrastructure it needs. Exact
    // geometry created after the runtime handle is destroyed must remain usable.
    ovspatialquery::detail::releaseDefaultBackendRuntime(runtime);
    runtime = nullptr;

    const auto makeGeometry = [&](const std::array<double, 16u>& worldFromLocal)
    {
        std::shared_ptr<const ExactGeometry> geometry;
        const auto                           outcome = createConvexExactGeometry(hull, worldFromLocal, geometry);
        if (outcome != ConvexConstructionOutcome::success)
            std::fprintf(stderr, "convex construction outcome=%d translation=%g,%g,%g\n", int(outcome),
                         worldFromLocal[3], worldFromLocal[7], worldFromLocal[11]);
        REQUIRE(outcome == ConvexConstructionOutcome::success);
        REQUIRE(geometry != nullptr);
        return geometry;
    };

    const auto requireTouchAndNearMiss =
        [&](const std::shared_ptr<const ExactGeometry>& geometry, const double center[3], double halfExtent)
    {
        ovspatialquery_overlap_shape_t shape{};
        shape.kind                  = OVSPATIALQUERY_OVERLAP_SPHERE;
        shape.data.sphere.center[0] = center[0] - halfExtent - 0.5;
        shape.data.sphere.center[1] = center[1];
        shape.data.sphere.center[2] = center[2];
        shape.data.sphere.radius    = 0.5;
        REQUIRE(geometry->overlaps(shape));
        shape.data.sphere.center[0] -= 0.001;
        REQUIRE(!geometry->overlaps(shape));

        shape                           = {};
        shape.kind                      = OVSPATIALQUERY_OVERLAP_BOX;
        shape.data.box.center[0]        = center[0] - halfExtent - 0.5;
        shape.data.box.center[1]        = center[1];
        shape.data.box.center[2]        = center[2];
        shape.data.box.rotation_xyzw[3] = 1.0f;
        shape.data.box.half_extents[0]  = 0.5;
        shape.data.box.half_extents[1]  = 0.25;
        shape.data.box.half_extents[2]  = 0.25;
        REQUIRE(geometry->overlaps(shape));
        shape.data.box.center[0] -= 0.001;
        REQUIRE(!geometry->overlaps(shape));

        shape                         = {};
        shape.kind                    = OVSPATIALQUERY_OVERLAP_AABB;
        shape.data.aabb.min_corner[0] = center[0] - halfExtent - 1.0;
        shape.data.aabb.min_corner[1] = center[1] - 0.25;
        shape.data.aabb.min_corner[2] = center[2] - 0.25;
        shape.data.aabb.max_corner[0] = center[0] - halfExtent;
        shape.data.aabb.max_corner[1] = center[1] + 0.25;
        shape.data.aabb.max_corner[2] = center[2] + 0.25;
        REQUIRE(geometry->overlaps(shape));
        shape.data.aabb.max_corner[0] -= 0.001;
        REQUIRE(!geometry->overlaps(shape));
    };

    const std::array<double, 16u> identity{
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
    };
    const std::shared_ptr<const ExactGeometry> identityGeometry = makeGeometry(identity);
    const double                               identityCenter[3]{0.0, 0.0, 0.0};
    requireTouchAndNearMiss(identityGeometry, identityCenter, 1.0);
    std::shared_ptr<const ExactGeometry> offsetGeometry;
    REQUIRE(createConvexExactGeometry(offsetHull, identity, offsetGeometry) == ConvexConstructionOutcome::success);
    const double offsetCenter[3]{100.0, -100.0, 100.0};
    requireTouchAndNearMiss(offsetGeometry, offsetCenter, 1.0);

    std::shared_ptr<const ExactGeometry> rejectedGeometry          = identityGeometry;
    constexpr double                     kRelativeDeterminantFloor = 256.0 * std::numeric_limits<double>::epsilon();
    const std::array<double, 16u>        nearSingular{
        1.0, 1.0, 0.0, 0.0, 0.0, kRelativeDeterminantFloor * 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    };
    REQUIRE(createConvexExactGeometry(hull, nearSingular, rejectedGeometry) == ConvexConstructionOutcome::malformed);
    REQUIRE(rejectedGeometry == identityGeometry);

    // The linear transform is well conditioned, but the finite world-space
    // representation cannot preserve unit-sized local vertex separation.
    const std::array<double, 16u> collapsedByTranslation{
        1.0, 0.0, 0.0, 1.0e20, 0.0, 1.0, 0.0, 1.0e20, 0.0, 0.0, 1.0, 1.0e20, 0.0, 0.0, 0.0, 1.0,
    };
    std::shared_ptr<const ExactGeometry> collapsedGeometry;
    REQUIRE(createConvexExactGeometry(hull, collapsedByTranslation, collapsedGeometry) ==
            ConvexConstructionOutcome::success);
    const double     collapsedOrigin[3]{1.0e20, 1.0e20, 1.0e20};
    const float      collapsedDirection[3]{1, 0, 0};
    ExactGeometryHit collapsedHit{};
    REQUIRE(collapsedGeometry->raycast(collapsedOrigin, collapsedDirection, 0, 10, false, collapsedHit));
    REQUIRE(collapsedHit.distance == 0);
    for (size_t axis = 0; axis < 3; ++axis)
        REQUIRE(collapsedHit.position[axis] == collapsedOrigin[axis]);

    const std::array<double, 16u> translatedSurface{
        1, 0, 0, 10, 0, 1, 0, -3, 0, 0, 1, 4, 0, 0, 0, 1,
    };
    const std::shared_ptr<const ExactGeometry> translatedGeometry = makeGeometry(translatedSurface);
    const float                                sweepDirection[3]{1.0f, 0.0f, 0.0f};
    const double                               identitySweepCenter[3]{-3.0, 0.0, 0.0};
    ExactGeometryHit                           identitySweepHit{};
    ovspatialquery_api_status_t                identitySweepStatus = OVSPATIALQUERY_API_INTERNAL;
    REQUIRE(identityGeometry->sweepSphere(identitySweepCenter, 0.25, sweepDirection, 0.0, 10.0, identitySweepHit,
                                          &sharedSurfaceLedger, &identitySweepStatus));
    REQUIRE(identitySweepStatus == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(std::abs(identitySweepHit.distance - 1.75) < 1.0e-6);
    // The hull's charge holds its surface's query structure, so building it on the first
    // sweep charges the reader nothing (ADR-0075).
    REQUIRE(sharedSurfaceLedger.total() == 0u);

    const double                translatedSweepCenter[3]{7.0, -3.0, 4.0};
    ExactGeometryHit            translatedSweepHit{};
    ovspatialquery_api_status_t translatedSweepStatus = OVSPATIALQUERY_API_INTERNAL;
    REQUIRE(translatedGeometry->sweepSphere(translatedSweepCenter, 0.25, sweepDirection, 0.0, 10.0, translatedSweepHit,
                                            &sharedSurfaceLedger, &translatedSweepStatus));
    REQUIRE(translatedSweepStatus == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(std::abs(translatedSweepHit.distance - 1.75) < 1.0e-6);
    REQUIRE(sharedSurfaceLedger.total() == 0u);

    // A representable query along Y still resolves local thickness when X collapses.
    auto collapsedX                     = identity;
    collapsedX[3]                       = 1.0e20;
    const auto       collapsedXGeometry = makeGeometry(collapsedX);
    const double     collapsedSweepCenter[3]{1.0e20, -3, 0};
    const float      collapsedSweepDirection[3]{0, 1, 0};
    ExactGeometryHit collapsedSweepHit{};
    REQUIRE(
        collapsedXGeometry->sweepSphere(collapsedSweepCenter, .25, collapsedSweepDirection, 0, 10, collapsedSweepHit));
    REQUIRE(std::abs(collapsedSweepHit.distance - 1.75) < 1e-6);
    REQUIRE(collapsedSweepHit.position[0] == 1.0e20);
    REQUIRE(std::abs(collapsedSweepHit.position[1] + 1) < 1e-6);
    REQUIRE(
        !collapsedXGeometry->sweepSphere(collapsedSweepCenter, .25, collapsedSweepDirection, 0, 1, collapsedSweepHit));
    const double farOrigin[3]{1.0e20, -0x1p60, 0};
    REQUIRE(collapsedXGeometry->raycast(farOrigin, collapsedSweepDirection, 0, 0x1p60, false, collapsedHit));
    REQUIRE(collapsedHit.distance == 0x1p60 && collapsedHit.position[0] == 1.0e20);

    // A local radius below the local predicates' range takes the world-space
    // fallback, whose temporary contact geometry must survive a large
    // translation and is charged to the ledger only while the query runs.
    auto tinyTranslatedPose = identity;
    tinyTranslatedPose[0] = tinyTranslatedPose[5] = tinyTranslatedPose[10] = 0x1p-60;
    tinyTranslatedPose[3]                                                  = 0x1p59;
    const auto       tinyTranslated                                        = makeGeometry(tinyTranslatedPose);
    const double     tinyOrigin[3]{0x1p59, -3, 0};
    ExactGeometryHit tinyHit{};
    REQUIRE(tinyTranslated->raycast(tinyOrigin, collapsedSweepDirection, 0, 4, false, tinyHit));
    REQUIRE(tinyHit.distance == 3 && tinyHit.position[0] == 0x1p59);
    ovspatialquery_overlap_shape_t tinyShape{};
    tinyShape.kind                  = OVSPATIALQUERY_OVERLAP_SPHERE;
    tinyShape.data.sphere.center[0] = 0x1p59;
    tinyShape.data.sphere.radius    = 1e-200;
    InstanceHostLedger          tinyDenied(0), tinyAllowed(1u << 20);
    ovspatialquery_api_status_t tinyStatus = OVSPATIALQUERY_API_SUCCESS;
    REQUIRE(!tinyTranslated->overlaps(tinyShape, nullptr, &tinyDenied, &tinyStatus));
    REQUIRE(tinyStatus == OVSPATIALQUERY_API_HOST_BUDGET && tinyDenied.total() == 0);
    tinyStatus = OVSPATIALQUERY_API_SUCCESS;
    REQUIRE(tinyTranslated->overlaps(tinyShape, nullptr, &tinyAllowed, &tinyStatus));
    REQUIRE(tinyStatus == OVSPATIALQUERY_API_SUCCESS && tinyAllowed.total() == 0);
    tinyShape.data.sphere.center[1] = 3;
    REQUIRE(!tinyTranslated->overlaps(tinyShape));
    tinyShape                           = {};
    tinyShape.kind                      = OVSPATIALQUERY_OVERLAP_BOX;
    tinyShape.data.box.center[0]        = 0x1p59;
    tinyShape.data.box.rotation_xyzw[3] = 1;
    for (double& half : tinyShape.data.box.half_extents)
        half = 1e-200;
    REQUIRE(tinyTranslated->overlaps(tinyShape));
    tinyShape.data.box.center[1] = 3;
    REQUIRE(!tinyTranslated->overlaps(tinyShape));
    tinyShape                         = {};
    tinyShape.kind                    = OVSPATIALQUERY_OVERLAP_AABB;
    tinyShape.data.aabb.min_corner[0] = tinyShape.data.aabb.max_corner[0] = 0x1p59;
    REQUIRE(tinyTranslated->overlaps(tinyShape));
    tinyShape.data.aabb.min_corner[1] = tinyShape.data.aabb.max_corner[1] = 3;
    REQUIRE(!tinyTranslated->overlaps(tinyShape));

    constexpr double              largeCenter[3]{1000000000.125, -1000000000.25, 16777217.5};
    const std::array<double, 16u> rotatedUniform{
        0.0, -2.0, 0.0, largeCenter[0], 2.0, 0.0, 0.0, largeCenter[1],
        0.0, 0.0,  2.0, largeCenter[2], 0.0, 0.0, 0.0, 1.0,
    };
    const std::shared_ptr<const ExactGeometry> rotatedGeometry = makeGeometry(rotatedUniform);
    requireTouchAndNearMiss(rotatedGeometry, largeCenter, 2.0);

    // AABB queries may be points or flat slabs. PhysX box geometry rejects
    // those extents, so the exact convex predicate remains the required path.
    ovspatialquery_overlap_shape_t aabb{};
    aabb.kind                    = OVSPATIALQUERY_OVERLAP_AABB;
    aabb.data.aabb.min_corner[0] = aabb.data.aabb.max_corner[0] = -1.0;
    aabb.data.aabb.min_corner[1] = aabb.data.aabb.max_corner[1] = 0.0;
    aabb.data.aabb.min_corner[2] = aabb.data.aabb.max_corner[2] = 0.0;
    REQUIRE(identityGeometry->overlaps(aabb));
    aabb.data.aabb.min_corner[0] = aabb.data.aabb.max_corner[0] = -1.001;
    REQUIRE(!identityGeometry->overlaps(aabb));

    aabb                         = {};
    aabb.kind                    = OVSPATIALQUERY_OVERLAP_AABB;
    aabb.data.aabb.min_corner[0] = -1.0;
    aabb.data.aabb.max_corner[0] = -1.0;
    aabb.data.aabb.min_corner[1] = -0.25;
    aabb.data.aabb.max_corner[1] = 0.25;
    aabb.data.aabb.min_corner[2] = -0.25;
    aabb.data.aabb.max_corner[2] = 0.25;
    REQUIRE(identityGeometry->overlaps(aabb));
    aabb.data.aabb.min_corner[0] = aabb.data.aabb.max_corner[0] = -1.001;
    REQUIRE(!identityGeometry->overlaps(aabb));

    aabb      = {};
    aabb.kind = OVSPATIALQUERY_OVERLAP_AABB;
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        aabb.data.aabb.min_corner[axis] = -0x1p60;
        aabb.data.aabb.max_corner[axis] = 0x1p60;
    }
    REQUIRE(identityGeometry->overlaps(aabb));
}

void testPreparedConvexContactsMatchWorldConstructionAndRetainPoses()
{
    using namespace ovspatialquery::detail;
    ovspatialquery_backend_runtime_t* runtime = nullptr;
    REQUIRE(ovspatialquery::detail::acquireDefaultBackendRuntime(&runtime) == OVSPATIALQUERY_API_SUCCESS);
    {
        std::vector<float> points;
        for (float z : {-1.f, 1.f})
            for (float y : {-1.f, 1.f})
                for (float x : {-1.f, 1.f})
                    points.insert(points.end(), {x, y, z});
        std::shared_ptr<const PreparedConvexHull> hull;
        REQUIRE(prepareConvexHull(runtime, points, UINT64_MAX, hull) == ConvexHullPreparationOutcome::success);
        ConvexGeometrySource source;
        for (double value : hull->vertices())
            source.vertices.push_back(static_cast<float>(value));
        source.faceVertexCounts  = hull->faceVertexCounts();
        source.faceVertexIndices = hull->faceVertexIndices();
        const std::array<std::array<double, 16>, 3>         poses{{
            {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1},
            {0, -2, 0, 10, 2, 0, 0, -3, 0, 0, 2, 4, 0, 0, 0, 1},
            {0, -2, 0, 1000000000.125, 2, 0, 0, -1000000000.25, 0, 0, 2, 16777217.5, 0, 0, 0, 1},
        }};
        std::array<std::shared_ptr<const ExactGeometry>, 2> retainedGeometries;
        for (size_t poseIndex = 0; poseIndex < poses.size(); ++poseIndex)
        {
            const auto&  pose  = poses[poseIndex];
            const double scale = poseIndex == 0 ? 1 : 2;
            const auto   world = [&](std::array<double, 3> point)
            {
                std::array<double, 3> result{};
                for (size_t axis = 0; axis < 3; ++axis)
                    result[axis] = std::fma(pose[axis * 4], point[0],
                                            std::fma(pose[axis * 4 + 1], point[1],
                                                     std::fma(pose[axis * 4 + 2], point[2], pose[axis * 4 + 3])));
                return result;
            };
            const float direction[3]{float(pose[0] / scale), float(pose[4] / scale), float(pose[8] / scale)};
            std::shared_ptr<const ExactGeometry> prepared, raw;
            REQUIRE(createConvexExactGeometry(hull, pose, prepared) == ConvexConstructionOutcome::success);
            source.worldFromLocal = pose;
            REQUIRE(createConvexExactGeometry(source, raw) == ConvexConstructionOutcome::success);
            if (poseIndex < retainedGeometries.size())
                retainedGeometries[poseIndex] = prepared;
            const auto contact = [&](const ExactGeometryHit& actual, const ExactGeometryHit& expected)
            {
                REQUIRE(std::abs(actual.distance - expected.distance) < 2.0e-6);
                for (size_t axis = 0; axis < 3; ++axis)
                {
                    REQUIRE(std::abs(actual.position[axis] - expected.position[axis]) < 2.0e-6);
                    REQUIRE(std::abs(actual.normal[axis] - expected.normal[axis]) < 2.0e-6);
                }
                REQUIRE(!actual.hasTriangleSource && !expected.hasTriangleSource);
            };
            const auto ray = [&](std::array<double, 3> local, double minimum, double maximum, bool expectedHit,
                                 double expectedDistance)
            {
                const auto       origin = world(local);
                ExactGeometryHit a{}, b{};
                REQUIRE(prepared->raycast(origin.data(), direction, minimum * scale, maximum * scale, false, a) ==
                        expectedHit);
                REQUIRE(raw->raycast(origin.data(), direction, minimum * scale, maximum * scale, false, b) ==
                        expectedHit);
                if (expectedHit)
                {
                    contact(a, b);
                    REQUIRE(std::abs(a.distance - expectedDistance * scale) < 2.0e-6);
                }
            };
            ray({-3, .25, .125}, 0, 10, true, 2);
            ray({-3, .25, .125}, 2.25, 10, true, 2.25); // Near plane inside; surface contact remains defined.
            ray({.5, .25, .125}, 0, 10, true, 0);       // Unique nearest face, avoiding interior ties.
            ray({-3, .25, .125}, 0, 1.99, false, 0);
            ray({-3, .25, .125}, 4.25, 10, false, 0);
            ray({-3, .25, .125}, 2, 2, true, 2);
            const auto sweep = [&](std::array<double, 3> local, double radius, double minimum, double expectedDistance)
            {
                const auto                  origin = world(local);
                ExactGeometryHit            a{}, b{};
                ovspatialquery_api_status_t aStatus = OVSPATIALQUERY_API_SUCCESS, bStatus = OVSPATIALQUERY_API_SUCCESS;
                REQUIRE(prepared->sweepSphere(origin.data(), radius * scale, direction, minimum * scale, 10 * scale, a,
                                              nullptr, &aStatus));
                REQUIRE(raw->sweepSphere(origin.data(), radius * scale, direction, minimum * scale, 10 * scale, b,
                                         nullptr, &bStatus));
                REQUIRE(aStatus == OVSPATIALQUERY_API_SUCCESS && bStatus == OVSPATIALQUERY_API_SUCCESS);
                contact(a, b);
                REQUIRE(std::abs(a.distance - expectedDistance * scale) < 2.0e-6);
            };
            sweep({-3, 1.375, 0}, .625, 0, 1.5);                     // Edge contact.
            sweep({-3, 1.25, 1.375}, .625, 0, 2 - std::sqrt(.1875)); // Vertex contact.
            sweep({.5, .25, .125}, .125, 0, 0);                      // Initial overlap, with full contact comparison.
            sweep({-3, .25, .125}, .125, 2.5, 2.5);                  // Initial overlap at a nonzero near plane.
            for (double localX : {1.125, 2.0})
            {
                ovspatialquery_overlap_shape_t box{};
                box.kind          = OVSPATIALQUERY_OVERLAP_BOX;
                const auto center = world({localX, .125, .0625});
                for (size_t axis = 0; axis < 3; ++axis)
                {
                    box.data.box.center[axis]       = center[axis];
                    box.data.box.half_extents[axis] = .25 * scale;
                }
                // The cube and equal box extents permit the same world rotation at each pose.
                box.data.box.rotation_xyzw[2] = float(std::sin(.17));
                box.data.box.rotation_xyzw[3] = float(std::cos(.17));
                const bool expected           = localX < 1.5;
                REQUIRE(prepared->overlaps(box) == expected);
                REQUIRE(raw->overlaps(box) == expected);
            }
        }
        // Publication retains separate poses while both instances share the same prepared definition.
        PopulationTable  table(runtime);
        PopulationObject object{};
        object.id.value          = 71;
        object.geometryReference = {ExactGeometryFamily::convex, 901};
        object.exactGeometry     = retainedGeometries[0];
        object.bounds            = object.exactGeometry->worldBounds();
        PopulationKey population{};
        REQUIRE(table.create(2, 10, {object}, population) == OVSPATIALQUERY_API_SUCCESS);
        const ovspatialquery_ordinal_t baseOrdinal = 10;
        GenerationLease                base, replacement;
        REQUIRE(table.beginUpdate(population, 11, &baseOrdinal, base) == OVSPATIALQUERY_API_SUCCESS);
        object.exactGeometry = retainedGeometries[1];
        object.bounds        = object.exactGeometry->worldBounds();
        REQUIRE(table.buildGeneration(11, ObjectTable({object}), {}, replacement) == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(table.publishUpdate(population, replacement) == OVSPATIALQUERY_API_SUCCESS);
        table.finishUpdate(population);
        GenerationLease old, head;
        REQUIRE(table.pinExact(population, 10, old) == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(table.pin(population, head) == OVSPATIALQUERY_API_SUCCESS);
        hull.reset();
        retainedGeometries = {};
        object             = {};
        base.reset();
        replacement.reset();
        REQUIRE(old.ordinal() == 10 && head.ordinal() == 11);
        REQUIRE(old.objects()[0].id.value == 71 && head.objects()[0].id.value == 71);
        REQUIRE(old.objects()[0].geometryReference == head.objects()[0].geometryReference);
        const double     origins[2][3]{{-3, .25, .125}, {9.5, -9, 4.25}};
        const float      directions[2][3]{{1, 0, 0}, {0, 1, 0}};
        ExactGeometryHit oldHit{}, newHit{};
        REQUIRE(old.objects()[0].exactGeometry->raycast(origins[0], directions[0], 0, 20, false, oldHit));
        REQUIRE(head.objects()[0].exactGeometry->raycast(origins[1], directions[1], 0, 20, false, newHit));
        REQUIRE(std::abs(oldHit.distance - 2) < 1.0e-12);
        REQUIRE(std::abs(newHit.distance - 4) < 1.0e-12);
        REQUIRE(!head.objects()[0].exactGeometry->raycast(origins[0], directions[0], 0, 20, false, oldHit));
        std::shared_ptr<void> deferred;
        REQUIRE(table.release(population, deferred) == OVSPATIALQUERY_API_SUCCESS);
    }
    ovspatialquery::detail::releaseDefaultBackendRuntime(runtime);
}

void testPreparedConvexRetainsLocalGeometryAfterPlacement()
{
    using namespace ovspatialquery::detail;
    const std::array<double, 16> identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    const auto                   rawSource = [&](const PreparedConvexHull& hull)
    {
        ConvexGeometrySource source;
        source.vertices.reserve(hull.vertices().size());
        for (const double value : hull.vertices())
            source.vertices.push_back(static_cast<float>(value));
        source.faceVertexCounts  = hull.faceVertexCounts();
        source.faceVertexIndices = hull.faceVertexIndices();
        source.worldFromLocal    = identity;
        return source;
    };
    const auto portable = [](const std::vector<float>& points, const std::vector<uint32_t>& triangles)
    {
        std::shared_ptr<const PreparedConvexHull> hull;
        REQUIRE(createPreparedConvexHull(std::vector<double>(points.begin(), points.end()),
                                         std::vector<uint32_t>(triangles.size() / 3, 3), triangles,
                                         hull) == ConvexHullPreparationOutcome::success);
        return hull;
    };
    // All pairwise distances are ordinary, but rounding the thin axis destroys volume.
    const auto thin = portable({0, 0, 0, 1, 0, 0, 0, 1, 0, .25f, .25f, 0x1p-20f}, {0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3});
    auto       thinSource = rawSource(*thin);
    std::shared_ptr<const ExactGeometry> thinIdentity;
    REQUIRE(createConvexExactGeometry(thinSource, thinIdentity) == ConvexConstructionOutcome::success);
    auto prepared = thinIdentity, raw = thinIdentity;
    thinSource.worldFromLocal[11] = 0x1p34; // ULP is 2^-18, larger than the tetrahedron height.
    REQUIRE(createConvexExactGeometry(thin, thinSource.worldFromLocal, prepared) == ConvexConstructionOutcome::success);
    REQUIRE(createConvexExactGeometry(thinSource, raw) == ConvexConstructionOutcome::malformed);
    REQUIRE(raw == thinIdentity);
    const double     thinOrigin[3]{.25, .25, 0x1p34 + 1};
    const float      thinDirection[3]{0, 0, -1};
    ExactGeometryHit thinHit{};
    REQUIRE(prepared->raycast(thinOrigin, thinDirection, 0, 2, false, thinHit));
    REQUIRE(std::abs(thinHit.distance - (1 - 0x1p-20)) < 1e-12);
    REQUIRE(!prepared->raycast(thinOrigin, thinDirection, 0, .5, false, thinHit));

    // Coplanar vertices outside each triangle are legitimate supporting-face vertices.
    const auto cube = portable(
        {-1, -1, -1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1},
        {0, 2, 3, 0, 3, 1, 4, 5, 7, 4, 7, 6, 0, 1, 5, 0, 5, 4, 2, 6, 7, 2, 7, 3, 0, 4, 6, 0, 6, 2, 1, 3, 7, 1, 7, 5});
    ovspatialquery_backend_runtime_t* runtime = nullptr;
    REQUIRE(ovspatialquery::detail::acquireDefaultBackendRuntime(&runtime) == OVSPATIALQUERY_API_SUCCESS);
    {
        std::vector<float> points;
        for (float z : {-1.f, 1.f})
            for (size_t side = 0; side < 16; ++side)
            {
                const double angle = double(side) * std::acos(-1.0) / 8;
                points.insert(points.end(), {float(std::cos(angle)), float(std::sin(angle)), z});
            }
        std::shared_ptr<const PreparedConvexHull> prism;
        REQUIRE(prepareConvexHull(runtime, points, UINT64_MAX, prism) == ConvexHullPreparationOutcome::success);
        for (const auto& hull : {cube, prism})
        {
            size_t successes = 0;
            for (const auto& configuration :
                 std::array<std::array<double, 3>, 4>{{{0, 0, 1}, {396, .25, 1}, {100, -.47, .5}, {1024, .61, 2}}})
            {
                const double t = configuration[0], angle = configuration[1], scale = configuration[2];
                const double c = std::cos(angle), sn = std::sin(angle);
                auto         source   = rawSource(*hull);
                source.worldFromLocal = {scale * c, -scale * sn, 0,     t,   scale * sn, scale * c, 0, -t,
                                         0,         0,           scale, .25, 0,          0,         0, 1};
                prepared = raw      = thinIdentity;
                const auto expected = createConvexExactGeometry(source, raw);
                REQUIRE(createConvexExactGeometry(hull, source.worldFromLocal, prepared) ==
                        ConvexConstructionOutcome::success);
                if (expected != ConvexConstructionOutcome::success)
                {
                    REQUIRE(raw == thinIdentity);
                    continue;
                }
                ++successes;
                const double     origin[3]{t + scale * (-3 * c - .173 * sn), -t + scale * (-3 * sn + .173 * c),
                                           .25 + scale * .217};
                const float      direction[3]{float(c), float(sn), 0};
                ExactGeometryHit a{}, b{};
                REQUIRE(raw->raycast(origin, direction, 0, 10 * scale, false, b));
                REQUIRE(prepared->raycast(origin, direction, 0, 10 * scale, false, a));
                REQUIRE(std::abs(a.distance - b.distance) < 2e-6);
                for (size_t axis = 0; axis < 3; ++axis)
                {
                    REQUIRE(std::abs(a.position[axis] - b.position[axis]) < 2e-6);
                    REQUIRE(std::abs(a.normal[axis] - b.normal[axis]) < 2e-6);
                }
                REQUIRE(!a.hasTriangleSource && !b.hasTriangleSource);
                REQUIRE(!raw->raycast(origin, direction, 0, scale, false, b));
                REQUIRE(!prepared->raycast(origin, direction, 0, scale, false, a));
                ovspatialquery_overlap_shape_t sphere{};
                sphere.kind                  = OVSPATIALQUERY_OVERLAP_SPHERE;
                sphere.data.sphere.radius    = .125 * scale;
                sphere.data.sphere.center[0] = t;
                sphere.data.sphere.center[1] = -t;
                sphere.data.sphere.center[2] = .25 + 1.0625 * scale;
                REQUIRE(raw->overlaps(sphere) && prepared->overlaps(sphere));
                sphere.data.sphere.center[2] = .25 + 1.25 * scale;
                REQUIRE(!raw->overlaps(sphere) && !prepared->overlaps(sphere));
            }
            REQUIRE(successes > 1); // Exercise translated contacts as well as identity.
        }
    }
    ovspatialquery::detail::releaseDefaultBackendRuntime(runtime);
}

void testTriangleRaycastMapsFacesAcrossTheDomainRange()
{
    using namespace ovspatialquery::detail;

    TriangleMeshSource mesh{};
    mesh.key           = 9103u;
    mesh.vertices      = {1.0f, -1.0f, -1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, -1.0f};
    mesh.triangles     = {0u, 1u, 2u};
    mesh.sourceFaces   = {43u};
    mesh.sourceCorners = {7u, 8u, 9u};

    std::shared_ptr<const PreparedTriangleMesh> prepared;
    REQUIRE(prepareTriangleMesh(mesh, prepared) == OVSPATIALQUERY_API_SUCCESS);
    const std::array<double, 16u> identity{
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    };
    std::shared_ptr<const TriangleObjectGeometry> oneSided;
    std::shared_ptr<const TriangleObjectGeometry> doubleSided;
    REQUIRE(bindPreparedTriangleGeometry(prepared, identity, false, oneSided) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(bindPreparedTriangleGeometry(prepared, identity, true, doubleSided) == OVSPATIALQUERY_API_SUCCESS);

    constexpr double kMaximumDistance = 0x1p60; // the input domain's longest ray
    const double     frontOrigin[3]{0.0, 0.0, 0.0};
    const double     backOrigin[3]{2.0, 0.0, 0.0};
    const float      frontDirection[3]{1.0f, 0.0f, 0.0f};
    const float      backDirection[3]{-1.0f, 0.0f, 0.0f};

    const auto requireMappedHit =
        [&](const TriangleObjectGeometry& object, const double origin[3], const float direction[3])
    {
        bool               hasHit = false;
        PreciseTriangleHit hit{};
        REQUIRE(raycastTriangleObject(object, origin, direction, 0.0, kMaximumDistance, false, nullptr, hasHit, hit) ==
                OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(hasHit);
        REQUIRE(std::abs(hit.distance - 1.0) < 1.0e-12);
        REQUIRE(std::abs(hit.position[0] - 1.0) < 1.0e-12);
        REQUIRE(hit.generatedTriangle == 0u);
        REQUIRE(hit.hasTriangleSource);
        REQUIRE(hit.faceIndex == 43u);
        REQUIRE(hit.sourceCorners[0] == 7u);
        REQUIRE(hit.sourceCorners[1] == 8u);
        REQUIRE(hit.sourceCorners[2] == 9u);
    };

    requireMappedHit(*oneSided, frontOrigin, frontDirection);

    bool               hasHit = true;
    PreciseTriangleHit hit{};
    REQUIRE(raycastTriangleObject(*oneSided, backOrigin, backDirection, 0.0, kMaximumDistance, false, nullptr, hasHit,
                                  hit) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(!hasHit);

    requireMappedHit(*doubleSided, backOrigin, backDirection);

    const double surfaceOrigin[3]{1.0, 0.0, 0.0};
    const float  tangentDirection[3]{0.0f, 1.0f, 0.0f};
    REQUIRE(raycastTriangleObject(*doubleSided, surfaceOrigin, tangentDirection, 0.0, kMaximumDistance, false, nullptr,
                                  hasHit, hit) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(!hasHit);
}

void testTriangleCandidateOverflowCompletesOrReportsBudget()
{
    using namespace ovspatialquery::detail;

    constexpr uint32_t kTriangleCount = 300u;
    TriangleMeshSource mesh{};
    mesh.key      = 9102u;
    mesh.vertices = {1.0f, -1.0f, -1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, -1.0f};
    mesh.triangles.reserve(kTriangleCount * 3u);
    mesh.sourceFaces.reserve(kTriangleCount);
    mesh.sourceCorners.reserve(kTriangleCount * 3u);
    for (uint32_t face = 0u; face < kTriangleCount; ++face)
    {
        mesh.triangles.insert(mesh.triangles.end(), {0u, 1u, 2u});
        mesh.sourceFaces.push_back(face);
        mesh.sourceCorners.insert(mesh.sourceCorners.end(), {0u, 1u, 2u});
    }

    std::shared_ptr<const PreparedTriangleMesh> prepared;
    REQUIRE(prepareTriangleMesh(mesh, prepared) == OVSPATIALQUERY_API_SUCCESS);
    const std::array<double, 16u> identity{
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    };
    std::shared_ptr<const TriangleObjectGeometry> object;
    REQUIRE(bindPreparedTriangleGeometry(prepared, identity, true, object) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(object != nullptr);

    // Cook before applying a query-time staging budget, so each budget result
    // below isolates the complete candidate-buffer retry.
    const double       rayOrigin[3]{0.0, 0.0, 0.0};
    const float        direction[3]{1.0f, 0.0f, 0.0f};
    bool               rayHit = false;
    PreciseTriangleHit warmHit{};
    REQUIRE(raycastTriangleObject(*object, rayOrigin, direction, 0.0, 4.0, false, nullptr, rayHit, warmHit) ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(rayHit);

    const double                   sphereCenter[3]{1.0, 0.0, 0.0};
    ovspatialquery_overlap_shape_t pointAabb{};
    pointAabb.kind = OVSPATIALQUERY_OVERLAP_AABB;
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        pointAabb.data.aabb.min_corner[axis] = sphereCenter[axis];
        pointAabb.data.aabb.max_corner[axis] = sphereCenter[axis];
    }

    const auto runSweep = [&](InstanceHostLedger* ledger, ovspatialquery_api_status_t& status, PreciseTriangleHit& hit)
    {
        status = OVSPATIALQUERY_API_SUCCESS;
        hit    = {};
        return sweepSphereTriangleObject(*object, sphereCenter, 0.25, direction, 0.0, 1.0, hit, ledger, &status);
    };
    const auto runOverlap = [&](InstanceHostLedger* ledger, ovspatialquery_api_status_t& status)
    {
        status = OVSPATIALQUERY_API_SUCCESS;
        return overlapTriangleObject(*object, pointAabb, nullptr, ledger, &status);
    };

    ovspatialquery_api_status_t status = OVSPATIALQUERY_API_SUCCESS;
    PreciseTriangleHit          hit{};
    REQUIRE(runSweep(nullptr, status, hit));
    REQUIRE(status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(hit.distance == 0.0);
    REQUIRE(hit.hasTriangleSource);
    REQUIRE(hit.faceIndex == 0u);
    REQUIRE(runOverlap(nullptr, status));
    REQUIRE(status == OVSPATIALQUERY_API_SUCCESS);

    InstanceHostLedger rejected(0u);
    REQUIRE(!runSweep(&rejected, status, hit));
    REQUIRE(status == OVSPATIALQUERY_API_HOST_BUDGET);
    REQUIRE(rejected.total() == 0u);
    REQUIRE(!runOverlap(&rejected, status));
    REQUIRE(status == OVSPATIALQUERY_API_HOST_BUDGET);
    REQUIRE(rejected.total() == 0u);

    // A sweep also orders its gathered triangles by entry bound, 16 bytes each.
    constexpr uint64_t kCandidateBytes = uint64_t(kTriangleCount) * sizeof(uint32_t);
    constexpr uint64_t kOrderBytes     = uint64_t(kTriangleCount) * 16u;
    InstanceHostLedger candidatesOnly(kCandidateBytes);
    REQUIRE(!runSweep(&candidatesOnly, status, hit));
    REQUIRE(status == OVSPATIALQUERY_API_HOST_BUDGET);
    REQUIRE(candidatesOnly.total() == 0u);
    InstanceHostLedger sweepAdmitted(kCandidateBytes + kOrderBytes);
    REQUIRE(runSweep(&sweepAdmitted, status, hit));
    REQUIRE(status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(hit.distance == 0.0);
    REQUIRE(hit.hasTriangleSource);
    REQUIRE(hit.faceIndex == 0u);
    REQUIRE(sweepAdmitted.total() == 0u);
    InstanceHostLedger admitted(kCandidateBytes);
    REQUIRE(runOverlap(&admitted, status));
    REQUIRE(status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(admitted.total() == 0u);
}

// Box definitions for the pair checks below: a unit cube and one twice as long in X.
constexpr uint64_t kSquareBoxDefinition = 1u; // half extents (0.5, 0.5, 0.5)
constexpr uint64_t kWideBoxDefinition   = 2u; // half extents (1.0, 0.5, 0.5)

void registerAabbBoxes(ovspatialquery_instance_handle_t instance)
{
    uint64_t values[2]      = {kSquareBoxDefinition, kWideBoxDefinition};
    double   halfExtents[6] = {
        0.5, 0.5, 0.5, 1.0, 0.5, 0.5,
    };
    InputArray                                  valueArray(values, 2, kDLUInt);
    InputArray                                  extentArray(halfExtents, 2, 3, kDLFloat);
    const ovspatialquery_box_definition_table_t boxes{&valueArray.value, &extentArray.value};
    waitAndDestroy(instance, ovspatialquery_enqueue_box_registration(instance, &boxes));
}

ovspatialquery_population_handle_t createBoxPopulation(ovspatialquery_instance_handle_t instance,
                                                       ovspatialquery_ordinal_t ordinal, InstanceRows& rows)
{
    const ovspatialquery_population_handle_t       population = createExplicitPopulation(instance, 0u);
    const ovspatialquery_explicit_population_set_t set{ordinal, rows.table(), nullptr, 0u};
    waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set));
    return population;
}

using PairSet = std::set<std::pair<uint64_t, uint64_t>>;

PairSet snapshotPairs(ovspatialquery_instance_handle_t instance, ovspatialquery_population_handle_t first,
                      ovspatialquery_population_handle_t second, ovspatialquery_ordinal_t expectedOrdinal)
{
    ovspatialquery_pair_snapshot_desc_t desc{};
    desc.first             = populationTarget(first);
    desc.second            = populationTarget(second);
    desc.contact_tolerance = 0.0;
    std::vector<ovspatialquery_pair_t>    rows(16u);
    ovspatialquery_pair_buffer_t          buffer{rows.data(), rows.size()};
    ovspatialquery_pair_snapshot_result_t result{};
    REQUIRE(ovspatialquery_query_pairs_sync(instance, &desc, &buffer, &result).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(result.first_source_ordinal == expectedOrdinal && result.second_source_ordinal == expectedOrdinal &&
            result.truncated == 0u && result.stored_pair_count <= rows.size());
    PairSet pairs;
    for (uint64_t index = 0u; index < result.stored_pair_count; ++index)
    {
        REQUIRE(pairs.insert({rows[index].first.value, rows[index].second.value}).second);
    }
    return pairs;
}

// Collects the candidate indices a query reports.
class IndexSink final : public ovspatialquery::detail::BackendCandidateSink
{
  public:
    bool reportCandidate(uint32_t objectIndex) override
    {
        indices.push_back(objectIndex);
        return true;
    }
    ovspatialquery_api_status_t status() const override { return OVSPATIALQUERY_API_SUCCESS; }
    std::vector<uint32_t>       indices;
};

} // namespace

// The tree bounds absorb float rounding of a query origin up to the reported
// tolerance, so a float origin displaced within it still yields the object the
// exact double ray touches; far outside, the inflation does not add candidates.
void testRayOriginToleranceCoversRoundedOrigins()
{
    using ovspatialquery::detail::BackendPopulation;
    using ovspatialquery::detail::FloatRaycastQuery;
    using ovspatialquery::detail::PopulationObject;

    ovspatialquery_backend_runtime_t* runtime = nullptr;
    REQUIRE(ovspatialquery::detail::acquireDefaultBackendRuntime(&runtime) == OVSPATIALQUERY_API_SUCCESS);
    std::unique_ptr<BackendPopulation> population;
    REQUIRE(ovspatialquery::detail::createBackendPopulation(runtime, population) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(population->rayOriginTolerance() == 0.0);

    PopulationObject box{};
    box.id.value = 5u;
    box.bounds   = {{1000.0, -1.0, -1.0}, {1001.0, 1.0, 1.0}};
    REQUIRE(population->update(ovspatialquery::detail::ObjectTable(std::vector<PopulationObject>{box})) ==
            OVSPATIALQUERY_API_SUCCESS);
    const double tolerance = population->rayOriginTolerance();
    // At least one float ulp of the largest coordinate the population reaches,
    // and not so large that it would stand in for a geometric expansion.
    REQUIRE(tolerance >= std::ldexp(1001.0, -24) && tolerance <= std::ldexp(1001.0, -21));

    const auto candidates = [&](double originX)
    {
        const FloatRaycastQuery query{{static_cast<float>(originX), 0.0f, -5.0f}, {0.0f, 0.0f, 1.0f}, 10.0f};
        IndexSink               sink;
        REQUIRE(population->raycastCandidates(query, sink) == OVSPATIALQUERY_API_SUCCESS);
        return sink.indices;
    };
    REQUIRE(static_cast<float>(1000.0 - 0.5 * tolerance) < 1000.0f);
    REQUIRE(candidates(1000.5) == std::vector<uint32_t>{0u});
    REQUIRE(candidates(1000.0 - 0.5 * tolerance) == std::vector<uint32_t>{0u});
    REQUIRE(candidates(1001.0 + 0.5 * tolerance) == std::vector<uint32_t>{0u});
    REQUIRE(candidates(1000.0 - 64.0 * tolerance).empty());
    REQUIRE(candidates(1001.0 + 64.0 * tolerance).empty());

    population.reset();
    ovspatialquery::detail::releaseDefaultBackendRuntime(runtime);
}

// A population derived from a previous version (ADR-0019) answers every query
// like one rebuilt from the same objects: box, ray and sphere sweep
// candidates that, filtered by the exact double predicate, are the objects a
// scan finds; pair sets that are equal once filtered by the exact bounds; and
// a ray-origin tolerance at least as large. Chains of derivations cross a
// re-cook (more than a quarter changed), an object reaching past the cook's
// largest coordinate and an appended object; a retained version keeps
// answering for its own objects.
void testDerivedPopulationsAnswerLikeRebuilds()
{
    using ovspatialquery::detail::BackendPairSink;
    using ovspatialquery::detail::BackendPopulation;
    using ovspatialquery::detail::FloatAabb;
    using ovspatialquery::detail::FloatBoxSweepQuery;
    using ovspatialquery::detail::FloatRaycastQuery;
    using ovspatialquery::detail::FloatSphereSweepQuery;
    using ovspatialquery::detail::PopulationObject;

    class PairSetSink final : public BackendPairSink
    {
      public:
        bool reportPair(uint32_t first, uint32_t second) override
        {
            pairs.insert({first, second});
            return true;
        }
        ovspatialquery_api_status_t             status() const override { return OVSPATIALQUERY_API_SUCCESS; }
        std::set<std::pair<uint32_t, uint32_t>> pairs;
    };

    uint64_t   state  = 12345u;
    const auto random = [&state](double low, double high)
    {
        state = state * 6364136223846793005u + 1442695040888963407u;
        return low + (high - low) * static_cast<double>(state >> 11) / static_cast<double>(uint64_t(1) << 53);
    };
    const auto place = [&](PopulationObject& object, double reach)
    {
        const double size = random(0.2, 2.0);
        for (uint32_t axis = 0u; axis < 3u; ++axis)
        {
            object.bounds.minimum[axis] = random(-reach, reach);
            object.bounds.maximum[axis] = object.bounds.minimum[axis] + size;
        }
    };
    const auto overlaps = [](const ovspatialquery_aabb_t& first, const ovspatialquery_aabb_t& second)
    {
        for (uint32_t axis = 0u; axis < 3u; ++axis)
        {
            if (first.minimum[axis] > second.maximum[axis] || second.minimum[axis] > first.maximum[axis])
            {
                return false;
            }
        }
        return true;
    };
    // Whether the segment from `origin` along `direction` for `length` touches
    // `bounds` grown by `radius` on every axis and by `half[axis]` along each,
    // in double arithmetic: what the ray (radius zero), sphere sweep and box
    // sweep candidate traversals must report.
    const auto segmentTouches = [](const ovspatialquery_aabb_t& bounds, const float origin[3], const float direction[3],
                                   float length, double radius, const float* half = nullptr)
    {
        double entry = 0.0;
        double exit  = length;
        for (uint32_t axis = 0u; axis < 3u; ++axis)
        {
            const double grow    = radius + (half ? half[axis] : 0.0);
            const double minimum = bounds.minimum[axis] - grow;
            const double maximum = bounds.maximum[axis] + grow;
            const double start   = origin[axis];
            if (direction[axis] == 0.0f)
            {
                if (start < minimum || start > maximum)
                {
                    return false;
                }
                continue;
            }
            const double first  = (minimum - start) / direction[axis];
            const double second = (maximum - start) / direction[axis];
            entry               = std::max(entry, std::min(first, second));
            exit                = std::min(exit, std::max(first, second));
        }
        return entry <= exit;
    };

    ovspatialquery_backend_runtime_t* runtime = nullptr;
    REQUIRE(ovspatialquery::detail::acquireDefaultBackendRuntime(&runtime) == OVSPATIALQUERY_API_SUCCESS);
    const auto build = [&](const std::vector<PopulationObject>& objects, const BackendPopulation* base,
                           const std::vector<uint32_t>& changed)
    {
        std::unique_ptr<BackendPopulation> population;
        REQUIRE(ovspatialquery::detail::createBackendPopulation(runtime, population) == OVSPATIALQUERY_API_SUCCESS);
        const ovspatialquery::detail::ObjectTable table(objects);
        REQUIRE((base != nullptr ? population->derive(*base, table, changed) : population->update(table)) ==
                OVSPATIALQUERY_API_SUCCESS);
        return population;
    };

    std::vector<PopulationObject> objects(400u);
    for (size_t index = 0u; index < objects.size(); ++index)
    {
        objects[index].id.value = index + 1u;
        place(objects[index], 50.0);
    }
    std::vector<PopulationObject> others(120u);
    for (size_t index = 0u; index < others.size(); ++index)
    {
        others[index].id.value = index + 1u;
        place(others[index], 50.0);
    }
    const std::unique_ptr<BackendPopulation> other = build(others, nullptr, {});

    const auto requireSameAnswers = [&](const BackendPopulation& derived, const std::vector<PopulationObject>& at)
    {
        const std::unique_ptr<BackendPopulation> rebuilt = build(at, nullptr, {});
        REQUIRE(derived.rayOriginTolerance() >= rebuilt->rayOriginTolerance());
        // A traversal's candidates from the derived and the rebuilt population,
        // filtered by the exact predicate, are both what a scan of `at` finds.
        const BackendPopulation* const populations[] = {&derived, rebuilt.get()};
        const auto                     sameAsScan    = [&](const auto& traverse, const auto& touches)
        {
            std::set<uint32_t> expected;
            for (uint32_t index = 0u; index < at.size(); ++index)
            {
                if (touches(at[index].bounds))
                {
                    expected.insert(index);
                }
            }
            for (const BackendPopulation* population : populations)
            {
                IndexSink sink;
                REQUIRE(traverse(*population, sink) == OVSPATIALQUERY_API_SUCCESS);
                std::set<uint32_t> matches;
                for (uint32_t index : sink.indices)
                {
                    REQUIRE(index < at.size());
                    if (touches(at[index].bounds))
                    {
                        matches.insert(index);
                    }
                }
                if (matches != expected)
                {
                    return false;
                }
            }
            return true;
        };
        for (uint32_t query = 0u; query < 24u; ++query)
        {
            PopulationObject box{};
            place(box, 55.0);
            FloatAabb floatBox{};
            for (uint32_t axis = 0u; axis < 3u; ++axis)
            {
                floatBox.minimum[axis]   = static_cast<float>(box.bounds.minimum[axis]);
                floatBox.maximum[axis]   = static_cast<float>(box.bounds.maximum[axis]);
                box.bounds.minimum[axis] = floatBox.minimum[axis];
                box.bounds.maximum[axis] = floatBox.maximum[axis];
            }
            REQUIRE(sameAsScan([&](const BackendPopulation& population, IndexSink& sink)
                               { return population.aabbOverlapCandidates(floatBox, sink); },
                               [&](const ovspatialquery_aabb_t& bounds) { return overlaps(bounds, box.bounds); }));

            const float direction[3] = {static_cast<float>(random(-1.0, 1.0)), static_cast<float>(random(-1.0, 1.0)),
                                        0.5f};
            const float length =
                std::sqrt(direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]);
            const FloatRaycastQuery ray{
                {static_cast<float>(random(-40.0, 40.0)), static_cast<float>(random(-40.0, 40.0)), -60.0f},
                {direction[0] / length, direction[1] / length, direction[2] / length},
                200.0f};
            REQUIRE(sameAsScan([&](const BackendPopulation& population, IndexSink& sink)
                               { return population.raycastCandidates(ray, sink); },
                               [&](const ovspatialquery_aabb_t& bounds) {
                                   return segmentTouches(bounds, ray.origin, ray.unitDirection, ray.maxDistance, 0.0);
                               }));
            const FloatSphereSweepQuery sweep{{ray.origin[0], ray.origin[1], ray.origin[2]},
                                              {ray.unitDirection[0], ray.unitDirection[1], ray.unitDirection[2]},
                                              0.75f,
                                              200.0f};
            REQUIRE(sameAsScan([&](const BackendPopulation& population, IndexSink& sink)
                               { return population.sphereSweepCandidates(sweep, sink); },
                               [&](const ovspatialquery_aabb_t& bounds) {
                                   return segmentTouches(bounds, sweep.center, sweep.unitDirection, sweep.maxDistance,
                                                         sweep.radius);
                               }));
            const FloatBoxSweepQuery boxSweep{{ray.origin[0], ray.origin[1], ray.origin[2]},
                                              {0.25f, 1.5f, 0.75f},
                                              {ray.unitDirection[0], ray.unitDirection[1], ray.unitDirection[2]},
                                              200.0f};
            REQUIRE(sameAsScan([&](const BackendPopulation& population, IndexSink& sink)
                               { return population.boxSweepCandidates(boxSweep, sink); },
                               [&](const ovspatialquery_aabb_t& bounds)
                               {
                                   return segmentTouches(bounds, boxSweep.center, boxSweep.unitDirection,
                                                         boxSweep.maxDistance, 0.0, boxSweep.halfExtents);
                               }));
        }
        PairSetSink pairsDerived;
        PairSetSink pairsRebuilt;
        REQUIRE(derived.findPairs(*other, pairsDerived) == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(rebuilt->findPairs(*other, pairsRebuilt) == OVSPATIALQUERY_API_SUCCESS);
        const auto exactPairs = [&](const std::set<std::pair<uint32_t, uint32_t>>& candidates)
        {
            std::set<std::pair<uint32_t, uint32_t>> matches;
            for (const std::pair<uint32_t, uint32_t>& pair : candidates)
            {
                if (overlaps(at[pair.first].bounds, others[pair.second].bounds))
                {
                    matches.insert(pair);
                }
            }
            return matches;
        };
        REQUIRE(exactPairs(pairsDerived.pairs) == exactPairs(pairsRebuilt.pairs));
    };

    struct Version
    {
        std::vector<PopulationObject>      objects;
        std::unique_ptr<BackendPopulation> population;
    };
    std::vector<Version> versions;
    versions.push_back({objects, build(objects, nullptr, {})});
    for (uint32_t step = 1u; step <= 40u; ++step)
    {
        std::set<uint32_t> changed;
        const uint32_t     count = step % 13u == 0u ? 150u : 3u + step % 5u;
        while (changed.size() < count)
        {
            changed.insert(static_cast<uint32_t>(random(0.0, static_cast<double>(objects.size()))));
        }
        for (uint32_t position : changed)
        {
            place(objects[position], 50.0);
        }
        if (step == 17u)
        {
            place(objects[*changed.begin()], 5000.0); // beyond the cook's largest coordinate
        }
        if (step == 24u)
        {
            objects[*changed.begin()].bounds.maximum[0] = objects[*changed.begin()].bounds.minimum[0] + 1.0;
        }
        if (step == 31u)
        {
            objects.push_back(objects.back());
            objects.back().id.value = objects.size();
            changed.insert(static_cast<uint32_t>(objects.size() - 1u));
        }
        const std::vector<uint32_t> positions(changed.begin(), changed.end());
        versions.push_back({objects, build(objects, versions.back().population.get(), positions)});
        requireSameAnswers(*versions.back().population, objects);
        if (versions.size() > 3u)
        {
            requireSameAnswers(*versions[versions.size() - 3u].population, versions[versions.size() - 3u].objects);
        }
    }
    versions.clear();
    ovspatialquery::detail::releaseDefaultBackendRuntime(runtime);
}

// ADR-0076: a nearly flat point set registers as the hull of its points or is refused, never as
// another shape. The cases are 8- to 127-sided prisms of radius 1 and random points over a 2 by 2
// square, from 1e-7 to 1e-2 thick; PhysX returns a thickened box or a hull that leaves points
// outside for the thinner ones; for denser clouds up to 1e-2 thick it returns polygons whose
// corners lie off one plane, which the library splits into triangles. A kept hull must have only
// input points as vertices and enclose every point, checked against the hull's own planes as
// above. Prisms and 16 points 1e-4 thick or more, and 64 or 255 points 1e-3 thick or more, must be
// kept, so refusing everything, or refusing a polygon instead of splitting it, fails. The prisms
// and 16-point sets that must be kept must also be kept 100 from the origin on every axis, where
// cooking about the origin instead of the points' centre refuses them 1e-4 thick. (Moving a denser
// set rounds it to other floats, and 1e-3 thick its outcome depends on the points.)
void testNearlyFlatConvexIsItsPointsHullOrRefused()
{
    using namespace ovspatialquery::detail;
    struct Case
    {
        std::vector<float> points;
        bool               mustKeep;
        bool               mustKeepAway = false;
    };
    std::vector<Case> cases;
    for (int sides : {8, 32, 64, 127})
        for (double thickness : {1e-7, 1e-6, 3e-6, 1e-5, 1e-4, 1e-3})
        {
            Case prism{{}, thickness >= 1e-4, thickness >= 1e-4};
            for (double z : {0.0, thickness})
                for (int side = 0; side < sides; ++side)
                {
                    const double angle = side * 6.283185307179586 / sides;
                    prism.points.insert(prism.points.end(),
                                        {static_cast<float>(std::cos(angle)), static_cast<float>(std::sin(angle)),
                                         static_cast<float>(z)});
                }
            cases.push_back(std::move(prism));
        }
    uint64_t state   = UINT64_C(0x2545F4914F6CDD1D);
    auto     uniform = [&state]()
    {
        state = state * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
        return static_cast<double>(state >> 11u) * (1.0 / 9007199254740992.0);
    };
    for (int count : {16, 64, 255})
        for (double thickness : {1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2})
        {
            Case cloud{{}, thickness >= (count == 16 ? 1e-4 : 1e-3), count == 16 && thickness >= 1e-4};
            for (int point = 0; point < count; ++point)
                cloud.points.insert(cloud.points.end(), {static_cast<float>(2.0 * uniform() - 1.0),
                                                         static_cast<float>(2.0 * uniform() - 1.0),
                                                         static_cast<float>(thickness * uniform())});
            cases.push_back(std::move(cloud));
        }
    const size_t centred = cases.size();
    for (size_t index = 0u; index < centred; ++index)
        if (cases[index].mustKeepAway)
        {
            Case shifted = cases[index];
            for (size_t coordinate = 0u; coordinate < shifted.points.size(); ++coordinate)
                shifted.points[coordinate] =
                    static_cast<float>(double(shifted.points[coordinate]) + (coordinate % 3u == 1u ? -100.0 : 100.0));
            cases.push_back(std::move(shifted));
        }

    ovspatialquery_backend_runtime_t* runtime = nullptr;
    REQUIRE(acquireDefaultBackendRuntime(&runtime) == OVSPATIALQUERY_API_SUCCESS);
    const std::array<double, 16u> identity = {
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    };
    size_t kept = 0u, refused = 0u;
    for (const Case& flat : cases)
    {
        const std::vector<float>&                 points = flat.points;
        std::shared_ptr<const PreparedConvexHull> hull;
        const ConvexHullPreparationOutcome        outcome = prepareConvexHull(runtime, points, UINT64_MAX, hull);
        REQUIRE(outcome == ConvexHullPreparationOutcome::success ||
                (!flat.mustKeep && outcome == ConvexHullPreparationOutcome::noEnclosingHull &&
                 convexHullPreparationStatus(outcome) == OVSPATIALQUERY_API_INVALID_ARGUMENT));
        if (outcome != ConvexHullPreparationOutcome::success)
        {
            ++refused;
            continue;
        }
        ++kept;
        const std::vector<double>& vertices = hull->vertices();
        for (size_t vertex = 0u; vertex < vertices.size(); vertex += 3u)
        {
            bool authored = false;
            for (size_t index = 0u; index < points.size() && !authored; index += 3u)
                authored = double(points[index]) == vertices[vertex] &&
                           double(points[index + 1u]) == vertices[vertex + 1u] &&
                           double(points[index + 2u]) == vertices[vertex + 2u];
            REQUIRE(authored);
        }
        std::shared_ptr<const ExactGeometry> geometry;
        REQUIRE(createConvexExactGeometry(hull, identity, geometry) == ConvexConstructionOutcome::success);
        ExactGeometryDescription description;
        REQUIRE(geometry->exportDescription(description));
        const auto& convex = std::get<ConvexExactGeometryDescription>(description);
        for (size_t index = 0u; index < points.size(); index += 3u)
            for (uint64_t plane = 0u; plane < convex.planeCount; ++plane)
            {
                const auto& face   = convex.planes[plane];
                double      height = -face.offset;
                for (uint32_t axis = 0u; axis < 3u; ++axis)
                    height += face.normal[axis] * double(points[index + axis]);
                REQUIRE(height <= std::ldexp(1.0, -20));
            }
    }
    REQUIRE(kept + refused == 53u && refused != 0u);
    releaseDefaultBackendRuntime(runtime);

    // Through the public API the refusal is INVALID_ARGUMENT before admission, naming the
    // definition.
    const ovspatialquery_instance_desc_t desc{2u, 1u, UINT64_C(64) * 1024u * 1024u, nullptr, 0u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&desc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    std::vector<float>& thin            = cases.front().points;
    uint64_t            definitionValue = 7u, vertexOffset = 0u, vertexCount = thin.size() / 3u;
    InputArray          values(&definitionValue, 1, kDLUInt);
    InputArray          offsets(&vertexOffset, 1, kDLUInt);
    InputArray          counts(&vertexCount, 1, kDLUInt);
    InputArray          vertices(thin.data(), static_cast<int64_t>(vertexCount), 3, kDLFloat);
    const ovspatialquery_convex_definition_table_t definitions{
        &values.value, &offsets.value, &counts.value, nullptr, nullptr, &vertices.value,
        nullptr,       nullptr,        nullptr,       0.0,     0u};
    REQUIRE(ovspatialquery_enqueue_convex_registration(instance, &definitions).status ==
            OVSPATIALQUERY_API_INVALID_ARGUMENT);
    const ovspatialquery_string_t detail = ovspatialquery_get_last_error();
    REQUIRE(std::string(detail.ptr, detail.length).rfind("convex definition 7: no hull", 0) == 0u);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

// The collider of a larger bevelled carton: 279 float32 points in metres, as bit patterns,
// whose own hull has 137 vertices and 255 faces.
const uint32_t kCartonBeyondTheFaceLimit[] = {
    0xbeb10549u, 0xbe7ddaadu, 0x3f033b72u, 0xbeafa917u, 0xbe808f22u, 0x3f040409u, 0xbeafd7c1u, 0xbe7e3e72u, 0x3f037dbau,
    0xbeb1dc4cu, 0xbe814cbau, 0x3f01ca16u, 0xbeb14320u, 0xbe7fcd60u, 0x3f021d56u, 0xbeb0e244u, 0xbe75f153u, 0x3f0373c9u,
    0xbeb10e00u, 0xbe7d6198u, 0x3f037048u, 0xbeaf6f18u, 0xbe749089u, 0x3f04fd1bu, 0xbeaf870eu, 0xbe7d4768u, 0x3f0491d4u,
    0xbeafc7b6u, 0xbe746293u, 0x3f03dc61u, 0xbeafcdefu, 0xbe7d9301u, 0x3f03be37u, 0xbeae22a8u, 0xbe828a9du, 0x3f020b2au,
    0xbeb04ef3u, 0xbe823b26u, 0x3f01d56cu, 0xbeafa093u, 0xbe8167a6u, 0x3f036934u, 0xbeafdaadu, 0xbe801527u, 0x3f0214e8u,
    0xbeb302d3u, 0xbe7b262du, 0x3f01bad8u, 0xbeb214c0u, 0xbe77c074u, 0x3f02f283u, 0xbeb2cab7u, 0xbe7f7ec4u, 0x3f0211ddu,
    0xbeb204ffu, 0xbe7d77a3u, 0x3f02f0ceu, 0xbeb1c66fu, 0xbe7e6b68u, 0x3f02b95bu, 0xbeb2275cu, 0xbe7f9f8eu, 0x3f0216a6u,
    0xbeb18854u, 0xbe81130bu, 0x3f01e738u, 0xbeb03989u, 0xbe81d5d8u, 0x3f020241u, 0xbeaf0588u, 0xbe81b4edu, 0x3f035f76u,
    0xbeaed50eu, 0xbe809442u, 0x3f042d1cu, 0xbeaed76eu, 0xbe7d6889u, 0x3f04b5f2u, 0xbeaed389u, 0xbe749089u, 0x3f0510e3u,
    0xbeaf9e54u, 0x3e74c653u, 0x3f04f07bu, 0xbeaed389u, 0x3e74c653u, 0x3f0510e6u, 0xbeafc7b6u, 0x3e74985du, 0x3f03b86fu,
    0xbeafcdefu, 0x3e7be753u, 0x3f0386bfu, 0xbeafb8d2u, 0x3e7d7d32u, 0x3f0491d8u, 0xbeaf947au, 0x3e7c2177u, 0x3f022585u,
    0xbeafc44au, 0x3e80aa07u, 0x3f04040du, 0xbeb0e244u, 0x3e762720u, 0x3f0373cdu, 0xbeb10e00u, 0x3e7b1623u, 0x3f032c9du,
    0xbeafa218u, 0x3e828a8du, 0x3f020b2eu, 0xbeaf0588u, 0x3e81cfd1u, 0x3f035f7au, 0xbeaff1bfu, 0x3e824099u, 0x3f01d56fu,
    0xbeafb623u, 0x3e81828bu, 0x3f036938u, 0xbeb2e90fu, 0x3e78bf2au, 0x3f01badcu, 0xbeb25faeu, 0x3e7c9d84u, 0x3f020f1fu,
    0xbeb204ffu, 0x3e7b5a18u, 0x3f029d76u, 0xbeb240f6u, 0x3e7779edu, 0x3f02c02cu, 0xbeb10af6u, 0x3e7bdb93u, 0x3f0238d7u,
    0xbeaed50eu, 0x3e80af26u, 0x3f042d20u, 0xbeafd280u, 0x3e8014f3u, 0x3f01db1au, 0xbeaed76eu, 0x3e7d9e51u, 0x3f04b5f6u,
    0xbeb13c37u, 0x3e7ef67cu, 0x3f01eaecu, 0xbeaf6f18u, 0xbe74907bu, 0x3a8ab1d8u, 0xbeaed389u, 0xbe74907bu, 0x3a4644e4u,
    0xbeafb30eu, 0xbe746285u, 0x3ba38828u, 0xbeafcdefu, 0xbe7d41b1u, 0x3bbf418fu, 0xbeaf870eu, 0xbe7c9964u, 0x3b26c331u,
    0xbeafd7c1u, 0xbe7f0238u, 0x3c0adc2au, 0xbeafa917u, 0xbe801663u, 0x3b938e56u, 0xbeb0d57fu, 0xbe75f145u, 0x3bf029c0u,
    0xbeb1013cu, 0xbe7d618au, 0x3bf1eb05u, 0xbeb0f885u, 0xbe7dda9fu, 0x3c062af5u, 0xbeae22a8u, 0xbe828a96u, 0x3c3367acu,
    0xbeaf0588u, 0xbe816ccdu, 0x3bd296a9u, 0xbeb00c51u, 0xbe823b1fu, 0x3c52b849u, 0xbeb028e3u, 0xbe81bba6u, 0x3c497c0cu,
    0xbeafdaadu, 0xbe801520u, 0x3c30f828u, 0xbeb0d4f6u, 0xbe7fcd52u, 0x3c3f8af2u, 0xbeb13e00u, 0xbe811304u, 0x3c544451u,
    0xbeafa093u, 0xbe811f88u, 0x3bcdb7f1u, 0xbeb30e2eu, 0xbe7b261fu, 0x3c501e29u, 0xbeb2cab7u, 0xbe7f7eb6u, 0x3c5a567bu,
    0xbeb1f2f2u, 0xbe7d7794u, 0x3c1788f6u, 0xbeb202b3u, 0xbe77c066u, 0x3c171babu, 0xbeb1d7d3u, 0xbe7f9f80u, 0x3c490fc1u,
    0xbeb188e1u, 0xbe7e6b5au, 0x3c1ea2c6u, 0xbeaed50eu, 0xbe801b83u, 0x3b7e0ae5u, 0xbeb1dc4cu, 0xbe814cb3u, 0x3c656144u,
    0xbeaed76eu, 0xbe7cba88u, 0x3b02a46du, 0xbeaf9e54u, 0x3e74c661u, 0x3aa7151fu, 0xbeafb8d2u, 0x3e7ca37du, 0x3b1ffb38u,
    0xbeafcdefu, 0x3e7b88b6u, 0x3bda2277u, 0xbeafc7b6u, 0x3e74986cu, 0x3bbf7fdau, 0xbeafc44au, 0x3e8018a6u, 0x3b924be3u,
    0xbeb10e00u, 0x3e7b0760u, 0x3c0238c0u, 0xbeb0e66cu, 0x3e754e1au, 0x3bef6d9du, 0xbeaf947au, 0x3e7c2185u, 0x3c2cd2bdu,
    0xbeb10af6u, 0x3e7bdba1u, 0x3c27fe17u, 0xbeafa218u, 0x3e828a94u, 0x3c425b3fu, 0xbeaff1bfu, 0x3e8240a0u, 0x3c4fcabdu,
    0xbeafa093u, 0x3e81432fu, 0x3bdd00a9u, 0xbeaf0588u, 0x3e819076u, 0x3be1df76u, 0xbeafd280u, 0x3e8014fau, 0x3c6410c4u,
    0xbeb13c37u, 0x3e7ef68au, 0x3c6094c3u, 0xbeb2dcfcu, 0x3e78bf38u, 0x3c50b894u, 0xbeb20eb8u, 0x3e7779fbu, 0x3c1b7053u,
    0xbeb1e5cdu, 0x3e7b5a26u, 0x3c221926u, 0xbeb25faeu, 0x3e7c9d92u, 0x3c51afb7u, 0xbeaed50eu, 0x3e801dc5u, 0x3b7b85d3u,
    0xbeaed389u, 0x3e74c661u, 0x3a4c7c29u, 0xbeaed76eu, 0x3e7cc49cu, 0x3af7b944u, 0x3eaf7f5cu, 0xbe749089u, 0x3f04dbf7u,
    0x3eaef557u, 0xbe749089u, 0x3f050612u, 0x3eaf8803u, 0xbe746293u, 0x3f03ca9au, 0x3eafbde1u, 0xbe7de72fu, 0x3f03cdb7u,
    0x3eaf9326u, 0xbe7d4768u, 0x3f0491d4u, 0x3eafc7b6u, 0xbe7e9298u, 0x3f038d3au, 0x3eaf86b3u, 0xbe808f22u, 0x3f040409u,
    0x3eb0bb5fu, 0xbe75f153u, 0x3f037806u, 0x3eb0dd31u, 0xbe7d9b7fu, 0x3f036bb7u, 0x3eb0e2d4u, 0xbe7e2b51u, 0x3f033b72u,
    0x3eaec90au, 0xbe827fbdu, 0x3f020b2au, 0x3eaee31fu, 0xbe81b4edu, 0x3f035f76u, 0x3eaffd94u, 0xbe823b26u, 0x3f01d858u,
    0x3eb01703u, 0xbe81d5d8u, 0x3f020241u, 0x3eafb840u, 0xbe801527u, 0x3f0214e8u, 0x3eb120b1u, 0xbe7fcd60u, 0x3f021d56u,
    0x3eb165e6u, 0xbe81130bu, 0x3f01e738u, 0x3eaf7e2eu, 0xbe8167a6u, 0x3f036934u, 0x3eb2f7c5u, 0xbe7b262du, 0x3f01bad8u,
    0x3eb2a836u, 0xbe7f7ec4u, 0x3f0211ddu, 0x3eb1d0fbu, 0xbe7de50eu, 0x3f02ed3au, 0x3eb1e0b6u, 0xbe77c074u, 0x3f02eeeeu,
    0x3eb204e4u, 0xbe7fc56fu, 0x3f022621u, 0x3eb1a408u, 0xbe7e96c4u, 0x3f02b95bu, 0x3eaf0ab6u, 0xbe809442u, 0x3f0428ecu,
    0x3eb1b9cau, 0xbe814cbau, 0x3f01ca16u, 0x3eaf0469u, 0xbe7d6889u, 0x3f04b486u, 0x3eaf7bd2u, 0x3e74c653u, 0x3f04f07bu,
    0x3eaf964du, 0x3e7d7d32u, 0x3f0491d8u, 0x3eaf9c97u, 0x3e7be753u, 0x3f0397e3u, 0x3eaf966eu, 0x3e74985du, 0x3f03c993u,
    0x3eafa1d4u, 0x3e80aa07u, 0x3f04040du, 0x3eb0ca52u, 0x3e7b1623u, 0x3f03498du, 0x3eb0ca73u, 0x3e762720u, 0x3f0375a9u,
    0x3eaf71fdu, 0x3e7c2177u, 0x3f022585u, 0x3eb0e876u, 0x3e7bdb93u, 0x3f0238d7u, 0x3eaf7f9fu, 0x3e828a8du, 0x3f020b2eu,
    0x3eafcf4fu, 0x3e824099u, 0x3f01d56fu, 0x3eaf7e2eu, 0x3e81828bu, 0x3f036938u, 0x3eaee31fu, 0x3e81cfd1u, 0x3f035f7au,
    0x3eafaffdu, 0x3e8014f3u, 0x3f01db1au, 0x3eb12221u, 0x3e7f195du, 0x3f01eaecu, 0x3eb2b0b8u, 0x3e78bf2au, 0x3f0213a3u,
    0x3eb241d7u, 0x3e7779edu, 0x3f02b104u, 0x3eb1ff40u, 0x3e7b5a18u, 0x3f02a16fu, 0x3eb260c5u, 0x3e7c78b7u, 0x3f020a3du,
    0x3eaeb2a1u, 0x3e80af26u, 0x3f042d20u, 0x3eaeb10du, 0x3e74c653u, 0x3f0510e6u, 0x3eaeb4fbu, 0x3e7d9e51u, 0x3f04b5f6u,
    0x3eaf76b6u, 0xbe74907bu, 0x3a8ab1d8u, 0x3eaf0f8fu, 0xbe74907bu, 0x3a4644e4u, 0x3eaf8005u, 0xbe7b72dfu, 0x3b1b5516u,
    0x3eafab88u, 0xbe7ddaa6u, 0x3bbf418fu, 0x3eaf9b05u, 0xbe746285u, 0x3bbb4823u, 0x3eafaf54u, 0xbe800573u, 0x3b97c196u,
    0x3eafb53bu, 0xbe7f0238u, 0x3c0adc2au, 0x3eb0dec3u, 0xbe7dbd5fu, 0x3bf1eb05u, 0x3eb0cd54u, 0xbe75f145u, 0x3be75921u,
    0x3eb0d61cu, 0xbe7e31a6u, 0x3c062af5u, 0x3eae003du, 0xbe82786fu, 0x3c3367acu, 0x3eafe9cau, 0xbe824f54u, 0x3c52b849u,
    0x3eaee31fu, 0xbe816ccdu, 0x3bd296a9u, 0x3eb00680u, 0xbe81bba6u, 0x3c497c0cu, 0x3eb11b94u, 0xbe811304u, 0x3c544451u,
    0x3eb0b276u, 0xbe7fcd52u, 0x3c3f8af2u, 0x3eafb840u, 0xbe801520u, 0x3c30f828u, 0x3eaf7e2eu, 0xbe8117e4u, 0x3be0deaau,
    0x3eb2f7c5u, 0xbe7b261fu, 0x3c501e29u, 0x3eb1f0b3u, 0xbe77c066u, 0x3c150d94u, 0x3eb1d076u, 0xbe7e0369u, 0x3c1788f6u,
    0x3eb2accau, 0xbe7fa1aeu, 0x3c5a567bu, 0x3eb1b554u, 0xbe7f9f80u, 0x3c490fc1u, 0x3eb1666cu, 0xbe7e6b5au, 0x3c1ea2c6u,
    0x3eaf47cau, 0xbe801b83u, 0x3b7e0ae5u, 0x3eb1b9cau, 0xbe81611fu, 0x3c656144u, 0x3eaf29e6u, 0xbe7b98acu, 0x3add252cu,
    0x3eaf7bd2u, 0x3e74c661u, 0x3aa7151fu, 0x3eaf74dfu, 0x3e7b0129u, 0x3bd14d51u, 0x3eafab88u, 0x3e7c1192u, 0x3be965a1u,
    0x3eaf964du, 0x3e7ca37du, 0x3b1ffb38u, 0x3eafa1d4u, 0x3e8018a6u, 0x3b924be3u, 0x3eb0db9fu, 0x3e7a07fbu, 0x3bf325fdu,
    0x3eb0ebdfu, 0x3e7b0760u, 0x3bfdc366u, 0x3eb0e876u, 0x3e7bdba1u, 0x3c27fe17u, 0x3eaf71fdu, 0x3e7c2185u, 0x3c2cd2bdu,
    0x3eaf7f9fu, 0x3e828a94u, 0x3c425b3fu, 0x3eaee31fu, 0x3e819076u, 0x3be1df76u, 0x3eaf7e2eu, 0x3e81432fu, 0x3bdd00a9u,
    0x3eafcf4fu, 0x3e8240a0u, 0x3c4fcabdu, 0x3eafaffdu, 0x3e8014fau, 0x3c6410c4u, 0x3eb12a85u, 0x3e7f0efbu, 0x3c6094c3u,
    0x3eb2caaeu, 0x3e78ce94u, 0x3c50b894u, 0x3eb251f6u, 0x3e7ca740u, 0x3c51afb7u, 0x3eb1cc66u, 0x3e7b5a26u, 0x3c208ee2u,
    0x3eb206b8u, 0x3e7a5b0fu, 0x3c2009ecu, 0x3eaeb2a1u, 0x3e801dc5u, 0x3b7b85d3u, 0x3eaeb10du, 0x3e74c661u, 0x3a4c7c29u,
    0x3eaeb4fbu, 0x3e7cc49cu, 0x3af7b944u, 0xbeaea61fu, 0x3d2e54c3u, 0x3f04ffc9u, 0xbeaed126u, 0x3d2e54c4u, 0x3f03e2c0u,
    0xbeb0ccccu, 0x3d2e54c4u, 0x3f037f7du, 0xbeb2f1a8u, 0x3d2e54c4u, 0x3f01ec03u, 0xbeb22621u, 0x3d2e54c4u, 0x3f02d6fdu,
    0xbeae2d0du, 0x3d2e54c3u, 0x3f05294eu, 0xbeb1f31au, 0x3d2e54fau, 0x3c17875bu, 0xbeb2b77cu, 0x3d2e54fau, 0x3c4afbd1u,
    0xbeb04a78u, 0x3d2e54fbu, 0x3bcd01bbu, 0xbeae2d0du, 0x3d2e54fbu, 0x3a1bb4a0u, 0xbeaea61fu, 0x3d2e54fbu, 0x3aa0d6f4u,
    0xbeaed126u, 0x3d2e54fbu, 0x3bafeeaeu, 0x3eb20394u, 0x3d3078a5u, 0x3f02d7d1u, 0x3eb2cc61u, 0x3d2f922au, 0x3f01fbbcu,
    0x3eb05f1au, 0x3d31ad25u, 0x3f03a13bu, 0x3eac75b6u, 0x3d3347f6u, 0x3f05294eu, 0x3eacef36u, 0x3d33b22eu, 0x3f04ffc9u,
    0x3ead348du, 0x3d331891u, 0x3f0409a0u, 0x3eb2b8fbu, 0x3d2e54fau, 0x3c49b43du, 0x3eb1f43du, 0x3d2e54fau, 0x3c150144u,
    0x3eb02d29u, 0x3d2e54fbu, 0x3bcde0f3u, 0x3eae528du, 0x3d2e54fbu, 0x3a148125u, 0x3eaeaa5fu, 0x3d2e54fbu, 0x3aa20a50u,
    0x3eaea824u, 0x3d2e54fbu, 0x3bb3335du, 0xbeb2fd2eu, 0xbd596593u, 0x3f01e921u, 0xbeb217dfu, 0xbd596593u, 0x3f02e43du,
    0xbeab4e4fu, 0xbd596594u, 0x3f0501a4u, 0xbeabef8eu, 0xbd596593u, 0x3f040361u, 0xbeb06c6eu, 0xbd596593u, 0x3f039deeu,
    0xbeaaa8ccu, 0xbd596594u, 0x3f0527cdu, 0xbeaecf2cu, 0xbd596593u, 0x3f03ce93u, 0xbeaac7d8u, 0xbd5ad966u, 0x3a1d79b8u,
    0xbeab6d6eu, 0xbd5ad966u, 0x3a9b4a52u, 0xbeab72b6u, 0xbd59893du, 0x3b9ca514u, 0xbeaede38u, 0xbd59d8b8u, 0x3baaa384u,
    0xbeb2ccf2u, 0xbd5ad968u, 0x3c4b2f09u, 0xbeb1f211u, 0xbd5a31bau, 0x3c16c9bcu, 0xbeb05072u, 0xbd59d8b8u, 0x3bcf54a9u,
    0x3eac8f88u, 0xbd5db1bdu, 0x3f054397u, 0x3ead00c8u, 0xbd5db1bdu, 0x3f05003au, 0x3eacc71cu, 0xbd5cd2d1u, 0x3f03e27au,
    0x3eb2daacu, 0xbd5b33b2u, 0x3f01e921u, 0x3eb1aa0fu, 0xbd5c1323u, 0x3f02b1dfu, 0x3eb06261u, 0xbd5cd29du, 0x3f039f13u,
    0x3eab67b3u, 0xbd61650au, 0x3ab1c538u, 0x3eab3043u, 0xbd61650au, 0x3a2d7952u, 0x3eab7988u, 0xbd5edb71u, 0x3b9808f8u,
    0x3eaebbcdu, 0xbd5db185u, 0x3baaa383u, 0x3eb1e41fu, 0xbd5b1abau, 0x3c133e37u, 0x3eb2aa6eu, 0xbd5a76a5u, 0x3c4b2f09u,
    0x3eb02df3u, 0xbd5db185u, 0x3bcf54a9u, 0xbeae447cu, 0xbd596594u, 0x3f052fe1u, 0xbeaf4877u, 0x3d2e54c3u, 0x3f04d1adu,
    0xbeaeb903u, 0x3d2e54c3u, 0x3f0529d4u, 0xbeac592du, 0xbd5a839fu, 0x3f056a33u, 0xbeb287dfu, 0xbd596593u, 0x3f02e53bu,
    0xbeb2fefau, 0x3d2e54c4u, 0x3f02c2bcu, 0xbeb144fau, 0x3d2e54ddu, 0x3f03b92cu, 0xbeb0b28du, 0xbd596593u, 0x3f03e85cu,
    0xbeadbb5bu, 0xbd59893du, 0x3abb9477u, 0xbeb04a13u, 0xbd59d8b8u, 0x3b9be7d4u, 0xbeb26192u, 0xbd5a31bau, 0x3c09659bu,
    0xbeb29ffau, 0x3d2e54fau, 0x3c0fd595u, 0xbeb18ee9u, 0x3d2e54fbu, 0x3bb06521u, 0xbeb033a9u, 0x3d2e54fbu, 0x3ad2983bu,
    0xbeac571du, 0xbd5ad966u, 0x39b944a4u, 0xbeaf5534u, 0x3d2e54fbu, 0x394ec206u, 0x3eaed18fu, 0xbd5d5238u, 0x3f04e237u,
    0x3eb095e4u, 0xbd5cd29du, 0x3f03f601u, 0x3eb24305u, 0xbd5c1323u, 0x3f02ffe9u, 0x3eb26278u, 0x3d3078a5u, 0x3f02ef52u,
    0x3eb0d22eu, 0x3d31ad25u, 0x3f040006u, 0x3eaf30b8u, 0x3d33188fu, 0x3f04f138u, 0x3eadbfc0u, 0xbd5db1bdu, 0x3f054c63u,
    0x3eadf3ecu, 0x3d33b22eu, 0x3f056772u, 0x3eaf960au, 0x3d2e54e1u, 0x32a75c29u, 0x3eb0cff3u, 0x3d2e54fbu, 0x3b1dc521u,
    0x3eb1edf3u, 0x3d2e54fbu, 0x3bacaa97u, 0x3eb30e2eu, 0x3d2e54fau, 0x3c0e1d0fu, 0x3eb2f008u, 0xbd5ac37cu, 0x3c1429afu,
    0x3eb083acu, 0xbd5db185u, 0x3ba0214au, 0x3eaf3571u, 0xbd5f2685u, 0x3b40cf96u, 0x3eacd34du, 0xbd619cb8u, 0x396a93aau,
};

// ADR-0078: under a hull vertex limit a registration answers for the hull PhysX simulates.
// The oracle is PhysX itself, called as Omni Physics cooks a convexHull collider: the points
// in their authored order and frame, eCOMPUTE_CONVEX, the vertex limit, default plane
// tolerance and GPU data. The registered hull has the oracle's vertices, and rays, rays onto
// the points the hull leaves outside, sphere sweeps and sphere and box overlaps through the
// public API answer as a brute-force test of the oracle's planes. A 130-point sphere is in
// the set because which points PhysX keeps for it depends on their order. Both point sets
// are refused without a limit, for more than 254 faces. Small point sets and point sets away
// from the origin get the oracle's hull under limits from 8 to 64.
void testAHullVertexLimitAnswersForTheHullPhysXSimulates()
{
    using namespace ovspatialquery::detail;
    std::vector<std::vector<float>> clouds(2u);
    clouds[0].resize(sizeof(kCartonBeyondTheFaceLimit) / sizeof(uint32_t));
    std::memcpy(clouds[0].data(), kCartonBeyondTheFaceLimit, sizeof(kCartonBeyondTheFaceLimit));
    for (int index = 0; index < 130; ++index)
    {
        const double z = 1.0 - 2.0 * (index + 0.5) / 130.0, angle = 2.3999632297286533 * index;
        const double radius = std::sqrt(1.0 - z * z);
        clouds[1].insert(clouds[1].end(), {static_cast<float>(radius * std::cos(angle)),
                                           static_cast<float>(radius * std::sin(angle)), static_cast<float>(z)});
    }
    constexpr uint32_t kLimit  = 64u;
    uint64_t           state   = UINT64_C(0x9E3779B97F4A7C15);
    auto               uniform = [&state](double low, double high)
    {
        state = state * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
        return low + (high - low) * static_cast<double>(state >> 11u) * (1.0 / 9007199254740992.0);
    };
    using Vector      = physx_convex_oracle::Vector;
    using Face        = physx_convex_oracle::Face;
    const auto dotV   = physx_convex_oracle::dot;
    const auto crossV = physx_convex_oracle::cross;
    const auto unitV  = physx_convex_oracle::unit;

    ovspatialquery_backend_runtime_t* runtime = nullptr;
    REQUIRE(acquireDefaultBackendRuntime(&runtime) == OVSPATIALQUERY_API_SUCCESS);
    const ovspatialquery_instance_desc_t desc{2u, 1u, UINT64_C(256) * 1024u * 1024u, nullptr, 0u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&desc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    uint64_t definitionValue = 80u;
    for (std::vector<float>& points : clouds)
    {
        const size_t count = points.size() / 3u;
        // The oracle hull.
        const std::vector<Vector> corners = physx_convex_oracle::cookAsOmniPhysics(points.data(), count, kLimit);
        REQUIRE(!corners.empty());
        Vector low{INFINITY, INFINITY, INFINITY}, high{-INFINITY, -INFINITY, -INFINITY};
        for (size_t index = 0u; index < points.size(); ++index)
        {
            low[index % 3u]  = std::min(low[index % 3u], double(points[index]));
            high[index % 3u] = std::max(high[index % 3u], double(points[index]));
        }
        const double            half   = 0.5 * std::max({high[0] - low[0], high[1] - low[1], high[2] - low[2]});
        const double            margin = 1e-6 * half;
        const std::vector<Face> faces  = physx_convex_oracle::hullFaces(corners, half);

        // The library's hull has the oracle's vertices.
        std::shared_ptr<const PreparedConvexHull> hull;
        ConvexHullCookingOptions                  options;
        options.vertexLimit = kLimit;
        REQUIRE(prepareConvexHull(runtime, points, UINT64_MAX, hull, nullptr, options) ==
                ConvexHullPreparationOutcome::success);
        std::vector<Vector> registered;
        for (size_t vertex = 0u; vertex < hull->vertices().size(); vertex += 3u)
            registered.push_back(
                {hull->vertices()[vertex], hull->vertices()[vertex + 1u], hull->vertices()[vertex + 2u]});
        std::sort(registered.begin(), registered.end());
        std::vector<Vector> expected = corners;
        std::sort(expected.begin(), expected.end());
        REQUIRE(registered == expected);

        // Through the public API: refused without a limit, registered with it.
        uint64_t                                 vertexOffset = 0u, vertexCount = count;
        InputArray                               values(&definitionValue, 1, kDLUInt);
        InputArray                               offsets(&vertexOffset, 1, kDLUInt);
        InputArray                               counts(&vertexCount, 1, kDLUInt);
        InputArray                               vertices(points.data(), static_cast<int64_t>(count), 3, kDLFloat);
        ovspatialquery_convex_definition_table_t definitions{
            &values.value, &offsets.value, &counts.value, nullptr, nullptr, &vertices.value,
            nullptr,       nullptr,        nullptr,       0.0,     0u};
        REQUIRE(ovspatialquery_enqueue_convex_registration(instance, &definitions).status ==
                OVSPATIALQUERY_API_INVALID_ARGUMENT);
        const ovspatialquery_string_t refusal = ovspatialquery_get_last_error();
        REQUIRE(std::string(refusal.ptr, refusal.length).find("more than 254 faces") != std::string::npos);
        definitions.hull_vertex_limit = kLimit;
        waitAndDestroy(instance, ovspatialquery_enqueue_convex_registration(instance, &definitions));
        InstanceRows rows;
        rows.add(1u, OVSPATIALQUERY_GEOMETRY_CONVEX, definitionValue, 0.0, 0.0, 0.0);
        const ovspatialquery_population_handle_t population = createBoxPopulation(instance, 1, rows);

        const Vector centre{0.5 * (low[0] + high[0]), 0.5 * (low[1] + high[1]), 0.5 * (low[2] + high[2])};
        const auto   clip = [&](const Vector& origin, const Vector& direction, double& entry, double& exit)
        { return physx_convex_oracle::clipRay(faces, origin, direction, 4.0 * half, entry, exit); };
        ovspatialquery_query_common_t common{};
        common.target      = populationTarget(population);
        common.precision   = OVSPATIALQUERY_QUERY_EXACT;
        const auto castRay = [&](const Vector& origin, const Vector& direction, double& distance)
        {
            ovspatialquery_raycast_desc_t ray{};
            ray.common = common;
            for (uint32_t axis = 0u; axis < 3u; ++axis)
            {
                ray.ray.origin[axis]         = origin[axis];
                ray.ray.unit_direction[axis] = static_cast<float>(direction[axis]);
            }
            ray.ray.max_distance = 4.0 * half;
            ovspatialquery_raycast_single_result_t result{};
            REQUIRE(ovspatialquery_raycast_closest_sync(instance, &ray, &result).status == OVSPATIALQUERY_API_SUCCESS);
            distance = result.has_hit != 0u ? result.hit.data.exact.distance : -1.0;
            return result.has_hit != 0u;
        };
        const auto floatDirection = [&](const Vector& direction)
        {
            const Vector unit = unitV(direction);
            return Vector{double(static_cast<float>(unit[0])), double(static_cast<float>(unit[1])),
                          double(static_cast<float>(unit[2]))};
        };
        size_t sweeps = 0u, boxed = 0u, rays = 0u;
        // Rays from a sphere of radius 2h toward random points of the box.
        for (int index = 0; index < 400; ++index)
        {
            const Vector from = unitV({uniform(-1, 1), uniform(-1, 1), uniform(-1, 1)});
            const Vector origin{centre[0] + 2 * half * from[0], centre[1] + 2 * half * from[1],
                                centre[2] + 2 * half * from[2]};
            const Vector aim{uniform(low[0], high[0]), uniform(low[1], high[1]), uniform(low[2], high[2])};
            const Vector direction = floatDirection({aim[0] - origin[0], aim[1] - origin[1], aim[2] - origin[2]});
            double       entry = 0.0, exit = 0.0, distance = 0.0;
            const bool   expectedHit = clip(origin, direction, entry, exit);
            if (expectedHit && exit - entry < margin)
                continue; // grazing
            REQUIRE(castRay(origin, direction, distance) == expectedHit);
            if (expectedHit)
                REQUIRE(std::abs(distance - entry) <= margin);
            rays += expectedHit ? 1u : 0u;
        }
        // Rays onto each point the hull leaves outside, along the normal of the plane it lies
        // farthest outside of, stop on the hull beyond the point.
        size_t outside = 0u;
        for (size_t index = 0u; index < points.size(); index += 3u)
        {
            const Vector point{points[index], points[index + 1u], points[index + 2u]};
            const Face*  farthest = &faces.front();
            for (const Face& face : faces)
                if (dotV(face.normal, point) - face.offset > dotV(farthest->normal, point) - farthest->offset)
                    farthest = &face;
            const double height = dotV(farthest->normal, point) - farthest->offset;
            if (height <= margin)
                continue;
            ++outside;
            const Vector direction = floatDirection({-farthest->normal[0], -farthest->normal[1], -farthest->normal[2]});
            const Vector origin{point[0] - 2 * half * direction[0], point[1] - 2 * half * direction[1],
                                point[2] - 2 * half * direction[2]};
            double       entry = 0.0, exit = 0.0, distance = 0.0;
            REQUIRE(clip(origin, direction, entry, exit));
            REQUIRE(castRay(origin, direction, distance));
            REQUIRE(std::abs(distance - entry) <= margin && distance > 2 * half + margin);
        }
        REQUIRE(outside != 0u);
        // Sphere sweeps and overlaps onto the middle of each face, where a sphere of radius
        // 0.02 h first touches the face itself: every other face plane lies more than two radii
        // beyond the middle.
        const double radius = 0.02 * half;
        for (const Face& face : faces)
        {
            Vector middle{0.0, 0.0, 0.0};
            for (const Vector& corner : face.corners)
                for (uint32_t axis = 0u; axis < 3u; ++axis)
                    middle[axis] += corner[axis] / double(face.corners.size());
            bool roomy = true;
            for (const Face& other : faces)
                roomy = roomy && (&other == &face || other.offset - dotV(other.normal, middle) > 2.0 * radius);
            if (!roomy)
                continue;
            ++sweeps;
            ovspatialquery_sphere_sweep_desc_t sweep{};
            sweep.common           = common;
            const Vector direction = floatDirection({-face.normal[0], -face.normal[1], -face.normal[2]});
            for (uint32_t axis = 0u; axis < 3u; ++axis)
            {
                sweep.sweep.center[axis]         = middle[axis] - 2 * half * direction[axis];
                sweep.sweep.unit_direction[axis] = static_cast<float>(direction[axis]);
            }
            sweep.sweep.radius       = radius;
            sweep.sweep.max_distance = 4.0 * half;
            ovspatialquery_sphere_sweep_closest_result_t swept{};
            REQUIRE(ovspatialquery_sweep_sphere_closest_sync(instance, &sweep, &swept).status ==
                    OVSPATIALQUERY_API_SUCCESS);
            const double expectedDistance =
                (dotV(face.normal, Vector{sweep.sweep.center[0], sweep.sweep.center[1], sweep.sweep.center[2]}) -
                 face.offset - radius) /
                -dotV(face.normal, direction);
            REQUIRE(swept.has_hit != 0u && std::abs(swept.hit.data.exact.distance - expectedDistance) <= 1e-5 * half);
            for (const double gap : {-margin, margin})
            {
                ovspatialquery_overlap_desc_t overlap{};
                overlap.common     = common;
                overlap.shape.kind = OVSPATIALQUERY_OVERLAP_SPHERE;
                for (uint32_t axis = 0u; axis < 3u; ++axis)
                    overlap.shape.data.sphere.center[axis] = middle[axis] + (radius + gap) * face.normal[axis];
                overlap.shape.data.sphere.radius = radius;
                ovspatialquery_overlap_any_result_t found{};
                REQUIRE(ovspatialquery_overlap_any_sync(instance, &overlap, &found).status ==
                        OVSPATIALQUERY_API_SUCCESS);
                REQUIRE((found.has_hit != 0u) == (gap < 0.0));
            }
        }
        // Boxes in random poses, decided by the separating-axis test on the oracle's faces,
        // the box's axes and their edges' cross products.
        std::vector<Vector> edges;
        for (size_t first = 0u; first < faces.size(); ++first)
            for (size_t second = first + 1u; second < faces.size(); ++second)
            {
                std::vector<Vector> shared;
                for (const Vector& corner : faces[first].corners)
                    if (std::abs(dotV(faces[second].normal, corner) - faces[second].offset) <= 1e-9 * half)
                        shared.push_back(corner);
                if (shared.size() >= 2u)
                    edges.push_back(
                        unitV({shared[1][0] - shared[0][0], shared[1][1] - shared[0][1], shared[1][2] - shared[0][2]}));
            }
        for (int index = 0; index < 200; ++index)
        {
            const Vector at{centre[0] + 1.3 * uniform(-1, 1) * (high[0] - low[0]) / 2,
                            centre[1] + 1.3 * uniform(-1, 1) * (high[1] - low[1]) / 2,
                            centre[2] + 1.3 * uniform(-1, 1) * (high[2] - low[2]) / 2};
            const Vector extent{half * std::pow(10.0, uniform(-3, -0.5)), half * std::pow(10.0, uniform(-3, -0.5)),
                                half * std::pow(10.0, uniform(-3, -0.5))};
            std::array<float, 4u> turn{};
            {
                const Vector axis  = unitV({uniform(-1, 1), uniform(-1, 1), uniform(-1, 1)});
                const double angle = uniform(0, 3.141592653589793);
                turn               = {static_cast<float>(axis[0] * std::sin(angle / 2)),
                                      static_cast<float>(axis[1] * std::sin(angle / 2)),
                                      static_cast<float>(axis[2] * std::sin(angle / 2)), static_cast<float>(std::cos(angle / 2))};
            }
            const double                 x = turn[0], y = turn[1], z = turn[2], w = turn[3];
            const double                 norm = x * x + y * y + z * z + w * w;
            const std::array<Vector, 3u> axes{
                Vector{(1 - 2 * (y * y + z * z) / norm), 2 * (x * y + z * w) / norm, 2 * (x * z - y * w) / norm},
                Vector{2 * (x * y - z * w) / norm, (1 - 2 * (x * x + z * z) / norm), 2 * (y * z + x * w) / norm},
                Vector{2 * (x * z + y * w) / norm, 2 * (y * z - x * w) / norm, (1 - 2 * (x * x + y * y) / norm)}};
            std::vector<Vector> candidates;
            for (const Face& face : faces)
                candidates.push_back(face.normal);
            for (const Vector& axis : axes)
            {
                candidates.push_back(axis);
                for (const Vector& edge : edges)
                {
                    const Vector across = crossV(edge, axis);
                    if (dotV(across, across) > 1e-18)
                        candidates.push_back(unitV(across));
                }
            }
            double separation = -INFINITY;
            for (const Vector& axis : candidates)
            {
                double hullLow = INFINITY, hullHigh = -INFINITY;
                for (const Vector& corner : corners)
                {
                    hullLow  = std::min(hullLow, dotV(axis, corner));
                    hullHigh = std::max(hullHigh, dotV(axis, corner));
                }
                double reach = 0.0;
                for (uint32_t k = 0u; k < 3u; ++k)
                    reach += std::abs(dotV(axis, axes[k])) * extent[k];
                const double middle = dotV(axis, at);
                separation = std::max(separation, std::max(hullLow - (middle + reach), (middle - reach) - hullHigh));
            }
            if (std::abs(separation) < margin)
                continue;
            ++boxed;
            ovspatialquery_overlap_desc_t overlap{};
            overlap.common     = common;
            overlap.shape.kind = OVSPATIALQUERY_OVERLAP_BOX;
            for (uint32_t axis = 0u; axis < 3u; ++axis)
            {
                overlap.shape.data.box.center[axis]       = at[axis];
                overlap.shape.data.box.half_extents[axis] = extent[axis];
            }
            for (uint32_t component = 0u; component < 4u; ++component)
                overlap.shape.data.box.rotation_xyzw[component] = turn[component];
            ovspatialquery_overlap_any_result_t found{};
            REQUIRE(ovspatialquery_overlap_any_sync(instance, &overlap, &found).status == OVSPATIALQUERY_API_SUCCESS);
            REQUIRE((found.has_hit != 0u) == (separation < 0.0));
        }
        // Each family compared some rows, so the skips above cannot make the test vacuous.
        REQUIRE(rays != 0u && sweeps != 0u && boxed != 0u);
        REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_CONVEX, definitionValue})
                    .status == OVSPATIALQUERY_API_SUCCESS);
        ++definitionValue;
    }

    // Where the points are, at every size: PhysX's hull tolerances have absolute floors, so
    // the hull it cooks depends on the points' own coordinates, and the simulation cooks them
    // there. A 1,000-point sphere and a 128-gon prism of radius 0.01, 0.1 and 1, about the
    // origin, (5, 2, 1) and (50, 20, 10), register under every limit from 8 to 64 with the
    // oracle's vertices, and rays from 3 radii toward random points of their box answer as
    // its planes. A cropped hull's corners need not be input points, and at (5, 2, 1) they
    // are off their polygons' planes by more than the polygon tolerance.
    for (const bool prism : {false, true})
        for (const double radius : {0.01, 0.1, 1.0})
            for (const Vector& at : {Vector{0.0, 0.0, 0.0}, Vector{5.0, 2.0, 1.0}, Vector{50.0, 20.0, 10.0}})
            {
                std::vector<float> points;
                const auto         add = [&](double x, double y, double z)
                {
                    points.insert(points.end(),
                                  {static_cast<float>(at[0] + radius * x), static_cast<float>(at[1] + radius * y),
                                   static_cast<float>(at[2] + radius * z)});
                };
                for (int index = 0; !prism && index < 1000; ++index)
                {
                    const double z = 1.0 - 2.0 * (index + 0.5) / 1000.0, angle = 2.3999632297286533 * index;
                    add(std::sqrt(1.0 - z * z) * std::cos(angle), std::sqrt(1.0 - z * z) * std::sin(angle), z);
                }
                for (const double height : {-0.5, 0.5})
                    for (int side = 0; prism && side < 128; ++side)
                        add(std::cos(6.283185307179586 * side / 128.0), std::sin(6.283185307179586 * side / 128.0),
                            height);
                const size_t count = points.size() / 3u;
                for (const uint32_t limit : {8u, 16u, 32u, 64u})
                {
                    const std::vector<Vector> corners =
                        physx_convex_oracle::cookAsOmniPhysics(points.data(), count, limit);
                    REQUIRE(!corners.empty());
                    const std::vector<Face>                   faces = physx_convex_oracle::hullFaces(corners, radius);
                    std::shared_ptr<const PreparedConvexHull> hull;
                    ConvexHullCookingOptions                  options;
                    options.vertexLimit = limit;
                    REQUIRE(prepareConvexHull(runtime, points, UINT64_MAX, hull, nullptr, options) ==
                            ConvexHullPreparationOutcome::success);
                    std::vector<Vector> registered;
                    for (size_t vertex = 0u; vertex < hull->vertices().size(); vertex += 3u)
                        registered.push_back(
                            {hull->vertices()[vertex], hull->vertices()[vertex + 1u], hull->vertices()[vertex + 2u]});
                    std::sort(registered.begin(), registered.end());
                    std::vector<Vector> expected = corners;
                    std::sort(expected.begin(), expected.end());
                    REQUIRE(registered == expected);

                    uint64_t   vertexOffset = 0u, vertexCount = count;
                    InputArray values(&definitionValue, 1, kDLUInt);
                    InputArray offsets(&vertexOffset, 1, kDLUInt);
                    InputArray counts(&vertexCount, 1, kDLUInt);
                    InputArray vertices(points.data(), static_cast<int64_t>(count), 3, kDLFloat);
                    ovspatialquery_convex_definition_table_t definitions{
                        &values.value, &offsets.value, &counts.value, nullptr, nullptr, &vertices.value,
                        nullptr,       nullptr,        nullptr,       0.0,     limit};
                    waitAndDestroy(instance, ovspatialquery_enqueue_convex_registration(instance, &definitions));
                    InstanceRows rows;
                    rows.add(1u, OVSPATIALQUERY_GEOMETRY_CONVEX, definitionValue, 0.0, 0.0, 0.0);
                    const ovspatialquery_population_handle_t population = createBoxPopulation(instance, 1, rows);
                    ovspatialquery_raycast_desc_t            ray{};
                    ray.common.target    = populationTarget(population);
                    ray.common.precision = OVSPATIALQUERY_QUERY_EXACT;
                    ray.ray.max_distance = 6.0 * radius;
                    for (int index = 0; index < 32; ++index)
                    {
                        const Vector from = unitV({uniform(-1, 1), uniform(-1, 1), uniform(-1, 1)});
                        const Vector origin{at[0] + 3 * radius * from[0], at[1] + 3 * radius * from[1],
                                            at[2] + 3 * radius * from[2]};
                        const Vector aim{at[0] + uniform(-radius, radius), at[1] + uniform(-radius, radius),
                                         at[2] + uniform(-radius, radius)};
                        Vector       direction = unitV({aim[0] - origin[0], aim[1] - origin[1], aim[2] - origin[2]});
                        for (uint32_t axis = 0u; axis < 3u; ++axis)
                        {
                            ray.ray.origin[axis]         = origin[axis];
                            ray.ray.unit_direction[axis] = static_cast<float>(direction[axis]);
                            direction[axis]              = ray.ray.unit_direction[axis];
                        }
                        double     entry = 0.0, exit = 0.0;
                        const bool expectedHit =
                            physx_convex_oracle::clipRay(faces, origin, direction, ray.ray.max_distance, entry, exit);
                        if (expectedHit && exit - entry < 1e-5 * radius)
                            continue; // grazing
                        ovspatialquery_raycast_single_result_t result{};
                        REQUIRE(ovspatialquery_raycast_closest_sync(instance, &ray, &result).status ==
                                OVSPATIALQUERY_API_SUCCESS);
                        REQUIRE((result.has_hit != 0u) == expectedHit);
                        REQUIRE(!expectedHit || std::abs(result.hit.data.exact.distance - entry) <= 1e-5 * radius);
                    }
                    REQUIRE(ovspatialquery_destroy_population(instance, population).status ==
                            OVSPATIALQUERY_API_SUCCESS);
                    REQUIRE(ovspatialquery_destroy_geometry_definition(
                                instance, {OVSPATIALQUERY_GEOMETRY_CONVEX, definitionValue++})
                                .status == OVSPATIALQUERY_API_SUCCESS);
                }
            }
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
    releaseDefaultBackendRuntime(runtime);
}

// ADR-0081: a convexHull collider of a centimetre stage, registered under a hull vertex limit
// from its local points in centimetres and placed by the prim's rotation and translation with a
// uniform scale of 0.01, answers for the hull the simulation collides with: PhysX's cook of those
// local points at the stage's tolerance scale of 100, rotated, scaled and moved. The collider is
// a 128-gon prism 0.4 cm across, the size of a bollard part, turned 30 degrees and 5.5 m from the
// origin. PhysX cooks another hull from the same points with the metres and the pose already
// applied, so a registration that applied the placement before cooking would fail the rays.
void testALocalColliderPlacedByAUniformScaleAnswersForTheSimulatedHull()
{
    using namespace ovspatialquery::detail;
    using Vector = physx_convex_oracle::Vector;
    using Face   = physx_convex_oracle::Face;
    std::vector<float> local;
    for (const double height : {-0.1, 0.1})
        for (int side = 0; side < 128; ++side)
            local.insert(local.end(), {static_cast<float>(0.2 * std::cos(6.283185307179586 * side / 128.0)),
                                       static_cast<float>(0.2 * std::sin(6.283185307179586 * side / 128.0)),
                                       static_cast<float>(height)});
    const size_t       count          = local.size() / 3u;
    constexpr uint32_t kLimit         = 64u;
    constexpr double   kMetersPerUnit = 0.01, kHalf = 0.2 * kMetersPerUnit;
    const Vector       axis      = physx_convex_oracle::unit({1.0, 2.0, 3.0}), at{5.0, 2.0, 1.0};
    const double       halfAngle = 0.5 * 0.5235987755982988;
    double rotation[4] = {axis[0] * std::sin(halfAngle), axis[1] * std::sin(halfAngle), axis[2] * std::sin(halfAngle),
                          std::cos(halfAngle)};
    const auto place   = [&](const Vector& point)
    {
        const Vector q{rotation[0], rotation[1], rotation[2]},
            scaled{kMetersPerUnit * point[0], kMetersPerUnit * point[1], kMetersPerUnit * point[2]};
        const Vector t = physx_convex_oracle::cross(q, scaled), u = physx_convex_oracle::cross(q, t);
        Vector       world{};
        for (uint32_t k = 0u; k < 3u; ++k)
            world[k] = at[k] + scaled[k] + 2.0 * (rotation[3] * t[k] + u[k]);
        return world;
    };
    const auto worldFaces = [&](const std::vector<Vector>& corners)
    {
        REQUIRE(!corners.empty());
        std::vector<Vector> placed;
        for (const Vector& corner : corners)
            placed.push_back(place(corner));
        return physx_convex_oracle::hullFaces(placed, kHalf);
    };
    // The simulation's hull, and PhysX's hull of the same points already in world metres.
    ovspatialquery_backend_runtime_t* runtime = nullptr;
    REQUIRE(acquireDefaultBackendRuntime(&runtime) == OVSPATIALQUERY_API_SUCCESS);
    const std::vector<Face> simulated =
        worldFaces(physx_convex_oracle::cookAsOmniPhysics(local.data(), count, kLimit, kMetersPerUnit));
    std::vector<float> baked;
    for (size_t index = 0u; index < local.size(); index += 3u)
        for (const double coordinate : place({local[index], local[index + 1u], local[index + 2u]}))
            baked.push_back(static_cast<float>(coordinate));
    const std::vector<Vector> bakedCorners = physx_convex_oracle::cookAsOmniPhysics(baked.data(), count, kLimit);
    REQUIRE(!bakedCorners.empty());
    const std::vector<Face> bakedFaces = physx_convex_oracle::hullFaces(bakedCorners, kHalf);

    const ovspatialquery_instance_desc_t desc{2u, 1u, UINT64_C(64) * 1024u * 1024u, nullptr, 0u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&desc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    uint64_t                                 definitionValue = 90u, vertexOffset = 0u, vertexCount = count;
    InputArray                               values(&definitionValue, 1, kDLUInt);
    InputArray                               offsets(&vertexOffset, 1, kDLUInt);
    InputArray                               counts(&vertexCount, 1, kDLUInt);
    InputArray                               vertices(local.data(), static_cast<int64_t>(count), 3, kDLFloat);
    ovspatialquery_convex_definition_table_t definitions{
        &values.value, &offsets.value, &counts.value, nullptr, nullptr, &vertices.value,
        nullptr,       nullptr,        nullptr,       0.0,     kLimit};
    waitAndDestroy(instance, ovspatialquery_enqueue_convex_registration(instance, &definitions));
    uint64_t                                       objectValue  = 1u;
    int32_t                                        geometryType = OVSPATIALQUERY_GEOMETRY_CONVEX;
    double                                         translation[3]{at[0], at[1], at[2]}, scale = kMetersPerUnit;
    InputArray                                     objects(&objectValue, 1, kDLUInt);
    InputArray                                     types(&geometryType, 1, kDLInt);
    InputArray                                     geometry(&definitionValue, 1, kDLUInt);
    InputArray                                     translations(translation, 1, 3, kDLFloat);
    InputArray                                     rotations(rotation, 1, 4, kDLFloat);
    InputArray                                     scales(&scale, 1, kDLFloat);
    const ovspatialquery_population_handle_t       population = createExplicitPopulation(instance, 0u);
    const ovspatialquery_explicit_population_set_t set{1,
                                                       {&objects.value, &types.value, &geometry.value,
                                                        &translations.value, &rotations.value, &scales.value, nullptr,
                                                        nullptr},
                                                       nullptr,
                                                       0u};
    waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set));

    // Rays from a sphere of radius 3h about the prism toward random points within h of its
    // centre answer as the simulated hull's planes; many of them meet the baked hull elsewhere.
    uint64_t state   = UINT64_C(0x9E3779B97F4A7C15);
    auto     uniform = [&state](double low, double high)
    {
        state = state * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
        return low + (high - low) * static_cast<double>(state >> 11u) * (1.0 / 9007199254740992.0);
    };
    ovspatialquery_raycast_desc_t ray{};
    ray.common.target     = populationTarget(population);
    ray.common.precision  = OVSPATIALQUERY_QUERY_EXACT;
    ray.ray.max_distance  = 6.0 * kHalf;
    const double margin   = 1e-5 * kHalf;
    size_t       compared = 0u, bakedDiffers = 0u;
    for (int index = 0; index < 400; ++index)
    {
        const Vector from = physx_convex_oracle::unit({uniform(-1, 1), uniform(-1, 1), uniform(-1, 1)});
        const Vector origin{at[0] + 3 * kHalf * from[0], at[1] + 3 * kHalf * from[1], at[2] + 3 * kHalf * from[2]};
        const Vector aim{at[0] + uniform(-kHalf, kHalf), at[1] + uniform(-kHalf, kHalf),
                         at[2] + uniform(-kHalf, kHalf)};
        Vector direction = physx_convex_oracle::unit({aim[0] - origin[0], aim[1] - origin[1], aim[2] - origin[2]});
        for (uint32_t k = 0u; k < 3u; ++k)
        {
            ray.ray.origin[k]         = origin[k];
            ray.ray.unit_direction[k] = static_cast<float>(direction[k]);
            direction[k]              = ray.ray.unit_direction[k];
        }
        double     entry = 0.0, exit = 0.0, bakedEntry = 0.0, bakedExit = 0.0;
        const bool expectedHit =
            physx_convex_oracle::clipRay(simulated, origin, direction, ray.ray.max_distance, entry, exit);
        const bool bakedHit =
            physx_convex_oracle::clipRay(bakedFaces, origin, direction, ray.ray.max_distance, bakedEntry, bakedExit);
        if ((expectedHit && exit - entry < margin) || (bakedHit && bakedExit - bakedEntry < margin))
            continue; // grazing
        ++compared;
        bakedDiffers += bakedHit != expectedHit || (expectedHit && std::abs(bakedEntry - entry) > margin) ? 1u : 0u;
        ovspatialquery_raycast_single_result_t result{};
        REQUIRE(ovspatialquery_raycast_closest_sync(instance, &ray, &result).status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE((result.has_hit != 0u) == expectedHit);
        REQUIRE(!expectedHit || std::abs(result.hit.data.exact.distance - entry) <= margin);
    }
    REQUIRE(compared > 300u && bakedDiffers > 0u);
    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_CONVEX, definitionValue})
                .status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
    releaseDefaultBackendRuntime(runtime);
}

int main()
{
    testPackagedPhysXConvexPreparationLimitsAndAtomicity();
    testPackagedPhysXConvexRegistrationAcceptsMergedCookerPolygons();
    testNearlyFlatConvexIsItsPointsHullOrRefused();
    testAHullVertexLimitAnswersForTheHullPhysXSimulates();
    testALocalColliderPlacedByAUniformScaleAnswersForTheSimulatedHull();
    testPackagedPhysXConvexOverlapParityAndLifetime();
    testPreparedConvexContactsMatchWorldConstructionAndRetainPoses();
    testPreparedConvexRetainsLocalGeometryAfterPlacement();
    testTriangleRaycastMapsFacesAcrossTheDomainRange();
    testTriangleCandidateOverflowCompletesOrReportsBudget();
    testRayOriginToleranceCoversRoundedOrigins();
    testDerivedPopulationsAnswerLikeRebuilds();

    const ovspatialquery_instance_desc_t instanceDesc{4u, 1u, UINT64_C(64) * 1024u * 1024u, nullptr, 0u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&instanceDesc, &instance).status == OVSPATIALQUERY_API_SUCCESS);

    // Self-pairs: an identical selection is unordered with no self-pairs.
    // Object 30 overlaps 31 but neither overlaps the isolated object 32.
    registerAabbBoxes(instance);
    InstanceRows selfRows;
    selfRows.add(30u, OVSPATIALQUERY_GEOMETRY_BOX, kWideBoxDefinition, 1.0, 0.5, 0.5);
    selfRows.add(31u, OVSPATIALQUERY_GEOMETRY_BOX, kWideBoxDefinition, 2.0, 0.5, 0.5);
    selfRows.add(32u, OVSPATIALQUERY_GEOMETRY_BOX, kSquareBoxDefinition, 5.5, 0.5, 0.5);
    const ovspatialquery_population_handle_t selfPopulation = createBoxPopulation(instance, 12, selfRows);
    REQUIRE(snapshotPairs(instance, selfPopulation, selfPopulation, 12) == PairSet{{30u, 31u}});

    REQUIRE(ovspatialquery_destroy_population(instance, selfPopulation).status == OVSPATIALQUERY_API_SUCCESS);
    for (uint64_t definition : {kSquareBoxDefinition, kWideBoxDefinition})
    {
        REQUIRE(
            ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_BOX, definition}).status ==
            OVSPATIALQUERY_API_SUCCESS);
    }
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);

    std::puts("PhysX backend contract test passed");
    return 0;
}
