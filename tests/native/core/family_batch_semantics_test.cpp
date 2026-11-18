/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "oracle_backend.h"
#include "ovspatialquery_family_batch_cpu.h"
#include "ovspatialquery_target_resolver.h"
#include "test_failure.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>

namespace
{

using ovspatialquery::detail::FamilyBatchOutcome;
using ovspatialquery::detail::GeometryFeatureKind;
using ovspatialquery::detail::GeometryReference;
using ovspatialquery::detail::HitMode;
using ovspatialquery::detail::HitTruth;
using ovspatialquery::detail::OverlapBatchLaunch;
using ovspatialquery::detail::OverlapBatchPlan;
using ovspatialquery::detail::OverlapBatchResult;
using ovspatialquery::detail::OverlapBatchRow;
using ovspatialquery::detail::PopulationKey;
using ovspatialquery::detail::PopulationObject;
using ovspatialquery::detail::PopulationTable;
using ovspatialquery::detail::QueryPrecision;
using ovspatialquery::detail::RaycastBatchLaunch;
using ovspatialquery::detail::RaycastBatchPlan;
using ovspatialquery::detail::RaycastBatchResult;
using ovspatialquery::detail::RaycastBatchRow;
using ovspatialquery::detail::ResolvedTarget;
using ovspatialquery::detail::RowOutcome;
using ovspatialquery::detail::SphereSweepBatchLaunch;
using ovspatialquery::detail::SphereSweepBatchPlan;
using ovspatialquery::detail::SphereSweepBatchResult;
using ovspatialquery::detail::SphereSweepBatchRow;
using ovspatialquery::detail::TargetRequest;
using ovspatialquery::detail::TargetResolveOutcome;
using ovspatialquery::detail::TargetResolver;
using ovspatialquery::detail::TargetSelectionKind;

const ovspatialquery::detail::SemanticHitIdentity& identity(const ovspatialquery::detail::RaycastSemanticHit& hit)
{
    return hit.truth == HitTruth::exactSurface ? hit.exact.identity : hit.boundsCandidate.identity;
}

const ovspatialquery::detail::SemanticHitIdentity& identity(const ovspatialquery::detail::SphereSweepSemanticHit& hit)
{
    return hit.truth == HitTruth::exactSurface ? hit.exact.identity : hit.boundsCandidate.identity;
}

PopulationObject object(uint64_t id, double minimumX, double maximumX, uint64_t participationBits,
                        uint64_t geometryValue)
{
    PopulationObject value{};
    value.id.value          = id;
    value.bounds.minimum[0] = minimumX;
    value.bounds.minimum[1] = -1.0;
    value.bounds.minimum[2] = -1.0;
    value.bounds.maximum[0] = maximumX;
    value.bounds.maximum[1] = 1.0;
    value.bounds.maximum[2] = 1.0;
    value.geometryReference = GeometryReference{ovspatialquery::detail::ExactGeometryFamily::sphere, geometryValue};
    value.participationBits = {{participationBits, 0u}};
    return value;
}

PopulationObject exactObject(uint64_t id, uint64_t geometryValue,
                             std::shared_ptr<const ovspatialquery::detail::ExactGeometry> geometry,
                             uint64_t participationBits = UINT64_MAX)
{
    PopulationObject value{};
    value.id.value          = id;
    value.bounds            = geometry->worldBounds();
    value.geometryReference = GeometryReference{geometry->family(), geometryValue};
    value.exactGeometry     = std::move(geometry);
    value.participationBits = {{participationBits, 0u}};
    return value;
}

ovspatialquery::detail::ConvexGeometrySource translatedCube(double x)
{
    ovspatialquery::detail::ConvexGeometrySource source{};
    source.vertices = {
        -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0,
        -1.0, -1.0, 1.0,  1.0, -1.0, 1.0,  1.0, 1.0, 1.0,  -1.0, 1.0, 1.0,
    };
    source.faceVertexCounts  = {4u, 4u, 4u, 4u, 4u, 4u};
    source.faceVertexIndices = {
        0u, 3u, 2u, 1u, 4u, 5u, 6u, 7u, 0u, 1u, 5u, 4u, 3u, 7u, 6u, 2u, 0u, 4u, 7u, 3u, 1u, 2u, 6u, 5u,
    };
    source.worldFromLocal = {
        1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    };
    return source;
}

std::shared_ptr<const ovspatialquery::detail::ExactGeometry> makeMeshGeometry()
{
    constexpr uint32_t                         sourceFace = 700u;
    constexpr float                            positionX  = 14.0f;
    ovspatialquery::detail::TriangleMeshSource mesh{};
    mesh.key      = 77u;
    mesh.vertices = {
        positionX, -1.0, -1.0, positionX, -1.0, 1.0, positionX, 1.0, 1.0, positionX, 1.0, -1.0,
    };
    mesh.triangles     = {0u, 1u, 2u, 0u, 2u, 3u};
    mesh.sourceFaces   = {sourceFace, sourceFace};
    mesh.sourceCorners = {0u, 1u, 2u, 0u, 2u, 3u};
    std::shared_ptr<const ovspatialquery::detail::PreparedTriangleMesh> preparedMesh;
    REQUIRE(ovspatialquery::detail::prepareTriangleMesh(mesh, preparedMesh) == OVSPATIALQUERY_API_SUCCESS);
    std::shared_ptr<const ovspatialquery::detail::TriangleObjectGeometry> prepared;
    REQUIRE(ovspatialquery::detail::bindPreparedTriangleGeometry(
                preparedMesh, {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0}, true,
                prepared) == OVSPATIALQUERY_API_SUCCESS);
    std::shared_ptr<const ovspatialquery::detail::ExactGeometry> geometry;
    REQUIRE(ovspatialquery::detail::createTriangleMeshExactGeometry(prepared, geometry) == OVSPATIALQUERY_API_SUCCESS);
    return geometry;
}

std::vector<PopulationObject> fiveExactFamilies()
{
    std::vector<PopulationObject>                                objects;
    std::shared_ptr<const ovspatialquery::detail::ExactGeometry> geometry;

    ovspatialquery::detail::SphereGeometrySource sphere{};
    sphere.center = {2.0, 0.0, 0.0};
    sphere.radius = 1.0;
    REQUIRE(ovspatialquery::detail::createSphereExactGeometry(sphere, geometry) == OVSPATIALQUERY_API_SUCCESS);
    objects.push_back(exactObject(1u, 101u, geometry));

    ovspatialquery::detail::BoxGeometrySource box{};
    box.center = {5.0, 0.0, 0.0};
    box.axes   = {
        std::array<double, 3u>{1.0, 0.0, 0.0},
        std::array<double, 3u>{0.0, 1.0, 0.0},
        std::array<double, 3u>{0.0, 0.0, 1.0},
    };
    box.halfExtents = {1.0, 1.0, 1.0};
    REQUIRE(ovspatialquery::detail::createBoxExactGeometry(box, geometry) == OVSPATIALQUERY_API_SUCCESS);
    objects.push_back(exactObject(2u, 202u, geometry));

    ovspatialquery::detail::CapsuleGeometrySource capsule{};
    capsule.endpointA = {8.0, -1.0, 0.0};
    capsule.endpointB = {8.0, 1.0, 0.0};
    capsule.radius    = 1.0;
    REQUIRE(ovspatialquery::detail::createCapsuleExactGeometry(capsule, geometry) == OVSPATIALQUERY_API_SUCCESS);
    objects.push_back(exactObject(3u, 303u, geometry));

    REQUIRE(ovspatialquery::detail::createConvexExactGeometry(translatedCube(11.0), geometry) ==
            ovspatialquery::detail::ConvexConstructionOutcome::success);
    objects.push_back(exactObject(4u, 404u, geometry));

    objects.push_back(exactObject(5u, 505u, makeMeshGeometry()));
    return objects;
}

ResolvedTarget makeResolvedTarget(PopulationTable& populations, PopulationKey population)
{
    const TargetResolver resolver(populations);
    ResolvedTarget       resolved;
    const TargetRequest  request{{population, {}}, {TargetSelectionKind::head, 0}};
    REQUIRE(resolver.resolve(request, resolved) == TargetResolveOutcome::success);
    return resolved;
}

std::shared_ptr<const std::vector<uint64_t>> selected(std::initializer_list<uint64_t> ids)
{
    return std::make_shared<const std::vector<uint64_t>>(ids);
}

void publish(PopulationTable& populations, PopulationKey population, ovspatialquery_ordinal_t ordinal,
             std::vector<PopulationObject>                                 objects,
             std::vector<ovspatialquery::detail::GenerationViewMembership> memberships)
{
    ovspatialquery::detail::GenerationLease base;
    REQUIRE(populations.beginUpdate(population, ordinal, nullptr, base) == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery::detail::GenerationLease generation;
    REQUIRE(populations.buildGeneration(ordinal, std::move(objects), std::move(memberships), generation) ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(populations.publishUpdate(population, generation) == OVSPATIALQUERY_API_SUCCESS);
    generation.reset();
    base.reset();
    populations.finishUpdate(population);
}

PopulationObject sphereObject(uint64_t id, double centerX, uint64_t bits, uint64_t geometryValue)
{
    ovspatialquery::detail::SphereGeometrySource source{};
    source.center = {centerX, 0.0, 0.0};
    source.radius = 1.0;
    std::shared_ptr<const ovspatialquery::detail::ExactGeometry> geometry;
    REQUIRE(ovspatialquery::detail::createSphereExactGeometry(source, geometry) == OVSPATIALQUERY_API_SUCCESS);
    return exactObject(id, geometryValue, std::move(geometry), bits);
}

RaycastBatchRow ray(double originX, uint64_t queryBits)
{
    RaycastBatchRow row{};
    row.query.origin          = {originX, 0.0, 0.0};
    row.query.unitDirection   = {1.0f, 0.0f, 0.0f};
    row.query.minimumDistance = 0.0;
    row.query.maximumDistance = 100.0;
    row.queryBits             = {{queryBits, 0u}};
    return row;
}

SphereSweepBatchRow sweep(double originX, uint64_t queryBits)
{
    SphereSweepBatchRow row{};
    row.query.center          = {originX, 0.0, 0.0};
    row.query.radius          = 0.25;
    row.query.unitDirection   = {1.0f, 0.0f, 0.0f};
    row.query.minimumDistance = 0.0;
    row.query.maximumDistance = 100.0;
    row.queryBits             = {{queryBits, 0u}};
    return row;
}

OverlapBatchRow overlapAll(uint64_t queryBits)
{
    OverlapBatchRow row{};
    row.query.geometry.kind                    = OVSPATIALQUERY_OVERLAP_AABB;
    row.query.geometry.data.aabb.min_corner[0] = 0.0;
    row.query.geometry.data.aabb.min_corner[1] = -2.0;
    row.query.geometry.data.aabb.min_corner[2] = -2.0;
    row.query.geometry.data.aabb.max_corner[0] = 20.0;
    row.query.geometry.data.aabb.max_corner[1] = 2.0;
    row.query.geometry.data.aabb.max_corner[2] = 2.0;
    row.queryBits                              = {{queryBits, 0u}};
    return row;
}

void testResolvedCandidatePreservesIdentityAndParticipation(ovspatialquery_backend_runtime_t* runtime)
{
    PopulationTable               populations(runtime);
    PopulationKey                 population;
    std::vector<PopulationObject> objects{
        object(11u, 2.0, 3.0, UINT64_C(0x1), 101u),
        object(22u, 4.0, 5.0, UINT64_C(0x2), 202u),
    };
    REQUIRE(populations.create(2u, 17, std::move(objects), population) == OVSPATIALQUERY_API_SUCCESS);

    ResolvedTarget   resolved = makeResolvedTarget(populations, population);
    RaycastBatchPlan plan{};
    plan.precision           = QueryPrecision::boundsCandidate;
    plan.maximumQueryCount   = 1u;
    plan.maximumHitsPerQuery = 1u;
    plan.hitCapacity         = 1u;
    RaycastBatchLaunch launch{};
    launch.rows.push_back(ray(0.0, UINT64_C(0x2)));

    RaycastBatchResult result;
    REQUIRE(ovspatialquery::detail::executeFamilyBatchCpu(resolved, plan, launch, result) ==
            FamilyBatchOutcome::success);
    REQUIRE(result.common.sourceOrdinal == 17);
    REQUIRE(result.common.sourcePopulation == population);
    REQUIRE(result.common.truth == HitTruth::boundsCandidate);
    REQUIRE(result.common.storedHitCount == 1u);
    REQUIRE(result.common.rowOutcomes == std::vector<RowOutcome>{RowOutcome::success});
    REQUIRE(result.hits.size() == 1u);
    REQUIRE(identity(result.hits[0]).objectId.value == 22u);
    const GeometryReference expectedGeometry{ovspatialquery::detail::ExactGeometryFamily::sphere, 202u};
    REQUIRE(identity(result.hits[0]).geometry == expectedGeometry);

    resolved = {};
    std::shared_ptr<void> retired;
    REQUIRE(populations.release(population, retired) == OVSPATIALQUERY_API_SUCCESS);
}

void testPopulationQualifiesAllFamilyCandidateAndExactIdentity(ovspatialquery_backend_runtime_t* runtime)
{
    PopulationTable populations(runtime);
    PopulationKey   firstPopulation;
    PopulationKey   secondPopulation;
    REQUIRE(populations.create(2u, 81, {sphereObject(77u, 2.0, UINT64_MAX, 9001u)}, firstPopulation) ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(populations.create(2u, 81, {sphereObject(77u, 2.0, UINT64_MAX, 9001u)}, secondPopulation) ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(!(firstPopulation == secondPopulation));
    ResolvedTarget first  = makeResolvedTarget(populations, firstPopulation);
    ResolvedTarget second = makeResolvedTarget(populations, secondPopulation);

    RaycastBatchPlan plan{};
    plan.maximumQueryCount = 1u;
    plan.hitCapacity       = 1u;
    RaycastBatchLaunch launch{};
    launch.rows.push_back(ray(0.0, 0u));
    for (QueryPrecision precision : {QueryPrecision::boundsCandidate, QueryPrecision::exact})
    {
        plan.precision = precision;
        RaycastBatchResult firstResult;
        RaycastBatchResult secondResult;
        REQUIRE(ovspatialquery::detail::executeFamilyBatchCpu(first, plan, launch, firstResult) ==
                FamilyBatchOutcome::success);
        REQUIRE(ovspatialquery::detail::executeFamilyBatchCpu(second, plan, launch, secondResult) ==
                FamilyBatchOutcome::success);
        REQUIRE(firstResult.common.sourceOrdinal == 81);
        REQUIRE(secondResult.common.sourceOrdinal == 81);
        REQUIRE(firstResult.common.sourcePopulation == firstPopulation);
        REQUIRE(secondResult.common.sourcePopulation == secondPopulation);
        REQUIRE(firstResult.hits.size() == 1u);
        REQUIRE(secondResult.hits.size() == 1u);
        const ovspatialquery::detail::SemanticHitIdentity& firstIdentity  = identity(firstResult.hits[0]);
        const ovspatialquery::detail::SemanticHitIdentity& secondIdentity = identity(secondResult.hits[0]);
        REQUIRE(firstIdentity.objectId.value == secondIdentity.objectId.value);
        REQUIRE(firstIdentity.geometry == secondIdentity.geometry);

        SphereSweepBatchPlan sweepPlan{};
        sweepPlan.precision         = precision;
        sweepPlan.maximumQueryCount = 1u;
        sweepPlan.hitCapacity       = 1u;
        SphereSweepBatchLaunch sweepLaunch{};
        sweepLaunch.rows.push_back(sweep(0.0, 0u));
        SphereSweepBatchResult firstSweep;
        SphereSweepBatchResult secondSweep;
        REQUIRE(ovspatialquery::detail::executeFamilyBatchCpu(first, sweepPlan, sweepLaunch, firstSweep) ==
                FamilyBatchOutcome::success);
        REQUIRE(ovspatialquery::detail::executeFamilyBatchCpu(second, sweepPlan, sweepLaunch, secondSweep) ==
                FamilyBatchOutcome::success);
        REQUIRE(firstSweep.common.sourcePopulation == firstPopulation);
        REQUIRE(secondSweep.common.sourcePopulation == secondPopulation);
        REQUIRE(firstSweep.hits.size() == 1u);
        REQUIRE(secondSweep.hits.size() == 1u);

        OverlapBatchPlan overlapPlan{};
        overlapPlan.precision         = precision;
        overlapPlan.maximumQueryCount = 1u;
        overlapPlan.hitCapacity       = 1u;
        OverlapBatchLaunch overlapLaunch{};
        overlapLaunch.rows.push_back(overlapAll(0u));
        OverlapBatchResult firstOverlap;
        OverlapBatchResult secondOverlap;
        REQUIRE(ovspatialquery::detail::executeFamilyBatchCpu(first, overlapPlan, overlapLaunch, firstOverlap) ==
                FamilyBatchOutcome::success);
        REQUIRE(ovspatialquery::detail::executeFamilyBatchCpu(second, overlapPlan, overlapLaunch, secondOverlap) ==
                FamilyBatchOutcome::success);
        REQUIRE(firstOverlap.common.sourcePopulation == firstPopulation);
        REQUIRE(secondOverlap.common.sourcePopulation == secondPopulation);
        REQUIRE(firstOverlap.hits.size() == 1u);
        REQUIRE(secondOverlap.hits.size() == 1u);
    }

    RaycastBatchLaunch noHitLaunch{};
    noHitLaunch.rows.push_back(ray(100.0, 0u));
    RaycastBatchResult noHitResult;
    REQUIRE(ovspatialquery::detail::executeFamilyBatchCpu(first, plan, noHitLaunch, noHitResult) ==
            FamilyBatchOutcome::success);
    REQUIRE(noHitResult.common.sourcePopulation == firstPopulation);
    REQUIRE(noHitResult.common.sourceOrdinal == 81);
    REQUIRE(noHitResult.common.truth == HitTruth::exactSurface);
    REQUIRE(noHitResult.hits.empty());

    first  = {};
    second = {};
    std::shared_ptr<void> retired;
    REQUIRE(populations.release(firstPopulation, retired) == OVSPATIALQUERY_API_SUCCESS);
    retired.reset();
    REQUIRE(populations.release(secondPopulation, retired) == OVSPATIALQUERY_API_SUCCESS);
}

void testExactRayMultipleCarriesFiveFamilySemantics(ovspatialquery_backend_runtime_t* runtime)
{
    PopulationTable populations(runtime);
    PopulationKey   population;
    REQUIRE(populations.create(2u, 31, fiveExactFamilies(), population) == OVSPATIALQUERY_API_SUCCESS);
    ResolvedTarget resolved = makeResolvedTarget(populations, population);

    RaycastBatchPlan plan{};
    plan.hitMode             = HitMode::multiple;
    plan.precision           = QueryPrecision::exact;
    plan.maximumQueryCount   = 1u;
    plan.maximumHitsPerQuery = 5u;
    plan.hitCapacity         = 5u;
    RaycastBatchLaunch launch{};
    launch.rows.push_back(ray(0.0, 0u));
    launch.rows[0].query.origin[1] = 0.75;
    launch.rows[0].maximumHits     = 5u;

    RaycastBatchResult result;
    REQUIRE(ovspatialquery::detail::executeFamilyBatchCpu(resolved, plan, launch, result) ==
            FamilyBatchOutcome::success);
    REQUIRE(result.common.truth == HitTruth::exactSurface);
    REQUIRE(result.common.sourcePopulation == population);
    REQUIRE(result.common.hitCounts == std::vector<uint64_t>{5u});
    REQUIRE(result.common.truncated == std::vector<uint8_t>{0u});
    REQUIRE(result.hits.size() == 5u);
    for (size_t index = 0u; index < 5u; ++index)
    {
        REQUIRE(result.hits[index].truth == HitTruth::exactSurface);
        REQUIRE(identity(result.hits[index]).objectId.value == index + 1u);
        REQUIRE(identity(result.hits[index]).geometry.value == (index + 1u) * 101u);
        REQUIRE(result.hits[index].exact.feature.kind ==
                (index == 4u ? GeometryFeatureKind::triangle : GeometryFeatureKind::none));
    }
    const ovspatialquery::detail::TriangleSourceFeature& triangle = result.hits[4].exact.feature.triangle;
    REQUIRE(triangle.sourceFace == 700u);
    const std::array<uint32_t, 3u> expectedCorners{0u, 2u, 3u};
    REQUIRE(triangle.sourceCorners == expectedCorners);
    const float barycentricSum = triangle.barycentrics[0] + triangle.barycentrics[1] + triangle.barycentrics[2];
    REQUIRE(std::abs(barycentricSum - 1.0f) < 1.0e-6f);
    REQUIRE(std::abs(triangle.barycentrics[0] - 0.125f) < 1.0e-6f);
    REQUIRE(std::abs(triangle.barycentrics[1] - 0.5f) < 1.0e-6f);
    REQUIRE(std::abs(triangle.barycentrics[2] - 0.375f) < 1.0e-6f);

    plan.precision = QueryPrecision::boundsCandidate;
    RaycastBatchResult candidateResult;
    REQUIRE(ovspatialquery::detail::executeFamilyBatchCpu(resolved, plan, launch, candidateResult) ==
            FamilyBatchOutcome::success);
    REQUIRE(candidateResult.hits.size() == 5u);
    for (size_t index = 0u; index < candidateResult.hits.size(); ++index)
    {
        REQUIRE(candidateResult.hits[index].truth == HitTruth::boundsCandidate);
        REQUIRE(identity(candidateResult.hits[index]).objectId.value == index + 1u);
        REQUIRE(identity(candidateResult.hits[index]).geometry == identity(result.hits[index]).geometry);
    }

    resolved = {};
    std::shared_ptr<void> retired;
    REQUIRE(populations.release(population, retired) == OVSPATIALQUERY_API_SUCCESS);
}

void testTypedFamilyModes(ovspatialquery_backend_runtime_t* runtime)
{
    PopulationTable populations(runtime);
    PopulationKey   population;
    REQUIRE(populations.create(2u, 41, fiveExactFamilies(), population) == OVSPATIALQUERY_API_SUCCESS);
    ResolvedTarget resolved = makeResolvedTarget(populations, population);

    for (HitMode mode : {HitMode::closest, HitMode::any})
    {
        RaycastBatchPlan rayPlan{};
        rayPlan.hitMode           = mode;
        rayPlan.precision         = QueryPrecision::exact;
        rayPlan.maximumQueryCount = 1u;
        rayPlan.hitCapacity       = 1u;
        RaycastBatchLaunch rayLaunch{};
        rayLaunch.rows.push_back(ray(0.0, 0u));
        RaycastBatchResult rayResult;
        REQUIRE(ovspatialquery::detail::executeFamilyBatchCpu(resolved, rayPlan, rayLaunch, rayResult) ==
                FamilyBatchOutcome::success);
        REQUIRE(rayResult.hits.size() == 1u);
        REQUIRE(rayResult.common.sourcePopulation == population);
        REQUIRE(identity(rayResult.hits[0]).objectId.value == 1u);
    }

    SphereSweepBatchPlan sweepPlan{};
    sweepPlan.precision         = QueryPrecision::exact;
    sweepPlan.maximumQueryCount = 1u;
    sweepPlan.hitCapacity       = 1u;
    SphereSweepBatchLaunch sweepLaunch{};
    sweepLaunch.rows.push_back(sweep(0.0, 0u));
    SphereSweepBatchResult sweepResult;
    REQUIRE(ovspatialquery::detail::executeFamilyBatchCpu(resolved, sweepPlan, sweepLaunch, sweepResult) ==
            FamilyBatchOutcome::success);
    REQUIRE(sweepResult.hits.size() == 1u);
    REQUIRE(sweepResult.common.sourcePopulation == population);
    REQUIRE(identity(sweepResult.hits[0]).objectId.value == 1u);
    REQUIRE(sweepResult.hits[0].exact.feature.kind == GeometryFeatureKind::none);
    REQUIRE(sweepResult.hits[0].exact.queryPosition[0] < sweepResult.hits[0].exact.objectPosition[0]);

    for (HitMode mode : {HitMode::any, HitMode::multiple})
    {
        OverlapBatchPlan overlapPlan{};
        overlapPlan.hitMode             = mode;
        overlapPlan.precision           = QueryPrecision::exact;
        overlapPlan.maximumQueryCount   = 1u;
        overlapPlan.maximumHitsPerQuery = 5u;
        overlapPlan.hitCapacity         = 5u;
        OverlapBatchLaunch overlapLaunch{};
        overlapLaunch.rows.push_back(overlapAll(0u));
        overlapLaunch.rows[0].maximumHits = 5u;
        OverlapBatchResult overlapResult;
        REQUIRE(ovspatialquery::detail::executeFamilyBatchCpu(resolved, overlapPlan, overlapLaunch, overlapResult) ==
                FamilyBatchOutcome::success);
        const size_t expectedCount = mode == HitMode::any ? 1u : 5u;
        REQUIRE(overlapResult.hits.size() == expectedCount);
        REQUIRE(overlapResult.common.sourcePopulation == population);
        for (const ovspatialquery::detail::OverlapSemanticHit& hit : overlapResult.hits)
        {
            REQUIRE(hit.truth == HitTruth::exactSurface);
            REQUIRE(hit.identity.geometry.value != 0u);
        }
    }

    resolved = {};
    std::shared_ptr<void> retired;
    REQUIRE(populations.release(population, retired) == OVSPATIALQUERY_API_SUCCESS);
}

void testPartitionsCapacityRowsAndAtomicFailure(ovspatialquery_backend_runtime_t* runtime)
{
    PopulationTable populations(runtime);
    PopulationKey   population;
    REQUIRE(populations.create(2u, 51, fiveExactFamilies(), population) == OVSPATIALQUERY_API_SUCCESS);
    ResolvedTarget   resolved = makeResolvedTarget(populations, population);
    RaycastBatchPlan plan{};
    plan.hitMode             = HitMode::multiple;
    plan.precision           = QueryPrecision::exact;
    plan.maximumQueryCount   = 3u;
    plan.maximumHitsPerQuery = 3u;
    plan.hitCapacity         = 4u;
    RaycastBatchLaunch launch{};
    launch.rows.push_back(ray(0.0, 0u));
    launch.rows.back().maximumHits = 2u;
    launch.rows.push_back(ray(0.0, 0u));
    launch.rows.back().query.unitDirection = {2.0f, 0.0f, 0.0f};
    launch.rows.push_back(ray(0.0, 0u));
    launch.rows.back().maximumHits = 3u;

    RaycastBatchResult first;
    REQUIRE(ovspatialquery::detail::executeFamilyBatchCpu(resolved, plan, launch, first) ==
            FamilyBatchOutcome::success);
    const std::vector<RowOutcome> expectedOutcomes{RowOutcome::success, RowOutcome::invalidArgument,
                                                   RowOutcome::success};
    REQUIRE(first.common.rowOutcomes == expectedOutcomes);
    REQUIRE(first.common.hitOffsets == std::vector<uint64_t>({0u, 2u, 2u}));
    REQUIRE(first.common.hitCounts == std::vector<uint64_t>({2u, 0u, 2u}));
    REQUIRE(first.common.truncated == std::vector<uint8_t>({1u, 0u, 1u}));
    REQUIRE(first.common.storedHitCount == first.common.hitCapacity);
    REQUIRE(first.hits.size() == first.common.storedHitCount);

    RaycastBatchResult repeated;
    REQUIRE(ovspatialquery::detail::executeFamilyBatchCpu(resolved, plan, launch, repeated) ==
            FamilyBatchOutcome::success);
    REQUIRE(repeated.common.hitOffsets == first.common.hitOffsets);
    REQUIRE(repeated.common.hitCounts == first.common.hitCounts);
    REQUIRE(repeated.common.truncated == first.common.truncated);
    REQUIRE(repeated.hits.size() == first.hits.size());
    for (size_t index = 0u; index < first.hits.size(); ++index)
    {
        REQUIRE(identity(repeated.hits[index]).objectId.value == identity(first.hits[index]).objectId.value);
    }

    RaycastBatchResult preserved    = first;
    ResolvedTarget     inconsistent = makeResolvedTarget(populations, population);
    inconsistent.ordinal += 1;
    REQUIRE(ovspatialquery::detail::executeFamilyBatchCpu(inconsistent, plan, launch, first) ==
            FamilyBatchOutcome::invalidArgument);
    REQUIRE(first.common.sourceOrdinal == preserved.common.sourceOrdinal);
    REQUIRE(first.common.sourcePopulation == preserved.common.sourcePopulation);
    REQUIRE(first.common.hitCounts == preserved.common.hitCounts);
    REQUIRE(first.hits.size() == preserved.hits.size());

    ResolvedTarget missingPopulation = makeResolvedTarget(populations, population);
    missingPopulation.population     = {};
    REQUIRE(ovspatialquery::detail::executeFamilyBatchCpu(missingPopulation, plan, launch, first) ==
            FamilyBatchOutcome::invalidArgument);
    REQUIRE(first.common.sourcePopulation == preserved.common.sourcePopulation);
    REQUIRE(first.common.hitCounts == preserved.common.hitCounts);
    REQUIRE(first.hits.size() == preserved.hits.size());

    SphereSweepBatchPlan sweepPlan{};
    sweepPlan.maximumQueryCount = 1u;
    sweepPlan.hitCapacity       = 1u;
    SphereSweepBatchLaunch sweepLaunch{};
    sweepLaunch.rows.push_back(sweep(0.0, 0u));
    SphereSweepBatchResult sweepResult{};
    sweepResult.common.sourceOrdinal = 701;
    sweepResult.hits.resize(1u);
    REQUIRE(ovspatialquery::detail::executeFamilyBatchCpu(inconsistent, sweepPlan, sweepLaunch, sweepResult) ==
            FamilyBatchOutcome::invalidArgument);
    REQUIRE(sweepResult.common.sourceOrdinal == 701);
    REQUIRE(sweepResult.hits.size() == 1u);

    OverlapBatchPlan overlapPlan{};
    overlapPlan.maximumQueryCount = 1u;
    overlapPlan.hitCapacity       = 1u;
    OverlapBatchLaunch overlapLaunch{};
    overlapLaunch.rows.push_back(overlapAll(0u));
    OverlapBatchResult overlapResult{};
    overlapResult.common.sourceOrdinal = 702;
    overlapResult.hits.resize(1u);
    REQUIRE(ovspatialquery::detail::executeFamilyBatchCpu(inconsistent, overlapPlan, overlapLaunch, overlapResult) ==
            FamilyBatchOutcome::invalidArgument);
    REQUIRE(overlapResult.common.sourceOrdinal == 702);
    REQUIRE(overlapResult.hits.size() == 1u);

    missingPopulation = {};
    inconsistent      = {};
    resolved          = {};
    std::shared_ptr<void> retired;
    REQUIRE(populations.release(population, retired) == OVSPATIALQUERY_API_SUCCESS);
}

template <typename Plan, typename Launch, typename Result>
void requireOversizedCapacityIsResourceExhaustionAndAtomic(const ResolvedTarget& resolved)
{
    Plan plan{};
    plan.hitCapacity = UINT64_MAX;
    Launch              launch{};
    Result              result{};
    const PopulationKey sentinelPopulation{91u, 7u};
    result.common.sourcePopulation = sentinelPopulation;
    result.common.sourceOrdinal    = 701;
    result.common.storedHitCount   = 9u;
    result.common.rowOutcomes.push_back(RowOutcome::notSupported);
    result.hits.resize(1u);
    REQUIRE(ovspatialquery::detail::executeFamilyBatchCpu(resolved, plan, launch, result) ==
            FamilyBatchOutcome::outOfMemory);
    REQUIRE(result.common.sourcePopulation == sentinelPopulation);
    REQUIRE(result.common.sourceOrdinal == 701);
    REQUIRE(result.common.storedHitCount == 9u);
    REQUIRE(result.common.rowOutcomes == std::vector<RowOutcome>{RowOutcome::notSupported});
    REQUIRE(result.hits.size() == 1u);
}

void testOversizedTypedCapacityIsResourceExhaustionAndAtomic(ovspatialquery_backend_runtime_t* runtime)
{
    PopulationTable populations(runtime);
    PopulationKey   population;
    REQUIRE(populations.create(2u, 52, {sphereObject(1u, 2.0, UINT64_MAX, 101u)}, population) ==
            OVSPATIALQUERY_API_SUCCESS);
    ResolvedTarget resolved = makeResolvedTarget(populations, population);

    requireOversizedCapacityIsResourceExhaustionAndAtomic<RaycastBatchPlan, RaycastBatchLaunch, RaycastBatchResult>(
        resolved);
    requireOversizedCapacityIsResourceExhaustionAndAtomic<SphereSweepBatchPlan, SphereSweepBatchLaunch,
                                                          SphereSweepBatchResult>(resolved);
    requireOversizedCapacityIsResourceExhaustionAndAtomic<OverlapBatchPlan, OverlapBatchLaunch, OverlapBatchResult>(
        resolved);

    resolved = {};
    std::shared_ptr<void> retired;
    REQUIRE(populations.release(population, retired) == OVSPATIALQUERY_API_SUCCESS);
}

void testZeroRowsZeroHitsAndIsolatedUnsupportedRow(ovspatialquery_backend_runtime_t* runtime)
{
    PopulationTable               populations(runtime);
    PopulationKey                 population;
    std::vector<PopulationObject> objects;
    objects.push_back(object(10u, 2.0, 3.0, UINT64_C(0x1), 10u));
    std::vector<PopulationObject> exact = fiveExactFamilies();
    exact[0].participationBits          = {{UINT64_C(0x2), 0u}};
    objects.push_back(exact[0]);
    REQUIRE(populations.create(2u, 61, std::move(objects), population) == OVSPATIALQUERY_API_SUCCESS);
    ResolvedTarget resolved = makeResolvedTarget(populations, population);

    RaycastBatchPlan   emptyPlan{};
    RaycastBatchLaunch emptyLaunch{};
    RaycastBatchResult emptyResult;
    REQUIRE(ovspatialquery::detail::executeFamilyBatchCpu(resolved, emptyPlan, emptyLaunch, emptyResult) ==
            FamilyBatchOutcome::success);
    REQUIRE(emptyResult.common.sourceOrdinal == 61);
    REQUIRE(emptyResult.common.sourcePopulation == population);
    REQUIRE(emptyResult.common.rowOutcomes.empty());
    REQUIRE(emptyResult.hits.empty());

    RaycastBatchPlan plan{};
    plan.precision         = QueryPrecision::exact;
    plan.maximumQueryCount = 3u;
    plan.hitCapacity       = 2u;
    RaycastBatchLaunch launch{};
    launch.rows.push_back(ray(0.0, UINT64_C(0x1)));
    launch.rows.push_back(ray(0.0, UINT64_C(0x2)));
    launch.rows.push_back(ray(100.0, UINT64_C(0x2)));
    RaycastBatchResult result;
    REQUIRE(ovspatialquery::detail::executeFamilyBatchCpu(resolved, plan, launch, result) ==
            FamilyBatchOutcome::success);
    const std::vector<RowOutcome> outcomes{RowOutcome::notSupported, RowOutcome::success, RowOutcome::success};
    REQUIRE(result.common.rowOutcomes == outcomes);
    REQUIRE(result.common.sourcePopulation == population);
    REQUIRE(result.common.hitOffsets == std::vector<uint64_t>({0u, 0u, 1u}));
    REQUIRE(result.common.hitCounts == std::vector<uint64_t>({0u, 1u, 0u}));
    REQUIRE(result.common.truncated == std::vector<uint8_t>({0u, 0u, 0u}));
    REQUIRE(result.hits.size() == 1u);

    resolved = {};
    std::shared_ptr<void> retired;
    REQUIRE(populations.release(population, retired) == OVSPATIALQUERY_API_SUCCESS);
}

void testResolvedRetentionMembershipAndParticipation(ovspatialquery_backend_runtime_t* runtime)
{
    PopulationTable populations(runtime);
    PopulationKey   population;
    REQUIRE(populations.create(1u, population) == OVSPATIALQUERY_API_SUCCESS);
    const ovspatialquery_view_handle_t                       viewHandle = ovspatialquery_view_handle_t{900u};
    const std::shared_ptr<ovspatialquery::detail::ViewState> view =
        std::make_shared<ovspatialquery::detail::ViewState>();
    view->handle    = viewHandle;
    view->parent    = population;
    view->automatic = true;

    publish(populations, population, 10, {sphereObject(101u, 2.0, UINT64_C(0x1), 1001u)},
            {{viewHandle, selected({101u})}});
    publish(populations, population, 20, {sphereObject(202u, 4.0, UINT64_C(0x2), 2002u)}, {{viewHandle, selected({})}});

    const TargetResolver                           resolver(populations);
    const ovspatialquery::detail::PopulationTarget viewTarget{population, view};
    ResolvedTarget                                 retained;
    REQUIRE(resolver.resolve({viewTarget, {TargetSelectionKind::retainedExact, 10}}, retained) ==
            TargetResolveOutcome::success);
    ResolvedTarget head;
    REQUIRE(resolver.resolve({viewTarget, {TargetSelectionKind::head, 0}}, head) == TargetResolveOutcome::success);

    RaycastBatchPlan plan{};
    plan.precision         = QueryPrecision::exact;
    plan.maximumQueryCount = 1u;
    plan.hitCapacity       = 1u;
    RaycastBatchLaunch retainedLaunch{};
    retainedLaunch.rows.push_back(ray(0.0, UINT64_C(0x1)));
    RaycastBatchResult retainedResult;
    REQUIRE(ovspatialquery::detail::executeFamilyBatchCpu(retained, plan, retainedLaunch, retainedResult) ==
            FamilyBatchOutcome::success);
    REQUIRE(retainedResult.common.sourceOrdinal == 10);
    REQUIRE(retainedResult.common.sourcePopulation == population);
    REQUIRE(retainedResult.hits.size() == 1u);
    REQUIRE(identity(retainedResult.hits[0]).objectId.value == 101u);

    RaycastBatchLaunch headLaunch{};
    headLaunch.rows.push_back(ray(0.0, 0u));
    RaycastBatchResult headResult;
    REQUIRE(ovspatialquery::detail::executeFamilyBatchCpu(head, plan, headLaunch, headResult) ==
            FamilyBatchOutcome::success);
    REQUIRE(headResult.common.sourceOrdinal == 20);
    REQUIRE(headResult.common.sourcePopulation == population);
    REQUIRE(headResult.hits.empty());

    publish(populations, population, 30, {sphereObject(303u, 6.0, UINT64_C(0x4), 3003u)},
            {{viewHandle, selected({303u})}});
    RaycastBatchResult heldResult;
    REQUIRE(ovspatialquery::detail::executeFamilyBatchCpu(retained, plan, retainedLaunch, heldResult) ==
            FamilyBatchOutcome::success);
    REQUIRE(heldResult.common.sourceOrdinal == 10);
    REQUIRE(heldResult.common.sourcePopulation == population);
    REQUIRE(heldResult.hits.size() == 1u);

    ResolvedTarget evicted;
    REQUIRE(resolver.resolve({viewTarget, {TargetSelectionKind::retainedExact, 10}}, evicted) ==
            TargetResolveOutcome::ordinalNotRetained);
    REQUIRE(!evicted.generation.initialized());

    retained = {};
    head     = {};
    std::shared_ptr<void> retired;
    REQUIRE(populations.release(population, retired) == OVSPATIALQUERY_API_SUCCESS);
}

void testExactEligibilityTransitionsPreserveBatchFallbacks(ovspatialquery_backend_runtime_t* runtime)
{
    PopulationTable populations(runtime);
    PopulationKey   population;
    REQUIRE(populations.create(4u, 0, {}, population) == OVSPATIALQUERY_API_SUCCESS);
    const ovspatialquery_view_handle_t                       viewHandle = ovspatialquery_view_handle_t{901u};
    const std::shared_ptr<ovspatialquery::detail::ViewState> view =
        std::make_shared<ovspatialquery::detail::ViewState>();
    view->handle    = viewHandle;
    view->parent    = population;
    view->automatic = true;
    const TargetResolver                           resolver(populations);
    const ovspatialquery::detail::PopulationTarget populationTarget{population, {}};
    const ovspatialquery::detail::PopulationTarget viewTarget{population, view};
    RaycastBatchPlan                               plan{};
    plan.precision         = QueryPrecision::exact;
    plan.maximumQueryCount = 2u;
    plan.hitCapacity       = 2u;
    RaycastBatchLaunch launch{};
    launch.rows.push_back(ray(0.0, UINT64_C(0x1)));

    ResolvedTarget resolved;
    REQUIRE(resolver.resolve({populationTarget, {TargetSelectionKind::head, 0}}, resolved) ==
            TargetResolveOutcome::success);
    RaycastBatchResult result;
    REQUIRE(ovspatialquery::detail::executeFamilyBatchCpu(resolved, plan, launch, result) ==
            FamilyBatchOutcome::success);
    REQUIRE(result.common.sourceOrdinal == 0 && result.hits.empty());

    publish(populations, population, 10, {sphereObject(101u, 2.0, UINT64_C(0x1), 1001u)},
            {{viewHandle, selected({101u})}});
    REQUIRE(resolver.resolve({populationTarget, {TargetSelectionKind::head, 0}}, resolved) ==
            TargetResolveOutcome::success);
    REQUIRE(ovspatialquery::detail::executeFamilyBatchCpu(resolved, plan, launch, result) ==
            FamilyBatchOutcome::success);
    REQUIRE(result.common.sourceOrdinal == 10 && result.hits.size() == 1u &&
            identity(result.hits[0]).objectId.value == 101u);

    publish(populations, population, 20,
            {sphereObject(202u, 4.0, UINT64_C(0x1), 2002u), object(203u, 5.0, 7.0, UINT64_C(0x2), 2003u)},
            {{viewHandle, selected({202u})}});
    REQUIRE(resolver.resolve({populationTarget, {TargetSelectionKind::head, 0}}, resolved) ==
            TargetResolveOutcome::success);
    launch.rows.clear();
    launch.rows.push_back(ray(0.0, UINT64_C(0x1)));
    launch.rows.push_back(ray(0.0, UINT64_C(0x2)));
    REQUIRE(ovspatialquery::detail::executeFamilyBatchCpu(resolved, plan, launch, result) ==
            FamilyBatchOutcome::success);
    REQUIRE((result.common.sourceOrdinal == 20 &&
             result.common.rowOutcomes == std::vector<RowOutcome>{RowOutcome::success, RowOutcome::notSupported}));
    REQUIRE(result.hits.size() == 1u && identity(result.hits[0]).objectId.value == 202u);

    REQUIRE(resolver.resolve({viewTarget, {TargetSelectionKind::head, 0}}, resolved) == TargetResolveOutcome::success);
    launch.rows = {ray(0.0, UINT64_C(0x2))};
    REQUIRE(ovspatialquery::detail::executeFamilyBatchCpu(resolved, plan, launch, result) ==
            FamilyBatchOutcome::success);
    REQUIRE((result.common.sourceOrdinal == 20 &&
             result.common.rowOutcomes == std::vector<RowOutcome>{RowOutcome::success} && result.hits.empty()));

    REQUIRE(resolver.resolve({populationTarget, {TargetSelectionKind::retainedExact, 10}}, resolved) ==
            TargetResolveOutcome::success);
    launch.rows = {ray(0.0, UINT64_C(0x1))};
    REQUIRE(ovspatialquery::detail::executeFamilyBatchCpu(resolved, plan, launch, result) ==
            FamilyBatchOutcome::success);
    REQUIRE(result.common.sourceOrdinal == 10 && result.hits.size() == 1u &&
            identity(result.hits[0]).objectId.value == 101u);

    publish(populations, population, 30, {sphereObject(303u, 8.0, UINT64_C(0x4), 3003u)},
            {{viewHandle, selected({303u})}});
    REQUIRE(resolver.resolve({populationTarget, {TargetSelectionKind::head, 0}}, resolved) ==
            TargetResolveOutcome::success);
    launch.rows = {ray(0.0, UINT64_C(0x4))};
    REQUIRE(ovspatialquery::detail::executeFamilyBatchCpu(resolved, plan, launch, result) ==
            FamilyBatchOutcome::success);
    REQUIRE(result.common.sourceOrdinal == 30 && result.hits.size() == 1u &&
            identity(result.hits[0]).objectId.value == 303u);

    REQUIRE(resolver.resolve({populationTarget, {TargetSelectionKind::retainedExact, 20}}, resolved) ==
            TargetResolveOutcome::success);
    launch.rows = {ray(0.0, UINT64_C(0x2))};
    REQUIRE(ovspatialquery::detail::executeFamilyBatchCpu(resolved, plan, launch, result) ==
            FamilyBatchOutcome::success);
    REQUIRE((result.common.rowOutcomes == std::vector<RowOutcome>{RowOutcome::notSupported}));

    resolved = {};
    std::shared_ptr<void> retired;
    REQUIRE(populations.release(population, retired) == OVSPATIALQUERY_API_SUCCESS);
}

} // namespace

int main()
{
    ovspatialquery_backend_runtime_t* runtime = nullptr;
    REQUIRE(ovspatialquery_test_oracle_create_runtime(&runtime) == OVSPATIALQUERY_API_SUCCESS);
    testResolvedCandidatePreservesIdentityAndParticipation(runtime);
    testPopulationQualifiesAllFamilyCandidateAndExactIdentity(runtime);
    testExactRayMultipleCarriesFiveFamilySemantics(runtime);
    testTypedFamilyModes(runtime);
    testPartitionsCapacityRowsAndAtomicFailure(runtime);
    testOversizedTypedCapacityIsResourceExhaustionAndAtomic(runtime);
    testZeroRowsZeroHitsAndIsolatedUnsupportedRow(runtime);
    testResolvedRetentionMembershipAndParticipation(runtime);
    testExactEligibilityTransitionsPreserveBatchFallbacks(runtime);
    ovspatialquery_test_oracle_destroy_runtime(runtime);
    return 0;
}
