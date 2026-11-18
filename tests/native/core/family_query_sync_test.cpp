/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "oracle_backend.h"
#include "ovspatialquery_family_batch_cpu.h"
#include "ovspatialquery_instance_host_ledger.h"
#include "ovspatialquery_typed_geometry.h"
#include "test_failure.h"

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <new>
#include <utility>
#include <vector>

namespace
{
// Live and peak bytes held through operator new, so a test can compare what a
// query really holds with what its admission charged.
constexpr std::size_t    kAllocationHeader = alignof(std::max_align_t);
std::atomic<std::size_t> gLiveBytes{0u};
std::atomic<std::size_t> gPeakBytes{0u};
} // namespace

// Do not inline the malloc/free replacements into unrelated new expressions:
// GCC otherwise diagnoses their matched storage as mismatched new/delete.
#if defined(__GNUC__)
__attribute__((noinline))
#endif
void* operator new(std::size_t size)
{
    void* block = std::malloc(size + kAllocationHeader);
    if (block == nullptr)
    {
        throw std::bad_alloc();
    }
    *static_cast<std::size_t*>(block) = size;
    const std::size_t live            = gLiveBytes.fetch_add(size) + size;
    std::size_t       peak            = gPeakBytes.load();
    while (live > peak && !gPeakBytes.compare_exchange_weak(peak, live)) {}
    return static_cast<unsigned char*>(block) + kAllocationHeader;
}

#if defined(__GNUC__)
__attribute__((noinline))
#endif
void operator delete(void* memory) noexcept
{
    if (memory == nullptr)
    {
        return;
    }
    void* block = static_cast<unsigned char*>(memory) - kAllocationHeader;
    gLiveBytes.fetch_sub(*static_cast<std::size_t*>(block));
    std::free(block);
}

#if defined(__GNUC__)
__attribute__((noinline))
#endif
void operator delete(void* memory, std::size_t) noexcept
{
    ::operator delete(memory);
}

namespace
{

using ovspatialquery::detail::ExactGeometry;
using ovspatialquery::detail::ExactGeometryFamily;
using ovspatialquery::detail::GeometryFeatureKind;
using ovspatialquery::detail::GeometryReference;
using ovspatialquery::detail::HitTruth;
using ovspatialquery::detail::InstanceHostLedger;
using ovspatialquery::detail::PopulationKey;
using ovspatialquery::detail::PopulationObject;
using ovspatialquery::detail::PopulationTable;
using ovspatialquery::detail::QueryPrecision;
using ovspatialquery::detail::TargetResolver;
using ovspatialquery::detail::TargetSelectionKind;

struct Common
{
    ovspatialquery::detail::TargetRequest target{};
    QueryPrecision                        precision        = QueryPrecision::boundsCandidate;
    uint64_t                              queryBits        = 0u;
    double                                contactTolerance = 0.0;
};

struct RaycastSyncRequest
{
    Common                               common{};
    ovspatialquery::detail::RaycastQuery query{};
};

struct SphereSweepSyncRequest
{
    Common                                   common{};
    ovspatialquery::detail::SphereSweepQuery query{};
};

struct OverlapSyncRequest
{
    Common                               common{};
    ovspatialquery::detail::OverlapQuery query{};
};

// The resolved target and the evaluated hits of one synchronous query.
template <typename Hit>
struct Rows
{
    ovspatialquery::detail::ResolvedTarget target;
    std::vector<Hit>                       hits;
};

using RaycastRows = Rows<ovspatialquery::detail::RaycastSemanticHit>;
using SweepRows   = Rows<ovspatialquery::detail::SphereSweepSemanticHit>;
using OverlapRows = Rows<ovspatialquery::detail::OverlapSemanticHit>;

// Runs a request as the one-row plan the public synchronous entry points use.
template <typename Row, typename Plan, typename Request, typename Hit>
ovspatialquery_api_status_t run(const TargetResolver& resolver, const Request& request, Plan plan, Rows<Hit>& out,
                                InstanceHostLedger* ledger)
{
    plan.precision        = request.common.precision;
    plan.contactTolerance = request.common.contactTolerance;
    plan.hostLedger       = ledger;
    Row row{};
    row.query     = request.query;
    row.queryBits = {{request.common.queryBits, 0u}};
    return ovspatialquery::detail::executeFamilyRowCpu(resolver, request.common.target, plan, row, out.target,
                                                       out.hits);
}

template <typename Plan>
Plan modePlan(ovspatialquery::detail::HitMode mode)
{
    Plan plan{};
    plan.hitMode             = mode;
    plan.maximumHitsPerQuery = UINT64_MAX;
    return plan;
}

ovspatialquery_api_status_t runClosest(const TargetResolver& resolver, const RaycastSyncRequest& request,
                                       RaycastRows& out, InstanceHostLedger* ledger = nullptr)
{
    return run<ovspatialquery::detail::RaycastBatchRow>(
        resolver, request, modePlan<ovspatialquery::detail::RaycastBatchPlan>(ovspatialquery::detail::HitMode::closest),
        out, ledger);
}

ovspatialquery_api_status_t runAny(const TargetResolver& resolver, const RaycastSyncRequest& request, RaycastRows& out)
{
    return run<ovspatialquery::detail::RaycastBatchRow>(
        resolver, request, modePlan<ovspatialquery::detail::RaycastBatchPlan>(ovspatialquery::detail::HitMode::any),
        out, nullptr);
}

ovspatialquery_api_status_t runMultiple(const TargetResolver& resolver, const RaycastSyncRequest& request,
                                        RaycastRows& out)
{
    return run<ovspatialquery::detail::RaycastBatchRow>(
        resolver, request,
        modePlan<ovspatialquery::detail::RaycastBatchPlan>(ovspatialquery::detail::HitMode::multiple), out, nullptr);
}

ovspatialquery_api_status_t runClosest(const TargetResolver& resolver, const SphereSweepSyncRequest& request,
                                       SweepRows& out)
{
    return run<ovspatialquery::detail::SphereSweepBatchRow>(
        resolver, request, ovspatialquery::detail::SphereSweepBatchPlan{}, out, nullptr);
}

ovspatialquery_api_status_t runAny(const TargetResolver& resolver, const OverlapSyncRequest& request, OverlapRows& out)
{
    return run<ovspatialquery::detail::OverlapBatchRow>(
        resolver, request, modePlan<ovspatialquery::detail::OverlapBatchPlan>(ovspatialquery::detail::HitMode::any),
        out, nullptr);
}

ovspatialquery_api_status_t runMultiple(const TargetResolver& resolver, const OverlapSyncRequest& request,
                                        OverlapRows& out)
{
    return run<ovspatialquery::detail::OverlapBatchRow>(
        resolver, request,
        modePlan<ovspatialquery::detail::OverlapBatchPlan>(ovspatialquery::detail::HitMode::multiple), out, nullptr);
}

PopulationObject exactObject(uint64_t objectId, uint64_t geometryValue, std::shared_ptr<const ExactGeometry> geometry,
                             uint64_t participationBits)
{
    PopulationObject object{};
    object.id.value          = objectId;
    object.bounds            = geometry->worldBounds();
    object.geometryReference = GeometryReference{geometry->family(), geometryValue};
    object.exactGeometry     = std::move(geometry);
    object.participationBits = {{participationBits, 0u}};
    return object;
}

PopulationObject sphereObject(uint64_t objectId, double centerX, uint64_t geometryValue, uint64_t participationBits)
{
    ovspatialquery::detail::SphereGeometrySource source{};
    source.center = {centerX, 0.0, 0.0};
    source.radius = 1.0;
    std::shared_ptr<const ExactGeometry> geometry;
    REQUIRE(ovspatialquery::detail::createSphereExactGeometry(source, geometry) == OVSPATIALQUERY_API_SUCCESS);
    return exactObject(objectId, geometryValue, std::move(geometry), participationBits);
}

PopulationObject missingExactObject(uint64_t objectId, double centerX, uint64_t geometryValue,
                                    uint64_t participationBits)
{
    PopulationObject object{};
    object.id.value          = objectId;
    object.bounds            = {{centerX - 1.0, -1.0, -1.0}, {centerX + 1.0, 1.0, 1.0}};
    object.geometryReference = {ExactGeometryFamily::sphere, geometryValue};
    object.participationBits = {{participationBits, 0u}};
    return object;
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

std::shared_ptr<const ExactGeometry> meshGeometry()
{
    ovspatialquery::detail::TriangleMeshSource mesh{};
    mesh.key      = 77u;
    mesh.vertices = {
        14.0, -1.0, -1.0, 14.0, -1.0, 1.0, 14.0, 1.0, 1.0, 14.0, 1.0, -1.0,
    };
    mesh.triangles     = {0u, 1u, 2u, 0u, 2u, 3u};
    mesh.sourceFaces   = {700u, 700u};
    mesh.sourceCorners = {0u, 1u, 2u, 0u, 2u, 3u};
    std::shared_ptr<const ovspatialquery::detail::PreparedTriangleMesh> preparedMesh;
    REQUIRE(ovspatialquery::detail::prepareTriangleMesh(mesh, preparedMesh) == OVSPATIALQUERY_API_SUCCESS);
    std::shared_ptr<const ovspatialquery::detail::TriangleObjectGeometry> prepared;
    REQUIRE(ovspatialquery::detail::bindPreparedTriangleGeometry(
                preparedMesh, {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0}, true,
                prepared) == OVSPATIALQUERY_API_SUCCESS);
    std::shared_ptr<const ExactGeometry> geometry;
    REQUIRE(ovspatialquery::detail::createTriangleMeshExactGeometry(prepared, geometry) == OVSPATIALQUERY_API_SUCCESS);
    return geometry;
}

std::vector<PopulationObject> fiveExactFamilies()
{
    std::vector<PopulationObject> objects;
    objects.push_back(sphereObject(1u, 2.0, 101u, UINT64_C(0x1)));

    ovspatialquery::detail::BoxGeometrySource box{};
    box.center = {5.0, 0.0, 0.0};
    box.axes   = {
        std::array<double, 3u>{1.0, 0.0, 0.0},
        std::array<double, 3u>{0.0, 1.0, 0.0},
        std::array<double, 3u>{0.0, 0.0, 1.0},
    };
    box.halfExtents = {1.0, 1.0, 1.0};
    std::shared_ptr<const ExactGeometry> geometry;
    REQUIRE(ovspatialquery::detail::createBoxExactGeometry(box, geometry) == OVSPATIALQUERY_API_SUCCESS);
    objects.push_back(exactObject(2u, 202u, geometry, UINT64_C(0x2)));

    ovspatialquery::detail::CapsuleGeometrySource capsule{};
    capsule.endpointA = {8.0, -1.0, 0.0};
    capsule.endpointB = {8.0, 1.0, 0.0};
    capsule.radius    = 1.0;
    REQUIRE(ovspatialquery::detail::createCapsuleExactGeometry(capsule, geometry) == OVSPATIALQUERY_API_SUCCESS);
    objects.push_back(exactObject(3u, 303u, geometry, UINT64_C(0x4)));

    REQUIRE(ovspatialquery::detail::createConvexExactGeometry(translatedCube(11.0), geometry) ==
            ovspatialquery::detail::ConvexConstructionOutcome::success);
    objects.push_back(exactObject(4u, 404u, geometry, UINT64_C(0x8)));
    objects.push_back(exactObject(5u, 505u, meshGeometry(), UINT64_C(0x10)));
    return objects;
}

Common common(PopulationKey population, QueryPrecision precision, uint64_t queryBits = 0u,
              double contactTolerance = 0.0)
{
    Common value{};
    value.target           = {{population, {}}, {TargetSelectionKind::head, 0}};
    value.precision        = precision;
    value.queryBits        = queryBits;
    value.contactTolerance = contactTolerance;
    return value;
}

RaycastSyncRequest rayRequest(PopulationKey population, QueryPrecision precision, uint64_t queryBits = 0u)
{
    RaycastSyncRequest request{};
    request.common                = common(population, precision, queryBits);
    request.query.origin          = {0.0, 0.0, 0.0};
    request.query.unitDirection   = {1.0f, 0.0f, 0.0f};
    request.query.minimumDistance = 0.0;
    request.query.maximumDistance = 20.0;
    return request;
}

OverlapSyncRequest overlapAround(PopulationKey population, QueryPrecision precision, double centerX, uint64_t queryBits)
{
    OverlapSyncRequest request{};
    request.common                                 = common(population, precision, queryBits);
    request.query.geometry.kind                    = OVSPATIALQUERY_OVERLAP_AABB;
    request.query.geometry.data.aabb.min_corner[0] = centerX - 2.1;
    request.query.geometry.data.aabb.max_corner[0] = centerX + 2.1;
    request.query.geometry.data.aabb.min_corner[1] = -2.1;
    request.query.geometry.data.aabb.max_corner[1] = 2.1;
    request.query.geometry.data.aabb.min_corner[2] = -2.1;
    request.query.geometry.data.aabb.max_corner[2] = 2.1;
    return request;
}

const ovspatialquery::detail::SemanticHitIdentity& identity(const ovspatialquery::detail::RaycastSemanticHit& hit)
{
    return hit.truth == HitTruth::exactSurface ? hit.exact.identity : hit.boundsCandidate.identity;
}

const ovspatialquery::detail::SemanticHitIdentity& identity(const ovspatialquery::detail::SphereSweepSemanticHit& hit)
{
    return hit.truth == HitTruth::exactSurface ? hit.exact.identity : hit.boundsCandidate.identity;
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

void testExactRayContactToleranceBoundary(ovspatialquery_backend_runtime_t* runtime)
{
    PopulationTable populations(runtime);
    PopulationKey   population;
    REQUIRE(populations.create(2u, 17, {sphereObject(41u, 5.0, 410u, UINT64_MAX)}, population) ==
            OVSPATIALQUERY_API_SUCCESS);
    const TargetResolver resolver(populations);
    RaycastSyncRequest   request  = rayRequest(population, QueryPrecision::exact);
    request.query.origin          = {0.0, 1.25, 0.0};
    request.query.maximumDistance = 10.0;

    RaycastRows miss{};
    request.common.contactTolerance = 0.249;
    REQUIRE(runClosest(resolver, request, miss) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(miss.target.ordinal == 17 && miss.hits.empty());

    RaycastRows result{};
    request.common.contactTolerance = 0.25;
    REQUIRE(runClosest(resolver, request, result) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(result.target.population == population);
    REQUIRE(result.target.ordinal == 17);
    REQUIRE(!result.hits.empty());
    REQUIRE(result.hits[0].truth == HitTruth::exactSurface);
    REQUIRE(result.hits[0].exact.identity.objectId.value == 41u);
}

void testAllModesAndFiveExactFamilies(ovspatialquery_backend_runtime_t* runtime)
{
    PopulationTable populations(runtime);
    PopulationKey   population;
    REQUIRE(populations.create(2u, 31, fiveExactFamilies(), population) == OVSPATIALQUERY_API_SUCCESS);
    const TargetResolver                      resolver(populations);
    const std::array<double, 5u>              centers{2.0, 5.0, 8.0, 11.0, 14.0};
    const std::array<ExactGeometryFamily, 5u> families{
        ExactGeometryFamily::sphere, ExactGeometryFamily::box,          ExactGeometryFamily::capsule,
        ExactGeometryFamily::convex, ExactGeometryFamily::triangleMesh,
    };
    for (size_t index = 0u; index < families.size(); ++index)
    {
        const uint64_t     bits = UINT64_C(1) << index;
        RaycastSyncRequest ray  = rayRequest(population, QueryPrecision::exact, bits);
        RaycastRows        closest{};
        REQUIRE(runClosest(resolver, ray, closest) == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(!closest.hits.empty());
        REQUIRE(identity(closest.hits[0]).objectId.value == index + 1u);
        REQUIRE(identity(closest.hits[0]).geometry.family == families[index]);

        RaycastRows any{};
        REQUIRE(runAny(resolver, ray, any) == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(!any.hits.empty() && identity(any.hits[0]).objectId.value == index + 1u);

        RaycastRows rayMultiple{};
        REQUIRE(runMultiple(resolver, ray, rayMultiple) == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(rayMultiple.hits.size() == 1u);
        REQUIRE(identity(rayMultiple.hits[0]).objectId.value == index + 1u);

        SphereSweepSyncRequest sweep{};
        sweep.common                = common(population, QueryPrecision::exact, bits);
        sweep.query.center          = {0.0, 0.0, 0.0};
        sweep.query.radius          = 0.25;
        sweep.query.unitDirection   = {1.0f, 0.0f, 0.0f};
        sweep.query.minimumDistance = 0.0;
        sweep.query.maximumDistance = 20.0;
        SweepRows sweepResult{};
        REQUIRE(runClosest(resolver, sweep, sweepResult) == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(!sweepResult.hits.empty());
        REQUIRE(identity(sweepResult.hits[0]).objectId.value == index + 1u);
        REQUIRE(identity(sweepResult.hits[0]).geometry.family == families[index]);

        // The same target families must accept an oriented box sweep. These
        // axis-aligned contacts have an independent analytic time of impact.
        sweep.query.box         = true;
        sweep.query.halfExtents = {0.25, 0.5, 0.75};
        const double contact    = centers[index] - (index == 4u ? 0.0 : 1.0) - 0.25;
        REQUIRE(runClosest(resolver, sweep, sweepResult) == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(!sweepResult.hits.empty());
        REQUIRE(identity(sweepResult.hits[0]).objectId.value == index + 1u);
        REQUIRE(std::abs(sweepResult.hits[0].exact.distance - contact) < 1.0e-7);
        REQUIRE(std::abs(sweepResult.hits[0].exact.queryPosition[0] - contact) < 1.0e-7);
        REQUIRE(sweepResult.hits[0].exact.normal[0] < -0.99f);
        if (index == 4u)
        {
            REQUIRE(sweepResult.hits[0].exact.feature.kind == GeometryFeatureKind::triangle);
            REQUIRE(sweepResult.hits[0].exact.feature.triangle.sourceFace == 700u);
        }
        sweep.query.maximumDistance = contact - 0.001;
        REQUIRE(runClosest(resolver, sweep, sweepResult) == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(sweepResult.hits.empty());
        sweep.query.maximumDistance = 20.0;
        sweep.query.minimumDistance = contact + 0.01;
        REQUIRE(runClosest(resolver, sweep, sweepResult) == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(!sweepResult.hits.empty());
        REQUIRE(std::abs(sweepResult.hits[0].exact.distance - sweep.query.minimumDistance) < 1.0e-7);

        const OverlapSyncRequest overlap = overlapAround(population, QueryPrecision::exact, centers[index], bits);
        OverlapRows              overlapAny{};
        REQUIRE(runAny(resolver, overlap, overlapAny) == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(!overlapAny.hits.empty());
        REQUIRE(overlapAny.hits[0].identity.objectId.value == index + 1u);
        REQUIRE(overlapAny.hits[0].identity.geometry.family == families[index]);

        OverlapRows overlapMultiple{};
        REQUIRE(runMultiple(resolver, overlap, overlapMultiple) == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(overlapMultiple.hits.size() == 1u);
        REQUIRE(overlapMultiple.hits[0].identity.objectId.value == index + 1u);
    }
}

void testCandidateTruthParticipationOrderingAndCapacity(ovspatialquery_backend_runtime_t* runtime)
{
    PopulationTable               populations(runtime);
    PopulationKey                 population;
    std::vector<PopulationObject> objects;
    objects.push_back(sphereObject(10u, 2.0, 100u, UINT64_C(0x1)));
    objects.push_back(sphereObject(20u, 5.0, 200u, UINT64_C(0x2)));
    objects.push_back(sphereObject(30u, 8.0, 300u, UINT64_C(0x2)));
    REQUIRE(populations.create(2u, 40, std::move(objects), population) == OVSPATIALQUERY_API_SUCCESS);
    const TargetResolver resolver(populations);

    RaycastSyncRequest ray = rayRequest(population, QueryPrecision::boundsCandidate, UINT64_C(0x2));
    RaycastRows        result{};
    REQUIRE(runMultiple(resolver, ray, result) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(result.hits.size() == 2u);
    REQUIRE(result.hits[0].truth == HitTruth::boundsCandidate);
    REQUIRE(result.hits[0].boundsCandidate.identity.objectId.value == 20u);

    OverlapSyncRequest overlap = overlapAround(population, QueryPrecision::boundsCandidate, 5.0, UINT64_C(0x2));
    overlap.query.geometry.data.aabb.min_corner[0] = 0.0;
    overlap.query.geometry.data.aabb.max_corner[0] = 10.0;
    OverlapRows overlapResult{};
    REQUIRE(runMultiple(resolver, overlap, overlapResult) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(overlapResult.hits.size() == 2u);
    REQUIRE(overlapResult.hits[0].truth == HitTruth::boundsCandidate);
    REQUIRE(overlapResult.hits[0].identity.objectId.value == 20u);
}

void testOverlapAnyFiltersBeforeStopping(ovspatialquery_backend_runtime_t* runtime)
{
    ovspatialquery::detail::SphereGeometrySource source{};
    source.center = {0.0, 0.8, 0.8};
    source.radius = 0.25;
    std::shared_ptr<const ExactGeometry> geometry;
    REQUIRE(ovspatialquery::detail::createSphereExactGeometry(source, geometry) == OVSPATIALQUERY_API_SUCCESS);
    PopulationTable populations(runtime);
    PopulationKey   population;
    // The first sphere's bounds overlap the box but its exact surface does not.
    // The next two exact hits are excluded by bits and view membership.
    REQUIRE(populations.create(2u, 1,
                               {sphereObject(10u, 0.0, 10u, 1u), exactObject(20u, 20u, geometry, 2u),
                                exactObject(50u, 50u, geometry, 1u), exactObject(40u, 40u, geometry, 1u),
                                exactObject(30u, 30u, geometry, 1u)},
                               population) == OVSPATIALQUERY_API_SUCCESS);
    const std::shared_ptr<ovspatialquery::detail::ViewState> view =
        std::make_shared<ovspatialquery::detail::ViewState>();
    view->parent      = population;
    view->handle      = ovspatialquery_view_handle_t{902u};
    view->selectedIds = selected({10u, 20u, 30u, 40u});
    const TargetResolver resolver(populations);
    OverlapSyncRequest   request      = overlapAround(population, QueryPrecision::exact, 0.0, 1u);
    request.common.target.target.view = view;
    request.query.geometry.data.aabb  = {{-0.05, 0.75, 0.75}, {0.05, 0.85, 0.85}};
    OverlapRows any{};
    REQUIRE(runAny(resolver, request, any) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(!any.hits.empty() &&
            (any.hits[0].identity.objectId.value == 30u || any.hits[0].identity.objectId.value == 40u));
    OverlapRows multiple{};
    REQUIRE(runMultiple(resolver, request, multiple) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(multiple.hits.size() == 2u);
    REQUIRE(multiple.hits[0].identity.objectId.value == 30u && multiple.hits[1].identity.objectId.value == 40u);
    request.common.queryBits = 4u;
    REQUIRE(runAny(resolver, request, any) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(any.hits.empty());
    std::shared_ptr<void> retired;
    REQUIRE(populations.release(population, retired) == OVSPATIALQUERY_API_SUCCESS);
}

void testMeshSourceFeatureAndPopulationQualification(ovspatialquery_backend_runtime_t* runtime)
{
    PopulationTable                            populations(runtime);
    const std::shared_ptr<const ExactGeometry> geometry = meshGeometry();
    PopulationKey                              first;
    PopulationKey                              second;
    REQUIRE(populations.create(2u, 51, {exactObject(7u, 707u, geometry, UINT64_MAX)}, first) ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(populations.create(2u, 51, {exactObject(7u, 707u, geometry, UINT64_MAX)}, second) ==
            OVSPATIALQUERY_API_SUCCESS);
    const TargetResolver resolver(populations);
    RaycastSyncRequest   request = rayRequest(first, QueryPrecision::exact);
    request.query.origin[1]      = 0.75;
    RaycastRows firstResult{};
    REQUIRE(runClosest(resolver, request, firstResult) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(!firstResult.hits.empty());
    REQUIRE(firstResult.hits[0].exact.feature.kind == GeometryFeatureKind::triangle);
    const ovspatialquery::detail::TriangleSourceFeature& feature = firstResult.hits[0].exact.feature.triangle;
    REQUIRE(feature.sourceFace == 700u);
    REQUIRE(feature.sourceCorners == (std::array<uint32_t, 3u>{0u, 2u, 3u}));
    const float barycentricSum = feature.barycentrics[0] + feature.barycentrics[1] + feature.barycentrics[2];
    REQUIRE(std::abs(barycentricSum - 1.0f) < 1.0e-6f);

    request.common.target.target.population = second;
    RaycastRows secondResult{};
    REQUIRE(runClosest(resolver, request, secondResult) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(!secondResult.hits.empty());
    // Both populations hold the same object, so only the source population shows which one was queried.
    REQUIRE(firstResult.target.population == first && secondResult.target.population == second);
}

void testHeadRetainedViewAndSelectedOrdinalOnMiss(ovspatialquery_backend_runtime_t* runtime)
{
    PopulationTable populations(runtime);
    PopulationKey   population;
    REQUIRE(populations.create(2u, population) == OVSPATIALQUERY_API_SUCCESS);
    const std::shared_ptr<ovspatialquery::detail::ViewState> view =
        std::make_shared<ovspatialquery::detail::ViewState>();
    view->handle    = ovspatialquery_view_handle_t{900u};
    view->parent    = population;
    view->automatic = true;
    publish(populations, population, 10,
            {sphereObject(1u, 2.0, 101u, UINT64_C(0x1)), sphereObject(2u, 5.0, 202u, UINT64_C(0x2))},
            {{view->handle, selected({1u})}});
    publish(populations, population, 20,
            {sphereObject(1u, 2.0, 101u, UINT64_C(0x1)), sphereObject(2u, 5.0, 202u, UINT64_C(0x2))},
            {{view->handle, selected({2u})}});
    const TargetResolver resolver(populations);

    RaycastSyncRequest request        = rayRequest(population, QueryPrecision::exact, UINT64_C(0x1));
    request.common.target.target.view = view;
    request.common.target.selection   = {TargetSelectionKind::retainedExact, 10};
    RaycastRows retained{};
    REQUIRE(runClosest(resolver, request, retained) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(retained.target.ordinal == 10);
    REQUIRE(!retained.hits.empty() && identity(retained.hits[0]).objectId.value == 1u);

    request.common.target.selection = {TargetSelectionKind::head, 0};
    request.common.queryBits        = UINT64_C(0x2);
    RaycastRows head{};
    REQUIRE(runClosest(resolver, request, head) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(head.target.ordinal == 20);
    REQUIRE(!head.hits.empty() && identity(head.hits[0]).objectId.value == 2u);

    request.common.target.selection = {TargetSelectionKind::retainedExact, 10};
    request.common.queryBits        = UINT64_C(0x4);
    RaycastRows miss{};
    REQUIRE(runClosest(resolver, request, miss) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(miss.target.ordinal == 10 && miss.hits.empty());
}

void testExactEligibilityTransitionsPreserveScalarFallbacks(ovspatialquery_backend_runtime_t* runtime)
{
    PopulationTable populations(runtime);
    PopulationKey   population;
    REQUIRE(populations.create(4u, 0, {}, population) == OVSPATIALQUERY_API_SUCCESS);
    const std::shared_ptr<ovspatialquery::detail::ViewState> view =
        std::make_shared<ovspatialquery::detail::ViewState>();
    view->handle    = ovspatialquery_view_handle_t{901u};
    view->parent    = population;
    view->automatic = true;
    const TargetResolver resolver(populations);
    RaycastSyncRequest   request = rayRequest(population, QueryPrecision::exact, UINT64_C(0x1));

    RaycastRows result{};
    REQUIRE(runClosest(resolver, request, result) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(result.target.ordinal == 0 && result.hits.empty());

    publish(populations, population, 10, {sphereObject(101u, 2.0, 1001u, UINT64_C(0x1))},
            {{view->handle, selected({101u})}});
    REQUIRE(runClosest(resolver, request, result) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(result.target.ordinal == 10 && !result.hits.empty() && identity(result.hits[0]).objectId.value == 101u);

    publish(populations, population, 20,
            {sphereObject(202u, 4.0, 2002u, UINT64_C(0x1)), missingExactObject(203u, 6.0, 2003u, UINT64_C(0x2))},
            {{view->handle, selected({202u})}});
    request.common.target.target.view = view;
    REQUIRE(runClosest(resolver, request, result) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(result.target.ordinal == 20 && !result.hits.empty() && identity(result.hits[0]).objectId.value == 202u);

    request.common.target.target.view.reset();
    request.common.queryBits = UINT64_C(0x2);
    REQUIRE(runClosest(resolver, request, result) == OVSPATIALQUERY_API_NOT_SUPPORTED);

    request.common.target.target.view = view;
    REQUIRE(runClosest(resolver, request, result) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(result.target.ordinal == 20 && result.hits.empty());

    request.common.target.target.view.reset();
    request.common.target.selection = {TargetSelectionKind::retainedExact, 10};
    request.common.queryBits        = UINT64_C(0x1);
    REQUIRE(runClosest(resolver, request, result) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(result.target.ordinal == 10 && !result.hits.empty() && identity(result.hits[0]).objectId.value == 101u);

    publish(populations, population, 30, {sphereObject(303u, 8.0, 3003u, UINT64_C(0x4))},
            {{view->handle, selected({303u})}});
    request.common.target.selection = {TargetSelectionKind::head, 0};
    request.common.queryBits        = UINT64_C(0x4);
    REQUIRE(runClosest(resolver, request, result) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(result.target.ordinal == 30 && !result.hits.empty() && identity(result.hits[0]).objectId.value == 303u);

    request.common.target.selection = {TargetSelectionKind::retainedExact, 20};
    request.common.queryBits        = UINT64_C(0x2);
    REQUIRE(runClosest(resolver, request, result) == OVSPATIALQUERY_API_NOT_SUPPORTED);

    std::shared_ptr<void> retired;
    REQUIRE(populations.release(population, retired) == OVSPATIALQUERY_API_SUCCESS);
}

void testSweepAndOverlapToleranceBoundaries(ovspatialquery_backend_runtime_t* runtime)
{
    PopulationTable populations(runtime);
    PopulationKey   population;
    REQUIRE(populations.create(2u, 61, {sphereObject(61u, 5.0, 610u, UINT64_MAX)}, population) ==
            OVSPATIALQUERY_API_SUCCESS);
    const TargetResolver resolver(populations);

    SphereSweepSyncRequest sweep{};
    sweep.common                = common(population, QueryPrecision::exact);
    sweep.query.center          = {0.0, 2.25, 0.0};
    sweep.query.radius          = 1.0;
    sweep.query.unitDirection   = {1.0f, 0.0f, 0.0f};
    sweep.query.minimumDistance = 0.0;
    sweep.query.maximumDistance = 10.0;
    SweepRows sweepResult{};
    sweep.common.contactTolerance = 0.249;
    REQUIRE(runClosest(resolver, sweep, sweepResult) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(sweepResult.hits.empty());
    sweep.common.contactTolerance = 0.25;
    REQUIRE(runClosest(resolver, sweep, sweepResult) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(!sweepResult.hits.empty());

    OverlapSyncRequest overlap{};
    overlap.common                               = common(population, QueryPrecision::exact);
    overlap.query.geometry.kind                  = OVSPATIALQUERY_OVERLAP_SPHERE;
    overlap.query.geometry.data.sphere.center[0] = 7.25;
    overlap.query.geometry.data.sphere.center[1] = 0.0;
    overlap.query.geometry.data.sphere.center[2] = 0.0;
    overlap.query.geometry.data.sphere.radius    = 1.0;
    OverlapRows overlapResult{};
    overlap.common.contactTolerance = 0.249;
    REQUIRE(runAny(resolver, overlap, overlapResult) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(overlapResult.hits.empty());
    overlap.common.contactTolerance = 0.25;
    REQUIRE(runAny(resolver, overlap, overlapResult) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(!overlapResult.hits.empty());

    overlap.query.geometry                         = {};
    overlap.query.geometry.kind                    = OVSPATIALQUERY_OVERLAP_AABB;
    overlap.query.geometry.data.aabb.min_corner[0] = 6.25;
    overlap.query.geometry.data.aabb.max_corner[0] = 7.25;
    overlap.query.geometry.data.aabb.min_corner[1] = -0.5;
    overlap.query.geometry.data.aabb.max_corner[1] = 0.5;
    overlap.query.geometry.data.aabb.min_corner[2] = -0.5;
    overlap.query.geometry.data.aabb.max_corner[2] = 0.5;
    overlap.common.contactTolerance                = 0.249;
    REQUIRE(runAny(resolver, overlap, overlapResult) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(overlapResult.hits.empty());
    overlap.common.contactTolerance = 0.25;
    REQUIRE(runAny(resolver, overlap, overlapResult) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(!overlapResult.hits.empty());

    overlap.query.geometry                           = {};
    overlap.query.geometry.kind                      = OVSPATIALQUERY_OVERLAP_BOX;
    overlap.query.geometry.data.box.center[0]        = 7.25;
    overlap.query.geometry.data.box.center[1]        = 0.0;
    overlap.query.geometry.data.box.center[2]        = 0.0;
    overlap.query.geometry.data.box.rotation_xyzw[3] = 1.0f;
    overlap.query.geometry.data.box.half_extents[0]  = 1.0;
    overlap.query.geometry.data.box.half_extents[1]  = 0.5;
    overlap.query.geometry.data.box.half_extents[2]  = 0.5;
    overlap.common.contactTolerance                  = 0.249;
    REQUIRE(runAny(resolver, overlap, overlapResult) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(overlapResult.hits.empty());
    overlap.common.contactTolerance = 0.25;
    REQUIRE(runAny(resolver, overlap, overlapResult) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(!overlapResult.hits.empty());
}

void testFailureClassification(ovspatialquery_backend_runtime_t* runtime)
{
    PopulationTable populations(runtime);
    PopulationKey   population;
    REQUIRE(populations.create(2u, 71, {sphereObject(71u, 2.0, 710u, UINT64_MAX)}, population) ==
            OVSPATIALQUERY_API_SUCCESS);
    PopulationKey notReadyPopulation;
    REQUIRE(populations.create(2u, notReadyPopulation) == OVSPATIALQUERY_API_SUCCESS);
    const TargetResolver resolver(populations);
    RaycastSyncRequest   request = rayRequest(population, QueryPrecision::exact);
    RaycastRows          result{};

    request.common.target.target.population = {};
    REQUIRE(runClosest(resolver, request, result) == OVSPATIALQUERY_API_INVALID_HANDLE);
    request.common.target.target.population = notReadyPopulation;
    REQUIRE(runClosest(resolver, request, result) == OVSPATIALQUERY_API_NOT_READY);
    request.common.target.target.population = population;
    request.common.target.selection         = {TargetSelectionKind::retainedExact, 999};
    REQUIRE(runClosest(resolver, request, result) == OVSPATIALQUERY_API_ORDINAL_NOT_RETAINED);
    REQUIRE(runMultiple(resolver, request, result) == OVSPATIALQUERY_API_ORDINAL_NOT_RETAINED);
}

void testSynchronousHostBudgetAdmission(ovspatialquery_backend_runtime_t* runtime)
{
    PopulationTable populations(runtime);
    PopulationKey   population;
    REQUIRE(populations.create(1u, population) == OVSPATIALQUERY_API_SUCCESS);
    publish(populations, population, 1, {sphereObject(1u, 2.0, 1u, UINT64_MAX), sphereObject(2u, 5.0, 2u, UINT64_MAX)},
            {});
    const TargetResolver resolver(populations);
    RaycastSyncRequest   request   = rayRequest(population, QueryPrecision::boundsCandidate);
    uint64_t             exactPeak = 0u;
    REQUIRE(ovspatialquery::detail::estimateSemanticRaycastPeakBytes(2u, exactPeak) && exactPeak > 0u);

    RaycastRows        result{};
    InstanceHostLedger rejected(exactPeak - 1u);
    REQUIRE(runClosest(resolver, request, result, &rejected) == OVSPATIALQUERY_API_HOST_BUDGET &&
            rejected.total() == 0u);

    InstanceHostLedger exact(exactPeak);
    REQUIRE(runClosest(resolver, request, result, &exact) == OVSPATIALQUERY_API_SUCCESS && exact.total() == 0u &&
            result.target.ordinal == 1);
    std::shared_ptr<void> retired;
    REQUIRE(populations.release(population, retired) == OVSPATIALQUERY_API_SUCCESS);
}

// A MULTIPLE sweep that reaches every object holds no more heap than its
// admission charged, for both sweep shapes, with every row kept and with none.
// The oracle reports each object twice, and 320 objects make a hit vector grown
// row by row reallocate past one hit per object.
void testMultipleSweepHoldsNoMoreThanItsAdmission(ovspatialquery_backend_runtime_t* runtime)
{
    constexpr uint64_t            kObjects = 320u;
    std::vector<PopulationObject> objects;
    for (uint64_t index = 0u; index < kObjects; ++index)
    {
        objects.push_back(sphereObject(index + 1u, 3.0 * static_cast<double>(index + 1u), index + 1u, UINT64_MAX));
    }
    PopulationTable populations(runtime);
    PopulationKey   population;
    REQUIRE(populations.create(1u, population) == OVSPATIALQUERY_API_SUCCESS);
    publish(populations, population, 1, std::move(objects), {});
    const TargetResolver resolver(populations);

    for (const bool box : {false, true})
    {
        for (const uint64_t limit : {UINT64_MAX, uint64_t{0u}})
        {
            SphereSweepSyncRequest sweep{};
            sweep.common                = common(population, QueryPrecision::exact);
            sweep.query.box             = box;
            sweep.query.radius          = 0.25;
            sweep.query.halfExtents     = {0.25, 0.25, 0.25};
            sweep.query.unitDirection   = {1.0f, 0.0f, 0.0f};
            sweep.query.maximumDistance = 3.0 * static_cast<double>(kObjects + 1u);
            auto plan =
                modePlan<ovspatialquery::detail::SphereSweepBatchPlan>(ovspatialquery::detail::HitMode::multiple);
            plan.maximumHitsPerQuery = limit;
            const auto runWithin     = [&](uint64_t budget, SweepRows& out)
            {
                InstanceHostLedger ledger(budget);
                return run<ovspatialquery::detail::SphereSweepBatchRow>(resolver, sweep, plan, out, &ledger);
            };

            // The admitted charge, measured: the smallest budget the query runs within.
            uint64_t refused = 0u;
            uint64_t charge  = uint64_t{1u} << 32u;
            {
                SweepRows out{};
                REQUIRE(runWithin(charge, out) == OVSPATIALQUERY_API_SUCCESS);
            }
            while (charge - refused > 1u)
            {
                const uint64_t                    budget = refused + (charge - refused) / 2u;
                SweepRows                         out{};
                const ovspatialquery_api_status_t status = runWithin(budget, out);
                REQUIRE(status == OVSPATIALQUERY_API_SUCCESS || status == OVSPATIALQUERY_API_HOST_BUDGET);
                (status == OVSPATIALQUERY_API_SUCCESS ? charge : refused) = budget;
            }

            SweepRows         out{};
            const std::size_t baseline = gLiveBytes.load();
            gPeakBytes.store(baseline);
            REQUIRE(runWithin(charge, out) == OVSPATIALQUERY_API_SUCCESS);
            const std::size_t held = gPeakBytes.load() - baseline;
            if (held > charge)
            {
                std::fprintf(stderr, "%s sweep, limit %llu: held %zu bytes, admitted %llu\n", box ? "box" : "sphere",
                             static_cast<unsigned long long>(limit), held, static_cast<unsigned long long>(charge));
            }
            REQUIRE(held <= charge);
            if (limit == UINT64_MAX)
            {
                // Every object once, nearest first, however often it was reported.
                REQUIRE(out.hits.size() == kObjects);
                for (uint64_t index = 0u; index < kObjects; ++index)
                {
                    REQUIRE(identity(out.hits[index]).objectId.value == index + 1u);
                }
            }
        }
    }
    std::shared_ptr<void> retired;
    REQUIRE(populations.release(population, retired) == OVSPATIALQUERY_API_SUCCESS);
}

} // namespace

int main()
{
    ovspatialquery_backend_runtime_t* runtime = nullptr;
    REQUIRE(ovspatialquery_test_oracle_create_runtime(&runtime) == OVSPATIALQUERY_API_SUCCESS);
    testExactRayContactToleranceBoundary(runtime);
    testAllModesAndFiveExactFamilies(runtime);
    testCandidateTruthParticipationOrderingAndCapacity(runtime);
    testOverlapAnyFiltersBeforeStopping(runtime);
    testMeshSourceFeatureAndPopulationQualification(runtime);
    testHeadRetainedViewAndSelectedOrdinalOnMiss(runtime);
    testExactEligibilityTransitionsPreserveScalarFallbacks(runtime);
    testSweepAndOverlapToleranceBoundaries(runtime);
    testFailureClassification(runtime);
    testSynchronousHostBudgetAdmission(runtime);
    testMultipleSweepHoldsNoMoreThanItsAdmission(runtime);
    ovspatialquery_test_oracle_destroy_runtime(runtime);
    return 0;
}
