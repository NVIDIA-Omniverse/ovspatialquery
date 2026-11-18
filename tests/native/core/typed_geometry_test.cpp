/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#define OVSPATIALQUERY_ENABLE_TEST_HOOKS
#include "../../../source/ovspatialquery_input.h"
#include "../../../source/ovspatialquery_internal.h"
#include "../../../source/ovspatialquery_population_table.h"
#include "../../../source/ovspatialquery_typed_geometry.h"
#include "../../support/native/oracle_backend.h"
#include "../../support/native/stored_sphere_paths.h"
#include "../../support/native/test_failure.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <memory_resource>
#include <new>
#include <thread>
#include <type_traits>
#include <utility>
#include <variant>

namespace
{

class AccountingScratchProbe final : public std::pmr::memory_resource
{
  public:
    size_t current  = 0u;
    size_t peak     = 0u;
    bool   reject   = false;
    bool   rejected = false;

  private:
    void* do_allocate(size_t bytes, size_t alignment) override
    {
        if (reject)
        {
            rejected = true;
            throw std::bad_alloc();
        }
        void* allocation = std::pmr::new_delete_resource()->allocate(bytes, alignment);
        current += bytes;
        peak = std::max(peak, current);
        return allocation;
    }

    void do_deallocate(void* allocation, size_t bytes, size_t alignment) override
    {
        REQUIRE(bytes <= current);
        current -= bytes;
        std::pmr::new_delete_resource()->deallocate(allocation, bytes, alignment);
    }

    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override { return this == &other; }
};

void testResidentAccountingDeduplicatesAndReleasesScratch()
{
    using ovspatialquery::detail::ExactGeometryResidentAccumulator;
    AccountingScratchProbe     probe;
    std::pmr::memory_resource* previous = std::pmr::set_default_resource(&probe);
    std::vector<uint64_t>      owners(10000u);
    {
        ExactGeometryResidentAccumulator accumulator;
        for (uint64_t& owner : owners)
        {
            REQUIRE(accumulator.add(&owner, sizeof(owner)));
        }
        for (uint64_t& owner : owners)
        {
            REQUIRE(accumulator.add(&owner, sizeof(owner)));
        }
        REQUIRE(accumulator.residentByteCount() == owners.size() * sizeof(uint64_t));
        REQUIRE(probe.current > 0u);
    }
    REQUIRE(probe.current == 0u);
    std::fprintf(stderr, "accounting scratch: 10000 owners, peak %zu bytes\n", probe.peak);
    {
        ExactGeometryResidentAccumulator accumulator;
        REQUIRE(accumulator.add(&owners[0], UINT64_MAX - 1u));
        REQUIRE(accumulator.add(&owners[0], UINT64_MAX));
        REQUIRE(accumulator.residentByteCount() == UINT64_MAX - 1u);
        REQUIRE(!accumulator.add(&owners[1], 2u));
        REQUIRE(accumulator.residentByteCount() == UINT64_MAX);
        REQUIRE(!accumulator.add(&owners[0], 0u));
    }
    REQUIRE(probe.current == 0u);
    {
        ExactGeometryResidentAccumulator accumulator;
        // Some standard libraries allocate during construction. Reject only
        // subsequent growth, allowing additions to consume existing arena space.
        probe.reject = true;
        for (uint64_t& owner : owners)
        {
            const bool added = accumulator.add(&owner, 1u);
            REQUIRE(added == !probe.rejected);
            if (probe.rejected)
                break;
        }
        REQUIRE(probe.rejected);
        REQUIRE(accumulator.residentByteCount() == UINT64_MAX);
        REQUIRE(!accumulator.add(&owners[1], 1u));
    }
    REQUIRE(probe.current == 0u);
    std::pmr::set_default_resource(previous);
}

bool near(double first, double second) { return std::abs(first - second) < 1.0e-12; }

class RefusingDescriptionGeometry final : public ovspatialquery::detail::ExactGeometry
{
  public:
    ovspatialquery::detail::ExactGeometryFamily family() const noexcept override
    {
        return ovspatialquery::detail::ExactGeometryFamily::sphere;
    }

    bool exportDescription(ovspatialquery::detail::ExactGeometryDescription&) const noexcept override { return false; }

    const ovspatialquery_aabb_t& worldBounds() const noexcept override { return mBounds; }

    bool raycast(const double[3], const float[3], double, double, bool, ovspatialquery::detail::ExactGeometryHit&,
                 ovspatialquery::detail::InstanceHostLedger*, ovspatialquery_api_status_t*, bool) const override
    {
        return false;
    }

    bool sweepSphere(const double[3], double, const float[3], double, double, ovspatialquery::detail::ExactGeometryHit&,
                     ovspatialquery::detail::InstanceHostLedger*, ovspatialquery_api_status_t*, bool,
                     bool) const override
    {
        return false;
    }

    bool overlaps(const ovspatialquery_overlap_shape_t&, const double (*)[3],
                  ovspatialquery::detail::InstanceHostLedger*, ovspatialquery_api_status_t*) const override
    {
        return false;
    }

    uint64_t residentByteCount() const noexcept override { return sizeof(RefusingDescriptionGeometry); }

    void accumulateResidentAllocations(ovspatialquery::detail::ExactGeometryResidentAccumulator& accumulator,
                                       bool) const noexcept override
    {
        (void)accumulator.add(this, sizeof(RefusingDescriptionGeometry));
    }

  private:
    ovspatialquery_aabb_t mBounds{};
};

void testFiveFamilyDescriptionExportRejectsMissingOwnerAtomically()
{
    using namespace ovspatialquery::detail;
    static_assert(std::variant_size<ExactGeometryDescription>::value == 5u,
                  "Every mandatory exact family needs one private description arm");

    BoxExactGeometryDescription sentinel{};
    sentinel.center                                   = {7.0, 8.0, 9.0};
    ExactGeometryDescription                   output = sentinel;
    const std::shared_ptr<const ExactGeometry> missing;
    REQUIRE(exportExactGeometryDescription(missing, output) == OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(std::holds_alternative<BoxExactGeometryDescription>(output));
    REQUIRE(std::get<BoxExactGeometryDescription>(output).center == sentinel.center);

    const std::shared_ptr<const ExactGeometry> refusing = std::make_shared<const RefusingDescriptionGeometry>();
    REQUIRE(exportExactGeometryDescription(refusing, output) == OVSPATIALQUERY_API_INTERNAL);
    REQUIRE(std::holds_alternative<BoxExactGeometryDescription>(output));
    REQUIRE(std::get<BoxExactGeometryDescription>(output).center == sentinel.center);
}

ovspatialquery::detail::ConvexGeometrySource unitCubeConvex()
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
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    };
    return source;
}

ovspatialquery::detail::ConvexGeometrySource triangulatedCubeConvex()
{
    ovspatialquery::detail::ConvexGeometrySource source = unitCubeConvex();
    source.faceVertexCounts.assign(12u, 3u);
    source.faceVertexIndices = {
        0u, 3u, 2u, 0u, 2u, 1u, 4u, 5u, 6u, 4u, 6u, 7u, 0u, 1u, 5u, 0u, 5u, 4u,
        3u, 7u, 6u, 3u, 6u, 2u, 0u, 4u, 7u, 0u, 7u, 3u, 1u, 2u, 6u, 1u, 6u, 5u,
    };
    return source;
}

std::shared_ptr<const ovspatialquery::detail::ExactGeometry> mappedTriangleMeshGeometry(double translation = 5.0)
{
    using namespace ovspatialquery::detail;
    TriangleMeshSource mesh{};
    mesh.key      = 404u;
    mesh.vertices = {
        0.0, -1.0, -1.0, 0.0, -1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, -1.0,
    };
    mesh.triangles     = {0u, 1u, 2u, 0u, 2u, 3u};
    mesh.sourceFaces   = {17u, 17u};
    mesh.sourceCorners = {0u, 1u, 2u, 0u, 2u, 3u};
    std::shared_ptr<const PreparedTriangleMesh> preparedMesh;
    REQUIRE(prepareTriangleMesh(mesh, preparedMesh) == OVSPATIALQUERY_API_SUCCESS);
    std::shared_ptr<const TriangleObjectGeometry> prepared;
    REQUIRE(bindPreparedTriangleGeometry(
                preparedMesh, {1.0, 0.0, 0.0, translation, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0},
                true, prepared) == OVSPATIALQUERY_API_SUCCESS);
    std::shared_ptr<const ExactGeometry> geometry;
    REQUIRE(createTriangleMeshExactGeometry(prepared, geometry) == OVSPATIALQUERY_API_SUCCESS);
    return geometry;
}

void testRetainedGenerationOwnsDescriptionAndKeepsIdentitySeparate()
{
    using namespace ovspatialquery::detail;
    ovspatialquery_backend_runtime_t* runtime = nullptr;
    REQUIRE(ovspatialquery_test_oracle_create_runtime(&runtime) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(runtime != nullptr);
    {
        PopulationTable                      table(runtime);
        std::shared_ptr<const ExactGeometry> oldGeometry = mappedTriangleMeshGeometry(5.0);
        PopulationObject                     oldObject{};
        oldObject.id.value          = 71u;
        oldObject.bounds            = oldGeometry->worldBounds();
        oldObject.exactGeometry     = oldGeometry;
        oldObject.geometryReference = {ExactGeometryFamily::triangleMesh, 901u};

        PopulationKey population{};
        REQUIRE(table.create(2u, 10, {oldObject}, population) == OVSPATIALQUERY_API_SUCCESS);

        GenerationLease                current;
        const ovspatialquery_ordinal_t baseOrdinal = 10;
        REQUIRE(table.beginUpdate(population, 11, &baseOrdinal, current) == OVSPATIALQUERY_API_SUCCESS);
        std::shared_ptr<const ExactGeometry> newGeometry = mappedTriangleMeshGeometry(9.0);
        PopulationObject                     newObject   = oldObject;
        newObject.bounds                                 = newGeometry->worldBounds();
        newObject.exactGeometry                          = newGeometry;
        GenerationLease replacement;
        REQUIRE(table.buildGeneration(11, ObjectTable({newObject}), {}, replacement) == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(table.publishUpdate(population, replacement) == OVSPATIALQUERY_API_SUCCESS);
        table.finishUpdate(population);

        GenerationLease retained;
        GenerationLease head;
        REQUIRE(table.pinExact(population, 10, retained) == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(table.pin(population, head) == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(retained.ordinal() == 10);
        REQUIRE(head.ordinal() == 11);
        oldGeometry.reset();
        newGeometry.reset();
        oldObject = {};
        newObject = {};
        current.reset();
        replacement.reset();

        ExactGeometryDescription retainedDescription{};
        ExactGeometryDescription headDescription{};
        REQUIRE(exportExactGeometryDescription(retained.objects()[0].exactGeometry, retainedDescription) ==
                OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(exportExactGeometryDescription(head.objects()[0].exactGeometry, headDescription) ==
                OVSPATIALQUERY_API_SUCCESS);
        const TriangleMeshExactGeometryDescription retainedMesh =
            std::get<TriangleMeshExactGeometryDescription>(retainedDescription);
        const TriangleMeshExactGeometryDescription headMesh =
            std::get<TriangleMeshExactGeometryDescription>(headDescription);
        REQUIRE(near(retainedMesh.worldFromLocal[3], 5.0));
        REQUIRE(near(headMesh.worldFromLocal[3], 9.0));
        REQUIRE(retainedMesh.localVertices != nullptr);
        REQUIRE(headMesh.localVertices != nullptr);
        REQUIRE(near(retainedMesh.localVertices[4], -1.0));
        REQUIRE(near(headMesh.localVertices[4], -1.0));
        REQUIRE(retained.objects()[0].id.value == 71u);
        REQUIRE(head.objects()[0].id.value == 71u);
        REQUIRE(retained.objects()[0].geometryReference == head.objects()[0].geometryReference);
        REQUIRE(retained.objects()[0].geometryReference.value == 901u);

        retained.reset();
        head.reset();
        std::shared_ptr<void> deferred;
        REQUIRE(table.release(population, deferred) == OVSPATIALQUERY_API_SUCCESS);
    }
    ovspatialquery_test_oracle_destroy_runtime(runtime);
}

void requireRayEquivalent(const std::shared_ptr<const ovspatialquery::detail::ExactGeometry>& first,
                          const std::shared_ptr<const ovspatialquery::detail::ExactGeometry>& second,
                          const double origin[3], const float direction[3])
{
    ovspatialquery::detail::ExactGeometryHit firstHit{};
    ovspatialquery::detail::ExactGeometryHit secondHit{};
    const bool firstHasHit  = first->raycast(origin, direction, 0.0, 20.0, false, firstHit);
    const bool secondHasHit = second->raycast(origin, direction, 0.0, 20.0, false, secondHit);
    if (!firstHasHit || !secondHasHit)
    {
        std::fprintf(stderr, "description ray equivalence missed for family %u (%u/%u)\n",
                     static_cast<unsigned>(first->family()), firstHasHit ? 1u : 0u, secondHasHit ? 1u : 0u);
    }
    REQUIRE(firstHasHit);
    REQUIRE(secondHasHit);
    REQUIRE(near(firstHit.distance, secondHit.distance));
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        REQUIRE(near(firstHit.position[axis], secondHit.position[axis]));
        REQUIRE(near(firstHit.normal[axis], secondHit.normal[axis]));
    }
    REQUIRE(firstHit.hasTriangleSource == secondHit.hasTriangleSource);
    REQUIRE(firstHit.sourceFace == secondHit.sourceFace);
    REQUIRE(firstHit.generatedTriangle == secondHit.generatedTriangle);
    for (uint32_t corner = 0u; corner < 3u; ++corner)
    {
        REQUIRE(firstHit.sourceCorners[corner] == secondHit.sourceCorners[corner]);
    }
    REQUIRE(near(firstHit.uv[0], secondHit.uv[0]));
    REQUIRE(near(firstHit.uv[1], secondHit.uv[1]));
}

void testFiveFamilyDescriptionsExposeExactTruthAndRebuildEquivalentKernels()
{
    using namespace ovspatialquery::detail;
    ExactGeometryDescription description{};
    const double             origin[3]    = {0.0, 0.0, 0.0};
    const float              direction[3] = {1.0f, 0.0f, 0.0f};

    SphereGeometrySource sphereSource{};
    sphereSource.center = {5.0, 0.25, -0.5};
    sphereSource.radius = 1.25;
    std::shared_ptr<const ExactGeometry> sphere;
    REQUIRE(createSphereExactGeometry(sphereSource, sphere) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(exportExactGeometryDescription(sphere, description) == OVSPATIALQUERY_API_SUCCESS);
    const SphereExactGeometryDescription sphereDescription = std::get<SphereExactGeometryDescription>(description);
    REQUIRE(sphereDescription.center == sphereSource.center);
    REQUIRE(near(sphereDescription.radius, sphereSource.radius));
    SphereGeometrySource rebuiltSphereSource{};
    rebuiltSphereSource.center = sphereDescription.center;
    rebuiltSphereSource.radius = sphereDescription.radius;
    std::shared_ptr<const ExactGeometry> rebuiltSphere;
    REQUIRE(createSphereExactGeometry(rebuiltSphereSource, rebuiltSphere) == OVSPATIALQUERY_API_SUCCESS);
    requireRayEquivalent(sphere, rebuiltSphere, origin, direction);

    BoxGeometrySource boxSource{};
    boxSource.center = {5.0, 0.0, 0.0};
    boxSource.axes   = {
        std::array<double, 3u>{0.0, 1.0, 0.0},
        std::array<double, 3u>{-1.0, 0.0, 0.0},
        std::array<double, 3u>{0.0, 0.0, 1.0},
    };
    boxSource.halfExtents = {1.0, 2.0, 0.5};
    std::shared_ptr<const ExactGeometry> box;
    REQUIRE(createBoxExactGeometry(boxSource, box) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(exportExactGeometryDescription(box, description) == OVSPATIALQUERY_API_SUCCESS);
    const BoxExactGeometryDescription boxDescription = std::get<BoxExactGeometryDescription>(description);
    REQUIRE(boxDescription.center == boxSource.center);
    REQUIRE(boxDescription.axes == boxSource.axes);
    REQUIRE(boxDescription.halfExtents == boxSource.halfExtents);
    BoxGeometrySource rebuiltBoxSource{};
    rebuiltBoxSource.center      = boxDescription.center;
    rebuiltBoxSource.axes        = boxDescription.axes;
    rebuiltBoxSource.halfExtents = boxDescription.halfExtents;
    std::shared_ptr<const ExactGeometry> rebuiltBox;
    REQUIRE(createBoxExactGeometry(rebuiltBoxSource, rebuiltBox) == OVSPATIALQUERY_API_SUCCESS);
    requireRayEquivalent(box, rebuiltBox, origin, direction);

    BoxGeometrySource sameBoundsBoxSource{};
    sameBoundsBoxSource.center = boxSource.center;
    sameBoundsBoxSource.axes   = {
        std::array<double, 3u>{1.0, 0.0, 0.0},
        std::array<double, 3u>{0.0, 1.0, 0.0},
        std::array<double, 3u>{0.0, 0.0, 1.0},
    };
    sameBoundsBoxSource.halfExtents = {2.0, 1.0, 0.5};
    std::shared_ptr<const ExactGeometry> sameBoundsBox;
    REQUIRE(createBoxExactGeometry(sameBoundsBoxSource, sameBoundsBox) == OVSPATIALQUERY_API_SUCCESS);
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        REQUIRE(near(sameBoundsBox->worldBounds().minimum[axis], box->worldBounds().minimum[axis]));
        REQUIRE(near(sameBoundsBox->worldBounds().maximum[axis], box->worldBounds().maximum[axis]));
    }
    ExactGeometryDescription sameBoundsDescription{};
    REQUIRE(exportExactGeometryDescription(sameBoundsBox, sameBoundsDescription) == OVSPATIALQUERY_API_SUCCESS);
    const BoxExactGeometryDescription sameBoundsBoxDescription =
        std::get<BoxExactGeometryDescription>(sameBoundsDescription);
    REQUIRE(sameBoundsBoxDescription.axes != boxDescription.axes);
    REQUIRE(sameBoundsBoxDescription.halfExtents != boxDescription.halfExtents);

    CapsuleGeometrySource capsuleSource{};
    capsuleSource.endpointA = {5.0, -1.0, 0.0};
    capsuleSource.endpointB = {5.0, 1.0, 0.0};
    capsuleSource.radius    = 0.5;
    std::shared_ptr<const ExactGeometry> capsule;
    REQUIRE(createCapsuleExactGeometry(capsuleSource, capsule) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(exportExactGeometryDescription(capsule, description) == OVSPATIALQUERY_API_SUCCESS);
    const CapsuleExactGeometryDescription capsuleDescription = std::get<CapsuleExactGeometryDescription>(description);
    REQUIRE(capsuleDescription.endpointA == capsuleSource.endpointA);
    REQUIRE(capsuleDescription.endpointB == capsuleSource.endpointB);
    REQUIRE(near(capsuleDescription.radius, capsuleSource.radius));
    CapsuleGeometrySource rebuiltCapsuleSource{};
    rebuiltCapsuleSource.endpointA = capsuleDescription.endpointA;
    rebuiltCapsuleSource.endpointB = capsuleDescription.endpointB;
    rebuiltCapsuleSource.radius    = capsuleDescription.radius;
    std::shared_ptr<const ExactGeometry> rebuiltCapsule;
    REQUIRE(createCapsuleExactGeometry(rebuiltCapsuleSource, rebuiltCapsule) == OVSPATIALQUERY_API_SUCCESS);
    requireRayEquivalent(capsule, rebuiltCapsule, origin, direction);

    ConvexGeometrySource convexSource = unitCubeConvex();
    convexSource.worldFromLocal       = {
        2.0, 0.0, 0.0, 5.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    };
    std::shared_ptr<const ExactGeometry> convex;
    REQUIRE(createConvexExactGeometry(convexSource, convex) == ConvexConstructionOutcome::success);
    REQUIRE(exportExactGeometryDescription(convex, description) == OVSPATIALQUERY_API_SUCCESS);
    const ConvexExactGeometryDescription convexDescription = std::get<ConvexExactGeometryDescription>(description);
    REQUIRE(convexDescription.vertexCount == 8u);
    REQUIRE(convexDescription.triangleCount == 12u);
    REQUIRE(convexDescription.planeCount == 6u);
    REQUIRE(convexDescription.edgeCount == 12u);
    REQUIRE(near(convexDescription.vertices[0], 3.0));
    REQUIRE(near(convexDescription.vertices[1], -2.0));
    REQUIRE(near(convexDescription.vertices[2], -2.0));
    REQUIRE(near(convexDescription.planes[0].normal[2], -1.0));
    REQUIRE(convexDescription.edges[0].first < convexDescription.edges[0].second);
    ConvexGeometrySource rebuiltConvexSource{};
    for (uint64_t index = 0; index < convexDescription.vertexCount * 3u; ++index)
        rebuiltConvexSource.vertices.push_back(static_cast<float>(convexDescription.vertices[index]));
    rebuiltConvexSource.faceVertexCounts.assign(static_cast<size_t>(convexDescription.triangleCount), 3u);
    rebuiltConvexSource.faceVertexIndices.assign(convexDescription.triangleVertexIndices,
                                                 convexDescription.triangleVertexIndices +
                                                     convexDescription.triangleCount * 3u);
    rebuiltConvexSource.worldFromLocal = {
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    };
    std::shared_ptr<const ExactGeometry> rebuiltConvex;
    REQUIRE(createConvexExactGeometry(rebuiltConvexSource, rebuiltConvex) == ConvexConstructionOutcome::success);
    requireRayEquivalent(convex, rebuiltConvex, origin, direction);

    std::shared_ptr<const ExactGeometry> mesh = mappedTriangleMeshGeometry();
    REQUIRE(exportExactGeometryDescription(mesh, description) == OVSPATIALQUERY_API_SUCCESS);
    const TriangleMeshExactGeometryDescription meshDescription =
        std::get<TriangleMeshExactGeometryDescription>(description);
    REQUIRE(meshDescription.vertexCount == 4u);
    REQUIRE(meshDescription.triangleCount == 2u);
    REQUIRE(near(meshDescription.localVertices[0], 0.0));
    REQUIRE(near(meshDescription.worldFromLocal[3], 5.0));
    REQUIRE(meshDescription.doubleSided);
    REQUIRE(meshDescription.sourceFaces[0] == 17u);
    REQUIRE(meshDescription.sourceFaces[1] == 17u);
    REQUIRE(meshDescription.sourceCorners[3] == 0u);
    REQUIRE(meshDescription.sourceCorners[4] == 2u);
    REQUIRE(meshDescription.sourceCorners[5] == 3u);

    TriangleMeshSource rebuiltMeshSource{};
    rebuiltMeshSource.key      = 405u;
    rebuiltMeshSource.vertices = std::vector<float>(meshDescription.localVertices,
                                                    meshDescription.localVertices + meshDescription.vertexCount * 3u);
    rebuiltMeshSource.triangles.assign(meshDescription.triangleVertexIndices,
                                       meshDescription.triangleVertexIndices + meshDescription.triangleCount * 3u);
    rebuiltMeshSource.sourceFaces.assign(meshDescription.sourceFaces,
                                         meshDescription.sourceFaces + meshDescription.triangleCount);
    rebuiltMeshSource.sourceCorners.assign(meshDescription.sourceCorners,
                                           meshDescription.sourceCorners + meshDescription.triangleCount * 3u);
    std::shared_ptr<const PreparedTriangleMesh> rebuiltPreparedMesh;
    REQUIRE(prepareTriangleMesh(rebuiltMeshSource, rebuiltPreparedMesh) == OVSPATIALQUERY_API_SUCCESS);
    std::shared_ptr<const TriangleObjectGeometry> rebuiltPrepared;
    REQUIRE(bindPreparedTriangleGeometry(rebuiltPreparedMesh, meshDescription.worldFromLocal,
                                         meshDescription.doubleSided, rebuiltPrepared) == OVSPATIALQUERY_API_SUCCESS);
    std::shared_ptr<const ExactGeometry> rebuiltMesh;
    REQUIRE(createTriangleMeshExactGeometry(rebuiltPrepared, rebuiltMesh) == OVSPATIALQUERY_API_SUCCESS);
    const double meshOrigin[3] = {0.0, 0.75, 0.0};
    requireRayEquivalent(mesh, rebuiltMesh, meshOrigin, direction);

    const uint64_t     meshBytes        = mesh->residentByteCount();
    const float* const retainedVertices = meshDescription.localVertices;
    for (uint32_t repetition = 0u; repetition < 64u; ++repetition)
    {
        ExactGeometryDescription repeated{};
        REQUIRE(exportExactGeometryDescription(mesh, repeated) == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(std::get<TriangleMeshExactGeometryDescription>(repeated).localVertices == retainedVertices);
    }
    REQUIRE(mesh->residentByteCount() == meshBytes);
}

void testConvexSupportsEveryExactQueryWithoutFeatures()
{
    const ovspatialquery::detail::ConvexGeometrySource           source = unitCubeConvex();
    std::shared_ptr<const ovspatialquery::detail::ExactGeometry> geometry;
    REQUIRE(ovspatialquery::detail::createConvexExactGeometry(source, geometry) ==
            ovspatialquery::detail::ConvexConstructionOutcome::success);
    REQUIRE(geometry != nullptr);
    REQUIRE(geometry->family() == ovspatialquery::detail::ExactGeometryFamily::convex);
    REQUIRE(near(geometry->worldBounds().minimum[0], -1.0));
    REQUIRE(near(geometry->worldBounds().maximum[2], 1.0));

    const double                             origin[3]    = {-3.0, 0.0, 0.0};
    const float                              direction[3] = {1.0f, 0.0f, 0.0f};
    ovspatialquery::detail::ExactGeometryHit hit{};
    REQUIRE(geometry->raycast(origin, direction, 0.0, 10.0, false, hit));
    REQUIRE(near(hit.distance, 2.0));
    REQUIRE(near(hit.position[0], -1.0));
    REQUIRE(near(hit.normal[0], -1.0));
    REQUIRE(!hit.hasTriangleSource);

    hit = {};
    REQUIRE(geometry->sweepSphere(origin, 0.5, direction, 0.0, 10.0, hit));
    REQUIRE(near(hit.distance, 1.5));
    REQUIRE(near(hit.position[0], -1.0));
    REQUIRE(near(hit.normal[0], -1.0));
    REQUIRE(!hit.hasTriangleSource);

    ovspatialquery_overlap_shape_t overlap{};
    overlap.kind                  = OVSPATIALQUERY_OVERLAP_SPHERE;
    overlap.data.sphere.center[0] = -1.5;
    overlap.data.sphere.radius    = 0.5;
    REQUIRE(geometry->overlaps(overlap));
    overlap.data.sphere.center[0] = -1.5001;
    REQUIRE(!geometry->overlaps(overlap));

    overlap                           = {};
    overlap.kind                      = OVSPATIALQUERY_OVERLAP_BOX;
    overlap.data.box.center[0]        = -1.5;
    overlap.data.box.rotation_xyzw[3] = 1.0f;
    overlap.data.box.half_extents[0]  = 0.5;
    overlap.data.box.half_extents[1]  = 0.25;
    overlap.data.box.half_extents[2]  = 0.25;
    REQUIRE(geometry->overlaps(overlap));
    overlap.data.box.center[0] = -1.5001;
    REQUIRE(!geometry->overlaps(overlap));

    overlap                         = {};
    overlap.kind                    = OVSPATIALQUERY_OVERLAP_AABB;
    overlap.data.aabb.min_corner[0] = -1.5;
    overlap.data.aabb.min_corner[1] = -0.25;
    overlap.data.aabb.min_corner[2] = -0.25;
    overlap.data.aabb.max_corner[0] = -1.0;
    overlap.data.aabb.max_corner[1] = 0.25;
    overlap.data.aabb.max_corner[2] = 0.25;
    REQUIRE(geometry->overlaps(overlap));
    overlap.data.aabb.max_corner[0] = -1.0001;
    REQUIRE(!geometry->overlaps(overlap));

    overlap      = {};
    overlap.kind = OVSPATIALQUERY_OVERLAP_AABB;
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        overlap.data.aabb.min_corner[axis] = -std::numeric_limits<double>::max();
        overlap.data.aabb.max_corner[axis] = std::numeric_limits<double>::max();
    }
    REQUIRE(geometry->overlaps(overlap));

    const double missOrigin[3] = {-3.0, 1.0001, 0.0};
    hit                        = {};
    REQUIRE(!geometry->raycast(missOrigin, direction, 0.0, 10.0, false, hit));
    REQUIRE(!geometry->sweepSphere(missOrigin, 0.00001, direction, 0.0, 10.0, hit));
}

// A face polygon accepted as planar within the hull tolerance keeps the plane of
// its first three corners, which can enclose a sliver outside the hull. The cube's
// corner 7 is lowered by 2^-21, so over (-0.5, 0.5) the top is z = 1 - 2^-22 under
// the polygon's plane z = 1, with normal along (-2^-22, 2^-22, 1). A box of half
// extent 2^-25 turned onto that normal lies 3/16 of 2^-21 above it at z = 1 - 2^-23
// and is apart, though its corners are inside every face plane; 2^-22 lower it overlaps.
void testBoxAboveANonplanarFaceIsApart()
{
    ovspatialquery::detail::ConvexGeometrySource source  = unitCubeConvex();
    const double                                 lowered = std::ldexp(1.0, -21);
    source.vertices[7u * 3u + 2u]                        = static_cast<float>(1.0 - lowered); // exact in float
    std::shared_ptr<const ovspatialquery::detail::ExactGeometry> geometry;
    REQUIRE(ovspatialquery::detail::createConvexExactGeometry(source, geometry) ==
            ovspatialquery::detail::ConvexConstructionOutcome::success);
    ovspatialquery_overlap_shape_t overlap{};
    overlap.kind                      = OVSPATIALQUERY_OVERLAP_BOX;
    overlap.data.box.center[0]        = -0.5;
    overlap.data.box.center[1]        = 0.5;
    overlap.data.box.center[2]        = 1.0 - lowered / 4.0;
    overlap.data.box.rotation_xyzw[0] = static_cast<float>(-lowered / 4.0);
    overlap.data.box.rotation_xyzw[1] = static_cast<float>(-lowered / 4.0);
    overlap.data.box.rotation_xyzw[3] = 1.0f;
    for (double& extent : overlap.data.box.half_extents)
    {
        extent = lowered / 16.0;
    }
    REQUIRE(!geometry->overlaps(overlap));
    overlap.data.box.center[2] = 1.0 - 3.0 * lowered / 4.0;
    REQUIRE(geometry->overlaps(overlap));
}

void testOraclePreparesAuthoredPointSetIntoExactConvexHull()
{
    ovspatialquery_backend_runtime_t* runtime = nullptr;
    REQUIRE(ovspatialquery_test_oracle_create_runtime(&runtime) == OVSPATIALQUERY_API_SUCCESS);

    std::vector<float> authoredPoints = unitCubeConvex().vertices;
    authoredPoints.insert(authoredPoints.end(), {0.0, 0.0, 0.0, -1.0, -1.0, -1.0});
    std::shared_ptr<const ovspatialquery::detail::PreparedConvexHull> hull;
    REQUIRE(ovspatialquery::detail::prepareConvexHull(runtime, authoredPoints, std::numeric_limits<uint64_t>::max(),
                                                      hull) ==
            ovspatialquery::detail::ConvexHullPreparationOutcome::success);
    REQUIRE(hull != nullptr);
    REQUIRE(hull->vertexCount() == 8u);

    std::shared_ptr<const ovspatialquery::detail::ExactGeometry> geometry;
    REQUIRE(ovspatialquery::detail::createConvexExactGeometry(hull,
                                                              std::array<double, 16u>{
                                                                  1.0,
                                                                  0.0,
                                                                  0.0,
                                                                  0.0,
                                                                  0.0,
                                                                  1.0,
                                                                  0.0,
                                                                  0.0,
                                                                  0.0,
                                                                  0.0,
                                                                  1.0,
                                                                  0.0,
                                                                  0.0,
                                                                  0.0,
                                                                  0.0,
                                                                  1.0,
                                                              },
                                                              geometry) ==
            ovspatialquery::detail::ConvexConstructionOutcome::success);
    const double                             origin[3]    = {-3.0, 0.0, 0.0};
    const float                              direction[3] = {1.0f, 0.0f, 0.0f};
    ovspatialquery::detail::ExactGeometryHit hit{};
    REQUIRE(geometry->raycast(origin, direction, 0.0, 10.0, false, hit));
    REQUIRE(near(hit.distance, 2.0));
    REQUIRE(!hit.hasTriangleSource);
    hit = {};
    REQUIRE(geometry->sweepSphere(origin, 0.5, direction, 0.0, 10.0, hit));
    REQUIRE(near(hit.distance, 1.5));
    REQUIRE(!hit.hasTriangleSource);

    ovspatialquery_overlap_shape_t overlap{};
    overlap.kind                  = OVSPATIALQUERY_OVERLAP_SPHERE;
    overlap.data.sphere.center[0] = -1.5;
    overlap.data.sphere.radius    = 0.5;
    REQUIRE(geometry->overlaps(overlap));
    overlap.data.sphere.center[0] = -1.5001;
    REQUIRE(!geometry->overlaps(overlap));
    overlap                           = {};
    overlap.kind                      = OVSPATIALQUERY_OVERLAP_BOX;
    overlap.data.box.center[0]        = -1.5;
    overlap.data.box.rotation_xyzw[3] = 1.0f;
    overlap.data.box.half_extents[0]  = 0.5;
    overlap.data.box.half_extents[1]  = 0.25;
    overlap.data.box.half_extents[2]  = 0.25;
    REQUIRE(geometry->overlaps(overlap));
    overlap.data.box.center[0] = -1.5001;
    REQUIRE(!geometry->overlaps(overlap));
    overlap                         = {};
    overlap.kind                    = OVSPATIALQUERY_OVERLAP_AABB;
    overlap.data.aabb.min_corner[0] = -1.5;
    overlap.data.aabb.min_corner[1] = -0.25;
    overlap.data.aabb.min_corner[2] = -0.25;
    overlap.data.aabb.max_corner[0] = -1.0;
    overlap.data.aabb.max_corner[1] = 0.25;
    overlap.data.aabb.max_corner[2] = 0.25;
    REQUIRE(geometry->overlaps(overlap));
    overlap.data.aabb.max_corner[0] = -1.0001;
    REQUIRE(!geometry->overlaps(overlap));

    ovspatialquery_test_oracle_destroy_runtime(runtime);
}

void testOracleKeepsEveryPointOfADenseHull()
{
    ovspatialquery_backend_runtime_t* runtime = nullptr;
    REQUIRE(ovspatialquery_test_oracle_create_runtime(&runtime) == OVSPATIALQUERY_API_SUCCESS);

    constexpr uint32_t kPointCount  = 192u;
    constexpr double   kGoldenAngle = 2.3999632297286533222;
    std::vector<float> authoredPoints;
    authoredPoints.reserve(kPointCount * 3u);
    for (uint32_t index = 0u; index < kPointCount; ++index)
    {
        const double z      = 1.0 - 2.0 * (static_cast<double>(index) + 0.5) / static_cast<double>(kPointCount);
        const double radius = std::sqrt(std::max(0.0, 1.0 - z * z));
        const double angle  = kGoldenAngle * static_cast<double>(index);
        authoredPoints.push_back(static_cast<float>(radius * std::cos(angle)));
        authoredPoints.push_back(static_cast<float>(radius * std::sin(angle)));
        authoredPoints.push_back(static_cast<float>(z));
    }

    std::shared_ptr<const ovspatialquery::detail::PreparedConvexHull> hull;
    REQUIRE(ovspatialquery::detail::prepareConvexHull(runtime, authoredPoints, std::numeric_limits<uint64_t>::max(),
                                                      hull) ==
            ovspatialquery::detail::ConvexHullPreparationOutcome::success);
    REQUIRE(hull->vertexCount() == kPointCount);

    std::shared_ptr<const ovspatialquery::detail::ExactGeometry> geometry;
    const std::array<double, 16u>                                identity = {
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    };
    REQUIRE(ovspatialquery::detail::createConvexExactGeometry(hull, identity, geometry) ==
            ovspatialquery::detail::ConvexConstructionOutcome::success);
    for (uint32_t index = 0u; index < kPointCount; ++index)
    {
        ovspatialquery_overlap_shape_t pointOverlap{};
        pointOverlap.kind                  = OVSPATIALQUERY_OVERLAP_SPHERE;
        pointOverlap.data.sphere.center[0] = authoredPoints[index * 3u];
        pointOverlap.data.sphere.center[1] = authoredPoints[index * 3u + 1u];
        pointOverlap.data.sphere.center[2] = authoredPoints[index * 3u + 2u];
        pointOverlap.data.sphere.radius    = 1.0e-12;
        REQUIRE(geometry->overlaps(pointOverlap));
    }

    ovspatialquery_test_oracle_destroy_runtime(runtime);
}

void testOracleConvexPreparationRejectsMalformedAndBudgetAtomically()
{
    ovspatialquery_backend_runtime_t* runtime = nullptr;
    REQUIRE(ovspatialquery_test_oracle_create_runtime(&runtime) == OVSPATIALQUERY_API_SUCCESS);
    const std::vector<float>                                          validPoints = unitCubeConvex().vertices;
    std::shared_ptr<const ovspatialquery::detail::PreparedConvexHull> preserved;
    REQUIRE(ovspatialquery::detail::prepareConvexHull(runtime, validPoints, std::numeric_limits<uint64_t>::max(),
                                                      preserved) ==
            ovspatialquery::detail::ConvexHullPreparationOutcome::success);
    std::shared_ptr<const ovspatialquery::detail::PreparedConvexHull> output = preserved;

    std::vector<float> malformed = validPoints;
    malformed[0]                 = std::numeric_limits<float>::infinity();
    REQUIRE(
        ovspatialquery::detail::prepareConvexHull(runtime, malformed, std::numeric_limits<uint64_t>::max(), output) ==
        ovspatialquery::detail::ConvexHullPreparationOutcome::invalidArgument);
    REQUIRE(output == preserved);

    malformed = {
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    };
    REQUIRE(
        ovspatialquery::detail::prepareConvexHull(runtime, malformed, std::numeric_limits<uint64_t>::max(), output) ==
        ovspatialquery::detail::ConvexHullPreparationOutcome::invalidArgument);
    REQUIRE(output == preserved);

    malformed.insert(malformed.end(), {1.0, 1.0, 0.0});
    REQUIRE(
        ovspatialquery::detail::prepareConvexHull(runtime, malformed, std::numeric_limits<uint64_t>::max(), output) ==
        ovspatialquery::detail::ConvexHullPreparationOutcome::invalidArgument);
    REQUIRE(output == preserved);

    REQUIRE(preserved->residentByteCount() > 0u);
    REQUIRE(
        ovspatialquery::detail::prepareConvexHull(runtime, validPoints, preserved->residentByteCount() - 1u, output) ==
        ovspatialquery::detail::ConvexHullPreparationOutcome::resourceExhausted);
    REQUIRE(output == preserved);
    std::shared_ptr<const ovspatialquery::detail::PreparedConvexHull> exactBudgetHull;
    REQUIRE(ovspatialquery::detail::prepareConvexHull(runtime, validPoints, preserved->residentByteCount(),
                                                      exactBudgetHull) ==
            ovspatialquery::detail::ConvexHullPreparationOutcome::success);
    REQUIRE(exactBudgetHull->residentByteCount() <= preserved->residentByteCount());

    ovspatialquery_test_oracle_destroy_runtime(runtime);
}

void testOracleConvexPreparationIsRepeatedAndConcurrentDeterministic()
{
    ovspatialquery_backend_runtime_t* runtime = nullptr;
    REQUIRE(ovspatialquery_test_oracle_create_runtime(&runtime) == OVSPATIALQUERY_API_SUCCESS);
    std::vector<float> authoredPoints = unitCubeConvex().vertices;
    authoredPoints.insert(authoredPoints.end(), {0.0, 0.0, 0.0});

    constexpr size_t                                                               kPreparationCount = 16u;
    std::vector<std::shared_ptr<const ovspatialquery::detail::PreparedConvexHull>> prepared(kPreparationCount);
    std::vector<ovspatialquery::detail::ConvexHullPreparationOutcome>              outcomes(
        kPreparationCount, ovspatialquery::detail::ConvexHullPreparationOutcome::internalFailure);
    std::vector<std::thread> workers;
    workers.reserve(kPreparationCount);
    for (size_t index = 0u; index < kPreparationCount; ++index)
    {
        workers.emplace_back(
            [&, index]()
            {
                outcomes[index] = ovspatialquery::detail::prepareConvexHull(
                    runtime, authoredPoints, std::numeric_limits<uint64_t>::max(), prepared[index]);
            });
    }
    for (std::thread& worker : workers)
    {
        worker.join();
    }
    for (size_t index = 0u; index < kPreparationCount; ++index)
    {
        REQUIRE(outcomes[index] == ovspatialquery::detail::ConvexHullPreparationOutcome::success);
        REQUIRE(prepared[index] != nullptr);
        REQUIRE(prepared[index]->vertices() == prepared[0]->vertices());
        REQUIRE(prepared[index]->faceVertexCounts() == prepared[0]->faceVertexCounts());
        REQUIRE(prepared[index]->faceVertexIndices() == prepared[0]->faceVertexIndices());
    }

    ovspatialquery_test_oracle_destroy_runtime(runtime);
}

void testConvexConstructionDistinguishesMalformedAndNonConvexAtomically()
{
    ovspatialquery::detail::SphereGeometrySource sentinelSource{};
    sentinelSource.radius = 1.0;
    std::shared_ptr<const ovspatialquery::detail::ExactGeometry> preserved;
    REQUIRE(ovspatialquery::detail::createSphereExactGeometry(sentinelSource, preserved) == OVSPATIALQUERY_API_SUCCESS);
    std::shared_ptr<const ovspatialquery::detail::ExactGeometry> output = preserved;

    ovspatialquery::detail::ConvexGeometrySource source = triangulatedCubeConvex();
    REQUIRE(ovspatialquery::detail::createConvexExactGeometry(source, output) ==
            ovspatialquery::detail::ConvexConstructionOutcome::success);
    REQUIRE(output != preserved);
    const double                             diagonalOrigin[3]    = {-3.0, 0.0, 0.0};
    const float                              diagonalDirection[3] = {1.0f, 0.0f, 0.0f};
    ovspatialquery::detail::ExactGeometryHit diagonalHit{};
    REQUIRE(output->sweepSphere(diagonalOrigin, 0.25, diagonalDirection, 0.0, 10.0, diagonalHit));
    REQUIRE(near(diagonalHit.distance, 1.75));
    REQUIRE(!diagonalHit.hasTriangleSource);

    output                        = preserved;
    source.vertices[6u * 3u]      = 0.25;
    source.vertices[6u * 3u + 1u] = 0.25;
    source.vertices[6u * 3u + 2u] = 0.25;
    REQUIRE(ovspatialquery::detail::createConvexExactGeometry(source, output) ==
            ovspatialquery::detail::ConvexConstructionOutcome::nonConvex);
    REQUIRE(output == preserved);

    source = unitCubeConvex();
    std::reverse(source.faceVertexIndices.begin(), source.faceVertexIndices.end());
    REQUIRE(ovspatialquery::detail::createConvexExactGeometry(source, output) ==
            ovspatialquery::detail::ConvexConstructionOutcome::malformed);
    REQUIRE(output == preserved);

    source = unitCubeConvex();
    source.faceVertexCounts.pop_back();
    source.faceVertexIndices.resize(source.faceVertexIndices.size() - 4u);
    REQUIRE(ovspatialquery::detail::createConvexExactGeometry(source, output) ==
            ovspatialquery::detail::ConvexConstructionOutcome::malformed);
    REQUIRE(output == preserved);

    source = unitCubeConvex();
    source.faceVertexCounts.push_back(4u);
    source.faceVertexIndices.insert(source.faceVertexIndices.end(), {0u, 3u, 2u, 1u});
    REQUIRE(ovspatialquery::detail::createConvexExactGeometry(source, output) ==
            ovspatialquery::detail::ConvexConstructionOutcome::malformed);
    REQUIRE(output == preserved);

    source = unitCubeConvex();
    for (size_t vertex = 0u; vertex < source.vertices.size() / 3u; ++vertex)
    {
        source.vertices[vertex * 3u + 2u] = 0.0;
    }
    REQUIRE(ovspatialquery::detail::createConvexExactGeometry(source, output) ==
            ovspatialquery::detail::ConvexConstructionOutcome::malformed);
    REQUIRE(output == preserved);

    source             = unitCubeConvex();
    source.vertices[0] = std::numeric_limits<float>::infinity();
    REQUIRE(ovspatialquery::detail::createConvexExactGeometry(source, output) ==
            ovspatialquery::detail::ConvexConstructionOutcome::malformed);
    REQUIRE(output == preserved);
}

void testConvexResidentAccountingIsCapacityBasedAndSaturating()
{
    const uint64_t maximum = std::numeric_limits<uint64_t>::max();
    REQUIRE(ovspatialquery::detail::saturatingResidentByteAdd(17u, 3u, 4u) == 29u);
    REQUIRE(ovspatialquery::detail::saturatingResidentByteAdd(maximum - 1u, 1u, 2u) == maximum);
    REQUIRE(ovspatialquery::detail::saturatingResidentByteAdd(17u, maximum, 2u) == maximum);

    std::shared_ptr<const ovspatialquery::detail::ExactGeometry> quadGeometry;
    REQUIRE(ovspatialquery::detail::createConvexExactGeometry(unitCubeConvex(), quadGeometry) ==
            ovspatialquery::detail::ConvexConstructionOutcome::success);
    std::shared_ptr<const ovspatialquery::detail::ExactGeometry> triangulatedGeometry;
    REQUIRE(ovspatialquery::detail::createConvexExactGeometry(triangulatedCubeConvex(), triangulatedGeometry) ==
            ovspatialquery::detail::ConvexConstructionOutcome::success);
    REQUIRE(quadGeometry->residentByteCount() > 0u);
    REQUIRE(triangulatedGeometry->residentByteCount() > quadGeometry->residentByteCount());
}

void testConvexSweepHandlesEdgeVertexAndInsideStarts()
{
    const ovspatialquery::detail::ConvexGeometrySource           source = unitCubeConvex();
    std::shared_ptr<const ovspatialquery::detail::ExactGeometry> geometry;
    REQUIRE(ovspatialquery::detail::createConvexExactGeometry(source, geometry) ==
            ovspatialquery::detail::ConvexConstructionOutcome::success);
    const float direction[3] = {1.0f, 0.0f, 0.0f};

    const double                             edgeOrigin[3] = {-3.0, 1.3, 0.0};
    ovspatialquery::detail::ExactGeometryHit hit{};
    REQUIRE(geometry->sweepSphere(edgeOrigin, 0.5, direction, 0.0, 10.0, hit));
    REQUIRE(near(hit.distance, 1.6));
    REQUIRE(near(hit.position[0], -1.0));
    REQUIRE(near(hit.position[1], 1.0));
    REQUIRE(std::abs(static_cast<double>(hit.normal[0]) + 0.8) < 1.0e-6);
    REQUIRE(std::abs(static_cast<double>(hit.normal[1]) - 0.6) < 1.0e-6);
    REQUIRE(!hit.hasTriangleSource);

    const double vertexOrigin[3] = {-3.0, 1.2, 1.3};
    hit                          = {};
    REQUIRE(geometry->sweepSphere(vertexOrigin, 0.5, direction, 0.0, 10.0, hit));
    REQUIRE(near(hit.distance, 2.0 - std::sqrt(0.12)));
    REQUIRE(near(hit.position[0], -1.0));
    REQUIRE(near(hit.position[1], 1.0));
    REQUIRE(near(hit.position[2], 1.0));
    REQUIRE(!hit.hasTriangleSource);

    const double insideOrigin[3] = {0.0, 0.0, 0.0};
    hit                          = {};
    REQUIRE(geometry->raycast(insideOrigin, direction, 0.0, 10.0, false, hit));
    REQUIRE(near(hit.distance, 0.0));
    REQUIRE(near(std::abs(hit.position[0]), 1.0) || near(std::abs(hit.position[1]), 1.0) ||
            near(std::abs(hit.position[2]), 1.0));
    REQUIRE(!hit.hasTriangleSource);

    hit = {};
    REQUIRE(geometry->sweepSphere(insideOrigin, 0.1, direction, 0.0, 10.0, hit));
    REQUIRE(near(hit.distance, 0.0));
    REQUIRE(near(std::abs(hit.position[0]), 1.0) || near(std::abs(hit.position[1]), 1.0) ||
            near(std::abs(hit.position[2]), 1.0));
    REQUIRE(!hit.hasTriangleSource);
}

void testConvexSatHandlesCrossAndNearParallelAxes()
{
    const ovspatialquery::detail::ConvexGeometrySource           source = unitCubeConvex();
    std::shared_ptr<const ovspatialquery::detail::ExactGeometry> geometry;
    REQUIRE(ovspatialquery::detail::createConvexExactGeometry(source, geometry) ==
            ovspatialquery::detail::ConvexConstructionOutcome::success);

    ovspatialquery_overlap_shape_t overlap{};
    overlap.kind                      = OVSPATIALQUERY_OVERLAP_BOX;
    overlap.data.box.center[0]        = -0.5175798560958356;
    overlap.data.box.center[1]        = 1.57037194352597;
    overlap.data.box.center[2]        = -0.46036257059313357;
    overlap.data.box.rotation_xyzw[0] = 0.9942325262f;
    overlap.data.box.rotation_xyzw[1] = 0.00000249289428f;
    overlap.data.box.rotation_xyzw[2] = -0.00000210393116f;
    overlap.data.box.rotation_xyzw[3] = -0.1072459034f;
    overlap.data.box.half_extents[0]  = 1.8050461880862714;
    overlap.data.box.half_extents[1]  = 0.4960647005471401;
    overlap.data.box.half_extents[2]  = 0.4019088161410764;
    REQUIRE(!geometry->overlaps(overlap));

    constexpr double kTinyAngle       = 1.0e-9;
    overlap                           = {};
    overlap.kind                      = OVSPATIALQUERY_OVERLAP_BOX;
    overlap.data.box.center[1]        = 2.0 + 0.5e-9;
    overlap.data.box.rotation_xyzw[2] = static_cast<float>(std::sin(kTinyAngle * 0.5));
    overlap.data.box.rotation_xyzw[3] = static_cast<float>(std::cos(kTinyAngle * 0.5));
    overlap.data.box.half_extents[0]  = 1.0;
    overlap.data.box.half_extents[1]  = 1.0;
    overlap.data.box.half_extents[2]  = 1.0;
    REQUIRE(geometry->overlaps(overlap));
    overlap.data.box.center[1] = 2.0 + 2.0e-9;
    REQUIRE(!geometry->overlaps(overlap));
}

void testSphereSupportsEveryFirstStageExactQuery()
{
    ovspatialquery::detail::SphereGeometrySource source{};
    source.center = {5.0, 0.0, 0.0};
    source.radius = 1.0;

    std::shared_ptr<const ovspatialquery::detail::ExactGeometry> geometry;
    REQUIRE(ovspatialquery::detail::createSphereExactGeometry(source, geometry) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(geometry != nullptr);
    REQUIRE(geometry->family() == ovspatialquery::detail::ExactGeometryFamily::sphere);
    REQUIRE(near(geometry->worldBounds().minimum[0], 4.0));
    REQUIRE(near(geometry->worldBounds().maximum[0], 6.0));

    const double                             origin[3]    = {0.0, 0.0, 0.0};
    const float                              direction[3] = {1.0f, 0.0f, 0.0f};
    ovspatialquery::detail::ExactGeometryHit hit{};
    REQUIRE(geometry->raycast(origin, direction, 0.0, 10.0, false, hit));
    REQUIRE(near(hit.distance, 4.0));
    REQUIRE(near(hit.position[0], 4.0));
    REQUIRE(near(hit.normal[0], -1.0));
    REQUIRE(!hit.hasTriangleSource);

    hit = {};
    REQUIRE(geometry->sweepSphere(origin, 0.5, direction, 0.0, 10.0, hit));
    REQUIRE(near(hit.distance, 3.5));
    REQUIRE(near(hit.position[0], 4.0));
    REQUIRE(near(hit.normal[0], -1.0));

    ovspatialquery_overlap_shape_t overlap{};
    overlap.kind                  = OVSPATIALQUERY_OVERLAP_SPHERE;
    overlap.data.sphere.center[0] = 6.5;
    overlap.data.sphere.radius    = 0.5;
    REQUIRE(geometry->overlaps(overlap));
    overlap.data.sphere.center[0] = 6.5001;
    REQUIRE(!geometry->overlaps(overlap));

    overlap                         = {};
    overlap.kind                    = OVSPATIALQUERY_OVERLAP_AABB;
    overlap.data.aabb.min_corner[0] = 3.5;
    overlap.data.aabb.min_corner[1] = -0.25;
    overlap.data.aabb.min_corner[2] = -0.25;
    overlap.data.aabb.max_corner[0] = 4.0;
    overlap.data.aabb.max_corner[1] = 0.25;
    overlap.data.aabb.max_corner[2] = 0.25;
    REQUIRE(geometry->overlaps(overlap));
    overlap.data.aabb.max_corner[0] = 3.9999;
    REQUIRE(!geometry->overlaps(overlap));

    overlap                           = {};
    overlap.kind                      = OVSPATIALQUERY_OVERLAP_BOX;
    overlap.data.box.center[0]        = 3.75;
    overlap.data.box.rotation_xyzw[3] = 1.0f;
    overlap.data.box.half_extents[0]  = 0.25;
    overlap.data.box.half_extents[1]  = 0.25;
    overlap.data.box.half_extents[2]  = 0.25;
    REQUIRE(geometry->overlaps(overlap));
    overlap.data.box.center[0] = 3.7499;
    REQUIRE(!geometry->overlaps(overlap));
}

// Stored-sphere rays, tubes and sweeps through the exact geometry's own raycast and sweep,
// judged by the independent oracle (see stored_sphere_paths.h).
void testSpherePathsMeetIndependentOracle()
{
    namespace paths = stored_sphere_paths;
    std::shared_ptr<const ovspatialquery::detail::ExactGeometry> spheres[3];
    for (uint32_t target = 0u; target < 3u; ++target)
    {
        ovspatialquery::detail::SphereGeometrySource source{};
        source.center = {paths::kTargets[target].center[0], paths::kTargets[target].center[1],
                         paths::kTargets[target].center[2]};
        source.radius = paths::kTargets[target].radius;
        REQUIRE(ovspatialquery::detail::createSphereExactGeometry(source, spheres[target]) ==
                OVSPATIALQUERY_API_SUCCESS);
    }
    const std::vector<paths::Row> rows = paths::rows();
    size_t                        hits = 0u;
    for (size_t index = 0u; index < rows.size(); ++index)
    {
        const paths::Row&                        row = rows[index];
        ovspatialquery::detail::ExactGeometryHit hit{};
        const bool  found   = row.radius == 0.0 ? spheres[row.target]->raycast(row.origin, row.direction, row.minimum,
                                                                               row.maximum, false, hit)
                                                : spheres[row.target]->sweepSphere(row.origin, row.radius, row.direction,
                                                                                   row.minimum, row.maximum, hit);
        const char* failure = paths::verdict(row, found, hit.distance, hit.position, hit.normal);
        if (failure != nullptr)
        {
            std::fprintf(stderr, "stored sphere path row %zu: %s\n", index, failure);
        }
        REQUIRE(failure == nullptr);
        hits += found ? 1u : 0u;
    }
    REQUIRE(hits > rows.size() / 2u);
}

// A ray or sphere already touching or inside a stored solid at a positive minimum reports it
// at that minimum, with the contact of a start moved there (ADR-0018), for every solid family:
// rays and sweeps whose center is inside, and sweeps overlapping with the center outside. The
// contact lies on the stored surface with the outward normal: just outside along the normal
// the solid is clear, just inside it is not.
void testSolidsReportInitialOverlapAtTheMinimum()
{
    using namespace ovspatialquery::detail;
    std::shared_ptr<const ExactGeometry> solids[4];
    SphereGeometrySource                 sphere{};
    sphere.center = {5.0, 0.0, 0.0};
    sphere.radius = 1.0;
    REQUIRE(createSphereExactGeometry(sphere, solids[0]) == OVSPATIALQUERY_API_SUCCESS);
    BoxGeometrySource box{};
    box.center = {5.0, 0.0, 0.0};
    box.axes   = {
        std::array<double, 3u>{0.0, 1.0, 0.0},
        std::array<double, 3u>{-1.0, 0.0, 0.0},
        std::array<double, 3u>{0.0, 0.0, 1.0},
    };
    box.halfExtents = {1.0, 0.75, 0.5};
    REQUIRE(createBoxExactGeometry(box, solids[1]) == OVSPATIALQUERY_API_SUCCESS);
    CapsuleGeometrySource capsule{};
    capsule.endpointA = {5.0, -1.0, 0.0};
    capsule.endpointB = {5.0, 1.0, 0.0};
    capsule.radius    = 0.5;
    REQUIRE(createCapsuleExactGeometry(capsule, solids[2]) == OVSPATIALQUERY_API_SUCCESS);
    ConvexGeometrySource convex = unitCubeConvex();
    convex.worldFromLocal       = {0.0, -0.75, 0.0, 5.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.75, 0.0, 0.0, 0.0, 0.0, 1.0};
    REQUIRE(createConvexExactGeometry(convex, solids[3]) == ConvexConstructionOutcome::success);

    const double origin[3]        = {1.0, 0.1, 0.05};
    const float  directions[2][3] = {
        {1.0f, 0.0f, 0.0f},
        {float(0.9985 * std::sqrt(1 + 9e-6)), float(0.05 * std::sqrt(1 + 9e-6)), float(-0.02 * std::sqrt(1 + 9e-6))},
    };
    const auto clear = [](const std::shared_ptr<const ExactGeometry>& solid, const double point[3])
    {
        ovspatialquery_overlap_shape_t probe{};
        probe.kind = OVSPATIALQUERY_OVERLAP_SPHERE;
        std::copy_n(point, 3, probe.data.sphere.center);
        probe.data.sphere.radius = 1e-9;
        return !solid->overlaps(probe);
    };
    size_t checked = 0u;
    for (const std::shared_ptr<const ExactGeometry>& solid : solids)
    {
        for (const float* direction : {directions[0], directions[1]})
        {
            for (const double radius : {0.0, 0.25})
            {
                const auto trace = [&](const double start[3], double minimum, ExactGeometryHit& hit)
                {
                    return radius == 0.0 ? solid->raycast(start, direction, minimum, 20.0, false, hit)
                                         : solid->sweepSphere(start, radius, direction, minimum, 20.0, hit);
                };
                ExactGeometryHit first{};
                REQUIRE(trace(origin, 0.0, first));
                // Just past the first contact (the center outside for a sweep) and deep inside.
                for (const double past : {0.1, 0.6})
                {
                    const double     minimum = first.distance + past;
                    ExactGeometryHit clipped{};
                    REQUIRE(trace(origin, minimum, clipped));
                    REQUIRE(clipped.distance == minimum);
                    double moved[3] = {};
                    for (uint32_t axis = 0u; axis < 3u; ++axis)
                    {
                        moved[axis] = std::fma(double(direction[axis]), minimum, origin[axis]);
                    }
                    ExactGeometryHit shifted{};
                    REQUIRE(trace(moved, 0.0, shifted));
                    REQUIRE(shifted.distance == 0.0);
                    double outside[3] = {}, inside[3] = {};
                    for (uint32_t axis = 0u; axis < 3u; ++axis)
                    {
                        REQUIRE(std::abs(clipped.position[axis] - shifted.position[axis]) <= 1e-12);
                        REQUIRE(std::abs(clipped.normal[axis] - shifted.normal[axis]) <= 1e-6f);
                        outside[axis] = clipped.position[axis] + 1e-6 * clipped.normal[axis];
                        inside[axis]  = clipped.position[axis] - 1e-6 * clipped.normal[axis];
                    }
                    REQUIRE(clear(solid, outside) && !clear(solid, inside));
                    ++checked;
                }
            }
        }
    }
    REQUIRE(checked == 32u);

    // A start on the capsule's segment, inside its span (where the foot rounds a little along
    // the segment) or at its end, has no unique nearest surface point. Its contact still lies on
    // the surface with the outward normal, whether the path crosses the segment, runs along it,
    // or runs into it from the end.
    const double onSegment[2][3] = {{5.0, 0.3, 0.0}, {5.0, 1.0, 0.0}};
    const float  motions[3][3]   = {{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, -1.0f, 0.0f}};
    for (const double* start : {onSegment[0], onSegment[1]})
    {
        for (const float* motion : {motions[0], motions[1], motions[2]})
        {
            for (const double radius : {0.0, 0.25})
            {
                ExactGeometryHit hit{};
                REQUIRE(radius == 0.0 ? solids[2]->raycast(start, motion, 0.0, 20.0, false, hit)
                                      : solids[2]->sweepSphere(start, radius, motion, 0.0, 20.0, hit));
                REQUIRE(hit.distance == 0.0);
                double outside[3] = {}, inside[3] = {};
                for (uint32_t axis = 0u; axis < 3u; ++axis)
                {
                    outside[axis] = hit.position[axis] + 1e-6 * hit.normal[axis];
                    inside[axis]  = hit.position[axis] - 1e-6 * hit.normal[axis];
                }
                REQUIRE(clear(solids[2], outside) && !clear(solids[2], inside));
            }
        }
    }
}

void testSphereRejectsUntruthfulDefinitions()
{
    ovspatialquery::detail::SphereGeometrySource source{};
    source.radius = 0.0;
    std::shared_ptr<const ovspatialquery::detail::ExactGeometry> preserved(
        reinterpret_cast<const ovspatialquery::detail::ExactGeometry*>(1u),
        [](const ovspatialquery::detail::ExactGeometry*) {});
    std::shared_ptr<const ovspatialquery::detail::ExactGeometry> geometry;
    geometry = preserved;
    REQUIRE(ovspatialquery::detail::createSphereExactGeometry(source, geometry) == OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(geometry == preserved);
    source.radius    = 1.0;
    source.center[1] = std::numeric_limits<double>::infinity();
    REQUIRE(ovspatialquery::detail::createSphereExactGeometry(source, geometry) == OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(geometry == preserved);
}

void testBoxSupportsEveryExactQueryWithoutTriangleFeatures()
{
    ovspatialquery::detail::BoxGeometrySource source{};
    source.center = {5.0, 0.0, 0.0};
    source.axes   = {
        std::array<double, 3u>{0.0, 1.0, 0.0},
        std::array<double, 3u>{-1.0, 0.0, 0.0},
        std::array<double, 3u>{0.0, 0.0, 1.0},
    };
    source.halfExtents = {1.0, 2.0, 0.5};

    std::shared_ptr<const ovspatialquery::detail::ExactGeometry> geometry;
    REQUIRE(ovspatialquery::detail::createBoxExactGeometry(source, geometry) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(geometry != nullptr);
    REQUIRE(geometry->family() == ovspatialquery::detail::ExactGeometryFamily::box);
    REQUIRE(near(geometry->worldBounds().minimum[0], 3.0));
    REQUIRE(near(geometry->worldBounds().maximum[0], 7.0));
    REQUIRE(near(geometry->worldBounds().minimum[1], -1.0));
    REQUIRE(near(geometry->worldBounds().maximum[1], 1.0));

    const double                             origin[3]    = {0.0, 0.0, 0.0};
    const float                              direction[3] = {1.0f, 0.0f, 0.0f};
    ovspatialquery::detail::ExactGeometryHit hit{};
    REQUIRE(geometry->raycast(origin, direction, 0.0, 10.0, false, hit));
    REQUIRE(near(hit.distance, 3.0));
    REQUIRE(near(hit.position[0], 3.0));
    REQUIRE(near(hit.normal[0], -1.0));
    REQUIRE(!hit.hasTriangleSource);

    hit = {};
    REQUIRE(geometry->sweepSphere(origin, 0.25, direction, 0.0, 10.0, hit));
    REQUIRE(near(hit.distance, 2.75));
    REQUIRE(near(hit.position[0], 3.0));
    REQUIRE(near(hit.normal[0], -1.0));
    REQUIRE(!hit.hasTriangleSource);

    ovspatialquery_overlap_shape_t overlap{};
    overlap.kind                  = OVSPATIALQUERY_OVERLAP_SPHERE;
    overlap.data.sphere.center[0] = 2.5;
    overlap.data.sphere.radius    = 0.5;
    REQUIRE(geometry->overlaps(overlap));
    overlap.data.sphere.center[0] = 2.4999;
    REQUIRE(!geometry->overlaps(overlap));

    overlap                         = {};
    overlap.kind                    = OVSPATIALQUERY_OVERLAP_AABB;
    overlap.data.aabb.min_corner[0] = 2.5;
    overlap.data.aabb.min_corner[1] = -0.25;
    overlap.data.aabb.min_corner[2] = -0.25;
    overlap.data.aabb.max_corner[0] = 3.0;
    overlap.data.aabb.max_corner[1] = 0.25;
    overlap.data.aabb.max_corner[2] = 0.25;
    REQUIRE(geometry->overlaps(overlap));
    overlap.data.aabb.max_corner[0] = 2.9999;
    REQUIRE(!geometry->overlaps(overlap));

    overlap                           = {};
    overlap.kind                      = OVSPATIALQUERY_OVERLAP_BOX;
    overlap.data.box.center[0]        = 2.5;
    overlap.data.box.rotation_xyzw[3] = 1.0f;
    overlap.data.box.half_extents[0]  = 0.5;
    overlap.data.box.half_extents[1]  = 0.25;
    overlap.data.box.half_extents[2]  = 0.25;
    REQUIRE(geometry->overlaps(overlap));
    overlap.data.box.center[0] = 2.4999;
    REQUIRE(!geometry->overlaps(overlap));
}

void testCapsuleSupportsEveryExactQueryWithoutTriangleFeatures()
{
    ovspatialquery::detail::CapsuleGeometrySource source{};
    source.endpointA = {5.0, -1.0, 0.0};
    source.endpointB = {5.0, 1.0, 0.0};
    source.radius    = 0.5;

    std::shared_ptr<const ovspatialquery::detail::ExactGeometry> geometry;
    REQUIRE(ovspatialquery::detail::createCapsuleExactGeometry(source, geometry) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(geometry != nullptr);
    REQUIRE(geometry->family() == ovspatialquery::detail::ExactGeometryFamily::capsule);
    REQUIRE(near(geometry->worldBounds().minimum[0], 4.5));
    REQUIRE(near(geometry->worldBounds().maximum[0], 5.5));
    REQUIRE(near(geometry->worldBounds().minimum[1], -1.5));
    REQUIRE(near(geometry->worldBounds().maximum[1], 1.5));

    const double                             origin[3]    = {0.0, 0.0, 0.0};
    const float                              direction[3] = {1.0f, 0.0f, 0.0f};
    ovspatialquery::detail::ExactGeometryHit hit{};
    REQUIRE(geometry->raycast(origin, direction, 0.0, 10.0, false, hit));
    REQUIRE(near(hit.distance, 4.5));
    REQUIRE(near(hit.position[0], 4.5));
    REQUIRE(near(hit.normal[0], -1.0));
    REQUIRE(!hit.hasTriangleSource);

    hit = {};
    REQUIRE(geometry->sweepSphere(origin, 0.25, direction, 0.0, 10.0, hit));
    REQUIRE(near(hit.distance, 4.25));
    REQUIRE(near(hit.position[0], 4.5));
    REQUIRE(near(hit.normal[0], -1.0));
    REQUIRE(!hit.hasTriangleSource);

    const double capOrigin[3]    = {5.0, -3.0, 0.0};
    const float  capDirection[3] = {0.0f, 1.0f, 0.0f};
    hit                          = {};
    REQUIRE(geometry->raycast(capOrigin, capDirection, 0.0, 10.0, false, hit));
    REQUIRE(near(hit.distance, 1.5));
    REQUIRE(near(hit.position[1], -1.5));
    REQUIRE(near(hit.normal[1], -1.0));

    // Tangency is a hit, but a ray just outside the cylinder must remain a miss.
    const double tangentOrigin[3] = {0.0, 0.0, 0.5};
    hit                           = {};
    REQUIRE(geometry->raycast(tangentOrigin, direction, 0.0, 10.0, false, hit));
    REQUIRE(near(hit.distance, 5.0));
    REQUIRE(near(hit.position[0], 5.0));
    REQUIRE(near(hit.position[2], 0.5));
    REQUIRE(near(hit.normal[2], 1.0));
    const double outsideOrigin[3] = {0.0, 0.0, 0.5000001};
    REQUIRE(!geometry->raycast(outsideOrigin, direction, 0.0, 10.0, false, hit));

    ovspatialquery_overlap_shape_t overlap{};
    overlap.kind                  = OVSPATIALQUERY_OVERLAP_SPHERE;
    overlap.data.sphere.center[0] = 4.0;
    overlap.data.sphere.radius    = 0.5;
    REQUIRE(geometry->overlaps(overlap));
    overlap.data.sphere.center[0] = 3.9999;
    REQUIRE(!geometry->overlaps(overlap));

    overlap                         = {};
    overlap.kind                    = OVSPATIALQUERY_OVERLAP_AABB;
    overlap.data.aabb.min_corner[0] = 4.0;
    overlap.data.aabb.min_corner[1] = -0.25;
    overlap.data.aabb.min_corner[2] = -0.25;
    overlap.data.aabb.max_corner[0] = 4.5;
    overlap.data.aabb.max_corner[1] = 0.25;
    overlap.data.aabb.max_corner[2] = 0.25;
    REQUIRE(geometry->overlaps(overlap));
    overlap.data.aabb.max_corner[0] = 4.4999;
    REQUIRE(!geometry->overlaps(overlap));

    overlap                           = {};
    overlap.kind                      = OVSPATIALQUERY_OVERLAP_BOX;
    overlap.data.box.center[0]        = 4.25;
    overlap.data.box.rotation_xyzw[3] = 1.0f;
    overlap.data.box.half_extents[0]  = 0.25;
    overlap.data.box.half_extents[1]  = 0.25;
    overlap.data.box.half_extents[2]  = 0.25;
    REQUIRE(geometry->overlaps(overlap));
    overlap.data.box.center[0] = 4.2499;
    REQUIRE(!geometry->overlaps(overlap));
}

void testAnalyticGeometryRejectsUntruthfulDefinitionsAtomically()
{
    ovspatialquery::detail::SphereGeometrySource sentinelSource{};
    sentinelSource.radius = 1.0;
    std::shared_ptr<const ovspatialquery::detail::ExactGeometry> preserved;
    REQUIRE(ovspatialquery::detail::createSphereExactGeometry(sentinelSource, preserved) == OVSPATIALQUERY_API_SUCCESS);
    std::shared_ptr<const ovspatialquery::detail::ExactGeometry> output = preserved;

    ovspatialquery::detail::BoxGeometrySource box{};
    box.axes = {
        std::array<double, 3u>{1.0, 0.0, 0.0},
        std::array<double, 3u>{0.0, 1.0, 0.0},
        std::array<double, 3u>{0.0, 0.0, 1.0},
    };
    box.halfExtents = {1.0, 1.0, 0.0};
    REQUIRE(ovspatialquery::detail::createBoxExactGeometry(box, output) == OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(output == preserved);
    box.halfExtents[2] = 1.0;
    box.axes[2]        = {0.0, 1.0, 0.0};
    REQUIRE(ovspatialquery::detail::createBoxExactGeometry(box, output) == OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(output == preserved);

    ovspatialquery::detail::CapsuleGeometrySource capsule{};
    capsule.endpointA = {1.0, 2.0, 3.0};
    capsule.endpointB = {1.0, 3.0, 3.0};
    capsule.radius    = std::numeric_limits<double>::infinity();
    REQUIRE(ovspatialquery::detail::createCapsuleExactGeometry(capsule, output) == OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(output == preserved);
}

void testBoxRoundedCornerSweepAndRotatedSatAreExact()
{
    ovspatialquery::detail::BoxGeometrySource source{};
    source.axes = {
        std::array<double, 3u>{1.0, 0.0, 0.0},
        std::array<double, 3u>{0.0, 1.0, 0.0},
        std::array<double, 3u>{0.0, 0.0, 1.0},
    };
    source.halfExtents = {1.0, 1.0, 1.0};
    std::shared_ptr<const ovspatialquery::detail::ExactGeometry> geometry;
    REQUIRE(ovspatialquery::detail::createBoxExactGeometry(source, geometry) == OVSPATIALQUERY_API_SUCCESS);

    const double                             origin[3]    = {-3.0, 2.0, 0.0};
    const float                              direction[3] = {1.0f, 0.0f, 0.0f};
    ovspatialquery::detail::ExactGeometryHit hit{};
    REQUIRE(geometry->sweepSphere(origin, 1.0, direction, 0.0, 10.0, hit));
    REQUIRE(near(hit.distance, 2.0));
    REQUIRE(near(hit.position[0], -1.0));
    REQUIRE(near(hit.position[1], 1.0));
    REQUIRE(near(hit.normal[0], 0.0));
    REQUIRE(near(hit.normal[1], 1.0));

    constexpr float                kSinPiOverEight = 0.38268343236508977173F;
    constexpr float                kCosPiOverEight = 0.92387953251128675613F;
    ovspatialquery_overlap_shape_t overlap{};
    overlap.kind                      = OVSPATIALQUERY_OVERLAP_BOX;
    overlap.data.box.center[0]        = 1.0 + std::sqrt(2.0) - 1.0e-6;
    overlap.data.box.rotation_xyzw[2] = kSinPiOverEight;
    overlap.data.box.rotation_xyzw[3] = kCosPiOverEight;
    overlap.data.box.half_extents[0]  = 1.0;
    overlap.data.box.half_extents[1]  = 1.0;
    overlap.data.box.half_extents[2]  = 0.5;
    REQUIRE(geometry->overlaps(overlap));
    overlap.data.box.center[0] += 2.0e-6;
    REQUIRE(!geometry->overlaps(overlap));

    const double nearMissOrigin[3] = {-3.0, 1.0 + 1.0e-7, 0.0};
    hit                            = {};
    REQUIRE(!geometry->raycast(nearMissOrigin, direction, 0.0, 10.0, false, hit));

    const double insideOrigin[3] = {0.0, 0.0, 0.0};
    hit                          = {};
    REQUIRE(geometry->raycast(insideOrigin, direction, 0.0, 10.0, false, hit));
    REQUIRE(near(hit.distance, 0.0));
    REQUIRE(near(std::abs(hit.position[0]), 1.0));

    overlap                           = {};
    overlap.kind                      = OVSPATIALQUERY_OVERLAP_BOX;
    constexpr float kNearUnitScale    = 1.000004F;
    overlap.data.box.center[0]        = 1.100005;
    overlap.data.box.rotation_xyzw[2] = 0.7071067811865475244F * kNearUnitScale;
    overlap.data.box.rotation_xyzw[3] = 0.7071067811865475244F * kNearUnitScale;
    overlap.data.box.half_extents[0]  = 1.0;
    overlap.data.box.half_extents[1]  = 0.1;
    overlap.data.box.half_extents[2]  = 0.1;
    REQUIRE(!geometry->overlaps(overlap));
}

void testRoundedPrimitivePathsKeepAnalyticContactRoots()
{
    ovspatialquery::detail::BoxGeometrySource boxSource{};
    boxSource.axes = {
        std::array<double, 3u>{1.0, 0.0, 0.0},
        std::array<double, 3u>{0.0, 1.0, 0.0},
        std::array<double, 3u>{0.0, 0.0, 1.0},
    };
    boxSource.halfExtents = {0.5, 0.375, 0.5};
    std::shared_ptr<const ovspatialquery::detail::ExactGeometry> box;
    REQUIRE(ovspatialquery::detail::createBoxExactGeometry(boxSource, box) == OVSPATIALQUERY_API_SUCCESS);

    const double                             origin[3]    = {1.3, -0.15, 0.0};
    const float                              direction[3] = {-1.0f, 0.0f, 0.0f};
    ovspatialquery::detail::ExactGeometryHit hit{};
    REQUIRE(box->sweepSphere(origin, 0.35, direction, 0.0, 1.6, hit));
    REQUIRE(near(hit.distance, 0.45));
    REQUIRE(near(hit.position[0], 0.5));

    const double tangentOrigin[3] = {1.3, 0.725, 0.0};
    hit                           = {};
    REQUIRE(box->sweepSphere(tangentOrigin, 0.35, direction, 0.0, 1.6, hit));
    REQUIRE(std::abs(hit.distance - 0.8) < 2.0e-8);

    const double missOrigin[3] = {1.3, 0.7250001, 0.0};
    hit                        = {};
    REQUIRE(!box->sweepSphere(missOrigin, 0.35, direction, 0.0, 1.6, hit));

    ovspatialquery::detail::CapsuleGeometrySource capsuleSource{};
    capsuleSource.endpointA = {0.0, -0.375, 0.0};
    capsuleSource.endpointB = {0.0, 0.375, 0.0};
    capsuleSource.radius    = 0.35;
    std::shared_ptr<const ovspatialquery::detail::ExactGeometry> capsule;
    REQUIRE(ovspatialquery::detail::createCapsuleExactGeometry(capsuleSource, capsule) == OVSPATIALQUERY_API_SUCCESS);
    hit = {};
    REQUIRE(capsule->raycast(origin, direction, 0.0, 1.6, false, hit));
    REQUIRE(near(hit.distance, 0.95));
    const double distantCapsuleMiss[3] = {1.0e8, 0.6, 0.3};
    REQUIRE(!capsule->raycast(distantCapsuleMiss, direction, 0.0, 2.0e8, false, hit));
    const double distantBoxMiss[3] = {1.0e8, 0.6, 0.8};
    REQUIRE(!box->sweepSphere(distantBoxMiss, 0.35, direction, 0.0, 2.0e8, hit));
}

void testBoxSweepKeepsFaceAndEdgeContactsAcrossApproachDistances()
{
    using namespace ovspatialquery::detail;
    // Float-authored scales from Ovstage must retain the sphere radius at contact.
    // The uncentered discriminant could reject the true root and report the box face.
    for (bool strip : {false, true})
    {
        BoxGeometrySource source{};
        source.axes        = {std::array<double, 3u>{1, 0, 0}, std::array<double, 3u>{0, 1, 0},
                              std::array<double, 3u>{0, 0, 1}};
        source.center      = strip ? std::array<double, 3u>{0, 0.06, 0.65} : std::array<double, 3u>{0, 0.8, -2};
        source.halfExtents = strip ? std::array<double, 3u>{double(2.4f) / 2, double(0.12f) / 2, double(0.36f) / 2}
                                   : std::array<double, 3u>{double(1.3f) / 2, double(1.6f) / 2, double(1.1f) / 2};
        std::shared_ptr<const ExactGeometry> geometry;
        REQUIRE(createBoxExactGeometry(source, geometry) == OVSPATIALQUERY_API_SUCCESS);
        const float direction[3] = {0, 0, -1};
        for (double radius : {0.18, 0.25})
        {
            const double y      = strip ? 0.25 : 0.7;
            const double height = std::max(0.0, y - source.center[1] - source.halfExtents[1]);
            const double reach  = std::sqrt(radius * radius - height * height);
            const double front  = source.center[2] + source.halfExtents[2];
            for (double z : {1.5, 5.0, 7.5, 15.0})
            {
                const double     origin[3] = {-0.35, y, z};
                const double     expected  = z - front - reach;
                ExactGeometryHit hit{};
                REQUIRE(geometry->sweepSphere(origin, radius, direction, 0, 20, hit));
                REQUIRE(std::abs(hit.distance - expected) < 1.0e-10);
                REQUIRE(std::abs(hit.position[2] - front) < 1.0e-10);
                REQUIRE(!geometry->sweepSphere(origin, radius, direction, 0, expected - 1.0e-5, hit));
            }
            const double     overlapping[3] = {0, y, front};
            ExactGeometryHit hit{};
            REQUIRE(geometry->sweepSphere(overlapping, radius, direction, 0, 20, hit));
            REQUIRE(hit.distance == 0);
        }
    }
}

void testCapsuleRayDoesNotInflateSmallGeometryByAbsoluteTolerance()
{
    ovspatialquery::detail::CapsuleGeometrySource source{};
    source.endpointA = {5.0, -1.0, 0.0};
    source.endpointB = {5.0, 1.0, 0.0};
    source.radius    = 1.0e-6;
    std::shared_ptr<const ovspatialquery::detail::ExactGeometry> geometry;
    REQUIRE(ovspatialquery::detail::createCapsuleExactGeometry(source, geometry) == OVSPATIALQUERY_API_SUCCESS);

    const double                             origin[3]    = {0.0, 0.0, 1.1e-6};
    const float                              direction[3] = {1.0f, 0.0f, 0.0f};
    ovspatialquery::detail::ExactGeometryHit hit{};
    REQUIRE(!geometry->raycast(origin, direction, 0.0, 10.0, false, hit));
}

void testCapsuleRayKeepsOrdinaryObliqueSurfaceContact()
{
    using namespace ovspatialquery::detail;
    // A float32 angular-lattice direction crossing the cylindrical interior.
    // Re-evaluating the quadratic root used to reject this ordinary surface hit.
    CapsuleGeometrySource source{};
    source.endpointA = {1.103371262550354, -2.375, 6.932626247406006};
    source.endpointB = {1.103371262550354, -3.625, 6.932626247406006};
    source.radius    = .5;
    std::shared_ptr<const ExactGeometry> geometry;
    REQUIRE(createCapsuleExactGeometry(source, geometry) == OVSPATIALQUERY_API_SUCCESS);
    const double     origin[3]{2, -3, 1};
    const float      direction[3]{-0.13820332288742065f, 0.011454295367002487f, 0.9903376698493958f};
    ExactGeometryHit hit{};
    REQUIRE(geometry->raycast(origin, direction, .25, 10, false, hit));
    constexpr double expectedDistance = 5.504628651126874;
    REQUIRE(std::abs(hit.distance - expectedDistance) < 1.0e-9);
    for (uint32_t axis = 0; axis < 3; ++axis)
        REQUIRE(std::abs(hit.position[axis] - std::fma(double(direction[axis]), expectedDistance, origin[axis])) <
                1.0e-9);
    const double dx = hit.position[0] - source.endpointA[0];
    const double dz = hit.position[2] - source.endpointA[2];
    REQUIRE(std::abs(dx * dx + dz * dz - source.radius * source.radius) < 1.0e-12);
    REQUIRE(std::abs(hit.normal[0] - dx / source.radius) < 1.0e-6);
    REQUIRE(std::abs(hit.normal[1]) < 1.0e-6);
    REQUIRE(std::abs(hit.normal[2] - dz / source.radius) < 1.0e-6);
    REQUIRE(!hit.hasTriangleSource);
    const double missOrigin[3]{4, -3, 1};
    REQUIRE(!geometry->raycast(missOrigin, direction, .25, 10, false, hit));
}

void testBoxRayKeepsSmallDirectionComponentsOverLongPaths()
{
    ovspatialquery::detail::BoxGeometrySource source{};
    source.axes = {
        std::array<double, 3u>{1.0, 0.0, 0.0},
        std::array<double, 3u>{0.0, 1.0, 0.0},
        std::array<double, 3u>{0.0, 0.0, 1.0},
    };
    source.halfExtents = {1.0e20, 1.0, 1.0e20};
    std::shared_ptr<const ovspatialquery::detail::ExactGeometry> geometry;
    REQUIRE(ovspatialquery::detail::createBoxExactGeometry(source, geometry) == OVSPATIALQUERY_API_SUCCESS);

    const double                             origin[3]    = {0.0, 2.0, 0.0};
    const float                              direction[3] = {1.0f, -1.0e-16f, 0.0f};
    ovspatialquery::detail::ExactGeometryHit hit{};
    REQUIRE(geometry->raycast(origin, direction, 0.0, 2.0e16, false, hit));
    REQUIRE(hit.distance > 9.0e15 && hit.distance < 1.1e16);
    REQUIRE(near(hit.position[1], 1.0));
}

void testAnalyticInstancesRemainImmutableAcrossRevisions()
{
    ovspatialquery::detail::BoxGeometrySource source{};
    source.center = {5.0, 0.0, 0.0};
    source.axes   = {
        std::array<double, 3u>{1.0, 0.0, 0.0},
        std::array<double, 3u>{0.0, 1.0, 0.0},
        std::array<double, 3u>{0.0, 0.0, 1.0},
    };
    source.halfExtents = {1.0, 1.0, 1.0};
    std::shared_ptr<const ovspatialquery::detail::ExactGeometry> retained;
    REQUIRE(ovspatialquery::detail::createBoxExactGeometry(source, retained) == OVSPATIALQUERY_API_SUCCESS);
    source.center[0] = 9.0;
    std::shared_ptr<const ovspatialquery::detail::ExactGeometry> replacement;
    REQUIRE(ovspatialquery::detail::createBoxExactGeometry(source, replacement) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(near(retained->worldBounds().minimum[0], 4.0));
    REQUIRE(near(replacement->worldBounds().minimum[0], 8.0));
}

void testPrimitiveTransformQualificationMatchesFamilyTruth()
{
    const std::array<double, 16u> rotatedUniformScale = {
        0.0, -2.0, 0.0, 3.0, 2.0, 0.0, 0.0, 4.0, 0.0, 0.0, 2.0, 5.0, 0.0, 0.0, 0.0, 1.0,
    };
    ovspatialquery::detail::QualifiedPrimitiveTransform qualified{};
    REQUIRE(ovspatialquery::detail::qualifyPrimitiveTransform(rotatedUniformScale,
                                                              ovspatialquery::detail::PrimitiveScaleRule::uniform,
                                                              qualified) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(near(qualified.center[0], 3.0));
    REQUIRE(near(qualified.center[1], 4.0));
    REQUIRE(near(qualified.center[2], 5.0));
    REQUIRE(near(qualified.scales[0], 2.0));
    REQUIRE(near(qualified.scales[1], 2.0));
    REQUIRE(near(qualified.scales[2], 2.0));
    REQUIRE(near(qualified.axes[0][1], 1.0));
    REQUIRE(near(qualified.axes[1][0], -1.0));
    REQUIRE(near(qualified.axes[2][2], 1.0));

    std::array<double, 16u> independentBoxScale = rotatedUniformScale;
    independentBoxScale[10]                     = 3.0;
    REQUIRE(ovspatialquery::detail::qualifyPrimitiveTransform(
                independentBoxScale, ovspatialquery::detail::PrimitiveScaleRule::independentOrthogonal, qualified) ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(near(qualified.scales[0], 2.0));
    REQUIRE(near(qualified.scales[1], 2.0));
    REQUIRE(near(qualified.scales[2], 3.0));

    std::array<double, 16u> unsupported                                 = rotatedUniformScale;
    unsupported[10]                                                     = 1.0;
    const ovspatialquery::detail::QualifiedPrimitiveTransform preserved = qualified;
    REQUIRE(ovspatialquery::detail::qualifyPrimitiveTransform(unsupported,
                                                              ovspatialquery::detail::PrimitiveScaleRule::uniform,
                                                              qualified) == OVSPATIALQUERY_API_NOT_SUPPORTED);
    REQUIRE(qualified.center == preserved.center);
    REQUIRE(qualified.axes == preserved.axes);
    REQUIRE(qualified.scales == preserved.scales);

    unsupported    = rotatedUniformScale;
    unsupported[2] = 0.5;
    REQUIRE(ovspatialquery::detail::qualifyPrimitiveTransform(
                unsupported, ovspatialquery::detail::PrimitiveScaleRule::independentOrthogonal, qualified) ==
            OVSPATIALQUERY_API_NOT_SUPPORTED);

    unsupported     = rotatedUniformScale;
    unsupported[12] = 0.25;
    REQUIRE(ovspatialquery::detail::qualifyPrimitiveTransform(unsupported,
                                                              ovspatialquery::detail::PrimitiveScaleRule::uniform,
                                                              qualified) == OVSPATIALQUERY_API_NOT_SUPPORTED);

    const std::array<double, 16u> tinyNonUniformScale = {
        1.0e-9, 0.0, 0.0, 0.0, 0.0, 2.0e-9, 0.0, 0.0, 0.0, 0.0, 3.0e-9, 0.0, 0.0, 0.0, 0.0, 1.0,
    };
    REQUIRE(ovspatialquery::detail::qualifyPrimitiveTransform(tinyNonUniformScale,
                                                              ovspatialquery::detail::PrimitiveScaleRule::uniform,
                                                              qualified) == OVSPATIALQUERY_API_NOT_SUPPORTED);
    REQUIRE(qualified.center == preserved.center);
    REQUIRE(qualified.axes == preserved.axes);
    REQUIRE(qualified.scales == preserved.scales);

    std::array<double, 16u> tinyShear = {
        1.0e-9, 1.0e-10, 0.0, 0.0, 0.0, 1.0e-9, 0.0, 0.0, 0.0, 0.0, 1.0e-9, 0.0, 0.0, 0.0, 0.0, 1.0,
    };
    REQUIRE(ovspatialquery::detail::qualifyPrimitiveTransform(
                tinyShear, ovspatialquery::detail::PrimitiveScaleRule::independentOrthogonal, qualified) ==
            OVSPATIALQUERY_API_NOT_SUPPORTED);
    tinyShear     = rotatedUniformScale;
    tinyShear[12] = 1.0e-15;
    REQUIRE(ovspatialquery::detail::qualifyPrimitiveTransform(tinyShear,
                                                              ovspatialquery::detail::PrimitiveScaleRule::uniform,
                                                              qualified) == OVSPATIALQUERY_API_NOT_SUPPORTED);
}

void testMeshHitPreservesSourceFaceAndCorners()
{
    ovspatialquery::detail::TriangleMeshSource mesh{};
    mesh.key      = 303u;
    mesh.vertices = {
        5.0, -1.0, -1.0, 5.0, -1.0, 1.0, 5.0, 1.0, 1.0, 5.0, 1.0, -1.0,
    };
    mesh.triangles     = {0u, 1u, 2u, 0u, 2u, 3u};
    mesh.sourceFaces   = {17u, 17u};
    mesh.sourceCorners = {0u, 1u, 2u, 0u, 2u, 3u};
    const std::array<double, 16u> identity{
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    };
    std::shared_ptr<const ovspatialquery::detail::PreparedTriangleMesh>   preparedMesh;
    std::shared_ptr<const ovspatialquery::detail::TriangleObjectGeometry> prepared;
    REQUIRE(ovspatialquery::detail::prepareTriangleMesh(mesh, preparedMesh) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery::detail::bindPreparedTriangleGeometry(preparedMesh, identity, false, prepared) ==
            OVSPATIALQUERY_API_SUCCESS);

    std::shared_ptr<const ovspatialquery::detail::ExactGeometry> geometry;
    REQUIRE(ovspatialquery::detail::createTriangleMeshExactGeometry(prepared, geometry) == OVSPATIALQUERY_API_SUCCESS);
    std::shared_ptr<const ovspatialquery::detail::ExactGeometry> secondGeometry;
    REQUIRE(ovspatialquery::detail::createTriangleMeshExactGeometry(prepared, secondGeometry) ==
            OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery::detail::PopulationObject firstObject{};
    firstObject.id.value                                  = 1u;
    firstObject.triangleGeometry                          = prepared;
    firstObject.exactGeometry                             = geometry;
    ovspatialquery::detail::PopulationObject secondObject = firstObject;
    secondObject.id.value                                 = 2u;
    secondObject.exactGeometry                            = secondGeometry;
    const uint64_t singleResident                         = ovspatialquery::detail::exactGeometryResidentByteCount(
        std::vector<ovspatialquery::detail::PopulationObject>{firstObject});
    const uint64_t sharedResident = ovspatialquery::detail::exactGeometryResidentByteCount(
        std::vector<ovspatialquery::detail::PopulationObject>{firstObject, secondObject});
    REQUIRE(singleResident == geometry->residentByteCount());
    REQUIRE(sharedResident > singleResident);
    REQUIRE(sharedResident < singleResident * 2u);

    ovspatialquery::detail::PreparedPopulationInput retainedInput;
    REQUIRE(ovspatialquery::detail::PreparedPopulationInput::prepareCopiedSet(
                9, {firstObject, secondObject}, retainedInput) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(retainedInput.retainedByteCount() >= sharedResident);
    ovspatialquery::detail::ObjectTable materialized;
    std::optional<uint64_t>             measuredGeometryCharge;
    REQUIRE(retainedInput.materialize({}, materialized, measuredGeometryCharge) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(measuredGeometryCharge.has_value());
    REQUIRE(*measuredGeometryCharge == sharedResident);
    REQUIRE(materialized.size() == 2u);
    REQUIRE(ovspatialquery::detail::exactGeometryResidentByteCount(materialized) == *measuredGeometryCharge);

    // Inputs materialized without admission accounting still require a recount.
    ovspatialquery::detail::PreparedPopulationInput unmeasuredInput;
    REQUIRE(ovspatialquery::detail::PreparedPopulationInput::prepareCopiedSet(10, {firstObject}, unmeasuredInput) ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(unmeasuredInput.materialize({}, materialized, measuredGeometryCharge) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(!measuredGeometryCharge.has_value());
    REQUIRE(ovspatialquery::detail::exactGeometryResidentByteCount(materialized) == singleResident);

    ovspatialquery::detail::ExactGeometryResidentAccumulator saturated;
    REQUIRE(saturated.add(&firstObject, UINT64_MAX));
    REQUIRE(!saturated.add(&secondObject, 1u));
    REQUIRE(saturated.residentByteCount() == UINT64_MAX);
    const double                             origin[3]    = {0.0, 0.75, 0.0};
    const float                              direction[3] = {1.0f, 0.0f, 0.0f};
    ovspatialquery::detail::ExactGeometryHit hit{};
    REQUIRE(geometry->raycast(origin, direction, 0.0, 10.0, false, hit));
    REQUIRE(hit.hasTriangleSource);
    REQUIRE(hit.generatedTriangle == 1u);
    REQUIRE(hit.sourceFace == 17u);
    REQUIRE(hit.sourceCorners[0] == 0u);
    REQUIRE(hit.sourceCorners[1] == 2u);
    REQUIRE(hit.sourceCorners[2] == 3u);

    ovspatialquery::detail::TriangleMeshSource wideFaceMesh = mesh;
    wideFaceMesh.key                                        = 304u;
    wideFaceMesh.sourceFaces                                = {UINT64_C(4294967296), UINT64_C(4294967296)};
    std::shared_ptr<const ovspatialquery::detail::PreparedTriangleMesh>   wideFaceMeshPrepared;
    std::shared_ptr<const ovspatialquery::detail::TriangleObjectGeometry> wideFacePrepared;
    REQUIRE(ovspatialquery::detail::prepareTriangleMesh(wideFaceMesh, wideFaceMeshPrepared) ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery::detail::bindPreparedTriangleGeometry(wideFaceMeshPrepared, identity, false,
                                                                 wideFacePrepared) == OVSPATIALQUERY_API_SUCCESS);
    std::shared_ptr<const ovspatialquery::detail::ExactGeometry> wideFaceGeometry;
    REQUIRE(ovspatialquery::detail::createTriangleMeshExactGeometry(wideFacePrepared, wideFaceGeometry) ==
            OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery::detail::ExactGeometryHit wideFaceHit{};
    REQUIRE(wideFaceGeometry->raycast(origin, direction, 0.0, 10.0, false, wideFaceHit));
    REQUIRE(wideFaceHit.sourceFace == UINT64_C(4294967296));
}

} // namespace

int main()
{
    testResidentAccountingDeduplicatesAndReleasesScratch();
    testFiveFamilyDescriptionExportRejectsMissingOwnerAtomically();
    testFiveFamilyDescriptionsExposeExactTruthAndRebuildEquivalentKernels();
    testRetainedGenerationOwnsDescriptionAndKeepsIdentitySeparate();
    testConvexSupportsEveryExactQueryWithoutFeatures();
    testBoxAboveANonplanarFaceIsApart();
    testOraclePreparesAuthoredPointSetIntoExactConvexHull();
    testOracleKeepsEveryPointOfADenseHull();
    testOracleConvexPreparationRejectsMalformedAndBudgetAtomically();
    testOracleConvexPreparationIsRepeatedAndConcurrentDeterministic();
    testConvexConstructionDistinguishesMalformedAndNonConvexAtomically();
    testConvexResidentAccountingIsCapacityBasedAndSaturating();
    testConvexSweepHandlesEdgeVertexAndInsideStarts();
    testConvexSatHandlesCrossAndNearParallelAxes();
    testSphereSupportsEveryFirstStageExactQuery();
    testSpherePathsMeetIndependentOracle();
    testSolidsReportInitialOverlapAtTheMinimum();
    testSphereRejectsUntruthfulDefinitions();
    testBoxSupportsEveryExactQueryWithoutTriangleFeatures();
    testCapsuleSupportsEveryExactQueryWithoutTriangleFeatures();
    testAnalyticGeometryRejectsUntruthfulDefinitionsAtomically();
    testBoxRoundedCornerSweepAndRotatedSatAreExact();
    testRoundedPrimitivePathsKeepAnalyticContactRoots();
    testBoxSweepKeepsFaceAndEdgeContactsAcrossApproachDistances();
    testCapsuleRayDoesNotInflateSmallGeometryByAbsoluteTolerance();
    testCapsuleRayKeepsOrdinaryObliqueSurfaceContact();
    testBoxRayKeepsSmallDirectionComponentsOverLongPaths();
    testAnalyticInstancesRemainImmutableAcrossRevisions();
    testPrimitiveTransformQualificationMatchesFamilyTruth();
    testMeshHitPreservesSourceFaceAndCorners();
    return 0;
}
