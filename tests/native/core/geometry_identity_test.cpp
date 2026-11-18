/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../../source/ovspatialquery_geometry_identity.h"
#include "../../../source/ovspatialquery_internal.h"
#include "../../support/native/test_failure.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <thread>
#include <vector>

namespace
{

using ovspatialquery::detail::BoxDefinitionValue;
using ovspatialquery::detail::CapsuleAxis;
using ovspatialquery::detail::CapsuleDefinitionValue;
using ovspatialquery::detail::ConvexDefinitionValue;
using ovspatialquery::detail::ConvexHullPreparationBudget;
using ovspatialquery::detail::ConvexHullPreparationOutcome;
using ovspatialquery::detail::ExactGeometryFamily;
using ovspatialquery::detail::GeometryDefinitionValue;
using ovspatialquery::detail::GeometryIdentityOutcome;
using ovspatialquery::detail::GeometryIdentityRegistry;
using ovspatialquery::detail::GeometryIdentityRegistryInspection;
using ovspatialquery::detail::GeometryReference;
using ovspatialquery::detail::kFirstAdapterGeometryValue;
using ovspatialquery::detail::MeshDefinitionValue;
using ovspatialquery::detail::SphereDefinitionValue;

struct ConvexChargeProbe
{
    uint64_t limit        = std::numeric_limits<uint64_t>::max();
    uint64_t current      = 0u;
    uint64_t peak         = 0u;
    uint64_t reservations = 0u;
    uint64_t releases     = 0u;
};

struct ConvexChargeToken
{
    ConvexChargeToken(ConvexChargeProbe* probeValue, uint64_t byteCount) : probe(probeValue), bytes(byteCount) {}

    ConvexChargeToken(const ConvexChargeToken&)            = delete;
    ConvexChargeToken& operator=(const ConvexChargeToken&) = delete;

    ConvexChargeProbe* probe = nullptr;
    uint64_t           bytes = 0u;

    ~ConvexChargeToken()
    {
        if (probe != nullptr)
        {
            REQUIRE(probe->current >= bytes);
            probe->current -= bytes;
            ++probe->releases;
        }
    }
};

ConvexHullPreparationOutcome reserveConvexCharge(void* state, uint64_t bytes, std::shared_ptr<void>& outOwner) noexcept
{
    outOwner.reset();
    ConvexChargeProbe* probe = static_cast<ConvexChargeProbe*>(state);
    if (probe == nullptr || bytes > probe->limit - std::min(probe->limit, probe->current))
    {
        return ConvexHullPreparationOutcome::resourceExhausted;
    }
    try
    {
        probe->current += bytes;
        probe->peak = std::max(probe->peak, probe->current);
        ++probe->reservations;
        outOwner = std::make_shared<ConvexChargeToken>(probe, bytes);
        return ConvexHullPreparationOutcome::success;
    }
    catch (...)
    {
        probe->current -= bytes;
        return ConvexHullPreparationOutcome::outOfMemory;
    }
}

GeometryDefinitionValue sphere(double radius) { return SphereDefinitionValue{radius}; }

GeometryDefinitionValue convex()
{
    ConvexDefinitionValue value;
    value.points = {
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
    };
    value.faceVertexCounts  = {3u, 3u, 3u, 3u};
    value.faceVertexIndices = {
        0u, 2u, 1u, 0u, 1u, 3u, 0u, 3u, 2u, 1u, 2u, 3u,
    };
    return value;
}

GeometryDefinitionValue mesh(bool doubleSided = false)
{
    MeshDefinitionValue value;
    value.points = {
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0,
    };
    value.faceVertexCounts  = {4u};
    value.faceVertexIndices = {0u, 1u, 2u, 3u};
    value.doubleSided       = doubleSided;
    value.sourceFaces       = {0u, 0u};
    value.sourceCorners     = {0u, 1u, 2u, 0u, 2u, 3u};
    return value;
}

std::vector<GeometryReference> intern(GeometryIdentityRegistry&                   registry,
                                      const std::vector<GeometryDefinitionValue>& definitions)
{
    GeometryIdentityRegistry::Batch      batch;
    std::vector<GeometryDefinitionValue> staged = definitions; // The registry moves what it keeps.
    REQUIRE(registry.stage(staged, batch) == GeometryIdentityOutcome::success);
    const std::vector<GeometryReference> references = batch.references();
    REQUIRE(!references.empty());
    REQUIRE(batch.stagingResidentByteCount() >= references.size() * sizeof(GeometryReference));
    registry.commit(batch);
    const uint64_t released = batch.releaseStaging();
    REQUIRE(released >= references.size() * sizeof(GeometryReference));
    REQUIRE(batch.stagingResidentByteCount() == 0u);
    return references;
}

GeometryIdentityRegistryInspection inspect(const GeometryIdentityRegistry& registry)
{
    GeometryIdentityRegistryInspection inspection;
    REQUIRE(registry.inspect(inspection));
    return inspection;
}

void testCanonicalSharingAndDefinitionTruth()
{
    GeometryIdentityRegistry             registry;
    const std::vector<GeometryReference> first = intern(registry, {
                                                                      sphere(2.0),
                                                                      BoxDefinitionValue{4.0},
                                                                      CapsuleDefinitionValue{0.5, 3.0, CapsuleAxis::y},
                                                                      convex(),
                                                                      mesh(),
                                                                  });
    REQUIRE(first.size() == 5u);
    for (const GeometryReference& reference : first)
    {
        REQUIRE(reference.value != 0u);
    }
    REQUIRE(first[0].family == ExactGeometryFamily::sphere);
    REQUIRE(first[1].family == ExactGeometryFamily::box);
    REQUIRE(first[2].family == ExactGeometryFamily::capsule);
    REQUIRE(first[3].family == ExactGeometryFamily::convex);
    REQUIRE(first[4].family == ExactGeometryFamily::triangleMesh);

    const std::vector<GeometryReference> repeated =
        intern(registry, {sphere(2.0), BoxDefinitionValue{4.0}, convex(), mesh()});
    REQUIRE(repeated[0] == first[0]);
    REQUIRE(repeated[1] == first[1]);
    REQUIRE(repeated[2] == first[3]);
    REQUIRE(repeated[3] == first[4]);

    const std::vector<GeometryReference> changed =
        intern(registry, {
                             sphere(3.0),
                             BoxDefinitionValue{5.0},
                             CapsuleDefinitionValue{0.5, 3.0, CapsuleAxis::x},
                             mesh(true),
                         });
    REQUIRE(changed[0] != first[0]);
    REQUIRE(changed[1] != first[1]);
    REQUIRE(changed[2] != first[2]);
    REQUIRE(changed[3] != first[4]);

    MeshDefinitionValue topologyChanged = std::get<MeshDefinitionValue>(mesh());
    topologyChanged.faceVertexIndices   = {0u, 3u, 2u, 1u};
    REQUIRE(intern(registry, {topologyChanged})[0] != first[4]);

    MeshDefinitionValue mappingChanged = std::get<MeshDefinitionValue>(mesh());
    mappingChanged.sourceCorners       = {0u, 2u, 1u, 0u, 3u, 2u};
    REQUIRE(intern(registry, {mappingChanged})[0] != first[4]);
    REQUIRE(inspect(registry).definitionCount == 11u);
    REQUIRE(inspect(registry).residentByteCount > 0u);
}

void testSignedZeroAndConcurrentSameDefinitionSharing()
{
    GeometryIdentityRegistry registry;
    MeshDefinitionValue      positiveZeroMesh = std::get<MeshDefinitionValue>(mesh());
    MeshDefinitionValue      negativeZeroMesh = positiveZeroMesh;
    negativeZeroMesh.points[0]                = -0.0f;
    const GeometryReference positiveZero      = intern(registry, {positiveZeroMesh})[0];
    const GeometryReference negativeZero      = intern(registry, {negativeZeroMesh})[0];
    REQUIRE(positiveZero == negativeZero);

    // Shape keys preserve the normalized metric's bits, including signed zero.
    positiveZeroMesh.shapeMetric    = {4, 0, 0, 0, 1, 0, 0, 0, 0.25};
    negativeZeroMesh                = positiveZeroMesh;
    negativeZeroMesh.shapeMetric[1] = -0.0;
    GeometryIdentityRegistry metricRegistry;
    REQUIRE(intern(metricRegistry, {positiveZeroMesh})[0] != intern(metricRegistry, {negativeZeroMesh})[0]);

    constexpr size_t                           threadCount = 8u;
    std::array<GeometryReference, threadCount> references{};
    std::array<std::thread, threadCount>       threads;
    for (size_t index = 0u; index < threadCount; ++index)
    {
        threads[index] =
            std::thread([&registry, &references, index]() { references[index] = intern(registry, {sphere(7.0)})[0]; });
    }
    for (std::thread& thread : threads)
    {
        thread.join();
    }
    for (size_t index = 1u; index < threadCount; ++index)
    {
        REQUIRE(references[index] == references[0]);
    }
    REQUIRE(inspect(registry).definitionCount == 2u);
}

void testBatchAndFailureAtomicity()
{
    GeometryIdentityRegistry        registry;
    GeometryIdentityRegistry::Batch empty;
    REQUIRE(registry.stage({}, empty) == GeometryIdentityOutcome::invalidArgument);
    REQUIRE(empty.references().empty());
    REQUIRE(inspect(registry).definitionCount == 0u);
    REQUIRE(inspect(registry).residentByteCount == 0u);

    // A malformed definition fails the whole batch: the valid one staged before it is not kept.
    GeometryIdentityRegistry::Batch malformed;
    REQUIRE(registry.stage({sphere(1.0), sphere(std::numeric_limits<double>::infinity())}, malformed) ==
            GeometryIdentityOutcome::invalidArgument);
    REQUIRE(inspect(registry).definitionCount == 0u);
    REQUIRE(inspect(registry).residentByteCount == 0u);

    const std::vector<GeometryReference> admitted = intern(registry, {sphere(1.0), sphere(2.0), sphere(1.0)});
    REQUIRE(admitted[0] == admitted[2]);
    REQUIRE(admitted[0] != admitted[1]);
    REQUIRE(inspect(registry).definitionCount == 2u);
}

void testRetiredBytesSurviveFailedStaging()
{
    GeometryIdentityRegistry                 registry;
    std::vector<std::shared_ptr<const void>> owners{std::make_shared<int>(1)};
    {
        GeometryIdentityRegistry::Batch batch;
        REQUIRE(registry.stage({sphere(1.0)}, batch, &owners) == GeometryIdentityOutcome::success);
        registry.commit(batch);
    }
    const uint64_t before = inspect(registry).residentByteCount;
    owners.clear();
    GeometryIdentityRegistry::Batch failed;
    REQUIRE(registry.stage({sphere(std::numeric_limits<double>::infinity())}, failed) ==
            GeometryIdentityOutcome::invalidArgument);
    const uint64_t released = before - inspect(registry).residentByteCount;
    REQUIRE(released > 0u);
    REQUIRE(failed.releasedResidentByteCount() == released);
    REQUIRE(inspect(registry).definitionCount == 0u);
    REQUIRE(intern(registry, {sphere(2.0)})[0].value == kFirstAdapterGeometryValue + 1u);
}

// A second live owner of a committed definition is charged when it is staged, joins
// only at commit, and its slots leave the ledger with the definition.
void testOwnerSlotsAreChargedAndReleasedWithTheDefinition()
{
    GeometryIdentityRegistry                 registry;
    std::vector<std::shared_ptr<const void>> first{std::make_shared<int>(1)};
    std::vector<std::shared_ptr<const void>> second{std::make_shared<int>(2)};
    GeometryIdentityRegistry::Batch          created;
    REQUIRE(registry.stage({sphere(1.0)}, created, &first) == GeometryIdentityOutcome::success);
    registry.commit(created);
    const uint64_t createdBytes = inspect(registry).residentByteCount;

    {
        GeometryIdentityRegistry::Batch abandoned;
        REQUIRE(registry.stage({sphere(1.0)}, abandoned, &second) == GeometryIdentityOutcome::success);
        REQUIRE(abandoned.additionalResidentByteCount() > 0u);
    }
    REQUIRE(inspect(registry).residentByteCount == createdBytes);
    // The abandoned staging recorded nothing: releasing the first owner retires it.
    first.clear();
    GeometryIdentityRegistry::Batch sweep;
    REQUIRE(registry.stage({sphere(2.0)}, sweep) == GeometryIdentityOutcome::success);
    REQUIRE(sweep.releasedResidentByteCount() == createdBytes);
    registry.commit(sweep);
    const uint64_t permanentBytes = inspect(registry).residentByteCount;

    first.push_back(std::make_shared<int>(3));
    GeometryIdentityRegistry::Batch recreated;
    REQUIRE(registry.stage({sphere(1.0)}, recreated, &first) == GeometryIdentityOutcome::success);
    registry.commit(recreated);
    GeometryIdentityRegistry::Batch shared;
    REQUIRE(registry.stage({sphere(1.0)}, shared, &second) == GeometryIdentityOutcome::success);
    const uint64_t growth = shared.additionalResidentByteCount();
    REQUIRE(growth > 0u);
    registry.commit(shared);
    const uint64_t sharedBytes = inspect(registry).residentByteCount;
    REQUIRE(sharedBytes == permanentBytes + createdBytes + growth);

    // Either owner keeps it; once both are released it leaves with every byte charged.
    first.clear();
    GeometryIdentityRegistry::Batch kept;
    REQUIRE(registry.stage({sphere(2.0)}, kept) == GeometryIdentityOutcome::success);
    REQUIRE(kept.releasedResidentByteCount() == 0u);
    registry.commit(kept);
    second.clear();
    GeometryIdentityRegistry::Batch retired;
    REQUIRE(registry.stage({sphere(2.0)}, retired) == GeometryIdentityOutcome::success);
    REQUIRE(retired.releasedResidentByteCount() == createdBytes + growth);
    registry.commit(retired);
    REQUIRE(inspect(registry).residentByteCount == permanentBytes);
}

void testInstanceIndependenceAndLifetimeMonotonicity()
{
    GeometryIdentityRegistry firstInstance;
    const GeometryReference  first  = intern(firstInstance, {sphere(1.0)})[0];
    const GeometryReference  second = intern(firstInstance, {sphere(2.0)})[0];
    REQUIRE(second.value > first.value);
    REQUIRE(intern(firstInstance, {sphere(1.0)})[0] == first);

    GeometryIdentityRegistry secondInstance;
    const GeometryReference  independent = intern(secondInstance, {sphere(1.0)})[0];
    REQUIRE(independent.value == kFirstAdapterGeometryValue);
    REQUIRE(inspect(secondInstance).definitionCount == 1u);
    REQUIRE(inspect(firstInstance).definitionCount == 2u);
}

void testStagedBatchSafelyOutlivesRegistry()
{
    GeometryIdentityRegistry::Batch batch;
    {
        std::unique_ptr<GeometryIdentityRegistry> registry(new GeometryIdentityRegistry());
        REQUIRE(registry->stage({sphere(4.0)}, batch) == GeometryIdentityOutcome::success);
        REQUIRE(batch.references().size() == 1u);
        registry.reset();
    }
    REQUIRE(batch.references().size() == 1u);
    REQUIRE(batch.releaseStaging() > 0u);
    REQUIRE(batch.references().empty());
}

void testTypedStagingCountsTemporaryConvexOwnerSlots()
{
    const std::shared_ptr<const ovspatialquery::detail::PreparedConvexHull> prepared =
        std::make_shared<const ovspatialquery::detail::PreparedConvexHull>(
            std::vector<double>{
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
            },
            std::vector<uint32_t>{3u, 3u, 3u, 3u},
            std::vector<uint32_t>{
                0u,
                2u,
                1u,
                0u,
                1u,
                3u,
                0u,
                3u,
                2u,
                1u,
                2u,
                3u,
            });
    std::vector<std::shared_ptr<const ovspatialquery::detail::PreparedConvexHull>> oneOwner;
    oneOwner.reserve(1u);
    oneOwner.push_back(prepared);
    std::vector<std::shared_ptr<const ovspatialquery::detail::PreparedConvexHull>> twoReferences;
    twoReferences.reserve(2u);
    twoReferences.push_back(prepared);
    twoReferences.push_back(prepared);

    const uint64_t oneOwnerBytes = ovspatialquery::detail::typedGeometryStagingResidentByteCount({}, {}, oneOwner);
    const uint64_t twoReferenceBytes =
        ovspatialquery::detail::typedGeometryStagingResidentByteCount({}, {}, twoReferences);
    REQUIRE(oneOwnerBytes == sizeof(oneOwner[0]));
    REQUIRE(twoReferenceBytes == oneOwnerBytes + sizeof(twoReferences[0]));
}

void testPreparedConvexOwnerIsAlwaysPairedWithResidentCharge()
{
    const std::vector<float> authoredPoints{
        0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
    };
    const std::vector<double>   vertices(authoredPoints.begin(), authoredPoints.end());
    const std::vector<uint32_t> faceVertexCounts{3u, 3u, 3u, 3u};
    const std::vector<uint32_t> faceVertexIndices{
        0u, 2u, 1u, 0u, 1u, 3u, 0u, 3u, 2u, 1u, 2u, 3u,
    };

    ConvexChargeProbe                                                 probe;
    const ConvexHullPreparationBudget                                 budget{&probe, reserveConvexCharge};
    std::shared_ptr<const ovspatialquery::detail::PreparedConvexHull> prepared;
    REQUIRE(ovspatialquery::detail::createPreparedConvexHull(vertices, faceVertexCounts, faceVertexIndices, prepared,
                                                             &budget) == ConvexHullPreparationOutcome::success);
    REQUIRE(prepared != nullptr);
    REQUIRE(probe.reservations == 1u);
    REQUIRE(probe.current == prepared->residentByteCount());
    REQUIRE(probe.peak == probe.current);
    prepared.reset();
    REQUIRE(probe.current == 0u);
    REQUIRE(probe.releases == 1u);

    ConvexChargeProbe rejectedProbe;
    rejectedProbe.limit = 0u;
    const ConvexHullPreparationBudget rejectedBudget{&rejectedProbe, reserveConvexCharge};
    REQUIRE(ovspatialquery::detail::createPreparedConvexHull(vertices, faceVertexCounts, faceVertexIndices, prepared,
                                                             &rejectedBudget) ==
            ConvexHullPreparationOutcome::resourceExhausted);
    REQUIRE(prepared == nullptr);
    REQUIRE(rejectedProbe.current == 0u);
    REQUIRE(rejectedProbe.reservations == 0u);
    REQUIRE(rejectedProbe.releases == 0u);
}

void testHoleyMeshStagingScalesWithFaceCount()
{
    const auto makeMesh = [](uint32_t faceCount)
    {
        MeshDefinitionValue value = std::get<MeshDefinitionValue>(mesh());
        value.faceVertexCounts.assign(faceCount, 3u);
        value.faceVertexIndices.clear();
        value.holeIndices.clear();
        value.sourceFaces.clear();
        value.sourceCorners.clear();
        for (uint32_t face = 0u; face < faceCount; ++face)
        {
            value.faceVertexIndices.insert(value.faceVertexIndices.end(), {0u, 1u, 2u});
            if (face % 2u == 0u)
                value.holeIndices.push_back(face);
            else
            {
                value.sourceFaces.push_back(face);
                value.sourceCorners.insert(value.sourceCorners.end(), {0u, 1u, 2u});
            }
        }
        return value;
    };
    const auto stageMicros = [](const MeshDefinitionValue& meshValue)
    {
        uint64_t best = std::numeric_limits<uint64_t>::max();
        for (unsigned repeat = 0u; repeat < 3u; ++repeat)
        {
            GeometryIdentityRegistry             registry;
            std::vector<GeometryDefinitionValue> definitions{meshValue};
            GeometryIdentityRegistry::Batch      batch;
            const auto                           start = std::chrono::steady_clock::now();
            REQUIRE(registry.stage(definitions, batch) == GeometryIdentityOutcome::success);
            const auto elapsed =
                std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start);
            best = std::min(best, static_cast<uint64_t>(elapsed.count()));
        }
        return best;
    };
    const uint64_t small = stageMicros(makeMesh(4000u));
    const uint64_t large = stageMicros(makeMesh(40000u));
    REQUIRE(large <= small * 30u + 20000u);
}

} // namespace

int main()
{
    testCanonicalSharingAndDefinitionTruth();
    testSignedZeroAndConcurrentSameDefinitionSharing();
    testBatchAndFailureAtomicity();
    testRetiredBytesSurviveFailedStaging();
    testOwnerSlotsAreChargedAndReleasedWithTheDefinition();
    testInstanceIndependenceAndLifetimeMonotonicity();
    testStagedBatchSafelyOutlivesRegistry();
    testTypedStagingCountsTemporaryConvexOwnerSlots();
    testPreparedConvexOwnerIsAlwaysPairedWithResidentCharge();
    testHoleyMeshStagingScalesWithFaceCount();
    return 0;
}
