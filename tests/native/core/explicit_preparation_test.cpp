/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "oracle_backend.h"
#include "ovspatialquery_cuda_backend.h"
#include "ovspatialquery_explicit_preparation.h"
#include "ovspatialquery_instance_host_ledger.h"
#include "ovspatialquery_prepared_array.h"
#include "test_failure.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

namespace
{
// Arm only around the input preparation call on this thread; backend allocations
// and the rest of the test use the normal allocation path.
thread_local int gAllocationsBeforeFailure = -1;
} // namespace

// Do not inline the malloc/free replacements into unrelated new expressions:
// GCC otherwise diagnoses their matched storage as mismatched new/delete.
#if defined(__GNUC__)
__attribute__((noinline))
#endif
void* operator new(std::size_t size)
{
    if (gAllocationsBeforeFailure == 0)
    {
        gAllocationsBeforeFailure = -1;
        throw std::bad_alloc();
    }
    if (gAllocationsBeforeFailure > 0)
    {
        --gAllocationsBeforeFailure;
    }
    if (void* memory = std::malloc(size == 0 ? 1 : size))
    {
        return memory;
    }
    throw std::bad_alloc();
}

#if defined(__GNUC__)
__attribute__((noinline))
#endif
void operator delete(void* memory) noexcept
{
    std::free(memory);
}

#if defined(__GNUC__)
__attribute__((noinline))
#endif
void operator delete(void* memory, std::size_t) noexcept
{
    ::operator delete(memory);
}

namespace ovspatialquery
{
namespace detail
{
void failNextConvexHullPreparationForTest(ovspatialquery_backend_runtime_t* backendRuntime,
                                          ConvexHullPreparationOutcome      failure);
}
} // namespace ovspatialquery

namespace
{

using ovspatialquery::detail::completePoseRows;
using ovspatialquery::detail::ConvexHullPreparationOutcome;
using ovspatialquery::detail::ExactGeometryFamily;
using ovspatialquery::detail::ExactGeometryHit;
using ovspatialquery::detail::ExplicitBoxDefinition;
using ovspatialquery::detail::ExplicitCapsuleDefinition;
using ovspatialquery::detail::ExplicitConvexDefinition;
using ovspatialquery::detail::ExplicitDefinitionInput;
using ovspatialquery::detail::ExplicitInstanceInput;
using ovspatialquery::detail::ExplicitPreparationRegistry;
using ovspatialquery::detail::ExplicitPreparedRegistration;
using ovspatialquery::detail::ExplicitPreparedUpdate;
using ovspatialquery::detail::ExplicitSphereDefinition;
using ovspatialquery::detail::ExplicitTriangleMeshDefinition;
using ovspatialquery::detail::GenerationLease;
using ovspatialquery::detail::GeometryReference;
using ovspatialquery::detail::HostLease;
using ovspatialquery::detail::HostTransaction;
using ovspatialquery::detail::InstanceHostLedger;
using ovspatialquery::detail::ObjectTable;
using ovspatialquery::detail::PopulationObject;
using ovspatialquery::detail::PopulationSourceBinding;
using ovspatialquery::detail::PopulationTable;
using ovspatialquery::detail::PreparedTriangleMesh;
using ovspatialquery::detail::triangleGeometryResidentByteCount;
using ovspatialquery::detail::TriangleMeshSource;
using ovspatialquery::detail::TriangleObjectGeometry;

ovspatialquery_backend_runtime_t* gBackendRuntime = nullptr;

void testPreparedInputAllocationFailures()
{
    using ovspatialquery::detail::PreparedInputArray;
    uint64_t                     original      = 7u;
    uint64_t                     replacement[] = {3u, 5u};
    int64_t                      shape         = 1;
    int64_t                      stride        = 1;
    ovspatialquery_input_array_t input{};
    input.tensor.data    = &original;
    input.tensor.device  = {kDLCPU, 0};
    input.tensor.dtype   = {kDLUInt, 64u, 1u};
    input.tensor.ndim    = 1;
    input.tensor.shape   = &shape;
    input.tensor.strides = &stride;

    for (const auto ownership :
         {OVSPATIALQUERY_INPUT_BORROWED_UNTIL_TERMINAL, OVSPATIALQUERY_INPUT_COPIED_ON_SUBMISSION})
    {
        input.ownership = ownership;
        for (int allowed = 0;; ++allowed)
        {
            PreparedInputArray output;
            input.tensor.data = &original;
            shape             = 1;
            REQUIRE(PreparedInputArray::prepare(&input, kDLUInt, 64u, 1, 1, "values", output) ==
                    OVSPATIALQUERY_API_SUCCESS);
            const uint8_t* originalData         = output.data();
            input.tensor.data                   = replacement;
            shape                               = 2;
            ovspatialquery_api_status_t status  = OVSPATIALQUERY_API_INTERNAL;
            bool                        escaped = false;
            gAllocationsBeforeFailure           = allowed;
            try
            {
                status = PreparedInputArray::prepare(&input, kDLUInt, 64u, 1, 1, "values", output);
            }
            catch (const std::bad_alloc&)
            {
                escaped = true;
            }
            gAllocationsBeforeFailure = -1;
            REQUIRE(!escaped);
            if (status == OVSPATIALQUERY_API_SUCCESS)
            {
                REQUIRE(output.shape(0) == 2);
                REQUIRE(output.byteCount() == sizeof(replacement));
                REQUIRE(std::memcmp(output.data(), replacement, sizeof(replacement)) == 0);
                break;
            }
            REQUIRE(status == OVSPATIALQUERY_API_OUT_OF_MEMORY);
            REQUIRE(output.initialized() && output.shape(0) == 1);
            REQUIRE(output.data() == originalData && output.byteCount() == sizeof(original));
            REQUIRE(std::memcmp(output.data(), &original, sizeof(original)) == 0);
        }
    }
}

GeometryReference reference(ExactGeometryFamily family, uint64_t value) { return GeometryReference{family, value}; }

ExplicitDefinitionInput sphere(uint64_t value, double radius)
{
    return {reference(ExactGeometryFamily::sphere, value), ExplicitSphereDefinition{radius}};
}

ExplicitDefinitionInput box(uint64_t value)
{
    return {reference(ExactGeometryFamily::box, value), ExplicitBoxDefinition{{1.0, 2.0, 3.0}}};
}

ExplicitDefinitionInput capsule(uint64_t value)
{
    return {reference(ExactGeometryFamily::capsule, value), ExplicitCapsuleDefinition{0.5, 2.0}};
}

ExplicitConvexDefinition tetrahedron()
{
    ExplicitConvexDefinition result;
    result.vertices = {
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
    };
    result.faceVertexCounts  = {3u, 3u, 3u, 3u};
    result.faceVertexIndices = {
        0u, 2u, 1u, 0u, 1u, 3u, 0u, 3u, 2u, 1u, 2u, 3u,
    };
    return result;
}

ExplicitDefinitionInput convex(uint64_t value)
{
    return {reference(ExactGeometryFamily::convex, value), tetrahedron()};
}

ExplicitTriangleMeshDefinition triangleMesh(bool doubleSided = false)
{
    ExplicitTriangleMeshDefinition result;
    result.vertices = {
        0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0,
    };
    result.triangleVertexIndices = {0u, 1u, 2u};
    result.doubleSided           = doubleSided;
    return result;
}

ExplicitDefinitionInput mesh(uint64_t value, bool doubleSided = false)
{
    return {reference(ExactGeometryFamily::triangleMesh, value), triangleMesh(doubleSided)};
}

ExplicitPreparedRegistration prepare(ExplicitPreparationRegistry&         registry,
                                     std::vector<ExplicitDefinitionInput> definitions)
{
    ExplicitPreparedRegistration registration;
    REQUIRE(registry.prepareRegistration(definitions, registration) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(registration.initialized());
    REQUIRE(registration.definitionCount() == definitions.size());
    return registration;
}

bool contains(const ExplicitPreparationRegistry& registry, GeometryReference geometry)
{
    std::vector<std::shared_ptr<const void>> owners;
    return registry.retainDefinitions({geometry}, owners) == OVSPATIALQUERY_API_SUCCESS;
}

ExplicitInstanceInput instance(uint64_t objectValue, GeometryReference geometry, double x = 0.0)
{
    ExplicitInstanceInput result;
    result.objectValue    = objectValue;
    result.geometry       = geometry;
    result.translation[0] = x;
    return result;
}

// The generation the worker publishes for `update` over `base`: its object-value
// history adds the update's IDs to the base's, and it records the update's
// exact-geometry charge, or the recount when the update derived none.
GenerationLease publish(const PopulationTable& table, ovspatialquery_ordinal_t ordinal,
                        const ExplicitPreparedUpdate& update, const GenerationLease& base = GenerationLease())
{
    std::vector<uint64_t> history = base.usedObjectValues();
    for (const PopulationObject& object : update.objects())
    {
        history.push_back(object.id.value);
    }
    std::sort(history.begin(), history.end());
    history.erase(std::unique(history.begin(), history.end()), history.end());
    GenerationLease generation;
    REQUIRE(table.buildGeneration(ordinal, update.objects(), {}, generation, {}, {}, false,
                                  std::make_shared<const std::vector<uint64_t>>(std::move(history)),
                                  update.exactGeometryCharge().value_or(
                                      ovspatialquery::detail::exactGeometryResidentByteCount(update.objects()))) ==
            OVSPATIALQUERY_API_SUCCESS);
    return generation;
}

bool near(double first, double second) { return std::abs(first - second) <= 1.0e-9; }

const PopulationObject& objectWithId(const ExplicitPreparedUpdate& prepared, uint64_t id)
{
    const auto found = std::find_if(prepared.objects().begin(), prepared.objects().end(),
                                    [id](const PopulationObject& object) { return object.id.value == id; });
    REQUIRE(found != prepared.objects().end());
    return *found;
}

void testRegistrationValidationAndAtomicCommit()
{
    ExplicitPreparationRegistry  accountingRegistry(gBackendRuntime, UINT64_MAX);
    ExplicitPreparedRegistration accountingSphere = prepare(accountingRegistry, {sphere(1000u, 1.0)});
    ExplicitPreparedRegistration accountingMesh   = prepare(accountingRegistry, {mesh(1001u)});
    REQUIRE(accountingSphere.definitionResidentByteCounts().size() == 1u);
    REQUIRE(accountingMesh.definitionRetainedByteCounts().size() == 1u);
    const ExplicitTriangleMeshDefinition accountingSource = triangleMesh();
    const uint64_t                       rawMeshPayload =
        static_cast<uint64_t>(accountingSource.vertices.capacity()) * sizeof(float) +
        static_cast<uint64_t>(accountingSource.triangleVertexIndices.capacity()) * sizeof(uint32_t);
    TriangleMeshSource accountingTriangle;
    accountingTriangle.key           = 1001u;
    accountingTriangle.vertices      = accountingSource.vertices;
    accountingTriangle.triangles     = accountingSource.triangleVertexIndices;
    accountingTriangle.sourceFaces   = {0u};
    accountingTriangle.sourceCorners = {0u, 1u, 2u};
    std::shared_ptr<const PreparedTriangleMesh> accountingMeshPrepared;
    REQUIRE(ovspatialquery::detail::prepareTriangleMesh(accountingTriangle, accountingMeshPrepared) ==
            OVSPATIALQUERY_API_SUCCESS);
    std::shared_ptr<const TriangleObjectGeometry> accountingPrepared;
    REQUIRE(ovspatialquery::detail::bindPreparedTriangleGeometry(
                accountingMeshPrepared,
                {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0}, false,
                accountingPrepared) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(accountingPrepared != nullptr);
    // The retained charge covers the raw payload and the prepared triangle geometry.
    REQUIRE(accountingMesh.definitionRetainedByteCounts()[0] >=
            accountingSphere.definitionResidentByteCounts()[0] + rawMeshPayload +
                triangleGeometryResidentByteCount(*accountingPrepared));
    // Destroying the definition releases the memory owners the registry held for it.
    InstanceHostLedger accountingLedger(UINT64_MAX);
    HostTransaction    accountingTransaction(accountingLedger);
    REQUIRE(accountingTransaction.reserve(accountingMesh.definitionRetainedByteCounts()[0]));
    std::shared_ptr<HostLease> accountingOwner = std::make_shared<HostLease>();
    REQUIRE(accountingTransaction.commit(*accountingOwner));
    REQUIRE(accountingRegistry.commitRegistration(accountingMesh, {accountingOwner}, {}) == OVSPATIALQUERY_API_SUCCESS);
    accountingMesh = ExplicitPreparedRegistration{};
    REQUIRE(accountingRegistry.destroyDefinition(reference(ExactGeometryFamily::triangleMesh, 1001u)) ==
            OVSPATIALQUERY_API_SUCCESS);
    accountingOwner.reset();
    REQUIRE(accountingLedger.total() == 0u);

    ExplicitPreparationRegistry  registry(gBackendRuntime, UINT64_MAX);
    ExplicitPreparedRegistration registration =
        prepare(registry, {sphere(7u, 2.0), box(7u), capsule(8u), convex(9u), mesh(10u)});
    REQUIRE(registry.definitionCount() == 0u);
    REQUIRE(registry.commitRegistration(registration) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(registry.definitionCount() == 5u);
    REQUIRE(contains(registry, reference(ExactGeometryFamily::sphere, 7u)));
    REQUIRE(contains(registry, reference(ExactGeometryFamily::box, 7u)));

    ExplicitPreparedRegistration duplicate;
    REQUIRE(registry.prepareRegistration({sphere(11u, 1.0), sphere(7u, 3.0)}, duplicate) ==
            OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(!duplicate.initialized());
    REQUIRE(registry.definitionCount() == 5u);
    REQUIRE(!contains(registry, reference(ExactGeometryFamily::sphere, 11u)));

    ExplicitPreparedRegistration preserved = std::move(registration);
    // A finite radius or vertex beyond the input domain's 2^60 is malformed too.
    ExplicitDefinitionInput farConvex                                        = convex(12u);
    std::get<ExplicitConvexDefinition>(farConvex.definition).vertices[3]     = 0x1p61f;
    ExplicitDefinitionInput farMesh                                          = mesh(12u);
    std::get<ExplicitTriangleMeshDefinition>(farMesh.definition).vertices[3] = 0x1p61f;
    const std::vector<std::vector<ExplicitDefinitionInput>> invalid          = {
        {sphere(0u, 1.0)},
        {sphere(12u, 0.0)},
        {sphere(12u, std::numeric_limits<double>::infinity())},
        {sphere(12u, 0x1p61)},
        {farConvex},
        {farMesh},
        {sphere(12u, 1.0), sphere(12u, 2.0)},
    };
    for (const std::vector<ExplicitDefinitionInput>& values : invalid)
    {
        REQUIRE(registry.prepareRegistration(values, preserved) == OVSPATIALQUERY_API_INVALID_ARGUMENT);
        REQUIRE(preserved.definitionCount() == 5u);
        REQUIRE(registry.definitionCount() == 5u);
    }
    REQUIRE(registry.prepareRegistration(
                {{reference(ExactGeometryFamily::box, 14u), ExplicitBoxDefinition{{1.0, -1.0, 1.0}}}}, preserved) ==
            OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(registry.prepareRegistration(
                {{reference(ExactGeometryFamily::capsule, 15u), ExplicitCapsuleDefinition{0.5, -0.25}}}, preserved) ==
            OVSPATIALQUERY_API_INVALID_ARGUMENT);

    ExplicitConvexDefinition nonConvex = tetrahedron();
    nonConvex.vertices.insert(nonConvex.vertices.end(), {0.2f, 0.2f, 0.2f});
    nonConvex.faceVertexCounts = {
        3u, 3u, 3u, 3u, 3u, 3u,
    };
    nonConvex.faceVertexIndices = {
        0u, 2u, 1u, 0u, 1u, 4u, 1u, 2u, 4u, 2u, 0u, 4u, 0u, 3u, 2u, 0u, 1u, 3u,
    };
    ExplicitPreparedRegistration arbitraryPointSet;
    REQUIRE(registry.prepareRegistration({{reference(ExactGeometryFamily::convex, 20u), nonConvex}},
                                         arbitraryPointSet) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(arbitraryPointSet.initialized());
    REQUIRE(arbitraryPointSet.definitionCount() == 1u);

    ExplicitTriangleMeshDefinition malformedMesh = triangleMesh();
    malformedMesh.triangleVertexIndices[2]       = 9u;
    REQUIRE(registry.prepareRegistration({{reference(ExactGeometryFamily::triangleMesh, 21u), malformedMesh}},
                                         preserved) == OVSPATIALQUERY_API_INVALID_ARGUMENT);
}

void testFiveFamilyInstancesAndRigidPoseValidation()
{
    ExplicitPreparationRegistry  registry(gBackendRuntime, UINT64_MAX);
    ExplicitPreparedRegistration registration =
        prepare(registry, {sphere(1u, 2.0), box(2u), capsule(3u), convex(4u), mesh(5u, true)});
    REQUIRE(registry.commitRegistration(registration) == OVSPATIALQUERY_API_SUCCESS);

    std::vector<ExplicitInstanceInput> instances = {
        instance(50u, reference(ExactGeometryFamily::triangleMesh, 5u), 10.0),
        instance(10u, reference(ExactGeometryFamily::sphere, 1u), 1.0),
        instance(20u, reference(ExactGeometryFamily::box, 2u), 2.0),
        instance(30u, reference(ExactGeometryFamily::capsule, 3u), 3.0),
        instance(40u, reference(ExactGeometryFamily::convex, 4u), 4.0),
    };
    instances[0].participationBits = {{UINT64_C(0x1234), UINT64_C(0x5678)}};
    instances[1].uniformScale      = 3.0;
    instances[2].rotationXyzw      = {0.0f, 0.0f, 0.70710677f, 0.70710677f};
    instances[3].rotationXyzw      = {0.0f, 0.0f, 0.70710677f, 0.70710677f};

    ExplicitPreparedUpdate prepared;
    REQUIRE(registry.prepareSet(instances, prepared) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(prepared.objects().size() == 5u);
    REQUIRE(std::is_sorted(prepared.objects().begin(), prepared.objects().end(),
                           [](const PopulationObject& first, const PopulationObject& second)
                           { return first.id.value < second.id.value; }));
    for (const PopulationObject& object : prepared.objects())
    {
        REQUIRE(object.exactGeometry != nullptr);
        REQUIRE(object.geometryReference.value != 0u);
        REQUIRE(object.bounds.minimum[0] <= object.bounds.maximum[0]);
    }
    REQUIRE(prepared.objects()[0].id.value == 10u);
    REQUIRE(prepared.objects()[0].geometryReference == reference(ExactGeometryFamily::sphere, 1u));
    REQUIRE(prepared.objects()[0].exactGeometry->family() == ExactGeometryFamily::sphere);
    REQUIRE(prepared.objects()[0].bounds.minimum[0] == -5.0);
    REQUIRE(prepared.objects()[0].bounds.maximum[0] == 7.0);
    REQUIRE(prepared.objects()[4].triangleGeometry != nullptr);
    REQUIRE(prepared.objects()[4].triangleGeometry->doubleSided);
    REQUIRE(ovspatialquery::detail::participationEqual(prepared.objects()[4].participationBits,
                                                       {{UINT64_C(0x1234), UINT64_C(0x5678)}}));

    ExplicitPreparedUpdate secondPopulation;
    REQUIRE(registry.prepareSet({instance(99u, reference(ExactGeometryFamily::sphere, 1u), 20.0)}, secondPopulation) ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(secondPopulation.objects()[0].geometryReference == prepared.objects()[0].geometryReference);
    REQUIRE(secondPopulation.objects()[0].id.value != prepared.objects()[0].id.value);

    const float      positiveX[3] = {1.0f, 0.0f, 0.0f};
    ExactGeometryHit hit{};
    const double     sphereOrigin[3] = {-10.0, 0.0, 0.0};
    REQUIRE(objectWithId(prepared, 10u).exactGeometry->raycast(sphereOrigin, positiveX, 0.0, 30.0, false, hit));
    REQUIRE(near(hit.distance, 5.0));
    REQUIRE(!hit.hasTriangleSource);

    hit = {};
    REQUIRE(objectWithId(prepared, 20u).exactGeometry->raycast(sphereOrigin, positiveX, 0.0, 30.0, false, hit));
    REQUIRE(near(hit.distance, 10.0));
    REQUIRE(!hit.hasTriangleSource);

    hit = {};
    REQUIRE(objectWithId(prepared, 30u).exactGeometry->raycast(sphereOrigin, positiveX, 0.0, 30.0, false, hit));
    REQUIRE(near(hit.distance, 10.5));
    REQUIRE(!hit.hasTriangleSource);

    const double convexOrigin[3] = {3.0, 0.2, 0.2};
    hit                          = {};
    REQUIRE(objectWithId(prepared, 40u).exactGeometry->raycast(convexOrigin, positiveX, 0.0, 10.0, false, hit));
    REQUIRE(near(hit.distance, 1.0));
    REQUIRE(!hit.hasTriangleSource);

    const double meshOrigin[3] = {10.25, 0.25, 3.0};
    const float  negativeZ[3]  = {0.0f, 0.0f, -1.0f};
    hit                        = {};
    REQUIRE(objectWithId(prepared, 50u).exactGeometry->raycast(meshOrigin, negativeZ, 0.0, 10.0, false, hit));
    REQUIRE(near(hit.distance, 3.0));
    REQUIRE(hit.hasTriangleSource);
    REQUIRE(hit.sourceFace == 0u);
    REQUIRE(hit.sourceCorners[0] == 0u);
    REQUIRE(hit.sourceCorners[1] == 1u);
    REQUIRE(hit.sourceCorners[2] == 2u);

    ExplicitPreparedUpdate             preserved = std::move(prepared);
    std::vector<ExplicitInstanceInput> invalid   = instances;
    invalid[1].objectValue                       = 0u;
    REQUIRE(registry.prepareSet(invalid, preserved) == OVSPATIALQUERY_API_INVALID_ARGUMENT);
    invalid                = instances;
    invalid[1].objectValue = invalid[0].objectValue;
    REQUIRE(registry.prepareSet(invalid, preserved) == OVSPATIALQUERY_API_INVALID_ARGUMENT);
    invalid                   = instances;
    invalid[1].geometry.value = 99u;
    REQUIRE(registry.prepareSet(invalid, preserved) == OVSPATIALQUERY_API_INVALID_ARGUMENT);
    invalid                   = instances;
    invalid[1].translation[0] = std::numeric_limits<double>::quiet_NaN();
    REQUIRE(registry.prepareSet(invalid, preserved) == OVSPATIALQUERY_API_INVALID_ARGUMENT);
    invalid                 = instances;
    invalid[1].rotationXyzw = {0.0f, 0.0f, 0.0f, 0.0f};
    REQUIRE(registry.prepareSet(invalid, preserved) == OVSPATIALQUERY_API_INVALID_ARGUMENT);
    invalid                 = instances;
    invalid[1].rotationXyzw = {0.0f, 0.0f, 0.0f, 2.0f};
    REQUIRE(registry.prepareSet(invalid, preserved) == OVSPATIALQUERY_API_INVALID_ARGUMENT);
    invalid                    = instances;
    invalid[1].rotationXyzw[0] = std::numeric_limits<float>::infinity();
    REQUIRE(registry.prepareSet(invalid, preserved) == OVSPATIALQUERY_API_INVALID_ARGUMENT);
    invalid                 = instances;
    invalid[1].uniformScale = 0.0;
    REQUIRE(registry.prepareSet(invalid, preserved) == OVSPATIALQUERY_API_INVALID_ARGUMENT);
    invalid                 = instances;
    invalid[1].uniformScale = std::numeric_limits<double>::infinity();
    REQUIRE(registry.prepareSet(invalid, preserved) == OVSPATIALQUERY_API_INVALID_ARGUMENT);
    // The input domain: translations within 2^60, scales within [2^-60, 2^60],
    // and the world AABB within 2^60 even when the translation is not past it.
    // A mesh far along -x in its own frame brings a translation of 2^61 back to a
    // world AABB at the origin; the translation alone is outside the domain.
    ExplicitTriangleMeshDefinition offsetMesh = triangleMesh();
    offsetMesh.vertices                       = {-0x1p59f, 0.0f, 0.0f, -0x1p59f, 2.0f, 0.0f, -0x1p59f, 0.0f, 2.0f};
    ExplicitPreparedRegistration offsetRegistration =
        prepare(registry, {{reference(ExactGeometryFamily::triangleMesh, 6u), offsetMesh}});
    REQUIRE(registry.commitRegistration(offsetRegistration) == OVSPATIALQUERY_API_SUCCESS);
    invalid                   = instances;
    invalid[0].geometry       = reference(ExactGeometryFamily::triangleMesh, 6u);
    invalid[0].translation[0] = 0x1p61;
    invalid[0].uniformScale   = 4.0;
    REQUIRE(registry.prepareSet(invalid, preserved) == OVSPATIALQUERY_API_INVALID_ARGUMENT);
    invalid[0].translation[0] = 0x1p60;
    invalid[0].uniformScale   = 2.0;
    ExplicitPreparedUpdate offset;
    REQUIRE(registry.prepareSet(invalid, offset) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(objectWithId(offset, 50u).bounds.minimum[0] == 0.0);
    invalid                 = instances;
    invalid[1].uniformScale = 0x1p61;
    REQUIRE(registry.prepareSet(invalid, preserved) == OVSPATIALQUERY_API_INVALID_ARGUMENT);
    invalid[1].uniformScale = 0x1p-61;
    REQUIRE(registry.prepareSet(invalid, preserved) == OVSPATIALQUERY_API_INVALID_ARGUMENT);
    invalid                   = instances;
    invalid[1].translation[0] = 0x1p60;
    invalid[1].uniformScale   = 1024.0; // the radius-6 sphere reaches 2^60 + 6144
    REQUIRE(registry.prepareSet(invalid, preserved) == OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(preserved.objects().size() == 5u);
    invalid[1].translation[0] = 0x1p59;
    invalid[1].uniformScale   = 0x1p-60;
    ExplicitPreparedUpdate boundary;
    REQUIRE(registry.prepareSet(invalid, boundary) == OVSPATIALQUERY_API_SUCCESS);
}

void testCopiedDefinitionsAndConcurrentInstanceIdentity()
{
    ExplicitPreparationRegistry          registry(gBackendRuntime, UINT64_MAX);
    std::vector<ExplicitDefinitionInput> source{sphere(30u, 1.0)};
    ExplicitPreparedRegistration         copied                     = prepare(registry, source);
    std::get<ExplicitSphereDefinition>(source[0].definition).radius = 9.0;
    source[0].reference.value                                       = 99u;
    REQUIRE(registry.commitRegistration(copied) == OVSPATIALQUERY_API_SUCCESS);

    ExplicitPreparedUpdate instanceSet;
    REQUIRE(registry.prepareSet({instance(1u, reference(ExactGeometryFamily::sphere, 30u))}, instanceSet) ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(near(instanceSet.objects()[0].bounds.minimum[0], -1.0));
    REQUIRE(near(instanceSet.objects()[0].bounds.maximum[0], 1.0));

    ExplicitConvexDefinition     convexSource = tetrahedron();
    ExplicitPreparedRegistration copiedConvex =
        prepare(registry, {{reference(ExactGeometryFamily::convex, 31u), convexSource}});
    std::fill(convexSource.vertices.begin(), convexSource.vertices.end(), 0.0f);
    convexSource.faceVertexCounts.clear();
    convexSource.faceVertexIndices.clear();
    REQUIRE(registry.commitRegistration(copiedConvex) == OVSPATIALQUERY_API_SUCCESS);
    ExplicitPreparedUpdate convexSet;
    REQUIRE(registry.prepareSet({instance(2u, reference(ExactGeometryFamily::convex, 31u))}, convexSet) ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(convexSet.objects()[0].bounds.minimum[0] == 0.0);
    REQUIRE(convexSet.objects()[0].bounds.maximum[0] == 1.0);

    ExplicitPreparedRegistration first = prepare(registry, {sphere(40u, 2.0)});
    ExplicitPreparedRegistration second;
    REQUIRE(registry.prepareRegistration({sphere(40u, 3.0)}, second) == OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(registry.commitRegistration(first) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(registry.definitionCount() == 3u);

    ExplicitPreparationRegistry  independent(gBackendRuntime, UINT64_MAX);
    ExplicitPreparedRegistration sameCallerIdentity = prepare(independent, {sphere(30u, 4.0), box(30u)});
    REQUIRE(independent.commitRegistration(sameCallerIdentity) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(independent.definitionCount() == 2u);
    REQUIRE(contains(independent, reference(ExactGeometryFamily::sphere, 30u)));
    REQUIRE(contains(independent, reference(ExactGeometryFamily::box, 30u)));
}

// Explicit values stay below kFirstAdapterGeometryValue and the typed registry assigns
// from it up, so whichever source registers first, the other keeps its values.
void testCrossSourceValuesArePartitioned()
{
    using ovspatialquery::detail::GeometryDefinitionValue;
    using ovspatialquery::detail::GeometryIdentityOutcome;
    using ovspatialquery::detail::GeometryIdentityRegistry;
    using ovspatialquery::detail::GeometryReferenceClaimDirectory;
    using ovspatialquery::detail::kFirstAdapterGeometryValue;
    using ovspatialquery::detail::SphereDefinitionValue;

    GeometryReferenceClaimDirectory::Batch ownerGoneBatch;
    {
        std::unique_ptr<GeometryReferenceClaimDirectory> ephemeralClaims(new GeometryReferenceClaimDirectory());
        REQUIRE(ephemeralClaims->stage({reference(ExactGeometryFamily::sphere, 99u)}, ownerGoneBatch) ==
                GeometryIdentityOutcome::success);
        ephemeralClaims.reset();
    }
    ownerGoneBatch = GeometryReferenceClaimDirectory::Batch{};

    const std::shared_ptr<GeometryReferenceClaimDirectory> typedFirstClaims =
        std::make_shared<GeometryReferenceClaimDirectory>();
    GeometryIdentityRegistry        typedFirst(typedFirstClaims);
    ExplicitPreparationRegistry     explicitAfterTyped(gBackendRuntime, UINT64_MAX, typedFirstClaims);
    GeometryIdentityRegistry::Batch typedBatch;
    REQUIRE(typedFirst.stage({GeometryDefinitionValue{SphereDefinitionValue{1.0}}}, typedBatch) ==
            GeometryIdentityOutcome::success);
    REQUIRE(typedBatch.references()[0] == reference(ExactGeometryFamily::sphere, kFirstAdapterGeometryValue));
    typedFirst.commit(typedBatch);
    ExplicitPreparedRegistration rejected;
    REQUIRE(explicitAfterTyped.prepareRegistration({sphere(kFirstAdapterGeometryValue + 1u, 2.0)}, rejected) ==
            OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(explicitAfterTyped.definitionCount() == 0u);
    REQUIRE(typedFirstClaims->claimCount() == 1u);
    ExplicitPreparedRegistration highest = prepare(explicitAfterTyped, {sphere(kFirstAdapterGeometryValue - 1u, 2.0)});
    REQUIRE(explicitAfterTyped.commitRegistration(highest) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(typedFirstClaims->claimCount() == 2u);

    const std::shared_ptr<GeometryReferenceClaimDirectory> explicitFirstClaims =
        std::make_shared<GeometryReferenceClaimDirectory>();
    ExplicitPreparationRegistry  explicitFirst(gBackendRuntime, UINT64_MAX, explicitFirstClaims);
    ExplicitPreparedRegistration explicitSphere = prepare(explicitFirst, {sphere(1u, 1.0)});
    REQUIRE(explicitFirst.commitRegistration(explicitSphere) == OVSPATIALQUERY_API_SUCCESS);
    GeometryIdentityRegistry        typedAfterExplicit(explicitFirstClaims);
    GeometryIdentityRegistry::Batch afterExplicit;
    REQUIRE(typedAfterExplicit.stage({GeometryDefinitionValue{SphereDefinitionValue{3.0}}}, afterExplicit) ==
            GeometryIdentityOutcome::success);
    REQUIRE(afterExplicit.references()[0] == reference(ExactGeometryFamily::sphere, kFirstAdapterGeometryValue));
    typedAfterExplicit.commit(afterExplicit);
    REQUIRE(explicitFirstClaims->claimCount() == 2u);
    explicitSphere = ExplicitPreparedRegistration{};
    REQUIRE(explicitFirst.destroyDefinition(reference(ExactGeometryFamily::sphere, 1u)) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(explicitFirst.prepareRegistration({sphere(1u, 4.0)}, rejected) == OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(explicitFirstClaims->claimCount() == 2u);
}

void testUpdateOrdinalsAndCompleteRows()
{
    ExplicitPreparationRegistry  registry(gBackendRuntime, UINT64_MAX);
    ExplicitPreparedRegistration registration = prepare(registry, {sphere(1u, 1.0), sphere(2u, 2.0), box(1u)});
    REQUIRE(registry.commitRegistration(registration) == OVSPATIALQUERY_API_SUCCESS);

    ExplicitPreparedUpdate initial;
    REQUIRE(registry.prepareSet({instance(3u, reference(ExactGeometryFamily::sphere, 1u)),
                                 instance(1u, reference(ExactGeometryFamily::box, 1u)),
                                 instance(2u, reference(ExactGeometryFamily::sphere, 1u))},
                                initial) == OVSPATIALQUERY_API_SUCCESS);

    ovspatialquery_backend_runtime_t* runtime = nullptr;
    REQUIRE(ovspatialquery_test_oracle_create_runtime(&runtime) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(runtime != nullptr);
    {
        PopulationTable       table(runtime);
        const GenerationLease base = publish(table, 0, initial);
        REQUIRE(base.ordinal() == 0);

        ExplicitPreparedUpdate update;
        REQUIRE(registry.prepareUpdate(base, 0,
                                       {instance(2u, reference(ExactGeometryFamily::sphere, 2u), 8.0),
                                        instance(3u, reference(ExactGeometryFamily::sphere, 1u), 5.0),
                                        instance(4u, reference(ExactGeometryFamily::box, 1u), 9.0)},
                                       {1u}, update) == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(update.objects().size() == 3u);
        REQUIRE(update.objects()[0].id.value == 2u);
        REQUIRE(update.objects()[1].id.value == 3u);
        REQUIRE(update.objects()[2].id.value == 4u);
        REQUIRE(update.objects()[0].geometryReference == reference(ExactGeometryFamily::sphere, 2u));
        REQUIRE(update.objects()[1].geometryReference == reference(ExactGeometryFamily::sphere, 1u));
        REQUIRE(objectWithId(initial, 2u).geometryReference == reference(ExactGeometryFamily::sphere, 1u));
        REQUIRE(objectWithId(initial, 3u).geometryReference == update.objects()[1].geometryReference);

        const GenerationLease  updatedGeneration = publish(table, 7, update, base);
        ExplicitPreparedUpdate rejectedReuse;
        REQUIRE(registry.prepareUpdate(updatedGeneration, 7, {instance(1u, reference(ExactGeometryFamily::sphere, 1u))},
                                       {}, rejectedReuse) == OVSPATIALQUERY_API_INVALID_ARGUMENT);

        ExplicitPreparedUpdate preserved = std::move(update);
        REQUIRE(registry.prepareUpdate(base, 1, {}, {}, preserved) == OVSPATIALQUERY_API_ORDINAL_MISMATCH);
        REQUIRE(registry.prepareUpdate(GenerationLease{}, 0, {}, {}, preserved) == OVSPATIALQUERY_API_INVALID_ARGUMENT);
        REQUIRE(registry.prepareUpdate(base, 0, {instance(2u, reference(ExactGeometryFamily::sphere, 1u))}, {2u},
                                       preserved) == OVSPATIALQUERY_API_INVALID_ARGUMENT);
        REQUIRE(registry.prepareUpdate(base, 0, {instance(0u, reference(ExactGeometryFamily::sphere, 1u))}, {},
                                       preserved) == OVSPATIALQUERY_API_INVALID_ARGUMENT);
        REQUIRE(registry.prepareUpdate(base, 0,
                                       {instance(2u, reference(ExactGeometryFamily::sphere, 1u)),
                                        instance(2u, reference(ExactGeometryFamily::sphere, 2u))},
                                       {}, preserved) == OVSPATIALQUERY_API_INVALID_ARGUMENT);
        REQUIRE(registry.prepareUpdate(base, 0, {}, {9u}, preserved) == OVSPATIALQUERY_API_INVALID_ARGUMENT);
        REQUIRE(registry.prepareUpdate(base, 0, {}, {1u, 1u}, preserved) == OVSPATIALQUERY_API_INVALID_ARGUMENT);
        REQUIRE(preserved.objects().size() == 3u);
    }
    ovspatialquery_test_oracle_destroy_runtime(runtime);
}

// A head records its exact-geometry charge. A sparse update adjusts that charge by the
// objects it replaces, removes and adds instead of recounting every object; the
// adjustment must equal the recount across the geometry families, and a dense update or
// a head without a recorded charge leaves the recount to admission.
void testUpdateAdjustsRecordedExactGeometryCharge()
{
    using ovspatialquery::detail::exactGeometryResidentByteCount;
    ExplicitPreparationRegistry  registry(gBackendRuntime, UINT64_MAX);
    ExplicitPreparedRegistration registration =
        prepare(registry, {sphere(1u, 2.0), box(2u), capsule(3u), convex(4u), mesh(5u)});
    REQUIRE(registry.commitRegistration(registration) == OVSPATIALQUERY_API_SUCCESS);
    const std::array<GeometryReference, 5u> families = {
        reference(ExactGeometryFamily::sphere, 1u), reference(ExactGeometryFamily::box, 2u),
        reference(ExactGeometryFamily::capsule, 3u), reference(ExactGeometryFamily::convex, 4u),
        reference(ExactGeometryFamily::triangleMesh, 5u)};
    std::vector<ExplicitInstanceInput> rows;
    for (uint64_t value = 1u; value <= 20u; ++value)
    {
        rows.push_back(instance(value, families[value % families.size()], 4.0 * static_cast<double>(value)));
    }
    ExplicitPreparedUpdate initial;
    REQUIRE(registry.prepareSet(rows, initial) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(!initial.exactGeometryCharge().has_value());

    ovspatialquery_backend_runtime_t* runtime = nullptr;
    REQUIRE(ovspatialquery_test_oracle_create_runtime(&runtime) == OVSPATIALQUERY_API_SUCCESS);
    {
        PopulationTable       table(runtime);
        const GenerationLease base = publish(table, 0, initial);

        // Five replacements, two of them changing family, two insertions and two removals.
        ExplicitPreparedUpdate sparse;
        REQUIRE(registry.prepareUpdate(base, 0,
                                       {instance(3u, families[4], 90.0), instance(5u, families[0], 91.0),
                                        instance(6u, families[1], 92.0), instance(7u, families[2], 93.0),
                                        instance(8u, families[3], 94.0), instance(21u, families[4], 95.0),
                                        instance(22u, families[2], 96.0)},
                                       {1u, 2u}, sparse) == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(sparse.exactGeometryCharge().has_value());
        REQUIRE(*sparse.exactGeometryCharge() == exactGeometryResidentByteCount(sparse.objects()));

        // The adjusted charge carries into the next head and adjusts again.
        const GenerationLease  next = publish(table, 1, sparse, base);
        ExplicitPreparedUpdate following;
        REQUIRE(registry.prepareUpdate(next, 1, {instance(21u, families[1], 97.0), instance(23u, families[3], 98.0)},
                                       {22u}, following) == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(following.exactGeometryCharge().has_value());
        REQUIRE(*following.exactGeometryCharge() == exactGeometryResidentByteCount(following.objects()));

        // Replacing or removing half of the base leaves the recount to admission.
        std::vector<ExplicitInstanceInput> half;
        for (uint64_t value = 11u; value <= 20u; ++value)
        {
            half.push_back(instance(value, families[value % families.size()], 100.0 + static_cast<double>(value)));
        }
        ExplicitPreparedUpdate dense;
        REQUIRE(registry.prepareUpdate(base, 0, half, {}, dense) == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(!dense.exactGeometryCharge().has_value());

        // A head built without a recorded charge also leaves the recount to admission.
        GenerationLease unrecorded;
        REQUIRE(table.buildGeneration(0, initial.objects(), {}, unrecorded) == OVSPATIALQUERY_API_SUCCESS);
        ExplicitPreparedUpdate unmeasured;
        REQUIRE(registry.prepareUpdate(unrecorded, 0, {instance(4u, families[0], 99.0)}, {}, unmeasured) ==
                OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(!unmeasured.exactGeometryCharge().has_value());
    }
    ovspatialquery_test_oracle_destroy_runtime(runtime);
}

// What the publication of a complete ON_DEVICE set leaves at the head; only
// its description matters to a host-prepared update.
class DevicePreparedVersion final : public ovspatialquery::detail::CudaPopulationVersion
{
  public:
    ovspatialquery::detail::CudaPopulationDescription description() const override
    {
        ovspatialquery::detail::CudaPopulationDescription result;
        result.devicePrepared = true;
        return result;
    }
    ovspatialquery::detail::CudaCompletionState resultState() const override
    {
        return ovspatialquery::detail::CudaCompletionState::complete;
    }
    void     releaseReadyDependencies() override {}
    uint64_t deviceByteCharge() const override { return 0u; }
};

// The update API refuses a host-prepared update once the head is device-prepared,
// but a device set can publish between that check and the pin; preparation then
// refuses the base itself (ADR-0024).
void testUpdateRejectsDevicePreparedBase()
{
    ExplicitPreparationRegistry  registry(gBackendRuntime, UINT64_MAX);
    ExplicitPreparedRegistration registration = prepare(registry, {sphere(1u, 1.0)});
    REQUIRE(registry.commitRegistration(registration) == OVSPATIALQUERY_API_SUCCESS);
    ExplicitPreparedUpdate initial;
    REQUIRE(registry.prepareSet({instance(1u, reference(ExactGeometryFamily::sphere, 1u))}, initial) ==
            OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_backend_runtime_t* runtime = nullptr;
    REQUIRE(ovspatialquery_test_oracle_create_runtime(&runtime) == OVSPATIALQUERY_API_SUCCESS);
    {
        PopulationTable table(runtime);
        GenerationLease base;
        REQUIRE(table.buildGeneration(0, initial.objects(), {}, base, {}, std::make_shared<DevicePreparedVersion>()) ==
                OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(base.preparedOnDevice());
        ExplicitPreparedUpdate update;
        REQUIRE(registry.prepareUpdate(base, 0, {instance(2u, reference(ExactGeometryFamily::sphere, 1u))}, {},
                                       update) == OVSPATIALQUERY_API_NOT_SUPPORTED);
        // A poses-only update completes its rows from the base first; a
        // complete ON_DEVICE set publishes no host rows to complete them from.
        GenerationLease deviceOnly;
        REQUIRE(table.buildGeneration(0, ObjectTable(), {}, deviceOnly, {},
                                      std::make_shared<DevicePreparedVersion>()) == OVSPATIALQUERY_API_SUCCESS);
        std::vector<ExplicitInstanceInput> poses{instance(1u, reference(ExactGeometryFamily::sphere, 1u))};
        REQUIRE(completePoseRows(deviceOnly, false, false, poses) == OVSPATIALQUERY_API_NOT_SUPPORTED);
    }
    ovspatialquery_test_oracle_destroy_runtime(runtime);
}

void testConvexPreparationFailureAndStoredHullReuse()
{
    ExplicitPreparationRegistry  registry(gBackendRuntime, UINT64_MAX);
    ExplicitPreparedRegistration preserved = prepare(registry, {sphere(3000u, 1.0)});

    const std::array<std::pair<ConvexHullPreparationOutcome, ovspatialquery_api_status_t>, 3u> failures{{
        {ConvexHullPreparationOutcome::resourceExhausted, OVSPATIALQUERY_API_HOST_BUDGET},
        {ConvexHullPreparationOutcome::outOfMemory, OVSPATIALQUERY_API_OUT_OF_MEMORY},
        {ConvexHullPreparationOutcome::internalFailure, OVSPATIALQUERY_API_INTERNAL},
    }};
    uint64_t                                                                                   value = 3001u;
    for (const auto& failure : failures)
    {
        ovspatialquery::detail::failNextConvexHullPreparationForTest(gBackendRuntime, failure.first);
        REQUIRE(registry.prepareRegistration({convex(value++)}, preserved) == failure.second);
        REQUIRE(preserved.initialized());
        REQUIRE(preserved.definitionCount() == 1u);
        REQUIRE(registry.definitionCount() == 0u);
    }

    ExplicitConvexDefinition openSource = tetrahedron();
    openSource.faceVertexCounts         = {3u, 3u};
    openSource.faceVertexIndices        = {0u, 1u, 2u, 0u, 1u, 2u};
    ExplicitPreparedRegistration prepared;
    REQUIRE(registry.prepareRegistration({{reference(ExactGeometryFamily::convex, value), openSource}}, prepared) ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(registry.commitRegistration(prepared) == OVSPATIALQUERY_API_SUCCESS);

    ovspatialquery::detail::failNextConvexHullPreparationForTest(gBackendRuntime,
                                                                 ConvexHullPreparationOutcome::outOfMemory);
    ExplicitPreparedUpdate first;
    REQUIRE(registry.prepareSet({instance(1u, reference(ExactGeometryFamily::convex, value))}, first) ==
            OVSPATIALQUERY_API_SUCCESS);
    ExplicitPreparedUpdate second;
    REQUIRE(registry.prepareSet({instance(1u, reference(ExactGeometryFamily::convex, value))}, second) ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(first.objects().front().exactGeometry != nullptr);
    REQUIRE(second.objects().front().exactGeometry != nullptr);

    ExplicitPreparedRegistration faultConsumedByRegistration;
    REQUIRE(registry.prepareRegistration({convex(value + 1u)}, faultConsumedByRegistration) ==
            OVSPATIALQUERY_API_OUT_OF_MEMORY);
    REQUIRE(!faultConsumedByRegistration.initialized());
    REQUIRE(registry.prepareRegistration({convex(value + 1u)}, faultConsumedByRegistration) ==
            OVSPATIALQUERY_API_SUCCESS);

    ExplicitPreparedRegistration                                      rejected;
    const ExplicitConvexDefinition                                    budgetSource = tetrahedron();
    std::shared_ptr<const ovspatialquery::detail::PreparedConvexHull> measuredHull;
    REQUIRE(ovspatialquery::detail::prepareConvexHull(gBackendRuntime, budgetSource.vertices, UINT64_MAX,
                                                      measuredHull) == ConvexHullPreparationOutcome::success);
    REQUIRE(measuredHull != nullptr);
    ExplicitPreparationRegistry oneByteShort(gBackendRuntime, measuredHull->residentByteCount() - 1u);
    REQUIRE(oneByteShort.prepareRegistration({{reference(ExactGeometryFamily::convex, 4000u), budgetSource}},
                                             rejected) == OVSPATIALQUERY_API_HOST_BUDGET);
    REQUIRE(!rejected.initialized());
    ExplicitPreparationRegistry exactBudget(gBackendRuntime, measuredHull->residentByteCount());
    REQUIRE(exactBudget.prepareRegistration({{reference(ExactGeometryFamily::convex, 4000u), budgetSource}},
                                            rejected) == OVSPATIALQUERY_API_SUCCESS);
}

void testEmptySetAndSourceBindingLifetime()
{
    PopulationSourceBinding binding;
    {
        ExplicitPreparationRegistry registry(gBackendRuntime, UINT64_MAX);
        binding = registry.sourceBinding();
        REQUIRE(binding.kind == OVSPATIALQUERY_SOURCE_EXPLICIT);
        REQUIRE(binding.instance == nullptr);
        REQUIRE(binding.pathDictionary == nullptr);
        REQUIRE(binding.acquireSnapshot == nullptr);
        REQUIRE(binding.sourceState != nullptr);

        ExplicitPreparedUpdate empty;
        REQUIRE(registry.prepareSet({}, empty) == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(empty.objects().empty());
    }
    REQUIRE(binding.sourceState != nullptr);
}

} // namespace

int main()
{
    testPreparedInputAllocationFailures();
    REQUIRE(ovspatialquery_test_oracle_create_runtime(&gBackendRuntime) == OVSPATIALQUERY_API_SUCCESS);
    testRegistrationValidationAndAtomicCommit();
    testFiveFamilyInstancesAndRigidPoseValidation();
    testConvexPreparationFailureAndStoredHullReuse();
    testCopiedDefinitionsAndConcurrentInstanceIdentity();
    testCrossSourceValuesArePartitioned();
    testUpdateOrdinalsAndCompleteRows();
    testUpdateAdjustsRecordedExactGeometryCharge();
    testUpdateRejectsDevicePreparedBase();
    testEmptySetAndSourceBindingLifetime();
    ovspatialquery_test_oracle_destroy_runtime(gBackendRuntime);
    gBackendRuntime = nullptr;
    return 0;
}
