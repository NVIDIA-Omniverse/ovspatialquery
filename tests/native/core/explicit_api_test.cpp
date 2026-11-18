/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../support/native/public_api_inputs.h"
#include "../../support/native/test_failure.h"
#include <ovspatialquery/ovspatialquery_explicit.h>

#include "ovspatialquery_instance_operation_runtime.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <future>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#if defined(OVSPATIALQUERY_TEST_HAS_CUDA_BACKEND)
#include "cuda_driver.h"
#endif

namespace
{

void waitOnly(ovspatialquery_instance_handle_t instance, ovspatialquery_enqueue_result_t admission)
{
    REQUIRE(admission.status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(admission.op_id != OVSPATIALQUERY_INVALID_OP);
    ovspatialquery_op_wait_result_t wait{};
    const ovspatialquery_result_t   waitStatus =
        ovspatialquery_wait_op(instance, admission.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &wait);
    if (waitStatus.status != OVSPATIALQUERY_API_SUCCESS)
    {
        const ovspatialquery_string_t diagnostic = ovspatialquery_get_last_op_error(instance, admission.op_id);
        std::fprintf(stderr, "explicit operation failed: %.*s\n", static_cast<int>(diagnostic.length), diagnostic.ptr);
    }
    REQUIRE(waitStatus.status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(wait.state == OVSPATIALQUERY_OP_COMPLETED);
}

#if defined(OVSPATIALQUERY_TEST_HAS_CUDA_BACKEND)
uint64_t deviceUsage(ovspatialquery_instance_handle_t instance, DLDevice device)
{
    uint64_t bytes = UINT64_MAX;
    REQUIRE(ovspatialquery_get_instance_memory_usage(instance, device, &bytes).status == OVSPATIALQUERY_API_SUCCESS);
    return bytes;
}
#endif

// A rejected registration, at operation capacity or at scheduling, leaves no
// operation ID, no host charge and no claimed definition value behind.
void testRegistrationCapacityAndSchedulingRollback()
{
    const ovspatialquery_instance_desc_t instanceDesc{1u, 1u, UINT64_C(16) * 1024u * 1024u, nullptr, 0u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&instanceDesc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    uint64_t                                       value  = 9100u;
    double                                         radius = 1.0;
    InputArray                                     values(&value, 1, kDLUInt);
    InputArray                                     radii(&radius, 1, kDLFloat);
    const ovspatialquery_sphere_definition_table_t spheres{&values.value, &radii.value};
    REQUIRE(ovspatialquery::detail::setInstanceOperationWorkersPausedForTest(instance, true) ==
            OVSPATIALQUERY_API_SUCCESS);
    const ovspatialquery_enqueue_result_t first        = ovspatialquery_enqueue_sphere_registration(instance, &spheres);
    const uint64_t                        pendingBytes = hostUsage(instance);
    value                                              = 9101u;
    const ovspatialquery_enqueue_result_t full         = ovspatialquery_enqueue_sphere_registration(instance, &spheres);
    const uint64_t                        fullBytes    = hostUsage(instance);
    // Assert only after resuming: a failure exits while the first is pending.
    REQUIRE(ovspatialquery::detail::setInstanceOperationWorkersPausedForTest(instance, false) ==
            OVSPATIALQUERY_API_SUCCESS);
    requireRejected(full, OVSPATIALQUERY_API_OPERATION_CAPACITY);
    REQUIRE(fullBytes == pendingBytes);
    waitAndDestroy(instance, first);

    const uint64_t idleBytes = hostUsage(instance);
    REQUIRE(ovspatialquery::detail::failNextInstanceOperationSchedulingForTest(instance) == OVSPATIALQUERY_API_SUCCESS);
    requireRejected(ovspatialquery_enqueue_sphere_registration(instance, &spheres), OVSPATIALQUERY_API_ERROR);
    REQUIRE(hostUsage(instance) == idleBytes);
    const ovspatialquery_enqueue_result_t retry = ovspatialquery_enqueue_sphere_registration(instance, &spheres);
    REQUIRE(retry.op_id != first.op_id);
    waitAndDestroy(instance, retry);
    for (const uint64_t registered : {UINT64_C(9100), UINT64_C(9101)})
    {
        REQUIRE(
            ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_SPHERE, registered}).status ==
            OVSPATIALQUERY_API_SUCCESS);
    }
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

void testConvexRegistrationAcceptsPointsOnly()
{
    const ovspatialquery_instance_desc_t instanceDesc{2u, 1u, UINT64_C(16) * 1024u * 1024u, nullptr, 0u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&instanceDesc, &instance).status == OVSPATIALQUERY_API_SUCCESS);

    uint64_t definitionValue = 7001u;
    uint64_t vertexOffset    = 0u;
    uint64_t vertexCount     = 5u;
    float    vertices[15]    = {
        0.0, 0.0, 1.0, 0.2f, 0.2f, 0.2f, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    };
    InputArray                                     values(&definitionValue, 1, kDLUInt);
    InputArray                                     offsets(&vertexOffset, 1, kDLUInt);
    InputArray                                     counts(&vertexCount, 1, kDLUInt);
    InputArray                                     points(vertices, 5, 3, kDLFloat);
    const ovspatialquery_convex_definition_table_t definitions{
        &values.value, &offsets.value, &counts.value, nullptr, nullptr, &points.value,
        nullptr,       nullptr,        nullptr,       0.0,     0u,
    };
    double unsupportedVertices[15] = {};
    std::copy(std::begin(vertices), std::end(vertices), std::begin(unsupportedVertices));
    InputArray                               unsupportedPoints(unsupportedVertices, 5, 3, kDLFloat);
    ovspatialquery_convex_definition_table_t unsupportedDefinitions = definitions;
    unsupportedDefinitions.vertices                                 = &unsupportedPoints.value;
    requireRejected(ovspatialquery_enqueue_convex_registration(instance, &unsupportedDefinitions),
                    OVSPATIALQUERY_API_NOT_SUPPORTED);

    waitAndDestroy(instance, ovspatialquery_enqueue_convex_registration(instance, &definitions));
    REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_CONVEX, definitionValue})
                .status == OVSPATIALQUERY_API_SUCCESS);

    definitionValue                                               = 7002u;
    uint64_t                                       faceOffset     = 0u;
    uint64_t                                       faceCount      = 2u;
    uint64_t                                       loopOffsets[2] = {0u, 3u};
    uint32_t                                       loopCounts[2]  = {3u, 3u};
    uint32_t                                       loopIndices[6] = {0u, 1u, 2u, 0u, 1u, 2u};
    InputArray                                     faceOffsets(&faceOffset, 1, kDLUInt);
    InputArray                                     faceCounts(&faceCount, 1, kDLUInt);
    InputArray                                     sourceLoopOffsets(loopOffsets, 2, kDLUInt);
    InputArray                                     sourceLoopCounts(loopCounts, 2, kDLUInt);
    InputArray                                     sourceLoopIndices(loopIndices, 6, kDLUInt);
    const ovspatialquery_convex_definition_table_t openNonmanifold{
        &values.value,
        &offsets.value,
        &counts.value,
        &faceOffsets.value,
        &faceCounts.value,
        &points.value,
        &sourceLoopOffsets.value,
        &sourceLoopCounts.value,
        &sourceLoopIndices.value,
        0.0,
        0u,
    };
    waitAndDestroy(instance, ovspatialquery_enqueue_convex_registration(instance, &openNonmanifold));
    REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_CONVEX, definitionValue})
                .status == OVSPATIALQUERY_API_SUCCESS);
    const uint64_t usageBeforeRejections = hostUsage(instance);

    ovspatialquery_convex_definition_table_t mixed = openNonmanifold;
    mixed.face_counts                              = nullptr;
    requireRejected(ovspatialquery_enqueue_convex_registration(instance, &mixed), OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(hostUsage(instance) == usageBeforeRejections);

    definitionValue    = 7003u;
    const float finite = vertices[0];
    vertices[0]        = std::numeric_limits<float>::infinity();
    requireRejected(ovspatialquery_enqueue_convex_registration(instance, &definitions),
                    OVSPATIALQUERY_API_INVALID_ARGUMENT);
    vertices[0] = finite;

    definitionValue            = 7004u;
    float coplanarVertices[12] = {
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0,
    };
    uint64_t                                       coplanarCount = 4u;
    InputArray                                     coplanarCounts(&coplanarCount, 1, kDLUInt);
    InputArray                                     coplanarPoints(coplanarVertices, 4, 3, kDLFloat);
    const ovspatialquery_convex_definition_table_t coplanar{&values.value,
                                                            &offsets.value,
                                                            &coplanarCounts.value,
                                                            nullptr,
                                                            nullptr,
                                                            &coplanarPoints.value,
                                                            nullptr,
                                                            nullptr,
                                                            nullptr,
                                                            0.0,
                                                            0u};
    requireRejected(ovspatialquery_enqueue_convex_registration(instance, &coplanar),
                    OVSPATIALQUERY_API_INVALID_ARGUMENT);

    definitionValue = 7005u;
    loopIndices[5]  = 5u;
    requireRejected(ovspatialquery_enqueue_convex_registration(instance, &openNonmanifold),
                    OVSPATIALQUERY_API_INVALID_ARGUMENT);
    loopIndices[5] = 2u;
    REQUIRE(hostUsage(instance) == usageBeforeRejections);

    uint64_t batchValuesData[2]  = {7006u, 7007u};
    uint64_t batchOffsetsData[2] = {0u, 5u};
    uint64_t batchCountsData[2]  = {5u, 4u};
    float    batchPointsData[27] = {};
    std::copy(vertices, vertices + 15, batchPointsData);
    std::copy(coplanarVertices, coplanarVertices + 12, batchPointsData + 15);
    InputArray                                     batchValues(batchValuesData, 2, kDLUInt);
    InputArray                                     batchOffsets(batchOffsetsData, 2, kDLUInt);
    InputArray                                     batchCounts(batchCountsData, 2, kDLUInt);
    InputArray                                     batchPoints(batchPointsData, 9, 3, kDLFloat);
    const ovspatialquery_convex_definition_table_t invalidBatch{&batchValues.value,
                                                                &batchOffsets.value,
                                                                &batchCounts.value,
                                                                nullptr,
                                                                nullptr,
                                                                &batchPoints.value,
                                                                nullptr,
                                                                nullptr,
                                                                nullptr,
                                                                0.0,
                                                                0u};
    requireRejected(ovspatialquery_enqueue_convex_registration(instance, &invalidBatch),
                    OVSPATIALQUERY_API_INVALID_ARGUMENT);
    for (uint64_t value : batchValuesData)
    {
        REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_CONVEX, value}).status ==
                OVSPATIALQUERY_API_INVALID_HANDLE);
    }
    REQUIRE(hostUsage(instance) == usageBeforeRejections);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

void testConvexRegistrationExactHostBudgetBoundary()
{
    uint64_t definitionValue = 7100u;
    uint64_t vertexOffset    = 0u;
    uint64_t vertexCount     = 5u;
    float    vertices[15]    = {
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.2f, 0.2f, 0.2f,
    };
    InputArray                                     values(&definitionValue, 1, kDLUInt);
    InputArray                                     offsets(&vertexOffset, 1, kDLUInt);
    InputArray                                     counts(&vertexCount, 1, kDLUInt);
    InputArray                                     points(vertices, 5, 3, kDLFloat);
    const ovspatialquery_convex_definition_table_t definitions{
        &values.value, &offsets.value, &counts.value, nullptr, nullptr, &points.value,
        nullptr,       nullptr,        nullptr,       0.0,     0u};
    const ovspatialquery_instance_desc_t generousDesc{2u, 1u, UINT64_C(16) * 1024u * 1024u, nullptr, 0u};
    ovspatialquery_instance_handle_t     calibration = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&generousDesc, &calibration).status == OVSPATIALQUERY_API_SUCCESS);
    const ovspatialquery_enqueue_result_t calibrationAdmission =
        ovspatialquery_enqueue_convex_registration(calibration, &definitions);
    waitOnly(calibration, calibrationAdmission);
    const uint64_t         exactPeak     = hostUsage(calibration);
    ovspatialquery_op_id_t calibrationId = calibrationAdmission.op_id;
    REQUIRE(ovspatialquery_destroy_op(calibration, &calibrationId).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_geometry_definition(calibration, {OVSPATIALQUERY_GEOMETRY_CONVEX, definitionValue})
                .status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(calibration).status == OVSPATIALQUERY_API_SUCCESS);

    ovspatialquery_instance_desc_t tightDesc = generousDesc;
    tightDesc.host_byte_budget               = exactPeak - 1u;
    ovspatialquery_instance_handle_t tight   = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&tightDesc, &tight).status == OVSPATIALQUERY_API_SUCCESS);
    const uint64_t tightUsage = hostUsage(tight);
    requireRejected(ovspatialquery_enqueue_convex_registration(tight, &definitions), OVSPATIALQUERY_API_HOST_BUDGET);
    REQUIRE(hostUsage(tight) == tightUsage);
    REQUIRE(
        ovspatialquery_destroy_geometry_definition(tight, {OVSPATIALQUERY_GEOMETRY_CONVEX, definitionValue}).status ==
        OVSPATIALQUERY_API_INVALID_HANDLE);
    REQUIRE(ovspatialquery_destroy_instance(tight).status == OVSPATIALQUERY_API_SUCCESS);

    ovspatialquery_instance_desc_t exactDesc = generousDesc;
    exactDesc.host_byte_budget               = exactPeak;
    ovspatialquery_instance_handle_t exact   = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&exactDesc, &exact).status == OVSPATIALQUERY_API_SUCCESS);
    waitAndDestroy(exact, ovspatialquery_enqueue_convex_registration(exact, &definitions));
    REQUIRE(
        ovspatialquery_destroy_geometry_definition(exact, {OVSPATIALQUERY_GEOMETRY_CONVEX, definitionValue}).status ==
        OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(exact).status == OVSPATIALQUERY_API_SUCCESS);
}

void testCpuStatusAndPublicationAtomicity()
{
    const ovspatialquery_instance_desc_t instanceDesc{3u, 1u, UINT64_C(16) * 1024u * 1024u, nullptr, 0u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&instanceDesc, &instance).status == OVSPATIALQUERY_API_SUCCESS);

    uint64_t definitionValue = 90u;
    registerSphere(instance, definitionValue, 1.0);

    const uint64_t usageAfterRegistration = hostUsage(instance);
    uint64_t       malformedMeshValue     = 91u;
    uint64_t       zero                   = 0u;
    uint64_t       three                  = 3u;
    uint64_t       one                    = 1u;
    float          malformedVertices[9]   = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0};
    uint32_t       malformedTriangle[3]   = {0u, 1u, 3u};
    InputArray     malformedMeshValues(&malformedMeshValue, 1, kDLUInt);
    InputArray     vertexOffsets(&zero, 1, kDLUInt);
    InputArray     vertexCounts(&three, 1, kDLUInt);
    InputArray     triangleOffsets(&zero, 1, kDLUInt);
    InputArray     triangleCounts(&one, 1, kDLUInt);
    InputArray     vertices(malformedVertices, 3, 3, kDLFloat);
    InputArray     triangles(malformedTriangle, 1, 3, kDLUInt);
    const ovspatialquery_triangle_mesh_definition_table_t malformedMesh{&malformedMeshValues.value,
                                                                        &vertexOffsets.value,
                                                                        &vertexCounts.value,
                                                                        &triangleOffsets.value,
                                                                        &triangleCounts.value,
                                                                        &vertices.value,
                                                                        &triangles.value,
                                                                        nullptr,
                                                                        nullptr,
                                                                        nullptr,
                                                                        nullptr};
    requireRejected(ovspatialquery_enqueue_triangle_mesh_registration(instance, &malformedMesh),
                    OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(hostUsage(instance) == usageAfterRegistration);
    REQUIRE(ovspatialquery_destroy_geometry_definition(instance,
                                                       {OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, malformedMeshValue})
                .status == OVSPATIALQUERY_API_INVALID_HANDLE);

    const ovspatialquery_population_handle_t population = createExplicitPopulation(instance, 2u);
    ovspatialquery_population_info_t         info{};
    REQUIRE(ovspatialquery_get_population_info(instance, population, &info).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(info.ready == 0u && info.object_count == 0u);

    uint64_t                                 objectValue    = 900u;
    int32_t                                  wrongFamily    = OVSPATIALQUERY_GEOMETRY_BOX;
    double                                   translation[3] = {2.0, 0.0, 0.0};
    float                                    rotation[4]    = {0.0f, 0.0f, 0.0f, 1.0f};
    InputArray                               objects(&objectValue, 1, kDLUInt);
    InputArray                               types(&wrongFamily, 1, kDLInt);
    InputArray                               geometry(&definitionValue, 1, kDLUInt);
    InputArray                               positions(translation, 1, 3, kDLFloat);
    InputArray                               rotations(rotation, 1, 4, kDLFloat);
    ovspatialquery_explicit_population_set_t set{
        3,
        {&objects.value, &types.value, &geometry.value, &positions.value, &rotations.value, nullptr, nullptr, nullptr},
        nullptr,
        0u};
    requireRejected(ovspatialquery_enqueue_explicit_population_set(instance, population, &set),
                    OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(ovspatialquery_get_population_info(instance, population, &info).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(info.ready == 0u && info.object_count == 0u);
    // USER is reserved: a row or a definition reference naming it is NOT_SUPPORTED.
    int32_t userFamily      = OVSPATIALQUERY_GEOMETRY_USER;
    types.value.tensor.data = &userFamily;
    requireRejected(ovspatialquery_enqueue_explicit_population_set(instance, population, &set),
                    OVSPATIALQUERY_API_NOT_SUPPORTED);
    REQUIRE(
        ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_USER, definitionValue}).status ==
        OVSPATIALQUERY_API_NOT_SUPPORTED);

    int32_t sphereFamily    = OVSPATIALQUERY_GEOMETRY_SPHERE;
    types.value.tensor.data = &sphereFamily;
    waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set));
    REQUIRE(ovspatialquery_get_population_info(instance, population, &info).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(info.ready != 0u && info.head_ordinal == 3 && info.object_count == 1u && info.retained_ordinal_count == 0u);
    requireRejected(ovspatialquery_enqueue_explicit_population_set(instance, population, &set),
                    OVSPATIALQUERY_API_INVALID_STATE);

    uint64_t                                    emptyObject      = 0u;
    int32_t                                     emptyType        = 0;
    uint64_t                                    emptyGeometry    = 0u;
    double                                      emptyPosition[3] = {};
    float                                       emptyRotation[4] = {};
    InputArray                                  changedObjects(&emptyObject, 0, kDLUInt);
    InputArray                                  changedTypes(&emptyType, 0, kDLInt);
    InputArray                                  changedGeometry(&emptyGeometry, 0, kDLUInt);
    InputArray                                  changedPositions(emptyPosition, 0, 3, kDLFloat);
    InputArray                                  changedRotations(emptyRotation, 0, 4, kDLFloat);
    uint64_t                                    removedObject = 999u;
    InputArray                                  removed(&removedObject, 1, kDLUInt);
    ovspatialquery_explicit_population_update_t update{2,
                                                       4,
                                                       OVSPATIALQUERY_EXPLICIT_UPDATE_COMPLETE_ROWS,
                                                       {&changedObjects.value, &changedTypes.value,
                                                        &changedGeometry.value, &changedPositions.value,
                                                        &changedRotations.value, nullptr, nullptr, nullptr},
                                                       &removed.value,
                                                       nullptr};
    requireRejected(ovspatialquery_enqueue_explicit_population_update(instance, population, &update),
                    OVSPATIALQUERY_API_ORDINAL_MISMATCH);
    update.base_ordinal   = 3;
    update.target_ordinal = 3;
    requireRejected(ovspatialquery_enqueue_explicit_population_update(instance, population, &update),
                    OVSPATIALQUERY_API_STALE_ORDINAL);
    update.target_ordinal = 4;
    requireRejected(ovspatialquery_enqueue_explicit_population_update(instance, population, &update),
                    OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(ovspatialquery_get_population_info(instance, population, &info).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(info.head_ordinal == 3 && info.object_count == 1u && info.retained_ordinal_count == 0u);
    // A column whose rows would run past the end of the address space is malformed.
    removed.value.tensor.data = reinterpret_cast<void*>(UINTPTR_MAX - 1u);
    requireRejected(ovspatialquery_enqueue_explicit_population_update(instance, population, &update),
                    OVSPATIALQUERY_API_INVALID_ARGUMENT);

    removed.set(&removedObject, 0, kDLUInt);
    // DLPack producers report any strides for a dimension with no elements
    // (NumPy reports 0); empty columns are accepted whatever they are.
    for (InputArray* empty :
         {&changedObjects, &changedTypes, &changedGeometry, &changedPositions, &changedRotations, &removed})
    {
        empty->strides = {0, 0};
    }
    waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_update(instance, population, &update));
    REQUIRE(ovspatialquery_get_population_info(instance, population, &info).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(info.head_ordinal == 4 && info.object_count == 1u && info.retained_ordinal_count == 1u);

    ovspatialquery_raycast_desc_t retainedRay{};
    retainedRay.common.target                        = populationTarget(population);
    retainedRay.common.target.ordinal.kind           = OVSPATIALQUERY_ORDINAL_RETAINED_EXACT;
    retainedRay.common.target.ordinal.source_ordinal = 3;
    retainedRay.common.precision                     = OVSPATIALQUERY_QUERY_EXACT;
    retainedRay.common.query_bits                    = participationBits(UINT64_MAX, UINT64_MAX);
    retainedRay.ray.unit_direction[0]                = 1.0f;
    retainedRay.ray.max_distance                     = 10.0;
    ovspatialquery_raycast_single_result_t retainedResult{};
    REQUIRE(ovspatialquery_raycast_closest_sync(instance, &retainedRay, &retainedResult).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(retainedResult.source_ordinal == 3 && retainedResult.has_hit == 1u &&
            retainedResult.hit.data.exact.identity.object_id.value == objectValue);

    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_SPHERE, definitionValue})
                .status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

void testParticipationChangesRemainVisibleThroughQueries()
{
    const ovspatialquery_instance_desc_t instanceDesc{4u, 1u, UINT64_C(16) * 1024u * 1024u, nullptr, 0u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&instanceDesc, &instance).status == OVSPATIALQUERY_API_SUCCESS);

    uint64_t definitionValue = 92u;
    registerSphere(instance, definitionValue, 1.0);

    const ovspatialquery_population_handle_t population = createExplicitPopulation(instance, 3u);

    uint64_t objectValues[2]   = {10u, 20u};
    int32_t  geometryTypes[2]  = {OVSPATIALQUERY_GEOMETRY_SPHERE, OVSPATIALQUERY_GEOMETRY_SPHERE};
    uint64_t geometryValues[2] = {definitionValue, definitionValue};
    double   translations[6]   = {0.0, 0.0, 0.0, 5.0, 0.0, 0.0};
    float    rotations[8]      = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    // One uint64[N,2] row per object, bits 0-63 first: 20 also holds bit 71.
    uint64_t   objectBits[4] = {1u, 0u, 2u, UINT64_C(1) << 7u};
    InputArray objects(objectValues, 2, kDLUInt);
    InputArray types(geometryTypes, 2, kDLInt);
    InputArray geometry(geometryValues, 2, kDLUInt);
    InputArray positions(translations, 2, 3, kDLFloat);
    InputArray orientations(rotations, 2, 4, kDLFloat);
    InputArray participation(objectBits, 2, 2, kDLUInt);
    // The one-word column of the 64-bit releases is refused rather than read as half the bits.
    InputArray                                     oneWord(objectBits, 2, kDLUInt);
    const ovspatialquery_explicit_population_set_t oneWordSet{1,
                                                              {&objects.value, &types.value, &geometry.value,
                                                               &positions.value, &orientations.value, nullptr,
                                                               &oneWord.value, nullptr},
                                                              nullptr,
                                                              0u};
    requireRejected(ovspatialquery_enqueue_explicit_population_set(instance, population, &oneWordSet),
                    OVSPATIALQUERY_API_NOT_SUPPORTED);
    const ovspatialquery_explicit_population_set_t set{1,
                                                       {&objects.value, &types.value, &geometry.value, &positions.value,
                                                        &orientations.value, nullptr, &participation.value, nullptr},
                                                       nullptr,
                                                       0u};
    ovspatialquery_enqueue_result_t                admission =
        ovspatialquery_enqueue_explicit_population_set(instance, population, &set);
    waitOnly(instance, admission);
    ovspatialquery_population_update_result_t publication{};
    REQUIRE(ovspatialquery_fetch_population_update(instance, admission.op_id, &publication).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(publication.published_ordinal == 1 && publication.object_count == 2u &&
            publication.excluded_object_count == 0u);
    REQUIRE(ovspatialquery_destroy_op(instance, &admission.op_id).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(admission.op_id == OVSPATIALQUERY_INVALID_OP);

    const auto query = [&](double origin, double maximumDistance, uint64_t queryBits, uint64_t highQueryBits = 0u)
    {
        ovspatialquery_raycast_desc_t ray{};
        ray.common.target         = populationTarget(population);
        ray.common.precision      = OVSPATIALQUERY_QUERY_EXACT;
        ray.common.query_bits     = participationBits(queryBits, highQueryBits);
        ray.ray.origin[0]         = origin;
        ray.ray.unit_direction[0] = 1.0f;
        ray.ray.max_distance      = maximumDistance;
        ovspatialquery_raycast_single_result_t result{};
        REQUIRE(ovspatialquery_raycast_closest_sync(instance, &ray, &result).status == OVSPATIALQUERY_API_SUCCESS);
        return result;
    };

    ovspatialquery_raycast_single_result_t queryResult = query(-2.0, 4.0, 1u);
    REQUIRE(queryResult.source_ordinal == 1 && queryResult.has_hit == 1u);
    REQUIRE(queryResult.hit.data.exact.identity.object_id.value == 10u);
    queryResult = query(3.0, 4.0, 2u);
    REQUIRE(queryResult.source_ordinal == 1 && queryResult.has_hit == 1u);
    REQUIRE(queryResult.hit.data.exact.identity.object_id.value == 20u);
    queryResult = query(-2.0, 10.0, 0u, UINT64_C(1) << 7u); // bit 71 passes over 10 to reach 20
    REQUIRE(queryResult.source_ordinal == 1 && queryResult.has_hit == 1u);
    REQUIRE(queryResult.hit.data.exact.identity.object_id.value == 20u);

    // A CPU sphere-sweep plan filters each row by its own query_bits row, as an avoidance step
    // does in a grounded scene: a row starting against 10 (a static ground) reports it at
    // distance zero (ADR-0018) unless its bits leave 10 out, and then it reaches 20.
    ovspatialquery_sphere_sweep_plan_desc_t sweepPlanDesc{};
    sweepPlanDesc.target                     = populationRef(population);
    sweepPlanDesc.precision                  = OVSPATIALQUERY_QUERY_EXACT;
    sweepPlanDesc.radius                     = 0.5;
    sweepPlanDesc.device                     = DLDevice{kDLCPU, 0};
    sweepPlanDesc.maximum_query_count        = 4u;
    sweepPlanDesc.result_slot_count          = 1u;
    sweepPlanDesc.result_fields              = OVSPATIALQUERY_BATCH_RESULT_DISTANCE;
    sweepPlanDesc.hit_capacity_per_slot      = 4u;
    ovspatialquery_batch_plan_handle_t sweep = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_sphere_sweep_plan(instance, &sweepPlanDesc, &sweep).status ==
            OVSPATIALQUERY_API_SUCCESS);
    double     sweepCenters[12]    = {-3.0, 0.0, 0.0, -3.0, 0.0, 0.0, -1.5, 0.0, 0.0, -1.5, 0.0, 0.0};
    float      sweepDirections[12] = {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    double     sweepMaxima[4]      = {10.0, 10.0, 10.0, 10.0};
    uint64_t   sweepBits[8]        = {2u, 0u, 1u, 0u, 3u, 0u, 2u, 0u};
    InputArray sweepCenterInput(sweepCenters, 4, 3, kDLFloat);
    InputArray sweepDirectionInput(sweepDirections, 4, 3, kDLFloat);
    InputArray sweepMaximumInput(sweepMaxima, 4, kDLFloat);
    InputArray sweepBitInput(sweepBits, 4, 2, kDLUInt);
    ovspatialquery_sphere_sweep_batch_t sweepBatch{};
    sweepBatch.ordinal         = {OVSPATIALQUERY_ORDINAL_RETAINED_EXACT, 1};
    sweepBatch.query_count     = 4u;
    sweepBatch.query_bits      = &sweepBitInput.value;
    sweepBatch.centers         = &sweepCenterInput.value;
    sweepBatch.unit_directions = &sweepDirectionInput.value;
    sweepBatch.max_distances   = &sweepMaximumInput.value;
    ovspatialquery_sphere_sweep_batch_reservation_t sweepReserved{};
    ovspatialquery_enqueue_result_t                 sweepLaunch =
        ovspatialquery_enqueue_sphere_sweep_plan(instance, sweep, &sweepBatch, &sweepReserved);
    waitOnly(instance, sweepLaunch);
    ovspatialquery_sphere_sweep_batch_result_t swept{};
    REQUIRE(ovspatialquery_fetch_sphere_sweep_batch(instance, sweepLaunch.op_id, &swept).status ==
            OVSPATIALQUERY_API_SUCCESS);
    const uint64_t expectedObjects[4]   = {20u, 10u, 10u, 20u};
    const double   expectedDistances[4] = {6.5, 1.5, 0.0, 5.0};
    for (uint64_t row = 0u; row < 4u; ++row)
    {
        REQUIRE(static_cast<const uint64_t*>(swept.columns.hit_ranges.hit_counts.data)[row] == 1u);
        const uint64_t hit = static_cast<const uint64_t*>(swept.columns.hit_ranges.hit_offsets.data)[row];
        REQUIRE(static_cast<const uint64_t*>(swept.columns.object_ids.data)[hit] == expectedObjects[row]);
        REQUIRE(std::abs(static_cast<const double*>(swept.columns.contact.exact.distances.data)[hit] -
                         expectedDistances[row]) <= 1e-12);
    }
    REQUIRE(ovspatialquery_destroy_op(instance, &sweepLaunch.op_id).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_batch_plan(instance, sweep).status == OVSPATIALQUERY_API_SUCCESS);

    uint64_t                                          changedObject           = 10u;
    int32_t                                           changedType             = OVSPATIALQUERY_GEOMETRY_SPHERE;
    uint64_t                                          changedGeometry         = definitionValue;
    double                                            changedTranslation[3]   = {0.0, 0.0, 0.0};
    float                                             changedRotation[4]      = {0.0f, 0.0f, 0.0f, 1.0f};
    uint64_t                                          changedParticipation[2] = {4u, 0u};
    uint64_t                                          removedObject           = 20u;
    InputArray                                        changedObjects(&changedObject, 1, kDLUInt);
    InputArray                                        changedTypes(&changedType, 1, kDLInt);
    InputArray                                        changedGeometryIds(&changedGeometry, 1, kDLUInt);
    InputArray                                        changedPositions(changedTranslation, 1, 3, kDLFloat);
    InputArray                                        changedOrientations(changedRotation, 1, 4, kDLFloat);
    InputArray                                        changedBits(changedParticipation, 1, 2, kDLUInt);
    InputArray                                        removedObjects(&removedObject, 1, kDLUInt);
    const ovspatialquery_explicit_population_update_t changeAndRemove{
        1,
        3,
        OVSPATIALQUERY_EXPLICIT_UPDATE_COMPLETE_ROWS,
        {&changedObjects.value, &changedTypes.value, &changedGeometryIds.value, &changedPositions.value,
         &changedOrientations.value, nullptr, &changedBits.value, nullptr},
        &removedObjects.value,
        nullptr};
    admission = ovspatialquery_enqueue_explicit_population_update(instance, population, &changeAndRemove);
    waitOnly(instance, admission);
    publication = {};
    REQUIRE(ovspatialquery_fetch_population_update(instance, admission.op_id, &publication).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(publication.published_ordinal == 3 && publication.object_count == 1u &&
            publication.excluded_object_count == 0u);
    REQUIRE(ovspatialquery_destroy_op(instance, &admission.op_id).status == OVSPATIALQUERY_API_SUCCESS);

    queryResult = query(-2.0, 4.0, 1u);
    REQUIRE(queryResult.source_ordinal == 3 && queryResult.has_hit == 0u);
    queryResult = query(-2.0, 4.0, 4u);
    REQUIRE(queryResult.source_ordinal == 3 && queryResult.has_hit == 1u);
    REQUIRE(queryResult.hit.data.exact.identity.object_id.value == 10u);
    queryResult = query(3.0, 4.0, 2u);
    REQUIRE(queryResult.source_ordinal == 3 && queryResult.has_hit == 0u);

    uint64_t                                          addedObject         = 30u;
    double                                            addedTranslation[3] = {9.0, 0.0, 0.0};
    InputArray                                        addedObjects(&addedObject, 1, kDLUInt);
    InputArray                                        addedPositions(addedTranslation, 1, 3, kDLFloat);
    uint64_t                                          noRemovedValue = 0u;
    InputArray                                        noRemoved(&noRemovedValue, 0, kDLUInt);
    const ovspatialquery_explicit_population_update_t add{
        3,
        4,
        OVSPATIALQUERY_EXPLICIT_UPDATE_COMPLETE_ROWS,
        {&addedObjects.value, &changedTypes.value, &changedGeometryIds.value, &addedPositions.value,
         &changedOrientations.value, nullptr, &changedBits.value, nullptr},
        &noRemoved.value,
        nullptr};
    admission = ovspatialquery_enqueue_explicit_population_update(instance, population, &add);
    waitOnly(instance, admission);
    publication = {};
    REQUIRE(ovspatialquery_fetch_population_update(instance, admission.op_id, &publication).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(publication.published_ordinal == 4 && publication.object_count == 2u &&
            publication.excluded_object_count == 0u);
    REQUIRE(ovspatialquery_destroy_op(instance, &admission.op_id).status == OVSPATIALQUERY_API_SUCCESS);

    queryResult = query(7.0, 4.0, 4u);
    REQUIRE(queryResult.source_ordinal == 4 && queryResult.has_hit == 1u);
    REQUIRE(queryResult.hit.data.exact.identity.object_id.value == 30u);
    queryResult = query(-2.0, 4.0, 4u);
    REQUIRE(queryResult.source_ordinal == 4 && queryResult.has_hit == 1u);
    REQUIRE(queryResult.hit.data.exact.identity.object_id.value == 10u);
    queryResult = query(-2.0, 4.0, 1u);
    REQUIRE(queryResult.source_ordinal == 4 && queryResult.has_hit == 0u);
    queryResult = query(3.0, 4.0, 2u);
    REQUIRE(queryResult.source_ordinal == 4 && queryResult.has_hit == 0u);

    ovspatialquery_population_info_t beforeRejection{};
    REQUIRE(ovspatialquery_get_population_info(instance, population, &beforeRejection).status ==
            OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_explicit_population_update_t staleBase = add;
    staleBase.target_ordinal                              = 5;
    requireRejected(ovspatialquery_enqueue_explicit_population_update(instance, population, &staleBase),
                    OVSPATIALQUERY_API_ORDINAL_MISMATCH);
    ovspatialquery_population_info_t afterRejection{};
    REQUIRE(ovspatialquery_get_population_info(instance, population, &afterRejection).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(afterRejection.ready == beforeRejection.ready &&
            afterRejection.head_ordinal == beforeRejection.head_ordinal &&
            afterRejection.object_count == beforeRejection.object_count &&
            afterRejection.retained_ordinal_count == beforeRejection.retained_ordinal_count &&
            afterRejection.update_pending == beforeRejection.update_pending);
    queryResult = query(7.0, 4.0, 4u);
    REQUIRE(queryResult.source_ordinal == 4 && queryResult.has_hit == 1u);
    REQUIRE(queryResult.hit.data.exact.identity.object_id.value == 30u);

    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_SPHERE, definitionValue})
                .status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

void testDefinitionAndObjectValuesAreNeverReused()
{
    const ovspatialquery_instance_desc_t instanceDesc{4u, 1u, UINT64_C(16) * 1024u * 1024u, nullptr, 0u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&instanceDesc, &instance).status == OVSPATIALQUERY_API_SUCCESS);

    uint64_t                                       definitionValue = 501u;
    double                                         radius          = 1.0;
    InputArray                                     definitionValues(&definitionValue, 1, kDLUInt);
    InputArray                                     radii(&radius, 1, kDLFloat);
    const ovspatialquery_sphere_definition_table_t spheres{&definitionValues.value, &radii.value};
    waitAndDestroy(instance, ovspatialquery_enqueue_sphere_registration(instance, &spheres));
    REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_SPHERE, definitionValue})
                .status == OVSPATIALQUERY_API_SUCCESS);
    const uint64_t afterDestroy = hostUsage(instance);
    requireRejected(ovspatialquery_enqueue_sphere_registration(instance, &spheres),
                    OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(hostUsage(instance) == afterDestroy);

    definitionValue = 502u;
    waitAndDestroy(instance, ovspatialquery_enqueue_sphere_registration(instance, &spheres));
    const ovspatialquery_population_handle_t population = createExplicitPopulation(instance, 1u);

    uint64_t                                       objectValue    = 77u;
    int32_t                                        family         = OVSPATIALQUERY_GEOMETRY_SPHERE;
    double                                         translation[3] = {};
    float                                          rotation[4]    = {0.0f, 0.0f, 0.0f, 1.0f};
    InputArray                                     objects(&objectValue, 1, kDLUInt);
    InputArray                                     types(&family, 1, kDLInt);
    InputArray                                     geometry(&definitionValue, 1, kDLUInt);
    InputArray                                     positions(translation, 1, 3, kDLFloat);
    InputArray                                     rotations(rotation, 1, 4, kDLFloat);
    const ovspatialquery_explicit_population_set_t set{
        1,
        {&objects.value, &types.value, &geometry.value, &positions.value, &rotations.value, nullptr, nullptr, nullptr},
        nullptr,
        0u};
    waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set));

    uint64_t   emptyValue       = 0u;
    int32_t    emptyType        = 0;
    double     emptyPosition[3] = {};
    float      emptyRotation[4] = {};
    InputArray emptyObjects(&emptyValue, 0, kDLUInt);
    InputArray emptyTypes(&emptyType, 0, kDLInt);
    InputArray emptyGeometry(&emptyValue, 0, kDLUInt);
    InputArray emptyPositions(emptyPosition, 0, 3, kDLFloat);
    InputArray emptyRotations(emptyRotation, 0, 4, kDLFloat);
    translation[0] = 2.0;
    InputArray                                        noRemoved(&emptyValue, 0, kDLUInt);
    const ovspatialquery_explicit_population_update_t sameId{
        1,
        2,
        OVSPATIALQUERY_EXPLICIT_UPDATE_COMPLETE_ROWS,
        {&objects.value, &types.value, &geometry.value, &positions.value, &rotations.value, nullptr, nullptr, nullptr},
        &noRemoved.value,
        nullptr};
    waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_update(instance, population, &sameId));

    // A value below every used one enters the used-value history in order; the checks below read that history.
    uint64_t                                          lowerObjectValue = 50u;
    InputArray                                        lowerObjects(&lowerObjectValue, 1, kDLUInt);
    const ovspatialquery_explicit_population_update_t lower{2,
                                                            3,
                                                            OVSPATIALQUERY_EXPLICIT_UPDATE_COMPLETE_ROWS,
                                                            {&lowerObjects.value, &types.value, &geometry.value,
                                                             &positions.value, &rotations.value, nullptr, nullptr,
                                                             nullptr},
                                                            &noRemoved.value,
                                                            nullptr};
    waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_update(instance, population, &lower));
    ovspatialquery_population_info_t info{};
    REQUIRE(ovspatialquery_get_population_info(instance, population, &info).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(info.head_ordinal == 3 && info.object_count == 2u && info.update_pending == 0u);

    uint64_t retiredObjectValue = objectValue;
    objectValue                 = 78u;
    InputArray                                  removed(&retiredObjectValue, 1, kDLUInt);
    ovspatialquery_explicit_population_update_t replace{
        3,
        4,
        OVSPATIALQUERY_EXPLICIT_UPDATE_COMPLETE_ROWS,
        {&objects.value, &types.value, &geometry.value, &positions.value, &rotations.value, nullptr, nullptr, nullptr},
        &removed.value,
        nullptr};
    waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_update(instance, population, &replace));

    objectValue = retiredObjectValue;
    const ovspatialquery_explicit_population_update_t readd{
        4,
        5,
        OVSPATIALQUERY_EXPLICIT_UPDATE_COMPLETE_ROWS,
        {&objects.value, &types.value, &geometry.value, &positions.value, &rotations.value, nullptr, nullptr, nullptr},
        &noRemoved.value,
        nullptr};
    const uint64_t beforeRejectedReadd = hostUsage(instance);
    requireRejected(ovspatialquery_enqueue_explicit_population_update(instance, population, &readd),
                    OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(hostUsage(instance) == beforeRejectedReadd);
    objectValue        = 78u;
    retiredObjectValue = objectValue;
    const ovspatialquery_explicit_population_update_t removeReplacement{
        4,
        5,
        OVSPATIALQUERY_EXPLICIT_UPDATE_COMPLETE_ROWS,
        {&emptyObjects.value, &emptyTypes.value, &emptyGeometry.value, &emptyPositions.value, &emptyRotations.value,
         nullptr, nullptr, nullptr},
        &removed.value,
        nullptr};
    waitAndDestroy(instance,
                   ovspatialquery_enqueue_explicit_population_update(instance, population, &removeReplacement));
    const ovspatialquery_explicit_population_update_t readdReplacement{
        5,
        6,
        OVSPATIALQUERY_EXPLICIT_UPDATE_COMPLETE_ROWS,
        {&objects.value, &types.value, &geometry.value, &positions.value, &rotations.value, nullptr, nullptr, nullptr},
        &noRemoved.value,
        nullptr};
    requireRejected(ovspatialquery_enqueue_explicit_population_update(instance, population, &readdReplacement),
                    OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(ovspatialquery_get_population_info(instance, population, &info).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(info.head_ordinal == 5 && info.object_count == 1u && info.update_pending == 0u);

    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_SPHERE, definitionValue})
                .status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

void testTriangleMeshAccountingReleasesPreparedStorage()
{
    const ovspatialquery_instance_desc_t instanceDesc{2u, 1u, UINT64_C(16) * 1024u * 1024u, nullptr, 0u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&instanceDesc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    const uint64_t beforeRegistration = hostUsage(instance);
    uint64_t       meshValue          = 590u;
    uint64_t       zero               = 0u;
    uint64_t       three              = 3u;
    uint64_t       one                = 1u;
    float          verticesData[9]    = {0.0f, 0.0f, 0.25f, 2.0f, 0.0f, 0.25f, 0.0f, 2.0f, 0.25f};
    const double   localHeight        = verticesData[2];
    uint32_t       triangleData[3]    = {0u, 1u, 2u};
    uint8_t        doubleSided        = 1u;
    InputArray     values(&meshValue, 1, kDLUInt);
    InputArray     vertexOffsets(&zero, 1, kDLUInt);
    InputArray     vertexCounts(&three, 1, kDLUInt);
    InputArray     triangleOffsets(&zero, 1, kDLUInt);
    InputArray     triangleCounts(&one, 1, kDLUInt);
    InputArray     vertices(verticesData, 3, 3, kDLFloat);
    InputArray     triangles(triangleData, 1, 3, kDLUInt);
    InputArray     sidedness(&doubleSided, 1, kDLUInt);
    const ovspatialquery_triangle_mesh_definition_table_t mesh{&values.value,
                                                               &vertexOffsets.value,
                                                               &vertexCounts.value,
                                                               &triangleOffsets.value,
                                                               &triangleCounts.value,
                                                               &vertices.value,
                                                               &triangles.value,
                                                               &sidedness.value,
                                                               nullptr,
                                                               nullptr,
                                                               nullptr};
    double                                                unsupportedVerticesData[9] = {};
    std::copy(std::begin(verticesData), std::end(verticesData), std::begin(unsupportedVerticesData));
    InputArray                                      unsupportedVertices(unsupportedVerticesData, 3, 3, kDLFloat);
    ovspatialquery_triangle_mesh_definition_table_t unsupportedMesh = mesh;
    unsupportedMesh.vertices                                        = &unsupportedVertices.value;
    requireRejected(ovspatialquery_enqueue_triangle_mesh_registration(instance, &unsupportedMesh),
                    OVSPATIALQUERY_API_NOT_SUPPORTED);
    waitAndDestroy(instance, ovspatialquery_enqueue_triangle_mesh_registration(instance, &mesh));
    const uint64_t afterRegistration = hostUsage(instance);
    REQUIRE(afterRegistration > beforeRegistration);
    // Registration owns the geometry after completion, independent of the input buffer lifetime.
    std::fill(std::begin(verticesData), std::end(verticesData), 0.0f);
    const ovspatialquery_population_handle_t population     = createExplicitPopulation(instance, 1u);
    uint64_t                                 objectValue    = 591u;
    int32_t                                  family         = OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH;
    double                                   translation[3] = {1000000000.125, 0.0, 16777217.0};
    double                                   rotation[4]    = {0.0, 0.0, 0.0, 1.0};
    InputArray objects(&objectValue, 1, kDLUInt), types(&family, 1, kDLInt), geometry(&meshValue, 1, kDLUInt);
    InputArray positions(translation, 1, 3, kDLFloat), rotations(rotation, 1, 4, kDLFloat);
    const ovspatialquery_explicit_population_set_t set{
        1,
        {&objects.value, &types.value, &geometry.value, &positions.value, &rotations.value, nullptr, nullptr, nullptr},
        nullptr,
        0u};
    waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set));
    ovspatialquery_raycast_desc_t ray{};
    ray.common.target         = populationTarget(population);
    ray.common.precision      = OVSPATIALQUERY_QUERY_EXACT;
    ray.common.query_bits     = participationBits(UINT64_MAX, UINT64_MAX);
    ray.ray.origin[0]         = translation[0] + 0.5;
    ray.ray.origin[1]         = 0.5;
    ray.ray.origin[2]         = translation[2] + localHeight + 4.0;
    ray.ray.unit_direction[2] = -1.0f;
    ray.ray.max_distance      = 10.0;
    ovspatialquery_raycast_single_result_t result{};
    REQUIRE(ovspatialquery_raycast_closest_sync(instance, &ray, &result).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(result.has_hit && result.hit.data.exact.distance == ray.ray.origin[2] - (translation[2] + localHeight));
    REQUIRE(result.hit.data.exact.position[2] == translation[2] + localHeight);
    const double originalDistance = result.hit.data.exact.distance;
    ray.ray.origin[0]             = translation[0] - 0.0625;
    REQUIRE(ovspatialquery_raycast_closest_sync(instance, &ray, &result).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(!result.has_hit); // A narrowed world translation would put this ray inside.
    ray.ray.origin[0]    = translation[0] + 0.5;
    ray.ray.max_distance = originalDistance - 0.25;
    REQUIRE(ovspatialquery_raycast_closest_sync(instance, &ray, &result).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(!result.has_hit);
    ray.ray.max_distance = 10.0;
    ray.ray.min_distance = originalDistance + 0.25;
    REQUIRE(ovspatialquery_raycast_closest_sync(instance, &ray, &result).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(!result.has_hit);
    ray.ray.min_distance = 0.0;
    translation[2] += 0.5;
    InputArray                                        removed(&zero, 0, kDLUInt);
    const ovspatialquery_explicit_population_update_t update{
        1, 2, OVSPATIALQUERY_EXPLICIT_UPDATE_COMPLETE_ROWS, set.instances, &removed.value, nullptr};
    waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_update(instance, population, &update));
    REQUIRE(ovspatialquery_raycast_closest_sync(instance, &ray, &result).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(result.source_ordinal == 2 && result.has_hit && result.hit.data.exact.distance == originalDistance - 0.5);
    ray.common.target.ordinal.kind           = OVSPATIALQUERY_ORDINAL_RETAINED_EXACT;
    ray.common.target.ordinal.source_ordinal = 1;
    REQUIRE(ovspatialquery_raycast_closest_sync(instance, &ray, &result).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(result.source_ordinal == 1 && result.has_hit && result.hit.data.exact.distance == originalDistance);
    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, meshValue})
                .status == OVSPATIALQUERY_API_SUCCESS);
    const uint64_t afterDestroy = hostUsage(instance);
    REQUIRE(afterDestroy < afterRegistration);
    REQUIRE(afterDestroy > beforeRegistration);
    requireRejected(ovspatialquery_enqueue_triangle_mesh_registration(instance, &mesh),
                    OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(hostUsage(instance) == afterDestroy);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

#if defined(OVSPATIALQUERY_TEST_HAS_CUDA_BACKEND)
template <typename Rotation>
void testCudaCopiedAndBorrowedPopulationCapture()
{
    CudaDriver                           cuda;
    const ovspatialquery_device_budget_t deviceBudget{{kDLCUDA, 0}, UINT64_C(64) * 1024u * 1024u};
    const ovspatialquery_instance_desc_t instanceDesc{4u, 1u, UINT64_C(16) * 1024u * 1024u, &deviceBudget, 1u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&instanceDesc, &instance).status == OVSPATIALQUERY_API_SUCCESS);

    uint64_t                                    definitionValue = 70u;
    double                                      halfExtents[3]  = {1.0, 1.0, 1.0};
    InputArray                                  definitionValues(&definitionValue, 1, kDLUInt);
    InputArray                                  extents(halfExtents, 1, 3, kDLFloat);
    const ovspatialquery_box_definition_table_t boxes{&definitionValues.value, &extents.value};
    waitAndDestroy(instance, ovspatialquery_enqueue_box_registration(instance, &boxes));

    const ovspatialquery_population_handle_t population = createExplicitPopulation(instance, 1u);

    uint64_t                                  objectValue    = 700u;
    int32_t                                   geometryType   = OVSPATIALQUERY_GEOMETRY_BOX;
    double                                    translation[3] = {3.0, 0.0, 0.0};
    Rotation                                  rotation[4]    = {0, 0, Rotation(std::sin(0.3)), Rotation(std::cos(0.3))};
    const double                              qz = rotation[2], qw = rotation[3];
    const double                              xExtent = 1.0 / (1.0 - 2.0 * qz * qz / (qz * qz + qw * qw));
    InputArray                                objects(&objectValue, 1, kDLUInt);
    InputArray                                types(&geometryType, 1, kDLInt);
    InputArray                                geometry(&definitionValue, 1, kDLUInt);
    InputArray                                positions(translation, 1, 3, kDLFloat);
    InputArray                                rotations(rotation, 1, 4, kDLFloat);
    std::array<CudaDriver::DevicePointer, 5u> allocations{
        cuda.allocate(sizeof(objectValue)), cuda.allocate(sizeof(geometryType)), cuda.allocate(sizeof(definitionValue)),
        cuda.allocate(sizeof(translation)), cuda.allocate(sizeof(rotation))};
    const CudaDriver::Stream stream = cuda.createStream();
    const CudaDriver::Event  ready  = cuda.createEvent();
    cuda.uploadAsync(allocations[0], &objectValue, sizeof(objectValue), stream);
    cuda.uploadAsync(allocations[1], &geometryType, sizeof(geometryType), stream);
    cuda.uploadAsync(allocations[2], &definitionValue, sizeof(definitionValue), stream);
    cuda.uploadAsync(allocations[3], translation, sizeof(translation), stream);
    cuda.uploadAsync(allocations[4], rotation, sizeof(rotation), stream);
    cuda.record(ready, stream);
    std::array<InputArray*, 5u> inputs{&objects, &types, &geometry, &positions, &rotations};
    for (size_t index = 0u; index < inputs.size(); ++index)
    {
        inputs[index]->value.tensor.data   = reinterpret_cast<void*>(static_cast<uintptr_t>(allocations[index]));
        inputs[index]->value.tensor.device = {kDLCUDA, 0};
        inputs[index]->value.ownership     = OVSPATIALQUERY_INPUT_COPIED_ON_SUBMISSION;
    }
    const ovspatialquery_dynamic_execution_t execution{
        {kDLCUDA, 0},
        {1u, reinterpret_cast<uintptr_t>(stream), reinterpret_cast<uintptr_t>(ready)},
        OVSPATIALQUERY_PREPARE_ON_HOST};
    const ovspatialquery_explicit_population_set_t set{
        1,
        {&objects.value, &types.value, &geometry.value, &positions.value, &rotations.value, nullptr, nullptr, nullptr},
        &execution,
        0u};

    // USER is reserved: CUDA rows are NOT_SUPPORTED with host or device preparation.
    const int32_t                   userType       = OVSPATIALQUERY_GEOMETRY_USER;
    const CudaDriver::DevicePointer userAllocation = cuda.allocate(sizeof(userType));
    cuda.uploadAsync(userAllocation, &userType, sizeof(userType), stream);
    cuda.record(ready, stream);
    InputArray userTypes(&geometryType, 1, kDLInt);
    userTypes.value.tensor.data   = reinterpret_cast<void*>(static_cast<uintptr_t>(userAllocation));
    userTypes.value.tensor.device = {kDLCUDA, 0};
    userTypes.value.ownership     = OVSPATIALQUERY_INPUT_COPIED_ON_SUBMISSION;
    const ovspatialquery_dynamic_execution_t onDevice{
        {kDLCUDA, 0},
        {1u, reinterpret_cast<uintptr_t>(stream), reinterpret_cast<uintptr_t>(ready)},
        OVSPATIALQUERY_PREPARE_ON_DEVICE};
    auto userSet                                            = set;
    userSet.instances.geometry_types                        = &userTypes.value;
    userSet.dynamic_execution                               = &onDevice;
    const ovspatialquery_population_handle_t userPopulation = createExplicitPopulation(instance, 1u);
    requireRejected(ovspatialquery_enqueue_explicit_population_set(instance, userPopulation, &userSet),
                    OVSPATIALQUERY_API_NOT_SUPPORTED);
    userSet.dynamic_execution = &execution;
    requireRejected(ovspatialquery_enqueue_explicit_population_set(instance, userPopulation, &userSet),
                    OVSPATIALQUERY_API_NOT_SUPPORTED);
    REQUIRE(ovspatialquery_destroy_population(instance, userPopulation).status == OVSPATIALQUERY_API_SUCCESS);
    cuda.free(userAllocation);

    auto missingRotation                     = set;
    missingRotation.instances.rotations_xyzw = nullptr;
    requireRejected(ovspatialquery_enqueue_explicit_population_set(instance, population, &missingRotation),
                    OVSPATIALQUERY_API_INVALID_ARGUMENT);
    auto missingTranslation                   = set;
    missingTranslation.instances.translations = nullptr;
    requireRejected(ovspatialquery_enqueue_explicit_population_set(instance, population, &missingTranslation),
                    OVSPATIALQUERY_API_INVALID_ARGUMENT);
    const ovspatialquery_device_budget_t tightDeviceBudget{{kDLCUDA, 0}, 1u};
    const ovspatialquery_instance_desc_t tightInstanceDesc{2u, 1u, UINT64_C(16) * 1024u * 1024u, &tightDeviceBudget,
                                                           1u};
    ovspatialquery_instance_handle_t     tightInstance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&tightInstanceDesc, &tightInstance).status == OVSPATIALQUERY_API_SUCCESS);
    waitAndDestroy(tightInstance, ovspatialquery_enqueue_box_registration(tightInstance, &boxes));
    const ovspatialquery_population_handle_t tightPopulation  = createExplicitPopulation(tightInstance, 1u);
    const uint64_t                           tightDeviceUsage = deviceUsage(tightInstance, {kDLCUDA, 0});
    requireRejected(ovspatialquery_enqueue_explicit_population_set(tightInstance, tightPopulation, &set),
                    OVSPATIALQUERY_API_DEVICE_BUDGET);
    REQUIRE(deviceUsage(tightInstance, {kDLCUDA, 0}) == tightDeviceUsage);
    ovspatialquery_population_info_t tightInfo{};
    REQUIRE(ovspatialquery_get_population_info(tightInstance, tightPopulation, &tightInfo).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(tightInfo.ready == 0u && tightInfo.object_count == 0u && tightInfo.update_pending == 0u);
    REQUIRE(ovspatialquery_destroy_population(tightInstance, tightPopulation).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_geometry_definition(tightInstance, {OVSPATIALQUERY_GEOMETRY_BOX, definitionValue})
                .status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(tightInstance).status == OVSPATIALQUERY_API_SUCCESS);

    // A copied set that fails on the worker, here on a zero identity, releases
    // its device staging and published-data charges.
    const uint64_t                  zeroValue      = 0u;
    const CudaDriver::DevicePointer zeroAllocation = cuda.allocate(sizeof(zeroValue));
    cuda.uploadAsync(zeroAllocation, &zeroValue, sizeof(zeroValue), stream);
    cuda.record(ready, stream);
    InputArray zeroObjects(&objectValue, 1, kDLUInt);
    zeroObjects.value.tensor.data                    = reinterpret_cast<void*>(static_cast<uintptr_t>(zeroAllocation));
    zeroObjects.value.tensor.device                  = {kDLCUDA, 0};
    zeroObjects.value.ownership                      = OVSPATIALQUERY_INPUT_COPIED_ON_SUBMISSION;
    auto zeroSet                                     = set;
    zeroSet.instances.object_values                  = &zeroObjects.value;
    const ovspatialquery_population_handle_t failing = createExplicitPopulation(instance, 1u);
    const uint64_t                           deviceBeforeFailure = deviceUsage(instance, {kDLCUDA, 0});
    const ovspatialquery_enqueue_result_t    zeroAdmission =
        ovspatialquery_enqueue_explicit_population_set(instance, failing, &zeroSet);
    REQUIRE(zeroAdmission.status == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_op_wait_result_t zeroWait{};
    REQUIRE(ovspatialquery_wait_op(instance, zeroAdmission.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &zeroWait).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(zeroWait.state == OVSPATIALQUERY_OP_FAILURE);
    REQUIRE(deviceUsage(instance, {kDLCUDA, 0}) == deviceBeforeFailure);
    ovspatialquery_op_id_t zeroId = zeroAdmission.op_id;
    REQUIRE(ovspatialquery_destroy_op(instance, &zeroId).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_population(instance, failing).status == OVSPATIALQUERY_API_SUCCESS);
    cuda.free(zeroAllocation);

    const ovspatialquery_enqueue_result_t setAdmission =
        ovspatialquery_enqueue_explicit_population_set(instance, population, &set);
    REQUIRE(setAdmission.status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(
        ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_BOX, definitionValue}).status ==
        OVSPATIALQUERY_API_BUSY);
    const double reusedTranslation[3] = {100.0, 0.0, 0.0};
    cuda.uploadAsync(allocations[3], reusedTranslation, sizeof(reusedTranslation), stream);
    cuda.record(ready, stream);
    cuda.synchronize(ready);
    waitOnly(instance, setAdmission);
    ovspatialquery_raycast_desc_t ray{};
    ray.common.target         = populationTarget(population);
    ray.common.precision      = OVSPATIALQUERY_QUERY_EXACT;
    ray.common.query_bits     = participationBits(UINT64_MAX, UINT64_MAX);
    ray.ray.unit_direction[0] = 1.0f;
    ray.ray.max_distance      = 10.0;
    ovspatialquery_raycast_single_result_t result{};
    REQUIRE(ovspatialquery_raycast_closest_sync(instance, &ray, &result).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(result.source_ordinal == 1 && result.has_hit == 1u &&
            std::abs(result.hit.data.exact.distance - (3.0 - xExtent)) < 1.0e-12);
    ovspatialquery_op_id_t setId = setAdmission.op_id;
    REQUIRE(ovspatialquery_destroy_op(instance, &setId).status == OVSPATIALQUERY_API_SUCCESS);

    translation[0] = 5.0;
    cuda.uploadAsync(allocations[3], translation, sizeof(translation), stream);
    cuda.record(ready, stream);
    for (InputArray* input : inputs)
    {
        input->value.ownership = OVSPATIALQUERY_INPUT_BORROWED_UNTIL_TERMINAL;
    }
    uint64_t   removedSentinel = 0u;
    InputArray removed(&removedSentinel, 0, kDLUInt, {kDLCUDA, 0});
    removed.value.ownership = OVSPATIALQUERY_INPUT_BORROWED_UNTIL_TERMINAL;
    removed.strides[0]      = 0; // a producer's stride for an empty dimension
    const ovspatialquery_explicit_population_update_t update{
        1,
        2,
        OVSPATIALQUERY_EXPLICIT_UPDATE_COMPLETE_ROWS,
        {&objects.value, &types.value, &geometry.value, &positions.value, &rotations.value, nullptr, nullptr, nullptr},
        &removed.value,
        &execution};
    const ovspatialquery_enqueue_result_t updateAdmission =
        ovspatialquery_enqueue_explicit_population_update(instance, population, &update);
    waitOnly(instance, updateAdmission);
    result = {};
    REQUIRE(ovspatialquery_raycast_closest_sync(instance, &ray, &result).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(result.source_ordinal == 2 && result.has_hit == 1u &&
            std::abs(result.hit.data.exact.distance - (5.0 - xExtent)) < 1.0e-12);

    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(
        ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_BOX, definitionValue}).status ==
        OVSPATIALQUERY_API_BUSY);
    ovspatialquery_op_id_t updateId = updateAdmission.op_id;
    REQUIRE(ovspatialquery_destroy_op(instance, &updateId).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(
        ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_BOX, definitionValue}).status ==
        OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
    for (CudaDriver::DevicePointer allocation : allocations)
    {
        cuda.free(allocation);
    }
    cuda.destroyEvent(ready);
    cuda.destroyStream(stream);
}
#endif

// A sphere swept past a box or capsule. Each path passes between a sixth and a
// half of the expanded radius inside the surface, so its first contact is well
// defined; the contacts were computed in exact rational arithmetic. The last path
// starts on the expanded capsule surface, 1e-16 outside it in squared distance,
// and then moves inward.
struct PenetratingSweep
{
    bool   capsule;
    double origin[3];
    float  direction[3];
    double translation[3];
    double halfExtents[3];
    double capsuleRadius;
    double halfSegment;
    double sweepRadius;
    double maximumDistance;
    double contact;
};

constexpr PenetratingSweep kPenetratingSweeps[] = {
    {false,
     {0x1.63adf3891cd22p+0, 0x1.b589ebf0ba562p+1, 0x1.f5e22bde474c8p+1},
     {-0x1.33003p-3f, -0x1.820cd6p-1f, -0x1.477154p-1f},
     {0.0, 0.0, 0.0},
     {0x1.9f817a38c210dp+0, 0x1.c7e6f3ec74571p-1, 0x1.0a05e7afb16f3p-2},
     0.0,
     0.0,
     0x1.a7ec528d6b8b6p-8,
     0x1.6dffe051589dep+3,
     5.714622378222896},
    {true,
     {-0x1.d6c505fc02004p+7, -0x1.863d4d3fe5c74p+10, 0x1.d791f9908b89cp+10},
     {-0x1.dcba3cp-1f, 0x1.0fb954p-5f, -0x1.73f2dep-2f},
     {-0x1.2bfe780149322p+10, -0x1.7da70509d662fp+10, 0x1.78e8009c35044p+10},
     {1.0, 1.0, 1.0},
     0x1.b3ff1815bcd4fp+0,
     0x1.463a78ba3d62cp+8,
     0x1.ef8371034d953p-1,
     0x1.033471dc91a5bp+11,
     1035.3058624251262},
    {true,
     {-0x1.103367dd4da36p+2, -0x1.610c9d62d52b8p+6, -0x1.99eda6d147392p+2},
     {0x1.10b00ap-2f, -0x1.ef6b82p-4f, -0x1.e99c86p-1f},
     {-0x1.1df6a25fce1e5p+2, -0x1.c02ed76314439p+6, -0x1.d77cc201a2662p+2},
     {1.0, 1.0, 1.0},
     0x1.1b217e965b8e3p-1,
     0x1.48d4283c9c1dbp+6,
     0x1.baff5253bab5ep-2,
     0x1.b8352d5f9ee7fp+6,
     6.047453018163348e-17},
};

// A contact agrees when it lies within 1024 ulps of the path's largest coordinate.
bool sameContact(const PenetratingSweep& sweep, double distance, double expected)
{
    double scale = sweep.maximumDistance;
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        scale = (std::max)({scale, std::abs(sweep.origin[axis]), std::abs(sweep.translation[axis])});
    }
    return std::abs(distance - expected) <= 1024.0 * std::numeric_limits<double>::epsilon() * scale;
}

#if defined(OVSPATIALQUERY_TEST_HAS_CUDA_BACKEND)
// The CUDA sphere-sweep plan's EXACT contact for one row, or a negative distance
// when it reports none.
double cudaSweepContact(ovspatialquery_instance_handle_t instance, ovspatialquery_population_handle_t population,
                        const PenetratingSweep& sweep)
{
    CudaDriver                              cuda;
    const CudaDriver::Stream                stream = cuda.createStream();
    ovspatialquery_sphere_sweep_plan_desc_t planDesc{};
    planDesc.target                         = populationRef(population);
    planDesc.precision                      = OVSPATIALQUERY_QUERY_EXACT;
    planDesc.radius                         = sweep.sweepRadius;
    planDesc.device                         = {kDLCUDA, 0};
    planDesc.maximum_query_count            = 1u;
    planDesc.result_slot_count              = 1u;
    planDesc.result_fields                  = OVSPATIALQUERY_BATCH_RESULT_DISTANCE;
    planDesc.hit_capacity_per_slot          = 1u;
    ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_sphere_sweep_plan(instance, &planDesc, &plan).status == OVSPATIALQUERY_API_SUCCESS);

    const CudaDriver::DevicePointer center    = cuda.allocate(sizeof(sweep.origin));
    const CudaDriver::DevicePointer direction = cuda.allocate(sizeof(sweep.direction));
    const CudaDriver::DevicePointer maximum   = cuda.allocate(sizeof(double));
    cuda.uploadAsync(center, sweep.origin, sizeof(sweep.origin), stream);
    cuda.uploadAsync(direction, sweep.direction, sizeof(sweep.direction), stream);
    cuda.uploadAsync(maximum, &sweep.maximumDistance, sizeof(double), stream);
    const DLDevice device{kDLCUDA, 0};
    InputArray     centers(reinterpret_cast<const double*>(static_cast<uintptr_t>(center)), 1, 3, kDLFloat, device);
    InputArray directions(reinterpret_cast<const float*>(static_cast<uintptr_t>(direction)), 1, 3, kDLFloat, device);
    InputArray maxima(reinterpret_cast<const double*>(static_cast<uintptr_t>(maximum)), 1, kDLFloat, device);
    ovspatialquery_sphere_sweep_batch_t batch{};
    batch.ordinal.kind         = OVSPATIALQUERY_ORDINAL_HEAD;
    batch.query_count          = 1u;
    batch.centers              = &centers.value;
    batch.unit_directions      = &directions.value;
    batch.max_distances        = &maxima.value;
    batch.cuda_sync.has_stream = 1u;
    batch.cuda_sync.stream     = reinterpret_cast<uintptr_t>(stream);
    ovspatialquery_sphere_sweep_batch_reservation_t reservation{};
    ovspatialquery_enqueue_result_t                 operation =
        ovspatialquery_enqueue_sphere_sweep_plan(instance, plan, &batch, &reservation);
    waitOnly(instance, operation);
    ovspatialquery_sphere_sweep_batch_result_t result{};
    REQUIRE(ovspatialquery_fetch_sphere_sweep_batch(instance, operation.op_id, &result).status ==
            OVSPATIALQUERY_API_SUCCESS);
    cuda.wait(stream, reinterpret_cast<CudaDriver::Event>(result.common.result_ready.value));
    const auto address = [](const DLTensor& tensor)
    { return static_cast<CudaDriver::DevicePointer>(reinterpret_cast<uintptr_t>(tensor.data)) + tensor.byte_offset; };
    uint32_t rowStatus = UINT32_MAX;
    uint64_t stored    = UINT64_MAX;
    double   distance  = -1.0;
    cuda.downloadAsync(&rowStatus, address(result.columns.hit_ranges.query_status), sizeof(rowStatus), stream);
    cuda.downloadAsync(&stored, address(result.columns.hit_ranges.stored_hit_count), sizeof(stored), stream);
    cuda.downloadAsync(&distance, address(result.columns.contact.exact.distances), sizeof(distance), stream);
    const CudaDriver::Event done = cuda.createEvent();
    cuda.record(done, stream);
    cuda.synchronize(done);
    REQUIRE(rowStatus == OVSPATIALQUERY_BATCH_ROW_SUCCESS);
    // The library destroys the transferred consumer event once the operation retires.
    ovspatialquery_cuda_event_t consumerDone{reinterpret_cast<uintptr_t>(done)};
    REQUIRE(ovspatialquery_destroy_op_after_cuda(instance, &operation.op_id, &consumerDone).status ==
            OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_api_status_t destroyed = OVSPATIALQUERY_API_BUSY;
    for (int attempt = 0; attempt < 1000 && destroyed == OVSPATIALQUERY_API_BUSY; ++attempt)
    {
        destroyed = ovspatialquery_destroy_batch_plan(instance, plan).status;
        if (destroyed == OVSPATIALQUERY_API_BUSY)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    REQUIRE(destroyed == OVSPATIALQUERY_API_SUCCESS);
    cuda.free(center);
    cuda.free(direction);
    cuda.free(maximum);
    cuda.destroyStream(stream);
    return stored == 1u ? distance : -1.0;
}
#endif

// EXACT CPU sphere sweeps report each penetrating contact at its exact first
// distance and agree with the CUDA plan. The CPU path used to reject such roots when
// its residual guard's tolerance, relative to the radius, fell below the roundoff
// of coordinates much larger than the radius, and when a root rounded behind a
// start already on the surface.
void testPrimitiveSweepsReportPenetratingContacts()
{
#if defined(OVSPATIALQUERY_TEST_HAS_CUDA_BACKEND)
    const ovspatialquery_device_budget_t  deviceBudget{{kDLCUDA, 0}, UINT64_C(64) * 1024u * 1024u};
    const ovspatialquery_device_budget_t* budgets     = &deviceBudget;
    const uint32_t                        budgetCount = 1u;
#else
    const ovspatialquery_device_budget_t* budgets     = nullptr;
    const uint32_t                        budgetCount = 0u;
#endif
    for (const PenetratingSweep& sweep : kPenetratingSweeps)
    {
        const ovspatialquery_instance_desc_t instanceDesc{4u, 1u, UINT64_C(16) * 1024u * 1024u, budgets, budgetCount};
        ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
        REQUIRE(ovspatialquery_create_instance(&instanceDesc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
        uint64_t                                        definitionValue = 1u;
        InputArray                                      definitionValues(&definitionValue, 1, kDLUInt);
        InputArray                                      halfExtents(sweep.halfExtents, 1, 3, kDLFloat);
        InputArray                                      radii(&sweep.capsuleRadius, 1, kDLFloat);
        InputArray                                      halfSegments(&sweep.halfSegment, 1, kDLFloat);
        const ovspatialquery_box_definition_table_t     boxes{&definitionValues.value, &halfExtents.value};
        const ovspatialquery_capsule_definition_table_t capsules{&definitionValues.value, &radii.value,
                                                                 &halfSegments.value};
        waitAndDestroy(instance, sweep.capsule ? ovspatialquery_enqueue_capsule_registration(instance, &capsules)
                                               : ovspatialquery_enqueue_box_registration(instance, &boxes));
        const ovspatialquery_population_handle_t population  = createExplicitPopulation(instance, 1u);
        uint64_t                                 objectValue = 100u;
        int32_t    geometryType = sweep.capsule ? OVSPATIALQUERY_GEOMETRY_CAPSULE : OVSPATIALQUERY_GEOMETRY_BOX;
        float      rotation[4]  = {0.0f, 0.0f, 0.0f, 1.0f};
        InputArray objects(&objectValue, 1, kDLUInt);
        InputArray types(&geometryType, 1, kDLInt);
        InputArray geometry(&definitionValue, 1, kDLUInt);
        InputArray positions(sweep.translation, 1, 3, kDLFloat);
        InputArray rotations(rotation, 1, 4, kDLFloat);
        const ovspatialquery_explicit_population_set_t set{1,
                                                           {&objects.value, &types.value, &geometry.value,
                                                            &positions.value, &rotations.value, nullptr, nullptr,
                                                            nullptr},
                                                           nullptr,
                                                           0u};
        waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set));

        ovspatialquery_sphere_sweep_desc_t desc{};
        desc.common.target     = populationTarget(population);
        desc.common.precision  = OVSPATIALQUERY_QUERY_EXACT;
        desc.common.query_bits = participationBits(UINT64_MAX, UINT64_MAX);
        std::copy(std::begin(sweep.origin), std::end(sweep.origin), desc.sweep.center);
        std::copy(std::begin(sweep.direction), std::end(sweep.direction), desc.sweep.unit_direction);
        desc.sweep.radius       = sweep.sweepRadius;
        desc.sweep.max_distance = sweep.maximumDistance;
        ovspatialquery_sphere_sweep_closest_result_t result{};
        REQUIRE(ovspatialquery_sweep_sphere_closest_sync(instance, &desc, &result).status ==
                OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(result.has_hit == 1u);
        REQUIRE(sameContact(sweep, result.hit.data.exact.distance, sweep.contact));
#if defined(OVSPATIALQUERY_TEST_HAS_CUDA_BACKEND)
        REQUIRE(sameContact(sweep, cudaSweepContact(instance, population, sweep), result.hit.data.exact.distance));
#endif
        REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
        const ovspatialquery_geometry_ref_t definition{
            sweep.capsule ? OVSPATIALQUERY_GEOMETRY_CAPSULE : OVSPATIALQUERY_GEOMETRY_BOX, definitionValue};
        REQUIRE(ovspatialquery_destroy_geometry_definition(instance, definition).status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
    }
}

} // namespace

// A POSES update moves existing objects and keeps their binding, bits and
// scale; the host path answers its admission rules like the device path.
void testPosesUpdateKeepsBindingsOnTheHost()
{
    const ovspatialquery_instance_desc_t instanceDesc{4u, 1u, UINT64_C(16) * 1024u * 1024u, nullptr, 0u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&instanceDesc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    uint64_t definitionValue = 40u;
    registerSphere(instance, definitionValue, 1.0);
    const ovspatialquery_population_handle_t population = createExplicitPopulation(instance, 2u);

    uint64_t   objectValues[2]   = {410u, 411u};
    int32_t    families[2]       = {OVSPATIALQUERY_GEOMETRY_SPHERE, OVSPATIALQUERY_GEOMETRY_SPHERE};
    uint64_t   geometryValues[2] = {definitionValue, definitionValue};
    double     translations[6]   = {3.0, 0.0, 0.0, -3.0, 0.0, 0.0};
    double     rotations[8]      = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    double     scales[2]         = {1.0, 2.0};
    InputArray objects(objectValues, 2, kDLUInt), types(families, 2, kDLInt), geometry(geometryValues, 2, kDLUInt);
    InputArray positions(translations, 2, 3, kDLFloat), orientations(rotations, 2, 4, kDLFloat);
    InputArray uniformScales(scales, 2, kDLFloat);
    const ovspatialquery_explicit_population_set_t set{1,
                                                       {&objects.value, &types.value, &geometry.value, &positions.value,
                                                        &orientations.value, &uniformScales.value, nullptr, nullptr},
                                                       nullptr,
                                                       0u};
    // ON_DEVICE needs CUDA rows, whatever the rows are.
    const ovspatialquery_dynamic_execution_t onDeviceFromCpu{
        {kDLCPU, 0}, {0u, 0u, 0u}, OVSPATIALQUERY_PREPARE_ON_DEVICE};
    ovspatialquery_explicit_population_set_t misplaced = set;
    misplaced.dynamic_execution                        = &onDeviceFromCpu;
    requireRejected(ovspatialquery_enqueue_explicit_population_set(instance, population, &misplaced),
                    OVSPATIALQUERY_API_INVALID_ARGUMENT);
    waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set));

    const auto closestDistance = [&](double directionX)
    {
        ovspatialquery_raycast_desc_t ray{};
        ray.common.target         = populationTarget(population);
        ray.common.precision      = OVSPATIALQUERY_QUERY_EXACT;
        ray.common.query_bits     = participationBits(UINT64_MAX, UINT64_MAX);
        ray.ray.unit_direction[0] = static_cast<float>(directionX);
        ray.ray.max_distance      = 100.0;
        ovspatialquery_raycast_single_result_t result{};
        REQUIRE(ovspatialquery_raycast_closest_sync(instance, &ray, &result).status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(result.has_hit == 1u);
        return result.hit.data.exact.distance;
    };
    REQUIRE(std::abs(closestDistance(1.0) - 2.0) < 1.0e-9);  // radius 1 at x = 3
    REQUIRE(std::abs(closestDistance(-1.0) - 1.0) < 1.0e-9); // radius 2 at x = -3

    uint64_t   movedValues[2]       = {411u, 410u}; // any order
    double     movedTranslations[6] = {-6.0, 0.0, 0.0, 5.0, 0.0, 0.0};
    InputArray movedObjects(movedValues, 2, kDLUInt), movedPositions(movedTranslations, 2, 3, kDLFloat);
    InputArray movedOrientations(rotations, 2, 4, kDLFloat);
    uint64_t   zero = 0u;
    InputArray noRemovals(&zero, 0, kDLUInt);
    ovspatialquery_explicit_population_update_t poses{1,
                                                      2,
                                                      OVSPATIALQUERY_EXPLICIT_UPDATE_POSES,
                                                      {&movedObjects.value, nullptr, nullptr, &movedPositions.value,
                                                       &movedOrientations.value, nullptr, nullptr, nullptr},
                                                      &noRemovals.value,
                                                      nullptr};
    // Binding columns, removals, unknown objects and an unknown kind are refused at admission.
    ovspatialquery_explicit_population_update_t withBindings = poses;
    withBindings.changed_instances.geometry_types            = &types.value;
    withBindings.changed_instances.geometry_values           = &geometry.value;
    requireRejected(ovspatialquery_enqueue_explicit_population_update(instance, population, &withBindings),
                    OVSPATIALQUERY_API_INVALID_ARGUMENT);
    uint64_t                                    removedValue = 410u;
    InputArray                                  removed(&removedValue, 1, kDLUInt);
    ovspatialquery_explicit_population_update_t withRemovals = poses;
    withRemovals.removed_object_values                       = &removed.value;
    requireRejected(ovspatialquery_enqueue_explicit_population_update(instance, population, &withRemovals),
                    OVSPATIALQUERY_API_INVALID_ARGUMENT);
    uint64_t                                    unknownValues[2] = {411u, 999u};
    InputArray                                  unknownObjects(unknownValues, 2, kDLUInt);
    ovspatialquery_explicit_population_update_t unknown = poses;
    unknown.changed_instances.object_values             = &unknownObjects.value;
    requireRejected(ovspatialquery_enqueue_explicit_population_update(instance, population, &unknown),
                    OVSPATIALQUERY_API_INVALID_ARGUMENT);
    ovspatialquery_explicit_population_update_t unknownKind = poses;
    unknownKind.kind                                        = static_cast<ovspatialquery_explicit_update_kind_t>(7);
    requireRejected(ovspatialquery_enqueue_explicit_population_update(instance, population, &unknownKind),
                    OVSPATIALQUERY_API_INVALID_ARGUMENT);
    ovspatialquery_explicit_population_update_t misplacedUpdate = poses;
    misplacedUpdate.dynamic_execution                           = &onDeviceFromCpu;
    requireRejected(ovspatialquery_enqueue_explicit_population_update(instance, population, &misplacedUpdate),
                    OVSPATIALQUERY_API_INVALID_ARGUMENT);

    waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_update(instance, population, &poses));
    ovspatialquery_population_info_t info{};
    REQUIRE(ovspatialquery_get_population_info(instance, population, &info).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(info.head_ordinal == 2 && info.object_count == 2u);
    REQUIRE(std::abs(closestDistance(1.0) - 4.0) < 1.0e-9);  // radius 1 kept at x = 5
    REQUIRE(std::abs(closestDistance(-1.0) - 4.0) < 1.0e-9); // radius 2 kept at x = -6

    // POSES_AND_BITS rows set their objects' bits for this version and the ones
    // built on it; the column is required there and refused with POSES, and the
    // retained ordinal 2 keeps the bits it was published with (ADR-0003).
    const auto hitsAlongX = [&](uint64_t queryBits, ovspatialquery_ordinal_selection_kind_t kind,
                                ovspatialquery_ordinal_t ordinal, double expectedDistance)
    {
        ovspatialquery_raycast_desc_t ray{};
        ray.common.target                        = populationTarget(population);
        ray.common.target.ordinal.kind           = kind;
        ray.common.target.ordinal.source_ordinal = ordinal;
        ray.common.precision                     = OVSPATIALQUERY_QUERY_EXACT;
        ray.common.query_bits                    = participationBits(queryBits);
        ray.ray.unit_direction[0]                = 1.0f;
        ray.ray.max_distance                     = 100.0;
        ovspatialquery_raycast_single_result_t result{};
        REQUIRE(ovspatialquery_raycast_closest_sync(instance, &ray, &result).status == OVSPATIALQUERY_API_SUCCESS);
        if (expectedDistance < 0.0)
        {
            return result.has_hit == 0u;
        }
        return result.has_hit == 1u && std::abs(result.hit.data.exact.distance - expectedDistance) < 1.0e-9;
    };
    uint64_t                                    newBits[4] = {2u, 0u, 2u, 0u};
    InputArray                                  rowBits(newBits, 2, 2, kDLUInt);
    ovspatialquery_explicit_population_update_t rebits = poses;
    rebits.base_ordinal                                = 2;
    rebits.target_ordinal                              = 3;
    rebits.kind                                        = OVSPATIALQUERY_EXPLICIT_UPDATE_POSES_AND_BITS;
    requireRejected(ovspatialquery_enqueue_explicit_population_update(instance, population, &rebits),
                    OVSPATIALQUERY_API_INVALID_ARGUMENT); // the bits column is required
    ovspatialquery_explicit_population_update_t posesWithBits = poses;
    posesWithBits.base_ordinal                                = 2;
    posesWithBits.target_ordinal                              = 3;
    posesWithBits.changed_instances.participation_bits        = &rowBits.value;
    requireRejected(ovspatialquery_enqueue_explicit_population_update(instance, population, &posesWithBits),
                    OVSPATIALQUERY_API_INVALID_ARGUMENT); // POSES refuses it
    rebits.changed_instances.participation_bits = &rowBits.value;
    waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_update(instance, population, &rebits));
    REQUIRE(hitsAlongX(1u, OVSPATIALQUERY_ORDINAL_HEAD, 0, -1.0));          // bit 1 left object 410
    REQUIRE(hitsAlongX(2u, OVSPATIALQUERY_ORDINAL_HEAD, 0, 4.0));           // bit 2 finds it at x = 5
    REQUIRE(hitsAlongX(1u, OVSPATIALQUERY_ORDINAL_RETAINED_EXACT, 2, 4.0)); // ordinal 2 keeps bit 1

    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_SPHERE, definitionValue})
                .status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

// ADR-0021: ADMITTED_EXACT belongs to CUDA plan launches. On the host path a
// synchronous query and a CPU plan reject it with INVALID_ARGUMENT, while the
// same ordinal stays selectable as RETAINED_EXACT, and an explicit population
// accepts a pending-publication capacity above one.
void testAdmittedSelectionIsRejectedOnTheHostPath()
{
    const ovspatialquery_instance_desc_t instanceDesc{3u, 1u, UINT64_C(16) * 1024u * 1024u, nullptr, 0u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&instanceDesc, &instance).status == OVSPATIALQUERY_API_SUCCESS);

    uint64_t definitionValue = 70u;
    registerSphere(instance, definitionValue, 1.0);

    const ovspatialquery_population_desc_t populationDesc{1u, 2u};
    ovspatialquery_population_handle_t     population = OVSPATIALQUERY_INVALID_POPULATION;
    REQUIRE(ovspatialquery_create_explicit_population(instance, &populationDesc, &population).status ==
            OVSPATIALQUERY_API_SUCCESS);

    uint64_t                                 objectValue    = 700u;
    int32_t                                  family         = OVSPATIALQUERY_GEOMETRY_SPHERE;
    double                                   translation[3] = {2.0, 0.0, 0.0};
    float                                    rotation[4]    = {0.0f, 0.0f, 0.0f, 1.0f};
    InputArray                               objects(&objectValue, 1, kDLUInt);
    InputArray                               types(&family, 1, kDLInt);
    InputArray                               geometry(&definitionValue, 1, kDLUInt);
    InputArray                               positions(translation, 1, 3, kDLFloat);
    InputArray                               rotations(rotation, 1, 4, kDLFloat);
    ovspatialquery_explicit_population_set_t set{
        3,
        {&objects.value, &types.value, &geometry.value, &positions.value, &rotations.value, nullptr, nullptr, nullptr},
        nullptr,
        0u};
    // A slot capacity sizes device tables; a host-prepared set has none (ADR-0024).
    set.slot_capacity = 2u;
    requireRejected(ovspatialquery_enqueue_explicit_population_set(instance, population, &set),
                    OVSPATIALQUERY_API_INVALID_ARGUMENT);
    set.slot_capacity = 0u;
    waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set));

    ovspatialquery_raycast_desc_t ray{};
    ray.common.target                        = populationTarget(population);
    ray.common.target.ordinal.kind           = OVSPATIALQUERY_ORDINAL_ADMITTED_EXACT;
    ray.common.target.ordinal.source_ordinal = 3;
    ray.common.precision                     = OVSPATIALQUERY_QUERY_EXACT;
    ray.common.query_bits                    = participationBits(UINT64_MAX, UINT64_MAX);
    ray.ray.unit_direction[0]                = 1.0f;
    ray.ray.max_distance                     = 10.0;
    ovspatialquery_raycast_single_result_t result{};
    REQUIRE(ovspatialquery_raycast_closest_sync(instance, &ray, &result).status == OVSPATIALQUERY_API_INVALID_ARGUMENT);
    ray.common.target.ordinal.kind = OVSPATIALQUERY_ORDINAL_RETAINED_EXACT;
    REQUIRE(ovspatialquery_raycast_closest_sync(instance, &ray, &result).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(result.has_hit == 1u && result.source_ordinal == 3);

    ovspatialquery_overlap_plan_desc_t planDesc{};
    planDesc.target                         = ray.common.target.target;
    planDesc.hit_mode                       = OVSPATIALQUERY_HIT_MODE_MULTIPLE;
    planDesc.precision                      = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    planDesc.shape.kind                     = OVSPATIALQUERY_OVERLAP_SPHERE;
    planDesc.shape.data.sphere.radius       = 0.5;
    planDesc.device                         = DLDevice{kDLCPU, 0};
    planDesc.maximum_query_count            = 1u;
    planDesc.maximum_hits_per_query         = 1u;
    planDesc.hit_capacity_per_slot          = 1u;
    planDesc.result_slot_count              = 1u;
    ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_overlap_plan(instance, &planDesc, &plan).status == OVSPATIALQUERY_API_SUCCESS);
    double                         center[3] = {2.0, 0.0, 0.0};
    uint64_t                       bits[2]   = {UINT64_MAX, UINT64_MAX};
    InputArray                     centers(center, 1, 3, kDLFloat);
    InputArray                     queryBits(bits, 1, 2, kDLUInt);
    ovspatialquery_overlap_batch_t batch{};
    batch.ordinal      = {OVSPATIALQUERY_ORDINAL_ADMITTED_EXACT, 3};
    batch.query_count  = 1u;
    batch.query_bits   = &queryBits.value;
    batch.translations = &centers.value;
    ovspatialquery_overlap_batch_reservation_t reservation{};
    requireRejected(ovspatialquery_enqueue_overlap_plan(instance, plan, &batch, &reservation),
                    OVSPATIALQUERY_API_INVALID_ARGUMENT);
    batch.ordinal.kind = OVSPATIALQUERY_ORDINAL_RETAINED_EXACT;
    waitAndDestroy(instance, ovspatialquery_enqueue_overlap_plan(instance, plan, &batch, &reservation));

    REQUIRE(ovspatialquery_destroy_batch_plan(instance, plan).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_SPHERE, definitionValue})
                .status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

// A host-prepared publication admitted within the host budget completes: its
// admission charge, measured while the workers are paused, covers the version
// the worker builds. One byte less refuses it up front with HOST_BUDGET.
// BH-033: a 256 x 256 grid mesh (131072 triangles) that registers and publishes under the
// default 64 MiB host budget answers its first exact query and an exact CPU batch launch
// within that budget, because its registration reserved the query structure the first
// reader builds (ADR-0075).
void testLargeMeshAnswersExactReadersUnderTheBudgetItRegisteredUnder()
{
    constexpr uint32_t    kCells = 256u;
    std::vector<float>    gridVertices;
    std::vector<uint32_t> gridTriangles;
    for (uint32_t row = 0u; row <= kCells; ++row)
        for (uint32_t column = 0u; column <= kCells; ++column)
            gridVertices.insert(gridVertices.end(), {float(column) / kCells, float(row) / kCells, 0.0f});
    for (uint32_t row = 0u; row < kCells; ++row)
        for (uint32_t column = 0u; column < kCells; ++column)
        {
            const uint32_t corner = row * (kCells + 1u) + column;
            gridTriangles.insert(gridTriangles.end(), {corner, corner + 1u, corner + kCells + 2u, corner,
                                                       corner + kCells + 2u, corner + kCells + 1u});
        }
    const ovspatialquery_instance_desc_t instanceDesc{8u, 1u, UINT64_C(64) << 20u, nullptr, 0u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&instanceDesc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    uint64_t   meshValue = 33u, zero = 0u;
    uint64_t   vertexCount = gridVertices.size() / 3u, triangleCount = gridTriangles.size() / 3u;
    InputArray values(&meshValue, 1, kDLUInt), vertexOffsets(&zero, 1, kDLUInt), vertexCounts(&vertexCount, 1, kDLUInt);
    InputArray triangleOffsets(&zero, 1, kDLUInt), triangleCounts(&triangleCount, 1, kDLUInt);
    InputArray vertices(gridVertices.data(), static_cast<int64_t>(vertexCount), 3, kDLFloat);
    InputArray triangles(gridTriangles.data(), static_cast<int64_t>(triangleCount), 3, kDLUInt);
    const ovspatialquery_triangle_mesh_definition_table_t mesh{&values.value,
                                                               &vertexOffsets.value,
                                                               &vertexCounts.value,
                                                               &triangleOffsets.value,
                                                               &triangleCounts.value,
                                                               &vertices.value,
                                                               &triangles.value,
                                                               nullptr,
                                                               nullptr,
                                                               nullptr,
                                                               nullptr};
    waitAndDestroy(instance, ovspatialquery_enqueue_triangle_mesh_registration(instance, &mesh));
    const ovspatialquery_population_handle_t population = createExplicitPopulation(instance, 1u);
    InstanceRows                             rows;
    rows.add(1u, OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, meshValue, 0.0, 0.0, 0.0);
    const ovspatialquery_explicit_population_set_t set{1, rows.table(), nullptr, 0u};
    waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set));

    // The batch launch is the first reader: its execution builds the mesh.
    ovspatialquery_raycast_plan_desc_t planDesc{};
    planDesc.target                         = populationRef(population);
    planDesc.hit_mode                       = OVSPATIALQUERY_HIT_MODE_CLOSEST;
    planDesc.precision                      = OVSPATIALQUERY_QUERY_EXACT;
    planDesc.device                         = {kDLCPU, 0};
    planDesc.maximum_query_count            = 1u;
    planDesc.maximum_hits_per_query         = 1u;
    planDesc.hit_capacity_per_slot          = 1u;
    planDesc.result_slot_count              = 1u;
    planDesc.result_fields                  = OVSPATIALQUERY_BATCH_RESULT_DISTANCE;
    ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_raycast_plan(instance, &planDesc, &plan).status == OVSPATIALQUERY_API_SUCCESS);
    double                         origin[3] = {0.3, 0.3, 1.0}, maximumDistance = 10.0;
    float                          direction[3] = {0.0f, 0.0f, -1.0f};
    InputArray                     origins(origin, 1, 3, kDLFloat), directions(direction, 1, 3, kDLFloat);
    InputArray                     maximums(&maximumDistance, 1, kDLFloat);
    ovspatialquery_raycast_batch_t batch{};
    batch.ordinal.kind    = OVSPATIALQUERY_ORDINAL_HEAD;
    batch.query_count     = 1u;
    batch.origins         = &origins.value;
    batch.unit_directions = &directions.value;
    batch.max_distances   = &maximums.value;
    ovspatialquery_raycast_batch_reservation_t reservation{};
    ovspatialquery_enqueue_result_t launch = ovspatialquery_enqueue_raycast_plan(instance, plan, &batch, &reservation);
    REQUIRE(launch.status == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_op_wait_result_t waited{};
    REQUIRE(ovspatialquery_wait_op(instance, launch.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(waited.state == OVSPATIALQUERY_OP_COMPLETED);
    ovspatialquery_raycast_batch_result_t fetched{};
    REQUIRE(ovspatialquery_fetch_raycast_batch(instance, launch.op_id, &fetched).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(static_cast<const uint64_t*>(fetched.columns.hit_ranges.hit_counts.data)[0] == 1u);
    REQUIRE(std::abs(static_cast<const double*>(fetched.columns.contact.exact.distances.data)[0] - 1.0) < 1.0e-6);
    REQUIRE(ovspatialquery_destroy_op(instance, &launch.op_id).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_batch_plan(instance, plan).status == OVSPATIALQUERY_API_SUCCESS);

    ovspatialquery_raycast_desc_t ray{};
    ray.common.target     = populationTarget(population);
    ray.common.precision  = OVSPATIALQUERY_QUERY_EXACT;
    ray.common.query_bits = participationBits(UINT64_MAX, UINT64_MAX);
    std::copy(std::begin(origin), std::end(origin), ray.ray.origin);
    std::copy(std::begin(direction), std::end(direction), ray.ray.unit_direction);
    ray.ray.max_distance = maximumDistance;
    ovspatialquery_raycast_single_result_t result{};
    REQUIRE(ovspatialquery_raycast_closest_sync(instance, &ray, &result).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(result.has_hit && std::abs(result.hit.data.exact.distance - 1.0) < 1.0e-6);
    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, meshValue})
                .status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

void testAdmittedPublicationCompletesWithinHostBudget()
{
    constexpr size_t      kRows           = 2000u;
    uint64_t              definitionValue = 7u;
    std::vector<uint64_t> objectValues(kRows);
    std::vector<int32_t>  geometryTypes(kRows, OVSPATIALQUERY_GEOMETRY_SPHERE);
    std::vector<uint64_t> geometryValues(kRows, definitionValue);
    std::vector<double>   translations(3u * kRows, 0.0);
    std::vector<float>    rotations(4u * kRows, 0.0f);
    for (size_t row = 0u; row < kRows; ++row)
    {
        objectValues[row]        = row + 1u;
        translations[3u * row]   = static_cast<double>(row);
        rotations[4u * row + 3u] = 1.0f;
    }
    InputArray                                     objects(objectValues.data(), kRows, kDLUInt);
    InputArray                                     types(geometryTypes.data(), kRows, kDLInt);
    InputArray                                     geometry(geometryValues.data(), kRows, kDLUInt);
    InputArray                                     positions(translations.data(), kRows, 3, kDLFloat);
    InputArray                                     orientations(rotations.data(), kRows, 4, kDLFloat);
    uint64_t                                       noRemoval = 0u;
    InputArray                                     removed(&noRemoval, 0, kDLUInt);
    const ovspatialquery_explicit_population_set_t set{1,
                                                       {&objects.value, &types.value, &geometry.value, &positions.value,
                                                        &orientations.value, nullptr, nullptr, nullptr},
                                                       nullptr,
                                                       0u};
    // Every row moves: the update rebuilds the whole version beside its base.
    const ovspatialquery_explicit_population_update_t update{1,
                                                             2,
                                                             OVSPATIALQUERY_EXPLICIT_UPDATE_COMPLETE_ROWS,
                                                             {&objects.value, &types.value, &geometry.value,
                                                              &positions.value, &orientations.value, nullptr, nullptr,
                                                              nullptr},
                                                             &removed.value,
                                                             nullptr};

    ovspatialquery_population_handle_t population = OVSPATIALQUERY_INVALID_POPULATION;
    const auto                         open       = [&](uint64_t budget)
    {
        const ovspatialquery_instance_desc_t desc{2u, 1u, budget, nullptr, 0u};
        ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
        REQUIRE(ovspatialquery_create_instance(&desc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
        registerSphere(instance, definitionValue, 1.0);
        population = createExplicitPopulation(instance, 0u);
        return instance;
    };
    const auto close = [&](ovspatialquery_instance_handle_t instance)
    {
        REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_SPHERE, definitionValue})
                    .status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
    };
    // The host usage while the admitted publication waits for a worker.
    const auto admittedUsage =
        [](ovspatialquery_instance_handle_t instance, const std::function<ovspatialquery_enqueue_result_t()>& enqueue)
    {
        REQUIRE(ovspatialquery::detail::setInstanceOperationWorkersPausedForTest(instance, true) ==
                OVSPATIALQUERY_API_SUCCESS);
        const ovspatialquery_enqueue_result_t admission = enqueue();
        const uint64_t                        usage     = hostUsage(instance);
        REQUIRE(ovspatialquery::detail::setInstanceOperationWorkersPausedForTest(instance, false) ==
                OVSPATIALQUERY_API_SUCCESS);
        waitAndDestroy(instance, admission);
        return usage;
    };
    const auto enqueueSet = [&](ovspatialquery_instance_handle_t instance)
    { return ovspatialquery_enqueue_explicit_population_set(instance, population, &set); };
    const auto enqueueUpdate = [&](ovspatialquery_instance_handle_t instance)
    { return ovspatialquery_enqueue_explicit_population_update(instance, population, &update); };

    ovspatialquery_instance_handle_t instance   = open(UINT64_C(64) << 20);
    const uint64_t                   setCharge  = admittedUsage(instance, [&] { return enqueueSet(instance); });
    const uint64_t                   fullCharge = admittedUsage(instance, [&] { return enqueueUpdate(instance); });
    close(instance);

    instance = open(setCharge - 1u);
    requireRejected(enqueueSet(instance), OVSPATIALQUERY_API_HOST_BUDGET);
    close(instance);
    instance = open(setCharge);
    waitAndDestroy(instance, enqueueSet(instance));
    close(instance);

    instance = open(fullCharge - 1u);
    waitAndDestroy(instance, enqueueSet(instance));
    requireRejected(enqueueUpdate(instance), OVSPATIALQUERY_API_HOST_BUDGET);
    close(instance);
    instance = open(fullCharge);
    waitAndDestroy(instance, enqueueSet(instance));
    waitAndDestroy(instance, enqueueUpdate(instance));
    close(instance);

    // The ready head answers while the update waits for a worker, given the
    // room the same query needs beside the head alone.
    const auto queryHead = [&](ovspatialquery_instance_handle_t queryInstance)
    {
        ovspatialquery_raycast_desc_t ray{};
        ray.common.target         = populationTarget(population);
        ray.common.precision      = OVSPATIALQUERY_QUERY_EXACT;
        ray.common.query_bits     = participationBits(UINT64_MAX, UINT64_MAX);
        ray.ray.origin[0]         = -2.0;
        ray.ray.unit_direction[0] = 1.0f;
        ray.ray.max_distance      = 4.0;
        ovspatialquery_raycast_single_result_t result{};
        const ovspatialquery_api_status_t      status =
            ovspatialquery_raycast_closest_sync(queryInstance, &ray, &result).status;
        REQUIRE(status == OVSPATIALQUERY_API_HOST_BUDGET ||
                (status == OVSPATIALQUERY_API_SUCCESS && result.source_ordinal == 1 && result.has_hit == 1u &&
                 result.hit.data.exact.identity.object_id.value == 1u));
        return status == OVSPATIALQUERY_API_SUCCESS;
    };
    instance = open(UINT64_C(64) << 20);
    waitAndDestroy(instance, enqueueSet(instance));
    const uint64_t headCharge = hostUsage(instance);
    close(instance);
    // The update moves the objects it prepares into its version, so beside the
    // head it never holds a second copy of that version, here as large as the
    // head: a materialization charged per object of base and update counted
    // one (BH-002).
    REQUIRE(fullCharge - headCharge < 2u * headCharge);
    uint64_t refused  = 0u;
    uint64_t answered = UINT64_C(64) << 20;
    while (answered - refused > 1u)
    {
        const uint64_t room = refused + (answered - refused) / 2u;
        instance            = open(headCharge + room);
        waitAndDestroy(instance, enqueueSet(instance));
        (queryHead(instance) ? answered : refused) = room;
        close(instance);
    }
    instance = open(fullCharge + answered);
    waitAndDestroy(instance, enqueueSet(instance));
    REQUIRE(ovspatialquery::detail::setInstanceOperationWorkersPausedForTest(instance, true) ==
            OVSPATIALQUERY_API_SUCCESS);
    const ovspatialquery_enqueue_result_t pending = enqueueUpdate(instance);
    REQUIRE(queryHead(instance));
    REQUIRE(ovspatialquery::detail::setInstanceOperationWorkersPausedForTest(instance, false) ==
            OVSPATIALQUERY_API_SUCCESS);
    waitAndDestroy(instance, pending);
    close(instance);
}

// Two spheres placed by one physics-style pose array, (2, 7) as x y z qx qy qz qw:
// object 10 at x = 5 and object 20 at y = 10. Each exact ray from the origin,
// along +x and along +y, must hit its object at the distance its radius times
// the shared scale gives.
void requireStridedSpheres(ovspatialquery_instance_handle_t instance, ovspatialquery_population_handle_t population,
                           ovspatialquery_ordinal_t ordinal, double shift = 0.0)
{
    // Radii 1 and 2 scaled by 0.5: object 10 is hit at 5 - 0.5, object 20 at 10 - 1,
    // each `shift` further when the poses moved it away.
    const std::array<std::pair<uint64_t, double>, 2u> expected{{{10u, 4.5 + shift}, {20u, 9.0 + shift}}};
    for (size_t axis = 0u; axis < expected.size(); ++axis)
    {
        ovspatialquery_raycast_desc_t ray{};
        ray.common.target            = populationTarget(population);
        ray.common.precision         = OVSPATIALQUERY_QUERY_EXACT;
        ray.common.query_bits        = participationBits(UINT64_MAX, UINT64_MAX);
        ray.ray.unit_direction[axis] = 1.0f;
        ray.ray.max_distance         = 20.0;
        ovspatialquery_raycast_single_result_t result{};
        REQUIRE(ovspatialquery_raycast_closest_sync(instance, &ray, &result).status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(result.source_ordinal == ordinal && result.has_hit == 1u &&
                result.hit.data.exact.identity.object_id.value == expected[axis].first &&
                std::abs(result.hit.data.exact.distance - expected[axis].second) < 1.0e-12);
    }
}

void destroySpheres(ovspatialquery_instance_handle_t instance, const uint64_t (&values)[2])
{
    for (const uint64_t value : values)
    {
        REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_SPHERE, value}).status ==
                OVSPATIALQUERY_API_SUCCESS);
    }
}

bool lastErrorNames(const char* column)
{
    const ovspatialquery_string_t error = ovspatialquery_get_last_error();
    return error.ptr != nullptr && std::string(error.ptr, error.length).find(column) != std::string::npos;
}

// Every nonnegative layout a DLPack producer hands over is read as the rows it
// describes (ADR-0053): radii and object values at every other element, NULL
// strides, translations sliced from a pose array, column-major rotations, and one
// scale repeated by a zero stride. A negative stride is refused and named.
void testExplicitRowsReadEveryNonnegativeLayout()
{
    const ovspatialquery_instance_desc_t instanceDesc{4u, 1u, UINT64_C(16) * 1024u * 1024u, nullptr, 0u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&instanceDesc, &instance).status == OVSPATIALQUERY_API_SUCCESS);

    uint64_t   definitionValues[2] = {31u, 32u};
    double     radiusElements[4]   = {1.0, 9.0, 2.0, 9.0};
    InputArray values(definitionValues, 2, kDLUInt);
    InputArray radii(radiusElements, 2, kDLFloat);
    radii.strides = {2, 0};
    const ovspatialquery_sphere_definition_table_t spheres{&values.value, &radii.value};
    radii.strides[0] = -2;
    requireRejected(ovspatialquery_enqueue_sphere_registration(instance, &spheres),
                    OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(lastErrorNames("radii"));
    radii.strides[0] = 2;
    waitAndDestroy(instance, ovspatialquery_enqueue_sphere_registration(instance, &spheres));

    // Convex registration reads every required column before checking any
    // status; the refused first column keeps the diagnostic.
    uint64_t   convexValues[2]  = {41u, 42u};
    uint64_t   vertexOffsets[2] = {0u, 0u};
    uint64_t   vertexCounts[2]  = {4u, 4u};
    float      vertices[12]     = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    InputArray convexIds(convexValues, 2, kDLUInt);
    InputArray offsets(vertexOffsets, 2, kDLUInt);
    InputArray counts(vertexCounts, 2, kDLUInt);
    InputArray points(vertices, 4, 3, kDLFloat);
    convexIds.strides[0] = -1;
    const ovspatialquery_convex_definition_table_t convexes{
        &convexIds.value, &offsets.value, &counts.value, nullptr, nullptr, &points.value,
        nullptr,          nullptr,        nullptr,       0.0,     0u};
    requireRejected(ovspatialquery_enqueue_convex_registration(instance, &convexes),
                    OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(lastErrorNames("definition_values"));

    uint64_t   objectElements[4] = {10u, 0u, 20u, 0u};
    int32_t    types[2]          = {OVSPATIALQUERY_GEOMETRY_SPHERE, OVSPATIALQUERY_GEOMETRY_SPHERE};
    double     pose[14]          = {5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0};
    double     scale             = 0.5;
    InputArray objects(objectElements, 2, kDLUInt);
    InputArray geometryTypes(types, 2, kDLInt);
    InputArray geometryValues(definitionValues, 2, kDLUInt);
    // Rotations in column-major order: each component's two rows are adjacent.
    double     columnMajorRotations[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0};
    InputArray translations(pose, 2, 3, kDLFloat);
    InputArray rotations(columnMajorRotations, 2, 4, kDLFloat);
    InputArray scales(&scale, 2, kDLFloat);
    objects.strides                     = {2, 0};
    geometryTypes.value.tensor.strides  = nullptr;
    geometryValues.value.tensor.strides = nullptr;
    translations.strides                = {7, 1};
    rotations.strides                   = {1, 2};
    scales.strides                      = {0, 0};
    // A borrowed column the library cannot read in place is copied all the same.
    translations.value.ownership = OVSPATIALQUERY_INPUT_BORROWED_UNTIL_TERMINAL;
    const ovspatialquery_explicit_population_set_t set{1,
                                                       {&objects.value, &geometryTypes.value, &geometryValues.value,
                                                        &translations.value, &rotations.value, &scales.value, nullptr,
                                                        nullptr},
                                                       nullptr,
                                                       0u};
    const ovspatialquery_population_handle_t       population = createExplicitPopulation(instance, 2u);
    translations.strides[0]                                   = -7;
    requireRejected(ovspatialquery_enqueue_explicit_population_set(instance, population, &set),
                    OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(lastErrorNames("instances.translations"));
    translations.strides[0] = 7;
    // A stride whose last element lies exactly UINT64_MAX elements in, so a
    // sum of element offsets would wrap to a small address.
    rotations.strides = {1, INT64_C(6148914691236517205)};
    requireRejected(ovspatialquery_enqueue_explicit_population_set(instance, population, &set),
                    OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(lastErrorNames("instances.rotations_xyzw"));
    rotations.strides = {1, 2};
    waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set));
    requireStridedSpheres(instance, population, 1);

    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    destroySpheres(instance, definitionValues);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

#if defined(OVSPATIALQUERY_TEST_HAS_CUDA_BACKEND)
// The same layouts from CUDA columns, prepared on the host and on the device:
// the library copies each non-compact column on the producer's stream before
// either preparation reads it (ADR-0053).
void testCudaRowsReadEveryNonnegativeLayout()
{
    CudaDriver                           cuda;
    const ovspatialquery_device_budget_t deviceBudget{{kDLCUDA, 0}, UINT64_C(64) * 1024u * 1024u};
    const ovspatialquery_instance_desc_t instanceDesc{4u, 1u, UINT64_C(16) * 1024u * 1024u, &deviceBudget, 1u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&instanceDesc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    uint64_t                                       definitionValues[2] = {31u, 32u};
    double                                         radiusValues[2]     = {1.0, 2.0};
    InputArray                                     values(definitionValues, 2, kDLUInt);
    InputArray                                     radii(radiusValues, 2, kDLFloat);
    const ovspatialquery_sphere_definition_table_t spheres{&values.value, &radii.value};
    waitAndDestroy(instance, ovspatialquery_enqueue_sphere_registration(instance, &spheres));

    const uint64_t           objectElements[4] = {10u, 0u, 20u, 0u};
    const int32_t            types[2]          = {OVSPATIALQUERY_GEOMETRY_SPHERE, OVSPATIALQUERY_GEOMETRY_SPHERE};
    const double             pose[14]        = {5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0};
    const double             movedPose[14]   = {6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 11.0, 0.0, 0.0, 0.0, 0.0, 1.0};
    const double             scale           = 0.5;
    uint64_t                 removedSentinel = 0u;
    InputArray               removed(&removedSentinel, 0, kDLUInt, {kDLCUDA, 0});
    const CudaDriver::Stream stream               = cuda.createStream();
    const CudaDriver::Event  ready                = cuda.createEvent();
    const CudaDriver::DevicePointer deviceObjects = cuda.allocate(sizeof(objectElements));
    const CudaDriver::DevicePointer deviceTypes   = cuda.allocate(sizeof(types));
    const CudaDriver::DevicePointer deviceValues  = cuda.allocate(sizeof(definitionValues));
    const CudaDriver::DevicePointer devicePose    = cuda.allocate(sizeof(pose));
    const CudaDriver::DevicePointer deviceScale   = cuda.allocate(sizeof(scale));
    cuda.uploadAsync(deviceObjects, objectElements, sizeof(objectElements), stream);
    cuda.uploadAsync(deviceTypes, types, sizeof(types), stream);
    cuda.uploadAsync(deviceValues, definitionValues, sizeof(definitionValues), stream);
    cuda.uploadAsync(devicePose, pose, sizeof(pose), stream);
    cuda.uploadAsync(deviceScale, &scale, sizeof(scale), stream);
    cuda.record(ready, stream);
    const auto onDevice = [](CudaDriver::DevicePointer pointer)
    { return reinterpret_cast<double*>(static_cast<uintptr_t>(pointer)); };
    InputArray objects(reinterpret_cast<uint64_t*>(onDevice(deviceObjects)), 2, kDLUInt, {kDLCUDA, 0});
    InputArray geometryTypes(reinterpret_cast<int32_t*>(onDevice(deviceTypes)), 2, kDLInt, {kDLCUDA, 0});
    InputArray geometryValues(reinterpret_cast<uint64_t*>(onDevice(deviceValues)), 2, kDLUInt, {kDLCUDA, 0});
    InputArray translations(onDevice(devicePose), 2, 3, kDLFloat, {kDLCUDA, 0});
    InputArray rotations(onDevice(devicePose), 2, 4, kDLFloat, {kDLCUDA, 0});
    InputArray scales(onDevice(deviceScale), 2, kDLFloat, {kDLCUDA, 0});
    rotations.value.tensor.byte_offset  = 3u * sizeof(double);
    objects.strides                     = {2, 0};
    geometryTypes.value.tensor.strides  = nullptr;
    geometryValues.value.tensor.strides = nullptr;
    translations.strides                = {7, 1};
    rotations.strides                   = {7, 1};
    scales.strides                      = {0, 0};
    for (InputArray* input : {&objects, &geometryTypes, &geometryValues, &translations, &rotations, &scales})
    {
        input->value.ownership = OVSPATIALQUERY_INPUT_BORROWED_UNTIL_TERMINAL;
    }
    for (const ovspatialquery_preparation_t preparation :
         {OVSPATIALQUERY_PREPARE_ON_HOST, OVSPATIALQUERY_PREPARE_ON_DEVICE_HOST_READABLE})
    {
        const ovspatialquery_dynamic_execution_t execution{
            {kDLCUDA, 0}, {1u, reinterpret_cast<uintptr_t>(stream), reinterpret_cast<uintptr_t>(ready)}, preparation};
        const ovspatialquery_explicit_population_set_t set{1,
                                                           {&objects.value, &geometryTypes.value, &geometryValues.value,
                                                            &translations.value, &rotations.value, &scales.value,
                                                            nullptr, nullptr},
                                                           &execution,
                                                           preparation == OVSPATIALQUERY_PREPARE_ON_HOST ? 0u : 2u};
        const ovspatialquery_population_handle_t       population = createExplicitPopulation(instance, 2u);
        translations.strides[0]                                   = -7;
        requireRejected(ovspatialquery_enqueue_explicit_population_set(instance, population, &set),
                        OVSPATIALQUERY_API_INVALID_ARGUMENT);
        REQUIRE(lastErrorNames("instances.translations"));
        translations.strides[0] = 7;
        waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set));
        requireStridedSpheres(instance, population, 1);

        // A POSES update from the same strided columns, with both spheres moved
        // one unit away, copies into the staging buffers the set gave back; the
        // rays must see the new poses, not the set's.
        cuda.uploadAsync(devicePose, movedPose, sizeof(movedPose), stream);
        cuda.record(ready, stream);
        const ovspatialquery_explicit_population_update_t update{
            1,
            2,
            OVSPATIALQUERY_EXPLICIT_UPDATE_POSES,
            {&objects.value, nullptr, nullptr, &translations.value, &rotations.value, &scales.value, nullptr, nullptr},
            &removed.value,
            &execution};
        waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_update(instance, population, &update));
        requireStridedSpheres(instance, population, 2, 1.0);
        cuda.uploadAsync(devicePose, pose, sizeof(pose), stream);
        cuda.record(ready, stream);
        REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    }

    destroySpheres(instance, definitionValues);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
    for (CudaDriver::DevicePointer allocation : {deviceObjects, deviceTypes, deviceValues, devicePose, deviceScale})
    {
        cuda.free(allocation);
    }
    cuda.destroyEvent(ready);
    cuda.destroyStream(stream);
}

// A device-prepared POSES update returns without waiting for its producer, and
// the caller may overwrite a copied input on `cuda_sync.stream` as soon as
// enqueue returns: the library queues its copy there first. The producer's
// event is held on another stream, so the copy cannot have run when the
// caller overwrites the translation.
void testCudaCopiedPosesAreCopiedOnTheCallerStream()
{
    CudaDriver                           cuda(CudaDriver::Ownership::tracked);
    const ovspatialquery_device_budget_t deviceBudget{{kDLCUDA, 0}, UINT64_C(64) * 1024u * 1024u};
    const ovspatialquery_instance_desc_t instanceDesc{4u, 1u, UINT64_C(16) * 1024u * 1024u, &deviceBudget, 1u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&instanceDesc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    uint64_t sphereValue = 41u;
    registerSphere(instance, sphereValue, 1.0);

    const CudaDriver::Stream stream   = cuda.createStream();
    const CudaDriver::Stream producer = cuda.createStream();
    const CudaDriver::Event  ready    = cuda.createEvent();
    const auto               upload   = [&cuda, stream](const auto& value)
    {
        const CudaDriver::DevicePointer pointer = cuda.allocate(sizeof(value));
        cuda.uploadAsync(pointer, &value, sizeof(value), stream);
        return pointer;
    };
    const auto onDevice = [](CudaDriver::DevicePointer pointer)
    { return reinterpret_cast<void*>(static_cast<uintptr_t>(pointer)); };
    const uint64_t                  objectValue       = 410u;
    const int32_t                   geometryType      = OVSPATIALQUERY_GEOMETRY_SPHERE;
    double                          translation[3]    = {3.0, 0.0, 0.0};
    const float                     rotation[4]       = {0.0f, 0.0f, 0.0f, 1.0f};
    const CudaDriver::DevicePointer deviceTranslation = upload(translation);
    InputArray objects(static_cast<uint64_t*>(onDevice(upload(objectValue))), 1, kDLUInt, {kDLCUDA, 0});
    InputArray types(static_cast<int32_t*>(onDevice(upload(geometryType))), 1, kDLInt, {kDLCUDA, 0});
    InputArray geometry(static_cast<uint64_t*>(onDevice(upload(sphereValue))), 1, kDLUInt, {kDLCUDA, 0});
    InputArray positions(static_cast<double*>(onDevice(deviceTranslation)), 1, 3, kDLFloat, {kDLCUDA, 0});
    InputArray rotations(static_cast<float*>(onDevice(upload(rotation))), 1, 4, kDLFloat, {kDLCUDA, 0});
    cuda.record(ready, stream);
    ovspatialquery_dynamic_execution_t execution{
        {kDLCUDA, 0},
        {1u, reinterpret_cast<uintptr_t>(stream), reinterpret_cast<uintptr_t>(ready)},
        OVSPATIALQUERY_PREPARE_ON_DEVICE_HOST_READABLE};
    const ovspatialquery_population_handle_t       population = createExplicitPopulation(instance, 2u);
    const ovspatialquery_explicit_population_set_t set{
        1,
        {&objects.value, &types.value, &geometry.value, &positions.value, &rotations.value, nullptr, nullptr, nullptr},
        &execution,
        1u};
    waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set));

    // Move the sphere to x = 5, then hold the producer's event behind a host
    // function; a watchdog releases it after ten seconds so a regression fails
    // instead of hanging.
    translation[0] = 5.0;
    cuda.uploadAsync(deviceTranslation, translation, sizeof(translation), stream);
    cuda.record(ready, stream);
    cuda.wait(producer, ready);
    std::promise<void> release, checked;
    std::future<void>  released = release.get_future();
    cuda.enqueueHostFunction(
        producer, [](void* state) { static_cast<std::future<void>*>(state)->wait(); }, &released);
    const CudaDriver::Event held = cuda.createEvent();
    cuda.record(held, producer);
    execution.cuda_sync.wait_event = reinterpret_cast<uintptr_t>(held);
    std::thread watchdog(
        [&release, done = checked.get_future()]
        {
            done.wait_for(std::chrono::seconds(10));
            release.set_value();
        });
    uint64_t                                          removedSentinel = 0u;
    InputArray                                        removed(&removedSentinel, 0, kDLUInt, {kDLCUDA, 0});
    const ovspatialquery_explicit_population_update_t update{
        1,
        2,
        OVSPATIALQUERY_EXPLICIT_UPDATE_POSES,
        {&objects.value, nullptr, nullptr, &positions.value, &rotations.value, nullptr, nullptr, nullptr},
        &removed.value,
        &execution};
    const ovspatialquery_enqueue_result_t admission =
        ovspatialquery_enqueue_explicit_population_update(instance, population, &update);
    const double overwrite[3] = {100.0, 0.0, 0.0};
    cuda.uploadAsync(deviceTranslation, overwrite, sizeof(overwrite), stream);
    const CudaDriver::Event overwritten = cuda.createEvent();
    cuda.record(overwritten, stream);
    // Release the producer once the overwrite has run, or after half a second
    // in which it waited behind the library's copy.
    const std::chrono::steady_clock::time_point deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while (!cuda.isComplete(overwritten) && std::chrono::steady_clock::now() < deadline)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    checked.set_value();
    watchdog.join();
    waitOnly(instance, admission);

    // The sphere of radius 1 at x = 5 is hit at 4; the overwrite would put it at 99.
    ovspatialquery_raycast_desc_t ray{};
    ray.common.target         = populationTarget(population);
    ray.common.precision      = OVSPATIALQUERY_QUERY_EXACT;
    ray.common.query_bits     = participationBits(UINT64_MAX, UINT64_MAX);
    ray.ray.unit_direction[0] = 1.0f;
    ray.ray.max_distance      = 200.0;
    ovspatialquery_raycast_single_result_t result{};
    REQUIRE(ovspatialquery_raycast_closest_sync(instance, &ray, &result).status == OVSPATIALQUERY_API_SUCCESS);
    if (result.has_hit == 1u && std::abs(result.hit.data.exact.distance - 4.0) >= 1.0e-12)
    {
        std::fprintf(stderr, "copied pose hit at %.17g, expected 4\n", result.hit.data.exact.distance);
    }
    REQUIRE(result.source_ordinal == 2 && result.has_hit == 1u &&
            std::abs(result.hit.data.exact.distance - 4.0) < 1.0e-12);

    ovspatialquery_op_id_t updateId = admission.op_id;
    REQUIRE(ovspatialquery_destroy_op(instance, &updateId).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(
        ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_SPHERE, sphereValue}).status ==
        OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}
#endif

int main()
{
    testRegistrationCapacityAndSchedulingRollback();
    testConvexRegistrationAcceptsPointsOnly();
    testConvexRegistrationExactHostBudgetBoundary();
    testAdmittedPublicationCompletesWithinHostBudget();
    testLargeMeshAnswersExactReadersUnderTheBudgetItRegisteredUnder();
    testCpuStatusAndPublicationAtomicity();
    testParticipationChangesRemainVisibleThroughQueries();
    testDefinitionAndObjectValuesAreNeverReused();
    testTriangleMeshAccountingReleasesPreparedStorage();
    testPosesUpdateKeepsBindingsOnTheHost();
    testAdmittedSelectionIsRejectedOnTheHostPath();
    testPrimitiveSweepsReportPenetratingContacts();
    testExplicitRowsReadEveryNonnegativeLayout();
#if defined(OVSPATIALQUERY_TEST_HAS_CUDA_BACKEND)
    testCudaRowsReadEveryNonnegativeLayout();
    testCudaCopiedPosesAreCopiedOnTheCallerStream();
    testCudaCopiedAndBorrowedPopulationCapture<float>();
    testCudaCopiedAndBorrowedPopulationCapture<double>();
#endif
    const ovspatialquery_instance_desc_t instanceDesc{4u, 1u, UINT64_C(16) * 1024u * 1024u, nullptr, 0u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&instanceDesc, &instance).status == OVSPATIALQUERY_API_SUCCESS);

    uint64_t definitionValue = 7u;
    registerSphere(instance, definitionValue, 2.0);

    uint64_t                                    boxValue       = 8u;
    double                                      halfExtents[3] = {1.0, 2.0, 3.0};
    InputArray                                  boxValues(&boxValue, 1, kDLUInt);
    InputArray                                  boxExtents(halfExtents, 1, 3, kDLFloat);
    const ovspatialquery_box_definition_table_t boxes{&boxValues.value, &boxExtents.value};
    waitAndDestroy(instance, ovspatialquery_enqueue_box_registration(instance, &boxes));

    uint64_t                                        capsuleValue  = 9u;
    double                                          capsuleRadius = 0.5;
    double                                          halfSegment   = 2.0;
    InputArray                                      capsuleValues(&capsuleValue, 1, kDLUInt);
    InputArray                                      capsuleRadii(&capsuleRadius, 1, kDLFloat);
    InputArray                                      halfSegments(&halfSegment, 1, kDLFloat);
    const ovspatialquery_capsule_definition_table_t capsules{&capsuleValues.value, &capsuleRadii.value,
                                                             &halfSegments.value};
    waitAndDestroy(instance, ovspatialquery_enqueue_capsule_registration(instance, &capsules));

    uint64_t   convexValue           = 10u;
    uint64_t   convexVertexOffset    = 0u;
    uint64_t   convexVertexCount     = 4u;
    uint64_t   convexFaceOffset      = 0u;
    uint64_t   convexFaceCount       = 4u;
    float      convexVertices[12]    = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    uint64_t   faceVertexOffsets[4]  = {0u, 3u, 6u, 9u};
    uint32_t   faceVertexCounts[4]   = {3u, 3u, 3u, 3u};
    uint32_t   faceVertexIndices[12] = {0u, 2u, 1u, 0u, 1u, 3u, 0u, 3u, 2u, 1u, 2u, 3u};
    InputArray convexValues(&convexValue, 1, kDLUInt);
    InputArray convexVertexOffsets(&convexVertexOffset, 1, kDLUInt);
    InputArray convexVertexCounts(&convexVertexCount, 1, kDLUInt);
    InputArray convexFaceOffsets(&convexFaceOffset, 1, kDLUInt);
    InputArray convexFaceCounts(&convexFaceCount, 1, kDLUInt);
    InputArray convexVertexData(convexVertices, 4, 3, kDLFloat);
    InputArray convexLoopOffsets(faceVertexOffsets, 4, kDLUInt);
    InputArray convexLoopCounts(faceVertexCounts, 4, kDLUInt);
    InputArray convexLoopIndices(faceVertexIndices, 12, kDLUInt);
    const ovspatialquery_convex_definition_table_t convexes{&convexValues.value,
                                                            &convexVertexOffsets.value,
                                                            &convexVertexCounts.value,
                                                            &convexFaceOffsets.value,
                                                            &convexFaceCounts.value,
                                                            &convexVertexData.value,
                                                            &convexLoopOffsets.value,
                                                            &convexLoopCounts.value,
                                                            &convexLoopIndices.value,
                                                            0.0,
                                                            0u};
    waitAndDestroy(instance, ovspatialquery_enqueue_convex_registration(instance, &convexes));

    uint64_t   meshValue          = 11u;
    uint64_t   meshVertexOffset   = 0u;
    uint64_t   meshVertexCount    = 3u;
    uint64_t   triangleOffset     = 0u;
    uint64_t   triangleCount      = 1u;
    float      meshVertices[9]    = {0.0, -1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0};
    uint32_t   triangleIndices[3] = {0u, 1u, 2u};
    uint8_t    doubleSided        = 1u;
    InputArray meshValues(&meshValue, 1, kDLUInt);
    InputArray meshVertexOffsets(&meshVertexOffset, 1, kDLUInt);
    InputArray meshVertexCounts(&meshVertexCount, 1, kDLUInt);
    InputArray triangleOffsets(&triangleOffset, 1, kDLUInt);
    InputArray triangleCounts(&triangleCount, 1, kDLUInt);
    InputArray meshVertexData(meshVertices, 3, 3, kDLFloat);
    InputArray meshTriangleData(triangleIndices, 1, 3, kDLUInt);
    InputArray meshDoubleSided(&doubleSided, 1, kDLUInt);
    const ovspatialquery_triangle_mesh_definition_table_t meshes{&meshValues.value,
                                                                 &meshVertexOffsets.value,
                                                                 &meshVertexCounts.value,
                                                                 &triangleOffsets.value,
                                                                 &triangleCounts.value,
                                                                 &meshVertexData.value,
                                                                 &meshTriangleData.value,
                                                                 &meshDoubleSided.value,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr};
    waitAndDestroy(instance, ovspatialquery_enqueue_triangle_mesh_registration(instance, &meshes));

    const ovspatialquery_population_handle_t population = createExplicitPopulation(instance, 2u);

    uint64_t   objectValues[5]   = {101u, 102u, 103u, 104u, 105u};
    int32_t    geometryTypes[5]  = {OVSPATIALQUERY_GEOMETRY_SPHERE, OVSPATIALQUERY_GEOMETRY_BOX,
                                    OVSPATIALQUERY_GEOMETRY_CAPSULE, OVSPATIALQUERY_GEOMETRY_CONVEX,
                                    OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH};
    uint64_t   geometryValues[5] = {definitionValue, boxValue, capsuleValue, convexValue, meshValue};
    double     translations[15]  = {0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 20.0, 0.0, 0.0, 30.0, 0.0, 0.0, 40.0, 0.0, 0.0};
    float      rotations[20]     = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                                    0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    double     scales[5]         = {1.0, 1.0, 1.0, 1.0, 1.0};
    uint64_t   bits[10]          = {1u, 0u, 1u, 0u, 1u, 0u, 1u, 0u, 1u, 0u};
    InputArray objects(objectValues, 5, kDLUInt);
    InputArray types(geometryTypes, 5, kDLInt);
    InputArray geometryIds(geometryValues, 5, kDLUInt);
    InputArray positions(translations, 5, 3, kDLFloat);
    InputArray orientations(rotations, 5, 4, kDLFloat);
    InputArray uniformScales(scales, 5, kDLFloat);
    InputArray participation(bits, 5, 2, kDLUInt);
    const ovspatialquery_explicit_population_set_t set{10,
                                                       {&objects.value, &types.value, &geometryIds.value,
                                                        &positions.value, &orientations.value, &uniformScales.value,
                                                        &participation.value, nullptr},
                                                       nullptr,
                                                       0u};
    const ovspatialquery_enqueue_result_t          setAdmission =
        ovspatialquery_enqueue_explicit_population_set(instance, population, &set);
    waitOnly(instance, setAdmission);
    ovspatialquery_population_update_result_t setResult{};
    REQUIRE(ovspatialquery_fetch_population_update(instance, setAdmission.op_id, &setResult).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(setResult.published_ordinal == 10 && setResult.object_count == 5u && setResult.excluded_object_count == 0u);
    ovspatialquery_raycast_desc_t ray{};
    ray.common.target         = populationTarget(population);
    ray.common.precision      = OVSPATIALQUERY_QUERY_EXACT;
    ray.common.query_bits     = participationBits(UINT64_MAX, UINT64_MAX);
    ray.ray.origin[0]         = 35.0;
    ray.ray.unit_direction[0] = 1.0f;
    ray.ray.max_distance      = 10.0;
    ovspatialquery_raycast_single_result_t rayResult{};
    REQUIRE(ovspatialquery_raycast_closest_sync(instance, &ray, &rayResult).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(rayResult.source_ordinal == 10 && rayResult.has_hit == 1u);
    REQUIRE(rayResult.hit.truth == OVSPATIALQUERY_HIT_EXACT_SURFACE);
    REQUIRE(rayResult.hit.data.exact.identity.object_id.value == 105u);
    REQUIRE(rayResult.hit.data.exact.identity.geometry.type == OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH);
    REQUIRE(rayResult.hit.data.exact.identity.geometry.value == meshValue);
    REQUIRE(rayResult.hit.data.exact.distance == 5.0);
    REQUIRE(rayResult.hit.data.exact.feature.kind == OVSPATIALQUERY_GEOMETRY_FEATURE_TRIANGLE);
    REQUIRE(rayResult.hit.data.exact.feature.data.triangle.source_face_index == 0u);
    REQUIRE(rayResult.hit.data.exact.feature.data.triangle.source_face_corner_indices[0] == 0u &&
            rayResult.hit.data.exact.feature.data.triangle.source_face_corner_indices[1] == 1u &&
            rayResult.hit.data.exact.feature.data.triangle.source_face_corner_indices[2] == 2u);

    ray.ray.origin[0] = 25.0;
    ray.ray.origin[1] = 0.2;
    ray.ray.origin[2] = 0.2;
    REQUIRE(ovspatialquery_raycast_closest_sync(instance, &ray, &rayResult).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(rayResult.has_hit == 1u);
    REQUIRE(rayResult.hit.data.exact.identity.object_id.value == 104u);
    REQUIRE(rayResult.hit.data.exact.identity.geometry.type == OVSPATIALQUERY_GEOMETRY_CONVEX);
    REQUIRE(rayResult.hit.data.exact.feature.kind == OVSPATIALQUERY_GEOMETRY_FEATURE_NONE);

    ovspatialquery_sphere_sweep_desc_t sweep{};
    sweep.common                  = ray.common;
    sweep.sweep.center[0]         = 25.0;
    sweep.sweep.center[1]         = 0.2;
    sweep.sweep.center[2]         = 0.2;
    sweep.sweep.radius            = 0.1;
    sweep.sweep.unit_direction[0] = 1.0f;
    sweep.sweep.max_distance      = 10.0;
    ovspatialquery_sphere_sweep_closest_result_t sweepResult{};
    REQUIRE(ovspatialquery_sweep_sphere_closest_sync(instance, &sweep, &sweepResult).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(sweepResult.has_hit == 1u);
    REQUIRE(sweepResult.hit.truth == OVSPATIALQUERY_HIT_EXACT_SURFACE);
    REQUIRE(sweepResult.hit.data.exact.identity.object_id.value == 104u);
    REQUIRE(sweepResult.hit.data.exact.feature.kind == OVSPATIALQUERY_GEOMETRY_FEATURE_NONE);

    ovspatialquery_overlap_desc_t overlap{};
    overlap.common                      = ray.common;
    overlap.shape.kind                  = OVSPATIALQUERY_OVERLAP_SPHERE;
    overlap.shape.data.sphere.center[0] = 30.2;
    overlap.shape.data.sphere.center[1] = 0.2;
    overlap.shape.data.sphere.center[2] = 0.2;
    overlap.shape.data.sphere.radius    = 0.05;
    ovspatialquery_overlap_any_result_t overlapResult{};
    REQUIRE(ovspatialquery_overlap_any_sync(instance, &overlap, &overlapResult).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(overlapResult.has_hit == 1u);
    REQUIRE(overlapResult.hit.truth == OVSPATIALQUERY_HIT_EXACT_SURFACE);
    REQUIRE(overlapResult.hit.identity.object_id.value == 104u);
    REQUIRE(overlapResult.hit.identity.geometry.type == OVSPATIALQUERY_GEOMETRY_CONVEX);

    // Object 104's hull is the unit tetrahedron at world offset (30,0,0); its face
    // {A, C, B} lies in the world z=0 plane. A sphere at (30.3, 0.3, -0.2) with
    // radius 0.21 reaches the face's interior point (30.3, 0.3, 0) at distance 0.2,
    // so the exact overlap hits; a closest point misplaced along AB to (30.4, 0.3, 0)
    // would be 0.2236 away and miss it.
    ovspatialquery_overlap_desc_t interiorFaceOverlap{};
    interiorFaceOverlap.common                      = ray.common;
    interiorFaceOverlap.shape.kind                  = OVSPATIALQUERY_OVERLAP_SPHERE;
    interiorFaceOverlap.shape.data.sphere.center[0] = 30.3;
    interiorFaceOverlap.shape.data.sphere.center[1] = 0.3;
    interiorFaceOverlap.shape.data.sphere.center[2] = -0.2;
    interiorFaceOverlap.shape.data.sphere.radius    = 0.21;
    ovspatialquery_overlap_any_result_t interiorFaceOverlapResult{};
    REQUIRE(ovspatialquery_overlap_any_sync(instance, &interiorFaceOverlap, &interiorFaceOverlapResult).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(interiorFaceOverlapResult.has_hit == 1u);
    REQUIRE(interiorFaceOverlapResult.hit.truth == OVSPATIALQUERY_HIT_EXACT_SURFACE);
    REQUIRE(interiorFaceOverlapResult.hit.identity.object_id.value == 104u);
    REQUIRE(interiorFaceOverlapResult.hit.identity.geometry.type == OVSPATIALQUERY_GEOMETRY_CONVEX);

    ovspatialquery_op_id_t setId = setAdmission.op_id;
    REQUIRE(ovspatialquery_destroy_op(instance, &setId).status == OVSPATIALQUERY_API_SUCCESS);

    uint64_t                                          changedObject         = 101u;
    int32_t                                           changedType           = OVSPATIALQUERY_GEOMETRY_SPHERE;
    uint64_t                                          changedGeometry       = definitionValue;
    double                                            changedTranslation[3] = {2.0, 0.0, 0.0};
    float                                             changedRotation[4]    = {0.0f, 0.0f, 0.0f, 1.0f};
    uint64_t                                          removedObject         = 105u;
    InputArray                                        changedObjects(&changedObject, 1, kDLUInt);
    InputArray                                        changedTypes(&changedType, 1, kDLInt);
    InputArray                                        changedGeometryIds(&changedGeometry, 1, kDLUInt);
    InputArray                                        changedPositions(changedTranslation, 1, 3, kDLFloat);
    InputArray                                        changedOrientations(changedRotation, 1, 4, kDLFloat);
    InputArray                                        removedObjects(&removedObject, 1, kDLUInt);
    const ovspatialquery_explicit_population_update_t update{10,
                                                             20,
                                                             OVSPATIALQUERY_EXPLICIT_UPDATE_COMPLETE_ROWS,
                                                             {&changedObjects.value, &changedTypes.value,
                                                              &changedGeometryIds.value, &changedPositions.value,
                                                              &changedOrientations.value, nullptr, nullptr, nullptr},
                                                             &removedObjects.value,
                                                             nullptr};
    REQUIRE(ovspatialquery::detail::setInstanceOperationWorkersPausedForTest(instance, true) ==
            OVSPATIALQUERY_API_SUCCESS);
    const ovspatialquery_enqueue_result_t updateAdmission =
        ovspatialquery_enqueue_explicit_population_update(instance, population, &update);
    std::atomic<bool>     stopReaders{false};
    std::atomic<bool>     readersPassed{true};
    std::atomic<uint32_t> headReads{0u};
    std::atomic<uint32_t> retainedReads{0u};
    const auto readConvex = [&](ovspatialquery_ordinal_selection_kind_t ordinalKind, ovspatialquery_ordinal_t ordinal,
                                std::atomic<uint32_t>& readCount)
    {
        ovspatialquery_raycast_desc_t concurrentRay        = ray;
        concurrentRay.common.target.ordinal.kind           = ordinalKind;
        concurrentRay.common.target.ordinal.source_ordinal = ordinal;
        while (!stopReaders.load(std::memory_order_acquire))
        {
            ovspatialquery_raycast_single_result_t concurrentResult{};
            const ovspatialquery_result_t          status =
                ovspatialquery_raycast_closest_sync(instance, &concurrentRay, &concurrentResult);
            const bool expectedOrdinal =
                ordinalKind == OVSPATIALQUERY_ORDINAL_HEAD
                    ? (concurrentResult.source_ordinal == 10 || concurrentResult.source_ordinal == 20)
                    : concurrentResult.source_ordinal == 10;
            if (status.status != OVSPATIALQUERY_API_SUCCESS || concurrentResult.has_hit == 0u || !expectedOrdinal ||
                concurrentResult.hit.truth != OVSPATIALQUERY_HIT_EXACT_SURFACE ||
                concurrentResult.hit.data.exact.identity.object_id.value != 104u)
            {
                readersPassed.store(false, std::memory_order_release);
                break;
            }
            readCount.fetch_add(1u, std::memory_order_relaxed);
        }
    };
    std::thread headReader(readConvex, OVSPATIALQUERY_ORDINAL_HEAD, 0, std::ref(headReads));
    std::thread retainedReader(readConvex, OVSPATIALQUERY_ORDINAL_RETAINED_EXACT, 10, std::ref(retainedReads));
    while (readersPassed.load(std::memory_order_acquire) &&
           (headReads.load(std::memory_order_relaxed) < 32u || retainedReads.load(std::memory_order_relaxed) < 32u))
    {
        std::this_thread::yield();
    }
    REQUIRE(ovspatialquery::detail::setInstanceOperationWorkersPausedForTest(instance, false) ==
            OVSPATIALQUERY_API_SUCCESS);
    waitOnly(instance, updateAdmission);
    stopReaders.store(true, std::memory_order_release);
    headReader.join();
    retainedReader.join();
    REQUIRE(readersPassed.load(std::memory_order_acquire));
    REQUIRE(headReads.load(std::memory_order_relaxed) >= 32u);
    REQUIRE(retainedReads.load(std::memory_order_relaxed) >= 32u);
    ovspatialquery_population_update_result_t updateResult{};
    REQUIRE(ovspatialquery_fetch_population_update(instance, updateAdmission.op_id, &updateResult).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(updateResult.published_ordinal == 20 && updateResult.object_count == 4u &&
            updateResult.excluded_object_count == 0u);
    ray.common.target.ordinal.kind           = OVSPATIALQUERY_ORDINAL_RETAINED_EXACT;
    ray.common.target.ordinal.source_ordinal = 10;
    REQUIRE(ovspatialquery_raycast_closest_sync(instance, &ray, &rayResult).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(rayResult.source_ordinal == 10 && rayResult.has_hit == 1u);
    REQUIRE(rayResult.hit.data.exact.identity.object_id.value == 104u);
    ovspatialquery_op_id_t updateId = updateAdmission.op_id;
    REQUIRE(ovspatialquery_destroy_op(instance, &updateId).status == OVSPATIALQUERY_API_SUCCESS);

    ovspatialquery_population_info_t info{};
    REQUIRE(ovspatialquery_get_population_info(instance, population, &info).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(info.ready != 0u && info.head_ordinal == 20 && info.object_count == 4u &&
            info.retained_ordinal_count == 1u);
    REQUIRE(
        ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_CONVEX, convexValue}).status ==
        OVSPATIALQUERY_API_BUSY);

    uint64_t                                  viewIds[2] = {101u, 105u};
    InputArray                                viewIdInput(viewIds, 2, kDLUInt);
    const ovspatialquery_explicit_view_desc_t viewDesc{&viewIdInput.value};
    ovspatialquery_view_handle_t              view = OVSPATIALQUERY_INVALID_VIEW;
    REQUIRE(ovspatialquery_create_explicit_view(instance, population, &viewDesc, &view).status ==
            OVSPATIALQUERY_API_SUCCESS);

    REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_SPHERE, definitionValue})
                .status == OVSPATIALQUERY_API_BUSY);
    REQUIRE(ovspatialquery_destroy_view(instance, view).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);

    const std::array<ovspatialquery_geometry_ref_t, 5u> references{
        ovspatialquery_geometry_ref_t{OVSPATIALQUERY_GEOMETRY_SPHERE, definitionValue},
        ovspatialquery_geometry_ref_t{OVSPATIALQUERY_GEOMETRY_BOX, boxValue},
        ovspatialquery_geometry_ref_t{OVSPATIALQUERY_GEOMETRY_CAPSULE, capsuleValue},
        ovspatialquery_geometry_ref_t{OVSPATIALQUERY_GEOMETRY_CONVEX, convexValue},
        ovspatialquery_geometry_ref_t{OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, meshValue}};
    for (const ovspatialquery_geometry_ref_t reference : references)
    {
        REQUIRE(ovspatialquery_destroy_geometry_definition(instance, reference).status == OVSPATIALQUERY_API_SUCCESS);
    }
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
    return 0;
}
