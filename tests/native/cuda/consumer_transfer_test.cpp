/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <ovspatialquery/ovspatialquery.h>
#include <ovspatialquery/ovspatialquery_explicit.h>

#include "../../support/native/cuda_public_test_helpers.h"
#include "../../support/native/public_api_inputs.h"
#include "cuda_driver.h"
#include "ovspatialquery_cuda_failure_points.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <thread>

extern "C" void     ovspatialquery_test_set_cuda_failure_point(uint32_t point);
extern "C" uint32_t ovspatialquery_test_get_cuda_failure_point();

namespace
{

const char* gPhase = "initialization";

void require(bool condition)
{
    if (!condition)
    {
        std::fprintf(stderr, "CUDA consumer transfer failure: %s\n", gPhase);
        std::fflush(stderr);
        std::_Exit(EXIT_FAILURE);
    }
}

ovspatialquery_input_array_t cudaArray(
    CudaDriver::DevicePointer pointer, DLDataType dtype, int32_t rank, int64_t* shape, int64_t* strides,
    ovspatialquery_input_ownership_t ownership = OVSPATIALQUERY_INPUT_COPIED_ON_SUBMISSION)
{
    ovspatialquery_input_array_t result{};
    result.tensor.data    = reinterpret_cast<void*>(static_cast<uintptr_t>(pointer));
    result.tensor.device  = {kDLCUDA, 0};
    result.tensor.ndim    = rank;
    result.tensor.dtype   = dtype;
    result.tensor.shape   = shape;
    result.tensor.strides = strides;
    result.ownership      = ownership;
    return result;
}

// Publishes one sphere prepared on the device, launches two bounds rays at it
// on a CUDA plan and hands the result to a consumer. Returns the instance's
// device usage while the launch is admitted and still pending.
uint64_t runConsumerTransfer(uint64_t deviceBudgetBytes = UINT64_C(64) << 20, bool expectDeviceBudget = false,
                             bool originInsideBounds = false, bool useForeignConsumerDone = false,
                             bool forcePostTransferDeviceLoss = false)
{
    gPhase = "instance";
    const ovspatialquery_device_budget_t budget{{kDLCUDA, 0}, deviceBudgetBytes};
    ovspatialquery_instance_desc_t       instanceDesc{};
    instanceDesc.operation_capacity           = 1u;
    instanceDesc.max_cpu_concurrency          = 1u;
    instanceDesc.host_byte_budget             = UINT64_C(64) << 20;
    instanceDesc.device_budgets               = &budget;
    instanceDesc.device_budget_count          = 1u;
    ovspatialquery_instance_handle_t instance = OVSPATIALQUERY_INVALID_INSTANCE;
    require(ovspatialquery_create_instance(&instanceDesc, &instance).status == OVSPATIALQUERY_API_SUCCESS);

    gPhase = "sphere registration";
    // The sphere sits at x = 1.5 with radius 0.5, so its world bounds span
    // [1, 2] on x: a ray from the origin along +x enters them through the -x
    // face at x = 1, and an origin at x = 1.5 starts inside them.
    uint64_t                                       sphereDefinition = 7001u;
    double                                         sphereRadius     = 0.5;
    InputArray                                     definitionValues(&sphereDefinition, 1, kDLUInt);
    InputArray                                     radii(&sphereRadius, 1, kDLFloat);
    const ovspatialquery_sphere_definition_table_t spheres{&definitionValues.value, &radii.value};
    waitAndDestroy(instance, ovspatialquery_enqueue_sphere_registration(instance, &spheres));

    CudaDriver cuda;
    gPhase                                                 = "population publication";
    const CudaDriver::Stream               producer        = cuda.createStream();
    const CudaDriver::Stream               consumer        = cuda.createStream();
    const CudaDriver::Event                populationReady = cuda.createEvent();
    const ovspatialquery_population_desc_t populationDesc{0u, 1u};
    ovspatialquery_population_handle_t     population = OVSPATIALQUERY_INVALID_POPULATION;
    require(ovspatialquery_create_explicit_population(instance, &populationDesc, &population).status ==
            OVSPATIALQUERY_API_SUCCESS);
    uint64_t                        objectId       = 42u;
    uint64_t                        objectBits[2]  = {0u, 0x4u}; // bit 66
    int32_t                         geometryKind   = OVSPATIALQUERY_GEOMETRY_SPHERE;
    double                          translation[3] = {1.5, 0.0, 0.0};
    float                           rotation[4]    = {0.0f, 0.0f, 0.0f, 1.0f};
    const CudaDriver::DevicePointer deviceIds      = cuda.allocate(sizeof(objectId));
    const CudaDriver::DevicePointer deviceBits     = cuda.allocate(sizeof(objectBits));
    const CudaDriver::DevicePointer deviceTypes    = cuda.allocate(sizeof(geometryKind));
    const CudaDriver::DevicePointer deviceValues   = cuda.allocate(sizeof(sphereDefinition));
    const CudaDriver::DevicePointer devicePosition = cuda.allocate(sizeof(translation));
    const CudaDriver::DevicePointer deviceRotation = cuda.allocate(sizeof(rotation));
    cuda.uploadAsync(deviceIds, &objectId, sizeof(objectId), producer);
    cuda.uploadAsync(deviceBits, objectBits, sizeof(objectBits), producer);
    cuda.uploadAsync(deviceTypes, &geometryKind, sizeof(geometryKind), producer);
    cuda.uploadAsync(deviceValues, &sphereDefinition, sizeof(sphereDefinition), producer);
    cuda.uploadAsync(devicePosition, translation, sizeof(translation), producer);
    cuda.uploadAsync(deviceRotation, rotation, sizeof(rotation), producer);
    cuda.record(populationReady, producer);
    InputArray rows[6];
    rows[0].set(reinterpret_cast<uint64_t*>(static_cast<uintptr_t>(deviceIds)), 1, kDLUInt, {kDLCUDA, 0});
    rows[1].set(reinterpret_cast<int32_t*>(static_cast<uintptr_t>(deviceTypes)), 1, kDLInt, {kDLCUDA, 0});
    rows[2].set(reinterpret_cast<uint64_t*>(static_cast<uintptr_t>(deviceValues)), 1, kDLUInt, {kDLCUDA, 0});
    rows[3].set(reinterpret_cast<double*>(static_cast<uintptr_t>(devicePosition)), 1, 3, kDLFloat, {kDLCUDA, 0});
    rows[4].set(reinterpret_cast<float*>(static_cast<uintptr_t>(deviceRotation)), 1, 4, kDLFloat, {kDLCUDA, 0});
    rows[5].set(reinterpret_cast<uint64_t*>(static_cast<uintptr_t>(deviceBits)), 1, 2, kDLUInt, {kDLCUDA, 0});
    for (InputArray& row : rows)
    {
        row.value.ownership = OVSPATIALQUERY_INPUT_BORROWED_UNTIL_TERMINAL;
    }
    const ovspatialquery_dynamic_execution_t onDevice{
        {kDLCUDA, 0},
        {1u, reinterpret_cast<uintptr_t>(producer), reinterpret_cast<uintptr_t>(populationReady)},
        OVSPATIALQUERY_PREPARE_ON_DEVICE};
    const ovspatialquery_explicit_population_set_t set{7,
                                                       {&rows[0].value, &rows[1].value, &rows[2].value, &rows[3].value,
                                                        &rows[4].value, nullptr, &rows[5].value, nullptr},
                                                       &onDevice,
                                                       0u};
    waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set));

    gPhase = "plan creation";
    ovspatialquery_raycast_plan_desc_t planDesc{};
    planDesc.target                 = populationRef(population);
    planDesc.hit_mode               = OVSPATIALQUERY_HIT_MODE_ANY;
    planDesc.precision              = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    planDesc.device                 = {kDLCUDA, 0};
    planDesc.maximum_query_count    = 2u;
    planDesc.maximum_hits_per_query = 1u;
    planDesc.hit_capacity_per_slot  = 1u;
    planDesc.result_slot_count      = 1u;
    planDesc.result_fields          = OVSPATIALQUERY_BATCH_RESULT_POSITION | OVSPATIALQUERY_BATCH_RESULT_NORMAL |
                             OVSPATIALQUERY_BATCH_RESULT_DISTANCE | OVSPATIALQUERY_BATCH_RESULT_GEOMETRY_IDENTITY;
    ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    require(ovspatialquery_create_raycast_plan(instance, &planDesc, &plan).status == OVSPATIALQUERY_API_SUCCESS);

    const double                    originX            = originInsideBounds ? 1.5 : 0.0;
    double                          origins[6]         = {originX, 0.0, 0.0, originX, 0.0, 0.0};
    float                           directions[6]      = {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    double                          maxDistances[2]    = {10.0, 10.0};
    uint64_t                        queryBits[4]       = {0u, objectBits[1], 0u, objectBits[1]};
    const CudaDriver::DevicePointer deviceOrigins      = cuda.allocate(sizeof(origins));
    const CudaDriver::DevicePointer deviceDirections   = cuda.allocate(sizeof(directions));
    const CudaDriver::DevicePointer deviceMaxDistances = cuda.allocate(sizeof(maxDistances));
    const CudaDriver::DevicePointer deviceQueryBits    = cuda.allocate(sizeof(queryBits));
    cuda.uploadAsync(deviceOrigins, origins, sizeof(origins), producer);
    cuda.uploadAsync(deviceDirections, directions, sizeof(directions), producer);
    cuda.uploadAsync(deviceMaxDistances, maxDistances, sizeof(maxDistances), producer);
    cuda.uploadAsync(deviceQueryBits, queryBits, sizeof(queryBits), producer);
    // Hold the producer until the admitted launch's device usage is read, so
    // no charge it took can be released before then.
    std::promise<void> releaseProducer;
    std::future<void>  producerReleased = releaseProducer.get_future();
    cuda.enqueueHostFunction(
        producer, [](void* state) { static_cast<std::future<void>*>(state)->wait(); }, &producerReleased);
    const CudaDriver::Event queryReady = cuda.createEvent();
    cuda.record(queryReady, producer);
    int64_t                      scalarShape[1]  = {2};
    int64_t                      scalarStride[1] = {1};
    int64_t                      vectorShape[2]  = {2, 3};
    int64_t                      vectorStride[2] = {3, 1};
    ovspatialquery_input_array_t originsInput =
        cudaArray(deviceOrigins, {kDLFloat, 64u, 1u}, 2, vectorShape, vectorStride);
    ovspatialquery_input_array_t directionsInput =
        cudaArray(deviceDirections, {kDLFloat, 32u, 1u}, 2, vectorShape, vectorStride);
    ovspatialquery_input_array_t distancesInput =
        cudaArray(deviceMaxDistances, {kDLFloat, 64u, 1u}, 1, scalarShape, scalarStride);
    int64_t                      bitsShape[2]  = {2, 2};
    int64_t                      bitsStride[2] = {2, 1};
    ovspatialquery_input_array_t bitsInput = cudaArray(deviceQueryBits, {kDLUInt, 64u, 1u}, 2, bitsShape, bitsStride);
    gPhase                                 = "enqueue";
    ovspatialquery_raycast_batch_t batch{};
    batch.ordinal.kind         = OVSPATIALQUERY_ORDINAL_HEAD;
    batch.query_count          = 2u;
    batch.query_bits           = &bitsInput;
    batch.origins              = &originsInput;
    batch.unit_directions      = &directionsInput;
    batch.max_distances        = &distancesInput;
    batch.cuda_sync.has_stream = 1u;
    batch.cuda_sync.stream     = reinterpret_cast<uintptr_t>(producer);
    batch.cuda_sync.wait_event = reinterpret_cast<uintptr_t>(queryReady);
    ovspatialquery_raycast_batch_reservation_t reservation{};
    ovspatialquery_enqueue_result_t            operation =
        ovspatialquery_enqueue_raycast_plan(instance, plan, &batch, &reservation);
    if (operation.status != OVSPATIALQUERY_API_SUCCESS)
    {
        std::fprintf(stderr, "candidate enqueue status=%u op=%llu\n", static_cast<unsigned int>(operation.status),
                     static_cast<unsigned long long>(operation.op_id.raw_id));
    }
    uint64_t launchUsage = 0u;
    require(ovspatialquery_get_instance_memory_usage(instance, {kDLCUDA, 0}, &launchUsage).status ==
            OVSPATIALQUERY_API_SUCCESS);
    releaseProducer.set_value();
    CudaDriver::Event        consumerDone                   = nullptr;
    uint64_t                 attemptsBeforeConsumerTransfer = 0u;
    CudaDriver::ForeignEvent foreignConsumerDone{};
    if (expectDeviceBudget)
    {
        require(operation.status == OVSPATIALQUERY_API_DEVICE_BUDGET && operation.op_id == OVSPATIALQUERY_INVALID_OP &&
                reservation.common.result_ready.value == 0u);
    }
    else
    {
        require(operation.status == OVSPATIALQUERY_API_SUCCESS && operation.op_id != OVSPATIALQUERY_INVALID_OP &&
                reservation.common.source_ordinal == 7 && reservation.common.execution_device.device_type == kDLCUDA &&
                reservation.common.result_ready.value != 0u);

        gPhase                                       = "pre-terminal consumer completion rejection";
        const CudaDriver::Event rejectedConsumerDone = cuda.createEvent();
        cuda.record(rejectedConsumerDone, consumer);
        cuda.synchronize(rejectedConsumerDone);
        const ovspatialquery_op_id_t rejectedOperation = operation.op_id;
        ovspatialquery_cuda_event_t  rejectedDone{reinterpret_cast<uintptr_t>(rejectedConsumerDone)};
        const uintptr_t              rejectedDoneValue       = rejectedDone.value;
        const uint64_t               attemptsBeforeRejection = transferredEventDestroyAttempts();
        require(ovspatialquery_destroy_op_after_cuda(instance, &operation.op_id, &rejectedDone).status ==
                    OVSPATIALQUERY_API_BUSY &&
                operation.op_id == rejectedOperation && rejectedDone.value == rejectedDoneValue &&
                transferredEventDestroyAttempts() == attemptsBeforeRejection);
        cuda.destroyEvent(rejectedConsumerDone);

        gPhase = "wait and fetch";
        ovspatialquery_op_wait_result_t waited{};
        const ovspatialquery_result_t   waitStatus =
            ovspatialquery_wait_op(instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited);
        require(waitStatus.status == OVSPATIALQUERY_API_SUCCESS && waited.state == OVSPATIALQUERY_OP_COMPLETED);
        ovspatialquery_raycast_batch_result_t result{};
        require(ovspatialquery_fetch_raycast_batch(instance, operation.op_id, &result).status ==
                    OVSPATIALQUERY_API_SUCCESS &&
                result.common.result_ready.value == reservation.common.result_ready.value);
        gPhase = "device result read";
        cuda.wait(consumer, reinterpret_cast<CudaDriver::Event>(result.common.result_ready.value));
        uint32_t queryStatus[2] = {UINT32_MAX, UINT32_MAX};
        uint64_t stored         = UINT64_MAX;
        uint64_t offsets[2]     = {UINT64_MAX, UINT64_MAX};
        uint64_t counts[2]      = {UINT64_MAX, UINT64_MAX};
        uint64_t id             = UINT64_MAX;
        int32_t  geometryType   = 0;
        uint64_t geometryValue  = UINT64_MAX;
        float    normal[3]      = {};
        uint8_t  truncated[2]   = {UINT8_MAX, UINT8_MAX};
        cuda.downloadAsync(&queryStatus, deviceAddress(result.columns.hit_ranges.query_status), sizeof(queryStatus),
                           consumer);
        cuda.downloadAsync(&stored, deviceAddress(result.columns.hit_ranges.stored_hit_count), sizeof(stored),
                           consumer);
        cuda.downloadAsync(&offsets, deviceAddress(result.columns.hit_ranges.hit_offsets), sizeof(offsets), consumer);
        cuda.downloadAsync(&counts, deviceAddress(result.columns.hit_ranges.hit_counts), sizeof(counts), consumer);
        cuda.downloadAsync(&truncated, deviceAddress(result.columns.hit_ranges.truncated), sizeof(truncated), consumer);
        cuda.downloadAsync(&id, deviceAddress(result.columns.object_ids), sizeof(id), consumer);
        cuda.downloadAsync(&geometryType, deviceAddress(result.columns.geometry.types), sizeof(geometryType), consumer);
        cuda.downloadAsync(&geometryValue, deviceAddress(result.columns.geometry.values), sizeof(geometryValue),
                           consumer);
        cuda.downloadAsync(normal, deviceAddress(result.columns.contact.candidate.bounds_normals), sizeof(normal),
                           consumer);
        if (useForeignConsumerDone)
        {
            const CudaDriver::Event consumed = cuda.createEvent();
            cuda.record(consumed, consumer);
            cuda.synchronize(consumed);
            cuda.destroyEvent(consumed);
            foreignConsumerDone = cuda.createForeignEvent();
            consumerDone        = foreignConsumerDone.event;
        }
        else
        {
            consumerDone = cuda.createEvent();
            cuda.record(consumerDone, consumer);
            cuda.synchronize(consumerDone);
        }
        require(queryStatus[0] == 0u && queryStatus[1] == 0u && stored == 1u && offsets[0] == 0u && offsets[1] == 1u &&
                counts[0] == 1u && counts[1] == 0u && truncated[0] == 0u && truncated[1] == 1u && id == objectId &&
                geometryType == OVSPATIALQUERY_GEOMETRY_SPHERE && geometryValue == sphereDefinition &&
                normal[0] == -1.0f && normal[1] == 0.0f && normal[2] == 0.0f);
        gPhase = "consumer completion transfer";
        ovspatialquery_cuda_event_t done{reinterpret_cast<uintptr_t>(consumerDone)};
        attemptsBeforeConsumerTransfer = transferredEventDestroyAttempts();
        if (forcePostTransferDeviceLoss)
        {
            ovspatialquery_test_set_cuda_failure_point(ovspatialquery::detail::kCudaFailureConsumerPollDeviceLost);
        }
        require(ovspatialquery_destroy_op_after_cuda(instance, &operation.op_id, &done).status ==
                    OVSPATIALQUERY_API_SUCCESS &&
                operation.op_id == OVSPATIALQUERY_INVALID_OP && done.value == 0u);
    }
    gPhase = "plan destruction";
    require(destroyPlanWhenRetired(instance, plan));
    plan   = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    gPhase = "instance destruction";
    require(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    require(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_SPHERE, sphereDefinition})
                .status == OVSPATIALQUERY_API_SUCCESS);
    require(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
    if (consumerDone != nullptr)
    {
        if (forcePostTransferDeviceLoss)
        {
            require(ovspatialquery_test_get_cuda_failure_point() == 0u &&
                    transferredEventDestroyAttempts() == attemptsBeforeConsumerTransfer);
        }
        else
        {
            require(transferredEventDestroysWereInvoked());
        }
        if (useForeignConsumerDone)
        {
            cuda.destroyForeignContext(foreignConsumerDone);
        }
    }
    cuda.destroyEvent(queryReady);
    cuda.destroyEvent(populationReady);
    cuda.free(deviceQueryBits);
    cuda.free(deviceMaxDistances);
    cuda.free(deviceDirections);
    cuda.free(deviceOrigins);
    cuda.free(deviceRotation);
    cuda.free(devicePosition);
    cuda.free(deviceValues);
    cuda.free(deviceTypes);
    cuda.free(deviceBits);
    cuda.free(deviceIds);
    cuda.destroyStream(consumer);
    cuda.destroyStream(producer);
    return launchUsage;
}

} // namespace

int main()
{
    // The device bytes one admitted launch needs on top of the published
    // population and the plan, measured at a generous budget: the launch is
    // admitted at exactly that budget and refused with DEVICE_BUDGET one byte
    // below it, after publication and plan creation fit.
    const uint64_t launchCharge = runConsumerTransfer();
    gPhase                      = "launch admitted at its measured device charge";
    require(runConsumerTransfer(launchCharge) == launchCharge);
    runConsumerTransfer(launchCharge - 1u, true);
    runConsumerTransfer(UINT64_C(64) << 20, false, true);
    runConsumerTransfer(UINT64_C(64) << 20, false, false, true);
    runConsumerTransfer(UINT64_C(64) << 20, false, false, false, true);
    return 0;
}
