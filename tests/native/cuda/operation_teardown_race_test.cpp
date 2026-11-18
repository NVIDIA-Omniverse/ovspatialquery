/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../support/native/public_api_inputs.h"
#include "../../support/native/test_failure.h"
#include "cuda_driver.h"
#include <ovspatialquery/ovspatialquery_explicit.h>

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

// fetch_raycast_batch, destroy_op and destroy_op_after_cuda called on a
// completed CUDA batch-plan operation while another thread retires the
// operation and then destroys its plan, through the public API only. Each call
// either acts on the live operation or, once it is gone, reports so; none
// reads the plan's operation owner after destroy_batch_plan released it.

namespace
{

constexpr uint64_t kSphereValue = 77u;

ovspatialquery_population_handle_t publishUnitSphere(ovspatialquery_instance_handle_t instance)
{
    uint64_t   value  = kSphereValue;
    double     radius = 0.5;
    InputArray valueArray;
    valueArray.set(&value, 1, kDLUInt);
    InputArray radiusArray;
    radiusArray.set(&radius, 1, kDLFloat);
    const ovspatialquery_sphere_definition_table_t definitions{&valueArray.value, &radiusArray.value};
    waitAndDestroy(instance, ovspatialquery_enqueue_sphere_registration(instance, &definitions));

    const ovspatialquery_population_desc_t desc{1u, 1u};
    ovspatialquery_population_handle_t     population = OVSPATIALQUERY_INVALID_POPULATION;
    REQUIRE(ovspatialquery_create_explicit_population(instance, &desc, &population).status ==
            OVSPATIALQUERY_API_SUCCESS);
    uint64_t   object         = kSphereValue;
    int32_t    type           = OVSPATIALQUERY_GEOMETRY_SPHERE;
    uint64_t   geometry       = kSphereValue;
    double     translation[3] = {2.0, 0.0, 0.0};
    float      rotation[4]    = {0.0f, 0.0f, 0.0f, 1.0f};
    InputArray objectArray;
    objectArray.set(&object, 1, kDLUInt);
    InputArray typeArray;
    typeArray.set(&type, 1, kDLInt);
    InputArray geometryArray;
    geometryArray.set(&geometry, 1, kDLUInt);
    InputArray translationArray;
    translationArray.set(translation, 1, 3, kDLFloat);
    InputArray rotationArray;
    rotationArray.set(rotation, 1, 4, kDLFloat);
    const ovspatialquery_explicit_instance_table_t rows{&objectArray.value,
                                                        &typeArray.value,
                                                        &geometryArray.value,
                                                        &translationArray.value,
                                                        &rotationArray.value,
                                                        nullptr,
                                                        nullptr,
                                                        nullptr};
    const ovspatialquery_explicit_population_set_t set{1, rows, nullptr, 0u};
    waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set));
    return population;
}

bool destroyPlanWhenRetired(ovspatialquery_instance_handle_t instance, ovspatialquery_batch_plan_handle_t plan)
{
    for (;;)
    {
        const ovspatialquery_api_status_t status = ovspatialquery_destroy_batch_plan(instance, plan).status;
        if (status != OVSPATIALQUERY_API_BUSY)
        {
            return status == OVSPATIALQUERY_API_SUCCESS;
        }
        std::this_thread::yield();
    }
}

void testCudaOperationCallsRaceOperationAndPlanTeardown()
{
    const ovspatialquery_device_budget_t budget{{kDLCUDA, 0}, UINT64_C(64) << 20};
    const ovspatialquery_instance_desc_t desc{8u, 2u, UINT64_C(64) << 20, &budget, 1u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&desc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    const ovspatialquery_population_handle_t population = publishUnitSphere(instance);

    CudaDriver                      cuda;
    const CudaDriver::Stream        producer              = cuda.createStream();
    const CudaDriver::Stream        consumer              = cuda.createStream();
    const CudaDriver::Event         queryReady            = cuda.createEvent();
    double                          origin[3]             = {0.0, 0.0, 0.0};
    float                           direction[3]          = {1.0f, 0.0f, 0.0f};
    double                          maximumDistance       = 10.0;
    const CudaDriver::DevicePointer deviceOrigin          = cuda.allocate(sizeof(origin));
    const CudaDriver::DevicePointer deviceDirection       = cuda.allocate(sizeof(direction));
    const CudaDriver::DevicePointer deviceMaximumDistance = cuda.allocate(sizeof(maximumDistance));
    cuda.uploadAsync(deviceOrigin, origin, sizeof(origin), producer);
    cuda.uploadAsync(deviceDirection, direction, sizeof(direction), producer);
    cuda.uploadAsync(deviceMaximumDistance, &maximumDistance, sizeof(maximumDistance), producer);
    cuda.record(queryReady, producer);
    const DLDevice onCuda{kDLCUDA, 0};
    InputArray     originArray;
    originArray.set(reinterpret_cast<double*>(static_cast<uintptr_t>(deviceOrigin)), 1, 3, kDLFloat, onCuda);
    InputArray directionArray;
    directionArray.set(reinterpret_cast<float*>(static_cast<uintptr_t>(deviceDirection)), 1, 3, kDLFloat, onCuda);
    InputArray maxArray;
    maxArray.set(reinterpret_cast<double*>(static_cast<uintptr_t>(deviceMaximumDistance)), 1, kDLFloat, onCuda);
    ovspatialquery_raycast_batch_t batch{};
    batch.ordinal.kind         = OVSPATIALQUERY_ORDINAL_HEAD;
    batch.query_count          = 1u;
    batch.origins              = &originArray.value;
    batch.unit_directions      = &directionArray.value;
    batch.max_distances        = &maxArray.value;
    batch.cuda_sync.has_stream = 1u;
    batch.cuda_sync.stream     = reinterpret_cast<uintptr_t>(producer);
    batch.cuda_sync.wait_event = reinterpret_cast<uintptr_t>(queryReady);

    ovspatialquery_raycast_plan_desc_t planDesc{};
    planDesc.target                 = populationRef(population);
    planDesc.hit_mode               = OVSPATIALQUERY_HIT_MODE_CLOSEST;
    planDesc.precision              = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    planDesc.device                 = onCuda;
    planDesc.maximum_query_count    = 1u;
    planDesc.maximum_hits_per_query = 1u;
    planDesc.hit_capacity_per_slot  = 1u;
    planDesc.result_slot_count      = 1u;
    planDesc.result_fields          = OVSPATIALQUERY_BATCH_RESULT_DISTANCE;

    // One poller per call keeps the plan's adapter lock contended, so the
    // operation and plan teardown often lands while a call is in flight.
    constexpr int kRounds  = 200;
    constexpr int kPollers = 3;
    for (int round = 0; round < kRounds; ++round)
    {
        ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
        REQUIRE(ovspatialquery_create_raycast_plan(instance, &planDesc, &plan).status == OVSPATIALQUERY_API_SUCCESS);
        ovspatialquery_raycast_batch_reservation_t reservation{};
        const ovspatialquery_enqueue_result_t      admission =
            ovspatialquery_enqueue_raycast_plan(instance, plan, &batch, &reservation);
        REQUIRE(admission.status == OVSPATIALQUERY_API_SUCCESS);
        ovspatialquery_op_wait_result_t wait{};
        REQUIRE(ovspatialquery_wait_op(instance, admission.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &wait).status ==
                OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(wait.state == OVSPATIALQUERY_OP_COMPLETED);
        cuda.wait(consumer, reinterpret_cast<CudaDriver::Event>(reservation.common.result_ready.value));
        const CudaDriver::Event consumerDone = cuda.createEvent();
        cuda.record(consumerDone, consumer);
        cuda.synchronize(consumerDone);

        // Each poller repeats its call while it reports the live operation,
        // then must report the operation gone.
        std::atomic<int>         polling{0};
        std::vector<std::thread> pollers;
        const auto               poll = [&](auto call, ovspatialquery_api_status_t whileLive)
        {
            pollers.emplace_back(
                [&polling, call, whileLive]()
                {
                    ovspatialquery_api_status_t status = call();
                    polling.fetch_add(1);
                    while (status == whileLive)
                    {
                        status = call();
                    }
                    REQUIRE(status == OVSPATIALQUERY_API_INVALID_HANDLE);
                });
        };
        poll(
            [&]()
            {
                ovspatialquery_raycast_batch_result_t result{};
                return ovspatialquery_fetch_raycast_batch(instance, admission.op_id, &result).status;
            },
            OVSPATIALQUERY_API_SUCCESS);
        // A CUDA result is retired only with consumer_done.
        poll(
            [&]()
            {
                ovspatialquery_op_id_t operation = admission.op_id;
                return ovspatialquery_destroy_op(instance, &operation).status;
            },
            OVSPATIALQUERY_API_BUSY);
        // result_ready is never a valid consumer_done, so this retirement is
        // always refused and leaves the operation live.
        poll(
            [&]()
            {
                ovspatialquery_op_id_t      operation = admission.op_id;
                ovspatialquery_cuda_event_t done      = reservation.common.result_ready;
                return ovspatialquery_destroy_op_after_cuda(instance, &operation, &done).status;
            },
            OVSPATIALQUERY_API_INVALID_STATE);
        while (polling.load() < kPollers)
        {
            std::this_thread::yield();
        }
        ovspatialquery_op_id_t      operation = admission.op_id;
        ovspatialquery_cuda_event_t done{reinterpret_cast<uintptr_t>(consumerDone)};
        REQUIRE(ovspatialquery_destroy_op_after_cuda(instance, &operation, &done).status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(destroyPlanWhenRetired(instance, plan));
        for (std::thread& poller : pollers)
        {
            poller.join();
        }
    }

    cuda.free(deviceMaximumDistance);
    cuda.free(deviceDirection);
    cuda.free(deviceOrigin);
    cuda.destroyEvent(queryReady);
    cuda.destroyStream(consumer);
    cuda.destroyStream(producer);
    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(
        ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_SPHERE, kSphereValue}).status ==
        OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

} // namespace

int main()
{
    testCudaOperationCallsRaceOperationAndPlanTeardown();
    return 0;
}
