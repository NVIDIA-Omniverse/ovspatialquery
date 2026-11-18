/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../support/native/public_api_inputs.h"
#include "../../support/native/test_failure.h"
#include "ovspatialquery_instance_operation_runtime.h"
#include <ovspatialquery/ovspatialquery_explicit.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

// Arms the next operation a CPU plan admits to fail with exhausted resources
// (built only into test runtimes): no real input produces that failure on demand.
extern "C" ovspatialquery_api_status_t ovspatialquery_test_fail_next_family_plan_resource(
    ovspatialquery_instance_handle_t instance, ovspatialquery_batch_plan_handle_t plan);

// query_op_status polled on a failed batch-plan operation while another thread
// destroys the operation and then its plan, through the public API only. Every
// poll either reports the failure with its terminal cause or, once the
// operation is gone, INVALID_HANDLE; it never reads the plan's operation owner
// after destroy_batch_plan released it.

namespace
{

constexpr uint64_t kBoxValue = 77u;

ovspatialquery_population_handle_t publishUnitBox(ovspatialquery_instance_handle_t instance)
{
    uint64_t   value          = kBoxValue;
    double     halfExtents[3] = {0.5, 0.5, 0.5};
    InputArray valueArray;
    valueArray.set(&value, 1, kDLUInt);
    InputArray extentArray;
    extentArray.set(halfExtents, 1, 3, kDLFloat);
    const ovspatialquery_box_definition_table_t definitions{&valueArray.value, &extentArray.value};
    waitAndDestroy(instance, ovspatialquery_enqueue_box_registration(instance, &definitions));

    const ovspatialquery_population_desc_t desc{1u, 1u};
    ovspatialquery_population_handle_t     population = OVSPATIALQUERY_INVALID_POPULATION;
    REQUIRE(ovspatialquery_create_explicit_population(instance, &desc, &population).status ==
            OVSPATIALQUERY_API_SUCCESS);
    uint64_t   object         = kBoxValue;
    int32_t    type           = OVSPATIALQUERY_GEOMETRY_BOX;
    uint64_t   geometry       = kBoxValue;
    double     translation[3] = {0.5, 0.5, 0.5};
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

// One ray through the unit box, and a one-query CPU plan for it.
struct RaycastInputs
{
    double                             origin[3]    = {-1.0, 0.5, 0.5};
    float                              direction[3] = {1.0f, 0.0f, 0.0f};
    double                             minDistance  = 0.0;
    double                             maxDistance  = 10.0;
    InputArray                         originArray;
    InputArray                         directionArray;
    InputArray                         minArray;
    InputArray                         maxArray;
    ovspatialquery_raycast_batch_t     batch{};
    ovspatialquery_raycast_plan_desc_t desc{};

    RaycastInputs()
    {
        originArray.set(origin, 1, 3, kDLFloat);
        directionArray.set(direction, 1, 3, kDLFloat);
        minArray.set(&minDistance, 1, kDLFloat);
        maxArray.set(&maxDistance, 1, kDLFloat);
        batch.ordinal.kind    = OVSPATIALQUERY_ORDINAL_HEAD;
        batch.query_count     = 1u;
        batch.origins         = &originArray.value;
        batch.unit_directions = &directionArray.value;
        batch.min_distances   = &minArray.value;
        batch.max_distances   = &maxArray.value;
    }

    const ovspatialquery_raycast_plan_desc_t* planDescFor(ovspatialquery_population_handle_t population)
    {
        desc.target                 = populationRef(population);
        desc.hit_mode               = OVSPATIALQUERY_HIT_MODE_CLOSEST;
        desc.precision              = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
        desc.device                 = DLDevice{kDLCPU, 0};
        desc.maximum_query_count    = 1u;
        desc.maximum_hits_per_query = 1u;
        desc.hit_capacity_per_slot  = 1u;
        desc.result_slot_count      = 1u;
        desc.result_fields          = OVSPATIALQUERY_BATCH_RESULT_DISTANCE;
        return &desc;
    }
};

// destroy_batch_plan is BUSY while a wait_op on one of the plan's operations
// is still running, even once that operation is destroyed: the waiter is
// held after its wake-up and before it rechecks the operation, the operation
// is observed and destroyed, and the plan must outlive the waiter.
void testPlanDestroyWaitsForActiveWait()
{
    using namespace ovspatialquery::detail;
    const ovspatialquery_instance_desc_t desc{8u, 2u, UINT64_C(64) << 20, nullptr, 0u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&desc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    const ovspatialquery_population_handle_t population = publishUnitBox(instance);
    RaycastInputs                            inputs;
    ovspatialquery_batch_plan_handle_t       plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_raycast_plan(instance, inputs.planDescFor(population), &plan).status ==
            OVSPATIALQUERY_API_SUCCESS);

    REQUIRE(setInstanceOperationWorkersPausedForTest(instance, true) == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_raycast_batch_reservation_t reservation{};
    const ovspatialquery_enqueue_result_t      admission =
        ovspatialquery_enqueue_raycast_plan(instance, plan, &inputs.batch, &reservation);
    REQUIRE(admission.status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(setInstanceOperationWaiterPausedBeforeRevalidationForTest(instance, true) == OVSPATIALQUERY_API_SUCCESS);
    const uint64_t                  entries    = instanceOperationWaiterEntryCountForTest(instance);
    ovspatialquery_api_status_t     waitStatus = OVSPATIALQUERY_API_INTERNAL;
    ovspatialquery_op_wait_result_t wait{};
    std::thread                     waiter(
        [&]() {
            waitStatus =
                ovspatialquery_wait_op(instance, admission.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &wait).status;
        });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (instanceOperationWaiterEntryCountForTest(instance) == entries && std::chrono::steady_clock::now() < deadline)
    {
        std::this_thread::yield();
    }
    REQUIRE(instanceOperationWaiterEntryCountForTest(instance) != entries);
    REQUIRE(setInstanceOperationWorkersPausedForTest(instance, false) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(waitForInstanceOperationWaiterAtRevalidationForTest(instance) == OVSPATIALQUERY_API_SUCCESS);

    ovspatialquery_op_status_t status{};
    REQUIRE(ovspatialquery_query_op_status(instance, admission.op_id, &status).status == OVSPATIALQUERY_API_SUCCESS &&
            status.state == OVSPATIALQUERY_OP_COMPLETED);
    ovspatialquery_op_id_t operation = admission.op_id;
    REQUIRE(ovspatialquery_destroy_op(instance, &operation).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_batch_plan(instance, plan).status == OVSPATIALQUERY_API_BUSY);

    REQUIRE(setInstanceOperationWaiterPausedBeforeRevalidationForTest(instance, false) == OVSPATIALQUERY_API_SUCCESS);
    waiter.join();
    REQUIRE(waitStatus == OVSPATIALQUERY_API_INVALID_HANDLE);
    REQUIRE(ovspatialquery_destroy_batch_plan(instance, plan).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_BOX, kBoxValue}).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

void testStatusPollRacesOperationAndPlanTeardown()
{
    const ovspatialquery_instance_desc_t desc{8u, 2u, UINT64_C(64) << 20, nullptr, 0u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&desc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    const ovspatialquery_population_handle_t population = publishUnitBox(instance);

    RaycastInputs inputs;

    // Two pollers keep the plan's adapter lock contended, so the operation and
    // plan teardown often lands while a poll is in flight.
    constexpr int kRounds  = 200;
    constexpr int kPollers = 2;
    for (int round = 0; round < kRounds; ++round)
    {
        ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
        REQUIRE(ovspatialquery_create_raycast_plan(instance, inputs.planDescFor(population), &plan).status ==
                OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(ovspatialquery_test_fail_next_family_plan_resource(instance, plan) == OVSPATIALQUERY_API_SUCCESS);
        ovspatialquery_raycast_batch_reservation_t reservation{};
        const ovspatialquery_enqueue_result_t      admission =
            ovspatialquery_enqueue_raycast_plan(instance, plan, &inputs.batch, &reservation);
        REQUIRE(admission.status == OVSPATIALQUERY_API_SUCCESS);
        ovspatialquery_op_wait_result_t wait{};
        REQUIRE(ovspatialquery_wait_op(instance, admission.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &wait).status ==
                OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(wait.state == OVSPATIALQUERY_OP_FAILURE);

        std::atomic<int>         polling{0};
        std::vector<std::thread> pollers;
        for (int poller = 0; poller < kPollers; ++poller)
        {
            pollers.emplace_back(
                [&]()
                {
                    bool counted = false;
                    for (;;)
                    {
                        ovspatialquery_op_status_t        status{};
                        const ovspatialquery_api_status_t result =
                            ovspatialquery_query_op_status(instance, admission.op_id, &status).status;
                        if (!counted)
                        {
                            counted = true;
                            polling.fetch_add(1);
                        }
                        if (result != OVSPATIALQUERY_API_SUCCESS)
                        {
                            REQUIRE(result == OVSPATIALQUERY_API_INVALID_HANDLE);
                            return;
                        }
                        REQUIRE(status.state == OVSPATIALQUERY_OP_FAILURE &&
                                status.terminal_status == OVSPATIALQUERY_API_OUT_OF_MEMORY);
                    }
                });
        }
        while (polling.load() < kPollers)
        {
            std::this_thread::yield();
        }
        ovspatialquery_op_id_t operation = admission.op_id;
        REQUIRE(ovspatialquery_destroy_op(instance, &operation).status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(ovspatialquery_destroy_batch_plan(instance, plan).status == OVSPATIALQUERY_API_SUCCESS);
        for (std::thread& poller : pollers)
        {
            poller.join();
        }
    }

    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_BOX, kBoxValue}).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

} // namespace

int main()
{
    testPlanDestroyWaitsForActiveWait();
    testStatusPollRacesOperationAndPlanTeardown();
    return 0;
}
