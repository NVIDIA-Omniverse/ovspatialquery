/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <ovspatialquery/ovspatialquery_explicit.h>

#include "../../support/native/public_api_inputs.h"
#include "oracle_backend.h"
#include "test_failure.h"

#include "ovspatialquery_instance_operation_runtime.h"

#include <array>
#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>

// Asynchronous operation lifecycle promises from ovspatialquery.h and
// ovspatialquery_explicit.h, ported off the retired generation-1 internal
// API (source/ovspatialquery_private_api.h) onto ovspatialquery_create_instance
// and the explicit population adapter. The oracle_backend.h hooks used below
// gate the same default backend runtime that ovspatialquery_create_instance
// acquires (see ovspatialquery_instance.cpp's acquireDefaultBackendRuntime),
// so they observe publicly created instances exactly as they observed the
// retired private API's instances.

namespace
{

ovspatialquery_instance_handle_t createInstance(uint32_t operationCapacity, uint32_t maxCpuConcurrency = 1u)
{
    const ovspatialquery_instance_desc_t desc{operationCapacity, maxCpuConcurrency, UINT64_C(16) * 1024u * 1024u,
                                              nullptr, 0u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&desc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    return instance;
}

uint64_t registerUnitSphere(ovspatialquery_instance_handle_t instance, uint64_t definitionValue)
{
    double                                         radius = 1.0;
    InputArray                                     values(&definitionValue, 1, kDLUInt);
    InputArray                                     radii(&radius, 1, kDLFloat);
    const ovspatialquery_sphere_definition_table_t spheres{&values.value, &radii.value};
    const ovspatialquery_enqueue_result_t admission = ovspatialquery_enqueue_sphere_registration(instance, &spheres);
    REQUIRE(admission.status == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_op_wait_result_t wait{};
    REQUIRE(ovspatialquery_wait_op(instance, admission.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &wait).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(wait.state == OVSPATIALQUERY_OP_COMPLETED);
    ovspatialquery_op_id_t opId = admission.op_id;
    REQUIRE(ovspatialquery_destroy_op(instance, &opId).status == OVSPATIALQUERY_API_SUCCESS);
    return definitionValue;
}

ovspatialquery_enqueue_result_t enqueueSet(ovspatialquery_instance_handle_t   instance,
                                           ovspatialquery_population_handle_t population,
                                           ovspatialquery_ordinal_t ordinal, InstanceRows& rows)
{
    const ovspatialquery_explicit_population_set_t set{ordinal, rows.table(), nullptr, 0u};
    return ovspatialquery_enqueue_explicit_population_set(instance, population, &set);
}

ovspatialquery_enqueue_result_t enqueueUpdate(ovspatialquery_instance_handle_t   instance,
                                              ovspatialquery_population_handle_t population,
                                              ovspatialquery_ordinal_t base, ovspatialquery_ordinal_t target,
                                              InstanceRows& changed, InputArray& removed)
{
    const ovspatialquery_explicit_population_update_t update{
        base, target, OVSPATIALQUERY_EXPLICIT_UPDATE_COMPLETE_ROWS, changed.table(), &removed.value, nullptr};
    return ovspatialquery_enqueue_explicit_population_update(instance, population, &update);
}

InputArray& emptyRemoved(InputArray& removed)
{
    static uint64_t zero = 0u;
    removed.set(&zero, 0, kDLUInt);
    return removed;
}

// Promise: destroy_population "Returns BUSY while ... an admitted operation
// ... still retains the population" (ovspatialquery.h), proven for a
// reusable raycast-plan launch: the plan launch itself, and then the plan,
// each independently retain the population until each is released.
void testRaycastPlanRetainsTargetPopulationUntilTerminal()
{
    ovspatialquery_instance_handle_t   instance         = createInstance(2u);
    const uint64_t                     sphereDefinition = registerUnitSphere(instance, 700u);
    ovspatialquery_population_handle_t population       = createExplicitPopulation(instance, 2u);
    InstanceRows                       rows;
    rows.add(700u, OVSPATIALQUERY_GEOMETRY_SPHERE, sphereDefinition, 5.0, 0.0, 0.0);
    waitAndDestroy(instance, enqueueSet(instance, population, 1, rows));

    ovspatialquery_raycast_plan_desc_t planDesc{};
    planDesc.target                         = populationRef(population);
    planDesc.hit_mode                       = OVSPATIALQUERY_HIT_MODE_ANY;
    planDesc.precision                      = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    planDesc.device                         = DLDevice{kDLCPU, 0};
    planDesc.maximum_query_count            = 1u;
    planDesc.maximum_hits_per_query         = 1u;
    planDesc.hit_capacity_per_slot          = 1u;
    planDesc.result_slot_count              = 1u;
    ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_raycast_plan(instance, &planDesc, &plan).status == OVSPATIALQUERY_API_SUCCESS);

    double                         origin[3]    = {-5.0, 0.0, 0.0};
    float                          direction[3] = {1.0f, 0.0f, 0.0f};
    double                         maxDistance  = 100.0;
    InputArray                     originInput(origin, 1, 3, kDLFloat);
    InputArray                     directionInput(direction, 1, 3, kDLFloat);
    InputArray                     maxDistanceInput(&maxDistance, 1, kDLFloat);
    ovspatialquery_raycast_batch_t batch{};
    batch.ordinal.kind    = OVSPATIALQUERY_ORDINAL_HEAD;
    batch.query_count     = 1u;
    batch.origins         = &originInput.value;
    batch.unit_directions = &directionInput.value;
    batch.max_distances   = &maxDistanceInput.value;

    // A scheduling failure is INTERNAL with no operation and releases the
    // plan's single result slot for the launch below.
    ovspatialquery_raycast_batch_reservation_t reservation{};
    REQUIRE(ovspatialquery::detail::failNextInstanceOperationSchedulingForTest(instance) == OVSPATIALQUERY_API_SUCCESS);
    const ovspatialquery_enqueue_result_t unscheduled =
        ovspatialquery_enqueue_raycast_plan(instance, plan, &batch, &reservation);
    REQUIRE(unscheduled.status == OVSPATIALQUERY_API_INTERNAL && unscheduled.op_id == OVSPATIALQUERY_INVALID_OP);

    // Plan launches run on the instance's operation runtime, which the oracle
    // backend gates never see; pausing execution holds the launch pending.
    REQUIRE(ovspatialquery::detail::setInstanceOperationExecutionsPausedForTest(instance, true) ==
            OVSPATIALQUERY_API_SUCCESS);
    const ovspatialquery_enqueue_result_t admission =
        ovspatialquery_enqueue_raycast_plan(instance, plan, &batch, &reservation);
    REQUIRE(admission.status == OVSPATIALQUERY_API_SUCCESS);

    // The launch retains its target population until its own terminal state
    // is reached.
    const ovspatialquery_result_t destroyWhileLaunched = ovspatialquery_destroy_population(instance, population);
    if (destroyWhileLaunched.status != OVSPATIALQUERY_API_BUSY)
    {
        std::fprintf(stderr, "destroy_population during a launch returned %u\n",
                     static_cast<unsigned>(destroyWhileLaunched.status));
    }
    REQUIRE(destroyWhileLaunched.status == OVSPATIALQUERY_API_BUSY);

    REQUIRE(ovspatialquery::detail::setInstanceOperationExecutionsPausedForTest(instance, false) ==
            OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_op_wait_result_t wait{};
    REQUIRE(ovspatialquery_wait_op(instance, admission.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &wait).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(wait.state == OVSPATIALQUERY_OP_COMPLETED);

    ovspatialquery_raycast_batch_result_t result{};
    REQUIRE(ovspatialquery_fetch_raycast_batch(instance, admission.op_id, &result).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(result.common.source_ordinal == 1);
    REQUIRE(result.columns.object_ids.shape[0] == 1);
    REQUIRE(static_cast<const uint64_t*>(result.columns.object_ids.data)[0] == 700u);
    ovspatialquery_op_id_t opId = admission.op_id;
    REQUIRE(ovspatialquery_destroy_op(instance, &opId).status == OVSPATIALQUERY_API_SUCCESS);

    // The op is gone, but the plan still retains the population.
    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_BUSY);
    REQUIRE(ovspatialquery_destroy_batch_plan(instance, plan).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_SPHERE, sphereDefinition})
                .status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

// Promise: "One publication may be pending per population" (ovspatialquery.h
// common call rules) and destroy_population's BUSY-while-retained rule,
// exercised across the full window a mutation is admitted-but-not-terminal.
void testCompletionLinearizesMutationAndPopulationRelease()
{
    ovspatialquery_instance_handle_t   instance   = createInstance(4u);
    ovspatialquery_backend_runtime_t*  runtime    = ovspatialquery_test_oracle_get_default_runtime();
    ovspatialquery_population_handle_t population = createExplicitPopulation(instance, 2u);
    InstanceRows                       initial;
    waitAndDestroy(instance, enqueueSet(instance, population, 1, initial));

    ovspatialquery_test_oracle_block_before_completion(runtime, 2u, 1u);
    InstanceRows                          changed;
    InputArray                            removed;
    const ovspatialquery_enqueue_result_t admission =
        enqueueUpdate(instance, population, 1, 2, changed, emptyRemoved(removed));
    REQUIRE(admission.status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_test_oracle_wait_before_completion(runtime) == 1u);

    InstanceRows                          competingRows;
    InputArray                            competingRemoved;
    const ovspatialquery_enqueue_result_t competing =
        enqueueUpdate(instance, population, 2, 3, competingRows, emptyRemoved(competingRemoved));
    if (competing.status != OVSPATIALQUERY_API_BUSY)
    {
        std::fprintf(stderr, "competing update status %u, expected BUSY\n", static_cast<unsigned>(competing.status));
    }
    REQUIRE(competing.status == OVSPATIALQUERY_API_BUSY);
    REQUIRE(competing.op_id == OVSPATIALQUERY_INVALID_OP);
    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_BUSY);

    ovspatialquery_test_oracle_release_before_completion(runtime);
    waitAndDestroy(instance, admission);

    ovspatialquery_population_info_t info{};
    REQUIRE(ovspatialquery_get_population_info(instance, population, &info).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(info.head_ordinal == 2);

    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

// Promise: destroy_op "Pending or terminal-unobserved work returns BUSY";
// wait_op is "repeatable with distinct writable output structs"; op_ids
// "are instance-scoped" so a stale/foreign id is INVALID_HANDLE.
void testOperationHandleBusyStatesAndCapacityRelease()
{
    ovspatialquery_instance_handle_t   instance        = createInstance(4u);
    ovspatialquery_instance_handle_t   foreignInstance = createInstance(4u);
    ovspatialquery_backend_runtime_t*  runtime         = ovspatialquery_test_oracle_get_default_runtime();
    ovspatialquery_population_handle_t population      = createExplicitPopulation(instance, 2u);
    InstanceRows                       initial;
    waitAndDestroy(instance, enqueueSet(instance, population, 1, initial));

    ovspatialquery_test_oracle_block_before_completion(runtime, 2u, 1u);
    InstanceRows                          changed;
    InputArray                            removed;
    const ovspatialquery_enqueue_result_t admission =
        enqueueUpdate(instance, population, 1, 2, changed, emptyRemoved(removed));
    REQUIRE(admission.status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_test_oracle_wait_before_completion(runtime) == 1u);

    ovspatialquery_op_id_t pendingId = admission.op_id;
    REQUIRE(ovspatialquery_destroy_op(instance, &pendingId).status == OVSPATIALQUERY_API_BUSY);
    REQUIRE(pendingId == admission.op_id);

    ovspatialquery_api_status_t     firstWaitStatus  = OVSPATIALQUERY_API_INTERNAL;
    ovspatialquery_api_status_t     secondWaitStatus = OVSPATIALQUERY_API_INTERNAL;
    ovspatialquery_op_wait_result_t firstWait{};
    ovspatialquery_op_wait_result_t secondWait{};
    std::thread                     firstWaiter(
        [&]()
        {
            firstWaitStatus =
                ovspatialquery_wait_op(instance, admission.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &firstWait).status;
        });
    std::thread secondWaiter(
        [&]()
        {
            secondWaitStatus =
                ovspatialquery_wait_op(instance, admission.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &secondWait).status;
        });
    ovspatialquery_test_oracle_release_before_completion(runtime);
    firstWaiter.join();
    secondWaiter.join();
    REQUIRE(firstWaitStatus == OVSPATIALQUERY_API_SUCCESS && firstWait.state == OVSPATIALQUERY_OP_COMPLETED);
    REQUIRE(secondWaitStatus == OVSPATIALQUERY_API_SUCCESS && secondWait.state == OVSPATIALQUERY_OP_COMPLETED);

    ovspatialquery_op_id_t opId = admission.op_id;
    REQUIRE(ovspatialquery_destroy_op(instance, &opId).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(opId == OVSPATIALQUERY_INVALID_OP);

    // The foreign instance holds a live operation in the same slot at the same
    // generation, so only the instance scope of the ID can reject it.
    ovspatialquery_population_handle_t foreignPopulation = createExplicitPopulation(foreignInstance, 2u);
    InstanceRows                       foreignInitial;
    waitAndDestroy(foreignInstance, enqueueSet(foreignInstance, foreignPopulation, 1, foreignInitial));
    InstanceRows                          foreignChanged;
    InputArray                            foreignRemoved;
    const ovspatialquery_enqueue_result_t foreignLive =
        enqueueUpdate(foreignInstance, foreignPopulation, 1, 2, foreignChanged, emptyRemoved(foreignRemoved));
    REQUIRE(foreignLive.status == OVSPATIALQUERY_API_SUCCESS);

    ovspatialquery_op_wait_result_t staleWait{};
    REQUIRE(ovspatialquery_wait_op(instance, admission.op_id, 0u, &staleWait).status ==
            OVSPATIALQUERY_API_INVALID_HANDLE);
    REQUIRE(ovspatialquery_wait_op(foreignInstance, admission.op_id, 0u, &staleWait).status ==
            OVSPATIALQUERY_API_INVALID_HANDLE);

    waitAndDestroy(foreignInstance, foreignLive);
    REQUIRE(ovspatialquery_destroy_population(foreignInstance, foreignPopulation).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(foreignInstance).status == OVSPATIALQUERY_API_SUCCESS);
}

// Promise: "OPERATION_CAPACITY: The instance has no free asynchronous-
// operation entry" and "a slot returns at op destruction"
// (ovspatialquery_instance_desc_t.operation_capacity).
// Operation IDs are instance-scoped (ovspatialquery.h). Two instances created
// one after the other, with one operation slot each, must not resolve each
// other's IDs at any generation the other slot reaches.
void testOperationIdsDoNotAliasAcrossInstances()
{
    ovspatialquery_instance_handle_t first  = createInstance(1u);
    ovspatialquery_instance_handle_t second = createInstance(1u);

    uint64_t                                       definition = 41u;
    double                                         radius     = 1.0;
    InputArray                                     values(&definition, 1, kDLUInt);
    InputArray                                     radii(&radius, 1, kDLFloat);
    const ovspatialquery_sphere_definition_table_t spheres{&values.value, &radii.value};
    const ovspatialquery_enqueue_result_t          held = ovspatialquery_enqueue_sphere_registration(first, &spheres);
    REQUIRE(held.status == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_op_wait_result_t wait{};
    REQUIRE(ovspatialquery_wait_op(first, held.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &wait).status ==
            OVSPATIALQUERY_API_SUCCESS);

    for (uint64_t value = 1000u; value < 1256u; ++value)
    {
        uint64_t                                       secondDefinition = value;
        InputArray                                     secondValues(&secondDefinition, 1, kDLUInt);
        InputArray                                     secondRadii(&radius, 1, kDLFloat);
        const ovspatialquery_sphere_definition_table_t secondSpheres{&secondValues.value, &secondRadii.value};
        const ovspatialquery_enqueue_result_t live = ovspatialquery_enqueue_sphere_registration(second, &secondSpheres);
        REQUIRE(live.status == OVSPATIALQUERY_API_SUCCESS);
        ovspatialquery_op_wait_result_t foreign{};
        REQUIRE(ovspatialquery_wait_op(second, held.op_id, 0u, &foreign).status == OVSPATIALQUERY_API_INVALID_HANDLE);
        REQUIRE(ovspatialquery_wait_op(second, live.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &wait).status ==
                OVSPATIALQUERY_API_SUCCESS);
        ovspatialquery_op_id_t liveId = live.op_id;
        REQUIRE(ovspatialquery_destroy_op(second, &liveId).status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(ovspatialquery_destroy_geometry_definition(second, {OVSPATIALQUERY_GEOMETRY_SPHERE, value}).status ==
                OVSPATIALQUERY_API_SUCCESS);
    }

    ovspatialquery_op_id_t heldId = held.op_id;
    REQUIRE(ovspatialquery_destroy_op(first, &heldId).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_geometry_definition(first, {OVSPATIALQUERY_GEOMETRY_SPHERE, definition}).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(second).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(first).status == OVSPATIALQUERY_API_SUCCESS);
}

void testOperationCapacityReuse()
{
    // Two workers, so both admitted updates can reach the completion gate
    // and hold both operation slots at once.
    ovspatialquery_instance_handle_t   instance    = createInstance(2u, 2u);
    ovspatialquery_population_handle_t populationA = createExplicitPopulation(instance, 2u);
    ovspatialquery_population_handle_t populationB = createExplicitPopulation(instance, 2u);
    InstanceRows                       initialA;
    waitAndDestroy(instance, enqueueSet(instance, populationA, 1, initialA));
    InstanceRows initialB;
    waitAndDestroy(instance, enqueueSet(instance, populationB, 1, initialB));

    // Pausing execution keeps both admitted updates pending, so both slots
    // stay occupied without depending on backend gates.
    REQUIRE(ovspatialquery::detail::setInstanceOperationExecutionsPausedForTest(instance, true) ==
            OVSPATIALQUERY_API_SUCCESS);
    InstanceRows                          changedA;
    InstanceRows                          changedB;
    InputArray                            removedA;
    InputArray                            removedB;
    const ovspatialquery_enqueue_result_t first =
        enqueueUpdate(instance, populationA, 1, 2, changedA, emptyRemoved(removedA));
    const ovspatialquery_enqueue_result_t second =
        enqueueUpdate(instance, populationB, 1, 2, changedB, emptyRemoved(removedB));
    REQUIRE(first.status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(second.status == OVSPATIALQUERY_API_SUCCESS);

    ovspatialquery_population_handle_t    rejectedPopulation = createExplicitPopulation(instance, 2u);
    InstanceRows                          rejectedRows;
    const ovspatialquery_enqueue_result_t rejected = enqueueSet(instance, rejectedPopulation, 5, rejectedRows);
    if (rejected.status != OVSPATIALQUERY_API_OPERATION_CAPACITY)
    {
        std::fprintf(stderr, "third admission status %u, expected OPERATION_CAPACITY\n",
                     static_cast<unsigned>(rejected.status));
    }
    REQUIRE(rejected.status == OVSPATIALQUERY_API_OPERATION_CAPACITY);
    REQUIRE(rejected.op_id == OVSPATIALQUERY_INVALID_OP);

    REQUIRE(ovspatialquery::detail::setInstanceOperationExecutionsPausedForTest(instance, false) ==
            OVSPATIALQUERY_API_SUCCESS);
    waitAndDestroy(instance, first);

    // The slot first freed by destroying the first update is available for
    // reuse; occupying both slots again rejects a further admission.
    ovspatialquery_population_handle_t    reusedPopulation = createExplicitPopulation(instance, 2u);
    InstanceRows                          reusedRows;
    const ovspatialquery_enqueue_result_t reused = enqueueSet(instance, reusedPopulation, 6, reusedRows);
    REQUIRE(reused.status == OVSPATIALQUERY_API_SUCCESS);

    ovspatialquery_population_handle_t    overflowPopulation = createExplicitPopulation(instance, 2u);
    InstanceRows                          overflowRows;
    const ovspatialquery_enqueue_result_t overflow = enqueueSet(instance, overflowPopulation, 7, overflowRows);
    REQUIRE(overflow.status == OVSPATIALQUERY_API_OPERATION_CAPACITY);
    REQUIRE(overflow.op_id == OVSPATIALQUERY_INVALID_OP);

    waitAndDestroy(instance, reused);
    waitAndDestroy(instance, second);

    REQUIRE(ovspatialquery_destroy_population(instance, populationA).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_population(instance, populationB).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_population(instance, rejectedPopulation).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_population(instance, reusedPopulation).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_population(instance, overflowPopulation).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

// Promise: for OVSPATIALQUERY_INPUT_COPIED_ON_SUBMISSION, "The library
// copies the logical contents before enqueue returns successfully ... the
// caller can immediately reuse a temporary input array."
void testUpdateInputCopy()
{
    ovspatialquery_instance_handle_t   instance         = createInstance(2u);
    ovspatialquery_backend_runtime_t*  runtime          = ovspatialquery_test_oracle_get_default_runtime();
    const uint64_t                     sphereDefinition = registerUnitSphere(instance, 600u);
    ovspatialquery_population_handle_t population       = createExplicitPopulation(instance, 2u);
    InstanceRows                       initial;
    waitAndDestroy(instance, enqueueSet(instance, population, 1, initial));

    ovspatialquery_test_oracle_block_before_completion(runtime, 2u, 1u);
    InstanceRows changed;
    changed.add(500u, OVSPATIALQUERY_GEOMETRY_SPHERE, sphereDefinition, 0.0, 0.0, 0.0);
    InputArray                            removed;
    const ovspatialquery_enqueue_result_t admission =
        enqueueUpdate(instance, population, 1, 2, changed, emptyRemoved(removed));
    REQUIRE(admission.status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_test_oracle_wait_before_completion(runtime) == 1u);

    // Admission already copied the columns; overwriting or freeing the
    // caller's buffers now must not affect the pending publication.
    changed.objects[0]      = 999u;
    changed.translations[0] = 500.0;
    std::vector<uint64_t>().swap(changed.objects);
    std::vector<double>().swap(changed.translations);

    ovspatialquery_test_oracle_release_before_completion(runtime);
    waitAndDestroy(instance, admission);

    ovspatialquery_raycast_desc_t query{};
    query.common.target         = populationTarget(population);
    query.common.precision      = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    query.ray.origin[0]         = -10.0;
    query.ray.unit_direction[0] = 1.0f;
    query.ray.max_distance      = 100.0;
    ovspatialquery_raycast_single_result_t result{};
    REQUIRE(ovspatialquery_raycast_closest_sync(instance, &query, &result).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(result.has_hit == 1u);
    REQUIRE(result.hit.truth == OVSPATIALQUERY_HIT_BOUNDS_CANDIDATE);
    REQUIRE(result.hit.data.bounds_candidate.identity.object_id.value == 500u);

    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_SPHERE, sphereDefinition})
                .status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

// Promise: publication boundaries ("Publishing a population creates an
// immutable state identified by an ordinal") are atomic and observable
// before the admitting operation itself becomes terminal; and "Publication
// also fails atomically" with isolated failures (ovspatialquery_explicit.h).
void testPublicationBoundariesAndFailureIsolation()
{
    ovspatialquery_instance_handle_t   instance         = createInstance(2u);
    ovspatialquery_backend_runtime_t*  runtime          = ovspatialquery_test_oracle_get_default_runtime();
    const uint64_t                     sphereDefinition = registerUnitSphere(instance, 110u);
    ovspatialquery_population_handle_t population       = createExplicitPopulation(instance, 2u);
    InstanceRows                       initial;
    initial.add(110u, OVSPATIALQUERY_GEOMETRY_SPHERE, sphereDefinition, 0.0, 0.0, 0.0);
    waitAndDestroy(instance, enqueueSet(instance, population, 10, initial));

    ovspatialquery_test_oracle_block_before_publish(runtime, 11);
    ovspatialquery_test_oracle_block_after_publish(runtime, 11);
    InstanceRows changed;
    changed.add(111u, OVSPATIALQUERY_GEOMETRY_SPHERE, sphereDefinition, 1.0, 0.0, 0.0);
    InputArray                            removed;
    const ovspatialquery_enqueue_result_t admission =
        enqueueUpdate(instance, population, 10, 11, changed, emptyRemoved(removed));
    REQUIRE(admission.status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_test_oracle_wait_before_publish(runtime) == 1u);

    ovspatialquery_population_info_t beforePublish{};
    REQUIRE(ovspatialquery_get_population_info(instance, population, &beforePublish).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(beforePublish.head_ordinal == 10);
    ovspatialquery_op_wait_result_t stillPending{};
    REQUIRE(ovspatialquery_wait_op(instance, admission.op_id, 0u, &stillPending).status == OVSPATIALQUERY_API_TIMEOUT);

    ovspatialquery_test_oracle_release_before_publish(runtime);
    REQUIRE(ovspatialquery_test_oracle_wait_after_publish(runtime) == 1u);

    ovspatialquery_population_info_t afterPublish{};
    REQUIRE(ovspatialquery_get_population_info(instance, population, &afterPublish).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(afterPublish.head_ordinal == 11);
    REQUIRE(ovspatialquery_wait_op(instance, admission.op_id, 0u, &stillPending).status == OVSPATIALQUERY_API_TIMEOUT);

    ovspatialquery_test_oracle_release_after_publish(runtime);
    waitAndDestroy(instance, admission);

    // A build failure after admission is atomic and isolated: the head stays
    // at the last successful publication and a later update proceeds.
    ovspatialquery_test_oracle_fail_next_build(runtime, OVSPATIALQUERY_API_INTERNAL);
    InstanceRows failing;
    failing.add(112u, OVSPATIALQUERY_GEOMETRY_SPHERE, sphereDefinition, 2.0, 0.0, 0.0);
    InputArray                            removedForFailure;
    const ovspatialquery_enqueue_result_t failingAdmission =
        enqueueUpdate(instance, population, 11, 12, failing, emptyRemoved(removedForFailure));
    REQUIRE(failingAdmission.status == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_op_wait_result_t failingWait{};
    REQUIRE(ovspatialquery_wait_op(instance, failingAdmission.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &failingWait)
                .status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(failingWait.state == OVSPATIALQUERY_OP_FAILURE);
    ovspatialquery_op_id_t failingId = failingAdmission.op_id;
    REQUIRE(ovspatialquery_destroy_op(instance, &failingId).status == OVSPATIALQUERY_API_SUCCESS);

    ovspatialquery_population_info_t infoAfterFailure{};
    REQUIRE(ovspatialquery_get_population_info(instance, population, &infoAfterFailure).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(infoAfterFailure.head_ordinal == 11 && infoAfterFailure.object_count == 2u);

    InstanceRows recovered;
    recovered.add(113u, OVSPATIALQUERY_GEOMETRY_SPHERE, sphereDefinition, 3.0, 0.0, 0.0);
    InputArray removedForRecovery;
    waitAndDestroy(instance, enqueueUpdate(instance, population, 11, 12, recovered, emptyRemoved(removedForRecovery)));
    ovspatialquery_population_info_t infoAfterRecovery{};
    REQUIRE(ovspatialquery_get_population_info(instance, population, &infoAfterRecovery).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(infoAfterRecovery.head_ordinal == 12 && infoAfterRecovery.object_count == 3u);

    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_SPHERE, sphereDefinition})
                .status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

} // namespace

int main()
{
    testRaycastPlanRetainsTargetPopulationUntilTerminal();
    testCompletionLinearizesMutationAndPopulationRelease();
    testOperationHandleBusyStatesAndCapacityRelease();
    testOperationIdsDoNotAliasAcrossInstances();
    testOperationCapacityReuse();
    testUpdateInputCopy();
    testPublicationBoundariesAndFailureIsolation();
    std::puts("operation acceptance API test passed");
    return 0;
}
