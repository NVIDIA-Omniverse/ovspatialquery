/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <ovspatialquery/ovspatialquery.h>
#include <ovspatialquery/ovspatialquery_explicit.h>
#include <ovspatialquery/ovspatialquery_pairs.h>

#include "../../support/native/public_api_inputs.h"
#include "../../support/native/test_failure.h"
#include "ovspatialquery_instance_operation_runtime.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>

extern "C" void ovspatialquery_test_current_api_pause_fetch_projection(uint32_t kind, bool paused);
extern "C" void ovspatialquery_test_current_api_wait_fetch_projection(uint32_t kind);

namespace
{

template <typename T>
bool hasIdenticalBytes(const T& left, const T& right)
{
    return std::memcmp(&left, &right, sizeof(T)) == 0;
}

bool waitForFlag(const std::atomic<bool>& flag)
{
    const std::chrono::steady_clock::time_point deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
    while (!flag.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline)
    {
        std::this_thread::yield();
    }
    return flag.load(std::memory_order_acquire);
}

void waitForWaiterEntry(ovspatialquery_instance_handle_t instance, uint64_t previousCount)
{
    const std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (ovspatialquery::detail::instanceOperationWaiterEntryCountForTest(instance) == previousCount &&
           std::chrono::steady_clock::now() < deadline)
    {
        std::this_thread::yield();
    }
    REQUIRE(ovspatialquery::detail::instanceOperationWaiterEntryCountForTest(instance) != previousCount);
}

void observeCompletedOperation(ovspatialquery_instance_handle_t instance, ovspatialquery_op_id_t operation)
{
    ovspatialquery_result_t    statusResult{};
    ovspatialquery_op_status_t status{};
    do
    {
        statusResult = ovspatialquery_query_op_status(instance, operation, &status);
        std::this_thread::yield();
    } while (statusResult.status == OVSPATIALQUERY_API_SUCCESS && status.state == OVSPATIALQUERY_OP_PENDING);
    REQUIRE(statusResult.status == OVSPATIALQUERY_API_SUCCESS && status.state == OVSPATIALQUERY_OP_COMPLETED);
}

uint64_t createAndMeasureInstance(uint32_t maxCpuConcurrency)
{
    ovspatialquery_instance_desc_t desc{};
    desc.operation_capacity                   = 2u;
    desc.max_cpu_concurrency                  = maxCpuConcurrency;
    desc.host_byte_budget                     = UINT64_C(64) << 20;
    ovspatialquery_instance_handle_t instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&desc, &instance).status == OVSPATIALQUERY_API_SUCCESS &&
            instance != OVSPATIALQUERY_INVALID_INSTANCE);
    uint64_t usage = 0u;
    REQUIRE(ovspatialquery_get_instance_memory_usage(instance, DLDevice{kDLCPU, 0}, &usage).status ==
                OVSPATIALQUERY_API_SUCCESS &&
            usage != 0u);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
    return usage;
}

ovspatialquery_instance_handle_t createInstance(uint64_t budget)
{
    ovspatialquery_instance_desc_t desc{};
    desc.operation_capacity                   = 1u;
    desc.max_cpu_concurrency                  = 1u;
    desc.host_byte_budget                     = budget;
    ovspatialquery_instance_handle_t instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&desc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    return instance;
}

ovspatialquery_population_handle_t createEmptyPopulation(ovspatialquery_instance_handle_t instance)
{
    const ovspatialquery_population_desc_t desc{1u, 1u};
    ovspatialquery_population_handle_t     population = OVSPATIALQUERY_INVALID_POPULATION;
    REQUIRE(ovspatialquery_create_explicit_population(instance, &desc, &population).status ==
                OVSPATIALQUERY_API_SUCCESS &&
            population != OVSPATIALQUERY_INVALID_POPULATION);
    return population;
}

constexpr uint64_t kUnitSphere = 1u;

// Registers the unit sphere every object published below binds.
void registerUnitSphere(ovspatialquery_instance_handle_t instance)
{
    uint64_t                                       value  = kUnitSphere;
    double                                         radius = 1.0;
    InputArray                                     values(&value, 1, kDLUInt);
    InputArray                                     radii(&radius, 1, kDLFloat);
    const ovspatialquery_sphere_definition_table_t spheres{&values.value, &radii.value};
    waitAndDestroy(instance, ovspatialquery_enqueue_sphere_registration(instance, &spheres));
}

void destroyUnitSphereAndInstance(ovspatialquery_instance_handle_t instance)
{
    REQUIRE(
        ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_SPHERE, kUnitSphere}).status ==
            OVSPATIALQUERY_API_SUCCESS &&
        ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

// Publishes `objectId` as the unit sphere centered at (center, center, center),
// so its bounds span [center - 1, center + 1] on every axis: as the first set
// at `ordinal` when `base` is negative, otherwise as an update of head `base`.
void publishUnitSphere(ovspatialquery_instance_handle_t instance, ovspatialquery_population_handle_t population,
                       ovspatialquery_ordinal_t base, ovspatialquery_ordinal_t ordinal, uint64_t objectId,
                       double center)
{
    int32_t                                        type           = OVSPATIALQUERY_GEOMETRY_SPHERE;
    uint64_t                                       geometry       = kUnitSphere;
    double                                         translation[3] = {center, center, center};
    float                                          rotation[4]    = {0.0f, 0.0f, 0.0f, 1.0f};
    uint64_t                                       none           = 0u;
    InputArray                                     objects(&objectId, 1, kDLUInt);
    InputArray                                     types(&type, 1, kDLInt);
    InputArray                                     geometries(&geometry, 1, kDLUInt);
    InputArray                                     translations(translation, 1, 3, kDLFloat);
    InputArray                                     rotations(rotation, 1, 4, kDLFloat);
    InputArray                                     removed(&none, 0, kDLUInt);
    const ovspatialquery_explicit_instance_table_t row{
        &objects.value,   &types.value, &geometries.value, &translations.value,
        &rotations.value, nullptr,      nullptr,           nullptr};
    if (base < 0)
    {
        const ovspatialquery_explicit_population_set_t set{ordinal, row, nullptr, 0u};
        waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set));
        return;
    }
    const ovspatialquery_explicit_population_update_t update{
        base, ordinal, OVSPATIALQUERY_EXPLICIT_UPDATE_COMPLETE_ROWS, row, &removed.value, nullptr};
    waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_update(instance, population, &update));
}

ovspatialquery_batch_plan_handle_t createOverlapPlan(ovspatialquery_instance_handle_t   instance,
                                                     ovspatialquery_population_handle_t population)
{
    ovspatialquery_overlap_plan_desc_t desc{};
    desc.target                             = populationRef(population);
    desc.hit_mode                           = OVSPATIALQUERY_HIT_MODE_ANY;
    desc.precision                          = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    desc.shape.kind                         = OVSPATIALQUERY_OVERLAP_SPHERE;
    desc.shape.data.sphere.radius           = 1.0;
    desc.device                             = DLDevice{kDLCPU, 0};
    desc.maximum_query_count                = 1u;
    desc.maximum_hits_per_query             = 1u;
    desc.hit_capacity_per_slot              = 1u;
    desc.result_slot_count                  = 1u;
    ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_overlap_plan(instance, &desc, &plan).status == OVSPATIALQUERY_API_SUCCESS);
    return plan;
}

void requireRaycastAndSweepReservationRuntime()
{
    const ovspatialquery_instance_handle_t instance = createInstance(UINT64_C(64) << 20);
    registerUnitSphere(instance);
    const ovspatialquery_population_handle_t population = createEmptyPopulation(instance);
    publishUnitSphere(instance, population, -1, 23, 1u, 0.0);

    double     points[3]     = {-2.0, 0.0, 0.0};
    float      directions[3] = {1.0f, 0.0f, 0.0f};
    double     distances[1]  = {10.0};
    InputArray pointsInput(points, 1, 3, kDLFloat);
    InputArray directionsInput(directions, 1, 3, kDLFloat);
    InputArray distancesInput(distances, 1, kDLFloat);

    ovspatialquery_raycast_plan_desc_t raycastDesc{};
    raycastDesc.target                         = populationRef(population);
    raycastDesc.hit_mode                       = OVSPATIALQUERY_HIT_MODE_ANY;
    raycastDesc.precision                      = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    raycastDesc.device                         = DLDevice{kDLCPU, 0};
    raycastDesc.maximum_query_count            = 1u;
    raycastDesc.maximum_hits_per_query         = 1u;
    raycastDesc.hit_capacity_per_slot          = 1u;
    raycastDesc.result_slot_count              = 1u;
    ovspatialquery_batch_plan_handle_t raycast = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_raycast_plan(instance, &raycastDesc, &raycast).status == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_raycast_batch_t raycastBatch{};
    raycastBatch.ordinal.kind    = OVSPATIALQUERY_ORDINAL_HEAD;
    raycastBatch.query_count     = 1u;
    raycastBatch.origins         = &pointsInput.value;
    raycastBatch.unit_directions = &directionsInput.value;
    raycastBatch.max_distances   = &distancesInput.value;
    ovspatialquery_raycast_batch_reservation_t raycastRejected{};
    raycastRejected.common.query_count                                = 77u;
    const ovspatialquery_raycast_batch_reservation_t preservedRaycast = raycastRejected;
    ovspatialquery_raycast_batch_t                   oversizedRaycast = raycastBatch;
    oversizedRaycast.query_count                                      = 2u;
    REQUIRE(ovspatialquery_enqueue_raycast_plan(instance, raycast, &oversizedRaycast, &raycastRejected).status ==
                OVSPATIALQUERY_API_INVALID_ARGUMENT &&
            hasIdenticalBytes(raycastRejected, preservedRaycast));
    ovspatialquery_raycast_batch_reservation_t raycastReservation{};
    const ovspatialquery_enqueue_result_t      raycastAdmission =
        ovspatialquery_enqueue_raycast_plan(instance, raycast, &raycastBatch, &raycastReservation);
    REQUIRE(raycastAdmission.status == OVSPATIALQUERY_API_SUCCESS && raycastReservation.common.query_count == 1u &&
            raycastReservation.common.source_ordinal == 23 &&
            raycastReservation.columns.hit_ranges.query_status.data != nullptr &&
            raycastReservation.columns.hit_ranges.query_status.shape[0] == 1 &&
            raycastReservation.columns.object_ids.data != nullptr &&
            raycastReservation.columns.object_ids.shape[0] == 1);
    const ovspatialquery_raycast_batch_reservation_t savedRaycast = raycastReservation;
    observeCompletedOperation(instance, raycastAdmission.op_id);
    ovspatialquery_raycast_batch_result_t raycastResult{};
    REQUIRE(ovspatialquery_fetch_raycast_batch(instance, raycastAdmission.op_id, &raycastResult).status ==
                OVSPATIALQUERY_API_SUCCESS &&
            raycastResult.common.query_count == 1u && hasIdenticalBytes(raycastReservation, savedRaycast));
    ovspatialquery_op_id_t raycastOperation = raycastAdmission.op_id;
    REQUIRE(ovspatialquery_destroy_op(instance, &raycastOperation).status == OVSPATIALQUERY_API_SUCCESS &&
            raycastOperation == OVSPATIALQUERY_INVALID_OP);

    directions[0] = 2.0f;
    ovspatialquery_raycast_batch_reservation_t invalidRowReservation{};
    const ovspatialquery_enqueue_result_t      invalidRowAdmission =
        ovspatialquery_enqueue_raycast_plan(instance, raycast, &raycastBatch, &invalidRowReservation);
    REQUIRE(invalidRowAdmission.status == OVSPATIALQUERY_API_SUCCESS);
    observeCompletedOperation(instance, invalidRowAdmission.op_id);
    ovspatialquery_raycast_batch_result_t invalidRowResult{};
    REQUIRE(ovspatialquery_fetch_raycast_batch(instance, invalidRowAdmission.op_id, &invalidRowResult).status ==
                OVSPATIALQUERY_API_SUCCESS &&
            invalidRowResult.columns.hit_ranges.query_status.shape[0] == 1 &&
            static_cast<const uint32_t*>(invalidRowResult.columns.hit_ranges.query_status.data)[0] ==
                OVSPATIALQUERY_BATCH_ROW_INVALID_ARGUMENT &&
            invalidRowResult.columns.hit_ranges.hit_counts.shape[0] == 1 &&
            static_cast<const uint64_t*>(invalidRowResult.columns.hit_ranges.hit_counts.data)[0] == 0u);
    ovspatialquery_op_id_t invalidRowOperation = invalidRowAdmission.op_id;
    REQUIRE(ovspatialquery_destroy_op(instance, &invalidRowOperation).status == OVSPATIALQUERY_API_SUCCESS &&
            invalidRowOperation == OVSPATIALQUERY_INVALID_OP &&
            ovspatialquery_destroy_batch_plan(instance, raycast).status == OVSPATIALQUERY_API_SUCCESS);

    ovspatialquery_sphere_sweep_plan_desc_t sweepDesc{};
    sweepDesc.target                         = populationRef(population);
    sweepDesc.precision                      = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    sweepDesc.radius                         = 0.5;
    sweepDesc.device                         = DLDevice{kDLCPU, 0};
    sweepDesc.maximum_query_count            = 1u;
    sweepDesc.result_slot_count              = 1u;
    sweepDesc.hit_capacity_per_slot          = 2u;
    ovspatialquery_batch_plan_handle_t sweep = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_sphere_sweep_plan(instance, &sweepDesc, &sweep).status ==
                OVSPATIALQUERY_API_INVALID_ARGUMENT &&
            sweep == OVSPATIALQUERY_INVALID_BATCH_PLAN);
    sweepDesc.hit_capacity_per_slot = sweepDesc.maximum_query_count;
    REQUIRE(ovspatialquery_create_sphere_sweep_plan(instance, &sweepDesc, &sweep).status == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_sphere_sweep_batch_t sweepBatch{};
    sweepBatch.ordinal.kind    = OVSPATIALQUERY_ORDINAL_HEAD;
    sweepBatch.query_count     = 1u;
    sweepBatch.centers         = &pointsInput.value;
    sweepBatch.unit_directions = &directionsInput.value;
    sweepBatch.max_distances   = &distancesInput.value;
    ovspatialquery_sphere_sweep_batch_reservation_t sweepRejected{};
    sweepRejected.common.query_count                                     = 88u;
    const ovspatialquery_sphere_sweep_batch_reservation_t preservedSweep = sweepRejected;
    ovspatialquery_sphere_sweep_batch_t                   oversizedSweep = sweepBatch;
    oversizedSweep.query_count                                           = 2u;
    REQUIRE(ovspatialquery_enqueue_sphere_sweep_plan(instance, sweep, &oversizedSweep, &sweepRejected).status ==
                OVSPATIALQUERY_API_INVALID_ARGUMENT &&
            hasIdenticalBytes(sweepRejected, preservedSweep));
    ovspatialquery_sphere_sweep_batch_reservation_t sweepReservation{};
    const ovspatialquery_enqueue_result_t           sweepAdmission =
        ovspatialquery_enqueue_sphere_sweep_plan(instance, sweep, &sweepBatch, &sweepReservation);
    REQUIRE(sweepAdmission.status == OVSPATIALQUERY_API_SUCCESS && sweepReservation.common.query_count == 1u &&
            sweepReservation.common.source_ordinal == 23 &&
            sweepReservation.columns.hit_ranges.query_status.data != nullptr &&
            sweepReservation.columns.hit_ranges.query_status.shape[0] == 1 &&
            sweepReservation.columns.object_ids.data != nullptr && sweepReservation.columns.object_ids.shape[0] == 1);
    const ovspatialquery_sphere_sweep_batch_reservation_t savedSweep = sweepReservation;
    observeCompletedOperation(instance, sweepAdmission.op_id);
    ovspatialquery_sphere_sweep_batch_result_t sweepResult{};
    REQUIRE(ovspatialquery_fetch_sphere_sweep_batch(instance, sweepAdmission.op_id, &sweepResult).status ==
                OVSPATIALQUERY_API_SUCCESS &&
            sweepResult.common.query_count == 1u && hasIdenticalBytes(sweepReservation, savedSweep));
    ovspatialquery_op_id_t sweepOperation = sweepAdmission.op_id;
    REQUIRE(ovspatialquery_destroy_op(instance, &sweepOperation).status == OVSPATIALQUERY_API_SUCCESS &&
            sweepOperation == OVSPATIALQUERY_INVALID_OP &&
            ovspatialquery_destroy_batch_plan(instance, sweep).status == OVSPATIALQUERY_API_SUCCESS &&
            ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    destroyUnitSphereAndInstance(instance);
}

void destroyEmptyPlanFixture(ovspatialquery_instance_handle_t instance, ovspatialquery_population_handle_t population,
                             ovspatialquery_batch_plan_handle_t plan)
{
    REQUIRE(ovspatialquery_destroy_batch_plan(instance, plan).status == OVSPATIALQUERY_API_SUCCESS &&
            ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS &&
            ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

void requirePlanAdmissionPrecedenceAndCheckedOffsets()
{
    constexpr uint64_t                 generousBudget = UINT64_C(64) << 20;
    ovspatialquery_instance_handle_t   instance       = createInstance(generousBudget);
    ovspatialquery_population_handle_t population     = createEmptyPopulation(instance);
    ovspatialquery_batch_plan_handle_t plan           = createOverlapPlan(instance, population);
    uint64_t                           exactBudget    = 0u;
    REQUIRE(ovspatialquery_get_instance_memory_usage(instance, DLDevice{kDLCPU, 0}, &exactBudget).status ==
            OVSPATIALQUERY_API_SUCCESS);
    destroyEmptyPlanFixture(instance, population, plan);

    instance            = createInstance(exactBudget);
    population          = createEmptyPopulation(instance);
    plan                = createOverlapPlan(instance, population);
    uint64_t exactUsage = 0u;
    REQUIRE(ovspatialquery_get_instance_memory_usage(instance, DLDevice{kDLCPU, 0}, &exactUsage).status ==
                OVSPATIALQUERY_API_SUCCESS &&
            exactUsage == exactBudget);

    ovspatialquery_overlap_batch_t empty{};
    empty.ordinal.kind = OVSPATIALQUERY_ORDINAL_HEAD;
    ovspatialquery_overlap_batch_reservation_t preserved{};
    preserved.common.query_count                                       = 77u;
    const ovspatialquery_overlap_batch_reservation_t notReadyPreserved = preserved;
    const ovspatialquery_enqueue_result_t            notReady =
        ovspatialquery_enqueue_overlap_plan(instance, plan, &empty, &preserved);
    REQUIRE(notReady.status == OVSPATIALQUERY_API_NOT_READY && notReady.op_id == OVSPATIALQUERY_INVALID_OP &&
            hasIdenticalBytes(preserved, notReadyPreserved));

    double                       translation[3]{};
    int64_t                      translationShape[2]   = {1, 3};
    int64_t                      translationStrides[2] = {std::numeric_limits<int64_t>::max(), 1};
    ovspatialquery_input_array_t translations{};
    translations.tensor.data        = translation;
    translations.tensor.device      = DLDevice{kDLCPU, 0};
    translations.tensor.ndim        = 2;
    translations.tensor.dtype       = DLDataType{kDLFloat, 64u, 1u};
    translations.tensor.shape       = translationShape;
    translations.tensor.strides     = translationStrides;
    translations.tensor.byte_offset = std::numeric_limits<uint64_t>::max();
    translations.ownership          = OVSPATIALQUERY_INPUT_COPIED_ON_SUBMISSION;
    ovspatialquery_overlap_batch_t malformed{};
    malformed.ordinal.kind                                            = OVSPATIALQUERY_ORDINAL_HEAD;
    malformed.query_count                                             = 1u;
    malformed.translations                                            = &translations;
    preserved.common.query_count                                      = 88u;
    const ovspatialquery_overlap_batch_reservation_t invalidPreserved = preserved;
    const ovspatialquery_enqueue_result_t            invalid =
        ovspatialquery_enqueue_overlap_plan(instance, plan, &malformed, &preserved);
    REQUIRE(invalid.status == OVSPATIALQUERY_API_INVALID_ARGUMENT && invalid.op_id == OVSPATIALQUERY_INVALID_OP &&
            hasIdenticalBytes(preserved, invalidPreserved));
    destroyEmptyPlanFixture(instance, population, plan);
}

void requirePairHostBudgetAdmissionAndRelease()
{
    constexpr uint64_t               generousBudget = UINT64_C(64) << 20;
    ovspatialquery_instance_handle_t instance       = createInstance(generousBudget);
    registerUnitSphere(instance);
    const ovspatialquery_population_handle_t first  = createEmptyPopulation(instance);
    const ovspatialquery_population_handle_t second = createEmptyPopulation(instance);
    publishUnitSphere(instance, first, -1, 7, 1u, 1.0);
    publishUnitSphere(instance, second, -1, 11, 2u, 2.0);

    ovspatialquery_pair_snapshot_desc_t desc{};
    desc.first  = populationTarget(first);
    desc.second = populationTarget(second);
    ovspatialquery_pair_t pair{};
    pair.first.value  = UINT64_C(0x1111222233334444);
    pair.second.value = UINT64_C(0x5555666677778888);
    ovspatialquery_pair_buffer_t buffer{};
    buffer.pairs    = &pair;
    buffer.capacity = UINT64_C(1) << 30;
    ovspatialquery_pair_snapshot_result_t result{};
    result.first_source_ordinal  = 101;
    result.second_source_ordinal = 202;
    result.stored_pair_count     = 303u;
    result.truncated             = 1u;
    uint64_t usageBefore         = 0u;
    REQUIRE(ovspatialquery_get_instance_memory_usage(instance, DLDevice{kDLCPU, 0}, &usageBefore).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_query_pairs_sync(instance, &desc, &buffer, &result).status ==
            OVSPATIALQUERY_API_HOST_BUDGET);
    uint64_t usageAfterRejection = 0u;
    REQUIRE(result.first_source_ordinal == 101 && result.second_source_ordinal == 202 &&
            result.stored_pair_count == 303u && result.truncated == 1u &&
            pair.first.value == UINT64_C(0x1111222233334444) && pair.second.value == UINT64_C(0x5555666677778888) &&
            ovspatialquery_get_instance_memory_usage(instance, DLDevice{kDLCPU, 0}, &usageAfterRejection).status ==
                OVSPATIALQUERY_API_SUCCESS &&
            usageAfterRejection == usageBefore);

    buffer.capacity = 1u;
    result          = {};
    REQUIRE(ovspatialquery_query_pairs_sync(instance, &desc, &buffer, &result).status == OVSPATIALQUERY_API_SUCCESS);
    uint64_t usageAfterSuccess = 0u;
    REQUIRE(result.first_source_ordinal == 7 && result.second_source_ordinal == 11 && result.stored_pair_count == 1u &&
            result.truncated == 0u && pair.first.value == 1u && pair.second.value == 2u &&
            ovspatialquery_get_instance_memory_usage(instance, DLDevice{kDLCPU, 0}, &usageAfterSuccess).status ==
                OVSPATIALQUERY_API_SUCCESS &&
            usageAfterSuccess == usageBefore);
    REQUIRE(ovspatialquery_destroy_population(instance, first).status == OVSPATIALQUERY_API_SUCCESS &&
            ovspatialquery_destroy_population(instance, second).status == OVSPATIALQUERY_API_SUCCESS);
    destroyUnitSphereAndInstance(instance);
}

void requireConcurrentPlanFacadeLifecycle()
{
    ovspatialquery_instance_handle_t instance = createInstance(UINT64_C(64) << 20);
    registerUnitSphere(instance);
    const ovspatialquery_population_handle_t population = createEmptyPopulation(instance);
    publishUnitSphere(instance, population, -1, 17, 1u, 0.0);

    double                         translation[3]{};
    InputArray                     translations(translation, 1, 3, kDLFloat);
    ovspatialquery_overlap_batch_t batch{};
    batch.ordinal.kind = OVSPATIALQUERY_ORDINAL_HEAD;
    batch.query_count  = 1u;
    batch.translations = &translations.value;

    for (uint32_t iteration = 0u; iteration < 100u; ++iteration)
    {
        const ovspatialquery_batch_plan_handle_t   plan = createOverlapPlan(instance, population);
        std::atomic<bool>                          start{false};
        ovspatialquery_enqueue_result_t            admission{};
        ovspatialquery_overlap_batch_reservation_t early{};
        ovspatialquery_result_t                    planDestroy{};
        std::thread                                enqueueThread(
            [&]()
            {
                while (!start.load(std::memory_order_acquire))
                {
                    std::this_thread::yield();
                }
                admission = ovspatialquery_enqueue_overlap_plan(instance, plan, &batch, &early);
            });
        std::thread planDestroyThread(
            [&]()
            {
                while (!start.load(std::memory_order_acquire))
                {
                    std::this_thread::yield();
                }
                planDestroy = ovspatialquery_destroy_batch_plan(instance, plan);
            });
        start.store(true, std::memory_order_release);
        enqueueThread.join();
        planDestroyThread.join();

        if (admission.status == OVSPATIALQUERY_API_SUCCESS)
        {
            REQUIRE(planDestroy.status == OVSPATIALQUERY_API_BUSY);
            const ovspatialquery_op_id_t operation = admission.op_id;
            std::atomic<bool>            statusDone{false};
            ovspatialquery_result_t      statusResult{};
            ovspatialquery_op_status_t   status{};
            std::thread                  statusThread(
                [&]()
                {
                    do
                    {
                        statusResult = ovspatialquery_query_op_status(instance, operation, &status);
                        std::this_thread::yield();
                    } while (statusResult.status == OVSPATIALQUERY_API_SUCCESS &&
                             status.state == OVSPATIALQUERY_OP_PENDING);
                    statusDone.store(true, std::memory_order_release);
                });
            ovspatialquery_op_id_t  destroyOperation = operation;
            ovspatialquery_result_t operationDestroy{};
            std::thread             operationDestroyThread(
                [&]()
                {
                    do
                    {
                        operationDestroy = ovspatialquery_destroy_op(instance, &destroyOperation);
                        std::this_thread::yield();
                    } while (operationDestroy.status == OVSPATIALQUERY_API_BUSY);
                });
            statusThread.join();
            operationDestroyThread.join();
            REQUIRE(statusDone.load(std::memory_order_acquire) && statusResult.status == OVSPATIALQUERY_API_SUCCESS &&
                    status.state == OVSPATIALQUERY_OP_COMPLETED &&
                    operationDestroy.status == OVSPATIALQUERY_API_SUCCESS &&
                    destroyOperation == OVSPATIALQUERY_INVALID_OP &&
                    ovspatialquery_destroy_batch_plan(instance, plan).status == OVSPATIALQUERY_API_SUCCESS);
        }
        else
        {
            REQUIRE(admission.status == OVSPATIALQUERY_API_INVALID_HANDLE &&
                    admission.op_id == OVSPATIALQUERY_INVALID_OP && planDestroy.status == OVSPATIALQUERY_API_SUCCESS);
        }
    }

    const ovspatialquery_batch_plan_handle_t waitingPlan = createOverlapPlan(instance, population);
    REQUIRE(ovspatialquery::detail::setInstanceOperationExecutionsPausedForTest(instance, true) ==
            OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_overlap_batch_reservation_t waitingEarly{};
    const ovspatialquery_enqueue_result_t      waitingAdmission =
        ovspatialquery_enqueue_overlap_plan(instance, waitingPlan, &batch, &waitingEarly);
    REQUIRE(waitingAdmission.status == OVSPATIALQUERY_API_SUCCESS);
    // The worker must own this paused operation before the infinite waiter
    // starts; otherwise the waiter can claim it and pause before its wait-entry
    // checkpoint, which this thread waits for before releasing execution.
    const auto workerDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    ovspatialquery::detail::InstanceOperationRuntimeInspection workerState{};
    do
    {
        REQUIRE(ovspatialquery::detail::inspectInstanceOperationRuntimeForTest(instance, workerState) ==
                OVSPATIALQUERY_API_SUCCESS);
        if (workerState.pendingOperationCount == 0u && workerState.runningOperationCount == 1u)
        {
            break;
        }
        std::this_thread::yield();
    } while (std::chrono::steady_clock::now() < workerDeadline);
    REQUIRE(workerState.pendingOperationCount == 0u && workerState.runningOperationCount == 1u);
    const uint64_t          waiterCount = ovspatialquery::detail::instanceOperationWaiterEntryCountForTest(instance);
    ovspatialquery_result_t waitResult{};
    ovspatialquery_op_wait_result_t waitState{};
    std::thread                     waitThread(
        [&]()
        {
            waitResult =
                ovspatialquery_wait_op(instance, waitingAdmission.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waitState);
        });
    waitForWaiterEntry(instance, waiterCount);
    std::atomic<bool>       destroyReturned{false};
    ovspatialquery_result_t waitingDestroy{};
    std::thread             destroyThread(
        [&]()
        {
            waitingDestroy = ovspatialquery_destroy_batch_plan(instance, waitingPlan);
            destroyReturned.store(true, std::memory_order_release);
        });
    const bool destroyWasPrompt = waitForFlag(destroyReturned);
    REQUIRE(ovspatialquery::detail::setInstanceOperationExecutionsPausedForTest(instance, false) ==
            OVSPATIALQUERY_API_SUCCESS);
    waitThread.join();
    destroyThread.join();
    REQUIRE(destroyWasPrompt && waitingDestroy.status == OVSPATIALQUERY_API_BUSY &&
            waitResult.status == OVSPATIALQUERY_API_SUCCESS && waitState.state == OVSPATIALQUERY_OP_COMPLETED);
    ovspatialquery_op_id_t waitingOperation = waitingAdmission.op_id;
    REQUIRE(ovspatialquery_destroy_op(instance, &waitingOperation).status == OVSPATIALQUERY_API_SUCCESS &&
            waitingOperation == OVSPATIALQUERY_INVALID_OP &&
            ovspatialquery_destroy_batch_plan(instance, waitingPlan).status == OVSPATIALQUERY_API_SUCCESS);

    const ovspatialquery_batch_plan_handle_t   fetchPlan = createOverlapPlan(instance, population);
    ovspatialquery_overlap_batch_reservation_t firstEarly{};
    const ovspatialquery_enqueue_result_t      firstAdmission =
        ovspatialquery_enqueue_overlap_plan(instance, fetchPlan, &batch, &firstEarly);
    REQUIRE(firstAdmission.status == OVSPATIALQUERY_API_SUCCESS && firstEarly.common.query_count == 1u &&
            firstEarly.common.source_ordinal == 17 && firstEarly.columns.hit_ranges.query_status.data != nullptr &&
            firstEarly.columns.hit_ranges.query_status.shape[0] == 1 && firstEarly.columns.object_ids.data != nullptr &&
            firstEarly.columns.object_ids.shape[0] == 1);
    const ovspatialquery_overlap_batch_reservation_t firstReservation = firstEarly;
    observeCompletedOperation(instance, firstAdmission.op_id);
    ovspatialquery_test_current_api_pause_fetch_projection(0u, true);
    ovspatialquery_overlap_batch_result_t firstResult{};
    ovspatialquery_result_t               firstFetch{};
    std::thread                           fetchThread(
        [&]() { firstFetch = ovspatialquery_fetch_overlap_batch(instance, firstAdmission.op_id, &firstResult); });
    ovspatialquery_test_current_api_wait_fetch_projection(0u);
    ovspatialquery_op_id_t  firstOperation = firstAdmission.op_id;
    ovspatialquery_result_t firstDestroy{};
    std::atomic<bool>       operationDestroyReturned{false};
    std::thread             operationDestroyThread(
        [&]()
        {
            firstDestroy = ovspatialquery_destroy_op(instance, &firstOperation);
            operationDestroyReturned.store(true, std::memory_order_release);
        });
    const bool destroyRacedPastFetch = waitForFlag(operationDestroyReturned);
    ovspatialquery_test_current_api_pause_fetch_projection(0u, false);
    fetchThread.join();
    operationDestroyThread.join();
    REQUIRE(!destroyRacedPastFetch && firstFetch.status == OVSPATIALQUERY_API_SUCCESS &&
            firstResult.common.query_count == 1u && hasIdenticalBytes(firstEarly, firstReservation) &&
            firstDestroy.status == OVSPATIALQUERY_API_SUCCESS && firstOperation == OVSPATIALQUERY_INVALID_OP);
    ovspatialquery_overlap_batch_reservation_t secondEarly{};
    const ovspatialquery_enqueue_result_t      secondAdmission =
        ovspatialquery_enqueue_overlap_plan(instance, fetchPlan, &batch, &secondEarly);
    REQUIRE(secondAdmission.status == OVSPATIALQUERY_API_SUCCESS);
    secondEarly                                                             = {};
    const ovspatialquery_overlap_batch_reservation_t overwrittenReservation = secondEarly;
    observeCompletedOperation(instance, secondAdmission.op_id);
    ovspatialquery_overlap_batch_result_t secondResult{};
    REQUIRE(ovspatialquery_fetch_overlap_batch(instance, secondAdmission.op_id, &secondResult).status ==
                OVSPATIALQUERY_API_SUCCESS &&
            secondResult.common.query_count == 1u && hasIdenticalBytes(secondEarly, overwrittenReservation));
    ovspatialquery_op_id_t secondOperation = secondAdmission.op_id;
    REQUIRE(ovspatialquery_destroy_op(instance, &secondOperation).status == OVSPATIALQUERY_API_SUCCESS &&
            secondOperation == OVSPATIALQUERY_INVALID_OP &&
            ovspatialquery_destroy_batch_plan(instance, fetchPlan).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    destroyUnitSphereAndInstance(instance);
}

void requireConcurrentReporterFacadeLifecycle()
{
    ovspatialquery_instance_handle_t instance = createInstance(UINT64_C(64) << 20);
    registerUnitSphere(instance);
    const ovspatialquery_population_handle_t first  = createEmptyPopulation(instance);
    const ovspatialquery_population_handle_t second = createEmptyPopulation(instance);
    publishUnitSphere(instance, first, -1, 3, 1u, 1.0);
    publishUnitSphere(instance, second, -1, 5, 2u, 2.0);
    ovspatialquery_pair_reporter_desc_t desc{};
    desc.first                = populationRef(first);
    desc.second               = populationRef(second);
    desc.active_pair_capacity = 1u;
    desc.delta_pair_capacity  = 1u;
    desc.result_slot_count    = 1u;
    desc.result_device        = DLDevice{kDLCPU, 0};
    ovspatialquery_pair_reporter_advance_t advance{};
    advance.first_ordinal.kind  = OVSPATIALQUERY_ORDINAL_HEAD;
    advance.second_ordinal.kind = OVSPATIALQUERY_ORDINAL_HEAD;

    for (uint32_t iteration = 0u; iteration < 100u; ++iteration)
    {
        ovspatialquery_pair_reporter_handle_t reporter{};
        REQUIRE(ovspatialquery_create_pair_reporter(instance, &desc, &reporter).status == OVSPATIALQUERY_API_SUCCESS &&
                reporter != OVSPATIALQUERY_INVALID_PAIR_REPORTER);
        std::atomic<bool>                          start{false};
        ovspatialquery_enqueue_result_t            admission{};
        ovspatialquery_pair_reporter_reservation_t early{};
        ovspatialquery_result_t                    reporterDestroy{};
        std::thread                                enqueueThread(
            [&]()
            {
                while (!start.load(std::memory_order_acquire))
                {
                    std::this_thread::yield();
                }
                admission = ovspatialquery_enqueue_pair_reporter_advance(instance, reporter, &advance, &early);
            });
        std::thread reporterDestroyThread(
            [&]()
            {
                while (!start.load(std::memory_order_acquire))
                {
                    std::this_thread::yield();
                }
                reporterDestroy = ovspatialquery_destroy_pair_reporter(instance, reporter);
            });
        start.store(true, std::memory_order_release);
        enqueueThread.join();
        reporterDestroyThread.join();

        if (admission.status == OVSPATIALQUERY_API_SUCCESS)
        {
            REQUIRE(reporterDestroy.status == OVSPATIALQUERY_API_BUSY);
            const ovspatialquery_op_id_t operation = admission.op_id;
            ovspatialquery_result_t      statusResult{};
            ovspatialquery_op_status_t   status{};
            std::thread                  statusThread(
                [&]()
                {
                    do
                    {
                        statusResult = ovspatialquery_query_op_status(instance, operation, &status);
                        std::this_thread::yield();
                    } while (statusResult.status == OVSPATIALQUERY_API_SUCCESS &&
                             status.state == OVSPATIALQUERY_OP_PENDING);
                });
            ovspatialquery_op_id_t  destroyOperation = operation;
            ovspatialquery_result_t operationDestroy{};
            std::thread             operationDestroyThread(
                [&]()
                {
                    do
                    {
                        operationDestroy = ovspatialquery_destroy_op(instance, &destroyOperation);
                        std::this_thread::yield();
                    } while (operationDestroy.status == OVSPATIALQUERY_API_BUSY);
                });
            statusThread.join();
            operationDestroyThread.join();
            REQUIRE(statusResult.status == OVSPATIALQUERY_API_SUCCESS && status.state == OVSPATIALQUERY_OP_COMPLETED &&
                    operationDestroy.status == OVSPATIALQUERY_API_SUCCESS &&
                    destroyOperation == OVSPATIALQUERY_INVALID_OP &&
                    ovspatialquery_destroy_pair_reporter(instance, reporter).status == OVSPATIALQUERY_API_SUCCESS);
        }
        else
        {
            REQUIRE(admission.status == OVSPATIALQUERY_API_INVALID_HANDLE &&
                    admission.op_id == OVSPATIALQUERY_INVALID_OP &&
                    reporterDestroy.status == OVSPATIALQUERY_API_SUCCESS);
        }
    }

    ovspatialquery_pair_reporter_handle_t waitingReporter{};
    REQUIRE(ovspatialquery_create_pair_reporter(instance, &desc, &waitingReporter).status ==
                OVSPATIALQUERY_API_SUCCESS &&
            waitingReporter != OVSPATIALQUERY_INVALID_PAIR_REPORTER);
    REQUIRE(ovspatialquery::detail::setInstanceOperationExecutionsPausedForTest(instance, true) ==
            OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_pair_reporter_reservation_t waitingEarly{};
    const ovspatialquery_enqueue_result_t      waitingAdmission =
        ovspatialquery_enqueue_pair_reporter_advance(instance, waitingReporter, &advance, &waitingEarly);
    REQUIRE(waitingAdmission.status == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_op_status_t pendingStatus{};
    REQUIRE(ovspatialquery_query_op_status(instance, waitingAdmission.op_id, &pendingStatus).status ==
                OVSPATIALQUERY_API_SUCCESS &&
            pendingStatus.state == OVSPATIALQUERY_OP_PENDING &&
            pendingStatus.terminal_status == OVSPATIALQUERY_API_NOT_READY &&
            ovspatialquery_grow_pair_reporter_capacity(instance, waitingReporter, 2u, 2u).status ==
                OVSPATIALQUERY_API_BUSY);
    const uint64_t          waiterCount = ovspatialquery::detail::instanceOperationWaiterEntryCountForTest(instance);
    ovspatialquery_result_t waitResult{};
    ovspatialquery_op_wait_result_t waitState{};
    std::thread                     waitThread(
        [&]()
        {
            waitResult =
                ovspatialquery_wait_op(instance, waitingAdmission.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waitState);
        });
    waitForWaiterEntry(instance, waiterCount);
    std::atomic<bool>       destroyReturned{false};
    ovspatialquery_result_t waitingDestroy{};
    std::thread             destroyThread(
        [&]()
        {
            waitingDestroy = ovspatialquery_destroy_pair_reporter(instance, waitingReporter);
            destroyReturned.store(true, std::memory_order_release);
        });
    const bool destroyWasPrompt = waitForFlag(destroyReturned);
    REQUIRE(ovspatialquery::detail::setInstanceOperationExecutionsPausedForTest(instance, false) ==
            OVSPATIALQUERY_API_SUCCESS);
    waitThread.join();
    destroyThread.join();
    REQUIRE(destroyWasPrompt && waitingDestroy.status == OVSPATIALQUERY_API_BUSY &&
            waitResult.status == OVSPATIALQUERY_API_SUCCESS && waitState.state == OVSPATIALQUERY_OP_COMPLETED);
    ovspatialquery_op_id_t waitingOperation = waitingAdmission.op_id;
    REQUIRE(ovspatialquery_destroy_op(instance, &waitingOperation).status == OVSPATIALQUERY_API_SUCCESS &&
            waitingOperation == OVSPATIALQUERY_INVALID_OP &&
            ovspatialquery_destroy_pair_reporter(instance, waitingReporter).status == OVSPATIALQUERY_API_SUCCESS);

    ovspatialquery_pair_reporter_handle_t fetchReporter{};
    REQUIRE(ovspatialquery_create_pair_reporter(instance, &desc, &fetchReporter).status == OVSPATIALQUERY_API_SUCCESS &&
            fetchReporter != OVSPATIALQUERY_INVALID_PAIR_REPORTER);
    const unsigned growFirst =
        static_cast<unsigned>(ovspatialquery_grow_pair_reporter_capacity(instance, fetchReporter, 2u, 2u).status);
    const unsigned growEqual =
        static_cast<unsigned>(ovspatialquery_grow_pair_reporter_capacity(instance, fetchReporter, 2u, 2u).status);
    const unsigned growShrink =
        static_cast<unsigned>(ovspatialquery_grow_pair_reporter_capacity(instance, fetchReporter, 1u, 2u).status);
    const unsigned growBudget = static_cast<unsigned>(
        ovspatialquery_grow_pair_reporter_capacity(instance, fetchReporter, UINT64_C(10000000), UINT64_C(10000000))
            .status);
    REQUIRE(growFirst == OVSPATIALQUERY_API_SUCCESS && growEqual == OVSPATIALQUERY_API_INVALID_ARGUMENT &&
            growShrink == OVSPATIALQUERY_API_INVALID_ARGUMENT && growBudget == OVSPATIALQUERY_API_HOST_BUDGET);
    ovspatialquery_pair_reporter_reservation_t firstEarly{};
    const ovspatialquery_enqueue_result_t      firstAdmission =
        ovspatialquery_enqueue_pair_reporter_advance(instance, fetchReporter, &advance, &firstEarly);
    REQUIRE(firstAdmission.status == OVSPATIALQUERY_API_SUCCESS && firstEarly.result_slot == 0u &&
            firstEarly.result_device.device_type == kDLCPU && firstEarly.delta_pair_capacity == 2u &&
            firstEarly.result_ready.value == 0u);
    const ovspatialquery_pair_reporter_reservation_t firstReservation = firstEarly;
    observeCompletedOperation(instance, firstAdmission.op_id);
    ovspatialquery_test_current_api_pause_fetch_projection(1u, true);
    ovspatialquery_pair_reporter_result_t firstResult{};
    ovspatialquery_result_t               firstFetch{};
    std::thread                           fetchThread(
        [&]() { firstFetch = ovspatialquery_fetch_pair_reporter(instance, firstAdmission.op_id, &firstResult); });
    ovspatialquery_test_current_api_wait_fetch_projection(1u);
    ovspatialquery_op_id_t  firstOperation = firstAdmission.op_id;
    ovspatialquery_result_t firstDestroy{};
    std::atomic<bool>       operationDestroyReturned{false};
    std::thread             operationDestroyThread(
        [&]()
        {
            firstDestroy = ovspatialquery_destroy_op(instance, &firstOperation);
            operationDestroyReturned.store(true, std::memory_order_release);
        });
    const bool destroyRacedPastFetch = waitForFlag(operationDestroyReturned);
    ovspatialquery_test_current_api_pause_fetch_projection(1u, false);
    fetchThread.join();
    operationDestroyThread.join();
    REQUIRE(!destroyRacedPastFetch && firstFetch.status == OVSPATIALQUERY_API_SUCCESS &&
            firstResult.accepted_active_pair_count == 1u && hasIdenticalBytes(firstEarly, firstReservation) &&
            firstDestroy.status == OVSPATIALQUERY_API_SUCCESS && firstOperation == OVSPATIALQUERY_INVALID_OP);
    REQUIRE(ovspatialquery_grow_pair_reporter_capacity(instance, fetchReporter, 3u, 3u).status ==
            OVSPATIALQUERY_API_SUCCESS);
    publishUnitSphere(instance, first, 3, 4, 1u, 1.0);
    publishUnitSphere(instance, second, 5, 6, 2u, 2.0);
    ovspatialquery_pair_reporter_reservation_t secondEarly{};
    const ovspatialquery_enqueue_result_t      secondAdmission =
        ovspatialquery_enqueue_pair_reporter_advance(instance, fetchReporter, &advance, &secondEarly);
    REQUIRE(secondAdmission.status == OVSPATIALQUERY_API_SUCCESS);
    secondEarly                                                             = {};
    const ovspatialquery_pair_reporter_reservation_t overwrittenReservation = secondEarly;
    observeCompletedOperation(instance, secondAdmission.op_id);
    ovspatialquery_pair_reporter_result_t secondResult{};
    REQUIRE(ovspatialquery_fetch_pair_reporter(instance, secondAdmission.op_id, &secondResult).status ==
                OVSPATIALQUERY_API_SUCCESS &&
            secondResult.accepted_active_pair_count == 1u && hasIdenticalBytes(secondEarly, overwrittenReservation));
    ovspatialquery_op_id_t secondOperation = secondAdmission.op_id;
    REQUIRE(ovspatialquery_destroy_op(instance, &secondOperation).status == OVSPATIALQUERY_API_SUCCESS &&
            secondOperation == OVSPATIALQUERY_INVALID_OP &&
            ovspatialquery_destroy_pair_reporter(instance, fetchReporter).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_population(instance, first).status == OVSPATIALQUERY_API_SUCCESS &&
            ovspatialquery_destroy_population(instance, second).status == OVSPATIALQUERY_API_SUCCESS);
    destroyUnitSphereAndInstance(instance);
}

} // namespace

extern "C" void ovspatialquery_test_current_api_instance_bootstrap_budget(void)
{
    const uint64_t oneWorkerCharge  = createAndMeasureInstance(1u);
    const uint64_t fourWorkerCharge = createAndMeasureInstance(4u);
    REQUIRE(fourWorkerCharge > oneWorkerCharge);

    ovspatialquery_instance_desc_t desc{};
    desc.operation_capacity                   = 2u;
    desc.max_cpu_concurrency                  = 4u;
    desc.host_byte_budget                     = fourWorkerCharge - 1u;
    ovspatialquery_instance_handle_t instance = ovspatialquery_instance_handle_t{UINT64_C(0xface)};
    REQUIRE(ovspatialquery_create_instance(&desc, &instance).status == OVSPATIALQUERY_API_HOST_BUDGET &&
            instance == OVSPATIALQUERY_INVALID_INSTANCE);

    desc.host_byte_budget = fourWorkerCharge;
    REQUIRE(ovspatialquery_create_instance(&desc, &instance).status == OVSPATIALQUERY_API_SUCCESS &&
            instance != OVSPATIALQUERY_INVALID_INSTANCE);
    uint64_t usage = 0u;
    REQUIRE(ovspatialquery_get_instance_memory_usage(instance, DLDevice{kDLCPU, 0}, &usage).status ==
                OVSPATIALQUERY_API_SUCCESS &&
            usage == fourWorkerCharge);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
    requirePlanAdmissionPrecedenceAndCheckedOffsets();
    requireRaycastAndSweepReservationRuntime();
    requirePairHostBudgetAdmissionAndRelease();
    requireConcurrentPlanFacadeLifecycle();
    requireConcurrentReporterFacadeLifecycle();
}
