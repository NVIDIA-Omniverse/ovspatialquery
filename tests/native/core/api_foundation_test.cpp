/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <ovspatialquery/ovspatialquery_explicit.h>

#include "../../support/native/public_api_inputs.h"
#include "oracle_backend.h"
#include "test_failure.h"

#include <array>
#include <cstdio>
#include <limits>
#include <vector>

// Instance, population and publication foundations from ovspatialquery.h and
// ovspatialquery_explicit.h, ported off the retired generation-1 internal
// API (source/ovspatialquery_private_api.h) onto ovspatialquery_create_instance
// and the explicit population adapter. The oracle_backend.h hooks used below
// gate the same default backend runtime that ovspatialquery_create_instance
// acquires (see ovspatialquery_instance.cpp's acquireDefaultBackendRuntime),
// so they observe publicly created instances exactly as they observed the
// retired private API's instances.

namespace
{

InputArray& emptyRemoved(InputArray& removed)
{
    static uint64_t zero = 0u;
    removed.set(&zero, 0, kDLUInt);
    return removed;
}

ovspatialquery_instance_handle_t createInstance(uint32_t operationCapacity)
{
    const ovspatialquery_instance_desc_t desc{operationCapacity, 1u, UINT64_C(16) * 1024u * 1024u, nullptr, 0u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&desc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    return instance;
}

ovspatialquery_enqueue_result_t enqueueSet(ovspatialquery_instance_handle_t   instance,
                                           ovspatialquery_population_handle_t population,
                                           ovspatialquery_ordinal_t ordinal, InstanceRows& rows)
{
    const ovspatialquery_explicit_population_set_t set{ordinal, rows.table(), nullptr, 0u};
    return ovspatialquery_enqueue_explicit_population_set(instance, population, &set);
}

// Promise: "A newly created population returns SUCCESS with ready == 0; it
// has no valid head and queries return NOT_READY"
// (ovspatialquery_get_population_info doc comment), proven for the primary
// synchronous raycast entry point. Already covered for pair snapshots by
// candidate_pairs_test.cpp's requireSnapshotRejected(..., unpublished, ...)
// at NOT_READY; this proves the same rule for ovspatialquery.h's own query
// path.
void testUnreadyPopulationQueriesReportNotReady()
{
    ovspatialquery_instance_handle_t   instance   = createInstance(2u);
    ovspatialquery_population_handle_t population = createExplicitPopulation(instance, 2u);

    ovspatialquery_population_info_t info{};
    REQUIRE(ovspatialquery_get_population_info(instance, population, &info).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(info.ready == 0u);

    ovspatialquery_raycast_desc_t query{};
    query.common.target         = populationTarget(population);
    query.common.precision      = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    query.ray.unit_direction[0] = 1.0f;
    query.ray.max_distance      = 10.0;
    ovspatialquery_raycast_single_result_t result{71, 72u, {}};
    const ovspatialquery_result_t          status = ovspatialquery_raycast_closest_sync(instance, &query, &result);
    if (status.status != OVSPATIALQUERY_API_NOT_READY)
    {
        std::fprintf(stderr, "unready raycast status %u, expected NOT_READY\n", static_cast<unsigned>(status.status));
    }
    REQUIRE(status.status == OVSPATIALQUERY_API_NOT_READY);
    REQUIRE(result.source_ordinal == 71 && result.has_hit == 72u);

    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

// Promise: ovspatialquery_fetch_population_update "returns NOT_READY while
// pending"; query_op_status "PENDING uses NOT_READY (7)" (ovspatialquery.h).
void testPopulationSetPendingFetchReturnsNotReady()
{
    ovspatialquery_instance_handle_t   instance   = createInstance(2u);
    ovspatialquery_backend_runtime_t*  runtime    = ovspatialquery_test_oracle_get_default_runtime();
    ovspatialquery_population_handle_t population = createExplicitPopulation(instance, 2u);

    ovspatialquery_test_oracle_block_before_completion(runtime, 1u, 1u);
    InstanceRows                          rows;
    const ovspatialquery_enqueue_result_t admission = enqueueSet(instance, population, 1, rows);
    REQUIRE(admission.status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_test_oracle_wait_before_completion(runtime) == 1u);

    ovspatialquery_population_update_result_t pending{81, 82u, 83u, {}, {}};
    const ovspatialquery_result_t             fetchStatus =
        ovspatialquery_fetch_population_update(instance, admission.op_id, &pending);
    if (fetchStatus.status != OVSPATIALQUERY_API_NOT_READY)
    {
        std::fprintf(stderr, "pending fetch status %u, expected NOT_READY\n",
                     static_cast<unsigned>(fetchStatus.status));
    }
    REQUIRE(fetchStatus.status == OVSPATIALQUERY_API_NOT_READY);
    REQUIRE(pending.published_ordinal == 81 && pending.object_count == 82u && pending.excluded_object_count == 83u);

    ovspatialquery_op_status_t status{};
    REQUIRE(ovspatialquery_query_op_status(instance, admission.op_id, &status).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(status.state == OVSPATIALQUERY_OP_PENDING);
    REQUIRE(status.terminal_status == OVSPATIALQUERY_API_NOT_READY);

    ovspatialquery_test_oracle_release_before_completion(runtime);
    waitAndDestroy(instance, admission);
    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

// Promise: "Success is atomic; terminal failure leaves the population
// NOT_READY" (ovspatialquery_enqueue_explicit_population_set doc comment)
// and query_op_status's documented FAILURE causes (OPERATION_FAILED,
// OUT_OF_MEMORY, INTERNAL, DEVICE_LOST). Also proves failure isolation: a
// later set on the still-NOT_READY population is unaffected.
void testPopulationSetFailureIsAtomicAndIsolated()
{
    ovspatialquery_instance_handle_t   instance   = createInstance(2u);
    ovspatialquery_backend_runtime_t*  runtime    = ovspatialquery_test_oracle_get_default_runtime();
    ovspatialquery_population_handle_t population = createExplicitPopulation(instance, 2u);

    ovspatialquery_test_oracle_fail_next_build(runtime, OVSPATIALQUERY_API_INTERNAL);
    InstanceRows                          rows;
    const ovspatialquery_enqueue_result_t admission = enqueueSet(instance, population, 1, rows);
    REQUIRE(admission.status == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_op_wait_result_t wait{};
    REQUIRE(ovspatialquery_wait_op(instance, admission.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &wait).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(wait.state == OVSPATIALQUERY_OP_FAILURE);

    ovspatialquery_op_status_t status{};
    REQUIRE(ovspatialquery_query_op_status(instance, admission.op_id, &status).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(status.state == OVSPATIALQUERY_OP_FAILURE);
    const bool isDocumentedFailure = status.terminal_status == OVSPATIALQUERY_API_OPERATION_FAILED ||
                                     status.terminal_status == OVSPATIALQUERY_API_OUT_OF_MEMORY ||
                                     status.terminal_status == OVSPATIALQUERY_API_INTERNAL ||
                                     status.terminal_status == OVSPATIALQUERY_API_DEVICE_LOST;
    if (!isDocumentedFailure)
    {
        std::fprintf(stderr,
                     "failed op terminal_status %u is not one of the documented "
                     "FAILURE causes\n",
                     static_cast<unsigned>(status.terminal_status));
    }
    REQUIRE(isDocumentedFailure);

    const ovspatialquery_string_t diagnostic = ovspatialquery_get_last_op_error(instance, admission.op_id);
    REQUIRE(diagnostic.ptr != nullptr && diagnostic.length != 0u);

    ovspatialquery_population_update_result_t failedResult{91, 92u, 93u, {}, {}};
    REQUIRE(ovspatialquery_fetch_population_update(instance, admission.op_id, &failedResult).status ==
            OVSPATIALQUERY_API_OPERATION_FAILED);
    REQUIRE(failedResult.published_ordinal == 91 && failedResult.object_count == 92u &&
            failedResult.excluded_object_count == 93u);

    ovspatialquery_population_info_t infoAfterFailure{};
    REQUIRE(ovspatialquery_get_population_info(instance, population, &infoAfterFailure).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(infoAfterFailure.ready == 0u);

    ovspatialquery_op_id_t opId = admission.op_id;
    REQUIRE(ovspatialquery_destroy_op(instance, &opId).status == OVSPATIALQUERY_API_SUCCESS);

    // Isolation: the failed set did not corrupt the still-NOT_READY
    // population, so a later set on it succeeds normally.
    InstanceRows retryRows;
    waitAndDestroy(instance, enqueueSet(instance, population, 1, retryRows));
    ovspatialquery_population_info_t infoAfterRetry{};
    REQUIRE(ovspatialquery_get_population_info(instance, population, &infoAfterRetry).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(infoAfterRetry.ready != 0u && infoAfterRetry.head_ordinal == 1);

    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

// Promise: ordinals are "Nonnegative population-scoped publication
// ordinal[s]" up to the int64_t ceiling, and an update's target_ordinal
// "Must be strictly greater than base_ordinal" (ovspatialquery.h,
// ovspatialquery_explicit.h), exercised at the signed ceiling where no
// further update can ever satisfy that rule.
void testOrdinalCeilingAndNegativeRejection()
{
    ovspatialquery_instance_handle_t   instance   = createInstance(1u);
    ovspatialquery_population_handle_t population = createExplicitPopulation(instance, 2u);

    InstanceRows                                   negativeRows;
    const ovspatialquery_explicit_population_set_t negativeSet{-1, negativeRows.table(), nullptr, 0u};
    const ovspatialquery_enqueue_result_t          rejected =
        ovspatialquery_enqueue_explicit_population_set(instance, population, &negativeSet);
    REQUIRE(rejected.status == OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(rejected.op_id == OVSPATIALQUERY_INVALID_OP);
    ovspatialquery_population_info_t infoAfterRejection{};
    REQUIRE(ovspatialquery_get_population_info(instance, population, &infoAfterRejection).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(infoAfterRejection.ready == 0u);

    InstanceRows                   ceilingRows;
    const ovspatialquery_ordinal_t ceiling = std::numeric_limits<ovspatialquery_ordinal_t>::max();
    waitAndDestroy(instance, enqueueSet(instance, population, ceiling, ceilingRows));
    ovspatialquery_population_info_t infoAtCeiling{};
    REQUIRE(ovspatialquery_get_population_info(instance, population, &infoAtCeiling).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(infoAtCeiling.head_ordinal == ceiling);

    InstanceRows                                      changed;
    InputArray                                        removed;
    const ovspatialquery_explicit_population_update_t atCeiling{
        ceiling, ceiling, OVSPATIALQUERY_EXPLICIT_UPDATE_COMPLETE_ROWS, changed.table(), &emptyRemoved(removed).value,
        nullptr};
    const ovspatialquery_enqueue_result_t stale =
        ovspatialquery_enqueue_explicit_population_update(instance, population, &atCeiling);
    if (stale.status != OVSPATIALQUERY_API_STALE_ORDINAL)
    {
        std::fprintf(stderr, "update at ceiling status %u, expected STALE_ORDINAL\n",
                     static_cast<unsigned>(stale.status));
    }
    REQUIRE(stale.status == OVSPATIALQUERY_API_STALE_ORDINAL);
    REQUIRE(stale.op_id == OVSPATIALQUERY_INVALID_OP);
    ovspatialquery_population_info_t infoAfterStale{};
    REQUIRE(ovspatialquery_get_population_info(instance, population, &infoAfterStale).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(infoAfterStale.head_ordinal == ceiling);

    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

// Promise: "`{ kDLCPU, 0 }` has every plan and pair-reporter key ... and not
// PREPARE_ON_DEVICE"; a CUDA device needs "an entry in the instance's
// `device_budgets`"; an unknown key is NOT_SUPPORTED, not an error; malformed
// devices and handles keep their usual statuses (ovspatialquery_query_capability).
void testCapabilityQueryAnswersForTheCpuAndUnconfiguredDevices()
{
    const ovspatialquery_instance_handle_t instance = createInstance(1u);
    const DLDevice                         cpu{kDLCPU, 0};
    for (uint32_t key = OVSPATIALQUERY_CAPABILITY_RAYCAST_PLAN; key <= OVSPATIALQUERY_CAPABILITY_PAIR_REPORTER; ++key)
    {
        const auto capability = static_cast<ovspatialquery_capability_t>(key);
        REQUIRE(ovspatialquery_query_capability(instance, cpu, capability).status ==
                (capability == OVSPATIALQUERY_CAPABILITY_PREPARE_ON_DEVICE ? OVSPATIALQUERY_API_NOT_SUPPORTED
                                                                           : OVSPATIALQUERY_API_SUCCESS));
        // The instance has no device budget, so no CUDA device is usable whatever the host has.
        REQUIRE(ovspatialquery_query_capability(instance, {kDLCUDA, 0}, capability).status ==
                OVSPATIALQUERY_API_NOT_SUPPORTED);
    }
    REQUIRE(ovspatialquery_query_capability(
                instance, cpu, static_cast<ovspatialquery_capability_t>(OVSPATIALQUERY_CAPABILITY_PAIR_REPORTER + 1))
                .status == OVSPATIALQUERY_API_NOT_SUPPORTED);
    REQUIRE(ovspatialquery_get_last_error().length != 0u);
    REQUIRE(ovspatialquery_query_capability(instance, {kDLCPU, 1}, OVSPATIALQUERY_CAPABILITY_RAYCAST_PLAN).status ==
            OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(ovspatialquery_query_capability(instance, {kDLCUDA, -1}, OVSPATIALQUERY_CAPABILITY_RAYCAST_PLAN).status ==
            OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(
        ovspatialquery_query_capability(instance, {kDLCUDAHost, 0}, OVSPATIALQUERY_CAPABILITY_RAYCAST_PLAN).status ==
        OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(
        ovspatialquery_query_capability(OVSPATIALQUERY_INVALID_INSTANCE, cpu, OVSPATIALQUERY_CAPABILITY_RAYCAST_PLAN)
            .status == OVSPATIALQUERY_API_INVALID_HANDLE);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

} // namespace

int main()
{
    testCapabilityQueryAnswersForTheCpuAndUnconfiguredDevices();
    testUnreadyPopulationQueriesReportNotReady();
    testPopulationSetPendingFetchReturnsNotReady();
    testPopulationSetFailureIsAtomicAndIsolated();
    testOrdinalCeilingAndNegativeRejection();
    std::puts("api foundation test passed");
    return 0;
}
