/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "oracle_backend.h"
#include "ovspatialquery_family_batch_plan_operation.h"
#include "ovspatialquery_instance_operation_runtime.h"
#include "ovspatialquery_instance_source.h"
#include "ovspatialquery_pair_reporter_operation.h"
#include "test_failure.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace
{

using namespace ovspatialquery::detail;

void waitPublicationOperation(ovspatialquery_instance_handle_t instance, ovspatialquery_op_id_t operation)
{
    ovspatialquery_op_wait_result_t wait{};
    REQUIRE(ovspatialquery_wait_op(instance, operation, UINT64_MAX, &wait).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(wait.state == OVSPATIALQUERY_OP_COMPLETED);
}

void releasePublicationOperation(ovspatialquery_instance_handle_t instance, ovspatialquery_op_id_t operation)
{
    REQUIRE(ovspatialquery_destroy_op(instance, &operation).status == OVSPATIALQUERY_API_SUCCESS);
}

PopulationObject object(uint64_t id, double minimumX = 0.0)
{
    PopulationObject value{};
    value.id     = {id};
    value.bounds = {
        {minimumX, -1.0, -1.0},
        {minimumX + 1.0, 1.0, 1.0},
    };
    value.participationBits = {{UINT64_MAX, UINT64_MAX}};
    return value;
}

struct SourceProbe
{
    std::map<ovspatialquery_ordinal_t, std::vector<PopulationObject>> snapshots;
};

ovspatialquery_api_status_t acquireSnapshot(const void*, const void*, const std::shared_ptr<void>& sourceState,
                                            const std::vector<SourceViewRequest>&,
                                            ovspatialquery_ordinal_t targetOrdinal, const SourceAcquisitionBase*,
                                            CopiedSourceSnapshot&    outSnapshot)
{
    const std::shared_ptr<SourceProbe> source = std::static_pointer_cast<SourceProbe>(sourceState);
    const auto                         found  = source->snapshots.find(targetOrdinal);
    if (found == source->snapshots.end())
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    outSnapshot              = {};
    outSnapshot.objects      = found->second;
    outSnapshot.indexedCount = outSnapshot.objects.size();
    return OVSPATIALQUERY_API_SUCCESS;
}

void publish(PopulationTable& populations, PopulationKey population, ovspatialquery_ordinal_t ordinal,
             std::vector<PopulationObject> objects)
{
    GenerationLease base;
    REQUIRE(populations.beginUpdate(population, ordinal, nullptr, base) == OVSPATIALQUERY_API_SUCCESS);
    GenerationLease generation;
    REQUIRE(populations.buildGeneration(ordinal, std::move(objects), {}, generation) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(populations.publishUpdate(population, generation) == OVSPATIALQUERY_API_SUCCESS);
    generation.reset();
    base.reset();
    populations.finishUpdate(population);
}

struct Fixture
{
    Fixture(uint32_t operationCapacity, uint32_t maxCpuConcurrency)
    {
        const SourceInstanceConfig config{
            operationCapacity, 0u, maxCpuConcurrency, std::numeric_limits<uint64_t>::max(), nullptr, 0u,
        };
        REQUIRE(createInstance(config, &instance) == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(ovspatialquery_test_oracle_create_runtime(&runtime) == OVSPATIALQUERY_API_SUCCESS);
        populations = std::make_unique<PopulationTable>(runtime);
        REQUIRE(populations->create(3u, first) == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(populations->create(3u, second) == OVSPATIALQUERY_API_SUCCESS);
        publish(*populations, first, 1, {object(1u, 2.0)});
        publish(*populations, second, 10, {object(2u, 2.0)});

        source = std::make_shared<SourceProbe>();
        source->snapshots.emplace(1, std::vector<PopulationObject>{object(7u)});
        PopulationSourceBinding binding{
            OVSPATIALQUERY_SOURCE_OVSTAGE, this, nullptr, source, acquireSnapshot,
        };
        REQUIRE(createEmptyPopulationForSource(instance, std::move(binding), 2u, 0u, sourcePopulation) ==
                OVSPATIALQUERY_API_SUCCESS);
    }

    ~Fixture()
    {
        if (sourcePopulation != OVSPATIALQUERY_INVALID_POPULATION)
        {
            REQUIRE(ovspatialquery_destroy_population(instance, sourcePopulation).status == OVSPATIALQUERY_API_SUCCESS);
        }
        populations.reset();
        ovspatialquery_test_oracle_destroy_runtime(runtime);
        REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
    }

    std::unique_ptr<FamilyBatchPlanOperationOwner> familyOwner(uint32_t resultSlots = 2u)
    {
        RaycastFamilyBatchPlanConfig config{};
        config.common.target.population = first;
        config.common.resultSlotCount   = resultSlots;
        config.plan.maximumQueryCount   = 1u;
        config.plan.hitCapacity         = 1u;
        std::unique_ptr<FamilyBatchPlanOperationOwner> owner;
        REQUIRE(FamilyBatchPlanOperationOwner::create(*populations, instance, config, owner) ==
                OVSPATIALQUERY_API_SUCCESS);
        return owner;
    }

    std::unique_ptr<PairReporterOperationOwner> reporterOwner()
    {
        PairReporterOperationConfig config{};
        config.first.target.population  = first;
        config.second.target.population = second;
        config.relation                 = PairEndpointRelation::distinctPopulations;
        config.activePairCapacity       = 4u;
        config.deltaPairCapacity        = 4u;
        config.resultSlotCount          = 2u;
        std::unique_ptr<PairReporterOperationOwner> owner;
        REQUIRE(PairReporterOperationOwner::create(*populations, instance, config, owner) ==
                OVSPATIALQUERY_API_SUCCESS);
        return owner;
    }

    FamilyBatchPlanAdmission enqueueFamily(FamilyBatchPlanOperationOwner& owner)
    {
        RaycastBatchLaunch launch;
        RaycastBatchRow    row{};
        row.query.origin          = {0.0, 0.0, 0.0};
        row.query.unitDirection   = {1.0f, 0.0f, 0.0f};
        row.query.maximumDistance = 100.0;
        launch.rows.push_back(row);
        ovspatialquery_raycast_batch_reservation_t reservation{};
        return owner.enqueueColumns({TargetSelectionKind::head, 0}, launch, reservation);
    }

    PairReporterAdmission enqueueReporter(PairReporterOperationOwner& owner)
    {
        PairReporterAdvance advance{};
        advance.first.kind  = TargetSelectionKind::head;
        advance.second.kind = TargetSelectionKind::head;
        return owner.enqueue(advance);
    }

    ovspatialquery_op_id_t enqueuePublication()
    {
        ovspatialquery_op_id_t operation = OVSPATIALQUERY_INVALID_OP;
        REQUIRE(submitPopulationSnapshotForSource(instance, OVSPATIALQUERY_SOURCE_OVSTAGE, sourcePopulation, nullptr, 1,
                                                  operation) == OVSPATIALQUERY_API_SUCCESS);
        return operation;
    }

    void closeOwners(FamilyBatchPlanOperationOwner& family, PairReporterOperationOwner& reporter)
    {
        REQUIRE(family.close() == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(reporter.close() == OVSPATIALQUERY_API_SUCCESS);
    }

    ovspatialquery_instance_handle_t   instance         = OVSPATIALQUERY_INVALID_INSTANCE;
    ovspatialquery_population_handle_t sourcePopulation = OVSPATIALQUERY_INVALID_POPULATION;
    ovspatialquery_backend_runtime_t*  runtime          = nullptr;
    std::unique_ptr<PopulationTable>   populations;
    PopulationKey                      first;
    PopulationKey                      second;
    std::shared_ptr<SourceProbe>       source;
};

void setWorkersPaused(Fixture& fixture, bool paused)
{
    REQUIRE(setInstanceOperationWorkersPausedForTest(fixture.instance, paused) == OVSPATIALQUERY_API_SUCCESS);
}

void waitAndDestroy(ovspatialquery_instance_handle_t instance, FamilyBatchPlanOperationOwner& owner,
                    FamilyBatchPlanOperationId& operation)
{
    InstanceOperationState state = InstanceOperationState::pending;
    REQUIRE(waitInstanceOperation(instance, operation.value, OVSPATIALQUERY_TIMEOUT_INFINITE, state) ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(state == InstanceOperationState::completed);
    REQUIRE(owner.destroyOperation(operation) == OVSPATIALQUERY_API_SUCCESS);
}

void waitAndDestroy(ovspatialquery_instance_handle_t instance, PairReporterOperationOwner& owner,
                    PairReporterOperationId& operation)
{
    InstanceOperationState state = InstanceOperationState::pending;
    REQUIRE(waitInstanceOperation(instance, operation.value, OVSPATIALQUERY_TIMEOUT_INFINITE, state) ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(state == InstanceOperationState::completed);
    REQUIRE(owner.destroyOperation(operation) == OVSPATIALQUERY_API_SUCCESS);
}

void testSharedCapacityIdsAndLifecycle()
{
    Fixture                                        fixture(3u, 1u);
    std::unique_ptr<FamilyBatchPlanOperationOwner> family   = fixture.familyOwner();
    std::unique_ptr<PairReporterOperationOwner>    reporter = fixture.reporterOwner();
    setWorkersPaused(fixture, true);

    ovspatialquery_op_id_t   publication       = fixture.enqueuePublication();
    FamilyBatchPlanAdmission familyAdmission   = fixture.enqueueFamily(*family);
    PairReporterAdmission    reporterAdmission = fixture.enqueueReporter(*reporter);
    REQUIRE(familyAdmission.outcome == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(reporterAdmission.outcome == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(publication != familyAdmission.operation.value);
    REQUIRE(publication != reporterAdmission.operation.value);
    REQUIRE(familyAdmission.operation.value != reporterAdmission.operation.value);

    const FamilyBatchPlanAdmission exhausted = fixture.enqueueFamily(*family);
    REQUIRE(exhausted.outcome == OVSPATIALQUERY_API_OPERATION_CAPACITY);
    REQUIRE(!exhausted.operation.valid() && exhausted.resultSlot == UINT32_MAX);
    InstanceOperationState reporterState = InstanceOperationState::failed;
    REQUIRE(queryInstanceOperation(fixture.instance, reporterAdmission.operation.value, reporterState) ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(reporterState == InstanceOperationState::pending);
    REQUIRE(ovspatialquery_get_last_op_error(fixture.instance, familyAdmission.operation.value).length == 0u);

    REQUIRE(ovspatialquery_destroy_instance(fixture.instance).status == OVSPATIALQUERY_API_BUSY);

    setWorkersPaused(fixture, false);
    waitPublicationOperation(fixture.instance, publication);

    ovspatialquery_raycast_batch_result_t firstFetch{};
    for (;;)
    {
        const ovspatialquery_api_status_t fetch = family->fetchColumns(familyAdmission.operation, firstFetch);
        if (fetch == OVSPATIALQUERY_API_SUCCESS)
        {
            break;
        }
        REQUIRE(fetch == OVSPATIALQUERY_API_NOT_READY);
        std::this_thread::yield();
    }
    ovspatialquery_raycast_batch_result_t repeated{};
    REQUIRE(family->fetchColumns(familyAdmission.operation, repeated) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(repeated.common.source_ordinal == firstFetch.common.source_ordinal);
    REQUIRE(ovspatialquery_get_last_op_error(fixture.instance, familyAdmission.operation.value).length == 0u);

    REQUIRE(family->destroyOperation(familyAdmission.operation) == OVSPATIALQUERY_API_SUCCESS);

    waitAndDestroy(fixture.instance, *reporter, reporterAdmission.operation);
    releasePublicationOperation(fixture.instance, publication);
    fixture.closeOwners(*family, *reporter);
}

void testInstanceConcurrencyBoundAcrossKinds()
{
    Fixture                                        fixture(8u, 2u);
    std::unique_ptr<FamilyBatchPlanOperationOwner> family   = fixture.familyOwner();
    std::unique_ptr<PairReporterOperationOwner>    reporter = fixture.reporterOwner();
    REQUIRE(setInstanceOperationExecutionsPausedForTest(fixture.instance, true) == OVSPATIALQUERY_API_SUCCESS);

    ovspatialquery_op_id_t   publication       = fixture.enqueuePublication();
    FamilyBatchPlanAdmission familyAdmission   = fixture.enqueueFamily(*family);
    PairReporterAdmission    reporterAdmission = fixture.enqueueReporter(*reporter);
    REQUIRE(familyAdmission.outcome == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(reporterAdmission.outcome == OVSPATIALQUERY_API_SUCCESS);

    InstanceOperationRuntimeInspection inspection{};
    for (;;)
    {
        REQUIRE(inspectInstanceOperationRuntimeForTest(fixture.instance, inspection) == OVSPATIALQUERY_API_SUCCESS);
        if (inspection.runningOperationCount == 2u)
        {
            break;
        }
        std::this_thread::yield();
    }
    REQUIRE(inspection.peakRunningOperationCount == 2u);
    REQUIRE(inspection.pendingOperationCount == 1u);

    // Callers waiting without a timeout may run a queued operation themselves, but
    // only in a free CPU job: with both running, the queued one stays queued.
    const uint64_t           waiterEntries = instanceOperationWaiterEntryCountForTest(fixture.instance);
    std::vector<std::thread> waiters;
    for (const ovspatialquery_op_id_t operation :
         {publication, familyAdmission.operation.value, reporterAdmission.operation.value})
    {
        waiters.emplace_back(
            [&fixture, operation]()
            {
                InstanceOperationState state = InstanceOperationState::pending;
                (void)waitInstanceOperation(fixture.instance, operation, OVSPATIALQUERY_TIMEOUT_INFINITE, state);
            });
    }
    const std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (instanceOperationWaiterEntryCountForTest(fixture.instance) < waiterEntries + 3u &&
           std::chrono::steady_clock::now() < deadline)
    {
        std::this_thread::yield();
    }
    REQUIRE(inspectInstanceOperationRuntimeForTest(fixture.instance, inspection) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(inspection.peakRunningOperationCount == 2u);
    REQUIRE(inspection.pendingOperationCount == 1u);
    REQUIRE(instanceOperationWaiterEntryCountForTest(fixture.instance) == waiterEntries + 3u);

    REQUIRE(setInstanceOperationExecutionsPausedForTest(fixture.instance, false) == OVSPATIALQUERY_API_SUCCESS);
    for (std::thread& waiter : waiters)
    {
        waiter.join();
    }
    waitPublicationOperation(fixture.instance, publication);
    waitAndDestroy(fixture.instance, *family, familyAdmission.operation);
    waitAndDestroy(fixture.instance, *reporter, reporterAdmission.operation);
    releasePublicationOperation(fixture.instance, publication);
    REQUIRE(inspectInstanceOperationRuntimeForTest(fixture.instance, inspection) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(inspection.peakRunningOperationCount == 2u);
    fixture.closeOwners(*family, *reporter);
}

void testPrivateSchedulingCannotLoseWakeupToInstanceReaper()
{
    Fixture                                        fixture(1u, 1u);
    std::unique_ptr<FamilyBatchPlanOperationOwner> family                 = fixture.familyOwner(1u);
    constexpr uint64_t                             kFiniteWaitNanoseconds = UINT64_C(1000000000);
    for (uint32_t iteration = 0u; iteration < 100u; ++iteration)
    {
        FamilyBatchPlanAdmission admission = fixture.enqueueFamily(*family);
        REQUIRE(admission.outcome == OVSPATIALQUERY_API_SUCCESS);
        InstanceOperationState state = InstanceOperationState::pending;
        REQUIRE(waitInstanceOperation(fixture.instance, admission.operation.value, kFiniteWaitNanoseconds, state) ==
                OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(state == InstanceOperationState::completed);
        REQUIRE(family->destroyOperation(admission.operation) == OVSPATIALQUERY_API_SUCCESS);
    }
    REQUIRE(family->close() == OVSPATIALQUERY_API_SUCCESS);
}

void testInstanceOwnedIdsRejectForeignOperations()
{
    Fixture                                        firstFixture(2u, 1u);
    Fixture                                        secondFixture(2u, 1u);
    std::unique_ptr<FamilyBatchPlanOperationOwner> firstOwner  = firstFixture.familyOwner();
    std::unique_ptr<FamilyBatchPlanOperationOwner> secondOwner = secondFixture.familyOwner();
    setWorkersPaused(firstFixture, true);
    setWorkersPaused(secondFixture, true);
    FamilyBatchPlanAdmission first  = firstFixture.enqueueFamily(*firstOwner);
    FamilyBatchPlanAdmission second = secondFixture.enqueueFamily(*secondOwner);
    REQUIRE(first.outcome == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(second.outcome == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(first.operation.value != second.operation.value);

    InstanceOperationState untouched = InstanceOperationState::failed;
    REQUIRE(queryInstanceOperation(secondFixture.instance, first.operation.value, untouched) ==
            OVSPATIALQUERY_API_INVALID_HANDLE);
    REQUIRE(untouched == InstanceOperationState::failed);

    setWorkersPaused(firstFixture, false);
    setWorkersPaused(secondFixture, false);
    waitAndDestroy(firstFixture.instance, *firstOwner, first.operation);
    waitAndDestroy(secondFixture.instance, *secondOwner, second.operation);
    REQUIRE(firstOwner->close() == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(secondOwner->close() == OVSPATIALQUERY_API_SUCCESS);
}

void testFaultRollbackAndPerOperationDiagnostics()
{
    Fixture                                        fixture(2u, 1u);
    std::unique_ptr<FamilyBatchPlanOperationOwner> family   = fixture.familyOwner();
    std::unique_ptr<PairReporterOperationOwner>    reporter = fixture.reporterOwner();

    REQUIRE(failNextInstanceOperationSchedulingForTest(fixture.instance) == OVSPATIALQUERY_API_SUCCESS);
    const FamilyBatchPlanAdmission scheduling = fixture.enqueueFamily(*family);
    REQUIRE(scheduling.outcome == OVSPATIALQUERY_API_INTERNAL);

    setWorkersPaused(fixture, true);
    family->failNextResourceForTest();
    reporter->failNextInternalForTest();
    FamilyBatchPlanAdmission familyFailure   = fixture.enqueueFamily(*family);
    PairReporterAdmission    reporterFailure = fixture.enqueueReporter(*reporter);
    REQUIRE(familyFailure.outcome == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(reporterFailure.outcome == OVSPATIALQUERY_API_SUCCESS);
    setWorkersPaused(fixture, false);

    InstanceOperationState familyState   = InstanceOperationState::pending;
    InstanceOperationState reporterState = InstanceOperationState::pending;
    REQUIRE(waitInstanceOperation(fixture.instance, familyFailure.operation.value, OVSPATIALQUERY_TIMEOUT_INFINITE,
                                  familyState) == OVSPATIALQUERY_API_OPERATION_FAILED);
    REQUIRE(waitInstanceOperation(fixture.instance, reporterFailure.operation.value, OVSPATIALQUERY_TIMEOUT_INFINITE,
                                  reporterState) == OVSPATIALQUERY_API_OPERATION_FAILED);

    const ovspatialquery_string_t familyText =
        ovspatialquery_get_last_op_error(fixture.instance, familyFailure.operation.value);
    const std::string             familyDiagnostic(familyText.ptr, familyText.length);
    const ovspatialquery_string_t reporterText =
        ovspatialquery_get_last_op_error(fixture.instance, reporterFailure.operation.value);
    const std::string reporterDiagnostic(reporterText.ptr, reporterText.length);
    REQUIRE(familyDiagnostic == "family batch forced resource failure");
    REQUIRE(reporterDiagnostic == "pair reporter forced internal failure");
    REQUIRE(familyDiagnostic != reporterDiagnostic);

    REQUIRE(family->destroyOperation(familyFailure.operation) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(reporter->destroyOperation(reporterFailure.operation) == OVSPATIALQUERY_API_SUCCESS);
    // The failed advance accepted nothing, so the same ordinals still progress.
    PairReporterAdmission retry = fixture.enqueueReporter(*reporter);
    REQUIRE(retry.outcome == OVSPATIALQUERY_API_SUCCESS);
    waitAndDestroy(fixture.instance, *reporter, retry.operation);
    fixture.closeOwners(*family, *reporter);
}

void testReporterCompletionPublishesBaselineWithTerminalState()
{
    Fixture                                     fixture(2u, 1u);
    std::unique_ptr<PairReporterOperationOwner> reporter = fixture.reporterOwner();
    REQUIRE(setInstanceOperationBeforeTerminalPausedForTest(fixture.instance, true) == OVSPATIALQUERY_API_SUCCESS);
    PairReporterAdmission admission = fixture.enqueueReporter(*reporter);
    REQUIRE(admission.outcome == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(waitForInstanceOperationBeforeTerminalForTest(fixture.instance) == OVSPATIALQUERY_API_SUCCESS);
    // BUSY, not NO_PROGRESS: the advance is still pending and has not yet
    // published its baseline.
    REQUIRE(fixture.enqueueReporter(*reporter).outcome == OVSPATIALQUERY_API_BUSY);
    InstanceOperationState state = InstanceOperationState::failed;
    REQUIRE(queryInstanceOperation(fixture.instance, admission.operation.value, state) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(state == InstanceOperationState::pending);

    REQUIRE(setInstanceOperationBeforeTerminalPausedForTest(fixture.instance, false) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(waitInstanceOperation(fixture.instance, admission.operation.value, OVSPATIALQUERY_TIMEOUT_INFINITE,
                                  state) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(state == InstanceOperationState::completed);
    // Once the wait returns, the baseline is accepted: the same ordinals no
    // longer progress.
    REQUIRE(fixture.enqueueReporter(*reporter).outcome == OVSPATIALQUERY_API_STALE_ORDINAL);
    REQUIRE(reporter->destroyOperation(admission.operation) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(reporter->close() == OVSPATIALQUERY_API_SUCCESS);
}

void testTerminalIsInvisibleUntilWorkerOwnershipRetires()
{
    Fixture                                        fixture(1u, 1u);
    std::unique_ptr<FamilyBatchPlanOperationOwner> owner = fixture.familyOwner(2u);
    REQUIRE(setInstanceOperationBeforeTerminalPausedForTest(fixture.instance, true) == OVSPATIALQUERY_API_SUCCESS);
    FamilyBatchPlanAdmission first = fixture.enqueueFamily(*owner);
    REQUIRE(first.outcome == OVSPATIALQUERY_API_SUCCESS);
    const FamilyBatchPlanOperationId firstId = first.operation;
    REQUIRE(waitForInstanceOperationBeforeTerminalForTest(fixture.instance) == OVSPATIALQUERY_API_SUCCESS);

    InstanceOperationState            state = InstanceOperationState::failed;
    const ovspatialquery_api_status_t statusOutcome =
        queryInstanceOperation(fixture.instance, first.operation.value, state);
    const ovspatialquery_api_status_t destroyOutcome = owner->destroyOperation(first.operation);
    const FamilyBatchPlanAdmission    prematureReuse = fixture.enqueueFamily(*owner);

    REQUIRE(statusOutcome == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(state == InstanceOperationState::pending);
    REQUIRE(destroyOutcome == OVSPATIALQUERY_API_BUSY);
    REQUIRE(first.operation.value == firstId.value);
    REQUIRE(prematureReuse.outcome == OVSPATIALQUERY_API_OPERATION_CAPACITY);

    REQUIRE(setInstanceOperationBeforeTerminalPausedForTest(fixture.instance, false) == OVSPATIALQUERY_API_SUCCESS);
    waitAndDestroy(fixture.instance, *owner, first.operation);

    setWorkersPaused(fixture, true);
    FamilyBatchPlanAdmission afterRetirement = fixture.enqueueFamily(*owner);
    REQUIRE(afterRetirement.outcome == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(afterRetirement.operation.value != firstId.value);
    InstanceOperationRuntimeInspection inspection{};
    REQUIRE(inspectInstanceOperationRuntimeForTest(fixture.instance, inspection) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(inspection.pendingOperationCount == 1u);
    REQUIRE(inspection.runningOperationCount == 0u);
    InstanceOperationState staleState = InstanceOperationState::failed;
    REQUIRE(queryInstanceOperation(fixture.instance, firstId.value, staleState) == OVSPATIALQUERY_API_INVALID_HANDLE);
    REQUIRE(staleState == InstanceOperationState::failed);
    setWorkersPaused(fixture, false);
    waitAndDestroy(fixture.instance, *owner, afterRetirement.operation);
    REQUIRE(owner->close() == OVSPATIALQUERY_API_SUCCESS);
}

void testExplicitOwnerDestructionIsNonWaitingAndRetainsBusyOwners()
{
    Fixture                            fixture(2u, 1u);
    InstanceOperationRuntimeInspection inspection{};

    setWorkersPaused(fixture, true);
    REQUIRE(setInstanceOperationBeforeTerminalPausedForTest(fixture.instance, true) == OVSPATIALQUERY_API_SUCCESS);
    std::unique_ptr<FamilyBatchPlanOperationOwner> family          = fixture.familyOwner();
    FamilyBatchPlanAdmission                       familyAdmission = fixture.enqueueFamily(*family);
    REQUIRE(familyAdmission.outcome == OVSPATIALQUERY_API_SUCCESS);
    const FamilyBatchPlanOperationId familyId = familyAdmission.operation;
    REQUIRE(FamilyBatchPlanOperationOwner::destroy(family) == OVSPATIALQUERY_API_BUSY);
    REQUIRE(family != nullptr && familyAdmission.operation.value == familyId.value);
    setWorkersPaused(fixture, false);
    REQUIRE(waitForInstanceOperationBeforeTerminalForTest(fixture.instance) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(FamilyBatchPlanOperationOwner::destroy(family) == OVSPATIALQUERY_API_BUSY);
    REQUIRE(family != nullptr);
    REQUIRE(inspectInstanceOperationRuntimeForTest(fixture.instance, inspection) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(inspection.runningOperationCount == 1u);
    REQUIRE(setInstanceOperationBeforeTerminalPausedForTest(fixture.instance, false) == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_raycast_batch_result_t familyResult{};
    ovspatialquery_api_status_t           fetchOutcome = OVSPATIALQUERY_API_BUSY;
    while ((fetchOutcome = family->fetchColumns(familyAdmission.operation, familyResult)) ==
           OVSPATIALQUERY_API_NOT_READY)
    {
        std::this_thread::yield();
    }
    REQUIRE(fetchOutcome == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(FamilyBatchPlanOperationOwner::destroy(family) == OVSPATIALQUERY_API_BUSY);
    REQUIRE(family != nullptr);
    InstanceOperationState familyState = InstanceOperationState::pending;
    REQUIRE(waitInstanceOperation(fixture.instance, familyAdmission.operation.value, OVSPATIALQUERY_TIMEOUT_INFINITE,
                                  familyState) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(familyState == InstanceOperationState::completed);
    REQUIRE(FamilyBatchPlanOperationOwner::destroy(family) == OVSPATIALQUERY_API_BUSY);
    REQUIRE(family->destroyOperation(familyAdmission.operation) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(FamilyBatchPlanOperationOwner::destroy(family) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(family == nullptr);

    setWorkersPaused(fixture, true);
    std::unique_ptr<PairReporterOperationOwner> reporter          = fixture.reporterOwner();
    PairReporterAdmission                       reporterAdmission = fixture.enqueueReporter(*reporter);
    REQUIRE(reporterAdmission.outcome == OVSPATIALQUERY_API_SUCCESS);
    const PairReporterOperationId reporterId = reporterAdmission.operation;
    REQUIRE(PairReporterOperationOwner::destroy(reporter) == OVSPATIALQUERY_API_BUSY);
    REQUIRE(reporter != nullptr && reporterAdmission.operation.value == reporterId.value);
    setWorkersPaused(fixture, false);
    InstanceOperationState reporterState = InstanceOperationState::pending;
    REQUIRE(waitInstanceOperation(fixture.instance, reporterAdmission.operation.value, OVSPATIALQUERY_TIMEOUT_INFINITE,
                                  reporterState) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(PairReporterOperationOwner::destroy(reporter) == OVSPATIALQUERY_API_BUSY);
    REQUIRE(reporter->destroyOperation(reporterAdmission.operation) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(PairReporterOperationOwner::destroy(reporter) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(reporter == nullptr);
}

void testReservationSerializesWithInstanceRelease()
{
    const SourceInstanceConfig instanceConfig{
        1u, 0u, 1u, std::numeric_limits<uint64_t>::max(), nullptr, 0u,
    };
    ovspatialquery_instance_handle_t instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(createInstance(instanceConfig, &instance) == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_backend_runtime_t* runtime = nullptr;
    REQUIRE(ovspatialquery_test_oracle_create_runtime(&runtime) == OVSPATIALQUERY_API_SUCCESS);
    PopulationTable populations(runtime);
    PopulationKey   population;
    REQUIRE(populations.create(2u, population) == OVSPATIALQUERY_API_SUCCESS);
    publish(populations, population, 1, {object(1u)});
    RaycastFamilyBatchPlanConfig planConfig{};
    planConfig.common.target.population = population;
    planConfig.common.resultSlotCount   = 1u;
    planConfig.plan.maximumQueryCount   = 1u;
    planConfig.plan.hitCapacity         = 1u;
    std::unique_ptr<FamilyBatchPlanOperationOwner> owner;
    REQUIRE(FamilyBatchPlanOperationOwner::create(populations, instance, planConfig, owner) ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(setInstanceOperationReservationPausedAfterLifecycleForTest(instance, true) == OVSPATIALQUERY_API_SUCCESS);

    FamilyBatchPlanAdmission admission{};
    std::thread              admissionThread(
        [&]()
        {
            RaycastBatchLaunch launch;
            RaycastBatchRow    row{};
            row.query.origin          = {0.0, 0.0, 0.0};
            row.query.unitDirection   = {1.0f, 0.0f, 0.0f};
            row.query.maximumDistance = 10.0;
            launch.rows.push_back(row);
            ovspatialquery_raycast_batch_reservation_t reservation{};
            admission = owner->enqueueColumns({TargetSelectionKind::head, 0}, launch, reservation);
        });
    REQUIRE(waitForInstanceOperationReservationAfterLifecycleForTest(instance) == OVSPATIALQUERY_API_SUCCESS);
    std::atomic<bool>           releaseFinished{false};
    ovspatialquery_api_status_t releaseStatus = OVSPATIALQUERY_API_SUCCESS;
    std::thread                 releaseThread(
        [&]()
        {
            releaseStatus = ovspatialquery_destroy_instance(instance).status;
            releaseFinished.store(true, std::memory_order_release);
        });
    REQUIRE(waitForInstanceReleaseBeforeLifecycleForTest(instance) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(!releaseFinished.load(std::memory_order_acquire));
    REQUIRE(setInstanceOperationReservationPausedAfterLifecycleForTest(instance, false) == OVSPATIALQUERY_API_SUCCESS);
    admissionThread.join();
    releaseThread.join();
    REQUIRE(admission.outcome == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(releaseStatus == OVSPATIALQUERY_API_BUSY);
    waitAndDestroy(instance, *owner, admission.operation);
    REQUIRE(owner->close() == OVSPATIALQUERY_API_SUCCESS);
    owner.reset();
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
    std::shared_ptr<void> deferredDestruction;
    REQUIRE(populations.release(population, deferredDestruction) == OVSPATIALQUERY_API_SUCCESS);
    deferredDestruction.reset();
    ovspatialquery_test_oracle_destroy_runtime(runtime);
}

} // namespace

int main()
{
    testSharedCapacityIdsAndLifecycle();
    testInstanceConcurrencyBoundAcrossKinds();
    testPrivateSchedulingCannotLoseWakeupToInstanceReaper();
    testInstanceOwnedIdsRejectForeignOperations();
    testFaultRollbackAndPerOperationDiagnostics();
    testReporterCompletionPublishesBaselineWithTerminalState();
    testTerminalIsInvisibleUntilWorkerOwnershipRetires();
    testExplicitOwnerDestructionIsNonWaitingAndRetainsBusyOwners();
    testReservationSerializesWithInstanceRelease();
    return 0;
}
