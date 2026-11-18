/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../support/native/test_failure.h"
#include "oracle_backend.h"
#include "ovspatialquery_instance_source.h"
#include "ovspatialquery_pair_reporter_operation.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

namespace
{

ovspatialquery::detail::InstanceHostLedger* gReporterTaskDestroyLedger         = nullptr;
uint64_t                                    gReporterTaskDestroyObservedCharge = UINT64_MAX;

void observeReporterTaskPayloadDestroy() { gReporterTaskDestroyObservedCharge = gReporterTaskDestroyLedger->total(); }

using namespace ovspatialquery::detail;

[[noreturn]] void fail(const char* message)
{
    std::fprintf(stderr, "FAIL: %s\n", message);
    OVSPATIALQUERY_TEST_EXIT_FAILURE();
}

void require(bool condition, const char* message)
{
    if (!condition)
    {
        fail(message);
    }
}

PopulationObject object(uint64_t id, double minimumX = 0.0, double maximumX = 1.0,
                        uint64_t participationBits = UINT64_MAX)
{
    PopulationObject value{};
    value.id     = {id};
    value.bounds = {
        {minimumX, 0.0, 0.0},
        {maximumX, 1.0, 1.0},
    };
    value.participationBits = {{participationBits, 0u}};
    return value;
}

ovspatialquery_pair_t pair(uint64_t first, uint64_t second) { return ovspatialquery_pair_t{{first}, {second}}; }

class Fixture
{
  public:
    explicit Fixture(uint32_t operationCapacity = 8u, uint32_t maxCpuConcurrency = 1u)
    {
        const SourceInstanceConfig instanceConfig{
            operationCapacity, 0u, maxCpuConcurrency, std::numeric_limits<uint64_t>::max(), nullptr, 0u,
        };
        require(createInstance(instanceConfig, &mInstance) == OVSPATIALQUERY_API_SUCCESS,
                "operation instance creation failed");
        require(ovspatialquery_test_oracle_create_runtime(&mRuntime) == OVSPATIALQUERY_API_SUCCESS,
                "oracle runtime creation failed");
        mPopulations = std::make_unique<PopulationTable>(mRuntime);
    }

    ~Fixture()
    {
        mPopulations.reset();
        ovspatialquery_test_oracle_destroy_runtime(mRuntime);
        require(ovspatialquery_destroy_instance(mInstance).status == OVSPATIALQUERY_API_SUCCESS,
                "operation instance release failed");
    }

    PopulationKey createPopulation(uint32_t retainedOrdinalCapacity = 4u)
    {
        PopulationKey population;
        require(mPopulations->create(retainedOrdinalCapacity, population) == OVSPATIALQUERY_API_SUCCESS,
                "population creation failed");
        return population;
    }

    void publish(PopulationKey population, ovspatialquery_ordinal_t ordinal, std::vector<PopulationObject> objects,
                 std::vector<GenerationViewMembership> memberships = {})
    {
        GenerationLease base;
        require(mPopulations->beginUpdate(population, ordinal, nullptr, base) == OVSPATIALQUERY_API_SUCCESS,
                "population update did not begin");
        GenerationLease generation;
        require(mPopulations->buildGeneration(ordinal, std::move(objects), std::move(memberships), generation) ==
                    OVSPATIALQUERY_API_SUCCESS,
                "generation build failed");
        require(mPopulations->publishUpdate(population, generation) == OVSPATIALQUERY_API_SUCCESS,
                "generation publication failed");
        generation.reset();
        base.reset();
        mPopulations->finishUpdate(population);
    }

    std::shared_ptr<ViewState> automaticView(PopulationKey population, ovspatialquery_view_handle_t handle)
    {
        std::shared_ptr<ViewState> view = std::make_shared<ViewState>();
        view->handle                    = handle;
        view->parent                    = population;
        view->automatic                 = true;
        return view;
    }

    PopulationTable& populations() { return *mPopulations; }

    ovspatialquery_instance_handle_t instance() const noexcept { return mInstance; }

  private:
    ovspatialquery_instance_handle_t  mInstance = OVSPATIALQUERY_INVALID_INSTANCE;
    ovspatialquery_backend_runtime_t* mRuntime  = nullptr;
    std::unique_ptr<PopulationTable>  mPopulations;
};

PairReporterOperationConfig config(PopulationKey first, PopulationKey second, uint64_t activeCapacity = 8u,
                                   uint64_t deltaCapacity = 8u, uint32_t resultSlotCount = 4u)
{
    PairReporterOperationConfig value{};
    value.first.target.population  = first;
    value.second.target.population = second;
    value.relation                 = PairEndpointRelation::distinctPopulations;
    value.activePairCapacity       = activeCapacity;
    value.deltaPairCapacity        = deltaCapacity;
    value.resultSlotCount          = resultSlotCount;
    return value;
}

void setWorkersPaused(const Fixture& fixture, bool paused)
{
    require(setInstanceOperationWorkersPausedForTest(fixture.instance(), paused) == OVSPATIALQUERY_API_SUCCESS,
            "instance worker pause control failed");
}

void failNextScheduling(const Fixture& fixture)
{
    require(failNextInstanceOperationSchedulingForTest(fixture.instance()) == OVSPATIALQUERY_API_SUCCESS,
            "instance scheduling fault control failed");
}

void setWaiterPaused(const Fixture& fixture, bool paused)
{
    require(setInstanceOperationWaiterPausedBeforeRevalidationForTest(fixture.instance(), paused) ==
                OVSPATIALQUERY_API_SUCCESS,
            "instance waiter pause control failed");
}

void waitForWaiter(const Fixture& fixture)
{
    require(waitForInstanceOperationWaiterAtRevalidationForTest(fixture.instance()) == OVSPATIALQUERY_API_SUCCESS,
            "instance waiter did not reach revalidation");
}

uint64_t waiterEntryCount(const Fixture& fixture)
{
    return instanceOperationWaiterEntryCountForTest(fixture.instance());
}

PairReporterAdvance headAdvance()
{
    PairReporterAdvance value{};
    value.first.kind  = TargetSelectionKind::head;
    value.second.kind = TargetSelectionKind::head;
    return value;
}

// Copies a completed CPU advance's result out of the storage view production
// fetches return; a CUDA reporter's rows are on the device, not in the view's host vectors.
ovspatialquery_api_status_t fetchCopy(const PairReporterOperationOwner& owner, PairReporterOperationId operation,
                                      PairReporterResult& outResult)
{
    PairReporterResultStorageView     view{};
    const ovspatialquery_api_status_t outcome = owner.fetchStorageView(operation, view);
    if (outcome == OVSPATIALQUERY_API_SUCCESS)
    {
        outResult.resultSlot              = view.resultSlot;
        outResult.outcome                 = view.outcome;
        outResult.firstSourceOrdinal      = view.firstSourceOrdinal;
        outResult.secondSourceOrdinal     = view.secondSourceOrdinal;
        outResult.acceptedActivePairCount = view.acceptedActivePairCount;
        outResult.entered.assign(view.entered, view.entered + view.enteredCount);
        outResult.exited.assign(view.exited, view.exited + view.exitedCount);
    }
    return outcome;
}

PairReporterResult complete(const Fixture& fixture, PairReporterOperationOwner& owner,
                            PairReporterOperationId operation)
{
    InstanceOperationState state = InstanceOperationState::pending;
    require(waitInstanceOperation(fixture.instance(), operation.value, OVSPATIALQUERY_TIMEOUT_INFINITE, state) ==
                    OVSPATIALQUERY_API_SUCCESS &&
                state == InstanceOperationState::completed,
            "reporter operation did not complete successfully");
    PairReporterResult result{};
    require(fetchCopy(owner, operation, result) == OVSPATIALQUERY_API_SUCCESS,
            "typed reporter result was not fetchable");
    return result;
}

void destroyObserved(PairReporterOperationOwner& owner, PairReporterOperationId& operation)
{
    require(owner.destroyOperation(operation) == OVSPATIALQUERY_API_SUCCESS && !operation.valid(),
            "observed terminal operation did not destroy");
}

void testGenerationTruthSamePopulationAndHeldLeases()
{
    Fixture                          fixture;
    const PopulationKey              population = fixture.createPopulation(1u);
    const std::shared_ptr<ViewState> firstView  = fixture.automaticView(population, ovspatialquery_view_handle_t{101u});
    const std::shared_ptr<ViewState> secondView = fixture.automaticView(population, ovspatialquery_view_handle_t{102u});
    fixture.publish(population, 1, {object(1u, 0.0, 1.0, UINT64_C(0x1)), object(2u, 0.0, 1.0, UINT64_C(0x2))},
                    {{{101u}, std::make_shared<const std::vector<uint64_t>>(std::initializer_list<uint64_t>{1u})},
                     {{102u}, std::make_shared<const std::vector<uint64_t>>(std::initializer_list<uint64_t>{2u})}});

    PairReporterOperationConfig directed = config(population, population);
    directed.first.target.view           = firstView;
    directed.first.queryBits             = {{UINT64_C(0x1), 0u}};
    directed.second.target.view          = secondView;
    directed.second.queryBits            = {{UINT64_C(0x2), 0u}};
    directed.relation                    = PairEndpointRelation::samePopulationDistinctSelections;
    std::unique_ptr<PairReporterOperationOwner> owner;
    require(PairReporterOperationOwner::create(fixture.populations(), fixture.instance(), directed, owner) ==
                OVSPATIALQUERY_API_SUCCESS,
            "same-population reporter creation failed");
    setWorkersPaused(fixture, true);
    PairReporterAdmission held = owner->enqueue(headAdvance());
    require(held.outcome == OVSPATIALQUERY_API_SUCCESS, "held same-population advance was not admitted");
    fixture.publish(population, 2, {object(1u, 0.0, 1.0, UINT64_C(0x4)), object(2u, 0.0, 1.0, UINT64_C(0x4))},
                    {{{101u}, std::make_shared<const std::vector<uint64_t>>()},
                     {{102u}, std::make_shared<const std::vector<uint64_t>>()}});
    fixture.publish(population, 3, {object(7u)});
    setWorkersPaused(fixture, false);
    PairReporterResult heldResult = complete(fixture, *owner, held.operation);
    require(heldResult.firstSourceOrdinal == 1 && heldResult.secondSourceOrdinal == 1 &&
                heldResult.entered == std::vector<ovspatialquery_pair_t>{pair(1u, 2u)},
            "admitted immutable lease lost generation membership or participation truth");
    destroyObserved(*owner, held.operation);

    PairReporterAdvance incoherent{};
    incoherent.first               = {TargetSelectionKind::retainedExact, 2};
    incoherent.second              = {TargetSelectionKind::head, 0};
    PairReporterAdmission mismatch = owner->enqueue(incoherent);
    require(mismatch.outcome == OVSPATIALQUERY_API_ORDINAL_MISMATCH,
            "same-population incoherent selections were not rejected before admission");
}

void testFailureAtomicityAndRetry()
{
    Fixture             fixture;
    const PopulationKey first  = fixture.createPopulation();
    const PopulationKey second = fixture.createPopulation();
    fixture.publish(first, 1, {object(1u)});
    fixture.publish(second, 10, {object(2u)});
    std::unique_ptr<PairReporterOperationOwner> owner;
    require(PairReporterOperationOwner::create(fixture.populations(), fixture.instance(), config(first, second),
                                               owner) == OVSPATIALQUERY_API_SUCCESS,
            "failure reporter creation failed");

    failNextScheduling(fixture);
    PairReporterAdmission scheduling = owner->enqueue(headAdvance());
    require(scheduling.outcome == OVSPATIALQUERY_API_INTERNAL && !scheduling.operation.valid(),
            "scheduling failure issued an operation");

    failNextPairMaterializationAllocationForTest();
    PairReporterAdmission  resource = owner->enqueue(headAdvance());
    InstanceOperationState state    = InstanceOperationState::pending;
    require(waitInstanceOperation(fixture.instance(), resource.operation.value, OVSPATIALQUERY_TIMEOUT_INFINITE,
                                  state) == OVSPATIALQUERY_API_OPERATION_FAILED &&
                state == InstanceOperationState::failed,
            "materialization OOM did not become a terminal operation failure");
    ovspatialquery_op_status_t failure{};
    require(ovspatialquery_query_op_status(fixture.instance(), resource.operation.value, &failure).status ==
                    OVSPATIALQUERY_API_SUCCESS &&
                failure.terminal_status == OVSPATIALQUERY_API_OUT_OF_MEMORY,
            "resource failure category was lost");
    destroyObserved(*owner, resource.operation);

    owner->failNextInternalForTest();
    PairReporterAdmission internal = owner->enqueue(headAdvance());
    require(waitInstanceOperation(fixture.instance(), internal.operation.value, OVSPATIALQUERY_TIMEOUT_INFINITE,
                                  state) == OVSPATIALQUERY_API_OPERATION_FAILED,
            "internal failure did not become terminal");
    destroyObserved(*owner, internal.operation);

    // The failures left no baseline: the retry at the same ordinals is admitted
    // and everything enters.
    PairReporterAdmission retry       = owner->enqueue(headAdvance());
    PairReporterResult    retryResult = complete(fixture, *owner, retry.operation);
    require(retryResult.outcome == PairReporterResultOutcome::advanced &&
                retryResult.entered == std::vector<ovspatialquery_pair_t>{pair(1u, 2u)},
            "a failed advance mutated the baseline or the retry was nondeterministic");
    destroyObserved(*owner, retry.operation);

    // Growth needs an idle reporter, so its success also shows that no failure
    // above left a result slot occupied. Successful, invalid and failed growth
    // all keep the accepted baseline, which the next advance's empty delta shows.
    require(owner->growCapacity(16u, 16u) == OVSPATIALQUERY_API_SUCCESS,
            "idle capacity growth failed or a failure leaked a result slot");
    require(owner->growCapacity(16u, 16u) == OVSPATIALQUERY_API_INVALID_ARGUMENT &&
                owner->growCapacity(15u, 16u) == OVSPATIALQUERY_API_INVALID_ARGUMENT,
            "non-growing capacity was accepted");
    require(owner->growCapacity(UINT64_MAX, UINT64_MAX) == OVSPATIALQUERY_API_OUT_OF_MEMORY,
            "unallocatable capacity growth did not fail");

    fixture.publish(second, 11, {object(2u)});
    setWorkersPaused(fixture, true);
    PairReporterAdmission liveGrowth = owner->enqueue(headAdvance());
    require(liveGrowth.outcome == OVSPATIALQUERY_API_SUCCESS &&
                owner->growCapacity(17u, 17u) == OVSPATIALQUERY_API_BUSY,
            "live advance did not reject reporter growth promptly");
    setWorkersPaused(fixture, false);
    PairReporterResult liveGrowthResult = complete(fixture, *owner, liveGrowth.operation);
    require(liveGrowthResult.outcome == PairReporterResultOutcome::advanced && liveGrowthResult.entered.empty() &&
                liveGrowthResult.exited.empty(),
            "growth lost the accepted baseline before a later advance");
    destroyObserved(*owner, liveGrowth.operation);

    PairReporterOperationConfig invalid                  = config(first, second);
    invalid.contactTolerance                             = std::numeric_limits<double>::quiet_NaN();
    std::unique_ptr<PairReporterOperationOwner> sentinel = std::move(owner);
    require(PairReporterOperationOwner::create(fixture.populations(), fixture.instance(), invalid, sentinel) ==
                    OVSPATIALQUERY_API_INVALID_ARGUMENT &&
                sentinel != nullptr,
            "invalid reporter creation replaced caller ownership");
}

void testConcurrentAdmissionAndSlotExhaustion()
{
    Fixture             fixture(3u);
    const PopulationKey first  = fixture.createPopulation();
    const PopulationKey second = fixture.createPopulation();
    fixture.publish(first, 1, {object(1u)});
    fixture.publish(second, 10, {object(2u)});
    std::unique_ptr<PairReporterOperationOwner> owner;
    require(PairReporterOperationOwner::create(fixture.populations(), fixture.instance(),
                                               config(first, second, 4u, 4u, 2u), owner) == OVSPATIALQUERY_API_SUCCESS,
            "concurrency reporter creation failed");
    setWorkersPaused(fixture, true);
    std::atomic<bool>     start{false};
    PairReporterAdmission firstAdmission{};
    PairReporterAdmission secondAdmission{};
    std::thread           firstThread(
        [&]()
        {
            while (!start.load(std::memory_order_acquire)) {}
            firstAdmission = owner->enqueue(headAdvance());
        });
    std::thread secondThread(
        [&]()
        {
            while (!start.load(std::memory_order_acquire)) {}
            secondAdmission = owner->enqueue(headAdvance());
        });
    start.store(true, std::memory_order_release);
    firstThread.join();
    secondThread.join();
    const uint32_t admittedCount = (firstAdmission.outcome == OVSPATIALQUERY_API_SUCCESS ? 1u : 0u) +
                                   (secondAdmission.outcome == OVSPATIALQUERY_API_SUCCESS ? 1u : 0u);
    const uint32_t busyCount = (firstAdmission.outcome == OVSPATIALQUERY_API_BUSY ? 1u : 0u) +
                               (secondAdmission.outcome == OVSPATIALQUERY_API_BUSY ? 1u : 0u);
    require(admittedCount == 1u && busyCount == 1u, "concurrent reporter admissions were not serialized");
    PairReporterOperationId live =
        firstAdmission.operation.valid() ? firstAdmission.operation : secondAdmission.operation;
    setWorkersPaused(fixture, false);
    const PairReporterResult firstLiveResult = complete(fixture, *owner, live);

    fixture.publish(second, 11, {object(2u)});
    PairReporterAdmission coexist       = owner->enqueue(headAdvance());
    PairReporterResult    coexistResult = complete(fixture, *owner, coexist.operation);
    PairReporterResult    firstLiveAgain{};
    require(coexistResult.resultSlot != firstLiveResult.resultSlot &&
                fetchCopy(*owner, live, firstLiveAgain) == OVSPATIALQUERY_API_SUCCESS &&
                firstLiveAgain.resultSlot == firstLiveResult.resultSlot &&
                firstLiveAgain.entered == firstLiveResult.entered,
            "coexisting terminal results did not retain distinct fixed slots");

    std::unique_ptr<PairReporterOperationOwner> operationLimited;
    require(PairReporterOperationOwner::create(fixture.populations(), fixture.instance(),
                                               config(first, second, 4u, 4u, 2u),
                                               operationLimited) == OVSPATIALQUERY_API_SUCCESS,
            "operation-limited reporter creation failed");
    PairReporterAdmission operationHeld = operationLimited->enqueue(headAdvance());
    (void)complete(fixture, *operationLimited, operationHeld.operation);
    fixture.publish(second, 12, {object(2u)});
    require(operationLimited->enqueue(headAdvance()).outcome == OVSPATIALQUERY_API_OPERATION_CAPACITY,
            "live terminal operation did not exhaust operation capacity");
    destroyObserved(*operationLimited, operationHeld.operation);
    destroyObserved(*owner, live);

    std::unique_ptr<PairReporterOperationOwner> resultLimited;
    require(PairReporterOperationOwner::create(fixture.populations(), fixture.instance(),
                                               config(first, second, 4u, 4u, 1u),
                                               resultLimited) == OVSPATIALQUERY_API_SUCCESS,
            "result-limited reporter creation failed");
    PairReporterAdmission resultHeld = resultLimited->enqueue(headAdvance());
    (void)complete(fixture, *resultLimited, resultHeld.operation);
    fixture.publish(second, 13, {object(2u)});
    require(resultLimited->enqueue(headAdvance()).outcome == OVSPATIALQUERY_API_RESULT_CAPACITY,
            "live result did not exhaust reporter result capacity");

    destroyObserved(*owner, coexist.operation);
    destroyObserved(*resultLimited, resultHeld.operation);

    // Destroying a published advance keeps a pending one the only advance.
    std::unique_ptr<PairReporterOperationOwner> gated;
    require(PairReporterOperationOwner::create(fixture.populations(), fixture.instance(),
                                               config(first, second, 4u, 4u, 3u), gated) == OVSPATIALQUERY_API_SUCCESS,
            "gated reporter creation failed");
    PairReporterAdmission published = gated->enqueue(headAdvance());
    (void)complete(fixture, *gated, published.operation);
    fixture.publish(second, 14, {object(2u)});
    setWorkersPaused(fixture, true);
    PairReporterAdmission pending = gated->enqueue(headAdvance());
    require(pending.outcome == OVSPATIALQUERY_API_SUCCESS, "pending reporter advance was not admitted");
    destroyObserved(*gated, published.operation);
    require(gated->enqueue(headAdvance()).outcome == OVSPATIALQUERY_API_BUSY,
            "destroying a published advance admitted a second pending advance");
    setWorkersPaused(fixture, false);
    (void)complete(fixture, *gated, pending.operation);
    destroyObserved(*gated, pending.operation);
}

void testWaitRevalidatesGenerationAndPreservesLargeTimeouts()
{
    Fixture             fixture;
    const PopulationKey first  = fixture.createPopulation();
    const PopulationKey second = fixture.createPopulation();
    fixture.publish(first, 1, {object(1u)});
    fixture.publish(second, 10, {object(2u)});
    std::unique_ptr<PairReporterOperationOwner> owner;
    require(PairReporterOperationOwner::create(fixture.populations(), fixture.instance(),
                                               config(first, second, 4u, 4u, 1u), owner) == OVSPATIALQUERY_API_SUCCESS,
            "wait-race reporter creation failed");

    setWorkersPaused(fixture, true);
    setWaiterPaused(fixture, true);
    const uint64_t              staleEntriesBefore = waiterEntryCount(fixture);
    PairReporterAdmission       firstAdmission     = owner->enqueue(headAdvance());
    ovspatialquery_api_status_t staleWaitOutcome   = OVSPATIALQUERY_API_SUCCESS;
    InstanceOperationState      staleWaitState     = InstanceOperationState::pending;
    std::thread                 staleWaiter(
        [&]()
        {
            staleWaitOutcome = waitInstanceOperation(fixture.instance(), firstAdmission.operation.value,
                                                                     OVSPATIALQUERY_TIMEOUT_INFINITE, staleWaitState);
        });
    while (waiterEntryCount(fixture) == staleEntriesBefore)
    {
        std::this_thread::yield();
    }
    setWorkersPaused(fixture, false);
    waitForWaiter(fixture);

    InstanceOperationState observed = InstanceOperationState::pending;
    require(queryInstanceOperation(fixture.instance(), firstAdmission.operation.value, observed) ==
                    OVSPATIALQUERY_API_SUCCESS &&
                observed == InstanceOperationState::completed,
            "race setup did not observe the first terminal operation");
    destroyObserved(*owner, firstAdmission.operation);
    fixture.publish(second, 11, {object(2u)});
    setWorkersPaused(fixture, true);
    PairReporterAdmission reused = owner->enqueue(headAdvance());
    require(reused.outcome == OVSPATIALQUERY_API_SUCCESS, "race setup did not reuse the released operation slot");

    setWaiterPaused(fixture, false);
    staleWaiter.join();
    require(staleWaitOutcome == OVSPATIALQUERY_API_INVALID_HANDLE,
            "stale waiter crossed an operation-slot generation reuse");

    const uint64_t              entriesBefore    = waiterEntryCount(fixture);
    ovspatialquery_api_status_t largeWaitOutcome = OVSPATIALQUERY_API_TIMEOUT;
    InstanceOperationState      largeWaitState   = InstanceOperationState::pending;
    std::thread                 largeWaiter(
        [&]()
        {
            largeWaitOutcome =
                waitInstanceOperation(fixture.instance(), reused.operation.value,
                                                      static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) + 1u, largeWaitState);
        });
    while (waiterEntryCount(fixture) == entriesBefore)
    {
        std::this_thread::yield();
    }
    setWorkersPaused(fixture, false);
    largeWaiter.join();
    require(largeWaitOutcome == OVSPATIALQUERY_API_SUCCESS && largeWaitState == InstanceOperationState::completed,
            "finite timeout above INT64_MAX narrowed or timed out prematurely");
    destroyObserved(*owner, reused.operation);
}

void testExecutionPeakUsesAtomicHostLeaseAndReleasesAfterBaselineSwap()
{
    Fixture             fixture(1u);
    const PopulationKey first  = fixture.createPopulation();
    const PopulationKey second = fixture.createPopulation();
    fixture.publish(first, 1, {object(1u)});
    fixture.publish(second, 10, {object(2u)});

    const auto makeOwner = [&](InstanceHostLedger& ledger, double tolerance = 0.0)
    {
        PairReporterOperationConfig value = config(first, second, 2u, 2u, 1u);
        value.hostLedger                  = &ledger;
        value.contactTolerance            = tolerance;
        std::unique_ptr<PairReporterOperationOwner> owner;
        require(PairReporterOperationOwner::create(fixture.populations(), fixture.instance(), value, owner) ==
                    OVSPATIALQUERY_API_SUCCESS,
                "ledger reporter creation failed");
        return owner;
    };

    InstanceHostLedger                          calibrationLedger(UINT64_MAX);
    std::unique_ptr<PairReporterOperationOwner> calibration = makeOwner(calibrationLedger);
    setWorkersPaused(fixture, true);
    PairReporterAdmission calibrationAdmission = calibration->enqueue(headAdvance());
    require(calibrationAdmission.outcome == OVSPATIALQUERY_API_SUCCESS, "ledger reporter calibration was not admitted");
    const uint64_t executionCharge = calibrationLedger.total();
    require(executionCharge != 0u, "ledger reporter peak was not charged");
    setWorkersPaused(fixture, false);
    (void)complete(fixture, *calibration, calibrationAdmission.operation);
    require(calibrationLedger.total() == executionCharge,
            "ledger reporter live task charge changed before operation destroy");
    destroyObserved(*calibration, calibrationAdmission.operation);
    require(calibrationLedger.total() == 0u && calibration->close() == OVSPATIALQUERY_API_SUCCESS,
            "ledger reporter task charge did not release at destroy");
    calibration.reset();

    // A tolerance query needs duplicate-safe traversal scratch in addition to
    // the zero-tolerance execution lease. Admission must reject before work.
    InstanceHostLedger traversalLedger(executionCharge);
    auto               traversal          = makeOwner(traversalLedger, 0.5);
    const auto         traversalAdmission = traversal->enqueue(headAdvance());
    require(traversalAdmission.outcome == OVSPATIALQUERY_API_HOST_BUDGET && !traversalAdmission.operation.valid() &&
                traversalLedger.total() == 0u,
            "pair traversal scratch bypassed the execution budget");
    require(traversal->close() == OVSPATIALQUERY_API_SUCCESS, "budget-rejected traversal reporter did not close");
    traversal.reset();

    InstanceHostLedger                          rejectedLedger(executionCharge - 1u);
    std::unique_ptr<PairReporterOperationOwner> rejected  = makeOwner(rejectedLedger);
    const PairReporterAdmission                 rejection = rejected->enqueue(headAdvance());
    require(rejection.outcome == OVSPATIALQUERY_API_HOST_BUDGET && !rejection.operation.valid() &&
                rejectedLedger.total() == 0u && rejected->close() == OVSPATIALQUERY_API_SUCCESS,
            "ledger reporter charge-1 rejection left a charge or a result slot");
    rejected.reset();

    InstanceHostLedger                          exactLedger(executionCharge);
    std::unique_ptr<PairReporterOperationOwner> exact = makeOwner(exactLedger);
    setWorkersPaused(fixture, true);
    PairReporterAdmission exactAdmission = exact->enqueue(headAdvance());
    require(exactAdmission.outcome == OVSPATIALQUERY_API_SUCCESS && exactLedger.total() == executionCharge,
            "ledger reporter exact-boundary advance was not admitted");
    setWorkersPaused(fixture, false);
    const PairReporterResult result = complete(fixture, *exact, exactAdmission.operation);
    require(result.outcome == PairReporterResultOutcome::advanced &&
                result.entered == std::vector<ovspatialquery_pair_t>{pair(1u, 2u)} &&
                exactLedger.total() == executionCharge,
            "ledger reporter exact-boundary live charge was incorrect");
    gReporterTaskDestroyLedger         = &exactLedger;
    gReporterTaskDestroyObservedCharge = UINT64_MAX;
    setPairReporterTaskPayloadDestroyHookForTest(&observeReporterTaskPayloadDestroy);
    destroyObserved(*exact, exactAdmission.operation);
    setPairReporterTaskPayloadDestroyHookForTest(nullptr);
    gReporterTaskDestroyLedger = nullptr;
    require(gReporterTaskDestroyObservedCharge == executionCharge,
            "reporter task charge released before charged task storage retired");
    require(exactLedger.total() == 0u && exact->close() == OVSPATIALQUERY_API_SUCCESS,
            "ledger reporter exact-boundary owner did not close");
}

} // namespace

int main()
{
    testGenerationTruthSamePopulationAndHeldLeases();
    testFailureAtomicityAndRetry();
    testConcurrentAdmissionAndSlotExhaustion();
    testWaitRevalidatesGenerationAndPreservesLargeTimeouts();
    testExecutionPeakUsesAtomicHostLeaseAndReleasesAfterBaselineSwap();
    return 0;
}
