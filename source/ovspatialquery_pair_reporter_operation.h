/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_PAIR_REPORTER_OPERATION_H
#define OVSPATIALQUERY_PAIR_REPORTER_OPERATION_H

#include "ovspatialquery_instance_host_ledger.h"
#include "ovspatialquery_instance_operation_runtime.h"
#include "ovspatialquery_pair_materialization.h"
#include "ovspatialquery_pair_reporter_transition.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ovspatialquery
{
namespace detail
{

struct PairReporterOperationId
{
    ovspatialquery_op_id_t value = OVSPATIALQUERY_INVALID_OP;

    bool valid() const noexcept { return value != OVSPATIALQUERY_INVALID_OP; }
};

struct PairReporterOperationEndpoint
{
    PopulationTarget  target;
    ParticipationBits queryBits{};
};

struct PairReporterOperationConfig
{
    PairReporterOperationEndpoint first;
    PairReporterOperationEndpoint second;
    PairEndpointRelation          relation           = PairEndpointRelation::distinctPopulations;
    double                        contactTolerance   = 0.0;
    uint64_t                      activePairCapacity = 0u;
    uint64_t                      deltaPairCapacity  = 0u;
    uint32_t                      resultSlotCount    = 0u;
    InstanceHostLedger*           hostLedger         = nullptr;
    // kDLCPU, or the CUDA device the reporter keeps its accepted set and
    // results on (ADR-0019 stage 3).
    DLDevice resultDevice{kDLCPU, 0};
};

struct PairReporterAdvance
{
    TargetSelection first;
    TargetSelection second;
    // A CUDA reporter's stream and wait event (ADR-0022 decision 1); without a
    // stream it keeps the reporter's own (ADR-0060). CPU reporters ignore them.
    bool      hasStream = false;
    uintptr_t stream    = 0u;
    uintptr_t waitEvent = 0u;
};

struct PairReporterAdmission
{
    ovspatialquery_api_status_t outcome = OVSPATIALQUERY_API_INTERNAL;
    PairReporterOperationId     operation;
    uint32_t                    resultSlot = UINT32_MAX;
};

struct PairReporterResult
{
    uint32_t                           resultSlot              = UINT32_MAX;
    PairReporterResultOutcome          outcome                 = PairReporterResultOutcome::advanced;
    ovspatialquery_ordinal_t           firstSourceOrdinal      = 0;
    ovspatialquery_ordinal_t           secondSourceOrdinal     = 0;
    uint64_t                           acceptedActivePairCount = 0u;
    std::vector<ovspatialquery_pair_t> entered;
    std::vector<ovspatialquery_pair_t> exited;
};

struct PairReporterResultStorageView
{
    uint32_t                     resultSlot              = UINT32_MAX;
    PairReporterResultOutcome    outcome                 = PairReporterResultOutcome::advanced;
    ovspatialquery_ordinal_t     firstSourceOrdinal      = 0;
    ovspatialquery_ordinal_t     secondSourceOrdinal     = 0;
    uint64_t                     acceptedActivePairCount = 0u;
    const ovspatialquery_pair_t* entered                 = nullptr;
    uint64_t                     enteredCount            = 0u;
    const ovspatialquery_pair_t* exited                  = nullptr;
    uint64_t                     exitedCount             = 0u;
    // A CUDA reporter's result: device rows (uint64[N,2]) instead of host pairs,
    // its counts (uint64[4]: outcome, entered, exited, accepted) and the event
    // the device signals once they are written. The reserved view names the same
    // storage before the advance completes (ADR-0022 decision 2).
    DLDevice                    device{kDLCPU, 0};
    const void*                 deviceEntered = nullptr;
    const void*                 deviceExited  = nullptr;
    const void*                 deviceCounts  = nullptr;
    ovspatialquery_cuda_event_t resultReady{};
};

#if defined(OVSPATIALQUERY_ENABLE_PAIR_REPORTER_OPERATION_TEST_HOOK)
using PairReporterTaskPayloadDestroyHook = void (*)();
void setPairReporterTaskPayloadDestroyHookForTest(PairReporterTaskPayloadDestroyHook hook) noexcept;
/** Advances in this process that patched their accepted set; each also matched a complete recomputation. */
uint64_t patchedPairReporterAdvanceCountForTest() noexcept;
/** CUDA reporter advances in this process that patched; each matched a CPU recomputation. */
uint64_t patchedCudaPairReporterAdvanceCountForTest() noexcept;
#endif

/**
 * Private CPU reporter admission, execution, typed-result, and lifecycle owner.
 *
 * Creation retains both target populations. Admission atomically resolves both
 * selections, applies backward-before-no-progress policy without reserving an
 * operation or result slot, and admits at most one advance at a time. An
 * admitted operation owns both immutable resolved leases and one fixed result
 * slot until terminal-observed destruction. Only an `advanced` terminal result
 * replaces the accepted baseline.
 *
 * A reporter with a CUDA result device keeps its accepted set and results on
 * the device and enqueues each advance's device work at admission, on the
 * caller's stream when the advance names one; its worker only waits for the
 * work and reads the outcome. Such a reporter may advance over
 * device-prepared versions on its device, admitted or complete (ADR-0024
 * decision 4, ADR-0022); an endpoint whose publication fails makes the
 * advance fail with the baseline preserved.
 *
 * An advance from an accepted baseline patches it from the change records of
 * the versions in between when it can (ADR-0019), evaluating only changed
 * objects, and otherwise recomputes the complete set; both give the same
 * result. Either way scratch is bounded by the configured active-pair
 * capacity, both delta capacities and the traversal scratch charged at
 * admission; the reporter keeps the previous accepted set's buffer for the
 * next patch and is charged for two. Target resolution, participation
 * truth, pair materialization, and baseline transition are delegated to their
 * existing private seams; every call returns the public API status, and
 * public handle projection and backend policy remain out of this owner. Explicit destruction is non-waiting:
 * `destroy()` preserves the owner and returns BUSY until every operation has been observed and destroyed.
 */
class PairReporterOperationOwner
{
  public:
    static bool estimateResidentBytes(const PairReporterOperationConfig& config, uint64_t& outBytes) noexcept;
    static ovspatialquery_api_status_t create(PopulationTable& populations, ovspatialquery_instance_handle_t instance,
                                              const PairReporterOperationConfig&           config,
                                              std::unique_ptr<PairReporterOperationOwner>& outOwner) noexcept;
    static ovspatialquery_api_status_t destroy(std::unique_ptr<PairReporterOperationOwner>& owner) noexcept;

    ~PairReporterOperationOwner();

    PairReporterOperationOwner(const PairReporterOperationOwner&)            = delete;
    PairReporterOperationOwner& operator=(const PairReporterOperationOwner&) = delete;

    PairReporterAdmission       enqueue(const PairReporterAdvance& advance) noexcept;
    ovspatialquery_api_status_t growCapacity(uint64_t activePairCapacity, uint64_t deltaPairCapacity) noexcept;
    ovspatialquery_api_status_t fetchStorageView(PairReporterOperationId        operation,
                                                 PairReporterResultStorageView& outView) const noexcept;
    ovspatialquery_api_status_t reservedStorageView(PairReporterOperationId        operation,
                                                    PairReporterResultStorageView& outView) const noexcept;
    /** Records that a typed fetch handed a CUDA advance's device views to the
     * caller, which then retires the advance after their consumers. */
    ovspatialquery_api_status_t noteViewsReturned(PairReporterOperationId operation) noexcept;
    ovspatialquery_api_status_t destroyOperation(PairReporterOperationId& operation) noexcept;
    /** Retires a CUDA reporter advance after the consumers `consumerDone` covers. */
    ovspatialquery_api_status_t destroyOperationAfterCuda(PairReporterOperationId&     operation,
                                                          ovspatialquery_cuda_event_t& consumerDone) noexcept;
    /** Checks explicit destruction eligibility without waiting. */
    ovspatialquery_api_status_t close() noexcept;

#if defined(OVSPATIALQUERY_ENABLE_PAIR_REPORTER_OPERATION_TEST_HOOK)
    void failNextInternalForTest() noexcept;
#endif

  private:
    friend struct PairReporterRuntimeTask;
    friend class PairReporterCudaRetirement;
    struct Impl;
    explicit PairReporterOperationOwner(std::shared_ptr<Impl> impl) noexcept;

    std::shared_ptr<Impl> mImpl;
};

} // namespace detail
} // namespace ovspatialquery

#endif
