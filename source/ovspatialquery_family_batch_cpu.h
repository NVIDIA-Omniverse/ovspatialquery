/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_FAMILY_BATCH_CPU_H
#define OVSPATIALQUERY_FAMILY_BATCH_CPU_H

#include "ovspatialquery_query_evaluation.h"
#include "ovspatialquery_shared_worker_loan.h"
#include "ovspatialquery_target_resolver.h"

#include <cstdint>
#include <vector>

namespace ovspatialquery
{
namespace detail
{

enum class FamilyBatchOutcome : uint8_t
{
    success,
    invalidArgument,
    hostBudget,
    outOfMemory,
    internalFailure,
};

enum class RowOutcome : uint8_t
{
    success,
    invalidArgument,
    notSupported,
    failed,
};

struct FamilyBatchResultCommon
{
    // Successful execution always records the exact resolved population,
    // including zero-row and no-hit results.
    PopulationKey            sourcePopulation{};
    ovspatialquery_ordinal_t sourceOrdinal  = 0;
    HitTruth                 truth          = HitTruth::boundsCandidate;
    uint64_t                 hitCapacity    = 0u;
    uint64_t                 storedHitCount = 0u;
    std::vector<RowOutcome>  rowOutcomes;
    std::vector<uint64_t>    hitOffsets;
    std::vector<uint64_t>    hitCounts;
    std::vector<uint8_t>     truncated;
};

struct RaycastBatchPlan
{
    HitMode hitMode = HitMode::closest;
    // MULTIPLE only: report every triangle a mesh's surface crosses, not only its nearest.
    bool                allMeshHits         = false;
    QueryPrecision      precision           = QueryPrecision::boundsCandidate;
    uint64_t            maximumQueryCount   = 0u;
    uint64_t            maximumHitsPerQuery = 1u;
    uint64_t            hitCapacity         = 0u;
    double              contactTolerance    = 0.0;
    InstanceHostLedger* hostLedger          = nullptr;
    SharedWorkerLoan    workers{};
};

struct RaycastBatchRow
{
    RaycastQuery      query{};
    ParticipationBits queryBits{};
    uint64_t          maximumHits = UINT64_MAX;
};

struct RaycastBatchLaunch
{
    std::vector<RaycastBatchRow> rows;
};

struct RaycastBatchResult
{
    FamilyBatchResultCommon         common;
    std::vector<RaycastSemanticHit> hits;
};

struct SphereSweepBatchPlan
{
    HitMode             hitMode             = HitMode::closest;
    QueryPrecision      precision           = QueryPrecision::boundsCandidate;
    bool                box                 = false;
    uint64_t            maximumQueryCount   = 0u;
    uint64_t            maximumHitsPerQuery = 1u;
    uint64_t            hitCapacity         = 0u;
    double              contactTolerance    = 0.0;
    InstanceHostLedger* hostLedger          = nullptr;
    SharedWorkerLoan    workers{};
};

struct SphereSweepBatchRow
{
    SphereSweepQuery  query{};
    ParticipationBits queryBits{};
};

struct SphereSweepBatchLaunch
{
    std::vector<SphereSweepBatchRow> rows;
};

struct SphereSweepBatchResult
{
    FamilyBatchResultCommon             common;
    std::vector<SphereSweepSemanticHit> hits;
};

struct OverlapBatchPlan
{
    HitMode             hitMode             = HitMode::any;
    QueryPrecision      precision           = QueryPrecision::boundsCandidate;
    uint64_t            maximumQueryCount   = 0u;
    uint64_t            maximumHitsPerQuery = 1u;
    uint64_t            hitCapacity         = 0u;
    double              contactTolerance    = 0.0;
    InstanceHostLedger* hostLedger          = nullptr;
    SharedWorkerLoan    workers{};
};

struct OverlapBatchRow
{
    OverlapQuery      query{};
    ParticipationBits queryBits{};
    uint64_t          maximumHits = UINT64_MAX;
};

struct OverlapBatchLaunch
{
    std::vector<OverlapBatchRow> rows;
};

struct OverlapBatchResult
{
    FamilyBatchResultCommon         common;
    std::vector<OverlapSemanticHit> hits;
};

// All overloads consume an already-resolved immutable A2 target. Structural
// failure preserves outResult. Work is O(rows * selected traversal) with
// O(rows + hitCapacity + one-row candidates) transient host memory. A launch of
// enough rows is split into chunks that plan.workers' idle workers evaluate
// beside the caller when plan.hostLedger admits their row candidates and staged
// hits; the result is the one an in-order evaluation stores.
FamilyBatchOutcome executeFamilyBatchCpu(const ResolvedTarget& target, const RaycastBatchPlan& plan,
                                         const RaycastBatchLaunch& launch, RaycastBatchResult& outResult) noexcept;

FamilyBatchOutcome executeFamilyBatchCpu(const ResolvedTarget& target, const SphereSweepBatchPlan& plan,
                                         const SphereSweepBatchLaunch& launch,
                                         SphereSweepBatchResult&       outResult) noexcept;

FamilyBatchOutcome executeFamilyBatchCpu(const ResolvedTarget& target, const OverlapBatchPlan& plan,
                                         const OverlapBatchLaunch& launch, OverlapBatchResult& outResult) noexcept;

// One synchronous query run as one row of `plan`: resolves and pins the
// target for the call, copies a readable device version to the host, charges
// the row's execution peak (and, in MULTIPLE mode, room for one hit per object
// up to maximumHitsPerQuery) to plan.hostLedger for the call, and evaluates the
// row as the batch executor does. Returns the public status; on SUCCESS
// outTarget is the resolved target and outHits the row's evaluated hits.
ovspatialquery_api_status_t executeFamilyRowCpu(const TargetResolver& resolver, const TargetRequest& request,
                                                const RaycastBatchPlan& plan, const RaycastBatchRow& row,
                                                ResolvedTarget&                  outTarget,
                                                std::vector<RaycastSemanticHit>& outHits) noexcept;
ovspatialquery_api_status_t executeFamilyRowCpu(const TargetResolver& resolver, const TargetRequest& request,
                                                const SphereSweepBatchPlan& plan, const SphereSweepBatchRow& row,
                                                ResolvedTarget&                      outTarget,
                                                std::vector<SphereSweepSemanticHit>& outHits) noexcept;
ovspatialquery_api_status_t executeFamilyRowCpu(const TargetResolver& resolver, const TargetRequest& request,
                                                const OverlapBatchPlan& plan, const OverlapBatchRow& row,
                                                ResolvedTarget&                  outTarget,
                                                std::vector<OverlapSemanticHit>& outHits) noexcept;

bool estimateFamilyBatchExecutionPeakBytes(const ResolvedTarget& target, const RaycastBatchPlan& plan,
                                           uint64_t& outBytes) noexcept;
bool estimateFamilyBatchExecutionPeakBytes(const ResolvedTarget& target, const SphereSweepBatchPlan& plan,
                                           uint64_t& outBytes) noexcept;
bool estimateFamilyBatchExecutionPeakBytes(const ResolvedTarget& target, const OverlapBatchPlan& plan,
                                           uint64_t& outBytes) noexcept;

} // namespace detail
} // namespace ovspatialquery

#endif
