/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_FAMILY_BATCH_PLAN_OPERATION_H
#define OVSPATIALQUERY_FAMILY_BATCH_PLAN_OPERATION_H

#include "ovspatialquery_cuda_family_plan.h"
#include "ovspatialquery_family_batch_columns.h"
#include "ovspatialquery_instance_host_ledger.h"
#include "ovspatialquery_instance_operation_runtime.h"

#include <cstdint>
#include <memory>
#include <string>

namespace ovspatialquery
{
namespace detail
{

struct FamilyBatchPlanOperationId
{
    ovspatialquery_op_id_t value = OVSPATIALQUERY_INVALID_OP;

    bool valid() const noexcept { return value != OVSPATIALQUERY_INVALID_OP; }
};

enum class FamilyBatchPlanBackend : uint8_t
{
    cpu,
    cuda,
};

struct FamilyBatchPlanCommonConfig
{
    PopulationTarget       target;
    FamilyBatchPlanBackend backend         = FamilyBatchPlanBackend::cpu;
    uint32_t               resultSlotCount = 0u;
    InstanceHostLedger*    hostLedger      = nullptr;
    DLDevice               cudaDevice{kDLCUDA, 0};
    uint32_t               resultFields     = 0u;
    uint32_t               cudaResultFields = 0u;
    uint32_t               cudaOverlapKind  = cuda_family::kOverlapSphere;
    // Result semantics remain exact; only the CUDA evaluator gets this permission.
    bool cudaAllowSinglePrecision = false;
    // The CUDA plan keeps its result columns in mapped host memory (ADR-0077).
    bool   cudaHostMapped         = false;
    double cudaPlanRadius         = 0.0;
    double cudaPlanHalfExtents[3] = {};
    // The instance's CUDA mesh definitions, shared by its CUDA plans.
    CudaMeshResidencyRegistry* cudaMeshResidency = nullptr;
};

struct RaycastFamilyBatchPlanConfig
{
    FamilyBatchPlanCommonConfig common;
    RaycastBatchPlan            plan;
};

struct SphereSweepFamilyBatchPlanConfig
{
    FamilyBatchPlanCommonConfig common;
    SphereSweepBatchPlan        plan;
};

struct OverlapFamilyBatchPlanConfig
{
    FamilyBatchPlanCommonConfig common;
    OverlapBatchPlan            plan;
};

struct FamilyBatchPlanAdmission
{
    ovspatialquery_api_status_t outcome = OVSPATIALQUERY_API_INTERNAL;
    FamilyBatchPlanOperationId  operation;
    uint32_t                    resultSlot = UINT32_MAX;
};

#if defined(OVSPATIALQUERY_ENABLE_FAMILY_BATCH_PLAN_OPERATION_TEST_HOOK)
using FamilyBatchPlanRetiredSlotHook = void (*)();
void setFamilyBatchPlanRetiredSlotHookForTest(FamilyBatchPlanRetiredSlotHook hook) noexcept;
using FamilyBatchPlanTaskPayloadDestroyHook = void (*)();
void setFamilyBatchPlanTaskPayloadDestroyHookForTest(FamilyBatchPlanTaskPayloadDestroyHook hook) noexcept;
#endif

/**
 * Private CPU family-plan target, fixed-storage, operation, and lifecycle owner.
 *
 * The owner retains one population/view target but resolves HEAD or one exact
 * retained ordinal only when a launch is admitted; every call returns the
 * public API status. Public handle and result projection remain outside this
 * private module. Explicit
 * destruction is non-waiting: `destroy()` preserves the owner and returns
 * BUSY until every operation has been observed and destroyed.
 */
class FamilyBatchPlanOperationOwner
{
  public:
    static bool estimateResidentBytes(const RaycastFamilyBatchPlanConfig& config, uint64_t& outBytes) noexcept;
    static bool estimateResidentBytes(const SphereSweepFamilyBatchPlanConfig& config, uint64_t& outBytes) noexcept;
    static bool estimateResidentBytes(const OverlapFamilyBatchPlanConfig& config, uint64_t& outBytes) noexcept;
    static ovspatialquery_api_status_t create(PopulationTable& populations, ovspatialquery_instance_handle_t instance,
                                              const RaycastFamilyBatchPlanConfig&             config,
                                              std::unique_ptr<FamilyBatchPlanOperationOwner>& outOwner) noexcept;
    static ovspatialquery_api_status_t create(PopulationTable& populations, ovspatialquery_instance_handle_t instance,
                                              const SphereSweepFamilyBatchPlanConfig&         config,
                                              std::unique_ptr<FamilyBatchPlanOperationOwner>& outOwner) noexcept;
    static ovspatialquery_api_status_t create(PopulationTable& populations, ovspatialquery_instance_handle_t instance,
                                              const OverlapFamilyBatchPlanConfig&             config,
                                              std::unique_ptr<FamilyBatchPlanOperationOwner>& outOwner) noexcept;
    static ovspatialquery_api_status_t destroy(std::unique_ptr<FamilyBatchPlanOperationOwner>& owner) noexcept;

    ~FamilyBatchPlanOperationOwner();

    FamilyBatchPlanOperationOwner(const FamilyBatchPlanOperationOwner&)            = delete;
    FamilyBatchPlanOperationOwner& operator=(const FamilyBatchPlanOperationOwner&) = delete;

    FamilyBatchPlanAdmission enqueueColumns(TargetSelection selection, const RaycastBatchLaunch& launch,
                                            ovspatialquery_raycast_batch_reservation_t& outReservation) noexcept;
    FamilyBatchPlanAdmission enqueueColumns(TargetSelection selection, const SphereSweepBatchLaunch& launch,
                                            ovspatialquery_sphere_sweep_batch_reservation_t& outReservation) noexcept;
    FamilyBatchPlanAdmission enqueueColumns(TargetSelection selection, const OverlapBatchLaunch& launch,
                                            ovspatialquery_overlap_batch_reservation_t& outReservation) noexcept;
    // The same admissions without a copy: an admitted launch's rows move into its result slot
    // and `launch` is left holding that slot's previous rows, storage the caller may reuse.
    FamilyBatchPlanAdmission    enqueueColumns(TargetSelection selection, RaycastBatchLaunch&& launch,
                                               ovspatialquery_raycast_batch_reservation_t& outReservation) noexcept;
    FamilyBatchPlanAdmission    enqueueColumns(TargetSelection selection, SphereSweepBatchLaunch&& launch,
                                               ovspatialquery_sphere_sweep_batch_reservation_t& outReservation) noexcept;
    FamilyBatchPlanAdmission    enqueueColumns(TargetSelection selection, OverlapBatchLaunch&& launch,
                                               ovspatialquery_overlap_batch_reservation_t& outReservation) noexcept;
    FamilyBatchPlanAdmission    enqueueCuda(TargetSelection selection, const CudaFamilyLaunch& launch,
                                            CudaFamilyResultColumns& outReservation) noexcept;
    ovspatialquery_api_status_t fetchCuda(FamilyBatchPlanOperationId operation,
                                          CudaFamilyResultColumns&   outResult) const noexcept;
    ovspatialquery_api_status_t fetchColumns(FamilyBatchPlanOperationId             operation,
                                             ovspatialquery_raycast_batch_result_t& outResult) const noexcept;
    ovspatialquery_api_status_t fetchColumns(FamilyBatchPlanOperationId                  operation,
                                             ovspatialquery_sphere_sweep_batch_result_t& outResult) const noexcept;
    ovspatialquery_api_status_t fetchColumns(FamilyBatchPlanOperationId             operation,
                                             ovspatialquery_overlap_batch_result_t& outResult) const noexcept;
    ovspatialquery_api_status_t destroyOperation(FamilyBatchPlanOperationId& operation) noexcept;
    ovspatialquery_api_status_t destroyOperationAfterCuda(FamilyBatchPlanOperationId&  operation,
                                                          ovspatialquery_cuda_event_t& consumerDone) noexcept;

    /** Checks explicit destruction eligibility without waiting. */
    ovspatialquery_api_status_t close() noexcept;

#if defined(OVSPATIALQUERY_ENABLE_FAMILY_BATCH_PLAN_OPERATION_TEST_HOOK)
    void failNextResourceForTest() noexcept;
    void failNextBackendForTest() noexcept;
    void failNextInternalForTest() noexcept;
#endif

  private:
    friend struct FamilyBatchPlanRuntimeTask;
    friend class FamilyBatchPlanCudaRetirement;
    struct Impl;
    template <typename Policy, typename Config>
    static ovspatialquery_api_status_t createTyped(PopulationTable&                 populations,
                                                   ovspatialquery_instance_handle_t instance, const Config& config,
                                                   std::unique_ptr<FamilyBatchPlanOperationOwner>& outOwner) noexcept;
    explicit FamilyBatchPlanOperationOwner(std::shared_ptr<Impl> impl) noexcept;

    std::shared_ptr<Impl> mImpl;
};

} // namespace detail
} // namespace ovspatialquery

#endif
