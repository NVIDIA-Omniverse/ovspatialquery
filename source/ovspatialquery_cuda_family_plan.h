/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_CUDA_FAMILY_PLAN_H
#define OVSPATIALQUERY_CUDA_FAMILY_PLAN_H

#include "ovspatialquery_cuda_backend.h"
#include "ovspatialquery_cuda_family_plan_types.h"
#include "ovspatialquery_cuda_mesh_residency.h"
#include "ovspatialquery_device_signal.h"
#include "ovspatialquery_input_layout.h"
#include "ovspatialquery_instance_host_ledger.h"
#include "ovspatialquery_population_table.h"
#include "ovspatialquery_shared_worker_loan.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace ovspatialquery
{
namespace detail
{

enum class CudaCompletionState : uint32_t;

enum class CudaFamilyKind : uint32_t
{
    raycast     = cuda_family::kFamilyRaycast,
    sphereSweep = cuda_family::kFamilySphereSweep,
    overlap     = cuda_family::kFamilyOverlap,
    boxSweep    = cuda_family::kFamilyBoxSweep,
};

enum class CudaFamilyPrecision : uint32_t
{
    boundsCandidate = cuda_family::kPrecisionBoundsCandidate,
    exact           = cuda_family::kPrecisionExact,
    exactSingle     = cuda_family::kPrecisionExactSingle,
};

enum class CudaFamilyHitMode : uint32_t
{
    closest  = cuda_family::kHitModeClosest,
    any      = cuda_family::kHitModeAny,
    multiple = cuda_family::kHitModeMultiple,
};

enum class CudaOverlapKind : uint32_t
{
    sphere = cuda_family::kOverlapSphere,
    box    = cuda_family::kOverlapBox,
    aabb   = cuda_family::kOverlapAabb,
};

constexpr uint32_t kCudaFamilyResultPosition         = cuda_family::kResultPosition;
constexpr uint32_t kCudaFamilyResultNormal           = cuda_family::kResultNormal;
constexpr uint32_t kCudaFamilyResultDistance         = cuda_family::kResultDistance;
constexpr uint32_t kCudaFamilyResultSourceFeature    = cuda_family::kResultSourceFeature;
constexpr uint32_t kCudaFamilyResultGeometryIdentity = cuda_family::kResultGeometryIdentity;
constexpr uint32_t kCudaFamilyResultQueryPosition    = cuda_family::kResultQueryPosition;
constexpr uint32_t kCudaFamilyResultMaterial         = cuda_family::kResultMaterial;

struct CudaFamilyPlanConfig
{
    CudaFamilyKind      family    = CudaFamilyKind::raycast;
    CudaFamilyPrecision precision = CudaFamilyPrecision::boundsCandidate;
    CudaFamilyHitMode   hitMode   = CudaFamilyHitMode::closest;
    // Raycast MULTIPLE only: every triangle a mesh's surface crosses is a hit.
    bool            allMeshHits = false;
    CudaOverlapKind overlapKind = CudaOverlapKind::sphere;
    DLDevice        device{kDLCUDA, 0};
    uint32_t        resultFields = 0u;
    // Every result column of every slot lives in mapped host memory (ADR-0077).
    bool     hostMapped          = false;
    uint64_t maximumQueryCount   = 0u;
    uint64_t maximumHitsPerQuery = 1u;
    uint64_t hitCapacityPerSlot  = 0u;
    uint32_t resultSlotCount     = 0u;
    double   contactTolerance    = 0.0;
    // Sweep and overlap plans carry one default shape; a row's own size column replaces it.
    double planRadius         = 0.0;
    double planHalfExtents[3] = {};
    // The instance's idle workers, which help build a launch's new mesh trees.
    SharedWorkerLoan workers{};
    // The instance's resident mesh definitions, which the plan shares with the
    // instance's other plans on its device; read only while the plan is created.
    // Null gives the plan an index of its own.
    CudaMeshResidencyRegistry* meshResidency = nullptr;
};

struct CudaFamilyInput;

// Reads a present input's layout (ADR-0053) while the caller's shape and
// strides are still live; `column` names it in a refusal's diagnostic.
void readCudaFamilyInputLayout(CudaFamilyInput& input, const char* column) noexcept;

struct CudaFamilyInput
{
    DLTensor tensor{};
    uint32_t ownership = UINT32_MAX;
    bool     present   = false;
    // Read at admission; a non-compact column is copied before the kernels run.
    InputColumnLayout layout{};
    bool              layoutValid = false;

#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
    // Test construction convenience: a direct tensor assignment is explicitly
    // borrowed; the descriptor remains wrapped so preparation can retain or
    // copy it under the canonical lifetime policy.  The engine sets the three
    // members itself, so the shipped library does not carry this operator.
    CudaFamilyInput& operator=(const DLTensor& value) noexcept
    {
        tensor    = value;
        ownership = 0u;
        present   = true;
        readCudaFamilyInputLayout(*this, "input");
        return *this;
    }
#endif

    operator const DLTensor&() const noexcept { return tensor; }
};

/**
 * The public batch API transports DLPack wrappers. Borrowed columns remain
 * live until terminal observation; copied columns are staged on the supplied
 * CUDA stream before enqueue returns. Keep the ownership metadata intact
 * until that private staging decision has been made.
 */
struct CudaFamilyLaunch
{
    uint64_t        queryCount = 0u;
    CudaFamilyInput queryBits;
    CudaFamilyInput minimumDistances;
    CudaFamilyInput maximumDistances;
    CudaFamilyInput radii;
    CudaFamilyInput positions;
    CudaFamilyInput directionsOrRotations;
    CudaFamilyInput rotations;
    CudaFamilyInput rowMaximumHits;
    CudaFamilyInput halfExtents;     // AABB/BOX overlap or box sweep rows' own half extents, optional
    uint32_t        queryFlags = 0u; // ovspatialquery_query_flag_t bits of the launch (ADR-0065)
    uintptr_t       stream     = 0u;
    uintptr_t       waitEvent  = 0u;
};

struct CudaFamilyResultColumns
{
    uint32_t                    resultSlot    = UINT32_MAX;
    uint64_t                    queryCount    = 0u;
    uint64_t                    hitCapacity   = 0u;
    uint32_t                    truth         = 0u;
    ovspatialquery_ordinal_t    sourceOrdinal = 0;
    DLDevice                    executionDevice{kDLCPU, 0};
    ovspatialquery_cuda_event_t resultReady{};
    DLTensor                    queryStatus{};
    DLTensor                    storedHitCount{};
    DLTensor                    hitOffsets{};
    DLTensor                    hitCounts{};
    DLTensor                    truncated{};
    DLTensor                    objectIds{};
    DLTensor                    positions{};
    DLTensor                    normals{};
    DLTensor                    distances{};
    DLTensor                    sourceFeatureKinds{};
    DLTensor                    sourceFaceIndices{};
    DLTensor                    sourceFaceCorners{};
    DLTensor                    sourceBarycentrics{};
    DLTensor                    geometryTypes{};
    DLTensor                    geometryValues{};
    DLTensor                    queryPositions{};
    DLTensor                    materialIds{};
};

/**
 * Measured admission of one launch. The published device bytes are the
 * generation's catalogue tables (zero when the plan's current catalogue already
 * describes the generation); each entry of publishedDefinitionBytes is a mesh
 * definition the launch must make resident, while definitions already resident
 * from an earlier generation are shared and cost nothing. The caller reserves
 * each part separately so a shared definition keeps its own charge for exactly
 * as long as any catalogue holds it.
 */
struct CudaFamilyStagingCharge
{
    uint64_t              hostBytes            = 0u;
    uint64_t              deviceBytes          = 0u;
    uint64_t              publishedDeviceBytes = 0u;
    std::vector<uint64_t> publishedDefinitionBytes;
    std::shared_ptr<void> staging;
};

enum class CudaFamilyCompletedResultOutcome : uint8_t
{
    success,
    unavailable,
    deviceLost,
    transferFailure,
    invariantFailure,
};

class CudaFamilyOperation
{
  public:
    virtual ~CudaFamilyOperation()                                        = default;
    virtual CudaFamilyResultColumns          reservation() const noexcept = 0;
    virtual CudaFamilyCompletedResultOutcome completedResult(CudaFamilyResultColumns& outResult) const noexcept = 0;
    virtual CudaCompletionState              resultState() const noexcept                                       = 0;
    virtual CudaCompletionState              consumerState() const noexcept                                     = 0;
    virtual bool                             consumerDoneOwnershipReleased() const noexcept                     = 0;
    virtual void                             releaseReadyDependencies() noexcept                                = 0;
    virtual ovspatialquery_api_status_t      transferConsumerDone(ovspatialquery_cuda_event_t event) noexcept   = 0;
    virtual void attachDeviceResultReservation(std::shared_ptr<void> reservation) noexcept                      = 0;
    virtual void attachStagingReservation(std::shared_ptr<void> reservation) noexcept                           = 0;
    virtual void attachHostStagingReservation(std::shared_ptr<HostLease> reservation) noexcept                  = 0;
    // Reservations in the order the measured charge listed them; attached
    // between prepare() and launch().
    virtual void attachPublishedReservations(std::shared_ptr<void>              catalogueReservation,
                                             std::vector<std::shared_ptr<void>> definitionReservations) noexcept = 0;

    // Blocks the calling thread until the launch's device work completes, or
    // returns at once when it cannot; either way resultState() decides.
    virtual void waitForResult() const noexcept = 0;
    // Rings `signal` once the consumer event transferConsumerDone() recorded
    // has completed. Best effort; a no-op before the transfer.
    virtual void ringWhenConsumed(const std::shared_ptr<DeviceSignal>& signal) const noexcept = 0;
};

class CudaFamilyPlan
{
  public:
    virtual ~CudaFamilyPlan()                                                                = default;
    virtual ovspatialquery_api_status_t measure(const CudaFamilyLaunch& launch, const GenerationLease& generation,
                                                CudaFamilyStagingCharge& outCharge) noexcept = 0;
    // Takes the launch, lease and staging that measure() accepted; returns INVALID_ARGUMENT without
    // staging and does not validate the launch or lease again.
    virtual ovspatialquery_api_status_t prepare(CudaFamilyLaunch launch, GenerationLease generation,
                                                std::shared_ptr<void>                 staging,
                                                std::shared_ptr<CudaFamilyOperation>& outPrepared) noexcept = 0;
    virtual ovspatialquery_api_status_t launch(uint32_t resultSlot, std::shared_ptr<CudaFamilyOperation> prepared,
                                               std::shared_ptr<CudaFamilyOperation>& outOperation) noexcept = 0;
    virtual uint64_t                    deviceByteCharge() const noexcept                                   = 0;
    // Whether this plan evaluates the generation's host objects rather than
    // borrowing a device-prepared version's tables: exact plans do, so the
    // caller materializes a readable version's host copy before `measure`.
    virtual bool needsHostObjects(const GenerationLease& generation) const noexcept = 0;
};

// The published columns of one result slot, in the order a HOST_MAPPED plan
// lays them out in mapped host memory (ADR-0077).
enum class CudaFamilyMappedColumn : uint32_t
{
    queryStatus,
    storedHitCount,
    hitOffsets,
    hitCounts,
    truncated,
    objectIds,
    geometryTypes,
    geometryValues,
    materialIds,
    positions,
    queryPositions,
    normals,
    distances,
    featureKinds,
    faceIndices,
    corners,
    barycentrics,
    count,
};

struct CudaFamilyMappedLayout
{
    static constexpr size_t kCount = static_cast<size_t>(CudaFamilyMappedColumn::count);
    // A column's byte offset in the slot's block, its bytes per row (zero when
    // the plan does not publish it) and how many rows it copies.
    uint64_t offsets[kCount]  = {};
    uint32_t rowBytes[kCount] = {};
    uint32_t rows[kCount]     = {};
    // The slot's block, a multiple of cuda_family::kMappedColumnAlignment.
    uint64_t bytes = 0u;
};

// The mapped block of one result slot for these sizes and result fields; false
// when it does not fit in 64 bits.
bool layoutCudaFamilyMappedColumns(uint64_t maximumQueryCount, uint64_t hitCapacityPerSlot, uint32_t resultFields,
                                   CudaFamilyMappedLayout& outLayout) noexcept;

// INVALID_ARGUMENT for a malformed configuration; DEVICE_BUDGET when the
// device byte charge does not fit in 64 bits, which exceeds any budget.
ovspatialquery_api_status_t validateCudaFamilyPlanConfig(const CudaFamilyPlanConfig& config,
                                                         uint64_t&                   outDeviceByteCharge) noexcept;
bool estimateCudaFamilyPlanFixedHostByteCharge(uint32_t resultSlotCount, uint64_t& outHostByteCharge) noexcept;
bool validateCudaFamilyLaunch(const CudaFamilyPlanConfig& config, const CudaFamilyLaunch& launch) noexcept;
ovspatialquery_api_status_t createCudaFamilyPlan(const CudaFamilyPlanConfig&      config,
                                                 std::shared_ptr<CudaFamilyPlan>& outPlan) noexcept;

} // namespace detail
} // namespace ovspatialquery

#endif
