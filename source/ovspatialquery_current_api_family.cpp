/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_float_bounds.h"
#include "ovspatialquery_instance_state.h"

#include "ovspatialquery_instance_operation_runtime.h"
#include "ovspatialquery_target_resolver.h"

#include <ovspatialquery_query_evaluation.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>

using namespace ovspatialquery::detail;

namespace
{

uint64_t apiFamilyRowSize(ApiFamily family) noexcept
{
    if (family == ApiFamily::sphereSweep || family == ApiFamily::boxSweep)
    {
        return sizeof(ovspatialquery::detail::SphereSweepBatchRow);
    }
    if (family == ApiFamily::overlap)
    {
        return sizeof(ovspatialquery::detail::OverlapBatchRow);
    }
    return sizeof(ovspatialquery::detail::RaycastBatchRow);
}

bool apiAddFamilyPlanStorageCharge(ApiFamily family, uint64_t maximumQueryCount, uint32_t resultSlotCount,
                                   uint64_t& total) noexcept
{
    return addHostCharge(1u, sizeof(ApiFamilyPlanState) + 3u * kApiSharedControlAllowance + kApiDirectoryNodeAllowance,
                         total) &&
           addHostCharge(maximumQueryCount, apiFamilyRowSize(family), total) &&
           addHostCharge(resultSlotCount,
                         sizeof(std::shared_ptr<ApiFamilyProjection>) + kApiSharedControlAllowance +
                             sizeof(ApiFamilyProjection),
                         total);
}

// Every family accepts HOST_MAPPED: a CUDA plan places its columns by it and a
// CPU plan ignores it (ADR-0077), so neither backend sees it as a column.
constexpr uint32_t kApiRaycastResultFieldMask =
    ovspatialquery::detail::kBatchResultPosition | ovspatialquery::detail::kBatchResultNormal |
    ovspatialquery::detail::kBatchResultDistance | ovspatialquery::detail::kBatchResultSourceFeature |
    ovspatialquery::detail::kBatchResultGeometryIdentity | ovspatialquery::detail::kBatchResultMaterial |
    ovspatialquery::detail::kBatchResultHostMapped;
constexpr uint32_t kApiSphereSweepResultFieldMask =
    kApiRaycastResultFieldMask | ovspatialquery::detail::kBatchResultQueryPosition;
constexpr uint32_t kApiOverlapResultFieldMask = ovspatialquery::detail::kBatchResultGeometryIdentity |
                                                ovspatialquery::detail::kBatchResultMaterial |
                                                ovspatialquery::detail::kBatchResultHostMapped;
constexpr uint32_t kApiColumnFields = ~ovspatialquery::detail::kBatchResultHostMapped;

// CUDA plans take the batch result-field bits unchanged.
static_assert(
    ovspatialquery::detail::kCudaFamilyResultPosition == ovspatialquery::detail::kBatchResultPosition &&
    ovspatialquery::detail::kCudaFamilyResultNormal == ovspatialquery::detail::kBatchResultNormal &&
    ovspatialquery::detail::kCudaFamilyResultDistance == ovspatialquery::detail::kBatchResultDistance &&
    ovspatialquery::detail::kCudaFamilyResultSourceFeature == ovspatialquery::detail::kBatchResultSourceFeature &&
    ovspatialquery::detail::kCudaFamilyResultGeometryIdentity == ovspatialquery::detail::kBatchResultGeometryIdentity &&
    ovspatialquery::detail::kCudaFamilyResultQueryPosition == ovspatialquery::detail::kBatchResultQueryPosition &&
    ovspatialquery::detail::kCudaFamilyResultMaterial == ovspatialquery::detail::kBatchResultMaterial);

template <typename Descriptor>
ovspatialquery_api_status_t apiCreateCudaFamilyPlan(const InstanceReference&         instance,
                                                    ovspatialquery_instance_handle_t instanceHandle,
                                                    std::shared_ptr<void> viewRetention, ApiFamily family,
                                                    const ovspatialquery::detail::CudaFamilyPlanConfig& cudaConfig,
                                                    Descriptor descriptor, ovspatialquery_batch_plan_handle_t* outPlan)
{
    uint64_t                          deviceCharge = 0u;
    const ovspatialquery_api_status_t validated =
        ovspatialquery::detail::validateCudaFamilyPlanConfig(cudaConfig, deviceCharge);
    if (validated != OVSPATIALQUERY_API_SUCCESS)
    {
        return validated;
    }
    std::shared_ptr<void>             deviceLease;
    const ovspatialquery_api_status_t reserved = ovspatialquery::detail::reserveInstanceDeviceMemory(
        instanceHandle, cudaConfig.device, deviceCharge, deviceLease);
    if (reserved != OVSPATIALQUERY_API_SUCCESS)
    {
        return reserved;
    }
    try
    {
        descriptor.common.backend           = ovspatialquery::detail::FamilyBatchPlanBackend::cuda;
        descriptor.common.resultSlotCount   = cudaConfig.resultSlotCount;
        descriptor.common.hostLedger        = &instance->hostLedger;
        descriptor.common.cudaDevice        = cudaConfig.device;
        descriptor.common.cudaResultFields  = cudaConfig.resultFields;
        descriptor.common.cudaHostMapped    = cudaConfig.hostMapped;
        descriptor.plan.workers             = instance->sharedWorkerLoan();
        descriptor.common.cudaMeshResidency = &instance->cudaMeshResidency;
        descriptor.common.cudaAllowSinglePrecision =
            cudaConfig.precision == ovspatialquery::detail::CudaFamilyPrecision::exactSingle;
        descriptor.common.cudaOverlapKind = static_cast<uint32_t>(cudaConfig.overlapKind);
        descriptor.common.cudaPlanRadius  = cudaConfig.planRadius;
        for (size_t axis = 0u; axis < 3u; ++axis)
        {
            descriptor.common.cudaPlanHalfExtents[axis] = cudaConfig.planHalfExtents[axis];
        }
        uint64_t hostCharge = 0u;
        if (!ovspatialquery::detail::FamilyBatchPlanOperationOwner::estimateResidentBytes(descriptor, hostCharge) ||
            !apiAddFamilyPlanStorageCharge(family, 0u, 0u, hostCharge))
        {
            return OVSPATIALQUERY_API_HOST_BUDGET;
        }
        std::shared_ptr<ovspatialquery::detail::HostLease> hostLease;
        const ovspatialquery_api_status_t leaseStatus = reserveSharedHostLease(instance, hostCharge, hostLease);
        if (leaseStatus != OVSPATIALQUERY_API_SUCCESS)
        {
            return leaseStatus;
        }
        std::unique_ptr<ovspatialquery::detail::FamilyBatchPlanOperationOwner> owner;
        const ovspatialquery_api_status_t create = ovspatialquery::detail::FamilyBatchPlanOperationOwner::create(
            instance->populations, instanceHandle, descriptor, owner);
        if (create != OVSPATIALQUERY_API_SUCCESS)
        {
            return create;
        }
        const std::shared_ptr<ApiFamilyPlanState> plan = std::make_shared<ApiFamilyPlanState>();
        plan->family                                   = family;
        plan->cudaPlan                                 = true;
        plan->cudaDeviceId                             = cudaConfig.device.device_id;
        plan->maximumQueryCount                        = cudaConfig.maximumQueryCount;
        plan->viewRetention                            = std::move(viewRetention);
        plan->owner                                    = std::move(owner);
        plan->cudaTargetRetention                      = std::move(deviceLease);
        plan->hostLease                                = std::move(hostLease);
        return apiPublishAdapter(*instance, plan, outPlan->raw_id);
    }
    catch (const std::bad_alloc&)
    {
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    catch (...)
    {
        return OVSPATIALQUERY_API_INTERNAL;
    }
}

void apiPrepareFamilyPlanStorage(ApiFamilyPlanState& plan, uint32_t resultSlotCount)
{
    if (plan.family == ApiFamily::raycast)
    {
        plan.raycastScratch.rows.reserve(static_cast<size_t>(plan.maximumQueryCount));
    }
    else if (plan.family == ApiFamily::sphereSweep || plan.family == ApiFamily::boxSweep)
    {
        plan.sweepScratch.rows.reserve(static_cast<size_t>(plan.maximumQueryCount));
    }
    else
    {
        plan.overlapScratch.rows.reserve(static_cast<size_t>(plan.maximumQueryCount));
    }
    plan.projections.reserve(resultSlotCount);
    for (uint32_t index = 0u; index < resultSlotCount; ++index)
    {
        const auto projection = std::make_shared<ApiFamilyProjection>();
        if (plan.family == ApiFamily::raycast)
        {
            projection->payload.emplace<ovspatialquery_raycast_batch_result_t>();
        }
        else if (plan.family == ApiFamily::sphereSweep || plan.family == ApiFamily::boxSweep)
        {
            projection->payload.emplace<ovspatialquery_sphere_sweep_batch_result_t>();
        }
        else
        {
            projection->payload.emplace<ovspatialquery_overlap_batch_result_t>();
        }
        plan.projections.push_back(projection);
    }
}

bool apiCheckedByteOffset(const void* data, uint64_t byteOffset, uint64_t element, size_t elementSize,
                          size_t& outOffset) noexcept
{
    const uint64_t sizeMaximum = static_cast<uint64_t>(std::numeric_limits<size_t>::max());
    if (byteOffset > sizeMaximum || (element != 0u && elementSize > sizeMaximum / element))
    {
        return false;
    }
    const uint64_t elementOffset = element * elementSize;
    if (elementOffset > sizeMaximum - byteOffset)
    {
        return false;
    }
    const uint64_t  offset         = byteOffset + elementOffset;
    const uintptr_t base           = reinterpret_cast<uintptr_t>(data);
    const uintptr_t addressMaximum = std::numeric_limits<uintptr_t>::max();
    if (offset > addressMaximum - base || elementSize > addressMaximum - base - offset)
    {
        return false;
    }
    outOffset = static_cast<size_t>(offset);
    return true;
}

template <typename T>
bool apiReadScalar(const void* sourceBytes, uint64_t index, uint8_t code, uint8_t bits, bool optional, T defaultValue,
                   T& out)
{
    if (sourceBytes == nullptr)
    {
        out = defaultValue;
        return optional;
    }
    ovspatialquery_input_array_t sourceStorage{};
    std::memcpy(&sourceStorage, sourceBytes, sizeof(sourceStorage));
    const ovspatialquery_input_array_t* source = &sourceStorage;
    const DLTensor&                     tensor = source->tensor;
    if (source->ownership > 1u || tensor.device.device_type != kDLCPU || tensor.device.device_id != 0 ||
        tensor.data == nullptr || tensor.dtype.code != code || tensor.dtype.bits != bits || tensor.dtype.lanes != 1u ||
        tensor.ndim != 1 || tensor.shape == nullptr || tensor.shape[0] < 0 ||
        index >= static_cast<uint64_t>(tensor.shape[0]) || tensor.byte_offset > std::numeric_limits<size_t>::max())
    {
        return false;
    }
    const int64_t stride = tensor.strides == nullptr ? 1 : tensor.strides[0];
    if (stride < 0)
    {
        return false;
    }
    const uint64_t unsignedStride = static_cast<uint64_t>(stride);
    if (index != 0u && unsignedStride > UINT64_MAX / index)
    {
        return false;
    }
    size_t offset = 0u;
    if (!apiCheckedByteOffset(tensor.data, tensor.byte_offset, index * unsignedStride, sizeof(T), offset))
    {
        return false;
    }
    const uint8_t* bytes = static_cast<const uint8_t*>(tensor.data) + offset;
    std::memcpy(&out, bytes, sizeof(T));
    return true;
}

// A launch's options (ADR-0065); overlap batches have none.
template <typename Batch>
uint32_t apiBatchQueryFlags(const Batch& batch)
{
    return batch.query_flags;
}

uint32_t apiBatchQueryFlags(const ovspatialquery_overlap_batch_t&) { return 0u; }

template <typename T, size_t N>
bool apiReadVector(const void* sourceBytes, uint64_t index, uint8_t code, uint8_t bits, std::array<T, N>& out)
{
    if (sourceBytes == nullptr)
    {
        return false;
    }
    ovspatialquery_input_array_t sourceStorage{};
    std::memcpy(&sourceStorage, sourceBytes, sizeof(sourceStorage));
    const ovspatialquery_input_array_t* source = &sourceStorage;
    const DLTensor&                     tensor = source->tensor;
    if (source->ownership > 1u || tensor.device.device_type != kDLCPU || tensor.device.device_id != 0 ||
        tensor.data == nullptr || tensor.dtype.code != code || tensor.dtype.bits != bits || tensor.dtype.lanes != 1u ||
        tensor.ndim != 2 || tensor.shape == nullptr || tensor.shape[0] < 0 ||
        tensor.shape[1] != static_cast<int64_t>(N) || index >= static_cast<uint64_t>(tensor.shape[0]))
    {
        return false;
    }
    const int64_t rowStride    = tensor.strides == nullptr ? static_cast<int64_t>(N) : tensor.strides[0];
    const int64_t columnStride = tensor.strides == nullptr ? 1 : tensor.strides[1];
    if (rowStride < 0 || columnStride < 0)
    {
        return false;
    }
    const uint64_t unsignedRowStride    = static_cast<uint64_t>(rowStride);
    const uint64_t unsignedColumnStride = static_cast<uint64_t>(columnStride);
    if ((index != 0u && unsignedRowStride > UINT64_MAX / index) ||
        tensor.byte_offset > std::numeric_limits<size_t>::max())
    {
        return false;
    }
    const uint64_t rowElement = index * unsignedRowStride;
    for (size_t column = 0u; column < N; ++column)
    {
        if (column != 0u && unsignedColumnStride > UINT64_MAX / column)
        {
            return false;
        }
        const uint64_t columnElement = column * unsignedColumnStride;
        if (columnElement > UINT64_MAX - rowElement)
        {
            return false;
        }
        size_t offset = 0u;
        if (!apiCheckedByteOffset(tensor.data, tensor.byte_offset, rowElement + columnElement, sizeof(T), offset))
        {
            return false;
        }
        std::memcpy(&out[column], static_cast<const uint8_t*>(tensor.data) + offset, sizeof(T));
    }
    return true;
}

// One CPU input column of a launch, checked once instead of once per row. Every check
// apiReadScalar and apiReadVector make depends on the row only through offsets that grow
// with it, so they accept every row below `rows` exactly when they accept the last one.
// A row read is then one offset and one copy per component. An absent optional column
// reads `fallback`; zero rows read nothing, so nothing is checked, as before.
template <typename T, size_t N>
class ApiInputColumn
{
  public:
    // A vector column is T[rows, N]; a scalar column (Scalar) is T[rows].
    template <bool Scalar = false>
    bool open(const void* sourceBytes, uint64_t rows, uint8_t code, uint8_t bits, bool optional,
              const std::array<T, N>& fallback = {})
    {
        mFallback = fallback;
        mData     = nullptr;
        if (rows == 0u)
        {
            return true;
        }
        if (sourceBytes == nullptr)
        {
            return optional;
        }
        std::array<T, N> last{};
        if constexpr (Scalar)
        {
            static_assert(N == 1u, "a scalar column has one component");
            if (!apiReadScalar(sourceBytes, rows - 1u, code, bits, optional, T{}, last[0]))
            {
                return false;
            }
        }
        else if (!apiReadVector(sourceBytes, rows - 1u, code, bits, last))
        {
            return false;
        }
        ovspatialquery_input_array_t source{};
        std::memcpy(&source, sourceBytes, sizeof(source));
        const DLTensor& tensor = source.tensor;
        mData                  = static_cast<const uint8_t*>(tensor.data) + static_cast<size_t>(tensor.byte_offset);
        mRow    = sizeof(T) * static_cast<size_t>(tensor.strides == nullptr ? static_cast<int64_t>(Scalar ? 1u : N)
                                                                            : tensor.strides[0]);
        mColumn = sizeof(T) * static_cast<size_t>(tensor.strides == nullptr || Scalar ? 1 : tensor.strides[1]);
        return true;
    }

    void read(uint64_t row, std::array<T, N>& out) const
    {
        if (mData == nullptr)
        {
            out = mFallback;
            return;
        }
        const uint8_t* bytes = mData + static_cast<size_t>(row) * mRow;
        for (size_t component = 0u; component < N; ++component)
        {
            std::memcpy(&out[component], bytes + component * mColumn, sizeof(T));
        }
    }

    T scalar(uint64_t row) const
    {
        std::array<T, N> value{};
        read(row, value);
        return value[0];
    }

  private:
    const uint8_t*   mData   = nullptr;
    size_t           mRow    = 0u;
    size_t           mColumn = 0u;
    std::array<T, N> mFallback{};
};

using ApiScalarColumn    = ApiInputColumn<double, 1u>;
using ApiCountColumn     = ApiInputColumn<uint64_t, 1u>;
using ApiQueryBitsColumn = ApiInputColumn<uint64_t, 2u>;

// The optional uint64[Q,2] participation column; a null column accepts every object.
bool apiOpenQueryBits(const void* sourceBytes, uint64_t rows, ApiQueryBitsColumn& out)
{
    return out.open(sourceBytes, rows, kDLUInt, 64u, true);
}

ovspatialquery::detail::ParticipationBits apiQueryBits(const ApiQueryBitsColumn& column, uint64_t row)
{
    std::array<uint64_t, 2u> words{};
    column.read(row, words);
    ovspatialquery::detail::ParticipationBits bits{};
    bits.words[0] = words[0];
    bits.words[1] = words[1];
    return bits;
}

template <typename ProjectionPayload, typename Reservation>
std::shared_ptr<ApiFamilyProjection> apiActivateFamilyProjection(const ApiFamilyPlanState& plan,
                                                                 const Reservation&        reservation)
{
    if (reservation.common.result_slot >= plan.projections.size())
    {
        return {};
    }
    const std::shared_ptr<ApiFamilyProjection> projection = plan.projections[reservation.common.result_slot];
    if (projection == nullptr)
    {
        return {};
    }
    std::lock_guard<std::mutex> lock(projection->mutex);
    ProjectionPayload* const    payload = std::get_if<ProjectionPayload>(&projection->payload);
    if (payload == nullptr)
    {
        return {};
    }
    *payload          = {};
    projection->ready = false;
    return projection;
}

template <typename Output>
void apiCudaBaseColumns(const ovspatialquery::detail::CudaFamilyResultColumns& columns, Output& out)
{
    out                                     = {};
    out.common.result_slot                  = columns.resultSlot;
    out.common.query_count                  = columns.queryCount;
    out.common.hit_capacity                 = columns.hitCapacity;
    out.common.truth                        = static_cast<ovspatialquery_hit_truth_t>(columns.truth);
    out.common.source_ordinal               = columns.sourceOrdinal;
    out.common.execution_device             = columns.executionDevice;
    out.common.result_ready                 = columns.resultReady;
    out.columns.hit_ranges.query_status     = columns.queryStatus;
    out.columns.hit_ranges.stored_hit_count = columns.storedHitCount;
    out.columns.hit_ranges.hit_offsets      = columns.hitOffsets;
    out.columns.hit_ranges.hit_counts       = columns.hitCounts;
    out.columns.hit_ranges.truncated        = columns.truncated;
    out.columns.object_ids                  = columns.objectIds;
    out.columns.geometry.types              = columns.geometryTypes;
    out.columns.geometry.values             = columns.geometryValues;
    out.columns.material_ids                = columns.materialIds;
}

template <typename Output>
void apiCudaRaycastColumns(const ovspatialquery::detail::CudaFamilyResultColumns& columns, Output& out)
{
    apiCudaBaseColumns(columns, out);
    if (columns.truth == OVSPATIALQUERY_HIT_BOUNDS_CANDIDATE)
    {
        out.columns.contact.candidate.bounds_positions = columns.positions;
        out.columns.contact.candidate.bounds_normals   = columns.normals;
        out.columns.contact.candidate.distances        = columns.distances;
    }
    else
    {
        out.columns.contact.exact.positions                      = columns.positions;
        out.columns.contact.exact.normals                        = columns.normals;
        out.columns.contact.exact.distances                      = columns.distances;
        out.columns.contact.exact.features.kinds                 = columns.sourceFeatureKinds;
        out.columns.contact.exact.features.source_face_indices   = columns.sourceFaceIndices;
        out.columns.contact.exact.features.source_corner_indices = columns.sourceFaceCorners;
        out.columns.contact.exact.features.barycentrics          = columns.sourceBarycentrics;
    }
}

// The raycast columns plus each arm's query positions.
template <typename Output>
void apiCudaSphereSweepColumns(const ovspatialquery::detail::CudaFamilyResultColumns& columns, Output& out)
{
    apiCudaRaycastColumns(columns, out);
    if (columns.truth == OVSPATIALQUERY_HIT_BOUNDS_CANDIDATE)
    {
        out.columns.contact.candidate.query_positions = columns.queryPositions;
    }
    else
    {
        out.columns.contact.exact.query_positions = columns.queryPositions;
    }
}

void apiCudaInput(const void* opaque, const char* column, ovspatialquery::detail::CudaFamilyInput& out) noexcept
{
    out = ovspatialquery::detail::CudaFamilyInput{};
    if (opaque == nullptr)
    {
        return;
    }
    const ovspatialquery_input_array_t* const input = static_cast<const ovspatialquery_input_array_t*>(opaque);
    out.tensor                                      = input->tensor;
    out.ownership                                   = input->ownership;
    out.present                                     = true;
    ovspatialquery::detail::readCudaFamilyInputLayout(out, column);
}

// Completes a CPU plan creation once the caller has filled the owner
// configuration both devices share: the CPU-only configuration fields, the
// storage charge and host lease, the owner, and the published plan handle are
// identical for all families. The row defaults are those ApiFamilyPlanState
// documents.
template <ApiFamily Family, typename Desc, typename Config>
ovspatialquery_result_t apiCreateCpuFamilyPlan(const InstanceReference&         instance,
                                               ovspatialquery_instance_handle_t instanceHandle, const Desc& desc,
                                               Config& config, std::shared_ptr<void> viewRetention,
                                               uint32_t overlapShapeKind, double radius,
                                               const std::array<double, 3u>&       halfExtents,
                                               ovspatialquery_batch_plan_handle_t* outPlan)
{
    config.common.resultSlotCount = desc.result_slot_count;
    config.common.resultFields    = desc.result_fields & kApiColumnFields;
    config.common.hostLedger      = &instance->hostLedger;
    config.plan.hostLedger        = &instance->hostLedger;
    config.plan.workers           = instance->sharedWorkerLoan();
    uint64_t fixedHostCharge      = 0u;
    // A charge that does not fit in 64 bits exceeds any host budget.
    if (!ovspatialquery::detail::FamilyBatchPlanOperationOwner::estimateResidentBytes(config, fixedHostCharge) ||
        !apiAddFamilyPlanStorageCharge(Family, desc.maximum_query_count, desc.result_slot_count, fixedHostCharge))
    {
        return apiResult(OVSPATIALQUERY_API_HOST_BUDGET);
    }
    std::shared_ptr<ovspatialquery::detail::HostLease> hostLease;
    const ovspatialquery_api_status_t hostLeaseStatus = reserveSharedHostLease(instance, fixedHostCharge, hostLease);
    if (hostLeaseStatus != OVSPATIALQUERY_API_SUCCESS)
    {
        return apiResult(hostLeaseStatus);
    }
    std::unique_ptr<ovspatialquery::detail::FamilyBatchPlanOperationOwner> owner;
    const ovspatialquery_api_status_t status = ovspatialquery::detail::FamilyBatchPlanOperationOwner::create(
        instance->populations, instanceHandle, config, owner);
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        return apiResult(status);
    }
    try
    {
        const auto plan         = std::make_shared<ApiFamilyPlanState>();
        plan->family            = Family;
        plan->overlapShapeKind  = overlapShapeKind;
        plan->radius            = radius;
        plan->halfExtents       = halfExtents;
        plan->maximumQueryCount = desc.maximum_query_count;
        plan->viewRetention     = std::move(viewRetention);
        plan->owner             = std::move(owner);
        apiPrepareFamilyPlanStorage(*plan, desc.result_slot_count);
        plan->hostLease = std::move(hostLease);
        return apiResult(apiPublishAdapter(*instance, plan, outPlan->raw_id));
    }
    catch (const std::bad_alloc&)
    {
        return apiResult(OVSPATIALQUERY_API_OUT_OF_MEMORY);
    }
    catch (...)
    {
        return apiResult(OVSPATIALQUERY_API_INTERNAL);
    }
}

// Resolves the instance and the plan a public enqueue entry point names.
// Returns OVSPATIALQUERY_API_SUCCESS with both outputs set, or the status the
// export must report alongside OVSPATIALQUERY_INVALID_OP.
ovspatialquery_api_status_t apiResolveFamilyPlan(ovspatialquery_instance_handle_t   instanceHandle,
                                                 ovspatialquery_batch_plan_handle_t planHandle, ApiFamily family,
                                                 const void* batch, const void* outReservation,
                                                 InstanceReference&                   outInstance,
                                                 std::shared_ptr<ApiFamilyPlanState>& outPlan)
{
    if (batch == nullptr || outReservation == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    outInstance = instanceState(instanceHandle);
    if (outInstance == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    outPlan = apiFindAdapter<ApiFamilyPlanState>(*outInstance, planHandle.raw_id);
    if (outPlan == nullptr || outPlan->family != family)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    return OVSPATIALQUERY_API_SUCCESS;
}

// Serves one public enqueue entry point. A CUDA plan takes the device launch
// fillCuda completes and projectCuda reserves; a CPU plan reads its rows with
// readRows into the plan scratch under the adapter lock, and readRows returns
// false to reject the batch. Statuses keep this order: plan resolution, a
// reserved input column, the CUDA stream and then whether it is capturing, the CPU stream, event and query count, a
// released plan, the ordinal, the family's column checks and row reads, then admission.
template <ApiFamily Family, typename ProjectionPayload, typename Batch, typename Reservation, typename Launch,
          typename FillCuda, typename ProjectCuda, typename ReadRows>
ovspatialquery_enqueue_result_t apiAdmitFamily(ovspatialquery_instance_handle_t   instanceHandle,
                                               ovspatialquery_batch_plan_handle_t planHandle, const Batch* batch,
                                               Reservation* outReservation, Launch ApiFamilyPlanState::*scratch,
                                               FillCuda fillCuda, ProjectCuda projectCuda, ReadRows readRows)
{
    ovspatialquery_enqueue_result_t rejected{OVSPATIALQUERY_API_INVALID_ARGUMENT, OVSPATIALQUERY_INVALID_OP};
    ovspatialquery::detail::clearInputLayoutDiagnostic();
    InstanceReference                   instance;
    std::shared_ptr<ApiFamilyPlanState> plan;
    const ovspatialquery_api_status_t   resolved =
        apiResolveFamilyPlan(instanceHandle, planHandle, Family, batch, outReservation, instance, plan);
    if (resolved != OVSPATIALQUERY_API_SUCCESS)
    {
        rejected.status = resolved;
        return rejected;
    }
    // A reserved input column or an unknown option belongs to a later version (ADR-0062, ADR-0065).
    if (batch->reserved[0] != nullptr || batch->reserved[1] != nullptr ||
        (apiBatchQueryFlags(*batch) & ~static_cast<uint32_t>(OVSPATIALQUERY_QUERY_MESH_BOTH_SIDES)) != 0u)
    {
        rejected.status = OVSPATIALQUERY_API_NOT_SUPPORTED;
        return rejected;
    }
    // A CUDA plan launches on the caller's stream, which may be stream 0 (ADR-0060).
    if (plan->cudaPlan ? batch->cuda_sync.has_stream != 1u
                       : (batch->cuda_sync.has_stream != 0u || batch->cuda_sync.wait_event != 0u ||
                          batch->query_count > plan->maximumQueryCount))
    {
        return rejected;
    }
    // Nothing is queued on a stream that is capturing a CUDA graph (ADR-0060).
    if (plan->cudaPlan && ovspatialquery::detail::cudaStreamIsCapturing(plan->cudaDeviceId, batch->cuda_sync.stream))
    {
        rejected.status = OVSPATIALQUERY_API_NOT_SUPPORTED;
        return rejected;
    }
    try
    {
        std::lock_guard<std::mutex> adapterLock(plan->adapterMutex);
        if (plan->owner == nullptr)
        {
            rejected.status = OVSPATIALQUERY_API_INVALID_HANDLE;
            return rejected;
        }
        bool                                          selectionValid = false;
        const ovspatialquery::detail::TargetSelection selection      = apiSelection(batch->ordinal, selectionValid);
        if (!selectionValid)
        {
            return rejected;
        }
        Reservation                                      reservation{};
        ovspatialquery::detail::CudaFamilyResultColumns  columns{};
        ovspatialquery::detail::FamilyBatchPlanAdmission admission{};
        if (plan->cudaPlan)
        {
            ovspatialquery::detail::CudaFamilyLaunch launch{};
            launch.queryCount = batch->query_count;
            apiCudaInput(batch->query_bits, "query_bits", launch.queryBits);
            fillCuda(launch);
            launch.stream    = batch->cuda_sync.stream;
            launch.waitEvent = batch->cuda_sync.wait_event;
            admission        = plan->owner->enqueueCuda(selection, launch, columns);
        }
        else
        {
            // CPU rows read any nonnegative strides in place; the same layout
            // reads as a CUDA launch name a refused column (ADR-0053).
            ovspatialquery::detail::CudaFamilyLaunch probe{};
            apiCudaInput(batch->query_bits, "query_bits", probe.queryBits);
            fillCuda(probe);
            if (ovspatialquery::detail::inputLayoutDiagnostic(OVSPATIALQUERY_API_INVALID_ARGUMENT) != nullptr)
            {
                return rejected;
            }
            Launch& launch = (*plan).*scratch;
            launch.rows.resize(static_cast<size_t>(batch->query_count));
            if (!readRows(static_cast<const ApiFamilyPlanState&>(*plan), launch))
            {
                return rejected;
            }
            admission = plan->owner->enqueueColumns(selection, std::move(launch), reservation);
        }
        rejected.status = admission.outcome;
        if (rejected.status != OVSPATIALQUERY_API_SUCCESS)
        {
            return rejected;
        }
        // A CUDA plan's results stay on its owner; only CPU plans keep a projection.
        ApiOperationOwner owner{};
        owner.plan = plan;
        if (plan->cudaPlan)
        {
            projectCuda(columns, reservation);
        }
        else
        {
            owner.projection = apiActivateFamilyProjection<ProjectionPayload>(*plan, reservation);
            if (owner.projection == nullptr)
            {
                rejected.status = OVSPATIALQUERY_API_INTERNAL;
                return rejected;
            }
        }
        apiSetOperationOwner(*instance, admission.operation.value, std::move(owner));
        *outReservation = reservation;
        return {OVSPATIALQUERY_API_SUCCESS, admission.operation.value};
    }
    catch (const std::bad_alloc&)
    {
        rejected.status = OVSPATIALQUERY_API_OUT_OF_MEMORY;
        return rejected;
    }
    catch (...)
    {
        rejected.status = OVSPATIALQUERY_API_INTERNAL;
        return rejected;
    }
}

// Every return of a family enqueue entry point sets the calling thread's diagnostic.
template <ApiFamily Family, typename ProjectionPayload, typename Batch, typename Reservation, typename Launch,
          typename FillCuda, typename ProjectCuda, typename ReadRows>
ovspatialquery_enqueue_result_t apiEnqueueFamily(ovspatialquery_instance_handle_t   instanceHandle,
                                                 ovspatialquery_batch_plan_handle_t planHandle, const Batch* batch,
                                                 Reservation* outReservation, Launch ApiFamilyPlanState::*scratch,
                                                 FillCuda fillCuda, ProjectCuda projectCuda, ReadRows readRows)
{
    const ovspatialquery_enqueue_result_t result = apiAdmitFamily<Family, ProjectionPayload>(
        instanceHandle, planHandle, batch, outReservation, scratch, fillCuda, projectCuda, readRows);
    return sourcePublicEnqueueResult(result.status, result.op_id, nullptr);
}

// Serves one public fetch entry point: CUDA plans hand their device columns to
// project(), CPU plans materialize the projection payload once and copy it.
template <ApiFamily Family, typename Result, typename ProjectCuda>
ovspatialquery_result_t apiFetchFamily(ovspatialquery_instance_handle_t instanceHandle,
                                       ovspatialquery_op_id_t operationId, Result* outResult, ProjectCuda project)
{
    if (outResult == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_HANDLE);
    }
    ApiOperationOwner owner = apiOperationOwner(*instance, operationId);
    if (owner.plan == nullptr || owner.plan->family != Family)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_HANDLE);
    }
    Result result{};
    if (owner.plan->cudaPlan)
    {
        ovspatialquery::detail::CudaFamilyResultColumns completed{};
        std::lock_guard<std::mutex>                     adapterLock(owner.plan->adapterMutex);
        if (owner.plan->owner == nullptr)
        {
            return apiResult(OVSPATIALQUERY_API_INVALID_HANDLE);
        }
        const ovspatialquery_api_status_t status = owner.plan->owner->fetchCuda({operationId}, completed);
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return apiResult(status);
        }
        project(completed, result);
        *outResult = result;
        return apiResult(OVSPATIALQUERY_API_SUCCESS);
    }
    {
        std::lock_guard<std::mutex>       adapterLock(owner.plan->adapterMutex);
        const ovspatialquery_api_status_t status = apiMaterializeFamilyLocked(owner, operationId);
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return apiResult(status);
        }
        const std::shared_ptr<ApiFamilyProjection> projection =
            std::static_pointer_cast<ApiFamilyProjection>(owner.projection);
        std::lock_guard<std::mutex> projectionLock(projection->mutex);
#if defined(OVSPATIALQUERY_ENABLE_INSTANCE_OPERATION_RUNTIME_TEST_HOOK)
        apiFetchProjectionPausePointForTest(0u);
#endif
        const Result* const payload = std::get_if<Result>(&projection->payload);
        if (payload == nullptr)
        {
            return apiResult(OVSPATIALQUERY_API_INTERNAL);
        }
        result = *payload;
    }
    *outResult = result;
    return apiResult(OVSPATIALQUERY_API_SUCCESS);
}

// Serves one public plan constructor once its descriptor is checked: resolves
// the device, the instance and the target under the lifecycle lock, fills the
// configuration every family shares, then creates the CUDA plan, whose
// family-specific fields cudaFamily sets, or the CPU plan.
template <ApiFamily Family, typename Desc, typename Config, typename CudaFamily>
ovspatialquery_result_t apiCreateFamilyPlan(ovspatialquery_instance_handle_t instanceHandle, const Desc& desc,
                                            Config& config, ovspatialquery::detail::CudaFamilyKind cudaKind,
                                            CudaFamily cudaFamily, uint32_t overlapShapeKind, double radius,
                                            const std::array<double, 3u>&       halfExtents,
                                            ovspatialquery_batch_plan_handle_t* outPlan)
{
    if (desc.device.device_type == kDLCUDA ? desc.device.device_id < 0
                                           : (desc.device.device_type != kDLCPU || desc.device.device_id != 0))
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_HANDLE);
    }
    std::lock_guard<std::mutex> lifecycle(instance->lifecycleMutex);
    std::shared_ptr<void>       viewRetention;
    ovspatialquery_api_status_t status = OVSPATIALQUERY_API_SUCCESS;
    config.common.target               = apiPlanTarget(instance, desc.target, false, viewRetention, status);
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        return apiResult(status);
    }
    config.plan.precision         = desc.precision == 0u ? ovspatialquery::detail::QueryPrecision::boundsCandidate
                                                         : ovspatialquery::detail::QueryPrecision::exact;
    config.plan.maximumQueryCount = desc.maximum_query_count;
    config.plan.hitCapacity       = desc.hit_capacity_per_slot;
    config.plan.contactTolerance  = desc.contact_tolerance;
    if (desc.device.device_type == kDLCUDA)
    {
        ovspatialquery::detail::CudaFamilyPlanConfig cudaConfig{};
        cudaConfig.family             = cudaKind;
        cudaConfig.precision          = static_cast<ovspatialquery::detail::CudaFamilyPrecision>(desc.precision);
        cudaConfig.device             = desc.device;
        cudaConfig.resultFields       = desc.result_fields & kApiColumnFields;
        cudaConfig.hostMapped         = (desc.result_fields & ovspatialquery::detail::kBatchResultHostMapped) != 0u;
        cudaConfig.maximumQueryCount  = desc.maximum_query_count;
        cudaConfig.hitCapacityPerSlot = desc.hit_capacity_per_slot;
        cudaConfig.resultSlotCount    = desc.result_slot_count;
        cudaConfig.contactTolerance   = desc.contact_tolerance;
        cudaFamily(cudaConfig);
        if (const char* unavailable = ovspatialquery::detail::cudaBackendUnavailableReason(
                cudaConfig.device.device_id, instance->cudaModulePins(cudaConfig.device.device_id)))
        {
            return apiResult(OVSPATIALQUERY_API_NOT_SUPPORTED, unavailable);
        }
        return apiResult(apiCreateCudaFamilyPlan(instance, instanceHandle, std::move(viewRetention), Family, cudaConfig,
                                                 config, outPlan));
    }
    return apiCreateCpuFamilyPlan<Family>(instance, instanceHandle, desc, config, std::move(viewRetention),
                                          overlapShapeKind, radius, halfExtents, outPlan);
}
} // namespace

namespace ovspatialquery
{
namespace detail
{

ovspatialquery_api_status_t apiMaterializeFamilyLocked(const ApiOperationOwner& owner,
                                                       ovspatialquery_op_id_t   operationId)
{
    if (owner.plan == nullptr || owner.projection == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    ApiFamilyProjection&        projection = *std::static_pointer_cast<ApiFamilyProjection>(owner.projection);
    std::lock_guard<std::mutex> projectionLock(projection.mutex);
    if (owner.plan->owner == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    if (projection.ready)
    {
        return OVSPATIALQUERY_API_SUCCESS;
    }
    const ovspatialquery::detail::FamilyBatchPlanOperationId operation{operationId};
    const ovspatialquery_api_status_t                        access = std::visit(
        [&](auto& payload)
        {
            if constexpr (std::is_same_v<std::decay_t<decltype(payload)>, std::monostate>)
            {
                return OVSPATIALQUERY_API_INTERNAL;
            }
            else
            {
                return owner.plan->owner->fetchColumns(operation, payload);
            }
        },
        projection.payload);
    if (access != OVSPATIALQUERY_API_SUCCESS)
    {
        return access;
    }
    projection.ready = true;
    return OVSPATIALQUERY_API_SUCCESS;
}

} // namespace detail
} // namespace ovspatialquery

namespace
{

// A sweep plan's mode and limit (ADR-0067): CLOSEST and ANY store one hit per
// row and accept a limit of zero or one; MULTIPLE needs a positive limit and
// any slot capacity.
template <typename Desc>
bool apiValidSweepMode(const Desc& desc)
{
    if (static_cast<uint32_t>(desc.hit_mode) > OVSPATIALQUERY_HIT_MODE_MULTIPLE)
    {
        return false;
    }
    return desc.hit_mode == OVSPATIALQUERY_HIT_MODE_MULTIPLE
               ? desc.maximum_hits_per_query != 0u
               : desc.hit_capacity_per_slot == desc.maximum_query_count && desc.maximum_hits_per_query <= 1u;
}

template <typename Desc>
void apiSweepMode(const Desc& desc, ovspatialquery::detail::SphereSweepBatchPlan& plan)
{
    plan.hitMode = static_cast<ovspatialquery::detail::HitMode>(desc.hit_mode);
    plan.maximumHitsPerQuery =
        desc.hit_mode == OVSPATIALQUERY_HIT_MODE_MULTIPLE ? desc.maximum_hits_per_query : uint64_t{1u};
}

} // namespace

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_create_raycast_plan(
    ovspatialquery_instance_handle_t instanceHandle, const ovspatialquery_raycast_plan_desc_t* desc,
    ovspatialquery_batch_plan_handle_t* outPlan)
{
    if (outPlan == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }
    *outPlan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    if (desc == nullptr || static_cast<uint32_t>(desc->precision) > OVSPATIALQUERY_QUERY_EXACT_SINGLE ||
        desc->hit_mode > 2u || desc->result_slot_count == 0u ||
        (desc->hit_mode != 2u && desc->maximum_hits_per_query != 1u) ||
        static_cast<uint32_t>(desc->mesh_hits) > OVSPATIALQUERY_MESH_HITS_NEAREST ||
        (desc->result_fields & ~kApiRaycastResultFieldMask) != 0u ||
        ((desc->result_fields & ovspatialquery::detail::kBatchResultSourceFeature) != 0u &&
         desc->precision == OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE) ||
        !inInputDomain(desc->contact_tolerance) || desc->contact_tolerance < 0.0)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }
    const bool allMeshHits =
        desc->hit_mode == OVSPATIALQUERY_HIT_MODE_MULTIPLE && desc->mesh_hits == OVSPATIALQUERY_MESH_HITS_ALL;
    if (allMeshHits && desc->contact_tolerance != 0.0)
    {
        return apiResult(OVSPATIALQUERY_API_NOT_SUPPORTED,
                         "MESH_HITS_ALL needs a zero contact_tolerance; ask for MESH_HITS_NEAREST");
    }
    ovspatialquery::detail::RaycastFamilyBatchPlanConfig config{};
    config.plan.hitMode             = static_cast<ovspatialquery::detail::HitMode>(desc->hit_mode);
    config.plan.allMeshHits         = allMeshHits;
    config.plan.maximumHitsPerQuery = desc->maximum_hits_per_query;
    return apiCreateFamilyPlan<ApiFamily::raycast>(
        instanceHandle, *desc, config, ovspatialquery::detail::CudaFamilyKind::raycast,
        [desc, allMeshHits](ovspatialquery::detail::CudaFamilyPlanConfig& cuda)
        {
            cuda.hitMode             = static_cast<ovspatialquery::detail::CudaFamilyHitMode>(desc->hit_mode);
            cuda.allMeshHits         = allMeshHits;
            cuda.maximumHitsPerQuery = desc->maximum_hits_per_query;
        },
        0u, 0.0, {}, outPlan);
}

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_create_sphere_sweep_plan(
    ovspatialquery_instance_handle_t instanceHandle, const ovspatialquery_sphere_sweep_plan_desc_t* desc,
    ovspatialquery_batch_plan_handle_t* outPlan)
{
    if (outPlan == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }
    *outPlan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    if (desc == nullptr || static_cast<uint32_t>(desc->precision) > OVSPATIALQUERY_QUERY_EXACT_SINGLE ||
        desc->result_slot_count == 0u || !apiValidSweepMode(*desc) ||
        (desc->result_fields & ~kApiSphereSweepResultFieldMask) != 0u ||
        ((desc->result_fields & ovspatialquery::detail::kBatchResultSourceFeature) != 0u &&
         desc->precision == OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE) ||
        !inInputDomain(desc->radius) || desc->radius <= 0.0 || !inInputDomain(desc->contact_tolerance) ||
        desc->contact_tolerance < 0.0)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }
    if (desc->precision == OVSPATIALQUERY_QUERY_EXACT_SINGLE)
    {
        return apiResult(OVSPATIALQUERY_API_NOT_SUPPORTED);
    }
    ovspatialquery::detail::SphereSweepFamilyBatchPlanConfig config{};
    apiSweepMode(*desc, config.plan);
    return apiCreateFamilyPlan<ApiFamily::sphereSweep>(
        instanceHandle, *desc, config, ovspatialquery::detail::CudaFamilyKind::sphereSweep,
        [desc, &config](ovspatialquery::detail::CudaFamilyPlanConfig& cuda)
        {
            cuda.planRadius          = desc->radius;
            cuda.hitMode             = static_cast<ovspatialquery::detail::CudaFamilyHitMode>(desc->hit_mode);
            cuda.maximumHitsPerQuery = config.plan.maximumHitsPerQuery;
        },
        0u, desc->radius, {}, outPlan);
}

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_create_box_sweep_plan(
    ovspatialquery_instance_handle_t instanceHandle, const ovspatialquery_box_sweep_plan_desc_t* desc,
    ovspatialquery_batch_plan_handle_t* outPlan)
{
    if (outPlan == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }
    *outPlan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    if (desc == nullptr || static_cast<uint32_t>(desc->precision) > OVSPATIALQUERY_QUERY_EXACT_SINGLE ||
        desc->result_slot_count == 0u || !apiValidSweepMode(*desc) ||
        (desc->result_fields & ~kApiSphereSweepResultFieldMask) != 0u ||
        ((desc->result_fields & ovspatialquery::detail::kBatchResultSourceFeature) != 0u &&
         desc->precision == OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE) ||
        !inInputDomain(desc->contact_tolerance) || desc->contact_tolerance < 0.0)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }
    for (double extent : desc->half_extents)
    {
        if (!inInputDomain(extent) || extent <= 0.0)
        {
            return apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT);
        }
    }
    if (desc->precision == OVSPATIALQUERY_QUERY_EXACT_SINGLE)
    {
        return apiResult(OVSPATIALQUERY_API_NOT_SUPPORTED);
    }
    ovspatialquery::detail::SphereSweepFamilyBatchPlanConfig config{};
    config.plan.box = true;
    apiSweepMode(*desc, config.plan);
    return apiCreateFamilyPlan<ApiFamily::boxSweep>(
        instanceHandle, *desc, config, ovspatialquery::detail::CudaFamilyKind::boxSweep,
        [desc, &config](ovspatialquery::detail::CudaFamilyPlanConfig& cuda)
        {
            std::copy(desc->half_extents, desc->half_extents + 3u, cuda.planHalfExtents);
            cuda.hitMode             = static_cast<ovspatialquery::detail::CudaFamilyHitMode>(desc->hit_mode);
            cuda.maximumHitsPerQuery = config.plan.maximumHitsPerQuery;
        },
        0u, 0.0, {desc->half_extents[0], desc->half_extents[1], desc->half_extents[2]}, outPlan);
}

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_create_overlap_plan(
    ovspatialquery_instance_handle_t instanceHandle, const ovspatialquery_overlap_plan_desc_t* desc,
    ovspatialquery_batch_plan_handle_t* outPlan)
{
    if (outPlan == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }
    *outPlan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    if (desc == nullptr || static_cast<uint32_t>(desc->precision) > OVSPATIALQUERY_QUERY_EXACT_SINGLE ||
        desc->hit_mode > 2u || desc->hit_mode == 0u || desc->result_slot_count == 0u ||
        (desc->hit_mode != 2u && desc->maximum_hits_per_query != 1u) ||
        (desc->result_fields & ~kApiOverlapResultFieldMask) != 0u || desc->shape.kind > 2u ||
        (desc->shape.kind == 0u &&
         (!inInputDomain(desc->shape.data.sphere.radius) || desc->shape.data.sphere.radius <= 0.0)) ||
        (desc->shape.kind == 1u &&
         (!inInputDomain(desc->shape.data.box.half_extents[0]) ||
          !inInputDomain(desc->shape.data.box.half_extents[1]) ||
          !inInputDomain(desc->shape.data.box.half_extents[2]) || desc->shape.data.box.half_extents[0] <= 0.0 ||
          desc->shape.data.box.half_extents[1] <= 0.0 || desc->shape.data.box.half_extents[2] <= 0.0)) ||
        (desc->shape.kind == 2u &&
         (!inInputDomain(desc->shape.data.aabb.half_extents[0]) ||
          !inInputDomain(desc->shape.data.aabb.half_extents[1]) ||
          !inInputDomain(desc->shape.data.aabb.half_extents[2]) || desc->shape.data.aabb.half_extents[0] <= 0.0 ||
          desc->shape.data.aabb.half_extents[1] <= 0.0 || desc->shape.data.aabb.half_extents[2] <= 0.0)) ||
        !inInputDomain(desc->contact_tolerance) || desc->contact_tolerance < 0.0)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }
    if (desc->precision == OVSPATIALQUERY_QUERY_EXACT_SINGLE)
    {
        return apiResult(OVSPATIALQUERY_API_NOT_SUPPORTED);
    }
    ovspatialquery::detail::OverlapFamilyBatchPlanConfig config{};
    config.plan.hitMode             = static_cast<ovspatialquery::detail::HitMode>(desc->hit_mode);
    config.plan.maximumHitsPerQuery = desc->maximum_hits_per_query;
    const double* const extents =
        desc->shape.kind == 1u ? desc->shape.data.box.half_extents : desc->shape.data.aabb.half_extents;
    return apiCreateFamilyPlan<ApiFamily::overlap>(
        instanceHandle, *desc, config, ovspatialquery::detail::CudaFamilyKind::overlap,
        [desc, extents](ovspatialquery::detail::CudaFamilyPlanConfig& cuda)
        {
            cuda.hitMode             = static_cast<ovspatialquery::detail::CudaFamilyHitMode>(desc->hit_mode);
            cuda.maximumHitsPerQuery = desc->maximum_hits_per_query;
            cuda.overlapKind         = static_cast<ovspatialquery::detail::CudaOverlapKind>(desc->shape.kind);
            if (desc->shape.kind == 0u)
            {
                cuda.planRadius = desc->shape.data.sphere.radius;
            }
            else
            {
                std::copy(extents, extents + 3u, cuda.planHalfExtents);
            }
        },
        desc->shape.kind, desc->shape.data.sphere.radius, {extents[0], extents[1], extents[2]}, outPlan);
}

extern "C" OVSPATIALQUERY_API ovspatialquery_enqueue_result_t ovspatialquery_enqueue_raycast_plan(
    ovspatialquery_instance_handle_t instanceHandle, ovspatialquery_batch_plan_handle_t planHandle,
    const ovspatialquery_raycast_batch_t* batch, ovspatialquery_raycast_batch_reservation_t* outReservation)
{
    return apiEnqueueFamily<ApiFamily::raycast, ovspatialquery_raycast_batch_result_t>(
        instanceHandle, planHandle, batch, outReservation, &ApiFamilyPlanState::raycastScratch,
        [batch](ovspatialquery::detail::CudaFamilyLaunch& launch)
        {
            apiCudaInput(batch->maximum_hits, "maximum_hits", launch.rowMaximumHits);
            apiCudaInput(batch->origins, "origins", launch.positions);
            apiCudaInput(batch->unit_directions, "unit_directions", launch.directionsOrRotations);
            apiCudaInput(batch->min_distances, "min_distances", launch.minimumDistances);
            apiCudaInput(batch->max_distances, "max_distances", launch.maximumDistances);
            launch.queryFlags = batch->query_flags;
        },
        [](const ovspatialquery::detail::CudaFamilyResultColumns& columns,
           ovspatialquery_raycast_batch_reservation_t& reservation) { apiCudaRaycastColumns(columns, reservation); },
        [batch](const ApiFamilyPlanState&, ovspatialquery::detail::RaycastBatchLaunch& launch)
        {
            const uint64_t             rows = batch->query_count;
            ApiQueryBitsColumn         bits;
            ApiInputColumn<double, 3u> origins;
            ApiInputColumn<float, 3u>  directions;
            ApiScalarColumn            minimums, maximums;
            ApiCountColumn             maximumHits;
            if (!apiOpenQueryBits(batch->query_bits, rows, bits) ||
                !origins.open(batch->origins, rows, kDLFloat, 64u, false) ||
                !directions.open(batch->unit_directions, rows, kDLFloat, 32u, false) ||
                !minimums.open<true>(batch->min_distances, rows, kDLFloat, 64u, true, {0.0}) ||
                !maximums.open<true>(batch->max_distances, rows, kDLFloat, 64u, false) ||
                !maximumHits.open<true>(batch->maximum_hits, rows, kDLUInt, 64u, true, {UINT64_MAX}))
            {
                return false;
            }
            const bool bothSides = (batch->query_flags & OVSPATIALQUERY_QUERY_MESH_BOTH_SIDES) != 0u;
            for (uint64_t index = 0u; index < rows; ++index)
            {
                ovspatialquery::detail::RaycastBatchRow& row = launch.rows[static_cast<size_t>(index)];
                row.query.bothSides                          = bothSides;
                row.queryBits                                = apiQueryBits(bits, index);
                origins.read(index, row.query.origin);
                directions.read(index, row.query.unitDirection);
                row.query.minimumDistance = minimums.scalar(index);
                row.query.maximumDistance = maximums.scalar(index);
                row.maximumHits           = maximumHits.scalar(index);
            }
            return true;
        });
}

extern "C" OVSPATIALQUERY_API ovspatialquery_enqueue_result_t ovspatialquery_enqueue_sphere_sweep_plan(
    ovspatialquery_instance_handle_t instanceHandle, ovspatialquery_batch_plan_handle_t planHandle,
    const ovspatialquery_sphere_sweep_batch_t* batch, ovspatialquery_sphere_sweep_batch_reservation_t* outReservation)
{
    return apiEnqueueFamily<ApiFamily::sphereSweep, ovspatialquery_sphere_sweep_batch_result_t>(
        instanceHandle, planHandle, batch, outReservation, &ApiFamilyPlanState::sweepScratch,
        [batch](ovspatialquery::detail::CudaFamilyLaunch& launch)
        {
            apiCudaInput(batch->centers, "centers", launch.positions);
            apiCudaInput(batch->unit_directions, "unit_directions", launch.directionsOrRotations);
            apiCudaInput(batch->min_distances, "min_distances", launch.minimumDistances);
            apiCudaInput(batch->max_distances, "max_distances", launch.maximumDistances);
            apiCudaInput(batch->radii, "radii", launch.radii);
            launch.queryFlags = batch->query_flags;
        },
        [](const ovspatialquery::detail::CudaFamilyResultColumns& columns,
           ovspatialquery_sphere_sweep_batch_reservation_t&       reservation)
        { apiCudaSphereSweepColumns(columns, reservation); },
        [batch](const ApiFamilyPlanState& plan, ovspatialquery::detail::SphereSweepBatchLaunch& launch)
        {
            // A row's own radius, when the batch carries radii, else the plan's.
            const uint64_t             rows = batch->query_count;
            ApiQueryBitsColumn         bits;
            ApiInputColumn<double, 3u> centers;
            ApiInputColumn<float, 3u>  directions;
            ApiScalarColumn            minimums, maximums, radii;
            if (!apiOpenQueryBits(batch->query_bits, rows, bits) ||
                !centers.open(batch->centers, rows, kDLFloat, 64u, false) ||
                !directions.open(batch->unit_directions, rows, kDLFloat, 32u, false) ||
                !minimums.open<true>(batch->min_distances, rows, kDLFloat, 64u, true, {0.0}) ||
                !maximums.open<true>(batch->max_distances, rows, kDLFloat, 64u, false) ||
                !radii.open<true>(batch->radii, rows, kDLFloat, 64u, true, {plan.radius}))
            {
                return false;
            }
            const bool bothSides = (batch->query_flags & OVSPATIALQUERY_QUERY_MESH_BOTH_SIDES) != 0u;
            for (uint64_t index = 0u; index < rows; ++index)
            {
                ovspatialquery::detail::SphereSweepBatchRow& row   = launch.rows[static_cast<size_t>(index)];
                ovspatialquery::detail::SphereSweepQuery&    query = row.query;
                query.bothSides                                    = bothSides;
                row.queryBits                                      = apiQueryBits(bits, index);
                centers.read(index, query.center);
                directions.read(index, query.unitDirection);
                query.minimumDistance = minimums.scalar(index);
                query.maximumDistance = maximums.scalar(index);
                query.radius          = radii.scalar(index);
            }
            return true;
        });
}

extern "C" OVSPATIALQUERY_API ovspatialquery_enqueue_result_t ovspatialquery_enqueue_box_sweep_plan(
    ovspatialquery_instance_handle_t instanceHandle, ovspatialquery_batch_plan_handle_t planHandle,
    const ovspatialquery_box_sweep_batch_t* batch, ovspatialquery_box_sweep_batch_reservation_t* outReservation)
{
    return apiEnqueueFamily<ApiFamily::boxSweep, ovspatialquery_sphere_sweep_batch_result_t>(
        instanceHandle, planHandle, batch, outReservation, &ApiFamilyPlanState::sweepScratch,
        [batch](ovspatialquery::detail::CudaFamilyLaunch& launch)
        {
            apiCudaInput(batch->centers, "centers", launch.positions);
            apiCudaInput(batch->unit_directions, "unit_directions", launch.directionsOrRotations);
            apiCudaInput(batch->rotations_xyzw, "rotations_xyzw", launch.rotations);
            apiCudaInput(batch->min_distances, "min_distances", launch.minimumDistances);
            apiCudaInput(batch->max_distances, "max_distances", launch.maximumDistances);
            apiCudaInput(batch->half_extents, "half_extents", launch.halfExtents);
            launch.queryFlags = batch->query_flags;
        },
        [](const ovspatialquery::detail::CudaFamilyResultColumns& columns,
           ovspatialquery_box_sweep_batch_reservation_t&          reservation)
        { apiCudaSphereSweepColumns(columns, reservation); },
        [batch](const ApiFamilyPlanState& plan, ovspatialquery::detail::SphereSweepBatchLaunch& launch)
        {
            // A row's own half extents, when the batch carries them, else the plan's.
            const uint64_t             rows = batch->query_count;
            ApiQueryBitsColumn         bits;
            ApiInputColumn<double, 3u> centers, halfExtents;
            ApiInputColumn<float, 4u>  rotations;
            ApiInputColumn<float, 3u>  directions;
            ApiScalarColumn            minimums, maximums;
            if (!apiOpenQueryBits(batch->query_bits, rows, bits) ||
                !centers.open(batch->centers, rows, kDLFloat, 64u, false) ||
                !rotations.open(batch->rotations_xyzw, rows, kDLFloat, 32u, false) ||
                !directions.open(batch->unit_directions, rows, kDLFloat, 32u, false) ||
                !minimums.open<true>(batch->min_distances, rows, kDLFloat, 64u, true, {0.0}) ||
                !maximums.open<true>(batch->max_distances, rows, kDLFloat, 64u, false) ||
                !halfExtents.open(batch->half_extents, rows, kDLFloat, 64u, true, plan.halfExtents))
            {
                return false;
            }
            const bool bothSides = (batch->query_flags & OVSPATIALQUERY_QUERY_MESH_BOTH_SIDES) != 0u;
            for (uint64_t index = 0u; index < rows; ++index)
            {
                ovspatialquery::detail::SphereSweepBatchRow& row   = launch.rows[static_cast<size_t>(index)];
                ovspatialquery::detail::SphereSweepQuery&    query = row.query;
                query.box                                          = true;
                query.bothSides                                    = bothSides;
                row.queryBits                                      = apiQueryBits(bits, index);
                centers.read(index, query.center);
                rotations.read(index, query.rotationXyzw);
                directions.read(index, query.unitDirection);
                query.minimumDistance = minimums.scalar(index);
                query.maximumDistance = maximums.scalar(index);
                halfExtents.read(index, query.halfExtents);
            }
            return true;
        });
}

extern "C" OVSPATIALQUERY_API ovspatialquery_enqueue_result_t ovspatialquery_enqueue_overlap_plan(
    ovspatialquery_instance_handle_t instanceHandle, ovspatialquery_batch_plan_handle_t planHandle,
    const ovspatialquery_overlap_batch_t* batch, ovspatialquery_overlap_batch_reservation_t* outReservation)
{
    return apiEnqueueFamily<ApiFamily::overlap, ovspatialquery_overlap_batch_result_t>(
        instanceHandle, planHandle, batch, outReservation, &ApiFamilyPlanState::overlapScratch,
        [batch](ovspatialquery::detail::CudaFamilyLaunch& launch)
        {
            apiCudaInput(batch->translations, "translations", launch.positions);
            apiCudaInput(batch->rotations_xyzw, "rotations_xyzw", launch.directionsOrRotations);
            apiCudaInput(batch->maximum_hits, "maximum_hits", launch.rowMaximumHits);
            apiCudaInput(batch->half_extents, "half_extents", launch.halfExtents);
            apiCudaInput(batch->radii, "radii", launch.radii);
        },
        [](const ovspatialquery::detail::CudaFamilyResultColumns& columns,
           ovspatialquery_overlap_batch_reservation_t& reservation) { apiCudaBaseColumns(columns, reservation); },
        [batch](const ApiFamilyPlanState& plan, ovspatialquery::detail::OverlapBatchLaunch& launch)
        {
            // Rotations orient a box; a sphere or AABB has none, so the column
            // must be absent for those shapes and is required for a box.
            const bool boxShape = plan.overlapShapeKind == 1u;
            if (!boxShape && batch->rotations_xyzw != nullptr)
            {
                return false;
            }
            // AABB and BOX rows may carry their own half extents; a sphere has none.
            const bool rowExtents =
                (plan.overlapShapeKind == 1u || plan.overlapShapeKind == 2u) && batch->half_extents != nullptr;
            if (!rowExtents && batch->half_extents != nullptr)
            {
                return false;
            }
            // SPHERE rows may carry their own radii; a box or AABB has none.
            const bool rowRadii = plan.overlapShapeKind == 0u && batch->radii != nullptr;
            if (!rowRadii && batch->radii != nullptr)
            {
                return false;
            }
            // A BOX row's rotation comes from a float64 or a float32 column (ADR-0026); an
            // AABB or BOX row's half extents and a SPHERE row's radius, when present, are its own.
            const uint64_t             rows = batch->query_count;
            ApiQueryBitsColumn         bits;
            ApiInputColumn<double, 3u> translations, halfExtents;
            ApiInputColumn<double, 4u> rotations64;
            ApiInputColumn<float, 4u>  rotations32;
            ApiScalarColumn            radii;
            ApiCountColumn             maximumHits;
            const bool                 float64Rotation =
                boxShape && rotations64.open(batch->rotations_xyzw, rows, kDLFloat, 64u, false);
            if (!apiOpenQueryBits(batch->query_bits, rows, bits) ||
                !translations.open(batch->translations, rows, kDLFloat, 64u, false) ||
                (boxShape && !float64Rotation &&
                 !rotations32.open(batch->rotations_xyzw, rows, kDLFloat, 32u, false)) ||
                !halfExtents.open(rowExtents ? batch->half_extents : nullptr, rows, kDLFloat, 64u, true,
                                  plan.halfExtents) ||
                !radii.open<true>(rowRadii ? batch->radii : nullptr, rows, kDLFloat, 64u, true, {plan.radius}) ||
                !maximumHits.open<true>(batch->maximum_hits, rows, kDLUInt, 64u, true, {UINT64_MAX}))
            {
                return false;
            }
            for (uint64_t index = 0u; index < rows; ++index)
            {
                ovspatialquery::detail::OverlapBatchRow& row = launch.rows[static_cast<size_t>(index)];
                row                                          = {};
                std::array<double, 3u> translation{};
                std::array<double, 4u> rotation{0.0, 0.0, 0.0, 1.0};
                std::array<double, 3u> half{};
                row.queryBits = apiQueryBits(bits, index);
                translations.read(index, translation);
                if (float64Rotation)
                {
                    rotations64.read(index, rotation);
                }
                else if (boxShape)
                {
                    std::array<float, 4u> single{};
                    rotations32.read(index, single);
                    for (size_t component = 0u; component < 4u; ++component)
                    {
                        rotation[component] = static_cast<double>(single[component]);
                    }
                }
                halfExtents.read(index, half);
                const double radius                      = radii.scalar(index);
                row.maximumHits                          = maximumHits.scalar(index);
                ovspatialquery_overlap_shape_t& geometry = row.query.geometry;
                if (plan.overlapShapeKind == 0u)
                {
                    // A bad row radius fails the shape's validation, so the row
                    // reports INVALID_ARGUMENT.
                    geometry.kind = OVSPATIALQUERY_OVERLAP_SPHERE;
                    std::copy(translation.begin(), translation.end(), geometry.data.sphere.center);
                    geometry.data.sphere.radius = radius;
                }
                else if (boxShape)
                {
                    // A bad half-extent triple fails the shape's validation, so the
                    // row reports INVALID_ARGUMENT. The float32 shape field keeps the
                    // row's rotation, exactly for a float32 column and as its nearest
                    // value for a float64 one; only a float64 row carries its rotation
                    // in double, so a float32 row evaluates as before.
                    geometry.kind = OVSPATIALQUERY_OVERLAP_BOX;
                    std::copy(translation.begin(), translation.end(), geometry.data.box.center);
                    std::copy(half.begin(), half.end(), geometry.data.box.half_extents);
                    for (size_t component = 0u; component < 4u; ++component)
                    {
                        geometry.data.box.rotation_xyzw[component] = static_cast<float>(rotation[component]);
                    }
                    row.query.hasBoxRotation  = float64Rotation;
                    row.query.boxRotationXyzw = rotation;
                }
                else
                {
                    // A row's own half extents must be a finite positive triple;
                    // otherwise the row's corners are left unordered and the row
                    // reports INVALID_ARGUMENT, as the header promises. Corners
                    // outside the input domain report it too.
                    geometry.kind = OVSPATIALQUERY_OVERLAP_AABB;
                    bool positive = true;
                    for (size_t axis = 0u; axis < 3u; ++axis)
                    {
                        positive = positive && std::isfinite(half[axis]) && half[axis] > 0.0;
                    }
                    for (size_t axis = 0u; axis < 3u; ++axis)
                    {
                        geometry.data.aabb.min_corner[axis] =
                            positive ? translation[axis] - half[axis] : std::numeric_limits<double>::infinity();
                        geometry.data.aabb.max_corner[axis] =
                            positive ? translation[axis] + half[axis] : -std::numeric_limits<double>::infinity();
                    }
                }
            }
            return true;
        });
}

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t
ovspatialquery_fetch_raycast_batch(ovspatialquery_instance_handle_t instanceHandle, ovspatialquery_op_id_t operationId,
                                   ovspatialquery_raycast_batch_result_t* outResult)
{
    return apiFetchFamily<ApiFamily::raycast>(instanceHandle, operationId, outResult,
                                              [](const ovspatialquery::detail::CudaFamilyResultColumns& columns,
                                                 ovspatialquery_raycast_batch_result_t&                 result)
                                              { apiCudaRaycastColumns(columns, result); });
}

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_fetch_sphere_sweep_batch(
    ovspatialquery_instance_handle_t instanceHandle, ovspatialquery_op_id_t operationId,
    ovspatialquery_sphere_sweep_batch_result_t* outResult)
{
    return apiFetchFamily<ApiFamily::sphereSweep>(instanceHandle, operationId, outResult,
                                                  [](const ovspatialquery::detail::CudaFamilyResultColumns& columns,
                                                     ovspatialquery_sphere_sweep_batch_result_t&            result)
                                                  { apiCudaSphereSweepColumns(columns, result); });
}

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_fetch_box_sweep_batch(
    ovspatialquery_instance_handle_t instanceHandle, ovspatialquery_op_id_t operationId,
    ovspatialquery_box_sweep_batch_result_t* outResult)
{
    return apiFetchFamily<ApiFamily::boxSweep>(instanceHandle, operationId, outResult,
                                               [](const ovspatialquery::detail::CudaFamilyResultColumns& columns,
                                                  ovspatialquery_box_sweep_batch_result_t&               result)
                                               { apiCudaSphereSweepColumns(columns, result); });
}

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t
ovspatialquery_fetch_overlap_batch(ovspatialquery_instance_handle_t instanceHandle, ovspatialquery_op_id_t operationId,
                                   ovspatialquery_overlap_batch_result_t* outResult)
{
    return apiFetchFamily<ApiFamily::overlap>(instanceHandle, operationId, outResult,
                                              [](const ovspatialquery::detail::CudaFamilyResultColumns& columns,
                                                 ovspatialquery_overlap_batch_result_t&                 result)
                                              { apiCudaBaseColumns(columns, result); });
}

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_destroy_batch_plan(
    ovspatialquery_instance_handle_t instanceHandle, ovspatialquery_batch_plan_handle_t planHandle)
{
    return apiDestroyAdapter<ApiFamilyPlanState>(instanceHandle, planHandle.raw_id);
}

#if defined(OVSPATIALQUERY_ENABLE_FAMILY_BATCH_PLAN_OPERATION_TEST_HOOK)
// Failure injection the real path cannot produce on demand: the next operation
// a CPU plan admits fails with exhausted resources.
extern "C" ovspatialquery_api_status_t ovspatialquery_test_fail_next_family_plan_resource(
    ovspatialquery_instance_handle_t instanceHandle, ovspatialquery_batch_plan_handle_t planHandle)
{
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    const std::shared_ptr<ApiFamilyPlanState> plan = apiFindAdapter<ApiFamilyPlanState>(*instance, planHandle.raw_id);
    if (plan == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    std::lock_guard<std::mutex> adapterLock(plan->adapterMutex);
    if (plan->cudaPlan || plan->owner == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    plan->owner->failNextResourceForTest();
    return OVSPATIALQUERY_API_SUCCESS;
}
#endif
