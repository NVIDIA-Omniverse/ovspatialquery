/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_float_bounds.h"
#include "ovspatialquery_instance_state.h"

#include "ovspatialquery_cuda_pair_reporter.h"
#include "ovspatialquery_instance_operation_runtime.h"
#include "ovspatialquery_pair_materialization.h"
#include "ovspatialquery_target_resolver.h"

#include <ovspatialquery_query_evaluation.h>

#include <algorithm>
#include <cmath>
#include <new>
#include <stdexcept>

using namespace ovspatialquery::detail;

namespace
{

bool apiAddReporterStorageCharge(uint64_t deltaPairCapacity, uint32_t resultSlotCount, uint64_t& total) noexcept
{
    uint64_t projectionCharge = sizeof(ApiReporterProjection);
    return addHostCharge(deltaPairCapacity, 2u * sizeof(ovspatialquery_pair_t), projectionCharge) &&
           addHostCharge(1u,
                         sizeof(ApiPairReporterState) + 3u * kApiSharedControlAllowance + kApiDirectoryNodeAllowance,
                         total) &&
           addHostCharge(resultSlotCount,
                         sizeof(std::shared_ptr<ApiReporterProjection>) + kApiSharedControlAllowance + projectionCharge,
                         total);
}

bool apiReporterCapacityCharge(uint64_t activePairCapacity, uint64_t deltaPairCapacity, uint32_t resultSlotCount,
                               uint64_t& outCharge) noexcept
{
    outCharge                             = 0u;
    const uint64_t ownerDeltaElementBytes = static_cast<uint64_t>(resultSlotCount) * 2u * sizeof(ovspatialquery_pair_t);
    const uint64_t projectionDeltaElementBytes =
        static_cast<uint64_t>(resultSlotCount) * 2u * sizeof(ovspatialquery_pair_t);
    // The CPU reporter keeps two accepted sets: the current one and the buffer
    // its next patch writes into.
    return addHostCharge(activePairCapacity, 2u * sizeof(ovspatialquery_pair_t), outCharge) &&
           addHostCharge(deltaPairCapacity, ownerDeltaElementBytes, outCharge) &&
           addHostCharge(deltaPairCapacity, projectionDeltaElementBytes, outCharge);
}

void apiPrepareReporterStorage(ApiPairReporterState& reporter, uint32_t resultSlotCount)
{
    reporter.projections.reserve(resultSlotCount);
    for (uint32_t index = 0u; index < resultSlotCount; ++index)
    {
        const auto projection = std::make_shared<ApiReporterProjection>();
        if (reporter.resultDevice.device_type != kDLCUDA)
        {
            projection->entered.reserve(static_cast<size_t>(reporter.deltaPairCapacity));
            projection->exited.reserve(static_cast<size_t>(reporter.deltaPairCapacity));
        }
        projection->reservedShape[0] = static_cast<int64_t>(reporter.deltaPairCapacity);
        reporter.projections.push_back(projection);
    }
}

// The device bytes a CUDA reporter keeps, charged as result storage while it lives.
ovspatialquery_api_status_t apiReserveReporterDevice(ovspatialquery_instance_handle_t instanceHandle, DLDevice device,
                                                     uint64_t activePairCapacity, uint64_t deltaPairCapacity,
                                                     uint32_t resultSlotCount, std::shared_ptr<void>& outLease)
{
    uint64_t bytes = 0u;
    if (!ovspatialquery::detail::CudaPairReporterDevice::residentDeviceBytes(activePairCapacity, deltaPairCapacity,
                                                                             resultSlotCount, bytes))
    {
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    return ovspatialquery::detail::reserveInstanceDeviceMemory(instanceHandle, device, bytes, outLease);
}

std::shared_ptr<ApiReporterProjection> apiActivateReporterProjection(const ApiPairReporterState& reporter,
                                                                     uint32_t                    resultSlot)
{
    if (resultSlot >= reporter.projections.size())
    {
        return {};
    }
    const std::shared_ptr<ApiReporterProjection> projection = reporter.projections[resultSlot];
    if (projection == nullptr)
    {
        return {};
    }
    std::lock_guard<std::mutex> lock(projection->mutex);
    projection->resultSlot              = resultSlot;
    projection->outcome                 = 0u;
    projection->firstSourceOrdinal      = 0;
    projection->secondSourceOrdinal     = 0;
    projection->acceptedActivePairCount = 0u;
    projection->entered.clear();
    projection->exited.clear();
    projection->enteredShape[0] = 0;
    projection->exitedShape[0]  = 0;
    projection->device          = DLDevice{kDLCPU, 0};
    projection->deviceEntered   = nullptr;
    projection->deviceExited    = nullptr;
    projection->deviceCounts    = nullptr;
    projection->resultReady     = {};
    projection->ready           = false;
    return projection;
}

DLTensor apiPairTensor(const std::vector<ovspatialquery_pair_t>& pairs, const void* deviceRows, DLDevice device,
                       const std::array<int64_t, 2u>& shape, const std::array<int64_t, 2u>& strides)
{
    DLTensor result{};
    result.data    = device.device_type == kDLCUDA ? const_cast<void*>(deviceRows)
                                                   : static_cast<void*>(const_cast<ovspatialquery_pair_t*>(pairs.data()));
    result.device  = device;
    result.ndim    = 2;
    result.dtype   = DLDataType{kDLUInt, 64u, 1u};
    result.shape   = const_cast<int64_t*>(shape.data());
    result.strides = const_cast<int64_t*>(strides.data());
    return result;
}

void apiReporterResult(const ApiReporterProjection& projection, ovspatialquery_pair_reporter_result_t& out)
{
    out                            = {};
    out.result_slot                = projection.resultSlot;
    out.outcome                    = static_cast<ovspatialquery_pair_reporter_outcome_t>(projection.outcome);
    out.first_source_ordinal       = projection.firstSourceOrdinal;
    out.second_source_ordinal      = projection.secondSourceOrdinal;
    out.accepted_active_pair_count = projection.acceptedActivePairCount;
    out.result_ready               = projection.resultReady;
    out.entered_pairs              = apiPairTensor(projection.entered, projection.deviceEntered, projection.device,
                                                   projection.enteredShape, projection.enteredStrides);
    out.exited_pairs               = apiPairTensor(projection.exited, projection.deviceExited, projection.device,
                                                   projection.exitedShape, projection.exitedStrides);
}

// A CUDA reporter's reservation describes the slot's capacity-shaped delta
// rows and its counts (ADR-0022 decision 2); a CPU reporter's leaves the
// three descriptors absent.
void apiReporterReservation(const ApiPairReporterState& reporter, ApiReporterProjection& projection,
                            const ovspatialquery::detail::PairReporterResultStorageView& early,
                            ovspatialquery_pair_reporter_reservation_t&                  out)
{
    out                     = {};
    out.result_slot         = projection.resultSlot;
    out.result_device       = reporter.resultDevice;
    out.delta_pair_capacity = reporter.deltaPairCapacity;
    out.result_ready        = early.resultReady;
    if (early.device.device_type != kDLCUDA || early.deviceEntered == nullptr || early.deviceExited == nullptr ||
        early.deviceCounts == nullptr)
    {
        return;
    }
    std::lock_guard<std::mutex> lock(projection.mutex);
    projection.device        = early.device;
    projection.deviceEntered = early.deviceEntered;
    projection.deviceExited  = early.deviceExited;
    projection.deviceCounts  = early.deviceCounts;
    projection.resultReady   = early.resultReady;
    out.entered_pairs  = apiPairTensor(projection.entered, early.deviceEntered, early.device, projection.reservedShape,
                                       projection.enteredStrides);
    out.exited_pairs   = apiPairTensor(projection.exited, early.deviceExited, early.device, projection.reservedShape,
                                       projection.exitedStrides);
    out.counts.data    = const_cast<void*>(early.deviceCounts);
    out.counts.device  = early.device;
    out.counts.ndim    = 1;
    out.counts.dtype   = DLDataType{kDLUInt, 64u, 1u};
    out.counts.shape   = projection.countsShape.data();
    out.counts.strides = projection.countsStrides.data();
}

} // namespace

namespace ovspatialquery
{
namespace detail
{

ovspatialquery_api_status_t apiMaterializeReporterLocked(const ApiOperationOwner& owner,
                                                         ovspatialquery_op_id_t   operationId)
{
    if (owner.reporter == nullptr || owner.projection == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    ApiReporterProjection&      projection = *std::static_pointer_cast<ApiReporterProjection>(owner.projection);
    std::lock_guard<std::mutex> projectionLock(projection.mutex);
    if (owner.reporter->owner == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    if (projection.ready)
    {
        return OVSPATIALQUERY_API_SUCCESS;
    }
    ovspatialquery::detail::PairReporterResultStorageView view{};
    const ovspatialquery_api_status_t access = owner.reporter->owner->fetchStorageView({operationId}, view);
    if (access != OVSPATIALQUERY_API_SUCCESS)
    {
        return access;
    }
    // A CUDA reporter's rows stay on the device and the projection describes
    // them; a CPU reporter's are copied.
    const bool cuda = view.device.device_type == kDLCUDA;
    if (cuda ? (view.deviceEntered == nullptr || view.deviceExited == nullptr ||
                view.enteredCount > owner.reporter->deltaPairCapacity ||
                view.exitedCount > owner.reporter->deltaPairCapacity)
             : (view.enteredCount > projection.entered.capacity() || view.exitedCount > projection.exited.capacity() ||
                (view.enteredCount != 0u && view.entered == nullptr) ||
                (view.exitedCount != 0u && view.exited == nullptr)))
    {
        return OVSPATIALQUERY_API_INTERNAL;
    }
    projection.resultSlot              = view.resultSlot;
    projection.outcome                 = static_cast<uint32_t>(view.outcome);
    projection.firstSourceOrdinal      = view.firstSourceOrdinal;
    projection.secondSourceOrdinal     = view.secondSourceOrdinal;
    projection.acceptedActivePairCount = view.acceptedActivePairCount;
    projection.enteredShape[0]         = static_cast<int64_t>(view.enteredCount);
    projection.exitedShape[0]          = static_cast<int64_t>(view.exitedCount);
    projection.ready                   = true;
    if (cuda)
    {
        projection.device        = view.device;
        projection.deviceEntered = view.deviceEntered;
        projection.deviceExited  = view.deviceExited;
        projection.deviceCounts  = view.deviceCounts;
        projection.resultReady   = view.resultReady;
        return OVSPATIALQUERY_API_SUCCESS;
    }
    projection.entered.assign(view.entered, view.entered + view.enteredCount);
    projection.exited.assign(view.exited, view.exited + view.exitedCount);
    return OVSPATIALQUERY_API_SUCCESS;
}

} // namespace detail
} // namespace ovspatialquery

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_create_pair_reporter(
    ovspatialquery_instance_handle_t instanceHandle, const ovspatialquery_pair_reporter_desc_t* desc,
    ovspatialquery_pair_reporter_handle_t* outReporter)
{
    if (outReporter == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }
    *outReporter = OVSPATIALQUERY_INVALID_PAIR_REPORTER;
    if (desc == nullptr || !inInputDomain(desc->contact_tolerance) || desc->contact_tolerance < 0.0 ||
        desc->result_slot_count == 0u)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }
    const bool cudaResults = desc->result_device.device_type == kDLCUDA;
    if ((cudaResults && desc->result_device.device_id < 0) ||
        (!cudaResults && (desc->result_device.device_type != kDLCPU || desc->result_device.device_id != 0)))
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_HANDLE);
    }
    if (const char* unavailable =
            cudaResults ? ovspatialquery::detail::cudaBackendUnavailableReason(
                              desc->result_device.device_id, instance->cudaModulePins(desc->result_device.device_id))
                        : nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_NOT_SUPPORTED, unavailable);
    }
    std::lock_guard<std::mutex>                         lifecycle(instance->lifecycleMutex);
    ovspatialquery_api_status_t                         status = OVSPATIALQUERY_API_SUCCESS;
    ovspatialquery::detail::PairReporterOperationConfig descriptor{};
    std::shared_ptr<void>                               firstViewRetention;
    std::shared_ptr<void>                               secondViewRetention;
    descriptor.first.target = apiPlanTarget(instance, desc->first, true, firstViewRetention, status);
    if (status == OVSPATIALQUERY_API_SUCCESS)
    {
        descriptor.second.target = apiPlanTarget(instance, desc->second, true, secondViewRetention, status);
    }
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        return apiResult(status);
    }
    descriptor.first.queryBits    = desc->first_query_bits;
    descriptor.second.queryBits   = desc->second_query_bits;
    descriptor.activePairCapacity = desc->active_pair_capacity;
    descriptor.deltaPairCapacity  = desc->delta_pair_capacity;
    descriptor.resultSlotCount    = desc->result_slot_count;
    descriptor.contactTolerance   = desc->contact_tolerance;
    descriptor.hostLedger         = &instance->hostLedger;
    descriptor.resultDevice       = desc->result_device;
    descriptor.relation           = ovspatialquery::detail::pairEndpointRelation(
        descriptor.first.target.population == descriptor.second.target.population, desc->first, desc->first_query_bits,
        desc->second, desc->second_query_bits);
    // A CUDA reporter keeps its accepted set and deltas on the device.
    uint64_t fixedHostCharge = 0u;
    if (!ovspatialquery::detail::PairReporterOperationOwner::estimateResidentBytes(descriptor, fixedHostCharge) ||
        !apiAddReporterStorageCharge(cudaResults ? 0u : desc->delta_pair_capacity, desc->result_slot_count,
                                     fixedHostCharge))
    {
        return apiResult(OVSPATIALQUERY_API_OUT_OF_MEMORY);
    }
    uint64_t capacityCharge = 0u;
    if (!cudaResults && (!apiReporterCapacityCharge(desc->active_pair_capacity, desc->delta_pair_capacity,
                                                    desc->result_slot_count, capacityCharge) ||
                         capacityCharge > fixedHostCharge))
    {
        return apiResult(OVSPATIALQUERY_API_OUT_OF_MEMORY);
    }
    std::shared_ptr<void> deviceLease;
    if (cudaResults)
    {
        const ovspatialquery_api_status_t reserved =
            apiReserveReporterDevice(instanceHandle, desc->result_device, desc->active_pair_capacity,
                                     desc->delta_pair_capacity, desc->result_slot_count, deviceLease);
        if (reserved != OVSPATIALQUERY_API_SUCCESS)
        {
            return apiResult(reserved);
        }
    }
    std::shared_ptr<ovspatialquery::detail::HostLease> hostLease;
    const ovspatialquery_api_status_t                  hostLeaseStatus =
        reserveSharedHostLease(instance, fixedHostCharge - capacityCharge, hostLease);
    if (hostLeaseStatus != OVSPATIALQUERY_API_SUCCESS)
    {
        return apiResult(hostLeaseStatus);
    }
    std::shared_ptr<ovspatialquery::detail::HostLease> capacityHostLease;
    const ovspatialquery_api_status_t                  capacityLeaseStatus =
        reserveSharedHostLease(instance, capacityCharge, capacityHostLease);
    if (capacityLeaseStatus != OVSPATIALQUERY_API_SUCCESS)
    {
        return apiResult(capacityLeaseStatus);
    }
    std::unique_ptr<ovspatialquery::detail::PairReporterOperationOwner> owner;
    status = ovspatialquery::detail::PairReporterOperationOwner::create(instance->populations, instanceHandle,
                                                                        descriptor, owner);
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        return apiResult(status);
    }
    try
    {
        const auto reporter          = std::make_shared<ApiPairReporterState>();
        reporter->activePairCapacity = desc->active_pair_capacity;
        reporter->deltaPairCapacity  = desc->delta_pair_capacity;
        reporter->resultSlotCount    = desc->result_slot_count;
        reporter->resultDevice       = cudaResults ? desc->result_device : DLDevice{kDLCPU, 0};
        reporter->deviceLease        = std::move(deviceLease);
        reporter->viewRetentions[0]  = std::move(firstViewRetention);
        reporter->viewRetentions[1]  = std::move(secondViewRetention);
        reporter->owner              = std::move(owner);
        apiPrepareReporterStorage(*reporter, desc->result_slot_count);
        reporter->hostLease         = std::move(hostLease);
        reporter->capacityHostLease = std::move(capacityHostLease);
        return apiResult(apiPublishAdapter(*instance, reporter, outReporter->raw_id));
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

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_grow_pair_reporter_capacity(
    ovspatialquery_instance_handle_t instanceHandle, ovspatialquery_pair_reporter_handle_t reporterHandle,
    uint64_t activePairCapacity, uint64_t deltaPairCapacity)
{
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_HANDLE);
    }
    const std::shared_ptr<ApiPairReporterState> reporter =
        apiFindAdapter<ApiPairReporterState>(*instance, reporterHandle.raw_id);
    if (reporter == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_HANDLE);
    }
    std::lock_guard<std::mutex> adapterLock(reporter->adapterMutex);
    if (activePairCapacity < reporter->activePairCapacity || deltaPairCapacity < reporter->deltaPairCapacity ||
        (activePairCapacity == reporter->activePairCapacity && deltaPairCapacity == reporter->deltaPairCapacity))
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }
    if (reporter->activeWaitCount != 0u || reporter->owner == nullptr)
    {
        return apiResult(reporter->owner == nullptr ? OVSPATIALQUERY_API_INVALID_HANDLE : OVSPATIALQUERY_API_BUSY);
    }

    const bool cudaResults    = reporter->resultDevice.device_type == kDLCUDA;
    uint64_t   capacityCharge = 0u;
    if (!cudaResults &&
        !apiReporterCapacityCharge(activePairCapacity, deltaPairCapacity, reporter->resultSlotCount, capacityCharge))
    {
        return apiResult(OVSPATIALQUERY_API_OUT_OF_MEMORY);
    }
    // Both device allocations are charged while the reporter moves its baseline.
    std::shared_ptr<void> deviceLease;
    if (cudaResults)
    {
        const ovspatialquery_api_status_t reserved =
            apiReserveReporterDevice(instanceHandle, reporter->resultDevice, activePairCapacity, deltaPairCapacity,
                                     reporter->resultSlotCount, deviceLease);
        if (reserved != OVSPATIALQUERY_API_SUCCESS)
        {
            return apiResult(reserved);
        }
    }
    std::shared_ptr<ovspatialquery::detail::HostLease> capacityHostLease;
    const ovspatialquery_api_status_t leaseStatus = reserveSharedHostLease(instance, capacityCharge, capacityHostLease);
    if (leaseStatus != OVSPATIALQUERY_API_SUCCESS)
    {
        return apiResult(leaseStatus);
    }

    try
    {
        ApiPairReporterState replacement{};
        replacement.deltaPairCapacity = deltaPairCapacity;
        replacement.resultDevice      = reporter->resultDevice;
        apiPrepareReporterStorage(replacement, reporter->resultSlotCount);
        const ovspatialquery_api_status_t status = reporter->owner->growCapacity(activePairCapacity, deltaPairCapacity);
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return apiResult(status);
        }
        reporter->projections.swap(replacement.projections);
        reporter->activePairCapacity = activePairCapacity;
        reporter->deltaPairCapacity  = deltaPairCapacity;
        reporter->capacityHostLease  = std::move(capacityHostLease);
        if (cudaResults)
        {
            reporter->deviceLease = std::move(deviceLease);
        }
        return apiResult(OVSPATIALQUERY_API_SUCCESS);
    }
    catch (const std::bad_alloc&)
    {
        return apiResult(OVSPATIALQUERY_API_OUT_OF_MEMORY);
    }
    catch (const std::length_error&)
    {
        return apiResult(OVSPATIALQUERY_API_OUT_OF_MEMORY);
    }
    catch (...)
    {
        return apiResult(OVSPATIALQUERY_API_INTERNAL);
    }
}

namespace
{

ovspatialquery_enqueue_result_t apiAdvancePairReporter(ovspatialquery_instance_handle_t              instanceHandle,
                                                       ovspatialquery_pair_reporter_handle_t         reporterHandle,
                                                       const ovspatialquery_pair_reporter_advance_t* advance,
                                                       ovspatialquery_pair_reporter_reservation_t*   outReservation)
{
    ovspatialquery_enqueue_result_t rejected{OVSPATIALQUERY_API_INVALID_ARGUMENT, OVSPATIALQUERY_INVALID_OP};
    if (advance == nullptr || outReservation == nullptr)
    {
        return rejected;
    }
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        rejected.status = OVSPATIALQUERY_API_INVALID_HANDLE;
        return rejected;
    }
    const std::shared_ptr<ApiPairReporterState> reporter =
        apiFindAdapter<ApiPairReporterState>(*instance, reporterHandle.raw_id);
    if (reporter == nullptr)
    {
        rejected.status = OVSPATIALQUERY_API_INVALID_HANDLE;
        return rejected;
    }
    bool                                        firstValid  = false;
    bool                                        secondValid = false;
    ovspatialquery::detail::PairReporterAdvance request{};
    request.first     = apiSelection(advance->first_ordinal, firstValid);
    request.second    = apiSelection(advance->second_ordinal, secondValid);
    request.hasStream = advance->cuda_sync.has_stream == 1u;
    request.stream    = advance->cuda_sync.stream;
    request.waitEvent = advance->cuda_sync.wait_event;
    // A CUDA reporter's advance runs on the device: it may take the caller's
    // stream and select admitted publications (ADR-0022). A CPU reporter needs
    // completed publications and no stream.
    const bool cudaReporter = reporter->resultDevice.device_type == kDLCUDA;
    const bool admitted     = request.first.kind == ovspatialquery::detail::TargetSelectionKind::admittedExact ||
                          request.second.kind == ovspatialquery::detail::TargetSelectionKind::admittedExact;
    if (!firstValid || !secondValid || advance->cuda_sync.has_stream > 1u ||
        (!cudaReporter && (admitted || advance->cuda_sync.has_stream != 0u || advance->cuda_sync.wait_event != 0u)))
    {
        return rejected;
    }
    // Nothing is queued on a stream that is capturing a CUDA graph (ADR-0060).
    if (request.hasStream &&
        ovspatialquery::detail::cudaStreamIsCapturing(reporter->resultDevice.device_id, request.stream))
    {
        rejected.status = OVSPATIALQUERY_API_NOT_SUPPORTED;
        return rejected;
    }
    try
    {
        std::lock_guard<std::mutex> adapterLock(reporter->adapterMutex);
        if (reporter->owner == nullptr)
        {
            rejected.status = OVSPATIALQUERY_API_INVALID_HANDLE;
            return rejected;
        }
        const ovspatialquery::detail::PairReporterAdmission admission = reporter->owner->enqueue(request);
        rejected.status                                               = admission.outcome;
        if (rejected.status != OVSPATIALQUERY_API_SUCCESS)
        {
            return rejected;
        }
        // Written only when the view is available.
        ovspatialquery::detail::PairReporterResultStorageView early{};
        (void)reporter->owner->reservedStorageView(admission.operation, early);
        const std::shared_ptr<ApiReporterProjection> projection =
            apiActivateReporterProjection(*reporter, early.resultSlot);
        if (projection == nullptr)
        {
            rejected.status = OVSPATIALQUERY_API_INTERNAL;
            return rejected;
        }
        ApiOperationOwner owner{};
        owner.reporter   = reporter;
        owner.projection = projection;
        apiSetOperationOwner(*instance, admission.operation.value, std::move(owner));
        ovspatialquery_pair_reporter_reservation_t reservation{};
        apiReporterReservation(*reporter, *projection, early, reservation);
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

} // namespace

extern "C" OVSPATIALQUERY_API ovspatialquery_enqueue_result_t ovspatialquery_enqueue_pair_reporter_advance(
    ovspatialquery_instance_handle_t instanceHandle, ovspatialquery_pair_reporter_handle_t reporterHandle,
    const ovspatialquery_pair_reporter_advance_t* advance, ovspatialquery_pair_reporter_reservation_t* outReservation)
{
    const ovspatialquery_enqueue_result_t result =
        apiAdvancePairReporter(instanceHandle, reporterHandle, advance, outReservation);
    return sourcePublicEnqueueResult(result.status, result.op_id, nullptr);
}

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t
ovspatialquery_fetch_pair_reporter(ovspatialquery_instance_handle_t instanceHandle, ovspatialquery_op_id_t operationId,
                                   ovspatialquery_pair_reporter_result_t* outResult)
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
    if (owner.reporter == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_HANDLE);
    }
    ovspatialquery_pair_reporter_result_t result{};
    {
        std::lock_guard<std::mutex>       adapterLock(owner.reporter->adapterMutex);
        const ovspatialquery_api_status_t status = apiMaterializeReporterLocked(owner, operationId);
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return apiResult(status);
        }
        // Device views handed to the caller retire only after their consumers.
        if (owner.reporter->resultDevice.device_type == kDLCUDA && owner.reporter->owner != nullptr &&
            owner.reporter->owner->noteViewsReturned({operationId}) != OVSPATIALQUERY_API_SUCCESS)
        {
            return apiResult(OVSPATIALQUERY_API_INTERNAL);
        }
        const std::shared_ptr<ApiReporterProjection> projection =
            std::static_pointer_cast<ApiReporterProjection>(owner.projection);
        std::lock_guard<std::mutex> projectionLock(projection->mutex);
#if defined(OVSPATIALQUERY_ENABLE_INSTANCE_OPERATION_RUNTIME_TEST_HOOK)
        apiFetchProjectionPausePointForTest(1u);
#endif
        apiReporterResult(*projection, result);
    }
    *outResult = result;
    return apiResult(OVSPATIALQUERY_API_SUCCESS);
}

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_destroy_pair_reporter(
    ovspatialquery_instance_handle_t instanceHandle, ovspatialquery_pair_reporter_handle_t reporterHandle)
{
    return apiDestroyAdapter<ApiPairReporterState>(instanceHandle, reporterHandle.raw_id);
}
