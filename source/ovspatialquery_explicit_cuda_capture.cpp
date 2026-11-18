/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_explicit_cuda_capture.h"

#include "ovspatialquery_cuda_driver.h"
#include "ovspatialquery_input_layout.h"
#include "ovspatialquery_instance_operation_runtime.h"
#include "ovspatialquery_instance_state.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <new>
#include <optional>
#include <utility>
#include <vector>

namespace ovspatialquery
{
namespace detail
{
namespace
{

#if defined(OVSPATIALQUERY_HAS_CUDA_BACKEND)

struct CapturedColumn
{
    uintptr_t source        = 0u;
    uintptr_t captured      = 0u;
    size_t    capturedBytes = 0u; // the staging buffer's size, at least byteCount
    uint64_t  byteCount     = 0u;
    uint64_t  rowCount      = 0u;
    uint8_t   bits          = 0u;
    uint32_t  ownership     = 0u;
    // Copied before any read: a COPIED_ON_SUBMISSION column, or one whose
    // layout the preparation cannot read in place (ADR-0053).
    bool                 copied = false;
    InputColumnLayout    layout{};
    std::vector<uint8_t> hostBytes;
};

bool addBytes(uint64_t value, uint64_t& total) noexcept
{
    if (value > UINT64_MAX - total)
    {
        total = UINT64_MAX;
        return false;
    }
    total += value;
    return true;
}

ovspatialquery_api_status_t validateColumn(const ovspatialquery_input_array_t* source, const char* column,
                                           DLDevice device, uint8_t code, uint8_t bits, int32_t rank, int64_t trailing,
                                           bool optional, CapturedColumn& out) noexcept
{
    if (source == nullptr)
    {
        return optional ? OVSPATIALQUERY_API_SUCCESS : OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    const DLTensor& tensor = source->tensor;
    if (source->ownership > 1u || tensor.device.device_type != kDLCUDA || tensor.device.device_id != device.device_id ||
        tensor.dtype.code != code || tensor.dtype.bits != bits || tensor.dtype.lanes != 1u || tensor.ndim != rank)
    {
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    if (tensor.shape == nullptr || tensor.shape[0] < 0 || (rank == 2 && tensor.shape[1] != trailing))
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    InputColumnLayout                 layout;
    const ovspatialquery_api_status_t layoutStatus = readInputColumnLayout(tensor, column, layout);
    if (layoutStatus != OVSPATIALQUERY_API_SUCCESS)
    {
        return layoutStatus;
    }
    const uint64_t byteCount = layout.compactBytes();
    if ((byteCount == 0u) != (tensor.data == nullptr) || (tensor.data == nullptr && tensor.byte_offset != 0u))
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    out.source    = reinterpret_cast<uintptr_t>(tensor.data) + static_cast<uintptr_t>(tensor.byte_offset);
    out.byteCount = byteCount;
    out.rowCount  = static_cast<uint64_t>(tensor.shape[0]);
    out.bits      = bits;
    out.ownership = source->ownership;
    out.layout    = layout;
    out.copied    = source->ownership == 1u || !layout.compact;
    return OVSPATIALQUERY_API_SUCCESS;
}

template <typename Value>
Value readValue(const CapturedColumn& column, size_t index) noexcept
{
    Value value{};
    std::memcpy(&value, column.hostBytes.data() + index * sizeof(Value), sizeof(Value));
    return value;
}

double readReal(const CapturedColumn& column, size_t index) noexcept
{
    return column.bits == 32u ? static_cast<double>(readValue<float>(column, index)) : readValue<double>(column, index);
}

using CuResult                               = int;
using CuContext                              = void*;
using CuStream                               = void*;
using CuEvent                                = void*;
using CuDevicePtr                            = unsigned long long;
constexpr CuResult     kCuSuccess            = 0;
constexpr unsigned int kCuStreamNonBlocking  = 1u;
constexpr unsigned int kCuEventDisableTiming = 2u;

class ExplicitCudaCapture final : public CudaPopulationCapture
{
  public:
    ~ExplicitCudaCapture() override
    {
        releaseCaptureState();
        if (retainedPrimary)
        {
            (void)driver.primaryRelease(primaryDevice);
        }
    }

    void releaseCaptureState() noexcept
    {
        if (driver.module == nullptr || cudaContext == nullptr)
        {
            return;
        }
        CuContext previous = nullptr;
        if (driver.contextGetCurrent(&previous) != kCuSuccess || driver.contextSetCurrent(cudaContext) != kCuSuccess)
        {
            return;
        }
        if (completion != nullptr)
        {
            (void)driver.eventSynchronize(completion);
        }
        // A kept buffer must be idle: every copy queued on the caller's stream has
        // finished, which `completion` shows once recorded, every read queued on
        // the capture stream has finished, and device preparation, which reads
        // on its own stream, has finished its version.
        const bool copiesDone = captureStream == nullptr || driver.streamSynchronize(captureStream) == kCuSuccess;
        const CudaCompletionState prepared = version != nullptr ? version->resultState() : CudaCompletionState::invalid;
        const bool                idle =
            callerCopiesIdle && copiesDone &&
            (!onDevice || prepared == CudaCompletionState::complete || prepared == CudaCompletionState::error);
        // Buffers kept for the next capture outlive this capture's retain of the
        // primary context, which may be the last one; without the pool's own
        // retain they are freed instead.
        const bool keep = idle && staging != nullptr && (!retainedPrimary || staging->keepPrimary(primaryDevice));
        for (CapturedColumn& column : columns)
        {
            if (column.captured == 0u)
            {
                continue;
            }
            if (keep)
            {
                staging->give(driver, cudaContext, static_cast<CuDevicePtr>(column.captured), column.capturedBytes);
            }
            else
            {
                (void)driver.memFree(static_cast<CuDevicePtr>(column.captured));
            }
        }
        if (completion != nullptr)
        {
            (void)driver.eventDestroy(completion);
        }
        if (captureStream != nullptr)
        {
            (void)driver.streamDestroy(captureStream);
        }
        (void)driver.contextSetCurrent(previous);
    }

    // Device preparation (ADR-0024): `boxes` is the population's fixed
    // membership for an update, or the membership this complete set defines
    // once `captureAdmissionInputs` has read its identity columns.
    bool                                 onDevice     = false;
    bool                                 hostReadable = false; // ON_DEVICE_HOST_READABLE
    bool                                 posesOnly    = false;
    std::shared_ptr<DeviceBoxPopulation> boxes;
    ExplicitCudaCaptureContext           captureContext;
    std::shared_ptr<CudaStagingPool>     staging; // the population's, when it has device state
    DLDevice                             device() const override { return captureDevice; }

    uint64_t deviceByteCharge() const override
    {
        if (onDevice)
        {
            // One pose, bounds and bits table per version, sized to the slot
            // capacity; a complete set also uploads the membership it defines.
            const uint64_t capacity = deviceSlotCapacity();
            if (capacity > UINT64_MAX / 512u)
                return UINT64_MAX;
            return DeviceBoxPopulation::versionBytes(capacity) +
                   (completeSet ? DeviceBoxPopulation::membershipBytes(capacity) : 0u);
        }
        if (!completeSet)
        {
            return 0u;
        }
        // The version prepared on the host uploads each object's ID,
        // participation bits and bounds.
        constexpr uint64_t kBytesPerObject =
            sizeof(uint64_t) + sizeof(ovspatialquery_participation_bits_t) + sizeof(double) * 6u;
        if (maximumOutputCount > UINT64_MAX / kBytesPerObject)
        {
            return UINT64_MAX;
        }
        return maximumOutputCount * kBytesPerObject;
    }

    uint64_t deviceStagingByteCharge() const override
    {
        uint64_t total = onDevice ? (2u * deviceSlotCapacity() + 1u) * sizeof(uint32_t) : 0u;
        for (const CapturedColumn& column : columns)
        {
            if (column.copied && !addBytes(column.byteCount, total))
            {
                return UINT64_MAX;
            }
        }
        return total;
    }

    uint64_t retainedOperationByteCharge() const override
    {
        if (onDevice)
            return 0u;
        const uint64_t rowCount = columns.empty() ? 0u : columns[0].rowCount;
        if (rowCount > UINT64_MAX / sizeof(std::shared_ptr<const void>))
        {
            return UINT64_MAX;
        }
        return rowCount * sizeof(std::shared_ptr<const void>);
    }

    uint64_t hostStagingByteCharge() const override
    {
        uint64_t total = static_cast<uint64_t>(sizeof(*this));
        if (columns.capacity() > static_cast<size_t>((UINT64_MAX - total) / sizeof(CapturedColumn)))
        {
            return UINT64_MAX;
        }
        total += static_cast<uint64_t>(columns.capacity()) * sizeof(CapturedColumn);
        if (onDevice)
            return total + 4096u;
        for (const CapturedColumn& column : columns)
        {
            if (!addBytes(column.byteCount, total))
            {
                return UINT64_MAX;
            }
        }
        constexpr uint64_t treeNodeAllowance      = 4u * sizeof(void*) + 32u;
        constexpr uint64_t exactInstanceAllowance = 1024u;
        constexpr uint64_t perOutputCandidate     = 3u * sizeof(PopulationObject) + sizeof(ExplicitInstanceInput) +
                                                2u * treeNodeAllowance + exactInstanceAllowance;
        if (maximumOutputCount > (UINT64_MAX - total) / perOutputCandidate)
        {
            return UINT64_MAX;
        }
        total += maximumOutputCount * perOutputCandidate;
        if (!completeSet)
        {
            const uint64_t removedCount = columns.back().rowCount;
            if (removedCount > (UINT64_MAX - total) / treeNodeAllowance)
            {
                return UINT64_MAX;
            }
            total += removedCount * treeNodeAllowance;
        }
        return total;
    }

    ovspatialquery_api_status_t prepareAdmission(const GenerationLease& base) override
    {
        if (!completeSet && (!base.initialized() || base.ordinal() != baseOrdinal))
        {
            return OVSPATIALQUERY_API_ORDINAL_MISMATCH;
        }
        if (onDevice)
        {
            const uint64_t rowCount = columns[0].rowCount;
            maximumOutputCount      = deviceSlotCapacity();
            if (rowCount > maximumOutputCount)
            {
                return OVSPATIALQUERY_API_INVALID_ARGUMENT;
            }
            deviceBase = base; // the head, or the newest admitted update (ADR-0021)
            return OVSPATIALQUERY_API_SUCCESS;
        }
        const uint64_t baseObjectCount = completeSet ? 0u : static_cast<uint64_t>(base.objects().size());
        if (columns[0].rowCount > UINT64_MAX - baseObjectCount)
        {
            return OVSPATIALQUERY_API_INVALID_ARGUMENT;
        }
        maximumOutputCount = baseObjectCount + columns[0].rowCount;
        return maximumOutputCount < static_cast<uint64_t>(INT64_MAX) ? OVSPATIALQUERY_API_SUCCESS
                                                                     : OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }

    ovspatialquery_api_status_t captureAdmissionInputs() override
    {
        if (!driver.open())
        {
            return OVSPATIALQUERY_API_NOT_SUPPORTED;
        }
        // The copies run on the caller's stream, after its event when it
        // supplies one, so the caller orders a later overwrite of its input on
        // that stream; the capture stream's later reads follow them through
        // `completion`.
        const CuStream callerStream  = reinterpret_cast<CuStream>(stream);
        int            currentDevice = -1;
        CuContext      current       = nullptr;
        if (driver.contextGetCurrent(&current) != kCuSuccess)
        {
            return OVSPATIALQUERY_API_NOT_SUPPORTED;
        }
        // The caller's stream belongs to the device's primary context, where the
        // library runs its own work (ovspatialquery_cuda_sync_t), so a thread with
        // no current context, or one current on another device, captures there
        // as plans and launches do. A device-prepared population lives in that
        // context too.
        if (current == nullptr || driver.contextGetDevice(&currentDevice) != kCuSuccess ||
            currentDevice != captureDevice.device_id || onDevice)
        {
            if (driver.deviceGet(&primaryDevice, captureDevice.device_id) != kCuSuccess ||
                driver.primaryRetain(&cudaContext, primaryDevice) != kCuSuccess)
            {
                return OVSPATIALQUERY_API_NOT_SUPPORTED;
            }
            retainedPrimary = true;
        }
        else
        {
            cudaContext = current;
        }
        CudaCurrentContext scope(driver, cudaContext, deviceLost);
        if (scope.status() != kCuSuccess)
        {
            return OVSPATIALQUERY_API_NOT_SUPPORTED;
        }
        // Refused before anything is queued on the caller's stream, which a
        // rejected publication must leave untouched.
        if (onDevice && boxes && !boxes->compatibleContext(cudaContext))
            return OVSPATIALQUERY_API_NOT_SUPPORTED;
        if (driver.streamCreate(&captureStream, kCuStreamNonBlocking) != kCuSuccess ||
            driver.eventCreate(&completion, kCuEventDisableTiming) != kCuSuccess ||
            (waitEvent != 0u &&
             driver.streamWaitEvent(callerStream, reinterpret_cast<CuEvent>(waitEvent), 0u) != kCuSuccess))
        {
            return OVSPATIALQUERY_API_NOT_SUPPORTED;
        }
        ovspatialquery_api_status_t copied = OVSPATIALQUERY_API_SUCCESS;
        for (CapturedColumn& column : columns)
        {
            if (!column.copied || column.byteCount == 0u || column.source == 0u)
            {
                continue;
            }
            CuDevicePtr  destination = 0u;
            const size_t bytes       = static_cast<size_t>(column.byteCount);
            column.capturedBytes     = bytes;
            const CuResult allocated =
                staging != nullptr ? staging->take(driver, cudaContext, bytes, destination, column.capturedBytes)
                                   : driver.memAlloc(&destination, bytes);
            if (allocated != kCuSuccess)
            {
                copied = OVSPATIALQUERY_API_OUT_OF_MEMORY;
                break;
            }
            column.captured = static_cast<uintptr_t>(destination);
            if (copyInputColumnAsync(driver, static_cast<CuDevicePtr>(column.source), column.layout, destination,
                                     callerStream) != kCuSuccess)
            {
                copied = OVSPATIALQUERY_API_INTERNAL;
                break;
            }
        }
        // Recorded after a failed column too, so teardown waits for the copies
        // already queued before it reuses their buffers.
        if (driver.eventRecord(completion, callerStream) != kCuSuccess)
        {
            // Teardown cannot see these copies without the event, and may not
            // use the caller's stream after enqueue: wait for them now.
            callerCopiesIdle = driver.streamSynchronize(callerStream) == kCuSuccess;
            return OVSPATIALQUERY_API_INTERNAL;
        }
        if (driver.streamWaitEvent(captureStream, completion, 0u) != kCuSuccess)
        {
            return OVSPATIALQUERY_API_INTERNAL;
        }
        if (copied != OVSPATIALQUERY_API_SUCCESS)
        {
            return copied;
        }
        if (onDevice)
        {
            // Only a complete set reads anything back here: its membership.
            // An update enqueues the version's kernels after the copies and
            // returns, so the host waits for neither the producer nor the
            // preparation (ADR-0021).
            if (completeSet || !posesOnly)
            {
                // A complete set defines the membership; complete rows may change
                // it (additions, removals). Both read identity columns back once,
                // at event rate; POSES never does.
                if (driver.eventSynchronize(completion) != kCuSuccess)
                {
                    return OVSPATIALQUERY_API_INTERNAL;
                }
                const ovspatialquery_api_status_t membership =
                    completeSet ? defineDeviceMembership() : changeDeviceMembership();
                if (membership != OVSPATIALQUERY_API_SUCCESS)
                {
                    return membership;
                }
            }
            return launchDeviceVersion();
        }
        if (driver.eventSynchronize(completion) != kCuSuccess)
        {
            return OVSPATIALQUERY_API_INTERNAL;
        }
        if (posesOnly)
        {
            return OVSPATIALQUERY_API_SUCCESS;
        }
        for (size_t index : std::array<size_t, 2u>{1u, 2u})
        {
            const ovspatialquery_api_status_t status = downloadColumn(columns[index]);
            if (status != OVSPATIALQUERY_API_SUCCESS)
            {
                return status;
            }
        }
        if (driver.eventRecord(completion, captureStream) != kCuSuccess ||
            driver.eventSynchronize(completion) != kCuSuccess)
        {
            return OVSPATIALQUERY_API_INTERNAL;
        }
        std::vector<GeometryReference> references;
        try
        {
            references.reserve(static_cast<size_t>(columns[0].rowCount));
            for (size_t row = 0u; row < columns[0].rowCount; ++row)
            {
                const int32_t  family = readValue<int32_t>(columns[1], row);
                const uint64_t value  = readValue<uint64_t>(columns[2], row);
                if (family < kGeometryTypeFirstBuiltin || family > kGeometryTypeLast || value == 0u)
                {
                    return family == OVSPATIALQUERY_GEOMETRY_USER ? OVSPATIALQUERY_API_NOT_SUPPORTED
                                                                  : OVSPATIALQUERY_API_INVALID_ARGUMENT;
                }
                references.push_back({static_cast<ExactGeometryFamily>(family - kGeometryTypeFirstBuiltin), value});
            }
        }
        catch (const std::bad_alloc&)
        {
            return OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
        return registry->retainDefinitions(references, definitionOwners);
    }

    bool materializesHostObjects() const override { return !onDevice; }

    bool requiresCpuQueryBackend() const override { return !onDevice; }

    void publicationCommitted() noexcept override
    {
        if (onDevice && completeSet && captureContext.deviceState != nullptr)
        {
            captureContext.deviceState->commit(boxes);
        }
        if (const std::shared_ptr<DevicePublicationOutcome> outcome = DeviceBoxPopulation::outcome(version))
        {
            outcome->finish(DevicePublicationOutcome::State::published);
        }
    }

    void publicationAbandoned() noexcept override
    {
        if (const std::shared_ptr<DevicePublicationOutcome> outcome = DeviceBoxPopulation::outcome(version))
        {
            outcome->finish(DevicePublicationOutcome::State::abandoned);
        }
    }

    std::shared_ptr<CudaPopulationVersion> admittedVersion() const override { return version; }

    void ringWhenReady(const std::shared_ptr<DeviceSignal>& signal) noexcept override
    {
        if (onDevice && boxes != nullptr && version != nullptr)
            boxes->ringWhenReady(version, signal);
    }

    // Enqueues the version's device work on the population's stream, ordered
    // after this capture's copies and the producer's event.
    ovspatialquery_api_status_t launchDeviceVersion()
    {
        std::array<DeviceColumn, 7> inputs{};
        for (size_t i = 0; i < 5; ++i)
            inputs[i] = {columns[i].captured ? columns[i].captured : columns[i].source, columns[i].bits};
        size_t next = 5;
        if (hasScales)
        {
            auto& c   = columns[next++];
            inputs[5] = {c.captured ? c.captured : c.source, c.bits};
        }
        if (hasParticipation)
        {
            auto& c   = columns[next];
            inputs[6] = {c.captured ? c.captured : c.source, c.bits};
        }
        return boxes->launch(targetOrdinal, deviceBase, columns[0].rowCount, inputs, posesOnly, false, hostReadable,
                             reinterpret_cast<uintptr_t>(completion), removedSlots, version);
    }

    uint64_t deviceSlotCapacity() const noexcept
    {
        if (boxes)
            return boxes->capacity();
        return slotCapacity > columns[0].rowCount ? slotCapacity : columns[0].rowCount;
    }

    // Reads this update's identity and removed columns back and resolves them
    // against the membership: known IDs keep their slots, unknown IDs are
    // added into free slots, removed IDs are retired. Applied at admission, so
    // the change stands even if the publication later fails validation.
    ovspatialquery_api_status_t changeDeviceMembership()
    {
        for (size_t index : std::array<size_t, 4u>{0u, 1u, 2u, columns.size() - 1u})
        {
            const ovspatialquery_api_status_t status = downloadColumn(columns[index]);
            if (status != OVSPATIALQUERY_API_SUCCESS)
                return status;
        }
        if (driver.eventRecord(completion, captureStream) != kCuSuccess ||
            driver.eventSynchronize(completion) != kCuSuccess)
        {
            return OVSPATIALQUERY_API_INTERNAL;
        }
        std::vector<ExplicitInstanceInput> rows;
        ovspatialquery_api_status_t        status = readBindingRows(rows);
        if (status != OVSPATIALQUERY_API_SUCCESS)
            return status;
        const CapturedColumn& removedColumn = columns.back();
        std::vector<uint64_t> removedIds;
        try
        {
            removedIds.reserve(static_cast<size_t>(removedColumn.rowCount));
            for (size_t row = 0u; row < removedColumn.rowCount; ++row)
                removedIds.push_back(readValue<uint64_t>(removedColumn, row));
        }
        catch (const std::bad_alloc&)
        {
            return OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
        DeviceMembershipChange change;
        status = boxes->planMembershipChange(rows, removedIds, change);
        if (status != OVSPATIALQUERY_API_SUCCESS)
            return status;
        std::vector<std::shared_ptr<const void>> definitions;
        if (!change.added.empty())
        {
            std::vector<GeometryReference> references;
            references.reserve(change.added.size());
            for (const ExplicitInstanceInput& added : change.added)
                references.push_back(added.geometry);
            status = registry->describeDeviceShapes(references, change.addedShapes, definitions);
            if (status != OVSPATIALQUERY_API_SUCCESS)
                return status;
        }
        // Hulls and meshes the population does not hold yet take device bytes
        // of their own, charged here at event rate.
        std::shared_ptr<void> poolLease;
        const uint64_t        poolBytes = boxes->additionalDefinitionBytes(change.addedShapes);
        if (poolBytes != 0u)
        {
            status = reserveInstanceDeviceMemory(captureContext.instance, captureDevice, poolBytes, poolLease);
            if (status != OVSPATIALQUERY_API_SUCCESS)
                return status;
        }
        status = boxes->applyMembershipChange(change, std::move(definitions), std::move(poolLease));
        if (status != OVSPATIALQUERY_API_SUCCESS)
            return status;
        removedSlots = std::move(change.removedSlots);
        return OVSPATIALQUERY_API_SUCCESS;
    }

    // A complete set prepared on the device defines the population's membership
    // from its identity and binding columns, read once here;
    // the poses stay on the device for the kernel.
    ovspatialquery_api_status_t defineDeviceMembership()
    {
        const InstanceReference instanceReference = instanceState(captureContext.instance);
        InstanceState* const    instance          = instanceReference;
        if (instance == nullptr)
        {
            return OVSPATIALQUERY_API_INVALID_HANDLE;
        }
        const std::array<size_t, 3u> identityColumns{0u, 1u, 2u};
        for (size_t index : identityColumns)
        {
            const ovspatialquery_api_status_t status = downloadColumn(columns[index]);
            if (status != OVSPATIALQUERY_API_SUCCESS)
            {
                return status;
            }
        }
        if (driver.eventRecord(completion, captureStream) != kCuSuccess ||
            driver.eventSynchronize(completion) != kCuSuccess)
        {
            return OVSPATIALQUERY_API_INTERNAL;
        }
        const uint64_t count    = columns[0].rowCount;
        const uint64_t capacity = deviceSlotCapacity();
        if (capacity > UINT32_MAX || capacity > (UINT64_MAX - 8192u) / 4096u)
        {
            return OVSPATIALQUERY_API_INVALID_ARGUMENT;
        }
        std::vector<ExplicitInstanceInput> rows;
        ovspatialquery_api_status_t        status = readBindingRows(rows);
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return status;
        }
        std::vector<GeometryReference> references;
        try
        {
            references.reserve(static_cast<size_t>(count));
            std::sort(rows.begin(), rows.end(),
                      [](const ExplicitInstanceInput& left, const ExplicitInstanceInput& right)
                      { return left.objectValue < right.objectValue; });
            for (size_t row = 0u; row < rows.size(); ++row)
            {
                if (rows[row].objectValue == 0u || (row != 0u && rows[row - 1u].objectValue == rows[row].objectValue))
                {
                    return OVSPATIALQUERY_API_INVALID_ARGUMENT;
                }
                references.push_back(rows[row].geometry);
            }
        }
        catch (const std::bad_alloc&)
        {
            return OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
        std::vector<std::shared_ptr<const void>> definitions;
        std::vector<DeviceShapeDescription>      shapes;
        status = registry->describeDeviceShapes(references, shapes, definitions);
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return status;
        }
        // The membership tables outlive every version, so they take their own
        // leases instead of the version's published-data charge.
        std::shared_ptr<HostLease>        hostLease;
        const ovspatialquery_api_status_t reserved =
            reserveSharedHostLease(instance, 8192u + capacity * 128u, hostLease);
        if (reserved != OVSPATIALQUERY_API_SUCCESS)
        {
            return reserved;
        }
        std::shared_ptr<void> deviceLease;
        status = reserveInstanceDeviceMemory(captureContext.instance, captureDevice,
                                             DeviceBoxPopulation::membershipBytes(capacity), deviceLease);
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return status;
        }
        // Hulls and meshes keep their local vertices or bounds on the device.
        std::shared_ptr<void> poolLease;
        const uint64_t        poolBytes = DeviceBoxPopulation::definitionBytes(shapes);
        if (poolBytes != 0u)
        {
            status = reserveInstanceDeviceMemory(captureContext.instance, captureDevice, poolBytes, poolLease);
            if (status != OVSPATIALQUERY_API_SUCCESS)
            {
                return status;
            }
        }
        std::shared_ptr<DeviceBoxPopulation> defined = std::make_shared<DeviceBoxPopulation>();
        status = defined->initialize(captureDevice, capacity, rows, shapes, std::move(definitions),
                                     std::move(hostLease), std::move(deviceLease), registry, std::move(poolLease));
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return status;
        }
        boxes = std::move(defined);
        return OVSPATIALQUERY_API_SUCCESS;
    }

    // The identity and binding rows read back from columns 0 to 2: a USER row
    // is NOT_SUPPORTED on the device, any other type outside the exact
    // families INVALID_ARGUMENT.
    ovspatialquery_api_status_t readBindingRows(std::vector<ExplicitInstanceInput>& rows) const
    {
        try
        {
            rows.reserve(static_cast<size_t>(columns[0].rowCount));
            for (size_t row = 0u; row < columns[0].rowCount; ++row)
            {
                ExplicitInstanceInput input;
                input.objectValue  = readValue<uint64_t>(columns[0], row);
                const int32_t type = readValue<int32_t>(columns[1], row);
                if (type < kGeometryTypeFirstBuiltin || type > kGeometryTypeLast)
                {
                    return type == OVSPATIALQUERY_GEOMETRY_USER ? OVSPATIALQUERY_API_NOT_SUPPORTED
                                                                : OVSPATIALQUERY_API_INVALID_ARGUMENT;
                }
                input.geometry = {static_cast<ExactGeometryFamily>(type - kGeometryTypeFirstBuiltin),
                                  readValue<uint64_t>(columns[2], row)};
                rows.push_back(input);
            }
            return OVSPATIALQUERY_API_SUCCESS;
        }
        catch (const std::bad_alloc&)
        {
            return OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
    }

    ovspatialquery_api_status_t downloadColumn(CapturedColumn& column)
    {
        try
        {
            column.hostBytes.resize(static_cast<size_t>(column.byteCount));
        }
        catch (const std::bad_alloc&)
        {
            return OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
        if (column.byteCount != 0u &&
            driver.copyDtoHAsync(column.hostBytes.data(),
                                 static_cast<CuDevicePtr>(column.captured != 0u ? column.captured : column.source),
                                 static_cast<size_t>(column.byteCount), captureStream) != kCuSuccess)
        {
            return OVSPATIALQUERY_API_INTERNAL;
        }
        return OVSPATIALQUERY_API_SUCCESS;
    }

    ovspatialquery_api_status_t materializeHostObjects(const GenerationLease& base, ObjectTable& outObjects,
                                                       std::optional<VersionChangeRecord>& outChanges,
                                                       std::optional<uint64_t>&            outGeometryCharge) override
    {
        CuContext previous = nullptr;
        if (cudaContext == nullptr || driver.contextGetCurrent(&previous) != kCuSuccess ||
            driver.contextSetCurrent(cudaContext) != kCuSuccess)
        {
            return OVSPATIALQUERY_API_INTERNAL;
        }
        ovspatialquery_api_status_t status = OVSPATIALQUERY_API_SUCCESS;
        for (CapturedColumn& column : columns)
        {
            status = downloadColumn(column); // nothing to copy for a binding column a poses-only update lacks
            if (status != OVSPATIALQUERY_API_SUCCESS)
            {
                break;
            }
        }
        if (status == OVSPATIALQUERY_API_SUCCESS && (driver.eventRecord(completion, captureStream) != kCuSuccess ||
                                                     driver.eventSynchronize(completion) != kCuSuccess))
        {
            status = OVSPATIALQUERY_API_INTERNAL;
        }
        (void)driver.contextSetCurrent(previous);
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return status;
        }
        std::vector<ExplicitInstanceInput> instances;
        status = decodeInstances(instances);
        if (status == OVSPATIALQUERY_API_SUCCESS && posesOnly)
        {
            status = completePoseRows(base, hasScales, hasParticipation, instances);
        }
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return status;
        }
        if (completeSet)
        {
            std::vector<PopulationObject> objects;
            status = registry->prepareObjects(instances, objects);
            if (status == OVSPATIALQUERY_API_SUCCESS)
            {
                outObjects = ObjectTable(std::move(objects));
            }
            return status;
        }
        // The same preparation as a CPU explicit update: a few changed rows
        // share the base's storage and the version records what changed.
        const CapturedColumn& removedColumn = columns.back();
        std::vector<uint64_t> removed;
        removed.reserve(static_cast<size_t>(removedColumn.rowCount));
        for (size_t row = 0u; row < removedColumn.rowCount; ++row)
        {
            removed.push_back(readValue<uint64_t>(removedColumn, row));
        }
        ExplicitPreparedUpdate prepared;
        status = registry->prepareUpdate(base, baseOrdinal, instances, removed, prepared);
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return status;
        }
        outGeometryCharge = prepared.exactGeometryCharge();
        outChanges        = prepared.releaseChanges();
        outObjects        = prepared.releaseObjects();
        return OVSPATIALQUERY_API_SUCCESS;
    }

    std::vector<std::shared_ptr<const void>> releaseDefinitionOwners() noexcept override
    {
        return std::move(definitionOwners);
    }

    // Only a complete set uploads a device version, which a CUDA reporter
    // borrows. An update uploads none: a reporter derives its copy from the
    // version's change record with the changed rows alone (ADR-0019), where
    // a whole upload per publication cost milliseconds.
    ovspatialquery_api_status_t execute(ovspatialquery_ordinal_t sourceOrdinal, const GenerationLease& base,
                                        const ObjectTable&                      objects,
                                        std::shared_ptr<CudaPopulationVersion>& outVersion) override
    {
        if (onDevice)
        {
            (void)sourceOrdinal;
            (void)base;
            // Launched at admission; the worker observes the outcome here, or
            // comes back later when the device has not finished (NOT_READY).
            const ovspatialquery_api_status_t status = boxes->complete(version);
            if (status == OVSPATIALQUERY_API_NOT_READY)
                return status;
            deviceBase.reset();
            if (status == OVSPATIALQUERY_API_SUCCESS)
                outVersion = version;
            return status;
        }
        if (!completeSet)
        {
            outVersion.reset();
            return OVSPATIALQUERY_API_SUCCESS;
        }
        return createCudaPopulationVersionFromHostObjects(captureDevice, sourceOrdinal, objects, outVersion);
    }

    ovspatialquery_api_status_t decodeInstances(std::vector<ExplicitInstanceInput>& out) const noexcept
    {
        try
        {
            const size_t                       count = static_cast<size_t>(columns[0].rowCount);
            std::vector<ExplicitInstanceInput> instances;
            instances.reserve(count);
            for (size_t row = 0u; row < count; ++row)
            {
                ExplicitInstanceInput instance;
                instance.objectValue = readValue<uint64_t>(columns[0], row);
                if (!posesOnly)
                {
                    // captureAdmissionInputs checked the family column; a borrowed
                    // column must not change until the operation is terminal.
                    instance.geometry = {static_cast<ExactGeometryFamily>(readValue<int32_t>(columns[1], row) -
                                                                          kGeometryTypeFirstBuiltin),
                                         readValue<uint64_t>(columns[2], row)};
                }
                for (size_t axis = 0u; axis < 3u; ++axis)
                {
                    instance.translation[axis] = readReal(columns[3], row * 3u + axis);
                }
                for (size_t component = 0u; component < 4u; ++component)
                {
                    instance.rotationXyzw[component] = readReal(columns[4], row * 4u + component);
                }
                if (hasScales)
                {
                    instance.uniformScale = readReal(columns[5], row);
                }
                if (hasParticipation)
                {
                    const CapturedColumn& bits = columns[hasScales ? 6u : 5u];
                    instance.participationBits = {
                        {readValue<uint64_t>(bits, row * 2u), readValue<uint64_t>(bits, row * 2u + 1u)}};
                }
                instances.push_back(instance);
            }
            out = std::move(instances);
            return OVSPATIALQUERY_API_SUCCESS;
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

    ovspatialquery::detail::Driver               driver;
    DLDevice                                     captureDevice{kDLCUDA, 0};
    uintptr_t                                    stream             = 0u; // the caller's, used only inside enqueue
    uintptr_t                                    waitEvent          = 0u;
    ovspatialquery_ordinal_t                     baseOrdinal        = 0;
    ovspatialquery_ordinal_t                     targetOrdinal      = 0;
    bool                                         completeSet        = true;
    bool                                         hasScales          = false;
    bool                                         hasParticipation   = false;
    uint64_t                                     maximumOutputCount = 0u;
    std::shared_ptr<ExplicitPreparationRegistry> registry;
    std::vector<std::shared_ptr<const void>>     definitionOwners;
    std::vector<CapturedColumn>                  columns;
    CuContext                                    cudaContext     = nullptr;
    CuDevice                                     primaryDevice   = 0;
    bool                                         retainedPrimary = false; // cudaContext is a primary retain
    std::atomic<bool>                            deviceLost{false};
    CuStream                                     captureStream    = nullptr;
    CuEvent                                      completion       = nullptr;
    bool                                         callerCopiesIdle = true; // false: copies may still be in flight
    // Device preparation (ADR-0024/0021): the membership, the version this one
    // is built on, and the version launched at admission.
    GenerationLease                        deviceBase;
    std::shared_ptr<CudaPopulationVersion> version;
    uint64_t                               slotCapacity = 0u; // declared by a complete set
    std::vector<uint32_t>                  removedSlots;      // retired by this update's change
};

#endif

} // namespace

#if defined(OVSPATIALQUERY_HAS_CUDA_BACKEND)

ovspatialquery_api_status_t prepareExplicitCudaCapture(
    const ovspatialquery_explicit_instance_table_t& instances, const ovspatialquery_input_array_t* removed,
    const ovspatialquery_dynamic_execution_t& execution, ovspatialquery_ordinal_t baseOrdinal,
    ovspatialquery_ordinal_t targetOrdinal, bool posesOnly, bool rowBits, uint64_t slotCapacity,
    std::shared_ptr<ExplicitPreparationRegistry> registry, const ExplicitCudaCaptureContext& captureContext,
    std::shared_ptr<CudaPopulationCapture>& outCapture) noexcept
{
    const bool completeSet = removed == nullptr;
    if (execution.device.device_id < 0 || execution.cuda_sync.has_stream != 1u ||
        (execution.preparation != OVSPATIALQUERY_PREPARE_ON_HOST &&
         execution.preparation != OVSPATIALQUERY_PREPARE_ON_DEVICE &&
         execution.preparation != OVSPATIALQUERY_PREPARE_ON_DEVICE_HOST_READABLE))
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    const bool onDevice = execution.preparation != OVSPATIALQUERY_PREPARE_ON_HOST;
    if (posesOnly && (instances.geometry_types != nullptr || instances.geometry_values != nullptr ||
                      instances.material_ids != nullptr || (instances.participation_bits != nullptr) != rowBits))
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    // CUDA rows carry no material column yet (ADR-0066): their objects report zero.
    if (instances.material_ids != nullptr)
    {
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    // Nothing is queued on a stream that is capturing a CUDA graph (ADR-0060).
    if (cudaStreamIsCapturing(execution.device.device_id, execution.cuda_sync.stream))
    {
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    try
    {
        std::shared_ptr<ExplicitCudaCapture> capture = std::make_shared<ExplicitCudaCapture>();
        capture->onDevice                            = onDevice;
        capture->hostReadable   = execution.preparation == OVSPATIALQUERY_PREPARE_ON_DEVICE_HOST_READABLE;
        capture->posesOnly      = posesOnly;
        capture->slotCapacity   = slotCapacity;
        capture->captureContext = captureContext;
        if (captureContext.deviceState != nullptr)
        {
            capture->staging = captureContext.deviceState->staging([] { return std::make_shared<CudaStagingPool>(); });
        }
        if (onDevice && !completeSet)
        {
            capture->boxes = captureContext.deviceState != nullptr ? captureContext.deviceState->current() : nullptr;
            if (capture->boxes == nullptr)
            {
                return OVSPATIALQUERY_API_NOT_SUPPORTED; // a host-prepared head cannot continue on the device
            }
        }
        capture->captureDevice    = execution.device;
        capture->stream           = execution.cuda_sync.stream;
        capture->waitEvent        = execution.cuda_sync.wait_event;
        capture->baseOrdinal      = baseOrdinal;
        capture->targetOrdinal    = targetOrdinal;
        capture->completeSet      = completeSet;
        capture->registry         = std::move(registry);
        capture->hasScales        = instances.uniform_scales != nullptr;
        capture->hasParticipation = instances.participation_bits != nullptr;
        capture->columns.resize(5u + (capture->hasScales ? 1u : 0u) + (capture->hasParticipation ? 1u : 0u) +
                                (completeSet ? 0u : 1u));
        size_t                      column = 0u;
        ovspatialquery_api_status_t status =
            validateColumn(instances.object_values, "instances.object_values", execution.device, kDLUInt, 64u, 1, 0,
                           false, capture->columns[column++]);
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            status = validateColumn(instances.geometry_types, "instances.geometry_types", execution.device, kDLInt, 32u,
                                    1, 0, posesOnly, capture->columns[column++]);
        }
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            status = validateColumn(instances.geometry_values, "instances.geometry_values", execution.device, kDLUInt,
                                    64u, 1, 0, posesOnly, capture->columns[column++]);
        }
        // A required real column: null is INVALID_ARGUMENT and a width other
        // than 32 or 64 bits NOT_SUPPORTED, as on the CPU path.
        const auto realColumn = [&execution, &capture, &column](const ovspatialquery_input_array_t* source,
                                                                const char* name, int32_t rank, int64_t trailing)
        {
            if (source == nullptr)
            {
                return OVSPATIALQUERY_API_INVALID_ARGUMENT;
            }
            const uint8_t bits = source->tensor.dtype.bits;
            if (bits != 32u && bits != 64u)
            {
                return OVSPATIALQUERY_API_NOT_SUPPORTED;
            }
            return validateColumn(source, name, execution.device, kDLFloat, bits, rank, trailing, false,
                                  capture->columns[column++]);
        };
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            status = realColumn(instances.translations, "instances.translations", 2, 3);
        }
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            status = realColumn(instances.rotations_xyzw, "instances.rotations_xyzw", 2, 4);
        }
        if (status == OVSPATIALQUERY_API_SUCCESS && capture->hasScales)
        {
            status = realColumn(instances.uniform_scales, "instances.uniform_scales", 1, 0);
        }
        if (status == OVSPATIALQUERY_API_SUCCESS && capture->hasParticipation)
        {
            status = validateColumn(instances.participation_bits, "instances.participation_bits", execution.device,
                                    kDLUInt, 64u, 2, 2, false, capture->columns[column++]);
        }
        if (status == OVSPATIALQUERY_API_SUCCESS && !completeSet)
        {
            status = validateColumn(removed, "removed_object_values", execution.device, kDLUInt, 64u, 1, 0, false,
                                    capture->columns[column++]);
        }
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return status;
        }
        const uint64_t rowCount            = capture->columns[0].rowCount;
        const size_t   instanceColumnCount = capture->columns.size() - (completeSet ? 0u : 1u);
        for (size_t index = 1u; index < instanceColumnCount; ++index)
        {
            if (posesOnly && (index == 1u || index == 2u))
            {
                capture->columns[index].rowCount = rowCount; // absent binding columns
                continue;
            }
            if (capture->columns[index].rowCount != rowCount)
            {
                return OVSPATIALQUERY_API_INVALID_ARGUMENT;
            }
        }
        if (!completeSet && posesOnly && capture->columns.back().rowCount != 0u)
        {
            return OVSPATIALQUERY_API_INVALID_ARGUMENT; // a poses-only update moves objects
        }
        if (capture->boxes && capture->boxes->device().device_id != execution.device.device_id)
            return OVSPATIALQUERY_API_NOT_SUPPORTED;
        outCapture = std::move(capture);
        return OVSPATIALQUERY_API_SUCCESS;
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

#else

ovspatialquery_api_status_t prepareExplicitCudaCapture(const ovspatialquery_explicit_instance_table_t&,
                                                       const ovspatialquery_input_array_t*,
                                                       const ovspatialquery_dynamic_execution_t&,
                                                       ovspatialquery_ordinal_t, ovspatialquery_ordinal_t, bool, bool,
                                                       uint64_t, std::shared_ptr<ExplicitPreparationRegistry>,
                                                       const ExplicitCudaCaptureContext&,
                                                       std::shared_ptr<CudaPopulationCapture>&) noexcept
{
    return OVSPATIALQUERY_API_NOT_SUPPORTED;
}

#endif

} // namespace detail
} // namespace ovspatialquery
