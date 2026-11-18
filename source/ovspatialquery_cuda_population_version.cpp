/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_cuda_backend.h"
#include "ovspatialquery_cuda_driver.h"

#include <cassert>
#include <cstring>
#include <mutex>
#include <vector>

namespace ovspatialquery
{
namespace detail
{

#if defined(OVSPATIALQUERY_HAS_CUDA_BACKEND)
namespace
{

class CudaPopulationVersionImpl final : public CudaPopulationVersion
{
  public:
    Driver                      driver;
    CuDevice                    device            = 0;
    CuContext                   context           = nullptr;
    CuContext                   submissionContext = nullptr;
    CuStream                    stream            = nullptr;
    CuEvent                     ready             = nullptr;
    int32_t                     deviceId          = 0;
    ovspatialquery_ordinal_t    sourceOrdinal     = 0;
    uint64_t                    objectCount       = 0u;
    CuDevicePtr                 objectIds         = 0u;
    CuDevicePtr                 objectBits        = 0u;
    CuDevicePtr                 objectMinimums    = 0u;
    CuDevicePtr                 objectMaximums    = 0u;
    std::vector<CuDevicePtr>    allocations;
    std::vector<void*>          staging;
    uint64_t                    byteCharge = 0u;
    mutable std::mutex          mutex;
    mutable bool                terminalObserved = false;
    mutable CudaCompletionState terminalState    = CudaCompletionState::pending;
    bool                        workEnqueued     = false;
    bool                        cleanupRecorded  = false;
    bool                        enqueueFailure   = false;

    ~CudaPopulationVersionImpl() override
    {
        const CudaCompletionState state = resultState();
        const bool                safe =
            !workEnqueued || state == CudaCompletionState::complete || state == CudaCompletionState::error;
        assert(safe);
        if (!safe)
        {
            std::terminate();
        }
        CuContext restoreContext   = nullptr;
        bool      cleanupSucceeded = true;
        if (driver.contextGetCurrent != nullptr)
        {
            cleanupSucceeded = driver.contextGetCurrent(&restoreContext) == kCuSuccess;
        }
        if (context != nullptr && driver.contextSetCurrent != nullptr)
        {
            cleanupSucceeded = driver.contextSetCurrent(context) == kCuSuccess && cleanupSucceeded;
        }
        if (ready != nullptr)
        {
            cleanupSucceeded = driver.eventDestroy(ready) == kCuSuccess && cleanupSucceeded;
        }
        for (CuDevicePtr allocation : allocations)
        {
            cleanupSucceeded = driver.memFree(allocation) == kCuSuccess && cleanupSucceeded;
        }
        for (void* pointer : staging)
        {
            cleanupSucceeded = driver.memFreeHost(pointer) == kCuSuccess && cleanupSucceeded;
        }
        if (stream != nullptr)
        {
            cleanupSucceeded = driver.streamDestroy(stream) == kCuSuccess && cleanupSucceeded;
        }
        if (context != nullptr)
        {
            cleanupSucceeded = driver.contextSetCurrent(restoreContext) == kCuSuccess && cleanupSucceeded;
            cleanupSucceeded = driver.primaryRelease(device) == kCuSuccess && cleanupSucceeded;
        }
        assert(cleanupSucceeded || state == CudaCompletionState::error);
    }

    ovspatialquery_api_status_t initializeHostAabb(DLDevice sourceDevice, ovspatialquery_ordinal_t ordinal,
                                                   const ObjectTable& objects)
    {
        deviceId      = sourceDevice.device_id;
        sourceOrdinal = ordinal;
        objectCount   = static_cast<uint64_t>(objects.size());
        std::vector<uint64_t> hostIds;
        std::vector<uint64_t> hostBits;
        std::vector<double>   hostMinimums;
        std::vector<double>   hostMaximums;
        try
        {
            hostIds.reserve(objects.size());
            hostBits.reserve(objects.size() * 2u);
            hostMinimums.reserve(objects.size() * 3u);
            hostMaximums.reserve(objects.size() * 3u);
            for (const PopulationObject& object : objects)
            {
                hostIds.push_back(object.id.value);
                hostBits.insert(hostBits.end(), object.participationBits.words, object.participationBits.words + 2u);
                for (size_t axis = 0u; axis < 3u; ++axis)
                {
                    hostMinimums.push_back(object.bounds.minimum[axis]);
                    hostMaximums.push_back(object.bounds.maximum[axis]);
                }
            }
        }
        catch (const std::bad_alloc&)
        {
            return OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
        if (!driver.open() || driver.deviceGet(&device, deviceId) != kCuSuccess ||
            driver.contextGetCurrent(&submissionContext) != kCuSuccess ||
            driver.primaryRetain(&context, device) != kCuSuccess || driver.contextSetCurrent(context) != kCuSuccess ||
            driver.streamCreate(&stream, kCuStreamNonBlocking) != kCuSuccess ||
            driver.eventCreate(&ready, kCuEventDisableTiming) != kCuSuccess)
        {
            if (context != nullptr)
            {
                (void)driver.contextSetCurrent(submissionContext);
            }
            return OVSPATIALQUERY_API_NOT_SUPPORTED;
        }
        workEnqueued    = true;
        CuResult status = upload(hostIds, objectIds, stream);
        if (status == kCuSuccess)
        {
            status = upload(hostBits, objectBits, stream);
        }
        if (status == kCuSuccess)
        {
            status = upload(hostMinimums, objectMinimums, stream);
        }
        if (status == kCuSuccess)
        {
            status = upload(hostMaximums, objectMaximums, stream);
        }
        if (status == kCuSuccess)
        {
            status          = driver.eventRecord(ready, stream);
            cleanupRecorded = status == kCuSuccess;
        }
        if (status != kCuSuccess)
        {
            enqueueFailure  = true;
            cleanupRecorded = driver.eventRecord(ready, stream) == kCuSuccess;
            status          = cleanupRecorded ? kCuSuccess : status;
        }
        const CuResult restoreStatus = driver.contextSetCurrent(submissionContext);
        if (status != kCuSuccess || restoreStatus != kCuSuccess)
        {
            return OVSPATIALQUERY_API_INTERNAL;
        }
        return OVSPATIALQUERY_API_SUCCESS;
    }

    CudaPopulationDescription description() const override
    {
        CudaPopulationDescription result{};
        result.device         = DLDevice{kDLCUDA, deviceId};
        result.sourceOrdinal  = sourceOrdinal;
        result.ready.value    = reinterpret_cast<uintptr_t>(ready);
        result.objectCount    = objectCount;
        result.objectIds      = reinterpret_cast<void*>(static_cast<uintptr_t>(objectIds));
        result.objectBits     = reinterpret_cast<void*>(static_cast<uintptr_t>(objectBits));
        result.objectMinimums = reinterpret_cast<void*>(static_cast<uintptr_t>(objectMinimums));
        result.objectMaximums = reinterpret_cast<void*>(static_cast<uintptr_t>(objectMaximums));
        return result;
    }

    CudaCompletionState resultState() const override
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (terminalObserved)
        {
            return terminalState;
        }
        if (context == nullptr || ready == nullptr)
        {
            return CudaCompletionState::invalid;
        }
        CuContext restoreContext = nullptr;
        if (driver.contextGetCurrent(&restoreContext) != kCuSuccess || driver.contextSetCurrent(context) != kCuSuccess)
        {
            return CudaCompletionState::invalid;
        }
        if (!workEnqueued)
        {
            driver.contextSetCurrent(restoreContext);
            return CudaCompletionState::complete;
        }
        const CuResult status = cleanupRecorded ? driver.eventQuery(ready) : driver.streamQuery(stream);
        driver.contextSetCurrent(restoreContext);
        CudaCompletionState state = CudaCompletionState::error;
        if (status == kCuSuccess)
        {
            state = enqueueFailure ? CudaCompletionState::error : CudaCompletionState::complete;
        }
        else if (status == kCuNotReady)
        {
            state = CudaCompletionState::pending;
        }
        else if (status == kCuInvalidHandle || status == 1 || status == 201)
        {
            state = CudaCompletionState::invalid;
        }
        if (state == CudaCompletionState::complete || state == CudaCompletionState::error)
        {
            terminalObserved = true;
            terminalState    = state;
        }
        return state;
    }

    void releaseReadyDependencies() override
    {
        const CudaCompletionState state = resultState();
        if (state != CudaCompletionState::complete && state != CudaCompletionState::error)
        {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex);
        size_t                      remaining      = 0u;
        CuContext                   restoreContext = nullptr;
        if (driver.contextGetCurrent(&restoreContext) != kCuSuccess || driver.contextSetCurrent(context) != kCuSuccess)
        {
            return;
        }
        for (void* pointer : staging)
        {
            if (driver.memFreeHost(pointer) != kCuSuccess)
            {
                staging[remaining++] = pointer;
            }
        }
        staging.resize(remaining);
        driver.contextSetCurrent(restoreContext);
    }

    uint64_t deviceByteCharge() const override { return byteCharge; }

  private:
    CuResult allocate(size_t bytes, CuDevicePtr& output)
    {
        if (bytes == 0u)
        {
            output = 0u;
            return kCuSuccess;
        }
        const CuResult status = driver.memAlloc(&output, bytes);
        if (status == kCuSuccess)
        {
            allocations.push_back(output);
            byteCharge += static_cast<uint64_t>(bytes);
        }
        return status;
    }

    template <typename Value>
    CuResult upload(const std::vector<Value>& values, CuDevicePtr& output, CuStream uploadStream)
    {
        const size_t bytes  = values.size() * sizeof(Value);
        CuResult     status = allocate(bytes, output);
        if (status != kCuSuccess || bytes == 0u)
        {
            return status;
        }
        void* host = nullptr;
        status     = driver.memHostAlloc(&host, bytes, kCuMemHostAllocPortable);
        if (status != kCuSuccess)
        {
            return status;
        }
        staging.push_back(host);
        std::memcpy(host, values.data(), bytes);
        return driver.copyHtoDAsync(output, host, bytes, uploadStream);
    }
};

} // namespace
#endif

ovspatialquery_api_status_t createCudaPopulationVersionFromHostObjects(
    DLDevice device, ovspatialquery_ordinal_t sourceOrdinal, const ObjectTable& objects,
    std::shared_ptr<CudaPopulationVersion>& outVersion)
{
#if defined(OVSPATIALQUERY_HAS_CUDA_BACKEND)
    if (device.device_type != kDLCUDA || device.device_id < 0)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    try
    {
        std::shared_ptr<CudaPopulationVersionImpl> version = std::make_shared<CudaPopulationVersionImpl>();
        const ovspatialquery_api_status_t          status = version->initializeHostAabb(device, sourceOrdinal, objects);
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return status;
        }
        outVersion = std::move(version);
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
#else
    (void)device;
    (void)sourceOrdinal;
    (void)objects;
    (void)outVersion;
    return OVSPATIALQUERY_API_NOT_SUPPORTED;
#endif
}

} // namespace detail
} // namespace ovspatialquery
