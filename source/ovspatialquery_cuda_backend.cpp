/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_cuda_backend.h"

#if defined(OVSPATIALQUERY_HAS_CUDA_BACKEND)
#include "ovspatialquery_cuda_driver.h"
#endif

#include <atomic>
#include <map>
#include <mutex>
#include <new>
#include <utility>

namespace ovspatialquery
{
namespace detail
{
#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
namespace
{

std::atomic<uint64_t> gDeviceToHostBytes{0u};
std::atomic<uint64_t> gDeviceToDeviceBytes{0u};
std::atomic<uint64_t> gModuleLoads{0u};
std::atomic<uint64_t> gModuleUnloads{0u};

} // namespace
#endif

void CudaModulePins::keep(int32_t deviceId, std::shared_ptr<void> module) noexcept
{
    const std::lock_guard<std::mutex> lock(mMutex);
    for (std::pair<int32_t, std::shared_ptr<void>>& kept : mModules)
    {
        if (kept.first == deviceId)
        {
            kept.second.swap(module);
            return;
        }
    }
    try
    {
        mModules.emplace_back(deviceId, std::move(module));
    }
    catch (const std::bad_alloc&)
    {
        // Unkept: the device's plans still load the module when they need it.
    }
}

const char* cudaBackendUnavailableReason(int32_t deviceId, CudaModulePins* pins)
{
    if (deviceId < 0)
    {
        return nullptr;
    }
    static std::mutex                     mutex;
    static std::map<int32_t, const char*> probed;
    static const char*                    everyDevice = nullptr;
    const char*                           reason      = nullptr;
    std::shared_ptr<void>                 module;
    {
        const std::lock_guard<std::mutex>              lock(mutex);
        const std::map<int32_t, const char*>::iterator found = probed.find(deviceId);
        if (everyDevice != nullptr || found != probed.end())
        {
            reason = everyDevice != nullptr ? everyDevice : found->second;
        }
        else
        {
            reason = cudaDispatchBackendUnavailableReason(deviceId, module);
            // A reason that holds for every device is remembered once, and an ID that
            // names no device is not remembered, so arbitrary IDs cannot grow the table.
            bool sharedReason = reason == kCudaNoBackend;
#if defined(OVSPATIALQUERY_HAS_CUDA_BACKEND)
            sharedReason = sharedReason || reason == kCudaNoDriver || reason == kCudaDriverInitFailed;
            if (reason == kCudaNoSuchDevice || reason == kCudaHostOutOfMemory)
            {
                return reason;
            }
#endif
            if (sharedReason)
            {
                everyDevice = reason;
                return reason;
            }
            try
            {
                probed.emplace(deviceId, reason);
            }
            catch (const std::bad_alloc&)
            {
                // Unremembered: the next request for this device probes again.
            }
        }
    }
    if (reason == nullptr && pins != nullptr)
    {
        // The probe's own module, or, once the probe is remembered, the device's
        // live one: a device the instance already keeps finds the same module.
        if (module == nullptr)
        {
            cudaDispatchAcquireModule(deviceId, module);
        }
        if (module != nullptr)
        {
            pins->keep(deviceId, std::move(module));
        }
    }
    return reason;
}

bool cudaStreamIsCapturing(int32_t deviceId, uintptr_t stream) noexcept
{
#if defined(OVSPATIALQUERY_HAS_CUDA_BACKEND)
    // One driver for the process, opened on first use and never closed, so the
    // check costs each CUDA submission a few driver calls and no library load.
    static Driver* const driver = []() noexcept -> Driver*
    {
        Driver* opened = new (std::nothrow) Driver();
        if (opened != nullptr && !opened->open())
        {
            delete opened;
            opened = nullptr;
        }
        return opened;
    }();
    if (driver == nullptr || driver->streamIsCapturing == nullptr || deviceId < 0)
    {
        return false;
    }
    CuDevice  device  = 0;
    CuContext context = nullptr;
    if (driver->deviceGet(&device, deviceId) != kCuSuccess || driver->primaryRetain(&context, device) != kCuSuccess)
    {
        return false;
    }
    std::atomic<bool> lost{false};
    int               captureStatus = 0;
    CuResult          result        = kCuSuccess;
    {
        CudaCurrentContext scope(*driver, context, lost);
        result = scope.status();
        if (result == kCuSuccess)
        {
            result = driver->streamIsCapturing(reinterpret_cast<CuStream>(stream), &captureStatus);
        }
    }
    (void)driver->primaryRelease(device);
    // A legacy-stream query while another stream captures in global mode
    // reports the implicit dependency the capture would refuse.
    return (result == kCuSuccess && captureStatus != 0) || result == kCuCaptureImplicit;
#else
    (void)deviceId;
    (void)stream;
    return false;
#endif
}

#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
void noteCudaDeviceToHost(uint64_t bytes) { gDeviceToHostBytes.fetch_add(bytes); }

void noteCudaDeviceToDevice(uint64_t bytes) { gDeviceToDeviceBytes.fetch_add(bytes); }

void noteCudaModuleLoad(int change) { (change > 0 ? gModuleLoads : gModuleUnloads).fetch_add(1u); }
#endif

} // namespace detail
} // namespace ovspatialquery

#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
// The bytes the CUDA backend copied to the host and between device buffers, and
// the modules every Driver of the process loaded and unloaded; null skips one.
extern "C" void ovspatialquery_test_get_cuda_dispatch_evidence(uint64_t* deviceToHostBytes,
                                                               uint64_t* deviceToDeviceBytes, uint64_t* moduleLoads,
                                                               uint64_t* moduleUnloads)
{
    const auto read = [](uint64_t* out, const std::atomic<uint64_t>& total)
    {
        if (out != nullptr)
        {
            *out = total.load();
        }
    };
    read(deviceToHostBytes, ovspatialquery::detail::gDeviceToHostBytes);
    read(deviceToDeviceBytes, ovspatialquery::detail::gDeviceToDeviceBytes);
    read(moduleLoads, ovspatialquery::detail::gModuleLoads);
    read(moduleUnloads, ovspatialquery::detail::gModuleUnloads);
}
#endif
