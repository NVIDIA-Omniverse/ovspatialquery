/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_cuda_family_plan_internal.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <utility>

namespace ovspatialquery
{
namespace detail
{

#if defined(OVSPATIALQUERY_HAS_CUDA_BACKEND)

#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
std::atomic<uint64_t> gCudaTransferredEventDestroyAttempts{0u};
std::atomic<uint64_t> gCudaTransferredEventDestroySuccesses{0u};
std::atomic<uint64_t> gCudaTransferredEventPreDestroyFailures{0u};
std::atomic<int>      gCudaTransferredEventDestroyLastStatus{0};
#endif

bool releaseTransferredConsumerDone(Driver& driver, CuContext ownerContext, CuEvent& ownedEvent,
                                    bool terminalDeviceLost) noexcept
{
    if (ownedEvent == nullptr)
    {
        return false;
    }
    if (terminalDeviceLost)
    {
        ownedEvent = nullptr;
        return true;
    }
    CuContext      restore = nullptr;
    const CuResult current = driver.contextGetCurrent(&restore);
    if (current != kCuSuccess)
    {
        if (current == kCuDeviceLost)
        {
            ownedEvent = nullptr;
            return true;
        }
#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
        gCudaTransferredEventPreDestroyFailures.fetch_add(1u);
#endif
        return false;
    }
    const bool     injectContextSelectionFailure = cudaFailurePointArmed(kCudaFailureConsumerDestroyContextSelection);
    const CuResult selected =
        injectContextSelectionFailure ? kCuInvalidContext : driver.contextSetCurrent(ownerContext);
    if (selected != kCuSuccess)
    {
        if (selected == kCuDeviceLost)
        {
            ownedEvent = nullptr;
            return true;
        }
#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
        gCudaTransferredEventPreDestroyFailures.fetch_add(1u);
#endif
        const CuResult restored = driver.contextSetCurrent(restore);
        if (restored == kCuDeviceLost)
        {
            ownedEvent = nullptr;
            return true;
        }
        return false;
    }
    const CuResult destroyed = driver.eventDestroy(ownedEvent);
    // CUDA documents that cuEventDestroy may return an error from an earlier
    // asynchronous launch.  Once the destroy call was issued under the owner
    // context, retrying this handle could double-destroy it even when the
    // returned status is not CUDA_SUCCESS.
    ownedEvent = nullptr;
#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
    gCudaTransferredEventDestroyLastStatus.store(destroyed);
    if (destroyed == kCuSuccess)
    {
        gCudaTransferredEventDestroySuccesses.fetch_add(1u);
    }
    // Publish the completed attempt last so a test snapshot that observes the
    // new attempt also observes its success count and status.
    gCudaTransferredEventDestroyAttempts.fetch_add(1u);
#else
    (void)destroyed;
#endif
    if (destroyed == kCuDeviceLost)
    {
        return true;
    }
    return driver.contextSetCurrent(restore) == kCuDeviceLost;
}

#endif // defined(OVSPATIALQUERY_HAS_CUDA_BACKEND)

const char* const kCudaNoBackend = "CUDA is unavailable: this ovspatialquery build has no CUDA backend";

// The backend is available when the PTX next to the library exposes the one
// kernel every plan launches.
const char* cudaDispatchBackendUnavailableReason(int32_t deviceId, std::shared_ptr<void>& outModule)
{
#if defined(OVSPATIALQUERY_HAS_CUDA_BACKEND)
    std::shared_ptr<CudaContextModule> module;
    const char* const reason = cudaKernelUnavailableReason(deviceId, "ovspatialquery_cuda_family_plan", module);
    outModule                = std::move(module);
    return reason;
#else
    (void)deviceId;
    outModule.reset();
    return kCudaNoBackend;
#endif
}

void cudaDispatchAcquireModule(int32_t deviceId, std::shared_ptr<void>& outModule) noexcept
{
    outModule.reset();
#if defined(OVSPATIALQUERY_HAS_CUDA_BACKEND)
    std::shared_ptr<CudaContextModule> module;
    if (acquireCudaContextModule(deviceId, module) == OVSPATIALQUERY_API_SUCCESS)
    {
        outModule = std::move(module);
    }
#else
    (void)deviceId;
#endif
}

#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
extern "C" void ovspatialquery_test_set_cuda_failure_point(uint32_t point)
{
#if defined(OVSPATIALQUERY_HAS_CUDA_BACKEND)
    gCudaFailurePoint.store(point);
#else
    (void)point;
#endif
}

// Returns whether a held ring gave up waiting since the previous call.
extern "C" bool ovspatialquery_test_hold_cuda_signal_rings(bool held)
{
#if defined(OVSPATIALQUERY_HAS_CUDA_BACKEND)
    gCudaHoldSignalRings.store(held);
    return gCudaSignalRingHoldExpired.exchange(false);
#else
    (void)held;
    return false;
#endif
}

extern "C" uint32_t ovspatialquery_test_get_cuda_failure_point()
{
#if defined(OVSPATIALQUERY_HAS_CUDA_BACKEND)
    return gCudaFailurePoint.load();
#else
    return 0u;
#endif
}

extern "C" void ovspatialquery_test_get_cuda_transferred_event_evidence(uint64_t* attempts, uint64_t* successes,
                                                                        int* lastStatus, uint64_t* preDestroyFailures)
{
#if defined(OVSPATIALQUERY_HAS_CUDA_BACKEND)
    if (attempts != nullptr)
    {
        *attempts = gCudaTransferredEventDestroyAttempts.load();
    }
    if (successes != nullptr)
    {
        *successes = gCudaTransferredEventDestroySuccesses.load();
    }
    if (lastStatus != nullptr)
    {
        *lastStatus = gCudaTransferredEventDestroyLastStatus.load();
    }
    if (preDestroyFailures != nullptr)
    {
        *preDestroyFailures = gCudaTransferredEventPreDestroyFailures.load();
    }
#else
    if (attempts != nullptr)
    {
        *attempts = 0u;
    }
    if (successes != nullptr)
    {
        *successes = 0u;
    }
    if (lastStatus != nullptr)
    {
        *lastStatus = 0;
    }
    if (preDestroyFailures != nullptr)
    {
        *preDestroyFailures = 0u;
    }
#endif
}
#endif

} // namespace detail
} // namespace ovspatialquery
